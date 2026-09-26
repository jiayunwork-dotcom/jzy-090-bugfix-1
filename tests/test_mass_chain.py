"""多级质量比链：m0 必须计入上方全部质量，mf = 结构 + 上方质量。

并锁定"多级总 Δv 是逐级相加"，绝不退化成整箭质量比的一次对数。
"""

from __future__ import annotations

import math

import pytest

from app.core.engine import StageInput, evaluate
from app.core.losses import DragSpec
from app.core.staging import build_mass_chain, liftoff_mass


def test_chain_includes_all_upper_masses():
    # 两级：一级 ms=2000/mp=8000，二级 ms=1000/mp=3000，载荷 500
    chain = build_mass_chain(
        structural_masses=[2000.0, 1000.0],
        propellant_masses=[8000.0, 3000.0],
        payload_mass=500.0,
    )
    top, bottom = chain[1], chain[0]

    # 上面级上方质量只有载荷
    assert top.upper_mass == 500.0
    assert top.mf == 1000.0 + 500.0
    assert top.m0 == top.mf + 3000.0

    # 一级的上方质量必须包含上面级的全部质量（结构 + 推进剂）+ 载荷：
    # 一级点火时上面级仍满载挂在箭上，其推进剂同样压在一级肩上
    assert bottom.upper_mass == 1000.0 + 3000.0 + 500.0
    assert bottom.mf == 2000.0 + 4500.0
    assert bottom.m0 == bottom.mf + 8000.0

    assert liftoff_mass(chain) == bottom.m0


def test_three_stage_chain_propagates_full_upper_mass():
    # 三级：逐级回推时，下方级必须扛起上方各级的结构 + 推进剂 + 载荷
    chain = build_mass_chain(
        structural_masses=[2500.0, 800.0, 300.0],
        propellant_masses=[20000.0, 5000.0, 1200.0],
        payload_mass=200.0,
    )
    first, second, third = chain[0], chain[1], chain[2]

    # 最上面级：上方只有载荷
    assert third.upper_mass == 200.0
    assert third.mf == 300.0 + 200.0
    assert third.m0 == 500.0 + 1200.0

    # 中间级：上方质量 = 上面级点火质量（含其推进剂）
    assert second.upper_mass == third.m0 == 1700.0
    assert second.mf == 800.0 + 1700.0
    assert second.m0 == 2500.0 + 5000.0

    # 最底一级：上方质量 = 中间级点火质量（含中间级与上面级的推进剂）
    assert first.upper_mass == second.m0 == 7500.0
    assert first.mf == 2500.0 + 7500.0
    assert first.m0 == 10000.0 + 20000.0

    # 起飞质量 = 全箭结构 + 推进剂 + 载荷的总和
    total = (2500.0 + 800.0 + 300.0) + (20000.0 + 5000.0 + 1200.0) + 200.0
    assert liftoff_mass(chain) == first.m0 == total == 30000.0


def test_total_is_sum_of_stages_not_single_log():
    # 非对称质量，避免级间质量比乘积碰巧约化成"起飞质量/载荷"
    stages = [
        StageInput(ve=3200.0, structural_mass=2000.0, propellant_mass=8000.0, burn_time=1.0),
        StageInput(ve=3200.0, structural_mass=1500.0, propellant_mass=5000.0, burn_time=1.0),
    ]
    result = evaluate(stages, payload_mass=1000.0, drag=DragSpec())

    expected_sum = math.fsum(s.ideal_delta_v for s in result.stages)
    assert result.ideal_total_delta_v == pytest.approx(expected_sum)

    # 逐级质量比的乘积
    ratio_product = math.prod(s.mass_ratio for s in result.stages)
    # 反例做法：整箭起飞质量除以最终净载荷做一次对数
    liftoff_over_payload = result.liftoff_mass / result.payload_mass
    assert ratio_product != pytest.approx(liftoff_over_payload)
    wrong_single_log = 3200.0 * math.log(liftoff_over_payload)
    assert result.ideal_total_delta_v != pytest.approx(wrong_single_log)
