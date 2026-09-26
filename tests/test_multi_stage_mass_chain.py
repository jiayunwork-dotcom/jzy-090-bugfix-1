"""回归测试：多级质量链必须把上方级的**满箱点火质量**计入本级 m0/mf。

曾经出现过的 bug：自顶向下回推时把下一级的"上方质量"取成了上级的耗尽
质量 mf（只剩结构 + 载荷），漏掉了上级自身的推进剂。后果是：

* 单级与最顶一级永远正确（没有/不经过这次回推）；
* 下面各级 m0、mf 同时偏小，且 mf 偏小更多，质量比与理想 Δv 被虚高；
* 起飞质量不等于全部结构 + 全部推进剂 + 载荷。

本文件用两级、三级构型逐级锁死 m0/mf/upper_mass，并强制
单次接口与批量接口对同一构型给出逐字节一致的结果。
"""

from __future__ import annotations

import math

import pytest

from app.core.staging import build_mass_chain, iterate_mass_chain, liftoff_mass


# ---- 复现用构型（与故障报告一致）----
TWO_STAGE_PAYLOAD = {
    "stages": [
        # 自下而上：一级
        {"ve": 3000.0, "structural_mass": 4000.0, "propellant_mass": 30000.0,
         "burn_time": 110.0},
        # 二级（最上面一级）
        {"ve": 3400.0, "structural_mass": 1000.0, "propellant_mass": 6000.0,
         "burn_time": 90.0},
    ],
    "payload_mass": 500.0,
}


def test_two_stage_chain_stage_by_stage():
    chain = build_mass_chain(
        structural_masses=[4000.0, 1000.0],
        propellant_masses=[30000.0, 6000.0],
        payload_mass=500.0,
    )

    # 上面级：上方质量只有载荷
    top = chain[1]
    assert top.upper_mass == 500.0
    assert top.mf == 1000.0 + 500.0          # 1500
    assert top.m0 == 1000.0 + 6000.0 + 500.0  # 7500
    assert top.mass_ratio == pytest.approx(5.0)

    # 一级点火时二级尚未工作、推进剂满箱：上方质量 = 二级完整点火质量
    bottom = chain[0]
    assert bottom.upper_mass == 7500.0
    assert bottom.mf == 4000.0 + 7500.0       # 11500（不是 4000+1500=5500）
    assert bottom.m0 == 11500.0 + 30000.0     # 41500（不是 35500）
    assert bottom.mass_ratio == pytest.approx(41500.0 / 11500.0)

    # 起飞质量 = 整箭所有结构 + 所有推进剂 + 载荷
    assert liftoff_mass(chain) == 41500.0
    assert liftoff_mass(chain) == 4000.0 + 30000.0 + 1000.0 + 6000.0 + 500.0


def test_two_stage_ideal_delta_v_per_stage_and_total():
    from app.core.engine import StageInput, evaluate
    from app.core.losses import DragSpec

    stages = [
        StageInput(ve=3000.0, structural_mass=4000.0,
                   propellant_mass=30000.0, burn_time=110.0),
        StageInput(ve=3400.0, structural_mass=1000.0,
                   propellant_mass=6000.0, burn_time=90.0),
    ]
    result = evaluate(stages, payload_mass=500.0, drag=DragSpec())

    assert result.liftoff_mass == 41500.0
    assert result.stages[0].m0 == 41500.0
    assert result.stages[0].mf == 11500.0
    assert result.stages[0].ideal_delta_v == pytest.approx(3850.0, abs=0.1)
    assert result.stages[0].ideal_delta_v == pytest.approx(
        3000.0 * math.log(41500.0 / 11500.0)
    )

    # 最上面一级的结果一个数都不能动
    assert result.stages[1].m0 == 7500.0
    assert result.stages[1].mf == 1500.0
    assert result.stages[1].mass_ratio == pytest.approx(5.0)
    assert result.stages[1].ideal_delta_v == pytest.approx(5472.1, abs=0.1)

    assert result.ideal_total_delta_v == pytest.approx(9322.1, abs=0.1)
    assert result.ideal_total_delta_v == pytest.approx(
        math.fsum(s.ideal_delta_v for s in result.stages)
    )


def test_two_stage_single_endpoint(client):
    resp = client.post("/api/v1/delta-v", json=TWO_STAGE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert body["liftoff_mass"] == 41500.0
    bottom, top = body["stages"]
    assert bottom["index"] == 0
    assert bottom["upper_mass"] == 7500.0
    assert bottom["m0"] == 41500.0
    assert bottom["mf"] == 11500.0
    assert bottom["mass_ratio"] == pytest.approx(41500.0 / 11500.0)
    assert bottom["ideal_delta_v"] == pytest.approx(3850.0, abs=0.1)

    assert top["m0"] == 7500.0
    assert top["mf"] == 1500.0
    assert top["ideal_delta_v"] == pytest.approx(5472.1, abs=0.1)
    assert body["ideal_total_delta_v"] == pytest.approx(9322.1, abs=0.1)


def test_single_and_batch_endpoints_agree(client):
    # 批量里把同一构型放两次，并与单次接口结果逐一比对
    single = client.post("/api/v1/delta-v", json=TWO_STAGE_PAYLOAD)
    assert single.status_code == 200
    expected = single.json()

    batch = client.post(
        "/api/v1/delta-v/batch",
        json={"configurations": [TWO_STAGE_PAYLOAD, TWO_STAGE_PAYLOAD]},
    )
    assert batch.status_code == 200
    body = batch.json()
    assert body["total"] == 2 and body["succeeded"] == 2 and body["failed"] == 0
    for item in body["items"]:
        assert item["ok"] is True
        assert item["error"] is None
        # 批量中的每一级 m0/mf 与单次接口完全一致（浮点逐值相等）
        assert item["result"] == expected


# ---- 三级构型：下面两级都曾被旧算法算错 ----
THREE_STAGE_PAYLOAD = {
    "stages": [
        {"ve": 3000.0, "structural_mass": 3000.0, "propellant_mass": 20000.0,
         "burn_time": 120.0},
        {"ve": 3300.0, "structural_mass": 1500.0, "propellant_mass": 8000.0,
         "burn_time": 90.0},
        {"ve": 3500.0, "structural_mass": 800.0, "propellant_mass": 3000.0,
         "burn_time": 60.0},
    ],
    "payload_mass": 400.0,
}
# 手算口径（自上而下，upper 取上级 m0）：
#   stage2: upper=400,    mf=1200,  m0=4200
#   stage1: upper=4200,   mf=5700,  m0=13700
#   stage0: upper=13700,  mf=16700, m0=36700
THREE_STAGE_EXPECTED = [
    # index, upper, m0, mf
    (0, 13700.0, 36700.0, 16700.0),
    (1, 4200.0, 13700.0, 5700.0),
    (2, 400.0, 4200.0, 1200.0),
]


def test_three_stage_chain_stage_by_stage():
    chain = build_mass_chain(
        structural_masses=[3000.0, 1500.0, 800.0],
        propellant_masses=[20000.0, 8000.0, 3000.0],
        payload_mass=400.0,
    )
    for stage, (index, upper, m0, mf) in zip(chain, THREE_STAGE_EXPECTED):
        assert stage.index == index
        assert stage.upper_mass == upper
        assert stage.m0 == m0
        assert stage.mf == mf
        assert stage.mass_ratio == pytest.approx(m0 / mf)

    assert liftoff_mass(chain) == 36700.0
    assert liftoff_mass(chain) == (
        3000.0 + 20000.0 + 1500.0 + 8000.0 + 800.0 + 3000.0 + 400.0
    )

    # iterate_mass_chain 与 build_mass_chain 必须同一口径
    tuples = list(iterate_mass_chain(
        [3000.0, 1500.0, 800.0], [20000.0, 8000.0, 3000.0], 400.0
    ))
    by_index = {t[0]: t for t in tuples}
    for stage in chain:
        _, _, _, upper, m0, mf = by_index[stage.index]
        assert (upper, m0, mf) == (stage.upper_mass, stage.m0, stage.mf)


def test_three_stage_endpoints_agree(client):
    single = client.post("/api/v1/delta-v", json=THREE_STAGE_PAYLOAD)
    assert single.status_code == 200
    expected = single.json()
    assert expected["liftoff_mass"] == 36700.0

    for stage, (_index, upper, m0, mf) in zip(
        expected["stages"], THREE_STAGE_EXPECTED
    ):
        assert stage["upper_mass"] == upper
        assert stage["m0"] == m0
        assert stage["mf"] == mf

    # 夹一支无关构型，批量结果也必须与单次完全一致
    other = {
        "stages": [
            {"ve": 2900.0, "structural_mass": 5000.0,
             "propellant_mass": 25000.0, "burn_time": 100.0},
            {"ve": 3100.0, "structural_mass": 1200.0,
             "propellant_mass": 5000.0, "burn_time": 70.0},
        ],
        "payload_mass": 300.0,
    }
    batch = client.post(
        "/api/v1/delta-v/batch",
        json={"configurations": [other, THREE_STAGE_PAYLOAD]},
    )
    assert batch.status_code == 200
    items = batch.json()["items"]
    assert items[0]["ok"] is True and items[1]["ok"] is True
    assert items[1]["result"] == expected


def test_liftoff_equals_total_mass_for_arbitrary_stage_count():
    # 任意级数：起飞质量 = Σms + Σmp + 载荷（一至四级）
    import random

    rng = random.Random(20260926)
    for n in range(1, 5):
        ms = [rng.uniform(500.0, 5000.0) for _ in range(n)]
        mp = [rng.uniform(1000.0, 20000.0) for _ in range(n)]
        payload = rng.uniform(0.0, 2000.0)
        chain = build_mass_chain(ms, mp, payload)
        assert liftoff_mass(chain) == pytest.approx(sum(ms) + sum(mp) + payload)
        # 逐级递推自洽：每级 mf = ms + 紧邻上级的 m0
        for i, stage in enumerate(chain):
            if i < n - 1:
                assert stage.mf == pytest.approx(
                    stage.structural_mass + chain[i + 1].m0
                )
            else:
                assert stage.mf == pytest.approx(
                    stage.structural_mass + payload
                )
            assert stage.m0 == pytest.approx(stage.mf + stage.propellant_mass)
