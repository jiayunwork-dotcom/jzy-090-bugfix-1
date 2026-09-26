"""回归：多级构型的质量链必须计入上方各级的全部质量（含其推进剂）。

锁定曾导致起飞质量少算、下方各级 Δv 虚高的缺陷：质量链自顶向下回推时，
"上方质量"必须按上方级的点火质量 m0（满载）传递，而非耗尽质量 mf。

覆盖：两级与三级构型逐级核对 m0/mf、起飞质量等于全箭总质量、
单次与批量接口结果一致、内置参考算例起飞质量与单级对照基准。
"""

from __future__ import annotations

import math

import pytest

# 组里复核用的两级构型（有效载荷 500 kg）
TWO_STAGE_CONFIG = {
    "stages": [
        {"ve": 3000.0, "structural_mass": 4000.0, "propellant_mass": 30000.0, "burn_time": 110.0},
        {"ve": 3400.0, "structural_mass": 1000.0, "propellant_mass": 6000.0, "burn_time": 90.0},
    ],
    "payload_mass": 500.0,
}

# 三级构型（有效载荷 200 kg），各级质量刻意取整便于手工复核
THREE_STAGE_CONFIG = {
    "stages": [
        {"ve": 3000.0, "structural_mass": 2500.0, "propellant_mass": 20000.0, "burn_time": 100.0},
        {"ve": 3200.0, "structural_mass": 800.0, "propellant_mass": 5000.0, "burn_time": 70.0},
        {"ve": 3100.0, "structural_mass": 300.0, "propellant_mass": 1200.0, "burn_time": 50.0},
    ],
    "payload_mass": 200.0,
}


def _total_mass(config: dict) -> float:
    return sum(
        s["structural_mass"] + s["propellant_mass"] for s in config["stages"]
    ) + config["payload_mass"]


def test_two_stage_per_stage_masses_and_delta_v(client):
    body = client.post("/api/v1/delta-v", json=TWO_STAGE_CONFIG).json()

    first, second = body["stages"]
    # 一级：点火时必须扛起满载的二级（1000 结构 + 6000 推进剂）与 500 载荷
    assert first["upper_mass"] == 7500.0
    assert first["m0"] == 41500.0
    assert first["mf"] == 11500.0
    assert first["mass_ratio"] == pytest.approx(41500.0 / 11500.0)
    assert first["ideal_delta_v"] == pytest.approx(3850.0, abs=0.05)
    # 二级（最上面一级）本就算对，回归后数值不得漂移
    assert second["m0"] == 7500.0
    assert second["mf"] == 1500.0
    assert second["mass_ratio"] == pytest.approx(5.0)
    assert second["ideal_delta_v"] == pytest.approx(5472.1, abs=0.05)

    assert body["liftoff_mass"] == 41500.0 == _total_mass(TWO_STAGE_CONFIG)
    assert body["ideal_total_delta_v"] == pytest.approx(9322.1, abs=0.05)


def test_three_stage_per_stage_masses(client):
    body = client.post("/api/v1/delta-v", json=THREE_STAGE_CONFIG).json()

    first, second, third = body["stages"]
    # 逐级核对：下方级的上方质量 = 紧邻上一级的点火质量
    assert third["upper_mass"] == 200.0
    assert third["m0"] == 1700.0 and third["mf"] == 500.0
    assert second["upper_mass"] == third["m0"]
    assert second["m0"] == 7500.0 and second["mf"] == 2500.0
    assert first["upper_mass"] == second["m0"]
    assert first["m0"] == 30000.0 and first["mf"] == 10000.0

    assert body["liftoff_mass"] == 30000.0 == _total_mass(THREE_STAGE_CONFIG)
    expected_total = (
        3000.0 * math.log(30000.0 / 10000.0)
        + 3200.0 * math.log(7500.0 / 2500.0)
        + 3100.0 * math.log(1700.0 / 500.0)
    )
    assert body["ideal_total_delta_v"] == pytest.approx(expected_total)


@pytest.mark.parametrize("n_stages", [1, 2, 3, 4])
def test_liftoff_equals_total_mass_any_stage_count(client, n_stages):
    # 任意级数：起飞质量必须等于全箭结构 + 推进剂 + 载荷之和
    config = {
        "stages": [
            {
                "ve": 3000.0 + 100.0 * i,
                "structural_mass": 500.0 * (i + 1),
                "propellant_mass": 4000.0 + 1500.0 * i,
                "burn_time": 60.0,
            }
            for i in range(n_stages)
        ],
        "payload_mass": 350.0,
    }
    body = client.post("/api/v1/delta-v", json=config).json()
    assert body["liftoff_mass"] == pytest.approx(_total_mass(config))
    # 逐级闭合：每级 mf + 本级推进剂 == 本级 m0，且下方级上方质量 == 上一级 m0
    for i, stage in enumerate(body["stages"]):
        assert stage["mf"] + config["stages"][i]["propellant_mass"] == pytest.approx(
            stage["m0"]
        )
        if i + 1 < n_stages:
            assert stage["upper_mass"] == pytest.approx(body["stages"][i + 1]["m0"])
        else:
            assert stage["upper_mass"] == pytest.approx(config["payload_mass"])


def test_single_and_batch_endpoints_agree(client):
    # 同一多级构型：单次核算与批量比选必须给出完全一致的结果
    single = client.post("/api/v1/delta-v", json=THREE_STAGE_CONFIG).json()
    batch_body = client.post(
        "/api/v1/delta-v/batch",
        json={"configurations": [THREE_STAGE_CONFIG, TWO_STAGE_CONFIG]},
    ).json()
    assert batch_body["succeeded"] == 2
    assert batch_body["items"][0]["result"] == single
    assert batch_body["items"][1]["result"]["liftoff_mass"] == 41500.0


def test_reference_example_masses(client):
    body = client.get("/api/v1/reference/example").json()
    two = body["two_stage"]
    # 参考算例：起飞质量 = 2×(2000 结构 + 8000 推进剂) + 2000 载荷
    assert two["liftoff_mass_kg"] == 22000.0
    assert two["stages"][0]["m0_kg"] == 22000.0
    assert two["stages"][0]["mf_kg"] == 14000.0
    assert two["stages"][1]["m0_kg"] == 12000.0
    assert two["stages"][1]["mf_kg"] == 4000.0
    # 单级对照基准不动：约 4157.7 m/s，且分级仍然明显更优
    single = body["single_stage_comparator"]
    assert single["ideal_total_delta_v_mps"] == pytest.approx(4157.7, abs=0.05)
    assert body["staging_better_on_ideal"] is True
    assert body["staging_better_on_net"] is True
