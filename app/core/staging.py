"""多级质量比链。

这是多级核算与单级核算的关键区别所在：

* 各级"点火质量 m0"必须把它上方所有级（含最终有效载荷）全部计入；
* "燃料耗尽质量 mf"等于本级结构质量加上全部上级质量（含有效载荷）；
* 各级独立求 Δv 后逐级相加，绝不能用整箭起飞质量除以最终净载荷做一次对数。

级列表约定自下而上（index 0 为最底部、最早点火的一级）。
本模块只处理质量链，不依赖速度/损失计算（``StageMass.mass_ratio`` 除外，
它只是 m0/mf 的纯数值商）；所有需要推算各级质量的地方都应复用
``iterate_mass_chain``，保证全服务同一口径。
"""

from __future__ import annotations

from dataclasses import dataclass

from .physics import mass_ratio


@dataclass(frozen=True)
class StageMass:
    """单级质量链上的中间量。"""

    index: int
    structural_mass: float   # ms：本级结构质量
    propellant_mass: float   # mp：本级推进剂质量
    upper_mass: float        # 本级点火时，上方所有级（满箱）+ 最终有效载荷
    m0: float                # 点火质量 = ms + mp + upper_mass
    mf: float                # 耗尽质量 = ms + upper_mass
    mass_ratio: float        # m0 / mf


def iterate_mass_chain(
    structural_masses: list[float],
    propellant_masses: list[float],
    payload_mass: float,
):
    """质量链的唯一递推口径，自最顶一级向最底一级产出。

    每一项为 ``(index, ms, mp, upper_mass, m0, mf)``，其中对任意一级：

    * upper_mass = 本级点火瞬间其上方的全部质量——即紧邻上级的**点火质量**
      （上级自身的结构与推进剂都还在），最顶一级则是最终有效载荷；
    * mf = ms + upper_mass；m0 = mf + mp。

    注意回推时必须把 m0（上级满箱点火质量）带给下一级，而不是 mf：
    下一级先点火、先耗尽，分离时上面级尚未工作、推进剂一斤都没烧掉。

    产出顺序为自上而下（index 从 n-1 到 0）。校验层按此顺序检查
    m0/mf 即可；需要自下而上顺序的调用方自行反转。
    """
    n = len(structural_masses)
    if len(propellant_masses) != n:
        raise ValueError("结构质量与推进剂质量的级数不一致")

    upper_mass = payload_mass
    for i in range(n - 1, -1, -1):
        ms = structural_masses[i]
        mp = propellant_masses[i]
        mf = ms + upper_mass
        m0 = mf + mp
        yield i, ms, mp, upper_mass, m0, mf
        # 本级分离后上方级原封不动（尚未点火）：对下一级而言"上方质量"
        # 是上一级的点火质量 m0，包含其全部推进剂。
        upper_mass = m0


def build_mass_chain(
    structural_masses: list[float],
    propellant_masses: list[float],
    payload_mass: float,
) -> list[StageMass]:
    """由自下而上的各级质量与顶部有效载荷，逐级构建质量链。

    递推复用 :func:`iterate_mass_chain`（自顶向下回推，上方质量唯一确定
    本级 m0/mf），最后再按自下而上的顺序返回，保证 index 0 是最早点火
    的一级。
    """
    chain: list[StageMass] = []
    for i, ms, mp, upper_mass, m0, mf in iterate_mass_chain(
        structural_masses, propellant_masses, payload_mass
    ):
        chain.append(
            StageMass(
                index=i,
                structural_mass=ms,
                propellant_mass=mp,
                upper_mass=upper_mass,
                m0=m0,
                mf=mf,
                mass_ratio=mass_ratio(m0, mf),
            )
        )
    chain.reverse()
    return chain


def liftoff_mass(chain: list[StageMass]) -> float:
    """整箭起飞质量 = 最底一级的点火质量。"""
    if not chain:
        raise ValueError("质量链为空，无起飞质量")
    return chain[0].m0
