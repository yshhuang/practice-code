from scipy.special import comb, perm
from math import pow


def yyl():
    """
    计算烟烟罗三技能对单体变形回合数的数学期望
    单次命中概率为p(初始10%*(1+命中加成))
    :return:
    """
    p = 0.1 # 单次命中概率
    n = 6  # 攻击次数
    e = 0  # 期望值
    for i in range(1, n + 1):
        e = e + i * pow(p, i) * pow(1 - p, n - i) * comb(n, i)
    return e


print(yyl())

