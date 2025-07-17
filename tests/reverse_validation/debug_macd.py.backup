#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
调试MACD形态生成

分析MACD金叉和死叉形态的生成和识别
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from smart_pattern_generator import Smart_pattern_generator
from technical_indicators import Technical_indicators


def debug_macd_golden_cross():
    """调试MACD金叉形态"""
    print("调试MACD金叉形态...")

    generator = Smart_pattern_generator()
    indicators = Technical_indicators()

    # 生成金叉数据
    data = generator.generate_macd_golden_cross_data()
    macd_data = indicators.calculate_macd(data)
    dif = macd_data['DIF']
    dea = macd_data['DEA']

    print(f"数据点数: {len(data)}")
    print(f"DIF范围: {dif.min():.3f} - {dif.max():.3f}")
    print(f"DEA范围: {dea.min():.3f} - {dea.max():.3f}")
    print(f"最终DIF: {dif.iloc[-1]:.3f}")
    print(f"最终DEA: {dea.iloc[-1]:.3f}")

    # 检查金叉 - 扩大搜索范围
    golden_cross_found = False
    cross_point = None
    cross_details = []

    # 检查整个序列中的交叉点
    for i in range(1, len(dif)):
        if dif.iloc[i-1] <= dea.iloc[i-1] and dif.iloc[i] > dea.iloc[i]:
            golden_cross_found = True
            cross_point = i
            cross_details.append(f"第{i}天: DIF从{dif.iloc[i-1]:.3f}上穿DEA{dea.iloc[i-1]:.3f}")

    print(f"金叉检测结果: {'✅ 成功' if golden_cross_found else '❌ 失败'}")

    if cross_details:
        print("发现的金叉点:")
        for detail in cross_details:
            print(f"  {detail}")

    # 显示关键时期的DIF和DEA值
    print("\n关键时期DIF和DEA值:")

    # 显示前10天（下跌期）
    print("前10天（下跌期）:")
    for i in range(10):
        diff = dif.iloc[i] - dea.iloc[i]
        relation = "DIF>DEA" if diff > 0 else "DIF<DEA" if diff < 0 else "DIF=DEA"
        print(f"  第{i+1}天: DIF={dif.iloc[i]:.3f}, DEA={dea.iloc[i]:.3f}, 差值={diff:.3f} ({relation})")

    # 显示中间10天（转折期）
    mid_start = len(dif) // 2 - 5
    print(f"\n中间10天（转折期，从第{mid_start+1}天开始）:")
    for i in range(mid_start, mid_start + 10):
        if i < len(dif):
            diff = dif.iloc[i] - dea.iloc[i]
            relation = "DIF>DEA" if diff > 0 else "DIF<DEA" if diff < 0 else "DIF=DEA"
            print(f"  第{i+1}天: DIF={dif.iloc[i]:.3f}, DEA={dea.iloc[i]:.3f}, 差值={diff:.3f} ({relation})")

    # 显示最后10天（上涨期）
    print("\n最后10天（上涨期）:")
    recent_dif = dif.iloc[-10:]
    recent_dea = dea.iloc[-10:]
    for i in range(len(recent_dif)):
        diff = recent_dif.iloc[i] - recent_dea.iloc[i]
        relation = "DIF>DEA" if diff > 0 else "DIF<DEA" if diff < 0 else "DIF=DEA"
        print(f"  第{len(dif)-10+i+1}天: DIF={recent_dif.iloc[i]:.3f}, DEA={recent_dea.iloc[i]:.3f}, 差值={diff:.3f} ({relation})")

    return golden_cross_found


def debug_macd_death_cross():
    """调试MACD死叉形态"""
    print("\n" + "="*50)
    print("调试MACD死叉形态...")

    generator = Smart_pattern_generator()
    indicators = Technical_indicators()

    # 生成死叉数据
    data = generator.generate_macd_death_cross_data()
    macd_data = indicators.calculate_macd(data)
    dif = macd_data['DIF']
    dea = macd_data['DEA']

    print(f"数据点数: {len(data)}")
    print(f"DIF范围: {dif.min():.3f} - {dif.max():.3f}")
    print(f"DEA范围: {dea.min():.3f} - {dea.max():.3f}")
    print(f"最终DIF: {dif.iloc[-1]:.3f}")
    print(f"最终DEA: {dea.iloc[-1]:.3f}")

    # 检查死叉
    death_cross_found = False
    cross_point = None
    cross_details = []

    # 检查整个序列中的交叉点
    for i in range(1, len(dif)):
        if dif.iloc[i-1] >= dea.iloc[i-1] and dif.iloc[i] < dea.iloc[i]:
            death_cross_found = True
            cross_point = i
            cross_details.append(f"第{i}天: DIF从{dif.iloc[i-1]:.3f}下穿DEA{dea.iloc[i-1]:.3f}")

    print(f"死叉检测结果: {'✅ 成功' if death_cross_found else '❌ 失败'}")

    if cross_details:
        print("发现的死叉点:")
        for detail in cross_details:
            print(f"  {detail}")

    # 显示最后10天的DIF和DEA值
    print("\n最后10天DIF和DEA值:")
    recent_dif = dif.iloc[-10:]
    recent_dea = dea.iloc[-10:]
    for i in range(len(recent_dif)):
        diff = recent_dif.iloc[i] - recent_dea.iloc[i]
        relation = "DIF>DEA" if diff > 0 else "DIF<DEA" if diff < 0 else "DIF=DEA"
        print(f"  第{len(dif)-10+i+1}天: DIF={recent_dif.iloc[i]:.3f}, DEA={recent_dea.iloc[i]:.3f}, 差值={diff:.3f} ({relation})")

    return death_cross_found


if __name__ == '__main__':
    golden_success = debug_macd_golden_cross()
    death_success = debug_macd_death_cross()

    print("\n" + "="*50)
    print("MACD调试总结:")
    print(f"金叉形态: {'✅ 成功' if golden_success else '❌ 失败'}")
    print(f"死叉形态: {'✅ 成功' if death_success else '❌ 失败'}")

    if golden_success and death_success:
        print("🎉 MACD金叉和死叉形态生成成功！")
    else:
        print("❌ MACD形态生成需要进一步优化。")