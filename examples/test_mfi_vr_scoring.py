#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MFI和VR指标评分功能

验证MFI（资金流向指标）和VR（成交量指标）的评分机制是否正常工作
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.complete_indicator_registry import complete_registry
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_test_data_Scoring_Test_Mfi_Vr_Scoring(length=100):
    """生成测试数据"""
    np.random.seed(42)  # 固定随机种子以确保结果可重现
    
    # 生成日期序列
    dates = [datetime.now() - timedelta(days=i) for i in range(length)]
    dates.reverse()
    
    # 生成价格数据（模拟股票价格走势）
    base_price = 10.0
    prices = []
    volumes = []
    
    for i in range(length):
        # 价格随机游走
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 0.02)  # 2%的日波动
            price = max(prices[-1] * (1 + change), 1.0)  # 价格不能为负
        
        # 生成OHLC数据
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        close_price = low + (high - low) * np.random.random()
        
        # 确保价格关系合理
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        prices.append(close_price)
        
        # 生成成交量（与价格变化相关）
        price_change = abs(change) if i > 0 else 0.01
        base_volume = 1000000
        volume = int(base_volume * (1 + price_change * 10) * (0.5 + np.random.random()))
        volumes.append(volume)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'trade_date': dates,
        'open': [prices[i] * (0.98 + 0.04 * np.random.random()) for i in range(length)],
        'high': [prices[i] * (1.01 + 0.02 * np.random.random()) for i in range(length)],
        'low': [prices[i] * (0.97 + 0.02 * np.random.random()) for i in range(length)],
        'close': prices,
        'volume': volumes
    })
    
    # 确保OHLC关系正确
    for i in range(length):
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'high'], 
                                 data.loc[i, 'low'], data.loc[i, 'close'])
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'high'], 
                                data.loc[i, 'low'], data.loc[i, 'close'])
    
    return data


def test_mfi_scoring_Scoring():
    """测试MFI指标评分功能"""
    print("\n=== 测试MFI指标评分功能 ===")
    
    try:
        # 生成测试数据
        data = generate_test_data_Scoring_Test_Mfi_Vr_Scoring(100)
        print(f"✓ 生成测试数据完成，数据长度: {len(data)}")
        
        # 创建MFI指标
        mfi = MFI(period=14)
        
        # 计算指标
        result = mfi.calculate(data)
        print(f"✓ MFI指标计算完成，数据长度: {len(result)}")
        
        # 测试评分功能
        score_result = mfi.calculate_score(data)
        print(f"✓ MFI评分计算完成")
        
        # 检查评分结果
        if 'raw_score' not in score_result:
            print("❌ 评分结果缺少raw_score")
            return False
        
        if 'final_score' not in score_result:
            print("❌ 评分结果缺少final_score")
            return False
        
        if 'patterns' not in score_result:
            print("❌ 评分结果缺少patterns")
            return False
        
        # 显示评分统计
        raw_scores = score_result['raw_score']
        final_scores = score_result['final_score']
        
        # 过滤有效评分
        valid_raw_scores = raw_scores.dropna()
        valid_final_scores = final_scores.dropna()
        
        if len(valid_raw_scores) == 0:
            print("❌ 没有有效的原始评分")
            return False
        
        print(f"✓ 原始评分统计:")
        print(f"  - 有效评分数量: {len(valid_raw_scores)}")
        print(f"  - 平均分: {valid_raw_scores.mean():.2f}")
        print(f"  - 最高分: {valid_raw_scores.max():.2f}")
        print(f"  - 最低分: {valid_raw_scores.min():.2f}")
        
        print(f"✓ 最终评分统计:")
        print(f"  - 有效评分数量: {len(valid_final_scores)}")
        print(f"  - 平均分: {valid_final_scores.mean():.2f}")
        print(f"  - 最高分: {valid_final_scores.max():.2f}")
        print(f"  - 最低分: {valid_final_scores.min():.2f}")
        
        # 显示识别的形态
        patterns = score_result['patterns']
        if patterns:
            print(f"✓ 识别的形态: {', '.join(patterns)}")
        else:
            print("✓ 未识别到特殊形态")
        
        # 显示置信度
        confidence = score_result.get('confidence', 0)
        print(f"✓ 置信度: {confidence:.2f}")
        
        # 检查评分范围
        if not all(0 <= score <= 100 for score in valid_raw_scores):
            print("❌ 原始评分超出0-100范围")
            return False
        
        if not all(0 <= score <= 100 for score in valid_final_scores):
            print("❌ 最终评分超出0-100范围")
            return False
        
        print("✅ MFI指标评分功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ MFI指标评分测试失败: {e}")
        logger.error(f"MFI指标评分测试失败: {e}", exc_info=True)
        return False


def test_vr_scoring():
    """测试VR指标评分功能"""
    print("\n=== 测试VR指标评分功能 ===")
    
    try:
        # 生成测试数据
        data = generate_test_data_Scoring_Test_Mfi_Vr_Scoring(100)
        print(f"✓ 生成测试数据完成，数据长度: {len(data)}")
        
        # 创建VR指标
        vr = VR(period=26, ma_period=6)
        
        # 计算指标
        result = vr.calculate(data)
        print(f"✓ VR指标计算完成，数据长度: {len(result)}")
        
        # 测试评分功能
        score_result = vr.calculate_score(data)
        print(f"✓ VR评分计算完成")
        
        # 检查评分结果
        if 'raw_score' not in score_result:
            print("❌ 评分结果缺少raw_score")
            return False
        
        if 'final_score' not in score_result:
            print("❌ 评分结果缺少final_score")
            return False
        
        if 'patterns' not in score_result:
            print("❌ 评分结果缺少patterns")
            return False
        
        # 显示评分统计
        raw_scores = score_result['raw_score']
        final_scores = score_result['final_score']
        
        # 过滤有效评分
        valid_raw_scores = raw_scores.dropna()
        valid_final_scores = final_scores.dropna()
        
        if len(valid_raw_scores) == 0:
            print("❌ 没有有效的原始评分")
            return False
        
        print(f"✓ 原始评分统计:")
        print(f"  - 有效评分数量: {len(valid_raw_scores)}")
        print(f"  - 平均分: {valid_raw_scores.mean():.2f}")
        print(f"  - 最高分: {valid_raw_scores.max():.2f}")
        print(f"  - 最低分: {valid_raw_scores.min():.2f}")
        
        print(f"✓ 最终评分统计:")
        print(f"  - 有效评分数量: {len(valid_final_scores)}")
        print(f"  - 平均分: {valid_final_scores.mean():.2f}")
        print(f"  - 最高分: {valid_final_scores.max():.2f}")
        print(f"  - 最低分: {valid_final_scores.min():.2f}")
        
        # 显示识别的形态
        patterns = score_result['patterns']
        if patterns:
            print(f"✓ 识别的形态: {', '.join(patterns)}")
        else:
            print("✓ 未识别到特殊形态")
        
        # 显示置信度
        confidence = score_result.get('confidence', 0)
        print(f"✓ 置信度: {confidence:.2f}")
        
        # 检查评分范围
        if not all(0 <= score <= 100 for score in valid_raw_scores):
            print("❌ 原始评分超出0-100范围")
            return False
        
        if not all(0 <= score <= 100 for score in valid_final_scores):
            print("❌ 最终评分超出0-100范围")
            return False
        
        print("✅ VR指标评分功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ VR指标评分测试失败: {e}")
        logger.error(f"VR指标评分测试失败: {e}", exc_info=True)
        return False


def test_mfi_vr_patterns():
    """测试MFI和VR指标形态识别"""
    print("\n=== 测试MFI和VR指标形态识别 ===")
    
    try:
        # 生成测试数据
        data = generate_test_data_Scoring_Test_Mfi_Vr_Scoring(100)
        print(f"✓ 生成测试数据完成，数据长度: {len(data)}")
        
        # 测试MFI形态识别
        mfi = MFI(period=14)
        mfi.calculate(data)
        mfi_patterns = mfi.identify_patterns(data)
        print(f"✓ MFI形态识别: {mfi_patterns}")
        
        # 测试VR形态识别
        vr = VR(period=26, ma_period=6)
        vr.calculate(data)
        vr_patterns = vr.identify_patterns(data)
        print(f"✓ VR形态识别: {vr_patterns}")
        
        print("✅ 形态识别功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 形态识别测试失败: {e}")
        logger.error(f"形态识别测试失败: {e}", exc_info=True)
        return False


def main_testmfivrscoring():
    """主函数"""
    print("开始测试MFI和VR指标评分功能...")
    
    # 测试结果统计
    test_results = []
    
    # 测试MFI评分
    test_results.append(test_mfi_scoring_Scoring())
    
    # 测试VR评分
    test_results.append(test_vr_scoring())
    
    # 测试形态识别
    test_results.append(test_mfi_vr_patterns())
    
    # 统计测试结果
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n=== 测试结果汇总 ===")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！MFI和VR指标评分功能正常工作")
    else:
        print("⚠️  部分测试失败，请检查相关功能")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main_testmfivrscoring() 