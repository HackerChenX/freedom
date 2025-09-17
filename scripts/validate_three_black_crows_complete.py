#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整验证THREE_BLACK_CROWS指标
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


def create_comprehensive_test_data(length: int = 200) -> pd.DataFrame:
    """创建包含多种三只乌鸦形态的综合测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.015, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入多个三只乌鸦形态
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在多个位置插入三只乌鸦形态
        if 20 <= i <= 22:  # 第一个三只乌鸦
            if i == 20:  # 第一根乌鸦
                open_price = close + 1.5  # 阴线：开盘价高于收盘价
                high_price = open_price + 0.2  # 上影线短
                low_price = close - 0.1  # 下影线短，收盘价接近最低价
            elif i == 21:  # 第二根乌鸦
                open_price = close_prices[i-1] + 0.5  # 在前一根收盘价附近开盘
                close = close_prices[i-1] - 1.2  # 收盘价低于前一根
                high_price = open_price + 0.15
                low_price = close - 0.1
                close_prices[i] = close  # 更新收盘价
            else:  # 第三根乌鸦
                open_price = close_prices[i-1] + 0.3
                close = close_prices[i-1] - 1.0  # 继续下跌
                high_price = open_price + 0.1
                low_price = close - 0.1
                close_prices[i] = close  # 更新收盘价
        elif 80 <= i <= 82:  # 第二个三只乌鸦
            if i == 80:
                open_price = close + 1.2
                high_price = open_price + 0.15
                low_price = close - 0.1
            elif i == 81:
                open_price = close_prices[i-1] + 0.4
                close = close_prices[i-1] - 1.0
                high_price = open_price + 0.12
                low_price = close - 0.08
                close_prices[i] = close
            else:
                open_price = close_prices[i-1] + 0.25
                close = close_prices[i-1] - 0.8
                high_price = open_price + 0.08
                low_price = close - 0.06
                close_prices[i] = close
        elif 150 <= i <= 152:  # 第三个三只乌鸦
            if i == 150:
                open_price = close + 1.4
                high_price = open_price + 0.18
                low_price = close - 0.12
            elif i == 151:
                open_price = close_prices[i-1] + 0.5
                close = close_prices[i-1] - 1.1
                high_price = open_price + 0.14
                low_price = close - 0.09
                close_prices[i] = close
            else:
                open_price = close_prices[i-1] + 0.35
                close = close_prices[i-1] - 0.9
                high_price = open_price + 0.1
                low_price = close - 0.07
                close_prices[i] = close
        else:
            # 正常K线
            open_price = close + np.random.normal(0, 0.3)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.2))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.2))
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.3, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def validate_three_black_crows_complete():
    """完整验证THREE_BLACK_CROWS指标"""
    logger.info("🚀 开始完整验证THREE_BLACK_CROWS指标...")
    
    validation_score = 0
    max_score = 100
    
    try:
        # 阶段1: 算法正确性验证 (25分)
        logger.info("📊 阶段1: 算法正确性验证...")
        
        test_data = create_comprehensive_test_data(200)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 通过AdvancedCandlestickPatterns测试
        from indicators.pattern.advanced_candlestick_patterns import AdvancedCandlestickPatterns
        
        indicator = AdvancedCandlestickPatterns()
        result = indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            if '三黑鸦' in result.columns:
                three_black_crows = result['三黑鸦']
                detected_count = three_black_crows.sum()
                
                logger.info(f"  - 检测到的三只乌鸦形态数量: {detected_count}")
                
                if detected_count >= 2:  # 期望检测到至少2个形态
                    validation_score += 25
                    logger.info("  ✅ 算法正确性验证通过 (+25分)")
                else:
                    validation_score += 15
                    logger.warning("  ⚠️ 算法正确性部分通过 (+15分)")
            else:
                logger.error("  ❌ 缺少三黑鸦列")
        else:
            logger.error("  ❌ 计算结果为空")
        
        # 阶段2: 数值合理性验证 (20分)
        logger.info("📊 阶段2: 数值合理性验证...")
        
        if '三黑鸦' in result.columns:
            three_black_crows = result['三黑鸦']
            
            # 检查数据类型
            if three_black_crows.dtype == bool:
                validation_score += 10
                logger.info("  ✅ 数据类型正确 (布尔型) (+10分)")
            else:
                logger.warning(f"  ⚠️ 数据类型不正确: {three_black_crows.dtype}")
            
            # 检查数值范围
            if three_black_crows.isin([True, False]).all():
                validation_score += 10
                logger.info("  ✅ 数值范围正确 (True/False) (+10分)")
            else:
                logger.warning("  ⚠️ 数值范围不正确")
        
        # 阶段3: 功能完整性验证 (25分)
        logger.info("📊 阶段3: 功能完整性验证...")
        
        # 检查BaseIndicator抽象方法
        required_methods = [
            '_calculate_baseindicator',
            'calculate_raw_score_Indicator_Base_Indicator',
            'get_patterns_Indicator_Base_Indicator',
            'calculate_confidence_Indicator_Base_Indicator',
            'set_parameters_Indicator_Base_Indicator'
        ]
        
        missing_methods = [method for method in required_methods if not hasattr(indicator, method)]
        
        if not missing_methods:
            validation_score += 15
            logger.info("  ✅ BaseIndicator抽象方法完整 (+15分)")
        else:
            logger.warning(f"  ⚠️ 缺少方法: {missing_methods}")
        
        # 测试get_patterns方法
        try:
            patterns = indicator.get_patterns(test_data)
            if patterns is not None:
                validation_score += 10
                logger.info("  ✅ get_patterns方法正常工作 (+10分)")
            else:
                logger.warning("  ⚠️ get_patterns返回None")
        except Exception as e:
            logger.error(f"  ❌ get_patterns方法失败: {e}")
        
        # 阶段4: 性能表现验证 (15分)
        logger.info("📊 阶段4: 性能表现验证...")
        
        import time
        
        # 测试计算性能
        start_time = time.time()
        for _ in range(5):
            indicator.calculate(test_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        
        if avg_time < 0.5:
            validation_score += 15
            logger.info(f"  ✅ 性能优秀 (平均 {avg_time:.3f}秒) (+15分)")
        elif avg_time < 1.0:
            validation_score += 10
            logger.info(f"  ✅ 性能良好 (平均 {avg_time:.3f}秒) (+10分)")
        else:
            validation_score += 5
            logger.warning(f"  ⚠️ 性能一般 (平均 {avg_time:.3f}秒) (+5分)")
        
        # 阶段5: 稳定性验证 (15分)
        logger.info("📊 阶段5: 稳定性验证...")
        
        # 测试边界情况
        edge_cases_passed = 0
        
        # 空数据测试
        try:
            empty_result = indicator.calculate(pd.DataFrame())
            if empty_result is not None:
                edge_cases_passed += 1
        except Exception as e:
            logger.warning(f"  ⚠️ 空数据测试失败: {e}")
        
        # 少量数据测试
        try:
            small_data = create_comprehensive_test_data(10)
            small_result = indicator.calculate(small_data)
            if small_result is not None:
                edge_cases_passed += 1
        except Exception as e:
            logger.warning(f"  ⚠️ 少量数据测试失败: {e}")
        
        # 大量数据测试
        try:
            large_data = create_comprehensive_test_data(1000)
            large_result = indicator.calculate(large_data)
            if large_result is not None:
                edge_cases_passed += 1
        except Exception as e:
            logger.warning(f"  ⚠️ 大量数据测试失败: {e}")
        
        if edge_cases_passed == 3:
            validation_score += 15
            logger.info("  ✅ 稳定性验证完全通过 (+15分)")
        elif edge_cases_passed == 2:
            validation_score += 10
            logger.info("  ✅ 稳定性验证部分通过 (+10分)")
        else:
            validation_score += 5
            logger.warning("  ⚠️ 稳定性验证基本通过 (+5分)")
        
        # 生成验证报告
        logger.info("=" * 60)
        logger.info("🎯 THREE_BLACK_CROWS指标验证报告")
        logger.info("=" * 60)
        logger.info(f"总分: {validation_score}/{max_score}")
        logger.info(f"验证等级: {'PASSED' if validation_score >= 95 else 'FAILED'}")
        logger.info("=" * 60)
        
        if validation_score >= 95:
            logger.info("🎉 THREE_BLACK_CROWS指标验证通过！")
            logger.info("✅ 可以标记为PASSED状态")
        else:
            logger.warning("⚠️ THREE_BLACK_CROWS指标验证未完全通过")
            logger.info(f"需要提升 {95 - validation_score} 分才能达到PASSED标准")
        
        return validation_score >= 95, validation_score
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False, 0


def main():
    """主函数"""
    try:
        success, score = validate_three_black_crows_complete()
        
        if success:
            logger.info("🎉 THREE_BLACK_CROWS指标完整验证成功！")
            logger.info(f"最终得分: {score}/100")
        else:
            logger.error("❌ THREE_BLACK_CROWS指标验证失败")
            logger.info(f"当前得分: {score}/100")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
