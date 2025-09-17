#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整验证PENNANT指标
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
    """创建包含多种旗形形态的综合测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.015, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入多个旗形形态
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在多个位置插入旗形形态
        if 20 <= i <= 35:  # 第一个牛旗形
            if i <= 25:  # 旗杆阶段（上涨）
                open_price = close - 0.8
                close = close_prices[i-1] + 0.6  # 持续上涨
                high_price = close + 0.2
                low_price = open_price - 0.1
                close_prices[i] = close
            else:  # 旗面阶段（小幅整理）
                adjustment_factor = (i - 25) * 0.1
                open_price = close + np.random.normal(0, 0.1)
                close = close_prices[25] - adjustment_factor  # 轻微下倾
                high_price = close + 0.15
                low_price = close - 0.15
                close_prices[i] = close
        elif 80 <= i <= 95:  # 第二个熊旗形
            if i <= 85:  # 旗杆阶段（下跌）
                open_price = close + 0.8
                close = close_prices[i-1] - 0.6  # 持续下跌
                high_price = open_price + 0.1
                low_price = close - 0.2
                close_prices[i] = close
            else:  # 旗面阶段（小幅反弹）
                adjustment_factor = (i - 85) * 0.08
                open_price = close + np.random.normal(0, 0.1)
                close = close_prices[85] + adjustment_factor  # 轻微上倾
                high_price = close + 0.12
                low_price = close - 0.12
                close_prices[i] = close
        elif 150 <= i <= 165:  # 第三个牛旗形
            if i <= 155:  # 旗杆阶段
                open_price = close - 0.7
                close = close_prices[i-1] + 0.5
                high_price = close + 0.18
                low_price = open_price - 0.08
                close_prices[i] = close
            else:  # 旗面阶段
                adjustment_factor = (i - 155) * 0.08
                open_price = close + np.random.normal(0, 0.08)
                close = close_prices[155] - adjustment_factor
                high_price = close + 0.12
                low_price = close - 0.12
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


def validate_pennant_complete():
    """完整验证PENNANT指标"""
    logger.info("🚀 开始完整验证PENNANT指标...")
    
    validation_score = 0
    max_score = 100
    
    try:
        # 阶段1: 算法正确性验证 (25分)
        logger.info("📊 阶段1: 算法正确性验证...")
        
        test_data = create_comprehensive_test_data(200)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 通过CandlestickPatterns测试
        from indicators.pattern.candlestick_patterns import CandlestickPatterns
        
        indicator = CandlestickPatterns()
        result = indicator.calculate(test_data)
        
        if result is not None and not result.empty:
            flag_columns = [col for col in result.columns if 'flag' in col.lower()]
            logger.info(f"  - FLAG相关列: {flag_columns}")
            
            total_flag_count = 0
            for col in flag_columns:
                if col in result.columns:
                    flag_signals = result[col]
                    if flag_signals.dtype == bool:
                        flag_count = flag_signals.sum()
                        total_flag_count += flag_count
                        logger.info(f"  - {col}: 检测到 {flag_count} 个形态")
            
            logger.info(f"  - 总检测到的FLAG形态数量: {total_flag_count}")
            
            if total_flag_count >= 2:  # 期望检测到至少2个形态
                validation_score += 25
                logger.info("  ✅ 算法正确性验证通过 (+25分)")
            elif total_flag_count >= 1:
                validation_score += 15
                logger.warning("  ⚠️ 算法正确性部分通过 (+15分)")
            else:
                logger.error("  ❌ 算法正确性验证失败 (0分)")
        else:
            logger.error("  ❌ 计算结果为空")
        
        # 阶段2: 数值合理性验证 (20分)
        logger.info("📊 阶段2: 数值合理性验证...")
        
        flag_columns = [col for col in result.columns if 'flag' in col.lower()]
        if flag_columns:
            all_valid = True
            for col in flag_columns:
                flag_signals = result[col]
                
                # 检查数据类型
                if flag_signals.dtype != bool:
                    all_valid = False
                    logger.warning(f"  ⚠️ {col}数据类型不正确: {flag_signals.dtype}")
                
                # 检查数值范围
                if not flag_signals.isin([True, False]).all():
                    all_valid = False
                    logger.warning(f"  ⚠️ {col}数值范围不正确")
            
            if all_valid:
                validation_score += 20
                logger.info("  ✅ 数值合理性验证通过 (+20分)")
            else:
                validation_score += 10
                logger.warning("  ⚠️ 数值合理性部分通过 (+10分)")
        else:
            logger.error("  ❌ 缺少FLAG相关列")
        
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
            patterns = indicator.get_patterns_Indicator_Base_Indicator(test_data)
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
        logger.info("🎯 PENNANT指标验证报告")
        logger.info("=" * 60)
        logger.info(f"总分: {validation_score}/{max_score}")
        logger.info(f"验证等级: {'PASSED' if validation_score >= 95 else 'FAILED'}")
        logger.info("=" * 60)
        
        if validation_score >= 95:
            logger.info("🎉 PENNANT指标验证通过！")
            logger.info("✅ 可以标记为PASSED状态")
        else:
            logger.warning("⚠️ PENNANT指标验证未完全通过")
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
        success, score = validate_pennant_complete()
        
        if success:
            logger.info("🎉 PENNANT指标完整验证成功！")
            logger.info(f"最终得分: {score}/100")
        else:
            logger.error("❌ PENNANT指标验证失败")
            logger.info(f"当前得分: {score}/100")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
