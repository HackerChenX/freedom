#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断PENNANT指标的实际状态
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def create_test_data_with_pennant(length: int = 100) -> pd.DataFrame:
    """创建包含三角旗形形态的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入三角旗形形态
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在第20-30位置插入三角旗形形态
        if 20 <= i <= 30:
            # 三角旗形：先有强势上涨，然后小幅整理，形成三角旗形
            if i == 20:  # 旗杆开始
                open_price = close - 2.0  # 强势上涨
                high_price = close + 0.5
                low_price = open_price - 0.2
            elif 21 <= i <= 29:  # 旗形整理
                # 逐渐收敛的三角形整理
                convergence_factor = (30 - i) / 9  # 收敛因子
                open_price = close + np.random.normal(0, 0.2 * convergence_factor)
                high_price = close + abs(np.random.normal(0, 0.3 * convergence_factor))
                low_price = close - abs(np.random.normal(0, 0.3 * convergence_factor))
            elif i == 30:  # 突破
                open_price = close_prices[i-1] + 0.5
                close = close_prices[i-1] + 1.5  # 向上突破
                high_price = close + 0.3
                low_price = open_price - 0.2
                close_prices[i] = close
        else:
            # 正常K线
            open_price = close + np.random.normal(0, 0.5)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.3))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.3))
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def diagnose_pennant():
    """诊断PENNANT指标的实际状态"""
    logger.info("🔍 诊断PENNANT指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_pennant(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 测试CandlestickPatterns类
        logger.info("📦 测试CandlestickPatterns类...")
        try:
            from indicators.pattern.candlestick_patterns import CandlestickPatterns
            
            basic_indicator = CandlestickPatterns()
            basic_result = basic_indicator.calculate(test_data)
            
            logger.info(f"✅ CandlestickPatterns计算成功，形状: {basic_result.shape}")
            logger.info(f"  - 列名: {list(basic_result.columns)}")
            
            # 检查是否有pennant相关列
            pennant_columns = [col for col in basic_result.columns if 'pennant' in col.lower() or 'flag' in col.lower()]
            logger.info(f"  - PENNANT相关列: {pennant_columns}")
            
            if pennant_columns:
                for col in pennant_columns:
                    if col in basic_result.columns:
                        pattern_values = basic_result[col]
                        if pattern_values.dtype == bool:
                            pattern_count = pattern_values.sum()
                            logger.info(f"  - {col}: 检测到 {pattern_count} 个形态")
                        else:
                            logger.info(f"  - {col}: 列类型 {pattern_values.dtype}")
            else:
                logger.warning("⚠️ CandlestickPatterns中未找到PENNANT相关列")
            
        except Exception as e:
            logger.error(f"❌ CandlestickPatterns测试失败: {e}")
        
        # 2. 测试AdvancedCandlestickPatterns类
        logger.info("📦 测试AdvancedCandlestickPatterns类...")
        try:
            from indicators.pattern.advanced_candlestick_patterns import AdvancedCandlestickPatterns
            
            advanced_indicator = AdvancedCandlestickPatterns()
            advanced_result = advanced_indicator.calculate(test_data)
            
            logger.info(f"✅ AdvancedCandlestickPatterns计算成功，形状: {advanced_result.shape}")
            logger.info(f"  - 列名: {list(advanced_result.columns)}")
            
            # 检查是否有pennant相关列
            pennant_columns = [col for col in advanced_result.columns if 'pennant' in col.lower() or '三角旗' in col or 'flag' in col.lower()]
            logger.info(f"  - PENNANT相关列: {pennant_columns}")
            
            if pennant_columns:
                for col in pennant_columns:
                    if col in advanced_result.columns:
                        pattern_values = advanced_result[col]
                        if pattern_values.dtype == bool:
                            pattern_count = pattern_values.sum()
                            logger.info(f"  - {col}: 检测到 {pattern_count} 个形态")
                        else:
                            logger.info(f"  - {col}: 列类型 {pattern_values.dtype}")
            else:
                logger.warning("⚠️ AdvancedCandlestickPatterns中也未找到PENNANT相关列")
            
        except Exception as e:
            logger.error(f"❌ AdvancedCandlestickPatterns测试失败: {e}")
        
        # 3. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator
            
            # 尝试获取PENNANT指标
            try:
                pennant_indicator = get_indicator('PENNANT')
                logger.info(f"✅ PENNANT指标已注册: {type(pennant_indicator)}")
                
                # 测试指标计算
                result = pennant_indicator.calculate(test_data)
                logger.info(f"✅ 注册的PENNANT指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的PENNANT指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 4. 手动验证三角旗形形态
        logger.info("🧪 手动验证三角旗形形态...")
        try:
            open_prices = test_data['open']
            high_prices = test_data['high']
            low_prices = test_data['low']
            close_prices = test_data['close']
            
            # 手动识别三角旗形形态
            pennant = pd.Series(False, index=test_data.index)
            
            # 简化的三角旗形识别逻辑
            for i in range(10, len(test_data) - 5):
                # 检查是否有旗杆（前期强势上涨）
                pole_start = max(0, i - 10)
                pole_gain = (close_prices.iloc[i] - close_prices.iloc[pole_start]) / close_prices.iloc[pole_start]
                
                if pole_gain > 0.05:  # 前期有5%以上的涨幅
                    # 检查后续是否有收敛整理
                    flag_end = min(len(test_data) - 1, i + 5)
                    flag_range = high_prices.iloc[i:flag_end+1].max() - low_prices.iloc[i:flag_end+1].min()
                    avg_price = close_prices.iloc[i:flag_end+1].mean()
                    
                    # 整理幅度相对较小
                    if flag_range / avg_price < 0.03:  # 整理幅度小于3%
                        pennant.iloc[flag_end] = True
            
            manual_count = pennant.sum()
            logger.info(f"  - 手动识别的PENNANT数量: {manual_count}")
            
            if manual_count > 0:
                logger.info("✅ 测试数据中确实包含三角旗形形态")
            else:
                logger.warning("⚠️ 测试数据中未检测到三角旗形形态")
            
        except Exception as e:
            logger.error(f"❌ 手动验证失败: {e}")
        
        # 5. 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        try:
            # 测试CandlestickPatterns的get_patterns
            if hasattr(basic_indicator, 'get_patterns'):
                patterns = basic_indicator.get_patterns(test_data)
                logger.info(f"✅ CandlestickPatterns.get_patterns方法存在，返回类型: {type(patterns)}")
                
                if isinstance(patterns, pd.DataFrame):
                    logger.info(f"  - 形态DataFrame形状: {patterns.shape}")
                    logger.info(f"  - 形态列名: {list(patterns.columns)}")
                    
                    # 检查PENNANT相关形态
                    pennant_pattern_columns = [col for col in patterns.columns if 'pennant' in col.lower() or 'flag' in col.lower()]
                    logger.info(f"  - PENNANT相关形态列: {pennant_pattern_columns}")
                else:
                    logger.warning(f"⚠️ get_patterns返回未知类型: {type(patterns)}")
            else:
                logger.warning("⚠️ CandlestickPatterns缺少get_patterns方法")
            
            # 测试AdvancedCandlestickPatterns的get_patterns
            if hasattr(advanced_indicator, 'get_patterns'):
                advanced_patterns = advanced_indicator.get_patterns(test_data)
                logger.info(f"✅ AdvancedCandlestickPatterns.get_patterns方法存在，返回类型: {type(advanced_patterns)}")
                
                if isinstance(advanced_patterns, pd.DataFrame):
                    logger.info(f"  - 高级形态DataFrame形状: {advanced_patterns.shape}")
                    logger.info(f"  - 高级形态列名: {list(advanced_patterns.columns)}")
                    
                    # 检查PENNANT相关形态
                    pennant_pattern_columns = [col for col in advanced_patterns.columns if 'pennant' in col.lower() or '三角旗' in col]
                    logger.info(f"  - PENNANT相关高级形态列: {pennant_pattern_columns}")
                else:
                    logger.warning(f"⚠️ 高级get_patterns返回未知类型: {type(advanced_patterns)}")
            else:
                logger.warning("⚠️ AdvancedCandlestickPatterns缺少get_patterns方法")
                
        except Exception as e:
            logger.error(f"❌ get_patterns方法测试失败: {e}")
        
        logger.info("🎯 PENNANT指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_pennant()
    
    if success:
        logger.info("✅ PENNANT指标诊断完成")
    else:
        logger.error("❌ PENNANT指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
