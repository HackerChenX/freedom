#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断THREE_BLACK_CROWS指标的实际状态
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


def create_test_data_with_three_black_crows(length: int = 100) -> pd.DataFrame:
    """创建包含三只乌鸦形态的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入三只乌鸦形态
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在第20-22位置插入三只乌鸦形态
        if 20 <= i <= 22:
            # 三只乌鸦：连续三根阴线，每根都收于接近最低点
            if i == 20:  # 第一根乌鸦
                open_price = close + 2.0  # 高开
                high_price = open_price + 0.5
                low_price = close - 0.5
            elif i == 21:  # 第二根乌鸦
                open_price = close_prices[i-1] + 1.0  # 在前一根收盘价附近开盘
                high_price = open_price + 0.3
                low_price = close - 0.8
            else:  # 第三根乌鸦
                open_price = close_prices[i-1] + 0.5
                high_price = open_price + 0.2
                low_price = close - 1.0
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


def diagnose_three_black_crows():
    """诊断THREE_BLACK_CROWS指标的实际状态"""
    logger.info("🔍 诊断THREE_BLACK_CROWS指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_three_black_crows(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 测试CandlestickPatterns类
        logger.info("📦 测试CandlestickPatterns类...")
        try:
            from indicators.pattern.candlestick_patterns import CandlestickPatterns
            
            basic_indicator = CandlestickPatterns()
            basic_result = basic_indicator.calculate(test_data)
            
            logger.info(f"✅ CandlestickPatterns计算成功，形状: {basic_result.shape}")
            logger.info(f"  - 列名: {list(basic_result.columns)}")
            
            # 检查是否有three_black_crows列
            three_black_crows_columns = [col for col in basic_result.columns if 'three_black_crows' in col.lower() or 'black_crows' in col.lower()]
            logger.info(f"  - THREE_BLACK_CROWS相关列: {three_black_crows_columns}")
            
            if not three_black_crows_columns:
                logger.warning("⚠️ CandlestickPatterns中未找到THREE_BLACK_CROWS相关列")
            
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
            
            # 检查是否有three_black_crows列
            three_black_crows_columns = [col for col in advanced_result.columns if 'three_black_crows' in col.lower() or '三黑鸦' in col or 'black_crows' in col.lower()]
            logger.info(f"  - THREE_BLACK_CROWS相关列: {three_black_crows_columns}")
            
            if three_black_crows_columns:
                for col in three_black_crows_columns:
                    if col in advanced_result.columns:
                        pattern_values = advanced_result[col]
                        if pattern_values.dtype == bool:
                            pattern_count = pattern_values.sum()
                            logger.info(f"  - {col}: 检测到 {pattern_count} 个形态")
                        else:
                            logger.info(f"  - {col}: 列类型 {pattern_values.dtype}")
            else:
                logger.warning("⚠️ AdvancedCandlestickPatterns中也未找到THREE_BLACK_CROWS相关列")
            
        except Exception as e:
            logger.error(f"❌ AdvancedCandlestickPatterns测试失败: {e}")
        
        # 3. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator
            
            # 尝试获取THREE_BLACK_CROWS指标
            try:
                three_black_crows_indicator = get_indicator('THREE_BLACK_CROWS')
                logger.info(f"✅ THREE_BLACK_CROWS指标已注册: {type(three_black_crows_indicator)}")
                
                # 测试指标计算
                result = three_black_crows_indicator.calculate(test_data)
                logger.info(f"✅ 注册的THREE_BLACK_CROWS指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的THREE_BLACK_CROWS指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 4. 手动验证三只乌鸦形态
        logger.info("🧪 手动验证三只乌鸦形态...")
        try:
            open_prices = test_data['open']
            high_prices = test_data['high']
            low_prices = test_data['low']
            close_prices = test_data['close']
            
            # 手动识别三只乌鸦形态
            three_black_crows = pd.Series(False, index=test_data.index)
            
            for i in range(2, len(test_data)):
                # 检查连续三根K线
                if i >= 2:
                    # 三根都是阴线
                    is_bearish_1 = close_prices.iloc[i-2] < open_prices.iloc[i-2]
                    is_bearish_2 = close_prices.iloc[i-1] < open_prices.iloc[i-1]
                    is_bearish_3 = close_prices.iloc[i] < open_prices.iloc[i]
                    
                    # 每根都收于接近最低点
                    close_near_low_1 = (close_prices.iloc[i-2] - low_prices.iloc[i-2]) / (high_prices.iloc[i-2] - low_prices.iloc[i-2] + 1e-10) < 0.3
                    close_near_low_2 = (close_prices.iloc[i-1] - low_prices.iloc[i-1]) / (high_prices.iloc[i-1] - low_prices.iloc[i-1] + 1e-10) < 0.3
                    close_near_low_3 = (close_prices.iloc[i] - low_prices.iloc[i]) / (high_prices.iloc[i] - low_prices.iloc[i] + 1e-10) < 0.3
                    
                    # 连续下跌
                    declining = close_prices.iloc[i-1] < close_prices.iloc[i-2] and close_prices.iloc[i] < close_prices.iloc[i-1]
                    
                    if is_bearish_1 and is_bearish_2 and is_bearish_3 and close_near_low_1 and close_near_low_2 and close_near_low_3 and declining:
                        three_black_crows.iloc[i] = True
            
            manual_count = three_black_crows.sum()
            logger.info(f"  - 手动识别的THREE_BLACK_CROWS数量: {manual_count}")
            
            if manual_count > 0:
                logger.info("✅ 测试数据中确实包含三只乌鸦形态")
            else:
                logger.warning("⚠️ 测试数据中未检测到三只乌鸦形态")
            
        except Exception as e:
            logger.error(f"❌ 手动验证失败: {e}")
        
        logger.info("🎯 THREE_BLACK_CROWS指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_three_black_crows()
    
    if success:
        logger.info("✅ THREE_BLACK_CROWS指标诊断完成")
    else:
        logger.error("❌ THREE_BLACK_CROWS指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
