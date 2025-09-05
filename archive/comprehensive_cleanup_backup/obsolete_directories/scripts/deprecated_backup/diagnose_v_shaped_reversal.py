#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断V_SHAPED_REVERSAL指标的实际状态
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


def create_test_data_with_v_reversal(length: int = 100) -> pd.DataFrame:
    """创建包含V型反转形态的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入V型反转形态
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在第20-30位置插入V型反转形态
        if 20 <= i <= 30:
            # V型反转：急速下跌后快速反弹
            if i <= 25:  # 急速下跌阶段
                decline_factor = (i - 20) * 0.8  # 逐日加速下跌
                open_price = close + 0.5
                close = close_prices[20] - decline_factor  # 持续下跌
                high_price = open_price + 0.2
                low_price = close - 0.3
                close_prices[i] = close
            elif i == 26:  # V型底部
                open_price = close_prices[25] + 0.2
                close = close_prices[25] - 0.5  # 最低点
                high_price = open_price + 0.1
                low_price = close - 0.1
                close_prices[i] = close
            else:  # 快速反弹阶段
                rebound_factor = (i - 26) * 0.9  # 快速反弹
                open_price = close_prices[26] + 0.3
                close = close_prices[26] + rebound_factor  # 快速上涨
                high_price = close + 0.4
                low_price = open_price - 0.2
                close_prices[i] = close
        elif 60 <= i <= 70:  # 第二个V型反转（倒V型）
            if i <= 65:  # 急速上涨阶段
                rise_factor = (i - 60) * 0.7
                open_price = close - 0.4
                close = close_prices[60] + rise_factor
                high_price = close + 0.3
                low_price = open_price - 0.1
                close_prices[i] = close
            elif i == 66:  # 倒V型顶部
                open_price = close_prices[65] - 0.2
                close = close_prices[65] + 0.4  # 最高点
                high_price = close + 0.1
                low_price = open_price - 0.1
                close_prices[i] = close
            else:  # 快速下跌阶段
                decline_factor = (i - 66) * 0.8
                open_price = close_prices[66] - 0.3
                close = close_prices[66] - decline_factor
                high_price = open_price + 0.2
                low_price = close - 0.4
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


def diagnose_v_shaped_reversal():
    """诊断V_SHAPED_REVERSAL指标的实际状态"""
    logger.info("🔍 诊断V_SHAPED_REVERSAL指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_v_reversal(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 测试CandlestickPatterns类
        logger.info("📦 测试CandlestickPatterns类...")
        try:
            from indicators.pattern.candlestick_patterns import CandlestickPatterns
            
            basic_indicator = CandlestickPatterns()
            basic_result = basic_indicator.calculate(test_data)
            
            logger.info(f"✅ CandlestickPatterns计算成功，形状: {basic_result.shape}")
            logger.info(f"  - 列名: {list(basic_result.columns)}")
            
            # 检查是否有v_reversal相关列
            v_reversal_columns = [col for col in basic_result.columns if 'v_reversal' in col.lower() or 'v_shaped' in col.lower()]
            logger.info(f"  - V_REVERSAL相关列: {v_reversal_columns}")
            
            if v_reversal_columns:
                for col in v_reversal_columns:
                    if col in basic_result.columns:
                        pattern_values = basic_result[col]
                        if pattern_values.dtype == bool:
                            pattern_count = pattern_values.sum()
                            logger.info(f"  - {col}: 检测到 {pattern_count} 个形态")
                        else:
                            logger.info(f"  - {col}: 列类型 {pattern_values.dtype}")
            else:
                logger.warning("⚠️ CandlestickPatterns中未找到V_REVERSAL相关列")
            
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
            
            # 检查是否有v_reversal相关列
            v_reversal_columns = [col for col in advanced_result.columns if 'v_reversal' in col.lower() or 'v型' in col or 'v_shaped' in col.lower()]
            logger.info(f"  - V_REVERSAL相关列: {v_reversal_columns}")
            
            if v_reversal_columns:
                for col in v_reversal_columns:
                    if col in advanced_result.columns:
                        pattern_values = advanced_result[col]
                        if pattern_values.dtype == bool:
                            pattern_count = pattern_values.sum()
                            logger.info(f"  - {col}: 检测到 {pattern_count} 个形态")
                        else:
                            logger.info(f"  - {col}: 列类型 {pattern_values.dtype}")
            else:
                logger.warning("⚠️ AdvancedCandlestickPatterns中也未找到V_REVERSAL相关列")
            
        except Exception as e:
            logger.error(f"❌ AdvancedCandlestickPatterns测试失败: {e}")
        
        # 3. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator
            
            # 尝试获取V_SHAPED_REVERSAL指标
            try:
                v_reversal_indicator = get_indicator('V_SHAPED_REVERSAL')
                logger.info(f"✅ V_SHAPED_REVERSAL指标已注册: {type(v_reversal_indicator)}")
                
                # 测试指标计算
                result = v_reversal_indicator.calculate(test_data)
                logger.info(f"✅ 注册的V_SHAPED_REVERSAL指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的V_SHAPED_REVERSAL指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 4. 手动验证V型反转形态
        logger.info("🧪 手动验证V型反转形态...")
        try:
            open_prices = test_data['open']
            high_prices = test_data['high']
            low_prices = test_data['low']
            close_prices = test_data['close']
            
            # 手动识别V型反转形态
            v_reversal = pd.Series(False, index=test_data.index)
            
            # V型反转识别逻辑：急速下跌后快速反弹
            for i in range(10, len(test_data)):
                # 检查前期是否有明显下跌
                lookback = min(10, i)
                start_idx = i - lookback
                
                # 计算前期跌幅
                if start_idx >= 0:
                    drop_pct = (close_prices.iloc[start_idx] - close_prices.iloc[i-1]) / close_prices.iloc[start_idx]
                    
                    # 计算当日涨幅
                    if i < len(close_prices):
                        rise_pct = (close_prices.iloc[i] - close_prices.iloc[i-1]) / close_prices.iloc[i-1]
                        
                        # V型反转条件：前期有明显下跌，当日有明显反弹
                        if drop_pct > 0.05 and rise_pct > 0.03:
                            v_reversal.iloc[i] = True
            
            manual_count = v_reversal.sum()
            logger.info(f"  - 手动识别的V_REVERSAL数量: {manual_count}")
            
            if manual_count > 0:
                logger.info("✅ 测试数据中确实包含V型反转形态")
                # 显示检测到的位置
                v_positions = v_reversal[v_reversal].index.tolist()
                logger.info(f"  - V型反转位置: {v_positions}")
            else:
                logger.warning("⚠️ 测试数据中未检测到V型反转形态")
            
        except Exception as e:
            logger.error(f"❌ 手动验证失败: {e}")
        
        # 5. 测试get_patterns方法
        logger.info("🧪 测试get_patterns方法...")
        try:
            # 测试CandlestickPatterns的get_patterns
            if hasattr(basic_indicator, 'get_patterns_Indicator_Base_Indicator'):
                patterns = basic_indicator.get_patterns_Indicator_Base_Indicator(test_data)
                logger.info(f"✅ CandlestickPatterns.get_patterns方法存在，返回类型: {type(patterns)}")
                
                if isinstance(patterns, pd.DataFrame):
                    logger.info(f"  - 形态DataFrame形状: {patterns.shape}")
                    logger.info(f"  - 形态列名: {list(patterns.columns)}")
                    
                    # 检查V_REVERSAL相关形态
                    v_reversal_pattern_columns = [col for col in patterns.columns if 'v_reversal' in col.lower() or 'v型' in col]
                    logger.info(f"  - V_REVERSAL相关形态列: {v_reversal_pattern_columns}")
                else:
                    logger.warning(f"⚠️ get_patterns返回未知类型: {type(patterns)}")
            else:
                logger.warning("⚠️ CandlestickPatterns缺少get_patterns方法")
                
        except Exception as e:
            logger.error(f"❌ get_patterns方法测试失败: {e}")
        
        logger.info("🎯 V_SHAPED_REVERSAL指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_v_shaped_reversal()
    
    if success:
        logger.info("✅ V_SHAPED_REVERSAL指标诊断完成")
    else:
        logger.error("❌ V_SHAPED_REVERSAL指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
