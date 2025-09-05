#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断ISLAND_REVERSAL指标的实际状态
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


def create_test_data_with_island_reversal(length: int = 100) -> pd.DataFrame:
    """创建包含岛形反转特征的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.015, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入岛形反转特征
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在多个位置插入岛形反转特征
        if 25 <= i <= 35:  # 顶部岛形反转
            if i == 25:  # 向上跳空
                open_price = close_prices[24] + 2.0  # 跳空高开
                close = open_price + 1.0
                high_price = close + 0.5
                low_price = open_price - 0.2
                volume = np.random.lognormal(11.5, 0.2)
                close_prices[i] = close
            elif 26 <= i <= 34:  # 岛形区域
                open_price = close + np.random.uniform(-0.3, 0.3)
                close = close_prices[25] + np.random.uniform(-0.5, 0.5)  # 在高位震荡
                high_price = max(open_price, close) + abs(np.random.normal(0, 0.2))
                low_price = min(open_price, close) - abs(np.random.normal(0, 0.2))
                volume = np.random.lognormal(11.0, 0.3)
                close_prices[i] = close
            elif i == 35:  # 向下跳空
                open_price = close_prices[34] - 2.0  # 跳空低开
                close = open_price - 1.0
                high_price = open_price + 0.2
                low_price = close - 0.5
                volume = np.random.lognormal(11.8, 0.2)
                close_prices[i] = close
        elif 65 <= i <= 75:  # 底部岛形反转
            if i == 65:  # 向下跳空
                open_price = close_prices[64] - 1.5  # 跳空低开
                close = open_price - 0.8
                high_price = open_price + 0.2
                low_price = close - 0.4
                volume = np.random.lognormal(11.6, 0.2)
                close_prices[i] = close
            elif 66 <= i <= 74:  # 岛形区域
                open_price = close + np.random.uniform(-0.2, 0.2)
                close = close_prices[65] + np.random.uniform(-0.3, 0.3)  # 在低位震荡
                high_price = max(open_price, close) + abs(np.random.normal(0, 0.15))
                low_price = min(open_price, close) - abs(np.random.normal(0, 0.15))
                volume = np.random.lognormal(10.8, 0.3)
                close_prices[i] = close
            elif i == 75:  # 向上跳空
                open_price = close_prices[74] + 1.5  # 跳空高开
                close = open_price + 0.8
                high_price = close + 0.4
                low_price = open_price - 0.2
                volume = np.random.lognormal(11.7, 0.2)
                close_prices[i] = close
        else:
            # 正常交易
            open_price = close + np.random.normal(0, 0.3)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.2))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.2))
            volume = np.random.lognormal(10.8, 0.4)
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'date': dates[:data_length],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def diagnose_island_reversal():
    """诊断ISLAND_REVERSAL指标的实际状态"""
    logger.info("🔍 诊断ISLAND_REVERSAL指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_island_reversal(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            
            # 尝试获取ISLAND_REVERSAL指标
            try:
                island_reversal_indicator = registry.create_indicator('ISLAND_REVERSAL')
                logger.info(f"✅ ISLAND_REVERSAL指标已注册: {type(island_reversal_indicator)}")
                
                # 测试指标计算
                result = island_reversal_indicator.calculate(test_data)
                logger.info(f"✅ 注册的ISLAND_REVERSAL指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
                # 检查岛形反转相关列
                island_columns = [col for col in result.columns if any(keyword in col.lower() for keyword in ['island', 'reversal', '岛形', '反转'])]
                logger.info(f"  - 岛形反转相关列: {island_columns}")
                
                if island_columns:
                    for col in island_columns:
                        if col in result.columns:
                            values = result[col]
                            if pd.api.types.is_numeric_dtype(values):
                                logger.info(f"    - {col}: 数值型，范围 [{values.min():.3f}, {values.max():.3f}]")
                            else:
                                logger.info(f"    - {col}: 类型 {values.dtype}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的ISLAND_REVERSAL指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 2. 检查形态识别指标实现文件
        logger.info("📦 检查形态识别指标实现文件...")
        try:
            pattern_files = [
                'indicators/patterns/island_reversal.py',
                'indicators/pattern/island_reversal.py',
                'indicators/reversal_patterns.py'
            ]
            
            for file_path in pattern_files:
                full_path = Path(root_dir) / file_path
                if full_path.exists():
                    logger.info(f"✅ 找到实现文件: {file_path}")
                    
                    # 尝试导入
                    try:
                        if 'island_reversal.py' in file_path:
                            from indicators.patterns.island_reversal import IslandReversal
                            indicator = IslandReversal()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ 直接导入失败: {e}")
                else:
                    logger.warning(f"⚠️ 实现文件不存在: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ 检查实现文件失败: {e}")
        
        # 3. 手动实现岛形反转分析
        logger.info("🧪 手动实现岛形反转分析...")
        try:
            # 计算岛形反转指标
            close_prices = test_data['close']
            high_prices = test_data['high']
            low_prices = test_data['low']
            volumes = test_data['volume']
            
            # 检测跳空
            gaps = pd.Series(False, index=test_data.index)
            gap_direction = pd.Series(0, index=test_data.index)  # 1=向上跳空, -1=向下跳空
            
            for i in range(1, len(test_data)):
                prev_high = high_prices.iloc[i-1]
                prev_low = low_prices.iloc[i-1]
                curr_low = low_prices.iloc[i]
                curr_high = high_prices.iloc[i]
                
                # 向上跳空：当前最低价 > 前一日最高价
                if curr_low > prev_high:
                    gaps.iloc[i] = True
                    gap_direction.iloc[i] = 1
                # 向下跳空：当前最高价 < 前一日最低价
                elif curr_high < prev_low:
                    gaps.iloc[i] = True
                    gap_direction.iloc[i] = -1
            
            # 检测岛形反转
            island_reversal_top = pd.Series(False, index=test_data.index)
            island_reversal_bottom = pd.Series(False, index=test_data.index)
            
            # 寻找岛形反转模式
            for i in range(5, len(test_data) - 5):
                # 顶部岛形反转：向上跳空 + 高位震荡 + 向下跳空
                if gap_direction.iloc[i] == 1:  # 向上跳空
                    # 寻找后续的向下跳空
                    for j in range(i+3, min(i+15, len(test_data))):
                        if gap_direction.iloc[j] == -1:  # 向下跳空
                            # 检查中间是否为高位震荡
                            island_prices = close_prices.iloc[i:j+1]
                            if island_prices.std() < island_prices.mean() * 0.05:  # 震荡幅度小
                                island_reversal_top.iloc[j] = True
                            break
                
                # 底部岛形反转：向下跳空 + 低位震荡 + 向上跳空
                if gap_direction.iloc[i] == -1:  # 向下跳空
                    # 寻找后续的向上跳空
                    for j in range(i+3, min(i+15, len(test_data))):
                        if gap_direction.iloc[j] == 1:  # 向上跳空
                            # 检查中间是否为低位震荡
                            island_prices = close_prices.iloc[i:j+1]
                            if island_prices.std() < island_prices.mean() * 0.05:  # 震荡幅度小
                                island_reversal_bottom.iloc[j] = True
                            break
            
            # 岛形反转强度
            island_strength = pd.Series(0.0, index=test_data.index)
            for i in range(len(test_data)):
                if island_reversal_top.iloc[i] or island_reversal_bottom.iloc[i]:
                    # 基于成交量和价格变化计算强度
                    volume_ratio = volumes.iloc[i] / volumes.iloc[max(0, i-5):i+1].mean()
                    island_strength.iloc[i] = min(100, volume_ratio * 20)
            
            logger.info(f"  - 检测到跳空: {gaps.sum()} 个")
            logger.info(f"  - 向上跳空: {(gap_direction == 1).sum()} 个")
            logger.info(f"  - 向下跳空: {(gap_direction == -1).sum()} 个")
            logger.info(f"  - 顶部岛形反转: {island_reversal_top.sum()} 个")
            logger.info(f"  - 底部岛形反转: {island_reversal_bottom.sum()} 个")
            logger.info(f"  - 岛形反转强度范围: [{island_strength.min():.3f}, {island_strength.max():.3f}]")
            
            if island_reversal_top.sum() > 0 or island_reversal_bottom.sum() > 0:
                logger.info("✅ 手动岛形反转分析检测到有效信号")
            else:
                logger.warning("⚠️ 手动岛形反转分析未检测到明显信号")
            
        except Exception as e:
            logger.error(f"❌ 手动岛形反转分析失败: {e}")
        
        # 4. 检查BaseIndicator抽象方法
        logger.info("🧪 检查BaseIndicator抽象方法...")
        try:
            if 'island_reversal_indicator' in locals():
                required_methods = [
                    '_calculate_baseindicator',
                    'calculate_raw_score_Indicator_Base_Indicator',
                    'get_patterns_Indicator_Base_Indicator',
                    'calculate_confidence_Indicator_Base_Indicator',
                    'set_parameters_Indicator_Base_Indicator'
                ]
                
                missing_methods = [method for method in required_methods if not hasattr(island_reversal_indicator, method)]
                
                if not missing_methods:
                    logger.info("  ✅ BaseIndicator抽象方法完整")
                else:
                    logger.warning(f"  ⚠️ 缺少方法: {missing_methods}")
            else:
                logger.warning("  ⚠️ 无法检查BaseIndicator方法，指标未成功创建")
                
        except Exception as e:
            logger.error(f"❌ BaseIndicator方法检查失败: {e}")
        
        logger.info("🎯 ISLAND_REVERSAL指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_island_reversal()
    
    if success:
        logger.info("✅ ISLAND_REVERSAL指标诊断完成")
    else:
        logger.error("❌ ISLAND_REVERSAL指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
