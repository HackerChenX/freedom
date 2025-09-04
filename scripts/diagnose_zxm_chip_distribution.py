#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断ZXM_CHIP_DISTRIBUTION指标的实际状态
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


def create_test_data_with_chip_distribution(length: int = 100) -> pd.DataFrame:
    """创建包含筹码分布特征的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据，特意插入筹码分布特征
    close_prices = np.array(prices[1:])
    data_length = len(close_prices)
    
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 在多个位置插入筹码分布特征
        if 20 <= i <= 30:  # 筹码集中阶段
            # 价格在窄幅震荡，成交量较大，筹码集中
            price_range = 2.0
            open_price = close + np.random.uniform(-price_range/2, price_range/2)
            close = close_prices[19] + np.random.uniform(-price_range/2, price_range/2)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.1))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.1))
            volume = np.random.lognormal(11.5, 0.2)  # 较大成交量
            close_prices[i] = close
        elif 60 <= i <= 70:  # 筹码分散阶段
            # 价格大幅波动，成交量放大，筹码分散
            open_price = close + np.random.normal(0, 1.0)
            close = close_prices[i-1] + np.random.normal(0, 1.5)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.5))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.5))
            volume = np.random.lognormal(12.0, 0.3)  # 大成交量
            close_prices[i] = close
        else:
            # 正常交易
            open_price = close + np.random.normal(0, 0.3)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.2))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.2))
            volume = np.random.lognormal(10.5, 0.4)
        
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


def diagnose_zxm_chip_distribution():
    """诊断ZXM_CHIP_DISTRIBUTION指标的实际状态"""
    logger.info("🔍 诊断ZXM_CHIP_DISTRIBUTION指标的实际状态...")
    
    try:
        # 创建测试数据
        test_data = create_test_data_with_chip_distribution(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 检查指标注册情况
        logger.info("📦 检查指标注册情况...")
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            
            # 尝试获取ZXM_CHIP_DISTRIBUTION指标
            try:
                chip_distribution_indicator = registry.create_indicator('ZXM_CHIP_DISTRIBUTION')
                logger.info(f"✅ ZXM_CHIP_DISTRIBUTION指标已注册: {type(chip_distribution_indicator)}")
                
                # 测试指标计算
                result = chip_distribution_indicator.calculate(test_data)
                logger.info(f"✅ 注册的ZXM_CHIP_DISTRIBUTION指标计算成功，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
                # 检查筹码分布相关列
                chip_columns = [col for col in result.columns if any(keyword in col.lower() for keyword in ['chip', 'distribution', '筹码', '分布', 'concentration'])]
                logger.info(f"  - 筹码分布相关列: {chip_columns}")
                
                if chip_columns:
                    for col in chip_columns:
                        if col in result.columns:
                            values = result[col]
                            if pd.api.types.is_numeric_dtype(values):
                                logger.info(f"    - {col}: 数值型，范围 [{values.min():.3f}, {values.max():.3f}]")
                            else:
                                logger.info(f"    - {col}: 类型 {values.dtype}")
                
            except Exception as e:
                logger.error(f"❌ 获取注册的ZXM_CHIP_DISTRIBUTION指标失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 检查指标注册失败: {e}")
        
        # 2. 检查ZXM指标实现文件
        logger.info("📦 检查ZXM指标实现文件...")
        try:
            zxm_files = [
                'indicators/zxm/chip_distribution.py',
                'indicators/zxm/zxm_chip_distribution.py',
                'indicators/zxm/chip_analysis.py'
            ]
            
            for file_path in zxm_files:
                full_path = Path(root_dir) / file_path
                if full_path.exists():
                    logger.info(f"✅ 找到实现文件: {file_path}")
                    
                    # 尝试导入
                    try:
                        if 'chip_distribution.py' in file_path:
                            from indicators.zxm.chip_distribution import ZXMChipDistribution
                            indicator = ZXMChipDistribution()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                        elif 'zxm_chip_distribution.py' in file_path:
                            from indicators.zxm.zxm_chip_distribution import ZXMChipDistribution
                            indicator = ZXMChipDistribution()
                            result = indicator.calculate(test_data)
                            logger.info(f"  - 直接导入成功，计算结果形状: {result.shape}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ 直接导入失败: {e}")
                else:
                    logger.warning(f"⚠️ 实现文件不存在: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ 检查实现文件失败: {e}")
        
        # 3. 手动实现筹码分布分析
        logger.info("🧪 手动实现筹码分布分析...")
        try:
            # 计算筹码分布指标
            close_prices = test_data['close']
            high_prices = test_data['high']
            low_prices = test_data['low']
            volumes = test_data['volume']
            
            # 价格区间分析（筹码分布的基础）
            price_bins = 20  # 分成20个价格区间
            min_price = close_prices.min()
            max_price = close_prices.max()
            price_range = max_price - min_price
            bin_size = price_range / price_bins
            
            # 筹码集中度计算
            chip_concentration = pd.Series(index=test_data.index, dtype=float)
            
            for i in range(10, len(test_data)):
                # 计算过去10天的价格分布
                recent_prices = close_prices.iloc[i-10:i+1]
                recent_volumes = volumes.iloc[i-10:i+1]
                
                # 计算价格标准差（筹码分散程度）
                price_std = recent_prices.std()
                
                # 计算成交量加权价格标准差
                weighted_avg_price = (recent_prices * recent_volumes).sum() / recent_volumes.sum()
                weighted_std = np.sqrt(((recent_prices - weighted_avg_price) ** 2 * recent_volumes).sum() / recent_volumes.sum())
                
                # 筹码集中度 = 1 / (1 + 标准差)，值越大表示筹码越集中
                chip_concentration.iloc[i] = 1 / (1 + weighted_std)
            
            # 筹码分布强度
            chip_strength = pd.Series(index=test_data.index, dtype=float)
            for i in range(5, len(test_data)):
                # 基于成交量和价格波动计算筹码分布强度
                recent_volumes = volumes.iloc[i-5:i+1]
                recent_price_range = high_prices.iloc[i-5:i+1] - low_prices.iloc[i-5:i+1]
                
                # 成交量密度
                volume_density = recent_volumes.mean() / recent_price_range.mean()
                chip_strength.iloc[i] = min(100, volume_density / 1000)  # 归一化到0-100
            
            # 筹码峰值检测
            chip_peaks = pd.Series(False, index=test_data.index)
            for i in range(10, len(test_data) - 10):
                # 检测筹码集中度的局部峰值
                window = chip_concentration.iloc[i-10:i+11]
                if chip_concentration.iloc[i] == window.max() and chip_concentration.iloc[i] > 0.7:
                    chip_peaks.iloc[i] = True
            
            # 筹码转移信号
            chip_transfer = pd.Series(0, index=test_data.index)
            for i in range(5, len(test_data)):
                # 检测筹码从集中到分散的转移
                if (chip_concentration.iloc[i] < chip_concentration.iloc[i-5] * 0.8 and 
                    volumes.iloc[i] > volumes.iloc[i-5:i].mean() * 1.5):
                    chip_transfer.iloc[i] = 1  # 筹码分散
                elif (chip_concentration.iloc[i] > chip_concentration.iloc[i-5] * 1.2 and
                      volumes.iloc[i] > volumes.iloc[i-5:i].mean() * 1.2):
                    chip_transfer.iloc[i] = -1  # 筹码集中
            
            logger.info(f"  - 筹码集中度计算完成，范围: [{chip_concentration.min():.3f}, {chip_concentration.max():.3f}]")
            logger.info(f"  - 筹码分布强度范围: [{chip_strength.min():.3f}, {chip_strength.max():.3f}]")
            logger.info(f"  - 检测到筹码峰值: {chip_peaks.sum()} 个")
            logger.info(f"  - 筹码转移信号统计: 集中 {(chip_transfer == -1).sum()} 个, 分散 {(chip_transfer == 1).sum()} 个")
            
            if chip_peaks.sum() > 0 or chip_transfer.abs().sum() > 0:
                logger.info("✅ 手动筹码分布分析检测到有效信号")
            else:
                logger.warning("⚠️ 手动筹码分布分析未检测到明显信号")
            
        except Exception as e:
            logger.error(f"❌ 手动筹码分布分析失败: {e}")
        
        # 4. 检查BaseIndicator抽象方法
        logger.info("🧪 检查BaseIndicator抽象方法...")
        try:
            if 'chip_distribution_indicator' in locals():
                required_methods = [
                    '_calculate_baseindicator',
                    'calculate_raw_score_Indicator_Base_Indicator',
                    'get_patterns_Indicator_Base_Indicator',
                    'calculate_confidence_Indicator_Base_Indicator',
                    'set_parameters_Indicator_Base_Indicator'
                ]
                
                missing_methods = [method for method in required_methods if not hasattr(chip_distribution_indicator, method)]
                
                if not missing_methods:
                    logger.info("  ✅ BaseIndicator抽象方法完整")
                else:
                    logger.warning(f"  ⚠️ 缺少方法: {missing_methods}")
            else:
                logger.warning("  ⚠️ 无法检查BaseIndicator方法，指标未成功创建")
                
        except Exception as e:
            logger.error(f"❌ BaseIndicator方法检查失败: {e}")
        
        logger.info("🎯 ZXM_CHIP_DISTRIBUTION指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_zxm_chip_distribution()
    
    if success:
        logger.info("✅ ZXM_CHIP_DISTRIBUTION指标诊断完成")
    else:
        logger.error("❌ ZXM_CHIP_DISTRIBUTION指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
