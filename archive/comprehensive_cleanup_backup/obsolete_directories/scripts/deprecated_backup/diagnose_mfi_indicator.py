#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断MFI指标失败原因
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


def create_test_data(length: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成价格序列
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, length)  # 日收益率
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    close_prices = np.array(prices[1:])  # 去掉初始价格
    data_length = len(close_prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, data_length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, data_length)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, data_length)
    
    data = pd.DataFrame({
        'date': dates[:data_length],  # 确保日期长度匹配
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def diagnose_mfi_indicator():
    """诊断MFI指标问题"""
    logger.info("🔍 开始诊断MFI指标...")
    
    try:
        # 尝试导入MFI指标类
        logger.info("📦 尝试导入MFI指标...")
        
        from indicators.mfi import Mfi
        
        indicator = Mfi()
        logger.info(f"✅ 成功导入: Mfi")
        
        # 检查指标属性
        logger.info("🔍 检查指标属性...")
        logger.info(f"  - 指标名称: {getattr(indicator, 'name', 'N/A')}")
        logger.info(f"  - 计算周期: {getattr(indicator, 'period', 'N/A')}")
        logger.info(f"  - 超买阈值: {getattr(indicator, 'overbought', 'N/A')}")
        logger.info(f"  - 超卖阈值: {getattr(indicator, 'oversold', 'N/A')}")
        
        # 创建测试数据
        test_data = create_test_data(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 测试calculate方法
        logger.info("🧪 测试calculate方法...")
        try:
            result = indicator.calculate(test_data)
            
            if result is None:
                logger.error("❌ calculate方法返回None")
                return False
            elif isinstance(result, pd.DataFrame):
                logger.info(f"✅ calculate返回DataFrame，形状: {result.shape}")
                logger.info(f"  - 列名: {list(result.columns)}")
                
                # 检查MFI相关列
                mfi_columns = [col for col in result.columns if 'mfi' in col.lower()]
                logger.info(f"  - MFI相关列: {mfi_columns}")
                
                if result.empty:
                    logger.warning("⚠️ calculate返回空DataFrame")
                else:
                    # 检查MFI值的合理性
                    if 'mfi' in result.columns:
                        mfi_values = result['mfi'].dropna()
                        if len(mfi_values) > 0:
                            logger.info(f"  - MFI值范围: {mfi_values.min():.2f} 到 {mfi_values.max():.2f}")
                            logger.info(f"  - MFI最终值: {mfi_values.iloc[-1]:.2f}")
                            
                            # 检查MFI值是否在0-100范围内
                            if not all(0 <= val <= 100 for val in mfi_values):
                                logger.warning("⚠️ MFI值超出0-100范围")
                        else:
                            logger.warning("⚠️ MFI列全为NaN")
            else:
                logger.warning(f"⚠️ calculate返回未知类型: {type(result)}")
                
        except Exception as e:
            logger.error(f"❌ calculate方法执行失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
        
        # 检查BaseIndicator抽象方法
        logger.info("🔍 检查BaseIndicator抽象方法...")
        
        required_methods = [
            '_calculate_baseindicator',
            'calculate_raw_score_Indicator_Base_Indicator',
            'get_patterns_Indicator_Base_Indicator',
            'calculate_confidence_Indicator_Base_Indicator',
            'set_parameters_Indicator_Base_Indicator'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(indicator, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            logger.error(f"❌ 缺少BaseIndicator抽象方法: {missing_methods}")
        else:
            logger.info("✅ 所有BaseIndicator抽象方法都存在")
        
        # 测试形态识别功能
        logger.info("🧪 测试形态识别功能...")
        try:
            # 检查是否有get_patterns方法
            if hasattr(indicator, 'get_patterns'):
                patterns = indicator.get_patterns(test_data)
                logger.info(f"✅ get_patterns方法存在，返回类型: {type(patterns)}")
                
                if isinstance(patterns, pd.DataFrame):
                    logger.info(f"  - 形态DataFrame形状: {patterns.shape}")
                    logger.info(f"  - 形态列名: {list(patterns.columns)}")
                    
                    # 检查形态数量
                    pattern_count = 0
                    for col in patterns.columns:
                        if patterns[col].dtype == bool:
                            pattern_count += patterns[col].sum()
                    logger.info(f"  - 检测到的形态数量: {pattern_count}")
                    
                elif isinstance(patterns, list):
                    logger.info(f"  - 形态列表长度: {len(patterns)}")
                else:
                    logger.warning(f"⚠️ get_patterns返回未知类型: {type(patterns)}")
            else:
                logger.warning("⚠️ 缺少get_patterns方法")
                
        except Exception as e:
            logger.error(f"❌ 形态识别测试失败: {e}")
        
        # 测试MFI计算的准确性
        logger.info("🧪 测试MFI计算准确性...")
        try:
            # 手动计算MFI验证
            period = getattr(indicator, 'period', 14)
            
            # 计算典型价格
            tp = (test_data['high'] + test_data['low'] + test_data['close']) / 3
            
            # 计算资金流量
            mf = tp * test_data['volume']
            
            # 计算价格变化
            tp_change = tp.diff()
            
            # 分离正负资金流量
            pmf = np.where(tp_change > 0, mf, 0)
            nmf = np.where(tp_change < 0, mf, 0)
            
            # 计算资金流量比率
            pmf_sum = pd.Series(pmf).rolling(window=period).sum()
            nmf_sum = pd.Series(nmf).rolling(window=period).sum()
            
            # 计算MFI
            mfi_ratio = pmf_sum / (nmf_sum + 1e-10)
            manual_mfi = 100 - (100 / (1 + mfi_ratio))
            
            # 获取指标计算的MFI
            if 'mfi' in result.columns:
                indicator_mfi = result['mfi']
                
                # 比较结果（忽略NaN值）
                manual_valid = manual_mfi.dropna()
                indicator_valid = indicator_mfi.dropna()
                
                if len(manual_valid) > 0 and len(indicator_valid) > 0:
                    # 找到共同的索引
                    common_idx = manual_valid.index.intersection(indicator_valid.index)
                    if len(common_idx) > 0:
                        manual_subset = manual_valid.loc[common_idx]
                        indicator_subset = indicator_valid.loc[common_idx]
                        
                        # 计算相关性
                        correlation = manual_subset.corr(indicator_subset)
                        logger.info(f"  - MFI计算相关性: {correlation:.4f}")
                        
                        if correlation > 0.95:
                            logger.info("✅ MFI计算准确性良好")
                        else:
                            logger.warning(f"⚠️ MFI计算准确性较低: {correlation:.4f}")
                    else:
                        logger.warning("⚠️ 无法找到共同的有效数据点")
                else:
                    logger.warning("⚠️ 缺少有效的MFI数据进行比较")
            else:
                logger.warning("⚠️ 结果中缺少MFI列")
                
        except Exception as e:
            logger.error(f"❌ MFI计算准确性测试失败: {e}")
        
        # 测试边界情况
        logger.info("🧪 测试边界情况...")
        
        # 测试空数据
        try:
            empty_result = indicator.calculate(pd.DataFrame())
            logger.info(f"✅ 空数据处理: {type(empty_result)}")
        except Exception as e:
            logger.error(f"❌ 空数据处理失败: {e}")
        
        # 测试少量数据
        try:
            small_data = create_test_data(5)
            small_result = indicator.calculate(small_data)
            logger.info(f"✅ 少量数据处理: {type(small_result)}")
        except Exception as e:
            logger.error(f"❌ 少量数据处理失败: {e}")
        
        logger.info("🎯 MFI指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_mfi_indicator()
    
    if success:
        logger.info("✅ MFI指标诊断完成")
    else:
        logger.error("❌ MFI指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
