#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断OBV指标失败原因
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


def diagnose_obv_indicator():
    """诊断OBV指标问题"""
    logger.info("🔍 开始诊断OBV指标...")
    
    try:
        # 尝试导入OBV指标类
        logger.info("📦 尝试导入OBV指标...")
        
        from indicators.obv import OnBalanceVolume
        
        indicator = OnBalanceVolume()
        logger.info(f"✅ 成功导入: OnBalanceVolume")
        
        # 检查指标属性
        logger.info("🔍 检查指标属性...")
        logger.info(f"  - 指标名称: {getattr(indicator, 'name', 'N/A')}")
        logger.info(f"  - 信号周期: {getattr(indicator, 'signal_period', 'N/A')}")
        
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
                
                # 检查OBV相关列
                obv_columns = [col for col in result.columns if 'obv' in col.lower()]
                logger.info(f"  - OBV相关列: {obv_columns}")
                
                if result.empty:
                    logger.warning("⚠️ calculate返回空DataFrame")
                else:
                    # 检查OBV值的合理性
                    if 'OBV' in result.columns:
                        obv_values = result['OBV'].dropna()
                        if len(obv_values) > 0:
                            logger.info(f"  - OBV值范围: {obv_values.min():.2f} 到 {obv_values.max():.2f}")
                            logger.info(f"  - OBV最终值: {obv_values.iloc[-1]:.2f}")
                        else:
                            logger.warning("⚠️ OBV列全为NaN")
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
        
        # 测试OBV计算的准确性
        logger.info("🧪 测试OBV计算准确性...")
        try:
            # 手动计算OBV验证
            close_prices = test_data['close']
            volumes = test_data['volume']
            
            # 手动计算OBV
            manual_obv = [0]  # 初始值为0
            for i in range(1, len(close_prices)):
                price_change = close_prices.iloc[i] - close_prices.iloc[i-1]
                if price_change > 0:
                    manual_obv.append(manual_obv[-1] + volumes.iloc[i])
                elif price_change < 0:
                    manual_obv.append(manual_obv[-1] - volumes.iloc[i])
                else:
                    manual_obv.append(manual_obv[-1])
            
            # 获取指标计算的OBV
            if 'OBV' in result.columns:
                indicator_obv = result['OBV']
                
                # 比较结果
                manual_obv_series = pd.Series(manual_obv, index=result.index)
                
                # 计算相关性
                correlation = manual_obv_series.corr(indicator_obv)
                logger.info(f"  - OBV计算相关性: {correlation:.4f}")
                
                if correlation > 0.95:
                    logger.info("✅ OBV计算准确性良好")
                else:
                    logger.warning(f"⚠️ OBV计算准确性较低: {correlation:.4f}")
            else:
                logger.warning("⚠️ 结果中缺少OBV列")
                
        except Exception as e:
            logger.error(f"❌ OBV计算准确性测试失败: {e}")
        
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
        
        logger.info("🎯 OBV指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_obv_indicator()
    
    if success:
        logger.info("✅ OBV指标诊断完成")
    else:
        logger.error("❌ OBV指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
