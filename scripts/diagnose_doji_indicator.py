#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断DOJI指标失败原因
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
    """创建测试数据，包含一些十字星形态"""
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
    
    # 生成高低价，确保有一些十字星形态
    high_prices = []
    low_prices = []
    open_prices = []
    
    for i in range(data_length):
        close = close_prices[i]
        
        # 每10个数据点插入一个十字星形态
        if i % 10 == 5:
            # 十字星：开盘价和收盘价相近，有上下影线
            open_price = close + np.random.normal(0, 0.001)  # 开盘价接近收盘价
            high_price = close + abs(np.random.normal(0, 0.02))  # 上影线
            low_price = close - abs(np.random.normal(0, 0.02))   # 下影线
        else:
            # 正常K线
            open_price = close + np.random.normal(0, 0.01)
            high_price = max(open_price, close) + abs(np.random.normal(0, 0.005))
            low_price = min(open_price, close) - abs(np.random.normal(0, 0.005))
        
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
    
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


def diagnose_doji_indicator():
    """诊断DOJI指标问题"""
    logger.info("🔍 开始诊断DOJI指标...")
    
    try:
        # 尝试导入DOJI指标类
        logger.info("📦 尝试导入DOJI指标...")
        
        from indicators.doji import Doji
        
        indicator = Doji()
        logger.info(f"✅ 成功导入: Doji")
        
        # 检查指标属性
        logger.info("🔍 检查指标属性...")
        logger.info(f"  - 指标名称: {getattr(indicator, 'name', 'N/A')}")
        logger.info(f"  - 十字星阈值: {getattr(indicator, 'doji_threshold', 'N/A')}")
        logger.info(f"  - 影线比例: {getattr(indicator, 'shadow_ratio', 'N/A')}")
        
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
                
                # 检查DOJI相关列
                doji_columns = [col for col in result.columns if 'doji' in col.lower()]
                logger.info(f"  - DOJI相关列: {doji_columns}")
                
                if result.empty:
                    logger.warning("⚠️ calculate返回空DataFrame")
                else:
                    # 检查DOJI形态识别结果
                    if 'doji' in result.columns:
                        doji_signals = result['doji']
                        doji_count = doji_signals.sum() if doji_signals.dtype == bool else len(doji_signals.dropna())
                        logger.info(f"  - 检测到的DOJI形态数量: {doji_count}")
                    else:
                        logger.warning("⚠️ 结果中缺少doji列")
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
        
        # 测试DOJI识别的准确性
        logger.info("🧪 测试DOJI识别准确性...")
        try:
            # 手动计算DOJI验证
            open_prices = test_data['open']
            high_prices = test_data['high']
            low_prices = test_data['low']
            close_prices = test_data['close']
            
            # 计算实体大小和影线长度
            body_size = abs(close_prices - open_prices)
            upper_shadow = high_prices - np.maximum(open_prices, close_prices)
            lower_shadow = np.minimum(open_prices, close_prices) - low_prices
            total_range = high_prices - low_prices
            
            # 十字星判断条件：实体很小，有上下影线
            doji_threshold = 0.1  # 实体占总区间的比例阈值
            manual_doji = (body_size / (total_range + 1e-10)) < doji_threshold
            
            manual_doji_count = manual_doji.sum()
            logger.info(f"  - 手动识别的DOJI数量: {manual_doji_count}")
            
            # 获取指标识别的DOJI
            if 'doji' in result.columns:
                indicator_doji = result['doji']
                
                if indicator_doji.dtype == bool:
                    indicator_doji_count = indicator_doji.sum()
                    logger.info(f"  - 指标识别的DOJI数量: {indicator_doji_count}")
                    
                    # 计算识别准确性
                    if manual_doji_count > 0:
                        accuracy = (manual_doji & indicator_doji).sum() / manual_doji_count
                        logger.info(f"  - DOJI识别准确率: {accuracy:.2%}")
                    else:
                        logger.info("  - 测试数据中没有明显的DOJI形态")
                else:
                    logger.warning("⚠️ DOJI列不是布尔类型")
            else:
                logger.warning("⚠️ 结果中缺少DOJI列")
                
        except Exception as e:
            logger.error(f"❌ DOJI识别准确性测试失败: {e}")
        
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
        
        logger.info("🎯 DOJI指标诊断完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 诊断过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = diagnose_doji_indicator()
    
    if success:
        logger.info("✅ DOJI指标诊断完成")
    else:
        logger.error("❌ DOJI指标诊断失败")
    
    return success


if __name__ == "__main__":
    main()
