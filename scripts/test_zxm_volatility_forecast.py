#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试ZXM_VOLATILITY_FORECAST指标
找出验证失败的原因并修复
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
from indicators.zxm.zxm_volatility_forecast import ZXMVolatilityForecast

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
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, length)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, length)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def test_zxm_volatility_forecast():
    """测试ZXM波动率预测指标"""
    logger.info("🧪 开始测试ZXM_VOLATILITY_FORECAST指标...")
    
    try:
        # 创建指标实例
        indicator = ZXMVolatilityForecast()
        logger.info(f"✅ 指标实例创建成功: {indicator.name}")
        
        # 创建测试数据
        test_data = create_test_data(100)
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 测试calculate方法
        logger.info("🔍 测试calculate方法...")
        result = indicator.calculate(test_data)
        
        if result:
            logger.info("✅ calculate方法执行成功")
            logger.info(f"📊 返回结果键: {list(result.keys())}")
            
            # 检查关键指标
            expected_keys = [
                'volatility_5d', 'volatility_10d', 'volatility_20d',
                'ewma_volatility', 'forecast_volatility', 'volatility_trend',
                'trend_strength', 'volatility_percentile', 'risk_level'
            ]
            
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                logger.warning(f"⚠️ 缺少预期的键: {missing_keys}")
            else:
                logger.info("✅ 所有预期的键都存在")
            
            # 显示部分结果
            for key, value in list(result.items())[:5]:
                logger.info(f"  {key}: {value}")
                
        else:
            logger.error("❌ calculate方法返回空结果")
            return False
        
        # 测试get_patterns方法
        logger.info("🔍 测试get_patterns方法...")
        patterns = indicator.get_patterns()
        
        if patterns:
            logger.info("✅ get_patterns方法执行成功")
            logger.info(f"📊 形态信息键: {list(patterns.keys())}")
        else:
            logger.error("❌ get_patterns方法返回空结果")
            return False
        
        # 测试minimum_periods属性
        logger.info("🔍 测试minimum_periods属性...")
        min_periods = indicator.minimum_periods
        logger.info(f"✅ minimum_periods: {min_periods}")
        
        # 测试边界情况
        logger.info("🔍 测试边界情况...")
        
        # 测试空数据
        empty_result = indicator.calculate(pd.DataFrame())
        if empty_result == {}:
            logger.info("✅ 空数据处理正确")
        else:
            logger.warning("⚠️ 空数据处理可能有问题")
        
        # 测试少量数据
        small_data = create_test_data(10)
        small_result = indicator.calculate(small_data)
        if small_result:
            logger.info("✅ 少量数据处理正确")
        else:
            logger.warning("⚠️ 少量数据处理可能有问题")
        
        logger.info("🎉 ZXM_VOLATILITY_FORECAST指标测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def analyze_validation_failure():
    """分析验证失败的原因"""
    logger.info("🔍 分析ZXM_VOLATILITY_FORECAST验证失败原因...")
    
    # 检查验证报告
    report_path = Path(root_dir) / "docs/finaltesting/indicators/all_zxm_indicators_95_validation_report.md"
    
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找ZXM_VOLATILITY_FORECAST相关内容
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'ZXM_VOLATILITY_FORECAST' in line:
                logger.info(f"📄 验证报告第{i+1}行: {line}")
                # 显示前后几行上下文
                start = max(0, i-2)
                end = min(len(lines), i+3)
                for j in range(start, end):
                    if j == i:
                        logger.info(f"  >>> {lines[j]}")
                    else:
                        logger.info(f"      {lines[j]}")
                break
    else:
        logger.warning("⚠️ 验证报告文件不存在")


def main():
    """主函数"""
    logger.info("🚀 开始ZXM_VOLATILITY_FORECAST指标诊断...")
    
    # 分析验证失败原因
    analyze_validation_failure()
    
    # 运行测试
    success = test_zxm_volatility_forecast()
    
    if success:
        logger.info("✅ 指标测试通过，可能需要检查验证标准")
    else:
        logger.error("❌ 指标测试失败，需要修复代码")
    
    return success


if __name__ == "__main__":
    main()
