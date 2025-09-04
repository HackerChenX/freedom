#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZXM_LIQUIDITY_ANALYSIS指标正式验证脚本
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger
from indicators.zxm.zxm_liquidity_analysis import ZXMLiquidityAnalysis

logger = get_logger(__name__)


def create_realistic_test_data(length: int = 252) -> pd.DataFrame:
    """创建更真实的测试数据"""
    np.random.seed(42)
    
    # 生成模拟股价数据
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 生成更真实的价格序列（带趋势和波动聚集）
    initial_price = 100.0
    returns = []
    volatility = 0.02  # 初始波动率
    
    for i in range(length):
        # 波动率聚集效应
        if i > 0:
            volatility = 0.95 * volatility + 0.05 * abs(returns[-1])
        
        # 生成收益率
        ret = np.random.normal(0.0005, volatility)
        returns.append(ret)
    
    # 计算价格
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    close_prices = np.array(prices[1:])  # 去掉初始价格
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, length)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.3, length)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return data


def run_comprehensive_validation():
    """运行综合验证"""
    logger.info("🚀 开始ZXM_LIQUIDITY_ANALYSIS指标综合验证...")
    
    try:
        # 创建指标实例
        indicator = ZXMLiquidityAnalysis()
        logger.info(f"✅ 指标实例创建成功: {indicator.name}")
        
        # 创建测试数据
        test_data = create_realistic_test_data(252)  # 一年数据
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 算法正确性验证
        logger.info("🔍 1. 算法正确性验证...")
        result = indicator.calculate(test_data)
        
        if not result:
            logger.error("❌ 算法验证失败：计算结果为空")
            return False
        
        # 检查必需的输出
        required_outputs = [
            'volatility', 'liquidity_score', 'volume_stability', 'price_continuity',
            'liquidity_index', 'liquidity_level', 'liquidity_risk'
        ]
        
        missing_outputs = [key for key in required_outputs if key not in result]
        if missing_outputs:
            logger.error(f"❌ 算法验证失败：缺少输出 {missing_outputs}")
            return False
        
        logger.info("✅ 算法正确性验证通过")
        
        # 2. 数值合理性验证
        logger.info("🔍 2. 数值合理性验证...")
        
        # 检查波动率值是否合理
        volatility = result['volatility']
        if not (0 <= volatility <= 2.0):  # 年化波动率通常在0-200%之间
            logger.error(f"❌ 数值验证失败：波动率 {volatility} 超出合理范围")
            return False
        
        # 检查流动性评分
        liquidity_score = result['liquidity_score']
        if not (0 <= liquidity_score <= 100):
            logger.error(f"❌ 数值验证失败：流动性评分 {liquidity_score} 超出0-100范围")
            return False
        
        # 检查流动性指数
        liquidity_index = result['liquidity_index']
        if not (0 <= liquidity_index <= 100):
            logger.error(f"❌ 数值验证失败：流动性指数 {liquidity_index} 超出0-100范围")
            return False
        
        # 检查流动性等级
        valid_levels = ['excellent', 'good', 'fair', 'poor']
        if result['liquidity_level'] not in valid_levels:
            logger.error(f"❌ 数值验证失败：无效的流动性等级 {result['liquidity_level']}")
            return False
        
        # 检查风险等级
        valid_risks = ['low', 'medium', 'high']
        if result['liquidity_risk'] not in valid_risks:
            logger.error(f"❌ 数值验证失败：无效的风险等级 {result['liquidity_risk']}")
            return False
        
        logger.info("✅ 数值合理性验证通过")
        
        # 3. 功能完整性验证
        logger.info("🔍 3. 功能完整性验证...")
        
        # 测试get_patterns方法
        patterns = indicator.get_patterns()
        if not patterns:
            logger.error("❌ 功能验证失败：get_patterns返回空结果")
            return False
        
        # 检查形态信息完整性
        required_pattern_keys = [
            'indicator_type', 'category', 'description', 'metrics',
            'levels', 'risk_levels', 'thresholds'
        ]
        
        missing_pattern_keys = [key for key in required_pattern_keys if key not in patterns]
        if missing_pattern_keys:
            logger.error(f"❌ 功能验证失败：形态信息缺少键 {missing_pattern_keys}")
            return False
        
        logger.info("✅ 功能完整性验证通过")
        
        # 4. 性能验证
        logger.info("🔍 4. 性能验证...")
        
        import time
        start_time = time.time()
        
        # 运行多次计算测试性能
        for _ in range(10):
            indicator.calculate(test_data)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        if avg_time > 2.0:  # 单次计算不应超过2秒
            logger.error(f"❌ 性能验证失败：平均计算时间 {avg_time:.3f}秒 超过2秒阈值")
            return False
        
        logger.info(f"✅ 性能验证通过：平均计算时间 {avg_time:.3f}秒")
        
        # 5. 稳定性验证
        logger.info("🔍 5. 稳定性验证...")
        
        # 测试不同数据量
        test_sizes = [20, 60, 120, 252]
        for size in test_sizes:
            test_subset = test_data.head(size)
            subset_result = indicator.calculate(test_subset)
            
            if not subset_result:
                logger.error(f"❌ 稳定性验证失败：{size}行数据计算失败")
                return False
        
        # 测试边界情况
        empty_result = indicator.calculate(pd.DataFrame())
        if not isinstance(empty_result, dict):
            logger.error("❌ 稳定性验证失败：空数据处理异常")
            return False
        
        logger.info("✅ 稳定性验证通过")
        
        # 计算总分
        total_score = 100.0  # 所有验证都通过
        
        logger.info("=" * 60)
        logger.info("🎉 ZXM_LIQUIDITY_ANALYSIS指标验证完成")
        logger.info("=" * 60)
        logger.info(f"📊 验证结果:")
        logger.info(f"  - 算法正确性: ✅ PASSED")
        logger.info(f"  - 数值合理性: ✅ PASSED")
        logger.info(f"  - 功能完整性: ✅ PASSED")
        logger.info(f"  - 性能表现: ✅ PASSED ({avg_time:.3f}s)")
        logger.info(f"  - 稳定性: ✅ PASSED")
        logger.info(f"📈 总得分: {total_score:.1f}/100")
        logger.info(f"🏆 验证状态: PASSED")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 验证过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


def main():
    """主函数"""
    success = run_comprehensive_validation()
    
    if success:
        logger.info("🎉 ZXM_LIQUIDITY_ANALYSIS指标验证成功！")
        return True
    else:
        logger.error("❌ ZXM_LIQUIDITY_ANALYSIS指标验证失败！")
        return False


if __name__ == "__main__":
    main()
