#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MFI指标正式验证脚本
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

from utils.logger import get_logger
from indicators.mfi import Mfi

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
    logger.info("🚀 开始MFI指标综合验证...")
    
    try:
        # 创建指标实例
        indicator = Mfi()
        logger.info(f"✅ 指标实例创建成功: {indicator.name}")
        
        # 创建测试数据
        test_data = create_realistic_test_data(252)  # 一年数据
        logger.info(f"✅ 测试数据创建成功: {len(test_data)} 行")
        
        # 1. 算法正确性验证
        logger.info("🔍 1. 算法正确性验证...")
        result = indicator.calculate(test_data)
        
        if result is None or result.empty:
            logger.error("❌ 算法验证失败：计算结果为空")
            return False
        
        # 检查必需的输出
        required_outputs = ['mfi', 'MFI14', 'mfi_signal', 'mfi_volatility']
        
        missing_outputs = [key for key in required_outputs if key not in result.columns]
        if missing_outputs:
            logger.error(f"❌ 算法验证失败：缺少输出列 {missing_outputs}")
            return False
        
        # 检查MFI值的合理性
        mfi_values = result['mfi'].dropna()
        if len(mfi_values) == 0:
            logger.error("❌ 算法验证失败：MFI值全为NaN")
            return False
        
        # MFI值应该在0-100范围内
        if not all(0 <= val <= 100 for val in mfi_values):
            logger.error("❌ 算法验证失败：MFI值超出0-100范围")
            return False
        
        logger.info("✅ 算法正确性验证通过")
        
        # 2. 数值合理性验证
        logger.info("🔍 2. 数值合理性验证...")
        
        # 验证MFI计算的准确性
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
                if correlation < 0.99:
                    logger.error(f"❌ 数值验证失败：MFI计算相关性过低 {correlation:.4f}")
                    return False
                
                logger.info(f"✅ MFI计算相关性: {correlation:.4f}")
        
        logger.info("✅ 数值合理性验证通过")
        
        # 3. 功能完整性验证
        logger.info("🔍 3. 功能完整性验证...")
        
        # 测试形态识别
        patterns = indicator.get_patterns(test_data)
        if patterns is None or patterns.empty:
            logger.error("❌ 功能验证失败：形态识别返回空结果")
            return False
        
        # 检查形态列数量
        expected_patterns = ['overbought', 'oversold', 'divergence_bullish', 'divergence_bearish']
        missing_patterns = [p for p in expected_patterns if p not in patterns.columns]
        if missing_patterns:
            logger.error(f"❌ 功能验证失败：缺少形态 {missing_patterns}")
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
        
        if avg_time > 0.1:  # 单次计算不应超过0.1秒
            logger.error(f"❌ 性能验证失败：平均计算时间 {avg_time:.3f}秒 超过0.1秒阈值")
            return False
        
        logger.info(f"✅ 性能验证通过：平均计算时间 {avg_time:.3f}秒")
        
        # 5. 稳定性验证
        logger.info("🔍 5. 稳定性验证...")
        
        # 测试不同数据量
        test_sizes = [20, 30, 60, 120, 252]  # MFI需要至少period+1个数据点
        for size in test_sizes:
            test_subset = test_data.head(size)
            subset_result = indicator.calculate(test_subset)
            
            if subset_result is None or subset_result.empty:
                logger.error(f"❌ 稳定性验证失败：{size}行数据计算失败")
                return False
        
        # 测试边界情况
        empty_result = indicator.calculate(pd.DataFrame())
        if not isinstance(empty_result, pd.DataFrame):
            logger.error("❌ 稳定性验证失败：空数据处理异常")
            return False
        
        logger.info("✅ 稳定性验证通过")
        
        # 计算总分
        total_score = 100.0  # 所有验证都通过
        
        logger.info("=" * 60)
        logger.info("🎉 MFI指标验证完成")
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
        logger.info("🎉 MFI指标验证成功！")
        return True
    else:
        logger.error("❌ MFI指标验证失败！")
        return False


if __name__ == "__main__":
    main()
