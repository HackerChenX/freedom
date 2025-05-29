#!/usr/bin/env python3
"""
测试统一评分系统

验证所有指标的评分功能是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.macd import MACD
from indicators.kdj import KDJ
from indicators.rsi import RSI
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger

logger = get_logger(__name__)


def test_unified_scoring():
    """测试统一评分系统"""
    logger.info("开始测试统一评分系统")
    
    # 获取测试数据
    db = get_clickhouse_db()
    sql = """
    SELECT date, open, high, low, close, volume
    FROM stock_info 
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2024-01-01'
    ORDER BY date
    LIMIT 100
    """
    
    data = db.query(sql)
    if data.empty:
        logger.error("没有获取到测试数据")
        return
    
    # 重新映射列名
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    logger.info(f"获取到 {len(data)} 条数据")
    
    # 测试各个指标的评分功能
    indicators = [
        ("MACD", MACD()),
        ("KDJ", KDJ()),
        ("RSI", RSI())
    ]
    
    for name, indicator in indicators:
        logger.info(f"\n=== 测试 {name} 指标评分 ===")
        
        try:
            # 计算指标
            result = indicator.calculate(data)
            logger.info(f"{name} 指标计算成功，结果列: {result.columns.tolist()}")
            
            # 计算评分
            score_result = indicator.calculate_score(data)
            
            # 输出评分结果
            logger.info(f"{name} 评分结果:")
            logger.info(f"  - 最新原始评分: {score_result['raw_score'].iloc[-1]:.2f}")
            logger.info(f"  - 最新最终评分: {score_result['final_score'].iloc[-1]:.2f}")
            logger.info(f"  - 市场环境: {score_result['market_environment'].value}")
            logger.info(f"  - 识别形态: {score_result['patterns']}")
            logger.info(f"  - 置信度: {score_result['confidence']:.2f}")
            
            # 测试形态识别
            patterns = indicator.identify_patterns(data)
            logger.info(f"  - 形态识别结果: {patterns}")
            
            # 测试信号生成
            signals = indicator.generate_trading_signals(data)
            logger.info(f"  - 生成信号类型: {list(signals.keys())}")
            
            # 显示最近5个周期的评分变化
            recent_scores = score_result['final_score'].tail(5)
            logger.info(f"  - 最近5个周期评分: {recent_scores.values}")
            
        except Exception as e:
            logger.error(f"测试 {name} 指标时出错: {e}")
            import traceback
            traceback.print_exc()


def test_market_environment_detection():
    """测试市场环境检测"""
    logger.info("\n=== 测试市场环境检测 ===")
    
    # 获取更长期的数据
    db = get_clickhouse_db()
    sql = """
    SELECT date, open, high, low, close, volume
    FROM stock_info 
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2023-01-01'
    ORDER BY date
    LIMIT 300
    """
    
    data = db.query(sql)
    if data.empty:
        logger.error("没有获取到测试数据")
        return
    
    # 重新映射列名
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # 创建指标实例
    macd = MACD()
    
    # 测试不同时期的市场环境
    periods = [
        ("2023年前60天", data.head(60)),
        ("2023年中60天", data.iloc[120:180]),
        ("最近60天", data.tail(60))
    ]
    
    for period_name, period_data in periods:
        if len(period_data) >= 60:
            market_env = macd.detect_market_environment(period_data)
            logger.info(f"{period_name} 市场环境: {market_env.value}")
            
            # 计算该时期的价格变化
            price_change = (period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0]
            logger.info(f"  - 价格变化: {price_change:.2%}")
            
            # 计算波动率
            returns = period_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            logger.info(f"  - 年化波动率: {volatility:.2%}")


def test_scoring_consistency():
    """测试评分一致性"""
    logger.info("\n=== 测试评分一致性 ===")
    
    # 获取测试数据
    db = get_clickhouse_db()
    sql = """
    SELECT date, open, high, low, close, volume
    FROM stock_info 
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2024-01-01'
    ORDER BY date
    LIMIT 50
    """
    
    data = db.query(sql)
    if data.empty:
        logger.error("没有获取到测试数据")
        return
    
    # 重新映射列名
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # 测试多次计算的一致性
    macd = MACD()
    
    scores1 = macd.calculate_score(data)
    scores2 = macd.calculate_score(data)
    
    # 检查两次计算结果是否一致
    score_diff = abs(scores1['final_score'].iloc[-1] - scores2['final_score'].iloc[-1])
    logger.info(f"两次计算评分差异: {score_diff:.6f}")
    
    if score_diff < 0.001:
        logger.info("✓ 评分计算一致性测试通过")
    else:
        logger.warning("✗ 评分计算一致性测试失败")


def test_edge_cases():
    """测试边界情况"""
    logger.info("\n=== 测试边界情况 ===")
    
    # 测试数据不足的情况
    small_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'open': [10, 11, 12, 11, 10],
        'high': [11, 12, 13, 12, 11],
        'low': [9, 10, 11, 10, 9],
        'close': [10.5, 11.5, 12.5, 11.5, 10.5],
        'volume': [1000, 1100, 1200, 1100, 1000]
    })
    
    macd = MACD()
    
    try:
        score_result = macd.calculate_score(small_data)
        logger.info(f"小数据集评分: {score_result['final_score'].iloc[-1]:.2f}")
        logger.info("✓ 小数据集处理测试通过")
    except Exception as e:
        logger.error(f"✗ 小数据集处理测试失败: {e}")
    
    # 测试异常数据
    abnormal_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=20),
        'open': [np.nan] * 20,
        'high': [100] * 20,
        'low': [0] * 20,
        'close': [50] * 20,
        'volume': [0] * 20
    })
    
    try:
        score_result = macd.calculate_score(abnormal_data)
        logger.info(f"异常数据评分: {score_result['final_score'].iloc[-1]:.2f}")
        logger.info("✓ 异常数据处理测试通过")
    except Exception as e:
        logger.error(f"✗ 异常数据处理测试失败: {e}")


def main():
    """主函数"""
    logger.info("开始统一评分系统测试")
    
    try:
        test_unified_scoring()
        test_market_environment_detection()
        test_scoring_consistency()
        test_edge_cases()
        
        logger.info("\n=== 测试完成 ===")
        logger.info("所有测试已完成，请查看上述结果")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 