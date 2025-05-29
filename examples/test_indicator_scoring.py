"""
测试指标评分系统

展示如何使用统一的指标评分框架
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.scoring_framework import IndicatorScoreManager
from indicators.macd_score import MACDScore
from indicators.kdj_score import KDJScore
from indicators.rsi_score import RSIScore
from indicators.boll_score import BOLLScore
from indicators.volume_score import VolumeScore
from indicators.indicator_registry import indicator_registry
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger

logger = get_logger(__name__)


def test_single_indicator_scoring():
    """测试单个指标评分"""
    logger.info("开始测试单个指标评分")
    
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
    
    # 测试MACD评分
    logger.info("测试MACD评分指标")
    macd_scorer = MACDScore(weight=1.5)
    macd_result = macd_scorer.calculate_final_score(data)
    
    logger.info(f"MACD评分结果:")
    logger.info(f"- 平均评分: {macd_result['final_score'].mean():.2f}")
    logger.info(f"- 最高评分: {macd_result['final_score'].max():.2f}")
    logger.info(f"- 最低评分: {macd_result['final_score'].min():.2f}")
    logger.info(f"- 市场环境: {macd_result['market_environment']}")
    logger.info(f"- 识别形态: {macd_result['patterns']}")
    logger.info(f"- 信号数量: {len(macd_result['signals'])}")
    
    # 测试KDJ评分
    logger.info("测试KDJ评分指标")
    kdj_scorer = KDJScore(weight=1.2)
    kdj_result = kdj_scorer.calculate_final_score(data)
    
    logger.info(f"KDJ评分结果:")
    logger.info(f"- 平均评分: {kdj_result['final_score'].mean():.2f}")
    logger.info(f"- 最高评分: {kdj_result['final_score'].max():.2f}")
    logger.info(f"- 最低评分: {kdj_result['final_score'].min():.2f}")
    logger.info(f"- 市场环境: {kdj_result['market_environment']}")
    logger.info(f"- 识别形态: {kdj_result['patterns']}")
    logger.info(f"- 信号数量: {len(kdj_result['signals'])}")


def test_comprehensive_scoring():
    """测试综合评分系统"""
    logger.info("开始测试综合评分系统")
    
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
    
    # 创建评分管理器
    score_manager = IndicatorScoreManager()
    
    # 注册指标
    macd_scorer = MACDScore()
    kdj_scorer = KDJScore()
    rsi_scorer = RSIScore()
    boll_scorer = BOLLScore()
    volume_scorer = VolumeScore()
    
    score_manager.register_indicator(macd_scorer, weight=1.5)  # MACD权重1.5
    score_manager.register_indicator(kdj_scorer, weight=1.2)   # KDJ权重1.2
    score_manager.register_indicator(rsi_scorer, weight=1.3)   # RSI权重1.3
    score_manager.register_indicator(boll_scorer, weight=1.0)  # BOLL权重1.0
    score_manager.register_indicator(volume_scorer, weight=1.1) # Volume权重1.1
    
    # 计算综合评分
    comprehensive_result = score_manager.calculate_comprehensive_score(data)
    
    logger.info(f"综合评分结果:")
    logger.info(f"- 平均综合评分: {comprehensive_result['comprehensive_score'].mean():.2f}")
    logger.info(f"- 最高综合评分: {comprehensive_result['comprehensive_score'].max():.2f}")
    logger.info(f"- 最低综合评分: {comprehensive_result['comprehensive_score'].min():.2f}")
    logger.info(f"- 总权重: {comprehensive_result['total_weight']}")
    logger.info(f"- 综合形态数量: {len(comprehensive_result['patterns'])}")
    logger.info(f"- 综合信号数量: {len(comprehensive_result['signals'])}")
    
    # 显示各指标的详细评分
    logger.info("各指标详细评分:")
    for indicator_name, result in comprehensive_result['indicator_results'].items():
        avg_score = result['final_score'].mean()
        patterns = result['patterns']
        logger.info(f"- {indicator_name}: 平均评分={avg_score:.2f}, 形态={patterns}")
    
    # 显示最近5天的详细评分
    logger.info("最近5天的详细评分:")
    recent_data = data.tail(5)
    recent_score = comprehensive_result['comprehensive_score'].tail(5)
    recent_signals = comprehensive_result['comprehensive_signal']
    
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        score = recent_score.iloc[i]
        strong_buy = recent_signals['strong_buy'].iloc[idx] if idx < len(recent_signals['strong_buy']) else False
        buy = recent_signals['buy'].iloc[idx] if idx < len(recent_signals['buy']) else False
        hold = recent_signals['hold'].iloc[idx] if idx < len(recent_signals['hold']) else False
        sell = recent_signals['sell'].iloc[idx] if idx < len(recent_signals['sell']) else False
        strong_sell = recent_signals['strong_sell'].iloc[idx] if idx < len(recent_signals['strong_sell']) else False
        
        signal_text = ""
        if strong_buy:
            signal_text = "强烈买入"
        elif buy:
            signal_text = "买入"
        elif hold:
            signal_text = "持有"
        elif sell:
            signal_text = "卖出"
        elif strong_sell:
            signal_text = "强烈卖出"
        else:
            signal_text = "无信号"
        
        logger.info(f"  {row['date']}: 评分={score:.2f}, 信号={signal_text}, 收盘价={row['close']:.2f}")


def test_indicator_registry():
    """测试指标注册机制"""
    logger.info("开始测试指标注册机制")
    
    # 获取可用的评分指标
    available_indicators = indicator_registry.get_available_scoring_indicators()
    logger.info(f"可用的评分指标: {available_indicators}")
    
    # 使用注册机制创建评分管理器
    indicator_configs = [
        {'name': 'macd_score', 'weight': 1.5, 'fast_period': 12, 'slow_period': 26},
        {'name': 'kdj_score', 'weight': 1.2, 'n': 9},
        {'name': 'rsi_score', 'weight': 1.3, 'period': 14},
        {'name': 'boll_score', 'weight': 1.0, 'period': 20, 'std_dev': 2.0},
        {'name': 'volume_score', 'weight': 1.1, 'period': 20}
    ]
    
    try:
        score_manager = indicator_registry.create_score_manager(indicator_configs)
        logger.info("成功通过注册机制创建评分管理器")
        
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
        if not data.empty:
            # 重新映射列名
            data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # 计算综合评分
            result = score_manager.calculate_comprehensive_score(data)
            logger.info(f"通过注册机制计算的综合评分平均值: {result['comprehensive_score'].mean():.2f}")
        
    except Exception as e:
        logger.error(f"测试指标注册机制失败: {e}")


def test_pattern_recognition():
    """测试形态识别功能"""
    logger.info("开始测试形态识别功能")
    
    # 获取更多测试数据用于形态识别
    db = get_clickhouse_db()
    sql = """
    SELECT date, open, high, low, close, volume
    FROM stock_info 
    WHERE code = '000001'
    AND level = '日线'
    AND date >= '2023-01-01'
    ORDER BY date
    LIMIT 500
    """
    
    data = db.query(sql)
    if data.empty:
        logger.error("没有获取到测试数据")
        return
    
    # 重新映射列名
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    logger.info(f"获取到 {len(data)} 条数据用于形态识别")
    
    # 测试所有指标的形态识别
    indicators = [
        ("MACD", MACDScore()),
        ("KDJ", KDJScore()),
        ("RSI", RSIScore()),
        ("BOLL", BOLLScore()),
        ("Volume", VolumeScore())
    ]
    
    for indicator_name, indicator in indicators:
        logger.info(f"\n{indicator_name}形态识别结果:")
        
        try:
            patterns = indicator.identify_patterns(data)
            signals = indicator.generate_signals(data)
            
            logger.info(f"- 识别形态: {patterns}")
            
            # 统计信号出现次数
            for signal_name, signal_series in signals.items():
                count = signal_series.sum() if hasattr(signal_series, 'sum') else 0
                logger.info(f"- {signal_name}: {count} 次")
                
        except Exception as e:
            logger.error(f"测试 {indicator_name} 形态识别失败: {e}")


def main():
    """主函数"""
    logger.info("开始测试指标评分系统")
    
    try:
        # 测试单个指标评分
        test_single_indicator_scoring()
        
        print("\n" + "="*50 + "\n")
        
        # 测试综合评分
        test_comprehensive_scoring()
        
        print("\n" + "="*50 + "\n")
        
        # 测试指标注册机制
        test_indicator_registry()
        
        print("\n" + "="*50 + "\n")
        
        # 测试形态识别
        test_pattern_recognition()
        
        logger.info("指标评分系统测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 