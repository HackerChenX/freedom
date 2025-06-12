"""
测试增强型震荡指标的脚本

用于验证增强型震荡指标的功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.enhanced_rsi import EnhancedRSI
from indicators.rsi import RSI
from indicators.oscillator.enhanced_kdj import EnhancedKDJ
from indicators.kdj import KDJ
from indicators.enhanced_factory import EnhancedIndicatorFactory
from indicators.market_env import MarketDetector
from indicators.base_indicator import MarketEnvironment

from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger

logger = get_logger(__name__)


def get_test_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取测试数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        pd.DataFrame: 测试数据
    """
    try:
        # 从数据库获取数据
        db = get_clickhouse_db()
        query = f"""
        SELECT 
            trade_date, 
            open, 
            high, 
            low, 
            close, 
            volume
        FROM 
            stock_daily_price
        WHERE 
            ts_code = '{stock_code}' AND 
            trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY 
            trade_date
        """
        
        data = db.query_dataframe(query)
        
        # 确保日期列是日期类型
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        
        # 设置日期为索引
        data.set_index('trade_date', inplace=True)
        
        return data
        
    except Exception as e:
        logger.error(f"获取测试数据失败: {e}")
        
        # 如果数据库查询失败，使用模拟数据
        logger.warning("使用模拟数据进行测试")
        
        # 创建日期范围
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # 创建模拟数据
        np.random.seed(42)  # 确保结果可重现
        n = len(date_range)
        
        # 创建模拟价格
        close = np.cumsum(np.random.normal(0, 1, n)) + 100
        high = close + np.random.uniform(0, 2, n)
        low = close - np.random.uniform(0, 2, n)
        open_price = low + np.random.uniform(0, high - low, n)
        volume = np.random.uniform(1000, 10000, n)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=date_range)
        
        return data


def test_enhanced_rsi():
    """测试增强型RSI指标"""
    logger.info("开始测试增强型RSI指标")
    
    # 获取测试数据
    data = get_test_data('000001.SZ', '20220101', '20221231')
    
    if data.empty:
        logger.error("获取测试数据失败")
        return
    
    logger.info(f"获取到 {len(data)} 条测试数据")
    
    # 创建标准RSI和增强型RSI
    std_rsi = RSI(period=14)
    enhanced_rsi = EnhancedRSI(period=14, multi_periods=[9, 14, 21])
    print(f"DEBUG: Indicator type is {enhanced_rsi.get_indicator_type()}")
    
    # 检测市场环境
    market_detector = MarketDetector()
    market_env = market_detector.detect_environment(data)
    enhanced_rsi.set_market_environment(market_env)
    
    logger.info(f"检测到市场环境: {market_env.value}")
    
    # 计算指标
    std_result = std_rsi.calculate(data)
    enhanced_result = enhanced_rsi.calculate(data)
    
    logger.info(f"标准RSI结果: {std_result.shape}, 增强型RSI结果: {enhanced_result.shape}")
    
    # 生成信号
    std_signals = std_rsi.generate_signals(data)
    enhanced_signals = enhanced_rsi.generate_signals(data)
    
    logger.info(f"标准RSI信号: {std_signals.shape}, 增强型RSI信号: {enhanced_signals.shape}")
    
    # 计算评分
    enhanced_score = enhanced_rsi.calculate_raw_score(data)
    
    # 计算信号统计
    std_buy_count = std_signals['buy_signal'].sum()
    std_sell_count = std_signals['sell_signal'].sum()
    
    enhanced_buy_count = enhanced_signals['buy_signal'].sum()
    enhanced_sell_count = enhanced_signals['sell_signal'].sum()
    
    logger.info(f"标准RSI买入信号: {std_buy_count}, 卖出信号: {std_sell_count}")
    logger.info(f"增强型RSI买入信号: {enhanced_buy_count}, 卖出信号: {enhanced_sell_count}")
    
    # 识别形态
    patterns = enhanced_rsi.identify_patterns(data)
    
    logger.info(f"识别到的形态: {patterns}")
    
    # 对比信号质量
    compare_signal_quality(data, std_signals, enhanced_signals)
    
    logger.info("增强型RSI测试完成")


def test_enhanced_kdj():
    """测试增强型KDJ指标"""
    logger.info("开始测试增强型KDJ指标")
    
    # 获取测试数据
    data = get_test_data('000001.SZ', '20220101', '20221231')
    
    if data.empty:
        logger.error("获取测试数据失败")
        return
    
    logger.info(f"获取到 {len(data)} 条测试数据")
    
    # 创建标准KDJ和增强型KDJ
    std_kdj = KDJ(n=9, m1=3, m2=3)
    enhanced_kdj = EnhancedKDJ(n=9, m1=3, m2=3, multi_periods=[5, 9, 14])
    
    # 检测市场环境
    market_detector = MarketDetector()
    market_env = market_detector.detect_environment(data)
    enhanced_kdj.set_market_environment(market_env)
    
    logger.info(f"检测到市场环境: {market_env.value}")
    
    # 计算指标
    std_result = std_kdj.calculate(data)
    enhanced_result = enhanced_kdj.calculate(data)
    
    logger.info(f"标准KDJ结果: {std_result.shape}, 增强型KDJ结果: {enhanced_result.shape}")
    
    # 生成信号
    std_signals = std_kdj.generate_signals(data)
    enhanced_signals = enhanced_kdj.generate_signals(data)
    
    logger.info(f"标准KDJ信号: {std_signals.shape}, 增强型KDJ信号: {enhanced_signals.shape}")
    
    # 计算评分
    enhanced_score = enhanced_kdj.calculate_raw_score(data)
    
    # 计算信号统计
    std_buy_count = std_signals['buy_signal'].sum()
    std_sell_count = std_signals['sell_signal'].sum()
    
    enhanced_buy_count = enhanced_signals['buy_signal'].sum()
    enhanced_sell_count = enhanced_signals['sell_signal'].sum()
    
    logger.info(f"标准KDJ买入信号: {std_buy_count}, 卖出信号: {std_sell_count}")
    logger.info(f"增强型KDJ买入信号: {enhanced_buy_count}, 卖出信号: {enhanced_sell_count}")
    
    # 识别形态
    patterns = enhanced_kdj.identify_patterns(data)
    
    logger.info(f"识别到的形态: {patterns}")
    
    # 分析高级特性
    analyze_kdj_advanced_features(enhanced_result)
    
    logger.info("增强型KDJ测试完成")


def analyze_kdj_advanced_features(result):
    """
    分析KDJ的高级特性
    
    Args:
        result: KDJ计算结果
    """
    # 分析J线加速度
    j_accel = result["j_acceleration"]
    logger.info(f"J线加速度统计: 均值={j_accel.mean():.4f}, 标准差={j_accel.std():.4f}")
    logger.info(f"J线加速度最大值: {j_accel.max():.4f}, 最小值: {j_accel.min():.4f}")
    
    # 分析KD交叉角度
    kd_angle = result["kd_cross_angle"]
    logger.info(f"KD交叉角度统计: 均值={kd_angle.mean():.4f}, 标准差={kd_angle.std():.4f}")
    logger.info(f"KD交叉角度最大值: {kd_angle.max():.4f}, 最小值: {kd_angle.min():.4f}")
    
    # 分析KD线距离
    kd_distance = result["kd_distance"]
    logger.info(f"KD距离统计: 均值={kd_distance.mean():.4f}, 标准差={kd_distance.std():.4f}")
    logger.info(f"KD距离最大值: {kd_distance.max():.4f}, 最小值: {kd_distance.min():.4f}")
    
    # 分析多周期KDJ
    k_columns = [col for col in result.columns if col.startswith("k_")]
    d_columns = [col for col in result.columns if col.startswith("d_")]
    j_columns = [col for col in result.columns if col.startswith("j_") and col != "j_acceleration" and col != "j_normalized"]
    
    logger.info(f"多周期K线: {k_columns}")
    logger.info(f"多周期D线: {d_columns}")
    logger.info(f"多周期J线: {j_columns}")


def compare_signal_quality(data, std_signals, enhanced_signals):
    """
    对比标准信号和增强信号质量
    
    Args:
        data: 价格数据
        std_signals: 标准信号
        enhanced_signals: 增强信号
    """
    # 计算标准信号的前瞻性能
    std_forward_returns = calculate_forward_returns(data, std_signals)
    
    # 计算增强信号的前瞻性能
    enhanced_forward_returns = calculate_forward_returns(data, enhanced_signals)
    
    logger.info(f"标准信号前瞻收益: {std_forward_returns}")
    logger.info(f"增强信号前瞻收益: {enhanced_forward_returns}")
    
    # 计算信号一致性
    consistency = calculate_signal_consistency(std_signals, enhanced_signals)
    
    logger.info(f"信号一致性: {consistency}")


def calculate_forward_returns(data, signals, periods=[5, 10, 20]):
    """
    计算信号的前瞻收益
    
    Args:
        data: 价格数据
        signals: 信号数据
        periods: 前瞻周期列表
        
    Returns:
        dict: 各类信号的前瞻收益
    """
    results = {"buy": {}, "sell": {}}
    
    for period in periods:
        # 计算未来收益率
        future_returns = data['close'].pct_change(period).shift(-period)
        
        # 买入信号的未来收益
        if signals['buy_signal'].any():
            buy_returns = future_returns[signals['buy_signal']]
            results["buy"][period] = buy_returns.mean()
        else:
            results["buy"][period] = np.nan
        
        # 卖出信号的未来收益（应为负值才是好的）
        if signals['sell_signal'].any():
            sell_returns = future_returns[signals['sell_signal']]
            results["sell"][period] = sell_returns.mean()
        else:
            results["sell"][period] = np.nan
    
    return results


def calculate_signal_consistency(std_signals, enhanced_signals):
    """
    计算两组信号的一致性
    
    Args:
        std_signals: 标准信号
        enhanced_signals: 增强信号
        
    Returns:
        dict: 信号一致性统计
    """
    # 计算买入信号一致性
    buy_agreement = (std_signals['buy_signal'] & enhanced_signals['buy_signal']).sum()
    std_only_buy = (std_signals['buy_signal'] & ~enhanced_signals['buy_signal']).sum()
    enhanced_only_buy = (~std_signals['buy_signal'] & enhanced_signals['buy_signal']).sum()
    
    # 计算卖出信号一致性
    sell_agreement = (std_signals['sell_signal'] & enhanced_signals['sell_signal']).sum()
    std_only_sell = (std_signals['sell_signal'] & ~enhanced_signals['sell_signal']).sum()
    enhanced_only_sell = (~std_signals['sell_signal'] & enhanced_signals['sell_signal']).sum()
    
    return {
        "buy": {
            "agreement": buy_agreement,
            "std_only": std_only_buy,
            "enhanced_only": enhanced_only_buy,
            "agreement_rate": buy_agreement / (buy_agreement + std_only_buy + enhanced_only_buy) if (buy_agreement + std_only_buy + enhanced_only_buy) > 0 else 0
        },
        "sell": {
            "agreement": sell_agreement,
            "std_only": std_only_sell,
            "enhanced_only": enhanced_only_sell,
            "agreement_rate": sell_agreement / (sell_agreement + std_only_sell + enhanced_only_sell) if (sell_agreement + std_only_sell + enhanced_only_sell) > 0 else 0
        }
    }


if __name__ == "__main__":
    """
    测试增强型震荡指标
    
    使用命令行参数指定要测试的指标:
    --rsi: 测试增强型RSI
    --kdj: 测试增强型KDJ
    --cci: 测试增强型CCI
    --all: 测试所有增强型震荡指标
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='测试增强型震荡指标')
    parser.add_argument('--rsi', action='store_true', help='测试增强型RSI')
    parser.add_argument('--kdj', action='store_true', help='测试增强型KDJ')
    parser.add_argument('--cci', action='store_true', help='测试增强型CCI')
    parser.add_argument('--all', action='store_true', help='测试所有增强型震荡指标')
    
    args = parser.parse_args()
    
    if args.all or args.rsi:
        test_enhanced_rsi()
    
    if args.all or args.kdj:
        test_enhanced_kdj()
    
    if args.all or args.cci:
        # 暂未实现
        logger.info("增强型CCI测试尚未实现")
    
    # 如果没有指定任何参数，默认测试RSI和KDJ
    if not (args.rsi or args.kdj or args.cci or args.all):
        test_enhanced_rsi()
        test_enhanced_kdj() 