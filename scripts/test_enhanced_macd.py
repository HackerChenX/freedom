"""
测试增强型MACD指标的脚本

用于验证增强型MACD指标的功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from indicators.macd import MACD
from db.clickhouse_db import get_clickhouse_db
from utils.logger import get_logger
from indicators.base_indicator import MarketEnvironment

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
            volume as vol
        FROM stock_kline_day
        WHERE code = '{stock_code}'
        AND trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date
        """
        data = db.query(query)
        
        # 设置日期索引
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        data.set_index('trade_date', inplace=True)
        
        return data
    except Exception as e:
        logger.error(f"获取测试数据失败: {e}")
        # 如果无法从数据库获取数据，创建模拟数据
        return generate_mock_data(start_date, end_date)


def generate_mock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    生成模拟数据用于测试
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        pd.DataFrame: 模拟数据
    """
    logger.info("使用模拟数据进行测试")
    
    # 转换日期
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 计算天数
    days = (end_date - start_date).days + 1
    
    # 创建日期范围
    date_range = [start_date + timedelta(days=i) for i in range(days)]
    
    # 生成模拟价格数据
    np.random.seed(42)  # 固定随机种子，便于复现
    
    # 初始价格
    initial_price = 100.0
    
    # 生成价格序列
    prices = [initial_price]
    for i in range(1, days):
        # 生成每日的随机价格变动（均值为0，标准差为1）
        change = np.random.normal(0, 1)
        
        # 加入趋势成分
        if i < days // 3:
            # 前1/3时间段为上升趋势
            trend = 0.1
        elif i < 2 * days // 3:
            # 中间1/3时间段为下降趋势
            trend = -0.15
        else:
            # 后1/3时间段为震荡趋势
            trend = 0.05
        
        # 计算新价格
        new_price = prices[-1] * (1 + change * 0.01 + trend * 0.01)
        # 确保价格为正
        new_price = max(new_price, 1.0)
        prices.append(new_price)
    
    # 创建OHLC数据
    data = pd.DataFrame(index=date_range)
    data['close'] = prices
    
    # 模拟开盘价、最高价、最低价
    data['open'] = data['close'].shift(1).fillna(data['close'])
    data['high'] = data['close'] * (1 + np.random.uniform(0, 0.02, size=days))
    data['low'] = data['close'] * (1 - np.random.uniform(0, 0.02, size=days))
    
    # 确保high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    # 模拟成交量
    data['vol'] = np.random.lognormal(10, 1, size=days)
    
    return data


def test_enhanced_macd():
    """
    测试增强型MACD指标
    """
    logger.info("开始测试增强型MACD指标...")
    
    # 获取测试数据（使用某一段时间的上证指数数据）
    # data = get_test_data("000001.SH", "2022-01-01", "2022-12-31")
    # 如果无法从数据库获取，使用模拟数据
    data = generate_mock_data("2022-01-01", "2022-12-31")
    
    logger.info(f"获取到{len(data)}条测试数据")
    
    # 1. 测试默认参数的MACD
    macd = MACD()
    macd_result = macd.calculate(data)
    
    # 3. 测试不同市场环境下的MACD
    # 3.1 牛市环境
    bull_macd = MACD()
    bull_macd.set_market_environment(MarketEnvironment.BULL_MARKET)
    bull_result = bull_macd.calculate(data)
    
    # 3.2 熊市环境
    bear_macd = MACD()
    bear_macd.set_market_environment(MarketEnvironment.BEAR_MARKET)
    bear_result = bear_macd.calculate(data)
    
    # 4. 测试信号生成
    signals = macd.get_signals(data)
    
    # 5. 测试评分计算
    score = macd.calculate_raw_score(data)
    
    # 6. 测试形态识别
    patterns = macd.get_patterns(data)
    
    # 打印测试结果摘要
    logger.info("MACD测试结果摘要:")
    logger.info(f"标准MACD计算结果: macd/signal/hist均值 = {macd_result['macd'].mean():.2f} / {macd_result['signal'].mean():.2f} / {macd_result['hist'].mean():.2f}")
    
    # 统计信号数量
    buy_signals = signals['buy_signal'].sum()
    sell_signals = signals['sell_signal'].sum()
    logger.info(f"信号统计: {buy_signals}个买入信号, {sell_signals}个卖出信号")
    
    # 分析评分分布
    score_stats = score.describe()
    logger.info(f"评分统计: 均值={score_stats['mean']:.2f}, 最小值={score_stats['min']:.2f}, 最大值={score_stats['max']:.2f}")
    
    # 检查识别到的形态
    logger.info(f"识别到的形态: {patterns}")
    
    # 分析信号和评分
    analyze_signals(data, signals, score)
    
    logger.info("MACD增强功能测试完成")
    return True


def analyze_signals(data, signals, score):
    """
    分析信号质量
    
    Args:
        data: 价格数据
        signals: 信号数据
        score: 评分数据
    """
    # 找出所有买入信号
    buy_dates = signals['buy_signal'][signals['buy_signal']].index
    
    # 找出所有卖出信号
    sell_dates = signals['sell_signal'][signals['sell_signal']].index
    
    if len(buy_dates) == 0 or len(sell_dates) == 0:
        logger.warning("没有足够的买入或卖出信号进行分析")
        return
    
    # 分析买入信号后的短期收益
    buy_returns = []
    for date in buy_dates:
        if date in data.index:
            # 计算买入后5个交易日的收益率
            idx = data.index.get_loc(date)
            if idx + 5 < len(data):
                ret = (data['close'].iloc[idx+5] / data['close'].iloc[idx] - 1) * 100
                buy_returns.append(ret)
    
    # 分析卖出信号后的短期收益（避免了损失）
    sell_returns = []
    for date in sell_dates:
        if date in data.index:
            # 计算卖出后5个交易日的收益率（负值表示避免了损失）
            idx = data.index.get_loc(date)
            if idx + 5 < len(data):
                ret = (data['close'].iloc[idx] / data['close'].iloc[idx+5] - 1) * 100
                sell_returns.append(ret)
    
    # 计算平均收益
    avg_buy_return = sum(buy_returns) / len(buy_returns) if buy_returns else 0
    avg_sell_return = sum(sell_returns) / len(sell_returns) if sell_returns else 0
    
    logger.info(f"买入信号后5日平均收益率: {avg_buy_return:.2f}%")
    logger.info(f"卖出信号后5日平均避免损失: {avg_sell_return:.2f}%")
    
    # 分析评分与实际收益的相关性
    if len(buy_dates) > 0:
        buy_scores = [score.loc[date] if date in score.index else None for date in buy_dates]
        buy_scores = [s for s in buy_scores if s is not None]
        
        if len(buy_scores) > 0 and len(buy_returns) == len(buy_scores):
            # 计算相关系数
            correlation = np.corrcoef(buy_scores, buy_returns)[0, 1]
            logger.info(f"买入信号评分与实际收益相关系数: {correlation:.4f}")
    
    # 分析高分信号与低分信号的表现差异
    if len(buy_returns) > 0 and len(buy_scores) > 0:
        high_score_indices = [i for i, s in enumerate(buy_scores) if s >= 70]
        low_score_indices = [i for i, s in enumerate(buy_scores) if s < 70]
        
        if high_score_indices and low_score_indices:
            high_score_returns = [buy_returns[i] for i in high_score_indices]
            low_score_returns = [buy_returns[i] for i in low_score_indices]
            
            avg_high_score_return = sum(high_score_returns) / len(high_score_returns)
            avg_low_score_return = sum(low_score_returns) / len(low_score_returns)
            
            logger.info(f"高分买入信号(>=70)平均收益率: {avg_high_score_return:.2f}%")
            logger.info(f"低分买入信号(<70)平均收益率: {avg_low_score_return:.2f}%")
            logger.info(f"高分信号相对低分信号收益增强: {avg_high_score_return - avg_low_score_return:.2f}%")


if __name__ == "__main__":
    test_enhanced_macd() 