"""
高级技术指标使用示例

本示例展示如何使用新实现的高级技术指标，包括K线形态识别、筹码分布、波浪分析等
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from db.clickhouse_db import get_clickhouse_db
from indicators.pattern.candlestick_patterns import CandlestickPatterns, PatternType
from indicators.zxm_washplate import ZXMWashPlate, WashPlateType
from indicators.chip_distribution import ChipDistribution
from indicators.fibonacci_tools import FibonacciTools, FibonacciType
from indicators.elliott_wave import ElliottWave, WavePattern, WaveDirection, WaveType
from indicators.gann_tools import GannTools, GannAngle, GannTimeCycle
from utils.logger import get_logger
from indicators.enhanced_factory import EnhancedIndicatorFactory
from indicators.market_env import MarketDetector

logger = get_logger(__name__)


def get_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从数据库获取股票数据
    
    Args:
        stock_code: 股票代码
        start_date: 起始日期
        end_date: 结束日期
        
    Returns:
        pd.DataFrame: 股票数据
    """
    try:
        # 连接ClickHouse数据库
        db = get_clickhouse_db()
        
        # 查询SQL
        sql = f"""
        SELECT 
            trade_date as date,
            open,
            high,
            low,
            close,
            volume,
            amount
        FROM stock_daily_data
        WHERE 
            ts_code = '{stock_code}' AND
            trade_date >= '{start_date}' AND
            trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        
        # 执行查询
        data = db.query(sql)
        
        # 添加换手率估算
        if 'turnover_rate' not in data.columns:
            # 这里使用成交量相对值估算换手率
            data['turnover_rate'] = data['volume'] / data['volume'].rolling(window=20).mean() * 5
            
        return data
    
    except Exception as e:
        logger.error(f"获取股票数据时出错: {e}")
        
        # 生成模拟数据用于演示
        logger.warning("使用模拟数据进行演示")
        
        # 生成日期序列
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        
        # 只保留交易日（周一至周五）
        dates = [date for date in dates if date.weekday() < 5]
        
        # 生成价格和成交量数据
        n = len(dates)
        close = np.zeros(n)
        close[0] = 100.0  # 起始价格
        
        # 生成随机价格序列
        for i in range(1, n):
            # 随机价格变动，带有一些趋势和波动
            change = np.random.normal(0, 1) * 2.0  # 日波动率
            trend = np.sin(i / 10) * 0.5  # 添加周期性趋势
            close[i] = close[i-1] * (1 + (change + trend) / 100)
        
        # 生成其他价格数据
        high = close * (1 + np.random.rand(n) * 0.03)
        low = close * (1 - np.random.rand(n) * 0.03)
        open_prices = low + (high - low) * np.random.rand(n)
        
        # 生成成交量数据
        volume = np.random.rand(n) * 1000000 * (1 + np.abs(np.diff(np.append(0, close)) / close) * 10)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'turnover_rate': volume / np.mean(volume) * 5
        })
        
        return df


def demo_candlestick_patterns(data: pd.DataFrame):
    """
    演示K线形态识别
    
    Args:
        data: 股票数据
    """
    print("\n=== K线形态识别示例 ===")
    
    # 创建K线形态识别指标
    candlestick = CandlestickPatterns()
    
    # 计算K线形态
    patterns = candlestick.calculate(data)
    
    # 分析最近形成的形态
    recent_patterns = candlestick.get_latest_patterns(data, lookback=10)
    
    # 显示结果
    print("\n最近10天内形成的K线形态:")
    for pattern, formed in recent_patterns.items():
        if formed:
            print(f"- {pattern}")
    
    # 查找指定形态
    interesting_patterns = [
        PatternType.HAMMER.value, 
        PatternType.MORNING_STAR.value,
        PatternType.ENGULFING_BULLISH.value,
        PatternType.DOUBLE_BOTTOM.value
    ]
    
    print("\n完整周期内的重要形态:")
    for pattern in interesting_patterns:
        if pattern in patterns.columns:
            pattern_dates = data.iloc[patterns[pattern].astype(bool)].index
            if len(pattern_dates) > 0:
                print(f"- {pattern}: 出现于 {pattern_dates.tolist()}")


def demo_zxm_washplate(data: pd.DataFrame):
    """
    演示ZXM洗盘形态识别
    
    Args:
        data: 股票数据
    """
    print("\n=== ZXM洗盘形态示例 ===")
    
    # 创建ZXM洗盘形态指标
    zxm_wash = ZXMWashPlate()
    
    # 计算洗盘形态
    wash_patterns = zxm_wash.calculate(data)
    
    # 分析最近的洗盘形态
    recent_wash = zxm_wash.get_recent_wash_plates(data, lookback=20)
    
    # 显示结果
    print("\n最近20天内形成的洗盘形态:")
    for pattern, formed in recent_wash.items():
        if formed:
            print(f"- {pattern}")
    
    # 统计各种洗盘形态出现的次数
    print("\n各洗盘形态出现次数统计:")
    for wash_type in WashPlateType:
        wash_name = wash_type.value
        if wash_name in wash_patterns.columns:
            count = wash_patterns[wash_name].sum()
            if count > 0:
                print(f"- {wash_name}: {count}次")


def demo_chip_distribution(data: pd.DataFrame):
    """
    演示筹码分布分析
    
    Args:
        data: 股票数据
    """
    print("\n=== 筹码分布分析示例 ===")
    
    # 创建筹码分布指标
    chip = ChipDistribution()
    
    # 计算筹码分布
    chip_result = chip.calculate(data)
    
    # 显示结果
    latest_result = chip_result.iloc[-1]
    
    print(f"当前筹码集中度: {latest_result['chip_concentration']:.4f}")
    print(f"获利盘比例: {latest_result['profit_ratio']*100:.2f}%")
    print(f"90%筹码区间宽度: {latest_result['chip_width_90pct']*100:.2f}%")
    print(f"平均成本: {latest_result['avg_cost']:.2f}")
    print(f"解套难度: {latest_result['untrapped_difficulty']:.4f}")
    
    # 分析筹码变动
    profit_change = chip_result['profit_ratio_change'].iloc[-5:].mean()
    if profit_change > 0:
        print("近期获利盘比例上升，筹码趋于松动")
    else:
        print("近期获利盘比例下降，筹码趋于沉淀")
    
    # 绘制筹码分布图
    try:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        chip.plot_chip_distribution(data.iloc[-60:], ax)
        plt.title("筹码分布热力图")
        plt.tight_layout()
        plt.savefig("chip_distribution.png")
        print("筹码分布图已保存为 chip_distribution.png")
    except Exception as e:
        print(f"绘制筹码分布图时出错: {e}")


def demo_fibonacci_tools(data: pd.DataFrame):
    """
    演示斐波那契工具
    
    Args:
        data: 股票数据
    """
    print("\n=== 斐波那契工具示例 ===")
    
    # 创建斐波那契工具指标
    fib = FibonacciTools()
    
    # 检测最近的高点和低点
    high_idx = data['high'].iloc[-60:].idxmax()
    low_idx = data['low'].iloc[-60:].idxmin()
    
    # 获取高低点的索引位置
    high_pos = data.index.get_loc(high_idx)
    low_pos = data.index.get_loc(low_idx)
    
    # 计算回调线
    retracements = fib.calculate(data, swing_high_idx=high_pos, swing_low_idx=low_pos, 
                               fib_type=FibonacciType.RETRACEMENT)
    
    # 计算扩展线
    extensions = fib.calculate(data, swing_high_idx=high_pos, swing_low_idx=low_pos, 
                             fib_type=FibonacciType.EXTENSION)
    
    # 显示结果
    print(f"高点日期: {data.iloc[high_pos]['date']}, 价格: {data.iloc[high_pos]['high']:.2f}")
    print(f"低点日期: {data.iloc[low_pos]['date']}, 价格: {data.iloc[low_pos]['low']:.2f}")
    
    print("\n斐波那契回调线:")
    for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
        level_name = f"fib_retracement_{level:.3f}".replace(".", "_")
        if level_name in retracements.columns:
            price = retracements[level_name].iloc[0]
            print(f"- {level:.3f} 回调线: {price:.2f}")
    
    print("\n斐波那契扩展线:")
    for level in [0.618, 1.0, 1.618, 2.618]:
        level_name = f"fib_extension_{level:.3f}".replace(".", "_")
        if level_name in extensions.columns:
            price = extensions[level_name].iloc[0]
            print(f"- {level:.3f} 扩展线: {price:.2f}")
    
    # 绘制斐波那契回调线
    try:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        fib.plot_fibonacci_levels(data.iloc[-60:], retracements, fib_type=FibonacciType.RETRACEMENT, ax=ax)
        plt.title("斐波那契回调线")
        plt.tight_layout()
        plt.savefig("fibonacci_retracement.png")
        print("斐波那契回调线图已保存为 fibonacci_retracement.png")
    except Exception as e:
        print(f"绘制斐波那契回调线时出错: {e}")


def demo_elliott_wave(data: pd.DataFrame):
    """
    演示艾略特波浪分析
    
    Args:
        data: 股票数据
    """
    print("\n=== 艾略特波浪分析示例 ===")
    
    # 创建艾略特波浪分析指标
    wave = ElliottWave()
    
    # 分析波浪结构
    wave_result = wave.calculate(data.iloc[-120:])
    
    # 显示结果
    pattern = wave_result["wave_pattern"].iloc[0]
    print(f"检测到的波浪形态: {pattern}")
    
    # 统计各波浪标签
    wave_labels = wave_result["wave_label"].dropna().unique()
    print("\n识别的波浪标签:")
    for label in sorted(wave_labels):
        if label:
            print(f"- 波浪 {label}")
    
    # 显示下一波浪预测
    if "next_wave_prediction" in wave_result.columns and not pd.isna(wave_result["next_wave_prediction"].iloc[0]):
        prediction = wave_result["next_wave_prediction"].iloc[0]
        current_price = data["close"].iloc[-1]
        change_pct = (prediction - current_price) / current_price * 100
        direction = "上涨" if change_pct > 0 else "下跌"
        print(f"\n下一波浪预测价格: {prediction:.2f} ({direction} {abs(change_pct):.2f}%)")
    
    # 绘制艾略特波浪
    try:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        wave.plot_elliott_waves(data.iloc[-120:], wave_result, ax=ax)
        plt.title("艾略特波浪分析")
        plt.tight_layout()
        plt.savefig("elliott_wave.png")
        print("艾略特波浪分析图已保存为 elliott_wave.png")
    except Exception as e:
        print(f"绘制艾略特波浪分析图时出错: {e}")


def demo_gann_tools(data: pd.DataFrame):
    """
    演示江恩理论工具
    
    Args:
        data: 股票数据
    """
    print("\n=== 江恩理论工具示例 ===")
    
    # 创建江恩理论工具指标
    gann = GannTools()
    
    # 选择支点位置
    pivot_idx = len(data) - 60  # 使用60天前的点作为支点
    
    # 计算江恩角度线
    gann_result = gann.calculate(data, pivot_idx=pivot_idx)
    
    # 计算江恩方格
    gann_square = gann.calculate_gann_square(data, pivot_idx=pivot_idx)
    
    # 显示结果
    pivot_price = data["close"].iloc[pivot_idx]
    pivot_date = data.iloc[pivot_idx]["date"]
    print(f"支点日期: {pivot_date}, 价格: {pivot_price:.2f}")
    
    print("\n江恩角度线 (当前价格):")
    latest = gann_result.iloc[-1]
    current_price = data["close"].iloc[-1]
    
    for angle in [GannAngle.ANGLE_1X1, GannAngle.ANGLE_1X2, GannAngle.ANGLE_2X1]:
        angle_name = angle.value
        if angle_name in latest.index:
            angle_price = latest[angle_name]
            diff_pct = (current_price - angle_price) / angle_price * 100
            position = "上方" if current_price > angle_price else "下方"
            print(f"- {angle_name}: {angle_price:.2f} (当前价格在{position}, 偏离 {abs(diff_pct):.2f}%)")
    
    print("\n重要时间周期:")
    time_cycles = [45, 90, 144, 360]
    for cycle in time_cycles:
        cycle_name = f"cycle_{cycle}"
        if cycle_name in gann_result.columns:
            cycle_points = gann_result[cycle_name].dropna()
            if not cycle_points.empty:
                dates = [data.iloc[idx]["date"] for idx in cycle_points.index]
                print(f"- {cycle}日周期: {', '.join(map(str, dates))}")
    
    # 绘制江恩角度线
    try:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        gann.plot_gann_angles(data.iloc[-60:], gann_result.iloc[-60:], ax=ax)
        plt.title("江恩角度线")
        plt.tight_layout()
        plt.savefig("gann_angles.png")
        print("江恩角度线图已保存为 gann_angles.png")
    except Exception as e:
        print(f"绘制江恩角度线时出错: {e}")


def main():
    """主函数"""
    # 获取样本数据
    stock_code = "000001.SZ"  # 平安银行
    start_date = "20220101"
    end_date = "20230630"
    
    print(f"获取 {stock_code} 从 {start_date} 到 {end_date} 的数据...")
    data = get_stock_data(stock_code, start_date, end_date)
    print(f"获取到 {len(data)} 条数据记录")
    
    if data.empty:
        logger.error("未能获取测试数据，无法继续")
        return

    # 2. 获取市场环境
    market_detector = MarketDetector()
    market_env = market_detector.detect_environment(data)
    logger.info(f"当前市场环境: {market_env.value}")

    # 3. 创建并计算指标
    indicator_names = ['EnhancedMACD', 'EnhancedRSI', 'EnhancedKDJ', 'EnhancedBOLL']

    for indicator_name in indicator_names:
        logger.info(f"--- 测试指标: {indicator_name} ---")
        try:
            # 创建指标实例
            indicator = EnhancedIndicatorFactory.create(indicator_name)
            
            # 设置市场环境
            indicator.set_market_environment(market_env)

            # 计算指标
            result_df = indicator.calculate(data)
            logger.info(f"{indicator_name} 计算结果 (前5行):\n{result_df.tail()}")

            # 生成信号
            signals_df = indicator.generate_signals(result_df)
            logger.info(f"{indicator_name} 信号 (前5行):\n{signals_df.tail()}")
            
            # 计算评分
            score_series = indicator.calculate_raw_score(result_df)
            logger.info(f"{indicator_name} 评分 (后5行):\n{score_series.tail()}")

        except Exception as e:
            logger.error(f"测试指标 {indicator_name} 时出错: {e}", exc_info=True)


if __name__ == "__main__":
    main() 