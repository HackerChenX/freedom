"""
ZXM体系指标使用示例

展示如何使用ZXM体系的各种指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入指标工厂和ZXM体系指标
from indicators.factory import IndicatorFactory
from indicators.zxm.trend_indicators import ZXMDailyTrendUp
from indicators.zxm.elasticity_indicators import ZXMAmplitudeElasticity
from indicators.zxm.buy_point_indicators import ZXMMACallback
from indicators.zxm.selection_model import ZXMSelectionModel


def generate_sample_data(days=180):
    """
    生成示例数据
    
    Args:
        days: 生成数据的天数
        
    Returns:
        pd.DataFrame: 示例数据
    """
    # 生成日期
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 生成模拟价格数据
    np.random.seed(42)  # 设置随机种子以保证结果可重现
    
    # 初始价格
    base_price = 50.0
    
    # 模拟价格趋势（带有一定的周期性和随机性）
    trend = np.linspace(0, 2*np.pi, days)
    price_trend = base_price + 10 * np.sin(trend) + np.cumsum(np.random.normal(0, 0.3, days))
    
    # 生成OHLCV数据
    high = price_trend + np.random.uniform(0, 2, days)
    low = price_trend - np.random.uniform(0, 2, days)
    open_price = low + np.random.uniform(0, high-low, days)
    close = low + np.random.uniform(0, high-low, days)
    volume = np.random.uniform(100000, 1000000, days) * (1 + 0.5 * np.sin(trend))
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': date_range,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'capital': volume * 100  # 模拟流通股本
    })
    
    # 设置日期为索引
    data.set_index('date', inplace=True)
    
    return data


def generate_weekly_data(daily_data):
    """
    从日线数据生成周线数据
    
    Args:
        daily_data: 日线数据
        
    Returns:
        pd.DataFrame: 周线数据
    """
    # 设置日期为索引（如果尚未设置）
    if 'date' in daily_data.columns:
        daily_data = daily_data.set_index('date')
    
    # 按周重采样
    weekly_data = pd.DataFrame()
    weekly_data['open'] = daily_data['open'].resample('W').first()
    weekly_data['high'] = daily_data['high'].resample('W').max()
    weekly_data['low'] = daily_data['low'].resample('W').min()
    weekly_data['close'] = daily_data['close'].resample('W').last()
    weekly_data['volume'] = daily_data['volume'].resample('W').sum()
    
    return weekly_data


def generate_monthly_data(daily_data):
    """
    从日线数据生成月线数据
    
    Args:
        daily_data: 日线数据
        
    Returns:
        pd.DataFrame: 月线数据
    """
    # 设置日期为索引（如果尚未设置）
    if 'date' in daily_data.columns:
        daily_data = daily_data.set_index('date')
    
    # 按月重采样
    monthly_data = pd.DataFrame()
    monthly_data['open'] = daily_data['open'].resample('M').first()
    monthly_data['high'] = daily_data['high'].resample('M').max()
    monthly_data['low'] = daily_data['low'].resample('M').min()
    monthly_data['close'] = daily_data['close'].resample('M').last()
    monthly_data['volume'] = daily_data['volume'].resample('M').sum()
    
    return monthly_data


def demo_zxm_daily_trend_up(data):
    """
    演示ZXM趋势-日线上移指标
    
    Args:
        data: 日线数据
    """
    print("\n=== ZXM趋势-日线上移指标示例 ===")
    
    # 创建指标实例
    zxm_daily_trend = ZXMDailyTrendUp()
    
    # 计算指标
    result = zxm_daily_trend.calculate(data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的结果:\n{recent_result[['MA60', 'MA120', 'XG']]}")
    
    # 统计指标信号次数
    signal_count = result['XG'].sum()
    total_days = len(result)
    print(f"日线上移信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")
    
    # 绘制结果图表
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(result.index, result['MA60'], label='MA60')
    plt.plot(result.index, result['MA120'], label='MA120')
    plt.title('ZXM趋势-日线上移指标')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.fill_between(result.index, 0, result['XG'], color='green', alpha=0.3, label='Trend Up Signal')
    plt.title('趋势上移信号')
    plt.tight_layout()
    plt.show()


def demo_zxm_amplitude_elasticity(data):
    """
    演示ZXM弹性-振幅指标
    
    Args:
        data: 日线数据
    """
    print("\n=== ZXM弹性-振幅指标示例 ===")
    
    # 创建指标实例
    zxm_amplitude = ZXMAmplitudeElasticity()
    
    # 计算指标
    result = zxm_amplitude.calculate(data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的结果:\n{recent_result[['Amplitude', 'A1', 'XG']]}")
    
    # 统计指标信号次数
    signal_count = result['XG'].sum()
    total_days = len(result)
    print(f"振幅弹性信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")
    
    # 绘制结果图表
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.title('收盘价')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(result.index, result['Amplitude'], label='Amplitude(%)')
    plt.axhline(y=8.1, color='r', linestyle='--', label='8.1% Threshold')
    plt.title('日振幅百分比')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.fill_between(result.index, 0, result['XG'], color='green', alpha=0.3, label='Elasticity Signal')
    plt.title('振幅弹性信号')
    plt.tight_layout()
    plt.show()


def demo_zxm_ma_callback(data):
    """
    演示ZXM买点-回踩均线指标
    
    Args:
        data: 日线数据
    """
    print("\n=== ZXM买点-回踩均线指标示例 ===")
    
    # 创建指标实例（默认回踩幅度为4%）
    zxm_callback = ZXMMACallback(callback_percent=4.0)
    
    # 计算指标
    result = zxm_callback.calculate(data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的结果:\n{recent_result[['A20', 'A30', 'A60', 'A120', 'XG']]}")
    
    # 统计指标信号次数
    signal_count = result['XG'].sum()
    total_days = len(result)
    print(f"回踩均线买点信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")
    
    # 绘制结果图表
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(result.index, result['MA20'], label='MA20')
    plt.plot(result.index, result['MA60'], label='MA60')
    plt.plot(result.index, result['MA120'], label='MA120')
    plt.title('ZXM买点-回踩均线指标')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.fill_between(result.index, 0, result['XG'], color='green', alpha=0.3, label='MA Callback Signal')
    plt.title('回踩均线买点信号')
    plt.tight_layout()
    plt.show()


def demo_zxm_selection_model(daily_data, weekly_data, monthly_data):
    """
    演示ZXM体系通用选股模型
    
    Args:
        daily_data: 日线数据
        weekly_data: 周线数据
        monthly_data: 月线数据
    """
    print("\n=== ZXM体系通用选股模型示例 ===")
    
    # 创建选股模型实例
    zxm_model = ZXMSelectionModel(callback_percent=4.0)
    
    # 计算选股模型
    result = zxm_model.calculate(daily_data, weekly_data, monthly_data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的总得分:\n{recent_result[['趋势指标得分', '弹性指标得分', '买点指标得分', 'ZXM选股总得分']]}")
    
    # 绘制结果图表
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(daily_data.index, daily_data['close'], label='Close Price')
    plt.title('收盘价')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(result.index, result['趋势指标得分'], label='趋势得分', color='blue')
    plt.plot(result.index, result['弹性指标得分'], label='弹性得分', color='orange')
    plt.plot(result.index, result['买点指标得分'], label='买点得分', color='green')
    plt.title('各类指标得分')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(result.index, result['ZXM选股总得分'], label='总得分', color='red')
    plt.axhline(y=60, color='green', linestyle='--', label='买入阈值(60分)')
    plt.title('ZXM选股总得分')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def demo_zxm_indicator_factory():
    """
    演示通过指标工厂创建ZXM指标
    """
    print("\n=== 通过指标工厂创建ZXM指标示例 ===")
    
    # 获取所有支持的指标类型
    supported_indicators = IndicatorFactory.get_supported_indicators()
    
    # 筛选ZXM相关指标
    zxm_indicators = [indicator for indicator in supported_indicators if indicator.startswith('ZXM')]
    
    print(f"系统支持的ZXM指标：")
    for i, indicator in enumerate(zxm_indicators, 1):
        print(f"{i}. {indicator}")
    
    # 演示使用工厂创建ZXM指标
    daily_trend = IndicatorFactory.create_indicator("ZXM_DAILY_TREND_UP")
    amplitude = IndicatorFactory.create_indicator("ZXM_AMPLITUDE_ELASTICITY")
    selection_model = IndicatorFactory.create_indicator("ZXM_SELECTION_MODEL")
    
    print("\n通过工厂成功创建以下指标：")
    print(f"- {daily_trend.name}: {daily_trend.description}")
    print(f"- {amplitude.name}: {amplitude.description}")
    print(f"- {selection_model.name}: {selection_model.description}")


def main():
    """主函数"""
    print("==== ZXM体系指标使用示例 ====")
    
    # 生成示例数据
    daily_data = generate_sample_data(days=180)
    weekly_data = generate_weekly_data(daily_data)
    monthly_data = generate_monthly_data(daily_data)
    
    print(f"生成的示例数据：{len(daily_data)}行日线数据，{len(weekly_data)}行周线数据，{len(monthly_data)}行月线数据")
    print(f"日线数据示例：\n{daily_data.head()}")
    
    # 演示各类指标
    demo_zxm_daily_trend_up(daily_data)
    demo_zxm_amplitude_elasticity(daily_data)
    demo_zxm_ma_callback(daily_data)
    demo_zxm_selection_model(daily_data, weekly_data, monthly_data)
    demo_zxm_indicator_factory()
    
    print("\n==== 示例结束 ====")


if __name__ == "__main__":
    main() 