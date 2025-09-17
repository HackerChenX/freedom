"""
ZXM体系指标使用示例

展示如何使用ZXM体系的各种指标
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 导入统一指标注册系统
from indicators.complete_indicator_registry import complete_registry
from db.sql_manager import SQLManager, QueryType


def generate_sample_data_Indicators(days=180):
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
    zxm_daily_trend = complete_registry.create_indicator('ZXM_DAILY_TREND_UP')
    if not zxm_daily_trend:
        print("无法创建ZXM_DAILY_TREND_UP指标")
        return

    # 计算指标
    result = zxm_daily_trend.calculate(data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的结果:\n{recent_result[['MA60', 'MA120', 'XG']]}")
    
    # 统计指标信号次数
    signal_count = result['XG'].sum()
    total_days = len(result)
    print(f"日线上移信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")


def demo_zxm_amplitude_elasticity(data):
    """
    演示ZXM弹性-振幅指标

    Args:
        data: 日线数据
    """
    print("\n=== ZXM弹性-振幅指标示例 ===")

    # 创建指标实例
    zxm_amplitude = complete_registry.create_indicator('ZXM_AMPLITUDE_ELASTICITY')
    if not zxm_amplitude:
        print("无法创建ZXM_AMPLITUDE_ELASTICITY指标")
        return

    # 计算指标
    result = zxm_amplitude.calculate(data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的结果:\n{recent_result[['Amplitude', 'A1', 'XG']]}")
    
    # 统计指标信号次数
    signal_count = result['XG'].sum()
    total_days = len(result)
    print(f"振幅弹性信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")


def demo_zxm_ma_callback(data):
    """
    演示ZXM买点-回踩均线指标

    Args:
        data: 日线数据
    """
    print("\n=== ZXM买点-回踩均线指标示例 ===")

    # 创建指标实例（默认回踩幅度为4%）
    zxm_callback = complete_registry.create_indicator('ZXM_MA_CALLBACK', callback_percent=4.0)
    if not zxm_callback:
        print("无法创建ZXM_MA_CALLBACK指标")
        return

    # 计算指标
    result = zxm_callback.calculate(data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的结果:\n{recent_result[['A20', 'A30', 'A60', 'A120', 'XG']]}")
    
    # 统计指标信号次数
    signal_count = result['XG'].sum()
    total_days = len(result)
    print(f"回踩均线买点信号出现次数: {signal_count}，占总天数的 {signal_count/total_days*100:.2f}%")


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
    zxm_model = complete_registry.create_indicator('ZXM_SELECTION_MODEL', callback_percent=4.0)
    if not zxm_model:
        print("无法创建ZXM_SELECTION_MODEL指标")
        return

    # 计算选股模型
    result = zxm_model.calculate(daily_data, weekly_data, monthly_data)
    
    # 打印最近的结果
    recent_result = result.tail(5)
    print(f"最近5天的总得分:\n{recent_result[['趋势指标得分', '弹性指标得分', '买点指标得分', 'ZXM选股总得分']]}")


def demo_zxm_indicator_factory():
    """
    演示通过统一注册系统创建ZXM指标
    """
    print("\n=== 通过统一注册系统创建ZXM指标示例 ===")

    # 获取所有支持的指标类型
    supported_indicators = complete_registry.get_indicator_names()

    # 筛选ZXM相关指标
    zxm_indicators = [indicator for indicator in supported_indicators if indicator.startswith('ZXM')]

    print(f"系统支持的ZXM指标：")
    for i, indicator in enumerate(zxm_indicators, 1):
        print(f"{i}. {indicator}")

    # 演示使用统一注册系统创建ZXM指标
    daily_trend = complete_registry.create_indicator("ZXM_DAILY_TREND_UP")
    amplitude = complete_registry.create_indicator("ZXM_AMPLITUDE_ELASTICITY")
    selection_model = complete_registry.create_indicator("ZXM_SELECTION_MODEL")

    print("\n通过统一注册系统成功创建以下指标：")
    if daily_trend:
        print(f"- {daily_trend.name}: {getattr(daily_trend, 'description', '无描述')}")
    if amplitude:
        print(f"- {amplitude.name}: {getattr(amplitude, 'description', '无描述')}")
    if selection_model:
        print(f"- {selection_model.name}: {getattr(selection_model, 'description', '无描述')}")


def mainUsezxmindicators():
    """主函数"""
    print("==== ZXM体系指标使用示例 ====")
    
    # 生成示例数据
    daily_data = generate_sample_data_Indicators(days=180)
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
    mainUsezxmindicators() 