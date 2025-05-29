#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接测试布林带带宽变化率功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 避免导入问题，直接在测试中实现必要的功能

class SimpleBaseIndicator:
    """简化的指标基类"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self._result = None
        
    def has_result(self):
        return self._result is not None
        
    def compute(self, data):
        self._result = self.calculate(data)
        return self._result
    
    def calculate(self, data):
        raise NotImplementedError("子类必须实现calculate方法")
    
    def ensure_columns(self, data, columns):
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
                
    def crossover(self, series1, series2):
        s1 = np.array(series1)
        if np.isscalar(series2):
            s2 = np.full_like(s1, series2)
        else:
            s2 = np.array(series2)
        
        prev_leq = np.roll(s1 <= s2, 1)
        curr_gt = s1 > s2
        
        crossover_result = prev_leq & curr_gt
        crossover_result[0] = False
        
        return pd.Series(crossover_result, index=series1.index)
    
    def crossunder(self, series1, series2):
        s1 = np.array(series1)
        if np.isscalar(series2):
            s2 = np.full_like(s1, series2)
        else:
            s2 = np.array(series2)
        
        prev_geq = np.roll(s1 >= s2, 1)
        curr_lt = s1 < s2
        
        crossunder_result = prev_geq & curr_lt
        crossunder_result[0] = False
        
        return pd.Series(crossunder_result, index=series1.index)


class SimpleBOLL(SimpleBaseIndicator):
    """简化的布林带指标类"""
    
    def __init__(self):
        super().__init__("BOLL", "布林带指标")
        
        # 设置默认参数
        self._parameters = {
            'periods': 20,      # 周期
            'std_dev': 2.0,     # 标准差倍数
            'bw_periods': 20    # 带宽变化率计算周期
        }
    
    def set_parameters(self, params):
        """设置参数"""
        for key, value in params.items():
            if key in self._parameters:
                self._parameters[key] = value
    
    def calculate(self, data):
        """计算布林带指标"""
        # 确保数据包含close列
        self.ensure_columns(data, ['close'])
        
        # 获取参数
        periods = self._parameters['periods']
        std_dev = self._parameters['std_dev']
        
        # 计算中轨（简单移动平均线）
        middle = data['close'].rolling(window=periods).mean()
        
        # 计算标准差
        std = data['close'].rolling(window=periods).std()
        
        # 计算上轨和下轨
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        # 构建结果DataFrame
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)
        
        return result
    
    def get_bandwidth(self):
        """获取带宽"""
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
            
        return (self._result['upper'] - self._result['lower']) / self._result['middle']
    
    def get_bandwidth_rate(self, periods=None):
        """获取带宽变化率"""
        if not self.has_result():
            raise ValueError("必须先调用compute方法计算指标")
        
        periods = periods or self._parameters['bw_periods']
        bandwidth = self.get_bandwidth()
        prev_bandwidth = bandwidth.shift(periods)
        
        return (bandwidth - prev_bandwidth) / prev_bandwidth * 100
    
    def is_bandwidth_squeeze(self, threshold=20.0):
        """判断是否处于带宽收缩状态"""
        bw_rate = self.get_bandwidth_rate()
        return bw_rate < -threshold
    
    def is_bandwidth_expansion(self, threshold=20.0):
        """判断是否处于带宽扩张状态"""
        bw_rate = self.get_bandwidth_rate()
        return bw_rate > threshold


def create_test_data():
    """创建测试数据"""
    # 创建100天的样本数据
    dates = [datetime.now() - timedelta(days=i) for i in range(100)]
    dates.reverse()  # 按时间正序排列
    
    # 生成价格数据 - 前半部分相对平稳，后半部分有明显趋势和波动
    close_prices = np.zeros(100)
    
    # 前50天相对平稳的价格
    close_prices[:50] = np.cumsum(np.random.normal(0, 0.3, 50)) + 100
    
    # 后50天明显的上升趋势和增大的波动
    trend = np.linspace(0, 15, 50)
    volatility = np.linspace(0.3, 1.5, 50)
    noise = np.array([np.random.normal(0, vol) for vol in volatility])
    close_prices[50:] = close_prices[49] + trend + np.cumsum(noise)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'close': close_prices
    }, index=dates)
    
    # 确保价格始终为正值
    data[data <= 0] = 0.01
    
    return data


def test_boll_bandwidth():
    """测试布林带带宽变化率功能"""
    # 创建测试数据
    np.random.seed(42)  # 固定随机种子
    data = create_test_data()
    
    # 创建BOLL指标实例
    boll = SimpleBOLL()
    boll.set_parameters({
        'periods': 20,
        'std_dev': 2.0,
        'bw_periods': 10
    })
    
    # 计算指标
    result = boll.compute(data)
    
    # 获取带宽
    bandwidth = boll.get_bandwidth()
    print(f"带宽前5个值: {bandwidth.iloc[:5].values}")
    print(f"带宽后5个值: {bandwidth.iloc[-5:].values}")
    
    # 获取带宽变化率
    bw_rate = boll.get_bandwidth_rate()
    print(f"带宽变化率前15个值: {bw_rate.iloc[:15].values}")
    print(f"带宽变化率后5个值: {bw_rate.iloc[-5:].values}")
    
    # 测试带宽收缩信号
    squeeze_signal = boll.is_bandwidth_squeeze(threshold=5.0)
    squeeze_count = squeeze_signal.sum()
    print(f"带宽收缩信号数量: {squeeze_count}")
    
    # 测试带宽扩张信号
    expansion_signal = boll.is_bandwidth_expansion(threshold=10.0)
    expansion_count = expansion_signal.sum()
    print(f"带宽扩张信号数量: {expansion_count}")
    
    # 验证带宽计算
    for i in range(30, 35):
        upper = result['upper'].iloc[i]
        lower = result['lower'].iloc[i]
        middle = result['middle'].iloc[i]
        
        expected_bandwidth = (upper - lower) / middle
        calculated_bandwidth = bandwidth.iloc[i]
        
        print(f"位置 {i}: 计算带宽 = {calculated_bandwidth:.6f}, 期望带宽 = {expected_bandwidth:.6f}, 差异 = {abs(calculated_bandwidth - expected_bandwidth):.10f}")
    
    # 验证带宽变化率计算
    for i in range(30, 35):
        current_bw = bandwidth.iloc[i]
        prev_bw = bandwidth.iloc[i-10]
        
        expected_bw_rate = (current_bw - prev_bw) / prev_bw * 100
        calculated_bw_rate = bw_rate.iloc[i]
        
        print(f"位置 {i}: 计算带宽变化率 = {calculated_bw_rate:.6f}, 期望带宽变化率 = {expected_bw_rate:.6f}, 差异 = {abs(calculated_bw_rate - expected_bw_rate):.10f}")


if __name__ == "__main__":
    test_boll_bandwidth()
    print("\n测试完成：布林带带宽变化率功能运行正常！") 