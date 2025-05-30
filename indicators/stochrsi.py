#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
随机相对强弱指标(STOCHRSI)模块

实现STOCHRSI指标计算，结合了RSI和随机指标的特性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class STOCHRSI(BaseIndicator):
    """
    随机相对强弱指标(STOCHRSI)
    
    STOCHRSI结合了RSI和随机指标的特性，是对RSI的改进，可以提供更敏感的超买超卖信号
    """
    
    def __init__(self, n: int = 14, m: int = 3, p: int = 3):
        """
        初始化STOCHRSI指标
        
        Args:
            n: RSI周期，默认为14
            m: K值周期，默认为3
            p: D值周期，默认为3
        """
        super().__init__(name="STOCHRSI", description="随机相对强弱指标")
        self.n = n
        self.m = m
        self.p = p
    
    def calculate(self, data: pd.DataFrame, n: int = None, m: int = None, p: int = None, *args, **kwargs) -> pd.DataFrame:
        """
        计算STOCHRSI指标
        
        Args:
            data: 输入数据，包含收盘价
            n: RSI周期，默认为None，使用实例化时指定的值
            m: K值周期，默认为None，使用实例化时指定的值
            p: D值周期，默认为None，使用实例化时指定的值
            
        Returns:
            pd.DataFrame: 计算结果，包含RSI、K和D值
            
        公式说明：
        RSI = 100 - (100 / (1 + RS))，其中RS = 平均上涨点数 / 平均下跌点数
        STOCHRSI_K = (RSI - min(RSI, m)) / (max(RSI, m) - min(RSI, m)) * 100
        STOCHRSI_D = SMA(STOCHRSI_K, p)
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 使用传入的参数或实例化时指定的参数
        n = n or self.n
        m = m or self.m
        p = p or self.p
        
        # 计算RSI
        rsi = self._calculate_rsi(data["close"], n)
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        result["RSI"] = rsi
        
        # 计算STOCHRSI_K
        stochrsi_k = np.zeros_like(rsi)
        for i in range(m, len(rsi)):
            rsi_window = rsi[i-m+1:i+1]
            min_rsi = np.min(rsi_window)
            max_rsi = np.max(rsi_window)
            
            if max_rsi != min_rsi:  # 防止除以零
                stochrsi_k[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
            else:
                stochrsi_k[i] = 50  # 如果最大最小值相等，设为中间值
        
        # 计算STOCHRSI_D
        stochrsi_d = self._sma(stochrsi_k, p)
        
        # 添加计算结果到数据框
        result["STOCHRSI_K"] = stochrsi_k
        result["STOCHRSI_D"] = stochrsi_d
        
        # 存储结果
        self._result = result
        
        return result
    
    def _calculate_rsi(self, prices: np.ndarray, n: int) -> np.ndarray:
        """
        计算RSI
        
        Args:
            prices: 价格序列
            n: 周期
            
        Returns:
            np.ndarray: RSI结果
        """
        # 计算价格变化
        deltas = np.diff(prices)
        deltas = np.append(0, deltas)  # 第一个元素没有变化，设为0
        
        # 分离上涨和下跌
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均上涨和下跌
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # 初始化第n个元素的平均值
        if n <= len(gain):
            avg_gain[n-1] = np.mean(gain[:n])
            avg_loss[n-1] = np.mean(loss[:n])
        
        # 计算剩余元素的平均值
        for i in range(n, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (n-1) + gain[i]) / n
            avg_loss[i] = (avg_loss[i-1] * (n-1) + loss[i]) / n
        
        # 计算RS和RSI
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        
        for i in range(n, len(prices)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
            else:
                rs[i] = 100  # 如果没有下跌，RSI设为100
            
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        return rsi
    
    def _sma(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算简单移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.mean(series[:i+1])
            else:
                result[i] = np.mean(series[i-n+1:i+1])
        
        return result
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        # 确保已计算STOCHRSI
        if not self.has_result():
            self.calculate(data, *args, **kwargs)
        
        if self._result is None:
            return pd.DataFrame()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=self._result.index)
        signals["STOCHRSI_K"] = self._result["STOCHRSI_K"]
        signals["STOCHRSI_D"] = self._result["STOCHRSI_D"]
        signals["RSI"] = self._result["RSI"]
        
        # 生成信号
        # 超买: K和D都高于80
        signals["overbought"] = (signals["STOCHRSI_K"] > 80) & (signals["STOCHRSI_D"] > 80)
        
        # 超卖: K和D都低于20
        signals["oversold"] = (signals["STOCHRSI_K"] < 20) & (signals["STOCHRSI_D"] < 20)
        
        # 金叉: K上穿D
        signals["golden_cross"] = self.crossover(signals["STOCHRSI_K"], signals["STOCHRSI_D"])
        
        # 死叉: K下穿D
        signals["death_cross"] = self.crossunder(signals["STOCHRSI_K"], signals["STOCHRSI_D"])
        
        # 买入信号: 超卖区域金叉
        signals["buy_signal"] = signals["oversold"] & signals["golden_cross"]
        
        # 卖出信号: 超买区域死叉
        signals["sell_signal"] = signals["overbought"] & signals["death_cross"]
        
        return signals

