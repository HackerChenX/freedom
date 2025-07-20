#!/usr/bin/env python3
from utils.dependency_injection import get_logger
# -*- coding: utf-8 -*-

"""
真实技术指标计算实现

完全替代所有模拟实现，使用真实的数学公式计算技术指标。

Author: AI Assistant
Date: 2025-07-19
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = get_logger(__name__)


class RealTechnicalIndicators:
    """真实技术指标计算器"""
    
    @staticmethod
    def calculate_ma(data: pd.Series, period: int = 20) -> pd.Series:
        """计算真实移动平均线"""
        return data.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
        """计算真实指数移动平均线"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算真实MACD指标"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """计算真实RSI指标"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """计算真实布林带"""
        ma = data.rolling(window=period, min_periods=period).mean()
        std = data.rolling(window=period, min_periods=period).std()
        
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': ma,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series, 
                     k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Dict[str, pd.Series]:
        """计算真实KDJ指标"""
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        k = rsv.ewm(alpha=1/d_period, adjust=False).mean()
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return {
            'k': k,
            'd': d,
            'j': j
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算真实平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算真实能量潮指标"""
        price_change = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算真实威廉指标"""
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return wr
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """计算真实顺势指标"""
        typical_price = (high + low + close) / 3
        ma_tp = typical_price.rolling(window=period, min_periods=period).mean()
        
        mad = typical_price.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - ma_tp) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """计算真实随机指标"""
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def calculate_momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """计算真实动量指标"""
        return data - data.shift(period)
    
    @staticmethod
    def calculate_roc(data: pd.Series, period: int = 10) -> pd.Series:
        """计算真实变化率指标"""
        return (data / data.shift(period) - 1) * 100


class RealIndicatorFactory:
    """真实指标工厂类"""
    
    def __init__(self):
        self.calculator = RealTechnicalIndicators()
    
    def create_indicator(self, name: str, **kwargs) -> Any:
        """创建真实指标实例"""
        
        class RealIndicator:
            def __init__(self, indicator_name: str, calculator: RealTechnicalIndicators):
                self.name = indicator_name
                self.calculator = calculator
                self.patterns = ['bullish', 'bearish', 'neutral']
            
            def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
                """计算真实指标值"""
                try:
                    if 'close' not in data.columns:
                        raise ValueError("数据中缺少收盘价列")
                    
                    close = pd.to_numeric(data['close'], errors='coerce').dropna()
                    
                    if len(close) < 20:
                        return {'error': 'insufficient_data'}
                    
                    result = {}
                    
                    if 'MA' in self.name:
                        result['ma5'] = self.calculator.calculate_ma(close, 5)
                        result['ma10'] = self.calculator.calculate_ma(close, 10)
                        result['ma20'] = self.calculator.calculate_ma(close, 20)
                        
                    elif 'EMA' in self.name:
                        result['ema12'] = self.calculator.calculate_ema(close, 12)
                        result['ema26'] = self.calculator.calculate_ema(close, 26)
                        
                    elif 'MACD' in self.name:
                        macd_data = self.calculator.calculate_macd(close)
                        result.update(macd_data)
                        
                    elif 'RSI' in self.name:
                        result['rsi'] = self.calculator.calculate_rsi(close)
                        
                    elif 'BOLL' in self.name:
                        boll_data = self.calculator.calculate_bollinger_bands(close)
                        result.update(boll_data)
                        
                    elif 'KDJ' in self.name:
                        if all(col in data.columns for col in ['high', 'low']):
                            high = pd.to_numeric(data['high'], errors='coerce').dropna()
                            low = pd.to_numeric(data['low'], errors='coerce').dropna()
                            kdj_data = self.calculator.calculate_kdj(high, low, close)
                            result.update(kdj_data)
                        
                    elif 'ATR' in self.name:
                        if all(col in data.columns for col in ['high', 'low']):
                            high = pd.to_numeric(data['high'], errors='coerce').dropna()
                            low = pd.to_numeric(data['low'], errors='coerce').dropna()
                            result['atr'] = self.calculator.calculate_atr(high, low, close)
                    
                    elif 'OBV' in self.name:
                        if 'volume' in data.columns:
                            volume = pd.to_numeric(data['volume'], errors='coerce').dropna()
                            result['obv'] = self.calculator.calculate_obv(close, volume)
                    
                    else:
                        # 默认计算基本指标
                        result['value'] = close
                        result['ma20'] = self.calculator.calculate_ma(close, 20)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"计算指标 {self.name} 失败: {e}")
                    return {'error': str(e)}
            
            def get_patterns(self) -> list:
                """获取支持的形态"""
                return self.patterns
        
        return RealIndicator(name, self.calculator)


# 全局真实指标工厂实例
real_indicator_factory = RealIndicatorFactory()
