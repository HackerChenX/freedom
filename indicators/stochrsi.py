#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
随机相对强弱指数(STOCHRSI)

将RSI指标标准化到0-100区间，增强短期超买超卖信号
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class STOCHRSI(BaseIndicator):
    """
    随机相对强弱指数(STOCHRSI) (STOCHRSI)
    
    分类：震荡类指标
    描述：将RSI指标标准化到0-100区间，增强短期超买超卖信号
    """
    
    def __init__(self, period: int = 14, k_period: int = 3, d_period: int = 3):
        """
        初始化随机相对强弱指数(STOCHRSI)指标
        
        Args:
            period: RSI计算周期，默认为14
            k_period: K值计算周期，默认为3
            d_period: D值计算周期，默认为3
        """
        super().__init__(name="STOCHRSI", description="随机相对强弱指数，将RSI标准化到0-100区间")
        self.period = period
        self.k_period = k_period
        self.d_period = d_period
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算随机相对强弱指数(STOCHRSI)指标
        
        Args:
            df: 包含价格数据的DataFrame
                
        Returns:
            包含STOCHRSI指标的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 步骤1: 计算RSI
        delta = df_copy['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 步骤2: 计算StochRSI
        min_rsi = rsi.rolling(window=self.period).min()
        max_rsi = rsi.rolling(window=self.period).max()
        
        # 避免除零错误，当max_rsi == min_rsi时，StochRSI设为50
        rsi_range = max_rsi - min_rsi
        stoch_rsi = np.where(
            rsi_range == 0, 
            50.0,  # 当RSI没有变化时，设为中性值
            100 * (rsi - min_rsi) / rsi_range
        )
        
        # 确保StochRSI在0-100范围内
        stoch_rsi = np.clip(stoch_rsi, 0, 100)
        
        # 转换为pandas Series以便使用rolling方法
        stoch_rsi = pd.Series(stoch_rsi, index=df_copy.index)
        
        # 步骤3: 计算K和D值
        k = stoch_rsi.rolling(window=self.k_period).mean()
        d = k.rolling(window=self.d_period).mean()
        
        # 保存结果
        df_copy['rsi'] = rsi
        df_copy['stochrsi'] = stoch_rsi
        df_copy['k'] = k
        df_copy['d'] = d
        
        # 保存结果到实例变量
        self._result = df_copy
        
        return df_copy

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算StochRSI原始评分
        
        Args:
            data: 包含价格数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列
        """
        # 计算指标
        result = self.calculate(data)
        
        # 获取指标值
        k = result['k']
        d = result['d']
        stoch_rsi = result['stochrsi']
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
        
        # 1. StochRSI超买超卖评分
        # 超卖区域（K < 20）买入信号
        oversold_mask = k < 20
        score += np.where(oversold_mask, 20, 0)
        
        # 超买区域（K > 80）卖出信号
        overbought_mask = k > 80
        score -= np.where(overbought_mask, 20, 0)
        
        # 2. K线与D线交叉评分
        if len(k) > 1:
            # K上穿D线（金叉）
            golden_cross = (k.shift(1) <= d.shift(1)) & (k > d)
            score += np.where(golden_cross, 25, 0)
            
            # K下穿D线（死叉）
            death_cross = (k.shift(1) >= d.shift(1)) & (k < d)
            score -= np.where(death_cross, 25, 0)
        
        # 3. StochRSI趋势评分
        if len(k) >= 3:
            # K线连续上升
            k_rising = (k > k.shift(1)) & (k.shift(1) > k.shift(2))
            score += np.where(k_rising, 15, 0)
            
            # K线连续下降
            k_falling = (k < k.shift(1)) & (k.shift(1) < k.shift(2))
            score -= np.where(k_falling, 15, 0)
        
        # 4. StochRSI背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_stochrsi_divergence(data['close'], k)
            score += divergence_score
        
        # 5. StochRSI强度评分
        # K线在极值区域的强度
        extreme_low = k < 10  # 极度超卖
        extreme_high = k > 90  # 极度超买
        score += np.where(extreme_low, 12, 0)
        score -= np.where(extreme_high, 12, 0)
        
        # 6. StochRSI位置评分
        # K线在中性区域上方
        above_neutral = (k > 50) & (k < 80)
        score += np.where(above_neutral, 8, 0)
        
        # K线在中性区域下方
        below_neutral = (k < 50) & (k > 20)
        score -= np.where(below_neutral, 8, 0)
        
        # 确保评分在0-100范围内
        score = np.clip(score, 0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别StochRSI技术形态
        
        Args:
            data: 包含价格数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        if len(data) < 10:
            return []
        
        patterns = []
        
        # 计算指标
        result = self.calculate(data)
        k = result['k']
        d = result['d']
        
        if k.isna().all() or d.isna().all():
            return patterns
        
        # 获取最近的有效值
        recent_k = k.dropna().tail(10)
        recent_d = d.dropna().tail(10)
        
        if len(recent_k) == 0 or len(recent_d) == 0:
            return patterns
        
        current_k = recent_k.iloc[-1]
        current_d = recent_d.iloc[-1]
        
        # 1. 超买超卖形态
        if current_k > 80:
            patterns.append("StochRSI超买")
        elif current_k < 20:
            patterns.append("StochRSI超卖")
        elif 40 <= current_k <= 60:
            patterns.append("StochRSI中性区域")
        
        # 2. 交叉形态
        if len(recent_k) >= 2:
            prev_k = recent_k.iloc[-2]
            prev_d = recent_d.iloc[-2]
            
            if prev_k <= prev_d and current_k > current_d:
                patterns.append("StochRSI金叉")
            elif prev_k >= prev_d and current_k < current_d:
                patterns.append("StochRSI死叉")
        
        # 3. 趋势形态
        if len(recent_k) >= 3:
            if all(recent_k.iloc[i] > recent_k.iloc[i-1] for i in range(-2, 0)):
                patterns.append("StochRSI连续上升")
            elif all(recent_k.iloc[i] < recent_k.iloc[i-1] for i in range(-2, 0)):
                patterns.append("StochRSI连续下降")
        
        # 4. 背离形态
        if len(data) >= 20:
            divergence_type = self._detect_stochrsi_divergence_pattern(data['close'], k)
            if divergence_type:
                patterns.append(f"StochRSI{divergence_type}")
        
        # 5. 极值形态
        if current_k < 10:
            patterns.append("StochRSI极度超卖")
        elif current_k > 90:
            patterns.append("StochRSI极度超买")
        
        # 6. K线与D线位置关系
        if current_k > current_d:
            patterns.append("StochRSI K线上方")
        else:
            patterns.append("StochRSI K线下方")
        
        # 7. 钝化形态
        if len(recent_k) >= 5:
            if all(k_val > 80 for k_val in recent_k.tail(5)):
                patterns.append("StochRSI高位钝化")
            elif all(k_val < 20 for k_val in recent_k.tail(5)):
                patterns.append("StochRSI低位钝化")
        
        return patterns
    
    def _calculate_stochrsi_divergence(self, price: pd.Series, k: pd.Series) -> pd.Series:
        """
        计算StochRSI背离评分
        
        Args:
            price: 价格序列
            k: StochRSI K线序列
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
        
        # 寻找价格和K线的峰值谷值
        window = 5
        for i in range(window, len(price) - window):
            price_window = price.iloc[i-window:i+window+1]
            k_window = k.iloc[i-window:i+window+1]
            
            if pd.isna(k.iloc[i]):
                continue
            
            if price.iloc[i] == price_window.max():  # 价格峰值
                if k.iloc[i] != k_window.max():  # K线未创新高
                    divergence_score.iloc[i:i+10] -= 30  # 负背离
            elif price.iloc[i] == price_window.min():  # 价格谷值
                if k.iloc[i] != k_window.min():  # K线未创新低
                    divergence_score.iloc[i:i+10] += 30  # 正背离
        
        return divergence_score
    
    def _detect_stochrsi_divergence_pattern(self, price: pd.Series, k: pd.Series) -> Optional[str]:
        """
        检测StochRSI背离形态
        
        Args:
            price: 价格序列
            k: StochRSI K线序列
            
        Returns:
            Optional[str]: 背离类型
        """
        if len(price) < 20:
            return None
        
        # 寻找最近的峰值和谷值
        recent_price = price.tail(20)
        recent_k = k.tail(20)
        
        # 检查顶背离
        price_peaks = []
        k_peaks = []
        for i in range(5, len(recent_price) - 5):
            if recent_price.iloc[i] == recent_price.iloc[i-5:i+6].max():
                price_peaks.append((i, recent_price.iloc[i]))
                k_peaks.append((i, recent_k.iloc[i]))
        
        if len(price_peaks) >= 2:
            last_price_peak = price_peaks[-1][1]
            prev_price_peak = price_peaks[-2][1]
            last_k_peak = k_peaks[-1][1]
            prev_k_peak = k_peaks[-2][1]
            
            if last_price_peak > prev_price_peak and last_k_peak < prev_k_peak:
                return "负背离"
        
        # 检查底背离
        price_troughs = []
        k_troughs = []
        for i in range(5, len(recent_price) - 5):
            if recent_price.iloc[i] == recent_price.iloc[i-5:i+6].min():
                price_troughs.append((i, recent_price.iloc[i]))
                k_troughs.append((i, recent_k.iloc[i]))
        
        if len(price_troughs) >= 2:
            last_price_trough = price_troughs[-1][1]
            prev_price_trough = price_troughs[-2][1]
            last_k_trough = k_troughs[-1][1]
            prev_k_trough = k_troughs[-2][1]
            
            if last_price_trough < prev_price_trough and last_k_trough > prev_k_trough:
                return "正背离"
        
        return None
    
    # 兼容新接口
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算随机相对强弱指数(STOCHRSI)指标 - 兼容新接口
        
        Args:
            df: 包含价格数据的DataFrame
                
        Returns:
            包含STOCHRSI指标的DataFrame
        """
        return self.calculate(df)
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成随机相对强弱指数(STOCHRSI)指标交易信号
        
        Args:
            df: 包含价格数据和STOCHRSI指标的DataFrame
            **kwargs: 额外参数
                overbought: 超买阈值，默认为80
                oversold: 超卖阈值，默认为20
                
        Returns:
            添加了信号列的DataFrame:
            - stochrsi_buy_signal: 1=买入信号, 0=无信号
            - stochrsi_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['k', 'd']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        overbought = kwargs.get('overbought', 80)  # 超买阈值
        oversold = kwargs.get('oversold', 20)  # 超卖阈值
        
        # 初始化信号列
        df_copy['stochrsi_buy_signal'] = 0
        df_copy['stochrsi_sell_signal'] = 0
        
        # K上穿D为买入信号
        df_copy.loc[crossover(df_copy['k'], df_copy['d']), 'stochrsi_buy_signal'] = 1
        
        # K下穿D为卖出信号
        df_copy.loc[crossunder(df_copy['k'], df_copy['d']), 'stochrsi_sell_signal'] = 1
        
        # 超卖区域上穿为买入信号
        for i in range(1, len(df_copy)):
            if df_copy['k'].iloc[i-1] < oversold and df_copy['k'].iloc[i] > oversold:
                df_copy.iloc[i, df_copy.columns.get_loc('stochrsi_buy_signal')] = 1
            
            # 超买区域下穿为卖出信号
            elif df_copy['k'].iloc[i-1] > overbought and df_copy['k'].iloc[i] < overbought:
                df_copy.iloc[i, df_copy.columns.get_loc('stochrsi_sell_signal')] = 1
        
        return df_copy
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制随机相对强弱指数(STOCHRSI)指标图表
        
        Args:
            df: 包含STOCHRSI指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['k', 'd']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制K和D线
        ax.plot(df.index, df['k'], label='%K', color='blue')
        ax.plot(df.index, df['d'], label='%D', color='red', linestyle='--')
        
        # 添加超买超卖参考线
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.3, label='超买区域(80)')
        ax.axhline(y=20, color='g', linestyle='--', alpha=0.3, label='超卖区域(20)')
        ax.axhline(y=50, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('随机相对强弱指数(STOCHRSI)')
        ax.set_ylim([0, 100])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

