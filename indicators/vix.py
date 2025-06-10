#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VIX恐慌指数指标

通过价格波动幅度衡量市场恐慌程度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class VIX(BaseIndicator):
    """
    VIX恐慌指数指标
    
    分类：波动类指标
    描述：通过价格波动幅度衡量市场恐慌程度
    """
    
    def __init__(self, period: int = 10, smooth_period: int = 5):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化VIX恐慌指数指标
        
        Args:
            period: 计算周期，默认为10
            smooth_period: 平滑周期，默认为5
        """
        super().__init__()
        self.period = period
        self.smooth_period = smooth_period
        self.name = "VIX"
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VIX指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX恐慌指数指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                
        Returns:
            添加了VIX指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算日内波动率：(high-low)/close
        df_copy['daily_range'] = (df_copy['high'] - df_copy['low']) / df_copy['close'] * 100
        
        # 计算N日平均波动率
        df_copy['vix'] = df_copy['daily_range'].rolling(window=self.period).mean()
        
        # 计算平滑后的VIX
        df_copy['vix_smooth'] = df_copy['vix'].rolling(window=self.smooth_period).mean()
        
        return df_copy
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含VIX指标的DataFrame
        
        Returns:
            添加了交易信号的DataFrame
        """
        # 先计算指标
        result = self.calculate(df)
        
        # 初始化信号列
        result['buy_signal'] = 0
        result['sell_signal'] = 0
        result['vix_buy_signal'] = 0  # 添加与指标名称相关的信号列
        result['vix_sell_signal'] = 0  # 添加与指标名称相关的信号列
        
        # 提取指标数据
        vix = result['vix'].values
        vix_smooth = result['vix_smooth'].values
        
        # VIX见顶回落买入信号
        for i in range(2, len(vix)):
            if vix[i-2] < vix[i-1] and vix[i] < vix[i-1]:
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_buy_signal')] = 1
        
        # VIX处于低位的买入信号
        vix_avg = result['vix'].rolling(window=20).mean()
        for i in range(20, len(vix)):
            if vix[i] < vix_avg[i] * 0.7:  # VIX低于20日均值的70%
                result.iloc[i, result.columns.get_loc('buy_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_buy_signal')] = 1
        
        # VIX急剧上升的卖出信号
        for i in range(1, len(vix)):
            if vix[i] > vix[i-1] * 1.5:  # VIX上升超过50%
                result.iloc[i, result.columns.get_loc('sell_signal')] = 1
                result.iloc[i, result.columns.get_loc('vix_sell_signal')] = 1
        
        return result
    
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

    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算VIX恐慌指数的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取VIX值
        vix = indicator_data['vix'].fillna(0)
        vix_smooth = indicator_data['vix_smooth'].fillna(0)
        
        # 1. VIX水平评分（-30到+30分）
        # 高恐慌（VIX>30）通常是买入机会
        high_panic_mask = vix > 30
        score.loc[high_panic_mask] += 20
        
        # 极度恐慌（VIX>50）是强烈买入信号
        extreme_panic_mask = vix > 50
        score.loc[extreme_panic_mask] += 30
        
        # 低恐慌（VIX<15）通常是风险信号
        low_panic_mask = vix < 15
        score.loc[low_panic_mask] -= 10
        
        # 极度乐观（VIX<10）是强烈风险信号
        extreme_optimism_mask = vix < 10
        score.loc[extreme_optimism_mask] -= 20
        
        # 2. VIX趋势评分（-25到+25分）
        vix_change = vix.pct_change().fillna(0)
        
        # VIX快速上升（恐慌增加）是买入机会
        rapid_rise_mask = vix_change > 0.3
        score.loc[rapid_rise_mask] += 25
        
        # VIX上升
        rise_mask = vix_change > 0.1
        score.loc[rise_mask] += 15
        
        # VIX快速下降（恐慌减少）可能是风险信号
        rapid_fall_mask = vix_change < -0.2
        score.loc[rapid_fall_mask] -= 15
        
        # VIX下降
        fall_mask = vix_change < -0.1
        score.loc[fall_mask] -= 10
        
        # 3. VIX反转评分（-25到+25分）
        # 检测VIX从高位回落（买入信号）
        if len(vix) >= 5:
            for i in range(4, len(vix)):
                # VIX见顶回落
                if (vix.iloc[i-2] < vix.iloc[i-1] and 
                    vix.iloc[i] < vix.iloc[i-1] and 
                    vix.iloc[i-1] > 25):
                    score.iloc[i] += 20
                
                # VIX见底回升（风险信号）
                if (vix.iloc[i-2] > vix.iloc[i-1] and 
                    vix.iloc[i] > vix.iloc[i-1] and 
                    vix.iloc[i-1] < 15):
                    score.iloc[i] -= 15
        
        # 4. VIX相对位置评分（-15到+15分）
        # 计算VIX的历史分位数
        if len(vix) >= 60:
            vix_60_max = vix.rolling(window=60).max()
            vix_60_min = vix.rolling(window=60).min()
            vix_percentile = (vix - vix_60_min) / (vix_60_max - vix_60_min)
            vix_percentile = vix_percentile.fillna(0.5)
            
            # VIX处于历史高位（买入机会）
            high_percentile_mask = vix_percentile > 0.8
            score.loc[high_percentile_mask] += 15
            
            # VIX处于历史低位（风险信号）
            low_percentile_mask = vix_percentile < 0.2
            score.loc[low_percentile_mask] -= 15
        
        # 5. VIX与价格背离评分（-25到+25分）
        if 'close' in data.columns:
            close_price = data['close']
            price_change = close_price.pct_change().fillna(0)
            
            # 检测背离
            for i in range(5, len(vix)):
                # 价格下跌但VIX下降（负背离，风险信号）
                if (price_change.iloc[i] < -0.02 and 
                    vix_change.iloc[i] < -0.1):
                    score.iloc[i] -= 25
                
                # 价格上涨但VIX上升（正背离，买入机会）
                if (price_change.iloc[i] > 0.02 and 
                    vix_change.iloc[i] > 0.1):
                    score.iloc[i] += 25
        
        # 6. VIX平滑线评分（-10到+10分）
        # VIX与平滑线的关系
        vix_above_smooth = vix > vix_smooth
        score.loc[vix_above_smooth] += 8
        
        vix_below_smooth = vix < vix_smooth
        score.loc[vix_below_smooth] -= 5
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别VIX恐慌指数相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 10:
            return patterns
        
        # 获取VIX数据
        vix = indicator_data['vix']
        vix_smooth = indicator_data['vix_smooth']
        
        # 获取最新数据
        latest_vix = vix.iloc[-1]
        latest_vix_smooth = vix_smooth.iloc[-1]
        
        # 1. VIX水平形态
        if pd.notna(latest_vix):
            if latest_vix > 50:
                patterns.append("极度恐慌")
            elif latest_vix > 30:
                patterns.append("高度恐慌")
            elif latest_vix > 20:
                patterns.append("中度恐慌")
            elif latest_vix < 10:
                patterns.append("极度乐观")
            elif latest_vix < 15:
                patterns.append("低恐慌")
        
        # 2. VIX趋势形态
        if len(vix) >= 5:
            recent_vix = vix.tail(5)
            vix_trend = recent_vix.iloc[-1] - recent_vix.iloc[0]
            
            if vix_trend > recent_vix.mean() * 0.3:
                patterns.append("VIX快速上升")
            elif vix_trend > recent_vix.mean() * 0.1:
                patterns.append("VIX上升")
            elif vix_trend < -recent_vix.mean() * 0.3:
                patterns.append("VIX快速下降")
            elif vix_trend < -recent_vix.mean() * 0.1:
                patterns.append("VIX下降")
        
        # 3. VIX反转形态
        if len(vix) >= 5:
            # VIX见顶回落
            if (vix.iloc[-3] < vix.iloc[-2] and 
                vix.iloc[-1] < vix.iloc[-2] and 
                vix.iloc[-2] > 25):
                patterns.append("VIX见顶回落")
            
            # VIX见底回升
            if (vix.iloc[-3] > vix.iloc[-2] and 
                vix.iloc[-1] > vix.iloc[-2] and 
                vix.iloc[-2] < 15):
                patterns.append("VIX见底回升")
        
        # 4. VIX极值形态
        if len(vix) >= 20:
            vix_20_max = vix.tail(20).max()
            vix_20_min = vix.tail(20).min()
            
            if pd.notna(latest_vix):
                if latest_vix >= vix_20_max:
                    patterns.append("VIX创20日新高")
                elif latest_vix <= vix_20_min:
                    patterns.append("VIX创20日新低")
        
        # 5. VIX与平滑线关系
        if pd.notna(latest_vix) and pd.notna(latest_vix_smooth):
            if latest_vix > latest_vix_smooth * 1.2:
                patterns.append("VIX大幅高于平滑线")
            elif latest_vix > latest_vix_smooth:
                patterns.append("VIX高于平滑线")
            elif latest_vix < latest_vix_smooth * 0.8:
                patterns.append("VIX大幅低于平滑线")
            elif latest_vix < latest_vix_smooth:
                patterns.append("VIX低于平滑线")
        
        # 6. VIX背离形态
        if 'close' in data.columns and len(data) >= 10:
            close_price = data['close']
            price_change = close_price.pct_change()
            vix_change = vix.pct_change()
            
            # 检测最近的背离
            if (pd.notna(price_change.iloc[-1]) and pd.notna(vix_change.iloc[-1])):
                if (price_change.iloc[-1] < -0.02 and vix_change.iloc[-1] < -0.1):
                    patterns.append("VIX负背离")
                elif (price_change.iloc[-1] > 0.02 and vix_change.iloc[-1] > 0.1):
                    patterns.append("VIX正背离")
        
        # 7. VIX历史分位数形态
        if len(vix) >= 60:
            vix_60_max = vix.tail(60).max()
            vix_60_min = vix.tail(60).min()
            if pd.notna(latest_vix) and vix_60_max > vix_60_min:
                vix_percentile = (latest_vix - vix_60_min) / (vix_60_max - vix_60_min)
                
                if vix_percentile > 0.9:
                    patterns.append("VIX历史高位")
                elif vix_percentile > 0.7:
                    patterns.append("VIX相对高位")
                elif vix_percentile < 0.1:
                    patterns.append("VIX历史低位")
                elif vix_percentile < 0.3:
                    patterns.append("VIX相对低位")
        
        return patterns

