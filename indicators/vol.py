#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
成交量(VOL)

市场活跃度、参与度直观体现
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class VOL(BaseIndicator):
    """
    成交量(VOL) (VOL)
    
    分类：量能类指标
    描述：市场活跃度、参与度直观体现
    """
    
    def __init__(self, period: int = 14):
        """
        初始化成交量(VOL)指标
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__()
        self.period = period
        self.name = "VOL"
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含VOL指标的DataFrame
        """
        return self.calculate(df)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量(VOL)指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含以下列：
                - volume: 成交量
                
        Returns:
            添加了VOL指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['volume']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 添加原始成交量
        df_copy['vol'] = df_copy['volume']
        
        # 计算成交量移动平均
        df_copy['vol_ma5'] = df_copy['volume'].rolling(window=5).mean()
        df_copy['vol_ma10'] = df_copy['volume'].rolling(window=10).mean()
        df_copy['vol_ma20'] = df_copy['volume'].rolling(window=20).mean()
        
        # 计算相对成交量（当前成交量与N日平均成交量的比值）
        df_copy['vol_ratio'] = df_copy['volume'] / df_copy['vol_ma5']
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成成交量(VOL)指标交易信号
        
        Args:
            df: 包含价格数据和VOL指标的DataFrame
            **kwargs: 额外参数
                vol_ratio_threshold: 相对成交量阈值，默认为1.5
                
        Returns:
            添加了信号列的DataFrame:
            - vol_signal: 1=放量信号, -1=缩量信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['vol', 'vol_ma5']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 获取参数
        vol_ratio_threshold = kwargs.get('vol_ratio_threshold', 1.5)  # 相对成交量阈值
        
        # 生成信号
        df_copy['vol_signal'] = 0
        
        # 放量信号（成交量大于N日平均的1.5倍）
        df_copy.loc[df_copy['vol_ratio'] > vol_ratio_threshold, 'vol_signal'] = 1
        
        # 缩量信号（成交量小于N日平均的0.5倍）
        df_copy.loc[df_copy['vol_ratio'] < 0.5, 'vol_signal'] = -1
        
        return df_copy
    
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
        
    def plot(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制成交量(VOL)指标图表
        
        Args:
            df: 包含VOL指标的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['vol', 'vol_ma5', 'vol_ma10']
        self._validate_dataframe(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制VOL指标线
        ax.bar(df.index, df['vol'], label='成交量', alpha=0.3, color='gray')
        ax.plot(df.index, df['vol_ma5'], label='5日均量', color='red')
        ax.plot(df.index, df['vol_ma10'], label='10日均量', color='blue')
        ax.plot(df.index, df['vol_ma20'], label='20日均量', color='green')
        
        ax.set_ylabel('成交量')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax

    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 获取成交量数据
        volume = indicator_data['vol']
        vol_ma5 = indicator_data['vol_ma5']
        vol_ma10 = indicator_data['vol_ma10']
        vol_ma20 = indicator_data['vol_ma20']
        vol_ratio = indicator_data['vol_ratio'].fillna(1.0)
        
        # 1. 成交量水平评分（-15到+25分）
        # 放量加分，缩量减分
        high_volume_mask = vol_ratio > 1.5
        score.loc[high_volume_mask] += 15
        
        very_high_volume_mask = vol_ratio > 2.0
        score.loc[very_high_volume_mask] += 25
        
        low_volume_mask = vol_ratio < 0.7
        score.loc[low_volume_mask] -= 10
        
        very_low_volume_mask = vol_ratio < 0.5
        score.loc[very_low_volume_mask] -= 15
        
        # 2. 量价配合评分（-20到+25分）
        if 'close' in data.columns:
            close_price = data['close']
            price_change = close_price.pct_change().fillna(0)
            
            # 价涨量增（理想状态）
            price_up_vol_up = (price_change > 0.02) & (vol_ratio > 1.2)
            score.loc[price_up_vol_up] += 20
            
            # 价涨量增（强势）
            strong_price_up_vol_up = (price_change > 0.05) & (vol_ratio > 1.5)
            score.loc[strong_price_up_vol_up] += 25
            
            # 价跌量增（警告信号）
            price_down_vol_up = (price_change < -0.02) & (vol_ratio > 1.2)
            score.loc[price_down_vol_up] -= 15
            
            # 价跌量增（恐慌信号）
            panic_price_down_vol_up = (price_change < -0.05) & (vol_ratio > 1.5)
            score.loc[panic_price_down_vol_up] -= 20
            
            # 价涨量缩（警告信号）
            price_up_vol_down = (price_change > 0.02) & (vol_ratio < 0.8)
            score.loc[price_up_vol_down] -= 10
            
            # 价跌量缩（可能见底）
            price_down_vol_down = (price_change < -0.02) & (vol_ratio < 0.8)
            score.loc[price_down_vol_down] += 10
        
        # 3. 成交量趋势评分（-15到+15分）
        # 计算成交量趋势
        vol_trend_5 = volume.rolling(window=5).mean().pct_change().fillna(0)
        vol_trend_10 = volume.rolling(window=10).mean().pct_change().fillna(0)
        
        # 持续放量
        sustained_volume_up = (vol_trend_5 > 0.1) & (vol_trend_10 > 0.05)
        score.loc[sustained_volume_up] += 10
        
        # 急剧放量
        sharp_volume_up = vol_trend_5 > 0.3
        score.loc[sharp_volume_up] += 15
        
        # 持续缩量
        sustained_volume_down = (vol_trend_5 < -0.1) & (vol_trend_10 < -0.05)
        score.loc[sustained_volume_down] -= 10
        
        # 急剧缩量
        sharp_volume_down = vol_trend_5 < -0.3
        score.loc[sharp_volume_down] -= 15
        
        # 4. 成交量突破评分（-10到+25分）
        # 突破性放量（成交量创近期新高）
        if len(volume) >= 20:
            vol_20_max = volume.rolling(window=20).max()
            vol_20_avg = volume.rolling(window=20).mean()
            
            # 创新高且大幅放量
            breakthrough_volume = (volume >= vol_20_max) & (volume > vol_20_avg * 2)
            score.loc[breakthrough_volume] += 25
            
            # 一般突破性放量
            moderate_breakthrough = (volume >= vol_20_max * 0.9) & (volume > vol_20_avg * 1.5)
            score.loc[moderate_breakthrough] += 15
        
        # 5. 成交量异常评分（-20到+20分）
        # 计算成交量的标准差
        if len(volume) >= 20:
            vol_std = volume.rolling(window=20).std()
            vol_mean = volume.rolling(window=20).mean()
            
            # 异常放量（超过2个标准差）
            abnormal_high_vol = volume > (vol_mean + 2 * vol_std)
            score.loc[abnormal_high_vol] += 20
            
            # 异常缩量（低于2个标准差）
            abnormal_low_vol = volume < (vol_mean - 2 * vol_std)
            score.loc[abnormal_low_vol] -= 15
            
            # 极端异常放量（超过3个标准差）
            extreme_high_vol = volume > (vol_mean + 3 * vol_std)
            score.loc[extreme_high_vol] += 15  # 额外加分
            
            # 极端异常缩量（低于3个标准差）
            extreme_low_vol = volume < (vol_mean - 3 * vol_std)
            score.loc[extreme_low_vol] -= 20
        
        # 6. 成交量均线关系评分（-10到+10分）
        # 短期均线高于长期均线
        vol_ma_bullish = (vol_ma5 > vol_ma10) & (vol_ma10 > vol_ma20)
        vol_ma_bullish = vol_ma_bullish.fillna(False)
        score.loc[vol_ma_bullish] += 8
        
        # 短期均线低于长期均线
        vol_ma_bearish = (vol_ma5 < vol_ma10) & (vol_ma10 < vol_ma20)
        vol_ma_bearish = vol_ma_bearish.fillna(False)
        score.loc[vol_ma_bearish] -= 8
        
        # 成交量在均线之上
        vol_above_ma = volume > vol_ma5
        vol_above_ma = vol_above_ma.fillna(False)
        score.loc[vol_above_ma] += 5
        
        # 成交量在均线之下
        vol_below_ma = volume < vol_ma5
        vol_below_ma = vol_below_ma.fillna(False)
        score.loc[vol_below_ma] -= 5
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别成交量相关的技术形态
        
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
        
        # 获取成交量数据
        volume = indicator_data['vol']
        vol_ma5 = indicator_data['vol_ma5']
        vol_ma10 = indicator_data['vol_ma10']
        vol_ma20 = indicator_data['vol_ma20']
        vol_ratio = indicator_data['vol_ratio']
        
        # 获取最新数据
        latest_vol = volume.iloc[-1]
        latest_vol_ratio = vol_ratio.iloc[-1]
        
        # 1. 成交量水平形态
        if pd.notna(latest_vol_ratio):
            if latest_vol_ratio > 3.0:
                patterns.append("巨量")
            elif latest_vol_ratio > 2.0:
                patterns.append("大幅放量")
            elif latest_vol_ratio > 1.5:
                patterns.append("放量")
            elif latest_vol_ratio < 0.3:
                patterns.append("极度缩量")
            elif latest_vol_ratio < 0.5:
                patterns.append("明显缩量")
            elif latest_vol_ratio < 0.7:
                patterns.append("缩量")
        
        # 2. 量价配合形态
        if 'close' in data.columns and len(data) >= 2:
            close_price = data['close']
            price_change = close_price.pct_change().iloc[-1]
            
            if pd.notna(price_change) and pd.notna(latest_vol_ratio):
                if price_change > 0.02 and latest_vol_ratio > 1.2:
                    if latest_vol_ratio > 2.0:
                        patterns.append("价涨量增强势")
                    else:
                        patterns.append("价涨量增")
                elif price_change < -0.02 and latest_vol_ratio > 1.2:
                    patterns.append("价跌量增")
                elif price_change > 0.02 and latest_vol_ratio < 0.8:
                    patterns.append("价涨量缩")
                elif price_change < -0.02 and latest_vol_ratio < 0.8:
                    patterns.append("价跌量缩")
        
        # 3. 成交量趋势形态
        if len(volume) >= 5:
            recent_vol_trend = volume.tail(5).pct_change().mean()
            
            if pd.notna(recent_vol_trend):
                if recent_vol_trend > 0.2:
                    patterns.append("成交量持续放大")
                elif recent_vol_trend > 0.1:
                    patterns.append("成交量温和放大")
                elif recent_vol_trend < -0.2:
                    patterns.append("成交量持续萎缩")
                elif recent_vol_trend < -0.1:
                    patterns.append("成交量温和萎缩")
        
        # 4. 成交量突破形态
        if len(volume) >= 20:
            vol_20_max = volume.tail(20).max()
            vol_20_avg = volume.tail(20).mean()
            
            if pd.notna(latest_vol) and pd.notna(vol_20_max) and pd.notna(vol_20_avg):
                if latest_vol >= vol_20_max:
                    patterns.append("成交量创新高")
                elif latest_vol >= vol_20_max * 0.9:
                    patterns.append("成交量接近新高")
                
                if latest_vol > vol_20_avg * 2:
                    patterns.append("突破性放量")
        
        # 5. 成交量异常形态
        if len(volume) >= 20:
            vol_std = volume.tail(20).std()
            vol_mean = volume.tail(20).mean()
            
            if pd.notna(latest_vol) and pd.notna(vol_std) and pd.notna(vol_mean):
                if latest_vol > vol_mean + 3 * vol_std:
                    patterns.append("极端异常放量")
                elif latest_vol > vol_mean + 2 * vol_std:
                    patterns.append("异常放量")
                elif latest_vol < vol_mean - 3 * vol_std:
                    patterns.append("极端异常缩量")
                elif latest_vol < vol_mean - 2 * vol_std:
                    patterns.append("异常缩量")
        
        # 6. 成交量均线形态
        if (pd.notna(vol_ma5.iloc[-1]) and pd.notna(vol_ma10.iloc[-1]) and 
            pd.notna(vol_ma20.iloc[-1])):
            
            if vol_ma5.iloc[-1] > vol_ma10.iloc[-1] > vol_ma20.iloc[-1]:
                patterns.append("成交量多头排列")
            elif vol_ma5.iloc[-1] < vol_ma10.iloc[-1] < vol_ma20.iloc[-1]:
                patterns.append("成交量空头排列")
            
            if latest_vol > vol_ma5.iloc[-1]:
                patterns.append("成交量高于短期均线")
            elif latest_vol < vol_ma5.iloc[-1]:
                patterns.append("成交量低于短期均线")
        
        # 7. 成交量变化形态
        if len(volume) >= 3:
            # 连续放量
            recent_3_vol = volume.tail(3)
            if (recent_3_vol.iloc[-1] > recent_3_vol.iloc[-2] and 
                recent_3_vol.iloc[-2] > recent_3_vol.iloc[-3]):
                patterns.append("连续放量")
            
            # 连续缩量
            if (recent_3_vol.iloc[-1] < recent_3_vol.iloc[-2] and 
                recent_3_vol.iloc[-2] < recent_3_vol.iloc[-3]):
                patterns.append("连续缩量")
        
        # 8. 成交量极值形态
        if len(volume) >= 60:
            vol_60_max = volume.tail(60).max()
            vol_60_min = volume.tail(60).min()
            
            if pd.notna(latest_vol):
                if latest_vol >= vol_60_max:
                    patterns.append("60日成交量新高")
                elif latest_vol <= vol_60_min:
                    patterns.append("60日成交量新低")
        
        return patterns

