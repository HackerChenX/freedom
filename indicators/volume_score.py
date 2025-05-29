"""
成交量评分指标

基于统一评分框架的成交量指标，提供评分、形态识别和信号生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.scoring_framework import IndicatorScoreBase, PatternRecognitionMixin
from indicators.common import ma
from utils.logger import get_logger

logger = get_logger(__name__)


class VolumeScore(IndicatorScoreBase, PatternRecognitionMixin):
    """
    成交量评分指标
    
    基于成交量相关指标的评分系统，包含OBV、VR等指标的综合评分
    """
    
    def __init__(self, period: int = 20, weight: float = 1.0):
        """
        初始化成交量评分指标
        
        Args:
            period: 计算周期
            weight: 指标权重
        """
        super().__init__(name="Volume", weight=weight)
        self.period = period
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算成交量原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. OBV评分（30分）
        obv_score = self._calculate_obv_score(data)
        score += obv_score
        
        # 2. 成交量比率评分（25分）
        vr_score = self._calculate_vr_score(data)
        score += vr_score
        
        # 3. 量价配合评分（20分）
        volume_price_score = self._calculate_volume_price_score(data)
        score += volume_price_score
        
        # 4. 成交量形态评分（15分）
        pattern_score = self._calculate_volume_pattern_score(data)
        score += pattern_score
        
        # 确保评分在0-100范围内
        return np.clip(score, 0, 100)
    
    def _calculate_obv_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算OBV评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: OBV评分（-30到30分）
        """
        score = pd.Series(0.0, index=data.index)
        
        # 计算OBV
        obv = self._calculate_obv(data['close'], data['volume'])
        
        # 计算OBV移动平均
        obv_ma = ma(obv.values, min(self.period, len(obv)))
        obv_ma_series = pd.Series(obv_ma, index=data.index)
        
        for i in range(1, len(data)):
            if i < self.period:
                continue
            
            current_obv = obv.iloc[i]
            prev_obv = obv.iloc[i-1]
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            
            # OBV与价格同向性评分
            obv_change = current_obv - prev_obv
            price_change = current_price - prev_price
            
            if obv_change > 0 and price_change > 0:
                score.iloc[i] += 15  # 量价齐升
            elif obv_change < 0 and price_change < 0:
                score.iloc[i] -= 15  # 量价齐跌
            elif obv_change > 0 and price_change < 0:
                score.iloc[i] += 10  # 价跌量增，可能是洗盘
            elif obv_change < 0 and price_change > 0:
                score.iloc[i] -= 10  # 价涨量缩，可能是假突破
            
            # OBV趋势评分
            if current_obv > obv_ma_series.iloc[i]:
                score.iloc[i] += 8   # OBV在均线上方
            else:
                score.iloc[i] -= 8   # OBV在均线下方
            
            # OBV突破评分
            if i >= 2:
                if (current_obv > obv_ma_series.iloc[i] and 
                    prev_obv <= obv_ma_series.iloc[i-1]):
                    score.iloc[i] += 12  # OBV上穿均线
                elif (current_obv < obv_ma_series.iloc[i] and 
                      prev_obv >= obv_ma_series.iloc[i-1]):
                    score.iloc[i] -= 12  # OBV下穿均线
        
        return score
    
    def _calculate_vr_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算VR（成交量比率）评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: VR评分（-25到25分）
        """
        score = pd.Series(0.0, index=data.index)
        
        # 计算VR指标
        vr = self._calculate_vr(data)
        
        for i in range(self.period, len(data)):
            current_vr = vr.iloc[i]
            
            # VR区间评分
            if current_vr < 70:
                score.iloc[i] += 20  # 超卖区域
            elif current_vr < 100:
                score.iloc[i] += 10  # 偏低区域
            elif current_vr > 200:
                score.iloc[i] -= 20  # 超买区域
            elif current_vr > 150:
                score.iloc[i] -= 10  # 偏高区域
            
            # VR趋势评分
            if i >= self.period + 5:
                vr_trend = current_vr - vr.iloc[i-5]
                if vr_trend > 20:
                    score.iloc[i] += 8   # VR快速上升
                elif vr_trend < -20:
                    score.iloc[i] -= 8   # VR快速下降
        
        return score
    
    def _calculate_volume_price_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算量价配合评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 量价配合评分（-20到20分）
        """
        score = pd.Series(0.0, index=data.index)
        
        # 计算成交量移动平均
        volume_ma = ma(data['volume'].values, self.period)
        volume_ma_series = pd.Series(volume_ma, index=data.index)
        
        for i in range(1, len(data)):
            if i < self.period:
                continue
            
            current_price = data['close'].iloc[i]
            prev_price = data['close'].iloc[i-1]
            current_volume = data['volume'].iloc[i]
            avg_volume = volume_ma_series.iloc[i]
            
            price_change_pct = (current_price - prev_price) / prev_price
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 量价配合评分
            if price_change_pct > 0.02:  # 价格上涨超过2%
                if volume_ratio > 1.5:
                    score.iloc[i] += 15  # 放量上涨
                elif volume_ratio > 1.2:
                    score.iloc[i] += 10  # 温和放量上涨
                elif volume_ratio < 0.8:
                    score.iloc[i] -= 5   # 缩量上涨，可能乏力
            elif price_change_pct < -0.02:  # 价格下跌超过2%
                if volume_ratio > 1.5:
                    score.iloc[i] -= 15  # 放量下跌
                elif volume_ratio > 1.2:
                    score.iloc[i] -= 10  # 温和放量下跌
                elif volume_ratio < 0.8:
                    score.iloc[i] += 5   # 缩量下跌，可能是技术调整
            
            # 突破确认
            if i >= 5:
                recent_high = data['high'].iloc[i-4:i].max()
                recent_low = data['low'].iloc[i-4:i].min()
                
                if current_price > recent_high and volume_ratio > 1.3:
                    score.iloc[i] += 12  # 放量突破新高
                elif current_price < recent_low and volume_ratio > 1.3:
                    score.iloc[i] -= 12  # 放量跌破新低
        
        return score
    
    def _calculate_volume_pattern_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量形态评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 成交量形态评分（-15到15分）
        """
        score = pd.Series(0.0, index=data.index)
        
        # 计算成交量移动平均
        volume_ma = ma(data['volume'].values, self.period)
        volume_ma_series = pd.Series(volume_ma, index=data.index)
        
        for i in range(self.period, len(data)):
            current_volume = data['volume'].iloc[i]
            avg_volume = volume_ma_series.iloc[i]
            
            # 天量检测
            if i >= 20:
                recent_max_volume = data['volume'].iloc[i-19:i].max()
                if current_volume > recent_max_volume * 1.5:
                    # 检查是否是天量天价或天量地价
                    recent_max_price = data['high'].iloc[i-4:i+1].max()
                    recent_min_price = data['low'].iloc[i-4:i+1].min()
                    current_price = data['close'].iloc[i]
                    
                    if current_price >= recent_max_price * 0.98:
                        score.iloc[i] -= 10  # 天量天价，可能见顶
                    elif current_price <= recent_min_price * 1.02:
                        score.iloc[i] += 10  # 天量地价，可能见底
            
            # 地量检测
            if current_volume < avg_volume * 0.5:
                score.iloc[i] += 5  # 地量，可能是变盘前兆
            
            # 成交量递增形态
            if i >= 3:
                volume_trend = all(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] 
                                 for j in range(3))
                if volume_trend:
                    price_trend = data['close'].iloc[i] > data['close'].iloc[i-3]
                    if price_trend:
                        score.iloc[i] += 8  # 量价齐升
                    else:
                        score.iloc[i] -= 5  # 量增价跌，可能是出货
        
        return score
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算OBV指标
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            
        Returns:
            pd.Series: OBV值
        """
        obv = pd.Series(0.0, index=close.index)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vr(self, data: pd.DataFrame) -> pd.Series:
        """
        计算VR指标
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: VR值
        """
        vr = pd.Series(100.0, index=data.index)
        
        for i in range(self.period, len(data)):
            up_volume = 0
            down_volume = 0
            equal_volume = 0
            
            for j in range(i - self.period + 1, i + 1):
                if data['close'].iloc[j] > data['close'].iloc[j-1]:
                    up_volume += data['volume'].iloc[j]
                elif data['close'].iloc[j] < data['close'].iloc[j-1]:
                    down_volume += data['volume'].iloc[j]
                else:
                    equal_volume += data['volume'].iloc[j]
            
            total_down = down_volume + equal_volume / 2
            if total_down > 0:
                vr.iloc[i] = (up_volume + equal_volume / 2) / total_down * 100
            else:
                vr.iloc[i] = 100
        
        return vr
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别成交量技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算相关指标
        obv = self._calculate_obv(data['close'], data['volume'])
        vr = self._calculate_vr(data)
        volume_ma = ma(data['volume'].values, self.period)
        volume_ma_series = pd.Series(volume_ma, index=data.index)
        
        # 检测放量突破
        volume_breakout = data['volume'] > volume_ma_series * 1.5
        price_breakout = data['close'] > data['close'].rolling(10).max().shift(1)
        
        if (volume_breakout & price_breakout).any():
            patterns.append("放量突破")
        
        # 检测缩量整理
        volume_shrink = data['volume'] < volume_ma_series * 0.7
        price_consolidation = (data['high'] - data['low']) / data['close'] < 0.03
        
        if (volume_shrink & price_consolidation).any():
            patterns.append("缩量整理")
        
        # 检测量价背离
        price_trend = data['close'].rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0])
        obv_trend = obv.rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0])
        
        price_up_obv_down = (price_trend > 0) & (obv_trend < 0)
        price_down_obv_up = (price_trend < 0) & (obv_trend > 0)
        
        if price_up_obv_down.any():
            patterns.append("量价负背离")
        if price_down_obv_up.any():
            patterns.append("量价正背离")
        
        # 检测VR极值
        vr_oversold = vr < 70
        vr_overbought = vr > 200
        
        if vr_oversold.any():
            patterns.append("VR超卖")
        if vr_overbought.any():
            patterns.append("VR超买")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成成交量交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 信号字典
        """
        signals = {}
        
        # 计算相关指标
        obv = self._calculate_obv(data['close'], data['volume'])
        vr = self._calculate_vr(data)
        volume_ma = ma(data['volume'].values, self.period)
        volume_ma_series = pd.Series(volume_ma, index=data.index)
        
        # 基本成交量信号
        signals['volume_surge'] = data['volume'] > volume_ma_series * 1.5
        signals['volume_dry'] = data['volume'] < volume_ma_series * 0.5
        
        # OBV信号
        obv_ma = ma(obv.values, self.period)
        obv_ma_series = pd.Series(obv_ma, index=data.index)
        
        signals['obv_bullish'] = obv > obv_ma_series
        signals['obv_bearish'] = obv < obv_ma_series
        signals['obv_cross_up'] = (obv > obv_ma_series) & (obv.shift(1) <= obv_ma_series.shift(1))
        signals['obv_cross_down'] = (obv < obv_ma_series) & (obv.shift(1) >= obv_ma_series.shift(1))
        
        # VR信号
        signals['vr_oversold'] = vr < 70
        signals['vr_overbought'] = vr > 200
        signals['vr_oversold_recovery'] = (vr > 70) & (vr.shift(1) <= 70)
        signals['vr_overbought_decline'] = (vr < 200) & (vr.shift(1) >= 200)
        
        # 量价配合信号
        price_up = data['close'] > data['close'].shift(1)
        price_down = data['close'] < data['close'].shift(1)
        volume_up = data['volume'] > volume_ma_series
        
        signals['volume_price_bullish'] = price_up & volume_up
        signals['volume_price_bearish'] = price_down & volume_up
        
        # 突破确认信号
        price_breakout_up = data['close'] > data['close'].rolling(10).max().shift(1)
        price_breakout_down = data['close'] < data['close'].rolling(10).min().shift(1)
        
        signals['volume_confirmed_breakout_up'] = price_breakout_up & signals['volume_surge']
        signals['volume_confirmed_breakout_down'] = price_breakout_down & signals['volume_surge']
        
        # 综合强势信号
        signals['strong_volume_buy'] = (signals['obv_cross_up'] | 
                                       signals['vr_oversold_recovery'] | 
                                       signals['volume_confirmed_breakout_up'])
        
        signals['strong_volume_sell'] = (signals['obv_cross_down'] | 
                                        signals['vr_overbought_decline'] | 
                                        signals['volume_confirmed_breakout_down'])
        
        return signals 