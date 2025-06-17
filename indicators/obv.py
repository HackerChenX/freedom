#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
能量潮(OBV)指标模块

实现能量潮指标的计算功能，用于判断资金流向与价格趋势的一致性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger
from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength, PatternPolarity
from indicators.common import crossover, crossunder
from utils.technical_utils import calculate_ma
from utils.decorators import log_calls, error_handling, cache_result

logger = get_logger(__name__)


class OBV(BaseIndicator):
    """
    能量潮(On Balance Volume)指标
    
    根据价格变动方向，计算成交量的累计值，用于判断资金流向与价格趋势的一致性
    """
    
    def __init__(self, ma_period: int = 30):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化OBV指标
        
        Args:
            ma_period: OBV均线周期，默认为30日
        """
        super().__init__(name="OBV", description="能量潮指标，根据价格变动方向，计算成交量的累计值")
        self.ma_period = ma_period
        self.crossover = crossover
        self.crossunder = crossunder
    
    def set_parameters(self, **kwargs):
        """设置指标参数，可设置 'ma_period'"""
        if 'ma_period' in kwargs:
            self.ma_period = int(kwargs['ma_period'])
    
    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算OBV指标的置信度。
        """
        return 0.5
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取OBV指标的技术形态
        """
        return pd.DataFrame(index=data.index)

    def register_patterns(self):
        """
        注册OBV指标的形态到全局形态注册表
        """
        # 注册OBV趋势形态
        self.register_pattern_to_registry(
            pattern_id="OBV_UPTREND",
            display_name="OBV上升趋势",
            description="OBV持续上升，表明买盘资金持续流入",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_DOWNTREND",
            display_name="OBV下降趋势",
            description="OBV持续下降，表明卖盘资金持续流出",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_SIDEWAYS",
            display_name="OBV横盘整理",
            description="OBV横盘整理，表明资金流向不明确",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册OBV均线交叉形态
        self.register_pattern_to_registry(
            pattern_id="OBV_GOLDEN_CROSS",
            display_name="OBV上穿均线",
            description="OBV上穿其均线，表明资金流入加速",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_DEATH_CROSS",
            display_name="OBV下穿均线",
            description="OBV下穿其均线，表明资金流出加速",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册OBV背离形态
        self.register_pattern_to_registry(
            pattern_id="OBV_BULLISH_DIVERGENCE",
            display_name="OBV正背离",
            description="价格创新低但OBV未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_BEARISH_DIVERGENCE",
            display_name="OBV负背离",
            description="价格创新高但OBV未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册OBV突破形态
        self.register_pattern_to_registry(
            pattern_id="OBV_BREAKOUT_HIGH",
            display_name="OBV突破前期高点",
            description="OBV突破前期高点，表明资金流入创新高",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=22.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_BREAKDOWN_LOW",
            display_name="OBV跌破前期低点",
            description="OBV跌破前期低点，表明资金流出创新低",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-22.0,
            polarity="NEGATIVE"
        )

        # 注册量价配合形态
        self.register_pattern_to_registry(
            pattern_id="OBV_VOLUME_PRICE_HARMONY",
            display_name="OBV量价配合良好",
            description="OBV与价格趋势一致，量价配合良好",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="OBV_VOLUME_PRICE_DIVERGENCE",
            display_name="OBV量价背离",
            description="OBV与价格趋势背离，需要谨慎",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,
            polarity="NEGATIVE"
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            添加了OBV指标列的DataFrame
        """
        if data.empty:
            return data
            
        # 确保数据包含必要的列
        required_columns = ['close', 'volume']
        
        # 验证输入数据
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少所需的列: {', '.join(missing_columns)}")
        
        df = data.copy()
        
        # 基础OBV计算
        df['price_change'] = df['close'].diff()
        df['obv'] = 0.0
        
        # 第一个值设为0
        df.loc[df.index[0], 'obv'] = 0
        
        # 迭代计算OBV
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['price_change'].iloc[i] < 0:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
        
        # 计算OBV均线
        df['obv_ma'] = df['obv'].rolling(window=self.ma_period).mean()
        
        # 计算OBV相对强弱
        if not df['obv'].empty and df['obv'].max() != df['obv'].min():
            df['obv_norm'] = (df['obv'] - df['obv'].min()) / (df['obv'].max() - df['obv'].min()) * 100
        else:
            df['obv_norm'] = 50.0  # 默认中性值
        
        # 计算OBV动量
        df['obv_momentum'] = df['obv'].diff(self.ma_period)
        
        # 计算OBV相对于均线的位置
        df['obv_position'] = df['obv'] - df['obv_ma']
        
        # OBV背离标志
        df['obv_divergence'] = 0
        
        # 保存结果
        self._result = df
        
        # 确保基础数据列被保留
        df = self._preserve_base_columns(data, df)
        
        return df
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算OBV原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算OBV
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        obv = self._result['obv']
        obv_ma = self._result['obv_ma']
        close = self._result['close']
        
        # 1. OBV趋势与价格趋势一致性评分
        obv_trend = obv > obv.shift(5)  # OBV上升趋势
        price_trend = close > close.shift(5)  # 价格上升趋势
        
        # 趋势一致+15分
        trend_consistency = (obv_trend & price_trend) | (~obv_trend & ~price_trend)
        score += trend_consistency * 15
        
        # 2. OBV与价格背离评分
        divergence_score = self._calculate_obv_divergence(close, obv)
        score += divergence_score
        
        # 3. OBV突破评分
        # OBV突破前期高点+20分
        obv_breakout_high = self._detect_obv_breakout(obv, direction='up')
        score += obv_breakout_high * 20
        
        # OBV跌破前期低点-20分
        obv_breakout_low = self._detect_obv_breakout(obv, direction='down')
        score -= obv_breakout_low * 20
        
        # 4. OBV均线交叉评分
        # OBV上穿均线+15分
        obv_cross_up_ma = self.crossover(obv, obv_ma)
        score += obv_cross_up_ma * 15
        
        # OBV下穿均线-15分
        obv_cross_down_ma = self.crossunder(obv, obv_ma)
        score -= obv_cross_down_ma * 15
        
        # 5. OBV能量强度评分
        volume_strength = self._calculate_volume_strength(data)
        score += volume_strength
        
        # 6. OBV斜率评分
        obv_slope = self._calculate_obv_slope(obv)
        score += obv_slope
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别OBV技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算OBV
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        obv = self._result['obv']
        obv_ma = self._result['obv_ma']
        close = self._result['close']
        
        # 检查最近的信号
        recent_periods = min(10, len(obv))
        if recent_periods == 0:
            return patterns
        
        recent_obv = obv.tail(recent_periods)
        recent_obv_ma = obv_ma.tail(recent_periods)
        recent_close = close.tail(recent_periods)
        
        # 1. OBV趋势形态
        if recent_obv.iloc[-1] > recent_obv.iloc[-5]:
            patterns.append("OBV上升趋势")
        elif recent_obv.iloc[-1] < recent_obv.iloc[-5]:
            patterns.append("OBV下降趋势")
        else:
            patterns.append("OBV横盘整理")
        
        # 2. OBV均线交叉形态
        if self.crossover(recent_obv, recent_obv_ma).any():
            patterns.append("OBV上穿均线")
        if self.crossunder(recent_obv, recent_obv_ma).any():
            patterns.append("OBV下穿均线")
        
        # 3. OBV背离形态
        divergence_type = self._detect_obv_divergence_pattern(recent_close, recent_obv)
        if divergence_type:
            patterns.append(f"OBV{divergence_type}")
        
        # 4. OBV突破形态
        if self._detect_obv_breakout(recent_obv, direction='up').any():
            patterns.append("OBV突破前期高点")
        if self._detect_obv_breakout(recent_obv, direction='down').any():
            patterns.append("OBV跌破前期低点")
        
        # 5. 量价配合形态
        volume_price_harmony = self._detect_volume_price_harmony(recent_close, recent_obv)
        if volume_price_harmony == 'positive':
            patterns.append("OBV量价配合良好")
        elif volume_price_harmony == 'negative':
            patterns.append("OBV量价背离")
        
        return patterns
    
    def _calculate_obv_divergence(self, price: pd.Series, obv: pd.Series) -> pd.Series:
        """
        计算OBV背离评分
        
        Args:
            price: 价格序列
            obv: OBV序列
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
        
        # 寻找价格和OBV的峰值谷值
        window = 5
        for i in range(window, len(price) - window):
            price_window = price.iloc[i-window:i+window+1]
            obv_window = obv.iloc[i-window:i+window+1]
            
            if price.iloc[i] == price_window.max():  # 价格峰值
                if obv.iloc[i] != obv_window.max():  # OBV未创新高
                    divergence_score.iloc[i:i+10] -= 25  # 负背离
            elif price.iloc[i] == price_window.min():  # 价格谷值
                if obv.iloc[i] != obv_window.min():  # OBV未创新低
                    divergence_score.iloc[i:i+10] += 25  # 正背离
        
        return divergence_score
    
    def _detect_obv_breakout(self, obv: pd.Series, direction: str, window: int = 20) -> pd.Series:
        """
        检测OBV突破
        
        Args:
            obv: OBV序列
            direction: 突破方向 ('up' 或 'down')
            window: 检测窗口
            
        Returns:
            pd.Series: 突破信号
        """
        breakout = pd.Series(False, index=obv.index)
        
        if len(obv) < window:
            return breakout
        
        if direction == 'up':
            # 突破前期高点
            rolling_max = obv.rolling(window=window).max()
            breakout = obv > rolling_max.shift(1)
        elif direction == 'down':
            # 跌破前期低点
            rolling_min = obv.rolling(window=window).min()
            breakout = obv < rolling_min.shift(1)
        
        return breakout
    
    def _calculate_volume_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量强度评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 成交量强度评分
        """
        volume_score = pd.Series(0.0, index=data.index)
        
        if 'volume' not in data.columns:
            return volume_score
        
        volume = data['volume']
        
        # 成交量相对强度
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma
        
        # 成交量放大+10分
        volume_score += np.where(volume_ratio > 1.5, 10, 0)
        # 成交量萎缩-5分
        volume_score -= np.where(volume_ratio < 0.5, 5, 0)
        
        return volume_score
    
    def _calculate_obv_slope(self, obv: pd.Series) -> pd.Series:
        """
        计算OBV斜率评分
        
        Args:
            obv: OBV序列
            
        Returns:
            pd.Series: 斜率评分
        """
        slope_score = pd.Series(0.0, index=obv.index)
        
        if len(obv) < 5:
            return slope_score
        
        # 计算5周期OBV斜率
        obv_slope = obv.diff(5)
        
        # 标准化斜率
        obv_std = obv.rolling(window=20).std()
        normalized_slope = obv_slope / obv_std
        
        # 斜率评分
        slope_score += np.where(normalized_slope > 1, 10, 0)   # 强烈上升+10分
        slope_score += np.where(normalized_slope > 0.5, 5, 0)  # 温和上升+5分
        slope_score -= np.where(normalized_slope < -1, 10, 0)  # 强烈下降-10分
        slope_score -= np.where(normalized_slope < -0.5, 5, 0) # 温和下降-5分
        
        return slope_score
    
    def _detect_obv_divergence_pattern(self, price: pd.Series, obv: pd.Series) -> Optional[str]:
        """
        检测OBV背离形态
        
        Args:
            price: 价格序列
            obv: OBV序列
            
        Returns:
            Optional[str]: 背离类型或None
        """
        if len(price) < 10:
            return None
        
        # 寻找最近的峰值和谷值
        price_extremes = []
        obv_extremes = []
        
        # 简化的极值检测
        for i in range(2, len(price) - 2):
            if (price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1]):
                price_extremes.append(price.iloc[i])
                obv_extremes.append(obv.iloc[i])
            elif (price.iloc[i] < price.iloc[i-1] and price.iloc[i] < price.iloc[i+1]):
                price_extremes.append(price.iloc[i])
                obv_extremes.append(obv.iloc[i])
        
        if len(price_extremes) >= 2:
            price_trend = price_extremes[-1] - price_extremes[-2]
            obv_trend = obv_extremes[-1] - obv_extremes[-2]
            
            # 正背离：价格创新低但OBV未创新低
            if price_trend < -0.01 and obv_trend > 0:
                return "正背离"
            # 负背离：价格创新高但OBV未创新高
            elif price_trend > 0.01 and obv_trend < 0:
                return "负背离"
        
        return None
    
    def _detect_volume_price_harmony(self, price: pd.Series, obv: pd.Series) -> Optional[str]:
        """
        检测量价配合情况
        
        Args:
            price: 价格序列
            obv: OBV序列
            
        Returns:
            Optional[str]: 配合情况
        """
        if len(price) < 5:
            return None
        
        # 计算价格和OBV的趋势
        price_trend = price.iloc[-1] - price.iloc[-5]
        obv_trend = obv.iloc[-1] - obv.iloc[-5]
        
        # 量价配合判断
        if price_trend > 0 and obv_trend > 0:
            return 'positive'  # 价涨量增
        elif price_trend < 0 and obv_trend < 0:
            return 'positive'  # 价跌量减
        elif (price_trend > 0 and obv_trend < 0) or (price_trend < 0 and obv_trend > 0):
            return 'negative'  # 量价背离
        
        return None

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成OBV信号
        
        Args:
            data: 输入数据，包含OBV指标
            
        Returns:
            pd.DataFrame: 包含OBV信号的数据框
        """
        if "obv" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["obv_trend"] = np.nan
        data["obv_divergence"] = np.nan
        
        # 计算OBV趋势
        for i in range(5, len(data)):
            # 检查OBV短期趋势
            if data["obv"].iloc[i] > data["obv"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("obv_trend")] = 1  # OBV上升
            elif data["obv"].iloc[i] < data["obv"].iloc[i-5]:
                data.iloc[i, data.columns.get_loc("obv_trend")] = -1  # OBV下降
            else:
                data.iloc[i, data.columns.get_loc("obv_trend")] = 0  # OBV横盘
        
        # 计算OBV与价格的背离
        for i in range(20, len(data)):
            # 判断近期是否创新高或新低
            is_price_high = data["close"].iloc[i] >= np.max(data["close"].iloc[i-20:i])
            is_price_low = data["close"].iloc[i] <= np.min(data["close"].iloc[i-20:i])
            
            is_obv_high = data["obv"].iloc[i] >= np.max(data["obv"].iloc[i-20:i])
            is_obv_low = data["obv"].iloc[i] <= np.min(data["obv"].iloc[i-20:i])
            
            # 价格创新高但OBV未创新高 -> 负背离
            if is_price_high and not is_obv_high:
                data.iloc[i, data.columns.get_loc("obv_divergence")] = -1
            
            # 价格创新低但OBV未创新低 -> 正背离
            elif is_price_low and not is_obv_low:
                data.iloc[i, data.columns.get_loc("obv_divergence")] = 1
            
            # 无背离
            else:
                data.iloc[i, data.columns.get_loc("obv_divergence")] = 0
        
        # 检测OBV突破OBV均线
        data["obv_ma_cross"] = np.nan
        
        for i in range(1, len(data)):
            if pd.notna(data["obv_ma"].iloc[i]) and pd.notna(data["obv_ma"].iloc[i-1]):
                # OBV上穿均线
                if data["obv"].iloc[i] > data["obv_ma"].iloc[i] and data["obv"].iloc[i-1] <= data["obv_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("obv_ma_cross")] = 1
                
                # OBV下穿均线
                elif data["obv"].iloc[i] < data["obv_ma"].iloc[i] and data["obv"].iloc[i-1] >= data["obv_ma"].iloc[i-1]:
                    data.iloc[i, data.columns.get_loc("obv_ma_cross")] = -1
                
                # 无交叉
                else:
                    data.iloc[i, data.columns.get_loc("obv_ma_cross")] = 0
        
        return data
    
    def get_obv_strength(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算OBV强度
        
        Args:
            data: 输入数据，包含OBV指标
            window: 计算窗口期，默认为20日
            
        Returns:
            pd.DataFrame: 包含OBV强度的数据框
        """
        if "obv" not in data.columns:
            data = self.calculate(data)
        
        # 计算OBV变化率
        data["obv_change_rate"] = np.nan
        
        for i in range(window, len(data)):
            # OBV变化率 = (当前OBV - N日前OBV) / N日前OBV
            data.iloc[i, data.columns.get_loc("obv_change_rate")] = (
                (data["obv"].iloc[i] - data["obv"].iloc[i-window]) / 
                abs(data["obv"].iloc[i-window]) if data["obv"].iloc[i-window] != 0 else 0
            )
        
        # 计算OBV强度
        data["obv_strength"] = np.nan
        
        for i in range(window, len(data)):
            # OBV强度 = OBV变化率 / 价格变化率
            price_change_rate = (data["close"].iloc[i] - data["close"].iloc[i-window]) / data["close"].iloc[i-window]
            
            if price_change_rate != 0:
                data.iloc[i, data.columns.get_loc("obv_strength")] = data["obv_change_rate"].iloc[i] / price_change_rate
            else:
                data.iloc[i, data.columns.get_loc("obv_strength")] = 0
        
        return data

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成能量潮(OBV)指标信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算OBV指标
        result = self.calculate(data)
        
        # 计算评分
        scores = self.calculate_raw_score(data)
        signals['score'] = scores
        
        # 识别形态
        patterns = {}
        for i, date in enumerate(data.index):
            if i < 20:  # 需要足够的历史数据来识别形态
                continue
            patterns[date] = self.identify_patterns(data.iloc[:i+1])
        
        # 基于评分和形态生成买卖信号
        for date, score in scores.items():
            if date not in data.index:
                continue
            
            # 获取当前形态
            current_patterns = patterns.get(date, [])
            
            # 设置信号类型和描述
            if score >= 70:
                signals.loc[date, 'buy_signal'] = True
                signals.loc[date, 'neutral_signal'] = False
                signals.loc[date, 'trend'] = 1
                
                # 根据不同形态设置信号类型和描述
                if "OBV正背离" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_bullish_divergence'
                    signals.loc[date, 'signal_desc'] = 'OBV正背离买入信号'
                    signals.loc[date, 'confidence'] = 80.0
                    signals.loc[date, 'position_size'] = 0.6
                elif "OBV上穿均线" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_golden_cross'
                    signals.loc[date, 'signal_desc'] = 'OBV上穿均线买入信号'
                    signals.loc[date, 'confidence'] = 70.0
                    signals.loc[date, 'position_size'] = 0.5
                elif "OBV突破前期高点" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_breakout'
                    signals.loc[date, 'signal_desc'] = 'OBV突破前期高点买入信号'
                    signals.loc[date, 'confidence'] = 75.0
                    signals.loc[date, 'position_size'] = 0.5
                elif "OBV量价配合良好" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_volume_price_harmony'
                    signals.loc[date, 'signal_desc'] = 'OBV量价配合良好买入信号'
                    signals.loc[date, 'confidence'] = 65.0
                    signals.loc[date, 'position_size'] = 0.4
                else:
                    signals.loc[date, 'signal_type'] = 'obv_bullish'
                    signals.loc[date, 'signal_desc'] = 'OBV看涨信号'
                    signals.loc[date, 'confidence'] = 60.0
                    signals.loc[date, 'position_size'] = 0.3
            elif score <= 30:
                signals.loc[date, 'sell_signal'] = True
                signals.loc[date, 'neutral_signal'] = False
                signals.loc[date, 'trend'] = -1
                
                # 根据不同形态设置信号类型和描述
                if "OBV负背离" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_bearish_divergence'
                    signals.loc[date, 'signal_desc'] = 'OBV负背离卖出信号'
                    signals.loc[date, 'confidence'] = 80.0
                    signals.loc[date, 'position_size'] = 0.0
                elif "OBV下穿均线" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_death_cross'
                    signals.loc[date, 'signal_desc'] = 'OBV下穿均线卖出信号'
                    signals.loc[date, 'confidence'] = 70.0
                    signals.loc[date, 'position_size'] = 0.0
                elif "OBV跌破前期低点" in current_patterns:
                    signals.loc[date, 'signal_type'] = 'obv_breakdown'
                    signals.loc[date, 'signal_desc'] = 'OBV跌破前期低点卖出信号'
                    signals.loc[date, 'confidence'] = 75.0
                    signals.loc[date, 'position_size'] = 0.0
                else:
                    signals.loc[date, 'signal_type'] = 'obv_bearish'
                    signals.loc[date, 'signal_desc'] = 'OBV看跌信号'
                    signals.loc[date, 'confidence'] = 60.0
                    signals.loc[date, 'position_size'] = 0.0
        
        # 设置止损位
        for date in signals.index:
            if signals.loc[date, 'buy_signal']:
                # 简单示例：使用当日最低价的95%作为止损
                if 'low' in data.columns:
                    signals.loc[date, 'stop_loss'] = data.loc[date, 'low'] * 0.95
        
        # 设置成交量确认
        # 简单判断：如果成交量大于20日均值，认为有成交量确认
        if 'volume' in data.columns:
            vol_ma = data['volume'].rolling(20).mean()
            signals['volume_confirmation'] = data['volume'] > vol_ma
        
        # 市场环境判断（简化示例）
        if 'close' in data.columns:
            ma20 = data['close'].rolling(20).mean()
            ma60 = data['close'].rolling(60).mean()
            
            bull_market = (data['close'] > ma20) & (ma20 > ma60)
            bear_market = (data['close'] < ma20) & (ma20 < ma60)
            
            signals.loc[bull_market.index, 'market_env'] = 'bull_market'
            signals.loc[bear_market.index, 'market_env'] = 'bear_market'
        
        # 设置风险等级
        for date in signals.index:
            if signals.loc[date, 'confidence'] >= 75:
                signals.loc[date, 'risk_level'] = '低'
            elif signals.loc[date, 'confidence'] <= 60:
                signals.loc[date, 'risk_level'] = '高'
        
        return signals

    def _register_obv_patterns(self):
        """
        注册OBV指标相关形态
        """
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 注册OBV与均线交叉形态
        registry.register(
            pattern_id="OBV_GOLDEN_CROSS",
            display_name="OBV金叉",
            description="OBV上穿其均线，表明买盘力量增强",
            indicator_id="OBV",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0,
            polarity=PatternPolarity.POSITIVE
        )

        registry.register(
            pattern_id="OBV_DEATH_CROSS",
            display_name="OBV死叉",
            description="OBV下穿其均线，表明卖盘力量增强",
            indicator_id="OBV",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0,
            polarity=PatternPolarity.NEGATIVE
        )

        # 注册OBV趋势形态
        registry.register(
            pattern_id="OBV_UPTREND",
            display_name="OBV上升趋势",
            description="OBV持续上升，表明资金持续流入",
            indicator_id="OBV",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0,
            polarity=PatternPolarity.POSITIVE
        )

        registry.register(
            pattern_id="OBV_DOWNTREND",
            display_name="OBV下降趋势",
            description="OBV持续下降，表明资金持续流出",
            indicator_id="OBV",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0,
            polarity=PatternPolarity.NEGATIVE
        )
        
        # 注册OBV背离形态
        registry.register(
            pattern_id="OBV_BULLISH_DIVERGENCE",
            display_name="OBV底背离",
            description="价格创新低，但OBV未创新低，表明卖盘力量减弱",
            indicator_id="OBV",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0,
            polarity=PatternPolarity.POSITIVE
        )

        registry.register(
            pattern_id="OBV_BEARISH_DIVERGENCE",
            display_name="OBV顶背离",
            description="价格创新高，但OBV未创新高，表明买盘力量减弱",
            indicator_id="OBV",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0,
            polarity=PatternPolarity.NEGATIVE
        )

        # 注册OBV爆量形态
        registry.register(
            pattern_id="OBV_RAPID_INCREASE",
            display_name="OBV快速上涨",
            description="OBV短期内快速上涨，表明买盘力量强劲",
            indicator_id="OBV",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0,
            polarity=PatternPolarity.POSITIVE
        )

        registry.register(
            pattern_id="OBV_RAPID_DECREASE",
            display_name="OBV快速下跌",
            description="OBV短期内快速下跌，表明卖盘力量强劲",
            indicator_id="OBV",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0,
            polarity=PatternPolarity.NEGATIVE
        )
        
        # 注册OBV稳定形态
        registry.register(
            pattern_id="OBV_STABLE_POSITIVE",
            display_name="OBV稳定正值",
            description="OBV保持在高位稳定，表明买盘力量持续存在",
            indicator_id="OBV",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0,
            polarity=PatternPolarity.POSITIVE
        )

        registry.register(
            pattern_id="OBV_STABLE_NEGATIVE",
            display_name="OBV稳定负值",
            description="OBV保持在低位稳定，表明卖盘力量持续存在",
            indicator_id="OBV",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0,
            polarity=PatternPolarity.NEGATIVE
        )

        # 注册OBV量价关系形态
        registry.register(
            pattern_id="OBV_PRICE_CONFIRMATION",
            display_name="OBV量价同步",
            description="OBV与价格同向变动，确认当前趋势",
            indicator_id="OBV",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.MEDIUM,
            score_impact=8.0,
            polarity=PatternPolarity.NEUTRAL
        )

        registry.register(
            pattern_id="OBV_PRICE_NON_CONFIRMATION",
            display_name="OBV量价不同步",
            description="OBV与价格反向变动，表明当前趋势可能不可靠",
            indicator_id="OBV",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.MEDIUM,
            score_impact=0.0,
            polarity=PatternPolarity.NEUTRAL
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
        
        # 在这里实现指标特定的信号生成逻辑
        
        # 此处提供默认实现
        return signals

    def _calculate_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        计算OBV与价格的背离
        
        Args:
            df: 包含OBV和价格数据的DataFrame
            
        Returns:
            pd.Series: 背离指示器，1表示看涨背离，-1表示看跌背离，0表示无背离
        """
        # 简单实现，实际应用中可能需要更复杂的逻辑
        divergence = pd.Series(0, index=df.index)
        
        # 这里只是一个占位符，未来可以实现完整的背离检测算法
        return divergence

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        pattern_info_map = {
            "OBV_GOLDEN_CROSS": {
                "id": "OBV_GOLDEN_CROSS",
                "name": "OBV金叉",
                "description": "OBV上穿其均线，表明买盘力量增强",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            },
            "OBV_DEATH_CROSS": {
                "id": "OBV_DEATH_CROSS",
                "name": "OBV死叉",
                "description": "OBV下穿其均线，表明卖盘力量增强",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -15.0
            },
            "OBV_UPTREND": {
                "id": "OBV_UPTREND",
                "name": "OBV上升趋势",
                "description": "OBV持续上升，表明资金持续流入",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 12.0
            },
            "OBV_DOWNTREND": {
                "id": "OBV_DOWNTREND",
                "name": "OBV下降趋势",
                "description": "OBV持续下降，表明资金持续流出",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -12.0
            },
            "OBV_BULLISH_DIVERGENCE": {
                "id": "OBV_BULLISH_DIVERGENCE",
                "name": "OBV底背离",
                "description": "价格创新低，但OBV未创新低，表明卖盘力量减弱",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "OBV_BEARISH_DIVERGENCE": {
                "id": "OBV_BEARISH_DIVERGENCE",
                "name": "OBV顶背离",
                "description": "价格创新高，但OBV未创新高，表明买盘力量减弱",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            },
            "OBV_RAPID_INCREASE": {
                "id": "OBV_RAPID_INCREASE",
                "name": "OBV快速上涨",
                "description": "OBV短期内快速上涨，表明买盘力量强劲",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 18.0
            },
            "OBV_RAPID_DECREASE": {
                "id": "OBV_RAPID_DECREASE",
                "name": "OBV快速下跌",
                "description": "OBV短期内快速下跌，表明卖盘力量强劲",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -18.0
            },
            "OBV_STABLE_POSITIVE": {
                "id": "OBV_STABLE_POSITIVE",
                "name": "OBV稳定正值",
                "description": "OBV保持在高位稳定，表明买盘力量持续存在",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "OBV_STABLE_NEGATIVE": {
                "id": "OBV_STABLE_NEGATIVE",
                "name": "OBV稳定负值",
                "description": "OBV保持在低位稳定，表明卖盘力量持续存在",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "OBV_PRICE_CONFIRMATION": {
                "id": "OBV_PRICE_CONFIRMATION",
                "name": "OBV量价同步",
                "description": "OBV与价格同向变动，确认当前趋势",
                "type": "NEUTRAL",
                "strength": "MEDIUM",
                "score_impact": 8.0
            },
            "OBV_PRICE_NON_CONFIRMATION": {
                "id": "OBV_PRICE_NON_CONFIRMATION",
                "name": "OBV量价不同步",
                "description": "OBV与价格反向变动，表明当前趋势可能不可靠",
                "type": "NEUTRAL",
                "strength": "MEDIUM",
                "score_impact": -8.0
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "未知形态",
            "description": "未定义的形态",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })
