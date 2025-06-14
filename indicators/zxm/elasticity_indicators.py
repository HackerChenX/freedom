"""
ZXM体系弹性指标模块

实现ZXM体系的2个弹性指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class AmplitudeElasticity(BaseIndicator):
    """
    ZXM弹性-振幅指标
    
    判断近120日内是否有日振幅超过8.1%的情况
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM弹性-振幅指标"""
        super().__init__(name="AmplitudeElasticity", description="ZXM弹性-振幅指标，判断近期是否有较大振幅")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM弹性-振幅指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含弹性信号
            
        公式说明：
        a1:=100*(H-L)/L>8.1;
        COUNT(a1,120)>1
        """
        # 确保数据包含必需的列
        required_cols = ["high", "low"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需的列: {missing_cols}")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算日振幅
        amplitude = 100 * (data["high"] - data["low"]) / data["low"]
        
        # 计算日振幅超过8.1%的情况
        a1 = amplitude > 8.1
        
        # 计算120日内是否有超过1次日振幅大于8.1%
        xg = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        
        for i in range(120, len(data)):
            xg.iloc[i] = np.sum(a1.iloc[i-119:i+1]) > 1
        
        # 添加计算结果到数据框
        result["Amplitude"] = amplitude
        result["A1"] = a1
        result["XG"] = xg
        
        return result



    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM振幅弹性指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 有振幅弹性信号时加分
        score[result["XG"]] += 40
        
        # 根据振幅大小给予额外加分
        if "Amplitude" in result.columns:
            # 振幅越大，加分越多（最多额外加10分）
            amplitude_bonus = result["Amplitude"].apply(lambda x: min(10, max(0, (x - 8.1) / 2)))
            score += amplitude_bonus
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM振幅弹性指标相关的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)

        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]

            # 基础形态判断
            if last_row["XG"]:
                patterns.append("振幅弹性信号")

            # 振幅大小判断
            amplitude = last_row["Amplitude"]
            if amplitude > 15:
                patterns.append("极大振幅(>15%)")
            elif amplitude > 12:
                patterns.append("大振幅(12%-15%)")
            elif amplitude > 8.1:
                patterns.append("中等振幅(8.1%-12%)")
            else:
                patterns.append("小振幅(<8.1%)")

            # 历史振幅判断
            if len(result) >= 120:
                recent_amplitude_count = result["A1"].iloc[-120:].sum()
                if recent_amplitude_count > 10:
                    patterns.append("频繁大振幅")
                elif recent_amplitude_count > 5:
                    patterns.append("偶尔大振幅")
                elif recent_amplitude_count > 1:
                    patterns.append("少量大振幅")

        return patterns

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1]

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "振幅弹性信号" in patterns:
            pattern_boost += 0.15
        if "极大振幅(>15%)" in patterns:
            pattern_boost += 0.15
        elif "大振幅(12%-15%)" in patterns:
            pattern_boost += 0.1
        if "频繁大振幅" in patterns:
            pattern_boost += 0.1

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        # 计算指标
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础形态
        patterns_df["振幅弹性信号"] = result["XG"]
        patterns_df["大振幅日"] = result["A1"]

        # 振幅大小形态
        amplitude = result["Amplitude"]
        patterns_df["极大振幅"] = amplitude > 15
        patterns_df["大振幅"] = (amplitude > 12) & (amplitude <= 15)
        patterns_df["中等振幅"] = (amplitude > 8.1) & (amplitude <= 12)
        patterns_df["小振幅"] = amplitude <= 8.1

        # 历史振幅统计形态
        if len(result) >= 120:
            amplitude_count_120 = result["A1"].rolling(window=120).sum()
            patterns_df["频繁大振幅"] = amplitude_count_120 > 10
            patterns_df["偶尔大振幅"] = (amplitude_count_120 > 5) & (amplitude_count_120 <= 10)
            patterns_df["少量大振幅"] = (amplitude_count_120 > 1) & (amplitude_count_120 <= 5)
            patterns_df["无大振幅"] = amplitude_count_120 <= 1

        return patterns_df

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - amplitude_threshold: 振幅阈值，默认8.1
                - count_period: 统计周期，默认120
        """
        self.amplitude_threshold = kwargs.get('amplitude_threshold', 8.1)
        self.count_period = kwargs.get('count_period', 120)
class ZXMRiseElasticity(BaseIndicator):
    """
    ZXM弹性-涨幅指标
    
    判断近80日内是否有日涨幅超过7%的情况
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM弹性-涨幅指标"""
        super().__init__(name="ZXMRiseElasticity", description="ZXM弹性-涨幅指标，判断近期是否有较大涨幅")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM弹性-涨幅指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含弹性信号
            
        公式说明：
        a1:=C/REF(C,1)>1.07;
        COUNT(a1,80)>0
        """
        # 确保数据包含必需的列
        if 'close' not in data.columns:
            raise ValueError("数据缺少必需的'close'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算日涨幅
        rise_ratio = data["close"] / data["close"].shift(1)
        
        # 计算日涨幅超过7%的情况
        a1 = rise_ratio > 1.07
        
        # 计算80日内是否有涨幅大于7%
        xg = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        
        for i in range(80, len(data)):
            xg.iloc[i] = np.sum(a1.iloc[i-79:i+1]) > 0
        
        # 添加计算结果到数据框
        result["RiseRatio"] = rise_ratio
        result["A1"] = a1
        result["XG"] = xg
        
        return result



    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM涨幅弹性指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 有涨幅弹性信号时加分
        score[result["XG"]] += 40
        
        # 根据涨幅大小给予额外加分
        if "RiseRatio" in result.columns:
            # 涨幅越大，加分越多（最多额外加10分）
            rise_bonus = result["RiseRatio"].apply(lambda x: min(10, max(0, (x - 1.07) * 100)))
            score += rise_bonus
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM涨幅弹性指标相关的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)

        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]

            # 基础形态判断
            if last_row["XG"]:
                patterns.append("涨幅弹性信号")

            # 涨幅大小判断
            rise_ratio = last_row["RiseRatio"]
            if rise_ratio > 1.15:
                patterns.append("极大涨幅(>15%)")
            elif rise_ratio > 1.10:
                patterns.append("大涨幅(10%-15%)")
            elif rise_ratio > 1.07:
                patterns.append("中等涨幅(7%-10%)")
            else:
                patterns.append("小涨幅(<7%)")

            # 历史涨幅判断
            if len(result) >= 80:
                recent_rise_count = result["A1"].iloc[-80:].sum()
                if recent_rise_count > 10:
                    patterns.append("频繁大涨")
                elif recent_rise_count > 5:
                    patterns.append("偶尔大涨")
                elif recent_rise_count > 0:
                    patterns.append("少量大涨")
                else:
                    patterns.append("无大涨")

        return patterns

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1]

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "涨幅弹性信号" in patterns:
            pattern_boost += 0.15
        if "极大涨幅(>15%)" in patterns:
            pattern_boost += 0.15
        elif "大涨幅(10%-15%)" in patterns:
            pattern_boost += 0.1
        if "频繁大涨" in patterns:
            pattern_boost += 0.1

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        # 计算指标
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础形态
        patterns_df["涨幅弹性信号"] = result["XG"]
        patterns_df["大涨日"] = result["A1"]

        # 涨幅大小形态
        rise_ratio = result["RiseRatio"]
        patterns_df["极大涨幅"] = rise_ratio > 1.15
        patterns_df["大涨幅"] = (rise_ratio > 1.10) & (rise_ratio <= 1.15)
        patterns_df["中等涨幅"] = (rise_ratio > 1.07) & (rise_ratio <= 1.10)
        patterns_df["小涨幅"] = rise_ratio <= 1.07

        # 历史涨幅统计形态
        if len(result) >= 80:
            rise_count_80 = result["A1"].rolling(window=80).sum()
            patterns_df["频繁大涨"] = rise_count_80 > 10
            patterns_df["偶尔大涨"] = (rise_count_80 > 5) & (rise_count_80 <= 10)
            patterns_df["少量大涨"] = (rise_count_80 > 0) & (rise_count_80 <= 5)
            patterns_df["无大涨"] = rise_count_80 == 0

        return patterns_df

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - rise_threshold: 涨幅阈值，默认1.07
                - count_period: 统计周期，默认80
        """
        self.rise_threshold = kwargs.get('rise_threshold', 1.07)
        self.count_period = kwargs.get('count_period', 80)
class Elasticity(BaseIndicator):
    """
    ZXM弹性指标
    
    检测股价弹性和反弹力度
    """
    
    def __init__(self, period: int = 20):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ZXM弹性指标
        
        Args:
            period: 计算周期，默认20天
        """
        super().__init__(name="Elasticity", description="ZXM弹性指标，检测股价弹性和反弹力度")
        self.period = period
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM弹性指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需的列: {missing_cols}")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算价格变动幅度
        price_change = data["close"].pct_change()
        
        # 计算弹性指标
        # 1. 上涨幅度 - 连续上涨天数累计涨幅
        up_move = price_change.apply(lambda x: max(0, x))  # 只保留上涨部分
        elasticity_up = up_move.rolling(window=self.period).sum()
        
        # 2. 下跌幅度 - 连续下跌天数累计跌幅（绝对值）
        down_move = price_change.apply(lambda x: max(0, -x))  # 只保留下跌部分，取绝对值
        elasticity_down = down_move.rolling(window=self.period).sum()
        
        # 3. 弹性比率 - 上涨幅度除以下跌幅度
        elasticity_ratio = elasticity_up / elasticity_down
        
        # 4. 弹性强度 - 当日价格相对于区间内最低价的反弹幅度
        lowest_low = data["low"].rolling(window=self.period).min()
        highest_high = data["high"].rolling(window=self.period).max()
        
        # 防止除以零
        range_pct = (highest_high - lowest_low) / lowest_low
        range_pct = range_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        bounce_strength = (data["close"] - lowest_low) / (highest_high - lowest_low)
        bounce_strength = bounce_strength.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        
        # 5. 反弹确认 - 成交量配合指标
        volume_ratio = data["volume"] / data["volume"].rolling(window=self.period).mean()
        
        # 添加计算结果到数据框
        result["ElasticityUp"] = elasticity_up
        result["ElasticityDown"] = elasticity_down
        result["ElasticityRatio"] = elasticity_ratio
        result["BounceStrength"] = bounce_strength
        result["RangePct"] = range_pct
        result["VolumeRatio"] = volume_ratio
        
        # 6. 弹性买点 - 同时满足以下条件：
        # - 弹性比率大于1.2（上涨幅度大于下跌幅度）
        # - 反弹强度大于0.3（从最低点反弹超过区间30%）
        # - 区间波动大于5%（有足够的波动空间）
        # - 成交量比率大于0.8（成交量相对活跃）
        elasticity_buy_signal = (
            (elasticity_ratio > 1.2) & 
            (bounce_strength > 0.3) & 
            (range_pct > 0.05) & 
            (volume_ratio > 0.8)
        )
        
        result["BuySignal"] = elasticity_buy_signal
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM弹性指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. 弹性买点信号：+30分
        score[result["BuySignal"]] += 30
        
        # 2. 弹性比率评分
        # 弹性比率越高，表示上涨能量越强
        ratio_score = result["ElasticityRatio"].apply(lambda x: min(20, max(0, (x-1)*10)))
        score += ratio_score
        
        # 3. 反弹强度评分
        # 反弹强度表示从最低点反弹的幅度，越高越好
        bounce_score = result["BounceStrength"].apply(lambda x: min(15, max(0, x*25)))
        score += bounce_score
        
        # 4. 成交量配合评分
        # 成交量放大，反弹更有力度
        volume_score = (result["VolumeRatio"] - 1).apply(lambda x: min(15, max(0, x*10)))
        score += volume_score
        
        # 5. 价格走势加分
        # 计算近期价格趋势
        price_trend = data["close"].pct_change(5)
        
        # 近期上涨趋势，加分
        score[price_trend > 0.05] += 10  # 5日涨幅超过5%
        score[price_trend > 0.10] += 10  # 5日涨幅超过10%
        
        # 6. 反弹持续性评分
        # 连续多日满足弹性买点，表示反弹较强
        bounce_days = pd.Series(0, index=data.index)
        for i in range(3, len(data)):
            if result["BuySignal"].iloc[i-3:i+1].sum() >= 2:
                bounce_days.iloc[i] = result["BuySignal"].iloc[i-3:i+1].sum()
        
        # 根据连续满足弹性买点的天数加分
        score[bounce_days == 2] += 5
        score[bounce_days == 3] += 10
        score[bounce_days == 4] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM弹性指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["BuySignal"]:
                patterns.append("弹性买点")
            
            # 弹性比率形态
            if last_row["ElasticityRatio"] > 2:
                patterns.append("高弹性比率(>2)")
            elif last_row["ElasticityRatio"] > 1.5:
                patterns.append("中等弹性比率(1.5-2)")
            elif last_row["ElasticityRatio"] > 1.2:
                patterns.append("轻微弹性比率(1.2-1.5)")
            elif last_row["ElasticityRatio"] < 0.8:
                patterns.append("低弹性比率(<0.8)")
            
            # 反弹强度形态
            if last_row["BounceStrength"] > 0.7:
                patterns.append("强反弹(>70%)")
            elif last_row["BounceStrength"] > 0.5:
                patterns.append("中等反弹(50%-70%)")
            elif last_row["BounceStrength"] > 0.3:
                patterns.append("轻微反弹(30%-50%)")
            elif last_row["BounceStrength"] < 0.2:
                patterns.append("接近低点(<20%)")
            
            # 成交量配合形态
            if last_row["VolumeRatio"] > 1.5:
                patterns.append("放量反弹")
            elif last_row["VolumeRatio"] < 0.7:
                patterns.append("缩量反弹")
            
            # 区间波动形态
            if last_row["RangePct"] > 0.15:
                patterns.append("大幅波动区间(>15%)")
            elif last_row["RangePct"] < 0.05:
                patterns.append("窄幅波动区间(<5%)")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals['buy_signal'] = result["BuySignal"]
        signals['sell_signal'] = False  # 弹性指标主要用于识别买点
        signals['neutral_signal'] = ~result["BuySignal"]
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["BuySignal"], 'trend'] = 1  # 弹性买点看涨
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["BuySignal"], 'signal_type'] = 'elasticity_buy'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 根据弹性强度和比率设置详细描述
        for i in signals.index[result["BuySignal"]]:
            ratio = result.loc[i, "ElasticityRatio"]
            bounce = result.loc[i, "BounceStrength"]
            vol_ratio = result.loc[i, "VolumeRatio"]
            
            desc_parts = []
            
            if ratio > 2:
                desc_parts.append("高弹性比率")
            elif ratio > 1.5:
                desc_parts.append("中等弹性比率")
            else:
                desc_parts.append("轻微弹性比率")
                
            if bounce > 0.7:
                desc_parts.append("强反弹")
            elif bounce > 0.5:
                desc_parts.append("中等反弹")
            else:
                desc_parts.append("轻微反弹")
                
            if vol_ratio > 1.5:
                desc_parts.append("放量确认")
            elif vol_ratio > 1:
                desc_parts.append("量能正常")
            else:
                desc_parts.append("缩量反弹")
            
            signals.loc[i, 'signal_desc'] = "弹性买点：" + "，".join(desc_parts)
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据弹性比率和反弹强度调整置信度
        for i in signals.index:
            if result.loc[i, "BuySignal"]:
                confidence_adj = 0
                
                # 弹性比率影响
                ratio = result.loc[i, "ElasticityRatio"]
                if ratio > 2:
                    confidence_adj += 15
                elif ratio > 1.5:
                    confidence_adj += 10
                elif ratio > 1.2:
                    confidence_adj += 5
                
                # 反弹强度影响
                bounce = result.loc[i, "BounceStrength"]
                if bounce > 0.7:
                    confidence_adj += 15
                elif bounce > 0.5:
                    confidence_adj += 10
                elif bounce > 0.3:
                    confidence_adj += 5
                
                # 成交量配合影响
                vol_ratio = result.loc[i, "VolumeRatio"]
                if vol_ratio > 1.5:
                    confidence_adj += 10
                elif vol_ratio > 1:
                    confidence_adj += 5
                
                signals.loc[i, 'confidence'] = min(95, 60 + confidence_adj)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[result["BuySignal"], 'position_size'] = 0.3  # 基础仓位
        signals.loc[(result["BuySignal"]) & (score > 70), 'position_size'] = 0.5  # 高分仓位
        signals.loc[(result["BuySignal"]) & (score > 85), 'position_size'] = 0.7  # 极高分仓位
        
        # 止损位 - 使用区间最低价
        signals['stop_loss'] = 0.0
        mask = result["BuySignal"]
        for i in data.index[mask]:
            period = self.period
            try:
                idx = data.index.get_loc(i)
                if idx >= period:
                    low_price = data.iloc[idx-period:idx+1]['low'].min()
                    signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
            except:
                continue
        
        # 市场环境和成交量确认
        signals['market_env'] = 'normal'
        signals['volume_confirmation'] = result["VolumeRatio"] > 1.0
        
        return signals

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1]

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "弹性买点" in patterns:
            pattern_boost += 0.2
        if "高弹性比率(>2)" in patterns:
            pattern_boost += 0.15
        elif "中等弹性比率(1.5-2)" in patterns:
            pattern_boost += 0.1
        if "强反弹(>70%)" in patterns:
            pattern_boost += 0.15
        elif "中等反弹(50%-70%)" in patterns:
            pattern_boost += 0.1
        if "放量反弹" in patterns:
            pattern_boost += 0.1

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        # 计算指标
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础形态
        patterns_df["弹性买点"] = result["BuySignal"]

        # 弹性比率形态
        elasticity_ratio = result["ElasticityRatio"]
        patterns_df["高弹性比率"] = elasticity_ratio > 2
        patterns_df["中等弹性比率"] = (elasticity_ratio > 1.5) & (elasticity_ratio <= 2)
        patterns_df["轻微弹性比率"] = (elasticity_ratio > 1.2) & (elasticity_ratio <= 1.5)
        patterns_df["低弹性比率"] = elasticity_ratio < 0.8

        # 反弹强度形态
        bounce_strength = result["BounceStrength"]
        patterns_df["强反弹"] = bounce_strength > 0.7
        patterns_df["中等反弹"] = (bounce_strength > 0.5) & (bounce_strength <= 0.7)
        patterns_df["轻微反弹"] = (bounce_strength > 0.3) & (bounce_strength <= 0.5)
        patterns_df["接近低点"] = bounce_strength < 0.2

        # 成交量配合形态
        volume_ratio = result["VolumeRatio"]
        patterns_df["放量反弹"] = volume_ratio > 1.5
        patterns_df["缩量反弹"] = volume_ratio < 0.7
        patterns_df["量能正常"] = (volume_ratio >= 0.7) & (volume_ratio <= 1.5)

        # 区间波动形态
        range_pct = result["RangePct"]
        patterns_df["大幅波动区间"] = range_pct > 0.15
        patterns_df["窄幅波动区间"] = range_pct < 0.05
        patterns_df["正常波动区间"] = (range_pct >= 0.05) & (range_pct <= 0.15)

        return patterns_df

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - period: 计算周期，默认20
                - elasticity_threshold: 弹性比率阈值，默认1.2
                - bounce_threshold: 反弹强度阈值，默认0.3
                - range_threshold: 区间波动阈值，默认0.05
                - volume_threshold: 成交量比率阈值，默认0.8
        """
        self.period = kwargs.get('period', 20)
        self.elasticity_threshold = kwargs.get('elasticity_threshold', 1.2)
        self.bounce_threshold = kwargs.get('bounce_threshold', 0.3)
        self.range_threshold = kwargs.get('range_threshold', 0.05)
        self.volume_threshold = kwargs.get('volume_threshold', 0.8)


class BounceDetector(BaseIndicator):
    """
    ZXM反弹检测器
    
    检测价格反弹和回调信号
    """
    
    def __init__(self, short_period: int = 5, long_period: int = 20):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ZXM反弹检测器
        
        Args:
            short_period: 短期周期，默认5天
            long_period: 长期周期，默认20天
        """
        super().__init__(name="BounceDetector", description="ZXM反弹检测器，检测价格反弹和回调信号")
        self.short_period = short_period
        self.long_period = long_period
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM反弹检测指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需的列: {missing_cols}")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算基础价格指标
        # 1. 计算短期和长期最低价
        short_low = data["low"].rolling(window=self.short_period).min()
        long_low = data["low"].rolling(window=self.long_period).min()
        
        # 2. 计算短期和长期最高价
        short_high = data["high"].rolling(window=self.short_period).max()
        long_high = data["high"].rolling(window=self.long_period).max()
        
        # 3. 计算当前价格相对于长期低点的反弹幅度
        bounce_from_low = (data["close"] - long_low) / long_low * 100
        
        # 4. 计算当前价格相对于长期高点的回调幅度
        pullback_from_high = (long_high - data["close"]) / long_high * 100
        
        # 记录这些基础指标
        result["ShortLow"] = short_low
        result["LongLow"] = long_low
        result["ShortHigh"] = short_high
        result["LongHigh"] = long_high
        result["BounceFromLow"] = bounce_from_low
        result["PullbackFromHigh"] = pullback_from_high
        
        # 5. 计算价格趋势方向
        # 使用短期均线的斜率
        ma5 = data["close"].rolling(window=5).mean()
        ma20 = data["close"].rolling(window=20).mean()
        
        price_trend = np.zeros(len(data))
        for i in range(5, len(data)):
            if ma5.iloc[i] > ma5.iloc[i-5]:
                price_trend[i] = 1  # 上升趋势
            elif ma5.iloc[i] < ma5.iloc[i-5]:
                price_trend[i] = -1  # 下降趋势
        
        result["PriceTrend"] = price_trend
        
        # 6. 计算量能变化
        volume_ma5 = data["volume"].rolling(window=5).mean()
        volume_ma20 = data["volume"].rolling(window=20).mean()
        
        volume_change = (data["volume"] / volume_ma5 - 1) * 100
        volume_trend = (volume_ma5 / volume_ma20 - 1) * 100
        
        result["VolumeChange"] = volume_change
        result["VolumeTrend"] = volume_trend
        
        # 7. 检测反弹信号
        # 反弹信号条件：
        # - 价格从长期低点反弹超过5%
        # - 短期趋势向上
        # - 近期成交量放大
        bounce_signal = (bounce_from_low > 5) & (price_trend == 1) & (volume_change > 10)
        
        # 8. 检测回调信号
        # 回调信号条件：
        # - 价格从长期高点回调超过5%
        # - 短期趋势仍向上（表明主趋势仍然看涨）
        # - 近期成交量萎缩
        pullback_signal = (pullback_from_high > 5) & (price_trend == 1) & (volume_change < -10)
        
        result["BounceSignal"] = bounce_signal
        result["PullbackSignal"] = pullback_signal
        
        # 9. 计算反弹强度 (0-100)
        bounce_strength = np.zeros(len(data))
        
        for i in range(self.long_period, len(data)):
            # 基础反弹分数 - 基于反弹幅度（最高40分）
            bounce_pct = bounce_from_low.iloc[i]
            base_score = min(40, bounce_pct * 2)
            
            # 趋势加分 - 趋势向上（最高20分）
            trend_score = 20 if price_trend[i] == 1 else 0
            
            # 成交量加分 - 成交量放大程度（最高20分）
            volume_score = min(20, max(0, volume_change.iloc[i]))
            
            # 反弹速度加分 - 基于短期涨幅（最高20分）
            if i >= 5:
                five_day_change = (data["close"].iloc[i] / data["close"].iloc[i-5] - 1) * 100
                speed_score = min(20, max(0, five_day_change * 2))
            else:
                speed_score = 0
            
            # 综合反弹强度得分
            bounce_strength[i] = base_score + trend_score + volume_score + speed_score
        
        result["BounceStrength"] = bounce_strength
        
        # 10. 计算回调买点
        # 回调买点条件：
        # - 长期趋势向上（MA20 > MA60）
        # - 价格处于回调状态（相对高点回调5%-20%）
        # - 回调到支撑位附近（如MA20或前期高点）
        # - 成交量萎缩
        ma60 = data["close"].rolling(window=60).mean()
        
        pullback_buy_point = pd.Series(False, index=data.index)
        
        for i in range(60, len(data)):
            if (ma20.iloc[i] > ma60.iloc[i] and  # 长期趋势向上
                pullback_from_high.iloc[i] >= 5 and pullback_from_high.iloc[i] <= 20 and  # 回调5%-20%
                abs(data["low"].iloc[i] - ma20.iloc[i]) / ma20.iloc[i] < 0.03 and  # 接近MA20
                volume_change.iloc[i] < 0):  # 成交量萎缩
                pullback_buy_point.iloc[i] = True
        
        result["PullbackBuyPoint"] = pullback_buy_point
        
        # 11. 计算反弹卖点
        # 反弹卖点条件：
        # - 长期趋势向下（MA20 < MA60）
        # - 价格处于反弹状态（相对低点反弹超过15%）
        # - 反弹到阻力位附近（如MA20或前期低点）
        # - 出现放量滞涨
        
        bounce_sell_point = pd.Series(False, index=data.index)
        
        for i in range(60, len(data)):
            if (ma20.iloc[i] < ma60.iloc[i] and  # 长期趋势向下
                bounce_from_low.iloc[i] >= 15 and  # 反弹超过15%
                abs(data["high"].iloc[i] - ma20.iloc[i]) / ma20.iloc[i] < 0.03 and  # 接近MA20
                volume_change.iloc[i] > 20 and  # 成交量放大
                data["close"].iloc[i] < data["open"].iloc[i]):  # 收阴（滞涨）
                bounce_sell_point.iloc[i] = True
        
        result["BounceSellPoint"] = bounce_sell_point
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM反弹检测器的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. 反弹信号：+20分
        score[result["BounceSignal"]] += 20
        
        # 2. 回调信号：-10分
        score[result["PullbackSignal"]] -= 10
        
        # 3. 回调买点：强烈买入信号，+30分
        score[result["PullbackBuyPoint"]] += 30
        
        # 4. 反弹卖点：强烈卖出信号，-30分
        score[result["BounceSellPoint"]] -= 30
        
        # 5. 根据反弹强度调整评分
        bounce_strength_score = result["BounceStrength"] / 5  # 除以5，让100分的强度贡献20分
        score += bounce_strength_score
        
        # 6. 根据价格趋势方向调整评分
        # 上升趋势+10分，下降趋势-10分
        score[result["PriceTrend"] == 1] += 10
        score[result["PriceTrend"] == -1] -= 10
        
        # 7. 反弹幅度评分
        # 低位小幅反弹，加分温和；大幅反弹后可能面临回调，分数减少
        for i in range(len(data)):
            bounce_pct = result["BounceFromLow"].iloc[i]
            if bounce_pct <= 10:  # 小幅反弹，看涨
                score.iloc[i] += bounce_pct
            elif bounce_pct > 20:  # 大幅反弹，可能回调
                score.iloc[i] -= (bounce_pct - 20) / 2
        
        # 8. 成交量配合评分
        # 反弹时放量，加分；回调时缩量，加分
        for i in range(len(data)):
            if result["BounceSignal"].iloc[i] and result["VolumeChange"].iloc[i] > 20:
                score.iloc[i] += 10  # 反弹放量，加分
            elif result["PullbackSignal"].iloc[i] and result["VolumeChange"].iloc[i] < -20:
                score.iloc[i] += 5  # 回调缩量，小幅加分
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM反弹检测器相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 判断反弹/回调状态
            bounce_pct = last_row["BounceFromLow"]
            pullback_pct = last_row["PullbackFromHigh"]
            
            # 反弹相关形态
            if bounce_pct > 5:
                if bounce_pct > 20:
                    patterns.append(f"大幅反弹(>20%)")
                elif bounce_pct > 10:
                    patterns.append(f"中等反弹(10-20%)")
                else:
                    patterns.append(f"小幅反弹(5-10%)")
                
                # 反弹信号
                if last_row["BounceSignal"]:
                    patterns.append("反弹确认信号")
                
                # 反弹卖点
                if last_row["BounceSellPoint"]:
                    patterns.append("反弹卖点")
            
            # 回调相关形态
            if pullback_pct > 5:
                if pullback_pct > 20:
                    patterns.append(f"深度回调(>20%)")
                elif pullback_pct > 10:
                    patterns.append(f"中等回调(10-20%)")
                else:
                    patterns.append(f"浅度回调(5-10%)")
                
                # 回调信号
                if last_row["PullbackSignal"]:
                    patterns.append("回调确认信号")
                
                # 回调买点
                if last_row["PullbackBuyPoint"]:
                    patterns.append("回调买点")
            
            # 价格趋势
            if last_row["PriceTrend"] == 1:
                patterns.append("短期上升趋势")
            elif last_row["PriceTrend"] == -1:
                patterns.append("短期下降趋势")
            
            # 成交量特征
            if last_row["VolumeChange"] > 20:
                patterns.append("明显放量")
            elif last_row["VolumeChange"] < -20:
                patterns.append("明显缩量")
            
            # 综合形态
            if last_row["PullbackBuyPoint"]:
                patterns.append("回调买入机会")
            elif last_row["BounceSellPoint"]:
                patterns.append("反弹卖出机会")
            elif bounce_pct > 5 and last_row["PriceTrend"] == 1 and last_row["VolumeChange"] > 0:
                patterns.append("强势反弹")
            elif pullback_pct > 5 and last_row["PriceTrend"] == 1 and last_row["VolumeChange"] < 0:
                patterns.append("健康回调")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals['buy_signal'] = result["PullbackBuyPoint"] | result["BounceSignal"]
        signals['sell_signal'] = result["BounceSellPoint"]
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result["PriceTrend"]
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["PullbackBuyPoint"], 'signal_type'] = 'pullback_buy'
        signals.loc[result["BounceSignal"] & ~result["PullbackBuyPoint"], 'signal_type'] = 'bounce_signal'
        signals.loc[result["BounceSellPoint"], 'signal_type'] = 'bounce_sell'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "PullbackBuyPoint"]:
                pullback_pct = result.loc[i, "PullbackFromHigh"]
                signals.loc[i, 'signal_desc'] = f"回调买点：回调{pullback_pct:.1f}%后筑底企稳"
            elif result.loc[i, "BounceSignal"]:
                bounce_pct = result.loc[i, "BounceFromLow"]
                signals.loc[i, 'signal_desc'] = f"反弹信号：从低点反弹{bounce_pct:.1f}%"
            elif result.loc[i, "BounceSellPoint"]:
                bounce_pct = result.loc[i, "BounceFromLow"]
                signals.loc[i, 'signal_desc'] = f"反弹卖点：反弹{bounce_pct:.1f}%后遇阻回落"
            elif result.loc[i, "PullbackSignal"]:
                pullback_pct = result.loc[i, "PullbackFromHigh"]
                signals.loc[i, 'signal_desc'] = f"回调信号：从高点回调{pullback_pct:.1f}%"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据反弹/回调幅度和成交量配合调整置信度
        for i in signals.index:
            if signals.loc[i, 'buy_signal']:
                confidence_adj = 0
                
                # 回调幅度影响
                pullback_pct = result.loc[i, "PullbackFromHigh"]
                if 10 <= pullback_pct <= 20:  # 适度回调，最佳买点
                    confidence_adj += 15
                elif pullback_pct > 20:  # 过度回调，可能有问题
                    confidence_adj += 5
                elif pullback_pct < 10:  # 回调不充分
                    confidence_adj += 10
                
                # 成交量配合影响
                vol_change = result.loc[i, "VolumeChange"]
                if result.loc[i, "PullbackBuyPoint"]:
                    # 回调买点希望先缩量再放量
                    if vol_change < -20:  # 明显缩量
                        confidence_adj += 10
                    elif vol_change > 0:  # 开始放量，确认买点
                        confidence_adj += 15
                else:  # 反弹信号
                    if vol_change > 20:  # 明显放量
                        confidence_adj += 15
                
                signals.loc[i, 'confidence'] = min(95, 60 + confidence_adj)
            
            elif signals.loc[i, 'sell_signal']:
                confidence_adj = 0
                
                # 反弹幅度影响
                bounce_pct = result.loc[i, "BounceFromLow"]
                if bounce_pct > 20:  # 大幅反弹，卖点可靠性高
                    confidence_adj += 15
                elif bounce_pct > 10:
                    confidence_adj += 10
                
                # 成交量配合影响
                vol_change = result.loc[i, "VolumeChange"]
                if vol_change > 20:  # 放量滞涨
                    confidence_adj += 15
                
                signals.loc[i, 'confidence'] = min(95, 60 + confidence_adj)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 标准回调买点风险较低
        perfect_pullback = result["PullbackBuyPoint"] & (result["PullbackFromHigh"] >= 10) & (result["PullbackFromHigh"] <= 20)
        signals.loc[perfect_pullback, 'risk_level'] = '低'
        
        # 过度回调或大幅反弹后的操作风险较高
        signals.loc[result["PullbackFromHigh"] > 30, 'risk_level'] = '高'
        signals.loc[result["BounceFromLow"] > 30, 'risk_level'] = '高'
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 标准回调买点可以加大仓位
        signals.loc[perfect_pullback, 'position_size'] = 0.5
        
        # 止损位
        signals['stop_loss'] = 0.0
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= self.long_period:
                    # 使用最近低点作为止损位
                    low_price = result.loc[i, "LongLow"]
                    signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        
        ma20 = data["close"].rolling(window=20).mean()
        ma60 = data["close"].rolling(window=60).mean()
        
        # 简单判断市场环境
        for i in signals.index:
            try:
                idx = data.index.get_loc(i)
                if idx >= 60:
                    if ma20.iloc[idx] > ma60.iloc[idx]:
                        signals.loc[i, 'market_env'] = 'bull_market'
                    elif ma20.iloc[idx] < ma60.iloc[idx]:
                        signals.loc[i, 'market_env'] = 'bear_market'
                    else:
                        signals.loc[i, 'market_env'] = 'sideways_market'
            except:
                continue
        
        # 成交量确认
        signals['volume_confirmation'] = result["VolumeChange"] > 0
        
        return signals

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1]

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "回调买入机会" in patterns:
            pattern_boost += 0.2
        elif "反弹卖出机会" in patterns:
            pattern_boost += 0.2
        elif "强势反弹" in patterns:
            pattern_boost += 0.15
        elif "健康回调" in patterns:
            pattern_boost += 0.15

        if "反弹确认信号" in patterns:
            pattern_boost += 0.1
        elif "回调确认信号" in patterns:
            pattern_boost += 0.1

        if "明显放量" in patterns:
            pattern_boost += 0.08
        elif "明显缩量" in patterns:
            pattern_boost += 0.05

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        # 计算指标
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础信号形态
        patterns_df["反弹确认信号"] = result["BounceSignal"]
        patterns_df["回调确认信号"] = result["PullbackSignal"]
        patterns_df["回调买入机会"] = result["PullbackBuyPoint"]
        patterns_df["反弹卖出机会"] = result["BounceSellPoint"]

        # 反弹幅度形态
        bounce_pct = result["BounceFromLow"]
        patterns_df["大幅反弹"] = bounce_pct > 20
        patterns_df["中等反弹"] = (bounce_pct > 10) & (bounce_pct <= 20)
        patterns_df["小幅反弹"] = (bounce_pct > 5) & (bounce_pct <= 10)

        # 回调幅度形态
        pullback_pct = result["PullbackFromHigh"]
        patterns_df["深度回调"] = pullback_pct > 20
        patterns_df["中等回调"] = (pullback_pct > 10) & (pullback_pct <= 20)
        patterns_df["浅度回调"] = (pullback_pct > 5) & (pullback_pct <= 10)

        # 趋势形态
        patterns_df["短期上升趋势"] = result["PriceTrend"] == 1
        patterns_df["短期下降趋势"] = result["PriceTrend"] == -1

        # 成交量形态
        vol_change = result["VolumeChange"]
        patterns_df["明显放量"] = vol_change > 20
        patterns_df["明显缩量"] = vol_change < -20

        # 综合形态
        patterns_df["强势反弹"] = (bounce_pct > 5) & (result["PriceTrend"] == 1) & (vol_change > 0)
        patterns_df["健康回调"] = (pullback_pct > 5) & (result["PriceTrend"] == 1) & (vol_change < 0)

        return patterns_df

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - short_period: 短期周期，默认5
                - long_period: 长期周期，默认20
        """
        self.short_period = kwargs.get('short_period', 5)
        self.long_period = kwargs.get('long_period', 20)