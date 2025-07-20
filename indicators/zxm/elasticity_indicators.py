"""
ZXM体系弹性指标模块

实现ZXM体系的2个弹性指标
"""

from utils.dependency_injection import get_logger

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class AmplitudeElasticity(BaseIndicator, PatternSignalMixin):
    """
    ZXM弹性-振幅指标
    
    判断近120日内是否有日振幅超过8.1%的情况
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM弹性-振幅指标"""
        super().__init__(name="AmplitudeElasticity", description="ZXM弹性-振幅指标，判断近期是否有较大振幅")
    
    def _calculate_elasticityindicators(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
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
        result.loc[:, "Amplitude"] = amplitude
        result.loc[:, "A1"] = a1
        result.loc[:, "XG"] = xg
        
        
        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        # 重写buy_signal逻辑，基于XG值而不是通用逻辑
        result.loc[:, 'buy_signal'] = result["XG"] == True
        result.loc[:, 'sell_signal'] = result["XG"] == False
        result.loc[:, 'hold_signal'] = result["XG"] == False

        return result



    def calculate_raw_score_Indicators_elasticityindicators(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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
        scores = pd.Series(50.0, index=data.index, dtype=float)
        
        # 获取振幅数据
        amplitude = result["Amplitude"]
        
        for i in range(len(data)):
            score = 50.0  # 基础分数
            current_amplitude = amplitude.iloc[i]
            
            # 1. 基于当前振幅的评分
            if current_amplitude >= 8.1:
                # 达到阈值，基础加分
                score += 40
                # 超过阈值的额外加分
                extra_bonus = min(10, (current_amplitude - 8.1) / 2)
                score += extra_bonus
            elif current_amplitude >= 6.0:
                # 接近阈值，适度加分
                score += 20 + (current_amplitude - 6.0) / 2.1 * 15  # 6%-8.1%之间线性加分
            elif current_amplitude >= 4.0:
                # 中等振幅，小幅加分
                score += 5 + (current_amplitude - 4.0) / 2.0 * 10   # 4%-6%之间线性加分
            elif current_amplitude >= 2.0:
                # 小振幅，维持中性
                score += (current_amplitude - 2.0) / 2.0 * 5       # 2%-4%之间小幅加分
            else:
                # 极小振幅，减分
                score -= (2.0 - current_amplitude) * 5
            
            # 2. 基于历史振幅的评分
            if i >= 10:
                # 计算10日内的平均振幅
                avg_amplitude_10 = amplitude.iloc[max(0, i-10):i+1].mean()
                if avg_amplitude_10 > 5.0:
                    score += 10
                elif avg_amplitude_10 > 4.0:
                    score += 5
                elif avg_amplitude_10 < 2.0:
                    score -= 5
                elif avg_amplitude_10 < 1.5:
                    score -= 10
            
            # 3. 基于振幅趋势的评分
            if i >= 5:
                # 计算振幅趋势
                recent_amplitude = amplitude.iloc[i-5:i+1].mean()
                earlier_amplitude = amplitude.iloc[max(0, i-10):max(1, i-5)].mean()
                
                if recent_amplitude > earlier_amplitude * 1.2:
                    score += 8  # 振幅放大
                elif recent_amplitude > earlier_amplitude * 1.1:
                    score += 5  # 振幅略有放大
                elif recent_amplitude < earlier_amplitude * 0.8:
                    score -= 8  # 振幅缩小
                elif recent_amplitude < earlier_amplitude * 0.9:
                    score -= 5  # 振幅略有缩小
            
            # 4. 基于振幅波动性的评分
            if i >= 10:
                # 计算振幅的标准差
                amplitude_std = amplitude.iloc[max(0, i-10):i+1].std()
                amplitude_mean = amplitude.iloc[max(0, i-10):i+1].mean()
                
                if amplitude_mean > 0:
                    cv = amplitude_std / amplitude_mean  # 变异系数
                    if cv > 0.5:
                        score += 8  # 高波动性
                    elif cv > 0.3:
                        score += 5  # 中等波动性
                    elif cv < 0.1:
                        score -= 5  # 低波动性
            
            # 5. 基于最高振幅的评分
            if i >= 20:
                # 计算20日内的最高振幅
                max_amplitude_20 = amplitude.iloc[max(0, i-20):i+1].max()
                if max_amplitude_20 >= 8.1:
                    score += 15  # 近期有大振幅
                elif max_amplitude_20 >= 6.0:
                    score += 10  # 近期有中等振幅
                elif max_amplitude_20 >= 4.0:
                    score += 5   # 近期有小振幅
            
            # 6. 基于振幅分布的评分
            if i >= 30:
                # 计算30日内振幅分布
                amplitude_30 = amplitude.iloc[max(0, i-30):i+1]
                high_amplitude_count = (amplitude_30 >= 5.0).sum()
                medium_amplitude_count = (amplitude_30 >= 3.0).sum()
                
                if high_amplitude_count >= 5:
                    score += 12  # 频繁高振幅
                elif high_amplitude_count >= 3:
                    score += 8   # 偶尔高振幅
                elif medium_amplitude_count >= 10:
                    score += 5   # 频繁中等振幅
                elif medium_amplitude_count <= 5:
                    score -= 5   # 缺乏振幅
            
            # 7. 基于振幅相对性的评分
            if i >= 60:
                # 与60日平均振幅比较
                avg_amplitude_60 = amplitude.iloc[max(0, i-60):i+1].mean()
                if current_amplitude > avg_amplitude_60 * 1.5:
                    score += 10  # 当前振幅显著高于平均
                elif current_amplitude > avg_amplitude_60 * 1.2:
                    score += 5   # 当前振幅高于平均
                elif current_amplitude < avg_amplitude_60 * 0.5:
                    score -= 10  # 当前振幅显著低于平均
                elif current_amplitude < avg_amplitude_60 * 0.8:
                    score -= 5   # 当前振幅低于平均
            
            # 确保分数在0-100范围内
            score = max(0, min(100, score))
            scores.iloc[i] = score
        
        return scores

    def identify_patterns_Indicators_elasticityindicators(self, data: pd.DataFrame, **kwargs) -> List[str]:
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

    def calculate_confidence_Indicators_elasticityindicators(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
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

    def get_patterns_Indicators_elasticityindicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的Data_frame
        """
        # 计算指标
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础形态
        patterns_df.loc[:, "振幅弹性信号"] = result["XG"]
        patterns_df.loc[:, "大振幅日"] = result["A1"]

        # 振幅大小形态
        amplitude = result["Amplitude"]
        patterns_df.loc[:, "极大振幅"] = amplitude > 15
        patterns_df.loc[:, "大振幅"] = (amplitude > 12) & (amplitude <= 15)
        patterns_df.loc[:, "中等振幅"] = (amplitude > 8.1) & (amplitude <= 12)
        patterns_df.loc[:, "小振幅"] = amplitude <= 8.1

        # 历史振幅统计形态
        if len(result) >= 120:
            amplitude_count_120 = result["A1"].rolling(window=120).sum()
            patterns_df.loc[:, "频繁大振幅"] = amplitude_count_120 > 10
            patterns_df.loc[:, "偶尔大振幅"] = (amplitude_count_120 > 5) & (amplitude_count_120 <= 10)
            patterns_df.loc[:, "少量大振幅"] = (amplitude_count_120 > 1) & (amplitude_count_120 <= 5)
            patterns_df.loc[:, "无大振幅"] = amplitude_count_120 <= 1

        return patterns_df

    def set_parameters_Indicators_elasticityindicators(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - amplitude_threshold: 振幅阈值，默认8.1
                - count_period: 统计周期，默认120
        """
        self.amplitude_threshold = kwargs.get('amplitude_threshold', 8.1)
        self.count_period = kwargs.get('count_period', 120)
class ZxmriseElasticity(BaseIndicator, PatternSignalMixin):
    """
    ZXM弹性-涨幅指标

    判断近80日内是否有日涨幅超过7%的情况
    """
    
    def calculate_raw_score_Indicators_elasticityindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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

    def identify_patterns_Indicators_elasticityindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> List[str]:
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
                patterns.append("温和上涨(<7%)")

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

    def calculate_confidence_Indicators_elasticityindicators_duplicate(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
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

    def get_patterns_Indicators_elasticityindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的Data_frame
        """
        # 计算指标
        result = self.calculate(data)

        # 初始化形态DataFrame
        patterns_df = pd.DataFrame(index=data.index)

        # 基础形态
        patterns_df.loc[:, "涨幅弹性信号"] = result["XG"]
        patterns_df.loc[:, "大涨日"] = result["A1"]

        # 涨幅大小形态
        rise_ratio = result["RiseRatio"]
        patterns_df.loc[:, "极大涨幅"] = rise_ratio > 1.15
        patterns_df.loc[:, "大涨幅"] = (rise_ratio > 1.10) & (rise_ratio <= 1.15)
        patterns_df.loc[:, "中等涨幅"] = (rise_ratio > 1.07) & (rise_ratio <= 1.10)
        patterns_df.loc[:, "温和上涨"] = rise_ratio <= 1.07

        # 历史涨幅统计形态
        if len(result) >= 80:
            rise_count_80 = result["A1"].rolling(window=80).sum()
            patterns_df.loc[:, "频繁大涨"] = rise_count_80 > 10
            patterns_df.loc[:, "偶尔大涨"] = (rise_count_80 > 5) & (rise_count_80 <= 10)
            patterns_df.loc[:, "少量大涨"] = (rise_count_80 > 0) & (rise_count_80 <= 5)
            patterns_df.loc[:, "无大涨"] = rise_count_80 == 0

        return patterns_df

    def set_parameters_Indicators_elasticityindicators_duplicate(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - rise_threshold: 涨幅阈值，默认1.07
                - count_period: 统计周期，默认80
        """
        self.rise_threshold = kwargs.get('rise_threshold', 1.07)
        self.count_period = kwargs.get('count_period', 80)
class Elasticity(BaseIndicator, PatternSignalMixin):
    """
    ZXM弹性指标

    检测股价弹性和反弹力度
    """
    
    def generate_signals_Indicators_elasticityindicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的Data_frame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score_Indicators_elasticityindicators(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals.loc[:, 'buy_signal'] = result["BuySignal"]
        signals.loc[:, 'sell_signal'] = False  # 弹性指标主要用于识别买点
        signals.loc[:, 'neutral_signal'] = ~result["BuySignal"]
        
        # 设置趋势
        signals.loc[:, 'trend'] = 0  # 默认中性
        signals.loc[result["BuySignal"], 'trend'] = 1  # 弹性买点看涨
        
        # 设置评分
        signals.loc[:, 'score'] = score
        
        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        signals.loc[result["BuySignal"], 'signal_type'] = 'elasticity_buy'
        
        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        
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
        signals.loc[:, 'confidence'] = 60  # 基础置信度
        
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
        signals.loc[:, 'risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位
        signals.loc[:, 'position_size'] = 0.0
        signals.loc[result["BuySignal"], 'position_size'] = 0.3  # 基础仓位
        signals.loc[(result["BuySignal"]) & (score > 70), 'position_size'] = 0.5  # 高分仓位
        signals.loc[(result["BuySignal"]) & (score > 85), 'position_size'] = 0.7  # 极高分仓位
        
        # 止损位 - 使用区间最低价
        signals.loc[:, 'stop_loss'] = 0.0
        mask = result["BuySignal"]
        for i in data.index[mask]:
            period = self.period
            try:
                idx = data.index.get_loc(i)
                if idx >= period:
                    low_price = data.iloc[idx-period:idx+1]['low'].min()
                    signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(f"计算止损价格时出错: {e}")
                continue
        
        # 市场环境和成交量确认
        signals.loc[:, 'market_env'] = 'normal'
        signals.loc[:, 'volume_confirmation'] = result["VolumeRatio"] > 1.0
        
        return signals

class BounceDetector(BaseIndicator, PatternSignalMixin):
    """
    ZXM反弹检测器

    检测价格反弹和回调信号
    """
    
    def generate_signals_Indicators_elasticityindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的Data_frame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score_Indicators_elasticityindicators(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals.loc[:, 'buy_signal'] = result["PullbackBuyPoint"] | result["BounceSignal"]
        signals.loc[:, 'sell_signal'] = result["BounceSellPoint"]
        signals.loc[:, 'neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals.loc[:, 'trend'] = result["PriceTrend"]
        
        # 设置评分
        signals.loc[:, 'score'] = score
        
        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        signals.loc[result["PullbackBuyPoint"], 'signal_type'] = 'pullback_buy'
        signals.loc[result["BounceSignal"] & ~result["PullbackBuyPoint"], 'signal_type'] = 'bounce_signal'
        signals.loc[result["BounceSellPoint"], 'signal_type'] = 'bounce_sell'
        
        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        
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
        signals.loc[:, 'confidence'] = 60  # 基础置信度
        
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
        signals.loc[:, 'risk_level'] = '中'  # 默认中等风险
        
        # 标准回调买点风险较低
        perfect_pullback = result["PullbackBuyPoint"] & (result["PullbackFromHigh"] >= 10) & (result["PullbackFromHigh"] <= 20)
        signals.loc[perfect_pullback, 'risk_level'] = '低'
        
        # 过度回调或大幅反弹后的操作风险较高
        signals.loc[result["PullbackFromHigh"] > 30, 'risk_level'] = '高'
        signals.loc[result["BounceFromLow"] > 30, 'risk_level'] = '高'
        
        # 建议仓位
        signals.loc[:, 'position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 标准回调买点可以加大仓位
        signals.loc[perfect_pullback, 'position_size'] = 0.5
        
        # 止损位
        signals.loc[:, 'stop_loss'] = 0.0
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= self.long_period:
                    # 使用最近低点作为止损位
                    low_price = result.loc[i, "LongLow"]
                    signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(f"计算止损价格时出错: {e}")
                continue
        
        # 市场环境
        signals.loc[:, 'market_env'] = 'normal'
        
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
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(f"计算市场环境时出错: {e}")
                continue
        
        # 成交量确认
        signals.loc[:, 'volume_confirmation'] = result["VolumeChange"] > 0
        
        return signals

    def get_pattern_info_Indicators_Elasticity_Indicators(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # ZXMRiseElasticity指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)