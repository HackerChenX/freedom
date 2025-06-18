"""
ZXM体系买点指标模块

实现ZXM体系的5个买点指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMDailyMACD(BaseIndicator):
    """
    ZXM买点-日MACD指标
    
    判断日线MACD指标是否小于0.9
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM买点-日MACD指标"""
        super().__init__(name="ZXMDailyMACD", description="ZXM买点-日MACD指标，判断日线MACD值是否小于0.9")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-日MACD指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        MACD:=2*(DIFF-DEA);
        xg:MACD<0.9
        """
        # 确保数据包含必需的列
        if 'close' not in data.columns:
            raise ValueError("数据缺少必需的'close'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd = 2 * (diff - dea)
        
        # 计算买点信号
        xg = macd < 0.9
        
        # 添加计算结果到数据框
        result.loc[:, "EMA12"] = ema12
        result.loc[:, "EMA26"] = ema26
        result.loc[:, "DIFF"] = diff
        result.loc[:, "DEA"] = dea
        result.loc[:, "MACD"] = macd
        result.loc[:, "XG"] = xg
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM日线MACD指标的原始评分
        
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

        # 主要信号评分规则
        # 1. MACD小于0.9的买点信号：+30分
        score[result["XG"]] += 30

        # 2. MACD为正值加分
        macd_positive = result["MACD"] > 0
        score[macd_positive] += 15

        # 3. MACD上升趋势加分
        macd_rising = result["MACD"] > result["MACD"].shift(1)
        score[macd_rising] += 15

        # 4. DIFF和DEA都为正值且DIFF>DEA（多头排列）
        bullish_alignment = (result["DIFF"] > 0) & (result["DEA"] > 0) & (result["DIFF"] > result["DEA"])
        score[bullish_alignment] += 10

        # 5. MACD金叉信号
        golden_cross = (result["DIFF"] > result["DEA"]) & (result["DIFF"].shift(1) <= result["DEA"].shift(1))
        score[golden_cross] += 20
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM日线MACD指标相关的技术形态

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
                patterns.append("日线MACD买点信号")

            # MACD值形态判断
            macd_value = last_row["MACD"]
            if macd_value > 0:
                patterns.append("日线MACD为正值")
            else:
                patterns.append("日线MACD为负值")

            # MACD趋势判断
            if len(result) > 1:
                prev_macd = result["MACD"].iloc[-2]
                if macd_value > prev_macd:
                    patterns.append("日线MACD上升趋势")
                elif macd_value < prev_macd:
                    patterns.append("日线MACD下降趋势")

            # DIFF和DEA关系判断
            diff_value = last_row["DIFF"]
            dea_value = last_row["DEA"]

            if diff_value > dea_value:
                patterns.append("日线MACD多头排列")
                # 检查是否为金叉
                if len(result) > 1:
                    prev_diff = result["DIFF"].iloc[-2]
                    prev_dea = result["DEA"].iloc[-2]
                    if prev_diff <= prev_dea:
                        patterns.append("日线MACD金叉形成")
            else:
                patterns.append("日线MACD空头排列")
                # 检查是否为死叉
                if len(result) > 1:
                    prev_diff = result["DIFF"].iloc[-2]
                    prev_dea = result["DEA"].iloc[-2]
                    if prev_diff >= prev_dea:
                        patterns.append("日线MACD死叉形成")

            # MACD值区间判断
            if abs(macd_value) < 0.5:
                patterns.append("日线MACD接近零轴")
            elif macd_value < -2:
                patterns.append("日线MACD严重超卖")
            elif macd_value > 2:
                patterns.append("日线MACD严重超买")

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
        signals.loc[:, 'buy_signal'] = result["XG"]
        signals.loc[:, 'sell_signal'] = ~result["XG"]
        signals.loc[:, 'neutral_signal'] = False

        # 设置趋势
        signals.loc[:, 'trend'] = 0  # 默认中性
        signals.loc[result["MACD"] > 0, 'trend'] = 1  # MACD为正看涨
        signals.loc[result["MACD"] < 0, 'trend'] = -1  # MACD为负看跌

        # 设置评分
        signals.loc[:, 'score'] = score

        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        signals.loc[result["XG"], 'signal_type'] = 'buy_point'

        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        for i in signals.index:
            if result.loc[i, "XG"]:
                signals.loc[i, 'signal_desc'] = f"日线MACD买点信号，MACD值{result.loc[i, 'MACD']:.3f}"
            else:
                signals.loc[i, 'signal_desc'] = f"日线MACD非买点，MACD值{result.loc[i, 'MACD']:.3f}"

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
        if "日线MACD买点信号" in patterns:
            pattern_boost += 0.2
        if "日线MACD金叉形成" in patterns:
            pattern_boost += 0.15
        if "日线MACD多头排列" in patterns:
            pattern_boost += 0.1
        if "日线MACD上升趋势" in patterns:
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

        # 基础形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_DAILY_MACD_BUY_POINT"] = result["XG"]
        patterns_df.loc[:, "ZXM_DAILY_MACD_POSITIVE"] = result["MACD"] > 0
        patterns_df.loc[:, "ZXM_DAILY_MACD_NEGATIVE"] = result["MACD"] < 0
        patterns_df.loc[:, "ZXM_DAILY_MACD_RISING"] = result["MACD"] > result["MACD"].shift(1)
        patterns_df.loc[:, "ZXM_DAILY_MACD_FALLING"] = result["MACD"] < result["MACD"].shift(1)

        # DIFF和DEA关系形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_DAILY_MACD_BULLISH_ALIGNMENT"] = result["DIFF"] > result["DEA"]
        patterns_df.loc[:, "ZXM_DAILY_MACD_BEARISH_ALIGNMENT"] = result["DIFF"] < result["DEA"]

        # 金叉死叉形态 - 使用注册的pattern_id
        diff_cross_above_dea = (result["DIFF"] > result["DEA"]) & (result["DIFF"].shift(1) <= result["DEA"].shift(1))
        diff_cross_below_dea = (result["DIFF"] < result["DEA"]) & (result["DIFF"].shift(1) >= result["DEA"].shift(1))

        patterns_df.loc[:, "ZXM_DAILY_MACD_GOLDEN_CROSS"] = diff_cross_above_dea
        patterns_df.loc[:, "ZXM_DAILY_MACD_DEATH_CROSS"] = diff_cross_below_dea

        # MACD值区间形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_DAILY_MACD_NEAR_ZERO"] = abs(result["MACD"]) < 0.5
        patterns_df.loc[:, "ZXM_DAILY_MACD_OVERSOLD"] = result["MACD"] < -2
        patterns_df.loc[:, "ZXM_DAILY_MACD_OVERBOUGHT"] = result["MACD"] > 2

        return patterns_df

    def register_patterns(self):
        """
        注册ZXMDailyMACD指标的形态到全局形态注册表
        """
        # 注册主要买点信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_BUY_POINT",
            display_name="ZXM日线MACD买点信号",
            description="日线MACD值小于0.9，ZXM体系买点信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        # 注册MACD值状态形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_POSITIVE",
            display_name="ZXM日线MACD为正值",
            description="日线MACD值为正，表明多头力量占优",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_NEGATIVE",
            display_name="ZXM日线MACD为负值",
            description="日线MACD值为负，表明空头力量占优",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册MACD趋势形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_RISING",
            display_name="ZXM日线MACD上升趋势",
            description="日线MACD呈上升趋势，动能增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_FALLING",
            display_name="ZXM日线MACD下降趋势",
            description="日线MACD呈下降趋势，动能减弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册MACD排列形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_BULLISH_ALIGNMENT",
            display_name="ZXM日线MACD多头排列",
            description="DIFF大于DEA，多头排列",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_BEARISH_ALIGNMENT",
            display_name="ZXM日线MACD空头排列",
            description="DIFF小于DEA，空头排列",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        # 注册MACD交叉形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_GOLDEN_CROSS",
            display_name="ZXM日线MACD金叉",
            description="DIFF上穿DEA，金叉形成",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_DEATH_CROSS",
            display_name="ZXM日线MACD死叉",
            description="DIFF下穿DEA，死叉形成",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册MACD极值形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_NEAR_ZERO",
            display_name="ZXM日线MACD接近零轴",
            description="MACD值接近零轴，可能变盘",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_OVERSOLD",
            display_name="ZXM日线MACD严重超卖",
            description="MACD值严重超卖，可能反弹",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_MACD_OVERBOUGHT",
            display_name="ZXM日线MACD严重超买",
            description="MACD值严重超买，可能回调",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - fast_period: 快线周期，默认12
                - slow_period: 慢线周期，默认26
                - signal_period: 信号线周期，默认9
                - threshold: MACD阈值，默认0.9
        """
        self.fast_period = kwargs.get('fast_period', 12)
        self.slow_period = kwargs.get('slow_period', 26)
        self.signal_period = kwargs.get('signal_period', 9)
        self.threshold = kwargs.get('threshold', 0.9)


class ZXMTurnover(BaseIndicator):
    """
    ZXM买点-换手率指标
    
    判断日线换手率是否大于0.7%
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM买点-换手率指标"""
        super().__init__(name="ZXMTurnover", description="ZXM买点-换手率指标，判断日线换手率是否大于0.7%")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-换手率指标
        
        Args:
            data: 输入数据，包含成交量和换手率数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        换手率>0.7;
        xg:换手;
        """
        # 确保数据包含必需的列
        if 'turnover_rate' not in data.columns:
            raise ValueError("数据缺少必需的'turnover_rate'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 直接使用数据库提供的换手率
        turnover = data["turnover_rate"]
        
        # 计算买点信号
        xg = turnover > 0.7
        
        # 添加计算结果到数据框
        result.loc[:, "Turnover"] = turnover
        result.loc[:, "XG"] = xg
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM换手率指标的原始评分
        
        Args:
            data: 输入数据，包含换手率数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)

        # 主要信号评分规则
        # 1. 换手率大于0.7%的买点信号：+30分
        score[result["XG"]] += 30

        # 2. 换手率活跃度评分
        turnover = result["Turnover"]

        # 换手率越高，活跃度越高
        score[turnover > 1.0] += 15  # 换手率>1%，非常活跃
        score[(turnover > 0.7) & (turnover <= 1.0)] += 10  # 换手率0.7%-1%，活跃
        score[(turnover > 0.5) & (turnover <= 0.7)] += 5   # 换手率0.5%-0.7%，一般活跃

        # 3. 相对换手率评分（与历史平均比较）
        if len(turnover) >= 20:
            avg_turnover_20 = turnover.rolling(window=20).mean()
            relative_active = turnover > avg_turnover_20 * 1.5
            score[relative_active] += 10

        # 4. 换手率过高风险评分
        score[turnover > 5.0] -= 10  # 换手率过高可能是炒作
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM换手率指标相关的技术形态

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
                patterns.append("换手率充分活跃")

            # 换手率活跃度判断
            turnover = last_row["Turnover"]
            if turnover > 5.0:
                patterns.append("换手率极度活跃")
            elif turnover > 2.0:
                patterns.append("换手率非常活跃")
            elif turnover > 1.0:
                patterns.append("换手率活跃")
            elif turnover > 0.7:
                patterns.append("换手率一般活跃")
            else:
                patterns.append("换手率低迷")

            # 相对活跃度判断
            if len(result) >= 20:
                avg_turnover_20 = result["Turnover"].rolling(window=20).mean().iloc[-1]
                if turnover > avg_turnover_20 * 2:
                    patterns.append("换手率相对历史极度活跃")
                elif turnover > avg_turnover_20 * 1.5:
                    patterns.append("换手率相对历史活跃")
                elif turnover < avg_turnover_20 * 0.5:
                    patterns.append("换手率相对历史低迷")

            # 换手率趋势判断
            if len(result) >= 5:
                recent_trend = result["Turnover"].iloc[-5:].mean()
                if turnover > recent_trend * 1.3:
                    patterns.append("换手率突然放大")
                elif turnover < recent_trend * 0.7:
                    patterns.append("换手率突然缩小")

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
        if "换手率买点信号" in patterns:
            pattern_boost += 0.15
        if "换手率相对历史活跃" in patterns:
            pattern_boost += 0.1
        if "换手率突然放大" in patterns:
            pattern_boost += 0.1
        if "换手率极度活跃" in patterns:
            pattern_boost -= 0.1  # 过度活跃可能是风险

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

        # 基础形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_TURNOVER_BUY_POINT"] = result["XG"]

        # 换手率活跃度形态 - 使用注册的pattern_id
        turnover = result["Turnover"]
        patterns_df.loc[:, "ZXM_TURNOVER_EXTREMELY_ACTIVE"] = turnover > 5.0
        patterns_df.loc[:, "ZXM_TURNOVER_VERY_ACTIVE"] = (turnover > 2.0) & (turnover <= 5.0)
        patterns_df.loc[:, "ZXM_TURNOVER_ACTIVE"] = (turnover > 1.0) & (turnover <= 2.0)
        patterns_df.loc[:, "ZXM_TURNOVER_NORMAL_ACTIVE"] = (turnover > 0.7) & (turnover <= 1.0)
        patterns_df.loc[:, "ZXM_TURNOVER_LOW"] = turnover <= 0.7

        # 相对活跃度形态 - 使用注册的pattern_id
        if len(result) >= 20:
            avg_turnover_20 = turnover.rolling(window=20).mean()
            patterns_df.loc[:, "ZXM_TURNOVER_RELATIVE_EXTREMELY_ACTIVE"] = turnover > avg_turnover_20 * 2
            patterns_df.loc[:, "ZXM_TURNOVER_RELATIVE_ACTIVE"] = (turnover > avg_turnover_20 * 1.5) & (turnover <= avg_turnover_20 * 2)
            patterns_df.loc[:, "ZXM_TURNOVER_RELATIVE_LOW"] = turnover < avg_turnover_20 * 0.5

        # 换手率趋势形态 - 使用注册的pattern_id
        if len(result) >= 5:
            recent_trend = turnover.rolling(window=5).mean()
            patterns_df.loc[:, "ZXM_TURNOVER_SUDDEN_INCREASE"] = turnover > recent_trend * 1.3
            patterns_df.loc[:, "ZXM_TURNOVER_SUDDEN_DECREASE"] = turnover < recent_trend * 0.7

        return patterns_df

    def register_patterns(self):
        """
        注册ZXMTurnover指标的形态到全局形态注册表
        """
        # 注册主要买点信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_BUY_POINT",
            display_name="ZXM换手率买点信号",
            description="换手率大于0.7%，ZXM体系买点信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册换手率活跃度形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_EXTREMELY_ACTIVE",
            display_name="ZXM换手率极度活跃",
            description="换手率>5%，极度活跃，需要谨慎",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_VERY_ACTIVE",
            display_name="ZXM换手率非常活跃",
            description="换手率2%-5%，非常活跃",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_ACTIVE",
            display_name="ZXM换手率活跃",
            description="换手率1%-2%，活跃",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=8.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_NORMAL_ACTIVE",
            display_name="ZXM换手率一般活跃",
            description="换手率0.7%-1%，一般活跃",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=5.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_LOW",
            display_name="ZXM换手率低迷",
            description="换手率≤0.7%，低迷",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-5.0,
            polarity="NEGATIVE"
        )

        # 注册相对活跃度形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_RELATIVE_EXTREMELY_ACTIVE",
            display_name="ZXM换手率相对历史极度活跃",
            description="换手率相对20日均值极度活跃",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_RELATIVE_ACTIVE",
            display_name="ZXM换手率相对历史活跃",
            description="换手率相对20日均值活跃",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_RELATIVE_LOW",
            display_name="ZXM换手率相对历史低迷",
            description="换手率相对20日均值低迷",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-8.0,
            polarity="NEGATIVE"
        )

        # 注册换手率变化形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_SUDDEN_INCREASE",
            display_name="ZXM换手率突然放大",
            description="换手率突然放大，关注资金流入",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_SUDDEN_DECREASE",
            display_name="ZXM换手率突然缩小",
            description="换手率突然缩小，资金流出",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-8.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_TURNOVER_SUDDEN_DECREASE",
            display_name="ZXM换手率突然缩小",
            description="换手率突然缩小，关注资金流出",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-8.0,
            polarity="NEGATIVE"
        )

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - threshold: 换手率阈值，默认0.7
        """
        self.threshold = kwargs.get('threshold', 0.7)


class ZXMVolumeShrink(BaseIndicator):
    """
    ZXM买点-缩量指标
    
    判断成交量是否较2日平均成交量缩减10%以上
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM买点-缩量指标"""
        super().__init__(name="ZXMVolumeShrink", description="ZXM买点-缩量指标，判断成交量是否明显缩量")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-缩量指标
        
        Args:
            data: 输入数据，包含成交量数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        VOL/MA(VOL,2)<0.9;
        """
        # 确保数据包含必需的列
        if 'volume' not in data.columns:
            raise ValueError("数据缺少必需的'volume'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算2日均量
        ma_vol_2 = data["volume"].rolling(window=2).mean()
        
        # 计算量比
        vol_ratio = data["volume"] / ma_vol_2
        
        # 计算买点信号
        xg = vol_ratio < 0.9
        
        # 添加计算结果到数据框
        result.loc[:, "MA_VOL_2"] = ma_vol_2
        result.loc[:, "VOL_RATIO"] = vol_ratio
        result.loc[:, "XG"] = xg
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM缩量指标的原始评分
        
        Args:
            data: 输入数据，包含成交量数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)

        # 主要信号评分规则
        # 1. 缩量买点信号：+30分
        score[result["XG"]] += 30

        # 2. 缩量程度评分
        vol_ratio = result["VOL_RATIO"]

        # 缩量越明显，评分越高
        score[vol_ratio < 0.7] += 15  # 严重缩量
        score[(vol_ratio >= 0.7) & (vol_ratio < 0.8)] += 10  # 明显缩量
        score[(vol_ratio >= 0.8) & (vol_ratio < 0.9)] += 5   # 轻微缩量

        # 3. 连续缩量评分
        if len(result) >= 3:
            consecutive_shrink = pd.Series(False, index=data.index)
            for i in range(2, len(result)):
                if all(result["XG"].iloc[i-2:i+1]):
                    consecutive_shrink.iloc[i] = True
            score[consecutive_shrink] += 15

        # 4. 缩量配合价格稳定评分
        if 'close' in data.columns and len(data) >= 3:
            price_stable = abs(data['close'].pct_change(3)) < 0.05  # 3日内价格变化小于5%
            volume_shrink_with_stable_price = result["XG"] & price_stable
            score[volume_shrink_with_stable_price] += 10
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM缩量指标相关的技术形态

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
                patterns.append("缩量买点信号")

            # 缩量程度判断
            vol_ratio = last_row["VOL_RATIO"]
            if vol_ratio < 0.5:
                patterns.append("严重缩量")
            elif vol_ratio < 0.7:
                patterns.append("明显缩量")
            elif vol_ratio < 0.9:
                patterns.append("轻微缩量")
            else:
                patterns.append("成交量正常")

            # 连续缩量判断
            if len(result) >= 3:
                if all(result["XG"].iloc[-3:]):
                    patterns.append("连续缩量")

            # 缩量配合价格稳定判断
            if 'close' in data.columns and len(data) >= 3:
                price_change = abs(data['close'].iloc[-1] / data['close'].iloc[-4] - 1)
                if last_row["XG"] and price_change < 0.05:
                    patterns.append("缩量整理")

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
        if "缩量买点信号" in patterns:
            pattern_boost += 0.15
        if "严重缩量" in patterns:
            pattern_boost += 0.15
        if "连续缩量" in patterns:
            pattern_boost += 0.1
        if "缩量整理" in patterns:
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

        # 基础形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_VOLUME_SHRINK_BUY_POINT"] = result["XG"]

        # 缩量程度形态 - 使用注册的pattern_id
        vol_ratio = result["VOL_RATIO"]
        patterns_df.loc[:, "ZXM_VOLUME_SEVERE_SHRINK"] = vol_ratio < 0.5
        patterns_df.loc[:, "ZXM_VOLUME_OBVIOUS_SHRINK"] = (vol_ratio >= 0.5) & (vol_ratio < 0.7)
        patterns_df.loc[:, "ZXM_VOLUME_SLIGHT_SHRINK"] = (vol_ratio >= 0.7) & (vol_ratio < 0.9)
        patterns_df.loc[:, "ZXM_VOLUME_NORMAL"] = vol_ratio >= 0.9

        # 连续缩量形态 - 使用注册的pattern_id
        if len(result) >= 3:
            consecutive_shrink = pd.Series(False, index=data.index)
            for i in range(2, len(result)):
                if all(result["XG"].iloc[i-2:i+1]):
                    consecutive_shrink.iloc[i] = True
            patterns_df.loc[:, "ZXM_VOLUME_CONSECUTIVE_SHRINK"] = consecutive_shrink

        # 缩量整理形态 - 使用注册的pattern_id
        if 'close' in data.columns and len(data) >= 3:
            price_stable = abs(data['close'].pct_change(3)) < 0.05
            patterns_df.loc[:, "ZXM_VOLUME_SHRINK_CONSOLIDATION"] = result["XG"] & price_stable

        return patterns_df

    def register_patterns(self):
        """
        注册ZXMVolumeShrink指标的形态到全局形态注册表
        """
        # 注册主要买点信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_SHRINK_BUY_POINT",
            display_name="ZXM缩量买点信号",
            description="成交量明显缩量，ZXM体系买点信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        # 注册缩量程度形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_SEVERE_SHRINK",
            display_name="ZXM严重缩量",
            description="成交量严重缩量，量比<0.5",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=5.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_OBVIOUS_SHRINK",
            display_name="ZXM明显缩量",
            description="成交量明显缩量，量比0.5-0.7",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=3.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_SLIGHT_SHRINK",
            display_name="ZXM轻微缩量",
            description="成交量轻微缩量，量比0.7-0.9",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=1.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_NORMAL",
            display_name="ZXM成交量正常",
            description="成交量正常，量比≥0.9",
            pattern_type="NEUTRAL",
            default_strength="WEAK",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册连续缩量形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_CONSECUTIVE_SHRINK",
            display_name="ZXM连续缩量",
            description="连续3日缩量，市场观望情绪浓厚",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=8.0,
            polarity="NEUTRAL"
        )

        # 注册缩量整理形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_SHRINK_CONSOLIDATION",
            display_name="ZXM缩量整理",
            description="缩量配合价格整理，蓄势待发",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - ma_period: 均量计算周期，默认2
                - shrink_threshold: 缩量阈值，默认0.9
        """
        self.ma_period = kwargs.get('ma_period', 2)
        self.shrink_threshold = kwargs.get('shrink_threshold', 0.9)


class ZXMMACallback(BaseIndicator):
    """
    ZXM买点-回踩均线指标
    
    判断收盘价是否回踩至20日、30日、60日或120日均线的N%以内
    """
    
    def __init__(self, callback_percent: float = 4.0):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ZXM买点-回踩均线指标
        
        Args:
            callback_percent: 回踩百分比，默认为4%
        """
        super().__init__(name="ZXMMACallback", description="ZXM买点-回踩均线指标，判断价格是否回踩至关键均线附近")
        self.callback_percent = callback_percent
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-回踩均线指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        A20:=ABS((C/MA(C,20)-1)*100)<= N;
        A30:=ABS((C/MA(C,30)-1)*100)<= N;
        A60:=ABS((C/MA(C,60)-1)*100)<= N;
        A120:=ABS((C/MA(C,120)-1)*100)<= N;
        XG:A20 OR A30 OR A60 OR A120;
        """
        # 确保数据包含必需的列
        if 'close' not in data.columns:
            raise ValueError("数据缺少必需的'close'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算各均线
        ma20 = data["close"].rolling(window=20).mean()
        ma30 = data["close"].rolling(window=30).mean()
        ma60 = data["close"].rolling(window=60).mean()
        ma120 = data["close"].rolling(window=120).mean()
        
        # 计算收盘价与各均线的偏离百分比
        a20 = abs((data["close"] / ma20 - 1) * 100) <= self.callback_percent
        a30 = abs((data["close"] / ma30 - 1) * 100) <= self.callback_percent
        a60 = abs((data["close"] / ma60 - 1) * 100) <= self.callback_percent
        a120 = abs((data["close"] / ma120 - 1) * 100) <= self.callback_percent
        
        # 计算买点信号
        xg = a20 | a30 | a60 | a120
        
        # 添加计算结果到数据框
        result.loc[:, "MA20"] = ma20
        result.loc[:, "MA30"] = ma30
        result.loc[:, "MA60"] = ma60
        result.loc[:, "MA120"] = ma120
        result.loc[:, "A20"] = a20
        result.loc[:, "A30"] = a30
        result.loc[:, "A60"] = a60
        result.loc[:, "A120"] = a120
        result.loc[:, "XG"] = xg
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM均线回调指标的原始评分
        
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

        # 主要信号评分规则
        # 1. 回踩均线买点信号：+30分
        score[result["XG"]] += 30

        # 2. 回踩到不同均线的评分
        score[result["A20"]] += 5   # 回踩20日线
        score[result["A30"]] += 8   # 回踩30日线
        score[result["A60"]] += 12  # 回踩60日线
        score[result["A120"]] += 15 # 回踩120日线

        # 3. 多条均线同时回踩加分
        ma_count = result["A20"].astype(int) + result["A30"].astype(int) + result["A60"].astype(int) + result["A120"].astype(int)
        score[ma_count >= 2] += 10  # 同时回踩2条以上均线
        score[ma_count >= 3] += 15  # 同时回踩3条以上均线

        # 4. 均线支撑强度评分
        if 'close' in data.columns:
            close_price = data['close']
            # 价格在均线上方但接近均线（支撑有效）
            for ma_col, a_col in [('MA20', 'A20'), ('MA30', 'A30'), ('MA60', 'A60'), ('MA120', 'A120')]:
                if ma_col in result.columns:
                    above_ma = close_price > result[ma_col]
                    near_ma = result[a_col]
                    valid_support = above_ma & near_ma
                    score[valid_support] += 5
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM均线回调指标相关的技术形态

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
                patterns.append("均线回调买点信号")

            # 具体回调均线判断
            if last_row["A20"]:
                patterns.append("回踩20日均线")
            if last_row["A30"]:
                patterns.append("回踩30日均线")
            if last_row["A60"]:
                patterns.append("回踩60日均线")
            if last_row["A120"]:
                patterns.append("回踩120日均线")

            # 多重回调判断
            ma_count = sum([last_row["A20"], last_row["A30"], last_row["A60"], last_row["A120"]])
            if ma_count >= 3:
                patterns.append("多重均线回调")
            elif ma_count >= 2:
                patterns.append("双重均线回调")

            # 支撑强度判断
            if 'close' in data.columns:
                close_price = data['close'].iloc[-1]
                for ma_name, ma_col, a_col in [
                    ("20日线", "MA20", "A20"), ("30日线", "MA30", "A30"),
                    ("60日线", "MA60", "A60"), ("120日线", "MA120", "A120")
                ]:
                    if ma_col in result.columns and last_row[a_col]:
                        ma_value = last_row[ma_col]
                        if close_price > ma_value:
                            patterns.append(f"{ma_name}有效支撑")
                        else:
                            patterns.append(f"{ma_name}跌破风险")

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
        if "均线回调买点信号" in patterns:
            pattern_boost += 0.15
        if "多重均线回调" in patterns:
            pattern_boost += 0.15
        elif "双重均线回调" in patterns:
            pattern_boost += 0.1

        # 长期均线支撑更可靠
        if "120日线有效支撑" in patterns:
            pattern_boost += 0.15
        elif "60日线有效支撑" in patterns:
            pattern_boost += 0.1

        # 跌破风险降低置信度
        if any("跌破风险" in p for p in patterns):
            pattern_boost -= 0.1

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

        # 基础形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_MA_CALLBACK_BUY_POINT"] = result["XG"]
        patterns_df.loc[:, "ZXM_MA20_CALLBACK"] = result["A20"]
        patterns_df.loc[:, "ZXM_MA30_CALLBACK"] = result["A30"]
        patterns_df.loc[:, "ZXM_MA60_CALLBACK"] = result["A60"]
        patterns_df.loc[:, "ZXM_MA120_CALLBACK"] = result["A120"]

        # 多重回调形态 - 使用注册的pattern_id
        ma_count = result["A20"].astype(int) + result["A30"].astype(int) + result["A60"].astype(int) + result["A120"].astype(int)
        patterns_df.loc[:, "ZXM_MULTIPLE_MA_CALLBACK"] = ma_count >= 3
        patterns_df.loc[:, "ZXM_DOUBLE_MA_CALLBACK"] = ma_count == 2
        patterns_df.loc[:, "ZXM_SINGLE_MA_CALLBACK"] = ma_count == 1

        # 支撑有效性形态 - 使用注册的pattern_id
        if 'close' in data.columns:
            close_price = data['close']
            for ma_col, a_col, pattern_name in [
                ("MA20", "A20", "ZXM_MA20_SUPPORT"), ("MA30", "A30", "ZXM_MA30_SUPPORT"),
                ("MA60", "A60", "ZXM_MA60_SUPPORT"), ("MA120", "A120", "ZXM_MA120_SUPPORT")
            ]:
                if ma_col in result.columns:
                    patterns_df[pattern_name] = (close_price > result[ma_col]) & result[a_col]

        return patterns_df

    def register_patterns(self):
        """
        注册ZXMMACallback指标的形态到全局形态注册表
        """
        # 注册主要买点信号
        self.register_pattern_to_registry(
            pattern_id="ZXM_MA_CALLBACK_BUY_POINT",
            display_name="ZXM均线回调买点信号",
            description="价格回踩至关键均线附近，ZXM体系买点信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册具体均线回调形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_MA20_CALLBACK",
            display_name="ZXM回踩20日均线",
            description="价格回踩至20日均线4%范围内",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=8.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MA30_CALLBACK",
            display_name="ZXM回踩30日均线",
            description="价格回踩至30日均线4%范围内",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MA60_CALLBACK",
            display_name="ZXM回踩60日均线",
            description="价格回踩至60日均线4%范围内",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MA120_CALLBACK",
            display_name="ZXM回踩120日均线",
            description="价格回踩至120日均线4%范围内",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册多重回调形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_MULTIPLE_MA_CALLBACK",
            display_name="ZXM多重均线回调",
            description="同时回踩3条以上均线，支撑强劲",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DOUBLE_MA_CALLBACK",
            display_name="ZXM双重均线回调",
            description="同时回踩2条均线，支撑较强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_SINGLE_MA_CALLBACK",
            display_name="ZXM单一均线回调",
            description="回踩单一均线，支撑一般",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册支撑有效性形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_MA20_SUPPORT",
            display_name="ZXM20日线有效支撑",
            description="价格在20日均线上方获得支撑",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=5.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MA30_SUPPORT",
            display_name="ZXM30日线有效支撑",
            description="价格在30日均线上方获得支撑",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=8.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MA60_SUPPORT",
            display_name="ZXM60日线有效支撑",
            description="价格在60日均线上方获得支撑",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_MA120_SUPPORT",
            display_name="ZXM120日线有效支撑",
            description="价格在120日均线上方获得支撑",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - callback_percent: 回踩百分比，默认4.0
        """
        self.callback_percent = kwargs.get('callback_percent', 4.0)
    
    
class ZXMBSAbsorb(BaseIndicator):
    """
    ZXM买点-BS吸筹指标
    
    判断60分钟级别是否存在低位吸筹特征
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM买点-BS吸筹指标"""
        super().__init__(name="ZXMBSAbsorb", description="ZXM买点-BS吸筹指标，判断60分钟级别是否存在低位吸筹特征")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点-BS吸筹指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为60分钟级别数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点信号
            
        公式说明：
        V11:=3*SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1)-2*SMA(SMA((C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100,5,1),3,1);
        V12:=(EMA(V11,3)-REF(EMA(V11,3),1))/REF(EMA(V11,3),1)*100;
        AA:=(EMA(V11,3)<=13) AND FILTER((EMA(V11,3)<=13),15);
        BB:=(EMA(V11,3)<=13 AND V12>13) AND FILTER((EMA(V11,3)<=13 AND V12>13),10);
        XG:COUNT(AA OR BB,6)
        
        注意：这里的FILTER函数表示在过去N周期内至少出现一次该条件
        """
        # 确保数据包含必需的列
        required_cols = ["close", "high", "low"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需的列: {missing_cols}")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算LLV和HHV
        llv_55 = data["low"].rolling(window=55).min()
        hhv_55 = data["high"].rolling(window=55).max()
        
        # 计算RSV变种
        rsv_55 = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = hhv_55 - llv_55
        valid_idx = divisor > 0
        rsv_55[valid_idx] = ((data["close"] - llv_55) / divisor * 100)[valid_idx]
        
        # 计算V11
        sma_rsv_5 = self._sma(rsv_55, 5, 1)
        sma_sma_3 = self._sma(sma_rsv_5, 3, 1)
        v11 = 3 * sma_rsv_5 - 2 * sma_sma_3
        
        # 计算V11的EMA
        ema_v11_3 = v11.ewm(span=3, adjust=False).mean()
        
        # 计算V12
        v12 = pd.Series(np.zeros(len(data)), index=data.index)
        valid_idx = ema_v11_3.shift(1) != 0
        v12[valid_idx] = ((ema_v11_3 - ema_v11_3.shift(1)) / ema_v11_3.shift(1) * 100)[valid_idx]
        
        # 计算AA和BB条件
        aa_base = ema_v11_3 <= 13
        aa_filter = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        for i in range(15, len(data)):
            aa_filter.iloc[i] = np.any(aa_base.iloc[i-14:i+1])
        aa = aa_base & aa_filter
        
        bb_base = (ema_v11_3 <= 13) & (v12 > 13)
        bb_filter = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
        for i in range(10, len(data)):
            bb_filter.iloc[i] = np.any(bb_base.iloc[i-9:i+1])
        bb = bb_base & bb_filter
        
        # 计算XG：近6周期内AA或BB条件满足的次数
        xg = pd.Series(np.zeros(len(data), dtype=int), index=data.index)
        for i in range(6, len(data)):
            xg.iloc[i] = np.sum((aa | bb).iloc[i-5:i+1])
        
        # 添加计算结果到数据框
        result.loc[:, "V11"] = v11
        result.loc[:, "EMA_V11_3"] = ema_v11_3
        result.loc[:, "V12"] = v12
        result.loc[:, "AA"] = aa
        result.loc[:, "BB"] = bb
        result.loc[:, "XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM庄家吸筹指标的原始评分
        
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
        # 1. BS吸筹信号强度评分
        xg_value = result["XG"]

        # 根据XG值（近6周期内满足条件的次数）评分
        score[xg_value >= 5] += 30  # 强烈吸筹信号
        score[(xg_value >= 3) & (xg_value < 5)] += 20  # 明显吸筹信号
        score[(xg_value >= 1) & (xg_value < 3)] += 10  # 轻微吸筹信号

        # 2. V11指标位置评分
        v11_ema = result["EMA_V11_3"]
        score[v11_ema <= 10] += 15  # V11极低位，强烈超卖
        score[(v11_ema > 10) & (v11_ema <= 13)] += 10  # V11低位，超卖
        score[v11_ema > 80] -= 10   # V11高位，可能超买

        # 3. V12动量评分
        v12_value = result["V12"]
        score[v12_value > 20] += 15  # 强烈上升动量
        score[(v12_value > 13) & (v12_value <= 20)] += 10  # 上升动量
        score[v12_value < -20] -= 10  # 下降动量

        # 4. AA和BB条件评分
        score[result["AA"]] += 10  # AA条件满足
        score[result["BB"]] += 15  # BB条件满足（更强信号）

        # 5. 连续满足条件加分
        if len(result) >= 3:
            consecutive_signal = pd.Series(False, index=data.index)
            for i in range(2, len(result)):
                if all((result["AA"] | result["BB"]).iloc[i-2:i+1]):
                    consecutive_signal.iloc[i] = True
            score[consecutive_signal] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM BS吸筹指标相关的技术形态

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

            # 吸筹强度判断
            xg_value = last_row["XG"]
            if xg_value >= 5:
                patterns.append("强烈吸筹信号")
            elif xg_value >= 3:
                patterns.append("明显吸筹信号")
            elif xg_value >= 1:
                patterns.append("轻微吸筹信号")
            else:
                patterns.append("无吸筹信号")

            # V11位置判断（基于吸筹技术含义）
            v11_ema = last_row["EMA_V11_3"]
            if v11_ema <= 10:
                patterns.append("主力大量吸筹区域")
            elif v11_ema <= 13:
                patterns.append("主力吸筹区域")
            elif v11_ema >= 80:
                patterns.append("高位调整区域")
            else:
                patterns.append("吸筹观察区间")

            # V12动量判断
            v12_value = last_row["V12"]
            if v12_value > 20:
                patterns.append("强烈上升动量")
            elif v12_value > 13:
                patterns.append("上升动量")
            elif v12_value < -20:
                patterns.append("下降动量")
            else:
                patterns.append("动量平稳")

            # AA和BB条件判断
            if last_row["AA"]:
                patterns.append("AA条件满足")
            if last_row["BB"]:
                patterns.append("BB条件满足")

            # 综合判断
            if last_row["AA"] and last_row["BB"]:
                patterns.append("双重吸筹确认")
            elif v11_ema <= 13 and v12_value > 13:
                patterns.append("低位反弹信号")

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
        if "强烈吸筹信号" in patterns:
            pattern_boost += 0.25
        elif "明显吸筹信号" in patterns:
            pattern_boost += 0.15
        elif "轻微吸筹信号" in patterns:
            pattern_boost += 0.1

        if "双重吸筹确认" in patterns:
            pattern_boost += 0.2
        elif "BB条件满足" in patterns:
            pattern_boost += 0.15
        elif "AA条件满足" in patterns:
            pattern_boost += 0.1

        if "V11极低位" in patterns:
            pattern_boost += 0.15
        elif "V11低位" in patterns:
            pattern_boost += 0.1

        if "强烈上升动量" in patterns:
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

        # 吸筹强度形态 - 使用注册的pattern_id
        xg_value = result["XG"]
        patterns_df.loc[:, "ZXM_BS_ABSORB_STRONG"] = xg_value >= 5
        patterns_df.loc[:, "ZXM_BS_ABSORB_OBVIOUS"] = (xg_value >= 3) & (xg_value < 5)
        patterns_df.loc[:, "ZXM_BS_ABSORB_SLIGHT"] = (xg_value >= 1) & (xg_value < 3)

        # V11位置形态（基于吸筹技术含义）- 使用注册的pattern_id
        v11_ema = result["EMA_V11_3"]
        patterns_df.loc[:, "ZXM_BS_ABSORB_HEAVY_ZONE"] = v11_ema <= 10
        patterns_df.loc[:, "ZXM_BS_ABSORB_ZONE"] = (v11_ema > 10) & (v11_ema <= 13)
        patterns_df.loc[:, "ZXM_BS_ABSORB_WATCH_ZONE"] = (v11_ema > 13) & (v11_ema < 80)
        patterns_df.loc[:, "ZXM_BS_HIGH_ADJUSTMENT"] = v11_ema >= 80

        # V12动量形态 - 使用注册的pattern_id
        v12_value = result["V12"]
        patterns_df.loc[:, "ZXM_BS_STRONG_MOMENTUM"] = v12_value > 20
        patterns_df.loc[:, "ZXM_BS_UP_MOMENTUM"] = (v12_value > 13) & (v12_value <= 20)
        patterns_df.loc[:, "ZXM_BS_STABLE_MOMENTUM"] = (v12_value >= -20) & (v12_value <= 13)

        # 条件满足形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_BS_DOUBLE_CONFIRM"] = result["AA"] & result["BB"]
        patterns_df.loc[:, "ZXM_BS_LOW_REBOUND"] = (v11_ema <= 13) & (v12_value > 13)

        return patterns_df

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - v11_threshold: V11阈值，默认13
                - v12_threshold: V12阈值，默认13
                - aa_filter_period: AA过滤周期，默认15
                - bb_filter_period: BB过滤周期，默认10
                - count_period: 计数周期，默认6
        """
        self.v11_threshold = kwargs.get('v11_threshold', 13)
        self.v12_threshold = kwargs.get('v12_threshold', 13)
        self.aa_filter_period = kwargs.get('aa_filter_period', 15)
        self.bb_filter_period = kwargs.get('bb_filter_period', 10)
        self.count_period = kwargs.get('count_period', 6)


class BuyPointDetector(BaseIndicator):
    """
    ZXM买点检测指标
    
    检测多种买点形态
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM买点检测指标"""
        super().__init__(name="BuyPointDetector", description="ZXM买点检测指标，检测多种买点形态")
        
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        """
        # 买点侦测器通常没有可变参数，但为了符合接口要求，提供此方法
        pass

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取买点侦测器的技术形态

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

        # 基础买点形态 - 使用注册的pattern_id
        patterns_df.loc[:, "ZXM_VOLUME_RISE_BUY"] = result["VolumeRiseBuyPoint"]
        patterns_df.loc[:, "ZXM_PULLBACK_STABILIZE_BUY"] = result["PullbackStabilizeBuyPoint"]
        patterns_df.loc[:, "ZXM_BREAKOUT_BUY"] = result["BreakoutBuyPoint"]
        patterns_df.loc[:, "ZXM_BOTTOM_VOLUME_BUY"] = result["BottomVolumeBuyPoint"]
        patterns_df.loc[:, "ZXM_VOLUME_SHRINK_BUY"] = result["VolumeShrinkBuyPoint"]
        patterns_df.loc[:, "ZXM_COMBINED_BUY"] = result["CombinedBuyPoint"]

        # 买点组合形态 - 使用注册的pattern_id
        buy_point_count = (
            result["VolumeRiseBuyPoint"].astype(int) +
            result["PullbackStabilizeBuyPoint"].astype(int) +
            result["BreakoutBuyPoint"].astype(int) +
            result["BottomVolumeBuyPoint"].astype(int) +
            result["VolumeShrinkBuyPoint"].astype(int)
        )

        patterns_df.loc[:, "ZXM_STRONG_MULTI_BUY"] = buy_point_count >= 3
        patterns_df.loc[:, "ZXM_DOUBLE_BUY"] = buy_point_count == 2
        patterns_df.loc[:, "ZXM_SINGLE_BUY"] = buy_point_count == 1

        return patterns_df

    def register_patterns(self):
        """
        注册BuyPointDetector指标的形态到全局形态注册表
        """
        # 注册基础买点形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_RISE_BUY",
            display_name="ZXM放量上涨买点",
            description="价格上涨配合成交量放大，涨幅适中的买点信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_PULLBACK_STABILIZE_BUY",
            display_name="ZXM回调企稳买点",
            description="前期上涨后小幅回调，缩量企稳回升的买点信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_BREAKOUT_BUY",
            display_name="ZXM突破买点",
            description="突破前期高点，放量确认的买点信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_BOTTOM_VOLUME_BUY",
            display_name="ZXM底部放量买点",
            description="底部横盘后突然放量，价格站上短期均线的买点信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_VOLUME_SHRINK_BUY",
            display_name="ZXM缩量整理买点",
            description="前期上涨后缩量整理，再次放量上涨的买点信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_COMBINED_BUY",
            display_name="ZXM组合买点",
            description="满足多种买点条件的综合买点信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        # 注册买点组合形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_STRONG_MULTI_BUY",
            display_name="ZXM强势多重买点组合",
            description="同时满足3种以上买点条件，买点信号极强",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_DOUBLE_BUY",
            display_name="ZXM双重买点组合",
            description="同时满足2种买点条件，买点信号较强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="ZXM_SINGLE_BUY",
            display_name="ZXM单一买点",
            description="满足单一买点条件，买点信号一般",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点指标
        
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
        
        # 计算各种买点
        
        # 1. 放量上涨买点
        result = self._calculate_volume_rise_buy_point(data, result)
        
        # 2. 回调企稳买点
        result = self._calculate_pullback_stabilize_buy_point(data, result)
        
        # 3. 突破买点
        result = self._calculate_breakout_buy_point(data, result)
        
        # 4. 底部放量买点
        result = self._calculate_bottom_volume_buy_point(data, result)
        
        # 5. 缩量整理买点
        result = self._calculate_volume_shrink_buy_point(data, result)
        
        # 6. 组合买点 - 满足多个买点的组合
        result.loc[:, "CombinedBuyPoint"] = (
            result["VolumeRiseBuyPoint"] | 
            result["PullbackStabilizeBuyPoint"] | 
            result["BreakoutBuyPoint"] | 
            result["BottomVolumeBuyPoint"] | 
            result["VolumeShrinkBuyPoint"]
        )
        
        return result
    
    def _calculate_volume_rise_buy_point(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算放量上涨买点
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        放量上涨买点 = 价格上涨 + 成交量放大 + 上涨幅度适中
        """
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        buy_signal = np.zeros(n, dtype=bool)
        
        # 计算放量上涨买点
        for i in range(5, n):
            # 条件1：价格上涨
            price_up = close[i] > close[i-1]
            
            # 条件2：成交量放大
            volume_up = volume[i] > volume[i-1] * 1.3  # 成交量放大30%以上
            
            # 条件3：5日内涨幅适中（3%-7%）
            five_day_change = (close[i] / close[i-5] - 1) * 100
            moderate_rise = 3 <= five_day_change <= 7
            
            if price_up and volume_up and moderate_rise:
                buy_signal[i] = True
        
        # 添加到结果
        result.loc[:, "VolumeRiseBuyPoint"] = buy_signal
        
        return result
    
    def _calculate_pullback_stabilize_buy_point(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算回调企稳买点
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        回调企稳买点 = 前期上涨 + 小幅回调 + 缩量 + 企稳回升
        """
        # 提取数据
        close = data["close"].values
        low = data["low"].values
        high = data["high"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        buy_signal = np.zeros(n, dtype=bool)
        
        # 计算回调企稳买点
        for i in range(10, n):
            # 条件1：前期上涨（前10日累计涨幅>8%）
            prev_rise = (close[i-3] / close[i-10] - 1) * 100 > 8
            
            # 条件2：近期回调（最近3日内最低点比前期高点回调3%-8%）
            recent_high = max(high[i-10:i-3])
            recent_low = min(low[i-3:i+1])
            pullback_pct = (recent_high - recent_low) / recent_high * 100
            moderate_pullback = 3 <= pullback_pct <= 8
            
            # 条件3：回调缩量（回调日成交量低于前期平均量）
            pullback_volume = np.mean(volume[i-3:i+1])
            prev_avg_volume = np.mean(volume[i-10:i-3])
            volume_shrink = pullback_volume < prev_avg_volume
            
            # 条件4：企稳回升（当日收盘价高于前一日，且高于3日低点5%以内）
            stabilize = close[i] > close[i-1] and (close[i] - recent_low) / recent_low < 0.05
            
            if prev_rise and moderate_pullback and volume_shrink and stabilize:
                buy_signal[i] = True
        
        # 添加到结果
        result.loc[:, "PullbackStabilizeBuyPoint"] = buy_signal
        
        return result
    
    def _calculate_breakout_buy_point(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算突破买点
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        突破买点 = 突破前期高点 + 放量 + 前期整理时间充分
        """
        # 提取数据
        close = data["close"].values
        high = data["high"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        buy_signal = np.zeros(n, dtype=bool)
        
        # 计算突破买点
        for i in range(20, n):
            # 条件1：突破前期高点（突破20日内最高点）
            prev_high = max(high[i-20:i-1])
            breakout = close[i] > prev_high
            
            # 条件2：放量（成交量大于前20日平均量的1.5倍）
            avg_volume = np.mean(volume[i-20:i])
            volume_surge = volume[i] > avg_volume * 1.5
            
            # 条件3：前期整理充分（前期10日振幅小于7%）
            prev_high_10d = max(high[i-10:i-1])
            prev_low_10d = min(data["low"].values[i-10:i-1])
            range_pct = (prev_high_10d - prev_low_10d) / prev_low_10d * 100
            sufficient_consolidation = range_pct < 7
            
            if breakout and volume_surge and sufficient_consolidation:
                buy_signal[i] = True
        
        # 添加到结果
        result.loc[:, "BreakoutBuyPoint"] = buy_signal
        
        return result
    
    def _calculate_bottom_volume_buy_point(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算底部放量买点
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        底部放量买点 = 前期下跌 + 底部横盘 + 突然放量 + 价格站上短期均线
        """
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 计算5日均线
        ma5 = np.zeros(len(close))
        for i in range(5, len(close)):
            ma5[i] = np.mean(close[i-5:i])
        
        # 初始化结果数组
        n = len(data)
        buy_signal = np.zeros(n, dtype=bool)
        
        # 计算底部放量买点
        for i in range(20, n):
            # 条件1：前期下跌（20日内下跌超过12%）
            max_close = max(close[i-20:i-10])
            min_close = min(close[i-10:i])
            decline_pct = (max_close - min_close) / max_close * 100
            previous_decline = decline_pct > 12
            
            # 条件2：底部横盘（近5日振幅小于5%）
            recent_range = (max(close[i-5:i]) - min(close[i-5:i])) / min(close[i-5:i]) * 100
            bottom_consolidation = recent_range < 5
            
            # 条件3：突然放量（当日成交量是前5日平均量的2倍以上）
            avg_volume_5d = np.mean(volume[i-5:i])
            sudden_volume_surge = volume[i] > avg_volume_5d * 2
            
            # 条件4：价格站上短期均线
            above_ma5 = close[i] > ma5[i]
            
            if previous_decline and bottom_consolidation and sudden_volume_surge and above_ma5:
                buy_signal[i] = True
        
        # 添加到结果
        result.loc[:, "BottomVolumeBuyPoint"] = buy_signal
        
        return result
    
    def _calculate_volume_shrink_buy_point(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算缩量整理买点
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        缩量整理买点 = 前期上涨 + 横盘整理 + 量能萎缩 + 再次放量上涨
        """
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        buy_signal = np.zeros(n, dtype=bool)
        
        # 计算缩量整理买点
        for i in range(15, n):
            # 条件1：前期上涨（前10-15日累计涨幅>10%）
            prev_rise = (close[i-5] / close[i-15] - 1) * 100 > 10
            
            # 条件2：近期横盘整理（最近5日振幅<6%）
            recent_range = (max(close[i-5:i]) - min(close[i-5:i])) / min(close[i-5:i]) * 100
            consolidation = recent_range < 6
            
            # 条件3：量能萎缩（近5日平均量低于前10日平均量的70%）
            recent_avg_volume = np.mean(volume[i-5:i])
            prev_avg_volume = np.mean(volume[i-15:i-5])
            volume_shrink = recent_avg_volume < prev_avg_volume * 0.7
            
            # 条件4：再次放量上涨（当日量能是近5日平均量的1.5倍以上且价格上涨）
            volume_expand = volume[i] > recent_avg_volume * 1.5
            price_up = close[i] > close[i-1]
            
            if prev_rise and consolidation and volume_shrink and volume_expand and price_up:
                buy_signal[i] = True
        
        # 添加到结果
        result.loc[:, "VolumeShrinkBuyPoint"] = buy_signal
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM买点指标的原始评分
        
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
        
        # 1. 各种买点基础得分
        # 放量上涨买点：中等强度买点
        score[result["VolumeRiseBuyPoint"]] += 25
        
        # 回调企稳买点：较强买点
        score[result["PullbackStabilizeBuyPoint"]] += 30
        
        # 突破买点：强买点
        score[result["BreakoutBuyPoint"]] += 35
        
        # 底部放量买点：强买点
        score[result["BottomVolumeBuyPoint"]] += 35
        
        # 缩量整理买点：中等强度买点
        score[result["VolumeShrinkBuyPoint"]] += 25
        
        # 2. 买点组合加分 - 同时满足多个买点形态时加分
        # 计算每天满足的买点数量
        buy_point_count = pd.Series(0, index=data.index)
        buy_point_columns = [col for col in result.columns if col.endswith("BuyPoint")]
        
        for i in range(len(data)):
            buy_point_count.iloc[i] = sum(result.iloc[i][col] for col in buy_point_columns)
        
        # 根据买点数量加分
        score[buy_point_count == 2] += 10  # 满足2种买点
        score[buy_point_count >= 3] += 15  # 满足3种及以上买点
        
        # 3. 技术形态加分 - 结合价格形态和均线系统
        
        # 计算均线系统
        ma5 = data["close"].rolling(window=5).mean()
        ma10 = data["close"].rolling(window=10).mean()
        ma20 = data["close"].rolling(window=20).mean()
        
        # 均线多头排列加分
        bullish_ma = (ma5 > ma10) & (ma10 > ma20)
        score[bullish_ma & result["CombinedBuyPoint"]] += 10
        
        # 价格站上所有均线加分
        price_above_all_ma = (data["close"] > ma5) & (data["close"] > ma10) & (data["close"] > ma20)
        score[price_above_all_ma & result["CombinedBuyPoint"]] += 5
        
        # 4. 连续性加分 - 如果近期已经出现过买点，当前买点可能更可靠
        recent_buy_points = pd.Series(0, index=data.index)
        for i in range(5, len(data)):
            recent_buy_points.iloc[i] = result["CombinedBuyPoint"].iloc[i-5:i].sum()
        
        # 近5日内有1个以上买点，当前买点评分加分
        score[(recent_buy_points >= 1) & result["CombinedBuyPoint"]] += 5
        
        # 5. 成交量配合加分
        # 计算成交量比率（当日成交量/20日平均量）
        volume_ratio = data["volume"] / data["volume"].rolling(window=20).mean()
        
        # 量能强劲配合买点加分
        score[(volume_ratio > 2) & result["CombinedBuyPoint"]] += 10  # 成交量是平均量2倍以上
        score[(volume_ratio > 1.5) & (volume_ratio <= 2) & result["CombinedBuyPoint"]] += 5  # 成交量是平均量1.5-2倍
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM买点指标相关的技术形态
        
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
            
            # 检查各类买点
            if last_row["VolumeRiseBuyPoint"]:
                patterns.append("放量上涨买点")
                
            if last_row["PullbackStabilizeBuyPoint"]:
                patterns.append("回调企稳买点")
                
            if last_row["BreakoutBuyPoint"]:
                patterns.append("突破买点")
                
            if last_row["BottomVolumeBuyPoint"]:
                patterns.append("底部放量买点")
                
            if last_row["VolumeShrinkBuyPoint"]:
                patterns.append("缩量整理买点")
            
            # 如果没有识别到任何买点
            if not patterns:
                patterns.append("无买点形态")
                
            # 买点组合形态
            buy_point_count = sum(1 for p in patterns if "买点" in p)
            if buy_point_count >= 3:
                patterns.append("强势多重买点组合")
            elif buy_point_count == 2:
                patterns.append("双重买点组合")
        
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
        signals.loc[:, 'buy_signal'] = result["CombinedBuyPoint"]
        signals.loc[:, 'sell_signal'] = False  # 买点检测器不生成卖出信号
        signals.loc[:, 'neutral_signal'] = ~result["CombinedBuyPoint"]
        
        # 设置趋势
        signals.loc[:, 'trend'] = 0  # 默认中性
        signals.loc[result["CombinedBuyPoint"], 'trend'] = 1  # 买点看涨
        
        # 设置评分
        signals.loc[:, 'score'] = score
        
        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        
        # 为每个买点设置特定的信号类型
        for i in signals.index:
            if result.loc[i, "CombinedBuyPoint"]:
                buy_types = []
                
                if result.loc[i, "VolumeRiseBuyPoint"]:
                    buy_types.append("volume_rise")
                    
                if result.loc[i, "PullbackStabilizeBuyPoint"]:
                    buy_types.append("pullback_stabilize")
                    
                if result.loc[i, "BreakoutBuyPoint"]:
                    buy_types.append("breakout")
                    
                if result.loc[i, "BottomVolumeBuyPoint"]:
                    buy_types.append("bottom_volume")
                    
                if result.loc[i, "VolumeShrinkBuyPoint"]:
                    buy_types.append("volume_shrink")
                
                if buy_types:
                    signals.loc[i, 'signal_type'] = f"buy_point_{'_'.join(buy_types)}"
        
        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        
        # 为每个买点设置详细描述
        for i in signals.index:
            if result.loc[i, "CombinedBuyPoint"]:
                desc_parts = []
                
                if result.loc[i, "VolumeRiseBuyPoint"]:
                    desc_parts.append("放量上涨")
                    
                if result.loc[i, "PullbackStabilizeBuyPoint"]:
                    desc_parts.append("回调企稳")
                    
                if result.loc[i, "BreakoutBuyPoint"]:
                    desc_parts.append("突破前高")
                    
                if result.loc[i, "BottomVolumeBuyPoint"]:
                    desc_parts.append("底部放量")
                    
                if result.loc[i, "VolumeShrinkBuyPoint"]:
                    desc_parts.append("缩量整理后上攻")
                
                if desc_parts:
                    signals.loc[i, 'signal_desc'] = "买点特征：" + "，".join(desc_parts)
        
        # 置信度设置
        signals.loc[:, 'confidence'] = 60  # 基础置信度
        
        # 计算每天满足的买点数量
        buy_point_count = pd.Series(0, index=data.index)
        buy_point_columns = [col for col in result.columns if col.endswith("BuyPoint") and col != "CombinedBuyPoint"]
        
        for i in range(len(data)):
            buy_point_count.iloc[i] = sum(result.iloc[i][col] for col in buy_point_columns)
        
        # 根据买点数量调整置信度
        signals.loc[buy_point_count == 1, 'confidence'] = 70  # 单一买点
        signals.loc[buy_point_count == 2, 'confidence'] = 80  # 双重买点
        signals.loc[buy_point_count >= 3, 'confidence'] = 90  # 三重及以上买点
        
        # 风险等级
        signals.loc[:, 'risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位
        signals.loc[:, 'position_size'] = 0.0
        signals.loc[result["CombinedBuyPoint"], 'position_size'] = 0.3  # 基础仓位
        signals.loc[(buy_point_count == 2), 'position_size'] = 0.5  # 双重买点
        signals.loc[(buy_point_count >= 3), 'position_size'] = 0.7  # 三重及以上买点
        
        # 止损位 - 使用近期低点
        signals.loc[:, 'stop_loss'] = 0.0
        mask = result["CombinedBuyPoint"]
        for i in data.index[mask]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 10:
                    low_price = data.iloc[idx-10:idx+1]['low'].min()
                    signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
            except:
                continue
        
        # 市场环境和成交量确认
        signals.loc[:, 'market_env'] = 'normal'
        
        # 成交量确认 - 当日成交量是否大于20日均量
        # 支持多种成交量列名格式
        volume_columns = ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL']
        volume_col = None

        for col in volume_columns:
            if col in data.columns:
                volume_col = col
                break

        if volume_col is not None:
            volume_ratio = data[volume_col] / data[volume_col].rolling(window=20).mean()
            signals.loc[:, 'volume_confirmation'] = volume_ratio > 1.2
        else:
            # 如果没有成交量数据，设置为False
            signals.loc[:, 'volume_confirmation'] = False
        
        return signals
    def get_pattern_info(self, pattern_id: str) -> dict:
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
            "name": "买点信号检测",
            "description": f"基于ZXM买点检测的技术分析: {pattern_id}",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # ZXMBuyPointScore指标特定的形态信息映射
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
        if "强势多重买点组合" in patterns:
            pattern_boost += 0.25
        elif "双重买点组合" in patterns:
            pattern_boost += 0.15

        # 具体买点形态调整
        if "放量上涨买点" in patterns:
            pattern_boost += 0.15
        if "突破买点" in patterns:
            pattern_boost += 0.15
        if "回调企稳买点" in patterns:
            pattern_boost += 0.12
        if "底部放量买点" in patterns:
            pattern_boost += 0.12
        if "缩量整理买点" in patterns:
            pattern_boost += 0.1

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence