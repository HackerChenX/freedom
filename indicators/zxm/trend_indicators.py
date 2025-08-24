"""
ZXM体系趋势识别指标模块

实现ZXM体系的7个趋势识别指标
"""

from utils.dependency_injection import get_logger

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from scipy.stats import linregress

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger
from indicators.score_manager import IndicatorScoreManager

logger = get_logger(__name__)


class ZxmdailyTrendUp(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ZXM趋势-日线上移指标
    
    判断60日或120日均线是否向上移动
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-日线上移指标"""
        super().__init__(name="ZXMDailyTrendUp", description="ZXM趋势-日线上移指标，判断日线均线是否向上")
    
    def _calculate_trendindicators(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-日线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        j1:MA(C,60)>=REF(MA(C,60),1);
        j2:MA(C,120)>=REF(MA(C,120),1);
        xg:j1 OR j2;
        """
        # 确保数据包含必需的列
        if 'close' not in data.columns:
            raise ValueError("数据缺少必需的'close'列")
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算60日均线和120日均线
        ma60 = data["close"].rolling(window=60).mean()
        ma120 = data["close"].rolling(window=120).mean()
        
        # 计算均线是否上移
        j1 = ma60 >= ma60.shift(1)
        j2 = ma120 >= ma120.shift(1)
        
        # 计算趋势信号
        xg = j1 | j2
        
        # 添加计算结果到数据框
        result.loc[:, "MA60"] = ma60
        result.loc[:, "MA120"] = ma120
        result.loc[:, "J1"] = j1
        result.loc[:, "J2"] = j2
        result.loc[:, "XG"] = xg
        
        
        # 添加形态识别和信号生成
        result = self.add_pattern_detection(result)
        result = self.add_signal_generation(result)

        # 重写buy_signal逻辑，基于XG值而不是通用逻辑
        result.loc[:, 'buy_signal'] = result["XG"] == True
        result.loc[:, 'sell_signal'] = result["XG"] == False
        result.loc[:, 'hold_signal'] = result["XG"] == False

        return result
    
    def calculate_raw_score_Indicators_trendindicators(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算日线上移指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        scores = pd.Series(50.0, index=data.index, dtype=float)
        
        close = data["close"]
        ma60 = result["MA60"]
        ma120 = result["MA120"]
        
        # 数据长度检查
        data_length = len(data)
        
        for i in range(len(data)):
            score = 50.0  # 基础分数
            
            # 跳过数据不足的开始部分
            if i < 5:
                scores.iloc[i] = score
                continue
            
            current_close = close.iloc[i]
            
            # 1. 基于价格趋势的评分（当MA数据不足时使用）
            if i >= 5:
                # 短期价格趋势
                price_trend_5 = (current_close - close.iloc[i-5]) / close.iloc[i-5] if close.iloc[i-5] > 0 else 0
                if price_trend_5 > 0.05:  # 5日涨幅>5%
                    score += 15
                elif price_trend_5 > 0.02:  # 5日涨幅>2%
                    score += 10
                elif price_trend_5 > 0:     # 5日涨幅>0%
                    score += 5
                elif price_trend_5 < -0.05: # 5日跌幅>5%
                    score -= 15
                elif price_trend_5 < -0.02: # 5日跌幅>2%
                    score -= 10
                elif price_trend_5 < 0:     # 5日跌幅>0%
                    score -= 5
            
            # 2. 基于10日趋势的评分
            if i >= 10:
                price_trend_10 = (current_close - close.iloc[i-10]) / close.iloc[i-10] if close.iloc[i-10] > 0 else 0
                if price_trend_10 > 0.10:   # 10日涨幅>10%
                    score += 20
                elif price_trend_10 > 0.05: # 10日涨幅>5%
                    score += 15
                elif price_trend_10 > 0:    # 10日涨幅>0%
                    score += 8
                elif price_trend_10 < -0.10: # 10日跌幅>10%
                    score -= 20
                elif price_trend_10 < -0.05: # 10日跌幅>5%
                    score -= 15
                elif price_trend_10 < 0:     # 10日跌幅>0%
                    score -= 8
            
            # 3. 基于20日趋势的评分
            if i >= 20:
                price_trend_20 = (current_close - close.iloc[i-20]) / close.iloc[i-20] if close.iloc[i-20] > 0 else 0
                if price_trend_20 > 0.20:   # 20日涨幅>20%
                    score += 25
                elif price_trend_20 > 0.10: # 20日涨幅>10%
                    score += 20
                elif price_trend_20 > 0:    # 20日涨幅>0%
                    score += 10
                elif price_trend_20 < -0.20: # 20日跌幅>20%
                    score -= 25
                elif price_trend_20 < -0.10: # 20日跌幅>10%
                    score -= 20
                elif price_trend_20 < 0:     # 20日跌幅>0%
                    score -= 10
            
            # 4. 如果有MA60数据，使用MA60评分
            if i >= 60 and not pd.isna(ma60.iloc[i]):
                current_ma60 = ma60.iloc[i]
                
                # 价格相对MA60位置
                if current_close > current_ma60 * 1.05:  # 价格在MA60上方5%以上
                    score += 15
                elif current_close > current_ma60:       # 价格在MA60上方
                    score += 10
                elif current_close > current_ma60 * 0.98: # 价格接近MA60（2%以内）
                    score += 5
                elif current_close < current_ma60 * 0.95: # 价格在MA60下方5%以上
                    score -= 10
                elif current_close < current_ma60:        # 价格在MA60下方
                    score -= 5
                
                # MA60趋势
                if i >= 65:
                    ma60_trend = (current_ma60 - ma60.iloc[i-5]) / ma60.iloc[i-5] if ma60.iloc[i-5] > 0 else 0
                    if ma60_trend > 0.02:    # MA60周涨幅>2%
                        score += 10
                    elif ma60_trend > 0:     # MA60上升
                        score += 5
                    elif ma60_trend < -0.02: # MA60周跌幅>2%
                        score -= 10
                    elif ma60_trend < 0:     # MA60下降
                        score -= 5
            
            # 5. 如果有MA120数据，使用MA120评分
            if i >= 120 and not pd.isna(ma120.iloc[i]):
                current_ma120 = ma120.iloc[i]
                
                # 价格相对MA120位置
                if current_close > current_ma120 * 1.10:  # 价格在MA120上方10%以上
                    score += 20
                elif current_close > current_ma120:       # 价格在MA120上方
                    score += 15
                elif current_close > current_ma120 * 0.95: # 价格接近MA120（5%以内）
                    score += 8
                elif current_close < current_ma120 * 0.90: # 价格在MA120下方10%以上
                    score -= 15
                elif current_close < current_ma120:        # 价格在MA120下方
                    score -= 10
                
                # MA120趋势
                if i >= 125:
                    ma120_trend = (current_ma120 - ma120.iloc[i-5]) / ma120.iloc[i-5] if ma120.iloc[i-5] > 0 else 0
                    if ma120_trend > 0.01:    # MA120周涨幅>1%
                        score += 15
                    elif ma120_trend > 0:     # MA120上升
                        score += 8
                    elif ma120_trend < -0.01: # MA120周跌幅>1%
                        score -= 15
                    elif ma120_trend < 0:     # MA120下降
                        score -= 8
            
            # 6. 波动率评分
            if i >= 10:
                # 计算10日波动率
                returns = close.pct_change()
                volatility = returns.iloc[i-10:i+1].std()
                
                # 低波动率加分，高波动率减分
                if volatility < 0.02:      # 日波动率<2%
                    score += 5
                elif volatility < 0.03:    # 日波动率<3%
                    score += 3
                elif volatility > 0.08:    # 日波动率>8%
                    score -= 8
                elif volatility > 0.05:    # 日波动率>5%
                    score -= 5
            
            # 7. 连续性评分
            if i >= 5:
                # 连续上涨天数
                consecutive_up = 0
                for j in range(min(5, i)):
                    if close.iloc[i-j] > close.iloc[i-j-1]:
                        consecutive_up += 1
                    else:
                        break
                
                if consecutive_up >= 3:
                    score += consecutive_up * 2  # 连续上涨加分
                
                # 连续下跌天数
                consecutive_down = 0
                for j in range(min(5, i)):
                    if close.iloc[i-j] < close.iloc[i-j-1]:
                        consecutive_down += 1
                    else:
                        break
                
                if consecutive_down >= 3:
                    score -= consecutive_down * 2  # 连续下跌减分
            
            # 确保分数在0-100范围内
            score = max(0, min(100, score))
            scores.iloc[i] = score
        
        return scores
    
    def identify_patterns_Indicators_trendindicators(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别日线上移指标相关的技术形态
        
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
                if last_row["J1"] and last_row["J2"]:
                    patterns.append("双均线上移")
                elif last_row["J1"]:
                    patterns.append("60日均线上移")
                elif last_row["J2"]:
                    patterns.append("120日均线上移")
            else:
                patterns.append("均线趋势走平或下移")
            
            # 均线位置关系
            if data["close"].iloc[-1] > last_row["MA60"] > last_row["MA120"]:
                patterns.append("价格站上双均线，多头排列")
            elif last_row["MA60"] > last_row["MA120"] and data["close"].iloc[-1] < last_row["MA60"]:
                patterns.append("均线多头排列，价格回踩60日线")
            elif last_row["MA60"] < last_row["MA120"] and data["close"].iloc[-1] < last_row["MA60"]:
                patterns.append("均线空头排列，价格在均线下方")
            
            # 近期趋势变化
            if len(result) >= 10:
                if not result["XG"].iloc[-10:-5].any() and result["XG"].iloc[-5:].all():
                    patterns.append("趋势由弱转强")
                elif result["XG"].iloc[-10:-5].all() and not result["XG"].iloc[-5:].any():
                    patterns.append("趋势由强转弱")
        
        return patterns
    
    def generate_signals_Indicators_trendindicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        score = self.calculate_raw_score_Indicators_trendindicators(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals.loc[:, 'buy_signal'] = result["XG"]
        signals.loc[:, 'sell_signal'] = ~result["XG"]
        signals.loc[:, 'neutral_signal'] = False
        
        # 设置趋势
        signals.loc[:, 'trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals.loc[:, 'score'] = score
        
        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        signals.loc[result["XG"], 'signal_type'] = 'daily_ma_up'
        signals.loc[~result["XG"], 'signal_type'] = 'daily_ma_down'
        
        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        signals.loc[result["XG"] & result["J1"] & result["J2"], 'signal_desc'] = '双均线上移，趋势强劲'
        signals.loc[result["XG"] & result["J1"] & ~result["J2"], 'signal_desc'] = '60日均线上移，中期趋势向好'
        signals.loc[result["XG"] & ~result["J1"] & result["J2"], 'signal_desc'] = '120日均线上移，长期趋势向好'
        signals.loc[~result["XG"], 'signal_desc'] = '均线不上移，趋势走弱'
        
        # 置信度设置
        signals.loc[:, 'confidence'] = 60  # 基础置信度
        # 双均线上移，置信度更高
        signals.loc[result["XG"] & result["J1"] & result["J2"], 'confidence'] = 80
        # 评分高的信号，置信度更高
        signals.loc[score > 70, 'confidence'] = 75
        signals.loc[score > 85, 'confidence'] = 90
        
        # 风险等级
        signals.loc[:, 'risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位
        signals.loc[:, 'position_size'] = 0.0
        signals.loc[result["XG"], 'position_size'] = 0.3  # 基础仓位
        signals.loc[(result["XG"]) & (score > 70), 'position_size'] = 0.5  # 高分仓位
        signals.loc[(result["XG"]) & (score > 85), 'position_size'] = 0.7  # 极高分仓位
        
        # 止损位 - 使用均线作为参考
        signals.loc[:, 'stop_loss'] = 0.0
        mask = result["XG"]
        for i in data.index[mask]:
            ma60_val = result.loc[i, "MA60"]
            close_val = data.loc[i, "close"]
            # 如果价格在均线上方，则以均线为止损位
            if close_val > ma60_val:
                signals.loc[i, 'stop_loss'] = ma60_val * 0.98  # 均线下方2%
            else:
                # 否则使用近期低点
                try:
                    idx = data.index.get_loc(i)
                    if idx >= 10:
                        low_price = data.iloc[idx-10:idx+1]['low'].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
                except:
                    continue
        
        # 市场环境和成交量确认
        signals.loc[:, 'market_env'] = 'normal'
        signals.loc[:, 'volume_confirmation'] = False
        
        return signals

    def calculate_confidence_Indicators_trendindicators(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
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
        if "双均线上移" in patterns:
            pattern_boost += 0.15
        elif "60日均线上移" in patterns or "120日均线上移" in patterns:
            pattern_boost += 0.1

        if "价格站上双均线，多头排列" in patterns:
            pattern_boost += 0.1
        elif "趋势由弱转强" in patterns:
            pattern_boost += 0.15

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence

    def get_patterns_Indicators_trendindicators(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        patterns_df.loc[:, "均线上移"] = result["XG"]
        patterns_df.loc[:, "双均线上移"] = result["J1"] & result["J2"]
        patterns_df.loc[:, "60日均线上移"] = result["J1"] & ~result["J2"]
        patterns_df.loc[:, "120日均线上移"] = ~result["J1"] & result["J2"]

        # 价格与均线关系
        price_above_ma60 = data["close"] > result["MA60"]
        price_above_ma120 = data["close"] > result["MA120"]
        ma_bull_alignment = result["MA60"] > result["MA120"]

        patterns_df.loc[:, "价格站上双均线"] = price_above_ma60 & price_above_ma120 & ma_bull_alignment
        patterns_df.loc[:, "价格回踩60日线"] = ma_bull_alignment & ~price_above_ma60 & (data["close"] > result["MA120"])
        patterns_df.loc[:, "均线空头排列"] = result["MA60"] < result["MA120"]

        # 趋势变化
        if len(result) >= 10:
            trend_weak_to_strong = pd.Series(False, index=data.index)
            trend_strong_to_weak = pd.Series(False, index=data.index)

            for i in range(10, len(result)):
                if not result["XG"].iloc[i-10:i-5].any() and result["XG"].iloc[i-5:i+1].all():
                    trend_weak_to_strong.iloc[i] = True
                elif result["XG"].iloc[i-10:i-5].all() and not result["XG"].iloc[i-5:i+1].any():
                    trend_strong_to_weak.iloc[i] = True

            patterns_df.loc[:, "趋势由弱转强"] = trend_weak_to_strong
            patterns_df.loc[:, "趋势由强转弱"] = trend_strong_to_weak

        return patterns_df

    def register_patterns_Indicators_Trend_Indicators_Trend_Indicators_trendindicators(self):
        """
        注册ZXMDaily_trend_up指标的形态到全局形态注册表
        """
        # 注册日线均线上移形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_TREND_UP",
            display_name="ZXM日线趋势向上",
            description="60日或120日均线向上移动，表明中长期趋势向好",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        # 注册双均线上移形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_DOUBLE_MA_UP",
            display_name="ZXM日线双均线上移",
            description="60日和120日均线同时向上移动，趋势更强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        # 注册单均线上移形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_DAILY_SINGLE_MA_UP",
            display_name="ZXM日线单均线上移",
            description="仅有一条均线向上移动，趋势较弱",
            pattern_type="BULLISH",
            default_strength="WEAK",
            score_impact=10.0,
            polarity="POSITIVE"
        )

    def set_parameters_Indicators_trendindicators(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 日线上移指标没有可调参数，保持默认实现
        pass
    
    @property
    def minimum_periods(self) -> int:
        """
        ZxmdailyTrendUp指标所需的最少数据周期数
        
        计算逻辑：使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30

class ZxmweeklyTrendUp(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势-周线上移指标

    判断周线10周、20周或30周均线是否向上移动
    """
    
    def calculate_raw_score_Indicators_trendindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算周线上移指标的原始评分
        
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
        # 1. 均线上移信号(XG)：周线级别上看涨，+30分
        score[result["XG"]] += 30
        
        # 2. 不同周期均线上移组合评分
        # 三均线同时上移是最强信号
        all_up = result["A1"] & result["B1"] & result["C1"]
        score[all_up] += 15
        
        # 只有一条均线上移是较弱信号，但不同均线有不同权重
        only_ma10_up = result["A1"] & ~result["B1"] & ~result["C1"]
        only_ma20_up = ~result["A1"] & result["B1"] & ~result["C1"]
        only_ma30_up = ~result["A1"] & ~result["B1"] & result["C1"]
        
        score[only_ma10_up] += 5   # 短期均线上移
        score[only_ma20_up] += 10  # 中期均线上移
        score[only_ma30_up] += 15  # 长期均线上移
        
        # 3. 价格与均线位置关系评分
        price_above_ma10 = data["close"] > result["MA10"]
        price_above_ma20 = data["close"] > result["MA20"]
        price_above_ma30 = data["close"] > result["MA30"]
        
        # 价格在均线上方，趋势更强
        score[price_above_ma10] += 5
        score[price_above_ma20] += 8
        score[price_above_ma30] += 12
        
        # 4. 均线多头排列评分
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        ma_bear_alignment = (result["MA10"] < result["MA20"]) & (result["MA20"] < result["MA30"])
        
        score[ma_bull_alignment] += 15  # 多头排列加分
        score[ma_bear_alignment] -= 15  # 空头排列减分
        
        # 5. 连续上移评分
        # 连续多周均线上移，表示趋势较强
        continuous_up_weeks = pd.Series(0, index=data.index)
        for i in range(4, len(data)):
            if all(result["XG"].iloc[i-4:i+1]):
                continuous_up_weeks.iloc[i] = 4
            elif all(result["XG"].iloc[i-2:i+1]):
                continuous_up_weeks.iloc[i] = 2
        
        # 根据连续上移周数加分
        score[continuous_up_weeks == 2] += 5
        score[continuous_up_weeks == 4] += 15
        
        # 6. 均线角度评分
        # 计算均线斜率
        ma10_slope = result["MA10"].diff(4) / result["MA10"].shift(4)
        ma20_slope = result["MA20"].diff(4) / result["MA20"].shift(4)
        ma30_slope = result["MA30"].diff(4) / result["MA30"].shift(4)
        
        # 根据斜率大小加分，斜率越大上升越快
        score[ma10_slope > 0.03] += 5  # 月涨幅>3%
        score[ma10_slope > 0.06] += 5  # 月涨幅>6%
        
        score[ma20_slope > 0.02] += 5  # 月涨幅>2%
        score[ma20_slope > 0.04] += 5  # 月涨幅>4%
        
        score[ma30_slope > 0.01] += 5  # 月涨幅>1%
        score[ma30_slope > 0.02] += 5  # 月涨幅>2%
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns_Indicators_trendindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别周线上移指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易周的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["XG"]:
                if last_row["A1"] and last_row["B1"] and last_row["C1"]:
                    patterns.append("三均线同时上移")
                elif last_row["A1"] and last_row["B1"]:
                    patterns.append("10周和20周均线上移")
                elif last_row["A1"] and last_row["C1"]:
                    patterns.append("10周和30周均线上移")
                elif last_row["B1"] and last_row["C1"]:
                    patterns.append("20周和30周均线上移")
                elif last_row["A1"]:
                    patterns.append("10周均线上移")
                elif last_row["B1"]:
                    patterns.append("20周均线上移")
                elif last_row["C1"]:
                    patterns.append("30周均线上移")
            else:
                patterns.append("周均线趋势走平或下移")
            
            # 均线位置关系
            close_value = data["close"].iloc[-1]
            if close_value > last_row["MA10"] > last_row["MA20"] > last_row["MA30"]:
                patterns.append("价格站上三均线，多头排列")
            elif last_row["MA10"] > last_row["MA20"] > last_row["MA30"]:
                if close_value < last_row["MA10"]:
                    patterns.append("均线多头排列，价格回踩10周线")
            elif last_row["MA10"] < last_row["MA20"] < last_row["MA30"]:
                if close_value < last_row["MA10"]:
                    patterns.append("均线空头排列，价格在均线下方")
                else:
                    patterns.append("均线空头排列，价格反弹站上10周线")
            
            # 近期趋势变化
            if len(result) >= 8:
                if not result["XG"].iloc[-8:-4].any() and result["XG"].iloc[-4:].all():
                    patterns.append("周线趋势由弱转强")
                elif result["XG"].iloc[-8:-4].all() and not result["XG"].iloc[-4:].any():
                    patterns.append("周线趋势由强转弱")
        
        return patterns
    
    def generate_signals_Indicators_trendindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        score = self.calculate_raw_score_Indicators_trendindicators(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals.loc[:, 'buy_signal'] = result["XG"]
        signals.loc[:, 'sell_signal'] = ~result["XG"]
        signals.loc[:, 'neutral_signal'] = False
        
        # 设置趋势
        signals.loc[:, 'trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals.loc[:, 'score'] = score
        
        # 设置信号类型
        signals.loc[:, 'signal_type'] = 'neutral'
        signals.loc[result["XG"] & result["A1"] & result["B1"] & result["C1"], 'signal_type'] = 'weekly_three_ma_up'
        signals.loc[result["XG"] & ~(result["A1"] & result["B1"] & result["C1"]), 'signal_type'] = 'weekly_some_ma_up'
        signals.loc[~result["XG"], 'signal_type'] = 'weekly_ma_down'
        
        # 设置信号描述
        signals.loc[:, 'signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "A1"] and result.loc[i, "B1"] and result.loc[i, "C1"]:
                signals.loc[i, 'signal_desc'] = "三周期均线同时上移，趋势强劲"
            elif result.loc[i, "XG"]:
                up_mas = []
                if result.loc[i, "A1"]: up_mas.append("10周")
                if result.loc[i, "B1"]: up_mas.append("20周")
                if result.loc[i, "C1"]: up_mas.append("30周")
                up_mas_str = "、".join(up_mas)
                signals.loc[i, 'signal_desc'] = f"{up_mas_str}均线上移，趋势向好"
            else:
                signals.loc[i, 'signal_desc'] = "周均线不上移，趋势走弱"
        
        # 置信度设置
        signals.loc[:, 'confidence'] = 60  # 基础置信度
        
        # 根据上移均线数量调整置信度
        up_ma_count = result["A1"].astype(int) + result["B1"].astype(int) + result["C1"].astype(int)
        for i in range(len(signals)):
            signals['confidence'].iloc[i] += up_ma_count.iloc[i] * 10
        
        # 均线排列影响置信度
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        signals.loc[ma_bull_alignment, 'confidence'] += 10
        
        # 确保置信度在0-100范围内
        signals.loc[:, 'confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals.loc[:, 'risk_level'] = '中'  # 默认中等风险
        signals.loc[score >= 75, 'risk_level'] = '低'
        signals.loc[score <= 30, 'risk_level'] = '高'
        
        # 建议仓位
        signals.loc[:, 'position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据上移均线数量和排列调整仓位
        three_ma_up = result["A1"] & result["B1"] & result["C1"]
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        
        signals.loc[three_ma_up, 'position_size'] = 0.5  # 三均线上移，加大仓位
        signals.loc[three_ma_up & ma_bull_alignment, 'position_size'] = 0.7  # 三均线上移且多头排列，大仓位
        
        # 止损位 - 使用均线作为参考
        signals.loc[:, 'stop_loss'] = 0.0
        for i in signals.index[signals['buy_signal']]:
            ma10_val = result.loc[i, "MA10"]
            ma20_val = result.loc[i, "MA20"]
            close_val = data.loc[i, "close"]
            
            # 如果价格在均线上方，则以均线为止损位
            if close_val > ma10_val:
                signals.loc[i, 'stop_loss'] = ma10_val * 0.95  # 10周均线下方5%
            elif close_val > ma20_val:
                signals.loc[i, 'stop_loss'] = ma20_val * 0.95  # 20周均线下方5%
            else:
                # 否则使用近期低点
                try:
                    idx = data.index.get_loc(i)
                    if idx >= 8:
                        low_price = data.iloc[idx-8:idx+1]['low'].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.95  # 最低点下方5%
                except:
                    continue
        
        # 市场环境
        signals.loc[:, 'market_env'] = 'normal'
        bull_market = result["XG"] & ma_bull_alignment
        bear_market = ~result["XG"] & (result["MA10"] < result["MA20"]) & (result["MA20"] < result["MA30"])
        
        signals.loc[bull_market, 'market_env'] = 'bull_market'
        signals.loc[bear_market, 'market_env'] = 'bear_market'
        signals.loc[~bull_market & ~bear_market, 'market_env'] = 'sideways_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals.loc[:, 'volume_confirmation'] = True
        
        return signals

    def calculate_confidence_Indicators_trendindicators_duplicate(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
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
        if "三均线同时上移" in patterns:
            pattern_boost += 0.2
        elif any(p in patterns for p in ["10周和20周均线上移", "20周和30周均线上移"]):
            pattern_boost += 0.15
        elif any(p in patterns for p in ["10周均线上移", "20周均线上移", "30周均线上移"]):
            pattern_boost += 0.1

        if "价格站上三均线，多头排列" in patterns:
            pattern_boost += 0.15
        elif "周线趋势由弱转强" in patterns:
            pattern_boost += 0.15

        # 最终置信度
        final_confidence = min(1.0, base_confidence + pattern_boost)
        return final_confidence

    def get_patterns_Indicators_trendindicators_duplicate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
        patterns_df.loc[:, "周均线上移"] = result["XG"]
        patterns_df.loc[:, "三均线同时上移"] = result["A1"] & result["B1"] & result["C1"]
        patterns_df.loc[:, "10周均线上移"] = result["A1"] & ~result["B1"] & ~result["C1"]
        patterns_df.loc[:, "20周均线上移"] = ~result["A1"] & result["B1"] & ~result["C1"]
        patterns_df.loc[:, "30周均线上移"] = ~result["A1"] & ~result["B1"] & result["C1"]
        patterns_df.loc[:, "10周和20周均线上移"] = result["A1"] & result["B1"] & ~result["C1"]
        patterns_df.loc[:, "10周和30周均线上移"] = result["A1"] & ~result["B1"] & result["C1"]
        patterns_df.loc[:, "20周和30周均线上移"] = ~result["A1"] & result["B1"] & result["C1"]

        # 价格与均线关系
        price_above_ma10 = data["close"] > result["MA10"]
        price_above_ma20 = data["close"] > result["MA20"]
        price_above_ma30 = data["close"] > result["MA30"]
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        ma_bear_alignment = (result["MA10"] < result["MA20"]) & (result["MA20"] < result["MA30"])

        patterns_df.loc[:, "价格站上三均线"] = price_above_ma10 & price_above_ma20 & price_above_ma30 & ma_bull_alignment
        patterns_df.loc[:, "均线多头排列"] = ma_bull_alignment
        patterns_df.loc[:, "均线空头排列"] = ma_bear_alignment
        patterns_df.loc[:, "价格回踩10周线"] = ma_bull_alignment & ~price_above_ma10 & price_above_ma20
        patterns_df.loc[:, "价格反弹站上10周线"] = ma_bear_alignment & price_above_ma10

        # 趋势变化
        if len(result) >= 8:
            trend_weak_to_strong = pd.Series(False, index=data.index)
            trend_strong_to_weak = pd.Series(False, index=data.index)

            for i in range(8, len(result)):
                if not result["XG"].iloc[i-8:i-4].any() and result["XG"].iloc[i-4:i+1].all():
                    trend_weak_to_strong.iloc[i] = True
                elif result["XG"].iloc[i-8:i-4].all() and not result["XG"].iloc[i-4:i+1].any():
                    trend_strong_to_weak.iloc[i] = True

            patterns_df.loc[:, "周线趋势由弱转强"] = trend_weak_to_strong
            patterns_df.loc[:, "周线趋势由强转弱"] = trend_strong_to_weak

        return patterns_df

    def register_patterns_Indicators_Trend_Indicators_Trend_Indicators_trendindicators_duplicate(self):
        """
        注册ZXMWeekly_trend_up指标的形态到全局形态注册表
        """
        # 注册周线均线上移形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_WEEKLY_TREND_UP",
            display_name="ZXM周线趋势向上",
            description="10周、20周或30周均线向上移动，表明中期趋势向好",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        # 注册三均线同时上移形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_WEEKLY_THREE_MA_UP",
            display_name="ZXM周线三均线上移",
            description="10周、20周、30周均线同时向上移动，趋势强劲",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0,
            polarity="POSITIVE"
        )

        # 注册均线多头排列形态
        self.register_pattern_to_registry(
            pattern_id="ZXM_WEEKLY_BULLISH_ALIGNMENT",
            display_name="ZXM周线多头排列",
            description="周线均线呈多头排列，价格站上均线",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

    def set_parameters_Indicators_trendindicators_duplicate(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 周线上移指标没有可调参数，保持默认实现
        pass


class ZxmmonthlyKdjtrendUp(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势-月KDJ·D及K上移指标

    判断月线KDJ指标的D值和K值是否同时向上移动
    """
    
    def _sma_Trend_Indicators_Trend_Indicators_trendindicators(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """计算移动平均线，类似于通达信中的SMA函数"""
        result = pd.Series(index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result
    
class ZxmweeklyKdjdorDeatrendUp(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势-周KDJ·D/DEA上移指标

    判断周线KDJ指标的D值或MACD的DEA值是否有一个向上移动
    """
    
class ZxmweeklyKdjdtrendUp(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势-周KDJ·D上移指标

    判断周线KDJ指标的D值是否向上移动
    """
    
class ZxmmonthlyMacd(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势-月MACD指标

    判断月线MACD金叉
    """
    
class TrendDetector(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势检测器

    识别价格趋势的方向和强度
    """
    
class TrendDuration(BaseIndicator, PatternSignalMixin):
    """
    ZXM趋势持续性指标

    分析价格趋势的持续时间和生命周期特征
    """
    
    def get_pattern_info_Indicators_Trend_Indicators_Trend_Indicators_trendindicators(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        pattern_info_map = {
            "超长期趋势": {
                "id": "超长期趋势",
                "name": "超长期趋势",
                "description": "趋势持续时间超过60个周期，表明强势趋势",
                "type": "BULLISH",
                "strength": "VERY_STRONG",
                "score_impact": 25.0
            },
            "长期趋势": {
                "id": "长期趋势",
                "name": "长期趋势",
                "description": "趋势持续时间30-60个周期，表明稳定趋势",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "中期趋势": {
                "id": "中期趋势",
                "name": "中期趋势",
                "description": "趋势持续时间10-30个周期，表明中等趋势",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            },
            "短期趋势": {
                "id": "短期趋势",
                "name": "短期趋势",
                "description": "趋势持续时间少于10个周期，表明短期趋势",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 10.0
            },
            "趋势转换": {
                "id": "趋势转换",
                "name": "趋势转换",
                "description": "趋势方向发生转换，需要关注",
                "type": "NEUTRAL",
                "strength": "MEDIUM",
                "score_impact": 0.0
            },
            "趋势加速": {
                "id": "趋势加速",
                "name": "趋势加速",
                "description": "趋势强度增强，动能增加",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 18.0
            },
            "趋势减速": {
                "id": "趋势减速",
                "name": "趋势减速",
                "description": "趋势强度减弱，动能减少",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -12.0
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "趋势生命周期分析",
            "description": f"基于趋势生命周期的持续性分析: {pattern_id}",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })

class ZxmweeklyMacd(BaseIndicator, PatternSignalMixin):
    """
    ZXM系统周线MACD指标
    用于检测中期趋势的变化和买卖信号
    特点是对中期趋势敏感，可靠性高于日线MACD
    """
    
    def _calculate_divergence_Trend_Indicators(self, data: pd.DataFrame) -> None:
        """
        计算MACD与价格的背离情况
        
        Args:
            data: Data_frame，包含MACD指标和价格数据
        """
        # 初始化背离列
        data['bullish_divergence'] = False
        data['bearish_divergence'] = False
        
        # 寻找价格和DIF的低点和高点
        window = 8  # 用于寻找局部极值的窗口大小
        
        # 确保数据量足够
        if len(data) < window * 2:
            return
        
        # 寻找局部低点（价格创新低但DIF不创新低）
        for i in range(window, len(data) - window):
            # 检查价格是否为局部低点
            if (data['close'].iloc[i] < data['close'].iloc[i-window:i].min()) and \
               (data['close'].iloc[i] < data['close'].iloc[i+1:i+window+1].min()):
                
                # 找前一个低点
                prev_low_idx = None
                for j in range(i-window, i):
                    if (data['close'].iloc[j] < data['close'].iloc[max(0, j-window):j].min()) and \
                       (data['close'].iloc[j] < data['close'].iloc[j+1:j+window+1].min()):
                        prev_low_idx = j
                        break
                
                if prev_low_idx is not None and data['close'].iloc[i] < data['close'].iloc[prev_low_idx]:
                    # 价格创新低
                    if data['DIF'].iloc[i] > data['DIF'].iloc[prev_low_idx]:
                        # DIF不创新低，形成底背离
                        data.loc[data.index[i], 'bullish_divergence'] = True
        
        # 寻找局部高点（价格创新高但DIF不创新高）
        for i in range(window, len(data) - window):
            # 检查价格是否为局部高点
            if (data['close'].iloc[i] > data['close'].iloc[i-window:i].max()) and \
               (data['close'].iloc[i] > data['close'].iloc[i+1:i+window+1].max()):
                
                # 找前一个高点
                prev_high_idx = None
                for j in range(i-window, i):
                    if (data['close'].iloc[j] > data['close'].iloc[max(0, j-window):j].max()) and \
                       (data['close'].iloc[j] > data['close'].iloc[j+1:j+window+1].max()):
                        prev_high_idx = j
                        break
                
                if prev_high_idx is not None and data['close'].iloc[i] > data['close'].iloc[prev_high_idx]:
                    # 价格创新高
                    if data['DIF'].iloc[i] < data['DIF'].iloc[prev_high_idx]:
                        # DIF不创新高，形成顶背离
                        data.loc[data.index[i], 'bearish_divergence'] = True
    
    def _calculate_trend_stability(self, result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算趋势稳定性
        
        Args:
            result: 包含趋势方向的Data_frame
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 添加了趋势稳定性的Data_frame
        """
        try:
            # 初始化趋势稳定性列
            result = result.copy()
            result.loc[:, 'TrendStability'] = 0.0
            
            if 'TrendDirection' not in result.columns:
                logger.warning("没有TrendDirection列，无法计算趋势稳定性")
                return result
                
            # 获取趋势方向列
            trend_direction = result['TrendDirection']
            
            # 计算趋势稳定性
            lookback = 5  # 回看窗口
            
            for i in range(lookback, len(result)):
                # 获取过去n天的趋势方向
                past_trend = trend_direction.values[i-lookback:i+1]
                
                # 计算一致性
                # 如果全部一致，返回1.0；如果完全相反，返回0.0
                if np.all(past_trend == 1) or np.all(past_trend == -1):
                    consistency = 1.0
                else:
                    # 计算上升趋势的比例
                    up_ratio = np.sum(past_trend == 1) / lookback
                    # 计算下降趋势的比例
                    down_ratio = np.sum(past_trend == -1) / lookback
                    # 取主导趋势的比例作为一致性
                    consistency = max(up_ratio, down_ratio)
                
                # 使用loc而不是链式索引，避免SettingWithCopyWarning
                result.loc[result.index[i], 'TrendStability'] = consistency
            
            return result
        except Exception as e:
            logger.error(f"计算趋势稳定性时出错: {e}")
            return result

    def get_pattern_info_Indicators_Trend_Indicators_Trend_Indicators_trendindicators_duplicate(self, pattern_id: str) -> dict:
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

        # ZXMWeeklyMACD指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "周线MACD零轴上方金叉": {
                "id": "周线MACD零轴上方金叉",
                "name": "周线MACD零轴上方金叉",
                "description": "周线MACD在零轴上方形成金叉，强势上涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 25.0
            },
            "周线MACD零轴下方金叉": {
                "id": "周线MACD零轴下方金叉",
                "name": "周线MACD零轴下方金叉",
                "description": "周线MACD在零轴下方形成金叉，反转向上信号",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0
            },
            "周线MACD零轴上方死叉": {
                "id": "周线MACD零轴上方死叉",
                "name": "周线MACD零轴上方死叉",
                "description": "周线MACD在零轴上方形成死叉，反转向下信号",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -15.0
            },
            "周线MACD零轴下方死叉": {
                "id": "周线MACD零轴下方死叉",
                "name": "周线MACD零轴下方死叉",
                "description": "周线MACD在零轴下方形成死叉，强势下跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -25.0
            },
            "周线MACD底背离": {
                "id": "周线MACD底背离",
                "name": "周线MACD底背离",
                "description": "周线MACD与价格形成底背离，强烈反转向上信号",
                "type": "BULLISH",
                "strength": "VERY_STRONG",
                "score_impact": 30.0
            },
            "周线MACD顶背离": {
                "id": "周线MACD顶背离",
                "name": "周线MACD顶背离",
                "description": "周线MACD与价格形成顶背离，强烈反转向下信号",
                "type": "BEARISH",
                "strength": "VERY_STRONG",
                "score_impact": -30.0
            }
        }

        return pattern_info_map.get(pattern_id, default_pattern)