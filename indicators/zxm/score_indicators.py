"""
ZXM体系评分指标模块

实现ZXM体系的评分指标，包括弹性评分和买点评分
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.zxm.elasticity_indicators import AmplitudeElasticity, ZXMRiseElasticity
from indicators.zxm.buy_point_indicators import ZXMDailyMACD, ZXMTurnover, ZXMMACallback
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMElasticityScore(BaseIndicator):
    """
    ZXM弹性评分指标
    
    综合评估股票的弹性表现，包括振幅弹性和涨幅弹性
    """
    
    def __init__(self, threshold: float = 75):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ZXM弹性评分指标
        
        Args:
            threshold: 弹性评分阈值，默认为75分
        """
        super().__init__(name="ZXMElasticityScore", description="ZXM弹性评分指标，综合评估股票的弹性表现")
        self.threshold = threshold
        
        # 初始化弹性子指标
        self.amplitude_elasticity = AmplitudeElasticity()
        self.rise_elasticity = ZXMRiseElasticity()
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM弹性评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含弹性评分和信号
        """
        # 初始化结果数据框
        result = data.copy()
        
        # 计算振幅弹性
        try:
            amplitude_result = self.amplitude_elasticity.calculate(data)
            amplitude_signal = amplitude_result["XG"]
        except Exception as e:
            logger.error(f"计算振幅弹性出错: {e}")
            amplitude_signal = pd.Series(False, index=data.index)
        
        # 计算涨幅弹性
        try:
            rise_result = self.rise_elasticity.calculate(data)
            rise_signal = rise_result["XG"]
        except Exception as e:
            logger.error(f"计算涨幅弹性出错: {e}")
            rise_signal = pd.Series(False, index=data.index)
        
        # 统计弹性指标
        elasticity_indicators = [amplitude_signal, rise_signal]
        
        # 计算弹性指标满足数量
        elasticity_count = pd.Series(0, index=data.index)
        for indicator in elasticity_indicators:
            elasticity_count += indicator.astype(int)
        
        # 计算弹性评分（0-100分）
        elasticity_score = (elasticity_count / len(elasticity_indicators)) * 100
        
        # 生成满足弹性条件的信号
        signal = elasticity_score >= self.threshold
        
        # 添加计算结果到数据框
        result["AmplitudeElasticity"] = amplitude_signal
        result["RiseElasticity"] = rise_signal
        result["ElasticityCount"] = elasticity_count
        result["ElasticityScore"] = elasticity_score
        result["Signal"] = signal
        
        return result



    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM弹性评分指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 直接使用计算的弹性评分作为原始评分
        return result["ElasticityScore"]
class ZXMBuyPointScore(BaseIndicator):
    """
    ZXM买点评分指标
    
    综合评估股票的买点表现，包括MACD买点、换手买点和回踩均线买点
    """
    
    def __init__(self, threshold: float = 75):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化ZXM买点评分指标
        
        Args:
            threshold: 买点评分阈值，默认为75分
        """
        super().__init__(name="ZXMBuyPointScore", description="ZXM买点评分指标，综合评估股票的买点表现")
        self.threshold = threshold
        
        # 初始化买点子指标
        self.daily_macd = ZXMDailyMACD()
        self.turnover = ZXMTurnover()
        self.ma_callback = ZXMMACallback()
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM买点评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含买点评分和信号
        """
        # 初始化结果数据框
        result = data.copy()
        
        # 计算MACD买点
        try:
            macd_result = self.daily_macd.calculate(data)
            macd_signal = macd_result["XG"]
        except Exception as e:
            logger.error(f"计算MACD买点出错: {e}")
            macd_signal = pd.Series(False, index=data.index)
        
        # 计算换手买点
        try:
            turnover_result = self.turnover.calculate(data)
            turnover_signal = turnover_result["XG"]
        except Exception as e:
            logger.error(f"计算换手买点出错: {e}")
            turnover_signal = pd.Series(False, index=data.index)
        
        # 计算回踩均线买点
        try:
            ma_callback_result = self.ma_callback.calculate(data)
            ma_callback_signal = ma_callback_result["XG"]
        except Exception as e:
            logger.error(f"计算回踩均线买点出错: {e}")
            ma_callback_signal = pd.Series(False, index=data.index)
        
        # 统计买点指标
        buy_point_indicators = [macd_signal, turnover_signal, ma_callback_signal]
        
        # 计算买点指标满足数量
        buy_point_count = pd.Series(0, index=data.index)
        for indicator in buy_point_indicators:
            buy_point_count += indicator.astype(int)
        
        # 计算买点评分（0-100分）
        buy_point_score = (buy_point_count / len(buy_point_indicators)) * 100
        
        # 生成满足买点条件的信号
        signal = buy_point_score >= self.threshold
        
        # 添加计算结果到数据框
        result["MACDBuyPoint"] = macd_signal
        result["TurnoverBuyPoint"] = turnover_signal
        result["MACallbackBuyPoint"] = ma_callback_signal
        result["BuyPointCount"] = buy_point_count
        result["BuyPointScore"] = buy_point_score
        result["Signal"] = signal
        
        return result



    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM买点评分指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 直接使用计算的买点评分作为原始评分
        return result["BuyPointScore"]
class StockScoreCalculator(BaseIndicator):
    """
    ZXM股票综合评分指标
    
    计算股票的综合评分
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM股票综合评分指标"""
        super().__init__(name="StockScoreCalculator", description="ZXM股票综合评分指标，计算股票的综合评分")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM股票综合评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算各类评分
        
        # 1. 趋势评分
        result = self._calculate_trend_score(data, result)
        
        # 2. 动量评分
        result = self._calculate_momentum_score(data, result)
        
        # 3. 波动率评分
        result = self._calculate_volatility_score(data, result)
        
        # 4. 成交量评分
        result = self._calculate_volume_score(data, result)
        
        # 5. 价值评分（如果有基本面数据）
        if "pe_ratio" in data.columns and "pb_ratio" in data.columns:
            result = self._calculate_value_score(data, result)
        
        # 6. 综合评分 - 加权平均
        weights = {
            "TrendScore": 0.35,
            "MomentumScore": 0.25,
            "VolatilityScore": 0.15,
            "VolumeScore": 0.25
        }
        
        # 如果有价值评分，则调整权重
        if "ValueScore" in result.columns:
            weights = {
                "TrendScore": 0.30,
                "MomentumScore": 0.20,
                "VolatilityScore": 0.10,
                "VolumeScore": 0.20,
                "ValueScore": 0.20
            }
        
        # 计算综合评分
        total_score = pd.Series(0.0, index=data.index)
        for score_name, weight in weights.items():
            if score_name in result.columns:
                total_score += result[score_name] * weight
        
        result["TotalScore"] = total_score
        
        # 7. 评分等级
        result["ScoreGrade"] = pd.cut(
            result["TotalScore"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["很差", "较差", "一般", "较好", "很好"]
        )
        
        # 8. 买入信号 - 当总分超过70分时
        result["BuySignal"] = result["TotalScore"] > 70
        
        # 9. 卖出信号 - 当总分低于30分时
        result["SellSignal"] = result["TotalScore"] < 30
        
        return result
    
    def _calculate_trend_score(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势评分
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 计算均线系统
        ma5 = data["close"].rolling(window=5).mean()
        ma10 = data["close"].rolling(window=10).mean()
        ma20 = data["close"].rolling(window=20).mean()
        ma60 = data["close"].rolling(window=60).mean()
        
        # 初始化趋势评分为50分（中性）
        trend_score = pd.Series(50.0, index=data.index)
        
        # 1. 均线多头排列得分
        bullish_ma = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
        trend_score[bullish_ma] += 20
        
        # 2. 均线空头排列扣分
        bearish_ma = (ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)
        trend_score[bearish_ma] -= 20
        
        # 3. 价格站上均线得分
        price_above_ma20 = data["close"] > ma20
        trend_score[price_above_ma20] += 10
        
        price_above_ma60 = data["close"] > ma60
        trend_score[price_above_ma60] += 10
        
        # 4. 价格跌破均线扣分
        price_below_ma20 = data["close"] < ma20
        trend_score[price_below_ma20] -= 10
        
        price_below_ma60 = data["close"] < ma60
        trend_score[price_below_ma60] -= 10
        
        # 5. 均线斜率得分
        ma20_slope = ma20.diff(5) / ma20.shift(5)
        ma60_slope = ma60.diff(5) / ma60.shift(5)
        
        trend_score[ma20_slope > 0.01] += 5  # 中期均线向上
        trend_score[ma20_slope < -0.01] -= 5  # 中期均线向下
        
        trend_score[ma60_slope > 0.005] += 5  # 长期均线向上
        trend_score[ma60_slope < -0.005] -= 5  # 长期均线向下
        
        # 确保评分在0-100范围内
        trend_score = trend_score.clip(0, 100)
        
        # 添加到结果
        result["TrendScore"] = trend_score
        
        return result
    
    def _calculate_momentum_score(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量评分
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 初始化动量评分为50分（中性）
        momentum_score = pd.Series(50.0, index=data.index)
        
        # 1. 计算各周期涨跌幅
        returns_5d = data["close"].pct_change(5) * 100
        returns_10d = data["close"].pct_change(10) * 100
        returns_20d = data["close"].pct_change(20) * 100
        
        # 2. 近期涨幅评分
        momentum_score[returns_5d > 5] += 10  # 5日涨幅大于5%
        momentum_score[returns_5d > 10] += 10  # 5日涨幅大于10%
        momentum_score[returns_5d < -5] -= 10  # 5日跌幅大于5%
        momentum_score[returns_5d < -10] -= 10  # 5日跌幅大于10%
        
        # 3. 中期涨幅评分
        momentum_score[returns_10d > 10] += 10  # 10日涨幅大于10%
        momentum_score[returns_10d < -10] -= 10  # 10日跌幅大于10%
        
        # 4. 长期涨幅评分
        momentum_score[returns_20d > 15] += 10  # 20日涨幅大于15%
        momentum_score[returns_20d < -15] -= 10  # 20日跌幅大于15%
        
        # 5. 动量加速/减速评分
        acceleration = returns_5d - returns_5d.shift(5)
        
        momentum_score[acceleration > 3] += 10  # 动量加速（5日涨幅比前5日提高3%以上）
        momentum_score[acceleration < -3] -= 10  # 动量减速（5日涨幅比前5日减少3%以上）
        
        # 确保评分在0-100范围内
        momentum_score = momentum_score.clip(0, 100)
        
        # 添加到结果
        result["MomentumScore"] = momentum_score
        
        return result
    
    def _calculate_volatility_score(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率评分
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 初始化波动率评分为50分（中性）
        volatility_score = pd.Series(50.0, index=data.index)
        
        # 1. 计算真实波幅（TR）
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift(1)).abs()
        low_close = (data["low"] - data["close"].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # 2. 计算平均真实波幅（ATR）
        atr_14 = tr.rolling(window=14).mean()
        
        # 3. 计算波动率（ATR/收盘价）
        volatility = atr_14 / data["close"] * 100
        
        # 4. 波动率评分
        # 高波动率（>3%）扣分，低波动率（<1%）加分
        volatility_score[volatility > 3] -= 15
        volatility_score[volatility > 5] -= 15  # 极端波动率（>5%）进一步扣分
        
        volatility_score[volatility < 1] += 15
        
        # 5. 波动率变化评分
        volatility_change = volatility - volatility.shift(5)
        
        volatility_score[volatility_change > 1] -= 10  # 波动率上升扣分
        volatility_score[volatility_change < -1] += 10  # 波动率下降加分
        
        # 6. 价格稳定性评分
        close_std = data["close"].rolling(window=20).std() / data["close"].rolling(window=20).mean() * 100
        
        volatility_score[close_std < 2] += 10  # 价格稳定加分
        volatility_score[close_std > 5] -= 10  # 价格不稳定扣分
        
        # 确保评分在0-100范围内
        volatility_score = volatility_score.clip(0, 100)
        
        # 添加到结果
        result["VolatilityScore"] = volatility_score
        
        return result
    
    def _calculate_volume_score(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量评分
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 初始化成交量评分为50分（中性）
        volume_score = pd.Series(50.0, index=data.index)
        
        # 1. 计算成交量变化率
        volume_change = data["volume"].pct_change() * 100
        
        # 2. 计算相对成交量（与20日均量比较）
        volume_ratio = data["volume"] / data["volume"].rolling(window=20).mean()
        
        # 3. 成交量放大评分
        volume_score[volume_ratio > 2] += 15  # 成交量是20日均量的2倍以上
        volume_score[volume_ratio > 3] += 10  # 成交量是20日均量的3倍以上
        
        # 4. 成交量萎缩评分
        volume_score[volume_ratio < 0.5] -= 15  # 成交量不足20日均量的50%
        
        # 5. 量价配合评分
        price_up = data["close"] > data["open"]
        price_down = data["close"] < data["open"]
        
        # 放量上涨加分
        volume_score[(price_up) & (volume_ratio > 1.5)] += 15
        
        # 放量下跌扣分
        volume_score[(price_down) & (volume_ratio > 1.5)] -= 15
        
        # 缩量上涨加分（潜在突破前蓄势）
        volume_score[(price_up) & (volume_ratio < 0.8)] += 10
        
        # 缩量下跌加分（可能企稳）
        volume_score[(price_down) & (volume_ratio < 0.8)] += 5
        
        # 6. 连续成交量变化评分
        volume_increase_days = pd.Series(0, index=data.index)
        volume_decrease_days = pd.Series(0, index=data.index)
        
        for i in range(5, len(data)):
            if all(data["volume"].iloc[i-j] > data["volume"].iloc[i-j-1] for j in range(3)):
                volume_increase_days.iloc[i] = 3
            if all(data["volume"].iloc[i-j] < data["volume"].iloc[i-j-1] for j in range(3)):
                volume_decrease_days.iloc[i] = 3
        
        # 连续放量加分（可能突破）
        volume_score[volume_increase_days == 3] += 10
        
        # 连续缩量扣分（可能无人问津）
        volume_score[volume_decrease_days == 3] -= 10
        
        # 确保评分在0-100范围内
        volume_score = volume_score.clip(0, 100)
        
        # 添加到结果
        result["VolumeScore"] = volume_score
        
        return result
    
    def _calculate_value_score(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算价值评分（基于基本面指标）
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 初始化价值评分为50分（中性）
        value_score = pd.Series(50.0, index=data.index)
        
        # 1. PE评分
        pe_ratio = data["pe_ratio"]
        
        value_score[pe_ratio < 15] += 10  # PE低于15
        value_score[pe_ratio < 10] += 10  # PE低于10
        value_score[pe_ratio > 30] -= 10  # PE高于30
        value_score[pe_ratio > 50] -= 10  # PE高于50
        
        # 2. PB评分
        pb_ratio = data["pb_ratio"]
        
        value_score[pb_ratio < 1.5] += 10  # PB低于1.5
        value_score[pb_ratio < 1] += 10  # PB低于1
        value_score[pb_ratio > 3] -= 10  # PB高于3
        value_score[pb_ratio > 5] -= 10  # PB高于5
        
        # 3. 如果有其他基本面指标，可以继续添加
        if "roe" in data.columns:
            roe = data["roe"]
            value_score[roe > 15] += 10  # ROE高于15%
            value_score[roe > 20] += 10  # ROE高于20%
        
        if "dividend_yield" in data.columns:
            dividend_yield = data["dividend_yield"]
            value_score[dividend_yield > 2] += 10  # 股息率高于2%
            value_score[dividend_yield > 4] += 10  # 股息率高于4%
        
        # 确保评分在0-100范围内
        value_score = value_score.clip(0, 100)
        
        # 添加到结果
        result["ValueScore"] = value_score
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM股票综合评分指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 直接使用计算的总分作为原始评分
        score = result["TotalScore"]
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM股票综合评分指标相关的技术形态
        
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
            
            # 评分等级形态
            if "ScoreGrade" in result.columns:
                patterns.append(f"综合评分等级：{last_row['ScoreGrade']}")
            
            # 总分形态
            total_score = last_row["TotalScore"]
            if total_score >= 80:
                patterns.append("高分股票(80+)")
            elif total_score >= 70:
                patterns.append("优质股票(70-80)")
            elif total_score <= 30:
                patterns.append("低分股票(<30)")
            
            # 各分项评分形态
            if last_row["TrendScore"] >= 70:
                patterns.append("趋势强劲")
            elif last_row["TrendScore"] <= 30:
                patterns.append("趋势疲软")
            
            if last_row["MomentumScore"] >= 70:
                patterns.append("动量强劲")
            elif last_row["MomentumScore"] <= 30:
                patterns.append("动量疲软")
            
            if last_row["VolatilityScore"] >= 70:
                patterns.append("低波动性")
            elif last_row["VolatilityScore"] <= 30:
                patterns.append("高波动性")
            
            if last_row["VolumeScore"] >= 70:
                patterns.append("成交量理想")
            elif last_row["VolumeScore"] <= 30:
                patterns.append("成交量不佳")
            
            if "ValueScore" in result.columns:
                if last_row["ValueScore"] >= 70:
                    patterns.append("高价值股")
                elif last_row["ValueScore"] <= 30:
                    patterns.append("低价值股")
            
            # 买卖信号形态
            if last_row["BuySignal"]:
                patterns.append("买入信号")
            elif last_row["SellSignal"]:
                patterns.append("卖出信号")
        
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
        score = result["TotalScore"]
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 设置买卖信号
        signals['buy_signal'] = result["BuySignal"]
        signals['sell_signal'] = result["SellSignal"]
        signals['neutral_signal'] = ~(result["BuySignal"] | result["SellSignal"])
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["TotalScore"] >= 60, 'trend'] = 1  # 高分看涨
        signals.loc[result["TotalScore"] <= 40, 'trend'] = -1  # 低分看跌
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["BuySignal"], 'signal_type'] = 'zxm_high_score_buy'
        signals.loc[result["SellSignal"], 'signal_type'] = 'zxm_low_score_sell'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            desc_parts = []
            
            # 添加总分
            desc_parts.append(f"总分{result.loc[i, 'TotalScore']:.1f}")
            
            # 添加评分等级
            if "ScoreGrade" in result.columns:
                desc_parts.append(f"等级{result.loc[i, 'ScoreGrade']}")
            
            # 添加优势项
            strengths = []
            if result.loc[i, "TrendScore"] >= 70:
                strengths.append("趋势")
            if result.loc[i, "MomentumScore"] >= 70:
                strengths.append("动量")
            if result.loc[i, "VolatilityScore"] >= 70:
                strengths.append("稳定性")
            if result.loc[i, "VolumeScore"] >= 70:
                strengths.append("成交量")
            if "ValueScore" in result.columns and result.loc[i, "ValueScore"] >= 70:
                strengths.append("价值")
            
            if strengths:
                desc_parts.append("优势:" + "/".join(strengths))
            
            # 添加劣势项
            weaknesses = []
            if result.loc[i, "TrendScore"] <= 30:
                weaknesses.append("趋势")
            if result.loc[i, "MomentumScore"] <= 30:
                weaknesses.append("动量")
            if result.loc[i, "VolatilityScore"] <= 30:
                weaknesses.append("稳定性")
            if result.loc[i, "VolumeScore"] <= 30:
                weaknesses.append("成交量")
            if "ValueScore" in result.columns and result.loc[i, "ValueScore"] <= 30:
                weaknesses.append("价值")
            
            if weaknesses:
                desc_parts.append("劣势:" + "/".join(weaknesses))
            
            signals.loc[i, 'signal_desc'] = "，".join(desc_parts)
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据分数和项目一致性调整置信度
        for i in signals.index:
            # 高分或低分的置信度更高
            if result.loc[i, "TotalScore"] >= 80:
                signals.loc[i, 'confidence'] = 80
            elif result.loc[i, "TotalScore"] <= 20:
                signals.loc[i, 'confidence'] = 80
            
            # 分项一致性高的置信度更高
            if signals.loc[i, 'buy_signal']:
                consistency_score = sum([
                    1 if result.loc[i, "TrendScore"] >= 60 else 0,
                    1 if result.loc[i, "MomentumScore"] >= 60 else 0,
                    1 if result.loc[i, "VolatilityScore"] >= 60 else 0,
                    1 if result.loc[i, "VolumeScore"] >= 60 else 0
                ])
                if consistency_score >= 3:  # 至少3个分项都支持买入
                    signals.loc[i, 'confidence'] = min(90, signals.loc[i, 'confidence'] + 20)
            
            elif signals.loc[i, 'sell_signal']:
                consistency_score = sum([
                    1 if result.loc[i, "TrendScore"] <= 40 else 0,
                    1 if result.loc[i, "MomentumScore"] <= 40 else 0,
                    1 if result.loc[i, "VolatilityScore"] <= 40 else 0,
                    1 if result.loc[i, "VolumeScore"] <= 40 else 0
                ])
                if consistency_score >= 3:  # 至少3个分项都支持卖出
                    signals.loc[i, 'confidence'] = min(90, signals.loc[i, 'confidence'] + 20)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        signals.loc[result["VolatilityScore"] <= 30, 'risk_level'] = '高'  # 高波动性，高风险
        signals.loc[result["VolatilityScore"] >= 70, 'risk_level'] = '低'  # 低波动性，低风险
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据总分和置信度调整仓位
        high_score_high_conf = (result["TotalScore"] >= 80) & (signals['confidence'] >= 80)
        signals.loc[high_score_high_conf, 'position_size'] = 0.5  # 高分高置信度，加大仓位
        
        exceptional_score = result["TotalScore"] >= 90
        signals.loc[exceptional_score, 'position_size'] = 0.7  # 极高分，最大仓位
        
        # 止损位 - 使用近期低点或移动平均线
        signals['stop_loss'] = 0.0
        ma20 = data["close"].rolling(window=20).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 20:
                    # 使用20日均线作为止损位基准
                    ma20_val = ma20.iloc[idx]
                    # 如果价格远高于均线，使用均线作为止损；否则使用近期低点
                    if data["close"].iloc[idx] > ma20_val * 1.1:
                        signals.loc[i, 'stop_loss'] = ma20_val * 0.95  # 均线下方5%
                    else:
                        # 使用近期低点
                        low_price = data.iloc[idx-20:idx+1]['low'].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
            except:
                continue
        
        # 市场环境和成交量确认
        signals['market_env'] = 'normal'
        
        # 判断市场环境
        # 使用市场趋势得分作为参考
        market_trend = pd.Series('normal', index=data.index)
        market_trend[result["TrendScore"] >= 75] = 'bull_market'
        market_trend[result["TrendScore"] <= 25] = 'bear_market'
        signals['market_env'] = market_trend
        
        # 成交量确认
        signals['volume_confirmation'] = result["VolumeScore"] >= 60
        
        return signals 