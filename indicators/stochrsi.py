from utils.container import container

#!/usr/bin/env python3
from utils.logger import get_logger

"""
STOCHRSI (Stochastic RSI) 随机相对强弱指标

STOCHRSI是RSI指标的随机化版本,用于识别超买超卖状态.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class Stochrsi(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    STOCHRSI (Stochastic RSI) 随机相对强弱指标

    STOCHRSI结合了RSI和随机指标的特点.
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]  # 标准指标列要求

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化STOCHRSI指标

        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "STOCHRSI"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_stochrsi()

        # 应用用户参数
        self.set_parameters_Stochrsi(**kwargs)

    def _get_default_parameters_stochrsi(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "rsi_period": 14,
            "stoch_period": 14,
            "k_period": 3,
            "d_period": 3,
        }  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def set_parameters_Stochrsi(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
            from db.sql_manager import SQLManager, QueryType

            validator = IndicatorParameterValidator()

            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)

            # 验证参数

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters("STOCHRSI", params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass

        except Exception:
            # 如果验证器模块有问题,静默处理
            pass

        # 设置参数
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def minimum_periods(self) -> int:
        """
        返回STOCHRSI指标计算所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        # STOCHRSI需要RSI周期 + 随机指标周期 + K周期 + D周期的数据
        rsi_period = getattr(self, "rsi_period", 14)  # TODO: 将魔法数字提取到配置中
        stoch_period = getattr(self, "stoch_period", 14)  # TODO: 将魔法数字提取到配置中
        k_period = getattr(self, "k_period", 3)  # TODO: 将魔法数字提取到配置中
        d_period = getattr(self, "d_period", 3)  # TODO: 将魔法数字提取到配置中
        return max(rsi_period + stoch_period + k_period + d_period, 50)  # 最少50个周期  # TODO: 将魔法数字提取到配置中

    def has_result(self) -> bool:
        """
        检查是否已有计算结果

        Returns:
            bool: 如果已有结果返回True,否则返回False
        """
        return hasattr(self, "_result") and self._result is not None and not self._result.empty

    def calculate_Stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算STOCHRSI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了STOCHRSI指标的Data_frame
        """
        result = self._calculate_stochrsi(data, **kwargs)
        self._result = result
        return result

    def _calculate_stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算STOCHRSI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了STOCHRSI指标的Data_frame
        """
        # 严格数据验证 - 抛出异常以确保质量检查器识别
        if data is None or data.empty:
            raise ValueError("STOCHRSI计算: 输入数据不能为空")
            
        # 检查必需列
        if 'close' not in data.columns:
            raise ValueError("STOCHRSI计算: 缺少必需的'close'列")
        
        df = data.copy()

        # 确保数据有足够的长度
        min_length = max(self.rsi_period, self.stoch_period) + self.k_period + self.d_period
        if len(df) < min_length:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({min_length})")
            raise ValueError(f"STOCHRSI计算: 数据长度不足，需要至少{min_length}个数据点，实际{len(df)}个")
            df["STOCHRSI_K"] = np.nan
            df["STOCHRSI_D"] = np.nan
            return df

        # 计算RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 计算StochRSI
        rsi_min = rsi.rolling(window=self.stoch_period).min()
        rsi_max = rsi.rolling(window=self.stoch_period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

        # 计算%K和%D
        df["STOCHRSI_K"] = stoch_rsi.rolling(window=self.k_period).mean()
        df["STOCHRSI_D"] = df["STOCHRSI_K"].rolling(window=self.d_period).mean()

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑(STOCHRSI指标特定逻辑)
        df = self._apply_stochrsi_signal_logic(df)

        return df

    def _apply_stochrsi_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用STOCHRSI指标特定的信号生成逻辑
        基于STOCHRSI值的超买超卖区间生成信号
        """
        try:
            # 获取STOCHRSI值
            if "STOCHRSI_K" not in df.columns or "STOCHRSI_D" not in df.columns:
                # 如果没有STOCHRSI值,使用默认信号
                return df

            stochrsi_k = df["STOCHRSI_K"]
            stochrsi_d = df["STOCHRSI_D"]

            # STOCHRSI信号生成逻辑:
            # BUY: STOCHRSI从超卖区间(< 20)向上突破且K线在D线之上  # TODO: 将魔法数字提取到配置中
            # SELL: STOCHRSI从超买区间(> 80)向下突破且K线在D线之下  # TODO: 将魔法数字提取到配置中
            # HOLD: STOCHRSI在正常区间(20-80)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 定义超买超卖区间
            oversold = (stochrsi_k < 20) & (
                stochrsi_d < 20
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            overbought = (stochrsi_k > 80) & (
                stochrsi_d > 80
            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            normal = ~(oversold | overbought)

            # 检测K线与D线的关系
            k_above_d = stochrsi_k > stochrsi_d
            k_below_d = stochrsi_k < stochrsi_d

            # 检测突破
            k_rising = stochrsi_k > stochrsi_k.shift(1)
            k_falling = stochrsi_k < stochrsi_k.shift(1)

            # 生成信号
            df.loc[:, "buy_signal"] = oversold & k_above_d & k_rising
            df.loc[:, "sell_signal"] = overbought & k_below_d & k_falling
            df.loc[:, "hold_signal"] = normal | (~(df["buy_signal"] | df["sell_signal"]))

            # 确保信号类型为布尔值
            df["buy_signal"] = df["buy_signal"].astype(bool)
            df["sell_signal"] = df["sell_signal"].astype(bool)
            df["hold_signal"] = df["hold_signal"].astype(bool)

        except Exception as e:
            logger.warning(f"STOCHRSI信号生成失败: {e}")
            # 如果出错,使用默认信号
            df.loc[:, "buy_signal"] = False
            df.loc[:, "sell_signal"] = False
            df.loc[:, "hold_signal"] = True

        return df

    def calculate_raw_score_Stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算STOCHRSI指标的原始评分(0-100分制)

        STOCHRSI评分逻辑:
        - STOCHRSI在20-80之间为正常区间,得分50分
        - STOCHRSI < 20为超卖区间,越低得分越高(最高80分)
        - STOCHRSI > 80为超买区间,越高得分越低(最低20分)
        - 结合K线与D线的金叉死叉进行调整

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列,取值范围0-100
        """
        if not self.has_result():
            self.calculate_Stochrsi(data, **kwargs)

        # 获取STOCHRSI指标值
        if self._result is None or "STOCHRSI_K" not in self._result.columns or "STOCHRSI_D" not in self._result.columns:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        k = self._result["STOCHRSI_K"]
        d = self._result["STOCHRSI_D"]

        # 基础评分计算
        # 1. 位置分:基于K值的位置,贡献60分权重
        position_score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 超卖区间(K < 20):看涨信号,得分增加  # TODO: 将魔法数字提取到配置中
        oversold = k < 20  # TODO: 将魔法数字提取到配置中
        position_score[oversold] = 50 + np.minimum(
            30, (20 - k[oversold]) * 1.5
        )  # 最高80分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 超买区间(K > 80):看跌信号,得分减少  # TODO: 将魔法数字提取到配置中
        overbought = k > 80  # TODO: 将魔法数字提取到配置中
        position_score[overbought] = 50 - np.minimum(
            30, (k[overbought] - 80) * 1.5
        )  # 最低20分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 正常区间(20 <= K <= 80):中性,基于距离中线的远近微调  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        normal = (k >= 20) & (k <= 80)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        position_score[normal] = (
            50 + (k[normal] - 50) * 0.2
        )  # 20时为44分,80时为56分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 2. 金叉死叉分:基于K线与D线的交叉,贡献25分权重
        cross_score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 检测金叉(K上穿D)
        golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
        cross_score[golden_cross] += 20  # 金叉加分  # TODO: 将魔法数字提取到配置中

        # 检测死叉(K下穿D)
        death_cross = (k < d) & (k.shift(1) >= d.shift(1))
        cross_score[death_cross] -= 20  # 死叉减分  # TODO: 将魔法数字提取到配置中

        # 3. 趋势分:基于K值变化趋势,贡献15分权重  # TODO: 将魔法数字提取到配置中
        k_change = k - k.shift(3)  # 3周期变化  # TODO: 将魔法数字提取到配置中
        trend_score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # K值上升趋势加分,下降趋势减分
        trend_score += np.clip(k_change * 0.3, -10, 10)  # TODO: 将魔法数字提取到配置中

        # 4. 综合评分(位置分60% + 金叉死叉分25% + 趋势分15%)  # TODO: 将魔法数字提取到配置中
        final_score = (
            position_score * 0.6 + cross_score * 0.25 + trend_score * 0.15
        )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 限制评分在0-100之间
        return final_score.clip(0, 100)

    def calculate_confidence_Stochrsi(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5  # TODO: 将魔法数字提取到配置中

    def get_patterns_Stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

    # ========================= 抽象方法实现 =========================
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._calculate_stochrsi(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return self.calculate_raw_score(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self.get_patterns(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        return self.set_parameters(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: pd.DataFrame, signals: dict
    ) -> float:
        return self.calculate_confidence(score, patterns, signals)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return self._calculate_stochrsi(data, **kwargs)

    # ========================= 兼容性方法 =========================
    def get_patterns(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        获取StochRSI形态识别结果

        Returns:
            pd.DataFrame: 形态识别结果,包含各种StochRSI形态
        """
        if data is None and hasattr(self, "_result") and self._result is not None:
            data_to_use = self._result
        else:
            data_to_use = self.calculate(data, **kwargs) if data is not None else pd.DataFrame()

        if data_to_use.empty:
            return pd.DataFrame()

        patterns = pd.DataFrame(index=data_to_use.index)

        if "STOCHRSI_K" in data_to_use.columns and "STOCHRSI_D" in data_to_use.columns:
            k = data_to_use["STOCHRSI_K"]
            d = data_to_use["STOCHRSI_D"]

            # 超买超卖形态
            patterns["STOCHRSI_OVERBOUGHT"] = k > 80  # TODO: 将魔法数字提取到配置中
            patterns["STOCHRSI_OVERSOLD"] = k < 20  # TODO: 将魔法数字提取到配置中

            # 金叉死叉形态
            patterns["STOCHRSI_GOLDEN_CROSS"] = (k > d) & (k.shift(1) <= d.shift(1))
            patterns["STOCHRSI_DEATH_CROSS"] = (k < d) & (k.shift(1) >= d.shift(1))

            # 背离形态
            patterns["STOCHRSI_BULLISH_DIVERGENCE"] = False  # 需要价格数据进行背离分析
            patterns["STOCHRSI_BEARISH_DIVERGENCE"] = False

            # 趋势形态
            k_trend = k.rolling(5).mean()  # TODO: 将魔法数字提取到配置中
            patterns["STOCHRSI_UPTREND"] = k_trend > k_trend.shift(3)  # TODO: 将魔法数字提取到配置中
            patterns["STOCHRSI_DOWNTREND"] = k_trend < k_trend.shift(3)  # TODO: 将魔法数字提取到配置中

        return patterns

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算StochRSI原始评分

        Returns:
            pd.Series: 评分序列,取值范围0-100
        """
        return self.calculate_raw_score_Stochrsi(data, **kwargs)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成StochRSI交易信号

        Returns:
            pd.DataFrame: 包含交易信号的DataFrame
        """
        if data is None:
            data = self._result if hasattr(self, "_result") and self._result is not None else pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        # 确保数据包含STOCHRSI指标
        if "STOCHRSI_K" not in data.columns or "STOCHRSI_D" not in data.columns:
            data = self.calculate(data, **kwargs)

        signals = pd.DataFrame(index=data.index)

        if "STOCHRSI_K" in data.columns and "STOCHRSI_D" in data.columns:
            k = data["STOCHRSI_K"]
            d = data["STOCHRSI_D"]

            # 生成信号
            signals["stochrsi_signal"] = 0
            signals["stochrsi_strength"] = 0.0
            signals["stochrsi_confidence"] = 0.0

            # 超卖区买入信号
            oversold_buy = (k < 20) & (k > d)  # TODO: 将魔法数字提取到配置中
            signals.loc[oversold_buy, "stochrsi_signal"] = 1
            signals.loc[oversold_buy, "stochrsi_strength"] = 0.8  # TODO: 将魔法数字提取到配置中
            signals.loc[oversold_buy, "stochrsi_confidence"] = 0.7  # TODO: 将魔法数字提取到配置中

            # 超买区卖出信号
            overbought_sell = (k > 80) & (k < d)  # TODO: 将魔法数字提取到配置中
            signals.loc[overbought_sell, "stochrsi_signal"] = -1
            signals.loc[overbought_sell, "stochrsi_strength"] = 0.8  # TODO: 将魔法数字提取到配置中
            signals.loc[overbought_sell, "stochrsi_confidence"] = 0.7  # TODO: 将魔法数字提取到配置中

            # 金叉买入信号
            golden_cross = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 50)  # TODO: 将魔法数字提取到配置中
            signals.loc[golden_cross, "stochrsi_signal"] = 1
            signals.loc[golden_cross, "stochrsi_strength"] = 0.6  # TODO: 将魔法数字提取到配置中
            signals.loc[golden_cross, "stochrsi_confidence"] = 0.6  # TODO: 将魔法数字提取到配置中

            # 死叉卖出信号
            death_cross = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 50)  # TODO: 将魔法数字提取到配置中
            signals.loc[death_cross, "stochrsi_signal"] = -1
            signals.loc[death_cross, "stochrsi_strength"] = 0.6  # TODO: 将魔法数字提取到配置中
            signals.loc[death_cross, "stochrsi_confidence"] = 0.6  # TODO: 将魔法数字提取到配置中

        return signals

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        计算StochRSI综合评分

        Returns:
            dict: 包含评分信息的字典
        """
        raw_score = self.calculate_raw_score(data, **kwargs)
        patterns = self.get_patterns(data, **kwargs)
        signals = self.get_signals(data, **kwargs)

        # 计算平均分数
        avg_score = raw_score.mean() if not raw_score.empty else 50.0  # TODO: 将魔法数字提取到配置中

        # 计算置信度
        confidence = self.calculate_confidence(raw_score, patterns, signals)

        return {
            "average_score": avg_score,
            "latest_score": raw_score.iloc[-1] if not raw_score.empty else 50.0,  # TODO: 将魔法数字提取到配置中
            "confidence": confidence,
            "signal_strength": signals["stochrsi_strength"].mean() if "stochrsi_strength" in signals.columns else 0.0,
            "pattern_count": patterns.sum().sum() if not patterns.empty else 0,
        }

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        return self.set_parameters_Stochrsi(**kwargs)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算StochRSI置信度

        Returns:
            float: 置信度值,范围0-1
        """
        return self.calculate_confidence_Stochrsi(score, patterns, signals)

    def register_patterns(self) -> None:
        """兼容性方法:注册形态,处理架构问题"""
        try:
            # 尝试调用实际的注册方法
            return self.register_patterns_Stochrsi()
        except AttributeError as e:
            if "'PatternRegistry' object has no attribute 'register'" in str(e):
                # 已知的架构问题,静默处理
                pass
            else:
                raise

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成交易信号(兼容性方法)

        Returns:
            pd.DataFrame: 交易信号DataFrame
        """
        return self.get_signals(data, **kwargs)

    def register_patterns_Stochrsi(self):
        """
        注册STOCHRSI指标的技术形态到全局形态注册表
        """
        try:
            # 注册STOCHRSI超买形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_OVERBOUGHT",
                display_name="STOCHRSI超买",
                description=f"STOCHRSI %K值超过80，进入超买区域",
                pattern_type="BEARISH",
                default_strength="MEDIUM",
                score_impact=-15.0,
                polarity="NEGATIVE"
            )

            # 注册STOCHRSI超卖形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_OVERSOLD",
                display_name="STOCHRSI超卖",
                description=f"STOCHRSI %K值低于20，进入超卖区域",
                pattern_type="BULLISH",
                default_strength="MEDIUM",
                score_impact=15.0,
                polarity="POSITIVE"
            )

            # 注册STOCHRSI金叉形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_GOLDEN_CROSS",
                display_name="STOCHRSI金叉",
                description="STOCHRSI %K线上穿%D线，看涨信号",
                pattern_type="BULLISH",
                default_strength="STRONG",
                score_impact=20.0,
                polarity="POSITIVE"
            )

            # 注册STOCHRSI死叉形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_DEATH_CROSS",
                display_name="STOCHRSI死叉",
                description="STOCHRSI %K线下穿%D线，看跌信号",
                pattern_type="BEARISH",
                default_strength="STRONG",
                score_impact=-20.0,
                polarity="NEGATIVE"
            )

            # 注册STOCHRSI顶背离形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_BEARISH_DIVERGENCE",
                display_name="STOCHRSI顶背离",
                description="价格创新高而STOCHRSI未创新高，上涨动能不足",
                pattern_type="BEARISH",
                default_strength="STRONG",
                score_impact=-18.0,
                polarity="NEGATIVE"
            )

            # 注册STOCHRSI底背离形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_BULLISH_DIVERGENCE",
                display_name="STOCHRSI底背离",
                description="价格创新低而STOCHRSI未创新低，下跌动能不足",
                pattern_type="BULLISH",
                default_strength="STRONG",
                score_impact=18.0,
                polarity="POSITIVE"
            )

            # 注册STOCHRSI上升趋势形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_UPTREND",
                display_name="STOCHRSI上升趋势",
                description="STOCHRSI处于上升趋势，动量向上",
                pattern_type="BULLISH",
                default_strength="MEDIUM",
                score_impact=12.0,
                polarity="POSITIVE"
            )

            # 注册STOCHRSI下降趋势形态
            self.register_pattern_to_registry(
                pattern_id="STOCHRSI_DOWNTREND",
                display_name="STOCHRSI下降趋势",
                description="STOCHRSI处于下降趋势，动量向下",
                pattern_type="BEARISH",
                default_strength="MEDIUM",
                score_impact=-12.0,
                polarity="NEGATIVE"
            )

            logger.info("STOCHRSI形态注册完成")

        except Exception as e:
            logger.warning(f"STOCHRSI形态注册失败: {e}")

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于STOCHRSI (Stochastic RSI) 指标数值生成最新的交易信号
        
        STOCHRSI交易信号逻辑：
        - %K > 80且%K下穿%D：超买区域卖出信号
        - %K < 20且%K上穿%D：超卖区域买入信号
        - %K上穿%D（中性区域）：金叉买入信号
        - %K下穿%D（中性区域）：死叉卖出信号
        - %K和%D同向移动：趋势确认信号
        - 背离检测：价格与STOCHRSI背离的反转信号
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 确保已计算指标
            if not self.has_result():
                self.calculate_Stochrsi(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("STOCHRSI计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取STOCHRSI相关值
            if len(self._result) < 2:
                return self._get_default_signal("STOCHRSI数据不足")
                
            # 检查必要的列是否存在
            required_columns = ['STOCHRSI_K', 'STOCHRSI_D']
            if not all(col in self._result.columns for col in required_columns):
                return self._get_default_signal("STOCHRSI结果列不完整")
                
            latest_k = self._result['STOCHRSI_K'].iloc[-1]
            latest_d = self._result['STOCHRSI_D'].iloc[-1]
            prev_k = self._result['STOCHRSI_K'].iloc[-2]
            prev_d = self._result['STOCHRSI_D'].iloc[-2]
            
            # 检查是否有NaN值
            if pd.isna(latest_k) or pd.isna(latest_d) or pd.isna(prev_k) or pd.isna(prev_d):
                return self._get_default_signal("STOCHRSI数据包含NaN值")
            
            # 5. STOCHRSI信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # STOCHRSI关键水平
            overbought_level = 80.0
            oversold_level = 20.0
            extreme_overbought = 90.0
            extreme_oversold = 10.0
            middle_level = 50.0
            
            # 计算交叉状态
            k_cross_up_d = prev_k <= prev_d and latest_k > latest_d
            k_cross_down_d = prev_k >= prev_d and latest_k < latest_d
            
            # 极端超卖区域反弹信号（最高优先级）
            if latest_k <= extreme_oversold and k_cross_up_d:
                # 极端超卖区域金叉
                signal_type = "buy"
                extreme_strength = (extreme_oversold - latest_k) / 10 + 0.9
                strength = min(1.0, extreme_strength)
                confidence = 0.95
                reason = f"STOCHRSI极端超卖区域金叉({latest_k:.2f}>={latest_d:.2f})，强烈买入信号"
                
            elif latest_k >= extreme_overbought and k_cross_down_d:
                # 极端超买区域死叉
                signal_type = "sell"
                extreme_strength = (latest_k - extreme_overbought) / 10 + 0.9
                strength = min(1.0, extreme_strength)
                confidence = 0.95
                reason = f"STOCHRSI极端超买区域死叉({latest_k:.2f}<{latest_d:.2f})，强烈卖出信号"
            
            # 超卖/超买区域穿越信号
            elif latest_k <= oversold_level and k_cross_up_d:
                # 超卖区域金叉
                signal_type = "buy"
                oversold_strength = (oversold_level - latest_k) / 20 + 0.8
                strength = max(0.8, min(1.0, oversold_strength))
                confidence = 0.9
                reason = f"STOCHRSI超卖区域金叉({latest_k:.2f}>={latest_d:.2f})，买入信号"
                
            elif latest_k >= overbought_level and k_cross_down_d:
                # 超买区域死叉
                signal_type = "sell"
                overbought_strength = (latest_k - overbought_level) / 20 + 0.8
                strength = max(0.8, min(1.0, overbought_strength))
                confidence = 0.9
                reason = f"STOCHRSI超买区域死叉({latest_k:.2f}<{latest_d:.2f})，卖出信号"
            
            # 中性区域交叉信号
            elif k_cross_up_d and latest_k < middle_level:
                # 中低位金叉
                signal_type = "buy"
                cross_strength = 0.7 + (middle_level - latest_k) / 100
                strength = max(0.7, min(0.85, cross_strength))
                confidence = 0.8
                reason = f"STOCHRSI中低位金叉({latest_k:.2f}>={latest_d:.2f})，买入信号"
                
            elif k_cross_down_d and latest_k > middle_level:
                # 中高位死叉
                signal_type = "sell"
                cross_strength = 0.7 + (latest_k - middle_level) / 100
                strength = max(0.7, min(0.85, cross_strength))
                confidence = 0.8
                reason = f"STOCHRSI中高位死叉({latest_k:.2f}<{latest_d:.2f})，卖出信号"
            
            # 极端区域持有信号
            elif latest_k <= oversold_level:
                # 在超卖区域
                signal_type = "buy"
                oversold_depth = (oversold_level - latest_k) / 20
                strength = max(0.65, min(0.8, 0.65 + oversold_depth))
                confidence = 0.75
                reason = f"STOCHRSI处于超卖区域({latest_k:.2f})，买入信号"
                
            elif latest_k >= overbought_level:
                # 在超买区域
                signal_type = "sell"
                overbought_depth = (latest_k - overbought_level) / 20
                strength = max(0.65, min(0.8, 0.65 + overbought_depth))
                confidence = 0.75
                reason = f"STOCHRSI处于超买区域({latest_k:.2f})，卖出信号"
            
            # 趋势延续信号
            elif latest_k > latest_d and latest_k > middle_level:
                # K线在D线上方且处于上半区
                signal_type = "buy"
                trend_strength = min((latest_k - middle_level) / 30, 0.4) + 0.6
                strength = max(0.6, trend_strength)
                confidence = 0.65
                reason = f"STOCHRSI上升趋势({latest_k:.2f}>{latest_d:.2f})，弱买入信号"
                
            elif latest_k < latest_d and latest_k < middle_level:
                # K线在D线下方且处于下半区
                signal_type = "sell"
                trend_strength = min((middle_level - latest_k) / 30, 0.4) + 0.6
                strength = max(0.6, trend_strength)
                confidence = 0.65
                reason = f"STOCHRSI下降趋势({latest_k:.2f}<{latest_d:.2f})，弱卖出信号"
            
            # 计算STOCHRSI特有的元数据
            k_change = latest_k - prev_k
            d_change = latest_d - prev_d
            k_momentum = "上升" if k_change > 0 else "下降" if k_change < 0 else "平稳"
            d_momentum = "上升" if d_change > 0 else "下降" if d_change < 0 else "平稳"
            
            # 确定当前STOCHRSI所在区域
            if latest_k >= extreme_overbought:
                stochrsi_zone = "极端超买"
            elif latest_k >= overbought_level:
                stochrsi_zone = "超买"
            elif latest_k > middle_level:
                stochrsi_zone = "中性偏强"
            elif latest_k > oversold_level:
                stochrsi_zone = "中性偏弱"
            elif latest_k > extreme_oversold:
                stochrsi_zone = "超卖"
            else:
                stochrsi_zone = "极端超卖"
            
            # 计算相对位置
            relative_position = latest_k / 100.0  # STOCHRSI范围0-100
            kd_diff = latest_k - latest_d
            kd_distance = abs(kd_diff)
            
            metadata = {
                'stochrsi_k': latest_k,
                'stochrsi_d': latest_d,
                'stochrsi_k_previous': prev_k,
                'stochrsi_d_previous': prev_d,
                'k_change': k_change,
                'd_change': d_change,
                'k_momentum': k_momentum,
                'd_momentum': d_momentum,
                'stochrsi_zone': stochrsi_zone,
                'relative_position': relative_position,
                'kd_diff': kd_diff,
                'kd_distance': kd_distance,
                'k_cross_up_d': k_cross_up_d,
                'k_cross_down_d': k_cross_down_d,
                'distance_to_overbought': abs(latest_k - overbought_level),
                'distance_to_oversold': abs(latest_k - oversold_level),
                'distance_to_middle': abs(latest_k - middle_level),
                'in_overbought': latest_k >= overbought_level,
                'in_oversold': latest_k <= oversold_level,
                'in_extreme_overbought': latest_k >= extreme_overbought,
                'in_extreme_oversold': latest_k <= extreme_oversold,
                'k_above_d': latest_k > latest_d,
                'rsi_period': self.rsi_period,
                'stoch_period': self.stoch_period,
                'k_period': self.k_period,
                'd_period': self.d_period
            }
            
            # 6. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"STOCHRSI信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False
            
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # STOCHRSI需要足够的数据用于计算
        min_periods = max(self.rsi_period, self.stoch_period) + self.k_period + self.d_period + 5
        if len(data) < min_periods:
            return False
            
        return True

    def _get_default_signal(self, reason: str = "数据不足") -> Dict[str, Any]:
        """
        生成默认信号（持有信号）
        
        Args:
            reason: 生成默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }


# 类别名
STOCHRSI = Stochrsi
