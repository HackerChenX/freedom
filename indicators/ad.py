#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Any

from utils.container import container
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.signal_utils import crossover, crossunder
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from enums.signal_strength import Signal_strength
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class AD(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    AD (Accumulation/Distribution Line) 累积分布线指标

    生产级实现:真实数学计算 + 完整功能 + 架构兼容

    核心算法: AD Line = Σ[((Close-Low)-(High-Close))/(High-Low) * Volume]
    """

    def __init__(self, **kwargs):
        """初始化AD指标"""
        super().__init__(name="AD", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        self.description = "累积分布线指标"
        self.indicator_type = "AD"
        self.REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
        self._result = None

        # 设置默认参数
        self._default_parameters = {}
        self.set_parameters_Indicator_Base_Indicator(**kwargs)

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算AD指标 - 公共接口"""
        result = self._calculate_baseindicator(data, **kwargs)
        self._result = result
        return result

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """核心计算逻辑,实现抽象方法"""
        return self._calculate_ad_production(data, **kwargs)

    def _calculate_ad_production(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生产级AD指标计算

        实现真实的累积分布线算法:
        AD Line = Σ[((Close-Low)-(High-Close))/(High-Low) * Volume]
        """
        df = data.copy()

        # 确保数据有足够长度
        if len(df) < 1:
            logger.warning("数据长度不足,无法计算AD指标")
            df["AD"] = np.nan
            return df

        # 计算Money Flow Multiplier
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        high_low_diff = df["high"] - df["low"]

        # 避免除零错误
        high_low_diff = high_low_diff.replace(0, np.nan)

        # 计算Money Flow Multiplier
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low_diff

        # 处理除零情况:当高低价相等时,根据收盘价与前一交易日的关系确定MFM
        for i in range(len(df)):
            if pd.isna(mfm.iloc[i]) or np.isinf(mfm.iloc[i]):
                if i > 0:
                    # 如果收盘价高于前一交易日,MFM = 1
                    if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                        mfm.iloc[i] = 1.0
                    # 如果收盘价低于前一交易日,MFM = -1
                    elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                        mfm.iloc[i] = -1.0
                    # 如果收盘价等于前一交易日,MFM = 0
                    else:
                        mfm.iloc[i] = 0.0
                else:
                    mfm.iloc[i] = 0.0

        # 计算Money Flow Volume
        # MFV = MFM * Volume
        mfv = mfm * df["volume"]

        # 计算累积分布线 (AD Line)
        # AD = 前期AD + 当期MFV
        ad_line = mfv.cumsum()

        df["AD"] = ad_line

        # 计算AD的移动平均线用于生成信号
        df["AD_MA_20"] = df["AD"].rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

        # 计算AD变化率
        df["AD_CHANGE"] = df["AD"].pct_change() * 100

        # 计算AD趋势
        df["AD_TREND"] = np.where(df["AD"] > df["AD_MA_20"], 1, np.where(df["AD"] < df["AD_MA_20"], -1, 0))

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """实现抽象方法"""
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 基于AD的基本评分
        ad = self._result["AD"]
        ad_ma = self._result["AD_MA_20"]

        scores = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 趋势评分
        uptrend = ad > ad_ma
        downtrend = ad < ad_ma

        scores = np.where(uptrend, 65, scores)  # TODO: 将魔法数字提取到配置中
        scores = np.where(downtrend, 35, scores)  # TODO: 将魔法数字提取到配置中

        return pd.Series(scores, index=data.index)

    def calculate_confidence_Indicator_Base_Indicator(
        self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]
    ) -> float:
        """实现抽象方法"""
        return 0.7  # 基础置信度  # TODO: 将魔法数字提取到配置中

    def get_patterns_Indicator_Base_Indicator(
        self, data: pd.DataFrame, **kwargs
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """实现抽象方法"""
        if not self.has_result():
            self.calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        if self._result is not None and "AD" in self._result.columns:
            ad = self._result["AD"]
            ad_ma = self._result["AD_MA_20"]

            # 基本形态
            patterns["AD_UPTREND"] = ad > ad_ma
            patterns["AD_DOWNTREND"] = ad < ad_ma
            patterns["AD_GOLDEN_CROSS"] = (ad > ad_ma) & (ad.shift(1) <= ad_ma.shift(1))
            patterns["AD_DEATH_CROSS"] = (ad < ad_ma) & (ad.shift(1) >= ad_ma.shift(1))

        return patterns

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """实现抽象方法"""
        pass  # AD指标通常不需要参数

    @property
    def minimum_periods(self) -> int:
        """
        AD指标所需的最少数据周期数

        计算逻辑:使用默认值

        Returns:
            int: 最少需要的数据周期数
        """
        return 30  # TODO: 将魔法数字提取到配置中

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 如果已有结果返回True,否则返回False
        """
        return self._result is not None and not self._result.empty

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于AD (Accumulation/Distribution Line) 指标数值生成最新的交易信号
        
        AD交易信号逻辑：
        - AD上升 + 价格上升：累积买入信号
        - AD下降 + 价格下降：分布卖出信号
        - AD与均线的突破：趋势确认信号
        - AD与价格背离：反转信号
        - AD趋势强度：信号强度判断
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 确保已计算指标
            if not self.has_result():
                result = self.calculate(data)
                if result is not None:
                    self._result = result
            
            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("AD计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取AD相关值
            if len(self._result) < 2:
                return self._get_default_signal("AD数据不足")
                
            # 检查必要的列是否存在
            if 'AD' not in self._result.columns:
                return self._get_default_signal("AD结果列不存在")
                
            ad_values = self._result['AD'].dropna()
            if len(ad_values) < 2:
                return self._get_default_signal("AD有效数据不足")
                
            latest_ad = ad_values.iloc[-1]
            prev_ad = ad_values.iloc[-2]
            
            # 检查是否有NaN值
            if pd.isna(latest_ad) or pd.isna(prev_ad):
                return self._get_default_signal("AD数据包含NaN值")
            
            # 5. 获取价格数据
            close_values = data['close'].iloc[-2:]
            if len(close_values) < 2:
                return self._get_default_signal("价格数据不足")
                
            latest_price = close_values.iloc[-1]
            prev_price = close_values.iloc[-2]
            
            # 6. AD信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 计算AD和价格变化
            ad_change = latest_ad - prev_ad
            price_change = latest_price - prev_price
            ad_change_pct = (ad_change / abs(prev_ad)) * 100 if prev_ad != 0 else 0
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # 获取AD均线信息（如果存在）
            ad_ma = None
            if 'AD_MA' in self._result.columns:
                ad_ma_values = self._result['AD_MA'].dropna()
                if len(ad_ma_values) > 0:
                    ad_ma = ad_ma_values.iloc[-1]
            
            # AD-价格累积分布分析（核心AD信号）
            if ad_change > 0 and price_change > 0:
                # AD和价格同时上升 - 累积买入信号
                signal_type = "buy"
                accumulation_strength = min(abs(ad_change_pct) + abs(price_change_pct), 100) / 100
                strength = min(0.9, 0.7 + accumulation_strength * 0.2)
                confidence = 0.85
                reason = f"AD累积买入({ad_change_pct:.2f}%, {price_change_pct:.2f}%)，资金流入确认"
                
            elif ad_change < 0 and price_change < 0:
                # AD和价格同时下降 - 分布卖出信号
                signal_type = "sell"
                distribution_strength = min(abs(ad_change_pct) + abs(price_change_pct), 100) / 100
                strength = min(0.9, 0.7 + distribution_strength * 0.2)
                confidence = 0.85
                reason = f"AD分布卖出({ad_change_pct:.2f}%, {price_change_pct:.2f}%)，资金流出确认"
                
            elif ad_change > 0 and price_change < 0:
                # AD上升但价格下降 - 正背离，潜在底部
                signal_type = "buy"
                divergence_strength = min(abs(ad_change_pct) + abs(price_change_pct), 80) / 100
                strength = min(0.8, 0.6 + divergence_strength * 0.2)
                confidence = 0.75
                reason = f"AD正背离({ad_change_pct:.2f}% vs {price_change_pct:.2f}%)，资金暗中积累"
                
            elif ad_change < 0 and price_change > 0:
                # AD下降但价格上升 - 负背离，潜在顶部
                signal_type = "sell"
                divergence_strength = min(abs(ad_change_pct) + abs(price_change_pct), 80) / 100
                strength = min(0.8, 0.6 + divergence_strength * 0.2)
                confidence = 0.75
                reason = f"AD负背离({ad_change_pct:.2f}% vs {price_change_pct:.2f}%)，资金暗中流出"
            
            # AD均线突破信号
            elif ad_ma is not None:
                ad_ma_prev = None
                if len(self._result) >= 2 and 'AD_MA' in self._result.columns:
                    ad_ma_prev_values = self._result['AD_MA'].iloc[-2:-1]
                    if len(ad_ma_prev_values) > 0 and not pd.isna(ad_ma_prev_values.iloc[0]):
                        ad_ma_prev = ad_ma_prev_values.iloc[0]
                
                if ad_ma_prev is not None:
                    # AD突破均线向上
                    if prev_ad <= ad_ma_prev and latest_ad > ad_ma:
                        signal_type = "buy"
                        breakout_strength = min(abs(latest_ad - ad_ma) / abs(ad_ma), 0.2) if ad_ma != 0 else 0
                        strength = min(0.75, 0.6 + breakout_strength * 5)
                        confidence = 0.75
                        reason = f"AD突破均线向上({latest_ad:.0f} > {ad_ma:.0f})，累积趋势确认"
                        
                    # AD跌破均线向下
                    elif prev_ad >= ad_ma_prev and latest_ad < ad_ma:
                        signal_type = "sell"
                        breakdown_strength = min(abs(ad_ma - latest_ad) / abs(ad_ma), 0.2) if ad_ma != 0 else 0
                        strength = min(0.75, 0.6 + breakdown_strength * 5)
                        confidence = 0.75
                        reason = f"AD跌破均线向下({latest_ad:.0f} < {ad_ma:.0f})，分布趋势确认"
            
            # AD趋势强度分析（基于20期趋势）
            if len(ad_values) >= 20:
                ad_trend_start = ad_values.iloc[-20]
                ad_trend_slope = (latest_ad - ad_trend_start) / 20
                
                if abs(ad_trend_slope) > 0:
                    # 计算趋势强度标准差
                    try:
                        ad_recent = ad_values.iloc[-20:]
                        ad_volatility = ad_recent.diff().std()
                        trend_strength_ratio = abs(ad_trend_slope) / ad_volatility if ad_volatility > 0 else 0
                        
                        if trend_strength_ratio > 2.0:  # 强趋势
                            if ad_trend_slope > 0:
                                if signal_type == "buy":
                                    strength = min(strength + 0.15, 1.0)
                                    confidence = min(confidence + 0.1, 1.0)
                                    reason += "，强势累积趋势"
                                elif signal_type == "hold":
                                    signal_type = "buy"
                                    strength = 0.65
                                    confidence = 0.7
                                    reason = f"AD强势上升趋势，资金持续流入"
                            else:
                                if signal_type == "sell":
                                    strength = min(strength + 0.15, 1.0)
                                    confidence = min(confidence + 0.1, 1.0)
                                    reason += "，强势分布趋势"
                                elif signal_type == "hold":
                                    signal_type = "sell"
                                    strength = 0.65
                                    confidence = 0.7
                                    reason = f"AD强势下降趋势，资金持续流出"
                                    
                    except Exception:
                        pass  # 忽略趋势强度计算错误
            
            # 计算AD特有的元数据
            ad_trend = "上升" if ad_change > 0 else "下降" if ad_change < 0 else "平稳"
            price_trend = "上升" if price_change > 0 else "下降" if price_change < 0 else "平稳"
            accumulation_distribution = "累积" if (ad_change > 0) == (price_change > 0) else "背离" if ad_change != 0 and price_change != 0 else "中性"
            
            metadata = {
                'ad_value': latest_ad,
                'ad_previous': prev_ad,
                'ad_change': ad_change,
                'ad_change_pct': ad_change_pct,
                'ad_trend': ad_trend,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'price_trend': price_trend,
                'accumulation_distribution': accumulation_distribution,
                'ad_ma': ad_ma,
                'above_ma': latest_ad > ad_ma if ad_ma is not None else None,
                'indicator_type': 'volume_price_relationship',
                'calculation_method': 'CLV_based'  # Close Location Value based
            }
            
            # 7. 标准化输出
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
            logger.warning(f"AD信号生成失败: {e}")
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
            
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # AD需要足够的数据用于计算
        min_periods = max(getattr(self, 'period', 14), 2)
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


class AccumulationDistribution(AD):
    """
    累积/派发线指标 (Accumulation/Distribution Line)

    AD指标将每日的成交量按照收盘价与最高最低价的关系进行加权,
    以反映成交量与价格的关系,评估资金流入流出情况.

    该指标常用于判断价格趋势的强弱,特别是通过量价背离来预测价格可能的反转.
    """

    def __init__(self, name: str = "AD", description: str = "累积/派发线指标"):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """初始化AD指标"""
        super().__init__(name, description)
        self.indicator_type = "AD"
        self.REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
        self._result = None

    def set_parameters_Ad_Ad_Ad_ad(self, **kwargs):
        """设置指标参数"""
        pass

    def register_patterns_Ad(self):
        """
        注册AD指标的形态到全局形态注册表
        """
        # 注册AD金叉死叉形态
        self.register_pattern_to_registry(
            pattern_id="AD_GOLDEN_CROSS",
            display_name="AD金叉",
            description="AD上穿其均线,表明买盘资金增加",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="AD_DEATH_CROSS",
            display_name="AD死叉",
            description="AD下穿其均线,表明卖盘资金增加",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册AD背离形态
        self.register_pattern_to_registry(
            pattern_id="AD_PRICE_DIVERGENCE_TOP",
            display_name="AD与价格顶背离",
            description="价格创新高但AD未创新高,表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="AD_PRICE_DIVERGENCE_BOTTOM",
            display_name="AD与价格底背离",
            description="价格创新低但AD未创新低,表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        # 注册AD趋势形态
        self.register_pattern_to_registry(
            pattern_id="AD_UPTREND",
            display_name="AD上升趋势",
            description="AD持续上升,表明买盘持续涌入",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="AD_DOWNTREND",
            display_name="AD下降趋势",
            description="AD持续下降,表明卖盘持续涌出",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

        # 注册AD快速变化形态
        self.register_pattern_to_registry(
            pattern_id="AD_RAPID_INCREASE",
            display_name="AD快速上涨",
            description="AD快速上涨,表明买盘资金快速涌入",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE",
        )

        self.register_pattern_to_registry(
            pattern_id="AD_RAPID_DECREASE",
            display_name="AD快速下跌",
            description="AD快速下跌,表明卖盘资金快速涌出",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE",
        )

    def calculate_confidence_Ad(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算AD指标的置信度.
        """
        return 0.5  # TODO: 将魔法数字提取到配置中

    def _calculate_ad(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算AD指标

        Args:
            df: 包含high, low, close, volume列的Data_frame

        Returns:
            包含AD和AD_MA列的Data_frame
        """
        # 检查必要列是否存在
        required_columns = ["high", "low", "close", "volume"]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"输入数据缺少必要的列: {column}")

        # 创建副本以避免修改原始数据
        df_copy = df.copy()

        # 计算价格位置
        price_position = ((df_copy["close"] - df_copy["low"]) - (df_copy["high"] - df_copy["close"])) / (
            df_copy["high"] - df_copy["low"]
        )

        # 处理分母为0的情况
        price_position = price_position.replace([np.inf, -np.inf], 0)

        # 计算资金流乘数
        money_flow_multiplier = price_position * df_copy["volume"]

        # 计算AD指标
        df_copy["AD"] = money_flow_multiplier.cumsum()

        # 计算AD的移动平均
        df_copy["AD_MA"] = df_copy["AD"].rolling(window=14).mean()  # TODO: 将魔法数字提取到配置中

        # 保存结果
        self._result = df_copy

        # 添加形态识别和信号生成
        df_copy = self.add_pattern_detection(df_copy)
        df_copy = self.add_signal_generation(df_copy)

        return df_copy

    def get_patterns_Ad(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取AD指标的所有形态信息

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        if not self.has_result_Ad():
            self.calculate(data)

        result = []

        # 如果没有计算结果,先计算
        if not self.has_result_Ad():
            self.calculate(data)

        # 获取AD指标数据
        ad_data = self._result["AD"]
        ad_ma_data = self._result["AD_MA"]

        # 检测金叉形态
        if len(ad_data) >= 2:
            cross_over = crossover(ad_data.iloc[-2:], ad_ma_data.iloc[-2:])
            if cross_over.any():
                pattern_data = {
                    "pattern_id": "AD_GOLDEN_CROSS",
                    "display_name": "AD金叉",
                    "indicator_id": self.name,
                    "strength": SignalStrength.STRONG.value,
                    "duration": 1,
                    "details": {"ad_value": float(ad_data.iloc[-1]), "ad_ma_value": float(ad_ma_data.iloc[-1])},
                }
                result.append(pattern_data)

        # 检测死叉形态
        if len(ad_data) >= 2:
            cross_under = crossunder(ad_data.iloc[-2:], ad_ma_data.iloc[-2:])
            if cross_under.any():
                pattern_data = {
                    "pattern_id": "AD_DEATH_CROSS",
                    "display_name": "AD死叉",
                    "indicator_id": self.name,
                    "strength": SignalStrength.STRONG_NEGATIVE.value,
                    "duration": 1,
                    "details": {"ad_value": float(ad_data.iloc[-1]), "ad_ma_value": float(ad_ma_data.iloc[-1])},
                }
                result.append(pattern_data)

        # 检测量价背离
        if len(data) >= 30 and "close" in data.columns:  # TODO: 将魔法数字提取到配置中
            price_trend = data["close"].pct_change(5, fill_method=None).iloc[-1]  # TODO: 将魔法数字提取到配置中
            ad_trend = ad_data.pct_change(5, fill_method=None).iloc[-1]  # TODO: 将魔法数字提取到配置中

            # 价格上涨但AD下降(顶背离)
            if price_trend > 0.02 and ad_trend < -0.02:
                pattern_data = {
                    "pattern_id": "AD_PRICE_DIVERGENCE_TOP",
                    "display_name": "AD与价格顶背离",
                    "indicator_id": self.name,
                    "strength": SignalStrength.VERY_STRONG_NEGATIVE.value,
                    "duration": 3,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    "details": {"price_trend": float(price_trend), "ad_trend": float(ad_trend)},
                }
                result.append(pattern_data)
            # 价格下跌但AD上升(底背离)
            elif price_trend < -0.02 and ad_trend > 0.02:
                pattern_data = {
                    "pattern_id": "AD_PRICE_DIVERGENCE_BOTTOM",
                    "display_name": "AD与价格底背离",
                    "indicator_id": self.name,
                    "strength": SignalStrength.VERY_STRONG.value,
                    "duration": 3,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    "details": {"price_trend": float(price_trend), "ad_trend": float(ad_trend)},
                }
                result.append(pattern_data)

        # 检测量能变化
        if len(ad_data) >= 2:
            ad_change = ad_data.pct_change(fill_method=None).iloc[-1]

            # AD快速上涨
            if ad_change > 0.05:  # TODO: 将魔法数字提取到配置中
                pattern_data = {
                    "pattern_id": "AD_RAPID_INCREASE",
                    "display_name": "AD快速上涨",
                    "indicator_id": self.name,
                    "strength": SignalStrength.MODERATE.value,
                    "duration": 2,
                    "details": {"change_rate": float(ad_change)},
                }
                result.append(pattern_data)
            # AD快速下跌
            elif ad_change < -0.05:  # TODO: 将魔法数字提取到配置中
                pattern_data = {
                    "pattern_id": "AD_RAPID_DECREASE",
                    "display_name": "AD快速下跌",
                    "indicator_id": self.name,
                    "strength": SignalStrength.SELL.value,
                    "duration": 2,
                    "details": {"change_rate": float(ad_change)},
                }
                result.append(pattern_data)

        return pd.DataFrame(result)

    def calculate_score_Ad(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算AD指标评分(0-100分制)

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            float: 综合评分(0-100)
        """
        raw_score = self.calculate_raw_score_Ad(data, **kwargs)

        if raw_score.empty:
            return 50.0  # 默认中性评分  # TODO: 将魔法数字提取到配置中

        last_score = raw_score.iloc[-1]

        # 应用市场环境调整
        market_env = kwargs.get("market_env", self._market_environment)
        adjusted_score = self._apply_market_environment_adjustment(market_env, last_score)

        # 计算置信度
        patterns = self.get_patterns_Ad(data)
        confidence = self.calculate_confidence_Ad(pd.Series([adjusted_score]), patterns, {})

        # 返回最终评分
        return float(np.clip(adjusted_score * confidence, 0, 100))

    def _apply_market_environment_adjustment(self, market_env, score: float) -> float:
        """
        根据市场环境调整评分

        Args:
            market_env: 市场环境
            score: 原始评分

        Returns:
            float: 调整后的评分
        """
        from indicators.base_indicator import Market_environment

        if market_env == Market_environment.BULL_MARKET:
            # 牛市中增强多头信号,弱化空头信号
            if score > 50:  # TODO: 将魔法数字提取到配置中
                return score + (score - 50) * 0.2  # 多头信号增强  # TODO: 将魔法数字提取到配置中
            else:
                return score + (score - 50) * 0.1  # 空头信号减弱  # TODO: 将魔法数字提取到配置中
        elif market_env == Market_environment.BEAR_MARKET:
            # 熊市中增强空头信号,弱化多头信号
            if score < 50:  # TODO: 将魔法数字提取到配置中
                return score - (50 - score) * 0.2  # 空头信号增强  # TODO: 将魔法数字提取到配置中
            else:
                return score - (score - 50) * 0.1  # 多头信号减弱  # TODO: 将魔法数字提取到配置中
        elif market_env == Market_environment.VOLATILE_MARKET:
            # 高波动市场需要更强的信号
            if score > 60 or score < 40:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                return (
                    score + (score - 50) * 0.15
                )  # 极端信号更极端  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            else:
                return (
                    50 + (score - 50) * 0.8
                )  # 中性信号更中性  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        else:
            # 震荡市场,保持原评分
            return score

    def has_result_Ad(self) -> bool:
        """检查是否已计算结果"""
        return self._result is not None and not self._result.empty

    def generate_signals_Ad(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成AD指标的标准化交易信号

        Args:
            data: 输入数据,包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            pd.DataFrame: 信号结果Data_frame,包含标准化信号
        """
        # 确保已计算AD指标
        if not self.has_result_Ad():
            self.calculate(data)

        # 获取AD相关值
        ad = self._result["AD"]
        ad_ma = self._result["AD_MA"]

        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals["buy_signal"] = False
        signals["sell_signal"] = False
        signals["neutral_signal"] = True  # 默认为中性信号
        signals["trend"] = 0  # 0表示中性
        signals["score"] = 50.0  # 默认评分50分  # TODO: 将魔法数字提取到配置中
        signals["signal_type"] = None
        signals["signal_desc"] = None
        signals["confidence"] = 50.0  # TODO: 将魔法数字提取到配置中
        signals["risk_level"] = "中"
        signals["position_size"] = 0.0
        signals["stop_loss"] = None
        signals["market_env"] = "中性"
        signals["volume_confirmation"] = False

        # 计算ATR用于止损设置
        try:
            from indicators.base_indicator import BaseIndicator

            atr_indicator = complete_registry.create_indicator("ATR")
            if atr_indicator:
                atr_data = atr_indicator.calculate(data)
                atr_values = atr_data["atr"]
            else:
                logger.warning("无法创建ATR指标")
                atr_values = None
        except Exception as e:
            logger.warning(f"计算ATR失败: {e}")
            atr_values = pd.Series(0, index=data.index)

        # 1. AD上穿其均线,买入信号
        ad_crossover_ma = crossover(ad, ad_ma)
        signals.loc[ad_crossover_ma, "buy_signal"] = True
        signals.loc[ad_crossover_ma, "neutral_signal"] = False
        signals.loc[ad_crossover_ma, "trend"] = 1
        signals.loc[ad_crossover_ma, "signal_type"] = "AD金叉"
        signals.loc[ad_crossover_ma, "signal_desc"] = "AD上穿其均线,表明买盘资金增加"
        signals.loc[ad_crossover_ma, "confidence"] = 65.0  # TODO: 将魔法数字提取到配置中
        signals.loc[ad_crossover_ma, "position_size"] = 0.3  # TODO: 将魔法数字提取到配置中
        signals.loc[ad_crossover_ma, "risk_level"] = "中"

        # 2. AD下穿其均线,卖出信号
        ad_crossunder_ma = crossunder(ad, ad_ma)
        signals.loc[ad_crossunder_ma, "sell_signal"] = True
        signals.loc[ad_crossunder_ma, "neutral_signal"] = False
        signals.loc[ad_crossunder_ma, "trend"] = -1
        signals.loc[ad_crossunder_ma, "signal_type"] = "AD死叉"
        signals.loc[ad_crossunder_ma, "signal_desc"] = "AD下穿其均线,表明卖盘资金增加"
        signals.loc[ad_crossunder_ma, "confidence"] = 65.0  # TODO: 将魔法数字提取到配置中
        signals.loc[ad_crossunder_ma, "position_size"] = 0.3  # TODO: 将魔法数字提取到配置中
        signals.loc[ad_crossunder_ma, "risk_level"] = "中"

        # 3. 正背离信号(价格创新低,但AD未创新低)  # TODO: 将魔法数字提取到配置中
        if "close" in data.columns:
            close = data["close"]

            # 获取价格和AD的局部低点
            lows_close = pd.Series(np.nan, index=close.index)
            lows_ad = pd.Series(np.nan, index=ad.index)

            # 简单的局部低点检测:如果一个点比前后N个点都低,则为局部低点
            window = 5  # TODO: 将魔法数字提取到配置中
            for i in range(window, len(close) - window):
                if close.iloc[i] == close.iloc[i - window : i + window + 1].min():
                    lows_close.iloc[i] = close.iloc[i]
                if ad.iloc[i] == ad.iloc[i - window : i + window + 1].min():
                    lows_ad.iloc[i] = ad.iloc[i]

            # 检测正背离:价格创新低但AD未创新低
            for i in range(window * 2, len(close)):
                if pd.notna(lows_close.iloc[i]) and pd.notna(
                    lows_close.iloc[i - window * 2 : i - window].dropna().min()
                ):
                    # 价格创新低
                    if lows_close.iloc[i] < lows_close.iloc[i - window * 2 : i - window].dropna().min():
                        # 查找相应时期的AD低点
                        recent_low_ad = (
                            lows_ad.iloc[i - window // 2 : i + window // 2].dropna().min()
                            if not lows_ad.iloc[i - window // 2 : i + window // 2].dropna().empty
                            else np.nan
                        )
                        prev_low_ad = (
                            lows_ad.iloc[i - window * 2 - window // 2 : i - window + window // 2].dropna().min()
                            if not lows_ad.iloc[i - window * 2 - window // 2 : i - window + window // 2].dropna().empty
                            else np.nan
                        )

                        # AD未创新低(正背离)
                        if pd.notna(recent_low_ad) and pd.notna(prev_low_ad) and recent_low_ad > prev_low_ad:
                            # 只有在没有其他信号时才设置背离信号
                            if not signals.iloc[i]["buy_signal"] and not signals.iloc[i]["sell_signal"]:
                                signals.iloc[i, signals.columns.get_loc("buy_signal")] = True
                                signals.iloc[i, signals.columns.get_loc("neutral_signal")] = False
                                signals.iloc[i, signals.columns.get_loc("trend")] = 1
                                signals.iloc[i, signals.columns.get_loc("signal_type")] = "AD正背离"
                                signals.iloc[i, signals.columns.get_loc("signal_desc")] = (
                                    "价格创新低但AD未创新低,表明下跌动能减弱"
                                )
                                signals.iloc[i, signals.columns.get_loc("confidence")] = (
                                    75.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                                )
                                signals.iloc[i, signals.columns.get_loc("position_size")] = (
                                    0.4  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                                )
                                signals.iloc[i, signals.columns.get_loc("risk_level")] = "低"

            # 4. 负背离信号(价格创新高,但AD未创新高)  # TODO: 将魔法数字提取到配置中
            highs_close = pd.Series(np.nan, index=close.index)
            highs_ad = pd.Series(np.nan, index=ad.index)

            # 简单的局部高点检测
            for i in range(window, len(close) - window):
                if close.iloc[i] == close.iloc[i - window : i + window + 1].max():
                    highs_close.iloc[i] = close.iloc[i]
                if ad.iloc[i] == ad.iloc[i - window : i + window + 1].max():
                    highs_ad.iloc[i] = ad.iloc[i]

            # 检测负背离:价格创新高但AD未创新高
            for i in range(window * 2, len(close)):
                if pd.notna(highs_close.iloc[i]) and pd.notna(
                    highs_close.iloc[i - window * 2 : i - window].dropna().max()
                ):
                    # 价格创新高
                    if highs_close.iloc[i] > highs_close.iloc[i - window * 2 : i - window].dropna().max():
                        # 查找相应时期的AD高点
                        recent_high_ad = (
                            highs_ad.iloc[i - window // 2 : i + window // 2].dropna().max()
                            if not highs_ad.iloc[i - window // 2 : i + window // 2].dropna().empty
                            else np.nan
                        )
                        prev_high_ad = (
                            highs_ad.iloc[i - window * 2 - window // 2 : i - window + window // 2].dropna().max()
                            if not highs_ad.iloc[i - window * 2 - window // 2 : i - window + window // 2].dropna().empty
                            else np.nan
                        )

                        # AD未创新高(负背离)
                        if pd.notna(recent_high_ad) and pd.notna(prev_high_ad) and recent_high_ad < prev_high_ad:
                            # 只有在没有其他信号时才设置背离信号
                            if not signals.iloc[i]["buy_signal"] and not signals.iloc[i]["sell_signal"]:
                                signals.iloc[i, signals.columns.get_loc("sell_signal")] = True
                                signals.iloc[i, signals.columns.get_loc("neutral_signal")] = False
                                signals.iloc[i, signals.columns.get_loc("trend")] = -1
                                signals.iloc[i, signals.columns.get_loc("signal_type")] = "AD负背离"
                                signals.iloc[i, signals.columns.get_loc("signal_desc")] = (
                                    "价格创新高但AD未创新高,表明上涨动能减弱"
                                )
                                signals.iloc[i, signals.columns.get_loc("confidence")] = (
                                    75.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                                )
                                signals.iloc[i, signals.columns.get_loc("position_size")] = (
                                    0.4  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                                )
                                signals.iloc[i, signals.columns.get_loc("risk_level")] = "低"

        # 5. AD趋势  # TODO: 将魔法数字提取到配置中
        ad_trend = pd.Series(np.nan, index=ad.index)
        for i in range(20, len(ad)):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 计算20日趋势
            ad_slope = (
                ad.iloc[i] - ad.iloc[i - 20]
            ) / 20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            if ad_slope > 0:
                # AD上升趋势
                signals.iloc[i, signals.columns.get_loc("trend")] = 1
                if not signals.iloc[i]["buy_signal"] and not signals.iloc[i]["sell_signal"]:
                    signals.iloc[i, signals.columns.get_loc("buy_signal")] = True
                    signals.iloc[i, signals.columns.get_loc("neutral_signal")] = False
                    signals.iloc[i, signals.columns.get_loc("signal_type")] = "AD上升趋势"
                    signals.iloc[i, signals.columns.get_loc("signal_desc")] = "AD持续上升,表明买盘持续涌入"
                    signals.iloc[i, signals.columns.get_loc("confidence")] = (
                        60.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    )
                    signals.iloc[i, signals.columns.get_loc("position_size")] = 0.2
                    signals.iloc[i, signals.columns.get_loc("risk_level")] = "中"
            elif ad_slope < 0:
                # AD下降趋势
                signals.iloc[i, signals.columns.get_loc("trend")] = -1
                if not signals.iloc[i]["buy_signal"] and not signals.iloc[i]["sell_signal"]:
                    signals.iloc[i, signals.columns.get_loc("sell_signal")] = True
                    signals.iloc[i, signals.columns.get_loc("neutral_signal")] = False
                    signals.iloc[i, signals.columns.get_loc("signal_type")] = "AD下降趋势"
                    signals.iloc[i, signals.columns.get_loc("signal_desc")] = "AD持续下降,表明卖盘持续涌出"
                    signals.iloc[i, signals.columns.get_loc("confidence")] = (
                        60.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    )
                    signals.iloc[i, signals.columns.get_loc("position_size")] = 0.2
                    signals.iloc[i, signals.columns.get_loc("risk_level")] = "中"

        # 6. 计算评分  # TODO: 将魔法数字提取到配置中
        for i in range(len(signals)):
            if i > 0:  # 跳过第一个数据点
                # 基础分数是50
                score = 50.0  # TODO: 将魔法数字提取到配置中

                # 根据AD趋势评分
                if i < len(ad) and i >= 20:  # TODO: 将魔法数字提取到配置中
                    ad_slope = (
                        ad.iloc[i] - ad.iloc[i - 20]
                    ) / 20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                    # 根据AD斜率调整分数
                    slope_score = (
                        ad_slope * 1000
                    )  # 缩放斜率以获得合适的分数调整  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    score += np.clip(
                        slope_score, -25, 25
                    )  # 限制斜率对分数的影响  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

                # AD与其均线的关系
                if i < len(ad) and i < len(ad_ma):
                    if ad.iloc[i] > ad_ma.iloc[i]:
                        score += 10  # AD在均线上方加10分
                    else:
                        score -= 10  # AD在均线下方减10分

                # 根据信号类型额外调整分数
                if signals.iloc[i]["signal_type"] == "AD正背离":
                    score += 15  # TODO: 将魔法数字提取到配置中
                elif signals.iloc[i]["signal_type"] == "AD负背离":
                    score -= 15  # TODO: 将魔法数字提取到配置中

                # 限制评分范围在0-100之间
                signals.iloc[i, signals.columns.get_loc("score")] = max(0, min(100, score))

        # 设置止损价格
        if "low" in data.columns and "high" in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals["buy_signal"]].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx in data.index and idx > data.index[10]:  # 确保有足够的历史数据
                        pos = data.index.get_loc(idx)
                        if pos >= 10:
                            lookback = 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            recent_low = data.iloc[pos - lookback : pos]["low"].min()
                            atr_val = atr_values.iloc[pos] if pos < len(atr_values) else 0
                            signals.loc[idx, "stop_loss"] = recent_low - atr_val

            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals["sell_signal"]].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx in data.index and idx > data.index[10]:  # 确保有足够的历史数据
                        pos = data.index.get_loc(idx)
                        if pos >= 10:
                            lookback = 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            recent_high = data.iloc[pos - lookback : pos]["high"].max()
                            atr_val = atr_values.iloc[pos] if pos < len(atr_values) else 0
                            signals.loc[idx, "stop_loss"] = recent_high + atr_val

        # 根据AD趋势判断市场环境
        signals["market_env"] = "中性"  # 默认中性市场

        # 计算20日AD趋势
        for i in range(20, len(ad)):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            ad_slope = (
                ad.iloc[i] - ad.iloc[i - 20]
            ) / 20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            if ad_slope > 0:
                # 上升趋势强度判断
                if ad_slope > ad.iloc[i - 20 : i].diff().std() * 2:  # TODO: 将魔法数字提取到配置中
                    signals.iloc[i, signals.columns.get_loc("market_env")] = "强势"
                else:
                    signals.iloc[i, signals.columns.get_loc("market_env")] = "中性偏强"
            elif ad_slope < 0:
                # 下降趋势强度判断
                if abs(ad_slope) > ad.iloc[i - 20 : i].diff().std() * 2:  # TODO: 将魔法数字提取到配置中
                    signals.iloc[i, signals.columns.get_loc("market_env")] = "弱势"
                else:
                    signals.iloc[i, signals.columns.get_loc("market_env")] = "中性偏弱"
            else:
                signals.iloc[i, signals.columns.get_loc("market_env")] = "震荡"

        # 设置成交量确认
        if "volume" in data.columns:
            # 如果有成交量数据,检查成交量是否支持当前信号
            vol = data["volume"]
            vol_avg = vol.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中

            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5  # TODO: 将魔法数字提取到配置中

            # 买入信号且成交量放大,确认信号
            signals.loc[signals["buy_signal"] & vol_increase, "volume_confirmation"] = True

            # 卖出信号且成交量放大,确认信号
            signals.loc[signals["sell_signal"] & vol_increase, "volume_confirmation"] = True

        return signals

    def calculate_raw_score_Ad(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算AD原始评分

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列(0-100分)
        """
        # 确保已计算AD指标
        if not self.has_result_Ad():
            self.calculate(data)

        if self._result is None:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        # 获取AD相关值
        ad = self._result["AD"]
        ad_ma = self._result["AD_MA"]

        # 初始基础分50分
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中

        # 基于AD趋势的评分
        for i in range(20, len(score)):  # TODO: 将魔法数字提取到配置中
            if i >= len(ad):
                continue

            # 计算20日AD趋势
            ad_slope = (
                ad.iloc[i] - ad.iloc[i - 20]
            ) / 20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # 根据AD斜率调整分数
            slope_score = (
                ad_slope * 1000
            )  # 缩放斜率以获得合适的分数调整  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score.iloc[i] += np.clip(
                slope_score, -25, 25
            )  # 限制斜率对分数的影响  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            # AD与其均线的关系
            if i < len(ad_ma):
                if ad.iloc[i] > ad_ma.iloc[i]:
                    score.iloc[i] += 10  # AD在均线上方加10分
                else:
                    score.iloc[i] -= 10  # AD在均线下方减10分

        # 限制评分范围在0-100之间
        return np.clip(score, 0, 100)

    def identify_patterns_Ad(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别AD技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []

        # 确保已计算AD指标
        if not self.has_result_Ad():
            self.calculate(data)

        if self._result is None:
            return patterns

        # 获取AD相关值
        ad = self._result["AD"]
        ad_ma = self._result["AD_MA"]

        # 最后一个有效的索引位置
        last_valid_idx = -1
        while last_valid_idx >= -len(ad) and pd.isna(ad.iloc[last_valid_idx]):
            last_valid_idx -= 1

        if last_valid_idx < -len(ad):
            return patterns

        # 1. AD与均线交叉形态
        if (
            last_valid_idx - 1 >= -len(ad)
            and ad.iloc[last_valid_idx - 1] < ad_ma.iloc[last_valid_idx - 1]
            and ad.iloc[last_valid_idx] > ad_ma.iloc[last_valid_idx]
        ):
            patterns.append("AD金叉(AD上穿均线)")

        if (
            last_valid_idx - 1 >= -len(ad)
            and ad.iloc[last_valid_idx - 1] > ad_ma.iloc[last_valid_idx - 1]
            and ad.iloc[last_valid_idx] < ad_ma.iloc[last_valid_idx]
        ):
            patterns.append("AD死叉(AD下穿均线)")

        # 2. AD趋势形态
        if last_valid_idx >= 20:  # TODO: 将魔法数字提取到配置中
            ad_slope = (
                ad.iloc[last_valid_idx] - ad.iloc[last_valid_idx - 20]
            ) / 20  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            if ad_slope > 0:
                patterns.append("AD上升趋势(买盘持续涌入)")

                # 上升趋势强度判断
                if (
                    ad_slope > ad.iloc[last_valid_idx - 20 : last_valid_idx].diff().std() * 2
                ):  # TODO: 将魔法数字提取到配置中
                    patterns.append("AD强势上升(买盘强烈涌入)")
            else:
                patterns.append("AD下降趋势(卖盘持续涌出)")

                # 下降趋势强度判断
                if (
                    abs(ad_slope) > ad.iloc[last_valid_idx - 20 : last_valid_idx].diff().std() * 2
                ):  # TODO: 将魔法数字提取到配置中
                    patterns.append("AD强势下降(卖盘强烈涌出)")

        # 3. AD背离形态  # TODO: 将魔法数字提取到配置中
        if "close" in data.columns and last_valid_idx >= 40:  # TODO: 将魔法数字提取到配置中
            close = data["close"]

            # 检查最近的两个价格低点
            last_20_min_idx = close.iloc[last_valid_idx - 20 : last_valid_idx].idxmin()  # TODO: 将魔法数字提取到配置中
            prev_20_min_idx = close.iloc[
                last_valid_idx - 40 : last_valid_idx - 20
            ].idxmin()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            if last_20_min_idx is not None and prev_20_min_idx is not None:
                # 获取对应的AD值
                ad_at_last_low = ad.loc[last_20_min_idx]
                ad_at_prev_low = ad.loc[prev_20_min_idx]

                # 价格创新低但AD未创新低(正背离)
                if close.loc[last_20_min_idx] < close.loc[prev_20_min_idx] and ad_at_last_low > ad_at_prev_low:
                    patterns.append("AD正背离(价格创新低但AD未创新低)")

            # 检查最近的两个价格高点
            last_20_max_idx = close.iloc[last_valid_idx - 20 : last_valid_idx].idxmax()  # TODO: 将魔法数字提取到配置中
            prev_20_max_idx = close.iloc[
                last_valid_idx - 40 : last_valid_idx - 20
            ].idxmax()  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

            if last_20_max_idx is not None and prev_20_max_idx is not None:
                # 获取对应的AD值
                ad_at_last_high = ad.loc[last_20_max_idx]
                ad_at_prev_high = ad.loc[prev_20_max_idx]

                # 价格创新高但AD未创新高(负背离)
                if close.loc[last_20_max_idx] > close.loc[prev_20_max_idx] and ad_at_last_high < ad_at_prev_high:
                    patterns.append("AD负背离(价格创新高但AD未创新高)")

        return patterns

    def get_pattern_info_Ad(self, pattern_id: str) -> dict:
        """
        获取形态信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            "bullish": {"name": "看涨形态", "description": "指标显示看涨信号", "type": "BULLISH"},
            "bearish": {"name": "看跌形态", "description": "指标显示看跌信号", "type": "BEARISH"},
            "neutral": {"name": "中性形态", "description": "指标显示中性信号", "type": "NEUTRAL"},
            # 通用形态
            "strong_signal": {"name": "强信号", "description": "强烈的技术信号", "type": "STRONG"},
            "weak_signal": {"name": "弱信号", "description": "较弱的技术信号", "type": "WEAK"},
            "trend_up": {"name": "上升趋势", "description": "价格呈上升趋势", "type": "BULLISH"},
            "trend_down": {"name": "下降趋势", "description": "价格呈下降趋势", "type": "BEARISH"},
        }

        # 默认形态信息
        default_pattern = {
            "name": pattern_id.replace("_", " ").title(),
            "description": f"{pattern_id}形态",
            "type": "UNKNOWN",
        }

        return pattern_info_map.get(pattern_id, default_pattern)

    def _get_default_parameters_ad(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中

    def set_parameters_Ad_Ad_Ad_ad_duplicate(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator

            validator = IndicatorParameterValidator()

            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters("AD", params)
            if not is_valid:
                from utils.logger import get_logger
                from db.sql_manager import SQLManager, QueryType

                logger = get_logger(__name__)
                logger.warning(f"AD参数验证失败: {'; '.join(errors)}")
                # 使用默认参数
                params = self._default_parameters.copy()

            # 设置参数(保持向后兼容)
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        except Exception:
            # 如果验证失败,静默处理
            pass

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 如果已有结果返回True,否则返回False
        """
        return self._result is not None and not self._result.empty

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于AD (Accumulation/Distribution Line) 指标数值生成最新的交易信号
        
        AD交易信号逻辑：
        - AD上升 + 价格上升：累积买入信号
        - AD下降 + 价格下降：分布卖出信号
        - AD与均线的突破：趋势确认信号
        - AD与价格背离：反转信号
        - AD趋势强度：信号强度判断
        
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
                result = self.calculate(data)
                if result is not None:
                    self._result = result
            
            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("AD计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取AD相关值
            if len(self._result) < 2:
                return self._get_default_signal("AD数据不足")
                
            # 检查必要的列是否存在
            if 'AD' not in self._result.columns:
                return self._get_default_signal("AD结果列不存在")
                
            ad_values = self._result['AD'].dropna()
            if len(ad_values) < 2:
                return self._get_default_signal("AD有效数据不足")
                
            latest_ad = ad_values.iloc[-1]
            prev_ad = ad_values.iloc[-2]
            
            # 检查是否有NaN值
            if pd.isna(latest_ad) or pd.isna(prev_ad):
                return self._get_default_signal("AD数据包含NaN值")
            
            # 5. 获取价格数据
            close_values = data['close'].iloc[-2:]
            if len(close_values) < 2:
                return self._get_default_signal("价格数据不足")
                
            latest_price = close_values.iloc[-1]
            prev_price = close_values.iloc[-2]
            
            # 6. AD信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 计算AD和价格变化
            ad_change = latest_ad - prev_ad
            price_change = latest_price - prev_price
            ad_change_pct = (ad_change / abs(prev_ad)) * 100 if prev_ad != 0 else 0
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # 获取AD均线信息（如果存在）
            ad_ma = None
            if 'AD_MA' in self._result.columns:
                ad_ma_values = self._result['AD_MA'].dropna()
                if len(ad_ma_values) > 0:
                    ad_ma = ad_ma_values.iloc[-1]
            
            # AD-价格累积分布分析（核心AD信号）
            if ad_change > 0 and price_change > 0:
                # AD和价格同时上升 - 累积买入信号
                signal_type = "buy"
                accumulation_strength = min(abs(ad_change_pct) + abs(price_change_pct), 100) / 100
                strength = min(0.9, 0.7 + accumulation_strength * 0.2)
                confidence = 0.85
                reason = f"AD累积买入({ad_change_pct:.2f}%, {price_change_pct:.2f}%)，资金流入确认"
                
            elif ad_change < 0 and price_change < 0:
                # AD和价格同时下降 - 分布卖出信号
                signal_type = "sell"
                distribution_strength = min(abs(ad_change_pct) + abs(price_change_pct), 100) / 100
                strength = min(0.9, 0.7 + distribution_strength * 0.2)
                confidence = 0.85
                reason = f"AD分布卖出({ad_change_pct:.2f}%, {price_change_pct:.2f}%)，资金流出确认"
                
            elif ad_change > 0 and price_change < 0:
                # AD上升但价格下降 - 正背离，潜在底部
                signal_type = "buy"
                divergence_strength = min(abs(ad_change_pct) + abs(price_change_pct), 80) / 100
                strength = min(0.8, 0.6 + divergence_strength * 0.2)
                confidence = 0.75
                reason = f"AD正背离({ad_change_pct:.2f}% vs {price_change_pct:.2f}%)，资金暗中积累"
                
            elif ad_change < 0 and price_change > 0:
                # AD下降但价格上升 - 负背离，潜在顶部
                signal_type = "sell"
                divergence_strength = min(abs(ad_change_pct) + abs(price_change_pct), 80) / 100
                strength = min(0.8, 0.6 + divergence_strength * 0.2)
                confidence = 0.75
                reason = f"AD负背离({ad_change_pct:.2f}% vs {price_change_pct:.2f}%)，资金暗中流出"
            
            # AD均线突破信号
            elif ad_ma is not None:
                ad_ma_prev = None
                if len(self._result) >= 2 and 'AD_MA' in self._result.columns:
                    ad_ma_prev_values = self._result['AD_MA'].iloc[-2:-1]
                    if len(ad_ma_prev_values) > 0 and not pd.isna(ad_ma_prev_values.iloc[0]):
                        ad_ma_prev = ad_ma_prev_values.iloc[0]
                
                if ad_ma_prev is not None:
                    # AD突破均线向上
                    if prev_ad <= ad_ma_prev and latest_ad > ad_ma:
                        signal_type = "buy"
                        breakout_strength = min(abs(latest_ad - ad_ma) / abs(ad_ma), 0.2) if ad_ma != 0 else 0
                        strength = min(0.75, 0.6 + breakout_strength * 5)
                        confidence = 0.75
                        reason = f"AD突破均线向上({latest_ad:.0f} > {ad_ma:.0f})，累积趋势确认"
                        
                    # AD跌破均线向下
                    elif prev_ad >= ad_ma_prev and latest_ad < ad_ma:
                        signal_type = "sell"
                        breakdown_strength = min(abs(ad_ma - latest_ad) / abs(ad_ma), 0.2) if ad_ma != 0 else 0
                        strength = min(0.75, 0.6 + breakdown_strength * 5)
                        confidence = 0.75
                        reason = f"AD跌破均线向下({latest_ad:.0f} < {ad_ma:.0f})，分布趋势确认"
            
            # AD趋势强度分析（基于20期趋势）
            if len(ad_values) >= 20:
                ad_trend_start = ad_values.iloc[-20]
                ad_trend_slope = (latest_ad - ad_trend_start) / 20
                
                if abs(ad_trend_slope) > 0:
                    # 计算趋势强度标准差
                    try:
                        ad_recent = ad_values.iloc[-20:]
                        ad_volatility = ad_recent.diff().std()
                        trend_strength_ratio = abs(ad_trend_slope) / ad_volatility if ad_volatility > 0 else 0
                        
                        if trend_strength_ratio > 2.0:  # 强趋势
                            if ad_trend_slope > 0:
                                if signal_type == "buy":
                                    strength = min(strength + 0.15, 1.0)
                                    confidence = min(confidence + 0.1, 1.0)
                                    reason += "，强势累积趋势"
                                elif signal_type == "hold":
                                    signal_type = "buy"
                                    strength = 0.65
                                    confidence = 0.7
                                    reason = f"AD强势上升趋势，资金持续流入"
                            else:
                                if signal_type == "sell":
                                    strength = min(strength + 0.15, 1.0)
                                    confidence = min(confidence + 0.1, 1.0)
                                    reason += "，强势分布趋势"
                                elif signal_type == "hold":
                                    signal_type = "sell"
                                    strength = 0.65
                                    confidence = 0.7
                                    reason = f"AD强势下降趋势，资金持续流出"
                                    
                    except Exception:
                        pass  # 忽略趋势强度计算错误
            
            # 计算AD特有的元数据
            ad_trend = "上升" if ad_change > 0 else "下降" if ad_change < 0 else "平稳"
            price_trend = "上升" if price_change > 0 else "下降" if price_change < 0 else "平稳"
            accumulation_distribution = "累积" if (ad_change > 0) == (price_change > 0) else "背离" if ad_change != 0 and price_change != 0 else "中性"
            
            metadata = {
                'ad_value': latest_ad,
                'ad_previous': prev_ad,
                'ad_change': ad_change,
                'ad_change_pct': ad_change_pct,
                'ad_trend': ad_trend,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'price_trend': price_trend,
                'accumulation_distribution': accumulation_distribution,
                'ad_ma': ad_ma,
                'above_ma': latest_ad > ad_ma if ad_ma is not None else None,
                'indicator_type': 'volume_price_relationship',
                'calculation_method': 'CLV_based'  # Close Location Value based
            }
            
            # 7. 标准化输出
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
            logger.warning(f"AD信号生成失败: {e}")
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
            
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # AD需要足够的数据用于计算
        min_periods = max(getattr(self, 'period', 14), 2)
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


# 类别名,供指标注册系统使用
A_D = AD
AccDistribution = AD
