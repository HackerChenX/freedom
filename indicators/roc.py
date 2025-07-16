import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class RateOfChange(BaseIndicator, PatternSignalMixin):
    """
    Rate of Change变化率指标

    ROC指标衡量价格在指定周期内的变化率，用于识别动量和趋势强度
    """

    def __init__(self, **kwargs):
        """
        初始化ROC指标

        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ROC"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_roc()

        # 应用用户参数
        self.set_parameters_Roc(**kwargs)

    def _get_default_parameters_roc(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}

    def set_parameters_Roc(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ROC', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass

        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass

        # 设置参数
        self.period = kwargs.get('period', 14)

    def calculate_Roc(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ROC指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了ROC指标的Data_frame
        """
        result = self._calculate_roc(data, **kwargs)
        self._result = result
        return result

    def _calculate_roc(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ROC指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了ROC指标的Data_frame
        """
        df = data.copy()

        # 计算ROC (Rate of Change)
        # ROC = (今日收盘价 - N日前收盘价) / N日前收盘价 * 100
        close = df['close']
        
        # 获取N日前的收盘价
        close_n_periods_ago = close.shift(self.period)
        
        # 计算ROC
        roc = ((close - close_n_periods_ago) / close_n_periods_ago) * 100
        
        # 保存计算结果
        df['roc'] = roc
        df['ROC_VALUE'] = roc  # 为了向后兼容
        
        # 计算ROC的移动平均（平滑处理）
        df['roc_ma'] = roc.rolling(window=5).mean()
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑（ROC指标特定逻辑）
        df = self._apply_roc_signal_logic(df)

        return df

    def _apply_roc_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用ROC指标特定的信号生成逻辑
        基于变化率的正负值和趋势生成信号
        """
        try:
            # 获取ROC值
            if 'roc' not in df.columns:
                # 如果没有ROC值，使用默认信号
                return df

            roc = df['roc']

            # ROC信号生成逻辑：
            # BUY: ROC值为正且上升（动量增强）
            # SELL: ROC值为负且下降（动量减弱）
            # HOLD: ROC值接近零或趋势不明确

            # 基本条件
            roc_positive = roc > 0
            roc_negative = roc < 0
            roc_strong_positive = roc > 5  # 强正动量
            roc_strong_negative = roc < -5  # 强负动量
            
            # 趋势条件
            roc_rising = roc > roc.shift(1)
            roc_falling = roc < roc.shift(1)
            
            # 连续上升/下降条件
            roc_continuous_rising = (roc > roc.shift(1)) & (roc.shift(1) > roc.shift(2))
            roc_continuous_falling = (roc < roc.shift(1)) & (roc.shift(1) < roc.shift(2))

            # 生成信号
            df.loc[:, 'buy_signal'] = (roc_positive & roc_rising) | roc_continuous_rising
            df.loc[:, 'sell_signal'] = (roc_negative & roc_falling) | roc_continuous_falling
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"ROC信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score_Roc(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ROC原始评分
        
        基于ROC指标的技术分析特点进行评分：
        1. ROC数值评分 (40%)
        2. ROC趋势评分 (30%)
        3. ROC动量强度 (20%)
        4. ROC稳定性 (10%)
        """
        if not self.has_result():
            self.calculate_Roc(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取ROC数据
        roc = self._result['roc']
        roc_ma = self._result['roc_ma']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 1. ROC数值评分 (40%)
        # ROC > 10: 强上涨动量 (+20分)
        # ROC 5-10: 中等上涨动量 (+15分)
        # ROC 0-5: 弱上涨动量 (+5分)
        # ROC -5-0: 弱下跌动量 (-5分)
        # ROC -10--5: 中等下跌动量 (-15分)
        # ROC < -10: 强下跌动量 (-20分)
        value_score = pd.Series(0.0, index=data.index)
        value_score = np.where(roc > 10, 20, value_score)
        value_score = np.where((roc >= 5) & (roc <= 10), 15, value_score)
        value_score = np.where((roc > 0) & (roc < 5), 5, value_score)
        value_score = np.where((roc >= -5) & (roc < 0), -5, value_score)
        value_score = np.where((roc >= -10) & (roc < -5), -15, value_score)
        value_score = np.where(roc < -10, -20, value_score)
        scores += value_score * 0.4
        
        # 2. ROC趋势评分 (30%)
        # ROC上升趋势加分，下降趋势减分
        roc_change = roc - roc.shift(1)
        roc_change_2 = roc.shift(1) - roc.shift(2)
        
        trend_score = pd.Series(0.0, index=data.index)
        # 连续上升
        trend_score = np.where((roc_change > 0) & (roc_change_2 > 0), 15, trend_score)
        # 单次上升
        trend_score = np.where((roc_change > 0) & (roc_change_2 <= 0), 8, trend_score)
        # 连续下降
        trend_score = np.where((roc_change < 0) & (roc_change_2 < 0), -15, trend_score)
        # 单次下降
        trend_score = np.where((roc_change < 0) & (roc_change_2 >= 0), -8, trend_score)
        scores += trend_score * 0.3
        
        # 3. ROC动量强度 (20%)
        # 基于ROC的绝对值评估动量强度
        roc_abs = abs(roc)
        momentum_score = pd.Series(0.0, index=data.index)
        momentum_score = np.where(roc_abs > 15, 10, momentum_score)
        momentum_score = np.where((roc_abs >= 10) & (roc_abs <= 15), 8, momentum_score)
        momentum_score = np.where((roc_abs >= 5) & (roc_abs < 10), 5, momentum_score)
        momentum_score = np.where(roc_abs < 2, -5, momentum_score)  # 动量太弱减分
        scores += momentum_score * 0.2
        
        # 4. ROC稳定性 (10%)
        # 基于ROC移动平均的稳定性
        if len(roc_ma.dropna()) > 0:
            roc_stability = abs(roc - roc_ma)
            stability_score = pd.Series(0.0, index=data.index)
            stability_score = np.where(roc_stability < 2, 5, stability_score)  # 稳定加分
            stability_score = np.where(roc_stability > 10, -5, stability_score)  # 不稳定减分
            scores += stability_score * 0.1
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores

    def calculate_confidence_Roc(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于ROC指标的明确性计算置信度
        roc = self._result['roc'].dropna()
        
        if len(roc) == 0:
            return 0.5
        
        # 计算最近的ROC值
        recent_roc = roc.iloc[-1] if len(roc) > 0 else 0
        
        # ROC绝对值越大，置信度越高
        roc_strength = min(abs(recent_roc) / 20, 1.0)  # 标准化到0-1
        
        # 趋势一致性提高置信度
        trend_consistency = 0
        if len(roc) >= 3:
            recent_trend = roc.iloc[-3:].diff().dropna()
            if len(recent_trend) > 0:
                # 如果趋势方向一致，提高置信度
                if all(recent_trend > 0) or all(recent_trend < 0):
                    trend_consistency = 0.2
        
        base_confidence = 0.3 + roc_strength * 0.5 + trend_consistency
        return min(max(base_confidence, 0.2), 0.9)

    def get_patterns_Roc(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取ROC相关形态"""
        if not self.has_result():
            self.calculate_Roc(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        roc = self._result['roc']
        
        # 基本形态
        patterns['ROC_POSITIVE'] = roc > 0
        patterns['ROC_NEGATIVE'] = roc < 0
        patterns['ROC_STRONG_POSITIVE'] = roc > 10
        patterns['ROC_STRONG_NEGATIVE'] = roc < -10
        patterns['ROC_NEUTRAL'] = (roc >= -2) & (roc <= 2)
        
        # 趋势形态
        roc_change = roc - roc.shift(1)
        patterns['ROC_RISING'] = roc_change > 0
        patterns['ROC_FALLING'] = roc_change < 0
        patterns['ROC_ACCELERATING'] = (roc_change > 0) & (roc_change > roc_change.shift(1))
        patterns['ROC_DECELERATING'] = (roc_change < 0) & (roc_change < roc_change.shift(1))
        
        # 极端形态
        patterns['ROC_EXTREME_HIGH'] = roc > 20
        patterns['ROC_EXTREME_LOW'] = roc < -20
        patterns['ROC_MOMENTUM_SHIFT'] = (roc > 0) & (roc.shift(1) < 0) | (roc < 0) & (roc.shift(1) > 0)
        
        return patterns
