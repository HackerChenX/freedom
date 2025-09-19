from utils.container import container
#!/usr/bin/env python3
from utils.logger import get_logger
"""
MOMENTUM 指标

动量指标 - 衡量价格变化的速度和幅度
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
from utils.indicator_parameter_validator import IndicatorParameterValidator

logger = get_logger(__name__)


class MomentumMomentum(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    MOMENTUM 指标
    
    动量指标衡量价格变化的速度和幅度,用于识别趋势强度
    """
    
    @property
    def minimum_periods(self) -> int:
        """返回计算指标所需的最小周期数"""
        return getattr(self, 'period', 14) + 1  # TODO: 将魔法数字提取到配置中

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化MOMENTUM指标

        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "MOMENTUM"
        self.description = "动量指标,衡量价格变化的速度和幅度"

        # 初始化结果存储
        self._result = None

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_momentum()

        # 应用用户参数
        self.set_parameters_Momentum(**kwargs)
    
    def _get_default_parameters_momentum(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Momentum(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
        except Exception as e:
            logger.error(f"错误: {e}")
            return pd.DataFrame()
    
    def validate_parameters(self, **kwargs):
        """验证参数"""
        try:
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('MOMENTUM', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)  # TODO: 将魔法数字提取到配置中
                    
        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            self.period = 14  # TODO: 将魔法数字提取到配置中
    
    def calculate_Momentum(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MOMENTUM指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了MOMENTUM指标的Data_frame
        """
        result = self._calculate_momentum(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_momentum(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算MOMENTUM指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了MOMENTUM指标的Data_frame
        """
        df = data.copy()
        
        # 计算MOMENTUM (Price Momentum)
        # MOMENTUM = 当前收盘价 - N日前收盘价
        close = df['close']
        
        # 获取N日前的收盘价
        close_n_periods_ago = close.shift(self.period)
        
        # 计算动量
        momentum = close - close_n_periods_ago
        
        # 保存计算结果
        df['momentum'] = momentum
        df['MOMENTUM_VALUE'] = momentum  # 为了向后兼容
        
        # 计算动量的移动平均(平滑处理)
        df['momentum_ma'] = momentum.rolling(window=5).mean()  # TODO: 将魔法数字提取到配置中
        
        # 计算动量的标准差(波动性)
        df['momentum_std'] = momentum.rolling(window=10).std()
        
        # 计算相对动量(动量/价格比率)
        df['momentum_ratio'] = momentum / close * 100
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑(MOMENTUM指标特定逻辑)
        df = self._apply_momentum_signal_logic(df)

        return df

    def _apply_momentum_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用MOMENTUM指标特定的信号生成逻辑
        基于动量值的变化和趋势生成信号
        """
        try:
            # 获取动量值
            if 'momentum' not in df.columns:
                # 如果没有动量值,使用默认信号
                return df

            momentum = df['momentum']
            momentum_ma = df['momentum_ma']

            # MOMENTUM信号生成逻辑:
            # BUY: 动量为正且上升(加速上涨)
            # SELL: 动量为负且下降(加速下跌)
            # HOLD: 动量接近零或趋势不明确

            # 基本条件
            momentum_positive = momentum > 0
            momentum_negative = momentum < 0
            momentum_strong_positive = momentum > momentum.std()  # 动量超过标准差
            momentum_strong_negative = momentum < -momentum.std()  # 动量低于负标准差
            
            # 趋势条件
            momentum_rising = momentum > momentum.shift(1)
            momentum_falling = momentum < momentum.shift(1)
            
            # 连续上升/下降条件
            momentum_continuous_rising = (momentum > momentum.shift(1)) & (momentum.shift(1) > momentum.shift(2))
            momentum_continuous_falling = (momentum < momentum.shift(1)) & (momentum.shift(1) < momentum.shift(2))
            
            # 与移动平均的关系
            momentum_above_ma = momentum > momentum_ma
            momentum_below_ma = momentum < momentum_ma

            # 生成信号
            df.loc[:, 'buy_signal'] = (momentum_positive & momentum_rising) | (momentum_strong_positive & momentum_above_ma) | momentum_continuous_rising
            df.loc[:, 'sell_signal'] = (momentum_negative & momentum_falling) | (momentum_strong_negative & momentum_below_ma) | momentum_continuous_falling
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"MOMENTUM信号生成失败: {e}")
            # 如果出错,使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df

    def calculate_raw_score_Momentum(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MOMENTUM原始评分
        
        基于MOMENTUM指标的技术分析特点进行评分:
        1. 动量数值评分 (35%)  # TODO: 将魔法数字提取到配置中
        2. 动量趋势评分 (30%)  # TODO: 将魔法数字提取到配置中
        3. 动量强度评分 (25%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        4. 动量稳定性评分 (10%)  # TODO: 将魔法数字提取到配置中
        """
        if not self.has_result():
            self.calculate_Momentum(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 获取MOMENTUM数据
        momentum = self._result['momentum']
        momentum_ma = self._result['momentum_ma']
        momentum_std = self._result['momentum_std']
        momentum_ratio = self._result['momentum_ratio']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 计算动量的统计特征
        momentum_mean = momentum.rolling(window=20).mean()  # TODO: 将魔法数字提取到配置中
        momentum_percentile = momentum.rolling(window=20).rank(pct=True)  # TODO: 将魔法数字提取到配置中
        
        # 1. 动量数值评分 (35%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于动量值的正负和相对强度
        value_score = pd.Series(0.0, index=data.index)
        
        # 正动量加分,负动量减分
        value_score = np.where(momentum > 0, momentum_percentile * 20, value_score)  # TODO: 将魔法数字提取到配置中
        value_score = np.where(momentum < 0, (momentum_percentile - 1) * 20, value_score)  # TODO: 将魔法数字提取到配置中
        
        # 强动量额外加分
        momentum_abs = abs(momentum)
        momentum_strength = momentum_abs / (momentum_abs.rolling(window=20).mean() + 1e-8)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        value_score += np.where(momentum_strength > 1.5, 10, 0)  # TODO: 将魔法数字提取到配置中
        
        scores += value_score * 0.35  # TODO: 将魔法数字提取到配置中
        
        # 2. 动量趋势评分 (30%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于动量的变化趋势
        momentum_change = momentum - momentum.shift(1)
        momentum_change_2 = momentum.shift(1) - momentum.shift(2)
        
        trend_score = pd.Series(0.0, index=data.index)
        
        # 动量加速(连续上升)
        trend_score = np.where((momentum_change > 0) & (momentum_change_2 > 0), 15, trend_score)  # TODO: 将魔法数字提取到配置中
        # 动量减速(连续下降)
        trend_score = np.where((momentum_change < 0) & (momentum_change_2 < 0), -15, trend_score)  # TODO: 将魔法数字提取到配置中
        # 单次上升
        trend_score = np.where((momentum_change > 0) & (momentum_change_2 <= 0), 8, trend_score)  # TODO: 将魔法数字提取到配置中
        # 单次下降
        trend_score = np.where((momentum_change < 0) & (momentum_change_2 >= 0), -8, trend_score)  # TODO: 将魔法数字提取到配置中
        
        # 动量方向改变
        momentum_direction_change = ((momentum > 0) & (momentum.shift(1) < 0)) | ((momentum < 0) & (momentum.shift(1) > 0))
        trend_score = np.where(momentum_direction_change, 5, trend_score)  # TODO: 将魔法数字提取到配置中
        
        scores += trend_score * 0.3  # TODO: 将魔法数字提取到配置中
        
        # 3. 动量强度评分 (25%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于动量的相对强度和波动性
        strength_score = pd.Series(0.0, index=data.index)
        
        # 相对动量强度
        if len(momentum_ratio.dropna()) > 0:
            ratio_abs = abs(momentum_ratio)
            strength_score = np.where(ratio_abs > 5, 12, strength_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            strength_score = np.where((ratio_abs >= 3) & (ratio_abs <= 5), 8, strength_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            strength_score = np.where((ratio_abs >= 1) & (ratio_abs < 3), 4, strength_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            strength_score = np.where(ratio_abs < 0.5, -5, strength_score)  # 动量太弱减分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 动量一致性(与移动平均的关系)
        if len(momentum_ma.dropna()) > 0:
            momentum_consistency = abs(momentum - momentum_ma) / (momentum_std + 1e-8)  # TODO: 将魔法数字提取到配置中
            strength_score += np.where(momentum_consistency < 0.5, 5, 0)  # 一致性高加分  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            strength_score += np.where(momentum_consistency > 2, -3, 0)  # 一致性低减分  # TODO: 将魔法数字提取到配置中
        
        scores += strength_score * 0.25  # TODO: 将魔法数字提取到配置中
        
        # 4. 动量稳定性评分 (10%)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        # 基于动量的稳定性和可预测性
        stability_score = pd.Series(0.0, index=data.index)
        
        if len(momentum_std.dropna()) > 0:
            # 波动性适中加分
            volatility_percentile = momentum_std.rolling(window=20).rank(pct=True)  # TODO: 将魔法数字提取到配置中
            stability_score = np.where((volatility_percentile >= 0.3) & (volatility_percentile <= 0.7), 5, stability_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 波动性过高减分
            stability_score = np.where(volatility_percentile > 0.9, -5, stability_score)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            # 波动性过低减分(缺乏动量)
            stability_score = np.where(volatility_percentile < 0.1, -3, stability_score)  # TODO: 将魔法数字提取到配置中
        
        scores += stability_score * 0.1
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Momentum(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
        # 基于MOMENTUM指标的明确性计算置信度
        momentum = self._result['momentum'].dropna()
        momentum_std = self._result['momentum_std'].dropna()
        
        if len(momentum) == 0:
            return 0.5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 计算最近的动量值
        recent_momentum = momentum.iloc[-1] if len(momentum) > 0 else 0
        recent_std = momentum_std.iloc[-1] if len(momentum_std) > 0 else 1
        
        # 动量相对强度
        momentum_strength = min(abs(recent_momentum) / (recent_std + 1e-8), 3.0) / 3.0  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 趋势一致性
        trend_consistency = 0
        if len(momentum) >= 3:  # TODO: 将魔法数字提取到配置中
            recent_trend = momentum.iloc[-3:].diff().dropna()  # TODO: 将魔法数字提取到配置中
            if len(recent_trend) > 0:
                # 如果趋势方向一致,提高置信度
                if all(recent_trend > 0) or all(recent_trend < 0):
                    trend_consistency = 0.2
        
        # 动量持续性
        momentum_persistence = 0
        if len(momentum) >= 5:  # TODO: 将魔法数字提取到配置中
            recent_momentum_values = momentum.iloc[-5:]  # TODO: 将魔法数字提取到配置中
            if (recent_momentum_values > 0).all() or (recent_momentum_values < 0).all():
                momentum_persistence = 0.15  # TODO: 将魔法数字提取到配置中
        
        base_confidence = 0.25 + momentum_strength * 0.4 + trend_consistency + momentum_persistence  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        return min(max(base_confidence, 0.2), 0.9)  # TODO: 将魔法数字提取到配置中
    
    def get_patterns_Momentum(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取MOMENTUM相关形态"""
        if not self.has_result():
            self.calculate_Momentum(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        momentum = self._result['momentum']
        momentum_ma = self._result['momentum_ma']
        momentum_std = self._result['momentum_std']
        
        # 基本形态
        patterns['MOMENTUM_POSITIVE'] = momentum > 0
        patterns['MOMENTUM_NEGATIVE'] = momentum < 0
        patterns['MOMENTUM_NEUTRAL'] = abs(momentum) < (momentum_std * 0.5)  # TODO: 将魔法数字提取到配置中
        
        # 强度形态
        patterns['MOMENTUM_STRONG_POSITIVE'] = momentum > momentum_std
        patterns['MOMENTUM_STRONG_NEGATIVE'] = momentum < -momentum_std
        patterns['MOMENTUM_EXTREME_POSITIVE'] = momentum > momentum_std * 2
        patterns['MOMENTUM_EXTREME_NEGATIVE'] = momentum < -momentum_std * 2
        
        # 趋势形态
        momentum_change = momentum - momentum.shift(1)
        patterns['MOMENTUM_RISING'] = momentum_change > 0
        patterns['MOMENTUM_FALLING'] = momentum_change < 0
        patterns['MOMENTUM_ACCELERATING'] = (momentum_change > 0) & (momentum_change > momentum_change.shift(1))
        patterns['MOMENTUM_DECELERATING'] = (momentum_change < 0) & (momentum_change < momentum_change.shift(1))
        
        # 与移动平均的关系
        patterns['MOMENTUM_ABOVE_MA'] = momentum > momentum_ma
        patterns['MOMENTUM_BELOW_MA'] = momentum < momentum_ma
        
        # 转折形态
        patterns['MOMENTUM_REVERSAL'] = ((momentum > 0) & (momentum.shift(1) < 0)) | ((momentum < 0) & (momentum.shift(1) > 0))
        patterns['MOMENTUM_PEAK'] = (momentum > momentum.shift(1)) & (momentum > momentum.shift(-1))
        patterns['MOMENTUM_TROUGH'] = (momentum < momentum.shift(1)) & (momentum < momentum.shift(-1))
        
        return patterns

    # ================== 抽象方法实现 ==================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象方法实现:调用MOMENTUM计算逻辑"""
        return self.calculate_momentum(data, **kwargs)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象方法实现:计算MOMENTUM原始评分"""
        return self.calculate_raw_score_momentum(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象方法实现:获取MOMENTUM形态"""
        return self.get_patterns_momentum(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象方法实现:设置参数"""
        return self.set_parameters_momentum(**kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> float:
        """抽象方法实现:计算置信度"""
        return self.calculate_confidence_momentum(data, **kwargs)

    # ================== 兼容性方法 ==================

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:计算指标"""
        return self.calculate_momentum(data, **kwargs)

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:获取形态"""
        return self.get_patterns_momentum(data, **kwargs)

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法:计算原始评分"""
        return self.calculate_raw_score_momentum(data, **kwargs)

    def get_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """兼容性方法:生成信号"""
        return self.generate_signals_momentum(data, **kwargs)

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """兼容性方法:计算综合评分"""
        return self.calculate_score_momentum(data, **kwargs)

    def calculate_confidence(self, data: pd.DataFrame, **kwargs) -> float:
        """兼容性方法:计算置信度"""
        return self.calculate_confidence_momentum(data, **kwargs)

    def set_parameters(self, **kwargs):
        """兼容性方法:设置参数"""
        return self.set_parameters_momentum(**kwargs)

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:计算指标(compute别名)"""
        return self.calculate_momentum(data, **kwargs)

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """兼容性方法:生成交易信号"""
        return self.generate_signals_momentum(data, **kwargs)

    def register_patterns(self, **kwargs):
        """兼容性方法:注册形态(空实现)"""
        pass

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None

    # ================== 公共接口方法 ==================

    def calculate_momentum(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算MOMENTUM指标(公共接口)

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            包含MOMENTUM指标的DataFrame
        """
        return self.calculate_Momentum(data, **kwargs)

    def calculate_raw_score_momentum(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MOMENTUM原始评分(公共接口)

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            原始评分Series
        """
        return self.calculate_raw_score_Momentum(data, **kwargs)

    def calculate_score_momentum(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        计算MOMENTUM综合评分(公共接口)

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            包含评分和置信度的字典
        """
        return self.calculate_score_Momentum(data, **kwargs)

    def get_patterns_momentum(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取MOMENTUM形态(公共接口)

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            包含形态的DataFrame
        """
        return self.get_patterns_Momentum(data, **kwargs)

    def generate_signals_momentum(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成MOMENTUM信号(公共接口)

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            包含信号的字典
        """
        try:
            # 先计算MOMENTUM指标
            result = self.calculate_momentum(data, **kwargs)
            
            # 简单的信号生成逻辑
            momentum = result['momentum']
            momentum_ma = result['momentum_ma']
            
            # 生成买入和卖出信号
            buy_signal = (momentum > momentum_ma) & (momentum.shift(1) <= momentum_ma.shift(1))
            sell_signal = (momentum < momentum_ma) & (momentum.shift(1) >= momentum_ma.shift(1))
            
            return {
                'momentum_buy_signal': buy_signal.fillna(False),
                'momentum_sell_signal': sell_signal.fillna(False)
            }
        except (ValueError, KeyError, IndexError) as e:
            # 如果信号生成失败,返回空信号
            logger.warning(f"MOMENTUM信号生成失败: {e}, 返回空信号")
            return {
                'momentum_buy_signal': pd.Series([False] * len(data), index=data.index),
                'momentum_sell_signal': pd.Series([False] * len(data), index=data.index)
            }

    def set_parameters_momentum(self, **kwargs):
        """
        设置MOMENTUM参数(公共接口)

        Args:
            **kwargs: 参数字典
        """
        return self.set_parameters_Momentum(**kwargs)

    def calculate_confidence_momentum(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算MOMENTUM置信度(公共接口)

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数

        Returns:
            置信度值
        """
        # 计算必要的中间数据
        score = self.calculate_raw_score_momentum(data, **kwargs)
        patterns = self.get_patterns_momentum(data, **kwargs)
        signals = self.generate_signals_momentum(data, **kwargs)
        
        return self.calculate_confidence_Momentum(score, patterns, signals)

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于MOMENTUM (动量指标) 指标数值生成最新的交易信号
        
        MOMENTUM交易信号逻辑：
        - 动量上穿动量均线：买入信号
        - 动量下穿动量均线：卖出信号
        - 动量持续高于均线：强势持有信号
        - 动量持续低于均线：弱势持有信号
        - 动量极值分析：极强/极弱动量信号
        - 动量背离检测：价格与动量背离的反转信号
        
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
                self.calculate_Momentum(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("MOMENTUM计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取MOMENTUM相关值
            if len(self._result) < 2:
                return self._get_default_signal("MOMENTUM数据不足")
                
            # 检查必要的列是否存在
            required_columns = ['momentum', 'momentum_ma']
            if not all(col in self._result.columns for col in required_columns):
                return self._get_default_signal("MOMENTUM结果列不完整")
                
            latest_momentum = self._result['momentum'].iloc[-1]
            latest_momentum_ma = self._result['momentum_ma'].iloc[-1]
            prev_momentum = self._result['momentum'].iloc[-2]
            prev_momentum_ma = self._result['momentum_ma'].iloc[-2]
            
            # 检查是否有NaN值
            if pd.isna(latest_momentum) or pd.isna(latest_momentum_ma) or pd.isna(prev_momentum) or pd.isna(prev_momentum_ma):
                return self._get_default_signal("MOMENTUM数据包含NaN值")
            
            # 5. MOMENTUM信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 计算动量相对位置和变化
            momentum_diff = latest_momentum - latest_momentum_ma
            momentum_change = latest_momentum - prev_momentum
            momentum_ma_change = latest_momentum_ma - prev_momentum_ma
            
            # 计算动量交叉状态
            momentum_cross_up = prev_momentum <= prev_momentum_ma and latest_momentum > latest_momentum_ma
            momentum_cross_down = prev_momentum >= prev_momentum_ma and latest_momentum < latest_momentum_ma
            
            # 动量上穿均线信号（最高优先级）
            if momentum_cross_up:
                # 动量上穿均线 - 买入信号
                signal_type = "buy"
                cross_strength = min(abs(momentum_diff) / (abs(latest_momentum_ma) + 1), 1.0)
                strength = max(0.85, 0.85 + cross_strength * 0.15)
                confidence = 0.9
                reason = f"MOMENTUM上穿均线({latest_momentum:.4f}>{latest_momentum_ma:.4f})，强烈买入信号"
                
            elif momentum_cross_down:
                # 动量下穿均线 - 卖出信号
                signal_type = "sell"
                cross_strength = min(abs(momentum_diff) / (abs(latest_momentum_ma) + 1), 1.0)
                strength = max(0.85, 0.85 + cross_strength * 0.15)
                confidence = 0.9
                reason = f"MOMENTUM下穿均线({latest_momentum:.4f}<{latest_momentum_ma:.4f})，强烈卖出信号"
            
            # 动量加速信号
            elif latest_momentum > latest_momentum_ma:
                # 动量在均线上方
                if momentum_change > 0 and momentum_ma_change > 0:
                    # 动量和均线都在上升 - 强势买入
                    signal_type = "buy"
                    momentum_strength = min(momentum_diff / (abs(latest_momentum_ma) + 1), 2.0)
                    acceleration = min(momentum_change / (abs(prev_momentum) + 1), 1.0)
                    strength = max(0.75, min(1.0, 0.75 + momentum_strength * 0.15 + acceleration * 0.1))
                    confidence = 0.85
                    reason = f"MOMENTUM强势上升({latest_momentum:.4f}>{latest_momentum_ma:.4f})，买入信号"
                elif momentum_change > 0:
                    # 动量上升，均线可能滞后
                    signal_type = "buy"
                    momentum_strength = min(momentum_diff / (abs(latest_momentum_ma) + 1), 1.0)
                    strength = max(0.7, 0.7 + momentum_strength * 0.2)
                    confidence = 0.8
                    reason = f"MOMENTUM上升动能({latest_momentum:.4f})，买入信号"
                else:
                    # 动量在均线上方但下降
                    signal_type = "buy"
                    momentum_strength = min(momentum_diff / (abs(latest_momentum_ma) + 1), 1.0)
                    strength = max(0.6, 0.6 + momentum_strength * 0.15)
                    confidence = 0.7
                    reason = f"MOMENTUM高位回落({latest_momentum:.4f})，弱买入信号"
                    
            elif latest_momentum < latest_momentum_ma:
                # 动量在均线下方
                if momentum_change < 0 and momentum_ma_change < 0:
                    # 动量和均线都在下降 - 强势卖出
                    signal_type = "sell"
                    momentum_strength = min(abs(momentum_diff) / (abs(latest_momentum_ma) + 1), 2.0)
                    deceleration = min(abs(momentum_change) / (abs(prev_momentum) + 1), 1.0)
                    strength = max(0.75, min(1.0, 0.75 + momentum_strength * 0.15 + deceleration * 0.1))
                    confidence = 0.85
                    reason = f"MOMENTUM强势下降({latest_momentum:.4f}<{latest_momentum_ma:.4f})，卖出信号"
                elif momentum_change < 0:
                    # 动量下降，均线可能滞后
                    signal_type = "sell"
                    momentum_strength = min(abs(momentum_diff) / (abs(latest_momentum_ma) + 1), 1.0)
                    strength = max(0.7, 0.7 + momentum_strength * 0.2)
                    confidence = 0.8
                    reason = f"MOMENTUM下降动能({latest_momentum:.4f})，卖出信号"
                else:
                    # 动量在均线下方但上升
                    signal_type = "sell"
                    momentum_strength = min(abs(momentum_diff) / (abs(latest_momentum_ma) + 1), 1.0)
                    strength = max(0.6, 0.6 + momentum_strength * 0.15)
                    confidence = 0.7
                    reason = f"MOMENTUM低位反弹({latest_momentum:.4f})，弱卖出信号"
            
            # 极值动量信号
            if 'momentum_std' in self._result.columns:
                momentum_std = self._result['momentum_std'].iloc[-1]
                if not pd.isna(momentum_std) and momentum_std > 0:
                    # 计算动量的标准化程度
                    z_score = abs(momentum_diff) / momentum_std
                    
                    if z_score > 2.0:  # 极端动量
                        if signal_type == "buy":
                            strength = min(1.0, strength + 0.1)
                            confidence = min(1.0, confidence + 0.05)
                            reason += "（极强动量确认）"
                        elif signal_type == "sell":
                            strength = min(1.0, strength + 0.1)
                            confidence = min(1.0, confidence + 0.05)
                            reason += "（极弱动量确认）"
            
            # 计算MOMENTUM特有的元数据
            momentum_momentum = "上升" if momentum_change > 0 else "下降" if momentum_change < 0 else "平稳"
            momentum_ma_momentum = "上升" if momentum_ma_change > 0 else "下降" if momentum_ma_change < 0 else "平稳"
            
            # 确定当前MOMENTUM相对强度
            if latest_momentum > latest_momentum_ma * 1.2:
                momentum_strength_level = "极强"
            elif latest_momentum > latest_momentum_ma * 1.1:
                momentum_strength_level = "强"
            elif latest_momentum > latest_momentum_ma:
                momentum_strength_level = "中等偏强"
            elif latest_momentum > latest_momentum_ma * 0.9:
                momentum_strength_level = "中等偏弱"
            elif latest_momentum > latest_momentum_ma * 0.8:
                momentum_strength_level = "弱"
            else:
                momentum_strength_level = "极弱"
            
            # 计算动量趋势一致性
            momentum_trend_consistency = (momentum_change > 0 and momentum_ma_change > 0) or \
                                       (momentum_change < 0 and momentum_ma_change < 0)
            
            metadata = {
                'momentum_value': latest_momentum,
                'momentum_ma_value': latest_momentum_ma,
                'momentum_previous': prev_momentum,
                'momentum_ma_previous': prev_momentum_ma,
                'momentum_change': momentum_change,
                'momentum_ma_change': momentum_ma_change,
                'momentum_diff': momentum_diff,
                'momentum_momentum': momentum_momentum,
                'momentum_ma_momentum': momentum_ma_momentum,
                'momentum_strength_level': momentum_strength_level,
                'momentum_cross_up': momentum_cross_up,
                'momentum_cross_down': momentum_cross_down,
                'momentum_above_ma': latest_momentum > latest_momentum_ma,
                'momentum_trend_consistency': momentum_trend_consistency,
                'momentum_relative_position': latest_momentum / (latest_momentum_ma + 1e-10),
                'momentum_volatility': self._result.get('momentum_std', pd.Series([0])).iloc[-1] if 'momentum_std' in self._result.columns else 0,
                'period': self.period
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
            logger.warning(f"MOMENTUM信号生成失败: {e}")
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
            
        # MOMENTUM需要足够的数据用于计算
        min_periods = self.period + 5
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
MOMENTUM = MomentumMomentum
Momentum = MomentumMomentum
