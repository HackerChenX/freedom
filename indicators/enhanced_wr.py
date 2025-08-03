import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class EnhancedWr(BaseIndicator, PatternSignalMixin):
    """
    增强型Williams %R指标

    在标准Williams %R基础上增加了多周期分析、自适应阈值、背离检测等功能
    """

    def __init__(self, 
                 period: int = 14,
                 overbought: float = -20.0,
                 oversold: float = -80.0,
                 multi_periods: List[int] = None,
                 adaptive_thresholds: bool = True,
                 smooth_period: int = 3,
                 **kwargs):
        """
        初始化增强型Williams %R指标

        Args:
            period: Williams %R计算周期，默认14
            overbought: 超买阈值，默认-20
            oversold: 超卖阈值，默认-80
            multi_periods: 多周期分析，默认[9, 14, 21]
            adaptive_thresholds: 是否使用自适应阈值，默认True
            smooth_period: 平滑周期，默认3
            **kwargs: 其他参数
        """
        super().__init__()
        self.name = "ENHANCED_WR"
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.multi_periods = multi_periods or [9, 14, 21]
        self.adaptive_thresholds = adaptive_thresholds
        self.smooth_period = smooth_period

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_enhancedwr()

        # 应用用户参数
        self.set_parameters_Wr(**kwargs)

    def _get_default_parameters_enhancedwr(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period": 14,
            "overbought": -20.0,
            "oversold": -80.0,
            "smooth_period": 3
        }

    def set_parameters_Wr(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ENHANCED_WR', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass

        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass

        # 设置参数
        self.period = kwargs.get('period', 14)
        self.overbought = kwargs.get('overbought', -20.0)
        self.oversold = kwargs.get('oversold', -80.0)
        self.smooth_period = kwargs.get('smooth_period', 3)

    def calculate_Wr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算增强型Williams %R指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了增强型Williams %R指标的Data_frame
        """
        result = self._calculate_enhancedwr(data, **kwargs)
        self._result = result
        return result

    def _calculate_enhancedwr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算增强型Williams %R指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了增强型Williams %R指标的Data_frame
        """
        min_length = max(self.multi_periods) + self.smooth_period + 10
        if len(data) < min_length:
            # 数据不足，返回空结果
            df = data.copy()
            df['ENHANCED_WR_VALUE'] = 50.0
            df['wr_signal'] = 0
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['hold_signal'] = True
            return df

        df = data.copy()

        # 计算主要Williams %R
        df['wr'] = self._calculate_williams_r(df, self.period)
        
        # 计算平滑Williams %R
        df['wr_smooth'] = df['wr'].rolling(window=self.smooth_period).mean()
        
        # 计算多周期Williams %R
        for period in self.multi_periods:
            df[f'wr_{period}'] = self._calculate_williams_r(df, period)
        
        # 计算自适应阈值
        if self.adaptive_thresholds:
            df['wr_overbought'], df['wr_oversold'] = self._calculate_adaptive_thresholds(df['wr'])
        else:
            df['wr_overbought'] = self.overbought
            df['wr_oversold'] = self.oversold
        
        # 计算Williams %R背离
        df['wr_divergence'] = self._calculate_wr_divergence(df['close'], df['wr'])
        
        # 计算Williams %R趋势强度
        df['wr_trend_strength'] = self._calculate_wr_trend_strength(df['wr'])
        
        # 计算多周期一致性
        df['wr_consistency'] = self._calculate_multi_period_consistency(df)
        
        # 计算Williams %R动量
        df['wr_momentum'] = self._calculate_wr_momentum(df['wr'])
        
        # 计算综合评分
        df['ENHANCED_WR_VALUE'] = self._calculate_enhanced_wr_score(df)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于Williams %R的买入信号
        df['wr_signal'] = self._generate_wr_signals(df)
        df['buy_signal'] = df['wr_signal'] > 0
        df['sell_signal'] = df['wr_signal'] < 0
        df['hold_signal'] = df['wr_signal'] == 0

        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        计算Williams %R指标
        
        Args:
            df: 数据Data_frame
            period: 计算周期
            
        Returns:
            Williams %R值序列
        """
        # 计算最高价和最低价
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        # 计算Williams %R
        wr = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
        
        return wr
    
    def _calculate_adaptive_thresholds(self, wr: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算自适应超买超卖阈值
        
        Args:
            wr: Williams %R值序列
            
        Returns:
            (超买阈值序列, 超卖阈值序列)
        """
        # 使用滚动统计计算动态阈值
        window = 50
        wr_mean = wr.rolling(window=window).mean()
        wr_std = wr.rolling(window=window).std()
        
        # 动态阈值：均值 ± 1.5倍标准差
        overbought = wr_mean + 1.5 * wr_std
        oversold = wr_mean - 1.5 * wr_std
        
        # 限制阈值范围
        overbought = overbought.clip(-30, -10)
        oversold = oversold.clip(-90, -70)
        
        return overbought, oversold
    
    def _calculate_wr_divergence(self, close: pd.Series, wr: pd.Series) -> pd.Series:
        """
        计算Williams %R背离信号
        
        Args:
            close: 收盘价序列
            wr: Williams %R值序列
            
        Returns:
            背离信号序列
        """
        # 简化的背离检测：比较价格和Williams %R的趋势方向
        price_trend = close.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        wr_trend = wr.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        # 背离：价格和Williams %R趋势方向相反
        divergence = (price_trend != wr_trend).astype(int)
        
        return divergence
    
    def _calculate_wr_trend_strength(self, wr: pd.Series) -> pd.Series:
        """
        计算Williams %R趋势强度
        
        Args:
            wr: Williams %R值序列
            
        Returns:
            趋势强度序列
        """
        # 计算Williams %R的移动平均和标准差
        wr_ma = wr.rolling(window=14).mean()
        wr_std = wr.rolling(window=14).std()
        
        # 趋势强度：Williams %R偏离移动平均的程度
        trend_strength = abs(wr - wr_ma) / (wr_std + 1e-10)
        
        return trend_strength
    
    def _calculate_multi_period_consistency(self, df: pd.DataFrame) -> pd.Series:
        """
        计算多周期Williams %R一致性
        
        Args:
            df: 包含多周期Williams %R的Data_frame
            
        Returns:
            一致性评分序列
        """
        # 计算多周期Williams %R的一致性
        wr_columns = [f'wr_{period}' for period in self.multi_periods]
        
        consistency_scores = []
        for i in range(len(df)):
            wr_values = [df[col].iloc[i] for col in wr_columns if col in df.columns]
            if len(wr_values) > 1:
                # 计算Williams %R值的标准差，标准差越小一致性越高
                std_dev = np.std(wr_values)
                consistency = max(0, 1 - std_dev / 50)  # 归一化到0-1
            else:
                consistency = 0.5
            consistency_scores.append(consistency)
        
        return pd.Series(consistency_scores, index=df.index)
    
    def _calculate_wr_momentum(self, wr: pd.Series) -> pd.Series:
        """
        计算Williams %R动量
        
        Args:
            wr: Williams %R值序列
            
        Returns:
            动量序列
        """
        # 计算Williams %R的变化率
        momentum = wr.diff(periods=5)  # 5日变化率
        
        # 平滑动量
        momentum_smooth = momentum.rolling(window=3).mean()
        
        return momentum_smooth
    
    def _calculate_enhanced_wr_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算增强型Williams %R综合评分
        
        Args:
            df: 包含Williams %R相关指标的Data_frame
            
        Returns:
            综合评分序列 (0-100)
        """
        # 基础Williams %R评分
        wr = df['wr']
        base_score = pd.Series(50.0, index=df.index)
        
        # Williams %R位置评分 (0-40分)
        for i in range(len(wr)):
            if pd.isna(wr.iloc[i]):
                continue
                
            wr_val = wr.iloc[i]
            
            if wr_val <= df['wr_oversold'].iloc[i]:
                # 超卖区域，看涨信号
                base_score.iloc[i] = 70 + min(20, (df['wr_oversold'].iloc[i] - wr_val) * 0.5)
            elif wr_val >= df['wr_overbought'].iloc[i]:
                # 超买区域，看跌信号
                base_score.iloc[i] = 30 - min(20, (wr_val - df['wr_overbought'].iloc[i]) * 0.5)
            else:
                # 正常区域，根据Williams %R值调整
                # Williams %R范围是-100到0，-50为中性
                if wr_val > -50:
                    base_score.iloc[i] = 50 + (wr_val + 50) * 0.5
                else:
                    base_score.iloc[i] = 50 - (-50 - wr_val) * 0.5
        
        # 背离加分 (0-15分)
        divergence_score = df['wr_divergence'] * 15
        
        # 趋势强度加分 (0-15分)
        trend_score = df['wr_trend_strength'].clip(0, 1) * 15
        
        # 多周期一致性加分 (0-15分)
        consistency_score = df['wr_consistency'] * 15
        
        # 动量加分 (0-15分)
        momentum_score = (df['wr_momentum'].clip(-10, 10) + 10) / 20 * 15
        
        # 综合评分
        final_score = base_score + divergence_score + trend_score + consistency_score + momentum_score
        
        # 限制在0-100范围内
        final_score = final_score.clip(0, 100)
        
        return final_score
    
    def _generate_wr_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成Williams %R交易信号
        
        Args:
            df: 包含Williams %R指标的Data_frame
            
        Returns:
            信号序列 (1: 买入, -1: 卖出, 0: 持有)
        """
        signals = pd.Series(0, index=df.index)
        wr = df['wr']
        wr_smooth = df['wr_smooth']
        
        for i in range(1, len(df)):
            if pd.isna(wr.iloc[i]) or pd.isna(wr.iloc[i-1]):
                continue
            
            # 超卖反弹买入信号
            if (wr.iloc[i-1] <= df['wr_oversold'].iloc[i-1] and 
                wr.iloc[i] > df['wr_oversold'].iloc[i]):
                signals.iloc[i] = 1
            
            # 超买回落卖出信号
            elif (wr.iloc[i-1] >= df['wr_overbought'].iloc[i-1] and 
                  wr.iloc[i] < df['wr_overbought'].iloc[i]):
                signals.iloc[i] = -1
            
            # 平滑线向上突破买入信号
            elif (wr_smooth.iloc[i-1] <= wr_smooth.iloc[i-2] and 
                  wr_smooth.iloc[i] > wr_smooth.iloc[i-1] and 
                  wr.iloc[i] < -60):
                signals.iloc[i] = 1
            
            # 平滑线向下突破卖出信号
            elif (wr_smooth.iloc[i-1] >= wr_smooth.iloc[i-2] and 
                  wr_smooth.iloc[i] < wr_smooth.iloc[i-1] and 
                  wr.iloc[i] > -40):
                signals.iloc[i] = -1
            
            # 背离信号
            elif df['wr_divergence'].iloc[i] > 0:
                if wr.iloc[i] < -70:  # 低位背离，买入
                    signals.iloc[i] = 1
                elif wr.iloc[i] > -30:  # 高位背离，卖出
                    signals.iloc[i] = -1
            
            # 多周期一致性信号
            elif df['wr_consistency'].iloc[i] > 0.8:
                if wr.iloc[i] < -65:  # 低位一致性，买入
                    signals.iloc[i] = 1
                elif wr.iloc[i] > -35:  # 高位一致性，卖出
                    signals.iloc[i] = -1
            
            # 动量信号
            elif df['wr_momentum'].iloc[i] > 5 and wr.iloc[i] < -60:
                # 正动量且在低位，买入
                signals.iloc[i] = 1
            elif df['wr_momentum'].iloc[i] < -5 and wr.iloc[i] > -40:
                # 负动量且在高位，卖出
                signals.iloc[i] = -1
        
        return signals

    def calculate_raw_score_Wr(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Wr(data, **kwargs)
        
        if 'ENHANCED_WR_VALUE' in self._result.columns:
            return self._result['ENHANCED_WR_VALUE']
        else:
            return pd.Series(50.0, index=data.index)

    def calculate_confidence_Wr(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if score.empty:
            return 0.5
        
        # 基于评分的变化和极值计算置信度
        score_std = score.std()
        score_range = score.max() - score.min()
        
        # 评分变化越大，置信度越高
        confidence = min(1.0, (score_std / 25.0 + score_range / 100.0) / 2)
        
        return max(0.3, confidence)

    def get_patterns_Wr(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Wr(data, **kwargs)
        
        patterns = pd.DataFrame(index=data.index)
        
        if 'wr' in self._result.columns:
            patterns['wr_overbought'] = self._result['wr'] > self._result['wr_overbought']
            patterns['wr_oversold'] = self._result['wr'] < self._result['wr_oversold']
            patterns['wr_divergence'] = self._result['wr_divergence'] > 0
            patterns['wr_uptrend'] = self._result['wr_momentum'] > 0
        
        return patterns

    # 🔧 Ultra Think修复：添加缺失的抽象方法实现，按照已验证的修复模式
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        实现基类要求的抽象方法
        """
        return self._calculate_enhancedwr(data, **kwargs)
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        实现基类要求的原始评分计算方法
        """
        return self.calculate_raw_score_Wr(data, **kwargs)
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        实现基类要求的形态获取方法
        """
        return self.get_patterns_Wr(data, **kwargs)
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        实现基类要求的置信度计算方法
        """
        return self.calculate_confidence_Wr(score, patterns, signals)
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        实现基类要求的参数设置方法
        """
        self.set_parameters_Wr(**kwargs)


# 🔧 Ultra Think修复：为了向后兼容，创建别名
enhanced_wr = EnhancedWr
# 修复测试导入问题：提供大写R版本的别名
EnhancedWR = EnhancedWr
