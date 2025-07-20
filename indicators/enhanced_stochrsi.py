import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class EnhancedStochasticRSI(BaseIndicator, PatternSignalMixin):
    """
    增强型Stoch_rSI指标

    在标准Stoch_rSI基础上增加了多周期分析、自适应阈值、背离检测等功能
    """

    def __init__(self, 
                 rsi_period: int = 14,
                 stoch_period: int = 14,
                 k_period: int = 3,
                 d_period: int = 3,
                 overbought: float = 80.0,
                 oversold: float = 20.0,
                 multi_periods: List[Tuple[int, int]] = None,
                 adaptive_thresholds: bool = True,
                 **kwargs):
        """
        初始化增强型Stoch_rSI指标

        Args:
            rsi_period: RSI计算周期，默认14
            stoch_period: Stochastic计算周期，默认14
            k_period: %K平滑周期，默认3
            d_period: %D平滑周期，默认3
            overbought: 超买阈值，默认80
            oversold: 超卖阈值，默认20
            multi_periods: 多周期分析，默认[(14,14), (21,21), (9,9)]
            adaptive_thresholds: 是否使用自适应阈值，默认True
            **kwargs: 其他参数
        """
        super().__init__()
        self.name = "ENHANCED_STOCHRSI"
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
        self.multi_periods = multi_periods or [(14, 14), (21, 21), (9, 9)]
        self.adaptive_thresholds = adaptive_thresholds

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_enhancedstochrsi()

        # 应用用户参数
        self.set_parameters_Stochrsi_Enhanced_Stochrsi(**kwargs)

    def _get_default_parameters_enhancedstochrsi(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "rsi_period": 14,
            "stoch_period": 14,
            "k_period": 3,
            "d_period": 3,
            "overbought": 80.0,
            "oversold": 20.0
        }

    def set_parameters_Stochrsi_Enhanced_Stochrsi(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ENHANCED_STOCHRSI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass

        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass

        # 设置参数
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.stoch_period = kwargs.get('stoch_period', 14)
        self.k_period = kwargs.get('k_period', 3)
        self.d_period = kwargs.get('d_period', 3)
        self.overbought = kwargs.get('overbought', 80.0)
        self.oversold = kwargs.get('oversold', 20.0)

    def calculate_Stochrsi_Enhanced_Stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算增强型Stoch_rSI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了增强型Stoch_rSI指标的Data_frame
        """
        result = self._calculate_enhancedstochrsi(data, **kwargs)
        self._result = result
        return result

    def _calculate_enhancedstochrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算增强型Stoch_rSI指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了增强型Stoch_rSI指标的Data_frame
        """
        min_length = max(self.rsi_period, self.stoch_period) + max(self.k_period, self.d_period) + 10
        if len(data) < min_length:
            # 数据不足，返回空结果
            df = data.copy()
            df['ENHANCED_STOCHRSI_VALUE'] = 50.0
            df['stochrsi_signal'] = 0
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['hold_signal'] = True
            return df

        df = data.copy()

        # 计算主要StochRSI
        df = self._calculate_stochrsi(df, self.rsi_period, self.stoch_period, self.k_period, self.d_period)
        
        # 计算多周期StochRSI
        for rsi_p, stoch_p in self.multi_periods:
            df = self._calculate_multi_period_stochrsi(df, rsi_p, stoch_p, self.k_period, self.d_period)
        
        # 计算自适应阈值
        if self.adaptive_thresholds:
            df['stochrsi_overbought'], df['stochrsi_oversold'] = self._calculate_adaptive_thresholds_Enhanced_Stochrsi(df['stochrsi_k'])
        else:
            df['stochrsi_overbought'] = self.overbought
            df['stochrsi_oversold'] = self.oversold
        
        # 计算StochRSI背离
        df['stochrsi_divergence'] = self._calculate_stochrsi_divergence(df['close'], df['stochrsi_k'])
        
        # 计算StochRSI趋势强度
        df['stochrsi_trend_strength'] = self._calculate_stochrsi_trend_strength(df['stochrsi_k'], df['stochrsi_d'])
        
        # 计算多周期一致性
        df['stochrsi_consistency'] = self._calculate_multi_period_consistency_Enhanced_Stochrsi(df)
        
        # 计算综合评分
        df['ENHANCED_STOCHRSI_VALUE'] = self._calculate_enhanced_stochrsi_score(df)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于StochRSI的买入信号
        df['stochrsi_signal'] = self._generate_stochrsi_signals(df)
        df['buy_signal'] = df['stochrsi_signal'] > 0
        df['sell_signal'] = df['stochrsi_signal'] < 0
        df['hold_signal'] = df['stochrsi_signal'] == 0

        return df
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """
        计算RSI指标
        
        Args:
            close: 收盘价序列
            period: 计算周期
            
        Returns:
            RSI值序列
        """
        # 计算价格变化
        delta = close.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均收益和平均损失
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算相对强弱指标
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochrsi(self, df: pd.DataFrame, rsi_period: int, stoch_period: int, k_period: int, d_period: int) -> pd.DataFrame:
        """
        计算Stoch_rSI指标
        
        Args:
            df: 数据Data_frame
            rsi_period: RSI周期
            stoch_period: Stochastic周期
            k_period: %K平滑周期
            d_period: %D平滑周期
            
        Returns:
            添加了Stoch_rSI指标的Data_frame
        """
        # 计算RSI
        rsi = self._calculate_rsi(df['close'], rsi_period)
        
        # 计算RSI的最高值和最低值
        rsi_high = rsi.rolling(window=stoch_period).max()
        rsi_low = rsi.rolling(window=stoch_period).min()
        
        # 计算StochRSI
        stochrsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)
        
        # 计算%K和%D
        stochrsi_k = stochrsi.rolling(window=k_period).mean()
        stochrsi_d = stochrsi_k.rolling(window=d_period).mean()
        
        # 保存结果
        df['rsi'] = rsi
        df['stochrsi'] = stochrsi
        df['stochrsi_k'] = stochrsi_k
        df['stochrsi_d'] = stochrsi_d
        
        return df
    
    def _calculate_multi_period_stochrsi(self, df: pd.DataFrame, rsi_period: int, stoch_period: int, k_period: int, d_period: int) -> pd.DataFrame:
        """
        计算多周期Stoch_rSI指标
        
        Args:
            df: 数据Data_frame
            rsi_period: RSI周期
            stoch_period: Stochastic周期
            k_period: %K平滑周期
            d_period: %D平滑周期
            
        Returns:
            添加了多周期Stoch_rSI指标的Data_frame
        """
        # 计算RSI
        rsi = self._calculate_rsi(df['close'], rsi_period)
        
        # 计算RSI的最高值和最低值
        rsi_high = rsi.rolling(window=stoch_period).max()
        rsi_low = rsi.rolling(window=stoch_period).min()
        
        # 计算StochRSI
        stochrsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10)
        
        # 计算%K和%D
        stochrsi_k = stochrsi.rolling(window=k_period).mean()
        stochrsi_d = stochrsi_k.rolling(window=d_period).mean()
        
        # 保存结果
        df[f'stochrsi_k_{rsi_period}_{stoch_period}'] = stochrsi_k
        df[f'stochrsi_d_{rsi_period}_{stoch_period}'] = stochrsi_d
        
        return df
    
    def _calculate_adaptive_thresholds_Enhanced_Stochrsi(self, stochrsi_k: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算自适应超买超卖阈值
        
        Args:
            stochrsi_k: Stoch_rSI %K值序列
            
        Returns:
            (超买阈值序列, 超卖阈值序列)
        """
        # 使用滚动统计计算动态阈值
        window = 50
        stochrsi_mean = stochrsi_k.rolling(window=window).mean()
        stochrsi_std = stochrsi_k.rolling(window=window).std()
        
        # 动态阈值：均值 ± 1.5倍标准差
        overbought = stochrsi_mean + 1.5 * stochrsi_std
        oversold = stochrsi_mean - 1.5 * stochrsi_std
        
        # 限制阈值范围
        overbought = overbought.clip(70, 90)
        oversold = oversold.clip(10, 30)
        
        return overbought, oversold
    
    def _calculate_stochrsi_divergence(self, close: pd.Series, stochrsi_k: pd.Series) -> pd.Series:
        """
        计算Stoch_rSI背离信号
        
        Args:
            close: 收盘价序列
            stochrsi_k: Stoch_rSI %K值序列
            
        Returns:
            背离信号序列
        """
        # 简化的背离检测：比较价格和StochRSI的趋势方向
        price_trend = close.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        stochrsi_trend = stochrsi_k.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        # 背离：价格和StochRSI趋势方向相反
        divergence = (price_trend != stochrsi_trend).astype(int)
        
        return divergence
    
    def _calculate_stochrsi_trend_strength(self, stochrsi_k: pd.Series, stochrsi_d: pd.Series) -> pd.Series:
        """
        计算Stoch_rSI趋势强度
        
        Args:
            stochrsi_k: Stoch_rSI %K值序列
            stochrsi_d: Stoch_rSI %D值序列
            
        Returns:
            趋势强度序列
        """
        # 计算K和D的差值
        kd_diff = abs(stochrsi_k - stochrsi_d)
        
        # 计算K线的变化率
        k_change = abs(stochrsi_k.diff())
        
        # 趋势强度：K和D的差值 + K线变化率
        trend_strength = (kd_diff + k_change) / 2
        
        # 归一化到0-1
        trend_strength = trend_strength / 100
        
        return trend_strength
    
    def _calculate_multi_period_consistency_Enhanced_Stochrsi(self, df: pd.DataFrame) -> pd.Series:
        """
        计算多周期Stoch_rSI一致性
        
        Args:
            df: 包含多周期Stoch_rSI的Data_frame
            
        Returns:
            一致性评分序列
        """
        # 计算多周期StochRSI的一致性
        stochrsi_columns = [f'stochrsi_k_{rsi_p}_{stoch_p}' for rsi_p, stoch_p in self.multi_periods]
        
        consistency_scores = []
        for i in range(len(df)):
            stochrsi_values = [df[col].iloc[i] for col in stochrsi_columns if col in df.columns]
            if len(stochrsi_values) > 1:
                # 计算StochRSI值的标准差，标准差越小一致性越高
                std_dev = np.std(stochrsi_values)
                consistency = max(0, 1 - std_dev / 50)  # 归一化到0-1
            else:
                consistency = 0.5
            consistency_scores.append(consistency)
        
        return pd.Series(consistency_scores, index=df.index)
    
    def _calculate_enhanced_stochrsi_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算增强型Stoch_rSI综合评分
        
        Args:
            df: 包含Stoch_rSI相关指标的Data_frame
            
        Returns:
            综合评分序列 (0-100)
        """
        # 基础StochRSI评分
        stochrsi_k = df['stochrsi_k']
        stochrsi_d = df['stochrsi_d']
        base_score = pd.Series(50.0, index=df.index)
        
        # StochRSI位置评分 (0-40分)
        for i in range(len(stochrsi_k)):
            if pd.isna(stochrsi_k.iloc[i]) or pd.isna(stochrsi_d.iloc[i]):
                continue
                
            k_val = stochrsi_k.iloc[i]
            d_val = stochrsi_d.iloc[i]
            
            if k_val <= df['stochrsi_oversold'].iloc[i]:
                # 超卖区域，看涨信号
                base_score.iloc[i] = 70 + min(20, (df['stochrsi_oversold'].iloc[i] - k_val) * 0.5)
            elif k_val >= df['stochrsi_overbought'].iloc[i]:
                # 超买区域，看跌信号
                base_score.iloc[i] = 30 - min(20, (k_val - df['stochrsi_overbought'].iloc[i]) * 0.5)
            else:
                # 正常区域，根据K和D的关系调整
                if k_val > d_val:
                    base_score.iloc[i] = 50 + (k_val - d_val) * 0.5
                else:
                    base_score.iloc[i] = 50 - (d_val - k_val) * 0.5
        
        # 背离加分 (0-15分)
        divergence_score = df['stochrsi_divergence'] * 15
        
        # 趋势强度加分 (0-15分)
        trend_score = df['stochrsi_trend_strength'].clip(0, 1) * 15
        
        # 多周期一致性加分 (0-15分)
        consistency_score = df['stochrsi_consistency'] * 15
        
        # 综合评分
        final_score = base_score + divergence_score + trend_score + consistency_score
        
        # 限制在0-100范围内
        final_score = final_score.clip(0, 100)
        
        return final_score
    
    def _generate_stochrsi_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成Stoch_rSI交易信号
        
        Args:
            df: 包含Stoch_rSI指标的Data_frame
            
        Returns:
            信号序列 (1: 买入, -1: 卖出, 0: 持有)
        """
        signals = pd.Series(0, index=df.index)
        stochrsi_k = df['stochrsi_k']
        stochrsi_d = df['stochrsi_d']
        
        for i in range(1, len(df)):
            if (pd.isna(stochrsi_k.iloc[i]) or pd.isna(stochrsi_d.iloc[i]) or 
                pd.isna(stochrsi_k.iloc[i-1]) or pd.isna(stochrsi_d.iloc[i-1])):
                continue
            
            # 超卖反弹买入信号
            if (stochrsi_k.iloc[i-1] <= df['stochrsi_oversold'].iloc[i-1] and 
                stochrsi_k.iloc[i] > df['stochrsi_oversold'].iloc[i]):
                signals.iloc[i] = 1
            
            # 超买回落卖出信号
            elif (stochrsi_k.iloc[i-1] >= df['stochrsi_overbought'].iloc[i-1] and 
                  stochrsi_k.iloc[i] < df['stochrsi_overbought'].iloc[i]):
                signals.iloc[i] = -1
            
            # K线上穿D线买入信号
            elif (stochrsi_k.iloc[i-1] <= stochrsi_d.iloc[i-1] and 
                  stochrsi_k.iloc[i] > stochrsi_d.iloc[i] and 
                  stochrsi_k.iloc[i] < 50):
                signals.iloc[i] = 1
            
            # K线下穿D线卖出信号
            elif (stochrsi_k.iloc[i-1] >= stochrsi_d.iloc[i-1] and 
                  stochrsi_k.iloc[i] < stochrsi_d.iloc[i] and 
                  stochrsi_k.iloc[i] > 50):
                signals.iloc[i] = -1
            
            # 背离信号
            elif df['stochrsi_divergence'].iloc[i] > 0:
                if stochrsi_k.iloc[i] < 30:  # 低位背离，买入
                    signals.iloc[i] = 1
                elif stochrsi_k.iloc[i] > 70:  # 高位背离，卖出
                    signals.iloc[i] = -1
            
            # 多周期一致性信号
            elif df['stochrsi_consistency'].iloc[i] > 0.8:
                if stochrsi_k.iloc[i] < 35:  # 低位一致性，买入
                    signals.iloc[i] = 1
                elif stochrsi_k.iloc[i] > 65:  # 高位一致性，卖出
                    signals.iloc[i] = -1
        
        return signals

    def calculate_raw_score_Stochrsi_Enhanced_Stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Stochrsi_Enhanced_Stochrsi(data, **kwargs)
        
        if 'ENHANCED_STOCHRSI_VALUE' in self._result.columns:
            return self._result['ENHANCED_STOCHRSI_VALUE']
        else:
            return pd.Series(50.0, index=data.index)

    def calculate_confidence_Stochrsi_Enhanced_Stochrsi(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if score.empty:
            return 0.5
        
        # 基于评分的变化和极值计算置信度
        score_std = score.std()
        score_range = score.max() - score.min()
        
        # 评分变化越大，置信度越高
        confidence = min(1.0, (score_std / 25.0 + score_range / 100.0) / 2)
        
        return max(0.3, confidence)

    def get_patterns_Stochrsi_Enhanced_Stochrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Stochrsi_Enhanced_Stochrsi(data, **kwargs)
        
        patterns = pd.DataFrame(index=data.index)
        
        if 'stochrsi_k' in self._result.columns:
            patterns['stochrsi_overbought'] = self._result['stochrsi_k'] > self._result['stochrsi_overbought']
            patterns['stochrsi_oversold'] = self._result['stochrsi_k'] < self._result['stochrsi_oversold']
            patterns['stochrsi_divergence'] = self._result['stochrsi_divergence'] > 0
            patterns['stochrsi_k_cross_d'] = self._result['stochrsi_k'] > self._result['stochrsi_d']
        
        return patterns


# 为了向后兼容，创建别名
enhanced_stochastic_rsi = EnhancedStochasticRSI
