import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedStochasticRSI(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
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

        # 应用用户参数，但保留构造函数中已设置的参数
        if kwargs:
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

    @property
    def minimum_periods(self) -> int:
        """
        Enhanced StochRSI指标所需的最少数据周期数

        计算逻辑：基于参数 rsi_period(14), stoch_period(14), k_period(3), d_period(3) 计算

        Returns:
            int: 最少需要的数据周期数
        """
        return max(self.rsi_period, self.stoch_period) + max(self.k_period, self.d_period) + 10

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Enhanced StochRSI指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced StochRSI指标的DataFrame
        """
        return self.calculate_Stochrsi_Enhanced_Stochrsi(data, **kwargs)

    def _get_default_parameters_enhancedstochrsi(self) -> dict:
        """
        获取Enhanced StochRSI指标的默认参数

        Returns:
            dict: 默认参数字典
        """
        return {
            'rsi_period': 14,
            'stoch_period': 14,
            'k_period': 3,
            'd_period': 3,
            'overbought': 80.0,
            'oversold': 20.0,
            'multi_periods': [(14, 14), (21, 21), (9, 9)],
            'adaptive_thresholds': True
        }

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced StochRSI指标的DataFrame
        """
        return self._calculate_enhancedstochrsi(data)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        BaseIndicator要求的置信度计算方法

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        return self.calculate_confidence_Enhanced_Stochrsi(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列
        """
        return self.calculate_raw_score_Enhanced_Stochrsi(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态DataFrame
        """
        return self.get_patterns_Enhanced_Stochrsi(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Stochrsi_Enhanced_Stochrsi(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        BaseIndicator要求的默认参数获取方法

        Returns:
            dict: 默认参数字典
        """
        return self._get_default_parameters_enhancedstochrsi()

    def set_parameters(self, **kwargs):
        """
        标准参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Stochrsi_Enhanced_Stochrsi(**kwargs)

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取增强型StochRSI交易信号
        
        增强型StochRSI结合了多周期分析、自适应阈值、背离检测等功能，
        提供更精确的超买超卖信号和趋势反转信号。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        # 数据验证
        if not self._validate_signal_data(data):
            return self._get_default_signal()
        
        # 确保有增强型StochRSI计算结果，如果没有则先计算
        required_columns = ['ENHANCED_STOCHRSI_VALUE', 'stochrsi_k', 'stochrsi_d']
        has_stochrsi_data = any(col in data.columns for col in required_columns)
        
        if not has_stochrsi_data:
            try:
                data = self.calculate_Stochrsi_Enhanced_Stochrsi(data)
            except Exception as e:
                logger.warning(f"计算增强型StochRSI失败: {e}")
                return self._get_default_signal()
        
        if data.empty:
            return self._get_default_signal()
        
        # 初始化信号参数
        signal_type = "hold"
        strength = 0.5
        confidence = 0.6
        reason = "无明确StochRSI信号"
        
        # 分析增强型StochRSI信号
        stochrsi_analysis = self._analyze_enhanced_stochrsi_signal(data)
        
        if stochrsi_analysis['signal_detected']:
            signal_type = stochrsi_analysis['signal_type']
            base_strength = stochrsi_analysis['base_strength']
            base_confidence = stochrsi_analysis['base_confidence']
            base_reason = stochrsi_analysis['reason']
            
            strength = base_strength
            confidence = base_confidence
            reason = base_reason
            
            # 多周期一致性增强
            consistency_score = stochrsi_analysis.get('consistency_score', 0.5)
            if consistency_score > 0.8:
                strength = min(1.0, strength + 0.15)
                confidence = min(1.0, confidence + 0.15)
                reason += "，多周期一致性强"
            elif consistency_score < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，多周期分歧"
            
            # 背离检测增强
            if stochrsi_analysis.get('bullish_divergence', False):
                if signal_type == "buy":
                    strength = min(1.0, strength + 0.2)
                    confidence = min(1.0, confidence + 0.2)
                    reason += "，牛背离确认"
                else:
                    strength = max(0.3, strength - 0.1)
                    confidence = max(0.4, confidence - 0.1)
            elif stochrsi_analysis.get('bearish_divergence', False):
                if signal_type == "sell":
                    strength = min(1.0, strength + 0.2)
                    confidence = min(1.0, confidence + 0.2)
                    reason += "，熊背离确认"
                else:
                    strength = max(0.3, strength - 0.1)
                    confidence = max(0.4, confidence - 0.1)
            
            # 自适应阈值优化
            adaptive_quality = stochrsi_analysis.get('adaptive_quality', 0.5)
            if adaptive_quality > 0.8:
                confidence = min(1.0, confidence + 0.1)
                reason += "，自适应阈值精确"
            elif adaptive_quality < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，阈值适应性差"
            
            # 趋势强度评估
            trend_strength = stochrsi_analysis.get('trend_strength', 0.5)
            if trend_strength > 0.85:
                strength = min(1.0, strength + 0.1)
                confidence = min(1.0, confidence + 0.1)
                reason += "，趋势强劲"
            elif trend_strength < 0.3:
                strength = max(0.3, strength - 0.1)
                confidence = max(0.4, confidence - 0.1)
                reason += "，趋势疲弱"
            
            # 区间突破评估
            breakthrough_strength = stochrsi_analysis.get('breakthrough_strength', 0.0)
            if breakthrough_strength > 0.8:
                strength = min(1.0, strength + 0.15)
                confidence = min(1.0, confidence + 0.15)
                reason += "，强势突破"
            elif breakthrough_strength > 0.5:
                strength = min(1.0, strength + 0.05)
                confidence = min(1.0, confidence + 0.05)
                reason += "，温和突破"
        
        # 构建标准化信号字典
        metadata = {
            'indicator_type': 'enhanced_stochrsi',
            'signal_detected': stochrsi_analysis.get('signal_detected', False),
            'stochrsi_k': stochrsi_analysis.get('stochrsi_k', 50.0),
            'stochrsi_d': stochrsi_analysis.get('stochrsi_d', 50.0),
            'overbought_threshold': stochrsi_analysis.get('overbought_threshold', self.overbought),
            'oversold_threshold': stochrsi_analysis.get('oversold_threshold', self.oversold),
            'consistency_score': stochrsi_analysis.get('consistency_score', 0.5),
            'adaptive_quality': stochrsi_analysis.get('adaptive_quality', 0.5),
            'trend_strength': stochrsi_analysis.get('trend_strength', 0.5),
            'bullish_divergence': stochrsi_analysis.get('bullish_divergence', False),
            'bearish_divergence': stochrsi_analysis.get('bearish_divergence', False),
            'breakthrough_strength': stochrsi_analysis.get('breakthrough_strength', 0.0),
            'multi_period_signals': stochrsi_analysis.get('multi_period_signals', []),
            'enhanced_score': stochrsi_analysis.get('enhanced_score', 50.0)
        }
        
        return {
            'signal_type': signal_type,
            'strength': max(0.0, min(1.0, strength)),
            'confidence': max(0.0, min(1.0, confidence)),
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': metadata
        }

    def _analyze_enhanced_stochrsi_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析增强型StochRSI信号
        
        Args:
            data: 包含StochRSI数据的DataFrame
            
        Returns:
            Dict: 包含信号分析结果的字典
        """
        analysis = {
            'signal_detected': False,
            'signal_type': 'hold',
            'base_strength': 0.5,
            'base_confidence': 0.6,
            'reason': '无明确信号',
            'stochrsi_k': 50.0,
            'stochrsi_d': 50.0,
            'overbought_threshold': self.overbought,
            'oversold_threshold': self.oversold,
            'consistency_score': 0.5,
            'adaptive_quality': 0.5,
            'trend_strength': 0.5,
            'bullish_divergence': False,
            'bearish_divergence': False,
            'breakthrough_strength': 0.0,
            'multi_period_signals': [],
            'enhanced_score': 50.0
        }
        
        try:
            if len(data) < 10:
                return analysis
            
            # 获取最新的StochRSI值
            latest_k = data['stochrsi_k'].iloc[-1] if 'stochrsi_k' in data.columns else 50.0
            latest_d = data['stochrsi_d'].iloc[-1] if 'stochrsi_d' in data.columns else 50.0
            enhanced_value = data['ENHANCED_STOCHRSI_VALUE'].iloc[-1] if 'ENHANCED_STOCHRSI_VALUE' in data.columns else 50.0
            
            # 获取自适应阈值
            current_overbought = data['stochrsi_overbought'].iloc[-1] if 'stochrsi_overbought' in data.columns else self.overbought
            current_oversold = data['stochrsi_oversold'].iloc[-1] if 'stochrsi_oversold' in data.columns else self.oversold
            
            analysis.update({
                'stochrsi_k': latest_k,
                'stochrsi_d': latest_d,
                'overbought_threshold': current_overbought,
                'oversold_threshold': current_oversold,
                'enhanced_score': enhanced_value
            })
            
            # 1. 基本超买超卖信号
            signal_type = "hold"
            base_strength = 0.5
            base_confidence = 0.6
            reason = "StochRSI中性区域"
            
            # K线与D线交叉分析
            if len(data) >= 2:
                prev_k = data['stochrsi_k'].iloc[-2] if 'stochrsi_k' in data.columns else latest_k
                prev_d = data['stochrsi_d'].iloc[-2] if 'stochrsi_d' in data.columns else latest_d
                
                # 金叉买入信号（超卖区域K线上穿D线）
                if (latest_k > latest_d and prev_k <= prev_d and 
                    latest_k < current_oversold + 10):  # 在超卖区域附近
                    signal_type = "buy"
                    base_strength = 0.75
                    base_confidence = 0.8
                    reason = "StochRSI超卖区金叉买入信号"
                    analysis['signal_detected'] = True
                
                # 死叉卖出信号（超买区域K线下穿D线）
                elif (latest_k < latest_d and prev_k >= prev_d and 
                      latest_k > current_overbought - 10):  # 在超买区域附近
                    signal_type = "sell"
                    base_strength = 0.75
                    base_confidence = 0.8
                    reason = "StochRSI超买区死叉卖出信号"
                    analysis['signal_detected'] = True
                
                # 极端超卖反弹信号
                elif latest_k < current_oversold and latest_k > prev_k:
                    signal_type = "buy"
                    base_strength = 0.7
                    base_confidence = 0.75
                    reason = "StochRSI极端超卖反弹信号"
                    analysis['signal_detected'] = True
                
                # 极端超买回落信号
                elif latest_k > current_overbought and latest_k < prev_k:
                    signal_type = "sell"
                    base_strength = 0.7
                    base_confidence = 0.75
                    reason = "StochRSI极端超买回落信号"
                    analysis['signal_detected'] = True
            
            analysis.update({
                'signal_type': signal_type,
                'base_strength': base_strength,
                'base_confidence': base_confidence,
                'reason': reason
            })
            
            # 2. 多周期一致性分析
            consistency_score = self._calculate_multi_period_consistency(data)
            analysis['consistency_score'] = consistency_score
            
            # 3. 背离检测
            bullish_div, bearish_div = self._detect_stochrsi_divergence(data)
            analysis['bullish_divergence'] = bullish_div
            analysis['bearish_divergence'] = bearish_div
            
            # 4. 自适应阈值质量评估
            adaptive_quality = self._evaluate_adaptive_threshold_quality(data)
            analysis['adaptive_quality'] = adaptive_quality
            
            # 5. 趋势强度评估
            trend_strength = self._calculate_trend_strength(data)
            analysis['trend_strength'] = trend_strength
            
            # 6. 突破强度分析
            breakthrough_strength = self._analyze_breakthrough_strength(data, latest_k, current_overbought, current_oversold)
            analysis['breakthrough_strength'] = breakthrough_strength
            
        except Exception as e:
            logger.warning(f"增强型StochRSI信号分析失败: {e}")
        
        return analysis

    def _calculate_multi_period_consistency(self, data: pd.DataFrame) -> float:
        """计算多周期一致性评分"""
        try:
            # 检查是否有多周期数据
            multi_period_columns = [col for col in data.columns if 'stochrsi_' in col and ('_k' in col or '_d' in col)]
            if len(multi_period_columns) < 4:  # 至少需要主周期的K和D线
                return 0.5
            
            # 简化的一致性计算：检查各周期的方向一致性
            signals = []
            for col in multi_period_columns:
                if col.endswith('_k') and len(data) >= 2:
                    current = data[col].iloc[-1]
                    previous = data[col].iloc[-2]
                    signals.append(1 if current > previous else -1)
            
            if not signals:
                return 0.5
            
            # 计算方向一致性
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            
            consistency = max(positive_signals, negative_signals) / len(signals)
            return consistency
            
        except Exception:
            return 0.5

    def _detect_stochrsi_divergence(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        """检测StochRSI背离"""
        try:
            if len(data) < 20 or 'stochrsi_divergence' not in data.columns:
                return False, False
            
            recent_divergence = data['stochrsi_divergence'].tail(5).values
            
            # 检查是否有明显的背离信号
            bullish_divergence = any(val > 0.5 for val in recent_divergence if not pd.isna(val))
            bearish_divergence = any(val < -0.5 for val in recent_divergence if not pd.isna(val))
            
            return bullish_divergence, bearish_divergence
            
        except Exception:
            return False, False

    def _evaluate_adaptive_threshold_quality(self, data: pd.DataFrame) -> float:
        """评估自适应阈值质量"""
        try:
            if not self.adaptive_thresholds or len(data) < 10:
                return 0.5
            
            # 检查阈值的稳定性和有效性
            if 'stochrsi_overbought' in data.columns and 'stochrsi_oversold' in data.columns:
                overbought_values = data['stochrsi_overbought'].tail(10).values
                oversold_values = data['stochrsi_oversold'].tail(10).values
                
                # 计算阈值稳定性（变化幅度小表示质量高）
                ob_stability = 1.0 - min(1.0, np.std(overbought_values) / 10)
                os_stability = 1.0 - min(1.0, np.std(oversold_values) / 10)
                
                return (ob_stability + os_stability) / 2
            
            return 0.5
            
        except Exception:
            return 0.5

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """计算趋势强度"""
        try:
            if 'stochrsi_trend_strength' in data.columns and len(data) >= 1:
                return min(1.0, max(0.0, data['stochrsi_trend_strength'].iloc[-1]))
            return 0.5
        except Exception:
            return 0.5

    def _analyze_breakthrough_strength(self, data: pd.DataFrame, current_k: float, 
                                     overbought: float, oversold: float) -> float:
        """分析突破强度"""
        try:
            if len(data) < 5:
                return 0.0
            
            # 检查是否突破关键阈值
            if current_k > overbought:
                # 超买突破强度
                excess = (current_k - overbought) / (100 - overbought)
                return min(1.0, excess * 2)
            elif current_k < oversold:
                # 超卖突破强度
                excess = (oversold - current_k) / oversold
                return min(1.0, excess * 2)
            
            return 0.0
            
        except Exception:
            return 0.0

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        
        if data.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # 检查数据长度（增强型StochRSI需要更多数据）
        min_length = max(self.rsi_period, self.stoch_period) + max(self.k_period, self.d_period) + 10
        if len(data) < min_length:
            return False
        
        return True

    def _get_default_signal(self, reason: str = "数据不足或无StochRSI信号") -> Dict[str, Any]:
        """
        生成默认的持有信号
        
        Args:
            reason: 默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号字典
        """
        return {
            'signal_type': 'hold',
            'strength': 0.5,
            'confidence': 0.5,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {
                'indicator_type': 'enhanced_stochrsi',
                'signal_detected': False,
                'stochrsi_k': 50.0,
                'stochrsi_d': 50.0,
                'overbought_threshold': self.overbought,
                'oversold_threshold': self.oversold,
                'consistency_score': 0.5,
                'adaptive_quality': 0.5,
                'trend_strength': 0.5,
                'bullish_divergence': False,
                'bearish_divergence': False,
                'breakthrough_strength': 0.0,
                'multi_period_signals': [],
                'enhanced_score': 50.0
            }
        }


# 为了向后兼容，创建别名
enhanced_stochastic_rsi = EnhancedStochasticRSI
ENHANCED_STOCHRSI = EnhancedStochasticRSI
