#!/usr/bin/env python3
"""
增强型RSI指标

实现真正的RSI计算逻辑，包括多周期分析、超买超卖判断、背离检测等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class EnhancedRsi(BaseIndicator, PatternSignalMixin):
    """
    增强型RSI指标
    
    在标准RSI基础上增加了多周期分析、背离检测、超买超卖区间优化等功能
    """
    
    def __init__(self, 
                 period: int = 14,
                 overbought: float = 70.0,
                 oversold: float = 30.0,
                 multi_periods: List[int] = None,
                 adaptive_thresholds: bool = True,
                 **kwargs):
        """
        初始化增强型RSI指标
        
        Args:
            period: RSI计算周期，默认14
            overbought: 超买阈值，默认70
            oversold: 超卖阈值，默认30
            multi_periods: 多周期分析，默认[9, 14, 21]
            adaptive_thresholds: 是否使用自适应阈值，默认True
            **kwargs: 其他参数
        """
        super().__init__()
        self.name = "ENHANCED_RSI"
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.multi_periods = multi_periods or [9, 14, 21]
        self.adaptive_thresholds = adaptive_thresholds
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_enhancedrsi()
        
        # 应用用户参数
        self.set_parameters_Rsi(**kwargs)
    
    def _get_default_parameters_enhancedrsi(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period": 14,
            "overbought": 70.0,
            "oversold": 30.0
        }
    
    def set_parameters_Rsi(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ENHANCED_RSI', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)
            self.overbought = params.get('overbought', 70.0)
            self.oversold = params.get('oversold', 30.0)
                    
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            self.period = 14
            self.overbought = 70.0
            self.oversold = 30.0
    
    def calculate_Rsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算增强型RSI指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了增强型RSI指标的Data_frame
        """
        result = self._calculate_enhancedrsi(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_enhancedrsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算增强型RSI指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了增强型RSI指标的Data_frame
        """
        if len(data) < max(self.multi_periods) + 1:
            # 数据不足，返回空结果
            df = data.copy()
            df['ENHANCED_RSI_VALUE'] = 50.0
            df['rsi_signal'] = 0
            df['buy_signal'] = False
            df['sell_signal'] = False
            df['hold_signal'] = True
            return df
        
        df = data.copy()
        
        # 计算主要RSI
        df['rsi'] = self._calculate_rsi_Enhanced_Rsi(df['close'], self.period)
        
        # 计算多周期RSI
        for period in self.multi_periods:
            df[f'rsi_{period}'] = self._calculate_rsi_Enhanced_Rsi(df['close'], period)
        
        # 计算自适应阈值
        if self.adaptive_thresholds:
            df['rsi_overbought'], df['rsi_oversold'] = self._calculate_adaptive_thresholds_Enhanced_Rsi(df['rsi'])
        else:
            df['rsi_overbought'] = self.overbought
            df['rsi_oversold'] = self.oversold
        
        # 计算RSI背离
        df['rsi_divergence'] = self._calculate_rsi_divergence(df['close'], df['rsi'])
        
        # 计算RSI趋势强度
        df['rsi_trend_strength'] = self._calculate_rsi_trend_strength(df['rsi'])
        
        # 计算多周期一致性
        df['rsi_consistency'] = self._calculate_multi_period_consistency_Enhanced_Rsi(df)
        
        # 计算综合评分
        df['ENHANCED_RSI_VALUE'] = self._calculate_enhanced_rsi_score(df)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)
        
        # 重写专用信号逻辑：基于RSI的买入信号
        df['rsi_signal'] = self._generate_rsi_signals(df)
        df['buy_signal'] = df['rsi_signal'] > 0
        df['sell_signal'] = df['rsi_signal'] < 0
        df['hold_signal'] = df['rsi_signal'] == 0
        
        return df
    
    def _calculate_rsi_Enhanced_Rsi(self, close: pd.Series, period: int) -> pd.Series:
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
    
    def _calculate_adaptive_thresholds_Enhanced_Rsi(self, rsi: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算自适应超买超卖阈值
        
        Args:
            rsi: RSI值序列
            
        Returns:
            (超买阈值序列, 超卖阈值序列)
        """
        # 使用滚动统计计算动态阈值
        window = 50
        rsi_mean = rsi.rolling(window=window).mean()
        rsi_std = rsi.rolling(window=window).std()
        
        # 动态阈值：均值 ± 1.5倍标准差
        overbought = rsi_mean + 1.5 * rsi_std
        oversold = rsi_mean - 1.5 * rsi_std
        
        # 限制阈值范围
        overbought = overbought.clip(65, 85)
        oversold = oversold.clip(15, 35)
        
        return overbought, oversold
    
    def _calculate_rsi_divergence(self, close: pd.Series, rsi: pd.Series) -> pd.Series:
        """
        计算RSI背离信号
        
        Args:
            close: 收盘价序列
            rsi: RSI值序列
            
        Returns:
            背离信号序列
        """
        # 简化的背离检测：比较价格和RSI的趋势方向
        price_trend = close.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        rsi_trend = rsi.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        # 背离：价格和RSI趋势方向相反
        divergence = (price_trend != rsi_trend).astype(int)
        
        return divergence
    
    def _calculate_rsi_trend_strength(self, rsi: pd.Series) -> pd.Series:
        """
        计算RSI趋势强度
        
        Args:
            rsi: RSI值序列
            
        Returns:
            趋势强度序列
        """
        # 计算RSI的移动平均和标准差
        rsi_ma = rsi.rolling(window=14).mean()
        rsi_std = rsi.rolling(window=14).std()
        
        # 趋势强度：RSI偏离移动平均的程度
        trend_strength = abs(rsi - rsi_ma) / (rsi_std + 1e-10)
        
        return trend_strength
    
    def _calculate_multi_period_consistency_Enhanced_Rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        计算多周期RSI一致性
        
        Args:
            df: 包含多周期RSI的Data_frame
            
        Returns:
            一致性评分序列
        """
        # 计算多周期RSI的一致性
        rsi_columns = [f'rsi_{period}' for period in self.multi_periods]
        
        consistency_scores = []
        for i in range(len(df)):
            rsi_values = [df[col].iloc[i] for col in rsi_columns if col in df.columns]
            if len(rsi_values) > 1:
                # 计算RSI值的标准差，标准差越小一致性越高
                std_dev = np.std(rsi_values)
                consistency = max(0, 1 - std_dev / 50)  # 归一化到0-1
            else:
                consistency = 0.5
            consistency_scores.append(consistency)
        
        return pd.Series(consistency_scores, index=df.index)
    
    def _calculate_enhanced_rsi_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算增强型RSI综合评分
        
        Args:
            df: 包含RSI相关指标的Data_frame
            
        Returns:
            综合评分序列 (0-100)
        """
        # 基础RSI评分
        rsi = df['rsi']
        base_score = pd.Series(50.0, index=df.index)
        
        # RSI位置评分 (0-40分)
        for i in range(len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue
                
            if rsi.iloc[i] <= df['rsi_oversold'].iloc[i]:
                # 超卖区域，看涨信号
                base_score.iloc[i] = 70 + min(20, (df['rsi_oversold'].iloc[i] - rsi.iloc[i]) * 2)
            elif rsi.iloc[i] >= df['rsi_overbought'].iloc[i]:
                # 超买区域，看跌信号
                base_score.iloc[i] = 30 - min(20, (rsi.iloc[i] - df['rsi_overbought'].iloc[i]) * 2)
            else:
                # 正常区域，根据RSI值调整
                if rsi.iloc[i] > 50:
                    base_score.iloc[i] = 50 + (rsi.iloc[i] - 50) * 0.5
                else:
                    base_score.iloc[i] = 50 - (50 - rsi.iloc[i]) * 0.5
        
        # 背离加分 (0-15分)
        divergence_score = df['rsi_divergence'] * 15
        
        # 趋势强度加分 (0-15分)
        trend_score = df['rsi_trend_strength'].clip(0, 1) * 15
        
        # 多周期一致性加分 (0-15分)
        consistency_score = df['rsi_consistency'] * 15
        
        # 综合评分
        final_score = base_score + divergence_score + trend_score + consistency_score
        
        # 限制在0-100范围内
        final_score = final_score.clip(0, 100)
        
        return final_score
    
    def _generate_rsi_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成RSI交易信号
        
        Args:
            df: 包含RSI指标的Data_frame
            
        Returns:
            信号序列 (1: 买入, -1: 卖出, 0: 持有)
        """
        signals = pd.Series(0, index=df.index)
        rsi = df['rsi']
        
        for i in range(1, len(df)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
                continue
            
            # 超卖反弹买入信号
            if (rsi.iloc[i-1] <= df['rsi_oversold'].iloc[i-1] and 
                rsi.iloc[i] > df['rsi_oversold'].iloc[i]):
                signals.iloc[i] = 1
            
            # 超买回落卖出信号
            elif (rsi.iloc[i-1] >= df['rsi_overbought'].iloc[i-1] and 
                  rsi.iloc[i] < df['rsi_overbought'].iloc[i]):
                signals.iloc[i] = -1
            
            # 背离信号
            elif df['rsi_divergence'].iloc[i] > 0:
                if rsi.iloc[i] < 40:  # 低位背离，买入
                    signals.iloc[i] = 1
                elif rsi.iloc[i] > 60:  # 高位背离，卖出
                    signals.iloc[i] = -1
            
            # 多周期一致性信号
            elif df['rsi_consistency'].iloc[i] > 0.8:
                if rsi.iloc[i] < 45:  # 低位一致性，买入
                    signals.iloc[i] = 1
                elif rsi.iloc[i] > 55:  # 高位一致性，卖出
                    signals.iloc[i] = -1
        
        return signals
    
    def calculate_raw_score_Rsi(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Rsi(data, **kwargs)
        
        if 'ENHANCED_RSI_VALUE' in self._result.columns:
            return self._result['ENHANCED_RSI_VALUE']
        else:
            return pd.Series(50.0, index=data.index)
    
    def calculate_confidence_Rsi(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if score.empty:
            return 0.5
        
        # 基于评分的变化和极值计算置信度
        score_std = score.std()
        score_range = score.max() - score.min()
        
        # 评分变化越大，置信度越高
        confidence = min(1.0, (score_std / 25.0 + score_range / 100.0) / 2)
        
        return max(0.3, confidence)
    
    def get_patterns_Rsi(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Rsi(data, **kwargs)
        
        patterns = pd.DataFrame(index=data.index)
        
        if 'rsi' in self._result.columns:
            patterns['rsi_overbought'] = self._result['rsi'] > self._result['rsi_overbought']
            patterns['rsi_oversold'] = self._result['rsi'] < self._result['rsi_oversold']
            patterns['rsi_divergence'] = self._result['rsi_divergence'] > 0
        
        return patterns

    def get_pattern_info(self) -> Dict[str, Any]:
        """
        获取指标模式信息
        
        Returns:
            Dict[str, Any]: 模式信息字典
        """
        return {
            'name': self.name,
            'type': 'ENHANCED_RSI',
            'period': self.period,
            'overbought': self.overbought,
            'oversold': self.oversold,
            'multi_periods': self.multi_periods,
            'adaptive_thresholds': self.adaptive_thresholds,
            'category': 'oscillator',
            'description': '增强型RSI指标，包含多周期分析和背离检测'
        }
    
    def ensure_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        确保数据包含必要的列
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            pd.DataFrame: 验证后的数据
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"EnhancedRSI缺少必需列: {missing_columns}")
        
        # 检查数据长度
        min_length = max(self.multi_periods) + 10
        if len(data) < min_length:
            raise ValueError(f"EnhancedRSI需要至少{min_length}行数据，当前只有{len(data)}行")
        
        return data
    
    def get_market_environment(self) -> str:
        """
        获取当前市场环境
        
        Returns:
            str: 市场环境类型
        """
        return getattr(self, 'market_environment', 'normal')
    
    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment: 市场环境类型
                - 'bull_market': 牛市
                - 'bear_market': 熊市  
                - 'sideways_market': 震荡市
                - 'volatile_market': 高波动市
                - 'normal': 正常市场
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment
        
        # 根据市场环境调整超买超卖阈值
        if environment == 'bull_market':
            self.overbought = 80.0
            self.oversold = 40.0
        elif environment == 'bear_market':
            self.overbought = 60.0
            self.oversold = 20.0
        elif environment == 'volatile_market':
            self.overbought = 75.0
            self.oversold = 25.0
        else:
            self.overbought = 70.0
            self.oversold = 30.0
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return 'ENHANCED_RSI'


# 为了向后兼容，创建别名
enhanced_rsi = EnhancedRsi
