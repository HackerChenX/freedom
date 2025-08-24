#!/usr/bin/env python3
from utils.dependency_injection import get_logger
"""
STOCK_VIX 指标 (股票波动率指标)

基于历史价格波动计算的股票波动率指标，类似于VIX指数的概念。
用于衡量股票价格的波动程度和市场恐慌情绪。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class StockVix(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    STOCK_VIX 指标 (股票波动率指标)
    
    特点:
    1. 基于历史价格波动计算隐含波动率
    2. 反映市场对未来波动的预期
    3. 高VIX值表示高波动和恐慌情绪
    4. 低VIX值表示低波动和平静市场
    
    计算方法:
    1. 计算对数收益率：ln(今日收盘价/昨日收盘价)
    2. 计算收益率的滚动标准差
    3. 年化波动率 = 标准差 * sqrt(252)
    4. VIX值 = 年化波动率 * 100
    
    参数:
    - period: 计算周期，默认为20
    - annualize_factor: 年化因子，默认为252（交易日）
    """
    
    def __init__(self, **kwargs):
        """
        初始化STOCK_VIX指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "STOCK_VIX"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_stockvix()
        
        # 应用用户参数
        self.set_parameters_Vix_Stock_Vix(**kwargs)
    
    def _get_default_parameters_stockvix(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 20, "annualize_factor": 252}
    
    def set_parameters_Vix_Stock_Vix(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('STOCK_VIX', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 20)
        self.annualize_factor = kwargs.get('annualize_factor', 252)
    
    def calculate_Vix_Stock_Vix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算STOCK_VIX指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了STOCK_VIX指标的Data_frame
        """
        result = self._calculate_stockvix(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_stockvix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算STOCK_VIX指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了STOCK_VIX指标的Data_frame
        """
        df = data.copy()
        
        # 获取收盘价
        close = df['close']
        
        # 1. 计算对数收益率
        log_returns = np.log(close / close.shift(1))
        
        # 2. 计算滚动标准差
        rolling_std = log_returns.rolling(window=self.period).std()
        
        # 3. 年化波动率
        annualized_volatility = rolling_std * np.sqrt(self.annualize_factor)
        
        # 4. VIX值（以百分比形式）
        vix_value = annualized_volatility * 100
        
        # 5. 计算其他相关指标
        # 短期波动率（5日）
        short_volatility = log_returns.rolling(window=5).std() * np.sqrt(self.annualize_factor) * 100
        
        # 长期波动率（60日）
        long_volatility = log_returns.rolling(window=60).std() * np.sqrt(self.annualize_factor) * 100
        
        # 波动率比率
        volatility_ratio = short_volatility / long_volatility.replace(0, np.nan)
        
        # 波动率变化率
        vix_change = vix_value.pct_change()
        
        # 波动率趋势
        vix_trend = vix_value.rolling(window=5).mean()
        
        # 保存计算结果
        df['STOCK_VIX_LOG_RETURNS'] = log_returns
        df['STOCK_VIX_ROLLING_STD'] = rolling_std
        df['STOCK_VIX_SHORT_VOL'] = short_volatility
        df['STOCK_VIX_LONG_VOL'] = long_volatility
        df['STOCK_VIX_RATIO'] = volatility_ratio
        df['STOCK_VIX_CHANGE'] = vix_change
        df['STOCK_VIX_TREND'] = vix_trend
        df['STOCK_VIX_VALUE'] = vix_value
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于波动率水平的信号
        df = self._apply_vix_signal_logic(df)

        return df
    
    def _apply_vix_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用VIX指标特定的信号生成逻辑
        基于波动率水平和趋势生成信号
        """
        try:
            if 'STOCK_VIX_VALUE' not in df.columns:
                return df

            vix_value = df['STOCK_VIX_VALUE']
            vix_trend = df['STOCK_VIX_TREND']
            
            # 计算VIX的分位数阈值
            vix_quantiles = vix_value.quantile([0.2, 0.8])
            low_threshold = vix_quantiles[0.2]
            high_threshold = vix_quantiles[0.8]
            
            # VIX信号生成逻辑：
            # 低波动率（VIX低）+ 波动率上升 = 买入信号（波动率从低位回升）
            # 高波动率（VIX高）+ 波动率下降 = 卖出信号（恐慌情绪缓解）
            
            low_vix = vix_value < low_threshold
            high_vix = vix_value > high_threshold
            vix_rising = vix_value > vix_trend
            vix_falling = vix_value < vix_trend
            
            # 生成信号
            df.loc[:, 'buy_signal'] = low_vix & vix_rising
            df.loc[:, 'sell_signal'] = high_vix & vix_falling
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"STOCK_VIX信号生成失败: {e}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df
    
    def calculate_raw_score_Vix_Stock_Vix(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算股票VIX指标的原始评分
        
        基于波动率水平、趋势和市场情绪进行评分：
        1. 波动率水平：适中波动率得高分，极端波动率得低分
        2. 波动率趋势：波动率变化的方向和幅度
        3. 相对波动率：与历史波动率的比较
        4. 波动率稳定性：波动率本身的波动程度
        """
        if not self.has_result():
            self.calculate_Vix_Stock_Vix(data, **kwargs)
        
        if 'STOCK_VIX_VALUE' not in self._result.columns:
            return pd.Series(50.0, index=data.index)
        
        vix_value = self._result['STOCK_VIX_VALUE'].fillna(0)
        vix_trend = self._result['STOCK_VIX_TREND'].fillna(0)
        vix_change = self._result['STOCK_VIX_CHANGE'].fillna(0)
        volatility_ratio = self._result['STOCK_VIX_RATIO'].fillna(1)
        
        scores = pd.Series(index=data.index, dtype=float)
        
        # 计算VIX的历史分位数
        vix_quantiles = vix_value.quantile([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        for i in range(len(vix_value)):
            if i < self.period:
                scores.iloc[i] = 50.0
                continue
            
            score = 50.0  # 基础分数
            
            # 获取当前数据
            current_vix = vix_value.iloc[i]
            current_trend = vix_trend.iloc[i]
            current_change = vix_change.iloc[i]
            current_ratio = volatility_ratio.iloc[i]
            
            # 1. 波动率水平评分 (30分)
            # 适中的波动率得高分，极端波动率得低分
            if current_vix < vix_quantiles[0.1]:
                # 极低波动率 - 可能预示变盘
                level_score = 15.0
            elif current_vix < vix_quantiles[0.2]:
                # 低波动率 - 相对稳定
                level_score = 25.0
            elif current_vix < vix_quantiles[0.3]:
                # 较低波动率 - 健康状态
                level_score = 30.0
            elif current_vix < vix_quantiles[0.7]:
                # 适中波动率 - 正常状态
                level_score = 25.0
            elif current_vix < vix_quantiles[0.8]:
                # 较高波动率 - 需要关注
                level_score = 20.0
            elif current_vix < vix_quantiles[0.9]:
                # 高波动率 - 市场紧张
                level_score = 10.0
            else:
                # 极高波动率 - 恐慌状态
                level_score = 5.0
            
            score += level_score - 20.0  # 调整基准
            
            # 2. 波动率趋势评分 (25分)
            if not pd.isna(current_change):
                if abs(current_change) < 0.05:
                    # 波动率稳定
                    trend_score = 25.0
                elif current_change > 0:
                    # 波动率上升
                    if current_change > 0.2:
                        trend_score = 10.0  # 急剧上升
                    elif current_change > 0.1:
                        trend_score = 15.0  # 明显上升
                    else:
                        trend_score = 20.0  # 温和上升
                else:
                    # 波动率下降
                    if current_change < -0.2:
                        trend_score = 25.0  # 急剧下降（恐慌缓解）
                    elif current_change < -0.1:
                        trend_score = 22.0  # 明显下降
                    else:
                        trend_score = 18.0  # 温和下降
            else:
                trend_score = 15.0
            
            score += trend_score - 15.0  # 调整基准
            
            # 3. 相对波动率评分 (25分)
            if not pd.isna(current_ratio) and current_ratio > 0:
                if current_ratio > 2.0:
                    # 短期波动率远高于长期
                    ratio_score = 5.0
                elif current_ratio > 1.5:
                    # 短期波动率明显高于长期
                    ratio_score = 10.0
                elif current_ratio > 1.2:
                    # 短期波动率略高于长期
                    ratio_score = 20.0
                elif current_ratio > 0.8:
                    # 短期波动率与长期接近
                    ratio_score = 25.0
                elif current_ratio > 0.6:
                    # 短期波动率略低于长期
                    ratio_score = 20.0
                else:
                    # 短期波动率远低于长期
                    ratio_score = 15.0
            else:
                ratio_score = 15.0
            
            score += ratio_score - 15.0  # 调整基准
            
            # 4. 波动率稳定性评分 (20分)
            if i >= 10:
                # 计算近期VIX的稳定性
                recent_vix = vix_value.iloc[max(0, i-9):i+1]
                vix_stability = 1.0 / (1.0 + recent_vix.std())
                stability_score = min(20.0, vix_stability * 40.0)
            else:
                stability_score = 15.0
            
            score += stability_score - 15.0  # 调整基准
            
            # 确保分数在合理范围内
            score = max(0, min(100, score))
            scores.iloc[i] = score
        
        return scores
    
    def calculate_confidence_Vix_Stock_Vix(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if not self.has_result():
            return 0.5
        
        # 基于VIX指标的稳定性和预测能力
        vix_value = self._result['STOCK_VIX_VALUE'].fillna(0)
        
        # 计算VIX的变异系数
        vix_cv = vix_value.std() / vix_value.mean() if vix_value.mean() > 0 else 1.0
        stability = 1.0 / (1.0 + vix_cv)
        
        # 计算信号的有效性
        signal_effectiveness = 0.5
        if 'buy_signal' in self._result.columns and 'sell_signal' in self._result.columns:
            buy_signals = self._result['buy_signal'].sum()
            sell_signals = self._result['sell_signal'].sum()
            total_signals = buy_signals + sell_signals
            if total_signals > 0:
                signal_effectiveness = min(buy_signals, sell_signals) / total_signals
        
        # 综合置信度
        confidence = (stability * 0.7 + signal_effectiveness * 0.3)
        return min(0.9, max(0.1, confidence))
    
    def get_patterns_Vix_Stock_Vix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Vix_Stock_Vix(data, **kwargs)
        
        patterns = pd.DataFrame(index=data.index)
        
        if 'STOCK_VIX_VALUE' in self._result.columns:
            vix_value = self._result['STOCK_VIX_VALUE']
            vix_trend = self._result['STOCK_VIX_TREND']
            
            # 计算分位数阈值
            vix_quantiles = vix_value.quantile([0.2, 0.8])
            
            # 识别关键形态
            patterns['low_volatility'] = vix_value < vix_quantiles[0.2]
            patterns['high_volatility'] = vix_value > vix_quantiles[0.8]
            patterns['volatility_spike'] = vix_value > vix_value.rolling(window=5).mean() * 1.5
            patterns['volatility_compression'] = vix_value < vix_value.rolling(window=20).mean() * 0.8
            patterns['volatility_rising'] = vix_value > vix_trend
            patterns['volatility_falling'] = vix_value < vix_trend
        
        return patterns


# 为了向后兼容，创建别名
stock_vix = STOCK_VIX
    @property
    def minimum_periods(self) -> int:
        """
        StockVix指标所需的最少数据周期数
        
        计算逻辑：使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30