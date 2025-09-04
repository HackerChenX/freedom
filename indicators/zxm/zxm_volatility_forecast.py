#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZXM波动率预测指标
基于历史数据预测未来波动率，用于风险管理和期权定价
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from indicators.base_indicator import BaseIndicator
from utils.dependency_injection import get_logger
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


class ZXMVolatilityForecast(BaseIndicator):
    """
    ZXM波动率预测指标
    
    使用GARCH模型思想预测未来波动率
    """
    
    def __init__(self):
        """初始化ZXM波动率预测指标"""
        # 直接设置属性，不调用super().__init__()
        self.name = "ZXM_VOLATILITY_FORECAST"
        self.description = "ZXM波动率预测指标，基于历史数据预测未来波动率"
        self.indicator_type = "ZXM_VOLATILITY_FORECAST"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

        # 设置最小周期数
        self._minimum_periods = 30

        # 初始化状态
        self._result = None
        self._error = None
        self.is_available = False

    @property
    def minimum_periods(self) -> int:
        """获取最小周期数"""
        return self._minimum_periods
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold=2.0)
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
        """
        计算ZXM波动率预测指标

        Args:
            data: 包含OHLCV数据的DataFrame
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Dict[str, Any]: 包含波动率预测指标的字典
        """
        if data is None or data.empty:
            logger.warning("ZXM_VOLATILITY_FORECAST: 输入数据为空")
            return self._get_default_result()

        # 验证必需的列
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"ZXM_VOLATILITY_FORECAST: 缺少必需的列: {missing_columns}")
            return self._get_default_result()

        result = {}

        try:
            # 计算收益率
            if len(data) >= 30:
                returns = data['close'].pct_change().dropna()

                if len(returns) < 5:
                    logger.warning("ZXM_VOLATILITY_FORECAST: 有效收益率数据不足")
                    return self._get_default_result()

                # 1. 历史波动率（不同周期）
                if len(returns) >= 5:
                    vol_5d = returns.tail(5).std() * np.sqrt(252)
                    result['volatility_5d'] = float(vol_5d) if not np.isnan(vol_5d) else 0.25

                if len(returns) >= 10:
                    vol_10d = returns.tail(10).std() * np.sqrt(252)
                    result['volatility_10d'] = float(vol_10d) if not np.isnan(vol_10d) else 0.25

                if len(returns) >= 20:
                    vol_20d = returns.tail(20).std() * np.sqrt(252)
                    result['volatility_20d'] = float(vol_20d) if not np.isnan(vol_20d) else 0.25

                # 2. EWMA波动率预测
                alpha = 0.94  # 衰减因子
                ewma_var = returns.var()
                if not np.isnan(ewma_var) and ewma_var > 0:
                    for ret in returns:
                        if not np.isnan(ret):
                            ewma_var = alpha * ewma_var + (1 - alpha) * ret**2

                    ewma_vol = np.sqrt(ewma_var * 252)
                    result['ewma_volatility'] = float(ewma_vol) if not np.isnan(ewma_vol) else 0.25
                else:
                    result['ewma_volatility'] = 0.25

                # 3. 简化GARCH预测
                if len(returns) >= 10:
                    recent_vol = returns.tail(10).std() * np.sqrt(252)
                    long_term_vol = returns.std() * np.sqrt(252)

                    if not np.isnan(recent_vol) and not np.isnan(long_term_vol):
                        # 均值回归预测
                        mean_reversion_speed = 0.1
                        forecast_vol = recent_vol * (1 - mean_reversion_speed) + long_term_vol * mean_reversion_speed
                        result['forecast_volatility'] = float(forecast_vol)
                    else:
                        result['forecast_volatility'] = 0.25
                else:
                    result['forecast_volatility'] = 0.25

                # 4. 波动率趋势
                if len(returns) >= 20:
                    recent_vol_raw = returns.tail(10).std()
                    previous_vol_raw = returns.tail(20).head(10).std()

                    if not np.isnan(recent_vol_raw) and not np.isnan(previous_vol_raw) and previous_vol_raw > 0:
                        if recent_vol_raw > previous_vol_raw * 1.2:
                            result['volatility_trend'] = 'increasing'
                            result['trend_strength'] = 'strong'
                        elif recent_vol_raw > previous_vol_raw * 1.1:
                            result['volatility_trend'] = 'increasing'
                            result['trend_strength'] = 'moderate'
                        elif recent_vol_raw < previous_vol_raw * 0.8:
                            result['volatility_trend'] = 'decreasing'
                            result['trend_strength'] = 'strong'
                        elif recent_vol_raw < previous_vol_raw * 0.9:
                            result['volatility_trend'] = 'decreasing'
                            result['trend_strength'] = 'moderate'
                        else:
                            result['volatility_trend'] = 'stable'
                            result['trend_strength'] = 'weak'
                    else:
                        result['volatility_trend'] = 'stable'
                        result['trend_strength'] = 'weak'
                else:
                    result['volatility_trend'] = 'stable'
                    result['trend_strength'] = 'weak'

                # 5. 波动率分位数
                if len(returns) >= 20:
                    current_vol = result.get('volatility_20d', 0.25)
                    rolling_vol = returns.rolling(min(252, len(returns))).std() * np.sqrt(252)
                    rolling_vol = rolling_vol.dropna()

                    if len(rolling_vol) > 0:
                        vol_percentile = (rolling_vol < current_vol).mean() * 100
                        result['volatility_percentile'] = float(vol_percentile) if not np.isnan(vol_percentile) else 50.0
                    else:
                        result['volatility_percentile'] = 50.0
                else:
                    result['volatility_percentile'] = 50.0

                # 6. 风险等级
                forecast_vol = result.get('forecast_volatility', 0.25)
                if forecast_vol > 0.6:
                    result['risk_level'] = 'very_high'
                elif forecast_vol > 0.4:
                    result['risk_level'] = 'high'
                elif forecast_vol > 0.25:
                    result['risk_level'] = 'medium'
                elif forecast_vol > 0.15:
                    result['risk_level'] = 'low'
                else:
                    result['risk_level'] = 'very_low'

            else:
                # 数据不足时的默认值
                result = self._get_default_result()

            # 确保所有必需的键都存在
            default_result = self._get_default_result()
            for key in default_result:
                if key not in result:
                    result[key] = default_result[key]

            return result

        except Exception as e:
            logger.error(f"ZXM_VOLATILITY_FORECAST计算失败: {e}")
            return self._get_default_result()

    def _get_default_result(self) -> Dict[str, Any]:
        """获取默认结果"""
        return {
            'volatility_5d': 0.25,
            'volatility_10d': 0.25,
            'volatility_20d': 0.25,
            'ewma_volatility': 0.25,
            'forecast_volatility': 0.25,
            'volatility_trend': 'stable',
            'trend_strength': 'weak',
            'volatility_percentile': 50.0,
            'risk_level': 'medium'
        }
    
    @exception_handler(reraise=False, default_return={})
    def get_patterns(self) -> Dict[str, Any]:
        """
        获取ZXM波动率预测指标的形态信息

        Returns:
            Dict[str, Any]: 包含形态信息的字典
        """
        return {
            'indicator_type': 'ZXM_VOLATILITY_FORECAST',
            'category': 'volatility_forecast',
            'description': 'ZXM波动率预测指标',
            'version': '1.0.0',
            'author': 'ZXM',
            'metrics': [
                'volatility_5d',
                'volatility_10d',
                'volatility_20d',
                'ewma_volatility',
                'forecast_volatility',
                'volatility_percentile'
            ],
            'trends': ['increasing', 'decreasing', 'stable'],
            'trend_strengths': ['strong', 'moderate', 'weak'],
            'risk_levels': ['very_low', 'low', 'medium', 'high', 'very_high'],
            'thresholds': {
                'very_high_risk': 0.6,
                'high_risk': 0.4,
                'medium_risk': 0.25,
                'low_risk': 0.15
            },
            'data_requirements': {
                'minimum_periods': 30,
                'required_columns': ['close'],
                'recommended_columns': ['open', 'high', 'low', 'close', 'volume']
            },
            'output_format': {
                'volatility_5d': 'float',
                'volatility_10d': 'float',
                'volatility_20d': 'float',
                'ewma_volatility': 'float',
                'forecast_volatility': 'float',
                'volatility_trend': 'string',
                'trend_strength': 'string',
                'volatility_percentile': 'float',
                'risk_level': 'string'
            }
        }

    # ==================== BaseIndicator抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现：核心计算逻辑"""
        result = self.calculate(data, **kwargs)

        # 将字典结果转换为DataFrame
        if isinstance(result, dict) and result:
            # 创建一个与输入数据长度相同的DataFrame
            df_result = pd.DataFrame(index=data.index)

            # 添加计算结果作为最后一行的值
            for key, value in result.items():
                df_result[key] = None
                if len(df_result) > 0:
                    df_result.iloc[-1, df_result.columns.get_loc(key)] = value

            return df_result
        else:
            # 如果计算失败，返回空DataFrame
            return pd.DataFrame(index=data.index)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """BaseIndicator抽象方法实现：计算原始评分"""
        try:
            result = self.calculate(data, **kwargs)

            if not result:
                return pd.Series([50.0], index=[data.index[-1]] if len(data) > 0 else [0])

            # 基于风险等级计算评分
            risk_level = result.get('risk_level', 'medium')
            forecast_vol = result.get('forecast_volatility', 0.25)

            # 风险等级评分映射
            risk_scores = {
                'very_low': 90.0,
                'low': 75.0,
                'medium': 50.0,
                'high': 25.0,
                'very_high': 10.0
            }

            base_score = risk_scores.get(risk_level, 50.0)

            # 根据预测波动率微调评分
            if forecast_vol < 0.15:
                base_score += 10
            elif forecast_vol > 0.4:
                base_score -= 10

            # 确保评分在0-100范围内
            final_score = max(0.0, min(100.0, base_score))

            return pd.Series([final_score], index=[data.index[-1]] if len(data) > 0 else [0])

        except Exception as e:
            logger.error(f"ZXM_VOLATILITY_FORECAST计算原始评分失败: {e}")
            return pd.Series([50.0], index=[data.index[-1]] if len(data) > 0 else [0])

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现：获取技术形态"""
        try:
            result = self.calculate(data, **kwargs)

            # 创建形态DataFrame
            patterns_df = pd.DataFrame(index=data.index)

            if result:
                # 基于波动率趋势识别形态
                volatility_trend = result.get('volatility_trend', 'stable')
                trend_strength = result.get('trend_strength', 'weak')
                risk_level = result.get('risk_level', 'medium')

                # 波动率上升形态
                if volatility_trend == 'increasing':
                    if trend_strength == 'strong':
                        patterns_df['VOLATILITY_SPIKE'] = False
                        patterns_df.iloc[-1, patterns_df.columns.get_loc('VOLATILITY_SPIKE')] = True
                    else:
                        patterns_df['VOLATILITY_RISING'] = False
                        patterns_df.iloc[-1, patterns_df.columns.get_loc('VOLATILITY_RISING')] = True

                # 波动率下降形态
                elif volatility_trend == 'decreasing':
                    if trend_strength == 'strong':
                        patterns_df['VOLATILITY_COLLAPSE'] = False
                        patterns_df.iloc[-1, patterns_df.columns.get_loc('VOLATILITY_COLLAPSE')] = True
                    else:
                        patterns_df['VOLATILITY_DECLINING'] = False
                        patterns_df.iloc[-1, patterns_df.columns.get_loc('VOLATILITY_DECLINING')] = True

                # 高风险形态
                if risk_level in ['high', 'very_high']:
                    patterns_df['HIGH_RISK_VOLATILITY'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('HIGH_RISK_VOLATILITY')] = True

                # 低风险形态
                elif risk_level in ['low', 'very_low']:
                    patterns_df['LOW_RISK_VOLATILITY'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('LOW_RISK_VOLATILITY')] = True

            return patterns_df

        except Exception as e:
            logger.error(f"ZXM_VOLATILITY_FORECAST获取形态失败: {e}")
            return pd.DataFrame(index=data.index)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """BaseIndicator抽象方法实现：计算置信度"""
        try:
            # 基础置信度
            base_confidence = 0.7

            # 根据数据量调整置信度
            data_length = len(score)
            if data_length >= 252:  # 一年数据
                data_confidence = 0.9
            elif data_length >= 60:  # 两个月数据
                data_confidence = 0.8
            elif data_length >= 30:  # 一个月数据
                data_confidence = 0.7
            else:
                data_confidence = 0.5

            # 根据形态数量调整置信度
            pattern_confidence = 0.7
            if isinstance(patterns, pd.DataFrame) and not patterns.empty:
                pattern_count = patterns.sum().sum()
                if pattern_count > 0:
                    pattern_confidence = min(0.9, 0.7 + pattern_count * 0.05)

            # 综合置信度
            final_confidence = (base_confidence + data_confidence + pattern_confidence) / 3

            return max(0.0, min(1.0, final_confidence))

        except Exception as e:
            logger.error(f"ZXM_VOLATILITY_FORECAST计算置信度失败: {e}")
            return 0.5

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """BaseIndicator抽象方法实现：设置参数"""
        try:
            # 更新参数
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"ZXM_VOLATILITY_FORECAST参数更新: {key} = {value}")

            # 重置结果，强制重新计算
            self._result = None

        except Exception as e:
            logger.error(f"ZXM_VOLATILITY_FORECAST设置参数失败: {e}")
    
    @property
    def minimum_periods(self) -> int:
        """
        ZXM波动率预测指标所需的最少数据周期数
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30
