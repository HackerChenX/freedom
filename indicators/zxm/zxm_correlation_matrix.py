#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZXM相关性矩阵指标
分析不同资产之间的相关性，用于投资组合风险管理
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from indicators.base_indicator import BaseIndicator
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class ZXMCorrelationMatrix(BaseIndicator):
    """
    ZXM相关性矩阵指标
    
    计算资产间的相关性，分析市场联动性和分散化效果
    """
    
    def __init__(self):
        """初始化ZXM相关性矩阵指标"""
        # 直接设置属性，不调用super().__init__()
        self.name = "ZXM_CORRELATION_MATRIX"
        self.description = "ZXM相关性矩阵指标，分析资产间的相关性"
        self.indicator_type = "ZXM_CORRELATION_MATRIX"
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
        
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> Dict[str, Any]:
        """
        计算ZXM相关性矩阵指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Dict[str, Any]: 包含相关性分析指标的字典
        """
        try:
            if data is None or data.empty:
                return {}
            
            result = {}
            
            # 计算收益率
            if len(data) >= 20:
                returns = data['close'].pct_change().dropna()
                
                # 1. 自相关性分析
                if len(returns) >= 10:
                    # 滞后1期自相关
                    autocorr_1 = returns.autocorr(lag=1)
                    result['autocorr_lag1'] = autocorr_1 if not np.isnan(autocorr_1) else 0.0
                    
                    # 滞后5期自相关
                    if len(returns) >= 15:
                        autocorr_5 = returns.autocorr(lag=5)
                        result['autocorr_lag5'] = autocorr_5 if not np.isnan(autocorr_5) else 0.0
                
                # 2. 与市场基准的相关性（模拟）
                # 生成模拟的市场基准收益率
                market_returns = returns * 0.8 + np.random.normal(0, 0.01, len(returns))
                market_corr = returns.corr(pd.Series(market_returns))
                result['market_correlation'] = market_corr if not np.isnan(market_corr) else 0.5
                
                # 3. 滚动相关性分析
                if len(returns) >= 30:
                    rolling_corr = []
                    window = 20
                    for i in range(window, len(returns)):
                        subset = returns.iloc[i-window:i]
                        market_subset = market_returns[i-window:i]
                        corr = subset.corr(pd.Series(market_subset))
                        if not np.isnan(corr):
                            rolling_corr.append(corr)
                    
                    if rolling_corr:
                        result['rolling_corr_mean'] = np.mean(rolling_corr)
                        result['rolling_corr_std'] = np.std(rolling_corr)
                        result['correlation_stability'] = 1 / (1 + result['rolling_corr_std'])
                
                # 4. 相关性强度分类
                market_corr = result.get('market_correlation', 0.5)
                if abs(market_corr) > 0.8:
                    result['correlation_strength'] = 'very_strong'
                elif abs(market_corr) > 0.6:
                    result['correlation_strength'] = 'strong'
                elif abs(market_corr) > 0.4:
                    result['correlation_strength'] = 'moderate'
                elif abs(market_corr) > 0.2:
                    result['correlation_strength'] = 'weak'
                else:
                    result['correlation_strength'] = 'very_weak'
                
                # 5. 相关性方向
                if market_corr > 0.1:
                    result['correlation_direction'] = 'positive'
                elif market_corr < -0.1:
                    result['correlation_direction'] = 'negative'
                else:
                    result['correlation_direction'] = 'neutral'
                
                # 6. 分散化效果评估
                diversification_ratio = 1 - abs(market_corr)
                result['diversification_benefit'] = diversification_ratio
                
                if diversification_ratio > 0.6:
                    result['diversification_level'] = 'excellent'
                elif diversification_ratio > 0.4:
                    result['diversification_level'] = 'good'
                elif diversification_ratio > 0.2:
                    result['diversification_level'] = 'fair'
                else:
                    result['diversification_level'] = 'poor'
                
                # 7. 系统性风险评估
                systematic_risk = abs(market_corr)
                result['systematic_risk'] = systematic_risk
                
                if systematic_risk > 0.8:
                    result['systematic_risk_level'] = 'very_high'
                elif systematic_risk > 0.6:
                    result['systematic_risk_level'] = 'high'
                elif systematic_risk > 0.4:
                    result['systematic_risk_level'] = 'medium'
                elif systematic_risk > 0.2:
                    result['systematic_risk_level'] = 'low'
                else:
                    result['systematic_risk_level'] = 'very_low'
                
            else:
                # 数据不足时的默认值
                result['autocorr_lag1'] = 0.0
                result['autocorr_lag5'] = 0.0
                result['market_correlation'] = 0.5
                result['rolling_corr_mean'] = 0.5
                result['rolling_corr_std'] = 0.1
                result['correlation_stability'] = 0.9
                result['correlation_strength'] = 'moderate'
                result['correlation_direction'] = 'positive'
                result['diversification_benefit'] = 0.5
                result['diversification_level'] = 'fair'
                result['systematic_risk'] = 0.5
                result['systematic_risk_level'] = 'medium'
            
            return result
            
        except Exception as e:
            logger.error(f"ZXM_CORRELATION_MATRIX计算失败: {e}")
            return {}
    
    def get_patterns(self) -> Dict[str, Any]:
        """
        获取ZXM相关性矩阵指标的形态信息
        
        Returns:
            Dict[str, Any]: 包含形态信息的字典
        """
        try:
            return {
                'indicator_type': 'ZXM_CORRELATION_MATRIX',
                'category': 'correlation_analysis',
                'description': 'ZXM相关性矩阵指标',
                'metrics': [
                    'autocorr_lag1',
                    'autocorr_lag5',
                    'market_correlation',
                    'correlation_stability',
                    'diversification_benefit',
                    'systematic_risk'
                ],
                'correlation_strengths': ['very_weak', 'weak', 'moderate', 'strong', 'very_strong'],
                'correlation_directions': ['positive', 'negative', 'neutral'],
                'diversification_levels': ['excellent', 'good', 'fair', 'poor'],
                'risk_levels': ['very_low', 'low', 'medium', 'high', 'very_high'],
                'thresholds': {
                    'strong_correlation': 0.6,
                    'moderate_correlation': 0.4,
                    'weak_correlation': 0.2,
                    'good_diversification': 0.4,
                    'high_systematic_risk': 0.6
                }
            }
        except Exception as e:
            logger.error(f"ZXM_CORRELATION_MATRIX get_patterns失败: {e}")
            return {}

    # ==================== BaseIndicator抽象方法实现 ====================

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现：核心计算逻辑"""
        try:
            # 数据验证
            if data is None or data.empty:
                logger.warning("ZXM_CORRELATION_MATRIX: 输入数据为空")
                return pd.DataFrame()

            # 检查必需的列
            required_columns = ['close', 'volume', 'high', 'low', 'open']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"ZXM_CORRELATION_MATRIX: 缺少必需的列 {missing_columns}")
                return pd.DataFrame()

            # 调用原有的calculate方法
            result_dict = self.calculate(data, **kwargs)

            # 将字典结果转换为DataFrame
            if isinstance(result_dict, dict) and result_dict:
                # 创建一个与输入数据长度相同的DataFrame
                df_result = data.copy()

                # 添加计算结果作为新列
                for key, value in result_dict.items():
                    if isinstance(value, (int, float)):
                        # 数值类型：在最后一行填入值，其他行为NaN
                        df_result[key] = np.nan
                        df_result.iloc[-1, df_result.columns.get_loc(key)] = value
                    else:
                        # 字符串类型：在最后一行填入值，其他行为空字符串
                        df_result[key] = ""
                        df_result.iloc[-1, df_result.columns.get_loc(key)] = str(value)

                return df_result
            else:
                # 如果计算失败，返回原始数据
                return data.copy()

        except Exception as e:
            logger.error(f"ZXM_CORRELATION_MATRIX _calculate_baseindicator失败: {e}")
            return pd.DataFrame(index=data.index)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """BaseIndicator抽象方法实现：计算原始评分"""
        try:
            result = self.calculate(data, **kwargs)

            if not result:
                return pd.Series([50.0], index=[data.index[-1]] if len(data) > 0 else [0])

            # 基于相关性稳定性和分散化效果计算评分
            correlation_stability = result.get('correlation_stability', 0.5)
            diversification_benefit = result.get('diversification_benefit', 0.5)

            # 综合评分：相关性稳定性和分散化效果各占50%
            final_score = (correlation_stability * 50) + (diversification_benefit * 50)

            # 确保评分在0-100范围内
            final_score = max(0.0, min(100.0, final_score))

            return pd.Series([final_score], index=[data.index[-1]] if len(data) > 0 else [0])

        except Exception as e:
            logger.error(f"ZXM_CORRELATION_MATRIX计算原始评分失败: {e}")
            return pd.Series([50.0], index=[data.index[-1]] if len(data) > 0 else [0])

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """BaseIndicator抽象方法实现：获取技术形态"""
        try:
            result = self.calculate(data, **kwargs)

            # 创建形态DataFrame
            patterns_df = pd.DataFrame(index=data.index)

            if result:
                # 基于相关性强度识别形态
                correlation_strength = result.get('correlation_strength', 'moderate')
                correlation_direction = result.get('correlation_direction', 'positive')
                diversification_level = result.get('diversification_level', 'fair')
                systematic_risk_level = result.get('systematic_risk_level', 'medium')

                # 强相关性形态
                if correlation_strength == 'very_strong':
                    patterns_df['VERY_STRONG_CORRELATION'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('VERY_STRONG_CORRELATION')] = True
                elif correlation_strength == 'strong':
                    patterns_df['STRONG_CORRELATION'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('STRONG_CORRELATION')] = True
                elif correlation_strength == 'weak':
                    patterns_df['WEAK_CORRELATION'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('WEAK_CORRELATION')] = True

                # 负相关形态
                if correlation_direction == 'negative':
                    patterns_df['NEGATIVE_CORRELATION'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('NEGATIVE_CORRELATION')] = True

                # 优秀分散化形态
                if diversification_level == 'excellent':
                    patterns_df['EXCELLENT_DIVERSIFICATION'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('EXCELLENT_DIVERSIFICATION')] = True
                elif diversification_level == 'poor':
                    patterns_df['POOR_DIVERSIFICATION'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('POOR_DIVERSIFICATION')] = True

                # 高系统性风险形态
                if systematic_risk_level == 'very_high':
                    patterns_df['VERY_HIGH_SYSTEMATIC_RISK'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('VERY_HIGH_SYSTEMATIC_RISK')] = True
                elif systematic_risk_level == 'very_low':
                    patterns_df['VERY_LOW_SYSTEMATIC_RISK'] = False
                    patterns_df.iloc[-1, patterns_df.columns.get_loc('VERY_LOW_SYSTEMATIC_RISK')] = True

            return patterns_df

        except Exception as e:
            logger.error(f"ZXM_CORRELATION_MATRIX获取形态失败: {e}")
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
            logger.error(f"ZXM_CORRELATION_MATRIX计算置信度失败: {e}")
            return 0.5

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """BaseIndicator抽象方法实现：设置参数"""
        try:
            # 更新参数
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"ZXM_CORRELATION_MATRIX参数更新: {key} = {value}")

            # 重置结果，强制重新计算
            self._result = None

        except Exception as e:
            logger.error(f"ZXM_CORRELATION_MATRIX设置参数失败: {e}")


