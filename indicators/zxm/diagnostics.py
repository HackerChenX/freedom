#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from indicators.base_indicator import BaseIndicator

class ZXMDiagnostics(BaseIndicator):
    """
    ZXM智能诊断器
    
    用于对股票的技术状态进行全面分析和诊断，检测潜在问题和交易机会，
    并生成标准化的诊断报告和评分。
    """
    
    def __init__(self):
        """初始化ZXM智能诊断器"""
        super().__init__()
        self.name = "ZXM智能诊断器"
        self.description = "智能分析股票技术状态的综合诊断工具"
        
        # 健康度评估指标权重
        self.health_weights = {
            'trend_health': 0.25,       # 趋势健康度
            'momentum_health': 0.20,    # 动量健康度
            'volume_health': 0.15,      # 成交量健康度
            'volatility_health': 0.10,  # 波动率健康度
            'support_resistance': 0.15, # 支撑阻力健康度
            'cycle_health': 0.15        # 周期健康度
        }
        
        # 机会评估指标权重
        self.opportunity_weights = {
            'breakout_potential': 0.20,   # 突破潜力
            'reversal_potential': 0.20,   # 反转潜力
            'trend_continuation': 0.15,   # 趋势延续性
            'value_potential': 0.15,      # 价值潜力
            'momentum_buildup': 0.15,     # 动量积累
            'divergence_signals': 0.15    # 背离信号
        } 

    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算股票诊断指标
        
        Args:
            data: DataFrame，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                lookback_period: 回溯分析周期，默认60个交易日
                require_volume: 是否要求成交量数据，默认True
                
        Returns:
            DataFrame: 包含诊断结果的DataFrame
        """
        if data.empty:
            return pd.DataFrame()
            
        # 获取参数
        lookback_period = kwargs.get('lookback_period', 60)
        require_volume = kwargs.get('require_volume', True)
        
        # 检查数据完整性
        if len(data) < lookback_period:
            return pd.DataFrame()
            
        if require_volume and 'volume' not in data.columns:
            return pd.DataFrame()
            
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.DataFrame()
            
        # 初始化结果DataFrame
        result = data.copy()
        
        # 1. 趋势健康度分析
        trend_health = self._analyze_trend_health(data, lookback_period)
        for col, values in trend_health.items():
            result[f'trend_{col}'] = values
            
        # 2. 动量健康度分析
        momentum_health = self._analyze_momentum_health(data, lookback_period)
        for col, values in momentum_health.items():
            result[f'momentum_{col}'] = values
            
        # 3. 成交量健康度分析
        if 'volume' in data.columns:
            volume_health = self._analyze_volume_health(data, lookback_period)
            for col, values in volume_health.items():
                result[f'volume_{col}'] = values
        
        # 4. 波动率健康度分析
        volatility_health = self._analyze_volatility_health(data, lookback_period)
        for col, values in volatility_health.items():
            result[f'volatility_{col}'] = values
            
        # 5. 支撑阻力分析
        support_resistance = self._analyze_support_resistance(data, lookback_period)
        for col, values in support_resistance.items():
            result[f'sr_{col}'] = values
            
        # 6. 周期健康度分析
        cycle_health = self._analyze_cycle_health(data, lookback_period)
        for col, values in cycle_health.items():
            result[f'cycle_{col}'] = values
            
        # 7. 突破潜力分析
        breakout_potential = self._analyze_breakout_potential(data, lookback_period)
        for col, values in breakout_potential.items():
            result[f'breakout_{col}'] = values
            
        # 8. 反转潜力分析
        reversal_potential = self._analyze_reversal_potential(data, lookback_period)
        for col, values in reversal_potential.items():
            result[f'reversal_{col}'] = values
            
        # 9. 综合健康度评分
        result['health_score'] = self._calculate_health_score(result)
        
        # 10. 综合机会评分
        result['opportunity_score'] = self._calculate_opportunity_score(result)
        
        # 11. 总体诊断评分 (0-100)
        result['diagnosis_score'] = self._calculate_diagnosis_score(result)
        
        # 12. 诊断结论
        result['diagnosis_conclusion'] = self._generate_diagnosis_conclusion(result)
        
        # 13. 主要问题和建议
        result['main_issues'] = self._identify_main_issues(result)
        result['recommendations'] = self._generate_recommendations(result)
        
        return result 

    def calculate_raw_score(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算股票诊断原始评分 (0-100分)
        
        Args:
            data: DataFrame，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                score_type: 评分类型，可选 'health'(健康度), 'opportunity'(机会度), 'overall'(综合)
                
        Returns:
            DataFrame: 包含原始评分的DataFrame
        """
        # 获取评分类型参数
        score_type = kwargs.get('score_type', 'overall')
        
        # 初始化评分DataFrame
        scores = pd.DataFrame(index=data.index)
        scores['raw_score'] = 50.0  # 默认评分50分（中性）
        
        # 计算诊断指标
        diagnosis_result = self.calculate(data, *args, **kwargs)
        
        if diagnosis_result.empty:
            return scores
            
        # 根据评分类型选择相应的评分
        if score_type == 'health':
            if 'health_score' in diagnosis_result.columns:
                scores['raw_score'] = diagnosis_result['health_score']
        elif score_type == 'opportunity':
            if 'opportunity_score' in diagnosis_result.columns:
                scores['raw_score'] = diagnosis_result['opportunity_score']
        else:  # 'overall' 或其他任何值
            if 'diagnosis_score' in diagnosis_result.columns:
                scores['raw_score'] = diagnosis_result['diagnosis_score']
        
        # 确保评分在0-100范围内
        scores['raw_score'] = scores['raw_score'].clip(0, 100)
        
        return scores
        
    def identify_patterns(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别股票技术形态和问题模式
        
        Args:
            data: DataFrame，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                min_pattern_strength: 最小形态强度阈值，默认0.6
                
        Returns:
            DataFrame: 包含识别出的形态的DataFrame
        """
        # 获取参数
        min_pattern_strength = kwargs.get('min_pattern_strength', 0.6)
        
        # 初始化形态DataFrame
        patterns = pd.DataFrame(index=data.index)
        patterns['pattern'] = None
        patterns['pattern_strength'] = 0.0
        patterns['pattern_desc'] = None
        patterns['pattern_type'] = None  # 问题类型或机会类型
        
        # 计算诊断指标
        diagnosis_result = self.calculate(data, *args, **kwargs)
        
        if diagnosis_result.empty:
            return patterns
            
        # 识别趋势相关形态
        self._identify_trend_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        # 识别动量相关形态
        self._identify_momentum_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        # 识别成交量相关形态
        if 'volume' in data.columns:
            self._identify_volume_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        # 识别波动率相关形态
        self._identify_volatility_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        # 识别支撑阻力相关形态
        self._identify_support_resistance_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        # 识别周期相关形态
        self._identify_cycle_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        # 识别综合技术形态
        self._identify_composite_patterns(data, diagnosis_result, patterns, min_pattern_strength)
        
        return patterns 

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成股票诊断信号
        
        Args:
            data: DataFrame，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                signal_threshold: 信号阈值，默认70
                
        Returns:
            DataFrame: 包含标准化信号的DataFrame
        """
        # 获取参数
        signal_threshold = kwargs.get('signal_threshold', 70)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算诊断结果
        diagnosis_result = self.calculate(data, *args, **kwargs)
        
        if diagnosis_result.empty:
            return signals
            
        # 识别形态
        patterns = self.identify_patterns(data, *args, **kwargs)
        
        # 生成买入信号
        buy_conditions = (
            (diagnosis_result['health_score'] > signal_threshold) & 
            (diagnosis_result['opportunity_score'] > signal_threshold)
        )
        
        # 生成卖出信号
        sell_conditions = (
            (diagnosis_result['health_score'] < (100 - signal_threshold)) & 
            (diagnosis_result['opportunity_score'] < 40)
        )
        
        # 应用信号条件
        for i in range(len(signals)):
            if i < 5:  # 跳过前几个数据点，因为指标计算需要足够的历史数据
                continue
                
            # 买入信号
            if buy_conditions.iloc[i]:
                signals.loc[signals.index[i], 'buy_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = 1
                signals.loc[signals.index[i], 'score'] = diagnosis_result['diagnosis_score'].iloc[i]
                signals.loc[signals.index[i], 'signal_type'] = '诊断买入'
                signals.loc[signals.index[i], 'signal_desc'] = '综合技术状态健康，存在买入机会'
                signals.loc[signals.index[i], 'confidence'] = min(90, diagnosis_result['health_score'].iloc[i])
                
                # 设置风险级别
                if diagnosis_result['health_score'].iloc[i] > 80:
                    signals.loc[signals.index[i], 'risk_level'] = '低'
                else:
                    signals.loc[signals.index[i], 'risk_level'] = '中'
                    
                # 设置仓位建议
                signals.loc[signals.index[i], 'position_size'] = min(1.0, diagnosis_result['health_score'].iloc[i] / 100)
                
                # 设置止损位
                if 'volatility_atr' in diagnosis_result.columns:
                    atr = diagnosis_result['volatility_atr'].iloc[i]
                    close = data['close'].iloc[i]
                    signals.loc[signals.index[i], 'stop_loss'] = close - (2 * atr)
            
            # 卖出信号
            elif sell_conditions.iloc[i]:
                signals.loc[signals.index[i], 'sell_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = -1
                signals.loc[signals.index[i], 'score'] = 100 - diagnosis_result['diagnosis_score'].iloc[i]
                signals.loc[signals.index[i], 'signal_type'] = '诊断卖出'
                signals.loc[signals.index[i], 'signal_desc'] = '综合技术状态不佳，建议卖出或规避'
                signals.loc[signals.index[i], 'confidence'] = min(90, 100 - diagnosis_result['health_score'].iloc[i])
                
                # 设置风险级别
                if diagnosis_result['health_score'].iloc[i] < 30:
                    signals.loc[signals.index[i], 'risk_level'] = '高'
                else:
                    signals.loc[signals.index[i], 'risk_level'] = '中'
                    
                # 设置仓位建议 (平仓)
                signals.loc[signals.index[i], 'position_size'] = 0.0
            
            # 使用识别出的形态增强信号
            if patterns['pattern'].iloc[i] is not None:
                pattern_strength = patterns['pattern_strength'].iloc[i]
                pattern_desc = patterns['pattern_desc'].iloc[i]
                
                # 如果形态强度足够高，使用形态描述作为信号描述
                if pattern_strength > 0.7:
                    signals.loc[signals.index[i], 'signal_desc'] = pattern_desc
                    
                    # 根据形态类型调整信号置信度
                    if 'pattern_type' in patterns.columns and patterns['pattern_type'].iloc[i] is not None:
                        if patterns['pattern_type'].iloc[i] == 'opportunity':
                            signals.loc[signals.index[i], 'confidence'] = min(95, signals['confidence'].iloc[i] + 10)
                        elif patterns['pattern_type'].iloc[i] == 'problem':
                            signals.loc[signals.index[i], 'confidence'] = min(95, signals['confidence'].iloc[i] + 10)
            
            # 设置市场环境
            if 'trend_direction' in diagnosis_result.columns:
                trend_direction = diagnosis_result['trend_direction'].iloc[i]
                trend_strength = diagnosis_result.get('trend_strength', pd.Series(0.5, index=diagnosis_result.index)).iloc[i]
                
                if trend_direction > 0.5 and trend_strength > 0.6:
                    signals.loc[signals.index[i], 'market_env'] = 'bull_market'
                elif trend_direction < -0.5 and trend_strength > 0.6:
                    signals.loc[signals.index[i], 'market_env'] = 'bear_market'
                elif trend_strength < 0.3:
                    signals.loc[signals.index[i], 'market_env'] = 'sideways_market'
            
            # 成交量确认
            if 'volume' in data.columns and 'volume_health' in diagnosis_result.columns:
                volume_health = diagnosis_result['volume_health'].iloc[i]
                if volume_health > 70:
                    signals.loc[signals.index[i], 'volume_confirmation'] = True
        
        return signals
        
    # 分析方法
    def _analyze_trend_health(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析趋势健康度"""
        result = {}
        
        close = data['close']
        
        # 计算短、中、长期均线
        ma20 = close.rolling(window=20).mean()
        ma60 = close.rolling(window=60).mean()
        
        # 趋势方向 (-1 到 1)
        result['direction'] = pd.Series(0.0, index=data.index)
        for i in range(lookback_period, len(data)):
            # 短期趋势：当前价格相对于20日均线
            short_trend = (close.iloc[i] / ma20.iloc[i] - 1) * 10
            # 中期趋势：20日均线相对于60日均线
            mid_trend = (ma20.iloc[i] / ma60.iloc[i] - 1) * 10
            # 长期趋势：通过线性回归斜率计算
            long_trend = (close.iloc[i] / close.iloc[i-lookback_period] - 1)
            
            # 综合趋势方向 (-1 到 1)
            trend_direction = (short_trend * 0.5 + mid_trend * 0.3 + long_trend * 0.2)
            result['direction'].iloc[i] = max(-1, min(1, trend_direction))
        
        # 趋势强度 (0 到 1)
        result['strength'] = pd.Series(0.0, index=data.index)
        for i in range(lookback_period, len(data)):
            # 均线排列情况
            ma_alignment = 1 if ma20.iloc[i] > ma60.iloc[i] else -1
            
            # 价格与均线的关系
            price_ma_relation = 1 if close.iloc[i] > ma20.iloc[i] else -1
            
            # 方向一致性
            direction_consistency = 1 if ma_alignment == price_ma_relation else -1
            
            # 趋势持续性（通过标准差/均值比率）
            recent_returns = close.pct_change().iloc[i-lookback_period:i]
            cv = recent_returns.std() / abs(recent_returns.mean()) if recent_returns.mean() != 0 else float('inf')
            trend_consistency = 1 / (1 + cv) if not np.isnan(cv) and not np.isinf(cv) else 0
            
            # 综合趋势强度
            trend_strength = (abs(result['direction'].iloc[i]) * 0.4 + 
                             (direction_consistency > 0) * 0.3 + 
                             trend_consistency * 0.3)
            result['strength'].iloc[i] = min(1, trend_strength)
        
        # 趋势健康度 (0 到 100)
        result['health'] = pd.Series(50.0, index=data.index)
        for i in range(lookback_period, len(data)):
            direction = result['direction'].iloc[i]
            strength = result['strength'].iloc[i]
            
            # 上升趋势健康度评分
            if direction > 0:
                result['health'].iloc[i] = 50 + direction * strength * 50
            # 下降趋势健康度评分
            else:
                result['health'].iloc[i] = 50 + direction * strength * 30  # 下降趋势健康度惩罚较小
        
        return result
        
    def _analyze_momentum_health(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析动量健康度"""
        result = {}
        
        close = data['close']
        
        # 计算RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 动量方向 (-1 到 1)
        result['direction'] = pd.Series(0.0, index=data.index)
        for i in range(lookback_period, len(data)):
            # RSI动量
            rsi_momentum = (rsi.iloc[i] - 50) / 50
            
            # 价格动量（通过ROC）
            price_roc = close.iloc[i] / close.iloc[i-20] - 1
            price_momentum = price_roc * 5  # 缩放到合理范围
            
            # 综合动量方向
            momentum_direction = rsi_momentum * 0.6 + price_momentum * 0.4
            result['direction'].iloc[i] = max(-1, min(1, momentum_direction))
        
        # 动量强度 (0 到 1)
        result['strength'] = pd.Series(0.0, index=data.index)
        for i in range(lookback_period, len(data)):
            # RSI位置
            rsi_extreme = abs(rsi.iloc[i] - 50) / 50
            
            # ROC强度
            recent_roc = close.iloc[i] / close.iloc[i-20] - 1
            roc_strength = min(1, abs(recent_roc) * 5)
            
            # 综合动量强度
            momentum_strength = rsi_extreme * 0.6 + roc_strength * 0.4
            result['strength'].iloc[i] = momentum_strength
        
        # 动量健康度 (0 到 100)
        result['health'] = pd.Series(50.0, index=data.index)
        for i in range(lookback_period, len(data)):
            direction = result['direction'].iloc[i]
            strength = result['strength'].iloc[i]
            
            # 正动量健康度评分
            if direction > 0:
                result['health'].iloc[i] = 50 + direction * strength * 50
            # 负动量健康度评分
            else:
                result['health'].iloc[i] = 50 + direction * strength * 30  # 负动量健康度惩罚较小
        
        return result 