#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)

class ZXMDiagnostics(BaseIndicator):
    """
    ZXM智能诊断器
    
    用于对股票的技术状态进行全面分析和诊断，检测潜在问题和交易机会，
    并生成标准化的诊断报告和评分。
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM智能诊断器"""
        super().__init__(name="ZXMDiagnostics", description="智能分析股票技术状态的综合诊断工具")
        self.indicator_type = "ZXM_DIAGNOSTICS"
        
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

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
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
        # 数据类型检查和转换
        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                logger.error(f"ZXMDiagnostics: 无法将字典转换为DataFrame: {e}")
                return pd.DataFrame()

        if not isinstance(data, pd.DataFrame):
            logger.error(f"ZXMDiagnostics: 输入数据类型错误，期望DataFrame，实际: {type(data)}")
            return pd.DataFrame()

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
            logger.warning(f"ZXMDiagnostics缺少必需列，尝试使用列名映射")

            # 尝试使用列名映射
            try:
                from utils.column_mapper import ColumnMapper
                data = ColumnMapper.standardize_columns(data, ['open', 'high', 'low', 'close', 'volume'])

                # 重新检查
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    logger.error(f"列名映射后仍缺少必需列，可用列: {list(data.columns)}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"列名映射失败: {e}")
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
        result.loc[:, 'health_score'] = self._calculate_health_score(result)
        
        # 10. 综合机会评分
        result.loc[:, 'opportunity_score'] = self._calculate_opportunity_score(result)
        
        # 11. 总体诊断评分 (0-100)
        result.loc[:, 'diagnosis_score'] = self._calculate_diagnosis_score(result)
        
        # 12. 诊断结论
        result.loc[:, 'diagnosis_conclusion'] = self._generate_diagnosis_conclusion(result)
        
        # 13. 主要问题和建议
        result.loc[:, 'main_issues'] = self._identify_main_issues(result)
        result.loc[:, 'recommendations'] = self._generate_recommendations(result)
        
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
        signals.loc[:, 'buy_signal'] = False
        signals.loc[:, 'sell_signal'] = False
        signals.loc[:, 'neutral_signal'] = True  # 默认为中性信号
        signals.loc[:, 'trend'] = 0  # 0表示中性
        signals.loc[:, 'score'] = 50.0  # 默认评分50分
        signals.loc[:, 'signal_type'] = None
        signals.loc[:, 'signal_desc'] = None
        signals.loc[:, 'confidence'] = 50.0
        signals.loc[:, 'risk_level'] = '中'
        signals.loc[:, 'position_size'] = 0.0
        signals.loc[:, 'stop_loss'] = None
        signals.loc[:, 'market_env'] = 'sideways_market'
        signals.loc[:, 'volume_confirmation'] = False
        
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
        
    def generate_trading_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成交易信号（满足BaseIndicator抽象类的要求）
        
        Args:
            data: DataFrame，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
                signal_threshold: 信号阈值，默认70
                
        Returns:
            DataFrame: 包含交易信号的DataFrame
        """
        # 获取参数
        signal_threshold = kwargs.get('signal_threshold', 70)
        
        # 初始化结果DataFrame
        signals = pd.DataFrame(index=data.index)
        signals.loc[:, 'buy_signal'] = False
        signals.loc[:, 'sell_signal'] = False
        signals.loc[:, 'signal_strength'] = 0.0
        
        # 计算诊断结果
        diagnosis_result = self.calculate(data, *args, **kwargs)
        
        if diagnosis_result.empty:
            return signals
            
        # 生成交易信号
        for i in range(len(signals)):
            if i < len(diagnosis_result):
                # 买入信号条件
                if diagnosis_result['diagnosis_score'].iloc[i] > signal_threshold:
                    signals.at[signals.index[i], 'buy_signal'] = True
                    signals.at[signals.index[i], 'signal_strength'] = (diagnosis_result['diagnosis_score'].iloc[i] - signal_threshold) / (100 - signal_threshold)
                
                # 卖出信号条件
                elif diagnosis_result['diagnosis_score'].iloc[i] < 30:
                    signals.at[signals.index[i], 'sell_signal'] = True
                    signals.at[signals.index[i], 'signal_strength'] = (30 - diagnosis_result['diagnosis_score'].iloc[i]) / 30
        
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

    def _analyze_volume_health(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """
        分析成交量健康度
        """
        volume_health = {}
        
        # 1. 量价关系
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        
        # 量价配合度 (价格上涨，成交量放大)
        volume_health['volume_price_coordination'] = (price_change > 0) & (volume_change > 0)
        
        # 2. 成交量激增
        avg_volume = data['volume'].rolling(window=lookback_period).mean()
        volume_health['volume_surge'] = data['volume'] > (avg_volume * 2)

        # 3. 成交量健康度评分
        volume_health['health'] = pd.Series(50.0, index=data.index)

        return volume_health

    def _analyze_support_resistance(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析支撑阻力"""
        result = {}

        high = data['high']
        low = data['low']
        close = data['close']

        # 计算支撑阻力位
        rolling_high = high.rolling(window=lookback_period).max()
        rolling_low = low.rolling(window=lookback_period).min()

        # 支撑阻力强度
        resistance_strength = (rolling_high - close) / close
        support_strength = (close - rolling_low) / close

        result['resistance_strength'] = resistance_strength.fillna(0)
        result['support_strength'] = support_strength.fillna(0)
        result['sr_health'] = pd.Series(50.0, index=data.index)

        return result

    def _analyze_cycle_health(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析周期健康度"""
        result = {}

        close = data['close']

        # 简单的周期分析
        result['cycle_position'] = pd.Series(0.5, index=data.index)  # 周期位置
        result['cycle_strength'] = pd.Series(0.5, index=data.index)  # 周期强度
        result['cycle_health'] = pd.Series(50.0, index=data.index)   # 周期健康度

        return result

    def _analyze_breakout_potential(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析突破潜力"""
        result = {}

        high = data['high']
        low = data['low']
        close = data['close']

        # 计算突破潜力
        rolling_high = high.rolling(window=lookback_period).max()
        rolling_low = low.rolling(window=lookback_period).min()

        # 突破潜力评分
        upward_potential = (rolling_high - close) / close
        downward_potential = (close - rolling_low) / close

        result['upward_potential'] = upward_potential.fillna(0)
        result['downward_potential'] = downward_potential.fillna(0)
        result['breakout_score'] = pd.Series(50.0, index=data.index)

        return result

    def _analyze_reversal_potential(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析反转潜力"""
        result = {}

        close = data['close']

        # 简单的反转潜力分析
        price_change = close.pct_change(lookback_period)

        result['reversal_signal'] = pd.Series(0.0, index=data.index)
        result['reversal_strength'] = pd.Series(0.5, index=data.index)
        result['reversal_score'] = pd.Series(50.0, index=data.index)

        return result

    def _analyze_volatility_health(self, data: pd.DataFrame, lookback_period: int) -> Dict[str, pd.Series]:
        """分析波动率健康度"""
        result = {}

        high = data['high']
        low = data['low']
        close = data['close']

        # 1. 计算ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        result['atr'] = atr  # 修复：使用字典赋值而不是.loc

        # 2. 计算布林带宽度 (Bollinger Band Width)
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper_band = ma20 + (2 * std20)
        lower_band = ma20 - (2 * std20)
        bbw = (upper_band - lower_band) / ma20
        result['bbw'] = bbw  # 修复：使用字典赋值而不是.loc

        # 3. 波动率水平 (0到1，相对于近期)
        min_atr = atr.rolling(window=lookback_period).min()
        max_atr = atr.rolling(window=lookback_period).max()
        # 避免除以零
        atr_range = max_atr - min_atr
        atr_range[atr_range == 0] = 1
        volatility_level = (atr - min_atr) / atr_range
        result['level'] = volatility_level.fillna(0.5).clip(0, 1)  # 修复：使用字典赋值而不是.loc

        # 4. 波动率健康度 (0到100)
        health = pd.Series(50.0, index=data.index)
        # 极低或极高的波动率都会降低健康度
        # 适中的波动率被认为是健康的
        # abs(volatility_level - 0.5) 是一个 V 形函数，在0.5处为0，在0和1处为0.5
        # 乘以100，将其惩罚范围扩大到0-50分
        penalty = abs(result['level'] - 0.5) * 100
        health -= penalty
        result['health'] = health.fillna(50.0).clip(0, 100)  # 修复：使用字典赋值而不是.loc

        return result
    
    def analyze_indicators_result(self, result, title="ZXM体系指标分析结果"):
        """
        分析ZXM体系指标结果
        
        Args:
            result: 分析结果
            title: 标题
            
        Returns:
            Dict: 分析结果字典
        """
        analysis_result = {
            "title": title,
            "indicators": {},
            "composite_score": 0,
            "composite_rating": "",
            "trend_consistency": {},
            "recommendation": ""
        }
        
        # 检查是否是ZXM指标分析结果
        zxm_indicators = ['ZXM_DK_SG', 'ZXM_MACD', 'ZXM_MOMENTUM', 'ZXM_KDJ', 'ZXM_BOLL']
        
        found_zxm = False
        for period_key, period_data in result.get('periods', {}).items():
            for indicator_name in period_data.get('indicators', {}).keys():
                if any(zxm_name in indicator_name for zxm_name in zxm_indicators):
                    found_zxm = True
                    break
            if found_zxm:
                break
        
        if not found_zxm:
            analysis_result["error"] = "未发现ZXM体系指标"
            return analysis_result
        
        # 分析ZXM指标结果
        period_results = {}
        for period_key, period_data in result.get('periods', {}).items():
            period_result = {
                "indicators": {}
            }
            
            for indicator_name, indicator_data in period_data.get('indicators', {}).items():
                if any(zxm_name in indicator_name for zxm_name in zxm_indicators):
                    score = indicator_data.get('score', 0) * 100
                    patterns = indicator_data.get('patterns', [])
                    
                    indicator_result = {
                        "score": score,
                        "patterns": patterns
                    }
                    
                    # 分析ZXM特有的值
                    if 'ZXM_DK_SG' in indicator_name:
                        indicator_result.update(self.analyze_dk_sg_values(indicator_data))
                    elif 'ZXM_MACD' in indicator_name:
                        indicator_result.update(self.analyze_macd_values(indicator_data))
                    elif 'ZXM_MOMENTUM' in indicator_name:
                        indicator_result.update(self.analyze_momentum_values(indicator_data))
                    elif 'ZXM_KDJ' in indicator_name:
                        indicator_result.update(self.analyze_kdj_values(indicator_data))
                    elif 'ZXM_BOLL' in indicator_name:
                        indicator_result.update(self.analyze_boll_values(indicator_data))
                    
                    period_result["indicators"][indicator_name] = indicator_result
            
            period_results[period_key] = period_result
        
        analysis_result["periods"] = period_results
        
        # 分析ZXM多周期合成
        composite_analysis = self.analyze_composite(result)
        analysis_result.update(composite_analysis)
        
        # 分析ZXM多周期趋势一致性
        trend_consistency = self.analyze_trend_consistency(result)
        analysis_result["trend_consistency"] = trend_consistency
        
        return analysis_result
    
    def analyze_dk_sg_values(self, indicator_data):
        """分析ZXM多空势格指标值"""
        values = indicator_data.get('values', {})
        if not values:
            return {}
        
        dk_value = values.get('dk_value', 0)
        sg_value = values.get('sg_value', 0)
        dk_status = "多头" if dk_value > 0 else "空头" if dk_value < 0 else "中性"
        sg_level = "强势" if sg_value > 0.5 else "弱势" if sg_value < -0.5 else "盘整"
        
        return {
            "dk_value": dk_value,
            "sg_value": sg_value,
            "dk_status": dk_status,
            "sg_level": sg_level
        }
    
    def analyze_macd_values(self, indicator_data):
        """分析ZXM MACD指标值"""
        values = indicator_data.get('values', {})
        if not values:
            return {}
        
        dif = values.get('dif', 0)
        dea = values.get('dea', 0)
        macd = values.get('macd', 0)
        
        # 判断MACD状态
        if dif > dea:
            if macd > 0:
                status = "金叉后多头加速"
            else:
                status = "金叉初期"
        else:
            if macd < 0:
                status = "死叉后空头加速"
            else:
                status = "死叉初期"
        
        return {
            "dif": dif,
            "dea": dea,
            "macd": macd,
            "status": status
        }
    
    def analyze_momentum_values(self, indicator_data):
        """分析ZXM动量指标值"""
        values = indicator_data.get('values', {})
        if not values:
            return {}
        
        momentum = values.get('momentum', 0)
        signal = values.get('signal', 0)
        
        # 判断动量状态
        if momentum > signal:
            if momentum > 0:
                status = "上升动量加速"
            else:
                status = "下降动量减缓"
        else:
            if momentum < 0:
                status = "下降动量加速"
            else:
                status = "上升动量减缓"
        
        return {
            "momentum": momentum,
            "signal": signal,
            "status": status
        }
    
    def analyze_kdj_values(self, indicator_data):
        """分析ZXM KDJ指标值"""
        values = indicator_data.get('values', {})
        if not values:
            return {}
        
        k = values.get('k', 0)
        d = values.get('d', 0)
        j = values.get('j', 0)
        
        # 判断KDJ状态
        if k > d:
            if k < 20:
                status = "超卖区金叉"
            elif k > 80:
                status = "超买区金叉"
            else:
                status = "多头行情中"
        else:
            if k < 20:
                status = "超卖区死叉"
            elif k > 80:
                status = "超买区死叉"
            else:
                status = "空头行情中"
        
        return {
            "k": k,
            "d": d,
            "j": j,
            "status": status
        }
    
    def analyze_boll_values(self, indicator_data):
        """分析ZXM布林带指标值"""
        values = indicator_data.get('values', {})
        if not values:
            return {}
        
        upper = values.get('upper', 0)
        middle = values.get('middle', 0)
        lower = values.get('lower', 0)
        price = values.get('price', 0)
        width = (upper - lower) / middle if middle != 0 else 0
        
        # 判断布林带状态
        if price > upper:
            status = "突破上轨"
        elif price < lower:
            status = "跌破下轨"
        elif price > middle:
            status = "运行于上轨与中轨之间"
        else:
            status = "运行于中轨与下轨之间"
        
        # 判断带宽状态
        width_status = "收缩" if width < 0.1 else "扩张" if width > 0.2 else "正常"
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "price": price,
            "width": width,
            "status": status,
            "width_status": width_status
        }
    
    def analyze_composite(self, result):
        """分析ZXM多周期合成"""
        # 周期权重
        period_weights = {
            'daily': 0.3,
            'weekly': 0.5,
            'monthly': 0.2
        }
        
        # 指标权重
        indicator_weights = {
            'ZXM_DK_SG': 0.3,
            'ZXM_MACD': 0.2,
            'ZXM_MOMENTUM': 0.2,
            'ZXM_KDJ': 0.15,
            'ZXM_BOLL': 0.15
        }
        
        # 计算综合得分
        composite_score = 0
        total_weight = 0
        
        for period_key, period_data in result.get('periods', {}).items():
            period_weight = period_weights.get(period_key, 0.1)
            
            for indicator_name, indicator_data in period_data.get('indicators', {}).items():
                for indicator_type, weight in indicator_weights.items():
                    if indicator_type in indicator_name:
                        score = indicator_data.get('score', 0)
                        indicator_weight = period_weight * weight
                        composite_score += score * indicator_weight
                        total_weight += indicator_weight
        
        # 归一化得分
        if total_weight > 0:
            normalized_score = composite_score / total_weight
        else:
            normalized_score = 0
        
        # 综合评级
        if normalized_score >= 0.8:
            rating = "强烈买入"
        elif normalized_score >= 0.6:
            rating = "买入"
        elif normalized_score >= 0.4:
            rating = "观望偏多"
        elif normalized_score >= 0.2:
            rating = "观望"
        elif normalized_score >= 0:
            rating = "观望偏空"
        else:
            rating = "回避"
        
        return {
            "composite_score": normalized_score * 100,
            "composite_rating": rating
        }
    
    def analyze_trend_consistency(self, result):
        """分析ZXM多周期趋势一致性"""
        # 记录各周期趋势
        trends = {}
        
        # 遍历各周期指标
        for period_key, period_data in result.get('periods', {}).items():
            period_trend = None
            trend_strength = 0
            
            # 分析该周期的趋势
            for indicator_name, indicator_data in period_data.get('indicators', {}).items():
                if 'ZXM_DK_SG' in indicator_name:
                    values = indicator_data.get('values', {})
                    dk_value = values.get('dk_value', 0)
                    
                    if dk_value > 0:
                        indicator_trend = "上升"
                        indicator_strength = dk_value
                    elif dk_value < 0:
                        indicator_trend = "下降"
                        indicator_strength = -dk_value
                    else:
                        indicator_trend = "盘整"
                        indicator_strength = 0
                    
                    # 使用多空势格作为主要趋势判断
                    if period_trend is None or indicator_strength > trend_strength:
                        period_trend = indicator_trend
                        trend_strength = indicator_strength
                
                elif 'ZXM_MACD' in indicator_name:
                    values = indicator_data.get('values', {})
                    dif = values.get('dif', 0)
                    dea = values.get('dea', 0)
                    
                    if dif > dea:
                        indicator_trend = "上升"
                        indicator_strength = abs(dif - dea)
                    elif dif < dea:
                        indicator_trend = "下降"
                        indicator_strength = abs(dif - dea)
                    else:
                        indicator_trend = "盘整"
                        indicator_strength = 0
                    
                    # 更新周期趋势（如果强度更大）
                    if period_trend is None or indicator_strength > trend_strength:
                        period_trend = indicator_trend
                        trend_strength = indicator_strength
            
            # 记录该周期的趋势
            if period_trend:
                trends[period_key] = {
                    "trend": period_trend,
                    "strength": trend_strength
                }
        
        # 分析趋势一致性
        unique_trends = set(trend_data["trend"] for trend_data in trends.values())
        
        if len(unique_trends) == 1:
            trend = next(iter(unique_trends))
            consistency = "高"
            consistency_description = f"所有周期都是{trend}趋势"
        elif len(unique_trends) == 2 and "盘整" in unique_trends:
            # 如果只有盘整和另一个趋势，判断为中等一致性
            other_trend = next(trend for trend in unique_trends if trend != "盘整")
            consistency = "中"
            consistency_description = f"部分周期为{other_trend}趋势，部分为盘整"
        else:
            consistency = "低"
            consistency_description = "各周期趋势不一致"
        
        # 判断多周期趋势方向
        daily_trend = trends.get("daily", {}).get("trend", "盘整")
        weekly_trend = trends.get("weekly", {}).get("trend", "盘整")
        monthly_trend = trends.get("monthly", {}).get("trend", "盘整")
        
        # 给出趋势方向建议
        if weekly_trend == "上升" and monthly_trend != "下降":
            direction = "上升"
        elif weekly_trend == "下降" and monthly_trend != "上升":
            direction = "下降"
        elif monthly_trend == "上升" and weekly_trend != "下降":
            direction = "潜在上升"
        elif monthly_trend == "下降" and weekly_trend != "上升":
            direction = "潜在下降"
        else:
            direction = "盘整"
        
        # 给出操作建议
        if direction == "上升":
            if daily_trend == "上升":
                advice = "可以积极买入"
            elif daily_trend == "盘整":
                advice = "回调时买入"
            else:
                advice = "等待日线回调结束后买入"
        elif direction == "潜在上升":
            if daily_trend == "上升":
                advice = "谨慎买入，设置止损"
            else:
                advice = "等待更明确的信号"
        elif direction == "下降":
            if daily_trend == "下降":
                advice = "回避"
            else:
                advice = "反弹时减仓"
        elif direction == "潜在下降":
            if daily_trend == "下降":
                advice = "回避"
            else:
                advice = "减仓或设置止损"
        else:
            advice = "观望，等待趋势明朗"
        
        return {
            "trends": trends,
            "consistency": consistency,
            "consistency_description": consistency_description,
            "direction": direction,
            "advice": advice
        }

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1] if hasattr(score, 'iloc') else score

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "健康状态良好" in patterns:
            pattern_boost += 0.15
        elif "健康状态优秀" in patterns:
            pattern_boost += 0.2
        elif "健康状态较差" in patterns:
            pattern_boost -= 0.1

        if "机会评分高" in patterns:
            pattern_boost += 0.15
        elif "机会评分低" in patterns:
            pattern_boost -= 0.1

        if "趋势健康" in patterns:
            pattern_boost += 0.1
        elif "动量健康" in patterns:
            pattern_boost += 0.1
        elif "成交量健康" in patterns:
            pattern_boost += 0.08

        # 最终置信度
        final_confidence = min(1.0, max(0.0, base_confidence + pattern_boost))
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        return self.identify_patterns(data, **kwargs)

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - lookback_period: 回溯分析周期，默认60
                - signal_threshold: 信号阈值，默认70
                - require_volume: 是否要求成交量数据，默认True
        """
        self.lookback_period = kwargs.get('lookback_period', 60)
        self.signal_threshold = kwargs.get('signal_threshold', 70)
        self.require_volume = kwargs.get('require_volume', True)

        # 更新权重配置
        if 'health_weights' in kwargs:
            self.health_weights.update(kwargs['health_weights'])
        if 'opportunity_weights' in kwargs:
            self.opportunity_weights.update(kwargs['opportunity_weights'])
    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # ZXMDiagnostics指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)

    def _calculate_health_score(self, result: pd.DataFrame) -> pd.Series:
        """计算综合健康度评分"""
        health_score = pd.Series(50.0, index=result.index)

        # 基于各个健康度指标计算综合评分
        if 'trend_health' in result.columns:
            health_score += (result['trend_health'] - 50) * 0.3
        if 'momentum_health' in result.columns:
            health_score += (result['momentum_health'] - 50) * 0.3
        if 'volume_health' in result.columns:
            health_score += (result['volume_health'] - 50) * 0.2
        if 'volatility_health' in result.columns:
            health_score += (result['volatility_health'] - 50) * 0.2

        return health_score.clip(0, 100)

    def _calculate_opportunity_score(self, result: pd.DataFrame) -> pd.Series:
        """计算综合机会评分"""
        opportunity_score = pd.Series(50.0, index=result.index)

        # 基于各个机会指标计算综合评分
        if 'breakout_score' in result.columns:
            opportunity_score += (result['breakout_score'] - 50) * 0.4
        if 'reversal_score' in result.columns:
            opportunity_score += (result['reversal_score'] - 50) * 0.3
        if 'sr_health' in result.columns:
            opportunity_score += (result['sr_health'] - 50) * 0.3

        return opportunity_score.clip(0, 100)

    def _calculate_diagnosis_score(self, result: pd.DataFrame) -> pd.Series:
        """计算总体诊断评分"""
        diagnosis_score = pd.Series(50.0, index=result.index)

        # 基于健康度和机会度计算总体评分
        if 'health_score' in result.columns:
            diagnosis_score += (result['health_score'] - 50) * 0.6
        if 'opportunity_score' in result.columns:
            diagnosis_score += (result['opportunity_score'] - 50) * 0.4

        return diagnosis_score.clip(0, 100)

    def _generate_diagnosis_conclusion(self, result: pd.DataFrame) -> pd.Series:
        """生成诊断结论"""
        conclusions = pd.Series("中性", index=result.index)

        if 'diagnosis_score' in result.columns:
            conclusions[result['diagnosis_score'] >= 70] = "健康"
            conclusions[result['diagnosis_score'] <= 30] = "不健康"

        return conclusions

    def _identify_main_issues(self, result: pd.DataFrame) -> pd.Series:
        """识别主要问题"""
        issues = pd.Series("无明显问题", index=result.index)

        # 基于各项指标识别问题
        if 'trend_health' in result.columns:
            issues[result['trend_health'] < 30] = "趋势问题"
        if 'momentum_health' in result.columns:
            issues[result['momentum_health'] < 30] = "动量问题"

        return issues

    def _generate_recommendations(self, result: pd.DataFrame) -> pd.Series:
        """生成建议"""
        recommendations = pd.Series("持续观察", index=result.index)

        if 'diagnosis_score' in result.columns:
            recommendations[result['diagnosis_score'] >= 70] = "可考虑买入"
            recommendations[result['diagnosis_score'] <= 30] = "建议规避"

        return recommendations

    # 形态识别方法
    def _identify_trend_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别趋势相关形态"""
        pass  # 简化实现

    def _identify_momentum_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别动量相关形态"""
        pass  # 简化实现

    def _identify_volume_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别成交量相关形态"""
        pass  # 简化实现

    def _identify_volatility_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别波动率相关形态"""
        pass  # 简化实现

    def _identify_support_resistance_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别支撑阻力相关形态"""
        pass  # 简化实现

    def _identify_cycle_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别周期相关形态"""
        pass  # 简化实现

    def _identify_composite_patterns(self, data: pd.DataFrame, diagnosis_result: pd.DataFrame, patterns: pd.DataFrame, min_strength: float):
        """识别综合技术形态"""
        pass  # 简化实现
