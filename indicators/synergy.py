"""
指标协同框架模块

提供技术指标之间协同工作的框架和功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple, Type
import copy

from indicators.base_indicator import BaseIndicator, MarketEnvironment
from indicators.market_env import MarketDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorSynergy:
    """
    指标协同框架
    
    用于协调多个技术指标之间的关系，提供综合信号和评分
    """
    
    def __init__(self, market_detector: Optional[MarketDetector] = None):
        """
        初始化指标协同框架
        
        Args:
            market_detector: 市场环境检测器，如未提供则创建默认检测器
        """
        self.indicators = {}  # 存储技术指标实例
        self.market_detector = market_detector or MarketDetector()
        self.correlation_matrix = None
        self.market_environment = MarketEnvironment.SIDEWAYS_MARKET
        self.weight_matrix = {}  # 存储各指标权重
    
    def add_indicator(self, indicator: BaseIndicator, weight: float = 1.0) -> None:
        """
        添加技术指标
        
        Args:
            indicator: 技术指标实例
            weight: 指标权重
        """
        if indicator.name in self.indicators:
            logger.warning(f"指标 '{indicator.name}' 已存在，将被替换")
            
        self.indicators[indicator.name] = indicator
        self.weight_matrix[indicator.name] = weight
        logger.debug(f"添加指标 '{indicator.name}'，权重 {weight}")
    
    def remove_indicator(self, indicator_name: str) -> bool:
        """
        移除技术指标
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            bool: 是否成功移除
        """
        if indicator_name in self.indicators:
            del self.indicators[indicator_name]
            if indicator_name in self.weight_matrix:
                del self.weight_matrix[indicator_name]
            logger.debug(f"移除指标 '{indicator_name}'")
            return True
        return False
    
    def get_indicator(self, indicator_name: str) -> Optional[BaseIndicator]:
        """
        获取技术指标实例
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Optional[BaseIndicator]: 指标实例，如不存在则返回None
        """
        return self.indicators.get(indicator_name)
    
    def detect_market_environment(self, data: pd.DataFrame) -> MarketEnvironment:
        """
        检测市场环境
        
        Args:
            data: 输入数据
            
        Returns:
            MarketEnvironment: 检测到的市场环境
        """
        env = self.market_detector.detect_environment(data)
        self.market_environment = env
        
        # 将环境设置到所有指标
        for indicator in self.indicators.values():
            indicator.set_market_environment(env)
            
        logger.info(f"检测到市场环境: {env.value}")
        return env
    
    def initialize_indicators(self, data: pd.DataFrame) -> None:
        """
        初始化所有指标
        
        Args:
            data: 输入数据
        """
        # 检测市场环境
        self.detect_market_environment(data)
        
        # 计算所有指标
        for name, indicator in self.indicators.items():
            try:
                indicator.safe_compute(data)
                logger.debug(f"初始化指标 '{name}' 完成")
            except Exception as e:
                logger.error(f"初始化指标 '{name}' 失败: {e}")
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, lookback_period: int = 60) -> pd.DataFrame:
        """
        计算指标相关性矩阵
        
        Args:
            data: 输入数据
            lookback_period: 回看周期
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        if len(data) < lookback_period:
            logger.warning(f"数据不足，需要至少 {lookback_period} 条数据")
            return pd.DataFrame()
        
        # 计算所有指标
        self.initialize_indicators(data)
        
        # 收集指标得分
        scores = {}
        recent_data = data.tail(lookback_period)
        
        for name, indicator in self.indicators.items():
            try:
                score_result = indicator.calculate_score(recent_data)
                scores[name] = score_result['final_score']
            except Exception as e:
                logger.error(f"计算指标 '{name}' 得分失败: {e}")
        
        # 创建得分DataFrame
        score_df = pd.DataFrame(scores)
        
        # 计算相关性矩阵
        self.correlation_matrix = score_df.corr()
        return self.correlation_matrix
    
    def get_correlated_indicators(self, indicator_name: str, threshold: float = 0.7) -> Dict[str, float]:
        """
        获取与指定指标高相关的其他指标
        
        Args:
            indicator_name: 指标名称
            threshold: 相关性阈值，绝对值大于此值视为高相关
            
        Returns:
            Dict[str, float]: 高相关指标及其相关系数
        """
        if self.correlation_matrix is None or indicator_name not in self.correlation_matrix:
            return {}
        
        correlations = self.correlation_matrix[indicator_name]
        
        # 筛选高相关指标
        high_corr = {}
        for name, corr in correlations.items():
            if name != indicator_name and abs(corr) >= threshold:
                high_corr[name] = corr
                
        return high_corr
    
    def adjust_weights_by_correlation(self, min_weight: float = 0.5) -> None:
        """
        基于相关性矩阵调整指标权重
        
        高相关指标的权重会被降低，以避免信息冗余
        
        Args:
            min_weight: 最小权重
        """
        if self.correlation_matrix is None:
            logger.warning("相关性矩阵未计算，无法调整权重")
            return
        
        # 创建权重调整因子
        weight_factors = {name: 1.0 for name in self.indicators}
        
        # 计算每对指标的相关性，并调整权重
        for i, name1 in enumerate(self.correlation_matrix.columns):
            for name2 in self.correlation_matrix.columns[i+1:]:
                corr = abs(self.correlation_matrix.loc[name1, name2])
                
                if corr > 0.7:
                    # 高相关指标，降低权重
                    reduction = (corr - 0.7) / 0.3 * 0.5  # 最多降低50%
                    weight_factors[name1] *= (1 - reduction / 2)
                    weight_factors[name2] *= (1 - reduction / 2)
        
        # 应用权重调整
        for name, factor in weight_factors.items():
            # 确保权重不低于最小值
            self.weight_matrix[name] = max(min_weight, self.weight_matrix[name] * factor)
        
        logger.info(f"基于相关性调整后的权重: {self.weight_matrix}")
    
    def generate_combined_score(self, data: pd.DataFrame) -> pd.Series:
        """
        生成综合评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 综合评分序列
        """
        # 计算所有指标的评分
        indicator_scores = {}
        
        for name, indicator in self.indicators.items():
            try:
                score_result = indicator.calculate_score(data)
                indicator_scores[name] = score_result['final_score']
            except Exception as e:
                logger.error(f"计算指标 '{name}' 得分失败: {e}")
        
        if not indicator_scores:
            logger.warning("没有有效的指标评分")
            return pd.Series(50.0, index=data.index)
        
        # 标准化权重
        total_weight = sum(self.weight_matrix.values())
        normalized_weights = {k: w / total_weight for k, w in self.weight_matrix.items()}
        
        # 加权平均
        combined_score = pd.Series(0.0, index=data.index)
        
        for name, score in indicator_scores.items():
            if name in normalized_weights:
                combined_score += score * normalized_weights[name]
        
        return combined_score
    
    def generate_combined_signals(self, data: pd.DataFrame, 
                                 buy_threshold: float = 70, 
                                 sell_threshold: float = 30) -> pd.DataFrame:
        """
        生成综合信号
        
        Args:
            data: 输入数据
            buy_threshold: 买入信号阈值
            sell_threshold: 卖出信号阈值
            
        Returns:
            pd.DataFrame: 信号DataFrame
        """
        # 计算综合评分
        combined_score = self.generate_combined_score(data)
        
        # 检测市场环境
        self.detect_market_environment(data)
        
        # 根据市场环境调整阈值
        if self.market_environment == MarketEnvironment.BULL_MARKET:
            buy_threshold -= 5  # 牛市降低买入门槛
            sell_threshold -= 10  # 牛市提高卖出门槛
        elif self.market_environment == MarketEnvironment.BEAR_MARKET:
            buy_threshold += 10  # 熊市提高买入门槛
            sell_threshold += 5  # 熊市降低卖出门槛
            
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['score'] = combined_score
        signals['buy_signal'] = combined_score > buy_threshold
        signals['sell_signal'] = combined_score < sell_threshold
        signals['bull_trend'] = combined_score > 60
        signals['bear_trend'] = combined_score < 40
        signals['neutral'] = (combined_score >= 40) & (combined_score <= 60)
        signals['market_environment'] = str(self.market_environment.value)
        
        # 计算置信度
        signals['confidence'] = self.calculate_combined_confidence(data)
        
        return signals
    
    def calculate_combined_confidence(self, data: pd.DataFrame) -> pd.Series:
        """
        计算综合置信度
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 置信度序列
        """
        confidences = {}
        
        for name, indicator in self.indicators.items():
            try:
                score_result = indicator.calculate_score(data)
                confidences[name] = score_result['confidence']
            except Exception as e:
                logger.error(f"计算指标 '{name}' 置信度失败: {e}")
        
        if not confidences:
            logger.warning("没有有效的指标置信度")
            return pd.Series(50.0, index=data.index)
        
        # 使用加权平均计算综合置信度
        combined_confidence = 0
        total_weight = 0
        
        for name, confidence in confidences.items():
            if name in self.weight_matrix:
                weight = self.weight_matrix[name]
                if isinstance(confidence, pd.Series):
                    confidence = confidence.iloc[-1]
                combined_confidence += confidence * weight
                total_weight += weight
        
        final_confidence = combined_confidence / total_weight if total_weight > 0 else 50.0
        
        # 返回作为序列
        return pd.Series(final_confidence, index=data.index)
    
    def find_conflicting_signals(self, data: pd.DataFrame, threshold: float = 0.3) -> Dict[str, List[str]]:
        """
        寻找冲突信号
        
        Args:
            data: 输入数据
            threshold: 冲突阈值
            
        Returns:
            Dict[str, List[str]]: 冲突指标组
        """
        # 计算所有指标的评分
        indicator_scores = {}
        
        for name, indicator in self.indicators.items():
            try:
                score_result = indicator.calculate_score(data)
                indicator_scores[name] = score_result['final_score'].iloc[-1]  # 最新评分
            except Exception as e:
                logger.error(f"计算指标 '{name}' 得分失败: {e}")
        
        if not indicator_scores:
            return {}
        
        # 寻找冲突指标
        conflicts = {'bullish': [], 'bearish': []}
        
        for name, score in indicator_scores.items():
            if score > 70:  # 看涨指标
                conflicts['bullish'].append(name)
            elif score < 30:  # 看跌指标
                conflicts['bearish'].append(name)
        
        # 判断是否存在明显冲突
        if conflicts['bullish'] and conflicts['bearish']:
            logger.info(f"检测到信号冲突 - 看涨: {conflicts['bullish']}, 看跌: {conflicts['bearish']}")
            return conflicts
        
        return {}
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"IndicatorSynergy(indicators={list(self.indicators.keys())}, market_environment={self.market_environment.value})"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__() 