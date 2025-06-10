"""
指标适配器辅助模块

提供各种指标接口的适配器，用于测试具有特殊接口的指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union


class DataFrameToZXMAdapter:
    """DataFrame到ZXM接口的适配器"""
    
    def __init__(self, zxm_indicator):
        """
        初始化适配器
        
        Args:
            zxm_indicator: ZXM形态指标实例
        """
        self.zxm_indicator = zxm_indicator
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用DataFrame计算ZXM形态指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含形态识别结果的DataFrame
        """
        # 提取需要的数据
        open_prices = data['open'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        close_prices = data['close'].values
        volumes = data['volume'].values
        
        # 调用ZXM指标计算
        result_dict = self.zxm_indicator.calculate(
            open_prices, high_prices, low_prices, close_prices, volumes
        )
        
        # 将结果添加到原始DataFrame
        result_df = data.copy()
        for key, value in result_dict.items():
            result_df[key] = value
            
        return result_df


class IndicatorToSelectorAdapter:
    """将指标适配为选股器的适配器"""
    
    def __init__(self, indicator, threshold: float = 70.0, signal_field: str = None):
        """
        初始化适配器
        
        Args:
            indicator: 要适配的指标实例
            threshold: 评分阈值，高于此值的股票将被选中
            signal_field: 要使用的信号字段，如不指定则使用原始评分
        """
        self.indicator = indicator
        self.threshold = threshold
        self.signal_field = signal_field
        
    def select(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        使用指标选择股票
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            选中的股票列表
        """
        # 计算指标
        result = self.indicator.calculate(data)
        
        # 获取原始评分
        if hasattr(self.indicator, 'calculate_raw_score'):
            scores = self.indicator.calculate_raw_score(result)
        else:
            scores = pd.Series(50, index=data.index)  # 默认中性评分
        
        # 如果指定了信号字段，使用信号字段
        if self.signal_field and self.signal_field in result.columns:
            signals = result[self.signal_field]
        else:
            signals = scores
        
        # 选择评分高于阈值的股票
        selected_indices = signals[signals > self.threshold].index
        
        # 创建选中结果
        selected = []
        for idx in selected_indices:
            row = data.loc[idx]
            
            # 获取最新评分
            latest_score = scores.loc[idx]
            
            # 创建选中项
            selected_item = {
                'code': row['code'] if 'code' in row else 'unknown',
                'date': idx if isinstance(idx, str) else str(idx),
                'score': float(latest_score),
                'signal': float(signals.loc[idx]),
                'price': float(row['close']),
                'indicator': self.indicator.name if hasattr(self.indicator, 'name') else 'unknown'
            }
            
            selected.append(selected_item)
            
        return selected


class MultiIndicatorAdapter:
    """多指标组合适配器"""
    
    def __init__(self, indicators: List, weights: List[float] = None):
        """
        初始化适配器
        
        Args:
            indicators: 要组合的指标列表
            weights: 各指标的权重列表，如不指定则平均分配
        """
        self.indicators = indicators
        
        # 设置权重
        if weights is None:
            self.weights = [1.0 / len(indicators)] * len(indicators)
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # 基础属性
        self.name = "CombinedIndicator"
        self.description = "多指标组合"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算组合指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含组合指标结果的DataFrame
        """
        result = data.copy()
        
        # 计算每个指标
        for indicator in self.indicators:
            indicator_result = indicator.calculate(data)
            
            # 如果返回的是DataFrame，合并结果
            if isinstance(indicator_result, pd.DataFrame):
                for col in indicator_result.columns:
                    if col not in result.columns:
                        result[col] = indicator_result[col]
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算组合指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            组合评分Series
        """
        # 初始化权重和
        weighted_sum = pd.Series(0, index=data.index)
        
        # 累加每个指标的加权评分
        for indicator, weight in zip(self.indicators, self.weights):
            if hasattr(indicator, 'calculate_raw_score'):
                indicator_result = indicator.calculate(data)
                score = indicator.calculate_raw_score(indicator_result)
                weighted_sum += score * weight
        
        return weighted_sum.clip(0, 100)  # 确保评分在0-100范围内


class PatternDetectionAdapter:
    """形态检测适配器"""
    
    def __init__(self, pattern_indicator, pattern_names: List[str] = None):
        """
        初始化适配器
        
        Args:
            pattern_indicator: 形态识别指标
            pattern_names: 要检测的形态名称列表，如不指定则检测所有形态
        """
        self.pattern_indicator = pattern_indicator
        self.pattern_names = pattern_names
        
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """
        检测形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            形态检测结果字典，键为形态名称，值为形态出现的位置列表
        """
        # 计算指标
        result = self.pattern_indicator.calculate(data)
        
        # 如果指标返回的是字典(如ZXM形态指标)
        if isinstance(result, dict):
            patterns = {}
            
            # 筛选要检测的形态
            for key, value in result.items():
                if self.pattern_names is None or key in self.pattern_names:
                    patterns[key] = np.where(value)[0].tolist()
            
            return patterns
        
        # 如果指标返回的是DataFrame
        elif isinstance(result, pd.DataFrame):
            patterns = {}
            
            # 查找形态列
            pattern_cols = [col for col in result.columns if 'pattern' in col.lower() or 'signal' in col.lower()]
            
            # 筛选要检测的形态
            for col in pattern_cols:
                if self.pattern_names is None or col in self.pattern_names:
                    # 找出列值为1或True的位置
                    patterns[col] = np.where(result[col].astype(bool))[0].tolist()
            
            return patterns
        
        # 如果指标有identify_patterns方法
        elif hasattr(self.pattern_indicator, 'identify_patterns'):
            patterns = self.pattern_indicator.identify_patterns(data)
            
            # 如果返回的是列表，转换为字典
            if isinstance(patterns, list):
                return {'pattern': [i for i in range(len(data)) if patterns[i]]}
            
            # 如果返回的是字典
            elif isinstance(patterns, dict):
                return patterns
        
        # 默认返回空结果
        return {} 