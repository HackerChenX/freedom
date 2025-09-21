#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ISLAND_REVERSAL 指标

岛型反转形态识别指标，用于识别价格的岛型反转模式
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from utils.container import container
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin

logger = get_logger(__name__)


class IslandReversal(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    ISLAND_REVERSAL 指标
    
    自动生成的最小化实现,支持参数标准化
    """
    
    def __init__(self, **kwargs):
        """
        初始化ISLAND_REVERSAL指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__(name="ISLAND_REVERSAL", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_islandreversal()
        
        # 应用用户参数
        self.set_parameters_Reversal(**kwargs)
    
    def _get_default_parameters_islandreversal(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}  # TODO: 将魔法数字提取到配置中
    
    def set_parameters_Reversal(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
        except Exception as e:
            logger.error(f"错误: {e}")
            return pd.DataFrame()
    
    def validate_parameters(self, **kwargs):
        """验证参数"""
        try:
            validator = IndicatorParameterValidator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('ISLAND_REVERSAL', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()
            
            # 设置参数
            self.period = params.get('period', 14)  # TODO: 将魔法数字提取到配置中
                    
        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            self.period = 14  # TODO: 将魔法数字提取到配置中
    
    def calculate_Reversal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ISLAND_REVERSAL指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了ISLAND_REVERSAL指标的Data_frame
        """
        result = self._calculate_islandreversal(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_islandreversal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ISLAND_REVERSAL指标

        Args:
            data: 包含OHLCV数据的Data_frame

        Returns:
            添加了ISLAND_REVERSAL指标的Data_frame
        """
        df = data.copy()

        # 真实的岛形反转算法实现
        # 1. 识别跳空缺口
        df['up_gap'] = (df['low'] > df['high'].shift(1)) & (df['low'].shift(1).notna())
        df['down_gap'] = (df['high'] < df['low'].shift(1)) & (df['high'].shift(1).notna())

        # 2. 识别岛形反转形态
        df['top_island_reversal'] = False
        df['bottom_island_reversal'] = False

        # 顶部岛形反转:向上跳空后又向下跳空
        for i in range(2, len(df)):
            # 检查是否有向上跳空
            if df['up_gap'].iloc[i-1]:
                # 检查后续是否有向下跳空
                for j in range(i, min(i+5, len(df))):  # 在5天内寻找  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    if df['down_gap'].iloc[j]:
                        df.iloc[j, df.columns.get_loc('top_island_reversal')] = True
                        break

        # 底部岛形反转:向下跳空后又向上跳空
        for i in range(2, len(df)):
            # 检查是否有向下跳空
            if df['down_gap'].iloc[i-1]:
                # 检查后续是否有向上跳空
                for j in range(i, min(i+5, len(df))):  # 在5天内寻找  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    if df['up_gap'].iloc[j]:
                        df.iloc[j, df.columns.get_loc('bottom_island_reversal')] = True
                        break

        # 3. 计算岛形反转强度  # TODO: 将魔法数字提取到配置中
        df['island_reversal_strength'] = 0.0
        df.loc[df['top_island_reversal'], 'island_reversal_strength'] = -1.0  # 看跌信号
        df.loc[df['bottom_island_reversal'], 'island_reversal_strength'] = 1.0  # 看涨信号

        # 4. 计算跳空幅度  # TODO: 将魔法数字提取到配置中
        df['gap_size'] = 0.0
        df.loc[df['up_gap'], 'gap_size'] = (df['low'] - df['high'].shift(1)) / df['close'].shift(1)
        df.loc[df['down_gap'], 'gap_size'] = (df['high'] - df['low'].shift(1)) / df['close'].shift(1)

        # 5. 岛形反转综合信号  # TODO: 将魔法数字提取到配置中
        df['island_reversal_signal'] = df['top_island_reversal'] | df['bottom_island_reversal']

        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def calculate_raw_score_Reversal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Reversal(data, **kwargs)
        return pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
    
    def calculate_confidence_Reversal(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5  # TODO: 将魔法数字提取到配置中
    
    def get_patterns_Reversal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算岛形反转指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含岛形反转指标的DataFrame
        """
        return self.calculate_Reversal(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置基础指标参数"""
        return self.set_parameters_Reversal(**kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """基础指标计算方法"""
        return self.calculate_Reversal(data, *args, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标置信度"""
        result = self._calculate_baseindicator(data)
        # 计算岛形反转的置信度
        score = self.calculate_raw_score_Reversal(data)
        confidence = score.mean() / 100.0  # 将评分转换为置信度
        result['confidence'] = confidence
        return result

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算原始评分"""
        result = self._calculate_baseindicator(data)
        score = self.calculate_raw_score_Reversal(data)
        result['raw_score'] = score
        return result

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取形态识别结果"""
        return self.get_patterns_Reversal(data)

    @property
    def minimum_periods(self) -> int:
        """
        IslandReversal指标所需的最少数据周期数
        
        计算逻辑:使用默认值
        
        Returns:
            int: 最少需要的数据周期数
        """
        return 30
    
    # ================== 标准抽象方法实现 ==================
    
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        获取岛型反转交易信号 - BaseIndicator抽象方法实现
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 标准化的交易信号字典
        """
        if data is None or len(data) < self.minimum_periods:
            return self._get_default_signal()
        
        try:
            # 计算岛型反转指标
            result = self.calculate(data, **kwargs)
            
            if result is None or len(result) == 0:
                return self._get_default_signal()
            
            # 获取最新的指标值
            latest_data = result.iloc[-1]
            
            # 基于岛型反转形态生成信号
            signal_type = "HOLD"
            strength = 50.0
            confidence = 50.0
            reason = "无明显岛型反转信号"
            
            # 检查岛型反转形态
            if 'island_reversal_value' in latest_data:
                island_value = latest_data['island_reversal_value']
                
                if island_value > 0.8:  # 强烈的岛型反转买入信号
                    signal_type = "BUY"
                    strength = min(100.0, 60 + island_value * 40)
                    confidence = min(100.0, 70 + island_value * 25)
                    reason = f"检测到强烈岛型反转买入信号，强度: {island_value:.2f}"
                elif island_value < -0.8:  # 强烈的岛型反转卖出信号
                    signal_type = "SELL"
                    strength = min(100.0, 60 + abs(island_value) * 40)
                    confidence = min(100.0, 70 + abs(island_value) * 25)
                    reason = f"检测到强烈岛型反转卖出信号，强度: {abs(island_value):.2f}"
                elif abs(island_value) > 0.4:  # 中等强度信号
                    signal_type = "BUY" if island_value > 0 else "SELL"
                    strength = min(100.0, 45 + abs(island_value) * 25)
                    confidence = min(100.0, 55 + abs(island_value) * 20)
                    reason = f"检测到中等岛型反转{'买入' if island_value > 0 else '卖出'}信号，强度: {abs(island_value):.2f}"
            
            # 检查是否有形态识别结果
            if hasattr(self, '_result') and self._result is not None:
                patterns = self.get_patterns_Reversal(data)
                if patterns is not None and len(patterns) > 0:
                    # 基于形态数量调整信号强度
                    pattern_count = patterns.sum().sum() if hasattr(patterns, 'sum') else 0
                    if pattern_count > 0:
                        strength = min(100.0, strength + pattern_count * 5)
                        confidence = min(100.0, confidence + pattern_count * 3)
                        reason += f"，发现{pattern_count}个相关形态"
            
            # 构建标准化信号字典
            signal = {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence,
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'indicator': 'ISLAND_REVERSAL',
                    'data_points': len(data),
                    'latest_value': latest_data.get('island_reversal_value', 0.0) if 'island_reversal_value' in latest_data else 0.0,
                    'pattern_detected': abs(latest_data.get('island_reversal_value', 0.0)) > 0.4 if 'island_reversal_value' in latest_data else False
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"岛型反转信号生成失败: {e}")
            return self._get_default_signal()
    
    def _get_default_signal(self) -> Dict[str, Any]:
        """获取默认信号"""
        return {
            'signal_type': 'HOLD',
            'strength': 50.0,
            'confidence': 50.0,
            'timestamp': pd.Timestamp.now(),
            'reason': '数据不足或计算失败',
            'metadata': {
                'indicator': 'ISLAND_REVERSAL',
                'data_points': 0,
                'latest_value': 0.0,
                'pattern_detected': False
            }
        }