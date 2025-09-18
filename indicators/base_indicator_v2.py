#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BaseIndicator V2.0 - 重构版技术指标基类

解决问题：
1. 方法命名混乱：get_signal vs get_signals
2. 返回格式不统一：DataFrame vs Dict
3. 缺乏统一标准：列名、数据类型不规范
4. 职责不清晰：calculate、signal、pattern方法关系模糊

新设计原则：
1. 清晰的方法职责划分
2. 统一的返回格式标准
3. 项目级命名规范
4. 完整的文档体系
"""

import abc
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.decorators import performance_monitor, exception_handler
from utils.container import container
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """交易信号类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = 0.3
    MEDIUM = 0.6
    STRONG = 0.9


@dataclass
class IndicatorSignal:
    """标准化的指标信号数据结构"""
    signal_type: SignalType
    strength: float
    confidence: float
    timestamp: datetime
    price: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'price': self.price,
            'metadata': self.metadata or {}
        }


@dataclass
class IndicatorMetadata:
    """指标元数据标准结构"""
    name: str
    display_name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    output_columns: List[str]
    calculation_period: int
    data_requirements: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'category': self.category,
            'description': self.description,
            'parameters': self.parameters,
            'output_columns': self.output_columns,
            'calculation_period': self.calculation_period,
            'data_requirements': self.data_requirements
        }


class StandardColumnNames:
    """项目级统一列名标准"""
    
    # 基础价格数据列
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TURNOVER_RATE = "turnover_rate"
    
    # 标准指标输出列（使用指标名前缀）
    # MACD系列
    MACD_DIF = "macd_dif"
    MACD_DEA = "macd_dea"
    MACD_HISTOGRAM = "macd_histogram"
    
    # RSI系列
    RSI_VALUE = "rsi_value"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    
    # KDJ系列
    KDJ_K = "kdj_k"
    KDJ_D = "kdj_d"
    KDJ_J = "kdj_j"
    
    # 通用信号列
    BUY_SIGNAL = "buy_signal"
    SELL_SIGNAL = "sell_signal"
    HOLD_SIGNAL = "hold_signal"
    SIGNAL_STRENGTH = "signal_strength"
    SIGNAL_CONFIDENCE = "signal_confidence"


class BaseIndicatorV2(abc.ABC):
    """
    技术指标基类 V2.0 - 重构版
    
    核心设计原则：
    1. 清晰的方法职责：calculate -> get_signals -> get_latest_signal
    2. 统一的返回格式：DataFrame for batch, IndicatorSignal for single
    3. 标准化列名：使用StandardColumnNames
    4. 完整的元数据：IndicatorMetadata
    
    数据流设计：
    原始数据 -> calculate() -> 指标DataFrame -> get_signals() -> 信号DataFrame -> get_latest_signal() -> IndicatorSignal
    """
    
    def __init__(self, name: str = "", **kwargs):
        """
        初始化指标
        
        Args:
            name: 指标名称
            **kwargs: 指标参数
        """
        self.name = name or self.__class__.__name__
        self.parameters = kwargs
        self._result_cache = None
        self._signals_cache = None
        
        # 严格依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        # 验证依赖注入
        if not self.data_access or not self.cache_service:
            raise RuntimeError("依赖注入失败，请检查容器配置")
        
        # 初始化指标特定配置
        self._initialize_indicator()
    
    def _initialize_indicator(self):
        """子类可重写的初始化方法"""
        pass
    
    # ==================== 核心抽象方法 ====================
    
    @abc.abstractmethod
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值 - 核心计算方法
        
        职责：
        1. 验证输入数据
        2. 执行指标计算
        3. 返回标准化的指标DataFrame
        
        Args:
            data: 输入数据，必须包含OHLCV字段
            
        Returns:
            pd.DataFrame: 包含指标计算结果的DataFrame
                         列名必须使用StandardColumnNames中定义的标准
                         
        Raises:
            ValueError: 输入数据不符合要求
            
        示例：
            对于MACD指标，返回的DataFrame应包含：
            - macd_dif: DIF值
            - macd_dea: DEA值  
            - macd_histogram: MACD柱状图
        """
        pass
    
    @abc.abstractmethod
    def get_metadata(self) -> IndicatorMetadata:
        """
        获取指标元数据
        
        职责：
        1. 提供指标的完整描述信息
        2. 定义输出列名标准
        3. 说明数据要求和参数
        
        Returns:
            IndicatorMetadata: 标准化的指标元数据
        """
        pass
    
    # ==================== 标准信号方法 ====================
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成批量交易信号 - 标准信号生成方法
        
        职责：
        1. 基于calculate()结果生成信号
        2. 返回标准化的信号DataFrame
        3. 包含买卖信号和强度信息
        
        Args:
            data: 原始数据或包含指标计算结果的数据
            
        Returns:
            pd.DataFrame: 包含以下标准列的信号DataFrame：
                - buy_signal: bool, 买入信号
                - sell_signal: bool, 卖出信号
                - hold_signal: bool, 持有信号
                - signal_strength: float, 信号强度(0-1)
                - signal_confidence: float, 信号置信度(0-1)
        """
        # 确保有指标计算结果
        if self._result_cache is None:
            self._result_cache = self.calculate(data)
        
        # 生成信号
        signals_df = self._generate_signals(self._result_cache, data)
        self._signals_cache = signals_df
        
        return signals_df
    
    def get_latest_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """
        获取最新交易信号 - 单一信号获取方法
        
        职责：
        1. 获取最新的交易信号
        2. 返回结构化的信号对象
        3. 提供详细的信号信息
        
        Args:
            data: 原始数据
            
        Returns:
            IndicatorSignal: 结构化的最新信号对象
        """
        signals_df = self.get_signals(data)
        
        if signals_df.empty:
            return IndicatorSignal(
                signal_type=SignalType.HOLD,
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                price=data[StandardColumnNames.CLOSE].iloc[-1] if not data.empty else 0.0
            )
        
        # 获取最新信号
        latest_row = signals_df.iloc[-1]
        
        # 确定信号类型
        if latest_row.get(StandardColumnNames.BUY_SIGNAL, False):
            signal_type = SignalType.BUY
        elif latest_row.get(StandardColumnNames.SELL_SIGNAL, False):
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        return IndicatorSignal(
            signal_type=signal_type,
            strength=latest_row.get(StandardColumnNames.SIGNAL_STRENGTH, 0.0),
            confidence=latest_row.get(StandardColumnNames.SIGNAL_CONFIDENCE, 0.0),
            timestamp=latest_row.name if hasattr(latest_row, 'name') else datetime.now(),
            price=data[StandardColumnNames.CLOSE].iloc[-1] if not data.empty else 0.0,
            metadata={'indicator': self.name}
        )
    
    # ==================== 子类可重写的方法 ====================
    
    def _generate_signals(self, indicator_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成信号的具体实现 - 子类可重写
        
        默认实现提供基础的信号生成逻辑，子类可以重写以实现特定的信号策略
        
        Args:
            indicator_data: 指标计算结果
            original_data: 原始价格数据
            
        Returns:
            pd.DataFrame: 标准化的信号DataFrame
        """
        signals_df = pd.DataFrame(index=indicator_data.index)
        
        # 默认实现：基于指标值的简单信号
        signals_df[StandardColumnNames.BUY_SIGNAL] = False
        signals_df[StandardColumnNames.SELL_SIGNAL] = False
        signals_df[StandardColumnNames.HOLD_SIGNAL] = True
        signals_df[StandardColumnNames.SIGNAL_STRENGTH] = 0.0
        signals_df[StandardColumnNames.SIGNAL_CONFIDENCE] = 0.0
        
        return signals_df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 验证结果
        """
        if data is None or data.empty:
            return False
        
        # 检查必需的列
        required_columns = [StandardColumnNames.CLOSE]
        return all(col in data.columns for col in required_columns)
    
    # ==================== 工具方法 ====================
    
    def clear_cache(self):
        """清除缓存"""
        self._result_cache = None
        self._signals_cache = None
    
    def get_result_cache(self) -> Optional[pd.DataFrame]:
        """获取计算结果缓存"""
        return self._result_cache
    
    def get_signals_cache(self) -> Optional[pd.DataFrame]:
        """获取信号缓存"""
        return self._signals_cache
