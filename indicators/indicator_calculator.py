#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标计算器实现

提供基础的指标计算器实现,满足依赖注入需求
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union

from utils.container import container
from indicators.base_indicator import BaseIndicator
from db.interfaces.indicator_calculator_interface import IindicatorCalculator, IIndicatorCalculator
from enums.indicator_types import Indicatortype_indicator_types as IndicatorType
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)


class IndicatorCalculator(BaseIndicator, IindicatorCalculator):
    """
    基础指标计算器实现
    
    提供基本的指标计算功能,可以被策略类使用
    """
    
    def __init__(self, **kwargs):
        """初始化指标计算器"""
        super().__init__(name="IndicatorCalculator", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        self.description = "基础指标计算器,提供基本的指标计算功能"
        self._indicators = {}
        logger.debug("指标计算器初始化完成")
    
    def calculate_Indicator_Calculator_Interface(self, 
                  data: pd.DataFrame, 
                  params: Optional[Dict[str, Any]] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        计算指标
        
        Args:
            data: 输入数据
            params: 计算参数
            
        Returns:
            Union[pd.Series, pd.DataFrame]: 计算结果
        """
        if params is None:
            params = {}
        
        # 简单的移动平均计算作为默认实现
        period = params.get('period', 20)  # TODO: 将魔法数字提取到配置中
        if 'close' in data.columns:
            return data['close'].rolling(window=period).mean()
        else:
            logger.warning("数据中没有找到 'close' 列,返回空序列")
            return pd.Series(index=data.index, dtype=float)
    
    def get_indicator_type_Indicator_Calculator_Interface(self) -> IndicatorType:
        """
        获取指标类型
        
        Returns:
            IndicatorType: 指标类型枚举
        """
        return IndicatorType.TREND
    
    def get_required_columns(self) -> List[str]:
        """
        获取计算所需的数据列
        
        Returns:
            List[str]: 必需的数据列名列表
        """
        return ['close']
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        获取默认参数
        
        Returns:
            Dict[str, Any]: 默认参数字典
        """
        return {'period': 20}  # TODO: 将魔法数字提取到配置中
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 验证结果
        """
        if data.empty:
            return False
        
        required_columns = self.get_required_columns()
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"数据中缺少必需的列: {col}")
                return False
        
        return True
    
    def calculate_ma_indicator_calculator(self, data: pd.DataFrame, period: int = 20) -> pd.Series:  # TODO: 将魔法数字提取到配置中
        """
        计算移动平均线
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            pd.Series: 移动平均线结果
        """
        if not self.validate_data(data):
            return pd.Series(index=data.index, dtype=float)
        
        return data['close'].rolling(window=period).mean()
    
    def calculate_rsi_indicator_calculator(self, data: pd.DataFrame, period: int = 14) -> pd.Series:  # TODO: 将魔法数字提取到配置中
        """
        计算RSI指标
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            pd.Series: RSI结果
        """
        if not self.validate_data(data):
            return pd.Series(index=data.index, dtype=float)
        
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd_indicator_calculator(self, data: pd.DataFrame, 
                      fast_period: int = 12,  # TODO: 将魔法数字提取到配置中 
                      slow_period: int = 26,  # TODO: 将魔法数字提取到配置中 
                      signal_period: int = 9) -> Dict[str, pd.Series]:  # TODO: 将魔法数字提取到配置中
        """
        计算MACD指标
        
        Args:
            data: 输入数据
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            Dict[str, pd.Series]: MACD结果字典
        """
        if not self.validate_data(data):
            empty_series = pd.Series(index=data.index, dtype=float)
            return {
                'macd': empty_series,
                'signal': empty_series,
                'histogram': empty_series
            }
        
        close = data['close']
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal
        
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值 - BaseIndicator抽象方法实现
        
        Args:
            data: 输入数据,包含OHLCV等字段
            
        Returns:
            pd.DataFrame: 包含指标计算结果的数据框
        """
        if not self.validate_data(data):
            raise ValueError("输入数据不符合要求")
        
        # 预处理数据
        processed_data = self.preprocess_data(data)
        
        # 使用移动平均作为默认计算
        result = processed_data.copy()
        period = getattr(self, 'period', 20)
        result['indicator_calculator_value'] = processed_data['close'].rolling(window=period).mean()
        
        # 后处理结果
        result = self.postprocess_result(result)
        
        # 保存结果
        self._result = result
        
        return result

    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取交易信号 - BaseIndicator抽象方法实现
        
        Args:
            data: 包含指标计算结果的数据
            
        Returns:
            Dict[str, Any]: 交易信号信息
        """
        try:
            if data.empty:
                return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}
            
            # 计算指标
            result = self.calculate(data)
            
            if result.empty or len(result) == 0:
                return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}
            
            # 获取最新的指标值
            latest = result.iloc[-1]
            indicator_value = latest.get('indicator_calculator_value', 0.0)
            close_price = data['close'].iloc[-1] if 'close' in data.columns else 0.0
            
            # 初始化信号
            signal = "HOLD"
            score = 50.0
            confidence = 0.5
            
            # 基于移动平均的信号逻辑
            if pd.notna(indicator_value) and indicator_value != 0:
                # 价格相对于移动平均线的位置
                price_ratio = close_price / indicator_value if indicator_value != 0 else 1.0
                
                if price_ratio > 1.01:  # 价格高于均线1%以上
                    signal = "BUY"
                    score = min(75.0, 50.0 + (price_ratio - 1.0) * 1000)
                    confidence = 0.6
                elif price_ratio < 0.99:  # 价格低于均线1%以上
                    signal = "SELL"
                    score = max(25.0, 50.0 - (1.0 - price_ratio) * 1000)
                    confidence = 0.6
                else:
                    signal = "HOLD"
                    score = 50.0
                    confidence = 0.5
            
            return {
                'signal': signal,
                'score': float(score),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.warning(f"IndicatorCalculator信号获取失败: {e}")
            return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}


# 兼容性别名
IIndicatorCalculator = IndicatorCalculator
