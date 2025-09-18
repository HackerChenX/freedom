#!/usr/bin/env python3
"""
OBV (On-Balance Volume) 能量潮指标

OBV指标通过累计成交量来反映资金流向.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from utils.container import container

logger = get_logger(__name__)


class OnBalanceVolume(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    OBV (On-Balance Volume) 能量潮指标
    
    OBV指标通过累计成交量变化来判断资金流向.
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, **kwargs):
        """
        初始化OBV指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "OBV"
        
        # 初始化结果存储
        self._result = None
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_obv()
        
        # 应用用户参数
        self.set_parameters_Obv(**kwargs)
    
    def _get_default_parameters_obv(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"signal_period": 10}
    
    def set_parameters_Obv(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('OBV', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                
        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            pass
        
        # 设置参数
        self.signal_period = kwargs.get('signal_period', 10)
    
    def calculate_Obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        result = self._calculate_obv(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算OBV指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了OBV指标的DataFrame
        """
        df = data.copy()
        
        # 确保数据有足够的长度
        if len(df) < 2:
            logger.warning(f"数据长度({len(df)})不足,返回原始数据")
            df['OBV'] = np.nan
            df['obv_ma'] = np.nan
            df['obv_signal'] = np.nan
            return df

        # 计算价格变化
        df['price_change'] = df['close'].diff()

        # 计算OBV (On-Balance Volume)
        obv = [0]  # 初始值为0
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0:
                # 价格上涨,加上成交量
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['price_change'].iloc[i] < 0:
                # 价格下跌,减去成交量
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                # 价格不变,OBV保持不变
                obv.append(obv[-1])
        
        df['OBV'] = obv
        df['obv'] = obv  # 为了一致性
        
        # 计算OBV移动平均线(信号线)
        df['obv_ma'] = df['OBV'].rolling(window=self.signal_period).mean()
        df[f'OBV_MA{self.signal_period}'] = df['obv_ma']  # 为了向后兼容
        
        # 计算OBV信号线(更短周期的移动平均)
        df['obv_signal'] = df['OBV'].rolling(window=5).mean()
        
        # 计算OBV变化率
        df['obv_change'] = df['OBV'].pct_change() * 100
        
        # 计算OBV波动率
        df['obv_volatility'] = df['obv_change'].rolling(window=10).std()
        
        # 计算OBV相对强度(与历史均值的关系)
        df['obv_strength'] = (df['OBV'] - df['OBV'].rolling(window=20).mean()) / (df['OBV'].rolling(window=20).std() + 1e-8)
        
        # 清理中间计算列
        df.drop(['price_change'], axis=1, inplace=True)
            
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写信号生成逻辑(OBV指标特定逻辑)
        df = self._apply_obv_signal_logic(df)

        return df

    def _apply_obv_signal_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用OBV指标特定的信号生成逻辑
        基于OBV值的变化和量价关系生成信号
        """
        try:
            # 获取OBV值
            if 'OBV' not in df.columns:
                # 如果没有OBV值,使用默认信号
                return df

            obv_value = df['OBV']
            obv_ma = df['obv_ma']
            obv_signal = df['obv_signal']
            close_price = df['close']

            # 计算OBV和价格的变化
            obv_rising = obv_value > obv_value.shift(1)
            obv_falling = obv_value < obv_value.shift(1)
            price_rising = close_price > close_price.shift(1)
            price_falling = close_price < close_price.shift(1)
            
            # OBV与移动平均线的关系
            obv_above_ma = obv_value > obv_ma
            obv_below_ma = obv_value < obv_ma
            
            # OBV突破移动平均线
            obv_breakout_up = obv_above_ma & (obv_value.shift(1) <= obv_ma.shift(1))
            obv_breakdown = obv_below_ma & (obv_value.shift(1) >= obv_ma.shift(1))
            
            # 强势OBV信号
            strong_obv_up = obv_rising & (obv_value > obv_signal)
            strong_obv_down = obv_falling & (obv_value < obv_signal)

            # 生成信号
            df.loc[:, 'buy_signal'] = (obv_rising & price_rising) | obv_breakout_up | strong_obv_up
            df.loc[:, 'sell_signal'] = (obv_falling & price_falling) | obv_breakdown | strong_obv_down
            df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])

            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)

        except Exception as e:
            logger.warning(f"OBV信号生成失败: {e}")
            # 如果出错,使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True

        return df
    
    def calculate_raw_score_Obv(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算OBV原始评分
        """
        if not self.has_result():
            self.calculate_Obv(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 获取OBV数据
        obv = self._result['OBV']
        obv_ma = self._result['obv_ma']
        obv_strength = self._result['obv_strength']
        
        # 初始化评分
        scores = pd.Series(50.0, index=data.index)
        
        # 简化的评分逻辑
        obv_change = obv - obv.shift(1)
        
        # OBV上升得分
        scores += np.where(obv_change > 0, 10, 0)
        # OBV下降扣分
        scores += np.where(obv_change < 0, -10, 0)
        
        # OBV与移动平均线的关系
        if len(obv_ma.dropna()) > 0:
            scores += np.where(obv > obv_ma, 5, -5)
        
        # 确保评分在合理范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def get_patterns_Obv(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取OBV相关形态"""
        if not self.has_result():
            self.calculate_Obv(data, **kwargs)
            
        if self._result is None:
            return pd.DataFrame(index=data.index)
            
        patterns = pd.DataFrame(index=data.index)
        
        obv = self._result['OBV']
        obv_ma = self._result['obv_ma']
        
        # 基本形态
        obv_change = obv - obv.shift(1)
        patterns['OBV_RISING'] = obv_change > 0
        patterns['OBV_FALLING'] = obv_change < 0
        patterns['OBV_STABLE'] = abs(obv_change) < (obv.std() * 0.1)
        
        # 与移动平均线的关系
        if len(obv_ma.dropna()) > 0:
            patterns['OBV_ABOVE_MA'] = obv > obv_ma
            patterns['OBV_BELOW_MA'] = obv < obv_ma
            patterns['OBV_BREAKOUT_UP'] = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
            patterns['OBV_BREAKDOWN'] = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))
        
        return patterns

    # ================== 抽象方法实现 ==================
    
    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象方法实现:调用OBV计算逻辑"""
        return self._calculate_obv(data, **kwargs)
    
    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """抽象方法实现:计算OBV原始评分"""
        return self.calculate_raw_score_Obv(data, **kwargs)
    
    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """抽象方法实现:获取OBV形态"""
        return self.get_patterns_Obv(data, **kwargs)
    
    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """抽象方法实现:设置参数"""
        return self.set_parameters_Obv(**kwargs)
    
    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """抽象方法实现:计算置信度"""
        return self.calculate_confidence_Obv(score, patterns, signals)
    
    # ================== 兼容性方法 ==================
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:计算指标"""
        return self.calculate_Obv(data, **kwargs)
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """兼容性方法:获取形态"""
        return self.get_patterns_Obv(data, **kwargs)
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """兼容性方法:计算原始评分"""
        return self.calculate_raw_score_Obv(data, **kwargs)
    
    def calculate_confidence_Obv(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if self._result is None:
            return 0.5
            
        # 基于OBV指标的明确性计算置信度
        try:
            obv = self._result['OBV'].dropna()
            if len(obv) == 0:
                return 0.5
                
            # 基础置信度
            confidence = 0.5
            
            # 根据OBV强度调整置信度
            if 'obv_strength' in self._result.columns:
                strength = abs(self._result['obv_strength'].iloc[-1])
                if pd.notna(strength):
                    confidence += min(strength * 0.1, 0.3)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"OBV置信度计算失败: {e}")
            return 0.0

    def has_result(self) -> bool:
        """检查是否有计算结果"""
        return self._result is not None

    @property
    def minimum_periods(self) -> int:
        """
        返回OBV指标计算所需的最少数据周期数

        Returns:
            int: 最少需要的数据周期数
        """
        return 2  # OBV只需要2个数据点就可以开始计算


# 类别名,供指标注册系统使用
OnBalanceVolumeOBV = OnBalanceVolume
Obv = OnBalanceVolume
OBV = OnBalanceVolume  # 添加OBV别名
