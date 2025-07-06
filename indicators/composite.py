import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class Composite(BaseIndicator, PatternSignalMixin):
    """
    COMPOSITE 指标
    
    自动生成的标准化实现
    """
    
    def __init__(self, **kwargs):
        """
        初始化COMPOSITE指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "COMPOSITE"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_composite()
        
        # 应用用户参数
        self.set_parameters_Composite(**kwargs)
    
    def _get_default_parameters_composite(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Composite(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import Indicator_parameter_validator
            validator = Indicator_parameter_validator()
            
            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)
            
            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('COMPOSITE', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 14)
    
    def calculate_Composite(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算COMPOSITE指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了COMPOSITE指标的Data_frame
        """
        result = self._calculate_composite(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_composite(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算COMPOSITE指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了COMPOSITE指标的Data_frame
        """
        df = data.copy()
        
        # 基本实现：返回原数据加上一个简单的计算列
        df[f'COMPOSITE_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def calculate_raw_score_Composite(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Composite(data, **kwargs)
        
        # 复合指标评分：结合多个技术指标
        df = data.copy()
        
        # 1. 移动平均线信号
        ma5 = df['close'].rolling(window=5).mean()
        ma10 = df['close'].rolling(window=10).mean()
        ma20 = df['close'].rolling(window=20).mean()
        
        # 2. RSI信号
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 3. MACD信号
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        
        # 4. 成交量信号
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # 复合评分计算
        scores = pd.Series(50.0, index=data.index)  # 基准分
        
        # MA信号 (权重: 25%)
        ma_bullish = (df['close'] > ma5) & (ma5 > ma10) & (ma10 > ma20)
        ma_bearish = (df['close'] < ma5) & (ma5 < ma10) & (ma10 < ma20)
        scores += np.where(ma_bullish, 15, 0)
        scores += np.where(ma_bearish, -15, 0)
        
        # RSI信号 (权重: 25%)
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_neutral = (rsi >= 30) & (rsi <= 70)
        scores += np.where(rsi_oversold, 15, 0)  # 超卖买入机会
        scores += np.where(rsi_overbought, -10, 0)  # 超买减分
        scores += np.where(rsi_neutral & (rsi > 50), 5, 0)  # 中性偏强
        
        # MACD信号 (权重: 25%)
        macd_golden_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        macd_death_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        macd_above_zero = macd > 0
        scores += np.where(macd_golden_cross, 20, 0)  # 金叉强烈买入
        scores += np.where(macd_death_cross, -15, 0)  # 死叉减分
        scores += np.where(macd_above_zero, 5, -5)  # 零轴位置
        
        # 成交量信号 (权重: 25%)
        volume_surge = volume_ratio > 2.0  # 放量
        volume_shrink = volume_ratio < 0.5  # 缩量
        scores += np.where(volume_surge & (df['close'] > df['close'].shift(1)), 10, 0)  # 放量上涨
        scores += np.where(volume_shrink & (df['close'] < df['close'].shift(1)), -5, 0)  # 缩量下跌
        
        # 趋势强度加成
        price_trend = (df['close'] / df['close'].shift(5) - 1) * 100  # 5日涨幅
        strong_uptrend = price_trend > 5
        strong_downtrend = price_trend < -5
        scores += np.where(strong_uptrend, 10, 0)
        scores += np.where(strong_downtrend, -10, 0)
        
        # 限制评分范围
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Composite(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Composite(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)
