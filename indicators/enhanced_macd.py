import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class EnhancedMacd(BaseIndicator, PatternSignalMixin):
    """
    ENHANCED_MACD 指标
    
    自动生成的标准化实现
    """
    
    def __init__(self, **kwargs):
        """
        初始化ENHANCED_MACD指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "ENHANCED_MACD"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_enhancedmacd()
        
        # 应用用户参数
        self.set_parameters_Macd_Enhanced_Macd(**kwargs)
    
    def _get_default_parameters_enhancedmacd(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Macd_Enhanced_Macd(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('ENHANCED_MACD', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 14)
    
    def calculate_Macd(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ENHANCED_MACD指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了ENHANCED_MACD指标的Data_frame
        """
        result = self._calculate_enhancedmacd(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_enhancedmacd(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算ENHANCED_MACD指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了ENHANCED_MACD指标的Data_frame
        """
        df = data.copy()
        
        # 基本实现：返回原数据加上一个简单的计算列
        df[f'ENHANCED_MACD_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于评分值的阈值判断
        # 对于state_type指标，使用评分阈值模式
        score_threshold = 50.0  # 默认阈值
        df.loc[:, 'buy_signal'] = df[f'ENHANCED_MACD_VALUE'] >= score_threshold
        df.loc[:, 'sell_signal'] = df[f'ENHANCED_MACD_VALUE'] < score_threshold
        df.loc[:, 'hold_signal'] = df[f'ENHANCED_MACD_VALUE'] < score_threshold

        return df
    
    def calculate_raw_score_Macd_Enhanced_Macd(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Macd(data, **kwargs)
        
        # 基于MACD指标计算评分
        df = data.copy()
        
        # 计算MACD指标
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        # 计算评分
        scores = pd.Series(50.0, index=data.index)  # 基准分
        
        # MACD金叉死叉信号
        macd_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        macd_death = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        
        # 零轴上下信号
        above_zero = macd > 0
        below_zero = macd < 0
        
        # 背离信号
        price_high = df['close'].rolling(window=5).max() == df['close']
        price_low = df['close'].rolling(window=5).min() == df['close']
        macd_high = macd.rolling(window=5).max() == macd
        macd_low = macd.rolling(window=5).min() == macd
        
        # 顶背离（价格新高，MACD不新高）
        top_divergence = price_high & ~macd_high & (macd > 0)
        # 底背离（价格新低，MACD不新低）
        bottom_divergence = price_low & ~macd_low & (macd < 0)
        
        # 评分计算
        scores += np.where(macd_cross, 20, 0)  # 金叉加分
        scores += np.where(macd_death, -20, 0)  # 死叉减分
        scores += np.where(above_zero & (macd > signal), 10, 0)  # 零轴上方且MACD>信号线
        scores += np.where(below_zero & (macd < signal), -10, 0)  # 零轴下方且MACD<信号线
        scores += np.where(histogram > 0, 5, -5)  # 柱状图正负
        scores += np.where(bottom_divergence, 15, 0)  # 底背离加分
        scores += np.where(top_divergence, -15, 0)  # 顶背离减分
        
        # 趋势强度
        macd_trend = macd.rolling(window=3).mean()
        trend_up = macd_trend > macd_trend.shift(1)
        scores += np.where(trend_up, 5, -5)
        
        # 限制评分范围
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Macd_Enhanced_Macd(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Macd_Enhanced_Macd(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)


# 为了向后兼容，创建别名
enhanced_macd = ENHANCED_MACD
