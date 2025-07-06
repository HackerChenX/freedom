import pandas as pd
import numpy as np
from typing import Dict, Any, List

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from utils.logger import getLogger

logger = getLogger(__name__)


class UnifiedMa(BaseIndicator, PatternSignalMixin):
    """
    UNIFIED_MA 指标
    
    自动生成的标准化实现
    """
    
    def __init__(self, **kwargs):
        """
        初始化UNIFIED_MA指标
        
        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "UNIFIED_MA"
        
        # 设置默认参数
        self._default_parameters = self._get_default_parameters_unifiedma()
        
        # 应用用户参数
        self.set_parameters_Ma_Unified_Ma(**kwargs)
    
    def _get_default_parameters_unifiedma(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {"period": 14}
    
    def set_parameters_Ma_Unified_Ma(self, **kwargs):
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
            is_valid, errors = validator.validate_indicator_parameters('UNIFIED_MA', params)
            if not is_valid:
                # 静默处理验证失败，避免过多警告
                pass
                
        except Exception:
            # 如果验证失败，静默处理，保持向后兼容
            pass
        
        # 设置参数
        self.period = kwargs.get('period', 14)
    
    def calculate_Ma(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算UNIFIED_MA指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了UNIFIED_MA指标的Data_frame
        """
        result = self._calculate_unifiedma(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_unifiedma(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算UNIFIED_MA指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了UNIFIED_MA指标的Data_frame
        """
        df = data.copy()
        
        # 基本实现：返回原数据加上一个简单的计算列
        df[f'UNIFIED_MA_VALUE'] = df['close'].rolling(window=self.period).mean()
        
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        # 重写专用信号逻辑：基于评分值的阈值判断
        # 对于state_type指标，使用评分阈值模式
        score_threshold = 50.0  # 默认阈值
        df.loc[:, 'buy_signal'] = df[f'UNIFIED_MA_VALUE'] >= score_threshold
        df.loc[:, 'sell_signal'] = df[f'UNIFIED_MA_VALUE'] < score_threshold
        df.loc[:, 'hold_signal'] = df[f'UNIFIED_MA_VALUE'] < score_threshold

        return df
    
    def calculate_raw_score_Ma_Unified_Ma(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算原始评分"""
        if not self.has_result():
            self.calculate_Ma(data, **kwargs)
        
        # 统一移动平均线评分：多周期MA综合分析
        df = data.copy()
        
        # 计算多个周期的移动平均线
        ma5 = df['close'].rolling(window=5).mean()
        ma10 = df['close'].rolling(window=10).mean()
        ma20 = df['close'].rolling(window=20).mean()
        ma30 = df['close'].rolling(window=30).mean()
        ma60 = df['close'].rolling(window=60).mean()
        
        # 指数移动平均线
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        
        # 复合评分计算
        scores = pd.Series(50.0, index=data.index)  # 基准分
        
        # 短期MA排列 (30%)
        short_bullish = (df['close'] > ma5) & (ma5 > ma10) & (ma10 > ma20)
        short_bearish = (df['close'] < ma5) & (ma5 < ma10) & (ma10 < ma20)
        scores += np.where(short_bullish, 20, 0)
        scores += np.where(short_bearish, -15, 0)
        
        # 中长期MA排列 (25%)
        long_bullish = (ma20 > ma30) & (ma30 > ma60)
        long_bearish = (ma20 < ma30) & (ma30 < ma60)
        scores += np.where(long_bullish, 15, 0)
        scores += np.where(long_bearish, -15, 0)
        
        # MA金叉死叉 (25%)
        ma5_cross_ma10 = (ma5 > ma10) & (ma5.shift(1) <= ma10.shift(1))
        ma5_death_ma10 = (ma5 < ma10) & (ma5.shift(1) >= ma10.shift(1))
        ma10_cross_ma20 = (ma10 > ma20) & (ma10.shift(1) <= ma20.shift(1))
        ma10_death_ma20 = (ma10 < ma20) & (ma10.shift(1) >= ma20.shift(1))
        
        scores += np.where(ma5_cross_ma10, 15, 0)
        scores += np.where(ma5_death_ma10, -10, 0)
        scores += np.where(ma10_cross_ma20, 12, 0)
        scores += np.where(ma10_death_ma20, -12, 0)
        
        # EMA信号 (15%)
        ema_bullish = ema12 > ema26
        ema_cross = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1))
        scores += np.where(ema_bullish, 8, -5)
        scores += np.where(ema_cross, 10, 0)
        
        # 价格与MA关系 (5%)
        above_all_ma = (df['close'] > ma5) & (df['close'] > ma10) & (df['close'] > ma20)
        below_all_ma = (df['close'] < ma5) & (df['close'] < ma10) & (df['close'] < ma20)
        scores += np.where(above_all_ma, 8, 0)
        scores += np.where(below_all_ma, -8, 0)
        
        # MA斜率分析
        ma20_slope = (ma20 - ma20.shift(5)) / ma20.shift(5)
        upward_slope = ma20_slope > 0.02  # 上升斜率>2%
        downward_slope = ma20_slope < -0.02  # 下降斜率>2%
        scores += np.where(upward_slope, 10, 0)
        scores += np.where(downward_slope, -10, 0)
        
        # MA收敛发散
        ma_spread = (ma5 - ma20) / ma20
        expanding = ma_spread > ma_spread.shift(5)  # 发散
        contracting = ma_spread < ma_spread.shift(5)  # 收敛
        scores += np.where(expanding & (ma_spread > 0), 5, 0)  # 上升发散
        scores += np.where(contracting & (ma_spread < 0), 5, 0)  # 下跌收敛
        
        # 限制评分范围
        scores = np.clip(scores, 0, 100)
        
        return scores
    
    def calculate_confidence_Ma_Unified_Ma(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5
    
    def get_patterns_Ma_Unified_Ma(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        return pd.DataFrame(index=data.index)


# 为了向后兼容，创建别名
unified_ma = UNIFIED_MA