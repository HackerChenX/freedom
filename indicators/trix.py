"""
TRIX三重指数平滑移动平均线模块

实现TRIX指标计算，用于过滤短期波动，捕捉中长期趋势
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class TRIX(BaseIndicator):
    """
    TRIX三重指数平滑移动平均线指标
    
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) × 100，其中TR = EMA(EMA(EMA(Close, N), N), N)
    过滤短期波动，捕捉中长期趋势变化
    """
    
    def __init__(self, n: int = 12, m: int = 9):
        """
        初始化TRIX指标
        
        Args:
            n: TRIX计算周期，默认为12
            m: 信号线周期，默认为9
        """
        super().__init__(name="TRIX", description="TRIX三重指数平滑移动平均线")
        self.n = n
        self.m = m
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算TRIX指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含TRIX指标的DataFrame
        """
        result = self.calculate(df, self.n, self.m)
        # 重命名列以符合标准
        result['trix'] = result['TRIX']
        result['signal'] = result['MATRIX']
        return result
    
    def sma(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算简单移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: SMA结果
        """
        result = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < n:
                result[i] = np.mean(series[:i+1])
            else:
                result[i] = np.mean(series[i-n+1:i+1])
        
        return result
    
    def calculate(self, data: pd.DataFrame, n: int = 12, m: int = 9, *args, **kwargs) -> pd.DataFrame:
        """
        计算TRIX指标
        
        Args:
            data: 输入数据，包含收盘价
            n: TRIX计算周期，默认为12
            m: MATRIX信号线周期，默认为9
            
        Returns:
            pd.DataFrame: 计算结果，包含TRIX和MATRIX
            
        公式说明：
        TR:=EMA(EMA(EMA(CLOSE,12),12),12);
        TRIX:(TR-REF(TR,1))/REF(TR,1)*100;
        MATRIX:MA(TRIX,9);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 提取数据
        close = data["close"].values
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算TR = EMA(EMA(EMA(Close, N), N), N)
        ema1 = self._ema(close, n)
        ema2 = self._ema(ema1, n)
        tr = self._ema(ema2, n)
        
        # 计算TRIX = (TR - REF(TR, 1)) / REF(TR, 1) × 100
        trix = np.zeros_like(close)
        for i in range(1, len(tr)):
            if tr[i-1] != 0:  # 防止除以零
                trix[i] = (tr[i] - tr[i-1]) / tr[i-1] * 100
        
        # 计算MATRIX = MA(TRIX, M)
        matrix = self.sma(trix, m)
        
        # 添加计算结果到数据框
        result["TR"] = tr
        result["TRIX"] = trix
        result["MATRIX"] = matrix
        
        # 存储结果
        self._result = result
        
        return result
    
    def _ema(self, series: np.ndarray, n: int) -> np.ndarray:
        """
        计算指数移动平均
        
        Args:
            series: 输入序列
            n: 周期
            
        Returns:
            np.ndarray: EMA结果
        """
        alpha = 2 / (n + 1)
        result = np.zeros_like(series)
        result[0] = series[0]
        
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算TRIX原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算TRIX
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. TRIX趋势评分
        trend_score = self._calculate_trix_trend_score()
        score += trend_score
        
        # 2. TRIX与信号线交叉评分
        cross_score = self._calculate_trix_cross_score()
        score += cross_score
        
        # 3. TRIX零轴穿越评分
        zero_cross_score = self._calculate_trix_zero_cross_score()
        score += zero_cross_score
        
        # 4. TRIX背离评分
        divergence_score = self._calculate_trix_divergence_score(data['close'])
        score += divergence_score
        
        # 5. TRIX强度评分
        strength_score = self._calculate_trix_strength_score()
        score += strength_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别TRIX技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算TRIX
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        close_price = data['close']
        
        # 1. 检测TRIX趋势形态
        trend_patterns = self._detect_trix_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 2. 检测TRIX交叉形态
        cross_patterns = self._detect_trix_cross_patterns()
        patterns.extend(cross_patterns)
        
        # 3. 检测TRIX零轴穿越形态
        zero_cross_patterns = self._detect_trix_zero_cross_patterns()
        patterns.extend(zero_cross_patterns)
        
        # 4. 检测TRIX背离形态
        divergence_patterns = self._detect_trix_divergence_patterns(close_price)
        patterns.extend(divergence_patterns)
        
        # 5. 检测TRIX强度形态
        strength_patterns = self._detect_trix_strength_patterns()
        patterns.extend(strength_patterns)
        
        return patterns
    
    def _calculate_trix_trend_score(self) -> pd.Series:
        """
        计算TRIX趋势评分
        
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=self._result.index)
        
        if 'TRIX' not in self._result.columns:
            return trend_score
        
        trix_values = self._result['TRIX']
        
        # TRIX上升趋势+12分
        trix_rising = trix_values > trix_values.shift(1)
        trend_score += trix_rising * 12
        
        # TRIX下降趋势-12分
        trix_falling = trix_values < trix_values.shift(1)
        trend_score -= trix_falling * 12
        
        # TRIX加速上升+18分
        if len(trix_values) >= 3:
            trix_accelerating = (trix_values.diff() > trix_values.shift(1).diff())
            trend_score += trix_accelerating * 18
        
        # TRIX加速下降-18分
        if len(trix_values) >= 3:
            trix_decelerating = (trix_values.diff() < trix_values.shift(1).diff())
            trend_score -= trix_decelerating * 18
        
        return trend_score
    
    def _calculate_trix_cross_score(self) -> pd.Series:
        """
        计算TRIX与信号线交叉评分
        
        Returns:
            pd.Series: 交叉评分
        """
        cross_score = pd.Series(0.0, index=self._result.index)
        
        if 'TRIX' not in self._result.columns or 'MATRIX' not in self._result.columns:
            return cross_score
        
        trix_values = self._result['TRIX']
        matrix_values = self._result['MATRIX']
        
        # TRIX上穿MATRIX（金叉）+25分
        golden_cross = self.crossover(trix_values, matrix_values)
        cross_score += golden_cross * 25
        
        # TRIX下穿MATRIX（死叉）-25分
        death_cross = self.crossunder(trix_values, matrix_values)
        cross_score -= death_cross * 25
        
        return cross_score
    
    def _calculate_trix_zero_cross_score(self) -> pd.Series:
        """
        计算TRIX零轴穿越评分
        
        Returns:
            pd.Series: 零轴穿越评分
        """
        zero_cross_score = pd.Series(0.0, index=self._result.index)
        
        if 'TRIX' not in self._result.columns:
            return zero_cross_score
        
        trix_values = self._result['TRIX']
        zero_line = pd.Series(0.0, index=self._result.index)
        
        # TRIX上穿零轴+20分
        zero_cross_up = self.crossover(trix_values, zero_line)
        zero_cross_score += zero_cross_up * 20
        
        # TRIX下穿零轴-20分
        zero_cross_down = self.crossunder(trix_values, zero_line)
        zero_cross_score -= zero_cross_down * 20
        
        # TRIX在零轴上方+8分
        above_zero = trix_values > 0
        zero_cross_score += above_zero * 8
        
        # TRIX在零轴下方-8分
        below_zero = trix_values < 0
        zero_cross_score -= below_zero * 8
        
        return zero_cross_score
    
    def _calculate_trix_divergence_score(self, close_price: pd.Series) -> pd.Series:
        """
        计算TRIX背离评分
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            pd.Series: 背离评分
        """
        divergence_score = pd.Series(0.0, index=close_price.index)
        
        if 'TRIX' not in self._result.columns or len(close_price) < 20:
            return divergence_score
        
        trix_values = self._result['TRIX']
        
        # 检测背离（简化版本）
        window = 10
        for i in range(window, len(close_price)):
            # 价格创新高但TRIX未创新高（顶背离）
            price_window = close_price.iloc[i-window:i+1]
            trix_window = trix_values.iloc[i-window:i+1]
            
            if (price_window.iloc[-1] == price_window.max() and 
                trix_window.iloc[-1] < trix_window.max()):
                divergence_score.iloc[i] -= 15  # 顶背离-15分
            
            # 价格创新低但TRIX未创新低（底背离）
            if (price_window.iloc[-1] == price_window.min() and 
                trix_window.iloc[-1] > trix_window.min()):
                divergence_score.iloc[i] += 15  # 底背离+15分
        
        return divergence_score
    
    def _calculate_trix_strength_score(self) -> pd.Series:
        """
        计算TRIX强度评分
        
        Returns:
            pd.Series: 强度评分
        """
        strength_score = pd.Series(0.0, index=self._result.index)
        
        if 'TRIX' not in self._result.columns:
            return strength_score
        
        trix_values = self._result['TRIX']
        
        # 计算TRIX的绝对值作为强度指标
        trix_abs = abs(trix_values)
        
        # 强度分级评分
        # 强度很高（>2%）+10分
        very_strong = trix_abs > 2
        strength_score += very_strong * 10
        
        # 强度高（1-2%）+5分
        strong = (trix_abs > 1) & (trix_abs <= 2)
        strength_score += strong * 5
        
        # 强度低（<0.5%）-5分
        weak = trix_abs < 0.5
        strength_score -= weak * 5
        
        return strength_score
    
    def _detect_trix_trend_patterns(self) -> List[str]:
        """
        检测TRIX趋势形态
        
        Returns:
            List[str]: 趋势形态列表
        """
        patterns = []
        
        if 'TRIX' not in self._result.columns or len(self._result) < 5:
            return patterns
        
        trix_values = self._result['TRIX']
        
        # 检查最近的趋势
        recent_periods = min(5, len(trix_values))
        recent_trix = trix_values.tail(recent_periods)
        
        rising_count = 0
        falling_count = 0
        
        for i in range(1, len(recent_trix)):
            if recent_trix.iloc[i] > recent_trix.iloc[i-1]:
                rising_count += 1
            elif recent_trix.iloc[i] < recent_trix.iloc[i-1]:
                falling_count += 1
        
        if rising_count >= 3:
            patterns.append("TRIX上升趋势")
        elif falling_count >= 3:
            patterns.append("TRIX下降趋势")
        else:
            patterns.append("TRIX震荡趋势")
        
        return patterns
    
    def _detect_trix_cross_patterns(self) -> List[str]:
        """
        检测TRIX交叉形态
        
        Returns:
            List[str]: 交叉形态列表
        """
        patterns = []
        
        if ('TRIX' not in self._result.columns or 
            'MATRIX' not in self._result.columns or 
            len(self._result) < 2):
            return patterns
        
        trix_values = self._result['TRIX']
        matrix_values = self._result['MATRIX']
        
        # 检查最近的交叉
        recent_periods = min(5, len(trix_values))
        recent_trix = trix_values.tail(recent_periods)
        recent_matrix = matrix_values.tail(recent_periods)
        
        if self.crossover(recent_trix, recent_matrix).any():
            patterns.append("TRIX金叉")
        
        if self.crossunder(recent_trix, recent_matrix).any():
            patterns.append("TRIX死叉")
        
        # 当前位置关系
        if len(trix_values) > 0 and len(matrix_values) > 0:
            current_trix = trix_values.iloc[-1]
            current_matrix = matrix_values.iloc[-1]
            
            if pd.notna(current_trix) and pd.notna(current_matrix):
                if current_trix > current_matrix:
                    patterns.append("TRIX在信号线上方")
                else:
                    patterns.append("TRIX在信号线下方")
        
        return patterns
    
    def _detect_trix_zero_cross_patterns(self) -> List[str]:
        """
        检测TRIX零轴穿越形态
        
        Returns:
            List[str]: 零轴穿越形态列表
        """
        patterns = []
        
        if 'TRIX' not in self._result.columns or len(self._result) < 2:
            return patterns
        
        trix_values = self._result['TRIX']
        zero_line = pd.Series(0.0, index=self._result.index)
        
        # 检查最近的零轴穿越
        recent_periods = min(5, len(trix_values))
        recent_trix = trix_values.tail(recent_periods)
        recent_zero = zero_line.tail(recent_periods)
        
        if self.crossover(recent_trix, recent_zero).any():
            patterns.append("TRIX上穿零轴")
        
        if self.crossunder(recent_trix, recent_zero).any():
            patterns.append("TRIX下穿零轴")
        
        # 当前位置
        if len(trix_values) > 0:
            current_trix = trix_values.iloc[-1]
            
            if pd.notna(current_trix):
                if current_trix > 0:
                    patterns.append("TRIX在零轴上方")
                else:
                    patterns.append("TRIX在零轴下方")
        
        return patterns
    
    def _detect_trix_divergence_patterns(self, close_price: pd.Series) -> List[str]:
        """
        检测TRIX背离形态
        
        Args:
            close_price: 收盘价序列
            
        Returns:
            List[str]: 背离形态列表
        """
        patterns = []
        
        if ('TRIX' not in self._result.columns or 
            len(close_price) < 20):
            return patterns
        
        trix_values = self._result['TRIX']
        
        # 检测最近的背离
        window = 10
        if len(close_price) >= window:
            recent_price = close_price.tail(window)
            recent_trix = trix_values.tail(window)
            
            # 顶背离：价格创新高但TRIX未创新高
            if (recent_price.iloc[-1] == recent_price.max() and 
                recent_trix.iloc[-1] < recent_trix.max()):
                patterns.append("TRIX顶背离")
            
            # 底背离：价格创新低但TRIX未创新低
            if (recent_price.iloc[-1] == recent_price.min() and 
                recent_trix.iloc[-1] > recent_trix.min()):
                patterns.append("TRIX底背离")
        
        return patterns
    
    def _detect_trix_strength_patterns(self) -> List[str]:
        """
        检测TRIX强度形态
        
        Returns:
            List[str]: 强度形态列表
        """
        patterns = []
        
        if 'TRIX' not in self._result.columns or len(self._result) == 0:
            return patterns
        
        trix_values = self._result['TRIX']
        current_trix = trix_values.iloc[-1]
        
        if pd.isna(current_trix):
            return patterns
        
        trix_abs = abs(current_trix)
        
        if trix_abs > 2:
            patterns.append("TRIX强势信号")
        elif trix_abs > 1:
            patterns.append("TRIX中等强度信号")
        elif trix_abs < 0.5:
            patterns.append("TRIX弱势信号")
        else:
            patterns.append("TRIX正常强度信号")
        
        return patterns

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成TRIX指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算TRIX指标
        if not self.has_result():
            self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算评分
        score = self.calculate_raw_score(data, **kwargs)
        signals['score'] = score
        
        # 检测形态
        patterns = self.identify_patterns(data, **kwargs)
        
        # 获取TRIX数据
        trix = self._result['TRIX']
        matrix = self._result['MATRIX']
        
        # 1. TRIX上穿信号线，买入信号
        trix_crossover_matrix = (trix > matrix) & (trix.shift(1) <= matrix.shift(1))
        signals.loc[trix_crossover_matrix, 'buy_signal'] = True
        signals.loc[trix_crossover_matrix, 'neutral_signal'] = False
        signals.loc[trix_crossover_matrix, 'trend'] = 1
        signals.loc[trix_crossover_matrix, 'signal_type'] = 'TRIX金叉'
        signals.loc[trix_crossover_matrix, 'signal_desc'] = 'TRIX上穿信号线，买入信号'
        signals.loc[trix_crossover_matrix, 'confidence'] = 70.0
        signals.loc[trix_crossover_matrix, 'position_size'] = 0.4
        signals.loc[trix_crossover_matrix, 'risk_level'] = '中'
        
        # 2. TRIX下穿信号线，卖出信号
        trix_crossunder_matrix = (trix < matrix) & (trix.shift(1) >= matrix.shift(1))
        signals.loc[trix_crossunder_matrix, 'sell_signal'] = True
        signals.loc[trix_crossunder_matrix, 'neutral_signal'] = False
        signals.loc[trix_crossunder_matrix, 'trend'] = -1
        signals.loc[trix_crossunder_matrix, 'signal_type'] = 'TRIX死叉'
        signals.loc[trix_crossunder_matrix, 'signal_desc'] = 'TRIX下穿信号线，卖出信号'
        signals.loc[trix_crossunder_matrix, 'confidence'] = 70.0
        signals.loc[trix_crossunder_matrix, 'position_size'] = 0.4
        signals.loc[trix_crossunder_matrix, 'risk_level'] = '中'
        
        # 3. TRIX上穿零轴，买入信号
        trix_crossover_zero = (trix > 0) & (trix.shift(1) <= 0)
        signals.loc[trix_crossover_zero, 'buy_signal'] = True
        signals.loc[trix_crossover_zero, 'neutral_signal'] = False
        signals.loc[trix_crossover_zero, 'trend'] = 1
        signals.loc[trix_crossover_zero, 'signal_type'] = 'TRIX零轴上穿'
        signals.loc[trix_crossover_zero, 'signal_desc'] = 'TRIX上穿零轴，转为上升趋势'
        signals.loc[trix_crossover_zero, 'confidence'] = 75.0
        signals.loc[trix_crossover_zero, 'position_size'] = 0.5
        signals.loc[trix_crossover_zero, 'risk_level'] = '中'
        
        # 4. TRIX下穿零轴，卖出信号
        trix_crossunder_zero = (trix < 0) & (trix.shift(1) >= 0)
        signals.loc[trix_crossunder_zero, 'sell_signal'] = True
        signals.loc[trix_crossunder_zero, 'neutral_signal'] = False
        signals.loc[trix_crossunder_zero, 'trend'] = -1
        signals.loc[trix_crossunder_zero, 'signal_type'] = 'TRIX零轴下穿'
        signals.loc[trix_crossunder_zero, 'signal_desc'] = 'TRIX下穿零轴，转为下降趋势'
        signals.loc[trix_crossunder_zero, 'confidence'] = 75.0
        signals.loc[trix_crossunder_zero, 'position_size'] = 0.5
        signals.loc[trix_crossunder_zero, 'risk_level'] = '中'
        
        # 5. TRIX持续上升且为正值，强势上涨信号
        strong_uptrend = (trix > 0) & (trix > trix.shift(1)) & (trix.shift(1) > trix.shift(2))
        signals.loc[strong_uptrend, 'buy_signal'] = True
        signals.loc[strong_uptrend, 'neutral_signal'] = False
        signals.loc[strong_uptrend, 'trend'] = 1
        signals.loc[strong_uptrend, 'signal_type'] = 'TRIX强势上涨'
        signals.loc[strong_uptrend, 'signal_desc'] = 'TRIX持续上升且为正值，强势上涨信号'
        signals.loc[strong_uptrend, 'confidence'] = 80.0
        signals.loc[strong_uptrend, 'position_size'] = 0.6
        signals.loc[strong_uptrend, 'risk_level'] = '低'
        
        # 6. TRIX持续下降且为负值，强势下跌信号
        strong_downtrend = (trix < 0) & (trix < trix.shift(1)) & (trix.shift(1) < trix.shift(2))
        signals.loc[strong_downtrend, 'sell_signal'] = True
        signals.loc[strong_downtrend, 'neutral_signal'] = False
        signals.loc[strong_downtrend, 'trend'] = -1
        signals.loc[strong_downtrend, 'signal_type'] = 'TRIX强势下跌'
        signals.loc[strong_downtrend, 'signal_desc'] = 'TRIX持续下降且为负值，强势下跌信号'
        signals.loc[strong_downtrend, 'confidence'] = 80.0
        signals.loc[strong_downtrend, 'position_size'] = 0.6
        signals.loc[strong_downtrend, 'risk_level'] = '低'
        
        # 7. TRIX高位死叉，顶部信号
        high_level_death_cross = trix_crossunder_matrix & (trix > 0.5)
        signals.loc[high_level_death_cross, 'sell_signal'] = True
        signals.loc[high_level_death_cross, 'neutral_signal'] = False
        signals.loc[high_level_death_cross, 'trend'] = -1
        signals.loc[high_level_death_cross, 'signal_type'] = 'TRIX高位死叉'
        signals.loc[high_level_death_cross, 'signal_desc'] = 'TRIX在高位下穿信号线，顶部信号'
        signals.loc[high_level_death_cross, 'confidence'] = 85.0
        signals.loc[high_level_death_cross, 'position_size'] = 0.7
        signals.loc[high_level_death_cross, 'risk_level'] = '低'
        
        # 8. TRIX低位金叉，底部信号
        low_level_golden_cross = trix_crossover_matrix & (trix < -0.5)
        signals.loc[low_level_golden_cross, 'buy_signal'] = True
        signals.loc[low_level_golden_cross, 'neutral_signal'] = False
        signals.loc[low_level_golden_cross, 'trend'] = 1
        signals.loc[low_level_golden_cross, 'signal_type'] = 'TRIX低位金叉'
        signals.loc[low_level_golden_cross, 'signal_desc'] = 'TRIX在低位上穿信号线，底部信号'
        signals.loc[low_level_golden_cross, 'confidence'] = 85.0
        signals.loc[low_level_golden_cross, 'position_size'] = 0.7
        signals.loc[low_level_golden_cross, 'risk_level'] = '低'
        
        # 9. TRIX区间震荡，无明显趋势
        sideways = (abs(trix) < 0.1) & (abs(trix - trix.shift(1)) < 0.05)
        signals.loc[sideways, 'neutral_signal'] = True
        signals.loc[sideways, 'buy_signal'] = False
        signals.loc[sideways, 'sell_signal'] = False
        signals.loc[sideways, 'trend'] = 0
        signals.loc[sideways, 'signal_type'] = 'TRIX区间震荡'
        signals.loc[sideways, 'signal_desc'] = 'TRIX在零轴附近小幅波动，无明显趋势'
        signals.loc[sideways, 'confidence'] = 60.0
        signals.loc[sideways, 'position_size'] = 0.0
        signals.loc[sideways, 'risk_level'] = '低'
        
        # 10. 根据形态设置更多信号
        for pattern in patterns:
            pattern_idx = signals.index[-5:]  # 假设形态影响最近5个周期
            
            if '上升' in pattern or '看涨' in pattern or '金叉' in pattern:
                signals.loc[pattern_idx, 'buy_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 1
                signals.loc[pattern_idx, 'signal_type'] = 'TRIX形态信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 70.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '下降' in pattern or '看跌' in pattern or '死叉' in pattern:
                signals.loc[pattern_idx, 'sell_signal'] = True
                signals.loc[pattern_idx, 'neutral_signal'] = False
                signals.loc[pattern_idx, 'trend'] = -1
                signals.loc[pattern_idx, 'signal_type'] = 'TRIX形态信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 70.0
                signals.loc[pattern_idx, 'position_size'] = 0.4
                signals.loc[pattern_idx, 'risk_level'] = '中'
            
            elif '震荡' in pattern or '平稳' in pattern:
                signals.loc[pattern_idx, 'neutral_signal'] = True
                signals.loc[pattern_idx, 'buy_signal'] = False
                signals.loc[pattern_idx, 'sell_signal'] = False
                signals.loc[pattern_idx, 'trend'] = 0
                signals.loc[pattern_idx, 'signal_type'] = 'TRIX震荡信号'
                signals.loc[pattern_idx, 'signal_desc'] = pattern
                signals.loc[pattern_idx, 'confidence'] = 60.0
                signals.loc[pattern_idx, 'position_size'] = 0.0
                signals.loc[pattern_idx, 'risk_level'] = '低'
        
        # 设置止损价格
        if 'low' in data.columns and 'high' in data.columns:
            # 买入信号的止损设为最近的低点
            buy_indices = signals[signals['buy_signal']].index
            if not buy_indices.empty:
                for idx in buy_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近低点作为止损
                        recent_low = data.loc[idx-lookback:idx, 'low'].min()
                        signals.loc[idx, 'stop_loss'] = recent_low
        
            # 卖出信号的止损设为最近的高点
            sell_indices = signals[signals['sell_signal']].index
            if not sell_indices.empty:
                for idx in sell_indices:
                    if idx > data.index[10]:  # 确保有足够的历史数据
                        lookback = 5
                        # 使用最近高点作为止损
                        recent_high = data.loc[idx-lookback:idx, 'high'].max()
                        signals.loc[idx, 'stop_loss'] = recent_high
        
        # 根据TRIX值判断市场环境
        signals['market_env'] = 'sideways_market'  # 默认震荡市场
        
        # TRIX > 0 且上升，牛市环境
        bull_market = (trix > 0) & (trix > trix.shift(1))
        signals.loc[bull_market, 'market_env'] = 'bull_market'
        
        # TRIX < 0 且下降，熊市环境
        bear_market = (trix < 0) & (trix < trix.shift(1))
        signals.loc[bear_market, 'market_env'] = 'bear_market'
        
        # TRIX接近0且波动小，震荡市场
        sideways_market = (abs(trix) < 0.1) & (abs(trix - trix.shift(1)) < 0.05)
        signals.loc[sideways_market, 'market_env'] = 'sideways_market'
        
        # 设置成交量确认
        if 'volume' in data.columns:
            # 如果有成交量数据，检查成交量是否支持当前信号
            vol = data['volume']
            vol_avg = vol.rolling(window=20).mean()
            
            # 成交量大于20日均量1.5倍为放量
            vol_increase = vol > vol_avg * 1.5
            
            # 买入信号且成交量放大，确认信号
            signals.loc[signals['buy_signal'] & vol_increase, 'volume_confirmation'] = True
            
            # 卖出信号且成交量放大，确认信号
            signals.loc[signals['sell_signal'] & vol_increase, 'volume_confirmation'] = True
        
        return signals 