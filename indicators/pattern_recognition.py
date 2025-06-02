"""
形态识别模块

提供技术形态识别和处理的功能
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = 0.2    # 弱信号
    MEDIUM = 0.5  # 中等信号
    STRONG = 0.8  # 强信号
    EXTREME = 1.0 # 极强信号

class PatternResult:
    """
    形态识别结果类
    
    记录识别到的形态信息和强度
    """
    
    def __init__(self, 
                pattern_id: str, 
                description: str = "", 
                strength: float = 0.5, 
                detail: Dict[str, Any] = None):
        """
        初始化形态结果
        
        Args:
            pattern_id: 形态ID
            description: 形态描述
            strength: 形态强度 (0.0-1.0)
            detail: 详细信息
        """
        self.pattern_id = pattern_id
        self.description = description
        self.strength = min(1.0, max(0.0, strength))  # 确保强度在0-1之间
        self.detail = detail or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict: 字典表示
        """
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "strength": self.strength,
            "detail": self.detail
        }

def detect_golden_cross(values: np.ndarray, signal: np.ndarray, 
                       lookback: int = 5) -> Optional[PatternResult]:
    """
    检测金叉形态
    
    Args:
        values: 主要值数组
        signal: 信号值数组
        lookback: 回溯周期
        
    Returns:
        PatternResult: 金叉形态结果，如果无金叉则返回None
    """
    if len(values) < lookback + 1 or len(signal) < lookback + 1:
        return None
    
    # 检查最近是否发生金叉
    for i in range(1, min(lookback, len(values) - 1)):
        # 金叉条件：前一个周期values < signal，当前周期values > signal
        if values[-i-1] <= signal[-i-1] and values[-i] > signal[-i]:
            # 计算金叉强度
            strength = min(1.0, abs(values[-i] - signal[-i]) / abs(signal[-i]) * 5)
            
            # 附加详细信息
            detail = {
                "cross_index": -i,
                "value_before": float(values[-i-1]),
                "signal_before": float(signal[-i-1]),
                "value_after": float(values[-i]),
                "signal_after": float(signal[-i]),
                "distance": float(abs(values[-i] - signal[-i]))
            }
            
            return PatternResult(
                pattern_id="golden_cross",
                description="金叉形态",
                strength=strength,
                detail=detail
            )
    
    return None

def detect_death_cross(values: np.ndarray, signal: np.ndarray, 
                      lookback: int = 5) -> Optional[PatternResult]:
    """
    检测死叉形态
    
    Args:
        values: 主要值数组
        signal: 信号值数组
        lookback: 回溯周期
        
    Returns:
        PatternResult: 死叉形态结果，如果无死叉则返回None
    """
    if len(values) < lookback + 1 or len(signal) < lookback + 1:
        return None
    
    # 检查最近是否发生死叉
    for i in range(1, min(lookback, len(values) - 1)):
        # 死叉条件：前一个周期values > signal，当前周期values < signal
        if values[-i-1] >= signal[-i-1] and values[-i] < signal[-i]:
            # 计算死叉强度
            strength = min(1.0, abs(values[-i] - signal[-i]) / abs(signal[-i]) * 5)
            
            # 附加详细信息
            detail = {
                "cross_index": -i,
                "value_before": float(values[-i-1]),
                "signal_before": float(signal[-i-1]),
                "value_after": float(values[-i]),
                "signal_after": float(signal[-i]),
                "distance": float(abs(values[-i] - signal[-i]))
            }
            
            return PatternResult(
                pattern_id="death_cross",
                description="死叉形态",
                strength=strength,
                detail=detail
            )
    
    return None

def detect_trend_change(values: np.ndarray, window: int = 10) -> Optional[PatternResult]:
    """
    检测趋势改变
    
    Args:
        values: 值数组
        window: 窗口大小
        
    Returns:
        PatternResult: 趋势改变结果，如果无趋势改变则返回None
    """
    if len(values) < window * 2:
        return None
    
    # 计算前窗口和当前窗口的趋势
    prev_window = values[-2*window:-window]
    curr_window = values[-window:]
    
    # 使用线性回归斜率估计趋势
    try:
        prev_x = np.arange(len(prev_window))
        prev_slope = np.polyfit(prev_x, prev_window, 1)[0]
        
        curr_x = np.arange(len(curr_window))
        curr_slope = np.polyfit(curr_x, curr_window, 1)[0]
        
        # 判断趋势改变
        if prev_slope * curr_slope < 0:  # 斜率符号变化意味着趋势改变
            # 计算趋势改变强度
            strength = min(1.0, (abs(prev_slope) + abs(curr_slope)) / 2)
            
            pattern_id = "trend_reversal_up" if curr_slope > 0 else "trend_reversal_down"
            description = "趋势反转向上" if curr_slope > 0 else "趋势反转向下"
            
            # 附加详细信息
            detail = {
                "prev_slope": float(prev_slope),
                "curr_slope": float(curr_slope),
                "prev_mean": float(np.mean(prev_window)),
                "curr_mean": float(np.mean(curr_window))
            }
            
            return PatternResult(
                pattern_id=pattern_id,
                description=description,
                strength=strength,
                detail=detail
            )
    except:
        # 如果拟合失败，返回None
        pass
    
    return None

def detect_overbought_oversold(values: np.ndarray, 
                             overbought_threshold: float = 80.0,
                             oversold_threshold: float = 20.0) -> Optional[PatternResult]:
    """
    检测超买超卖
    
    Args:
        values: 值数组（如RSI、KDJ的K值等）
        overbought_threshold: 超买阈值
        oversold_threshold: 超卖阈值
        
    Returns:
        PatternResult: 超买超卖结果，如果无超买超卖则返回None
    """
    if len(values) < 2:
        return None
    
    # 获取最新值
    latest_value = values[-1]
    prev_value = values[-2]
    
    # 检测超买
    if latest_value >= overbought_threshold:
        # 计算超买强度
        strength = min(1.0, (latest_value - overbought_threshold) / (100 - overbought_threshold))
        
        # 检测是否从超买区域回落
        if latest_value < prev_value and prev_value >= overbought_threshold:
            return PatternResult(
                pattern_id="overbought_reversal",
                description="超买区域回落",
                strength=strength,
                detail={
                    "value": float(latest_value),
                    "prev_value": float(prev_value),
                    "threshold": float(overbought_threshold)
                }
            )
        else:
            return PatternResult(
                pattern_id="overbought",
                description="超买区域",
                strength=strength,
                detail={
                    "value": float(latest_value),
                    "threshold": float(overbought_threshold)
                }
            )
    
    # 检测超卖
    if latest_value <= oversold_threshold:
        # 计算超卖强度
        strength = min(1.0, (oversold_threshold - latest_value) / oversold_threshold)
        
        # 检测是否从超卖区域反弹
        if latest_value > prev_value and prev_value <= oversold_threshold:
            return PatternResult(
                pattern_id="oversold_rebound",
                description="超卖区域反弹",
                strength=strength,
                detail={
                    "value": float(latest_value),
                    "prev_value": float(prev_value),
                    "threshold": float(oversold_threshold)
                }
            )
        else:
            return PatternResult(
                pattern_id="oversold",
                description="超卖区域",
                strength=strength,
                detail={
                    "value": float(latest_value),
                    "threshold": float(oversold_threshold)
                }
            )
    
    return None

def detect_divergence(price: np.ndarray, indicator: np.ndarray, 
                     window: int = 20) -> Optional[PatternResult]:
    """
    检测背离
    
    Args:
        price: 价格数组
        indicator: 指标数组
        window: 检测窗口
        
    Returns:
        PatternResult: 背离结果，如果无背离则返回None
    """
    if len(price) < window or len(indicator) < window:
        return None
    
    try:
        # 获取窗口内的价格和指标
        price_window = price[-window:]
        indicator_window = indicator[-window:]
        
        # 获取价格的高点和低点
        price_max_idx = np.argmax(price_window)
        price_min_idx = np.argmin(price_window)
        
        # 获取指标的高点和低点
        indicator_max_idx = np.argmax(indicator_window)
        indicator_min_idx = np.argmin(indicator_window)
        
        # 检测顶背离：价格创新高，但指标未创新高
        if price_max_idx > indicator_max_idx and price_max_idx >= len(price_window) - 5:
            strength = min(1.0, abs(price_window[price_max_idx] - price_window[indicator_max_idx]) / price_window[price_max_idx])
            
            return PatternResult(
                pattern_id="bearish_divergence",
                description="顶背离（看跌）",
                strength=strength,
                detail={
                    "price_max_idx": int(price_max_idx),
                    "indicator_max_idx": int(indicator_max_idx),
                    "price_max": float(price_window[price_max_idx]),
                    "indicator_at_price_max": float(indicator_window[price_max_idx]),
                    "indicator_max": float(indicator_window[indicator_max_idx])
                }
            )
        
        # 检测底背离：价格创新低，但指标未创新低
        if price_min_idx > indicator_min_idx and price_min_idx >= len(price_window) - 5:
            strength = min(1.0, abs(price_window[price_min_idx] - price_window[indicator_min_idx]) / price_window[price_min_idx])
            
            return PatternResult(
                pattern_id="bullish_divergence",
                description="底背离（看涨）",
                strength=strength,
                detail={
                    "price_min_idx": int(price_min_idx),
                    "indicator_min_idx": int(indicator_min_idx),
                    "price_min": float(price_window[price_min_idx]),
                    "indicator_at_price_min": float(indicator_window[price_min_idx]),
                    "indicator_min": float(indicator_window[indicator_min_idx])
                }
            )
    except:
        # 如果计算过程出错，返回None
        pass
    
    return None 