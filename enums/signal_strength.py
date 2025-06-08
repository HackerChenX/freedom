from enum import Enum

class SignalStrength(Enum):
    """
    信号强度枚举
    """
    STRONG_BUY = "强势买入"
    BUY = "买入"
    NEUTRAL = "中性"
    SELL = "卖出"
    STRONG_SELL = "强势卖出" 