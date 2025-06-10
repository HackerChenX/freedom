from enum import Enum

class SignalStrength(Enum):
    """
    统一信号强度枚举

    数值越大，信号越强。正数代表看涨/积极，负数代表看跌/消极。
    """
    VERY_STRONG = 2.0
    STRONG = 1.5
    MODERATE = 1.0
    WEAK = 0.5
    VERY_WEAK = 0.25
    
    NEUTRAL = 0.0
    
    VERY_STRONG_NEGATIVE = -2.0
    STRONG_NEGATIVE = -1.5
    MODERATE_NEGATIVE = -1.0
    WEAK_NEGATIVE = -0.5
    VERY_WEAK_NEGATIVE = -0.25

    # 为了兼容旧的基于字符串的定义
    STRONG_BUY = "强势买入"
    BUY = "买入"
    SELL = "卖出"
    STRONG_SELL = "强势卖出" 