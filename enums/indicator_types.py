"""
技术指标类型枚举模块

使用枚举定义所有支持的技术指标类型
"""

from enum import Enum, auto


class IndicatorType(Enum):
    """技术指标类型枚举"""
    
    # 趋势指标
    MA = auto()         # 移动平均线
    EMA = auto()        # 指数移动平均线
    MACD = auto()       # 移动平均线收敛散度
    BOLL = auto()       # 布林带
    
    # 动量指标
    RSI = auto()        # 相对强弱指数
    KDJ = auto()        # 随机指标
    STOCH = auto()      # 随机指标(另一种)
    
    # 波动性指标
    ATR = auto()        # 平均真实波幅
    
    # 成交量指标
    OBV = auto()        # 能量潮
    VOL = auto()        # 成交量
    
    # 自定义指标
    CUSTOM = auto()     # 自定义指标
    
    # 之前添加的指标
    WMA = "WMA"  # 加权移动平均线(WMA)
    SAR = "SAR"  # 抛物线转向(SAR)
    DMI = "DMI"  # 趋向指标(DMI)
    BIAS = "BIAS"  # 均线多空指标(BIAS)
    WR = "WR"  # 威廉指标(WR)
    CCI = "CCI"  # 顺势指标(CCI)
    MTM = "MTM"  # 动量指标(MTM)
    ROC = "ROC"  # 变动率(ROC)
    STOCHRSI = "STOCHRSI"  # 随机相对强弱指标(StochRSI)
    VOSC = "VOSC"  # 成交量变异率(VOSC)
    MFI = "MFI"  # 资金流向指标(MFI)
    VR = "VR"  # 成交量指标(VR)
    PVT = "PVT"  # 价量趋势指标(PVT)
    EMV = "EMV"  # 指数平均数指标(EMV)
    VIX = "VIX"  # 恐慌指数(VIX)
    
    # 之前新增实现的指标
    TRIX = "TRIX"  # TRIX三重指数平滑移动平均线
    ZXM_ABSORB = "ZXM_ABSORB"  # ZXM核心吸筹公式
    DIVERGENCE = "DIVERGENCE"  # 量价背离指标
    MULTI_PERIOD_RESONANCE = "MULTI_PERIOD_RESONANCE"  # 多周期共振分析指标
    
    # 新实现的指标
    CANDLESTICK_PATTERNS = "CANDLESTICK_PATTERNS"  # K线形态识别指标
    ZXM_WASHPLATE = "ZXM_WASHPLATE"  # ZXM洗盘形态指标
    CHIP_DISTRIBUTION = "CHIP_DISTRIBUTION"  # 筹码分布指标
    FIBONACCI_TOOLS = "FIBONACCI_TOOLS"  # 斐波那契工具指标
    ELLIOTT_WAVE = "ELLIOTT_WAVE"  # 艾略特波浪理论分析指标
    GANN_TOOLS = "GANN_TOOLS"  # 江恩理论工具指标
    
    # 本次新增实现的指标
    VOLUME_RATIO = "VOLUME_RATIO"  # 量比指标
    PLATFORM_BREAKOUT = "PLATFORM_BREAKOUT"  # 平台突破指标
    INTRADAY_VOLATILITY = "INTRADAY_VOLATILITY"  # 日内波动率指标
    V_SHAPED_REVERSAL = "V_SHAPED_REVERSAL"  # V形反转指标
    ISLAND_REVERSAL = "ISLAND_REVERSAL"  # 岛型反转指标
    TIME_CYCLE_ANALYSIS = "TIME_CYCLE_ANALYSIS"  # 时间周期分析指标
    
    # 新增高级指标
    MOMENTUM = "Momentum"  # 动量指标(Momentum)
    RSIMA = "RSIMA"  # RSI均线系统指标
    
    # ZXM体系指标
    # 趋势指标
    ZXM_DAILY_TREND_UP = "ZXM_DAILY_TREND_UP"  # ZXM趋势-日线上移指标
    ZXM_WEEKLY_TREND_UP = "ZXM_WEEKLY_TREND_UP"  # ZXM趋势-周线上移指标
    ZXM_MONTHLY_KDJ_TREND_UP = "ZXM_MONTHLY_KDJ_TREND_UP"  # ZXM趋势-月KDJ·D及K上移指标
    ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP = "ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP"  # ZXM趋势-周KDJ·D/DEA上移指标
    ZXM_WEEKLY_KDJ_D_TREND_UP = "ZXM_WEEKLY_KDJ_D_TREND_UP"  # ZXM趋势-周KDJ·D上移指标
    ZXM_MONTHLY_MACD = "ZXM_MONTHLY_MACD"  # ZXM趋势-月MACD<1.5指标
    ZXM_WEEKLY_MACD = "ZXM_WEEKLY_MACD"  # ZXM趋势-周MACD<2指标
    
    # 弹性指标
    ZXM_AMPLITUDE_ELASTICITY = "ZXM_AMPLITUDE_ELASTICITY"  # ZXM弹性-振幅指标
    ZXM_RISE_ELASTICITY = "ZXM_RISE_ELASTICITY"  # ZXM弹性-涨幅指标
    
    # 买点指标
    ZXM_DAILY_MACD = "ZXM_DAILY_MACD"  # ZXM买点-日MACD指标
    ZXM_TURNOVER = "ZXM_TURNOVER"  # ZXM买点-换手率指标
    ZXM_VOLUME_SHRINK = "ZXM_VOLUME_SHRINK"  # ZXM买点-缩量指标
    ZXM_MA_CALLBACK = "ZXM_MA_CALLBACK"  # ZXM买点-回踩均线指标
    ZXM_BS_ABSORB = "ZXM_BS_ABSORB"  # ZXM买点-BS吸筹指标
    
    # 综合指标
    ZXM_SELECTION_MODEL = "ZXM_SELECTION_MODEL"  # ZXM体系通用选股模型


class TimeFrame(Enum):
    """时间周期枚举"""
    
    MINUTE_1 = "1min"       # 1分钟
    MINUTE_5 = "5min"       # 5分钟
    MINUTE_15 = "15min"     # 15分钟
    MINUTE_30 = "30min"     # 30分钟
    MINUTE_60 = "60min"     # 60分钟
    
    DAILY = "daily"         # 日线
    WEEKLY = "weekly"       # 周线
    MONTHLY = "monthly"     # 月线
    
    QUARTERLY = "quarterly" # 季线
    YEARLY = "yearly"       # 年线


class CrossType(Enum):
    """交叉类型枚举"""
    
    GOLDEN_CROSS = auto()   # 金叉，第一条线从下方穿过第二条线
    DEATH_CROSS = auto()    # 死叉，第一条线从上方穿过第二条线


class TrendType(Enum):
    """趋势类型枚举"""
    
    UP = auto()      # 上升趋势
    DOWN = auto()    # 下降趋势
    FLAT = auto()    # 震荡趋势 