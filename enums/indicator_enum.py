from enum import Enum


class IndicatorEnum(str, Enum):
    """指标常量枚举"""

    # ZXM系列指标
    ZXM_ABSORB = "ZXM_ABSORB"  # ZXM吸筹
    ZXM_TURNOVER = "ZXM_TURNOVER"  # ZXM换手买点
    ZXM_DAILY_MACD = "ZXM_DAILY_MACD"  # ZXM日MACD买点
    ZXM_MA_CALLBACK = "ZXM_MA_CALLBACK"  # ZXM回踩均线买点
    ZXM_RISE_ELASTICITY = "ZXM_RISE_ELASTICITY"  # ZXM涨幅弹性
    ZXM_AMPLITUDE_ELASTICITY = "ZXM_AMPLITUDE_ELASTICITY"  # ZXM振幅弹性
    ZXM_ELASTICITY_SCORE = "ZXM_ELASTICITY_SCORE"  # ZXM弹性满足
    ZXM_BUYPOINT_SCORE = "ZXM_BUYPOINT_SCORE"  # ZXM买点满足
    ZXM_DAILY_TREND_UP = "ZXM_DAILY_TREND_UP"  # ZXM趋势满足

    # 经典技术指标
    MACD = "MACD"  # MACD指标
    KDJ = "KDJ"  # KDJ指标
    MA = "MA"  # 移动平均线
    BOLL = "BOLL"  # 布林带
    VR = "VR"  # 成交量比率
    OBV = "OBV"  # 能量潮
    RSI = "RSI"  # 相对强弱指数
    DMI = "DMI"  # 趋向指标
    CCI = "CCI"  # 顺势指标
    ATR = "ATR"  # 平均真实波幅
    BIAS = "BIAS"  # 乖离率
    WMA = "WMA"  # 加权移动平均线
    EMA = "EMA"  # 指数移动平均线
    PVT = "PVT"  # 价量趋势指标
    MFI = "MFI"  # 资金流向指标
    VOSC = "VOSC"  # 成交量变异率
    ROC = "ROC"  # 变动率指标
    MTM = "MTM"  # 动量指标
    WR = "WR"  # 威廉指标
    SAR = "SAR"  # 抛物线转向指标

    # 新增指标
    TRIX = "TRIX"  # TRIX三重指数平滑移动平均线
    CMO = "CMO"  # 钱德动量摆动指标
    DMA = "DMA"  # 动态移动平均线
    KC = "KC"  # 肯特纳通道
    EMV = "EMV"  # 简易波动指标
    ICHIMOKU = "ICHIMOKU"  # 一目均衡表
    VORTEX = "VORTEX"  # 涡旋指标
    STOCHRSI = "STOCHRSI"  # 随机RSI

    # 新增增强指标
    UNIFIED_MA = "UNIFIED_MA"  # 统一移动平均线
    ENHANCED_MACD = "ENHANCED_MACD"  # 增强版MACD
    ENHANCED_RSI = "ENHANCED_RSI"  # 增强版RSI 