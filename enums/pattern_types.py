"""
K线形态和吸筹买点模式的枚举类型

定义各种K线形态和吸筹买点模式的枚举类型
"""

from enum import Enum, auto


class CandlePatternType(Enum):
    """K线形态枚举"""
    
    # 单K线形态
    DOJI = auto()               # 十字星
    HAMMER = auto()             # 锤子线
    INVERTED_HAMMER = auto()    # 倒锤子线
    SHOOTING_STAR = auto()      # 流星线
    MARUBOZU = auto()           # 光头光脚阳线/阴线
    LONG_LEGGED_DOJI = auto()   # 长腿十字
    DRAGONFLY_DOJI = auto()     # T字线（蜻蜓十字）
    GRAVESTONE_DOJI = auto()    # 倒T字线（墓碑十字）
    
    # 双K线形态
    BULLISH_ENGULFING = auto()  # 看涨吞没
    BEARISH_ENGULFING = auto()  # 看跌吞没
    TWEEZER_TOP = auto()        # 顶部镊子
    TWEEZER_BOTTOM = auto()     # 底部镊子
    PIERCING_LINE = auto()      # 刺透形态
    DARK_CLOUD_COVER = auto()   # 乌云盖顶
    
    # 三K线形态
    MORNING_STAR = auto()       # 早晨之星
    EVENING_STAR = auto()       # 黄昏之星
    THREE_WHITE_SOLDIERS = auto() # 三白兵
    THREE_BLACK_CROWS = auto()  # 三黑鸦
    THREE_INSIDE_UP = auto()    # 三内涨
    THREE_INSIDE_DOWN = auto()  # 三内跌
    
    # 多K线形态
    HEAD_AND_SHOULDERS = auto() # 头肩顶
    INVERSE_HEAD_AND_SHOULDERS = auto() # 头肩底
    DOUBLE_TOP = auto()         # 双重顶
    DOUBLE_BOTTOM = auto()      # 双重底
    TRIPLE_TOP = auto()         # 三重顶
    TRIPLE_BOTTOM = auto()      # 三重底
    CUP_AND_HANDLE = auto()     # 杯柄形态
    TRIANGLE = auto()           # 三角形
    FLAG = auto()               # 旗形
    WEDGE = auto()              # 楔形
    RECTANGLE = auto()          # 矩形


class AbsorptionPatternType(Enum):
    """吸筹形态枚举"""
    
    # 初期吸筹特征
    VOLUME_DECREASE = auto()    # 缩量阴线
    DECLINE_SLOW_DOWN = auto()  # 均线下趋势变缓
    DECLINE_REDUCE = auto()     # 股价下跌幅度递减
    LOW_LEVEL_DOJI = auto()     # 低位十字星频现
    VOLUME_PRICE_DIVERGE = auto() # 成交量与股价背离
    
    # 中期吸筹特征
    STEP_UP_DOWN = auto()       # 阶梯式放量上涨后回落
    HIGH_TURNOVER_LOW_VOLATILITY = auto() # 高换手低波动
    KEY_SUPPORT_HOLD = auto()   # 关键价位精准支撑
    VOLUME_PRICE_IMPROVE = auto() # 量价关系恶化后好转
    MACD_DOUBLE_DIVERGE = auto() # MACD二次背离
    
    # 后期吸筹特征
    VOLUME_SHRINK_RANGE = auto() # 缩量横盘整理
    MA_CONVERGENCE = auto()     # 均线开始粘合
    MACD_ZERO_HOVER = auto()    # MACD零轴附近徘徊
    BOTTOM_BOX_LIFT = auto()    # 底部箱体逐渐抬高
    FAKE_BREAK_RECOVER = auto() # 假突破后快速回补
    
    # ZXM特有吸筹形态
    LONG_LOWER_SHADOW = auto()  # 长下影线收单阳
    CONTINUOUS_INTRADAY_RED = auto() # 连续多日分时收红
    MA_PRECISE_SUPPORT = auto() # 沿均线回调精准支撑
    SMALL_ALTERNATING = auto()  # 连续阴阳小实体交替
    INTRADAY_PATTERN = auto()   # 分时图早跌午拉尾盘突破


class BuyPointType(Enum):
    """买点形态枚举"""
    
    # 基本买点类型
    CLASS_ONE = auto()          # 一类买点(主升浪启动)
    CLASS_TWO = auto()          # 二类买点(回调支撑)
    CLASS_THREE = auto()        # 三类买点(超跌反弹)
    
    # ZXM特有买点
    BREAK_THROUGH_PULLBACK = auto() # 强势突破回踩型
    VOLUME_SHRINK_PLATFORM = auto() # 连续缩量平台型
    LONG_SHADOW_SUPPORT = auto()    # 长下影线支撑型
    MA_CONVERGE_DIVERGE = auto()    # 均线粘合发散型
    
    # 洗盘形态
    SHAKEOUT_SIDEWAYS = auto()      # 横盘震荡洗盘
    SHAKEOUT_PULLBACK = auto()      # 回调洗盘
    SHAKEOUT_FAKE_BREAK = auto()    # 假突破洗盘
    SHAKEOUT_TIME = auto()          # 时间洗盘
    SHAKEOUT_CONTINUOUS_DOWN = auto() # 连续阴线洗盘


class KLinePosition(Enum):
    """K线位置枚举"""
    
    TOP = auto()        # 顶部
    BOTTOM = auto()     # 底部
    UPTREND = auto()    # 上升趋势中
    DOWNTREND = auto()  # 下降趋势中
    SIDEWAYS = auto()   # 横盘整理中
    SUPPORT = auto()    # 支撑位附近
    RESISTANCE = auto() # 压力位附近


class VolumePattern(Enum):
    """成交量形态枚举"""
    
    VOLUME_EXPANSION = auto()    # 放量
    VOLUME_SHRINK = auto()       # 缩量
    VOLUME_STEADY = auto()       # 量能平稳
    VOLUME_DIVERGENCE_UP = auto() # 量价背离(价跌量增)
    VOLUME_DIVERGENCE_DOWN = auto() # 量价背离(价涨量减)
    VOLUME_BREAKOUT = auto()     # 突破性放量
    VOLUME_CLIMAX = auto()       # 量能顶峰(快速放大后见顶)
    VOLUME_DRY_UP = auto()       # 量能枯竭(持续萎缩至极低)
    
    
class PriceActionType(Enum):
    """价格行为形态枚举"""
    
    BREAKOUT = auto()            # 突破
    BREAKDOWN = auto()           # 跌破
    PULLBACK = auto()            # 回调
    BOUNCE = auto()              # 反弹
    CONSOLIDATION = auto()       # 盘整
    GAP_UP = auto()              # 跳空上涨
    GAP_DOWN = auto()            # 跳空下跌
    SQUEEZE = auto()             # 压缩(波动逐渐减小)
    EXPANSION = auto()           # 扩张(波动逐渐增大)
    TREND_CHANGE = auto()        # 趋势改变
    V_REVERSAL = auto()          # V型反转
    W_REVERSAL = auto()          # W型反转
    INVERTED_V_REVERSAL = auto() # 倒V型反转
    INVERTED_W_REVERSAL = auto() # 倒W型反转 