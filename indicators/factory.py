"""
指标工厂模块

提供统一的指标创建接口
"""

from typing import Dict, Type, Any, Optional

from enums.indicator_types import IndicatorType
from indicators.base_indicator import BaseIndicator
from indicators.ma import MA
from indicators.ema import EMA
from indicators.wma import WMA
from indicators.macd import MACD
from indicators.bias import BIAS
from indicators.boll import BOLL
from indicators.sar import SAR
from indicators.dmi import DMI
from indicators.rsi import RSI
from indicators.kdj import KDJ
from indicators.wr import WR
from indicators.cci import CCI
from indicators.mtm import MTM
from indicators.roc import ROC
from indicators.stochrsi import STOCHRSI
from indicators.vol import VOL
from indicators.obv import OBV
from indicators.vosc import VOSC
from indicators.mfi import MFI
from indicators.vr import VR
from indicators.pvt import PVT
from indicators.volume_ratio import VolumeRatio
from indicators.platform_breakout import PlatformBreakout
from indicators.pattern.candlestick_patterns import CandlestickPatterns
from indicators.zxm_washplate import ZXMWashPlate
from indicators.chip_distribution import ChipDistribution
from indicators.fibonacci_tools import FibonacciTools
from indicators.elliott_wave import ElliottWave
from indicators.gann_tools import GannTools
from indicators.emv import EMV
from indicators.intraday_volatility import IntradayVolatility
from indicators.v_shaped_reversal import VShapedReversal
from indicators.island_reversal import IslandReversal
from indicators.time_cycle_analysis import TimeCycleAnalysis
from indicators.momentum import Momentum
from indicators.rsima import RSIMA
# 添加已有但未集成的指标
from indicators.trix import TRIX
from indicators.vix import VIX
from indicators.divergence import DIVERGENCE
from indicators.multi_period_resonance import MULTI_PERIOD_RESONANCE
from indicators.zxm_absorb import ZXM_ABSORB

# 导入ZXM体系指标
from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp, 
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp,
    ZXMMonthlyMACD, ZXMWeeklyMACD
)
from indicators.zxm.elasticity_indicators import (
    ZXMAmplitudeElasticity, ZXMRiseElasticity
)
from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb
)
from indicators.zxm.selection_model import ZXMSelectionModel

from utils.logger import get_logger

logger = get_logger(__name__)


class IndicatorFactory:
    """
    指标工厂类
    
    提供统一的指标创建方法
    """
    
    # 指标类型映射表
    _indicators: Dict[str, Type[BaseIndicator]] = {
        IndicatorType.MA.name: MA,
        IndicatorType.EMA.name: EMA,
        IndicatorType.WMA.name: WMA,
        IndicatorType.MACD.name: MACD,
        IndicatorType.BIAS.name: BIAS,
        IndicatorType.BOLL.name: BOLL,
        IndicatorType.SAR.name: SAR,
        IndicatorType.DMI.name: DMI,
        IndicatorType.RSI.name: RSI,
        IndicatorType.KDJ.name: KDJ,
        IndicatorType.WR.name: WR,
        IndicatorType.CCI.name: CCI,
        IndicatorType.MTM.name: MTM,
        IndicatorType.ROC.name: ROC,
        IndicatorType.STOCHRSI.name: STOCHRSI,
        IndicatorType.VOL.name: VOL,
        IndicatorType.OBV.name: OBV,
        IndicatorType.VOSC.name: VOSC,
        IndicatorType.MFI.name: MFI,
        IndicatorType.VR.name: VR,
        IndicatorType.PVT.name: PVT,
        IndicatorType.VOLUME_RATIO.name: VolumeRatio,
        IndicatorType.PLATFORM_BREAKOUT.name: PlatformBreakout,
        IndicatorType.CANDLESTICK_PATTERNS.name: CandlestickPatterns,
        IndicatorType.ZXM_WASHPLATE.name: ZXMWashPlate,
        IndicatorType.CHIP_DISTRIBUTION.name: ChipDistribution,
        IndicatorType.FIBONACCI_TOOLS.name: FibonacciTools,
        IndicatorType.ELLIOTT_WAVE.name: ElliottWave,
        IndicatorType.GANN_TOOLS.name: GannTools,
        IndicatorType.EMV.name: EMV,
        IndicatorType.INTRADAY_VOLATILITY.name: IntradayVolatility,
        IndicatorType.V_SHAPED_REVERSAL.name: VShapedReversal,
        IndicatorType.ISLAND_REVERSAL.name: IslandReversal,
        IndicatorType.TIME_CYCLE_ANALYSIS.name: TimeCycleAnalysis,
        IndicatorType.MOMENTUM.name: Momentum,
        IndicatorType.RSIMA.name: RSIMA,
        # 添加已有但未集成的指标
        IndicatorType.TRIX.name: TRIX,
        IndicatorType.VIX.name: VIX,
        IndicatorType.DIVERGENCE.name: DIVERGENCE,
        IndicatorType.MULTI_PERIOD_RESONANCE.name: MULTI_PERIOD_RESONANCE,
        IndicatorType.ZXM_ABSORB.name: ZXM_ABSORB,
        
        # 添加ZXM体系指标
        # 趋势指标
        IndicatorType.ZXM_DAILY_TREND_UP.name: ZXMDailyTrendUp,
        IndicatorType.ZXM_WEEKLY_TREND_UP.name: ZXMWeeklyTrendUp,
        IndicatorType.ZXM_MONTHLY_KDJ_TREND_UP.name: ZXMMonthlyKDJTrendUp,
        IndicatorType.ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP.name: ZXMWeeklyKDJDOrDEATrendUp,
        IndicatorType.ZXM_WEEKLY_KDJ_D_TREND_UP.name: ZXMWeeklyKDJDTrendUp,
        IndicatorType.ZXM_MONTHLY_MACD.name: ZXMMonthlyMACD,
        IndicatorType.ZXM_WEEKLY_MACD.name: ZXMWeeklyMACD,
        
        # 弹性指标
        IndicatorType.ZXM_AMPLITUDE_ELASTICITY.name: ZXMAmplitudeElasticity,
        IndicatorType.ZXM_RISE_ELASTICITY.name: ZXMRiseElasticity,
        
        # 买点指标
        IndicatorType.ZXM_DAILY_MACD.name: ZXMDailyMACD,
        IndicatorType.ZXM_TURNOVER.name: ZXMTurnover,
        IndicatorType.ZXM_VOLUME_SHRINK.name: ZXMVolumeShrink,
        IndicatorType.ZXM_MA_CALLBACK.name: ZXMMACallback,
        IndicatorType.ZXM_BS_ABSORB.name: ZXMBSAbsorb,
        
        # 综合指标
        IndicatorType.ZXM_SELECTION_MODEL.name: ZXMSelectionModel,
    }
    
    @classmethod
    def create(cls, indicator_type: str, **params) -> Optional[BaseIndicator]:
        """
        创建指标实例
        
        Args:
            indicator_type: 指标类型
            **params: 指标参数
            
        Returns:
            BaseIndicator: 指标实例
        """
        # 获取指标类
        indicator_class = cls._indicators.get(indicator_type)
        
        if indicator_class is None:
            logger.error(f"未知的指标类型: {indicator_type}")
            return None
        
        # 创建指标实例
        try:
            return indicator_class(**params)
        except Exception as e:
            logger.error(f"创建指标实例失败: {e}")
            return None
    
    @classmethod
    def register_indicator(cls, indicator_type: str, indicator_class: Type[BaseIndicator]) -> None:
        """
        注册指标类型
        
        Args:
            indicator_type: 指标类型
            indicator_class: 指标类
        """
        cls._indicators[indicator_type] = indicator_class
        logger.info(f"已注册指标: {indicator_type}")
    
    @classmethod
    def get_indicator_types(cls) -> Dict[str, Type[BaseIndicator]]:
        """
        获取所有指标类型
        
        Returns:
            Dict[str, Type[BaseIndicator]]: 指标类型映射表
        """
        return cls._indicators.copy()

    @classmethod
    def create_indicator(cls, indicator_type: str, **params) -> Optional[BaseIndicator]:
        """
        兼容旧接口，调用create方法
        Args:
            indicator_type: 指标类型
            **params: 指标参数
        Returns:
            BaseIndicator: 指标实例
        """
        return cls.create(indicator_type, **params)
        
    @classmethod
    def get_supported_indicators(cls) -> list:
        """
        获取所有支持的指标类型名称
        
        Returns:
            list: 指标类型名称列表
        """
        return list(cls._indicators.keys())