import pandas as pd
from utils.container import container
from indicators.base_indicator import BaseIndicator

"""
完整的88+指标注册管理器

支持系统中所有88+个技术指标的注册和管理,实现生产级的指标体系
"""

import logging
import importlib
from typing import Dict, Any, List, Set, Optional
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class CompleteIndicatorRegistry:
    """完整的88+指标注册管理器"""

    def __init__(self):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        self._indicators = {}
        self._failed_indicators = []
        self._registration_log = []

    def register_all_indicators(self):
        """注册所有88+个指标"""
        logger.info("=== 开始注册全部88+个指标 ===")

        total_registered = 0
        total_failed = 0

        # 1. 核心指标(6个)
        total_registered += self._register_core_indicators()

        # 2. 趋势指标(10个)
        total_registered += self._register_trend_indicators()

        # 3. 振荡器指标(9个)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_oscillator_indicators()

        # 4. 成交量指标(9个)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_volume_indicators()

        # 5. 波动性指标(4个)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_volatility_indicators()

        # 6. ZXM体系指标(35个)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_zxm_indicators()

        # 7. 形态识别指标(21个)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_pattern_indicators()

        # 8. 增强指标(3个)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_enhanced_indicators()

        # 9. 其他专业指标(剩余)  # TODO: 将魔法数字提取到配置中
        total_registered += self._register_professional_indicators()

        total_failed = len(self._failed_indicators)
        success_rate = (
            (total_registered / (total_registered + total_failed)) * 100 if (total_registered + total_failed) > 0 else 0
        )

        logger.info(f"✅ 指标注册完成: 成功 {total_registered} 个, 失败 {total_failed} 个")
        logger.info(f"📊 注册成功率: {success_rate:.1f}%")

        if self._failed_indicators:
            logger.warning(f"❌ 注册失败的指标: {', '.join(self._failed_indicators)}")

        return total_registered

    def _register_core_indicators(self) -> int:
        """注册核心指标"""
        logger.info("注册核心指标...")

        core_indicators = {
            "MA": "indicators.ma.MaMa",
            "EMA": "indicators.ema.EmaEma",
            "MACD": "indicators.macd.MacdMacd",
            "RSI": "indicators.rsi.RsiRsi",
            "BOLL": "indicators.boll.BOLL",
            "PSY": "indicators.psy.PSY",
        }

        return self._batch_register(core_indicators, "核心指标")

    def _register_trend_indicators(self) -> int:
        """注册趋势指标"""
        logger.info("注册趋势指标...")

        trend_indicators = {
            "DMA": "indicators.dma.DMA",
            "DMI": "indicators.dmi.DMI",
            "ADX": "indicators.adx.AverageDirectionalIndex",
            "AROON": "indicators.aroon.AROON",
            "SAR": "indicators.sar.Sar",
            "PSAR": "indicators.sar.Sar",  # 别名
            "TRIX": "indicators.trix.TripleExponentialAverage",
            "CCI": "indicators.cci.CCI",
            "ENHANCED_CCI": "indicators.trend.enhanced_cci.EnhancedCci",
            "ENHANCED_TRIX": "indicators.trend.enhanced_trix.EnhancedTrix",
            "WMA": "indicators.wma.WMA",
            "SUPERTREND": "indicators.supertrend.SuperTrend",
            "SMA": "indicators.unified_calculator.SimpleMovingAverageCalculator",
        }

        return self._batch_register(trend_indicators, "趋势指标")

    def _register_oscillator_indicators(self) -> int:
        """注册振荡器指标"""
        logger.info("注册振荡器指标...")

        oscillator_indicators = {
            "KDJ": "indicators.kdj.KDJ",
            "WR": "indicators.wr.WrWr",  # 修复类名: WR -> WrWr
            "WILLR": "indicators.wr.WrWr",  # 添加WILLR别名
            "WILLIAMS_R": "indicators.wr.WrWr",  # 修复类名: WR -> WrWr
            "CMO": "indicators.cmo.ChandeMomentumOscillator",
            "STOCHRSI": "indicators.stochrsi.Stochrsi",
            "STOCH": "indicators.stochrsi.Stochrsi",  # 别名
            # "ENHANCED_RSI": "indicators.enhanced_rsi.EnhancedRSI",  # P1.1.2: 已整合到主RSI实现
            # "ENHANCED_KDJ": "indicators.oscillator.enhanced_kdj.EnhancedKdj",  # P1.1.3: 已整合到主KDJ实现
            "ENHANCED_WR": "indicators.enhanced_wr.EnhancedWR",
            "MOMENTUM": "indicators.momentum.MOMENTUM",
            "ROC": "indicators.roc.RateOfChange",
            "ROC_OSCILLATOR": "indicators.roc.RateOfChange",  # 别名
            "ULTIMATE": "indicators.ultimate.Ultimate",
        }

        return self._batch_register(oscillator_indicators, "振荡器指标")

    def _register_volume_indicators(self) -> int:
        """注册成交量指标"""
        logger.info("注册成交量指标...")

        volume_indicators = {
            "OBV": "indicators.obv.OBV",
            "AD": "indicators.ad.AD",
            "EMV": "indicators.emv.EmvEmv",
            "VOL": "indicators.vol.VOL",
            "VR": "indicators.vr.VR",
            "VOSC": "indicators.vosc.VOSC",
            "MFI": "indicators.mfi.MFI",
            "PVT": "indicators.pvt.PVT",
            "CHAIKIN": "indicators.chaikin.CHAIKIN",
            "FORCE_INDEX": "indicators.force_index.ForceIndex",
        }

        return self._batch_register(volume_indicators, "成交量指标")

    def _register_volatility_indicators(self) -> int:
        """注册波动性指标"""
        logger.info("注册波动性指标...")

        volatility_indicators = {
            "ATR": "indicators.atr.ATR",
            "KC": "indicators.kc.KC",
            "VIX": "indicators.vix.VIX",
            "STDDEV": "indicators.vol.STDDEV",
            "VOLATILITY": "indicators.vol.STDDEV",  # 别名
            "CHAIKIN_VOLATILITY": "indicators.chaikin_volatility.ChaikinVolatility",
            "GARMAN_KLASS": "indicators.garman_klass.GarmanKlass",
        }

        return self._batch_register(volatility_indicators, "波动性指标")

    def _register_zxm_indicators(self) -> int:
        """注册ZXM体系指标"""
        logger.info("注册ZXM体系指标...")

        zxm_indicators = {
            # 买点指标
            "ZXM_DAILY_MACD": "indicators.zxm.buy_point_indicators.ZXMDailyMACD",
            "ZXM_turnover_rate": "indicators.zxm.buy_point_indicators.ZXMTurnover",
            "ZXM_VOLUME_SHRINK": "indicators.zxm.buy_point_indicators.ZXMVolumeShrink",
            "ZXM_MA_CALLBACK": "indicators.zxm.buy_point_indicators.ZXMMACallback",
            "ZXM_BS_ABSORB": "indicators.zxm.buy_point_indicators.ZXMBSAbsorb",
            # 趋势指标
            "ZXM_DAILY_TREND_UP": "indicators.zxm.trend_indicators.ZxmdailyTrendUp",
            "ZXM_WEEKLY_TREND_UP": "indicators.zxm.trend_indicators.ZxmweeklyTrendUp",
            "ZXM_MONTHLY_KDJ_TREND_UP": "indicators.zxm.trend_indicators.ZxmmonthlyKdjtrendUp",
            "ZXM_WEEKLY_MACD": "indicators.zxm.trend_indicators.ZxmweeklyMacd",  # 修复类名:ZXMWeeklyMACD -> ZxmweeklyMacd
            "ZXM_MONTHLY_MACD": "indicators.zxm.trend_indicators.ZxmmonthlyMacd",  # 修复类名:ZXMMonthlyMACD -> ZxmmonthlyMacd
            # 弹性指标
            "ZXM_AMPLITUDE_ELASTICITY": "indicators.zxm.elasticity_indicators.AmplitudeElasticity",
            "ZXM_RISE_ELASTICITY": "indicators.zxm.elasticity_indicators.ZxmriseElasticity",
            "ZXM_ELASTICITY": "indicators.zxm.elasticity_indicators.Elasticity",
            "ZXM_BOUNCE_DETECTOR": "indicators.zxm.elasticity_indicators.BounceDetector",
            # 评分指标
            "ZXM_BUYPOINT_SCORE": "indicators.zxm.score_indicators.ZxmbuyPointScore",
            "ZXM_TREND_SCORE": "indicators.zxm.score_indicators.StockScoreCalculator",  # 修复类名
            "ZXM_ELASTIC_SCORE": "indicators.zxm.score_indicators.ZxmelasticityScore",
            # 其他专业指标(修复路径和类名)
            "ZXM_VOLUME_ENERGY": "indicators.zxm.market_breadth.ZxmmarketBreadth",
            "ZXM_PRICE_POSITION": "indicators.zxm.diagnostics.ZXMDiagnostics",
            "ZXM_TECHNICAL_FORM": "indicators.zxm.selection_model.SelectionModel",
            "ZXM_MARKET_SENTIMENT": "indicators.zxm.market_breadth.ZxmmarketBreadth",
            "ZXM_CHIP_DISTRIBUTION": "indicators.chip_distribution.ChipDistribution",
            "ZXM_FUND_FLOW": "indicators.institutional_behavior.FundFlow",
            "ZXM_INSTITUTION_BEHAVIOR": "indicators.institutional_behavior.InstitutionalBehavior",
            "ZXM_HOT_SPOT": "indicators.zxm.hot_spot_indicators.ZXMHotSpot",  # 修复路径
            "ZXM__ROTATION": "indicators.zxm.industry_rotation_indicators.ZXMRotation",  # 修复路径
            "ZXM_CYCLE_POSITION": "indicators.zxm.cycle_position_indicators.ZXMCyclePosition",  # 修复路径
            "ZXM_RISK_CONTROL": "indicators.zxm.risk_control_indicators.ZXMRiskControl",  # 修复路径
            "ZXM_TIMING_SIGNAL": "indicators.zxm.timing_signal_indicators.ZXMTimingSignal",  # 修复路径
            "ZXM_POSITION_MANAGEMENT": "indicators.zxm.position_management_indicators.ZXMPositionManagement",  # 修复路径
            "ZXM_PORTFOLIO_OPTIMIZATION": "indicators.zxm.portfolio_optimization_indicators.ZXMPortfolioOptimization",  # 修复路径
            "ZXM_STRATEGY_COMBINATION": "indicators.zxm.strategy_combination_indicators.ZXMStrategyCombination",  # 修复路径
            "ZXM_PERFORMANCE_ATTRIBUTION": "indicators.zxm.performance_attribution_indicators.ZXMPerformanceAttribution",  # 修复路径
            "ZXM_ALPHA_GENERATION": "indicators.zxm.alpha_generation_indicators.ZXMAlphaGeneration",  # 修复路径
            "ZXM_BETA_HEDGING": "indicators.zxm.beta_hedging_indicators.ZXMBetaHedging",  # 修复路径
            # 新增的ZXM指标
            "ZXM_LIQUIDITY_ANALYSIS": "indicators.zxm.zxm_liquidity_analysis.ZXMLiquidityAnalysis",
            "ZXM_VOLATILITY_FORECAST": "indicators.zxm.zxm_volatility_forecast.ZXMVolatilityForecast",
            "ZXM_CORRELATION_MATRIX": "indicators.zxm.zxm_correlation_matrix.ZXMCorrelationMatrix",
        }

        return self._batch_register(zxm_indicators, "ZXM体系指标")

    def _register_pattern_indicators(self) -> int:
        """注册形态识别指标"""
        logger.info("注册形态识别指标...")

        pattern_indicators = {
            "CANDLESTICK_PATTERNS": "indicators.pattern.candlestick_patterns.CandlestickPatterns",
            "DOJI": "indicators.pattern.candlestick_patterns.Doji",
            "HAMMER": "indicators.pattern.candlestick_patterns.Hammer",
            "SHOOTING_STAR": "indicators.pattern.candlestick_patterns.ShootingStar",
            "ENGULFING": "indicators.pattern.candlestick_patterns.Engulfing",
            "HARAMI": "indicators.pattern.candlestick_patterns.Harami",
            "PIERCING_LINE": "indicators.pattern.candlestick_patterns.PiercingLine",
            "DARK_CLOUD_COVER": "indicators.pattern.candlestick_patterns.DarkCloudCover",
            "MORNING_STAR": "indicators.pattern.candlestick_patterns.MorningStar",
            "EVENING_STAR": "indicators.pattern.candlestick_patterns.EveningStar",
            "THREE_BLACK_CROWS": "indicators.pattern.candlestick_patterns.ThreeBlackCrows",
            "THREE_WHITE_SOLDIERS": "indicators.pattern.candlestick_patterns.ThreeWhiteSoldiers",
            "ISLAND_REVERSAL": "indicators.island_reversal.IslandReversal",
            "V_SHAPED_REVERSAL": "indicators.v_shaped_reversal.VShapedReversal",
            "HEAD_SHOULDERS": "indicators.pattern.advanced_candlestick_patterns.HeadShoulders",
            "DOUBLE_TOP": "indicators.pattern.advanced_candlestick_patterns.DoubleTop",
            "DOUBLE_BOTTOM": "indicators.pattern.advanced_candlestick_patterns.DoubleBottom",
            "TRIANGLE": "indicators.pattern.advanced_candlestick_patterns.Triangle",
            "WEDGE": "indicators.pattern.advanced_candlestick_patterns.Wedge",
            "FLAG": "indicators.pattern.advanced_candlestick_patterns.Flag",
            "PENNANT": "indicators.pattern.advanced_candlestick_patterns.Pennant",
            "RECTANGLE": "indicators.pattern.rectangle.Rectangle",
            "CUP_AND_HANDLE": "indicators.pattern.cup_and_handle.CupAndHandle",
        }

        return self._batch_register(pattern_indicators, "形态识别指标")

    def _register_enhanced_indicators(self) -> int:
        """注册增强指标"""
        logger.info("注册增强指标...")

        enhanced_indicators = {
            # "ENHANCED_MACD": "indicators.enhanced_macd.EnhancedMACD",  # P1.1.1: 已整合到主MACD实现
            "ENHANCED_BOLL": "indicators.trend.enhanced_boll_indicators.EnhancedBoll",
            "ENHANCED_STOCHRSI": "indicators.enhanced_stochrsi.EnhancedStochasticRSI",
        }

        return self._batch_register(enhanced_indicators, "增强指标")

    def _register_professional_indicators(self) -> int:
        """注册其他专业指标"""
        logger.info("注册其他专业指标...")

        professional_indicators = {
            # 高级技术分析
            "FIBONACCI": "indicators.fibonacci.Fibonacci",
            "ELLIOTT_WAVE": "indicators.elliott_wave.ElliottWave",
            "GANN": "indicators.gann_tools.GannTools",
            "ICHIMOKU": "indicators.ichimoku.Ichimoku",
            "VORTEX": "indicators.vortex.Vortex",
            # 市场微观结构
            "BIAS": "indicators.bias.BIAS",
            "MTM": "indicators.mtm.Mtm",  # 修复类名:MTM -> Mtm
            "RSIMA": "indicators.rsima.Rsima",
            # 复合指标
            "COMPOSITE": "indicators.composite.COMPOSITE",  # 修复类名:Composite -> COMPOSITE
            "SYNERGY": "indicators.synergy.Synergy",
            "UNIFIED_MA": "indicators.unified_ma.UNIFIED_MA",
            # 评分框架
            "MACD_SCORE": "indicators.macd_score.MACDScore",
            # "RSI_SCORE": "indicators.rsi_score.RSIScore",  # P1.1.2: 已整合到主RSI实现
            "BOLL_SCORE": "indicators.boll_score.BOLLScore",
            # "KDJ_SCORE": "indicators.kdj_score.KDJScore",  # P1.1.3: 已整合到主KDJ实现
            "VOLUME_SCORE": "indicators.volume_score.VolumeScore",
        }

        return self._batch_register(professional_indicators, "其他专业指标")

    def _batch_register(self, indicators: Dict[str, str], category: str) -> int:
        """批量注册指标"""
        registered_count = 0

        for name, class_path in indicators.items():
            try:
                # 尝试动态导入和验证 - 严格模式,不允许任何回退
                if self._validate_indicator_path(class_path):
                    self._indicators[name] = class_path
                    self._registration_log.append(f"✅ {category}: {name}")
                    logger.debug(f"✅ 成功注册 {category}: {name}")
                    registered_count += 1
                else:
                    # 严格模式:导入失败直接记录为失败,不使用任何回退
                    self._failed_indicators.append(f"{name}: 指标路径验证失败 - {class_path}")
                    logger.error(f"❌ 指标验证失败 {category}: {name} - 路径: {class_path}")
                    # 不增加registered_count,确保失败的指标不被注册

            except Exception as e:
                self._failed_indicators.append(f"{name}: {e}")
                logger.error(f"❌ 注册失败 {category}: {name} - {e}")

        logger.info(f"  {category}: 注册 {registered_count}/{len(indicators)} 个指标")
        return registered_count

    def _validate_indicator_path(self, class_path: str) -> bool:
        """验证指标类路径是否存在"""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name)
            return True
        except (ImportError, AttributeError, ValueError):
            return False

    def get_indicator(self, name: str):
        """获取指标实例"""
        indicator_path = self._indicators.get(name)
        if not indicator_path:
            return None

        try:
            # 如果已经是实例,直接返回
            if hasattr(indicator_path, "calculate"):
                return indicator_path

            # 如果是字符串路径,创建实例
            if isinstance(indicator_path, str):
                module_path, class_name = indicator_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                indicator_class = getattr(module, class_name)

                # 尝试不同的实例化方式
                try:
                    # 首先尝试无参数实例化
                    return indicator_class()
                except Exception as e1:
                    try:
                        # 尝试使用默认参数实例化
                        return indicator_class(period=20)  # TODO: 将魔法数字提取到配置中
                    except Exception as e2:
                        try:
                            # 尝试使用更多默认参数
                            return indicator_class(
                                n=9, m1=3, m2=3
                            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        except Exception as e3:
                            # 严格模式:如果都失败,返回None,不使用任何回退
                            logger.error(
                                f"指标 {name} 实例化完全失败: 无参数失败, period=20失败, n=9,m1=3,m2=3失败"
                            )  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            return None

            return indicator_path
        except Exception as e:
            logger.error(f"创建指标 {name} 实例失败: {e}")
            # 严格模式:不使用任何回退,直接返回None
            return None

    def create_indicator(self, name: str, **kwargs):
        """
        创建指标实例

        Args:
            name: 指标名称
            **kwargs: 指标参数

        Returns:
            指标实例

        Raises:
            ValueError: 如果指标不存在
        """
        indicator_path = self._indicators.get(name)
        if indicator_path is None:
            raise ValueError(f"未注册的指标: {name}")

        # 严格模式:不允许real.前缀的指标
        if indicator_path.startswith("real."):
            raise ValueError(f"不允许使用real前缀的指标: {name}")

        # 动态导入并创建指标实例 - 严格模式,失败即抛出异常
        try:
            module_path, class_name = indicator_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name)
            return indicator_class(**kwargs)
        except Exception as e:
            logger.error(f"创建指标 {name} 实例失败: {e}")
            # 严格模式:失败直接抛出异常,不使用任何回退
            raise ValueError(f"指标 {name} 创建失败: {e}")

    # _create_real_indicator方法已移除 - 严格模式不允许任何real回退

    @property
    def indicators(self) -> Dict[str, Any]:
        """获取所有指标(属性访问)"""
        return self._indicators.copy()

    def get_all_indicators(self) -> Dict[str, Any]:
        """获取所有指标类对象"""
        indicator_classes = {}
        for name, class_path in self._indicators.items():
            try:
                if isinstance(class_path, str):
                    module_path, class_name = class_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name)
                    if issubclass(indicator_class, BaseIndicator):
                        indicator_classes[name] = indicator_class
                else:
                    indicator_classes[name] = class_path
            except Exception as e:
                logger.warning(f"无法加载指标类 {name}: {e}")
                continue
        return indicator_classes

    def get_all_indicator_paths(self) -> Dict[str, str]:
        """获取所有指标路径(原始方法)"""
        return self._indicators.copy()

    def get_indicator_count(self) -> int:
        """获取已注册指标数量"""
        return len(self._indicators)

    def get_failed_indicators(self) -> List[str]:
        """获取注册失败的指标列表"""
        return self._failed_indicators.copy()

    def get_registration_log(self) -> List[str]:
        """获取注册日志"""
        return self._registration_log.copy()

    def register_core_indicators(self):
        """向后兼容方法"""
        return self.register_all_indicators()

    def get_registration_stats(self) -> Dict[str, Any]:
        """获取注册统计信息"""
        total_indicators = len(self._indicators) + len(self._failed_indicators)
        success_count = len(self._indicators)
        failed_count = len(self._failed_indicators)

        return {
            "total_indicators": total_indicators,
            "successful_indicators": success_count,
            "failed_indicators": failed_count,
            "success_rate": success_count / total_indicators if total_indicators > 0 else 0.0,
            "indicator_names": list(self._indicators.keys()),
            "failed_names": self._failed_indicators,
        }


# 创建全局实例
_instance = CompleteIndicatorRegistry()


def get_indicator_registry():
    """获取指标注册表实例"""
    return _instance


def get_indicator(name: str):
    """获取指标实例"""
    return _instance.get_indicator(name)


# 执行注册
def initialize_indicators():
    """初始化所有指标"""
    try:
        registered_count = _instance.register_all_indicators()
        logger.info(f"指标系统初始化完成,共注册 {registered_count} 个指标")
        return registered_count
    except Exception as e:
        logger.error(f"指标注册失败: {e}")
        return 0


# 向后兼容的导出变量
complete_registry = _instance
INDICATOR_REGISTRY = _instance._indicators  # 导出指标注册表字典


def get_all_indicators() -> Dict[str, Any]:
    """获取所有已注册的指标(全局函数)"""
    return _instance.get_all_indicators()


def get_indicator_count() -> int:
    """获取已注册指标数量(全局函数)"""
    return _instance.get_indicator_count()


def get_failed_indicators() -> List[str]:
    """获取注册失败的指标列表(全局函数)"""
    return _instance.get_failed_indicators()


# 自动初始化
if __name__ != "__main__":
    initialize_indicators()
