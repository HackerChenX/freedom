#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
指标注册表模块

用于统一管理所有指标的唯一标识和创建函数
"""

from enum import Enum
import importlib
from typing import Dict, Type, Callable, Any, Optional, List

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger
from indicators.macd_score import MACDScore
from indicators.kdj_score import KDJScore
from indicators.rsi_score import RSIScore
from indicators.boll_score import BOLLScore
from indicators.scoring_framework import IndicatorScoreBase, IndicatorScoreManager
from indicators.volume_score import VolumeScore
# from indicators.trend.trend_strength import TrendStrength
from indicators.unified_ma import UnifiedMA
from indicators.vosc import VOSC
from indicators.volume_ratio import VR
from indicators.vortex import Vortex
from indicators.wma import WMA
from indicators.wr import WR
from enums.indicator_enum import IndicatorEnum
from indicators.macd import MACD
from indicators.rsi import RSI
from indicators.boll import BOLL as BollingerBands
from indicators.vol import VOL as Volume
from indicators.obv import OBV
# 暂时注释掉依赖talib的指标
# from indicators.atr import ATR
try:
    from indicators.atr import ATR
except ImportError:
    ATR = None
from indicators.chaikin import Chaikin as ChaikinVolatility
from indicators.roc import ROC
from indicators.trix import TRIX
from indicators.momentum import Momentum
# 逐步添加已验证可用的指标
try:
    from indicators.bias import BIAS
except ImportError:
    BIAS = None

try:
    from indicators.cci import CCI
except ImportError:
    CCI = None

try:
    from indicators.chaikin import Chaikin
except ImportError:
    Chaikin = None

try:
    from indicators.dmi import DMI
except ImportError:
    DMI = None

try:
    from indicators.emv import EMV
except ImportError:
    EMV = None

try:
    from indicators.ichimoku import Ichimoku
except ImportError:
    Ichimoku = None

try:
    from indicators.trend.enhanced_trix import EnhancedTRIX
except ImportError:
    EnhancedTRIX = None

try:
    from indicators.oscillator.enhanced_kdj import EnhancedKDJ
except ImportError:
    EnhancedKDJ = None

try:
    from indicators.trend.enhanced_macd import EnhancedMACD
except ImportError:
    EnhancedMACD = None

try:
    from indicators.kdj import KDJ
except ImportError:
    KDJ = None

try:
    from indicators.stochrsi import StochasticRSI
except ImportError:
    StochasticRSI = None

try:
    from indicators.cmo import CMO
except ImportError:
    CMO = None

try:
    from indicators.dma import DMA
except ImportError:
    DMA = None

# 导入公式指标
try:
    from indicators.formula_indicators import CrossOver, KDJCondition, MACDCondition, MACondition, GenericCondition
except ImportError:
    CrossOver = None
    KDJCondition = None
    MACDCondition = None
    MACondition = None
    GenericCondition = None

logger = get_logger(__name__)

class IndicatorRegistry:
    """指标注册表，管理所有可用的技术指标"""
    
    _indicators = {}
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(IndicatorRegistry, cls).__new__(cls)
            cls._instance._initialize_default_indicators()
        return cls._instance
    
    def _initialize_default_indicators(self):
        """初始化默认指标"""
        # 这里可以初始化系统默认提供的指标
        # 将在后面使用register_indicator注册
        pass
    
    def register_indicator(self, indicator_class, name=None, description=None, overwrite=False):
        """
        注册一个技术指标
        
        Args:
            indicator_class: 指标类
            name: 指标名称，如果为None则使用类名
            description: 指标描述
            overwrite: 是否覆盖已存在的指标
        """
        if name is None:
            name = indicator_class.__name__
            
        if name in self._indicators and not overwrite:
            logger.warning(f"指标 {name} 已存在，将不会覆盖。如需覆盖请设置overwrite=True")
            return False
            
        self._indicators[name] = {
            'class': indicator_class,
            'description': description or getattr(indicator_class, 'description', ''),
            'name': name
        }
        
        logger.info(f"注册指标: {name}")
        return True
    
    def get_indicator_class(self, name):
        """获取指标类"""
        if name not in self._indicators:
            logger.error(f"指标 {name} 未注册")
            return None
        return self._indicators[name]['class']
    
    def create_indicator(self, name, **kwargs):
        """
        创建指标实例
        
        Args:
            name: 指标名称
            **kwargs: 传递给指标构造函数的参数
            
        Returns:
            指标实例或None
        """
        indicator_class = self.get_indicator_class(name)
        if indicator_class is None:
            return None
            
        try:
            return indicator_class(**kwargs)
        except Exception as e:
            logger.error(f"创建指标 {name} 实例失败: {e}")
            return None
    
    def get_all_indicators(self):
        """获取所有注册的指标信息"""
        return self._indicators.copy()
    
    def get_indicator_names(self):
        """获取所有注册的指标名称"""
        return list(self._indicators.keys())

    def register_standard_indicators(self):
        """注册标准指标集"""
        # 注册基础指标
        self.register_indicator(MACD, name="MACD", description="移动平均线收敛散度指标")
        self.register_indicator(RSI, name="RSI", description="相对强弱指数")
        self.register_indicator(BollingerBands, name="BOLL", description="布林带")
        self.register_indicator(Volume, name="Volume", description="成交量指标")

        # 注册已修复的技术指标（逐步集成策略）
        # 第一批：已验证有get_pattern_info方法的增强指标
        if EnhancedKDJ:
            try:
                # 注册多个名称以确保兼容性
                self.register_indicator(EnhancedKDJ, name="EnhancedKDJ", description="增强版KDJ指标")
                self.register_indicator(EnhancedKDJ, name="ENHANCEDKDJ", description="增强版KDJ指标", overwrite=True)
                logger.info("成功注册EnhancedKDJ指标")
            except Exception as e:
                logger.warning(f"注册EnhancedKDJ指标失败: {e}")

        if EnhancedMACD:
            try:
                # 注册多个名称以确保兼容性
                self.register_indicator(EnhancedMACD, name="EnhancedMACD", description="增强版MACD指标")
                self.register_indicator(EnhancedMACD, name="ENHANCEDMACD", description="增强版MACD指标", overwrite=True)
                logger.info("成功注册EnhancedMACD指标")
            except Exception as e:
                logger.warning(f"注册EnhancedMACD指标失败: {e}")

        # 第二批：基础技术指标（需要验证get_pattern_info方法）
        if BIAS:
            try:
                self.register_indicator(BIAS, name="BIAS", description="均线多空指标")
                logger.info("成功注册BIAS指标")
            except Exception as e:
                logger.warning(f"注册BIAS指标失败: {e}")

        if CCI:
            try:
                self.register_indicator(CCI, name="CCI", description="顺势指标")
                logger.info("成功注册CCI指标")
            except Exception as e:
                logger.warning(f"注册CCI指标失败: {e}")

        if Chaikin:
            try:
                self.register_indicator(Chaikin, name="Chaikin", description="Chaikin振荡器")
                logger.info("成功注册Chaikin指标")
            except Exception as e:
                logger.warning(f"注册Chaikin指标失败: {e}")

        if DMI:
            try:
                self.register_indicator(DMI, name="DMI", description="动向指标")
                logger.info("成功注册DMI指标")
            except Exception as e:
                logger.warning(f"注册DMI指标失败: {e}")

        if EMV:
            try:
                self.register_indicator(EMV, name="EMV", description="简易波动指标")
                logger.info("成功注册EMV指标")
            except Exception as e:
                logger.warning(f"注册EMV指标失败: {e}")

        if Ichimoku:
            try:
                self.register_indicator(Ichimoku, name="Ichimoku", description="一目均衡表")
                logger.info("成功注册Ichimoku指标")
            except Exception as e:
                logger.warning(f"注册Ichimoku指标失败: {e}")

        if EnhancedTRIX:
            try:
                self.register_indicator(EnhancedTRIX, name="EnhancedTRIX", description="增强版TRIX指标")
                logger.info("成功注册EnhancedTRIX指标")
            except Exception as e:
                logger.warning(f"注册EnhancedTRIX指标失败: {e}")

        if StochasticRSI:
            try:
                self.register_indicator(StochasticRSI, name="StochasticRSI", description="随机RSI指标")
                logger.info("成功注册StochasticRSI指标")
            except Exception as e:
                logger.warning(f"注册StochasticRSI指标失败: {e}")

        # 尝试注册其他可能可用的指标
        if CMO:
            try:
                self.register_indicator(CMO, name="CMO", description="钱德动量摆动指标")
                logger.info("成功注册CMO指标")
            except Exception as e:
                logger.warning(f"注册CMO指标失败: {e}")

        if DMA:
            try:
                self.register_indicator(DMA, name="DMA", description="平行线差指标")
                logger.info("成功注册DMA指标")
            except Exception as e:
                logger.warning(f"注册DMA指标失败: {e}")

        # 注册公式指标
        if CrossOver:
            try:
                self.register_indicator(CrossOver, name="CROSS_OVER", description="交叉条件指标")
                logger.info("成功注册CROSS_OVER指标")
            except Exception as e:
                logger.warning(f"注册CROSS_OVER指标失败: {e}")

        if KDJCondition:
            try:
                self.register_indicator(KDJCondition, name="KDJ_CONDITION", description="KDJ条件指标")
                logger.info("成功注册KDJ_CONDITION指标")
            except Exception as e:
                logger.warning(f"注册KDJ_CONDITION指标失败: {e}")

        if MACDCondition:
            try:
                self.register_indicator(MACDCondition, name="MACD_CONDITION", description="MACD条件指标")
                logger.info("成功注册MACD_CONDITION指标")
            except Exception as e:
                logger.warning(f"注册MACD_CONDITION指标失败: {e}")

        if MACondition:
            try:
                self.register_indicator(MACondition, name="MA_CONDITION", description="MA条件指标")
                logger.info("成功注册MA_CONDITION指标")
            except Exception as e:
                logger.warning(f"注册MA_CONDITION指标失败: {e}")

        if GenericCondition:
            try:
                self.register_indicator(GenericCondition, name="GENERIC_CONDITION", description="通用条件指标")
                logger.info("成功注册GENERIC_CONDITION指标")
            except Exception as e:
                logger.warning(f"注册GENERIC_CONDITION指标失败: {e}")

        # 注册基础KDJ指标
        if KDJ:
            try:
                self.register_indicator(KDJ, name="KDJ", description="KDJ随机指标")
                logger.info("成功注册KDJ指标")
            except Exception as e:
                logger.warning(f"注册KDJ指标失败: {e}")

        # 扩展注册更多可用指标
        self._register_missing_indicators()

        # 批量注册核心指标
        self._batch_register_core_indicators()

        # 批量注册ZXM体系指标
        self._batch_register_zxm_indicators()

        # 修复并注册问题指标
        self._fix_and_register_problematic_indicators()

        # 注册最后的缺失指标
        self._register_final_missing_indicators()

        # 注册新发现的指标
        self._register_discovered_indicators()

    def _register_additional_indicators(self):
        """注册所有已验证通过的技术指标"""
        registered_count = 0

        # 注册已导入的指标
        additional_indicators = [
            (ROC, "ROC", "变动率指标"),
            (TRIX, "TRIX", "TRIX指标"),
            (Momentum, "MOMENTUM", "动量指标"),
            (OBV, "OBV", "能量潮指标"),
            (ChaikinVolatility, "CHAIKIN_VOLATILITY", "Chaikin波动率"),
            (UnifiedMA, "UNIFIED_MA", "统一移动平均线"),
            (VOSC, "VOSC", "成交量震荡器"),
            (VR, "VR", "成交量比率"),
            (Vortex, "VORTEX", "涡流指标"),
            (WMA, "WMA", "加权移动平均线"),
            (WR, "WR", "威廉指标"),
        ]

        for indicator_class, name, description in additional_indicators:
            try:
                if indicator_class and hasattr(indicator_class, '__bases__'):
                    from indicators.base_indicator import BaseIndicator
                    if issubclass(indicator_class, BaseIndicator):
                        if name not in self._indicators:
                            self.register_indicator(indicator_class, name=name, description=description)
                            registered_count += 1
                            logger.info(f"成功注册额外指标: {name}")
                    else:
                        logger.debug(f"跳过非BaseIndicator子类: {indicator_class.__name__}")
            except Exception as e:
                logger.warning(f"注册指标失败 {name}: {e}")

        # 尝试动态导入核心指标
        core_indicators = [
            ('indicators.ma', 'MA', 'MA', '移动平均线'),
            ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
            ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
            ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
            ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
            ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
            ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
            ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
            ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
            ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
            ('indicators.ad', 'AD', 'AD', '累积/派发线'),
            ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
            ('indicators.stochrsi', 'StochasticRSI', 'STOCHRSI', '随机RSI'),
            ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
        ]

        for module_path, class_name, indicator_name, description in core_indicators:
            try:
                module = importlib.import_module(module_path)
                indicator_class = getattr(module, class_name, None)

                if indicator_class and hasattr(indicator_class, '__bases__'):
                    from indicators.base_indicator import BaseIndicator
                    if issubclass(indicator_class, BaseIndicator):
                        if indicator_name not in self._indicators:
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"成功注册核心指标: {indicator_name}")

            except ImportError:
                logger.debug(f"导入指标失败: {module_path}.{class_name}")
            except Exception as e:
                logger.warning(f"注册指标失败 {indicator_name}: {e}")

        logger.info(f"额外注册了 {registered_count} 个技术指标")

        # 输出注册统计
        total_indicators = len(self._indicators)
        logger.info(f"指标注册完成，共注册 {total_indicators} 个技术指标")

    def _register_safe_indicators(self):
        """最保守的指标注册方法"""
        registered_count = 0

        # 只注册已经成功导入的指标
        try:
            safe_indicators = [
                (ROC, "ROC", "变动率指标"),
                (TRIX, "TRIX", "TRIX指标"),
                (Momentum, "MOMENTUM", "动量指标"),
                (OBV, "OBV", "能量潮指标"),
                (ChaikinVolatility, "CHAIKIN_VOLATILITY", "Chaikin波动率"),
                (UnifiedMA, "UNIFIED_MA", "统一移动平均线"),
                (VOSC, "VOSC", "成交量震荡器"),
                (VR, "VR", "成交量比率"),
                (Vortex, "VORTEX", "涡流指标"),
                (WMA, "WMA", "加权移动平均线"),
                (WR, "WR", "威廉指标"),
            ]

            for indicator_class, name, description in safe_indicators:
                try:
                    if indicator_class and name not in self._indicators:
                        self.register_indicator(indicator_class, name=name, description=description)
                        registered_count += 1
                        logger.info(f"✅ 安全注册指标: {name}")
                except Exception as e:
                    logger.debug(f"跳过指标 {name}: {e}")
        except Exception as e:
            logger.warning(f"安全注册过程出错: {e}")

        logger.info(f"安全注册完成，新增 {registered_count} 个指标")

    def initialize_pattern_registry(self):
        """初始化形态注册表，将所有指标的形态导入到形态注册表中"""
        from indicators.pattern_registry import PatternRegistry
        
        # 创建所有指标的实例
        indicators = []
        for name, info in self._indicators.items():
            try:
                indicator = self.create_indicator(name)
                if indicator:
                    indicators.append(indicator)
            except Exception as e:
                logger.error(f"创建指标 {name} 实例失败: {e}")
                
        # 将所有指标的形态注册到PatternRegistry
        PatternRegistry.auto_register_from_indicators(indicators)
        
        # 记录已注册的形态
        pattern_count = len(PatternRegistry.get_all_pattern_ids())
        logger.info(f"从指标中注册了 {pattern_count} 个形态")

    def _register_missing_indicators(self):
        """注册缺失的指标"""
        registered_count = 0

        # 第一批：核心技术指标
        core_indicators = [
            ('indicators.ma', 'MA', 'MA', '移动平均线'),
            ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
            ('indicators.wma', 'WMA', 'WMA', '加权移动平均线'),
            ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
            ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
            ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
            ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
            ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
            ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
            ('indicators.momentum', 'Momentum', 'MOMENTUM', '动量指标'),
            ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
            ('indicators.obv', 'OBV', 'OBV', '能量潮指标'),
            ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
            ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
            ('indicators.roc', 'ROC', 'ROC', '变动率指标'),
            ('indicators.trix', 'TRIX', 'TRIX', 'TRIX指标'),
            ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
            ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO', '量比指标'),
            ('indicators.vosc', 'VOSC', 'VOSC', '成交量震荡器'),
            ('indicators.vr', 'VR', 'VR', '成交量比率'),
            ('indicators.vortex', 'Vortex', 'VORTEX', '涡流指标'),
            ('indicators.wr', 'WR', 'WR', '威廉指标'),
            ('indicators.ad', 'AD', 'AD', '累积/派发线'),
        ]

        for module_path, class_name, indicator_name, description in core_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 注册核心指标: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        # 第二批：增强型指标
        enhanced_indicators = [
            ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI', '增强版CCI'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI', '增强版DMI'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI', '增强版MFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV', '增强版OBV'),
            ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI', '增强版RSI'),
            ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR', '增强版威廉指标'),
        ]

        for module_path, class_name, indicator_name, description in enhanced_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 注册增强指标: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        # 第三批：复合指标
        composite_indicators = [
            ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE', '复合指标'),
            ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA', '统一移动平均线'),
            ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION', '筹码分布'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR', '机构行为'),
            ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX', '个股恐慌指数'),
        ]

        for module_path, class_name, indicator_name, description in composite_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 注册复合指标: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        logger.info(f"缺失指标注册完成，新增 {registered_count} 个指标")

    def _batch_register_core_indicators(self):
        """批量注册核心指标"""
        registered_count = 0

        # 核心指标列表
        core_indicators = [
            ('indicators.ad', 'AD', 'AD', '累积/派发线'),
            ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
            ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
            ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
            ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
            ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
            ('indicators.ma', 'MA', 'MA', '移动平均线'),
            ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
            ('indicators.momentum', 'Momentum', 'MOMENTUM', '动量指标'),
            ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
            ('indicators.obv', 'OBV', 'OBV', '能量潮指标'),
            ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
            ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
            ('indicators.roc', 'ROC', 'ROC', '变动率指标'),
            ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
            ('indicators.trix', 'TRIX', 'TRIX', 'TRIX指标'),
            ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
            ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO', '量比指标'),
            ('indicators.vosc', 'VOSC', 'VOSC', '成交量震荡器'),
            ('indicators.vr', 'VR', 'VR', '成交量比率'),
            ('indicators.vortex', 'Vortex', 'VORTEX', '涡流指标'),
            ('indicators.wma', 'WMA', 'WMA', '加权移动平均线'),
            ('indicators.wr', 'WR', 'WR', '威廉指标'),
        ]

        logger.info(f"开始批量注册 {len(core_indicators)} 个核心指标...")

        for module_path, class_name, indicator_name, description in core_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 批量注册核心指标: {indicator_name}")
                        else:
                            logger.debug(f"跳过非BaseIndicator: {class_name}")
                    else:
                        logger.debug(f"未找到类: {class_name}")
                else:
                    logger.debug(f"指标已存在: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        logger.info(f"核心指标批量注册完成，新增 {registered_count} 个指标")

    def _batch_register_zxm_indicators(self):
        """批量注册ZXM体系指标"""
        registered_count = 0

        # ZXM体系指标列表 (25个)
        zxm_indicators = [
            # ZXM Trend (9个)
            ('indicators.zxm.trend_indicators', 'ZXMDailyTrendUp', 'ZXM_DAILY_TREND_UP', 'ZXM日趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyTrendUp', 'ZXM_WEEKLY_TREND_UP', 'ZXM周趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyKDJTrendUp', 'ZXM_MONTHLY_KDJ_TREND_UP', 'ZXM月KDJ趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDOrDEATrendUp', 'ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP', 'ZXM周KDJ D或DEA趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDTrendUp', 'ZXM_WEEKLY_KDJ_D_TREND_UP', 'ZXM周KDJ D趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyMACD', 'ZXM_MONTHLY_MACD', 'ZXM月MACD'),
            ('indicators.zxm.trend_indicators', 'TrendDetector', 'ZXM_TREND_DETECTOR', 'ZXM趋势检测器'),
            ('indicators.zxm.trend_indicators', 'TrendDuration', 'ZXM_TREND_DURATION', 'ZXM趋势持续时间'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyMACD', 'ZXM_WEEKLY_MACD', 'ZXM周MACD'),
            # ZXM Buy Points (5个)
            ('indicators.zxm.buy_point_indicators', 'ZXMDailyMACD', 'ZXM_DAILY_MACD', 'ZXM日MACD买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMTurnover', 'ZXM_TURNOVER', 'ZXM换手率买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMVolumeShrink', 'ZXM_VOLUME_SHRINK', 'ZXM缩量买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMMACallback', 'ZXM_MA_CALLBACK', 'ZXM均线回踩买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMBSAbsorb', 'ZXM_BS_ABSORB', 'ZXM吸筹买点'),
            # ZXM Elasticity (4个)
            ('indicators.zxm.elasticity_indicators', 'AmplitudeElasticity', 'ZXM_AMPLITUDE_ELASTICITY', 'ZXM振幅弹性'),
            ('indicators.zxm.elasticity_indicators', 'ZXMRiseElasticity', 'ZXM_RISE_ELASTICITY', 'ZXM涨幅弹性'),
            ('indicators.zxm.elasticity_indicators', 'Elasticity', 'ZXM_ELASTICITY', 'ZXM弹性'),
            ('indicators.zxm.elasticity_indicators', 'BounceDetector', 'ZXM_BOUNCE_DETECTOR', 'ZXM反弹检测器'),
            # ZXM Score (3个)
            ('indicators.zxm.score_indicators', 'ZXMElasticityScore', 'ZXM_ELASTICITY_SCORE', 'ZXM弹性评分'),
            ('indicators.zxm.score_indicators', 'ZXMBuyPointScore', 'ZXM_BUYPOINT_SCORE', 'ZXM买点评分'),
            ('indicators.zxm.score_indicators', 'StockScoreCalculator', 'ZXM_STOCK_SCORE', 'ZXM股票评分'),
            # ZXM其他 (4个)
            ('indicators.zxm.market_breadth', 'ZXMMarketBreadth', 'ZXM_MARKET_BREADTH', 'ZXM市场宽度'),
            ('indicators.zxm.selection_model', 'SelectionModel', 'ZXM_SELECTION_MODEL', 'ZXM选股模型'),
            ('indicators.zxm.diagnostics', 'ZXMDiagnostics', 'ZXM_DIAGNOSTICS', 'ZXM诊断'),
            ('indicators.zxm.buy_point_indicators', 'BuyPointDetector', 'ZXM_BUYPOINT_DETECTOR', 'ZXM买点检测器'),
        ]

        logger.info(f"开始批量注册 {len(zxm_indicators)} 个ZXM体系指标...")

        for module_path, class_name, indicator_name, description in zxm_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 批量注册ZXM指标: {indicator_name}")
                        else:
                            logger.debug(f"跳过非BaseIndicator: {class_name}")
                    else:
                        logger.debug(f"未找到类: {class_name}")
                else:
                    logger.debug(f"指标已存在: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        logger.info(f"ZXM指标批量注册完成，新增 {registered_count} 个指标")

    def _fix_and_register_problematic_indicators(self):
        """修复并注册有问题的指标"""
        registered_count = 0

        # 尝试修复的指标列表
        problematic_indicators = [
            # 修复CHAIKIN - 使用正确的类名
            ('indicators.chaikin', 'Chaikin', 'CHAIKIN', 'Chaikin波动率'),
            # 修复VOL - 使用正确的类名
            ('indicators.vol', 'VOL', 'VOL', '成交量指标'),
            # 尝试修复BOLL - 使用不同路径
            ('indicators.boll', 'BOLL', 'BOLL', '布林带'),
            ('indicators.bollinger', 'BollingerBands', 'BOLL', '布林带'),
            # 尝试修复StochRSI
            ('indicators.stochrsi', 'StochRSI', 'STOCHRSI', '随机RSI'),
        ]

        logger.info(f"开始修复并注册 {len(set([item[2] for item in problematic_indicators]))} 个问题指标...")

        registered_indicators = set()

        for module_path, class_name, indicator_name, description in problematic_indicators:
            try:
                # 避免重复注册
                if indicator_name in registered_indicators or indicator_name in self._indicators:
                    continue

                module = importlib.import_module(module_path)
                indicator_class = getattr(module, class_name, None)

                if indicator_class:
                    from indicators.base_indicator import BaseIndicator
                    if issubclass(indicator_class, BaseIndicator):
                        self.register_indicator(indicator_class, name=indicator_name, description=description)
                        registered_count += 1
                        registered_indicators.add(indicator_name)
                        logger.info(f"✅ 修复并注册指标: {indicator_name}")
                    else:
                        logger.debug(f"跳过非BaseIndicator: {class_name}")
                else:
                    logger.debug(f"未找到类: {class_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"修复失败 {indicator_name}: {e}")

        logger.info(f"问题指标修复完成，新增 {registered_count} 个指标")

    def _register_final_missing_indicators(self):
        """注册最后的4个缺失指标"""
        registered_count = 0

        # 最后4个需要修复的指标
        final_indicators = [
            ('indicators.boll', 'BOLL', 'BOLL', '布林带指标'),
            ('indicators.dmi', 'DMI', 'DMI', '趋向指标'),
            ('indicators.stochrsi', 'STOCHRSI', 'STOCHRSI', '随机RSI指标'),
            ('indicators.pattern.zxm_patterns', 'ZXMPatternIndicator', 'ZXM_PATTERNS', 'ZXM形态指标'),
        ]

        logger.info(f"开始注册最后 {len(final_indicators)} 个缺失指标...")

        for module_path, class_name, indicator_name, description in final_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 注册最后缺失指标: {indicator_name}")
                        else:
                            logger.debug(f"跳过非BaseIndicator: {class_name}")
                    else:
                        logger.debug(f"未找到类: {class_name}")
                else:
                    logger.debug(f"指标已存在: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        logger.info(f"最后缺失指标注册完成，新增 {registered_count} 个指标")

    def _register_discovered_indicators(self):
        """注册新发现的指标"""
        registered_count = 0

        # 新发现的可用指标
        discovered_indicators = [
            ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI', '增强型相对强弱指标'),
            ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR', '增强型威廉指标'),
        ]

        logger.info(f"开始注册新发现的 {len(discovered_indicators)} 个指标...")

        for module_path, class_name, indicator_name, description in discovered_indicators:
            try:
                if indicator_name not in self._indicators:
                    module = importlib.import_module(module_path)
                    indicator_class = getattr(module, class_name, None)

                    if indicator_class:
                        from indicators.base_indicator import BaseIndicator
                        if issubclass(indicator_class, BaseIndicator):
                            self.register_indicator(indicator_class, name=indicator_name, description=description)
                            registered_count += 1
                            logger.info(f"✅ 注册新发现指标: {indicator_name}")
                        else:
                            logger.debug(f"跳过非BaseIndicator: {class_name}")
                    else:
                        logger.debug(f"未找到类: {class_name}")
                else:
                    logger.debug(f"指标已存在: {indicator_name}")

            except ImportError:
                logger.debug(f"导入失败: {module_path}")
            except Exception as e:
                logger.debug(f"注册失败 {indicator_name}: {e}")

        logger.info(f"新发现指标注册完成，新增 {registered_count} 个指标")

# 延迟初始化以避免循环导入
indicator_registry = None

def get_registry():
    """获取指标注册表实例（延迟初始化）"""
    global indicator_registry
    if indicator_registry is None:
        indicator_registry = IndicatorRegistry()
        indicator_registry.register_standard_indicators()
    return indicator_registry

def get_indicator(name, **kwargs):
    """
    获取指标实例的便捷函数

    Args:
        name: 指标名称
        **kwargs: 传递给指标构造函数的参数

    Returns:
        指标实例或None
    """
    return get_registry().create_indicator(name, **kwargs)

def register_indicator(indicator_class, name=None, description=None, overwrite=False):
    """
    注册指标的便捷函数

    Args:
        indicator_class: 指标类
        name: 指标名称
        description: 指标描述
        overwrite: 是否覆盖已存在的指标

    Returns:
        bool: 是否成功注册
    """
    return get_registry().register_indicator(indicator_class, name, description, overwrite)

def get_all_indicator_names():
    """获取所有注册的指标名称"""
    return get_registry().get_indicator_names()

def create_indicator_score_manager(indicators=None, weights=None, market_environment=None):
    """
    创建指标评分管理器
    
    Args:
        indicators: 指标名称列表，如果为None则使用所有注册的指标
        weights: 指标权重字典，键为指标名称，值为权重
        market_environment: 市场环境
        
    Returns:
        IndicatorScoreManager: 指标评分管理器
    """
    from indicators.scoring_framework import IndicatorScoreManager, MarketEnvironment
    
    # 创建管理器
    manager = IndicatorScoreManager()
    
    # 设置市场环境
    if market_environment:
        manager.set_market_environment(market_environment)
        
    # 如果没有指定指标，使用所有注册的指标
    if indicators is None:
        indicators = get_all_indicator_names()
        
    # 如果没有指定权重，使用默认权重1.0
    if weights is None:
        weights = {name: 1.0 for name in indicators}
        
    # 添加指标
    for name in indicators:
        indicator = get_indicator(name)
        if indicator:
            weight = weights.get(name, 1.0)
            manager.add_indicator(indicator, weight)
            
    return manager

# 定义指标常量枚举
class IndicatorEnum_DEPRECATED(str, Enum):
    """指标常量枚举"""
    
    # ZXM系列指标
    ZXM_ABSORB = "ZXM_ABSORB"                 # ZXM吸筹
    ZXM_TURNOVER = "ZXM_TURNOVER"             # ZXM换手买点
    ZXM_DAILY_MACD = "ZXM_DAILY_MACD"         # ZXM日MACD买点
    ZXM_MA_CALLBACK = "ZXM_MA_CALLBACK"       # ZXM回踩均线买点
    ZXM_RISE_ELASTICITY = "ZXM_RISE_ELASTICITY"  # ZXM涨幅弹性
    ZXM_AMPLITUDE_ELASTICITY = "ZXM_AMPLITUDE_ELASTICITY"  # ZXM振幅弹性
    ZXM_ELASTICITY_SCORE = "ZXM_ELASTICITY_SCORE"  # ZXM弹性满足
    ZXM_BUYPOINT_SCORE = "ZXM_BUYPOINT_SCORE"  # ZXM买点满足
    ZXM_DAILY_TREND_UP = "ZXM_DAILY_TREND_UP"  # ZXM趋势满足
    
    # 经典技术指标
    MACD = "MACD"                             # MACD指标
    KDJ = "KDJ"                               # KDJ指标
    MA = "MA"                                 # 移动平均线
    BOLL = "BOLL"                             # 布林带
    VR = "VR"                                 # 成交量比率
    OBV = "OBV"                               # 能量潮
    RSI = "RSI"                               # 相对强弱指数
    DIVERGENCE = "DIVERGENCE"                 # 背离
    
    # 新增增强指标
    UNIFIED_MA = "UNIFIED_MA"                 # 统一移动平均线
    ENHANCED_MACD = "ENHANCED_MACD"           # 增强版MACD
    ENHANCED_RSI = "ENHANCED_RSI"             # 增强版RSI

# 定义标准指标参数映射
STANDARD_PARAMETER_MAPPING = {
    IndicatorEnum_DEPRECATED.ZXM_TURNOVER: {
        "turnover_threshold": "threshold"
    },
    IndicatorEnum_DEPRECATED.ZXM_DAILY_MACD: {
        "macd_threshold": "threshold"
    },
    IndicatorEnum_DEPRECATED.ZXM_MA_CALLBACK: {
        "ma_periods": "periods"
    },
    IndicatorEnum_DEPRECATED.ZXM_RISE_ELASTICITY: {
        "threshold": "rise_threshold"
    },
    IndicatorEnum_DEPRECATED.ZXM_AMPLITUDE_ELASTICITY: {
        "threshold": "amplitude_threshold"
    },
    IndicatorEnum_DEPRECATED.ZXM_ABSORB: {
        "threshold": "absorb_threshold"
    },
    IndicatorEnum_DEPRECATED.ZXM_ELASTICITY_SCORE: {
        "threshold": "threshold"
    },
    IndicatorEnum_DEPRECATED.ZXM_BUYPOINT_SCORE: {
        "threshold": "threshold"
    },
    # 新增增强指标参数映射
    IndicatorEnum_DEPRECATED.UNIFIED_MA: {
        "periods": "periods",
        "ma_type": "ma_type",
        "price_column": "price_col"
    },
    IndicatorEnum_DEPRECATED.ENHANCED_MACD: {
        "fast_period": "fast_period",
        "slow_period": "slow_period",
        "signal_period": "signal_period",
        "price_column": "price_col",
        "use_dual_macd": "use_secondary_macd"
    },
    IndicatorEnum_DEPRECATED.ENHANCED_RSI: {
        "periods": "periods",
        "price_column": "price_col",
        "use_multi_period": "use_multi_period"
    }
}
