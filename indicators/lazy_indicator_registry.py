"""
延迟加载的指标注册器
解决循环导入问题，实现分批注册策略
"""

import importlib
from typing import Dict, List, Tuple, Optional
import logging

# 获取logger
logger = logging.getLogger(__name__)

class LazyIndicatorRegistry:
    """延迟加载的指标注册器"""
    
    def __init__(self):
        self._indicator_specs = {}  # 存储指标规格，不立即导入
        self._loaded_indicators = {}  # 存储已加载的指标类
        self._registration_stats = {
            'total_specs': 0,
            'loaded_count': 0,
            'failed_count': 0,
            'failed_indicators': []
        }
    
    def register_indicator_spec(self, module_path: str, class_name: str, 
                               indicator_name: str, description: str, 
                               category: str = "core"):
        """注册指标规格（不立即导入）"""
        self._indicator_specs[indicator_name] = {
            'module_path': module_path,
            'class_name': class_name,
            'description': description,
            'category': category,
            'loaded': False
        }
        self._registration_stats['total_specs'] += 1
        logger.debug(f"注册指标规格: {indicator_name} ({category})")
    
    def load_indicator(self, indicator_name: str):
        """延迟加载单个指标"""
        if indicator_name in self._loaded_indicators:
            return self._loaded_indicators[indicator_name]
        
        if indicator_name not in self._indicator_specs:
            logger.error(f"指标规格不存在: {indicator_name}")
            return None
        
        spec = self._indicator_specs[indicator_name]
        try:
            # 动态导入模块
            module = importlib.import_module(spec['module_path'])
            indicator_class = getattr(module, spec['class_name'], None)
            
            if indicator_class:
                # 验证是否为BaseIndicator子类
                try:
                    from indicators.base_indicator import BaseIndicator
                    if issubclass(indicator_class, BaseIndicator):
                        self._loaded_indicators[indicator_name] = indicator_class
                        spec['loaded'] = True
                        self._registration_stats['loaded_count'] += 1
                        logger.info(f"✅ 成功加载指标: {indicator_name}")
                        return indicator_class
                    else:
                        logger.warning(f"❌ {indicator_name} 不是BaseIndicator子类")
                except Exception as e:
                    logger.warning(f"❌ 验证指标类型失败 {indicator_name}: {e}")
            else:
                logger.error(f"❌ 未找到指标类: {spec['module_path']}.{spec['class_name']}")
                
        except ImportError as e:
            logger.warning(f"❌ 导入失败 {indicator_name}: {e}")
        except Exception as e:
            logger.error(f"❌ 加载指标失败 {indicator_name}: {e}")
        
        # 记录失败
        self._registration_stats['failed_count'] += 1
        self._registration_stats['failed_indicators'].append(indicator_name)
        return None
    
    def load_category(self, category: str) -> List[str]:
        """加载指定类别的所有指标"""
        loaded_indicators = []
        category_specs = {name: spec for name, spec in self._indicator_specs.items() 
                         if spec['category'] == category}
        
        logger.info(f"开始加载 {category} 类别的 {len(category_specs)} 个指标...")
        
        for indicator_name in category_specs:
            if self.load_indicator(indicator_name):
                loaded_indicators.append(indicator_name)
        
        logger.info(f"{category} 类别加载完成: {len(loaded_indicators)}/{len(category_specs)} 个指标成功")
        return loaded_indicators
    
    def load_all_indicators(self) -> Dict[str, any]:
        """加载所有指标"""
        logger.info(f"开始加载所有 {self._registration_stats['total_specs']} 个指标...")
        
        for indicator_name in self._indicator_specs:
            self.load_indicator(indicator_name)
        
        self._print_loading_stats()
        return self._loaded_indicators
    
    def get_loaded_indicators(self) -> Dict[str, any]:
        """获取已加载的指标"""
        return self._loaded_indicators.copy()
    
    def get_indicator_names(self) -> List[str]:
        """获取所有指标名称（包括未加载的）"""
        return list(self._indicator_specs.keys())
    
    def get_loaded_indicator_names(self) -> List[str]:
        """获取已加载的指标名称"""
        return list(self._loaded_indicators.keys())
    
    def _print_loading_stats(self):
        """打印加载统计信息"""
        stats = self._registration_stats
        total = stats['total_specs']
        loaded = stats['loaded_count']
        failed = stats['failed_count']
        
        logger.info(f"指标加载统计:")
        logger.info(f"  📊 总指标数: {total}")
        logger.info(f"  ✅ 成功加载: {loaded} ({loaded/total*100:.1f}%)")
        logger.info(f"  ❌ 加载失败: {failed} ({failed/total*100:.1f}%)")
        
        if stats['failed_indicators']:
            logger.warning(f"  失败指标: {stats['failed_indicators']}")
    
    def register_core_indicators(self):
        """注册核心技术指标"""
        core_indicators = [
            # 基础指标
            ('indicators.ma', 'MA', 'MA', '移动平均线'),
            ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
            ('indicators.wma', 'WMA', 'WMA', '加权移动平均线'),
            ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
            ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
            ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
            ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
            ('indicators.bias', 'BIAS', 'BIAS', '乖离率'),
            ('indicators.cci', 'CCI', 'CCI', '顺势指标'),
            ('indicators.chaikin', 'ChaikinVolatility', 'CHAIKIN', 'Chaikin波动率'),
            ('indicators.cmo', 'CMO', 'CMO', '钱德动量摆动指标'),
            ('indicators.dma', 'DMA', 'DMA', '动态移动平均线'),
            ('indicators.dmi', 'DMI', 'DMI', '趋向指标'),
            ('indicators.emv', 'EMV', 'EMV', '简易波动指标'),
            ('indicators.ichimoku', 'Ichimoku', 'ICHIMOKU', '一目均衡表'),
            ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
            ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
            ('indicators.momentum', 'Momentum', 'MOMENTUM', '动量指标'),
            ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
            ('indicators.obv', 'OBV', 'OBV', '能量潮指标'),
            ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
            ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
            ('indicators.roc', 'ROC', 'ROC', '变动率指标'),
            ('indicators.stochrsi', 'StochasticRSI', 'STOCHRSI', '随机RSI'),
            ('indicators.trix', 'TRIX', 'TRIX', 'TRIX指标'),
            ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
            ('indicators.vol', 'Volume', 'VOL', '成交量指标'),
            ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO', '量比指标'),
            ('indicators.vosc', 'VOSC', 'VOSC', '成交量震荡器'),
            ('indicators.vr', 'VR', 'VR', '成交量比率'),
            ('indicators.vortex', 'Vortex', 'VORTEX', '涡流指标'),
            ('indicators.wr', 'WR', 'WR', '威廉指标'),
            ('indicators.ad', 'AD', 'AD', '累积/派发线'),
        ]
        
        for module_path, class_name, indicator_name, description in core_indicators:
            self.register_indicator_spec(module_path, class_name, indicator_name, description, "core")
        
        logger.info(f"注册了 {len(core_indicators)} 个核心指标规格")

    def register_enhanced_indicators(self):
        """注册增强型指标"""
        enhanced_indicators = [
            ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI', '增强版CCI'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI', '增强版DMI'),
            ('indicators.trend.enhanced_macd', 'EnhancedMACD', 'ENHANCED_MACD_TREND', '增强版MACD(趋势)'),
            ('indicators.trend.enhanced_trix', 'EnhancedTRIX', 'ENHANCED_TRIX', '增强版TRIX'),
            ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ', 'ENHANCED_KDJ_OSC', '增强版KDJ(震荡)'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI', '增强版MFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV', '增强版OBV'),
            ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI', '增强版RSI'),
            ('indicators.enhanced_stochrsi', 'EnhancedStochasticRSI', 'ENHANCED_STOCHRSI', '增强版随机RSI'),
            ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR', '增强版威廉指标'),
        ]

        for module_path, class_name, indicator_name, description in enhanced_indicators:
            self.register_indicator_spec(module_path, class_name, indicator_name, description, "enhanced")

        logger.info(f"注册了 {len(enhanced_indicators)} 个增强指标规格")

    def register_composite_indicators(self):
        """注册复合指标"""
        composite_indicators = [
            ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE', '复合指标'),
            ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA', '统一移动平均线'),
            ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION', '筹码分布'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR', '机构行为'),
            ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX', '个股恐慌指数'),
        ]

        for module_path, class_name, indicator_name, description in composite_indicators:
            self.register_indicator_spec(module_path, class_name, indicator_name, description, "composite")

        logger.info(f"注册了 {len(composite_indicators)} 个复合指标规格")

    def register_pattern_indicators(self):
        """注册形态指标"""
        pattern_indicators = [
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS', 'K线形态'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK', '高级K线形态'),
            ('indicators.pattern.zxm_patterns', 'ZXMPatterns', 'ZXM_PATTERNS', 'ZXM形态'),
        ]

        for module_path, class_name, indicator_name, description in pattern_indicators:
            self.register_indicator_spec(module_path, class_name, indicator_name, description, "pattern")

        logger.info(f"注册了 {len(pattern_indicators)} 个形态指标规格")

    def register_tool_indicators(self):
        """注册特色工具指标"""
        tool_indicators = [
            ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS', '斐波那契工具'),
            ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS', '江恩工具'),
            ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE', '艾略特波浪'),
        ]

        for module_path, class_name, indicator_name, description in tool_indicators:
            self.register_indicator_spec(module_path, class_name, indicator_name, description, "tools")

        logger.info(f"注册了 {len(tool_indicators)} 个工具指标规格")

    def register_all_specs(self):
        """注册所有指标规格"""
        logger.info("开始注册所有指标规格...")
        self.register_core_indicators()
        self.register_enhanced_indicators()
        self.register_composite_indicators()
        self.register_pattern_indicators()
        self.register_tool_indicators()

        total_specs = self._registration_stats['total_specs']
        logger.info(f"所有指标规格注册完成，共 {total_specs} 个指标")

# 创建全局实例
lazy_registry = LazyIndicatorRegistry()

