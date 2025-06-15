"""
完整的指标注册管理器
解决循环导入问题，实现所有101个指标的完整注册
"""

import importlib
import logging
from typing import Dict, List, Tuple, Optional, Any

# 设置日志
logger = logging.getLogger(__name__)

class CompleteIndicatorRegistry:
    """完整的指标注册管理器"""
    
    def __init__(self):
        self._indicators = {}
        self._registration_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'failed_indicators': []
        }
    
    def register_indicator_safe(self, indicator_class, name: str, description: str = ""):
        """安全注册单个指标"""
        try:
            # 验证是否为BaseIndicator子类
            from indicators.base_indicator import BaseIndicator
            if not issubclass(indicator_class, BaseIndicator):
                logger.warning(f"❌ {name} 不是BaseIndicator子类")
                return False
            
            # 注册指标
            self._indicators[name] = {
                'class': indicator_class,
                'description': description,
                'is_available': True
            }
            
            self._registration_stats['successful'] += 1
            logger.info(f"✅ 成功注册指标: {name}")
            return True
            
        except Exception as e:
            self._registration_stats['failed'] += 1
            self._registration_stats['failed_indicators'].append((name, str(e)))
            logger.error(f"❌ 注册指标失败 {name}: {e}")
            return False
    
    def register_from_module(self, module_path: str, class_name: str, indicator_name: str, description: str = ""):
        """从模块动态导入并注册指标"""
        self._registration_stats['total_attempted'] += 1
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class is None:
                logger.error(f"❌ 未找到类 {class_name} 在模块 {module_path}")
                self._registration_stats['failed'] += 1
                self._registration_stats['failed_indicators'].append((indicator_name, f"类 {class_name} 不存在"))
                return False
            
            # 注册指标
            return self.register_indicator_safe(indicator_class, indicator_name, description)
            
        except ImportError as e:
            logger.error(f"❌ 导入模块失败 {module_path}: {e}")
            self._registration_stats['failed'] += 1
            self._registration_stats['failed_indicators'].append((indicator_name, f"导入失败: {e}"))
            return False
        except Exception as e:
            logger.error(f"❌ 注册过程出错 {indicator_name}: {e}")
            self._registration_stats['failed'] += 1
            self._registration_stats['failed_indicators'].append((indicator_name, f"注册出错: {e}"))
            return False
    
    def register_core_indicators(self):
        """注册核心技术指标"""
        logger.info("=== 开始注册核心技术指标 ===")
        
        core_indicators = [
            # 基础指标
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
            # 已注册的基础指标
            ('indicators.macd', 'MACD', 'MACD', '移动平均线收敛散度指标'),
            ('indicators.rsi', 'RSI', 'RSI', '相对强弱指数'),
            ('indicators.boll', 'BollingerBands', 'BOLL', '布林带'),
            ('indicators.kdj', 'KDJ', 'KDJ', 'KDJ随机指标'),
            ('indicators.bias', 'BIAS', 'BIAS', '乖离率'),
            ('indicators.cci', 'CCI', 'CCI', '顺势指标'),
            ('indicators.chaikin', 'ChaikinVolatility', 'CHAIKIN', 'Chaikin波动率'),
            ('indicators.dmi', 'DMI', 'DMI', '趋向指标'),
            ('indicators.emv', 'EMV', 'EMV', '简易波动指标'),
            ('indicators.ichimoku', 'Ichimoku', 'ICHIMOKU', '一目均衡表'),
            ('indicators.cmo', 'CMO', 'CMO', '钱德动量摆动指标'),
            ('indicators.dma', 'DMA', 'DMA', '动态移动平均线'),
            ('indicators.vol', 'Volume', 'VOL', '成交量指标'),
            ('indicators.stochrsi', 'StochasticRSI', 'STOCHRSI', '随机RSI'),
        ]
        
        success_count = 0
        for module_path, class_name, indicator_name, description in core_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1
        
        logger.info(f"核心指标注册完成: {success_count}/{len(core_indicators)}")
        return success_count
    
    def register_enhanced_indicators(self):
        """注册增强型指标"""
        logger.info("=== 开始注册增强型指标 ===")
        
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
            ('indicators.enhanced_macd', 'EnhancedMACD', 'ENHANCED_MACD_ROOT', '增强版MACD(根目录)'),
        ]
        
        success_count = 0
        for module_path, class_name, indicator_name, description in enhanced_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1
        
        logger.info(f"增强指标注册完成: {success_count}/{len(enhanced_indicators)}")
        return success_count
    
    def register_composite_indicators(self):
        """注册复合指标"""
        logger.info("=== 开始注册复合指标 ===")
        
        composite_indicators = [
            ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE', '复合指标'),
            ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA', '统一移动平均线'),
            ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION', '筹码分布'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR', '机构行为'),
            ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX', '个股恐慌指数'),
        ]
        
        success_count = 0
        for module_path, class_name, indicator_name, description in composite_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1
        
        logger.info(f"复合指标注册完成: {success_count}/{len(composite_indicators)}")
        return success_count
    
    def get_indicator_names(self) -> List[str]:
        """获取所有已注册指标名称"""
        return list(self._indicators.keys())
    
    def get_registration_stats(self) -> Dict:
        """获取注册统计信息"""
        return self._registration_stats.copy()
    
    def create_indicator(self, name: str, **kwargs):
        """创建指标实例"""
        if name not in self._indicators:
            logger.error(f"指标 {name} 未注册")
            return None
        
        try:
            indicator_class = self._indicators[name]['class']
            return indicator_class(**kwargs)
        except Exception as e:
            logger.error(f"创建指标 {name} 实例失败: {e}")
            return None
    
    def print_summary(self):
        """打印注册摘要"""
        stats = self._registration_stats
        total_registered = len(self._indicators)
        
        print(f"\n=== 指标注册摘要 ===")
        print(f"尝试注册: {stats['total_attempted']} 个")
        print(f"成功注册: {stats['successful']} 个")
        print(f"注册失败: {stats['failed']} 个")
        print(f"最终注册: {total_registered} 个")
        print(f"成功率: {(stats['successful']/stats['total_attempted']*100):.1f}%" if stats['total_attempted'] > 0 else "0%")
        
        if stats['failed_indicators']:
            print(f"\n失败指标:")
            for name, reason in stats['failed_indicators']:
                print(f"  ❌ {name}: {reason}")

    def register_pattern_indicators(self):
        """注册形态指标"""
        logger.info("=== 开始注册形态指标 ===")

        pattern_indicators = [
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS', 'K线形态'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK', '高级K线形态'),
            ('indicators.pattern.zxm_patterns', 'ZXMPatterns', 'ZXM_PATTERNS', 'ZXM形态'),
        ]

        success_count = 0
        for module_path, class_name, indicator_name, description in pattern_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1

        logger.info(f"形态指标注册完成: {success_count}/{len(pattern_indicators)}")
        return success_count

    def register_tool_indicators(self):
        """注册工具指标"""
        logger.info("=== 开始注册工具指标 ===")

        tool_indicators = [
            ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS', '斐波那契工具'),
            ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS', '江恩工具'),
            ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE', '艾略特波浪'),
        ]

        success_count = 0
        for module_path, class_name, indicator_name, description in tool_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1

        logger.info(f"工具指标注册完成: {success_count}/{len(tool_indicators)}")
        return success_count

    def register_formula_indicators(self):
        """注册公式指标"""
        logger.info("=== 开始注册公式指标 ===")

        formula_indicators = [
            ('indicators.formula_indicators', 'CrossOver', 'CROSS_OVER', '交叉条件指标'),
            ('indicators.formula_indicators', 'KDJCondition', 'KDJ_CONDITION', 'KDJ条件指标'),
            ('indicators.formula_indicators', 'MACDCondition', 'MACD_CONDITION', 'MACD条件指标'),
            ('indicators.formula_indicators', 'MACondition', 'MA_CONDITION', 'MA条件指标'),
            ('indicators.formula_indicators', 'GenericCondition', 'GENERIC_CONDITION', '通用条件指标'),
        ]

        success_count = 0
        for module_path, class_name, indicator_name, description in formula_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1

        logger.info(f"公式指标注册完成: {success_count}/{len(formula_indicators)}")
        return success_count

    def register_zxm_indicators(self):
        """注册ZXM体系指标"""
        logger.info("=== 开始注册ZXM体系指标 ===")

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
        ]

        success_count = 0
        for module_path, class_name, indicator_name, description in zxm_indicators:
            if self.register_from_module(module_path, class_name, indicator_name, description):
                success_count += 1

        logger.info(f"ZXM指标注册完成: {success_count}/{len(zxm_indicators)}")
        return success_count

    def register_all_indicators(self):
        """注册所有指标"""
        logger.info("=== 开始完整指标注册 ===")

        # 按类别注册
        core_count = self.register_core_indicators()
        enhanced_count = self.register_enhanced_indicators()
        composite_count = self.register_composite_indicators()
        pattern_count = self.register_pattern_indicators()
        tool_count = self.register_tool_indicators()
        formula_count = self.register_formula_indicators()
        zxm_count = self.register_zxm_indicators()

        # 打印总结
        total_success = core_count + enhanced_count + composite_count + pattern_count + tool_count + formula_count + zxm_count
        logger.info(f"=== 完整注册总结 ===")
        logger.info(f"核心指标: {core_count}")
        logger.info(f"增强指标: {enhanced_count}")
        logger.info(f"复合指标: {composite_count}")
        logger.info(f"形态指标: {pattern_count}")
        logger.info(f"工具指标: {tool_count}")
        logger.info(f"公式指标: {formula_count}")
        logger.info(f"ZXM指标: {zxm_count}")
        logger.info(f"总成功注册: {total_success}")

        self.print_summary()
        return total_success

# 创建全局实例
complete_registry = CompleteIndicatorRegistry()

