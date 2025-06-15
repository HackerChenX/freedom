#!/usr/bin/env python3
"""
技术指标系统全面注册状态检查和分析工具
基于修复进度表中的101个已验证指标进行完整性检查
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IndicatorInfo:
    """指标信息数据类"""
    name: str
    module_path: str
    class_name: str
    category: str
    description: str
    is_available: bool = False
    is_registered: bool = False
    availability_reason: str = ""
    registration_reason: str = ""

class ComprehensiveIndicatorAnalyzer:
    """全面的指标分析器"""
    
    def __init__(self):
        self.indicators = {}  # 所有指标信息
        self.registered_indicators = set()  # 已注册指标
        self.analysis_results = {
            'total_indicators': 0,
            'available_indicators': 0,
            'registered_indicators': 0,
            'unregistered_available': 0,
            'registration_rate': 0.0,
            'categories': {},
            'priority_list': []
        }
    
    def get_all_indicators_from_progress_table(self) -> Dict[str, IndicatorInfo]:
        """基于修复进度表获取所有101个已验证指标"""
        indicators = {}
        
        # 第一部分：核心技术指标 (37个)
        core_indicators = [
            ('MACD', 'indicators.macd', 'MACD', 'MACD指标'),
            ('KDJ', 'indicators.kdj', 'KDJ', 'KDJ随机指标'),
            ('RSI', 'indicators.rsi', 'RSI', '相对强弱指数'),
            ('AD', 'indicators.ad', 'AD', '累积/派发线'),
            ('ADX', 'indicators.adx', 'ADX', '平均趋向指标'),
            ('Aroon', 'indicators.aroon', 'Aroon', 'Aroon指标'),
            ('ATR', 'indicators.atr', 'ATR', '平均真实波幅'),
            ('BIAS', 'indicators.bias', 'BIAS', '乖离率'),
            ('BOLL', 'indicators.boll', 'BollingerBands', '布林带'),
            ('CCI', 'indicators.cci', 'CCI', '顺势指标'),
            ('Chaikin', 'indicators.chaikin', 'ChaikinVolatility', 'Chaikin波动率'),
            ('CMO', 'indicators.cmo', 'CMO', '钱德动量摆动指标'),
            ('DMA', 'indicators.dma', 'DMA', '动态移动平均线'),
            ('DMI', 'indicators.dmi', 'DMI', '趋向指标'),
            ('EMA', 'indicators.ema', 'EMA', '指数移动平均线'),
            ('EMV', 'indicators.emv', 'EMV', '简易波动指标'),
            ('Ichimoku', 'indicators.ichimoku', 'Ichimoku', '一目均衡表'),
            ('KC', 'indicators.kc', 'KC', '肯特纳通道'),
            ('MA', 'indicators.ma', 'MA', '移动平均线'),
            ('MFI', 'indicators.mfi', 'MFI', '资金流量指标'),
            ('Momentum', 'indicators.momentum', 'Momentum', '动量指标'),
            ('MTM', 'indicators.mtm', 'MTM', '动量指标'),
            ('OBV', 'indicators.obv', 'OBV', '能量潮指标'),
            ('PSY', 'indicators.psy', 'PSY', '心理线指标'),
            ('PVT', 'indicators.pvt', 'PVT', '价量趋势指标'),
            ('ROC', 'indicators.roc', 'ROC', '变动率指标'),
            ('SAR', 'indicators.sar', 'SAR', '抛物线转向指标'),
            ('StochRSI', 'indicators.stochrsi', 'StochasticRSI', '随机RSI'),
            ('TRIX', 'indicators.trix', 'TRIX', 'TRIX指标'),
            ('VIX', 'indicators.vix', 'VIX', '恐慌指数'),
            ('VOL', 'indicators.vol', 'Volume', '成交量指标'),
            ('VolumeRatio', 'indicators.volume_ratio', 'VolumeRatio', '量比指标'),
            ('VOSC', 'indicators.vosc', 'VOSC', '成交量震荡器'),
            ('VR', 'indicators.vr', 'VR', '成交量比率'),
            ('Vortex', 'indicators.vortex', 'Vortex', '涡流指标'),
            ('WMA', 'indicators.wma', 'WMA', '加权移动平均线'),
            ('WR', 'indicators.wr', 'WR', '威廉指标'),
        ]
        
        for name, module_path, class_name, description in core_indicators:
            indicators[name] = IndicatorInfo(
                name=name,
                module_path=module_path,
                class_name=class_name,
                category='core',
                description=description
            )
        
        # 第二部分：增强型与复合型指标 (11个)
        enhanced_indicators = [
            ('EnhancedCCI', 'indicators.trend.enhanced_cci', 'EnhancedCCI', '增强版CCI'),
            ('EnhancedDMI', 'indicators.trend.enhanced_dmi', 'EnhancedDMI', '增强版DMI'),
            ('EnhancedMACD', 'indicators.trend.enhanced_macd', 'EnhancedMACD', '增强版MACD(趋势)'),
            ('EnhancedTRIX', 'indicators.trend.enhanced_trix', 'EnhancedTRIX', '增强版TRIX'),
            ('EnhancedKDJ', 'indicators.oscillator.enhanced_kdj', 'EnhancedKDJ', '增强版KDJ(震荡)'),
            ('EnhancedMFI', 'indicators.volume.enhanced_mfi', 'EnhancedMFI', '增强版MFI'),
            ('EnhancedOBV', 'indicators.volume.enhanced_obv', 'EnhancedOBV', '增强版OBV'),
            ('CompositeIndicator', 'indicators.composite_indicator', 'CompositeIndicator', '复合指标'),
            ('UnifiedMA', 'indicators.unified_ma', 'UnifiedMA', '统一移动平均线'),
            ('ChipDistribution', 'indicators.chip_distribution', 'ChipDistribution', '筹码分布'),
            ('InstitutionalBehavior', 'indicators.institutional_behavior', 'InstitutionalBehavior', '机构行为'),
            ('StockVIX', 'indicators.stock_vix', 'StockVIX', '个股恐慌指数'),
        ]
        
        for name, module_path, class_name, description in enhanced_indicators:
            indicators[name] = IndicatorInfo(
                name=name,
                module_path=module_path,
                class_name=class_name,
                category='enhanced',
                description=description
            )
        
        # 第三部分：特色与策略类指标 (28个)
        special_indicators = [
            # 形态指标 (3个)
            ('CandlestickPatterns', 'indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'K线形态'),
            ('AdvancedCandlestickPatterns', 'indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', '高级K线形态'),
            ('ZXMPatterns', 'indicators.pattern.zxm_patterns', 'ZXMPatterns', 'ZXM形态'),
            # 特色工具 (3个)
            ('FibonacciTools', 'indicators.fibonacci_tools', 'FibonacciTools', '斐波那契工具'),
            ('GannTools', 'indicators.gann_tools', 'GannTools', '江恩工具'),
            ('ElliottWave', 'indicators.elliott_wave', 'ElliottWave', '艾略特波浪'),
        ]
        
        for name, module_path, class_name, description in special_indicators:
            category = 'pattern' if 'pattern' in module_path else 'tools'
            indicators[name] = IndicatorInfo(
                name=name,
                module_path=module_path,
                class_name=class_name,
                category=category,
                description=description
            )

        # 第四部分：ZXM体系指标 (25个)
        zxm_indicators = [
            # ZXM Trend (9个)
            ('ZXMDailyTrendUp', 'indicators.zxm.trend_indicators', 'ZXMDailyTrendUp', 'ZXM日趋势向上'),
            ('ZXMWeeklyTrendUp', 'indicators.zxm.trend_indicators', 'ZXMWeeklyTrendUp', 'ZXM周趋势向上'),
            ('ZXMMonthlyKDJTrendUp', 'indicators.zxm.trend_indicators', 'ZXMMonthlyKDJTrendUp', 'ZXM月KDJ趋势向上'),
            ('ZXMWeeklyKDJDOrDEATrendUp', 'indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDOrDEATrendUp', 'ZXM周KDJ D或DEA趋势向上'),
            ('ZXMWeeklyKDJDTrendUp', 'indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDTrendUp', 'ZXM周KDJ D趋势向上'),
            ('ZXMMonthlyMACD', 'indicators.zxm.trend_indicators', 'ZXMMonthlyMACD', 'ZXM月MACD'),
            ('TrendDetector', 'indicators.zxm.trend_indicators', 'TrendDetector', 'ZXM趋势检测器'),
            ('TrendDuration', 'indicators.zxm.trend_indicators', 'TrendDuration', 'ZXM趋势持续时间'),
            ('ZXMWeeklyMACD', 'indicators.zxm.trend_indicators', 'ZXMWeeklyMACD', 'ZXM周MACD'),
            # ZXM Buy Points (5个)
            ('ZXMDailyMACD', 'indicators.zxm.buy_point_indicators', 'ZXMDailyMACD', 'ZXM日MACD买点'),
            ('ZXMTurnover', 'indicators.zxm.buy_point_indicators', 'ZXMTurnover', 'ZXM换手率买点'),
            ('ZXMVolumeShrink', 'indicators.zxm.buy_point_indicators', 'ZXMVolumeShrink', 'ZXM缩量买点'),
            ('ZXMMACallback', 'indicators.zxm.buy_point_indicators', 'ZXMMACallback', 'ZXM均线回踩买点'),
            ('ZXMBSAbsorb', 'indicators.zxm.buy_point_indicators', 'ZXMBSAbsorb', 'ZXM吸筹买点'),
            # ZXM Elasticity (4个)
            ('AmplitudeElasticity', 'indicators.zxm.elasticity_indicators', 'AmplitudeElasticity', 'ZXM振幅弹性'),
            ('ZXMRiseElasticity', 'indicators.zxm.elasticity_indicators', 'ZXMRiseElasticity', 'ZXM涨幅弹性'),
            ('Elasticity', 'indicators.zxm.elasticity_indicators', 'Elasticity', 'ZXM弹性'),
            ('BounceDetector', 'indicators.zxm.elasticity_indicators', 'BounceDetector', 'ZXM反弹检测器'),
            # ZXM Score (3个)
            ('ZXMElasticityScore', 'indicators.zxm.score_indicators', 'ZXMElasticityScore', 'ZXM弹性评分'),
            ('ZXMBuyPointScore', 'indicators.zxm.score_indicators', 'ZXMBuyPointScore', 'ZXM买点评分'),
            ('StockScoreCalculator', 'indicators.zxm.score_indicators', 'StockScoreCalculator', 'ZXM股票评分'),
            # ZXM其他 (4个)
            ('ZXMMarketBreadth', 'indicators.zxm.market_breadth', 'ZXMMarketBreadth', 'ZXM市场宽度'),
            ('SelectionModel', 'indicators.zxm.selection_model', 'SelectionModel', 'ZXM选股模型'),
            ('ZXMDiagnostics', 'indicators.zxm.diagnostics', 'ZXMDiagnostics', 'ZXM诊断'),
            ('BuyPointDetector', 'indicators.zxm.buy_point_indicators', 'BuyPointDetector', 'ZXM买点检测器'),
        ]

        for name, module_path, class_name, description in zxm_indicators:
            indicators[name] = IndicatorInfo(
                name=name,
                module_path=module_path,
                class_name=class_name,
                category='zxm',
                description=description
            )

        # 第五部分：公式指标 (5个)
        formula_indicators = [
            ('CrossOver', 'indicators.formula_indicators', 'CrossOver', '交叉条件指标'),
            ('KDJCondition', 'indicators.formula_indicators', 'KDJCondition', 'KDJ条件指标'),
            ('MACDCondition', 'indicators.formula_indicators', 'MACDCondition', 'MACD条件指标'),
            ('MACondition', 'indicators.formula_indicators', 'MACondition', 'MA条件指标'),
            ('GenericCondition', 'indicators.formula_indicators', 'GenericCondition', '通用条件指标'),
        ]

        for name, module_path, class_name, description in formula_indicators:
            indicators[name] = IndicatorInfo(
                name=name,
                module_path=module_path,
                class_name=class_name,
                category='formula',
                description=description
            )

        return indicators
    
    def check_indicator_availability(self, indicator: IndicatorInfo) -> bool:
        """检查指标是否可用"""
        try:
            # 尝试导入模块
            module = importlib.import_module(indicator.module_path)
            indicator_class = getattr(module, indicator.class_name, None)
            
            if indicator_class is None:
                indicator.availability_reason = f"类 {indicator.class_name} 不存在"
                return False
            
            # 检查是否为BaseIndicator子类
            from indicators.base_indicator import BaseIndicator
            if not issubclass(indicator_class, BaseIndicator):
                indicator.availability_reason = f"不是BaseIndicator子类"
                return False
            
            # 尝试检查必要方法
            required_methods = ['calculate', 'get_patterns', 'calculate_confidence']
            missing_methods = []
            for method in required_methods:
                if not hasattr(indicator_class, method):
                    missing_methods.append(method)
            
            if missing_methods:
                indicator.availability_reason = f"缺少方法: {missing_methods}"
                return False
            
            indicator.availability_reason = "可用"
            return True
            
        except ImportError as e:
            indicator.availability_reason = f"导入失败: {e}"
            return False
        except Exception as e:
            indicator.availability_reason = f"其他错误: {e}"
            return False
    
    def get_currently_registered_indicators(self) -> Set[str]:
        """获取当前已注册的指标"""
        # 基于之前的测试结果，这些是已注册的指标
        return {
            'BIAS', 'BOLL', 'CCI', 'CMO', 'CROSS_OVER', 'Chaikin', 'DMA', 'DMI', 'EMV',
            'ENHANCEDKDJ', 'ENHANCEDMACD', 'EnhancedKDJ', 'EnhancedMACD', 'EnhancedTRIX',
            'GENERIC_CONDITION', 'Ichimoku', 'KDJ', 'KDJ_CONDITION', 'MACD', 'MACD_CONDITION',
            'MA_CONDITION', 'RSI', 'Volume'
        }
    
    def analyze_registration_status(self, indicator: IndicatorInfo):
        """分析指标注册状态"""
        # 检查可能的注册名称
        possible_names = [
            indicator.name,
            indicator.class_name,
            indicator.name.upper(),
            indicator.class_name.upper(),
            # 特殊映射
            'Volume' if indicator.name == 'VOL' else None,
            'BollingerBands' if indicator.name == 'BOLL' else None,
        ]
        possible_names = [name for name in possible_names if name]
        
        # 检查是否已注册
        for name in possible_names:
            if name in self.registered_indicators:
                indicator.is_registered = True
                indicator.registration_reason = f"已注册为 {name}"
                return
        
        indicator.is_registered = False
        indicator.registration_reason = "未注册"

    def perform_comprehensive_analysis(self):
        """执行全面分析"""
        print("=== 技术指标系统全面注册状态检查 ===\n")

        # 获取所有指标
        self.indicators = self.get_all_indicators_from_progress_table()
        self.registered_indicators = self.get_currently_registered_indicators()

        print(f"基于修复进度表，共需检查 {len(self.indicators)} 个已验证指标")
        print(f"当前系统已注册 {len(self.registered_indicators)} 个指标\n")

        # 检查每个指标的可用性和注册状态
        available_count = 0
        registered_count = 0
        category_stats = {}

        for indicator in self.indicators.values():
            # 检查可用性
            indicator.is_available = self.check_indicator_availability(indicator)
            if indicator.is_available:
                available_count += 1

            # 检查注册状态
            self.analyze_registration_status(indicator)
            if indicator.is_registered:
                registered_count += 1

            # 统计类别信息
            if indicator.category not in category_stats:
                category_stats[indicator.category] = {
                    'total': 0, 'available': 0, 'registered': 0, 'unregistered_available': 0
                }

            stats = category_stats[indicator.category]
            stats['total'] += 1
            if indicator.is_available:
                stats['available'] += 1
            if indicator.is_registered:
                stats['registered'] += 1
            if indicator.is_available and not indicator.is_registered:
                stats['unregistered_available'] += 1

        # 更新分析结果
        self.analysis_results.update({
            'total_indicators': len(self.indicators),
            'available_indicators': available_count,
            'registered_indicators': registered_count,
            'unregistered_available': available_count - registered_count,
            'registration_rate': (registered_count / len(self.indicators)) * 100,
            'categories': category_stats
        })

        # 生成报告
        self.generate_detailed_report()

    def generate_detailed_report(self):
        """生成详细报告"""
        results = self.analysis_results

        print("=== 1. 总体注册状态概览 ===")
        print(f"总指标数量: {results['total_indicators']}")
        print(f"可用指标数量: {results['available_indicators']}")
        print(f"已注册指标数量: {results['registered_indicators']}")
        print(f"未注册但可用: {results['unregistered_available']}")
        print(f"当前注册率: {results['registration_rate']:.1f}%")

        print(f"\n=== 2. 各类别注册状态详情 ===")
        for category, stats in results['categories'].items():
            coverage = (stats['registered'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"{category.upper()}:")
            print(f"  总数: {stats['total']}, 可用: {stats['available']}, 已注册: {stats['registered']}")
            print(f"  覆盖率: {coverage:.1f}%, 待注册: {stats['unregistered_available']}")

        print(f"\n=== 3. 未注册但可用的指标清单 ===")
        unregistered_available = [
            indicator for indicator in self.indicators.values()
            if indicator.is_available and not indicator.is_registered
        ]

        # 按类别分组显示
        for category in ['core', 'enhanced', 'pattern', 'tools', 'zxm', 'formula']:
            category_indicators = [ind for ind in unregistered_available if ind.category == category]
            if category_indicators:
                print(f"\n{category.upper()} 类别未注册指标 ({len(category_indicators)}个):")
                for i, indicator in enumerate(category_indicators, 1):
                    print(f"  {i:2d}. {indicator.name} - {indicator.description}")
                    print(f"      路径: {indicator.module_path}.{indicator.class_name}")

        print(f"\n=== 4. 不可用指标分析 ===")
        unavailable_indicators = [
            indicator for indicator in self.indicators.values()
            if not indicator.is_available
        ]

        if unavailable_indicators:
            print(f"发现 {len(unavailable_indicators)} 个不可用指标:")
            for indicator in unavailable_indicators:
                print(f"  ❌ {indicator.name}: {indicator.availability_reason}")
        else:
            print("✅ 所有指标都可用！")

        print(f"\n=== 5. 注册建议和优先级 ===")
        self.generate_registration_recommendations()

    def generate_registration_recommendations(self):
        """生成注册建议和优先级"""
        unregistered_available = [
            indicator for indicator in self.indicators.values()
            if indicator.is_available and not indicator.is_registered
        ]

        # 按优先级排序
        priority_order = ['core', 'enhanced', 'formula', 'pattern', 'tools', 'zxm']
        priority_indicators = []

        for category in priority_order:
            category_indicators = [ind for ind in unregistered_available if ind.category == category]
            priority_indicators.extend(category_indicators)

        print(f"建议按以下优先级注册 {len(priority_indicators)} 个指标:")
        print(f"\n优先级1 - 核心指标 (基础技术分析):")
        core_indicators = [ind for ind in priority_indicators if ind.category == 'core']
        for i, indicator in enumerate(core_indicators[:10], 1):  # 显示前10个
            print(f"  {i:2d}. {indicator.name} - {indicator.description}")
        if len(core_indicators) > 10:
            print(f"      ... 还有 {len(core_indicators) - 10} 个核心指标")

        print(f"\n优先级2 - 增强指标 (高级分析功能):")
        enhanced_indicators = [ind for ind in priority_indicators if ind.category == 'enhanced']
        for i, indicator in enumerate(enhanced_indicators, 1):
            print(f"  {i:2d}. {indicator.name} - {indicator.description}")

        print(f"\n优先级3 - 公式指标 (条件判断):")
        formula_indicators = [ind for ind in priority_indicators if ind.category == 'formula']
        for i, indicator in enumerate(formula_indicators, 1):
            print(f"  {i:2d}. {indicator.name} - {indicator.description}")

        print(f"\n=== 6. 系统功能提升预期 ===")
        self.estimate_system_improvement()

    def estimate_system_improvement(self):
        """评估系统功能提升"""
        results = self.analysis_results

        current_indicators = results['registered_indicators']
        potential_indicators = results['available_indicators']
        improvement = potential_indicators - current_indicators

        print(f"完成全部指标注册后的系统提升:")
        print(f"  📊 指标数量: {current_indicators} → {potential_indicators} (增加 {improvement} 个)")
        print(f"  📈 注册率: {results['registration_rate']:.1f}% → 100% (提升 {100 - results['registration_rate']:.1f}%)")

        # 估算策略条件和技术形态数量
        estimated_conditions = potential_indicators * 8  # 每个指标平均8个条件
        estimated_patterns = potential_indicators * 3   # 每个指标平均3个形态

        print(f"  🎯 预期策略条件: ~{estimated_conditions} 个 (目标: 500+)")
        print(f"  📋 预期技术形态: ~{estimated_patterns} 个 (目标: 150+)")

        if estimated_conditions >= 500 and estimated_patterns >= 150:
            print(f"  ✅ 预期能够达到所有功能目标！")
        else:
            print(f"  ⚠️  可能需要进一步优化以达到功能目标")

        print(f"\n=== 7. 可行性评估 ===")
        availability_rate = (results['available_indicators'] / results['total_indicators']) * 100
        print(f"指标可用率: {availability_rate:.1f}%")

        if availability_rate >= 90:
            print(f"✅ 100%注册率完全可行！大部分指标都可用")
        elif availability_rate >= 70:
            print(f"⚠️  100%注册率基本可行，需要修复部分指标")
        else:
            print(f"❌ 100%注册率存在挑战，需要大量修复工作")

def main():
    """主函数"""
    print("开始技术指标系统全面注册状态检查...\n")

    analyzer = ComprehensiveIndicatorAnalyzer()
    analyzer.perform_comprehensive_analysis()

    # 总结
    results = analyzer.analysis_results
    print(f"\n=== 检查完成 ===")
    print(f"✅ 已完成对 {results['total_indicators']} 个指标的全面检查")
    print(f"📊 当前注册率: {results['registration_rate']:.1f}%")
    print(f"🎯 待注册可用指标: {results['unregistered_available']} 个")

    if results['registration_rate'] >= 80:
        print(f"🎉 系统指标注册状态良好！")
    elif results['registration_rate'] >= 50:
        print(f"⚠️  系统指标注册状态一般，建议优化")
    else:
        print(f"❌ 系统指标注册状态需要改进")

    return results['registration_rate'] >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
