#!/usr/bin/env python3
"""
集中式映射状态分析器 - 分析COMPLETE_INDICATOR_PATTERNS_MAP中的重复定义和迁移状态
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.buypoints.buypoint_batch_analyzer import COMPLETE_INDICATOR_PATTERNS_MAP
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger
import importlib

# 设置日志级别
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class CentralizedMappingAnalyzer:
    """集中式映射状态分析器"""

    def __init__(self):
        self.registry = PatternRegistry()
        self._initialize_indicators()
        
        # 已知已实现register_patterns()方法的指标
        self.indicators_with_patterns = {
            # 核心技术指标
            'KDJ': 'indicators.kdj',
            'RSI': 'indicators.rsi', 
            'MACD': 'indicators.macd',
            'BOLL': 'indicators.boll',
            'MA': 'indicators.ma',
            'EMA': 'indicators.ema',
            
            # 趋势指标
            'SAR': 'indicators.sar',
            'ADX': 'indicators.adx',
            'DMI': 'indicators.dmi',
            
            # 动量指标
            'TRIX': 'indicators.trix',
            'ROC': 'indicators.roc',
            'CMO': 'indicators.cmo',
            'STOCHRSI': 'indicators.stochrsi',
            'PSY': 'indicators.psy',
            'WR': 'indicators.wr',
            'BIAS': 'indicators.bias',
            
            # 成交量指标
            'VOL': 'indicators.vol',
            'OBV': 'indicators.obv',
            'MFI': 'indicators.mfi',
            'EMV': 'indicators.emv',
            
            # 波动率指标
            'ATR': 'indicators.atr',
            'KC': 'indicators.kc',
            'Vortex': 'indicators.vortex',
            'CCI': 'indicators.cci',
        }
        
        # 指标优先级分类
        self.priority_classification = {
            'P0': ['KDJ', 'RSI', 'MACD', 'BOLL', 'MA', 'EMA'],  # 核心指标
            'P1': ['SAR', 'ADX', 'DMI', 'TRIX', 'ROC', 'CMO'],  # 重要指标
            'P2': ['STOCHRSI', 'PSY', 'WR', 'BIAS', 'VOL', 'OBV', 'MFI', 'EMV'],  # 常用指标
            'P3': ['ATR', 'KC', 'Vortex', 'CCI', 'VOSC', 'VR', 'PVT'],  # 专业指标
            'P4': ['ZXMDailyMACD', 'ZXMTurnover', 'ZXMVolumeShrink', 'ZXMMACallback', 
                   'ZXMBuyPointScore', 'ZXMPattern', 'ZXMRiseElasticity', 'ZXMElasticityScore'],  # ZXM系列
            'P5': ['StockScoreCalculator', 'BounceDetector', 'TrendDetector', 'TrendDuration',
                   'AmplitudeElasticity', 'Elasticity', 'InstitutionalBehavior', 'ChipDistribution',
                   'SelectionModel', 'StockVIX']  # 系统分析指标
        }

    def _initialize_indicators(self):
        """初始化所有指标以注册形态到PatternRegistry"""
        print("🔄 初始化指标并注册形态...")

        indicators_to_init = [
            ('indicators.kdj', 'KDJ'),
            ('indicators.rsi', 'RSI'),
            ('indicators.trix', 'TRIX'),
            ('indicators.roc', 'ROC'),
            ('indicators.cmo', 'CMO'),
            ('indicators.vol', 'VOL'),
            ('indicators.atr', 'ATR'),
            ('indicators.kc', 'KC'),
            ('indicators.mfi', 'MFI'),
            ('indicators.vortex', 'Vortex'),
            ('indicators.obv', 'OBV'),
            ('indicators.ma', 'MA'),
            ('indicators.ema', 'EMA'),
            ('indicators.cci', 'CCI'),
            ('indicators.sar', 'SAR'),
            ('indicators.adx', 'ADX'),
            ('indicators.psy', 'PSY'),
            ('indicators.bias', 'BIAS'),
            ('indicators.dmi', 'DMI'),
            ('indicators.emv', 'EMV'),
            ('indicators.wr', 'WR'),
        ]

        initialized_count = 0
        for module_name, class_name in indicators_to_init:
            try:
                module = importlib.import_module(module_name)
                indicator_class = getattr(module, class_name)

                # 尝试不同的初始化方式
                indicator = None
                try:
                    indicator = indicator_class()
                except:
                    try:
                        indicator = indicator_class(period=14)
                    except:
                        try:
                            indicator = indicator_class(periods=[5, 10, 20])
                        except:
                            pass

                if indicator and hasattr(indicator, 'register_patterns'):
                    indicator.register_patterns()
                    initialized_count += 1

            except Exception as e:
                pass  # 静默处理错误

        print(f"✅ 成功初始化{initialized_count}个指标")

    def get_indicator_priority(self, indicator_name: str) -> str:
        """获取指标优先级"""
        for priority, indicators in self.priority_classification.items():
            if indicator_name in indicators:
                return priority
        return 'P6'  # 未分类
    
    def check_indicator_implementation(self, indicator_name: str) -> dict:
        """检查指标是否已实现register_patterns()方法"""
        result = {
            'has_implementation': False,
            'module_path': None,
            'patterns_count': 0,
            'error': None
        }

        # 首先检查PatternRegistry中是否已有该指标的形态
        patterns = self.registry.get_patterns_by_indicator(indicator_name)
        if patterns and len(patterns) > 0:
            result['has_implementation'] = True
            result['patterns_count'] = len(patterns)
            print(f"✅ {indicator_name}: 在PatternRegistry中找到{len(patterns)}个形态")

        # 然后检查是否有对应的模块实现
        if indicator_name in self.indicators_with_patterns:
            module_path = self.indicators_with_patterns[indicator_name]
            result['module_path'] = module_path

            try:
                # 尝试导入模块
                module = importlib.import_module(module_path)

                # 查找指标类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (hasattr(attr, '__bases__') and
                        hasattr(attr, 'register_patterns') and
                        callable(getattr(attr, 'register_patterns'))):

                        if not result['has_implementation']:
                            result['has_implementation'] = True
                            print(f"✅ {indicator_name}: 找到register_patterns()方法")
                        break

            except Exception as e:
                result['error'] = str(e)
                print(f"⚠️ {indicator_name}: 模块导入失败 - {e}")
        else:
            print(f"❌ {indicator_name}: 未找到对应的模块实现")

        return result
    
    def analyze_centralized_mapping(self) -> dict:
        """分析集中式映射状态"""
        print("\n=== 集中式映射状态分析 ===")
        
        analysis_result = {
            'total_indicators': len(COMPLETE_INDICATOR_PATTERNS_MAP),
            'total_patterns': 0,
            'migrated_indicators': [],
            'duplicate_indicators': [],
            'unmigrated_indicators': [],
            'priority_distribution': {},
            'migration_recommendations': []
        }
        
        # 统计总形态数量
        for indicator_name, patterns in COMPLETE_INDICATOR_PATTERNS_MAP.items():
            if isinstance(patterns, dict):
                analysis_result['total_patterns'] += len(patterns)
        
        print(f"📊 集中式映射总览:")
        print(f"   指标总数: {analysis_result['total_indicators']}")
        print(f"   形态总数: {analysis_result['total_patterns']}")
        
        # 分析每个指标的状态
        for indicator_name, patterns in COMPLETE_INDICATOR_PATTERNS_MAP.items():
            if not isinstance(patterns, dict):
                continue
                
            pattern_count = len(patterns)
            priority = self.get_indicator_priority(indicator_name)
            implementation_status = self.check_indicator_implementation(indicator_name)
            
            indicator_info = {
                'name': indicator_name,
                'pattern_count': pattern_count,
                'priority': priority,
                'has_implementation': implementation_status['has_implementation'],
                'registry_patterns_count': implementation_status['patterns_count'],
                'module_path': implementation_status['module_path'],
                'error': implementation_status['error']
            }
            
            # 分类指标
            if implementation_status['has_implementation']:
                if implementation_status['patterns_count'] > 0:
                    analysis_result['migrated_indicators'].append(indicator_info)
                    # 如果在集中式映射中还有定义，则为重复
                    if pattern_count > 0:
                        analysis_result['duplicate_indicators'].append(indicator_info)
                else:
                    analysis_result['unmigrated_indicators'].append(indicator_info)
            else:
                analysis_result['unmigrated_indicators'].append(indicator_info)
            
            # 统计优先级分布
            if priority not in analysis_result['priority_distribution']:
                analysis_result['priority_distribution'][priority] = []
            analysis_result['priority_distribution'][priority].append(indicator_info)
        
        return analysis_result
    
    def generate_migration_recommendations(self, analysis_result: dict) -> list:
        """生成迁移建议"""
        recommendations = []
        
        # 1. 清理重复定义的建议
        if analysis_result['duplicate_indicators']:
            recommendations.append({
                'type': 'cleanup_duplicates',
                'priority': 'HIGH',
                'title': '清理重复定义',
                'description': f"移除{len(analysis_result['duplicate_indicators'])}个已迁移指标的重复定义",
                'indicators': [ind['name'] for ind in analysis_result['duplicate_indicators']],
                'estimated_effort': 'LOW'
            })
        
        # 2. 按优先级生成迁移建议
        priority_order = ['P0', 'P1', 'P2', 'P3']
        for priority in priority_order:
            if priority in analysis_result['priority_distribution']:
                unmigrated_in_priority = [
                    ind for ind in analysis_result['priority_distribution'][priority]
                    if not ind['has_implementation'] or ind['registry_patterns_count'] == 0
                ]
                
                if unmigrated_in_priority:
                    priority_names = {
                        'P0': '核心指标',
                        'P1': '重要指标',
                        'P2': '常用指标',
                        'P3': '专业指标'
                    }
                    
                    recommendations.append({
                        'type': 'migrate_indicators',
                        'priority': priority,
                        'title': f'迁移{priority_names[priority]}',
                        'description': f"迁移{len(unmigrated_in_priority)}个{priority_names[priority]}到PatternRegistry",
                        'indicators': [ind['name'] for ind in unmigrated_in_priority],
                        'estimated_effort': 'MEDIUM' if len(unmigrated_in_priority) <= 3 else 'HIGH'
                    })
        
        return recommendations

def main():
    """主函数"""
    print("开始分析集中式映射状态...")
    
    analyzer = CentralizedMappingAnalyzer()
    
    # 分析集中式映射
    analysis_result = analyzer.analyze_centralized_mapping()
    
    # 显示分析结果
    print(f"\n📈 分析结果详情:")
    print(f"   已迁移指标: {len(analysis_result['migrated_indicators'])}")
    print(f"   重复定义指标: {len(analysis_result['duplicate_indicators'])}")
    print(f"   未迁移指标: {len(analysis_result['unmigrated_indicators'])}")
    
    # 显示重复定义的指标
    if analysis_result['duplicate_indicators']:
        print(f"\n⚠️ 发现{len(analysis_result['duplicate_indicators'])}个重复定义的指标:")
        for ind in analysis_result['duplicate_indicators']:
            print(f"   - {ind['name']} ({ind['priority']}): 集中式映射{ind['pattern_count']}个形态, "
                  f"PatternRegistry{ind['registry_patterns_count']}个形态")
    
    # 显示未迁移指标按优先级分布
    print(f"\n📋 未迁移指标按优先级分布:")
    for priority in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        if priority in analysis_result['priority_distribution']:
            unmigrated = [ind for ind in analysis_result['priority_distribution'][priority]
                         if not ind['has_implementation'] or ind['registry_patterns_count'] == 0]
            if unmigrated:
                priority_names = {
                    'P0': '核心指标', 'P1': '重要指标', 'P2': '常用指标', 'P3': '专业指标',
                    'P4': 'ZXM系列', 'P5': '系统分析', 'P6': '其他指标'
                }
                print(f"   {priority} ({priority_names.get(priority, '其他')}): {len(unmigrated)}个")
                for ind in unmigrated:
                    print(f"     - {ind['name']} ({ind['pattern_count']}个形态)")
    
    # 生成迁移建议
    recommendations = analyzer.generate_migration_recommendations(analysis_result)
    
    print(f"\n🎯 迁移建议 (共{len(recommendations)}项):")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} ({rec['priority']}优先级)")
        print(f"   描述: {rec['description']}")
        print(f"   工作量: {rec['estimated_effort']}")
        print(f"   涉及指标: {', '.join(rec['indicators'][:5])}")
        if len(rec['indicators']) > 5:
            print(f"   (还有{len(rec['indicators']) - 5}个...)")
        print()
    
    print("✅ 集中式映射状态分析完成！")
    
    return analysis_result

if __name__ == "__main__":
    result = main()
    sys.exit(0)
