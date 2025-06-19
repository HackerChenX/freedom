#!/usr/bin/env python3
"""
迁移候选分析器 - 分析COMPLETE_INDICATOR_PATTERNS_MAP中哪些指标适合迁移到PatternRegistry
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.buypoints.buypoint_batch_analyzer import COMPLETE_INDICATOR_PATTERNS_MAP
from utils.logger import get_logger

# 设置日志级别
import logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class MigrationCandidateAnalyzer:
    """迁移候选分析器"""
    
    def __init__(self):
        self.migration_priorities = {
            # P0: 核心技术指标 - 最高优先级
            'P0': ['MA', 'EMA', 'MACD', 'KDJ', 'RSI', 'BOLL'],
            
            # P1: 重要技术指标 - 高优先级
            'P1': ['SAR', 'CCI', 'DMI', 'ADX', 'DMA', 'MTM'],
            
            # P2: 常用技术指标 - 中等优先级
            'P2': ['VOSC', 'VR', 'PVT', 'EMV', 'BIAS', 'Momentum'],
            
            # P3: 专业技术指标 - 低优先级
            'P3': ['Aroon', 'Ichimoku', 'StockVIX', 'EnhancedMACD', 'EnhancedTRIX', 'TRIX'],
            
            # P4: ZXM系列指标 - 特殊处理
            'P4': ['ZXMDailyMACD', 'ZXMTurnover', 'ZXMVolumeShrink', 'ZXMMACallback', 
                   'ZXMBuyPointScore', 'ZXMPattern', 'ZXMRiseElasticity', 'ZXMElasticityScore'],
            
            # P5: 系统分析指标 - 保留在集中式映射
            'P5': ['StockScoreCalculator', 'BounceDetector', 'TrendDetector', 'TrendDuration',
                   'AmplitudeElasticity', 'Elasticity', 'InstitutionalBehavior', 'ChipDistribution',
                   'SelectionModel']
        }
        
        # 已迁移的指标
        self.migrated_indicators = [
            'KDJ', 'RSI', 'TRIX', 'ROC', 'CMO', 'VOL', 'ATR', 'KC', 'MFI', 'Vortex', 'OBV'
        ]
    
    def get_indicator_priority(self, indicator_name: str) -> str:
        """获取指标的迁移优先级"""
        for priority, indicators in self.migration_priorities.items():
            if indicator_name in indicators:
                return priority
        return 'P6'  # 未分类
    
    def analyze_migration_candidates(self) -> dict:
        """分析迁移候选指标"""
        print("\n=== 迁移候选指标分析 ===")
        
        candidates = {}
        total_patterns = 0
        
        for indicator_name, patterns in COMPLETE_INDICATOR_PATTERNS_MAP.items():
            if isinstance(patterns, dict):
                pattern_count = len(patterns)
                total_patterns += pattern_count
                priority = self.get_indicator_priority(indicator_name)
                is_migrated = indicator_name in self.migrated_indicators
                
                candidates[indicator_name] = {
                    'pattern_count': pattern_count,
                    'priority': priority,
                    'is_migrated': is_migrated,
                    'patterns': list(patterns.keys())
                }
        
        return {
            'candidates': candidates,
            'total_patterns': total_patterns,
            'total_indicators': len(candidates)
        }
    
    def generate_migration_recommendations(self, analysis_result: dict) -> list:
        """生成迁移建议"""
        print("\n=== 迁移建议生成 ===")
        
        recommendations = []
        candidates = analysis_result['candidates']
        
        # 按优先级分组
        priority_groups = {}
        for indicator_name, info in candidates.items():
            if not info['is_migrated']:  # 只考虑未迁移的指标
                priority = info['priority']
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append((indicator_name, info))
        
        # 生成建议
        for priority in ['P0', 'P1', 'P2', 'P3', 'P4']:
            if priority in priority_groups:
                group = priority_groups[priority]
                # 按形态数量排序（形态多的优先）
                group.sort(key=lambda x: x[1]['pattern_count'], reverse=True)
                
                for indicator_name, info in group:
                    recommendation = {
                        'indicator_name': indicator_name,
                        'priority': priority,
                        'pattern_count': info['pattern_count'],
                        'migration_reason': self._get_migration_reason(indicator_name, info),
                        'estimated_effort': self._estimate_effort(info['pattern_count']),
                        'patterns': info['patterns']
                    }
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _get_migration_reason(self, indicator_name: str, info: dict) -> str:
        """获取迁移原因"""
        priority = info['priority']
        pattern_count = info['pattern_count']
        
        if priority == 'P0':
            return f"核心技术指标，{pattern_count}个形态，使用频率极高"
        elif priority == 'P1':
            return f"重要技术指标，{pattern_count}个形态，使用频率高"
        elif priority == 'P2':
            return f"常用技术指标，{pattern_count}个形态，使用频率中等"
        elif priority == 'P3':
            return f"专业技术指标，{pattern_count}个形态，专业用户使用"
        elif priority == 'P4':
            return f"ZXM系列指标，{pattern_count}个形态，特殊买点体系"
        else:
            return f"其他指标，{pattern_count}个形态"
    
    def _estimate_effort(self, pattern_count: int) -> str:
        """估算迁移工作量"""
        if pattern_count <= 3:
            return "低"
        elif pattern_count <= 8:
            return "中"
        elif pattern_count <= 15:
            return "高"
        else:
            return "很高"

def main():
    """主函数"""
    print("开始分析迁移候选指标...")
    
    analyzer = MigrationCandidateAnalyzer()
    
    # 分析候选指标
    analysis_result = analyzer.analyze_migration_candidates()
    
    # 显示当前状态
    print(f"\n📊 当前状态:")
    print(f"   集中式映射中的指标总数: {analysis_result['total_indicators']}")
    print(f"   集中式映射中的形态总数: {analysis_result['total_patterns']}")
    print(f"   已迁移指标数量: {len(analyzer.migrated_indicators)}")
    
    # 生成迁移建议
    recommendations = analyzer.generate_migration_recommendations(analysis_result)
    
    # 显示迁移建议
    print(f"\n🎯 迁移建议 (共{len(recommendations)}个候选指标):")
    print("="*80)
    
    current_priority = None
    for i, rec in enumerate(recommendations, 1):
        if rec['priority'] != current_priority:
            current_priority = rec['priority']
            priority_names = {
                'P0': '核心技术指标',
                'P1': '重要技术指标', 
                'P2': '常用技术指标',
                'P3': '专业技术指标',
                'P4': 'ZXM系列指标'
            }
            print(f"\n### {current_priority}: {priority_names.get(current_priority, '其他指标')}")
        
        print(f"{i:2d}. {rec['indicator_name']}")
        print(f"    形态数量: {rec['pattern_count']}")
        print(f"    工作量: {rec['estimated_effort']}")
        print(f"    迁移原因: {rec['migration_reason']}")
        
        # 显示前3个形态作为示例
        if rec['patterns']:
            sample_patterns = rec['patterns'][:3]
            print(f"    示例形态: {', '.join(sample_patterns)}")
            if len(rec['patterns']) > 3:
                print(f"    (还有{len(rec['patterns']) - 3}个形态...)")
        print()
    
    # 生成迁移计划
    print("\n📋 建议的迁移计划:")
    print("="*50)
    
    # 第一阶段：P0和P1指标
    phase1 = [r for r in recommendations if r['priority'] in ['P0', 'P1']]
    if phase1:
        print(f"\n🚀 第一阶段 (核心指标): {len(phase1)}个指标")
        for rec in phase1:
            print(f"   - {rec['indicator_name']} ({rec['pattern_count']}个形态)")
    
    # 第二阶段：P2指标
    phase2 = [r for r in recommendations if r['priority'] == 'P2']
    if phase2:
        print(f"\n⚡ 第二阶段 (常用指标): {len(phase2)}个指标")
        for rec in phase2:
            print(f"   - {rec['indicator_name']} ({rec['pattern_count']}个形态)")
    
    # 第三阶段：P3和P4指标
    phase3 = [r for r in recommendations if r['priority'] in ['P3', 'P4']]
    if phase3:
        print(f"\n🔧 第三阶段 (专业指标): {len(phase3)}个指标")
        for rec in phase3:
            print(f"   - {rec['indicator_name']} ({rec['pattern_count']}个形态)")
    
    print(f"\n✅ 分析完成！建议优先迁移第一阶段的{len(phase1)}个核心指标。")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
