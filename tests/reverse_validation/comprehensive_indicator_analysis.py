#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
全面技术指标分析和分类

分析系统中所有82个技术指标，按优先级分类，识别关键技术形态
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


@dataclass
class IndicatorInfo:
    """指标信息数据类"""
    name: str
    priority: str
    category: str
    file_path: str
    class_name: str
    patterns: List[str]
    description: str
    is_implemented: bool
    has_patterns: bool


class ComprehensiveIndicatorAnalyzer:
    """全面指标分析器"""
    
    def __init__(self):
        self.indicators = {}
        self.priority_classification = {
            'P0': ['RSI', 'MACD', 'KDJ', 'BOLL', 'MA', 'EMA'],  # 核心指标（已完成）
            'P1': ['SAR', 'ADX', 'DMI', 'TRIX', 'ROC', 'CMO', 'DMA', 'MTM'],  # 重要指标
            'P2': ['STOCHRSI', 'PSY', 'WR', 'BIAS', 'VOL', 'OBV', 'MFI', 'EMV', 'CCI', 'MOMENTUM', 'VOSC', 'VR', 'PVT'],  # 常用指标
            'P3': ['ATR', 'KC', 'VORTEX', 'AROON', 'ICHIMOKU', 'WMA', 'AD', 'CHAIKIN', 'VIX', 'VOLUME_RATIO'],  # 专业指标
            'P4': [  # ZXM系列指标
                'ZXM_DAILY_MACD', 'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK',
                'ZXM_BS_ABSORB', 'ZXM_AMPLITUDE_ELASTICITY', 'ZXM_RISE_ELASTICITY', 'ZXM_ELASTICITY',
                'ZXM_BOUNCE_DETECTOR', 'ZXM_ELASTICITY_SCORE', 'ZXM_BUYPOINT_SCORE', 'ZXM_STOCK_SCORE',
                'ZXM_DAILY_TREND_UP', 'ZXM_WEEKLY_TREND_UP', 'ZXM_MONTHLY_KDJ_TREND_UP',
                'ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP', 'ZXM_WEEKLY_KDJ_D_TREND_UP', 'ZXM_MONTHLY_MACD',
                'ZXM_TREND_DETECTOR', 'ZXM_TREND_DURATION', 'ZXM_WEEKLY_MACD'
            ],
            'P5': [  # 系统分析指标
                'STOCK_SCORE_CALCULATOR', 'BOUNCE_DETECTOR', 'TREND_DETECTOR', 'TREND_DURATION',
                'AMPLITUDE_ELASTICITY', 'ELASTICITY', 'INSTITUTIONAL_BEHAVIOR', 'CHIP_DISTRIBUTION',
                'SELECTION_MODEL', 'STOCK_VIX', 'ZXM_MARKET_BREADTH', 'ZXM_SELECTION_MODEL',
                'ZXM_DIAGNOSTICS', 'ZXM_BUYPOINT_DETECTOR', 'ENHANCED_CCI', 'ENHANCED_DMI',
                'ENHANCED_MFI', 'ENHANCED_OBV', 'COMPOSITE', 'UNIFIED_MA'
            ]
        }
        
        # 指标形态定义
        self.indicator_patterns = {
            # P1重要指标形态
            'SAR': ['SAR上升趋势', 'SAR下降趋势', 'SAR转向信号', 'SAR支撑', 'SAR阻力'],
            'ADX': ['ADX强趋势', 'ADX弱趋势', 'ADX上升', 'ADX下降', 'ADX背离'],
            'DMI': ['DMI多头', 'DMI空头', 'DMI金叉', 'DMI死叉', 'DMI背离'],
            'TRIX': ['TRIX金叉', 'TRIX死叉', 'TRIX零轴上', 'TRIX零轴下', 'TRIX背离'],
            'ROC': ['ROC超买', 'ROC超卖', 'ROC金叉', 'ROC死叉', 'ROC背离'],
            'CMO': ['CMO超买', 'CMO超卖', 'CMO金叉', 'CMO死叉', 'CMO背离'],
            'DMA': ['DMA金叉', 'DMA死叉', 'DMA多头排列', 'DMA空头排列', 'DMA背离'],
            'MTM': ['MTM上升', 'MTM下降', 'MTM零轴上', 'MTM零轴下', 'MTM背离'],
            
            # P2常用指标形态
            'STOCHRSI': ['STOCHRSI超买', 'STOCHRSI超卖', 'STOCHRSI金叉', 'STOCHRSI死叉', 'STOCHRSI背离'],
            'PSY': ['PSY超买', 'PSY超卖', 'PSY上升', 'PSY下降', 'PSY极值'],
            'WR': ['WR超买', 'WR超卖', 'WR金叉', 'WR死叉', 'WR背离'],
            'BIAS': ['BIAS正乖离', 'BIAS负乖离', 'BIAS极值', 'BIAS收敛', 'BIAS背离'],
            'VOL': ['VOL放量', 'VOL缩量', 'VOL异常', 'VOL趋势', 'VOL背离'],
            'OBV': ['OBV上升', 'OBV下降', 'OBV背离', 'OBV突破', 'OBV支撑'],
            'MFI': ['MFI超买', 'MFI超卖', 'MFI金叉', 'MFI死叉', 'MFI背离'],
            'EMV': ['EMV上升', 'EMV下降', 'EMV零轴上', 'EMV零轴下', 'EMV背离'],
            'CCI': ['CCI超买', 'CCI超卖', 'CCI金叉', 'CCI死叉', 'CCI背离'],
            'MOMENTUM': ['MOMENTUM上升', 'MOMENTUM下降', 'MOMENTUM零轴上', 'MOMENTUM零轴下', 'MOMENTUM背离'],
            'VOSC': ['VOSC上升', 'VOSC下降', 'VOSC金叉', 'VOSC死叉', 'VOSC背离'],
            'VR': ['VR超买', 'VR超卖', 'VR上升', 'VR下降', 'VR极值'],
            'PVT': ['PVT上升', 'PVT下降', 'PVT背离', 'PVT突破', 'PVT支撑'],
            
            # P3专业指标形态
            'ATR': ['ATR高波动', 'ATR低波动', 'ATR上升', 'ATR下降', 'ATR极值'],
            'KC': ['KC上轨突破', 'KC下轨突破', 'KC收口', 'KC开口', 'KC中轨支撑'],
            'VORTEX': ['VORTEX多头', 'VORTEX空头', 'VORTEX金叉', 'VORTEX死叉', 'VORTEX背离'],
            'AROON': ['AROON多头', 'AROON空头', 'AROON上升', 'AROON下降', 'AROON平行'],
            'ICHIMOKU': ['ICHIMOKU多头', 'ICHIMOKU空头', 'ICHIMOKU云上', 'ICHIMOKU云下', 'ICHIMOKU转换'],
            'WMA': ['WMA金叉', 'WMA死叉', 'WMA多头排列', 'WMA空头排列', 'WMA支撑'],
            'AD': ['AD上升', 'AD下降', 'AD背离', 'AD突破', 'AD支撑'],
            'CHAIKIN': ['CHAIKIN上升', 'CHAIKIN下降', 'CHAIKIN金叉', 'CHAIKIN死叉', 'CHAIKIN背离'],
            'VIX': ['VIX恐慌', 'VIX贪婪', 'VIX上升', 'VIX下降', 'VIX极值'],
            'VOLUME_RATIO': ['VOLUME_RATIO放量', 'VOLUME_RATIO缩量', 'VOLUME_RATIO异常', 'VOLUME_RATIO趋势', 'VOLUME_RATIO背离']
        }
    
    def analyze_all_indicators_Analysis(self) -> Dict:
        """分析所有指标"""
        print("开始分析系统中的所有技术指标...")
        
        # 扫描indicators目录
        indicators_dir = Path("indicators")
        if not indicators_dir.exists():
            print("❌ indicators目录不存在")
            return {}
        
        # 收集所有指标文件
        indicator_files = self._collect_indicator_files(indicators_dir)
        print(f"发现 {len(indicator_files)} 个指标文件")
        
        # 分析每个指标
        for file_path in indicator_files:
            self._analyze_indicator_file(file_path)
        
        # 生成分析报告
        analysis_result = self._generate_analysis_report()
        
        return analysis_result
    
    def _collect_indicator_files(self, indicators_dir: Path) -> List[Path]:
        """收集所有指标文件"""
        indicator_files = []
        
        # 跳过的文件
        skip_files = {
            '__init__.py', 'base_indicator.py', 'common.py', 
            'pattern_registry.py', 'indicator_registry.py',
            'lazy_indicator_registry.py', 'complete_indicator_registry.py',
            'factory.py', 'enhanced_factory.py', 'adapter.py'
        }
        
        # 扫描主目录
        for py_file in indicators_dir.glob("*.py"):
            if py_file.name not in skip_files and not py_file.name.endswith('.backup'):
                indicator_files.append(py_file)
        
        # 扫描子目录
        for subdir in indicators_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('__'):
                for py_file in subdir.glob("*.py"):
                    if py_file.name not in skip_files and not py_file.name.endswith('.backup'):
                        indicator_files.append(py_file)
        
        return sorted(indicator_files)
    
    def _analyze_indicator_file(self, file_path: Path):
        """分析单个指标文件"""
        try:
            # 获取指标名称
            relative_path = file_path.relative_to(Path("indicators"))
            if relative_path.parent.name != ".":
                module_name = f"indicators.{relative_path.parent.name}.{relative_path.stem}"
            else:
                module_name = f"indicators.{relative_path.stem}"
            
            indicator_name = relative_path.stem.upper()
            
            # 获取优先级
            priority = self._get_indicator_priority(indicator_name)
            
            # 获取类别
            category = self._get_indicator_category(file_path)
            
            # 获取形态
            patterns = self.indicator_patterns.get(indicator_name, [])
            
            # 检查实现状态
            is_implemented = self._check_implementation(file_path)
            has_patterns = len(patterns) > 0
            
            # 创建指标信息
            indicator_info = IndicatorInfo(
                name=indicator_name,
                priority=priority,
                category=category,
                file_path=str(file_path),
                class_name=self._extract_class_name(file_path),
                patterns=patterns,
                description=self._extract_description(file_path),
                is_implemented=is_implemented,
                has_patterns=has_patterns
            )
            
            self.indicators[indicator_name] = indicator_info
            
        except Exception as e:
            print(f"分析 {file_path} 时出错: {e}")
    
    def _get_indicator_priority(self, indicator_name: str) -> str:
        """获取指标优先级"""
        for priority, indicators in self.priority_classification.items():
            if any(indicator_name.startswith(ind.upper()) or ind.upper().startswith(indicator_name) 
                   for ind in indicators):
                return priority
        return 'P6'  # 未分类
    
    def _get_indicator_category(self, file_path: Path) -> str:
        """获取指标类别"""
        if 'zxm' in str(file_path).lower():
            return 'ZXM'
        elif 'trend' in str(file_path).lower():
            return 'Trend'
        elif 'volume' in str(file_path).lower():
            return 'Volume'
        elif 'oscillator' in str(file_path).lower():
            return 'Oscillator'
        elif 'pattern' in str(file_path).lower():
            return 'Pattern'
        else:
            return 'General'
    
    def _check_implementation(self, file_path: Path) -> bool:
        """检查指标是否已实现"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return 'class ' in content and 'def calculate_Comprehensive_Indicator_Analysis' in content
        except:
            return False
    
    def _extract_class_name(self, file_path: Path) -> str:
        """提取类名"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith('class ') and '(' in line:
                        class_name = line.split('class ')[1].split('(')[0].strip()
                        return class_name
        except:
            pass
        return file_path.stem.title()
    
    def _extract_description(self, file_path: Path) -> str:
        """提取描述"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '"""' in line and i < 10:
                        desc_lines = []
                        for j in range(i+1, min(i+5, len(lines))):
                            if '"""' in lines[j]:
                                break
                            desc_lines.append(lines[j].strip())
                        return ' '.join(desc_lines)
        except:
            pass
        return f"{file_path.stem}技术指标"
    
    def _generate_analysis_report(self) -> Dict:
        """生成分析报告"""
        total_indicators = len(self.indicators)
        
        # 按优先级统计
        priority_stats = {}
        for priority in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            indicators_in_priority = [ind for ind in self.indicators.values() if ind.priority == priority]
            priority_stats[priority] = {
                'count': len(indicators_in_priority),
                'implemented': len([ind for ind in indicators_in_priority if ind.is_implemented]),
                'has_patterns': len([ind for ind in indicators_in_priority if ind.has_patterns]),
                'indicators': indicators_in_priority
            }
        
        # 按类别统计
        category_stats = {}
        categories = set(ind.category for ind in self.indicators.values())
        for category in categories:
            indicators_in_category = [ind for ind in self.indicators.values() if ind.category == category]
            category_stats[category] = {
                'count': len(indicators_in_category),
                'indicators': indicators_in_category
            }
        
        return {
            'total_indicators': total_indicators,
            'priority_stats': priority_stats,
            'category_stats': category_stats,
            'indicators': self.indicators
        }


def mainComprehensiveindicatoranalysis():
    """主函数"""
    print("=" * 80)
    print("全面技术指标分析和分类")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    analyzer = ComprehensiveIndicatorAnalyzer()
    analysis_result = analyzer.analyze_all_indicators_Analysis()
    
    # 显示分析结果
    print("=" * 80)
    print("📊 指标分析结果")
    print("=" * 80)
    print(f"总指标数: {analysis_result['total_indicators']}")
    print()
    
    print("📋 按优先级分布:")
    priority_names = {
        'P0': '核心指标(已完成)', 'P1': '重要指标', 'P2': '常用指标', 
        'P3': '专业指标', 'P4': 'ZXM系列', 'P5': '系统分析', 'P6': '其他指标'
    }
    
    for priority in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        stats = analysis_result['priority_stats'][priority]
        if stats['count'] > 0:
            print(f"  {priority} ({priority_names.get(priority, '其他')}): {stats['count']}个")
            print(f"    - 已实现: {stats['implemented']}个")
            print(f"    - 有形态定义: {stats['has_patterns']}个")
    
    print()
    print("📂 按类别分布:")
    for category, stats in analysis_result['category_stats'].items():
        print(f"  {category}: {stats['count']}个")
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_indicator_analysis_{timestamp}.json"
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        serializable_result = {
            'total_indicators': analysis_result['total_indicators'],
            'priority_stats': {
                priority: {
                    'count': stats['count'],
                    'implemented': stats['implemented'],
                    'has_patterns': stats['has_patterns'],
                    'indicators': [
                        {
                            'name': ind.name,
                            'priority': ind.priority,
                            'category': ind.category,
                            'patterns': ind.patterns,
                            'is_implemented': ind.is_implemented,
                            'has_patterns': ind.has_patterns
                        } for ind in stats['indicators']
                    ]
                } for priority, stats in analysis_result['priority_stats'].items()
            },
            'category_stats': {
                category: {
                    'count': stats['count'],
                    'indicators': [ind.name for ind in stats['indicators']]
                } for category, stats in analysis_result['category_stats'].items()
            }
        }
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细分析结果已保存到: {output_file}")
    
    return analysis_result


if __name__ == '__main__':
    mainComprehensiveindicatoranalysis()
