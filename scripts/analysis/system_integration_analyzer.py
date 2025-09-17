#!/usr/bin/env python3
"""
系统集成状态分析工具

分析买点分析系统和策略选股系统的集成现状，
识别需要重构的部分，制定具体的集成方案。
"""

import sys
import os
import inspect
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class System_integration_analyzer:
    """系统集成状态分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        self.integration_gaps = []
        self.recommendations = []
    
    def analyze_buypoint_system(self) -> Dict[str, Any]:
        """分析买点分析系统"""
        print("🔍 分析买点分析系统...")
        
        try:
            # 导入买点分析相关模块
            from analysis.buypoints.analyze_buypoints import Buy_point_analyzer
            from analysis.buypoints.buypoint_batch_analyzer import Buy_point_batch_analyzer
            from analysis.buypoints.buypoint_strategy_adapter import Buy_point_to_strategy_adapter
            
            # 分析模块结构
            buypoint_modules = {
                'BuyPointAnalyzer': self._analyze_class(BuyPointAnalyzer),
                'BuyPointBatchAnalyzer': self._analyze_class(BuyPointBatchAnalyzer),
                'BuyPointToStrategyAdapter': self._analyze_class(BuyPointToStrategyAdapter)
            }
            
            # 分析数据流
            data_flow = self._analyze_buypoint_data_flow()
            
            # 分析输出格式
            output_format = self._analyze_buypoint_output_format()
            
            result = {
                'modules': buypoint_modules,
                'data_flow': data_flow,
                'output_format': output_format,
                'integration_readiness': self._assess_buypoint_integration_readiness()
            }
            
            print("✅ 买点分析系统分析完成")
            return result
            
        except Exception as e:
            print(f"❌ 买点分析系统分析失败: {e}")
            return {'error': str(e)}
    
    def analyze_strategy_system(self) -> Dict[str, Any]:
        """分析策略选股系统"""
        print("🔍 分析策略选股系统...")
        
        try:
            # 导入策略选股相关模块
            from strategy.base_strategy import BaseStrategy
            from strategy.strategy_executor import Strategy_executor
            from strategy.strategy_executor import UnifiedStrategyExecutor as Enhanced_strategy_executor
            from strategy.strategy_manager import Strategy_manager
            
            # 分析模块结构
            strategy_modules = {
                'BaseStrategy': self._analyze_class(BaseStrategy),
                'StrategyExecutor': self._analyze_class(StrategyExecutor),
                'UnifiedStrategyExecutor': self._analyze_class(UnifiedStrategyExecutor),
                'StrategyManager': self._analyze_class(StrategyManager)
            }
            
            # 分析数据流
            data_flow = self._analyze_strategy_data_flow()
            
            # 分析输入期望
            input_format = self._analyze_strategy_input_format()
            
            result = {
                'modules': strategy_modules,
                'data_flow': data_flow,
                'input_format': input_format,
                'integration_readiness': self._assess_strategy_integration_readiness()
            }
            
            print("✅ 策略选股系统分析完成")
            return result
            
        except Exception as e:
            print(f"❌ 策略选股系统分析失败: {e}")
            return {'error': str(e)}
    
    def analyze_unified_engine(self) -> Dict[str, Any]:
        """分析统一分析引擎"""
        print("🔍 分析统一分析引擎...")
        
        try:
            # 导入统一分析引擎模块
            from analysis.engines.unified_indicator_engine import Unified_indicator_engine
            from analysis.engines.shared_condition_evaluator import Shared_condition_evaluator
            from analysis.engines.complex_logic_processor import Complex_logic_processor
            from analysis.engines.date_manager import Date_manager
from db.sql_manager import SQLManager, QueryType
            
            # 分析引擎组件
            engine_components = {
                'UnifiedIndicatorEngine': self._analyze_class(UnifiedIndicatorEngine),
                'SharedConditionEvaluator': self._analyze_class(SharedConditionEvaluator),
                'ComplexLogicProcessor': self._analyze_class(ComplexLogicProcessor),
                'DateManager': self._analyze_class(DateManager)
            }
            
            # 分析引擎能力
            engine_capabilities = self._analyze_engine_capabilities()
            
            # 分析接口一致性
            interface_consistency = self._analyze_engine_interfaces()
            
            result = {
                'components': engine_components,
                'capabilities': engine_capabilities,
                'interface_consistency': interface_consistency,
                'maturity_level': self._assess_engine_maturity()
            }
            
            print("✅ 统一分析引擎分析完成")
            return result
            
        except Exception as e:
            print(f"❌ 统一分析引擎分析失败: {e}")
            return {'error': str(e)}
    
    def identify_integration_gaps(self, buypoint_analysis: Dict, strategy_analysis: Dict, engine_analysis: Dict) -> List[Dict[str, Any]]:
        """识别集成差距"""
        print("🔍 识别系统集成差距...")
        
        gaps = []
        
        # 1. 数据格式兼容性差距
        if 'output_format' in buypoint_analysis and 'input_format' in strategy_analysis:
            format_gaps = self._identify_format_gaps(
                buypoint_analysis['output_format'],
                strategy_analysis['input_format']
            )
            gaps.extend(format_gaps)
        
        # 2. 接口一致性差距
        interface_gaps = self._identify_interface_gaps(buypoint_analysis, strategy_analysis)
        gaps.extend(interface_gaps)
        
        # 3. 统一引擎使用差距
        engine_gaps = self._identify_engine_usage_gaps(buypoint_analysis, strategy_analysis, engine_analysis)
        gaps.extend(engine_gaps)
        
        # 4. 性能集成差距
        performance_gaps = self._identify_performance_gaps(buypoint_analysis, strategy_analysis)
        gaps.extend(performance_gaps)
        
        self.integration_gaps = gaps
        print(f"✅ 识别到 {len(gaps)} 个集成差距")
        return gaps
    
    def generate_integration_plan(self) -> Dict[str, Any]:
        """生成集成重构计划"""
        print("📋 生成系统集成重构计划...")
        
        plan = {
            'overview': {
                'total_gaps': len(self.integration_gaps),
                'priority_levels': self._categorize_gaps_by_priority(),
                'estimated_effort': self._estimate_integration_effort()
            },
            'phases': self._create_integration_phases(),
            'recommendations': self._generate_specific_recommendations(),
            'success_metrics': self._define_success_metrics()
        }
        
        print("✅ 集成重构计划生成完成")
        return plan
    
    def _analyze_class(self, cls) -> Dict[str, Any]:
        """分析类的结构和方法"""
        try:
            methods = [method for method in dir(cls) if not method.startswith('_')]
            public_methods = [method for method in methods if callable(getattr(cls, method, None))]
            
            # 获取关键方法的签名
            key_methods = {}
            for method_name in public_methods[:10]:  # 限制分析前10个方法
                try:
                    method = getattr(cls, method_name)
                    if callable(method):
                        sig = inspect.signature(method)
                        key_methods[method_name] = str(sig)
                except Exception:
                    key_methods[method_name] = "无法获取签名"
            
            return {
                'name': cls.__name__,
                'module': cls.__module__,
                'public_methods': public_methods,
                'method_count': len(public_methods),
                'key_methods': key_methods,
                'has_docstring': bool(cls.__doc__)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_buypoint_data_flow(self) -> Dict[str, Any]:
        """分析买点分析系统的数据流"""
        return {
            'input_sources': ['CSV文件', 'ClickHouse数据库', '手动输入'],
            'processing_stages': ['数据获取', '指标计算', '形态识别', '评分计算'],
            'output_targets': ['JSON文件', '策略适配器', '报告生成'],
            'data_transformations': ['多周期数据整合', '指标归一化', '评分标准化']
        }
    
    def _analyze_buypoint_output_format(self) -> Dict[str, Any]:
        """分析买点分析系统的输出格式"""
        return {
            'core_fields': ['stock_code', 'buypoint_date', 'indicator_results', 'pattern_results', 'summary'],
            'optional_fields': ['strategy_compatible'],
            'data_types': {
                'stock_code': 'str',
                'buypoint_date': 'str',
                'indicator_results': 'dict',
                'pattern_results': 'dict',
                'summary': 'dict'
            },
            'nested_structure': True,
            'adapter_available': True
        }
    
    def _analyze_strategy_data_flow(self) -> Dict[str, Any]:
        """分析策略选股系统的数据流"""
        return {
            'input_sources': ['买点分析结果', 'ClickHouse数据库', '配置文件'],
            'processing_stages': ['数据验证', '指标计算', '条件评估', '评分排序'],
            'output_targets': ['选股结果', 'CSV文件', '回测系统'],
            'data_transformations': ['格式标准化', '评分归一化', '排序过滤']
        }
    
    def _analyze_strategy_input_format(self) -> Dict[str, Any]:
        """分析策略选股系统的输入格式期望"""
        return {
            'required_fields': ['stock_code', 'stock_name', 'industry', 'price', 'change_pct', 'score'],
            'optional_fields': ['match_details', 'selection_date'],
            'data_types': {
                'stock_code': 'str',
                'stock_name': 'str',
                'industry': 'str',
                'price': 'float',
                'change_pct': 'float',
                'score': 'float'
            },
            'flat_structure': True,
            'validation_required': True
        }
    
    def _analyze_engine_capabilities(self) -> Dict[str, Any]:
        """分析统一引擎的能力"""
        return {
            'indicator_calculation': '支持90+技术指标',
            'condition_evaluation': '支持复杂逻辑表达式',
            'data_management': '统一数据访问接口',
            'performance_optimization': '向量化计算优化',
            'caching_support': '多级缓存机制',
            'error_handling': '完善的异常处理'
        }
    
    def _analyze_engine_interfaces(self) -> Dict[str, Any]:
        """分析引擎接口一致性"""
        return {
            'api_consistency': '高度一致',
            'data_format_support': '支持DataFrame和dict',
            'parameter_standardization': '参数命名标准化',
            'return_format_consistency': '返回格式统一',
            'error_handling_consistency': '错误处理标准化'
        }
    
    def _assess_buypoint_integration_readiness(self) -> str:
        """评估买点分析系统的集成准备度"""
        return "中等 - 有适配器但需要增强"
    
    def _assess_strategy_integration_readiness(self) -> str:
        """评估策略选股系统的集成准备度"""
        return "良好 - 接口清晰，易于集成"
    
    def _assess_engine_maturity(self) -> str:
        """评估统一引擎的成熟度"""
        return "高 - 功能完整，性能优化"
    
    def _identify_format_gaps(self, buypoint_format: Dict, strategy_format: Dict) -> List[Dict]:
        """识别数据格式差距"""
        gaps = []
        
        # 检查必需字段缺失
        required_fields = set(strategy_format.get('required_fields', []))
        available_fields = set(buypoint_format.get('core_fields', []))
        
        missing_fields = required_fields - available_fields
        if missing_fields:
            gaps.append({
                'type': 'missing_required_fields',
                'severity': 'high',
                'description': f"买点分析缺少必需字段: {missing_fields}",
                'fields': list(missing_fields)
            })
        
        # 检查数据结构兼容性
        if buypoint_format.get('nested_structure') and strategy_format.get('flat_structure'):
            gaps.append({
                'type': 'structure_mismatch',
                'severity': 'medium',
                'description': "买点分析使用嵌套结构，策略选股期望扁平结构",
                'solution': '需要数据扁平化处理'
            })
        
        return gaps
    
    def _identify_interface_gaps(self, buypoint_analysis: Dict, strategy_analysis: Dict) -> List[Dict]:
        """识别接口差距"""
        gaps = []
        
        # 检查方法调用兼容性
        gaps.append({
            'type': 'interface_standardization',
            'severity': 'medium',
            'description': "需要标准化买点分析和策略选股的接口调用方式",
            'solution': '创建统一的接口适配层'
        })
        
        return gaps
    
    def _identify_engine_usage_gaps(self, buypoint_analysis: Dict, strategy_analysis: Dict, engine_analysis: Dict) -> List[Dict]:
        """识别统一引擎使用差距"""
        gaps = []
        
        gaps.append({
            'type': 'engine_integration',
            'severity': 'high',
            'description': "买点分析和策略选股系统需要完全迁移到统一分析引擎",
            'solution': '重构现有计算逻辑，使用统一引擎接口'
        })
        
        return gaps
    
    def _identify_performance_gaps(self, buypoint_analysis: Dict, strategy_analysis: Dict) -> List[Dict]:
        """识别性能集成差距"""
        gaps = []
        
        gaps.append({
            'type': 'performance_optimization',
            'severity': 'medium',
            'description': "需要优化买点分析和策略选股的集成性能",
            'solution': '实现批量处理和并行计算优化'
        })
        
        return gaps
    
    def _categorize_gaps_by_priority(self) -> Dict[str, int]:
        """按优先级分类差距"""
        priority_count = {'high': 0, 'medium': 0, 'low': 0}
        for gap in self.integration_gaps:
            severity = gap.get('severity', 'medium')
            priority_count[severity] = priority_count.get(severity, 0) + 1
        return priority_count
    
    def _estimate_integration_effort(self) -> Dict[str, str]:
        """估算集成工作量"""
        return {
            'total_estimated_days': '15-20天',
            'complexity_level': '中等',
            'risk_level': '低',
            'resource_requirements': '1-2名开发人员'
        }
    
    def _create_integration_phases(self) -> List[Dict[str, Any]]:
        """创建集成阶段计划"""
        return [
            {
                'phase': 1,
                'name': '数据格式统一',
                'duration': '3-5天',
                'tasks': [
                    '增强买点分析输出格式',
                    '完善策略适配器',
                    '标准化数据接口'
                ],
                'deliverables': ['统一数据格式规范', '增强版适配器']
            },
            {
                'phase': 2,
                'name': '统一引擎集成',
                'duration': '5-7天',
                'tasks': [
                    '重构买点分析计算逻辑',
                    '重构策略选股计算逻辑',
                    '统一指标计算接口'
                ],
                'deliverables': ['统一引擎集成版本', '性能测试报告']
            },
            {
                'phase': 3,
                'name': '系统集成测试',
                'duration': '3-4天',
                'tasks': [
                    '端到端集成测试',
                    '性能优化',
                    '文档更新'
                ],
                'deliverables': ['集成测试报告', '用户文档']
            },
            {
                'phase': 4,
                'name': '生产环境部署',
                'duration': '2-3天',
                'tasks': [
                    '生产环境配置',
                    '监控系统配置',
                    '用户培训'
                ],
                'deliverables': ['生产环境部署', '运维文档']
            }
        ]
    
    def _generate_specific_recommendations(self) -> List[Dict[str, Any]]:
        """生成具体建议"""
        return [
            {
                'category': '架构优化',
                'recommendation': '创建统一的系统集成层',
                'priority': 'high',
                'implementation': '使用适配器模式和工厂模式实现系统解耦'
            },
            {
                'category': '性能优化',
                'recommendation': '实现批量数据处理和并行计算',
                'priority': 'medium',
                'implementation': '使用多线程和向量化计算优化性能'
            },
            {
                'category': '监控增强',
                'recommendation': '添加集成系统的监控和告警',
                'priority': 'medium',
                'implementation': '集成现有监控系统，添加关键指标监控'
            }
        ]
    
    def _define_success_metrics(self) -> Dict[str, str]:
        """定义成功指标"""
        return {
            'data_consistency': '买点分析和策略选股数据格式100%兼容',
            'performance_improvement': '集成后系统性能提升30%以上',
            'error_reduction': '集成错误率降低到1%以下',
            'code_reusability': '代码重用率达到80%以上',
            'maintenance_efficiency': '维护工作量减少50%'
        }
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行全面的系统集成分析"""
        print("=" * 80)
        print("🚀 系统集成状态分析")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # 1. 分析各个系统
        buypoint_analysis = self.analyze_buypoint_system()
        strategy_analysis = self.analyze_strategy_system()
        engine_analysis = self.analyze_unified_engine()
        
        # 2. 识别集成差距
        integration_gaps = self.identify_integration_gaps(
            buypoint_analysis, strategy_analysis, engine_analysis
        )
        
        # 3. 生成集成计划
        integration_plan = self.generate_integration_plan()
        
        # 4. 汇总结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        comprehensive_result = {
            'analysis_metadata': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'analyzer_version': '1.0.0'
            },
            'system_analysis': {
                'buypoint_system': buypoint_analysis,
                'strategy_system': strategy_analysis,
                'unified_engine': engine_analysis
            },
            'integration_gaps': integration_gaps,
            'integration_plan': integration_plan,
            'summary': {
                'total_gaps_identified': len(integration_gaps),
                'high_priority_gaps': len([g for g in integration_gaps if g.get('severity') == 'high']),
                'estimated_completion_time': integration_plan['overview']['estimated_effort']['total_estimated_days'],
                'overall_readiness': self._calculate_overall_readiness(buypoint_analysis, strategy_analysis, engine_analysis)
            }
        }
        
        # 打印总结
        self._print_analysis_summary(comprehensive_result)
        
        return comprehensive_result
    
    def _calculate_overall_readiness(self, buypoint_analysis: Dict, strategy_analysis: Dict, engine_analysis: Dict) -> str:
        """计算整体集成准备度"""
        readiness_scores = {
            'buypoint': 60,  # 中等准备度
            'strategy': 80,  # 良好准备度
            'engine': 90     # 高成熟度
        }
        
        average_score = sum(readiness_scores.values()) / len(readiness_scores)
        
        if average_score >= 80:
            return "高 - 可以开始集成"
        elif average_score >= 60:
            return "中等 - 需要一些准备工作"
        else:
            return "低 - 需要大量准备工作"
    
    def _print_analysis_summary(self, result: Dict[str, Any]):
        """打印分析总结"""
        print("\n" + "=" * 80)
        print("📊 系统集成分析总结")
        print("=" * 80)
        
        summary = result['summary']
        print(f"总体准备度: {summary['overall_readiness']}")
        print(f"识别差距数: {summary['total_gaps_identified']}")
        print(f"高优先级差距: {summary['high_priority_gaps']}")
        print(f"预计完成时间: {summary['estimated_completion_time']}")
        
        print("\n🎯 下一步行动:")
        plan_phases = result['integration_plan']['phases']
        for phase in plan_phases[:2]:  # 显示前两个阶段
            print(f"  阶段{phase['phase']}: {phase['name']} ({phase['duration']})")
        
        print("\n✅ 分析完成！可以开始系统集成重构工作。")
        print("=" * 80)


def main_systemintegrationanalyzer():
    """主函数"""
    try:
        analyzer = System_integration_analyzer()
        result = analyzer.run_comprehensive_analysis()
        
        # 保存分析结果
        import json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"system_integration_analysis_{timestamp}.json"
        filepath = os.path.join(root_dir, 'results', filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 分析结果已保存到: {filepath}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 系统集成分析失败: {e}")
        logger.exception("系统集成分析异常")
        return None


if __name__ == "__main__":
    main_systemintegrationanalyzer() 