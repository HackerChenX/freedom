#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
查询性能优化和数据库索引优化系统

分析查询性能、优化索引策略、监控查询效率，确保数据库查询的最佳性能。
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.clickhouse_db import get_clickhouse_db
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class QueryPerformanceAnalyzer:
    """查询性能分析器"""
    
    def __init__(self):
        self.db = None
        self.query_stats = defaultdict(list)
        self.slow_queries = []
        self.optimization_suggestions = []
        
    @exception_handler(reraise=True)
    def initialize_database(self) -> bool:
        """初始化数据库连接"""
        try:
            self.db = get_clickhouse_db()
            
            # 测试连接
            result = self.db.query("SELECT 1 as test")
            if not result.empty:
                logger.info("✅ 查询性能分析器数据库连接成功")
                return True
                
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False
        
        return False
    
    @exception_handler(reraise=True)
    def analyze_table_structure(self) -> Dict[str, Any]:
        """分析表结构和索引情况"""
        try:
            logger.info("🔍 分析数据库表结构...")
            
            # 获取表信息
            table_info_query = """
            SELECT 
                name,
                engine,
                total_rows,
                total_bytes
            FROM system.tables 
            WHERE database = 'stock' AND name = 'stock_info'
            """
            
            table_info = self.db.query(table_info_query)
            
            # 获取列信息
            columns_query = """
            SELECT 
                name,
                type,
                is_in_partition_key,
                is_in_sorting_key,
                is_in_primary_key
            FROM system.columns 
            WHERE database = 'stock' AND table = 'stock_info'
            ORDER BY position
            """
            
            columns_info = self.db.query(columns_query)
            
            # 获取分区信息
            partitions_query = """
            SELECT 
                partition,
                rows,
                bytes_on_disk
            FROM system.parts 
            WHERE database = 'stock' AND table = 'stock_info' AND active
            ORDER BY partition
            LIMIT 10
            """
            
            partitions_info = self.db.query(partitions_query)
            
            result = {
                'table_info': table_info.to_dict('records') if not table_info.empty else [],
                'columns_info': columns_info.to_dict('records') if not columns_info.empty else [],
                'partitions_info': partitions_info.to_dict('records') if not partitions_info.empty else [],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ 表结构分析完成")
            return result
            
        except Exception as e:
            logger.error(f"❌ 表结构分析失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    def benchmark_common_queries(self) -> Dict[str, Any]:
        """基准测试常见查询"""
        try:
            logger.info("📊 执行常见查询基准测试...")
            
            # 定义测试查询
            test_queries = {
                'simple_count': {
                    'query': "SELECT COUNT(*) FROM stock.stock_info",
                    'description': "简单计数查询"
                },
                'count_by_level': {
                    'query': "SELECT level, COUNT(*) FROM stock.stock_info GROUP BY level",
                    'description': "按级别分组计数"
                },
                'single_stock_query': {
                    'query': "SELECT * FROM stock.stock_info WHERE code = '000001' AND level = '日线' ORDER BY date DESC LIMIT 100",
                    'description': "单只股票查询"
                },
                'date_range_query': {
                    'query': "SELECT COUNT(*) FROM stock.stock_info WHERE date >= '2024-01-01' AND date <= '2024-12-31'",
                    'description': "日期范围查询"
                },
                'complex_aggregation': {
                    'query': """
                    SELECT 
                        code,
                        COUNT(*) as record_count,
                        AVG(close) as avg_price,
                        MAX(high) as max_high,
                        MIN(low) as min_low
                    FROM stock.stock_info 
                    WHERE level = '日线' AND date >= '2024-06-01'
                    GROUP BY code 
                    HAVING record_count > 10
                    ORDER BY avg_price DESC 
                    LIMIT 100
                    """,
                    'description': "复杂聚合查询"
                },
                'join_like_query': {
                    'query': """
                    SELECT DISTINCT s1.code
                    FROM stock.stock_info s1
                    WHERE s1.level = '日线' 
                    AND s1.date >= '2024-01-01'
                    AND s1.code IN (
                        SELECT s2.code 
                        FROM stock.stock_info s2 
                        WHERE s2.level = '日线' 
                        AND s2.date >= '2024-06-01'
                        GROUP BY s2.code 
                        HAVING COUNT(*) > 50
                    )
                    LIMIT 1000
                    """,
                    'description': "子查询连接"
                }
            }
            
            results = {}
            
            for query_name, query_info in test_queries.items():
                logger.info(f"  测试查询: {query_info['description']}")
                
                # 执行多次测试取平均值
                execution_times = []
                for i in range(3):
                    start_time = time.time()
                    try:
                        result = self.db.query(query_info['query'])
                        execution_time = time.time() - start_time
                        execution_times.append(execution_time)
                        
                        # 记录查询统计
                        self.query_stats[query_name].append({
                            'execution_time': execution_time,
                            'rows_returned': len(result) if not result.empty else 0,
                            'timestamp': time.time()
                        })
                        
                    except Exception as e:
                        logger.warning(f"查询 {query_name} 第 {i+1} 次执行失败: {e}")
                        execution_times.append(float('inf'))
                
                # 计算统计信息
                valid_times = [t for t in execution_times if t != float('inf')]
                if valid_times:
                    results[query_name] = {
                        'description': query_info['description'],
                        'average_time': np.mean(valid_times),
                        'min_time': np.min(valid_times),
                        'max_time': np.max(valid_times),
                        'executions': len(valid_times),
                        'failed_executions': len(execution_times) - len(valid_times)
                    }
                    
                    # 标记慢查询
                    if results[query_name]['average_time'] > 5.0:
                        self.slow_queries.append({
                            'query_name': query_name,
                            'description': query_info['description'],
                            'average_time': results[query_name]['average_time'],
                            'query': query_info['query']
                        })
                else:
                    results[query_name] = {
                        'description': query_info['description'],
                        'error': "所有执行均失败"
                    }
            
            logger.info(f"✅ 基准测试完成，发现 {len(self.slow_queries)} 个慢查询")
            return results
            
        except Exception as e:
            logger.error(f"❌ 基准测试失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    def analyze_query_execution_plans(self) -> Dict[str, Any]:
        """分析查询执行计划"""
        try:
            logger.info("🔍 分析查询执行计划...")
            
            execution_plans = {}
            
            # 分析慢查询的执行计划
            for slow_query in self.slow_queries[:3]:  # 只分析前3个慢查询
                query_name = slow_query['query_name']
                query = slow_query['query']
                
                try:
                    # 获取查询执行计划
                    explain_query = f"EXPLAIN {query}"
                    plan_result = self.db.query(explain_query)
                    
                    execution_plans[query_name] = {
                        'description': slow_query['description'],
                        'average_time': slow_query['average_time'],
                        'execution_plan': plan_result.to_dict('records') if not plan_result.empty else []
                    }
                    
                except Exception as e:
                    logger.warning(f"获取查询 {query_name} 执行计划失败: {e}")
                    execution_plans[query_name] = {
                        'description': slow_query['description'],
                        'average_time': slow_query['average_time'],
                        'error': str(e)
                    }
            
            logger.info(f"✅ 执行计划分析完成，分析了 {len(execution_plans)} 个查询")
            return execution_plans
            
        except Exception as e:
            logger.error(f"❌ 执行计划分析失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    def test_index_effectiveness(self) -> Dict[str, Any]:
        """测试索引有效性"""
        try:
            logger.info("📊 测试索引有效性...")
            
            # 测试不同查询条件的性能
            index_tests = {
                'code_index': {
                    'with_index': "SELECT COUNT(*) FROM stock.stock_info WHERE code = '000001'",
                    'description': "股票代码索引测试"
                },
                'date_index': {
                    'with_index': "SELECT COUNT(*) FROM stock.stock_info WHERE date = '2024-01-01'",
                    'description': "日期索引测试"
                },
                'level_index': {
                    'with_index': "SELECT COUNT(*) FROM stock.stock_info WHERE level = '日线'",
                    'description': "级别索引测试"
                },
                'compound_index': {
                    'with_index': "SELECT COUNT(*) FROM stock.stock_info WHERE code = '000001' AND level = '日线'",
                    'description': "复合索引测试"
                },
                'range_query': {
                    'with_index': "SELECT COUNT(*) FROM stock.stock_info WHERE date >= '2024-01-01' AND date <= '2024-01-31'",
                    'description': "范围查询测试"
                }
            }
            
            results = {}
            
            for test_name, test_info in index_tests.items():
                logger.info(f"  测试: {test_info['description']}")
                
                # 执行查询多次取平均值
                execution_times = []
                for i in range(5):
                    start_time = time.time()
                    try:
                        result = self.db.query(test_info['with_index'])
                        execution_time = time.time() - start_time
                        execution_times.append(execution_time)
                    except Exception as e:
                        logger.warning(f"索引测试 {test_name} 第 {i+1} 次执行失败: {e}")
                
                if execution_times:
                    results[test_name] = {
                        'description': test_info['description'],
                        'average_time': np.mean(execution_times),
                        'min_time': np.min(execution_times),
                        'max_time': np.max(execution_times),
                        'std_time': np.std(execution_times),
                        'executions': len(execution_times)
                    }
                    
                    # 评估性能
                    avg_time = results[test_name]['average_time']
                    if avg_time < 0.1:
                        results[test_name]['performance'] = 'Excellent'
                    elif avg_time < 0.5:
                        results[test_name]['performance'] = 'Good'
                    elif avg_time < 2.0:
                        results[test_name]['performance'] = 'Fair'
                    else:
                        results[test_name]['performance'] = 'Poor'
            
            logger.info("✅ 索引有效性测试完成")
            return results
            
        except Exception as e:
            logger.error(f"❌ 索引有效性测试失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    def generate_optimization_suggestions(self, table_analysis: Dict[str, Any], 
                                        query_benchmark: Dict[str, Any],
                                        index_tests: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        try:
            logger.info("💡 生成查询性能优化建议...")
            
            suggestions = []
            
            # 基于慢查询生成建议
            if self.slow_queries:
                suggestions.append(f"发现 {len(self.slow_queries)} 个慢查询，建议优化：")
                for slow_query in self.slow_queries[:3]:
                    suggestions.append(f"  - {slow_query['description']}: {slow_query['average_time']:.2f}秒")
            
            # 基于索引测试生成建议
            poor_performance_tests = [
                test_name for test_name, test_data in index_tests.items()
                if test_data.get('performance') in ['Poor', 'Fair']
            ]
            
            if poor_performance_tests:
                suggestions.append("以下查询类型性能较差，建议优化索引：")
                for test_name in poor_performance_tests:
                    test_data = index_tests[test_name]
                    suggestions.append(f"  - {test_data['description']}: {test_data['average_time']:.3f}秒 ({test_data['performance']})")
            
            # 基于表结构生成建议
            if table_analysis.get('table_info'):
                table_info = table_analysis['table_info'][0] if table_analysis['table_info'] else {}
                total_rows = table_info.get('total_rows', 0)
                
                if total_rows > 10000000:  # 1000万行以上
                    suggestions.append("表数据量较大，建议考虑：")
                    suggestions.append("  - 实施分区策略优化查询性能")
                    suggestions.append("  - 考虑数据压缩以减少存储空间")
                    suggestions.append("  - 定期清理历史数据")
            
            # 通用优化建议
            general_suggestions = [
                "考虑在经常查询的字段上创建索引",
                "优化查询语句，避免使用SELECT *",
                "对于大数据量查询，考虑使用LIMIT限制结果集",
                "定期分析查询模式，调整索引策略",
                "监控查询性能，及时发现性能瓶颈"
            ]
            
            suggestions.extend(general_suggestions)
            
            self.optimization_suggestions = suggestions
            logger.info(f"✅ 生成了 {len(suggestions)} 条优化建议")
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ 生成优化建议失败: {e}")
            return []


class QueryPerformanceOptimizationService:
    """查询性能优化器"""
    
    def __init__(self):
        self.analyzer = QueryPerformanceAnalyzer()
        self.optimization_results = {}
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1800.0)
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """运行综合查询优化分析"""
        start_time = datetime.now()
        
        comprehensive_result = {
            'test_start_time': start_time.isoformat(),
            'test_end_time': None,
            'total_test_time': 0.0,
            'database_connection': False,
            'table_analysis': {},
            'query_benchmark': {},
            'execution_plans': {},
            'index_effectiveness': {},
            'optimization_suggestions': [],
            'performance_grade': 'Unknown',
            'overall_recommendations': []
        }
        
        try:
            logger.info("🚀 开始查询性能优化综合分析...")
            
            # 1. 初始化数据库连接
            logger.info("📝 Step 1: 初始化数据库连接...")
            comprehensive_result['database_connection'] = self.analyzer.initialize_database()
            
            if not comprehensive_result['database_connection']:
                comprehensive_result['overall_recommendations'].append("数据库连接失败，需要检查连接配置")
                return comprehensive_result
            
            # 2. 分析表结构
            logger.info("📝 Step 2: 分析数据库表结构...")
            comprehensive_result['table_analysis'] = self.analyzer.analyze_table_structure()
            
            # 3. 执行查询基准测试
            logger.info("📝 Step 3: 执行查询基准测试...")
            comprehensive_result['query_benchmark'] = self.analyzer.benchmark_common_queries()
            
            # 4. 分析执行计划
            logger.info("📝 Step 4: 分析查询执行计划...")
            comprehensive_result['execution_plans'] = self.analyzer.analyze_query_execution_plans()
            
            # 5. 测试索引有效性
            logger.info("📝 Step 5: 测试索引有效性...")
            comprehensive_result['index_effectiveness'] = self.analyzer.test_index_effectiveness()
            
            # 6. 生成优化建议
            logger.info("📝 Step 6: 生成优化建议...")
            comprehensive_result['optimization_suggestions'] = self.analyzer.generate_optimization_suggestions(
                comprehensive_result['table_analysis'],
                comprehensive_result['query_benchmark'],
                comprehensive_result['index_effectiveness']
            )
            
            # 7. 综合评估
            logger.info("📝 Step 7: 综合性能评估...")
            self._evaluate_overall_performance(comprehensive_result)
            
            end_time = datetime.now()
            comprehensive_result['test_end_time'] = end_time.isoformat()
            comprehensive_result['total_test_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ 查询性能优化分析完成，总耗时: {comprehensive_result['total_test_time']:.1f}秒")
            logger.info(f"📊 性能评级: {comprehensive_result['performance_grade']}")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ 综合优化分析失败: {e}")
            comprehensive_result['overall_recommendations'].append(f"分析过程异常: {str(e)}")
            return comprehensive_result
    
    def _evaluate_overall_performance(self, result: Dict[str, Any]) -> None:
        """评估整体性能"""
        score = 0
        recommendations = []
        
        # 评估查询基准测试
        query_benchmark = result.get('query_benchmark', {})
        if query_benchmark:
            # 统计查询性能
            excellent_queries = 0
            good_queries = 0
            fair_queries = 0
            poor_queries = 0
            
            for query_name, query_data in query_benchmark.items():
                if 'average_time' in query_data:
                    avg_time = query_data['average_time']
                    if avg_time < 0.5:
                        excellent_queries += 1
                    elif avg_time < 2.0:
                        good_queries += 1
                    elif avg_time < 5.0:
                        fair_queries += 1
                    else:
                        poor_queries += 1
            
            total_queries = len(query_benchmark)
            if total_queries > 0:
                excellent_ratio = excellent_queries / total_queries
                good_ratio = good_queries / total_queries
                
                if excellent_ratio >= 0.8:
                    score += 40
                    recommendations.append("查询性能表现优秀")
                elif excellent_ratio + good_ratio >= 0.7:
                    score += 30
                    recommendations.append("查询性能表现良好")
                elif excellent_ratio + good_ratio >= 0.5:
                    score += 20
                    recommendations.append("查询性能表现一般，有优化空间")
                else:
                    score += 10
                    recommendations.append("查询性能较差，需要重点优化")
        
        # 评估索引有效性
        index_tests = result.get('index_effectiveness', {})
        if index_tests:
            excellent_indexes = 0
            good_indexes = 0
            
            for test_name, test_data in index_tests.items():
                performance = test_data.get('performance', 'Unknown')
                if performance == 'Excellent':
                    excellent_indexes += 1
                elif performance == 'Good':
                    good_indexes += 1
            
            total_indexes = len(index_tests)
            if total_indexes > 0:
                excellent_ratio = excellent_indexes / total_indexes
                good_ratio = good_indexes / total_indexes
                
                if excellent_ratio >= 0.7:
                    score += 30
                    recommendations.append("索引配置优秀")
                elif excellent_ratio + good_ratio >= 0.6:
                    score += 25
                    recommendations.append("索引配置良好")
                elif excellent_ratio + good_ratio >= 0.4:
                    score += 15
                    recommendations.append("索引配置需要优化")
                else:
                    score += 5
                    recommendations.append("索引配置需要重新设计")
        
        # 评估慢查询数量
        slow_queries_count = len(self.analyzer.slow_queries)
        if slow_queries_count == 0:
            score += 30
            recommendations.append("未发现慢查询")
        elif slow_queries_count <= 2:
            score += 20
            recommendations.append("少量慢查询，建议优化")
        elif slow_queries_count <= 5:
            score += 10
            recommendations.append("存在多个慢查询，需要优化")
        else:
            score += 0
            recommendations.append("存在大量慢查询，需要系统性优化")
        
        # 确定性能评级
        if score >= 85:
            result['performance_grade'] = 'A+'
        elif score >= 75:
            result['performance_grade'] = 'A'
        elif score >= 65:
            result['performance_grade'] = 'B+'
        elif score >= 55:
            result['performance_grade'] = 'B'
        elif score >= 45:
            result['performance_grade'] = 'C+'
        elif score >= 35:
            result['performance_grade'] = 'C'
        else:
            result['performance_grade'] = 'D'
        
        result['overall_recommendations'] = recommendations
    
    def generate_report(self, result: Dict[str, Any]) -> str:
        """生成优化报告"""
        report_lines = [
            "=" * 80,
            "查询性能优化综合报告",
            "=" * 80,
            f"分析时间: {result.get('test_start_time', 'Unknown')}",
            f"分析耗时: {result.get('total_test_time', 0):.1f}秒",
            f"性能评级: {result.get('performance_grade', 'Unknown')}",
            "",
            "📊 详细分析结果:",
            "-" * 40,
        ]
        
        # 数据库连接状态
        conn_status = "✅ 成功" if result.get('database_connection', False) else "❌ 失败"
        report_lines.append(f"数据库连接: {conn_status}")
        
        # 表结构分析
        table_analysis = result.get('table_analysis', {})
        if table_analysis and table_analysis.get('table_info'):
            table_info = table_analysis['table_info'][0]
            report_lines.extend([
                "",
                "📋 表结构分析:",
                f"  表引擎: {table_info.get('engine', 'Unknown')}",
                f"  总行数: {table_info.get('total_rows', 0):,}",
                f"  数据大小: {table_info.get('total_bytes', 0):,} 字节",
                f"  列数量: {len(table_analysis.get('columns_info', []))}"
            ])
        
        # 查询基准测试
        query_benchmark = result.get('query_benchmark', {})
        if query_benchmark:
            report_lines.extend([
                "",
                "📈 查询基准测试结果:"
            ])
            for query_name, query_data in query_benchmark.items():
                if 'average_time' in query_data:
                    report_lines.append(
                        f"  {query_data['description']}: "
                        f"平均 {query_data['average_time']:.3f}秒 "
                        f"(范围: {query_data['min_time']:.3f}-{query_data['max_time']:.3f}秒)"
                    )
        
        # 索引有效性测试
        index_tests = result.get('index_effectiveness', {})
        if index_tests:
            report_lines.extend([
                "",
                "🔍 索引有效性测试:"
            ])
            for test_name, test_data in index_tests.items():
                performance = test_data.get('performance', 'Unknown')
                avg_time = test_data.get('average_time', 0)
                report_lines.append(
                    f"  {test_data['description']}: {avg_time:.3f}秒 ({performance})"
                )
        
        # 慢查询
        if self.analyzer.slow_queries:
            report_lines.extend([
                "",
                f"🐌 慢查询 ({len(self.analyzer.slow_queries)} 个):"
            ])
            for slow_query in self.analyzer.slow_queries[:5]:  # 只显示前5个
                report_lines.append(
                    f"  {slow_query['description']}: {slow_query['average_time']:.2f}秒"
                )
        
        # 优化建议
        optimization_suggestions = result.get('optimization_suggestions', [])
        if optimization_suggestions:
            report_lines.extend([
                "",
                "💡 优化建议:"
            ])
            for i, suggestion in enumerate(optimization_suggestions[:10], 1):  # 只显示前10个
                report_lines.append(f"{i}. {suggestion}")
        
        # 整体建议
        overall_recommendations = result.get('overall_recommendations', [])
        if overall_recommendations:
            report_lines.extend([
                "",
                "🎯 整体评估:"
            ])
            for i, rec in enumerate(overall_recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    print("🚀 启动查询性能优化综合分析...")
    
    optimizer = QueryPerformanceOptimizationService()
    
    try:
        # 运行综合优化分析
        result = optimizer.run_comprehensive_optimization()
        
        # 生成报告
        report = optimizer.generate_report(result)
        print(report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"query_performance_optimization_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 返回测试结果
        if result.get('performance_grade') in ['A+', 'A', 'B+']:
            print("🎉 查询性能优化分析完成，性能表现良好！")
            return 0
        else:
            print("⚠️  查询性能有优化空间，建议根据报告进行改进。")
            return 1
            
    except Exception as e:
        print(f"❌ 分析执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 