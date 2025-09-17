"""
数据一致性验证器
确保多次查询结果的一致性
"""

import os
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger
from config.unified_config_manager import get_config
from enums.test_status import TestStatus
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


@dataclass
class ConsistencyTestCase:
    """一致性测试用例"""
    test_name: str
    test_description: str
    query_function: Callable
    query_params: Dict[str, Any]
    expected_consistency: str
    tolerance_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyTestResult:
    """一致性测试结果"""
    test_name: str
    test_case_name: str
    total_executions: int
    consistent_results: int
    inconsistent_results: int
    consistency_rate: float
    execution_times: List[float]
    result_hashes: List[str]
    data_variations: List[Dict[str, Any]]
    test_status: TestStatus
    error_message: Optional[str] = None
    detailed_analysis: Optional[Dict[str, Any]] = None


@dataclass
class DataConsistencyReport:
    """数据一致性报告"""
    report_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_consistency_score: float
    test_results: List[ConsistencyTestResult] = field(default_factory=list)
    consistency_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class DataHasher:
    """数据哈希计算器"""
    
    @staticmethod
    def hash_dataframe(df: pd.DataFrame, ignore_index: bool = True) -> str:
        """计算DataFrame的哈希值"""
        if df.empty:
            return "empty_dataframe"
        
        # 排序以确保一致性
        if ignore_index:
            df_sorted = df.sort_values(by=list(df.columns), ignore_index=True)
        else:
            df_sorted = df.sort_values(by=list(df.columns))
        
        # 转换为字符串并计算哈希
        data_string = df_sorted.to_string(index=False)
        return hashlib.md5(data_string.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_dict(data: dict) -> str:
        """计算字典的哈希值"""
        # 按键排序以确保一致性
        sorted_items = sorted(data.items())
        data_string = str(sorted_items)
        return hashlib.md5(data_string.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_list(data: list) -> str:
        """计算列表的哈希值"""
        data_string = str(sorted(data) if all(isinstance(x, (int, float, str)) for x in data) else data)
        return hashlib.md5(data_string.encode('utf-8')).hexdigest()


class DataVariationAnalyzer:
    """数据差异分析器"""
    
    @staticmethod
    def analyze_dataframe_differences(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """分析两个DataFrame之间的差异"""
        differences = {
            'shape_difference': df1.shape != df2.shape,
            'column_difference': set(df1.columns) != set(df2.columns),
            'data_differences': []
        }
        
        if df1.shape == df2.shape and set(df1.columns) == set(df2.columns):
            # 比较数据内容
            try:
                comparison = df1.compare(df2)
                if not comparison.empty:
                    differences['data_differences'] = comparison.to_dict()
            except Exception as e:
                differences['comparison_error'] = str(e)
                
                # 手动比较
                for col in df1.columns:
                    if col in df2.columns:
                        col_diff = (df1[col] != df2[col]).sum()
                        if col_diff > 0:
                            differences['data_differences'].append({
                                'column': col,
                                'different_rows': int(col_diff)
                            })
        
        return differences
    
    @staticmethod
    def analyze_dict_differences(dict1: dict, dict2: dict) -> Dict[str, Any]:
        """分析两个字典之间的差异"""
        differences = {
            'key_differences': {
                'only_in_first': set(dict1.keys()) - set(dict2.keys()),
                'only_in_second': set(dict2.keys()) - set(dict1.keys())
            },
            'value_differences': {}
        }
        
        common_keys = set(dict1.keys()) & set(dict2.keys())
        for key in common_keys:
            if dict1[key] != dict2[key]:
                differences['value_differences'][key] = {
                    'first': dict1[key],
                    'second': dict2[key]
                }
        
        return differences
    
    @staticmethod
    def calculate_similarity_score(data1: Any, data2: Any) -> float:
        """计算两个数据结构的相似度"""
        if type(data1) != type(data2):
            return 0.0
        
        if isinstance(data1, pd.DataFrame):
            if data1.shape != data2.shape:
                return 0.0
            
            if data1.empty and data2.empty:
                return 1.0
            
            try:
                # 比较数据内容
                equal_mask = data1 == data2
                similarity = equal_mask.sum().sum() / (data1.shape[0] * data1.shape[1])
                return float(similarity)
            except:
                return 0.0 if not data1.equals(data2) else 1.0
        
        elif isinstance(data1, dict):
            if len(data1) == 0 and len(data2) == 0:
                return 1.0
            
            common_keys = set(data1.keys()) & set(data2.keys())
            all_keys = set(data1.keys()) | set(data2.keys())
            
            if len(all_keys) == 0:
                return 1.0
            
            key_similarity = len(common_keys) / len(all_keys)
            
            value_matches = 0
            for key in common_keys:
                if data1[key] == data2[key]:
                    value_matches += 1
            
            value_similarity = value_matches / len(common_keys) if common_keys else 0
            
            return (key_similarity + value_similarity) / 2
        
        else:
            return 1.0 if data1 == data2 else 0.0


class DataConsistencyValidator:
    """数据一致性验证器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.hasher = DataHasher()
        self.analyzer = DataVariationAnalyzer()
        self.test_cases = self._define_consistency_test_cases()
    
    def _define_consistency_test_cases(self) -> Dict[str, ConsistencyTestCase]:
        """定义一致性测试用例"""
        return {
            'stock_data_query': ConsistencyTestCase(
                test_name='stock_data_query_consistency',
                test_description='股票数据查询一致性测试',
                query_function=self._mock_stock_data_query,
                query_params={
                    'stock_code': '000001',
                    'start_date': '2023-01-01',
                    'end_date': '2023-12-31'
                },
                expected_consistency='完全一致',
                tolerance_settings={'precision': 1e-6}
            ),
            'indicator_calculation': ConsistencyTestCase(
                test_name='indicator_calculation_consistency',
                test_description='技术指标计算一致性测试',
                query_function=self._mock_indicator_calculation,
                query_params={
                    'stock_code': '000001',
                    'indicator_type': 'MA',
                    'period': 20
                },
                expected_consistency='数值一致',
                tolerance_settings={'precision': 1e-4}
            ),
            'strategy_analysis': ConsistencyTestCase(
                test_name='strategy_analysis_consistency',
                test_description='策略分析结果一致性测试',
                query_function=self._mock_strategy_analysis,
                query_params={
                    'stock_code': '000001',
                    'strategy_type': 'dual_ma'
                },
                expected_consistency='逻辑一致',
                tolerance_settings={'score_tolerance': 0.01}
            ),
            'batch_processing': ConsistencyTestCase(
                test_name='batch_processing_consistency',
                test_description='批量处理一致性测试',
                query_function=self._mock_batch_processing,
                query_params={
                    'stock_codes': ['000001', '000002', '600000'],
                    'operation': 'analyze'
                },
                expected_consistency='顺序无关',
                tolerance_settings={'order_sensitive': False}
            ),
            'concurrent_access': ConsistencyTestCase(
                test_name='concurrent_access_consistency',
                test_description='并发访问一致性测试',
                query_function=self._mock_concurrent_operation,
                query_params={
                    'operation_type': 'read',
                    'resource_id': 'shared_data'
                },
                expected_consistency='读取一致',
                tolerance_settings={'concurrent_safe': True}
            )
        }
    
    def _mock_stock_data_query(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """模拟股票数据查询"""
        # 模拟稳定的数据查询
        np.random.seed(hash(stock_code + start_date + end_date) % 1000)
        
        dates = pd.date_range(start_date, end_date, freq='D')[:100]  # 限制数据量
        data = []
        
        base_price = 10.0
        for i, date in enumerate(dates):
            price = base_price + i * 0.01  # 确定性价格变化
            data.append({
                'code': stock_code,
                'date': date.strftime('%Y-%m-%d'),
                'open': round(price, 2),
                'high': round(price * 1.05, 2),
                'low': round(price * 0.95, 2),
                'close': round(price + 0.01, 2),
                'volume': 1000000 + i * 1000
            })
        
        return pd.DataFrame(data)
    
    def _mock_indicator_calculation(self, stock_code: str, indicator_type: str, period: int) -> Dict[str, Any]:
        """模拟技术指标计算"""
        # 获取基础数据
        data = self._mock_stock_data_query(stock_code, '2023-01-01', '2023-12-31')
        
        if indicator_type == 'MA':
            # 计算移动平均
            ma_values = data['close'].rolling(window=period).mean().dropna()
            return {
                'indicator': 'MA',
                'period': period,
                'values': ma_values.tolist(),
                'latest_value': float(ma_values.iloc[-1]) if not ma_values.empty else None
            }
        else:
            return {
                'indicator': indicator_type,
                'period': period,
                'values': [],
                'latest_value': None
            }
    
    def _mock_strategy_analysis(self, stock_code: str, strategy_type: str) -> Dict[str, Any]:
        """模拟策略分析"""
        # 确定性的策略分析结果
        score_seed = hash(stock_code + strategy_type) % 100
        
        return {
            'stock_code': stock_code,
            'strategy': strategy_type,
            'signal': 'BUY' if score_seed > 60 else 'HOLD' if score_seed > 30 else 'SELL',
            'score': score_seed,
            'confidence': min(score_seed / 100.0, 1.0),
            'reason': f'技术分析评分: {score_seed}'
        }
    
    def _mock_batch_processing(self, stock_codes: List[str], operation: str) -> List[Dict[str, Any]]:
        """模拟批量处理"""
        results = []
        for code in stock_codes:
            result = self._mock_strategy_analysis(code, 'batch_analysis')
            results.append(result)
        
        return results
    
    def _mock_concurrent_operation(self, operation_type: str, resource_id: str) -> Dict[str, Any]:
        """模拟并发操作"""
        # 模拟读取共享数据
        return {
            'operation': operation_type,
            'resource': resource_id,
            'timestamp': time.time(),
            'data': f'shared_data_value_{hash(resource_id) % 1000}',
            'version': 1
        }
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def test_data_query_consistency(self, test_case: ConsistencyTestCase, execution_count: int = 5) -> ConsistencyTestResult:
        """测试数据查询一致性"""
        test_start_time = time.time()
        
        results = []
        execution_times = []
        result_hashes = []
        data_variations = []
        
        # 多次执行相同查询
        for i in range(execution_count):
            execution_start = time.time()
            
            try:
                result = test_case.query_function(**test_case.query_params)
                execution_time = time.time() - execution_start
                
                results.append(result)
                execution_times.append(execution_time)
                
                # 计算结果哈希
                if isinstance(result, pd.DataFrame):
                    result_hash = self.hasher.hash_dataframe(result)
                elif isinstance(result, dict):
                    result_hash = self.hasher.hash_dict(result)
                elif isinstance(result, list):
                    result_hash = self.hasher.hash_list(result)
                else:
                    result_hash = hashlib.md5(str(result).encode('utf-8')).hexdigest()
                
                result_hashes.append(result_hash)
                
            except Exception as e:
                self.logger.error(f"查询执行失败 (第{i+1}次): {e}")
                execution_times.append(time.time() - execution_start)
                result_hashes.append("error")
                results.append(None)
        
        # 分析一致性
        unique_hashes = set(result_hashes)
        consistent_results = sum(1 for h in result_hashes if h == result_hashes[0] and h != "error")
        inconsistent_results = execution_count - consistent_results
        consistency_rate = consistent_results / execution_count if execution_count > 0 else 0
        
        # 分析数据差异
        if len(results) > 1:
            base_result = results[0]
            for i, result in enumerate(results[1:], 1):
                if base_result is not None and result is not None:
                    if isinstance(base_result, pd.DataFrame) and isinstance(result, pd.DataFrame):
                        differences = self.analyzer.analyze_dataframe_differences(base_result, result)
                        similarity = self.analyzer.calculate_similarity_score(base_result, result)
                    elif isinstance(base_result, dict) and isinstance(result, dict):
                        differences = self.analyzer.analyze_dict_differences(base_result, result)
                        similarity = self.analyzer.calculate_similarity_score(base_result, result)
                    else:
                        differences = {'different': base_result != result}
                        similarity = 1.0 if base_result == result else 0.0
                    
                    data_variations.append({
                        'comparison_index': i,
                        'differences': differences,
                        'similarity_score': similarity
                    })
        
        # 确定测试状态
        if consistency_rate >= 0.9:
            test_status = TestStatus.PASSED
        elif consistency_rate >= 0.7:
            test_status = TestStatus.WARNING
        else:
            test_status = TestStatus.FAILED
        
        # 详细分析
        detailed_analysis = {
            'unique_hash_count': len(unique_hashes),
            'hash_distribution': {hash_val: result_hashes.count(hash_val) for hash_val in unique_hashes},
            'execution_time_stats': {
                'min': min(execution_times) if execution_times else 0,
                'max': max(execution_times) if execution_times else 0,
                'avg': sum(execution_times) / len(execution_times) if execution_times else 0,
                'std': np.std(execution_times) if execution_times else 0
            },
            'consistency_score': consistency_rate * 100
        }
        
        return ConsistencyTestResult(
            test_name=test_case.test_name,
            test_case_name=test_case.test_description,
            total_executions=execution_count,
            consistent_results=consistent_results,
            inconsistent_results=inconsistent_results,
            consistency_rate=consistency_rate,
            execution_times=execution_times,
            result_hashes=result_hashes,
            data_variations=data_variations,
            test_status=test_status,
            detailed_analysis=detailed_analysis
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=120.0)
    def test_concurrent_consistency(self, test_case: ConsistencyTestCase, thread_count: int = 5) -> ConsistencyTestResult:
        """测试并发一致性"""
        test_start_time = time.time()
        
        results = []
        execution_times = []
        result_hashes = []
        data_variations = []
        
        def execute_query():
            execution_start = time.time()
            try:
                result = test_case.query_function(**test_case.query_params)
                execution_time = time.time() - execution_start
                return result, execution_time, None
            except Exception as e:
                execution_time = time.time() - execution_start
                return None, execution_time, str(e)
        
        # 并发执行查询
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(execute_query) for _ in range(thread_count)]
            
            for future in as_completed(futures):
                result, execution_time, error = future.result()
                
                results.append(result)
                execution_times.append(execution_time)
                
                if error:
                    result_hashes.append("error")
                    self.logger.error(f"并发查询失败: {error}")
                else:
                    # 计算结果哈希
                    if isinstance(result, pd.DataFrame):
                        result_hash = self.hasher.hash_dataframe(result)
                    elif isinstance(result, dict):
                        result_hash = self.hasher.hash_dict(result)
                    elif isinstance(result, list):
                        result_hash = self.hasher.hash_list(result)
                    else:
                        result_hash = hashlib.md5(str(result).encode('utf-8')).hexdigest()
                    
                    result_hashes.append(result_hash)
        
        # 分析一致性
        error_count = result_hashes.count("error")
        valid_hashes = [h for h in result_hashes if h != "error"]
        unique_hashes = set(valid_hashes)
        
        if valid_hashes:
            most_common_hash = max(valid_hashes, key=valid_hashes.count)
            consistent_results = valid_hashes.count(most_common_hash)
            inconsistent_results = len(valid_hashes) - consistent_results
            consistency_rate = consistent_results / len(valid_hashes)
        else:
            consistent_results = 0
            inconsistent_results = thread_count
            consistency_rate = 0
        
        # 确定测试状态
        if error_count > 0:
            test_status = TestStatus.ERROR
        elif consistency_rate >= 0.9:
            test_status = TestStatus.PASSED
        elif consistency_rate >= 0.7:
            test_status = TestStatus.WARNING
        else:
            test_status = TestStatus.FAILED
        
        # 详细分析
        detailed_analysis = {
            'thread_count': thread_count,
            'error_count': error_count,
            'unique_hash_count': len(unique_hashes),
            'hash_distribution': {hash_val: valid_hashes.count(hash_val) for hash_val in unique_hashes},
            'execution_time_stats': {
                'min': min(execution_times) if execution_times else 0,
                'max': max(execution_times) if execution_times else 0,
                'avg': sum(execution_times) / len(execution_times) if execution_times else 0,
                'std': np.std(execution_times) if execution_times else 0
            },
            'consistency_score': consistency_rate * 100,
            'concurrent_safety': error_count == 0 and consistency_rate >= 0.9
        }
        
        return ConsistencyTestResult(
            test_name=test_case.test_name + "_concurrent",
            test_case_name=test_case.test_description + " (并发)",
            total_executions=thread_count,
            consistent_results=consistent_results,
            inconsistent_results=inconsistent_results,
            consistency_rate=consistency_rate,
            execution_times=execution_times,
            result_hashes=result_hashes,
            data_variations=data_variations,
            test_status=test_status,
            detailed_analysis=detailed_analysis
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=300.0)
    def run_all_consistency_tests(self) -> DataConsistencyReport:
        """运行所有一致性测试"""
        report = DataConsistencyReport(
            report_name="data_consistency_validation",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            overall_consistency_score=0.0
        )
        
        all_test_results = []
        
        # 串行一致性测试
        for test_case_name, test_case in self.test_cases.items():
            try:
                self.logger.info(f"开始一致性测试: {test_case_name}")
                
                # 串行测试
                serial_result = self.test_data_query_consistency(test_case, execution_count=5)
                all_test_results.append(serial_result)
                
                # 并发测试（仅对支持并发的测试用例）
                if test_case.tolerance_settings.get('concurrent_safe', False):
                    concurrent_result = self.test_concurrent_consistency(test_case, thread_count=3)
                    all_test_results.append(concurrent_result)
                
                self.logger.info(f"一致性测试 {test_case_name} 完成")
                
            except Exception as e:
                self.logger.error(f"一致性测试 {test_case_name} 失败: {e}")
                
                # 创建失败结果
                failed_result = ConsistencyTestResult(
                    test_name=test_case.test_name,
                    test_case_name=test_case.test_description,
                    total_executions=0,
                    consistent_results=0,
                    inconsistent_results=1,
                    consistency_rate=0.0,
                    execution_times=[],
                    result_hashes=[],
                    data_variations=[],
                    test_status=TestStatus.ERROR,
                    error_message=str(e)
                )
                all_test_results.append(failed_result)
        
        # 汇总结果
        report.test_results = all_test_results
        report.total_tests = len(all_test_results)
        
        for result in all_test_results:
            if result.test_status == TestStatus.PASSED:
                report.passed_tests += 1
            elif result.test_status in [TestStatus.FAILED, TestStatus.ERROR]:
                report.failed_tests += 1
        
        # 计算总体一致性评分
        if all_test_results:
            consistency_scores = [
                result.consistency_rate * 100 
                for result in all_test_results 
                if result.consistency_rate is not None
            ]
            report.overall_consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # 一致性分析
        report.consistency_analysis = {
            'average_consistency_rate': report.overall_consistency_score,
            'tests_with_perfect_consistency': sum(1 for r in all_test_results if r.consistency_rate >= 1.0),
            'tests_with_high_consistency': sum(1 for r in all_test_results if 0.9 <= r.consistency_rate < 1.0),
            'tests_with_medium_consistency': sum(1 for r in all_test_results if 0.7 <= r.consistency_rate < 0.9),
            'tests_with_low_consistency': sum(1 for r in all_test_results if r.consistency_rate < 0.7),
            'concurrent_safety_score': sum(
                1 for r in all_test_results 
                if r.detailed_analysis and r.detailed_analysis.get('concurrent_safety', False)
            ) / len([r for r in all_test_results if '_concurrent' in r.test_name]) if any('_concurrent' in r.test_name for r in all_test_results) else 0
        }
        
        # 生成建议
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: DataConsistencyReport) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if report.overall_consistency_score < 70:
            recommendations.append("系统一致性较低，建议检查数据访问逻辑和缓存机制")
        
        if report.consistency_analysis['tests_with_low_consistency'] > 0:
            recommendations.append("存在低一致性测试，建议优化相关功能模块")
        
        concurrent_tests = [r for r in report.test_results if '_concurrent' in r.test_name]
        if concurrent_tests:
            failed_concurrent_tests = [r for r in concurrent_tests if r.test_status == TestStatus.FAILED]
            if failed_concurrent_tests:
                recommendations.append("并发访问存在一致性问题，建议加强并发控制")
        
        high_variance_tests = [
            r for r in report.test_results 
            if r.detailed_analysis and r.detailed_analysis.get('execution_time_stats', {}).get('std', 0) > 1.0
        ]
        if high_variance_tests:
            recommendations.append("部分操作执行时间波动较大，建议优化性能稳定性")
        
        if not recommendations:
            recommendations.append("数据一致性表现良好，建议保持当前的数据管理策略")
        
        return recommendations
    
    def generate_consistency_report(self, report: DataConsistencyReport) -> str:
        """生成一致性测试报告"""
        report_lines = [
            "# 数据一致性验证报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试汇总",
            f"- 总测试数: {report.total_tests}",
            f"- 通过测试: {report.passed_tests}",
            f"- 失败测试: {report.failed_tests}",
            f"- 成功率: {(report.passed_tests/report.total_tests*100):.2f}%" if report.total_tests > 0 else "- 成功率: 0%",
            f"- 总体一致性评分: {report.overall_consistency_score:.2f}/100",
            ""
        ]
        
        # 一致性分析
        analysis = report.consistency_analysis
        report_lines.extend([
            "## 一致性分析",
            f"- 平均一致性率: {analysis.get('average_consistency_rate', 0):.2f}%",
            f"- 完全一致测试: {analysis.get('tests_with_perfect_consistency', 0)}",
            f"- 高一致性测试: {analysis.get('tests_with_high_consistency', 0)}",
            f"- 中等一致性测试: {analysis.get('tests_with_medium_consistency', 0)}",
            f"- 低一致性测试: {analysis.get('tests_with_low_consistency', 0)}",
            f"- 并发安全评分: {analysis.get('concurrent_safety_score', 0):.2f}",
            ""
        ])
        
        # 详细测试结果
        for result in report.test_results:
            status_icon = {
                TestStatus.PASSED: "✓",
                TestStatus.FAILED: "✗",
                TestStatus.WARNING: "⚠",
                TestStatus.ERROR: "❌"
            }.get(result.test_status, "?")
            
            report_lines.extend([
                f"## {status_icon} {result.test_name}",
                f"- 测试描述: {result.test_case_name}",
                f"- 执行次数: {result.total_executions}",
                f"- 一致结果: {result.consistent_results}",
                f"- 不一致结果: {result.inconsistent_results}",
                f"- 一致性率: {result.consistency_rate*100:.2f}%",
                f"- 测试状态: {result.test_status}",
                ""
            ])
            
            if result.error_message:
                report_lines.extend([
                    f"**错误信息**: {result.error_message}",
                    ""
                ])
            
            if result.detailed_analysis:
                analysis = result.detailed_analysis
                report_lines.extend([
                    "### 详细分析",
                    f"- 唯一哈希数量: {analysis.get('unique_hash_count', 0)}",
                    f"- 一致性评分: {analysis.get('consistency_score', 0):.2f}%",
                    ""
                ])
                
                # 执行时间统计
                time_stats = analysis.get('execution_time_stats', {})
                if time_stats:
                    report_lines.extend([
                        "### 执行时间统计",
                        f"- 最小时间: {time_stats.get('min', 0):.3f}秒",
                        f"- 最大时间: {time_stats.get('max', 0):.3f}秒",
                        f"- 平均时间: {time_stats.get('avg', 0):.3f}秒",
                        f"- 标准差: {time_stats.get('std', 0):.3f}秒",
                        ""
                    ])
        
        # 改进建议
        if report.recommendations:
            report_lines.extend([
                "## 改进建议",
                ""
            ])
            for i, recommendation in enumerate(report.recommendations, 1):
                report_lines.append(f"{i}. {recommendation}")
            report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    validator = DataConsistencyValidator()
    
    # 运行所有一致性测试
    report = validator.run_all_consistency_tests()
    
    # 生成报告
    report_content = validator.generate_consistency_report(report)
    
    # 保存报告
    report_file = f"data_consistency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"数据一致性验证完成，报告已保存到: {report_file}")
    print(f"测试结果: {report.passed_tests}/{report.total_tests} 通过")
    print(f"总体一致性评分: {report.overall_consistency_score:.2f}/100")


if __name__ == "__main__":
    main() 