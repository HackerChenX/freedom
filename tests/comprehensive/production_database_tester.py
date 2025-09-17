#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
生产环境数据库连接测试器

验证ClickHouse数据库连接、数据质量和系统兼容性，确保生产环境就绪。
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
from functools import wraps

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.clickhouse_db import get_clickhouse_db
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class ProductionDatabaseTester:
    """
    生产环境数据库连接测试器
    
    功能：
    1. 数据库连接验证
    2. 数据质量检查
    3. 并发性能测试
    4. 系统兼容性验证
    5. 生产环境就绪性评估
    """
    
    def __init__(self):
        """初始化生产环境数据库测试器"""
        self.db = None
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = None
        self.total_queries = 0
        self.failed_queries = 0
        
    @exception_handler(reraise=True)
    def setup_database_connection(self) -> bool:
        """
        建立数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.db = get_clickhouse_db()
            
            # 测试基本连接
            test_query = "SELECT 1 as test"
            result = self.db.query(test_query)
            
            if result is not None and len(result) > 0:
                logger.info("✅ 数据库连接成功建立")
                return True
            else:
                logger.error("❌ 数据库连接测试失败：查询无结果")
                return False
                
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def test_database_basic_info(self) -> Dict[str, Any]:
        """
        测试数据库基本信息
        
        Returns:
            Dict[str, Any]: 数据库基本信息
        """
        info = {}
        
        try:
            # 1. 数据库版本
            version_query = "SELECT version() as version"
            version_result = self.db.query(version_query)
            info['clickhouse_version'] = version_result.iloc[0]['version'] if not version_result.empty else 'Unknown'
            
            # 2. 数据库列表
            databases_query = "SHOW DATABASES"
            databases_result = self.db.query(databases_query)
            info['databases'] = databases_result['name'].tolist() if not databases_result.empty else []
            
            # 3. stock数据库表信息
            tables_query = "SHOW TABLES FROM stock"
            tables_result = self.db.query(tables_query)
            info['stock_tables'] = tables_result['name'].tolist() if not tables_result.empty else []
            
            # 4. stock_info表统计
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT code) as unique_stocks,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(DISTINCT level) as data_levels
            FROM stock.stock_info
            """
            stats_result = self.db.query(stats_query)
            if not stats_result.empty:
                info.update(stats_result.iloc[0].to_dict())
            
            logger.info(f"✅ 数据库基本信息获取成功：{info}")
            return info
            
        except Exception as e:
            logger.error(f"❌ 获取数据库基本信息失败: {e}")
            return {}
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=10.0)
    def test_data_quality(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        测试数据质量
        
        Args:
            sample_size: 样本大小
            
        Returns:
            Dict[str, Any]: 数据质量报告
        """
        quality_report = {
            'sample_size': sample_size,
            'quality_issues': [],
            'quality_score': 0.0,
            'data_integrity': True,
            'recommendations': []
        }
        
        try:
            # 1. 随机抽样检查
            sample_query = f"""
            SELECT code, name, date, open, high, low, close, volume, turnover_rate, level
            FROM stock.stock_info 
            ORDER BY rand() 
            LIMIT {sample_size}
            """
            sample_data = self.db.query(sample_query)
            
            if sample_data.empty:
                quality_report['quality_issues'].append("无法获取数据样本")
                quality_report['data_integrity'] = False
                return quality_report
            
            df = sample_data
            
            # 2. 空值检查
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                quality_report['quality_issues'].append(f"发现空值: {null_counts.to_dict()}")
            
            # 3. OHLC数据一致性检查
            ohlc_issues = 0
            for _, row in df.iterrows():
                if pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['open']) and pd.notna(row['close']):
                    if not (row['low'] <= row['open'] <= row['high'] and 
                           row['low'] <= row['close'] <= row['high']):
                        ohlc_issues += 1
            
            if ohlc_issues > 0:
                quality_report['quality_issues'].append(f"OHLC数据不一致: {ohlc_issues}条记录")
            
            # 4. 价格合理性检查
            price_issues = 0
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    invalid_prices = (df[col] <= 0).sum()
                    if invalid_prices > 0:
                        price_issues += invalid_prices
            
            if price_issues > 0:
                quality_report['quality_issues'].append(f"价格数据异常: {price_issues}条记录")
            
            # 5. 成交量合理性检查
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    quality_report['quality_issues'].append(f"成交量为负值: {negative_volume}条记录")
            
            # 6. 计算质量分数
            total_issues = len(quality_report['quality_issues'])
            if total_issues == 0:
                quality_report['quality_score'] = 100.0
            else:
                quality_report['quality_score'] = max(0, 100 - total_issues * 10)
            
            # 7. 推荐建议
            if quality_report['quality_score'] < 90:
                quality_report['recommendations'].append("建议进行数据清洗")
            if total_issues > 0:
                quality_report['recommendations'].append("建议添加数据验证规则")
            
            logger.info(f"✅ 数据质量检查完成，质量分数: {quality_report['quality_score']}")
            return quality_report
            
        except Exception as e:
            logger.error(f"❌ 数据质量检查失败: {e}")
            quality_report['quality_issues'].append(f"检查过程异常: {str(e)}")
            quality_report['data_integrity'] = False
            return quality_report
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def test_query_performance(self, test_stock_codes: List[str] = None) -> Dict[str, Any]:
        """
        测试查询性能
        
        Args:
            test_stock_codes: 测试股票代码列表
            
        Returns:
            Dict[str, Any]: 性能测试报告
        """
        performance_report = {
            'test_cases': [],
            'average_query_time': 0.0,
            'max_query_time': 0.0,
            'min_query_time': 0.0,
            'total_queries': 0,
            'failed_queries': 0,
            'queries_per_second': 0.0,
            'performance_grade': 'Unknown'
        }
        
        if not test_stock_codes:
            # 获取测试股票代码
            codes_query = """
            SELECT DISTINCT code 
            FROM stock.stock_info 
            WHERE date >= today() - INTERVAL 100 DAY 
            LIMIT 10
            """
            codes_result = self.db.query(codes_query)
            test_stock_codes = codes_result['code'].tolist() if not codes_result.empty else ['000001']
        
        try:
            query_times = []
            
            for code in test_stock_codes:
                # 测试案例1：基本股票数据查询
                start_time = time.time()
                query1 = f"""
                SELECT code, date, open, high, low, close, volume
                FROM stock.stock_info
                WHERE code = '{code}' AND level = '日线'
                ORDER BY date DESC
                LIMIT 100
                """
                result1 = self.db.query(query1)
                query_time1 = time.time() - start_time
                query_times.append(query_time1)
                
                performance_report['test_cases'].append({
                    'stock_code': code,
                    'query_type': 'basic_stock_data',
                    'query_time': query_time1,
                    'records_returned': len(result1) if not result1.empty else 0,
                    'success': not result1.empty
                })
                
                if result1.empty:
                    performance_report['failed_queries'] += 1
                
                # 测试案例2：技术指标计算查询
                start_time = time.time()
                query2 = f"""
                SELECT 
                    code, date, close,
                    avg(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20
                FROM stock.stock_info
                WHERE code = '{code}' AND level = '日线'
                ORDER BY date DESC
                LIMIT 50
                """
                result2 = self.db.query(query2)
                query_time2 = time.time() - start_time
                query_times.append(query_time2)
                
                performance_report['test_cases'].append({
                    'stock_code': code,
                    'query_type': 'technical_indicator',
                    'query_time': query_time2,
                    'records_returned': len(result2) if not result2.empty else 0,
                    'success': not result2.empty
                })
                
                if result2.empty:
                    performance_report['failed_queries'] += 1
            
            # 统计性能指标
            if query_times:
                performance_report['average_query_time'] = np.mean(query_times)
                performance_report['max_query_time'] = np.max(query_times)
                performance_report['min_query_time'] = np.min(query_times)
                performance_report['total_queries'] = len(query_times)
                
                total_time = sum(query_times)
                if total_time > 0:
                    performance_report['queries_per_second'] = len(query_times) / total_time
                
                # 性能等级评估
                avg_time = performance_report['average_query_time']
                if avg_time < 0.1:
                    performance_report['performance_grade'] = 'Excellent'
                elif avg_time < 0.5:
                    performance_report['performance_grade'] = 'Good'
                elif avg_time < 2.0:
                    performance_report['performance_grade'] = 'Fair'
                else:
                    performance_report['performance_grade'] = 'Poor'
            
            logger.info(f"✅ 查询性能测试完成，平均查询时间: {performance_report['average_query_time']:.3f}秒")
            return performance_report
            
        except Exception as e:
            logger.error(f"❌ 查询性能测试失败: {e}")
            performance_report['failed_queries'] = performance_report['total_queries']
            return performance_report
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=60.0)
    def test_concurrent_access(self, num_threads: int = 5, queries_per_thread: int = 5) -> Dict[str, Any]:
        """
        测试并发访问能力
        
        Args:
            num_threads: 并发线程数
            queries_per_thread: 每线程查询数
            
        Returns:
            Dict[str, Any]: 并发测试报告
        """
        concurrent_report = {
            'num_threads': num_threads,
            'queries_per_thread': queries_per_thread,
            'total_queries': num_threads * queries_per_thread,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_time': 0.0,
            'average_response_time': 0.0,
            'queries_per_second': 0.0,
            'success_rate': 0.0,
            'thread_results': [],
            'concurrent_grade': 'Unknown'
        }
        
        def worker_thread(thread_id: int) -> Dict[str, Any]:
            """工作线程"""
            thread_result = {
                'thread_id': thread_id,
                'successful_queries': 0,
                'failed_queries': 0,
                'query_times': [],
                'errors': []
            }
            
            try:
                # 每个线程获取独立的数据库连接
                thread_db = get_clickhouse_db()
                
                for i in range(queries_per_thread):
                    try:
                        start_time = time.time()
                        
                        # 执行简单查询
                        query = f"""
                        SELECT COUNT(*) as count
                        FROM stock.stock_info
                        WHERE date >= today() - INTERVAL {i + 1} DAY
                        LIMIT 10
                        """
                        
                        result = thread_db.query(query)
                        query_time = time.time() - start_time
                        
                        if not result.empty:
                            thread_result['successful_queries'] += 1
                            thread_result['query_times'].append(query_time)
                        else:
                            thread_result['failed_queries'] += 1
                            thread_result['errors'].append(f"Query {i+1}: No result")
                            
                    except Exception as e:
                        thread_result['failed_queries'] += 1
                        thread_result['errors'].append(f"Query {i+1}: {str(e)}")
                        
            except Exception as e:
                thread_result['errors'].append(f"Thread setup error: {str(e)}")
                thread_result['failed_queries'] = queries_per_thread
            
            return thread_result
        
        try:
            start_time = time.time()
            
            # 使用线程池执行并发查询
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
                
                for future in as_completed(futures):
                    thread_result = future.result()
                    concurrent_report['thread_results'].append(thread_result)
                    concurrent_report['successful_queries'] += thread_result['successful_queries']
                    concurrent_report['failed_queries'] += thread_result['failed_queries']
            
            concurrent_report['total_time'] = time.time() - start_time
            
            # 计算统计指标
            all_query_times = []
            for result in concurrent_report['thread_results']:
                all_query_times.extend(result['query_times'])
            
            if all_query_times:
                concurrent_report['average_response_time'] = np.mean(all_query_times)
            
            if concurrent_report['total_time'] > 0:
                concurrent_report['queries_per_second'] = concurrent_report['successful_queries'] / concurrent_report['total_time']
            
            if concurrent_report['total_queries'] > 0:
                concurrent_report['success_rate'] = concurrent_report['successful_queries'] / concurrent_report['total_queries']
            
            # 并发能力等级评估
            success_rate = concurrent_report['success_rate']
            if success_rate >= 0.95:
                concurrent_report['concurrent_grade'] = 'Excellent'
            elif success_rate >= 0.8:
                concurrent_report['concurrent_grade'] = 'Good'
            elif success_rate >= 0.6:
                concurrent_report['concurrent_grade'] = 'Fair'
            else:
                concurrent_report['concurrent_grade'] = 'Poor'
            
            logger.info(f"✅ 并发访问测试完成，成功率: {concurrent_report['success_rate']:.1%}")
            return concurrent_report
            
        except Exception as e:
            logger.error(f"❌ 并发访问测试失败: {e}")
            concurrent_report['failed_queries'] = concurrent_report['total_queries']
            concurrent_report['success_rate'] = 0.0
            concurrent_report['concurrent_grade'] = 'Failed'
            return concurrent_report
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=120.0)
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        运行综合测试
        
        Returns:
            Dict[str, Any]: 综合测试报告
        """
        self.start_time = datetime.now()
        
        comprehensive_report = {
            'test_start_time': self.start_time.isoformat(),
            'test_end_time': None,
            'total_test_time': 0.0,
            'database_connection': False,
            'database_info': {},
            'data_quality': {},
            'query_performance': {},
            'concurrent_access': {},
            'overall_grade': 'Unknown',
            'production_ready': False,
            'recommendations': []
        }
        
        try:
            logger.info("🚀 开始生产环境数据库综合测试...")
            
            # 1. 数据库连接测试
            logger.info("📝 Step 1: 测试数据库连接...")
            comprehensive_report['database_connection'] = self.setup_database_connection()
            
            if not comprehensive_report['database_connection']:
                comprehensive_report['recommendations'].append("需要修复数据库连接问题")
                logger.error("❌ 数据库连接失败，无法继续测试")
                return comprehensive_report
            
            # 2. 数据库信息获取
            logger.info("📝 Step 2: 获取数据库基本信息...")
            comprehensive_report['database_info'] = self.test_database_basic_info()
            
            # 3. 数据质量检查
            logger.info("📝 Step 3: 执行数据质量检查...")
            comprehensive_report['data_quality'] = self.test_data_quality(sample_size=2000)
            
            # 4. 查询性能测试
            logger.info("📝 Step 4: 执行查询性能测试...")
            comprehensive_report['query_performance'] = self.test_query_performance()
            
            # 5. 并发访问测试
            logger.info("📝 Step 5: 执行并发访问测试...")
            comprehensive_report['concurrent_access'] = self.test_concurrent_access(num_threads=3, queries_per_thread=3)
            
            # 6. 综合评估
            logger.info("📝 Step 6: 综合评估...")
            self._evaluate_overall_grade(comprehensive_report)
            
            end_time = datetime.now()
            comprehensive_report['test_end_time'] = end_time.isoformat()
            comprehensive_report['total_test_time'] = (end_time - self.start_time).total_seconds()
            
            logger.info(f"✅ 生产环境数据库综合测试完成，总耗时: {comprehensive_report['total_test_time']:.1f}秒")
            logger.info(f"📊 综合评级: {comprehensive_report['overall_grade']}")
            logger.info(f"🚀 生产环境就绪: {'是' if comprehensive_report['production_ready'] else '否'}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"❌ 综合测试失败: {e}")
            comprehensive_report['recommendations'].append(f"测试过程异常: {str(e)}")
            return comprehensive_report
    
    def _evaluate_overall_grade(self, report: Dict[str, Any]) -> None:
        """评估综合等级"""
        scores = []
        
        # 数据库连接分数
        if report['database_connection']:
            scores.append(100)
        else:
            scores.append(0)
        
        # 数据质量分数
        if 'quality_score' in report['data_quality']:
            scores.append(report['data_quality']['quality_score'])
        else:
            scores.append(0)
        
        # 查询性能分数
        perf_grade = report['query_performance'].get('performance_grade', 'Poor')
        if perf_grade == 'Excellent':
            scores.append(100)
        elif perf_grade == 'Good':
            scores.append(80)
        elif perf_grade == 'Fair':
            scores.append(60)
        else:
            scores.append(40)
        
        # 并发能力分数
        concurrent_grade = report['concurrent_access'].get('concurrent_grade', 'Poor')
        if concurrent_grade == 'Excellent':
            scores.append(100)
        elif concurrent_grade == 'Good':
            scores.append(80)
        elif concurrent_grade == 'Fair':
            scores.append(60)
        else:
            scores.append(40)
        
        # 计算综合分数
        if scores:
            overall_score = np.mean(scores)
            
            if overall_score >= 90:
                report['overall_grade'] = 'A+'
                report['production_ready'] = True
            elif overall_score >= 80:
                report['overall_grade'] = 'A'
                report['production_ready'] = True
            elif overall_score >= 70:
                report['overall_grade'] = 'B'
                report['production_ready'] = True
                report['recommendations'].append("建议优化性能后投入生产使用")
            elif overall_score >= 60:
                report['overall_grade'] = 'C'
                report['production_ready'] = False
                report['recommendations'].append("需要解决性能问题后才能投入生产使用")
            else:
                report['overall_grade'] = 'D'
                report['production_ready'] = False
                report['recommendations'].append("存在严重问题，不建议投入生产使用")
        
        else:
            report['overall_grade'] = 'F'
            report['production_ready'] = False
            report['recommendations'].append("无法完成评估，请检查测试环境")
    
    def generate_detailed_report(self, report: Dict[str, Any]) -> str:
        """生成详细报告"""
        report_lines = [
            "=" * 80,
            "生产环境数据库综合测试报告",
            "=" * 80,
            f"测试时间: {report.get('test_start_time', 'Unknown')}",
            f"测试耗时: {report.get('total_test_time', 0):.1f}秒",
            f"综合评级: {report.get('overall_grade', 'Unknown')}",
            f"生产就绪: {'✅ 是' if report.get('production_ready', False) else '❌ 否'}",
            "",
            "📊 详细测试结果:",
            "-" * 40,
        ]
        
        # 数据库连接
        conn_status = "✅ 成功" if report.get('database_connection', False) else "❌ 失败"
        report_lines.append(f"数据库连接: {conn_status}")
        
        # 数据库信息
        if 'database_info' in report:
            info = report['database_info']
            report_lines.extend([
                f"ClickHouse版本: {info.get('clickhouse_version', 'Unknown')}",
                f"总记录数: {info.get('total_records', 0):,}条",
                f"股票数量: {info.get('unique_stocks', 0):,}只",
                f"数据时间范围: {info.get('earliest_date', 'Unknown')} 至 {info.get('latest_date', 'Unknown')}",
                ""
            ])
        
        # 数据质量
        if 'data_quality' in report:
            quality = report['data_quality']
            quality_score = quality.get('quality_score', 0)
            issues_count = len(quality.get('quality_issues', []))
            report_lines.extend([
                f"数据质量分数: {quality_score:.1f}/100",
                f"质量问题数量: {issues_count}个",
                ""
            ])
        
        # 查询性能
        if 'query_performance' in report:
            perf = report['query_performance']
            report_lines.extend([
                f"查询性能等级: {perf.get('performance_grade', 'Unknown')}",
                f"平均查询时间: {perf.get('average_query_time', 0):.3f}秒",
                f"每秒查询数: {perf.get('queries_per_second', 0):.1f}",
                f"失败查询数: {perf.get('failed_queries', 0)}个",
                ""
            ])
        
        # 并发访问
        if 'concurrent_access' in report:
            concurrent = report['concurrent_access']
            success_rate = concurrent.get('success_rate', 0) * 100
            report_lines.extend([
                f"并发能力等级: {concurrent.get('concurrent_grade', 'Unknown')}",
                f"并发成功率: {success_rate:.1f}%",
                f"并发查询速度: {concurrent.get('queries_per_second', 0):.1f} 查询/秒",
                ""
            ])
        
        # 建议
        if 'recommendations' in report and report['recommendations']:
            report_lines.append("💡 改进建议:")
            for i, rec in enumerate(report['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """主函数 - 运行生产环境数据库测试"""
    print("🚀 启动生产环境数据库综合测试...")
    
    tester = ProductionDatabaseTester()
    
    try:
        # 运行综合测试
        report = tester.run_comprehensive_test()
        
        # 生成详细报告
        detailed_report = tester.generate_detailed_report(report)
        print(detailed_report)
        
        # 保存报告到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"production_database_test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 返回测试结果
        if report.get('production_ready', False):
            print("🎉 生产环境数据库测试通过！系统已准备好投入生产使用。")
            return 0
        else:
            print("⚠️  生产环境数据库测试未完全通过，请根据建议进行优化。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 