#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
4000只股票全市场扫描性能基准测试系统

测试系统在真实生产规模下处理4000只股票的性能极限、稳定性和资源使用情况。
"""

import os
import sys
import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import multiprocessing as mp

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.clickhouse_db import get_clickhouse_db
from config.unified_config_manager import get_config
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.start_time = None
        self.end_time = None
        self.total_stocks_processed = 0
        self.successful_stocks = 0
        self.failed_stocks = 0
        self.total_records_processed = 0
        self.query_times = []
        self.batch_times = []
        self.memory_usage_samples = []
        self.cpu_usage_samples = []
        self.error_messages = []
        
    def add_query_time(self, duration: float):
        """添加查询时间"""
        self.query_times.append(duration)
    
    def add_batch_time(self, duration: float):
        """添加批次处理时间"""
        self.batch_times.append(duration)
    
    def add_system_sample(self, cpu: float, memory: float):
        """添加系统资源样本"""
        self.cpu_usage_samples.append(cpu)
        self.memory_usage_samples.append(memory)
    
    def add_error(self, error_msg: str):
        """添加错误信息"""
        self.error_messages.append(error_msg)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_time = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        stats = {
            'execution_time': total_time,
            'total_stocks': self.total_stocks_processed,
            'successful_stocks': self.successful_stocks,
            'failed_stocks': self.failed_stocks,
            'success_rate': (self.successful_stocks / max(self.total_stocks_processed, 1)) * 100,
            'total_records': self.total_records_processed,
            'stocks_per_second': self.successful_stocks / max(total_time, 0.001),
            'records_per_second': self.total_records_processed / max(total_time, 0.001),
            'query_performance': {},
            'batch_performance': {},
            'system_performance': {},
            'error_count': len(self.error_messages)
        }
        
        # 查询性能统计
        if self.query_times:
            stats['query_performance'] = {
                'count': len(self.query_times),
                'average': np.mean(self.query_times),
                'median': np.median(self.query_times),
                'min': np.min(self.query_times),
                'max': np.max(self.query_times),
                'p95': np.percentile(self.query_times, 95),
                'p99': np.percentile(self.query_times, 99),
                'std': np.std(self.query_times)
            }
        
        # 批次性能统计
        if self.batch_times:
            stats['batch_performance'] = {
                'count': len(self.batch_times),
                'average': np.mean(self.batch_times),
                'median': np.median(self.batch_times),
                'min': np.min(self.batch_times),
                'max': np.max(self.batch_times),
                'std': np.std(self.batch_times)
            }
        
        # 系统性能统计
        if self.cpu_usage_samples and self.memory_usage_samples:
            stats['system_performance'] = {
                'cpu_average': np.mean(self.cpu_usage_samples),
                'cpu_max': np.max(self.cpu_usage_samples),
                'cpu_min': np.min(self.cpu_usage_samples),
                'memory_average': np.mean(self.memory_usage_samples),
                'memory_max': np.max(self.memory_usage_samples),
                'memory_min': np.min(self.memory_usage_samples),
                'samples_count': len(self.cpu_usage_samples)
            }
        
        return stats


class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitor_interval = 1.0  # 1秒采样间隔
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🔍 系统资源监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("✅ 系统资源监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 获取CPU和内存使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # 添加到指标中
                self.metrics.add_system_sample(cpu_percent, memory.percent)
                
                # 等待下次采样
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.warning(f"资源监控异常: {e}")
                time.sleep(self.monitor_interval)


class FullMarketScanner:
    """全市场扫描器"""
    
    def __init__(self):
        self.db = None
        self.metrics = PerformanceMetrics()
        self.resource_monitor = SystemResourceMonitor(self.metrics)
        
    @exception_handler(reraise=True)
    def initialize_database(self) -> bool:
        """初始化数据库连接"""
        try:
            self.db = get_clickhouse_db()
            
            # 测试连接
            result = self.db.query("SELECT 1 as test")
            if not result.empty:
                logger.info("✅ 全市场扫描器数据库连接成功")
                return True
                
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False
        
        return False
    
    @exception_handler(reraise=False)
    def get_all_stock_codes(self, limit: int = 4000) -> List[str]:
        """获取所有股票代码"""
        try:
            logger.info(f"📋 获取全市场股票代码 (最多 {limit} 只)...")
            
            query = f"""
            SELECT DISTINCT code
            FROM stock.stock_info
            WHERE level = '日线'
            AND date >= '2024-01-01'
            ORDER BY code
            LIMIT {limit}
            """
            
            result = self.db.query(query)
            
            if result.empty:
                logger.warning("未找到股票代码")
                return []
            
            stock_codes = result['code'].tolist()
            logger.info(f"✅ 获取到 {len(stock_codes)} 只股票代码")
            
            return stock_codes
            
        except Exception as e:
            logger.error(f"❌ 获取股票代码失败: {e}")
            return []
    
    @exception_handler(reraise=False)
    def process_single_stock(self, stock_code: str) -> Dict[str, Any]:
        """处理单只股票"""
        start_time = time.time()
        
        result = {
            'stock_code': stock_code,
            'success': False,
            'records_processed': 0,
            'processing_time': 0.0,
            'error_message': None,
            'basic_stats': {}
        }
        
        try:
            # 创建独立的数据库连接
            db = get_clickhouse_db()
            
            # 查询股票基础数据
            query = f"""
            SELECT 
                code, date, open, high, low, close, volume,
                COUNT(*) OVER () as total_count
            FROM stock.stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '2024-01-01'
            ORDER BY date DESC
            LIMIT 1000
            """
            
            query_start = time.time()
            stock_data = db.query(query)
            query_time = time.time() - query_start
            
            if not stock_data.empty:
                # 计算基本统计信息
                result['records_processed'] = len(stock_data)
                result['basic_stats'] = {
                    'latest_date': stock_data['date'].iloc[0] if not stock_data.empty else None,
                    'total_records': stock_data['total_count'].iloc[0] if 'total_count' in stock_data.columns else len(stock_data),
                    'price_range': {
                        'max_high': float(stock_data['high'].max()),
                        'min_low': float(stock_data['low'].min()),
                        'latest_close': float(stock_data['close'].iloc[0])
                    },
                    'volume_stats': {
                        'avg_volume': float(stock_data['volume'].mean()),
                        'max_volume': float(stock_data['volume'].max()),
                        'total_volume': float(stock_data['volume'].sum())
                    }
                }
                
                # 计算简单技术指标
                if len(stock_data) >= 20:
                    # 20日均线
                    ma20 = stock_data['close'].rolling(window=20).mean().iloc[-1]
                    # 价格相对位置
                    price_position = (stock_data['close'].iloc[0] - stock_data['low'].min()) / (stock_data['high'].max() - stock_data['low'].min())
                    
                    result['basic_stats']['technical_indicators'] = {
                        'ma20': float(ma20) if not pd.isna(ma20) else None,
                        'price_position': float(price_position) if not pd.isna(price_position) else None
                    }
                
                result['success'] = True
                self.metrics.add_query_time(query_time)
                
            else:
                result['error_message'] = "无数据"
                
        except Exception as e:
            result['error_message'] = str(e)
            self.metrics.add_error(f"{stock_code}: {str(e)}")
            
        finally:
            result['processing_time'] = time.time() - start_time
            
        return result
    
    @exception_handler(reraise=False)
    def process_stock_batch(self, stock_codes: List[str], batch_id: int, 
                          max_workers: int = 5) -> Dict[str, Any]:
        """处理股票批次"""
        batch_start_time = time.time()
        
        batch_result = {
            'batch_id': batch_id,
            'batch_size': len(stock_codes),
            'successful_stocks': 0,
            'failed_stocks': 0,
            'total_records': 0,
            'processing_time': 0.0,
            'stock_results': [],
            'performance_stats': {}
        }
        
        logger.info(f"📊 处理批次 {batch_id}: {len(stock_codes)} 只股票")
        
        try:
            # 使用线程池处理批次内的股票
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_stock = {
                    executor.submit(self.process_single_stock, stock_code): stock_code
                    for stock_code in stock_codes
                }
                
                # 收集结果
                for future in as_completed(future_to_stock):
                    stock_code = future_to_stock[future]
                    try:
                        stock_result = future.result(timeout=30)  # 30秒超时
                        batch_result['stock_results'].append(stock_result)
                        
                        if stock_result['success']:
                            batch_result['successful_stocks'] += 1
                            batch_result['total_records'] += stock_result['records_processed']
                        else:
                            batch_result['failed_stocks'] += 1
                            
                    except Exception as e:
                        batch_result['failed_stocks'] += 1
                        self.metrics.add_error(f"批次 {batch_id} 股票 {stock_code}: {str(e)}")
                        logger.warning(f"股票 {stock_code} 处理异常: {e}")
            
            # 计算批次性能统计
            processing_times = [r['processing_time'] for r in batch_result['stock_results']]
            if processing_times:
                batch_result['performance_stats'] = {
                    'avg_processing_time': np.mean(processing_times),
                    'max_processing_time': np.max(processing_times),
                    'min_processing_time': np.min(processing_times),
                    'total_processing_time': np.sum(processing_times)
                }
            
        except Exception as e:
            logger.error(f"批次 {batch_id} 处理异常: {e}")
            batch_result['failed_stocks'] = len(stock_codes)
            self.metrics.add_error(f"批次 {batch_id}: {str(e)}")
        
        batch_result['processing_time'] = time.time() - batch_start_time
        self.metrics.add_batch_time(batch_result['processing_time'])
        
        logger.info(f"✅ 批次 {batch_id} 完成: {batch_result['successful_stocks']}/{batch_result['batch_size']} 成功, "
                   f"耗时 {batch_result['processing_time']:.1f}秒")
        
        return batch_result
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=3600.0)  # 1小时阈值
    def run_full_market_scan(self, target_stock_count: int = 4000, 
                           batch_size: int = 50, max_workers: int = 4) -> Dict[str, Any]:
        """运行全市场扫描"""
        self.metrics.reset()
        self.metrics.start_time = datetime.now()
        
        scan_result = {
            'scan_start_time': self.metrics.start_time.isoformat(),
            'scan_end_time': None,
            'target_stock_count': target_stock_count,
            'actual_stock_count': 0,
            'configuration': {
                'batch_size': batch_size,
                'max_workers': max_workers,
                'max_concurrent_batches': mp.cpu_count()
            },
            'batch_results': [],
            'performance_summary': {},
            'system_resources': {},
            'recommendations': [],
            'scan_status': 'Unknown'
        }
        
        try:
            logger.info(f"🚀 开始全市场扫描测试 (目标: {target_stock_count} 只股票)...")
            
            # 1. 初始化数据库连接
            logger.info("📝 Step 1: 初始化数据库连接...")
            if not self.initialize_database():
                scan_result['scan_status'] = 'Failed'
                scan_result['recommendations'].append("数据库连接失败，检查连接配置")
                return scan_result
            
            # 2. 获取股票代码列表
            logger.info("📝 Step 2: 获取股票代码列表...")
            all_stock_codes = self.get_all_stock_codes(target_stock_count)
            scan_result['actual_stock_count'] = len(all_stock_codes)
            
            if not all_stock_codes:
                scan_result['scan_status'] = 'Failed'
                scan_result['recommendations'].append("未获取到股票代码，检查数据库数据")
                return scan_result
            
            # 3. 启动系统资源监控
            logger.info("📝 Step 3: 启动系统资源监控...")
            self.resource_monitor.start_monitoring()
            
            # 4. 分批处理股票
            logger.info(f"📝 Step 4: 分批处理股票 (批次大小: {batch_size})...")
            
            # 将股票分成批次
            batches = [all_stock_codes[i:i + batch_size] 
                      for i in range(0, len(all_stock_codes), batch_size)]
            
            logger.info(f"📊 总计 {len(batches)} 个批次，每批次 {batch_size} 只股票")
            
            # 顺序处理批次 (避免过度并发)
            for batch_id, batch_codes in enumerate(batches, 1):
                try:
                    batch_result = self.process_stock_batch(
                        batch_codes, batch_id, max_workers
                    )
                    scan_result['batch_results'].append(batch_result)
                    
                    # 更新总体指标
                    self.metrics.total_stocks_processed += batch_result['batch_size']
                    self.metrics.successful_stocks += batch_result['successful_stocks']
                    self.metrics.failed_stocks += batch_result['failed_stocks']
                    self.metrics.total_records_processed += batch_result['total_records']
                    
                    # 定期垃圾回收
                    if batch_id % 10 == 0:
                        gc.collect()
                        logger.info(f"📊 已处理 {batch_id}/{len(batches)} 个批次，"
                                   f"累计成功 {self.metrics.successful_stocks} 只股票")
                    
                    # 检查内存使用情况
                    memory = psutil.virtual_memory()
                    if memory.percent > 90:
                        logger.warning(f"⚠️ 内存使用率过高: {memory.percent:.1f}%")
                        scan_result['recommendations'].append("内存使用率过高，建议增加内存或减少批次大小")
                    
                except Exception as e:
                    logger.error(f"批次 {batch_id} 处理失败: {e}")
                    self.metrics.add_error(f"批次 {batch_id}: {str(e)}")
                    continue
            
            # 5. 停止资源监控
            logger.info("📝 Step 5: 停止系统资源监控...")
            self.resource_monitor.stop_monitoring()
            
            # 6. 生成性能摘要
            logger.info("📝 Step 6: 生成性能摘要...")
            self.metrics.end_time = datetime.now()
            scan_result['performance_summary'] = self.metrics.get_statistics()
            
            # 7. 评估扫描状态
            success_rate = self.metrics.successful_stocks / max(self.metrics.total_stocks_processed, 1) * 100
            if success_rate >= 95:
                scan_result['scan_status'] = 'Excellent'
            elif success_rate >= 85:
                scan_result['scan_status'] = 'Good'
            elif success_rate >= 70:
                scan_result['scan_status'] = 'Fair'
            else:
                scan_result['scan_status'] = 'Poor'
            
            # 8. 生成建议
            self._generate_recommendations(scan_result)
            
            scan_result['scan_end_time'] = self.metrics.end_time.isoformat()
            
            logger.info(f"✅ 全市场扫描完成！")
            logger.info(f"📊 处理结果: {self.metrics.successful_stocks}/{self.metrics.total_stocks_processed} 成功")
            logger.info(f"⏱️ 总耗时: {scan_result['performance_summary']['execution_time']:.1f}秒")
            logger.info(f"🎯 状态: {scan_result['scan_status']}")
            
            return scan_result
            
        except Exception as e:
            logger.error(f"❌ 全市场扫描失败: {e}")
            scan_result['scan_status'] = 'Failed'
            scan_result['recommendations'].append(f"扫描过程异常: {str(e)}")
            return scan_result
        finally:
            self.resource_monitor.stop_monitoring()
    
    def _generate_recommendations(self, scan_result: Dict[str, Any]):
        """生成优化建议"""
        recommendations = []
        performance = scan_result.get('performance_summary', {})
        
        # 成功率分析
        success_rate = performance.get('success_rate', 0)
        if success_rate < 95:
            recommendations.append(f"成功率 {success_rate:.1f}% 需要提升，检查数据库连接稳定性")
        
        # 性能分析
        stocks_per_second = performance.get('stocks_per_second', 0)
        if stocks_per_second < 5:
            recommendations.append("处理速度较慢，建议增加并发数或优化查询")
        elif stocks_per_second > 20:
            recommendations.append("处理速度优秀，可以处理更大规模数据")
        
        # 查询性能分析
        query_perf = performance.get('query_performance', {})
        if query_perf:
            avg_query_time = query_perf.get('average', 0)
            if avg_query_time > 1.0:
                recommendations.append("平均查询时间较长，建议优化SQL或增加索引")
            
            p95_query_time = query_perf.get('p95', 0)
            if p95_query_time > 3.0:
                recommendations.append("存在慢查询，建议检查查询计划")
        
        # 系统资源分析
        system_perf = performance.get('system_performance', {})
        if system_perf:
            cpu_avg = system_perf.get('cpu_average', 0)
            memory_avg = system_perf.get('memory_average', 0)
            
            if cpu_avg > 80:
                recommendations.append("CPU使用率较高，考虑增加处理节点")
            if memory_avg > 85:
                recommendations.append("内存使用率较高，考虑增加内存或优化数据处理")
        
        # 错误分析
        error_count = performance.get('error_count', 0)
        if error_count > 0:
            recommendations.append(f"存在 {error_count} 个错误，建议检查错误日志")
        
        # 通用建议
        if not recommendations:
            recommendations.append("系统性能表现优秀，可以投入生产使用")
        
        scan_result['recommendations'] = recommendations
    
    def generate_report(self, scan_result: Dict[str, Any]) -> str:
        """生成扫描报告"""
        report_lines = [
            "=" * 80,
            "4000只股票全市场扫描性能基准测试报告",
            "=" * 80,
            f"扫描时间: {scan_result.get('scan_start_time', 'Unknown')}",
            f"目标股票数: {scan_result.get('target_stock_count', 0):,}",
            f"实际股票数: {scan_result.get('actual_stock_count', 0):,}",
            f"扫描状态: {scan_result.get('scan_status', 'Unknown')}",
            "",
            "⚙️ 测试配置:",
            "-" * 40,
        ]
        
        config = scan_result.get('configuration', {})
        if config:
            report_lines.extend([
                f"批次大小: {config.get('batch_size', 0)}",
                f"最大工作线程: {config.get('max_workers', 0)}",
                f"最大并发批次: {config.get('max_concurrent_batches', 0)}"
            ])
        
        # 性能摘要
        performance = scan_result.get('performance_summary', {})
        if performance:
            report_lines.extend([
                "",
                "📊 性能摘要:",
                "-" * 40,
                f"总执行时间: {performance.get('execution_time', 0):.1f}秒",
                f"成功处理股票: {performance.get('successful_stocks', 0):,}",
                f"失败股票: {performance.get('failed_stocks', 0):,}",
                f"成功率: {performance.get('success_rate', 0):.1f}%",
                f"总处理记录数: {performance.get('total_records', 0):,}",
                f"股票处理速度: {performance.get('stocks_per_second', 0):.1f} 股票/秒",
                f"记录处理速度: {performance.get('records_per_second', 0):.1f} 记录/秒"
            ])
        
        # 查询性能
        query_perf = performance.get('query_performance', {})
        if query_perf:
            report_lines.extend([
                "",
                "🔍 查询性能分析:",
                "-" * 40,
                f"查询总数: {query_perf.get('count', 0):,}",
                f"平均查询时间: {query_perf.get('average', 0):.3f}秒",
                f"P95查询时间: {query_perf.get('p95', 0):.3f}秒",
                f"P99查询时间: {query_perf.get('p99', 0):.3f}秒",
                f"最快查询: {query_perf.get('min', 0):.3f}秒",
                f"最慢查询: {query_perf.get('max', 0):.3f}秒"
            ])
        
        # 系统资源
        system_perf = performance.get('system_performance', {})
        if system_perf:
            report_lines.extend([
                "",
                "💻 系统资源使用:",
                "-" * 40,
                f"CPU使用率: 平均 {system_perf.get('cpu_average', 0):.1f}%, 最大 {system_perf.get('cpu_max', 0):.1f}%",
                f"内存使用率: 平均 {system_perf.get('memory_average', 0):.1f}%, 最大 {system_perf.get('memory_max', 0):.1f}%",
                f"监控样本数: {system_perf.get('samples_count', 0):,}"
            ])
        
        # 批次统计
        batch_results = scan_result.get('batch_results', [])
        if batch_results:
            successful_batches = len([b for b in batch_results if b['successful_stocks'] > 0])
            total_batches = len(batch_results)
            avg_batch_time = np.mean([b['processing_time'] for b in batch_results])
            
            report_lines.extend([
                "",
                "📦 批次处理统计:",
                "-" * 40,
                f"总批次数: {total_batches}",
                f"成功批次: {successful_batches}",
                f"批次成功率: {(successful_batches/max(total_batches,1)*100):.1f}%",
                f"平均批次时间: {avg_batch_time:.1f}秒"
            ])
        
        # 优化建议
        recommendations = scan_result.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "",
                "💡 优化建议:",
                "-" * 40,
            ])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    print("🚀 启动4000只股票全市场扫描性能基准测试...")
    
    scanner = FullMarketScanner()
    
    try:
        # 运行全市场扫描 (测试版本使用较小规模)
        target_count = 100  # 测试版本先用100只股票，生产版本可设为4000
        result = scanner.run_full_market_scan(
            target_stock_count=target_count,
            batch_size=20,
            max_workers=4
        )
        
        # 生成报告
        report = scanner.generate_report(result)
        print(report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"full_market_scan_benchmark_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 保存详细结果
        import json
        result_file = f"full_market_scan_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 详细结果已保存到: {result_file}")
        
        # 返回测试结果
        if result.get('scan_status') in ['Excellent', 'Good']:
            print("🎉 全市场扫描性能基准测试通过！")
            return 0
        else:
            print("⚠️  性能表现需要优化，请查看报告建议。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 