#!/usr/bin/env python3
"""
4000股选股性能测试

目标：从30分钟优化到5分钟
包含超时控制和实时监控
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import get_config
from strategy.advanced_strategy_executor import AdvancedStrategyExecutor
from analysis.engines.unified_analysis_engine import UnifiedAnalysisEngine
from monitoring.production_performance_monitor import ProductionPerformanceMonitor
from utils.logger import get_logger

logger = get_logger(__name__)


class StockSelectionPerformanceTest:
    """4000股选股性能测试"""
    
    # 性能目标和阈值
    PERFORMANCE_TARGETS = {
        'target_time': 300,          # 目标时间：5分钟（300秒）
        'warning_time': 600,         # 警告时间：10分钟
        'max_time': 1800,            # 最大允许时间：30分钟
        'batch_size': 100,           # 批处理大小
        'memory_limit': 85.0,        # 内存使用限制（%）
        'cpu_limit': 90.0            # CPU使用限制（%）
    }
    
    def __init__(self):
        """初始化测试器"""
        self.test_start_time = None
        self.is_stopping = False
        self._stop_event = threading.Event()
        
        # 初始化组件
        try:
            self.strategy_executor = AdvancedStrategyExecutor()
            self.analysis_engine = UnifiedAnalysisEngine()
            self.performance_monitor = ProductionPerformanceMonitor()
            logger.info("所有组件初始化成功")
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
        
        # 测试结果
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'batch_results': [],
            'performance_metrics': {},
            'timeout_events': [],
            'final_status': 'PENDING'
        }
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.warning(f"接收到停止信号 {signum}，开始停止测试...")
        self.stop_test("手动停止")
    
    def stop_test(self, reason: str = "超时"):
        """停止测试"""
        if not self.is_stopping:
            self.is_stopping = True
            self._stop_event.set()
            logger.warning(f"测试被停止: {reason}")
            
            self.test_results['timeout_events'].append({
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'elapsed_time': time.time() - self.test_start_time if self.test_start_time else 0
            })
    
    def _monitor_system_resources(self):
        """监控系统资源"""
        def monitor():
            while not self._stop_event.wait(5.0):  # 每5秒检查一次
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    if memory_percent > self.PERFORMANCE_TARGETS['memory_limit']:
                        self.stop_test(f"内存使用率过高: {memory_percent:.1f}%")
                        break
                    
                    if cpu_percent > self.PERFORMANCE_TARGETS['cpu_limit']:
                        logger.warning(f"CPU使用率较高: {cpu_percent:.1f}%")
                    
                    # 检查总测试时间
                    if self.test_start_time:
                        elapsed = time.time() - self.test_start_time
                        if elapsed > self.PERFORMANCE_TARGETS['max_time']:
                            self.stop_test(f"总测试时间超限: {elapsed:.1f}s")
                            break
                        elif elapsed > self.PERFORMANCE_TARGETS['warning_time']:
                            logger.warning(f"测试时间接近警告线: {elapsed:.1f}s")
                
                except Exception as e:
                    logger.error(f"资源监控失败: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def _get_test_stock_list(self, count: int = 4000) -> List[str]:
        """获取测试股票列表"""
        try:
            # 从数据库获取真实股票代码
            from db.unified_data_manager import get_unified_data_manager
            data_manager = get_unified_data_manager()
            
            # 获取最近有数据的股票
            query_result = data_manager.get_available_stocks(
                start_date='2024-01-01',
                limit=count
            )
            
            if query_result and hasattr(query_result, 'data'):
                stock_codes = query_result.data['code'].unique().tolist()[:count]
                logger.info(f"获取到 {len(stock_codes)} 个测试股票代码")
                return stock_codes
            else:
                logger.warning("无法从数据库获取股票列表，使用模拟数据")
                return self._generate_mock_stock_list(count)
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}，使用模拟数据")
            return self._generate_mock_stock_list(count)
    
    def _generate_mock_stock_list(self, count: int) -> List[str]:
        """生成模拟股票列表"""
        stock_codes = []
        
        # 沪A股票 (000001-002999)
        for i in range(1, min(3000, count + 1)):
            stock_codes.append(f"{i:06d}")
        
        # 深A股票 (600000-999999) 
        if len(stock_codes) < count:
            remaining = count - len(stock_codes)
            for i in range(600000, 600000 + remaining):
                stock_codes.append(f"{i}")
        
        return stock_codes[:count]
    
    def _process_stock_batch(self, stocks: List[str], batch_id: int) -> Dict[str, Any]:
        """处理股票批次"""
        logger.info(f"处理批次 {batch_id}，股票数量: {len(stocks)}")
        batch_start = time.time()
        
        batch_result = {
            'batch_id': batch_id,
            'stock_count': len(stocks),
            'processed_count': 0,
            'successful_count': 0,
            'failed_count': 0,
            'duration': 0,
            'errors': []
        }
        
        try:
            # 使用策略执行器批量处理
            for i, stock_code in enumerate(stocks):
                if self.is_stopping or self._stop_event.is_set():
                    logger.info(f"批次 {batch_id} 处理被中断")
                    break
                
                try:
                    # 模拟股票分析处理
                    analysis_result = self.analysis_engine.analyze_stock(
                        stock_code=stock_code,
                        start_date='2024-01-01',
                        end_date='2024-01-31'
                    )
                    
                    if analysis_result and analysis_result.get('success', False):
                        batch_result['successful_count'] += 1
                    else:
                        batch_result['failed_count'] += 1
                    
                    batch_result['processed_count'] += 1
                    
                    # 定期检查时间
                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - batch_start
                        progress = (i + 1) / len(stocks)
                        estimated_total = elapsed / progress if progress > 0 else 0
                        
                        if estimated_total > self.PERFORMANCE_TARGETS['target_time'] / 10:  # 每批次不应超过目标时间的1/10
                            logger.warning(f"批次 {batch_id} 处理速度较慢，预计总时间: {estimated_total:.1f}s")
                    
                except Exception as e:
                    batch_result['errors'].append(f"股票 {stock_code}: {str(e)}")
                    batch_result['failed_count'] += 1
                    batch_result['processed_count'] += 1
                    logger.debug(f"处理股票 {stock_code} 失败: {e}")
            
        except Exception as e:
            logger.error(f"批次 {batch_id} 处理异常: {e}")
            batch_result['errors'].append(f"批次异常: {str(e)}")
        
        finally:
            batch_result['duration'] = time.time() - batch_start
            batch_result['success_rate'] = (
                batch_result['successful_count'] / batch_result['processed_count'] 
                if batch_result['processed_count'] > 0 else 0
            )
        
        logger.info(f"批次 {batch_id} 完成，耗时: {batch_result['duration']:.2f}s，"
                   f"成功率: {batch_result['success_rate']:.2%}")
        
        return batch_result
    
    def run_4000_stock_selection_test(self, stock_count: int = 4000) -> Dict[str, Any]:
        """运行4000股选股测试"""
        logger.info("=" * 80)
        logger.info(f"开始{stock_count}股选股性能测试")
        logger.info(f"目标时间: {self.PERFORMANCE_TARGETS['target_time']}秒")
        logger.info("=" * 80)
        
        self.test_start_time = time.time()
        self.test_results['start_time'] = datetime.now().isoformat()
        
        # 启动资源监控
        monitor_thread = self._monitor_system_resources()
        
        try:
            # 获取股票列表
            logger.info("获取测试股票列表...")
            stock_list = self._get_test_stock_list(stock_count)
            
            if len(stock_list) < stock_count:
                logger.warning(f"实际获取股票数量 {len(stock_list)} 少于目标 {stock_count}")
            
            # 分批处理
            batch_size = self.PERFORMANCE_TARGETS['batch_size']
            total_batches = (len(stock_list) + batch_size - 1) // batch_size
            
            logger.info(f"开始分批处理，批次大小: {batch_size}，总批次: {total_batches}")
            
            for batch_id in range(total_batches):
                if self.is_stopping:
                    logger.info("测试被中断，停止处理")
                    break
                
                start_idx = batch_id * batch_size
                end_idx = min(start_idx + batch_size, len(stock_list))
                batch_stocks = stock_list[start_idx:end_idx]
                
                # 处理批次
                batch_result = self._process_stock_batch(batch_stocks, batch_id + 1)
                self.test_results['batch_results'].append(batch_result)
                
                # 检查批次性能
                if batch_result['duration'] > 60:  # 单批次超过1分钟
                    logger.warning(f"批次 {batch_id + 1} 处理时间过长: {batch_result['duration']:.2f}s")
                
                # 计算进度和预估时间
                elapsed_time = time.time() - self.test_start_time
                processed_stocks = sum(b['processed_count'] for b in self.test_results['batch_results'])
                
                if processed_stocks > 0:
                    progress = processed_stocks / len(stock_list)
                    estimated_total_time = elapsed_time / progress
                    
                    logger.info(f"进度: {progress:.1%}，已耗时: {elapsed_time:.1f}s，"
                               f"预计总时间: {estimated_total_time:.1f}s")
                    
                    # 检查是否超出目标时间
                    if estimated_total_time > self.PERFORMANCE_TARGETS['warning_time']:
                        logger.warning(f"预计总时间超出警告线: {estimated_total_time:.1f}s")
                    
                    if estimated_total_time > self.PERFORMANCE_TARGETS['max_time']:
                        self.stop_test(f"预计总时间超出最大限制: {estimated_total_time:.1f}s")
                        break
            
            # 计算最终结果
            total_duration = time.time() - self.test_start_time
            total_processed = sum(b['processed_count'] for b in self.test_results['batch_results'])
            total_successful = sum(b['successful_count'] for b in self.test_results['batch_results'])
            
            self.test_results['performance_metrics'] = {
                'total_duration': total_duration,
                'total_processed': total_processed,
                'total_successful': total_successful,
                'overall_success_rate': total_successful / total_processed if total_processed > 0 else 0,
                'stocks_per_second': total_processed / total_duration if total_duration > 0 else 0,
                'target_achieved': total_duration <= self.PERFORMANCE_TARGETS['target_time'],
                'warning_exceeded': total_duration > self.PERFORMANCE_TARGETS['warning_time'],
                'max_time_exceeded': total_duration > self.PERFORMANCE_TARGETS['max_time']
            }
            
            # 设置最终状态
            if self.is_stopping:
                self.test_results['final_status'] = 'STOPPED'
            elif total_duration <= self.PERFORMANCE_TARGETS['target_time']:
                self.test_results['final_status'] = 'TARGET_ACHIEVED'
            elif total_duration <= self.PERFORMANCE_TARGETS['warning_time']:
                self.test_results['final_status'] = 'ACCEPTABLE'
            elif total_duration <= self.PERFORMANCE_TARGETS['max_time']:
                self.test_results['final_status'] = 'WARNING'
            else:
                self.test_results['final_status'] = 'FAILED'
            
            logger.info(f"测试完成，总耗时: {total_duration:.2f}s，"
                       f"处理股票: {total_processed}/{len(stock_list)}，"
                       f"状态: {self.test_results['final_status']}")
            
        except Exception as e:
            logger.error(f"测试过程发生异常: {e}")
            self.test_results['final_status'] = 'EXCEPTION'
            self.test_results['exception'] = str(e)
        
        finally:
            self.test_results['end_time'] = datetime.now().isoformat()
            self._stop_event.set()  # 停止监控线程
        
        return self.test_results
    
    def generate_performance_report(self) -> str:
        """生成性能测试报告"""
        if not self.test_results['batch_results']:
            return "❌ 没有可用的测试结果"
        
        metrics = self.test_results.get('performance_metrics', {})
        
        report = []
        report.append("=" * 80)
        report.append("4000股选股性能测试报告")
        report.append("=" * 80)
        report.append(f"测试开始时间: {self.test_results['start_time']}")
        report.append(f"测试结束时间: {self.test_results['end_time']}")
        report.append(f"最终状态: {self.test_results['final_status']}")
        report.append("")
        
        # 性能指标
        report.append("📊 性能指标:")
        report.append(f"  总耗时: {metrics.get('total_duration', 0):.2f}秒")
        report.append(f"  目标时间: {self.PERFORMANCE_TARGETS['target_time']}秒")
        report.append(f"  处理股票数: {metrics.get('total_processed', 0)}")
        report.append(f"  成功股票数: {metrics.get('total_successful', 0)}")
        report.append(f"  成功率: {metrics.get('overall_success_rate', 0):.2%}")
        report.append(f"  处理速度: {metrics.get('stocks_per_second', 0):.1f} 股票/秒")
        report.append("")
        
        # 目标达成情况
        report.append("🎯 目标达成情况:")
        target_achieved = metrics.get('target_achieved', False)
        warning_exceeded = metrics.get('warning_exceeded', False)
        max_exceeded = metrics.get('max_time_exceeded', False)
        
        if target_achieved:
            report.append("  ✅ 已达成5分钟目标")
        elif not warning_exceeded:
            report.append("  ⚠️ 未达成5分钟目标，但在可接受范围内")
        elif not max_exceeded:
            report.append("  ❌ 超出警告时间，需要优化")
        else:
            report.append("  💥 严重超时，需要重大优化")
        
        # 批次性能
        if self.test_results['batch_results']:
            avg_batch_time = sum(b['duration'] for b in self.test_results['batch_results']) / len(self.test_results['batch_results'])
            best_batch_time = min(b['duration'] for b in self.test_results['batch_results'])
            worst_batch_time = max(b['duration'] for b in self.test_results['batch_results'])
            
            report.append("📈 批次性能分析:")
            report.append(f"  总批次数: {len(self.test_results['batch_results'])}")
            report.append(f"  平均批次时间: {avg_batch_time:.2f}秒")
            report.append(f"  最快批次: {best_batch_time:.2f}秒")
            report.append(f"  最慢批次: {worst_batch_time:.2f}秒")
        
        # 超时事件
        if self.test_results['timeout_events']:
            report.append("")
            report.append("⚠️ 超时事件:")
            for event in self.test_results['timeout_events']:
                report.append(f"  - {event['reason']} (时间: {event['elapsed_time']:.1f}s)")
        
        # 优化建议
        report.append("")
        report.append("💡 优化建议:")
        
        if target_achieved:
            report.append("  - 性能表现优秀，可考虑增加功能复杂度")
        else:
            total_time = metrics.get('total_duration', 0)
            target_time = self.PERFORMANCE_TARGETS['target_time']
            improvement_needed = (total_time - target_time) / target_time * 100
            
            report.append(f"  - 需要提升性能 {improvement_needed:.1f}% 以达成目标")
            
            if improvement_needed > 100:
                report.append("  - 建议重构核心算法和数据访问层")
                report.append("  - 考虑增加缓存和并行处理")
            elif improvement_needed > 50:
                report.append("  - 优化数据库查询和批处理逻辑")
                report.append("  - 增强内存管理和数据预加载")
            else:
                report.append("  - 微调算法参数和查询优化")
        
        return "\n".join(report)


def main():
    """主函数"""
    print("=" * 80)
    print("4000股选股性能测试")
    print("目标：从30分钟优化到5分钟")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 创建测试实例
        test_framework = StockSelectionPerformanceTest()
        
        # 显示性能目标
        print("🎯 性能目标:")
        for key, value in test_framework.PERFORMANCE_TARGETS.items():
            print(f"  {key}: {value}")
        print()
        
        # 运行测试
        print("🚀 开始执行4000股选股测试...")
        results = test_framework.run_4000_stock_selection_test(stock_count=4000)
        
        # 显示测试报告
        print()
        print(test_framework.generate_performance_report())
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results/4000_stock_selection_test_{timestamp}.json"
        
        try:
            import json
            os.makedirs("results", exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"📄 详细结果已保存到: {result_file}")
        except Exception as e:
            print(f"⚠️ 结果保存失败: {e}")
        
        # 返回状态码
        if results['final_status'] in ['TARGET_ACHIEVED', 'ACCEPTABLE']:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        return 130
    except Exception as e:
        print(f"\n💥 测试执行异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 