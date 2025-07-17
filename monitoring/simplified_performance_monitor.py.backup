#!/usr/bin/env python3
"""
简化版生产级性能监控系统

核心功能：
1. 性能指标记录和监控
2. 告警机制
3. 数据导出和报告
4. 实时状态监控
"""

import time
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

@dataclass
class PerformanceRecord:
    """性能记录"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    status: str  # 'normal', 'warning', 'critical'

@dataclass 
class Alert:
    """告警信息"""
    timestamp: datetime
    level: str
    metric_name: str
    message: str
    value: float
    threshold: float

class SimplifiedPerformanceMonitor:
    """简化版性能监控器"""
    
    def __init__(self):
        # 性能数据存储
        self.performance_records = deque(maxlen=10000)
        self.alerts = deque(maxlen=1000)
        self.metric_stats = defaultdict(dict)
        
        # 性能阈值配置
        self.thresholds = {
            # 股票分析性能阈值 (秒)
            'stock_processing_time': {'warning': 1.0, 'critical': 3.0},
            'indicator_calculation_time': {'warning': 0.1, 'critical': 0.5},
            'database_query_time': {'warning': 0.5, 'critical': 2.0},
            'vectorization_speedup': {'warning': 2.0, 'critical': 1.5},  # 低于阈值告警
            
            # 业务指标阈值 (百分比)
            'cache_hit_rate': {'warning': 60.0, 'critical': 40.0},  # 低于阈值告警
            'success_rate': {'warning': 90.0, 'critical': 80.0},    # 低于阈值告警
            'error_rate': {'warning': 5.0, 'critical': 10.0},       # 高于阈值告警
            
            # 系统性能阈值
            'memory_usage_mb': {'warning': 1000, 'critical': 2000},
            'concurrent_operations': {'warning': 40, 'critical': 50},
            'response_time_ms': {'warning': 500, 'critical': 1000},
        }
        
        print("🎯 简化版性能监控系统初始化完成")
        print(f"📊 已配置 {len(self.thresholds)} 个性能指标阈值")
    
    def record_metric(self, metric_name: str, value: float, unit: str = "") -> PerformanceRecord:
        """记录性能指标"""
        timestamp = datetime.now()
        
        # 判断状态
        status = self._evaluate_metric_status(metric_name, value)
        
        # 创建记录
        record = PerformanceRecord(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            unit=unit,
            status=status
        )
        
        # 存储记录
        self.performance_records.append(record)
        
        # 更新统计信息
        self._update_metric_stats(metric_name, value, status)
        
        # 检查告警
        if status in ['warning', 'critical']:
            self._create_alert(record)
        
        return record
    
    def _evaluate_metric_status(self, metric_name: str, value: float) -> str:
        """评估指标状态"""
        if metric_name not in self.thresholds:
            return 'normal'
        
        thresholds = self.thresholds[metric_name]
        
        # 对于一些指标，低值是告警条件
        reverse_logic_metrics = ['cache_hit_rate', 'success_rate', 'vectorization_speedup']
        
        if metric_name in reverse_logic_metrics:
            # 值越低越危险
            if value <= thresholds['critical']:
                return 'critical'
            elif value <= thresholds['warning']:
                return 'warning'
            else:
                return 'normal'
        else:
            # 值越高越危险
            if value >= thresholds['critical']:
                return 'critical'
            elif value >= thresholds['warning']:
                return 'warning'
            else:
                return 'normal'
    
    def _update_metric_stats(self, metric_name: str, value: float, status: str):
        """更新指标统计"""
        if metric_name not in self.metric_stats:
            self.metric_stats[metric_name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'status_counts': {'normal': 0, 'warning': 0, 'critical': 0}
            }
        
        stats = self.metric_stats[metric_name]
        stats['count'] += 1
        stats['sum'] += value
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        stats['status_counts'][status] += 1
    
    def _create_alert(self, record: PerformanceRecord):
        """创建告警"""
        thresholds = self.thresholds[record.metric_name]
        threshold = thresholds['critical'] if record.status == 'critical' else thresholds['warning']
        
        alert = Alert(
            timestamp=record.timestamp,
            level=record.status,
            metric_name=record.metric_name,
            message=f"{record.metric_name} {record.status}: {record.value:.2f} {record.unit}",
            value=record.value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # 打印告警
        emoji = {'warning': '⚠️', 'critical': '🚨'}
        print(f"{emoji.get(record.status, '📊')} [{record.status.upper()}] {alert.message}")
        
        # 添加建议
        suggestions = self._get_suggestion(record.metric_name)
        if suggestions:
            print(f"   💡 {suggestions}")
    
    def _get_suggestion(self, metric_name: str) -> str:
        """获取优化建议"""
        suggestions = {
            'stock_processing_time': "建议使用并行处理、缓存或算法优化",
            'indicator_calculation_time': "建议使用向量化计算优化指标算法",
            'database_query_time': "建议优化SQL查询、使用索引或连接池",
            'cache_hit_rate': "建议调整缓存策略、增加缓存容量",
            'success_rate': "建议检查业务逻辑、优化策略参数",
            'error_rate': "建议检查错误日志、修复代码问题",
            'vectorization_speedup': "建议增加向量化指标覆盖率",
            'memory_usage_mb': "建议优化内存使用、检查内存泄漏",
            'response_time_ms': "建议优化响应时间、使用异步处理"
        }
        return suggestions.get(metric_name, "建议进一步分析性能瓶颈")
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """获取性能摘要"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 筛选时间范围内的数据
        recent_records = [r for r in self.performance_records 
                         if start_time <= r.timestamp <= end_time]
        recent_alerts = [a for a in self.alerts 
                        if start_time <= a.timestamp <= end_time]
        
        # 按指标分组统计
        metric_summary = defaultdict(list)
        for record in recent_records:
            metric_summary[record.metric_name].append(record.value)
        
        # 计算统计信息
        stats = {}
        for metric_name, values in metric_summary.items():
            if values:
                stats[metric_name] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        # 告警统计
        alert_stats = defaultdict(int)
        for alert in recent_alerts:
            alert_stats[alert.level] += 1
        
        summary = {
            'period': f"最近{hours}小时",
            'total_records': len(recent_records),
            'total_alerts': len(recent_alerts),
            'metrics_summary': stats,
            'alert_summary': dict(alert_stats),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能建议"""
        recommendations = []
        
        # 基于告警历史生成建议
        recent_alerts = [a for a in self.alerts 
                        if a.timestamp > datetime.now() - timedelta(hours=1)]
        
        if not recent_alerts:
            recommendations.append("✅ 系统运行良好，所有指标正常")
            return recommendations
        
        # 按指标分组告警
        alert_by_metric = defaultdict(list)
        for alert in recent_alerts:
            alert_by_metric[alert.metric_name].append(alert)
        
        # 生成针对性建议
        for metric_name, alerts in alert_by_metric.items():
            critical_count = sum(1 for a in alerts if a.level == 'critical')
            warning_count = sum(1 for a in alerts if a.level == 'warning')
            
            if critical_count > 0:
                recommendations.append(f"🚨 {metric_name}有{critical_count}个严重告警，需要立即处理")
            elif warning_count > 0:
                recommendations.append(f"⚠️ {metric_name}有{warning_count}个警告，建议关注")
        
        # 系统级建议
        if len(recent_alerts) > 10:
            recommendations.append("📊 告警频繁，建议全面检查系统性能")
        
        return recommendations
    
    def export_report(self, file_path: str = None) -> str:
        """导出性能报告"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"reports/performance_report_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 准备报告数据
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': self.get_performance_summary(24),  # 24小时摘要
            'thresholds': self.thresholds,
            'recent_records': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'metric_name': r.metric_name,
                    'value': r.value,
                    'unit': r.unit,
                    'status': r.status
                }
                for r in list(self.performance_records)[-100:]  # 最近100条记录
            ],
            'recent_alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'level': a.level,
                    'metric_name': a.metric_name,
                    'message': a.message,
                    'value': a.value,
                    'threshold': a.threshold
                }
                for a in list(self.alerts)[-50:]  # 最近50个告警
            ]
        }
        
        # 保存报告
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 性能报告已导出: {file_path}")
        return file_path
    
    def benchmark_system_performance(self) -> Dict[str, Any]:
        """系统性能基准测试"""
        print("🧪 开始系统性能基准测试...")
        
        # 模拟股票处理性能测试
        print("  📈 测试股票处理性能...")
        stock_times = []
        for i in range(10):
            start_time = time.time()
            # 模拟股票处理
            time.sleep(0.01)  # 模拟10ms处理时间
            processing_time = time.time() - start_time
            stock_times.append(processing_time)
            self.record_metric('stock_processing_time', processing_time, '秒/股')
        
        # 模拟指标计算性能测试
        print("  📊 测试指标计算性能...")
        indicator_times = []
        for i in range(20):
            start_time = time.time()
            # 模拟指标计算
            time.sleep(0.005)  # 模拟5ms计算时间
            calc_time = time.time() - start_time
            indicator_times.append(calc_time)
            self.record_metric('indicator_calculation_time', calc_time, '秒/指标')
        
        # 模拟缓存命中率测试
        print("  💾 测试缓存性能...")
        import random
        for i in range(30):
            hit_rate = random.uniform(70, 95)  # 模拟70-95%的命中率
            self.record_metric('cache_hit_rate', hit_rate, '%')
        
        # 模拟向量化性能
        print("  ⚡ 测试向量化性能...")
        for i in range(15):
            speedup = random.uniform(2.5, 4.0)  # 模拟2.5-4.0倍加速
            self.record_metric('vectorization_speedup', speedup, '倍')
        
        benchmark_results = {
            'stock_processing': {
                'avg_time': sum(stock_times) / len(stock_times),
                'max_time': max(stock_times),
                'throughput_per_hour': 3600 / (sum(stock_times) / len(stock_times))
            },
            'indicator_calculation': {
                'avg_time': sum(indicator_times) / len(indicator_times),
                'max_time': max(indicator_times),
                'indicators_per_second': 1 / (sum(indicator_times) / len(indicator_times))
            },
            'test_summary': self.get_performance_summary(1)
        }
        
        print("✅ 性能基准测试完成")
        return benchmark_results


# 全局监控器实例
_monitor = None

def get_monitor() -> SimplifiedPerformanceMonitor:
    """获取监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = SimplifiedPerformanceMonitor()
    return _monitor

def record_performance(metric_name: str, value: float, unit: str = ""):
    """便捷的性能记录函数"""
    monitor = get_monitor()
    return monitor.record_metric(metric_name, value, unit)


def main():
    """测试主函数"""
    print("🚀 开始简化版性能监控测试...")
    
    # 获取监控器
    monitor = get_monitor()
    
    # 执行基准测试
    benchmark_results = monitor.benchmark_system_performance()
    
    print(f"\n📊 基准测试结果:")
    print(f"股票处理: 平均 {benchmark_results['stock_processing']['avg_time']*1000:.1f}ms/股, "
          f"吞吐量 {benchmark_results['stock_processing']['throughput_per_hour']:.0f}股/小时")
    print(f"指标计算: 平均 {benchmark_results['indicator_calculation']['avg_time']*1000:.1f}ms/指标, "
          f"速度 {benchmark_results['indicator_calculation']['indicators_per_second']:.0f}指标/秒")
    
    # 模拟一些异常情况
    print(f"\n⚠️  模拟异常情况...")
    monitor.record_metric('stock_processing_time', 1.5, '秒/股')  # 超过warning
    monitor.record_metric('error_rate', 8.0, '%')                # 超过warning
    monitor.record_metric('cache_hit_rate', 35.0, '%')           # 低于critical
    
    # 获取性能摘要
    summary = monitor.get_performance_summary(1)
    
    print(f"\n📈 性能摘要:")
    print(f"记录总数: {summary['total_records']}")
    print(f"告警总数: {summary['total_alerts']}")
    
    if summary['alert_summary']:
        print("告警分布:", summary['alert_summary'])
    
    print("\n💡 系统建议:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # 导出报告
    report_file = monitor.export_report()
    
    print(f"\n✅ 性能监控测试完成")
    print(f"📄 详细报告: {report_file}")


if __name__ == "__main__":
    main() 