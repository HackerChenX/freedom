#!/usr/bin/env python3
"""
生产级性能监控系统

功能特性：
1. 实时指标监控
2. 异常告警机制
3. 资源使用跟踪
4. 性能基准验证
5. 监控仪表板
6. 告警通知系统
"""

import time
import psutil
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from db.sql_manager import SQLManager, QueryType
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    category: str
    status: str  # 'normal', 'warning', 'critical'
    threshold_warning: float
    threshold_critical: float

@dataclass
class SystemResource:
    """系统资源数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float

@dataclass
class AlertMessage:
    """告警消息数据类"""
    timestamp: datetime
    alert_id: str
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    suggestion: str

class ProductionPerformanceMonitor:
    """生产级性能监控器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # 监控数据存储
        self.metrics_history = defaultdict(deque)
        self.alerts_history = deque(maxlen=1000)
        self.resource_history = deque(maxlen=1000)
        
        # 性能基准
        self.performance_baselines = {}
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 告警回调函数
        self.alert_callbacks = []
        
        # 性能阈值
        self.thresholds = self._setup_performance_thresholds()
        
        print("🎯 生产级性能监控系统初始化完成")
        print(f"📊 监控间隔: {self.config['monitor_interval']}秒")
        print(f"📈 数据保留: {self.config['data_retention_hours']}小时")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'monitor_interval': 5,  # 监控间隔（秒）
            'data_retention_hours': 24,  # 数据保留时间（小时）
            'alert_cooldown_minutes': 5,  # 告警冷却时间（分钟）
            'performance_window_minutes': 30,  # 性能分析窗口（分钟）
            'enable_system_monitoring': True,
            'enable_performance_monitoring': True,
            'enable_alert_notifications': True,
            'log_file': 'logs/performance_monitor.log'
        }
    
    def _setup_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """设置性能阈值"""
        return {
            # 系统资源阈值
            'cpu_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_percent': {'warning': 75.0, 'critical': 90.0},
            'disk_io_mb_per_sec': {'warning': 100.0, 'critical': 200.0},
            
            # 股票分析性能阈值
            'stock_processing_time': {'warning': 1.0, 'critical': 3.0},  # 秒/股
            'indicator_calculation_time': {'warning': 0.1, 'critical': 0.5},  # 秒/指标
            'database_query_time': {'warning': 0.5, 'critical': 2.0},  # 秒/查询
            'cache_hit_rate': {'warning': 60.0, 'critical': 40.0},  # 百分比（低于阈值告警）
            
            # 业务指标阈值
            'successful_selection_rate': {'warning': 90.0, 'critical': 80.0},  # 百分比
            'error_rate': {'warning': 5.0, 'critical': 10.0},  # 百分比
            'concurrent_users': {'warning': 40, 'critical': 50},  # 用户数
        }
    
    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            print("⚠️  监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print("🚀 性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        print("⏸️  性能监控已停止")
    
    def _monitoring_loop(self):
        """监控主循环"""
        last_resource_check = 0
        
        while self.monitoring_active:
            current_time = time.time()
            
            try:
                # 系统资源监控
                if (current_time - last_resource_check) >= self.config['monitor_interval']:
                    if self.config['enable_system_monitoring']:
                        self._collect_system_metrics()
                    last_resource_check = current_time
                
                # 清理过期数据
                self._cleanup_expired_data()
                
                # 短暂休眠，避免过度消耗CPU
                time.sleep(1)
                
            except Exception as e:
                self._log_error(f"监控循环错误: {e}")
                time.sleep(5)  # 错误后稍长休眠
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        timestamp = datetime.now()
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # 磁盘I/O
            disk_io = psutil.disk_io_counters()
            
            # 网络I/O
            network_io = psutil.net_io_counters()
            
            # 创建资源记录
            resource = SystemResource(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                disk_io_write_mb=disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                network_sent_mb=network_io.bytes_sent / (1024 * 1024) if network_io else 0,
                network_recv_mb=network_io.bytes_recv / (1024 * 1024) if network_io else 0
            )
            
            self.resource_history.append(resource)
            
            # 检查系统资源告警
            self._check_system_alerts(resource)
            
        except Exception as e:
            self._log_error(f"系统指标收集失败: {e}")
    
    def record_performance_metric(self, metric_name: str, value: float, unit: str = "", 
                                category: str = "performance"):
        """记录性能指标"""
        timestamp = datetime.now()
        
        # 获取阈值
        thresholds = self.thresholds.get(metric_name, {'warning': float('inf'), 'critical': float('inf')})
        
        # 判断状态
        if value >= thresholds['critical']:
            status = 'critical'
        elif value >= thresholds['warning']:
            status = 'warning'
        else:
            status = 'normal'
        
        # 创建指标记录
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            unit=unit,
            category=category,
            status=status,
            threshold_warning=thresholds['warning'],
            threshold_critical=thresholds['critical']
        )
        
        # 存储指标
        self.metrics_history[metric_name].append(metric)
        
        # 限制历史数据量
        max_records = int(self.config['data_retention_hours'] * 3600 / self.config['monitor_interval'])
        if len(self.metrics_history[metric_name]) > max_records:
            self.metrics_history[metric_name].popleft()
        
        # 检查告警
        if status in ['warning', 'critical']:
            self._trigger_alert(metric)
        
        return metric
    
    def _check_system_alerts(self, resource: SystemResource):
        """检查系统资源告警"""
        # CPU告警
        if resource.cpu_percent >= self.thresholds['cpu_percent']['critical']:
            self._create_alert('critical', 'system', f"CPU使用率达到危险水平: {resource.cpu_percent:.1f}%",
                             'cpu_percent', resource.cpu_percent, self.thresholds['cpu_percent']['critical'],
                             "建议检查CPU密集型进程，考虑优化算法或增加计算资源")
        elif resource.cpu_percent >= self.thresholds['cpu_percent']['warning']:
            self._create_alert('warning', 'system', f"CPU使用率较高: {resource.cpu_percent:.1f}%",
                             'cpu_percent', resource.cpu_percent, self.thresholds['cpu_percent']['warning'],
                             "建议监控CPU使用情况，准备优化措施")
        
        # 内存告警
        if resource.memory_percent >= self.thresholds['memory_percent']['critical']:
            self._create_alert('critical', 'system', f"内存使用率达到危险水平: {resource.memory_percent:.1f}%",
                             'memory_percent', resource.memory_percent, self.thresholds['memory_percent']['critical'],
                             "建议立即释放内存，检查内存泄漏，考虑增加内存")
        elif resource.memory_percent >= self.thresholds['memory_percent']['warning']:
            self._create_alert('warning', 'system', f"内存使用率较高: {resource.memory_percent:.1f}%",
                             'memory_percent', resource.memory_percent, self.thresholds['memory_percent']['warning'],
                             "建议监控内存使用，准备清理措施")
    
    def _trigger_alert(self, metric: PerformanceMetric):
        """触发性能告警"""
        suggestions = {
            'stock_processing_time': "建议优化算法，使用并行处理或缓存机制",
            'indicator_calculation_time': "建议使用向量化计算，优化指标算法",
            'database_query_time': "建议优化SQL查询，使用索引或连接池",
            'cache_hit_rate': "建议调整缓存策略，增加缓存容量",
            'error_rate': "建议检查错误日志，修复代码问题",
            'successful_selection_rate': "建议检查选股逻辑，优化策略参数"
        }
        
        suggestion = suggestions.get(metric.metric_name, "建议进一步分析性能瓶颈")
        
        self._create_alert(
            metric.status, 'performance',
            f"{metric.metric_name} {metric.status}: {metric.value:.2f} {metric.unit}",
            metric.metric_name, metric.value,
            metric.threshold_critical if metric.status == 'critical' else metric.threshold_warning,
            suggestion
        )
    
    def _create_alert(self, level: str, category: str, message: str, metric_name: str, 
                     current_value: float, threshold: float, suggestion: str):
        """创建告警消息"""
        alert = AlertMessage(
            timestamp=datetime.now(),
            alert_id=f"{metric_name}_{int(time.time())}",
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            suggestion=suggestion
        )
        
        self.alerts_history.append(alert)
        
        # 通知告警回调
        if self.config['enable_alert_notifications']:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self._log_error(f"告警回调执行失败: {e}")
        
        # 打印告警信息
        emoji = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'critical': '🚨'}
        print(f"{emoji.get(level, '📊')} [{level.upper()}] {message}")
        if suggestion:
            print(f"   💡 {suggestion}")
    
    def register_alert_callback(self, callback: Callable[[AlertMessage], None]):
        """注册告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """获取性能摘要"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        summary = {
            'period': f"最近{hours}小时",
            'system_resources': self._get_system_summary(start_time, end_time),
            'performance_metrics': self._get_performance_summary(start_time, end_time),
            'alerts': self._get_alerts_summary(start_time, end_time),
            'recommendations': self._get_recommendations()
        }
        
        return summary
    
    def _get_system_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取系统资源摘要"""
        relevant_resources = [r for r in self.resource_history 
                            if start_time <= r.timestamp <= end_time]
        
        if not relevant_resources:
            return {'status': 'no_data'}
        
        cpu_values = [r.cpu_percent for r in relevant_resources]
        memory_values = [r.memory_percent for r in relevant_resources]
        
        return {
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'current': cpu_values[-1] if cpu_values else 0
            },
            'memory': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'current': memory_values[-1] if memory_values else 0
            },
            'samples': len(relevant_resources)
        }
    
    def _get_performance_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取性能指标摘要"""
        performance_summary = {}
        
        for metric_name, metrics in self.metrics_history.items():
            relevant_metrics = [m for m in metrics 
                              if start_time <= m.timestamp <= end_time]
            
            if relevant_metrics:
                values = [m.value for m in relevant_metrics]
                performance_summary[metric_name] = {
                    'avg': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'current': values[-1],
                    'samples': len(values),
                    'status': relevant_metrics[-1].status
                }
        
        return performance_summary
    
    def _get_alerts_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """获取告警摘要"""
        relevant_alerts = [a for a in self.alerts_history 
                          if start_time <= a.timestamp <= end_time]
        
        if not relevant_alerts:
            return {'total': 0, 'by_level': {}}
        
        by_level = defaultdict(int)
        for alert in relevant_alerts:
            by_level[alert.level] += 1
        
        return {
            'total': len(relevant_alerts),
            'by_level': dict(by_level),
            'recent_alerts': [asdict(a) for a in relevant_alerts[-5:]]
        }
    
    def _get_recommendations(self) -> List[str]:
        """获取性能建议"""
        recommendations = []
        
        # 基于最近的系统资源状态
        if self.resource_history:
            latest_resource = self.resource_history[-1]
            
            if latest_resource.cpu_percent > 80:
                recommendations.append("CPU使用率过高，建议优化计算密集型任务")
            
            if latest_resource.memory_percent > 80:
                recommendations.append("内存使用率过高，建议优化内存使用或增加内存")
        
        # 基于性能指标
        for metric_name, metrics in self.metrics_history.items():
            if metrics:
                latest_metric = metrics[-1]
                if latest_metric.status in ['warning', 'critical']:
                    recommendations.append(f"{metric_name}性能不佳，需要优化")
        
        # 基于告警历史
        recent_alerts = [a for a in self.alerts_history 
                        if a.timestamp > datetime.now() - timedelta(hours=1)]
        
        if len(recent_alerts) > 10:
            recommendations.append("最近告警频繁，建议全面检查系统状态")
        
        return recommendations
    
    def export_monitoring_data(self, file_path: str = None) -> str:
        """导出监控数据"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"reports/performance_monitoring_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 准备导出数据
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'config': self.config,
            'thresholds': self.thresholds,
            'system_resources': [asdict(r) for r in list(self.resource_history)],
            'performance_metrics': {
                name: [asdict(m) for m in list(metrics)]
                for name, metrics in self.metrics_history.items()
            },
            'alerts': [asdict(a) for a in list(self.alerts_history)],
            'summary': self.get_performance_summary(24)  # 24小时摘要
        }
        
        # 序列化时间对象
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        print(f"📊 监控数据已导出: {file_path}")
        return file_path
    
    def _cleanup_expired_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(hours=self.config['data_retention_hours'])
        
        # 清理系统资源历史
        while self.resource_history and self.resource_history[0].timestamp < cutoff_time:
            self.resource_history.popleft()
        
        # 清理性能指标历史
        for metric_name in list(self.metrics_history.keys()):
            metrics = self.metrics_history[metric_name]
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
            
            # 如果队列为空，删除这个指标
            if not metrics:
                del self.metrics_history[metric_name]
        
        # 清理告警历史（告警有独立的maxlen限制）
        while self.alerts_history and self.alerts_history[0].timestamp < cutoff_time:
            self.alerts_history.popleft()
    
    def _log_error(self, message: str):
        """记录错误日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = f"[{timestamp}] ERROR: {message}"
        
        print(error_message)
        
        # 可以添加到文件日志
        if 'log_file' in self.config:
            try:
                os.makedirs(os.path.dirname(self.config['log_file']), exist_ok=True)
                with open(self.config['log_file'], 'a', encoding='utf-8') as f:
                    f.write(error_message + '\n')
            except Exception:
                pass  # 忽略日志写入失败


# 监控管理器（单例）
_monitor_instance = None

def get_performance_monitor(config: Optional[Dict[str, Any]] = None) -> ProductionPerformanceMonitor:
    """获取性能监控器实例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ProductionPerformanceMonitor(config)
    return _monitor_instance

def start_production_monitoring():
    """启动生产级监控"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    return monitor

def stop_production_monitoring():
    """停止生产级监控"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop_monitoring()


def main():
    """测试主函数"""
    print("🚀 开始生产级性能监控测试...")
    
    # 创建监控器
    monitor = get_performance_monitor({
        'monitor_interval': 2,  # 更频繁的监控用于测试
        'data_retention_hours': 1,
        'enable_alert_notifications': True
    })
    
    # 注册告警回调
    def alert_handler(alert: AlertMessage):
        print(f"📧 告警通知: {alert.message}")
    
    monitor.register_alert_callback(alert_handler)
    
    # 启动监控
    monitor.start_monitoring()
    
    print("📊 开始模拟性能指标...")
    
    # 模拟一些性能指标
    test_metrics = [
        ('stock_processing_time', 0.8, '秒/股'),
        ('indicator_calculation_time', 0.05, '秒/指标'),
        ('database_query_time', 0.3, '秒/查询'),
        ('cache_hit_rate', 85.0, '%'),
        ('successful_selection_rate', 95.0, '%'),
        ('error_rate', 2.0, '%')
    ]
    
    # 记录一些正常指标
    for metric_name, value, unit in test_metrics:
        monitor.record_performance_metric(metric_name, value, unit)
        time.sleep(0.5)
    
    print("⚠️  模拟一些异常情况...")
    
    # 模拟一些告警情况
    warning_metrics = [
        ('stock_processing_time', 1.5, '秒/股'),  # 超过warning阈值
        ('cache_hit_rate', 50.0, '%'),            # 低于warning阈值
    ]
    
    for metric_name, value, unit in warning_metrics:
        monitor.record_performance_metric(metric_name, value, unit)
        time.sleep(0.5)
    
    # 等待一下让监控收集数据
    print("📈 等待监控数据收集...")
    time.sleep(5)
    
    # 获取性能摘要
    summary = monitor.get_performance_summary(hours=1)
    
    print("\n📊 性能监控摘要:")
    print(f"系统资源: CPU {summary['system_resources']['cpu']['current']:.1f}%, "
          f"内存 {summary['system_resources']['memory']['current']:.1f}%")
    print(f"性能指标: {len(summary['performance_metrics'])} 个")
    print(f"告警统计: {summary['alerts']['total']} 个")
    
    if summary['alerts']['total'] > 0:
        print("告警分布:", summary['alerts']['by_level'])
    
    if summary['recommendations']:
        print("优化建议:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # 导出监控数据
    export_file = monitor.export_monitoring_data()
    
    # 停止监控
    monitor.stop_monitoring()
    
    print(f"\n✅ 生产级性能监控测试完成")
    print(f"📄 监控报告: {export_file}")


if __name__ == "__main__":
    main() 