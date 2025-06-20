"""
系统健康监控器

监控选股系统的运行状态、性能指标和数据质量
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import json

from utils.logger import get_logger

logger = get_logger(__name__)


class SystemHealthMonitor:
    """系统健康监控器"""
    
    def __init__(self):
        """初始化监控器"""
        self.metrics = {
            'analysis_time': [],
            'memory_usage': [],
            'error_count': 0,
            'success_count': 0,
            'match_rates': [],
            'last_update': None
        }
        
        self.thresholds = {
            'analysis_time': 300,  # 5分钟
            'memory_usage': 0.8,   # 80%
            'error_rate': 0.05,    # 5%
            'match_rate': 0.4      # 40%
        }
        
        self.alerts = []
        
    def monitor_analysis_performance(self, analysis_func: Callable) -> Callable:
        """监控分析性能装饰器"""
        @wraps(analysis_func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_percent()
            
            try:
                result = analysis_func(*args, **kwargs)
                
                # 记录成功指标
                self._record_success_metrics(start_time, start_memory, result)
                
                return result
                
            except Exception as e:
                # 记录错误指标
                self._record_error_metrics(start_time, start_memory, e)
                raise
                
        return wrapper
    
    def _record_success_metrics(self, start_time: float, start_memory: float, result: Any):
        """记录成功执行的指标"""
        end_time = time.time()
        end_memory = psutil.Process().memory_percent()
        
        analysis_time = end_time - start_time
        memory_usage = max(start_memory, end_memory) / 100.0
        
        # 更新指标
        self.metrics['analysis_time'].append(analysis_time)
        self.metrics['memory_usage'].append(memory_usage)
        self.metrics['success_count'] += 1
        self.metrics['last_update'] = datetime.now()
        
        # P2级任务：优化内存使用 - 保持最近50次记录以减少内存占用
        max_records = 50
        if len(self.metrics['analysis_time']) > max_records:
            self.metrics['analysis_time'] = self.metrics['analysis_time'][-max_records:]
            self.metrics['memory_usage'] = self.metrics['memory_usage'][-max_records:]
        
        # 检查阈值
        self._check_thresholds(analysis_time, memory_usage)
        
        # 如果结果包含匹配率，记录它
        if isinstance(result, dict) and 'match_analysis' in result:
            match_rate = result['match_analysis'].get('match_rate', 0)
            self.metrics['match_rates'].append(match_rate)
            # P2级任务：优化内存使用
            if len(self.metrics['match_rates']) > max_records:
                self.metrics['match_rates'] = self.metrics['match_rates'][-max_records:]
    
    def _record_error_metrics(self, start_time: float, start_memory: float, error: Exception):
        """记录错误指标"""
        end_time = time.time()
        analysis_time = end_time - start_time
        
        self.metrics['error_count'] += 1
        self.metrics['last_update'] = datetime.now()
        
        # 记录错误告警
        self.alerts.append({
            'type': 'error',
            'timestamp': datetime.now(),
            'message': f"分析执行失败: {str(error)}",
            'analysis_time': analysis_time,
            'severity': 'high'
        })
        
        # P2级任务：优化告警存储 - 保持最近20个告警以减少内存占用
        max_alerts = 20
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]
    
    def _check_thresholds(self, analysis_time: float, memory_usage: float):
        """检查性能阈值"""
        current_time = datetime.now()
        
        # 检查分析时间
        if analysis_time > self.thresholds['analysis_time']:
            self.alerts.append({
                'type': 'performance',
                'timestamp': current_time,
                'message': f"分析时间超过阈值: {analysis_time:.2f}秒 > {self.thresholds['analysis_time']}秒",
                'value': analysis_time,
                'threshold': self.thresholds['analysis_time'],
                'severity': 'medium'
            })
        
        # 检查内存使用
        if memory_usage > self.thresholds['memory_usage']:
            self.alerts.append({
                'type': 'resource',
                'timestamp': current_time,
                'message': f"内存使用超过阈值: {memory_usage:.1%} > {self.thresholds['memory_usage']:.1%}",
                'value': memory_usage,
                'threshold': self.thresholds['memory_usage'],
                'severity': 'high'
            })
        
        # 检查错误率
        total_operations = self.metrics['success_count'] + self.metrics['error_count']
        if total_operations > 10:  # 至少10次操作后才检查错误率
            error_rate = self.metrics['error_count'] / total_operations
            if error_rate > self.thresholds['error_rate']:
                self.alerts.append({
                    'type': 'reliability',
                    'timestamp': current_time,
                    'message': f"错误率超过阈值: {error_rate:.1%} > {self.thresholds['error_rate']:.1%}",
                    'value': error_rate,
                    'threshold': self.thresholds['error_rate'],
                    'severity': 'high'
                })
        
        # 检查匹配率
        if self.metrics['match_rates']:
            avg_match_rate = np.mean(self.metrics['match_rates'][-10:])  # 最近10次的平均值
            if avg_match_rate < self.thresholds['match_rate']:
                self.alerts.append({
                    'type': 'quality',
                    'timestamp': current_time,
                    'message': f"平均匹配率低于阈值: {avg_match_rate:.1%} < {self.thresholds['match_rate']:.1%}",
                    'value': avg_match_rate,
                    'threshold': self.thresholds['match_rate'],
                    'severity': 'medium'
                })
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        current_time = datetime.now()
        
        # 计算统计指标
        stats = self._calculate_statistics()
        
        # 评估整体健康状态
        overall_status = self._assess_overall_health(stats)
        
        # 获取最近的告警
        recent_alerts = [alert for alert in self.alerts 
                        if (current_time - alert['timestamp']).total_seconds() < 3600]  # 最近1小时
        
        return {
            'timestamp': current_time,
            'overall_status': overall_status,
            'statistics': stats,
            'recent_alerts': recent_alerts,
            'alert_count': len(recent_alerts),
            'thresholds': self.thresholds,
            'uptime': self._calculate_uptime()
        }
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算统计指标"""
        stats = {
            'total_operations': self.metrics['success_count'] + self.metrics['error_count'],
            'success_count': self.metrics['success_count'],
            'error_count': self.metrics['error_count'],
            'error_rate': 0.0,
            'avg_analysis_time': 0.0,
            'max_analysis_time': 0.0,
            'avg_memory_usage': 0.0,
            'max_memory_usage': 0.0,
            'avg_match_rate': 0.0,
            'min_match_rate': 0.0
        }
        
        # 计算错误率
        if stats['total_operations'] > 0:
            stats['error_rate'] = stats['error_count'] / stats['total_operations']
        
        # 计算分析时间统计
        if self.metrics['analysis_time']:
            stats['avg_analysis_time'] = np.mean(self.metrics['analysis_time'])
            stats['max_analysis_time'] = np.max(self.metrics['analysis_time'])
        
        # 计算内存使用统计
        if self.metrics['memory_usage']:
            stats['avg_memory_usage'] = np.mean(self.metrics['memory_usage'])
            stats['max_memory_usage'] = np.max(self.metrics['memory_usage'])
        
        # 计算匹配率统计
        if self.metrics['match_rates']:
            stats['avg_match_rate'] = np.mean(self.metrics['match_rates'])
            stats['min_match_rate'] = np.min(self.metrics['match_rates'])
        
        return stats
    
    def _assess_overall_health(self, stats: Dict[str, Any]) -> str:
        """评估整体健康状态"""
        issues = []
        
        # 检查错误率
        if stats['error_rate'] > self.thresholds['error_rate']:
            issues.append('high_error_rate')
        
        # 检查性能
        if stats['avg_analysis_time'] > self.thresholds['analysis_time']:
            issues.append('slow_performance')
        
        # 检查资源使用
        if stats['avg_memory_usage'] > self.thresholds['memory_usage']:
            issues.append('high_memory_usage')
        
        # 检查质量
        if stats['avg_match_rate'] < self.thresholds['match_rate']:
            issues.append('low_match_rate')
        
        # 评估状态
        if not issues:
            return 'healthy'
        elif len(issues) == 1:
            return 'warning'
        else:
            return 'critical'
    
    def _calculate_uptime(self) -> Dict[str, Any]:
        """计算系统运行时间"""
        if self.metrics['last_update']:
            # 简化的运行时间计算
            uptime_seconds = (datetime.now() - self.metrics['last_update']).total_seconds()
            return {
                'last_activity': self.metrics['last_update'],
                'seconds_since_last_activity': uptime_seconds,
                'status': 'active' if uptime_seconds < 300 else 'idle'  # 5分钟内有活动算作活跃
            }
        else:
            return {
                'last_activity': None,
                'seconds_since_last_activity': None,
                'status': 'unknown'
            }
    
    def generate_health_report(self, output_file: str) -> None:
        """生成健康状态报告"""
        try:
            health_data = self.get_system_health()
            
            # 生成Markdown格式报告
            report_content = self._format_health_report(health_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"健康状态报告已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"生成健康状态报告失败: {e}")
    
    def _format_health_report(self, health_data: Dict[str, Any]) -> str:
        """格式化健康状态报告"""
        stats = health_data['statistics']
        
        report = f"""# 系统健康状态报告

## 概览
- **报告时间**: {health_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
- **整体状态**: {health_data['overall_status'].upper()}
- **最近告警数量**: {health_data['alert_count']}

## 性能统计
- **总操作次数**: {stats['total_operations']}
- **成功次数**: {stats['success_count']}
- **错误次数**: {stats['error_count']}
- **错误率**: {stats['error_rate']:.2%}
- **平均分析时间**: {stats['avg_analysis_time']:.2f}秒
- **最大分析时间**: {stats['max_analysis_time']:.2f}秒
- **平均内存使用**: {stats['avg_memory_usage']:.1%}
- **最大内存使用**: {stats['max_memory_usage']:.1%}

## 质量指标
- **平均匹配率**: {stats['avg_match_rate']:.2%}
- **最低匹配率**: {stats['min_match_rate']:.2%}

## 最近告警
"""
        
        recent_alerts = health_data['recent_alerts']
        if recent_alerts:
            for alert in recent_alerts[-10:]:  # 最近10个告警
                report += f"- **{alert['type'].upper()}** ({alert['severity']}): {alert['message']}\n"
        else:
            report += "无最近告警\n"
        
        return report
    
    def reset_metrics(self):
        """重置监控指标"""
        self.metrics = {
            'analysis_time': [],
            'memory_usage': [],
            'error_count': 0,
            'success_count': 0,
            'match_rates': [],
            'last_update': None
        }
        self.alerts = []
        logger.info("监控指标已重置")
