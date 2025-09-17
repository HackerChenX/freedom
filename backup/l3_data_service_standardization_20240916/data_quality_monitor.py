"""
数据质量监控和报告系统
提供实时数据质量监控、报告生成和质量趋势分析
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging

from db.unified_data_quality_manager import (
    UnifiedDataQualityManager, 
    DataQualityReport, 
    DataQualityLevel,
    get_data_quality_manager
)
from utils.enhanced_performance_monitor import performance_monitor
from utils.enhanced_exception_handler import exception_handler, ErrorSeverity, ErrorCategory
from config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """质量告警"""
    alert_id: str
    severity: str
    message: str
    dataset_id: str
    quality_score: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class DataQualityMonitor:
    """
    数据质量监控器
    
    功能：
    - 实时质量监控
    - 质量告警
    - 趋势分析
    - 报告生成
    """
    
    def __init__(self, 
                 alert_threshold: float = 70.0,
                 critical_threshold: float = 50.0,
                 monitoring_interval: int = 300,
                 enable_real_time: bool = True):
        """
        初始化数据质量监控器
        
        Args:
            alert_threshold: 告警阈值
            critical_threshold: 严重告警阈值
            monitoring_interval: 监控间隔（秒）
            enable_real_time: 是否启用实时监控
        """
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_interval = monitoring_interval
        self.enable_real_time = enable_real_time
        
        self.quality_manager = get_data_quality_manager()
        
        # 监控数据
        self.quality_metrics = []
        self.alerts = []
        self.monitoring_stats = {
            'total_checks': 0,
            'alerts_generated': 0,
            'critical_alerts': 0,
            'avg_quality_score': 0.0,
            'last_check_time': None
        }
        
        # 线程控制
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        # 报告配置
        self.report_config = {
            'output_dir': get_config('data_quality.report_dir', 'reports/data_quality'),
            'auto_generate': get_config('data_quality.auto_generate_reports', True),
            'report_frequency': get_config('data_quality.report_frequency', 'daily')
        }
        
        # 确保报告目录存在
        Path(self.report_config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据质量监控器初始化完成 - 告警阈值: {alert_threshold}, 监控间隔: {monitoring_interval}秒")
    
    def start_monitoring(self):
        """启动实时监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("数据质量实时监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("数据质量实时监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._perform_quality_check()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续
    
    @performance_monitor(threshold_seconds=10.0)
    def _perform_quality_check(self):
        """执行质量检查"""
        # 这里可以集成实际的数据源检查
        # 目前作为示例，生成模拟数据
        
        current_time = datetime.now()
        
        # 模拟质量检查结果
        mock_quality_score = 85.0 + (time.time() % 30 - 15) * 0.5  # 模拟波动
        
        quality_metric = {
            'timestamp': current_time,
            'quality_score': mock_quality_score,
            'dataset_count': 10,
            'issues_found': max(0, int((100 - mock_quality_score) / 10)),
            'processing_time': 2.5
        }
        
        with self.lock:
            self.quality_metrics.append(quality_metric)
            
            # 保留最近1000条记录
            if len(self.quality_metrics) > 1000:
                self.quality_metrics = self.quality_metrics[-1000:]
            
            # 更新统计
            self.monitoring_stats['total_checks'] += 1
            self.monitoring_stats['last_check_time'] = current_time
            
            # 计算平均质量分数
            recent_scores = [m['quality_score'] for m in self.quality_metrics[-10:]]
            self.monitoring_stats['avg_quality_score'] = sum(recent_scores) / len(recent_scores)
            
            # 检查是否需要告警
            self._check_alerts(quality_metric)
        
        logger.debug(f"质量检查完成 - 分数: {mock_quality_score:.2f}")
    
    def _check_alerts(self, quality_metric: Dict[str, Any]):
        """检查告警条件"""
        quality_score = quality_metric['quality_score']
        
        if quality_score < self.critical_threshold:
            self._generate_alert(
                severity='critical',
                message=f"数据质量严重下降: {quality_score:.2f}% (阈值: {self.critical_threshold}%)",
                quality_score=quality_score
            )
        elif quality_score < self.alert_threshold:
            self._generate_alert(
                severity='warning',
                message=f"数据质量低于阈值: {quality_score:.2f}% (阈值: {self.alert_threshold}%)",
                quality_score=quality_score
            )
    
    def _generate_alert(self, severity: str, message: str, quality_score: float, dataset_id: str = 'system'):
        """生成告警"""
        alert = QualityAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            severity=severity,
            message=message,
            dataset_id=dataset_id,
            quality_score=quality_score,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.monitoring_stats['alerts_generated'] += 1
        
        if severity == 'critical':
            self.monitoring_stats['critical_alerts'] += 1
        
        logger.warning(f"质量告警 [{severity.upper()}]: {message}")
        
        # 这里可以集成告警通知系统（邮件、短信、钉钉等）
        self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: QualityAlert):
        """发送告警通知"""
        # 示例：记录到文件
        try:
            alert_file = Path(self.report_config['output_dir']) / 'alerts.log'
            with open(alert_file, 'a', encoding='utf-8') as f:
                f.write(f"{alert.timestamp.isoformat()} [{alert.severity.upper()}] {alert.message}\n")
        except Exception as e:
            logger.error(f"记录告警失败: {e}")
    
    @performance_monitor(threshold_seconds=5.0)
    def check_dataset_quality(self, data: pd.DataFrame, dataset_id: str) -> DataQualityReport:
        """
        检查特定数据集质量
        
        Args:
            data: 数据集
            dataset_id: 数据集标识
            
        Returns:
            DataQualityReport: 质量报告
        """
        try:
            # 使用质量管理器进行检查
            quality_result = self.quality_manager.ensure_data_quality(data, dataset_id)
            quality_report = quality_result['quality_report']
            
            # 记录质量指标
            quality_metric = {
                'timestamp': datetime.now(),
                'dataset_id': dataset_id,
                'quality_score': quality_report.quality_score,
                'quality_level': quality_report.quality_level.value,
                'total_records': quality_report.total_records,
                'valid_records': quality_report.valid_records,
                'issues_count': len(quality_report.issues),
                'processing_time': quality_report.processing_time
            }
            
            with self.lock:
                self.quality_metrics.append(quality_metric)
                
                # 检查告警
                if quality_report.quality_score < self.alert_threshold:
                    self._generate_alert(
                        severity='critical' if quality_report.quality_score < self.critical_threshold else 'warning',
                        message=f"数据集 {dataset_id} 质量问题: {quality_report.quality_score:.2f}%",
                        quality_score=quality_report.quality_score,
                        dataset_id=dataset_id
                    )
            
            return quality_report
            
        except Exception as e:
            logger.error(f"数据集质量检查失败 {dataset_id}: {e}")
            raise
    
    def get_quality_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取质量趋势分析
        
        Args:
            hours: 分析时间范围（小时）
            
        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_metrics = [
                m for m in self.quality_metrics 
                if m['timestamp'] > cutoff_time
            ]
        
        if not recent_metrics:
            return {'message': f'最近{hours}小时内无质量数据'}
        
        # 计算趋势
        scores = [m['quality_score'] for m in recent_metrics]
        timestamps = [m['timestamp'] for m in recent_metrics]
        
        trend_analysis = {
            'time_range': f'{hours}小时',
            'data_points': len(recent_metrics),
            'avg_quality_score': sum(scores) / len(scores),
            'min_quality_score': min(scores),
            'max_quality_score': max(scores),
            'quality_variance': np.var(scores) if len(scores) > 1 else 0,
            'trend_direction': self._calculate_trend_direction(scores),
            'quality_distribution': self._calculate_quality_distribution(recent_metrics),
            'alert_count': len([a for a in self.alerts if a.timestamp > cutoff_time])
        }
        
        return trend_analysis
    
    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """计算趋势方向"""
        if len(scores) < 2:
            return 'stable'
        
        # 简单线性趋势
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        diff = second_avg - first_avg
        
        if diff > 2:
            return 'improving'
        elif diff < -2:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_quality_distribution(self, metrics: List[Dict]) -> Dict[str, int]:
        """计算质量分布"""
        distribution = {
            'excellent': 0,  # 95-100
            'good': 0,       # 85-94
            'fair': 0,       # 70-84
            'poor': 0,       # 50-69
            'critical': 0    # 0-49
        }
        
        for metric in metrics:
            score = metric['quality_score']
            if score >= 95:
                distribution['excellent'] += 1
            elif score >= 85:
                distribution['good'] += 1
            elif score >= 70:
                distribution['fair'] += 1
            elif score >= 50:
                distribution['poor'] += 1
            else:
                distribution['critical'] += 1
        
        return distribution
    
    @performance_monitor(threshold_seconds=10.0)
    def generate_quality_report(self, report_type: str = 'daily', output_format: str = 'json') -> str:
        """
        生成质量报告
        
        Args:
            report_type: 报告类型 (daily, weekly, monthly)
            output_format: 输出格式 (json, html, csv)
            
        Returns:
            str: 报告文件路径
        """
        # 确定时间范围
        time_ranges = {
            'daily': 24,
            'weekly': 168,
            'monthly': 720
        }
        
        hours = time_ranges.get(report_type, 24)
        
        # 获取数据
        trends = self.get_quality_trends(hours)
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=hours)]
        
        # 生成报告内容
        report_data = {
            'report_info': {
                'type': report_type,
                'generated_at': datetime.now().isoformat(),
                'time_range_hours': hours,
                'monitoring_stats': self.monitoring_stats
            },
            'quality_trends': trends,
            'alerts': [asdict(alert) for alert in recent_alerts],
            'summary': {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
                'avg_quality_score': trends.get('avg_quality_score', 0),
                'trend_direction': trends.get('trend_direction', 'unknown')
            }
        }
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quality_report_{report_type}_{timestamp}.{output_format}"
        filepath = Path(self.report_config['output_dir']) / filename
        
        try:
            if output_format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            elif output_format == 'csv':
                # 转换为CSV格式
                df = pd.DataFrame(self.quality_metrics)
                df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"质量报告已生成: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            raise
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        with self.lock:
            return {
                'monitoring_active': self.monitoring_active,
                'monitoring_stats': self.monitoring_stats.copy(),
                'alert_summary': {
                    'total_alerts': len(self.alerts),
                    'unresolved_alerts': len([a for a in self.alerts if not a.resolved]),
                    'recent_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)])
                },
                'configuration': {
                    'alert_threshold': self.alert_threshold,
                    'critical_threshold': self.critical_threshold,
                    'monitoring_interval': self.monitoring_interval
                }
            }


# 全局监控器实例
_quality_monitor = None
_monitor_lock = threading.Lock()


def get_data_quality_monitor() -> DataQualityMonitor:
    """获取全局数据质量监控器实例"""
    global _quality_monitor
    
    if _quality_monitor is None:
        with _monitor_lock:
            if _quality_monitor is None:
                _quality_monitor = DataQualityMonitor()
    
    return _quality_monitor


# 导出主要类和函数
__all__ = [
    'DataQualityMonitor',
    'QualityAlert',
    'get_data_quality_monitor'
]
