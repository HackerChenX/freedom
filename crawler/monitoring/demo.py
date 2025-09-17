"""
监控系统演示

展示监控与调度系统的功能
"""

import sys
import os
import time
import random
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crawler.monitoring.performance_monitor import Performance_monitor
from crawler.monitoring.alert_manager import Alert_manager
from crawler.monitoring.data_quality_checker import Data_quality_checker
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


def demo_performance_monitor():
    """演示性能监控"""
    print("\n=== 性能监控演示 ===")

    # 创建性能监控器
    monitor = Performance_monitor(update_interval=5)

    # 启动监控
    monitor.start()

    # 模拟一些请求
    print("模拟爬虫请求...")
    for i in range(10):
        # 模拟请求
        success = random.choice([True, True, True, False])  # 75%成功率
        response_time = random.uniform(1.0, 5.0)
        error_type = None if success else random.choice(['timeout', 'connection_error', 'parse_error'])

        monitor.record_request(success, response_time, error_type)
        print(f"请求 {i+1}: {'成功' if success else '失败'} - {response_time:.2f}秒")
        time.sleep(0.5)

    # 获取监控指标
    metrics = monitor.get_metrics()
    print(f"\n监控指标:")
    print(f"- 总请求数: {metrics['total_requests']}")
    print(f"- 成功率: {metrics['success_rate']:.2f}%")
    print(f"- 错误率: {metrics['error_rate']:.2f}%")
    print(f"- 平均响应时间: {metrics['avg_response_time']:.2f}秒")
    print(f"- 每小时请求数: {metrics['requests_per_hour']:.2f}")

    # 获取健康状态
    health = monitor.get_health_status()
    print(f"\n健康状态:")
    print(f"- 状态: {health['status']}")
    print(f"- 健康分数: {health['health_score']:.1f}")
    if health['issues']:
        print(f"- 问题: {'; '.join(health['issues'])}")

    # 停止监控
    monitor.stop()

    return metrics