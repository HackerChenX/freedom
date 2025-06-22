#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
监控系统测试

验证性能监控和告警系统功能
"""

import sys
import os
import time
import threading
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from monitoring.performance_monitor import (
    get_performance_monitor, 
    get_health_checker,
    start_monitoring,
    stop_monitoring,
    AlertRule
)
from utils.logger import get_logger

logger = get_logger(__name__)


def test_performance_monitoring():
    """测试性能监控"""
    logger.info("开始测试性能监控...")
    
    monitor = get_performance_monitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 等待收集一些数据
    time.sleep(15)
    
    # 获取当前指标
    current_metrics = monitor.get_current_metrics()
    logger.info(f"当前指标数量: {len(current_metrics)}")
    
    for name, metric in current_metrics.items():
        logger.info(f"  {name}: {metric.value}")
    
    # 获取统计信息
    stats = monitor.get_stats()
    logger.info(f"监控统计: {stats}")
    
    # 停止监控
    monitor.stop_monitoring()
    
    return len(current_metrics) > 0


def test_health_checks():
    """测试健康检查"""
    logger.info("开始测试健康检查...")
    
    health_checker = get_health_checker()
    
    # 运行所有健康检查
    results = health_checker.run_all_checks()
    
    logger.info(f"整体健康状态: {results['overall_status']}")
    
    for check_name, result in results['checks'].items():
        status = result.get('status', 'unknown')
        message = result.get('message', 'No message')
        logger.info(f"  {check_name}: {status} - {message}")
    
    return results['overall_status'] in ['healthy', 'warning']


def test_custom_alert_rule():
    """测试自定义告警规则"""
    logger.info("开始测试自定义告警规则...")
    
    monitor = get_performance_monitor()
    
    # 创建一个测试告警规则
    test_alert_triggered = threading.Event()
    
    def alert_callback(alert):
        logger.info(f"测试告警触发: {alert['message']}")
        test_alert_triggered.set()
    
    test_rule = AlertRule(
        name='测试CPU告警',
        metric_name='cpu_usage_percent',
        condition='gt',
        threshold=0.0,  # 设置很低的阈值确保触发
        severity='info',
        duration=5,  # 5秒持续时间
        callback=alert_callback
    )
    
    monitor.add_alert_rule(test_rule)
    
    # 启动监控
    monitor.start_monitoring()
    
    # 等待告警触发
    triggered = test_alert_triggered.wait(timeout=20)
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 移除测试规则
    monitor.remove_alert_rule('测试CPU告警')
    
    if triggered:
        logger.info("自定义告警规则测试成功")
    else:
        logger.warning("自定义告警规则未在预期时间内触发")
    
    return triggered


def test_metrics_export():
    """测试指标导出"""
    logger.info("开始测试指标导出...")
    
    monitor = get_performance_monitor()
    
    # 启动监控收集一些数据
    monitor.start_monitoring()
    time.sleep(10)
    monitor.stop_monitoring()
    
    # 导出指标
    export_file = "test_reports/metrics_export_test.json"
    os.makedirs("test_reports", exist_ok=True)
    
    try:
        monitor.export_metrics(export_file, hours=1)
        
        # 检查文件是否存在
        if os.path.exists(export_file):
            file_size = os.path.getsize(export_file)
            logger.info(f"指标导出成功，文件大小: {file_size} 字节")
            return True
        else:
            logger.error("指标导出文件不存在")
            return False
            
    except Exception as e:
        logger.error(f"指标导出失败: {e}")
        return False


def test_concurrent_monitoring():
    """测试并发监控"""
    logger.info("开始测试并发监控...")
    
    monitor = get_performance_monitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 模拟并发添加指标
    def add_test_metrics(thread_id):
        for i in range(10):
            monitor.add_metric(f'test_metric_{thread_id}', i * thread_id)
            time.sleep(0.1)
    
    # 创建多个线程
    threads = []
    for i in range(3):
        thread = threading.Thread(target=add_test_metrics, args=(i,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 等待一点时间让监控收集数据
    time.sleep(5)
    
    # 检查指标
    current_metrics = monitor.get_current_metrics()
    test_metrics = {k: v for k, v in current_metrics.items() if k.startswith('test_metric_')}
    
    monitor.stop_monitoring()
    
    logger.info(f"并发测试完成，收集到 {len(test_metrics)} 个测试指标")
    
    return len(test_metrics) >= 3


def run_comprehensive_monitoring_test():
    """运行全面的监控测试"""
    logger.info("=" * 80)
    logger.info("开始监控系统全面测试")
    logger.info("=" * 80)
    
    test_results = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {},
        'overall_success': True
    }
    
    # 测试项目列表
    tests = [
        ('性能监控基础功能', test_performance_monitoring),
        ('健康检查系统', test_health_checks),
        ('自定义告警规则', test_custom_alert_rule),
        ('指标数据导出', test_metrics_export),
        ('并发监控处理', test_concurrent_monitoring)
    ]
    
    # 执行所有测试
    for test_name, test_func in tests:
        logger.info(f"\n执行测试: {test_name}")
        
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            test_results['tests'][test_name] = {
                'success': success,
                'duration': duration,
                'status': '通过' if success else '失败'
            }
            
            if not success:
                test_results['overall_success'] = False
            
            logger.info(f"测试 '{test_name}' {'通过' if success else '失败'}，耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"测试 '{test_name}' 执行出错: {e}")
            test_results['tests'][test_name] = {
                'success': False,
                'error': str(e),
                'status': '错误'
            }
            test_results['overall_success'] = False
    
    return test_results


def main():
    """主函数"""
    print("=" * 80)
    print("监控系统功能测试")
    print("验证性能监控和告警系统")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 运行全面测试
        results = run_comprehensive_monitoring_test()
        
        # 显示结果摘要
        print("\n" + "=" * 80)
        print("测试结果摘要")
        print("=" * 80)
        
        total_tests = len(results['tests'])
        passed_tests = sum(1 for test in results['tests'].values() if test['success'])
        
        print(f"📊 总测试数: {total_tests}")
        print(f"✅ 通过测试: {passed_tests}")
        print(f"❌ 失败测试: {total_tests - passed_tests}")
        print(f"🎯 成功率: {(passed_tests / total_tests * 100):.1f}%")
        print()
        
        # 详细结果
        print("📋 详细测试结果:")
        for test_name, result in results['tests'].items():
            status_emoji = "✅" if result['success'] else "❌"
            duration = result.get('duration', 0)
            print(f"  {status_emoji} {test_name}: {result['status']} ({duration:.2f}秒)")
            
            if 'error' in result:
                print(f"    错误: {result['error']}")
        
        print()
        
        # 保存详细结果
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_reports/monitoring_test_{timestamp}.json"
        
        os.makedirs("test_reports", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📄 详细结果已保存: {output_file}")
        
        # 判断测试结果
        if results['overall_success']:
            print("\n🎉 监控系统测试全部通过！")
            return 0
        else:
            print("\n⚠️ 部分监控功能测试失败，请检查相关组件。")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
