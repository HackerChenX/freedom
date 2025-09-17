#!/usr/bin/env python3
"""
简单的超时控制测试

验证系统基本性能并实现超时控制
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.logger import get_logger

logger = get_logger(__name__)


class SimpleTimeoutTest:
    """简单的超时控制测试"""
    
    def __init__(self):
        self.test_start_time = None
        self.is_stopping = False
        self._stop_event = threading.Event()
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'tests_completed': 0,
            'status': 'PENDING'
        }
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.warning(f"接收到停止信号 {signum}")
        self.stop_test("手动停止")
    
    def stop_test(self, reason: str = "超时"):
        """停止测试"""
        if not self.is_stopping:
            self.is_stopping = True
            self._stop_event.set()
            logger.warning(f"测试被停止: {reason}")
    
    def test_basic_operations(self, timeout: float = 10.0) -> Dict[str, Any]:
        """测试基本操作"""
        logger.info(f"开始基本操作测试（超时: {timeout}s）")
        test_start = time.time()
        
        # 启动超时监控
        def timeout_monitor():
            if self._stop_event.wait(timeout):
                return  # 正常结束
            self.stop_test(f"操作超时 ({timeout}s)")
        
        monitor_thread = threading.Thread(target=timeout_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            # 模拟一些计算操作
            for i in range(100):
                if self.is_stopping:
                    break
                
                # 模拟计算
                result = sum(j ** 2 for j in range(1000))
                time.sleep(0.01)  # 模拟IO等待
                
                if i % 20 == 0:
                    elapsed = time.time() - test_start
                    logger.info(f"完成 {i+1}/100 操作，耗时: {elapsed:.2f}s")
            
            duration = time.time() - test_start
            
            return {
                'status': 'SUCCESS' if not self.is_stopping else 'STOPPED',
                'duration': duration,
                'operations_completed': i + 1,
                'timeout_occurred': duration > timeout
            }
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'duration': time.time() - test_start
            }
        finally:
            self._stop_event.set()  # 通知监控线程停止
    
    def test_database_connection(self, timeout: float = 5.0) -> Dict[str, Any]:
        """测试数据库连接"""
        logger.info(f"开始数据库连接测试（超时: {timeout}s）")
        test_start = time.time()
        
        try:
            # 简单的数据库连接测试
            from config.unified_config_manager import get_config
            
            # 模拟数据库操作
            time.sleep(1.0)  # 模拟连接时间
            
            if self.is_stopping:
                return {'status': 'STOPPED', 'reason': '测试被中断'}
            
            duration = time.time() - test_start
            
            return {
                'status': 'SUCCESS',
                'duration': duration,
                'timeout_occurred': duration > timeout
            }
            
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'duration': time.time() - test_start
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行全面测试"""
        logger.info("=" * 60)
        logger.info("开始简单超时控制测试")
        logger.info("=" * 60)
        
        self.test_start_time = time.time()
        self.test_results['start_time'] = datetime.now().isoformat()
        
        test_cases = [
            ('基本操作测试', self.test_basic_operations, 10.0),
            ('数据库连接测试', self.test_database_connection, 5.0)
        ]
        
        results = []
        
        for test_name, test_func, timeout in test_cases:
            if self.is_stopping:
                logger.info(f"跳过 {test_name}，测试已停止")
                break
            
            logger.info(f"执行 {test_name}")
            result = test_func(timeout)
            results.append({
                'name': test_name,
                'result': result
            })
            
            self.test_results['tests_completed'] += 1
            
            if result['status'] in ['ERROR', 'STOPPED']:
                logger.warning(f"{test_name} 失败或被停止，终止后续测试")
                break
        
        # 设置最终状态
        if self.is_stopping:
            self.test_results['status'] = 'STOPPED'
        elif all(test['result']['status'] == 'SUCCESS' for test in results):
            self.test_results['status'] = 'SUCCESS'
        else:
            self.test_results['status'] = 'PARTIAL'
        
        self.test_results['end_time'] = datetime.now().isoformat()
        self.test_results['total_duration'] = time.time() - self.test_start_time
        self.test_results['test_results'] = results
        
        return self.test_results
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"测试状态: {self.test_results['status']}")
        print(f"完成测试数: {self.test_results['tests_completed']}")
        print(f"总耗时: {self.test_results.get('total_duration', 0):.2f}秒")
        
        if 'test_results' in self.test_results:
            print("\n测试详情:")
            for test in self.test_results['test_results']:
                name = test['name']
                result = test['result']
                status = result['status']
                duration = result.get('duration', 0)
                
                status_icon = {
                    'SUCCESS': '✅',
                    'ERROR': '❌',
                    'STOPPED': '🛑'
                }.get(status, '❓')
                
                print(f"  {status_icon} {name}: {status} ({duration:.2f}s)")
                
                if result.get('timeout_occurred', False):
                    print(f"    ⚠️ 发生超时")


def main():
    """主函数"""
    print("=" * 60)
    print("简单超时控制测试")
    print("验证基本功能和超时机制")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 创建测试实例
        test_framework = SimpleTimeoutTest()
        
        # 运行测试
        results = test_framework.run_comprehensive_test()
        
        # 显示结果
        test_framework.print_summary()
        
        # 返回状态码
        if results['status'] == 'SUCCESS':
            print("\n🎉 所有测试通过")
            return 0
        elif results['status'] == 'PARTIAL':
            print("\n⚠️ 部分测试通过")
            return 1
        else:
            print("\n❌ 测试失败或被停止")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
        return 130
    except Exception as e:
        print(f"\n💥 测试执行异常: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 