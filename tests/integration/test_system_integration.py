#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统集成测试
"""

import unittest
import tempfile
import os
from pathlib import Path

from tests.comprehensive.system_manager import initialize_system, shutdown_system
from tests.comprehensive.test_config_manager import get_config_manager
from db.sql_manager import SQLManager, QueryType


class SystemIntegrationTest(unittest.TestCase):
    """系统集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时工作空间
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_dir = self.temp_dir.name
        
        # 创建配置目录
        config_dir = os.path.join(self.workspace_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        # 创建配置文件
        self.config_path = os.path.join(config_dir, "test_config.yaml")
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("""
test_config:
  execution:
    timeout_seconds: 300
    performance_threshold: 0.8
    max_workers: 20
    batch_size: 1000
  
  test_scope:
    date_range:
      start_date: "20240101"
      end_date: "20241231"
    stock_universe: "ALL"
    min_volume: 1000000
    min_price: 1.0
            """)
        
        # 初始化系统
        self.components = initialize_system(
            config_path=self.config_path,
            workspace_dir=self.workspace_dir,
            enable_monitoring=True,
            enable_dashboard=False
        )
    
    def tearDown(self):
        """清理测试环境"""
        # 关闭系统
        shutdown_system()
        
        # 清理临时目录
        self.temp_dir.cleanup()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsNotNone(self.components)
        self.assertIn('config_manager', self.components)
        self.assertIn('config', self.components)
        self.assertIn('monitoring_system', self.components)
        self.assertIn('performance_monitor', self.components)
        self.assertIn('tester', self.components)
        self.assertIn('orchestrator', self.components)
    
    def test_config_loading(self):
        """测试配置加载"""
        config = self.components['config']
        
        self.assertEqual(config.execution.timeout_seconds, 300)
        self.assertEqual(config.execution.max_workers, 20)
        self.assertEqual(config.test_scope.date_range.start_date, "20240101")
        self.assertEqual(config.test_scope.date_range.end_date, "20241231")
        self.assertEqual(config.test_scope.stock_universe, "ALL")
        self.assertEqual(config.test_scope.min_volume, 1000000)
        self.assertEqual(config.test_scope.min_price, 1.0)
    
    def test_monitoring_system(self):
        """测试监控系统"""
        monitoring_system = self.components['monitoring_system']
        
        # 检查监控系统是否运行
        self.assertTrue(monitoring_system.running)
        
        # 获取系统状态
        status = monitoring_system.get_system_status()
        self.assertIsNotNone(status)
        self.assertTrue(status['monitoring_active'])
    
    def test_performance_monitor(self):
        """测试性能监控"""
        performance_monitor = self.components['performance_monitor']
        
        # 启动监控
        performance_monitor.start_monitoring(total_tasks=1)
        
        # 检查是否正在运行
        self.assertTrue(performance_monitor.is_monitoring())
        
        # 停止监控
        performance_monitor.stop_monitoring()
        
        # 检查是否已停止
        self.assertFalse(performance_monitor.is_monitoring())
    
    def test_config_update(self):
        """测试配置更新"""
        config_manager = self.components['config_manager']
        
        # 更新配置
        config_updates = {
            'execution': {
                'timeout_seconds': 600,
                'max_workers': 30
            },
            'test_scope': {
                'date_range': {
                    'start_date': '20240201',
                    'end_date': '20240331'
                }
            }
        }
        
        config_manager.update_config(config_updates)
        config = config_manager.get_config()
        
        # 检查配置是否已更新
        self.assertEqual(config.execution.timeout_seconds, 600)
        self.assertEqual(config.execution.max_workers, 30)
        self.assertEqual(config.test_scope.date_range.start_date, "20240201")
        self.assertEqual(config.test_scope.date_range.end_date, "20240331")


if __name__ == '__main__':
    unittest.main()