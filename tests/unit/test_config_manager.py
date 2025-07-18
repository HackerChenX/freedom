#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理器单元测试
"""

import os
import unittest
import tempfile
from pathlib import Path

from tests.comprehensive.test_config_manager import TestConfigManager, TestConfig


class TestConfigManagerTest(unittest.TestCase):
    """配置管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时配置文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        
        # 创建测试配置
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
    
    def tearDown(self):
        """清理测试环境"""
        self.temp_dir.cleanup()
    
    def test_load_config(self):
        """测试加载配置"""
        config_manager = TestConfigManager(self.config_path)
        config = config_manager.get_config()
        
        self.assertIsNotNone(config)
        self.assertEqual(config.execution.timeout_seconds, 300)
        self.assertEqual(config.execution.max_workers, 20)
        self.assertEqual(config.test_scope.date_range.start_date, "20240101")
        self.assertEqual(config.test_scope.date_range.end_date, "20241231")
        self.assertEqual(config.test_scope.stock_universe, "ALL")
        self.assertEqual(config.test_scope.min_volume, 1000000)
        self.assertEqual(config.test_scope.min_price, 1.0)
    
    def test_validate_config(self):
        """测试验证配置"""
        config_manager = TestConfigManager(self.config_path)
        errors = config_manager.validate_config()
        
        self.assertEqual(len(errors), 0)
    
    def test_validate_invalid_config(self):
        """测试验证无效配置"""
        # 创建无效配置
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("""
test_config:
  execution:
    timeout_seconds: -1  # 无效的超时时间
    max_workers: 0       # 无效的工作线程数
  
  test_scope:
    date_range:
      start_date: "20241231"  # 开始日期晚于结束日期
      end_date: "20240101"
            """)
        
        config_manager = TestConfigManager(self.config_path)
        errors = config_manager.validate_config()
        
        self.assertGreater(len(errors), 0)
    
    def test_update_config(self):
        """测试更新配置"""
        config_manager = TestConfigManager(self.config_path)
        
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
        
        self.assertEqual(config.execution.timeout_seconds, 600)
        self.assertEqual(config.execution.max_workers, 30)
        self.assertEqual(config.test_scope.date_range.start_date, "20240201")
        self.assertEqual(config.test_scope.date_range.end_date, "20240331")
    
    def test_save_config(self):
        """测试保存配置"""
        config_manager = TestConfigManager(self.config_path)
        
        # 更新配置
        config_updates = {
            'execution': {
                'timeout_seconds': 600,
                'max_workers': 30
            }
        }
        
        config_manager.update_config(config_updates)
        
        # 保存到新文件
        new_config_path = os.path.join(self.temp_dir.name, "new_config.yaml")
        config_manager.save_config(new_config_path)
        
        # 加载新配置
        new_config_manager = TestConfigManager(new_config_path)
        new_config = new_config_manager.get_config()
        
        self.assertEqual(new_config.execution.timeout_seconds, 600)
        self.assertEqual(new_config.execution.max_workers, 30)


if __name__ == '__main__':
    unittest.main()