#!/usr/bin/env python3
"""
快速策略测试器 - 专注于修复核心策略问题
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# 确保能够导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import get_logger
from utils.dependency_injection import get_container
from db.clickhouse_db import get_clickhouse_db

logger = get_logger(__name__)

class QuickStrategyTester:
    """快速策略测试器"""
    
    def __init__(self):
        self.db = get_clickhouse_db()
        self.container = get_container()
        self._setup_mock_services()
        
    def _setup_mock_services(self):
        """设置模拟服务"""
        try:
            from db.interfaces.indicator_calculator_interface import IIndicatorCalculator
            
            # 创建简单但完整的模拟指标计算器
            class SimpleIndicatorCalculator:
                def calculate_ma(self, data, period=20):
                    """计算移动平均线"""
                    if 'close' in data.columns and len(data) >= period:
                        return data['close'].rolling(window=period).mean()
                    else:
                        return data['close'] * 0  # 返回零向量
                    
                def calculate_macd(self, data):
                    """计算MACD"""
                    return {
                        'macd': data['close'] * 0,
                        'signal': data['close'] * 0,
                        'histogram': data['close'] * 0
                    }
                    
                def calculate_rsi(self, data, period=14):
                    """计算RSI"""
                    return data['close'] * 0.5  # 返回中性值
                    
                def calculate_kdj(self, data):
                    """计算KDJ"""
                    return {
                        'K': data['close'] * 0.5,
                        'D': data['close'] * 0.5,
                        'J': data['close'] * 0.5
                    }
            
            # 使用正确的注册方法
            simple_mock = SimpleIndicatorCalculator()
            self.container.register_singleton(IIndicatorCalculator, factory=lambda: simple_mock)
            logger.info("已注册简单模拟指标计算器")
        except Exception as e:
            logger.error(f"注册模拟服务失败: {e}")
    
    def get_test_stocks(self) -> List[str]:
        """获取测试股票"""
        try:
            query = """
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE level = '日线'
            LIMIT 10
            """
            result = self.db.query(query)
            if not result.empty:
                return result['code'].tolist()
            else:
                return []
        except Exception as e:
            logger.error(f"获取测试股票失败: {e}")
            return []
    
    def test_strategy(self, strategy_class, strategy_name: str):
        """测试单个策略"""
        try:
            logger.info(f"测试策略: {strategy_name}")
            
            # 初始化策略
            strategy = strategy_class()
            logger.info(f"✅ 策略初始化成功: {strategy_name}")
            
            # 获取测试数据
            test_stocks = self.get_test_stocks()
            if not test_stocks:
                logger.error("无法获取测试股票")
                return False
                
            logger.info(f"获取到 {len(test_stocks)} 只测试股票")
            
            # 测试选股功能
            start_date = "2025-06-01"
            end_date = "2025-07-18"
            
            start_time = time.time()
            selected_stocks = strategy.select_stocks(test_stocks, start_date, end_date)
            execution_time = time.time() - start_time
            
            if selected_stocks is not None and not selected_stocks.empty:
                logger.info(f"✅ 策略 {strategy_name} 成功选出 {len(selected_stocks)} 只股票, 耗时 {execution_time:.2f}秒")
                return True
            else:
                logger.warning(f"⚠️ 策略 {strategy_name} 未选出任何股票, 耗时 {execution_time:.2f}秒")
                return False
                
        except Exception as e:
            logger.error(f"❌ 策略 {strategy_name} 测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有策略测试"""
        logger.info("=== 开始快速策略测试 ===")
        
        strategies = [
            ('strategy.dual_ma_strategy', 'DualMAStrategy', '双均线策略'),
            ('strategy.momentum_strategy', 'MomentumStrategy', '动量策略'),
            ('strategy.breakout_strategy', 'BreakoutStrategy', '突破策略'),
            ('strategy.rebound_strategy', 'ReboundStrategy', '反弹策略'),
        ]
        
        success_count = 0
        total_count = 0
        
        for module_name, class_name, display_name in strategies:
            try:
                # 动态导入策略类
                module = __import__(module_name, fromlist=[class_name])
                strategy_class = getattr(module, class_name)
                
                # 测试策略
                if self.test_strategy(strategy_class, display_name):
                    success_count += 1
                total_count += 1
                
            except Exception as e:
                logger.error(f"无法导入策略 {display_name}: {e}")
                total_count += 1
        
        logger.info(f"=== 测试完成: {success_count}/{total_count} 个策略成功 ===")
        
        if success_count > 0:
            logger.info("🎉 有策略成功运行并选出股票！")
        else:
            logger.warning("❌ 没有策略成功选出股票，需要进一步修复")
        
        return success_count, total_count


if __name__ == "__main__":
    tester = QuickStrategyTester()
    success, total = tester.run_all_tests()
    
    if success > 0:
        print(f"\n✅ 快速测试完成: {success}/{total} 个策略成功")
        exit(0)
    else:
        print(f"\n❌ 快速测试完成: {success}/{total} 个策略成功")
        exit(1) 