#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
WebSocket实时推送系统综合测试
测试WebSocket服务器、实时数据推送、客户端连接管理等功能
"""

import asyncio
import json
import time
import unittest
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import websockets
from utils.logger import get_logger

logger = get_logger(__name__)

class WebSocketSystemTest(unittest.TestCase):
    """WebSocket系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_results = []
        self.start_time = time.time()
        logger.info("🧪 开始WebSocket系统测试")
    
    def tearDown(self):
        """测试后清理"""
        end_time = time.time()
        duration = end_time - self.start_time
        logger.info(f"⏱️ 测试完成，耗时: {duration:.2f}秒")
    
    def test_websocket_server_functionality(self):
        """测试WebSocket服务器功能"""
        logger.info("🔧 测试WebSocket服务器功能")
        
        async def run_test():
            try:
                # 导入WebSocket服务器
                from api.websocket_server import WebSocketServer, WebSocketConnectionManager
                
                # 测试连接管理器
                manager = WebSocketConnectionManager()
                self.assertIsNotNone(manager)
                self.assertEqual(len(manager.connections), 0)
                
                # 测试服务器初始化
                server = WebSocketServer("localhost", 8765)
                self.assertIsNotNone(server)
                self.assertEqual(server.host, "localhost")
                self.assertEqual(server.port, 8765)
                
                logger.info("✅ WebSocket服务器功能测试通过")
                return True
                
            except Exception as e:
                logger.error(f"❌ WebSocket服务器功能测试失败: {e}")
                return False
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
        self.test_results.append(("WebSocket服务器功能", result))
    
    def test_realtime_data_pusher_functionality(self):
        """测试实时数据推送器功能"""
        logger.info("📡 测试实时数据推送器功能")
        
        async def run_test():
            try:
                # 导入实时数据推送器
                from api.realtime_data_pusher import (
                    StockPricePusher, AlertPusher, MonitoringStatusPusher, 
                    RealtimeDataPusher
                )
                
                # 测试股票价格推送器
                stock_pusher = StockPricePusher()
                self.assertIsNotNone(stock_pusher)
                self.assertFalse(stock_pusher.is_running)
                
                # 测试预警推送器
                alert_pusher = AlertPusher()
                self.assertIsNotNone(alert_pusher)
                self.assertFalse(alert_pusher.is_running)
                
                # 测试监控状态推送器
                monitoring_pusher = MonitoringStatusPusher()
                self.assertIsNotNone(monitoring_pusher)
                self.assertFalse(monitoring_pusher.is_running)
                
                # 测试主控制器
                realtime_pusher = RealtimeDataPusher()
                self.assertIsNotNone(realtime_pusher)
                self.assertFalse(realtime_pusher.is_running)
                
                logger.info("✅ 实时数据推送器功能测试通过")
                return True
                
            except Exception as e:
                logger.error(f"❌ 实时数据推送器功能测试失败: {e}")
                return False
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
        self.test_results.append(("实时数据推送器功能", result))
    
    def test_websocket_message_handling(self):
        """测试WebSocket消息处理"""
        logger.info("💬 测试WebSocket消息处理")
        
        async def run_test():
            try:
                from api.websocket_server import WebSocketConnectionManager, WebSocketMessageHandler
                
                # 创建连接管理器和消息处理器
                manager = WebSocketConnectionManager()
                handler = WebSocketMessageHandler(manager)
                
                self.assertIsNotNone(handler)
                
                # 测试连接统计
                stats = manager.get_connection_stats()
                self.assertIsInstance(stats, dict)
                self.assertIn('total_connections', stats)
                self.assertIn('total_subscriptions', stats)
                self.assertIn('topics', stats)
                
                logger.info("✅ WebSocket消息处理测试通过")
                return True
                
            except Exception as e:
                logger.error(f"❌ WebSocket消息处理测试失败: {e}")
                return False
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
        self.test_results.append(("WebSocket消息处理", result))
    
    def test_data_generation_functionality(self):
        """测试数据生成功能"""
        logger.info("🎲 测试数据生成功能")
        
        async def run_test():
            try:
                from api.realtime_data_pusher import StockPricePusher, AlertPusher, MonitoringStatusPusher
                
                # 测试股票价格数据生成
                stock_pusher = StockPricePusher()
                price_data = await stock_pusher._generate_stock_prices()
                
                self.assertIsInstance(price_data, list)
                self.assertGreater(len(price_data), 0)
                
                for price in price_data:
                    self.assertIn('code', price)
                    self.assertIn('name', price)
                    self.assertIn('price', price)
                    self.assertIn('change_percent', price)
                    self.assertIn('volume', price)
                
                # 测试预警数据生成
                alert_pusher = AlertPusher()
                alerts = await alert_pusher._check_alerts()
                self.assertIsInstance(alerts, list)
                
                # 测试监控状态数据生成
                monitoring_pusher = MonitoringStatusPusher()
                status = await monitoring_pusher._get_monitoring_status()
                
                self.assertIsInstance(status, dict)
                self.assertIn('system_status', status)
                self.assertIn('websocket_connections', status)
                self.assertIn('cpu_usage', status)
                self.assertIn('memory_usage', status)
                
                logger.info("✅ 数据生成功能测试通过")
                return True
                
            except Exception as e:
                logger.error(f"❌ 数据生成功能测试失败: {e}")
                return False
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
        self.test_results.append(("数据生成功能", result))
    
    def test_websocket_integration_with_fastapi(self):
        """测试WebSocket与FastAPI集成"""
        logger.info("🔗 测试WebSocket与FastAPI集成")
        
        try:
            # 检查FastAPI主应用是否包含WebSocket端点
            from api.main import app
            
            # 检查路由中是否包含WebSocket端点
            websocket_routes = [route for route in app.routes if hasattr(route, 'path') and route.path == '/ws']
            self.assertGreater(len(websocket_routes), 0, "FastAPI应用应包含WebSocket端点")
            
            # 检查是否有WebSocket统计端点
            stats_routes = [route for route in app.routes if hasattr(route, 'path') and route.path == '/ws/stats']
            self.assertGreater(len(stats_routes), 0, "FastAPI应用应包含WebSocket统计端点")
            
            logger.info("✅ WebSocket与FastAPI集成测试通过")
            result = True
            
        except Exception as e:
            logger.error(f"❌ WebSocket与FastAPI集成测试失败: {e}")
            result = False
        
        self.assertTrue(result)
        self.test_results.append(("WebSocket与FastAPI集成", result))
    
    def test_websocket_error_handling(self):
        """测试WebSocket错误处理"""
        logger.info("🛡️ 测试WebSocket错误处理")
        
        async def run_test():
            try:
                from api.websocket_server import WebSocketConnectionManager, WebSocketMessageHandler
                
                manager = WebSocketConnectionManager()
                handler = WebSocketMessageHandler(manager)
                
                # 测试无效JSON处理
                # 这里我们只能测试函数存在性，因为需要真实的WebSocket连接才能完全测试
                self.assertTrue(hasattr(handler, 'handle_message'))
                self.assertTrue(hasattr(handler, '_send_error'))
                self.assertTrue(hasattr(handler, '_handle_unknown_message'))
                
                # 测试连接管理器的错误处理方法
                self.assertTrue(hasattr(manager, 'unregister_connection'))
                self.assertTrue(hasattr(manager, 'broadcast_to_topic'))
                
                logger.info("✅ WebSocket错误处理测试通过")
                return True
                
            except Exception as e:
                logger.error(f"❌ WebSocket错误处理测试失败: {e}")
                return False
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
        self.test_results.append(("WebSocket错误处理", result))

def run_websocket_system_tests():
    """运行WebSocket系统测试"""
    print("🚀 开始WebSocket实时推送系统综合测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(WebSocketSystemTest)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 统计结果
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print("\n" + "=" * 60)
    print("📊 WebSocket系统测试结果汇总:")
    print(f"   总测试数: {total_tests}")
    print(f"   通过: {passed}")
    print(f"   失败: {failures}")
    print(f"   错误: {errors}")
    print(f"   成功率: {passed/total_tests*100:.1f}%")
    
    # 详细结果
    if hasattr(result, 'test_results'):
        print("\n📋 详细测试结果:")
        for test_name, test_result in result.test_results:
            status = "✅ 通过" if test_result else "❌ 失败"
            print(f"   {test_name}: {status}")
    
    # 失败详情
    if result.failures:
        print("\n❌ 失败详情:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n💥 错误详情:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    print("\n" + "=" * 60)
    
    # 返回测试是否全部通过
    return passed == total_tests

def run_performance_test():
    """运行性能测试"""
    print("\n🏃 WebSocket性能测试")
    print("-" * 40)
    
    async def performance_test():
        try:
            from api.realtime_data_pusher import StockPricePusher, AlertPusher, MonitoringStatusPusher
            
            # 测试数据生成性能
            stock_pusher = StockPricePusher()
            
            # 测试股票价格生成性能
            start_time = time.time()
            for _ in range(100):
                await stock_pusher._generate_stock_prices()
            stock_time = time.time() - start_time
            
            # 测试预警生成性能
            alert_pusher = AlertPusher()
            start_time = time.time()
            for _ in range(100):
                await alert_pusher._check_alerts()
            alert_time = time.time() - start_time
            
            # 测试监控状态生成性能
            monitoring_pusher = MonitoringStatusPusher()
            start_time = time.time()
            for _ in range(100):
                await monitoring_pusher._get_monitoring_status()
            monitoring_time = time.time() - start_time
            
            print(f"📊 性能测试结果:")
            print(f"   股票价格生成: {stock_time:.3f}秒 (100次)")
            print(f"   预警消息生成: {alert_time:.3f}秒 (100次)")
            print(f"   监控状态生成: {monitoring_time:.3f}秒 (100次)")
            print(f"   平均每次: {(stock_time + alert_time + monitoring_time)/300:.4f}秒")
            
            # 性能要求检查
            avg_time = (stock_time + alert_time + monitoring_time) / 300
            performance_ok = avg_time < 0.01  # 每次生成应小于10ms
            
            print(f"   性能评估: {'✅ 优秀' if performance_ok else '⚠️ 需要优化'}")
            
            return performance_ok
            
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            return False
    
    result = asyncio.run(performance_test())
    return result

def main():
    """主函数"""
    print("🧪 WebSocket实时推送系统测试套件")
    print("=" * 60)
    
    # 运行功能测试
    functional_result = run_websocket_system_tests()
    
    # 运行性能测试
    performance_result = run_performance_test()
    
    # 总体结果
    print("\n🎯 总体测试结果:")
    print(f"   功能测试: {'✅ 通过' if functional_result else '❌ 失败'}")
    print(f"   性能测试: {'✅ 通过' if performance_result else '❌ 失败'}")
    
    overall_success = functional_result and performance_result
    print(f"   整体评估: {'✅ 系统就绪' if overall_success else '❌ 需要修复'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
