#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
WebSocket客户端测试工具
用于测试WebSocket实时推送功能
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, Any
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketTestClient:
    """WebSocket测试客户端"""
    
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        """初始化WebSocket测试客户端"""
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        logger.info(f"WebSocket测试客户端初始化: {uri}")
    
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            logger.info(f"✅ 成功连接到WebSocket服务器: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"❌ 连接WebSocket服务器失败: {e}")
            return False
    
    async def disconnect(self):
        """断开WebSocket连接"""
        if self.websocket and self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            logger.info("✅ WebSocket连接已断开")
    
    async def send_message(self, message: Dict[str, Any]):
        """发送消息到WebSocket服务器"""
        if not self.is_connected or not self.websocket:
            logger.error("❌ WebSocket未连接")
            return False
        
        try:
            message_str = json.dumps(message, ensure_ascii=False)
            await self.websocket.send(message_str)
            logger.info(f"📤 发送消息: {message}")
            return True
        except Exception as e:
            logger.error(f"❌ 发送消息失败: {e}")
            return False
    
    async def receive_messages(self, duration: int = 30):
        """接收WebSocket消息"""
        if not self.is_connected or not self.websocket:
            logger.error("❌ WebSocket未连接")
            return
        
        logger.info(f"📥 开始接收消息，持续时间: {duration}秒")
        start_time = asyncio.get_event_loop().time()
        message_count = 0
        
        try:
            while True:
                # 检查是否超时
                if asyncio.get_event_loop().time() - start_time > duration:
                    logger.info(f"⏰ 接收消息超时，共接收 {message_count} 条消息")
                    break
                
                try:
                    # 设置接收超时
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    data = json.loads(message)
                    message_count += 1
                    
                    # 打印接收到的消息
                    self._print_received_message(data, message_count)
                    
                except asyncio.TimeoutError:
                    # 接收超时，继续循环
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️ WebSocket连接已关闭")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON解析失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ 接收消息时出错: {e}")
        
        logger.info(f"📊 消息接收完成，总计: {message_count} 条")
    
    def _print_received_message(self, data: Dict[str, Any], count: int):
        """打印接收到的消息"""
        msg_type = data.get('type', 'unknown')
        timestamp = data.get('timestamp', '')
        
        print(f"\n📨 消息 #{count} [{msg_type}] - {timestamp}")
        
        if msg_type == 'welcome':
            print(f"   欢迎消息: {data.get('message', '')}")
            print(f"   可用主题: {data.get('available_topics', [])}")
        
        elif msg_type == 'stock_prices':
            prices = data.get('data', [])
            print(f"   股票价格数据: {len(prices)} 只股票")
            for price in prices[:3]:  # 只显示前3只股票
                print(f"     {price.get('code')} {price.get('name')}: "
                      f"¥{price.get('price')} ({price.get('change_percent', 0):+.2f}%)")
        
        elif msg_type == 'alerts':
            alerts = data.get('data', [])
            print(f"   预警消息: {len(alerts)} 条")
            for alert in alerts:
                print(f"     [{alert.get('level')}] {alert.get('title')}: {alert.get('message')}")
        
        elif msg_type == 'monitoring_status':
            status = data.get('data', {})
            print(f"   系统状态: {status.get('system_status')}")
            print(f"   WebSocket连接: {status.get('websocket_connections')}")
            print(f"   CPU使用率: {status.get('cpu_usage')}%")
            print(f"   内存使用率: {status.get('memory_usage')}%")
        
        elif msg_type == 'subscription_confirmed':
            print(f"   订阅确认: {data.get('topic')}")
        
        elif msg_type == 'error':
            print(f"   错误消息: {data.get('message')}")
        
        else:
            print(f"   数据: {data}")
    
    async def subscribe_to_topic(self, topic: str):
        """订阅主题"""
        message = {
            'type': 'subscribe',
            'topic': topic
        }
        success = await self.send_message(message)
        if success:
            logger.info(f"📡 订阅主题: {topic}")
        return success
    
    async def unsubscribe_from_topic(self, topic: str):
        """取消订阅主题"""
        message = {
            'type': 'unsubscribe',
            'topic': topic
        }
        success = await self.send_message(message)
        if success:
            logger.info(f"📡 取消订阅主题: {topic}")
        return success
    
    async def send_ping(self):
        """发送心跳"""
        message = {
            'type': 'ping'
        }
        success = await self.send_message(message)
        if success:
            logger.info("💓 发送心跳")
        return success
    
    async def get_stats(self):
        """获取统计信息"""
        message = {
            'type': 'get_stats'
        }
        success = await self.send_message(message)
        if success:
            logger.info("📊 请求统计信息")
        return success

async def test_basic_connection():
    """测试基本连接功能"""
    print("\n🧪 测试1: 基本连接功能")
    client = WebSocketTestClient()
    
    # 连接
    if not await client.connect():
        return False
    
    # 接收欢迎消息
    await client.receive_messages(duration=3)
    
    # 断开连接
    await client.disconnect()
    return True

async def test_subscription_functionality():
    """测试订阅功能"""
    print("\n🧪 测试2: 订阅功能")
    client = WebSocketTestClient()
    
    if not await client.connect():
        return False
    
    # 订阅股票价格
    await client.subscribe_to_topic('stock_prices')
    await asyncio.sleep(1)
    
    # 订阅预警消息
    await client.subscribe_to_topic('alerts')
    await asyncio.sleep(1)
    
    # 订阅监控状态
    await client.subscribe_to_topic('monitoring')
    await asyncio.sleep(1)
    
    # 接收消息
    await client.receive_messages(duration=15)
    
    # 取消订阅
    await client.unsubscribe_from_topic('stock_prices')
    await asyncio.sleep(1)
    
    await client.disconnect()
    return True

async def test_heartbeat_functionality():
    """测试心跳功能"""
    print("\n🧪 测试3: 心跳功能")
    client = WebSocketTestClient()
    
    if not await client.connect():
        return False
    
    # 发送心跳
    for i in range(3):
        await client.send_ping()
        await asyncio.sleep(2)
    
    # 接收响应
    await client.receive_messages(duration=5)
    
    await client.disconnect()
    return True

async def test_stats_functionality():
    """测试统计信息功能"""
    print("\n🧪 测试4: 统计信息功能")
    client = WebSocketTestClient()
    
    if not await client.connect():
        return False
    
    # 获取统计信息
    await client.get_stats()
    
    # 接收响应
    await client.receive_messages(duration=3)
    
    await client.disconnect()
    return True

async def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始WebSocket综合测试")
    print("=" * 50)
    
    tests = [
        ("基本连接功能", test_basic_connection),
        ("订阅功能", test_subscription_functionality),
        ("心跳功能", test_heartbeat_functionality),
        ("统计信息功能", test_stats_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: 失败 - {e}")
        
        # 测试间隔
        await asyncio.sleep(2)
    
    # 测试结果汇总
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    return passed == total

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='WebSocket客户端测试工具')
    parser.add_argument('--uri', default='ws://localhost:8000/ws', help='WebSocket服务器URI')
    parser.add_argument('--test', choices=['basic', 'subscription', 'heartbeat', 'stats', 'all'], 
                       default='all', help='要运行的测试类型')
    
    args = parser.parse_args()
    
    # 设置WebSocket URI
    WebSocketTestClient.__init__ = lambda self, uri=args.uri: setattr(self, 'uri', uri) or setattr(self, 'websocket', None) or setattr(self, 'is_connected', False) or logger.info(f"WebSocket测试客户端初始化: {uri}")
    
    # 运行测试
    if args.test == 'all':
        asyncio.run(run_comprehensive_test())
    elif args.test == 'basic':
        asyncio.run(test_basic_connection())
    elif args.test == 'subscription':
        asyncio.run(test_subscription_functionality())
    elif args.test == 'heartbeat':
        asyncio.run(test_heartbeat_functionality())
    elif args.test == 'stats':
        asyncio.run(test_stats_functionality())

if __name__ == "__main__":
    main()
