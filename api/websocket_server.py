#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
WebSocket实时推送服务器
实现股票数据、预警消息、监控状态的实时推送功能
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class WebSocketConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        """初始化连接管理器"""
        self.connections: Set[WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.connection_info: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}
        logger.info("WebSocket连接管理器初始化完成")
    
    @exception_handler(reraise=True)
    async def register_connection(self, websocket: WebSocketServerProtocol, path: str):
        """注册新连接"""
        self.connections.add(websocket)
        self.connection_info[websocket] = {
            'path': path,
            'connected_at': datetime.now(),
            'last_ping': datetime.now(),
            'subscriptions': set()
        }
        logger.info(f"新WebSocket连接注册: {websocket.remote_address}, 路径: {path}")
        logger.info(f"当前连接数: {len(self.connections)}")
    
    @exception_handler(reraise=True)
    async def unregister_connection(self, websocket: WebSocketServerProtocol):
        """注销连接"""
        if websocket in self.connections:
            self.connections.remove(websocket)
            
            # 从所有订阅中移除
            for topic in list(self.subscriptions.keys()):
                if websocket in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(websocket)
                    if not self.subscriptions[topic]:
                        del self.subscriptions[topic]
            
            # 移除连接信息
            if websocket in self.connection_info:
                del self.connection_info[websocket]
            
            logger.info(f"WebSocket连接注销: {websocket.remote_address}")
            logger.info(f"当前连接数: {len(self.connections)}")
    
    @exception_handler(reraise=True)
    async def subscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """订阅主题"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        
        self.subscriptions[topic].add(websocket)
        
        if websocket in self.connection_info:
            self.connection_info[websocket]['subscriptions'].add(topic)
        
        logger.info(f"连接 {websocket.remote_address} 订阅主题: {topic}")
        logger.info(f"主题 {topic} 订阅者数量: {len(self.subscriptions[topic])}")
    
    @exception_handler(reraise=True)
    async def unsubscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """取消订阅主题"""
        if topic in self.subscriptions and websocket in self.subscriptions[topic]:
            self.subscriptions[topic].remove(websocket)
            
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            
            if websocket in self.connection_info:
                self.connection_info[websocket]['subscriptions'].discard(topic)
            
            logger.info(f"连接 {websocket.remote_address} 取消订阅主题: {topic}")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """向主题订阅者广播消息"""
        if topic not in self.subscriptions:
            logger.debug(f"主题 {topic} 没有订阅者")
            return
        
        message_str = json.dumps(message, ensure_ascii=False, default=str)
        disconnected = set()
        
        for websocket in self.subscriptions[topic]:
            try:
                await websocket.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"发送消息到 {websocket.remote_address} 失败: {e}")
                disconnected.add(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            await self.unregister_connection(websocket)
        
        logger.debug(f"向主题 {topic} 广播消息，成功发送: {len(self.subscriptions[topic]) - len(disconnected)}, 失败: {len(disconnected)}")
    
    @exception_handler(reraise=True)
    async def send_to_connection(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """向特定连接发送消息"""
        try:
            message_str = json.dumps(message, ensure_ascii=False, default=str)
            await websocket.send(message_str)
            logger.debug(f"向连接 {websocket.remote_address} 发送消息成功")
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_connection(websocket)
        except Exception as e:
            logger.error(f"向连接 {websocket.remote_address} 发送消息失败: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            'total_connections': len(self.connections),
            'total_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'topics': list(self.subscriptions.keys()),
            'topic_stats': {topic: len(subs) for topic, subs in self.subscriptions.items()}
        }

class WebSocketMessageHandler:
    """WebSocket消息处理器"""
    
    def __init__(self, connection_manager: WebSocketConnectionManager):
        """初始化消息处理器"""
        self.connection_manager = connection_manager
        logger.info("WebSocket消息处理器初始化完成")
    
    @exception_handler(reraise=True)
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self._handle_subscribe(websocket, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscribe(websocket, data)
            elif message_type == 'ping':
                await self._handle_ping(websocket, data)
            elif message_type == 'get_stats':
                await self._handle_get_stats(websocket, data)
            else:
                await self._handle_unknown_message(websocket, data)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            await self._send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self._send_error(websocket, "Message processing failed")
    
    async def _handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理订阅请求"""
        topic = data.get('topic')
        if not topic:
            await self._send_error(websocket, "Topic is required for subscription")
            return
        
        await self.get_connection_pool().subscribe(websocket, topic)
        await self.get_connection_pool().send_to_connection(websocket, {
            'type': 'subscription_confirmed',
            'topic': topic,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理取消订阅请求"""
        topic = data.get('topic')
        if not topic:
            await self._send_error(websocket, "Topic is required for unsubscription")
            return
        
        await self.get_connection_pool().unsubscribe(websocket, topic)
        await self.get_connection_pool().send_to_connection(websocket, {
            'type': 'unsubscription_confirmed',
            'topic': topic,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_ping(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理心跳请求"""
        if websocket in self.get_connection_pool().connection_info:
            self.get_connection_pool().connection_info[websocket]['last_ping'] = datetime.now()
        
        await self.get_connection_pool().send_to_connection(websocket, {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_get_stats(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理获取统计信息请求"""
        stats = self.get_connection_pool().get_connection_stats()
        await self.get_connection_pool().send_to_connection(websocket, {
            'type': 'stats',
            'data': stats,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_unknown_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """处理未知消息类型"""
        logger.warning(f"收到未知消息类型: {data.get('type')}")
        await self._send_error(websocket, f"Unknown message type: {data.get('type')}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str):
        """发送错误消息"""
        await self.get_connection_pool().send_to_connection(websocket, {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        })

class WebSocketServer:
    """WebSocket服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """初始化WebSocket服务器"""
        self.host = host
        self.port = port
        self.connection_manager = WebSocketget_connection_pool()
        self.message_handler = WebSocketMessageHandler(self.connection_manager)
        self.server = None
        logger.info(f"WebSocket服务器初始化完成: {host}:{port}")
    
    @exception_handler(reraise=True)
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        await self.get_connection_pool().register_connection(websocket, path)
        
        try:
            # 发送欢迎消息
            await self.get_connection_pool().send_to_connection(websocket, {
                'type': 'welcome',
                'message': '欢迎连接到股票分析系统WebSocket服务器',
                'server_time': datetime.now().isoformat(),
                'available_topics': [
                    'stock_prices',      # 股票价格推送
                    'alerts',           # 预警消息推送
                    'monitoring',       # 监控状态推送
                    'indicators',       # 技术指标推送
                    'risk_updates'      # 风险更新推送
                ]
            })
            
            # 处理客户端消息
            async for message in websocket:
                await self.message_handler.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端连接关闭: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"处理客户端连接时出错: {e}")
        finally:
            await self.get_connection_pool().unregister_connection(websocket)
    
    @exception_handler(reraise=True)
    async def start_server(self):
        """启动WebSocket服务器"""
        logger.info(f"启动WebSocket服务器: {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,  # 30秒心跳间隔
            ping_timeout=10,   # 10秒心跳超时
            max_size=1024*1024,  # 1MB最大消息大小
            max_queue=32       # 最大队列大小
        )
        
        logger.info(f"WebSocket服务器启动成功: ws://{self.host}:{self.port}")
        return self.server
    
    @exception_handler(reraise=True)
    async def stop_server(self):
        """停止WebSocket服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket服务器已停止")
    
    def get_connection_manager(self) -> WebSocketConnectionManager:
        """获取连接管理器"""
        return self.connection_manager

# 全局WebSocket服务器实例
_websocket_server: Optional[WebSocketServer] = None

def get_websocket_server() -> WebSocketServer:
    """获取WebSocket服务器实例"""
    global _websocket_server
    if _websocket_server is None:
        _websocket_server = WebSocketServer()
    return _websocket_server

@exception_handler(reraise=True)
async def start_websocket_server(host: str = "localhost", port: int = 8765):
    """启动WebSocket服务器"""
    server = get_websocket_server()
    server.host = host
    server.port = port
    return await server.start_server()

@exception_handler(reraise=True)
async def stop_websocket_server():
    """停止WebSocket服务器"""
    server = get_websocket_server()
    await server.stop_server()

if __name__ == "__main__":
    # 测试WebSocket服务器
    async def main():
        server = await start_websocket_server()
        logger.info("WebSocket服务器运行中，按Ctrl+C停止...")
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            await stop_websocket_server()
    
    asyncio.run(main())
