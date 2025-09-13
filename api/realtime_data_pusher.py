#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
实时数据推送服务
负责向WebSocket客户端推送股票价格、预警消息、监控状态等实时数据
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container
from api.websocket_server import get_websocket_server

logger = get_logger(__name__)

class StockPricePusher:
    """股票价格推送器"""
    
    def __init__(self):
        """初始化股票价格推送器"""
        self.websocket_server = get_websocket_server()
        self.is_running = False
        self.push_interval = 2.0  # 2秒推送间隔
        self.stock_codes = ['000001', '000002', '000858', '600000', '600036']  # 示例股票代码
        logger.info("股票价格推送器初始化完成")
    
    @exception_handler(reraise=True)
    async def start_pushing(self):
        """开始推送股票价格"""
        self.is_running = True
        logger.info("开始推送股票价格数据")
        
        while self.is_running:
            try:
                # 生成模拟股票价格数据
                price_data = await self._generate_stock_prices()
                
                # 推送到WebSocket客户端
                await self.websocket_server.connection_manager.broadcast_to_topic(
                    'stock_prices', 
                    {
                        'type': 'stock_prices',
                        'data': price_data,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                await asyncio.sleep(self.push_interval)
                
            except Exception as e:
                logger.error(f"推送股票价格数据时出错: {e}")
                await asyncio.sleep(1)
    
    @exception_handler(reraise=True)
    async def stop_pushing(self):
        """停止推送股票价格"""
        self.is_running = False
        logger.info("停止推送股票价格数据")
    
    @exception_handler(reraise=True)
    async def _generate_stock_prices(self) -> List[Dict[str, Any]]:
        """生成模拟股票价格数据"""
        prices = []
        
        for code in self.stock_codes:
            # 生成模拟价格数据
            base_price = random.uniform(10, 100)
            change_percent = random.uniform(-5, 5)
            volume = random.randint(1000000, 50000000)
            
            price_data = {
                'code': code,
                'name': f'股票{code}',
                'price': round(base_price, 2),
                'change': round(base_price * change_percent / 100, 2),
                'change_percent': round(change_percent, 2),
                'volume': volume,
                'turnover': round(base_price * volume, 2),
                'high': round(base_price * random.uniform(1.0, 1.05), 2),
                'low': round(base_price * random.uniform(0.95, 1.0), 2),
                'open': round(base_price * random.uniform(0.98, 1.02), 2),
                'timestamp': datetime.now().isoformat()
            }
            prices.append(price_data)
        
        return prices

class AlertPusher:
    """预警消息推送器"""
    
    def __init__(self):
        """初始化预警消息推送器"""
        self.websocket_server = get_websocket_server()
        self.is_running = False
        self.push_interval = 10.0  # 10秒检查间隔
        logger.info("预警消息推送器初始化完成")
    
    @exception_handler(reraise=True)
    async def start_pushing(self):
        """开始推送预警消息"""
        self.is_running = True
        logger.info("开始推送预警消息")
        
        while self.is_running:
            try:
                # 检查是否有新的预警消息
                alerts = await self._check_alerts()
                
                if alerts:
                    # 推送预警消息到WebSocket客户端
                    await self.websocket_server.connection_manager.broadcast_to_topic(
                        'alerts',
                        {
                            'type': 'alerts',
                            'data': alerts,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                
                await asyncio.sleep(self.push_interval)
                
            except Exception as e:
                logger.error(f"推送预警消息时出错: {e}")
                await asyncio.sleep(1)
    
    @exception_handler(reraise=True)
    async def stop_pushing(self):
        """停止推送预警消息"""
        self.is_running = False
        logger.info("停止推送预警消息")
    
    @exception_handler(reraise=True)
    async def _check_alerts(self) -> List[Dict[str, Any]]:
        """检查预警消息"""
        alerts = []
        
        # 模拟生成预警消息
        if random.random() < 0.3:  # 30%概率生成预警
            alert_types = ['price_alert', 'volume_alert', 'technical_alert', 'risk_alert']
            alert_type = random.choice(alert_types)
            
            alert = {
                'id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                'type': alert_type,
                'stock_code': random.choice(['000001', '000002', '000858', '600000', '600036']),
                'title': self._get_alert_title(alert_type),
                'message': self._get_alert_message(alert_type),
                'level': random.choice(['INFO', 'WARNING', 'CRITICAL']),
                'timestamp': datetime.now().isoformat()
            }
            alerts.append(alert)
        
        return alerts
    
    def _get_alert_title(self, alert_type: str) -> str:
        """获取预警标题"""
        titles = {
            'price_alert': '价格预警',
            'volume_alert': '成交量异常',
            'technical_alert': '技术指标信号',
            'risk_alert': '风险预警'
        }
        return titles.get(alert_type, '系统预警')
    
    def _get_alert_message(self, alert_type: str) -> str:
        """获取预警消息"""
        messages = {
            'price_alert': '股票价格出现异常波动，请关注',
            'volume_alert': '成交量异常放大，可能有重要消息',
            'technical_alert': 'MACD金叉信号出现，建议关注',
            'risk_alert': '风险指标超过阈值，请注意风险控制'
        }
        return messages.get(alert_type, '系统检测到异常情况')

class MonitoringStatusPusher:
    """监控状态推送器"""
    
    def __init__(self):
        """初始化监控状态推送器"""
        self.websocket_server = get_websocket_server()
        self.is_running = False
        self.push_interval = 5.0  # 5秒推送间隔
        logger.info("监控状态推送器初始化完成")
    
    @exception_handler(reraise=True)
    async def start_pushing(self):
        """开始推送监控状态"""
        self.is_running = True
        logger.info("开始推送监控状态")
        
        while self.is_running:
            try:
                # 获取系统监控状态
                status_data = await self._get_monitoring_status()
                
                # 推送到WebSocket客户端
                await self.websocket_server.connection_manager.broadcast_to_topic(
                    'monitoring',
                    {
                        'type': 'monitoring_status',
                        'data': status_data,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                
                await asyncio.sleep(self.push_interval)
                
            except Exception as e:
                logger.error(f"推送监控状态时出错: {e}")
                await asyncio.sleep(1)
    
    @exception_handler(reraise=True)
    async def stop_pushing(self):
        """停止推送监控状态"""
        self.is_running = False
        logger.info("停止推送监控状态")
    
    @exception_handler(reraise=True)
    async def _get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        # 获取WebSocket连接统计
        ws_stats = self.websocket_server.connection_manager.get_connection_stats()
        
        # 生成系统状态数据
        status = {
            'system_status': 'running',
            'websocket_connections': ws_stats['total_connections'],
            'active_subscriptions': ws_stats['total_subscriptions'],
            'cpu_usage': round(random.uniform(10, 80), 1),
            'memory_usage': round(random.uniform(30, 70), 1),
            'database_status': 'connected',
            'indicator_system_status': 'active',
            'last_update': datetime.now().isoformat(),
            'uptime': str(timedelta(seconds=random.randint(3600, 86400)))
        }
        
        return status

class RealtimeDataPusher:
    """实时数据推送服务主控制器"""
    
    def __init__(self):
        """初始化实时数据推送服务"""
        self.stock_pusher = StockPricePusher()
        self.alert_pusher = AlertPusher()
        self.monitoring_pusher = MonitoringStatusPusher()
        self.is_running = False
        logger.info("实时数据推送服务初始化完成")
    
    @exception_handler(reraise=True)
    async def start_all_pushers(self):
        """启动所有推送器"""
        if self.is_running:
            logger.warning("实时数据推送服务已在运行")
            return
        
        self.is_running = True
        logger.info("启动所有实时数据推送器")
        
        # 并发启动所有推送器
        await asyncio.gather(
            self.stock_pusher.start_pushing(),
            self.alert_pusher.start_pushing(),
            self.monitoring_pusher.start_pushing(),
            return_exceptions=True
        )
    
    @exception_handler(reraise=True)
    async def stop_all_pushers(self):
        """停止所有推送器"""
        if not self.is_running:
            logger.warning("实时数据推送服务未在运行")
            return
        
        self.is_running = False
        logger.info("停止所有实时数据推送器")
        
        # 停止所有推送器
        await asyncio.gather(
            self.stock_pusher.stop_pushing(),
            self.alert_pusher.stop_pushing(),
            self.monitoring_pusher.stop_pushing(),
            return_exceptions=True
        )
    
    @exception_handler(reraise=True)
    async def push_custom_message(self, topic: str, message: Dict[str, Any]):
        """推送自定义消息"""
        websocket_server = get_websocket_server()
        await websocket_server.connection_manager.broadcast_to_topic(topic, message)
        logger.info(f"推送自定义消息到主题 {topic}")

# 全局实时数据推送服务实例
_realtime_pusher: Optional[RealtimeDataPusher] = None

def get_realtime_pusher() -> RealtimeDataPusher:
    """获取实时数据推送服务实例"""
    global _realtime_pusher
    if _realtime_pusher is None:
        _realtime_pusher = RealtimeDataPusher()
    return _realtime_pusher

@exception_handler(reraise=True)
async def start_realtime_pushing():
    """启动实时数据推送服务"""
    pusher = get_realtime_pusher()
    await pusher.start_all_pushers()

@exception_handler(reraise=True)
async def stop_realtime_pushing():
    """停止实时数据推送服务"""
    pusher = get_realtime_pusher()
    await pusher.stop_all_pushers()

if __name__ == "__main__":
    # 测试实时数据推送服务
    async def main():
        logger.info("启动实时数据推送服务测试...")
        
        # 启动WebSocket服务器
        from api.websocket_server import start_websocket_server
        await start_websocket_server()
        
        # 启动实时数据推送
        await start_realtime_pushing()
    
    asyncio.run(main())
