#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
监控仪表板

提供实时监控仪表板，显示测试执行状态、系统资源使用情况和性能指标。
支持Web界面和命令行界面。
"""

import os
import sys
import json
import time
import threading
import webbrowser
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import http.server
import socketserver
import urllib.parse

from .monitoring import get_test_monitoring_system
from .logging_config import get_test_logger

logger = get_test_logger('monitoring')


class DashboardDataProvider:
    """仪表板数据提供者"""
    
    def __init__(self):
        """初始化数据提供者"""
        self.monitoring_system = get_test_monitoring_system()
        self.data_cache = {}
        self.cache_timestamp = {}
        self.cache_ttl = 5  # 缓存有效期（秒）
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            Dict[str, Any]: 系统状态数据
        """
        cache_key = 'system_status'
        current_time = time.time()
        
        # 检查缓存
        if cache_key in self.cache_timestamp and \
           current_time - self.cache_timestamp[cache_key] < self.cache_ttl:
            return self.data_cache[cache_key]
        
        # 获取新数据
        status = self.monitoring_system.get_system_status()
        
        # 更新缓存
        self.data_cache[cache_key] = status
        self.cache_timestamp[cache_key] = current_time
        
        return status
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        获取活跃会话
        
        Returns:
            List[Dict[str, Any]]: 活跃会话列表
        """
        cache_key = 'active_sessions'
        current_time = time.time()
        
        # 检查缓存
        if cache_key in self.cache_timestamp and \
           current_time - self.cache_timestamp[cache_key] < self.cache_ttl:
            return self.data_cache[cache_key]
        
        # 获取活跃会话
        sessions = []
        for session_id, session_info in self.monitoring_system.test_sessions.items():
            sessions.append({
                'session_id': session_id,
                'test_name': session_info.get('test_name', 'Unknown'),
                'start_time': session_info.get('start_time', datetime.now()).isoformat(),
                'duration': (datetime.now() - session_info.get('start_time', datetime.now())).total_seconds(),
                'metrics': {k: v[-1] if v else None for k, v in session_info.get('metrics', {}).items()}
            })
        
        # 更新缓存
        self.data_cache[cache_key] = sessions
        self.cache_timestamp[cache_key] = current_time
        
        return sessions
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        获取告警
        
        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        cache_key = 'alerts'
        current_time = time.time()
        
        # 检查缓存
        if cache_key in self.cache_timestamp and \
           current_time - self.cache_timestamp[cache_key] < self.cache_ttl:
            return self.data_cache[cache_key]
        
        # 获取告警
        alerts = []
        for alert in self.monitoring_system.alert_manager.get_active_alerts():
            alerts.append({
                'name': alert.name,
                'level': alert.level,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'tags': alert.tags
            })
        
        # 更新缓存
        self.data_cache[cache_key] = alerts
        self.cache_timestamp[cache_key] = current_time
        
        return alerts
    
    def get_metrics_history(self, metric_name: str, minutes: int = 30) -> List[Dict[str, Any]]:
        """
        获取指标历史数据
        
        Args:
            metric_name: 指标名称
            minutes: 时间范围（分钟）
            
        Returns:
            List[Dict[str, Any]]: 指标历史数据
        """
        cache_key = f'metrics_history_{metric_name}_{minutes}'
        current_time = time.time()
        
        # 检查缓存
        if cache_key in self.cache_timestamp and \
           current_time - self.cache_timestamp[cache_key] < self.cache_ttl:
            return self.data_cache[cache_key]
        
        # 获取指标历史
        metrics = []
        for point in self.monitoring_system.metrics_collector.get_metric(metric_name, minutes):
            metrics.append({
                'timestamp': point.timestamp.isoformat(),
                'value': point.value,
                'tags': point.tags
            })
        
        # 更新缓存
        self.data_cache[cache_key] = metrics
        self.cache_timestamp[cache_key] = current_time
        
        return metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表板完整数据
        
        Returns:
            Dict[str, Any]: 仪表板数据
        """
        return {
            'system_status': self.get_system_status(),
            'active_sessions': self.get_active_sessions(),
            'alerts': self.get_alerts(),
            'metrics': {
                'cpu_usage': self.get_metrics_history('cpu_usage', 10),
                'memory_usage': self.get_metrics_history('memory_usage', 10),
                'disk_io': self.get_metrics_history('disk_io', 10),
                'network_io': self.get_metrics_history('network_io', 10)
            },
            'timestamp': datetime.now().isoformat()
        }


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    """仪表板请求处理器"""
    
    def __init__(self, *args, **kwargs):
        self.data_provider = DashboardDataProvider()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # API请求
        if parsed_path.path.startswith('/api/'):
            self.handle_api_request(parsed_path.path)
            return
        
        # 静态文件请求
        if parsed_path.path == '/':
            self.path = '/dashboard.html'
        
        # 查找静态文件
        static_dir = Path(__file__).parent / 'static'
        file_path = static_dir / self.path.lstrip('/')
        
        if file_path.exists() and file_path.is_file():
            self.send_response(200)
            
            # 设置内容类型
            if file_path.suffix == '.html':
                self.send_header('Content-type', 'text/html')
            elif file_path.suffix == '.js':
                self.send_header('Content-type', 'application/javascript')
            elif file_path.suffix == '.css':
                self.send_header('Content-type', 'text/css')
            elif file_path.suffix == '.json':
                self.send_header('Content-type', 'application/json')
            else:
                self.send_header('Content-type', 'text/plain')
            
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            # 生成默认仪表板
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.generate_default_dashboard().encode('utf-8'))
    
    def handle_api_request(self, path: str):
        """
        处理API请求
        
        Args:
            path: 请求路径
        """
        if path == '/api/status':
            data = self.data_provider.get_system_status()
        elif path == '/api/sessions':
            data = self.data_provider.get_active_sessions()
        elif path == '/api/alerts':
            data = self.data_provider.get_alerts()
        elif path == '/api/dashboard':
            data = self.data_provider.get_dashboard_data()
        else:
            self.send_error(404, 'API endpoint not found')
            return
        
        # 发送JSON响应
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def generate_default_dashboard(self) -> str:
        """
        生成默认仪表板HTML
        
        Returns:
            str: 仪表板HTML
        """
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>测试监控仪表板</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                h1, h2, h3 {{ color: #333; }}
                .dashboard {{ display: flex; flex-wrap: wrap; }}
                .card {{ background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                       margin: 10px; padding: 15px; flex: 1; min-width: 300px; }}
                .status-ok {{ color: green; }}
                .status-warning {{ color: orange; }}
                .status-error {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .refresh-button {{ background-color: #4CAF50; color: white; padding: 10px 15px; 
                                 border: none; border-radius: 4px; cursor: pointer; }}
                .refresh-button:hover {{ background-color: #45a049; }}
            </style>
        </head>
        <body>
            <h1>测试监控仪表板</h1>
            <p>最后更新时间: <span id="update-time">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
            <button class="refresh-button" onclick="refreshDashboard()">刷新数据</button>
            
            <div class="dashboard">
                <div class="card">
                    <h2>系统状态</h2>
                    <div id="system-status">加载中...</div>
                </div>
                
                <div class="card">
                    <h2>资源使用</h2>
                    <div id="resource-usage">加载中...</div>
                </div>
                
                <div class="card">
                    <h2>活跃会话</h2>
                    <div id="active-sessions">加载中...</div>
                </div>
                
                <div class="card">
                    <h2>告警</h2>
                    <div id="alerts">加载中...</div>
                </div>
            </div>
            
            <script>
                // 页面加载完成后获取数据
                document.addEventListener('DOMContentLoaded', function() {{
                    refreshDashboard();
                    // 每30秒自动刷新一次
                    setInterval(refreshDashboard, 30000);
                }});
                
                // 刷新仪表板数据
                function refreshDashboard() {{
                    fetch('/api/dashboard')
                        .then(response => response.json())
                        .then(data => {{
                            updateDashboard(data);
                        }})
                        .catch(error => {{
                            console.error('获取仪表板数据失败:', error);
                        }});
                }}
                
                // 更新仪表板
                function updateDashboard(data) {{
                    // 更新时间
                    document.getElementById('update-time').textContent = new Date().toLocaleString();
                    
                    // 更新系统状态
                    const systemStatus = data.system_status;
                    let statusHtml = `
                        <p>监控状态: <span class="status-${{systemStatus.monitoring_active ? 'ok' : 'error'}}">
                            ${{systemStatus.monitoring_active ? '活跃' : '未运行'}}</span></p>
                        <p>活跃会话数: ${{systemStatus.active_sessions}}</p>
                        <p>活跃告警数: <span class="${{systemStatus.active_alerts > 0 ? 'status-warning' : 'status-ok'}}">
                            ${{systemStatus.active_alerts}}</span></p>
                    `;
                    document.getElementById('system-status').innerHTML = statusHtml;
                    
                    // 更新资源使用
                    const metrics = systemStatus.system_metrics;
                    let resourceHtml = '';
                    if (metrics.cpu_usage) {{
                        const cpuValue = metrics.cpu_usage.value;
                        resourceHtml += `
                            <p>CPU使用率: <span class="metric-value ${{cpuValue > 80 ? 'status-error' : 
                                                                     cpuValue > 60 ? 'status-warning' : 'status-ok'}}">
                                ${{cpuValue.toFixed(1)}}%</span></p>
                        `;
                    }}
                    if (metrics.memory_usage) {{
                        const memValue = metrics.memory_usage.value;
                        resourceHtml += `
                            <p>内存使用: <span class="metric-value ${{memValue > 6 ? 'status-error' : 
                                                                   memValue > 4 ? 'status-warning' : 'status-ok'}}">
                                ${{memValue.toFixed(2)}} GB</span></p>
                        `;
                    }}
                    if (metrics.disk_io) {{
                        resourceHtml += `
                            <p>磁盘I/O: <span class="metric-value">
                                ${{metrics.disk_io.value.toFixed(2)}} MB/s</span></p>
                        `;
                    }}
                    if (metrics.network_io) {{
                        resourceHtml += `
                            <p>网络I/O: <span class="metric-value">
                                ${{metrics.network_io.value.toFixed(2)}} MB/s</span></p>
                        `;
                    }}
                    document.getElementById('resource-usage').innerHTML = resourceHtml || '无资源使用数据';
                    
                    // 更新活跃会话
                    const sessions = data.active_sessions;
                    if (sessions.length > 0) {{
                        let sessionsHtml = '<table><tr><th>会话ID</th><th>测试名称</th><th>开始时间</th><th>运行时间</th></tr>';
                        sessions.forEach(session => {{
                            const startTime = new Date(session.start_time);
                            const duration = formatDuration(session.duration);
                            sessionsHtml += `
                                <tr>
                                    <td>${{session.session_id}}</td>
                                    <td>${{session.test_name}}</td>
                                    <td>${{startTime.toLocaleString()}}</td>
                                    <td>${{duration}}</td>
                                </tr>
                            `;
                        }});
                        sessionsHtml += '</table>';
                        document.getElementById('active-sessions').innerHTML = sessionsHtml;
                    }} else {{
                        document.getElementById('active-sessions').innerHTML = '<p>无活跃会话</p>';
                    }}
                    
                    // 更新告警
                    const alerts = data.alerts;
                    if (alerts.length > 0) {{
                        let alertsHtml = '<table><tr><th>级别</th><th>名称</th><th>消息</th><th>时间</th></tr>';
                        alerts.forEach(alert => {{
                            const alertTime = new Date(alert.timestamp);
                            alertsHtml += `
                                <tr>
                                    <td class="status-${{alert.level === 'ERROR' ? 'error' : 
                                                      alert.level === 'WARNING' ? 'warning' : 'ok'}}">${{alert.level}}</td>
                                    <td>${{alert.name}}</td>
                                    <td>${{alert.message}}</td>
                                    <td>${{alertTime.toLocaleString()}}</td>
                                </tr>
                            `;
                        }});
                        alertsHtml += '</table>';
                        document.getElementById('alerts').innerHTML = alertsHtml;
                    }} else {{
                        document.getElementById('alerts').innerHTML = '<p>无活跃告警</p>';
                    }}
                }}
                
                // 格式化持续时间
                function formatDuration(seconds) {{
                    const hours = Math.floor(seconds / 3600);
                    const minutes = Math.floor((seconds % 3600) / 60);
                    const secs = Math.floor(seconds % 60);
                    
                    let result = '';
                    if (hours > 0) result += `${{hours}}小时 `;
                    if (minutes > 0) result += `${{minutes}}分钟 `;
                    result += `${{secs}}秒`;
                    
                    return result;
                }}
            </script>
        </body>
        </html>
        """
    
    def log_message(self, format, *args):
        """重写日志方法，使用自定义日志器"""
        logger.debug(f"{self.address_string()} - {format % args}")


class MonitoringDashboard:
    """监控仪表板"""
    
    def __init__(self, host: str = 'localhost', port: int = 8080):
        """
        初始化监控仪表板
        
        Args:
            host: 主机地址
            port: 端口号
        """
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
    
    def start(self, open_browser: bool = True) -> None:
        """
        启动仪表板服务器
        
        Args:
            open_browser: 是否自动打开浏览器
        """
        if self.running:
            logger.warning("仪表板服务器已经在运行")
            return
        
        try:
            # 创建静态文件目录
            static_dir = Path(__file__).parent / 'static'
            static_dir.mkdir(exist_ok=True)
            
            # 创建服务器
            self.server = socketserver.ThreadingTCPServer((self.host, self.port), DashboardRequestHandler)
            self.server.daemon_threads = True
            
            # 启动服务器线程
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            logger.info(f"监控仪表板服务器已启动: http://{self.host}:{self.port}")
            
            # 打开浏览器
            if open_browser:
                webbrowser.open(f"http://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"启动仪表板服务器失败: {e}")
            raise
    
    def stop(self) -> None:
        """停止仪表板服务器"""
        if not self.running:
            return
        
        try:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            logger.info("监控仪表板服务器已停止")
            
        except Exception as e:
            logger.error(f"停止仪表板服务器失败: {e}")
    
    def is_running(self) -> bool:
        """
        检查仪表板服务器是否运行
        
        Returns:
            bool: 是否运行
        """
        return self.running


class AlertNotifier:
    """告警通知器"""
    
    def __init__(self):
        """初始化告警通知器"""
        self.monitoring_system = get_test_monitoring_system()
        self.notifiers = {}
        self.running = False
        self.thread = None
    
    def register_notifier(self, name: str, notifier_func: Callable[[Dict[str, Any]], None]) -> None:
        """
        注册通知器
        
        Args:
            name: 通知器名称
            notifier_func: 通知函数
        """
        self.notifiers[name] = notifier_func
        logger.info(f"注册告警通知器: {name}")
    
    def start(self) -> None:
        """启动告警通知器"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._notification_loop, daemon=True)
        self.thread.start()
        
        logger.info("告警通知器已启动")
    
    def stop(self) -> None:
        """停止告警通知器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("告警通知器已停止")
    
    def _notification_loop(self) -> None:
        """通知循环"""
        last_check_time = datetime.now()
        
        while self.running:
            try:
                # 获取新告警
                current_time = datetime.now()
                new_alerts = [
                    alert for alert in self.monitoring_system.alert_manager.get_active_alerts()
                    if alert.timestamp > last_check_time
                ]
                
                # 发送通知
                for alert in new_alerts:
                    self._send_notification(alert)
                
                last_check_time = current_time
                time.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"告警通知循环错误: {e}")
                time.sleep(30)  # 出错后等待30秒
    
    def _send_notification(self, alert: Any) -> None:
        """
        发送告警通知
        
        Args:
            alert: 告警对象
        """
        alert_data = {
            'name': alert.name,
            'level': alert.level,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'tags': alert.tags
        }
        
        for name, notifier in self.notifiers.items():
            try:
                notifier(alert_data)
            except Exception as e:
                logger.error(f"发送告警通知失败 ({name}): {e}")
    
    def console_notifier(self, alert_data: Dict[str, Any]) -> None:
        """
        控制台通知器
        
        Args:
            alert_data: 告警数据
        """
        level = alert_data['level']
        name = alert_data['name']
        message = alert_data['message']
        timestamp = datetime.fromisoformat(alert_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n[{level}] {name} - {timestamp}")
        print(f"  {message}")
        print("-" * 50)
    
    def log_notifier(self, alert_data: Dict[str, Any]) -> None:
        """
        日志通知器
        
        Args:
            alert_data: 告警数据
        """
        level = alert_data['level']
        name = alert_data['name']
        message = alert_data['message']
        
        if level == 'ERROR':
            logger.error(f"告警: {name} - {message}")
        elif level == 'WARNING':
            logger.warning(f"告警: {name} - {message}")
        else:
            logger.info(f"告警: {name} - {message}")


# 全局仪表板实例
_monitoring_dashboard = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """
    获取全局监控仪表板实例
    
    Returns:
        MonitoringDashboard: 监控仪表板实例
    """
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    return _monitoring_dashboard


# 全局告警通知器实例
_alert_notifier = None


def get_alert_notifier() -> AlertNotifier:
    """
    获取全局告警通知器实例
    
    Returns:
        AlertNotifier: 告警通知器实例
    """
    global _alert_notifier
    if _alert_notifier is None:
        _alert_notifier = AlertNotifier()
        # 注册默认通知器
        _alert_notifier.register_notifier('console', _alert_notifier.console_notifier)
        _alert_notifier.register_notifier('log', _alert_notifier.log_notifier)
    return _alert_notifier


def start_monitoring_services(start_dashboard: bool = True, 
                            start_notifier: bool = True) -> None:
    """
    启动监控服务
    
    Args:
        start_dashboard: 是否启动仪表板
        start_notifier: 是否启动通知器
    """
    # 启动监控系统
    monitoring_system = get_test_monitoring_system()
    monitoring_system.start_monitoring()
    
    # 启动仪表板
    if start_dashboard:
        dashboard = get_monitoring_dashboard()
        dashboard.start(open_browser=False)
    
    # 启动通知器
    if start_notifier:
        notifier = get_alert_notifier()
        notifier.start()
    
    logger.info("监控服务已启动")


def stop_monitoring_services() -> None:
    """停止监控服务"""
    # 停止通知器
    global _alert_notifier
    if _alert_notifier:
        _alert_notifier.stop()
    
    # 停止仪表板
    global _monitoring_dashboard
    if _monitoring_dashboard:
        _monitoring_dashboard.stop()
    
    # 停止监控系统
    monitoring_system = get_test_monitoring_system()
    monitoring_system.stop_monitoring()
    
    logger.info("监控服务已停止")


if __name__ == "__main__":
    # 启动监控服务
    start_monitoring_services()
    
    try:
        print("监控仪表板已启动，按Ctrl+C退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在停止监控服务...")
        stop_monitoring_services()
        print("监控服务已停止")