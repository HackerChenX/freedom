#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
综合选股测试系统管理器

提供系统组件集成、初始化和管理功能。
整合测试执行、监控、日志和告警系统。
遵循L2基础设施层规范。
"""

import os
import sys
import signal
import atexit
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime

from utils.logger import getLogger
from .stock_selection_tester import ComprehensiveStockSelectionTester as StockSelectionTester
from .test_config_manager import TestConfigManager, get_config_manager
from .performance_monitor import PerformanceMonitor
from .logging_config import get_test_log_manager
from .monitoring import get_test_monitoring_system
from .monitoring_dashboard import get_monitoring_dashboard, get_alert_notifier, start_monitoring_services, stop_monitoring_services
from .test_orchestrator import TestOrchestrator, create_orchestrator
from .system_integration import get_system_integrator
from db.sql_manager import SQLManager, QueryType

logger = getLogger(__name__)


class SystemManager:
    """系统管理器"""
    
    def __init__(self, config_path: Optional[str] = None, 
                workspace_dir: str = "test_workspace"):
        """
        初始化系统管理器
        
        Args:
            config_path: 配置文件路径
            workspace_dir: 工作空间目录
        """
        self.config_path = config_path
        self.workspace_dir = Path(workspace_dir)
        self.components = {}
        self.initialized = False
        self.shutdown_hooks = []
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 注册退出处理
        atexit.register(self.shutdown)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.warning(f"收到信号 {signum}，正在关闭系统")
        self.shutdown()
        sys.exit(1)
    
    def initialize(self, enable_monitoring: bool = True, 
                 enable_dashboard: bool = False) -> Dict[str, Any]:
        """
        初始化系统组件
        
        Args:
            enable_monitoring: 是否启用监控
            enable_dashboard: 是否启用仪表板
            
        Returns:
            Dict[str, Any]: 系统组件字典
        """
        if self.initialized:
            return self.components
        
        logger.info("初始化系统组件")
        
        try:
            # 确保工作空间目录存在
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化日志系统
            log_dir = self.workspace_dir / "logs"
            log_manager = get_test_log_manager()
            self.components['log_manager'] = log_manager
            
            # 初始化配置管理器
            config_manager = get_config_manager(self.config_path)
            config = config_manager.get_config()
            self.components['config_manager'] = config_manager
            self.components['config'] = config
            
            # 初始化监控系统
            if enable_monitoring:
                # 启动监控服务
                start_monitoring_services(start_dashboard=enable_dashboard)
                
                monitoring_system = get_test_monitoring_system()
                self.components['monitoring_system'] = monitoring_system
                
                # 注册关闭钩子
                self.register_shutdown_hook(lambda: stop_monitoring_services())
            
            # 初始化性能监控
            performance_monitor = PerformanceMonitor()
            self.components['performance_monitor'] = performance_monitor
            
            # 初始化测试器
            tester = StockSelectionTester()
            self.components['tester'] = tester
            
            # 初始化测试编排器
            orchestrator = create_orchestrator(config, str(self.workspace_dir))
            self.components['orchestrator'] = orchestrator
            
            # 初始化系统集成器
            system_integrator = get_system_integrator()
            system_integrator.initialize_system()
            self.components['system_integrator'] = system_integrator
            
            self.initialized = True
            logger.info("系统组件初始化完成")
            
            # 记录审计日志
            if 'log_manager' in self.components and hasattr(log_manager, 'log_audit'):
                log_manager.log_audit(
                    action="SYSTEM_INIT",
                    resource="system",
                    result="success",
                    details={
                        "components": list(self.components.keys()),
                        "monitoring_enabled": enable_monitoring,
                        "dashboard_enabled": enable_dashboard
                    }
                )
            
            return self.components
            
        except Exception as e:
            logger.error(f"初始化系统组件失败: {e}")
            # 尝试关闭已初始化的组件
            self.shutdown()
            raise
    
    def register_shutdown_hook(self, hook: Callable[[], None]) -> None:
        """
        注册关闭钩子
        
        Args:
            hook: 关闭钩子函数
        """
        self.shutdown_hooks.append(hook)
    
    def shutdown(self) -> None:
        """关闭系统组件"""
        if not self.initialized:
            return
        
        logger.info("关闭系统组件")
        
        # 执行关闭钩子
        for hook in reversed(self.shutdown_hooks):
            try:
                hook()
            except Exception as e:
                logger.error(f"执行关闭钩子失败: {e}")
        
        # 关闭性能监控
        if 'performance_monitor' in self.components:
            try:
                self.components['performance_monitor'].stop_monitoring()
            except Exception as e:
                logger.error(f"关闭性能监控失败: {e}")
        
        # 记录审计日志
        if 'log_manager' in self.components and hasattr(self.components['log_manager'], 'log_audit'):
            try:
                self.components['log_manager'].log_audit(
                    action="SYSTEM_SHUTDOWN",
                    resource="system",
                    result="success"
                )
            except Exception as e:
                logger.error(f"记录审计日志失败: {e}")
        
        self.initialized = False
        logger.info("系统组件已关闭")
    
    def get_component(self, name: str) -> Any:
        """
        获取系统组件
        
        Args:
            name: 组件名称
            
        Returns:
            Any: 组件实例
        """
        if not self.initialized:
            raise RuntimeError("系统未初始化")
        
        if name not in self.components:
            raise KeyError(f"组件不存在: {name}")
        
        return self.components[name]
    
    def run_test(self, progress_callback: Optional[Callable] = None) -> Any:
        """
        运行测试
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            Any: 测试结果
        """
        if not self.initialized:
            raise RuntimeError("系统未初始化")
        
        orchestrator = self.components['orchestrator']
        
        # 记录审计日志
        if 'log_manager' in self.components and hasattr(self.components['log_manager'], 'log_audit'):
            self.components['log_manager'].log_audit(
                action="TEST_START",
                resource="test",
                result="success"
            )
        
        # 启动性能监控
        if 'performance_monitor' in self.components:
            self.components['performance_monitor'].start_monitoring()
        
        # 记录测试开始
        start_time = datetime.now()
        logger.info(f"开始执行测试，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行测试
        try:
            session = orchestrator.execute_comprehensive_test(progress_callback)
            
            # 记录测试结束
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"测试执行完成，状态: {session.status}，耗时: {duration:.1f}秒")
            
            # 记录审计日志
            if 'log_manager' in self.components and hasattr(self.components['log_manager'], 'log_audit'):
                self.components['log_manager'].log_audit(
                    action="TEST_COMPLETE",
                    resource="test",
                    result="success",
                    details={
                        "session_id": session.session_id,
                        "status": session.status,
                        "duration": duration
                    }
                )
            
            return session
            
        except Exception as e:
            # 记录测试失败
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"测试执行失败，耗时: {duration:.1f}秒，错误: {e}")
            
            # 记录审计日志
            if 'log_manager' in self.components and hasattr(self.components['log_manager'], 'log_audit'):
                self.components['log_manager'].log_audit(
                    action="TEST_FAILED",
                    resource="test",
                    result="failure",
                    details={
                        "error": str(e),
                        "duration": duration
                    }
                )
            
            # 记录错误堆栈
            if 'log_manager' in self.components:
                self.components['log_manager'].log_stack_trace(e, "测试执行失败")
            
            raise
        
        finally:
            # 停止性能监控
            if 'performance_monitor' in self.components:
                self.components['performance_monitor'].stop_monitoring()


# 全局系统管理器实例
_system_manager = None


def get_system_manager(config_path: Optional[str] = None, 
                     workspace_dir: str = "test_workspace") -> SystemManager:
    """
    获取全局系统管理器实例
    
    Args:
        config_path: 配置文件路径
        workspace_dir: 工作空间目录
        
    Returns:
        SystemManager: 系统管理器实例
    """
    global _system_manager
    if _system_manager is None:
        _system_manager = SystemManager(config_path, workspace_dir)
    return _system_manager


def initialize_system(config_path: Optional[str] = None, 
                    workspace_dir: str = "test_workspace",
                    enable_monitoring: bool = True,
                    enable_dashboard: bool = False) -> Dict[str, Any]:
    """
    初始化系统组件
    
    Args:
        config_path: 配置文件路径
        workspace_dir: 工作空间目录
        enable_monitoring: 是否启用监控
        enable_dashboard: 是否启用仪表板
        
    Returns:
        Dict[str, Any]: 系统组件字典
    """
    manager = get_system_manager(config_path, workspace_dir)
    return manager.initialize(enable_monitoring, enable_dashboard)


def shutdown_system() -> None:
    """关闭系统组件"""
    global _system_manager
    if _system_manager:
        _system_manager.shutdown()


def main():
    """系统管理器测试"""
    print("系统管理器测试...")
    
    # 初始化系统
    components = initialize_system(enable_dashboard=True)
    
    print("系统组件:")
    for name, component in components.items():
        print(f"  - {name}: {type(component).__name__}")
    
    # 等待用户输入
    input("按Enter键关闭系统...")
    
    # 关闭系统
    shutdown_system()
    
    print("系统管理器测试完成")


if __name__ == "__main__":
    main()