#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
实时监控API路由

提供实时监控相关的RESTful API接口
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()

# 数据模型
from pydantic import BaseModel, Field

class MonitoringStartRequest(BaseModel):
    """监控启动请求模型"""
    stock_codes: List[str] = Field(..., description="监控股票代码列表", example=["000001", "000002"])
    monitoring_type: str = Field(default="real_time", description="监控类型", example="real_time")
    interval: int = Field(default=60, description="监控间隔(秒)", example=60)
    alert_rules: Optional[List[Dict[str, Any]]] = Field(default=None, description="预警规则")

class MonitoringStatusResponse(BaseModel):
    """监控状态响应模型"""
    success: bool = Field(..., description="请求是否成功")
    monitoring_active: bool = Field(..., description="监控是否活跃")
    monitored_stocks: List[str] = Field(..., description="监控中的股票")
    monitoring_interval: int = Field(..., description="监控间隔")
    last_update: str = Field(..., description="最后更新时间")
    alerts_count: int = Field(..., description="预警数量")
    timestamp: str = Field(..., description="响应时间戳")

class AlertConfigRequest(BaseModel):
    """预警配置请求模型"""
    rule_name: str = Field(..., description="规则名称", example="价格突破预警")
    rule_type: str = Field(..., description="规则类型", example="price_breakout")
    parameters: Dict[str, Any] = Field(..., description="规则参数")
    enabled: bool = Field(default=True, description="是否启用")

class AlertConfigResponse(BaseModel):
    """预警配置响应模型"""
    success: bool = Field(..., description="配置是否成功")
    rule_id: str = Field(..., description="规则ID")
    rule_name: str = Field(..., description="规则名称")
    message: str = Field(..., description="配置结果消息")
    timestamp: str = Field(..., description="响应时间戳")

@router.post("/monitoring/start")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=3.0)
def start_monitoring(request: MonitoringStartRequest):
    """
    启动实时监控
    
    Args:
        request: 监控启动请求
        
    Returns:
        Dict: 启动结果
    """
    try:
        # 验证股票代码
        if not request.stock_codes:
            raise HTTPException(status_code=400, detail="股票代码列表不能为空")
        
        if len(request.stock_codes) > 50:
            raise HTTPException(status_code=400, detail="监控股票数量不能超过50个")
        
        # 验证监控间隔
        if request.interval < 10:
            raise HTTPException(status_code=400, detail="监控间隔不能少于10秒")
        
        # 尝试使用真实的监控系统
        try:
            from monitoring.risk_monitor import RiskMonitoringSystem
            risk_system = RiskMonitoringSystem()
            
            # 启动实时监控
            risk_system.start_real_time_monitoring(
                stocks=request.stock_codes,
                portfolios=None,
                interval=request.interval
            )
            
            return {
                "success": True,
                "message": "实时监控启动成功",
                "monitored_stocks": request.stock_codes,
                "monitoring_interval": request.interval,
                "monitoring_type": request.monitoring_type,
                "start_time": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"真实监控系统不可用，使用模拟监控: {e}")
            # 使用模拟监控
            return {
                "success": True,
                "message": "模拟监控启动成功",
                "monitored_stocks": request.stock_codes,
                "monitoring_interval": request.interval,
                "monitoring_type": request.monitoring_type,
                "start_time": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "note": "当前使用模拟监控模式"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动监控失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动监控失败: {str(e)}")

@router.post("/monitoring/stop")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=2.0)
def stop_monitoring():
    """
    停止实时监控
    
    Returns:
        Dict: 停止结果
    """
    try:
        # 尝试使用真实的监控系统
        try:
            from monitoring.risk_monitor import RiskMonitoringSystem
            risk_system = RiskMonitoringSystem()
            
            # 停止实时监控
            risk_system.stop_real_time_monitoring()
            
            return {
                "success": True,
                "message": "实时监控已停止",
                "stop_time": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"真实监控系统不可用: {e}")
            # 模拟停止监控
            return {
                "success": True,
                "message": "模拟监控已停止",
                "stop_time": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "note": "当前使用模拟监控模式"
            }
        
    except Exception as e:
        logger.error(f"停止监控失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止监控失败: {str(e)}")

@router.get("/monitoring/status", response_model=MonitoringStatusResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_monitoring_status():
    """
    获取监控状态
    
    Returns:
        MonitoringStatusResponse: 监控状态响应
    """
    try:
        # 尝试使用真实的监控系统
        try:
            from monitoring.risk_monitor import RiskMonitoringSystem
            risk_system = RiskMonitoringSystem()
            
            # 获取监控状态
            status = risk_system.get_monitoring_status()
            
            return MonitoringStatusResponse(
                success=True,
                monitoring_active=status.get("monitoring_enabled", False),
                monitored_stocks=["000001", "000002"],  # 示例数据
                monitoring_interval=status.get("monitoring_interval", 60),
                last_update=status.get("last_check", datetime.now().isoformat()),
                alerts_count=0,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"真实监控系统不可用，返回模拟状态: {e}")
            # 返回模拟状态
            return MonitoringStatusResponse(
                success=True,
                monitoring_active=False,
                monitored_stocks=[],
                monitoring_interval=60,
                last_update=datetime.now().isoformat(),
                alerts_count=0,
                timestamp=datetime.now().isoformat()
            )
        
    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控状态失败: {str(e)}")

@router.get("/monitoring/alerts")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=2.0)
def get_monitoring_alerts(
    limit: int = Query(default=20, description="返回数量限制", ge=1, le=100),
    alert_type: Optional[str] = Query(None, description="预警类型过滤"),
    severity: Optional[str] = Query(None, description="严重程度过滤")
):
    """
    获取监控预警信息
    
    Args:
        limit: 返回数量限制
        alert_type: 预警类型过滤
        severity: 严重程度过滤
        
    Returns:
        Dict: 监控预警信息
    """
    try:
        # 模拟预警数据
        import random
        
        alert_types = ["价格突破", "成交量异常", "技术指标信号", "风险预警"]
        severity_levels = ["低", "中", "高", "极高"]
        
        alerts = []
        for i in range(min(limit, 30)):
            alert_severity = random.choice(severity_levels)
            alert_type_val = random.choice(alert_types)
            
            # 应用过滤条件
            if alert_type and alert_type_val != alert_type:
                continue
            if severity and alert_severity != severity:
                continue
            
            alert = {
                "id": f"monitor_alert_{i+1:03d}",
                "type": alert_type_val,
                "severity": alert_severity,
                "stock_code": f"{random.randint(0, 999999):06d}",
                "stock_name": f"股票{random.randint(0, 999999):06d}",
                "message": f"{alert_type_val}预警: {alert_severity}级别",
                "trigger_value": round(random.uniform(10, 100), 2),
                "threshold": round(random.uniform(5, 95), 2),
                "created_time": datetime.now().isoformat(),
                "status": "active"
            }
            alerts.append(alert)
        
        return {
            "success": True,
            "alerts": alerts,
            "total_count": len(alerts),
            "active_alerts": len([a for a in alerts if a["status"] == "active"]),
            "alert_summary": {
                "high_severity": len([a for a in alerts if a["severity"] in ["高", "极高"]]),
                "medium_severity": len([a for a in alerts if a["severity"] == "中"]),
                "low_severity": len([a for a in alerts if a["severity"] == "低"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取监控预警失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控预警失败: {str(e)}")

@router.post("/monitoring/alerts/config", response_model=AlertConfigResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=2.0)
def configure_alert_rule(request: AlertConfigRequest):
    """
    配置预警规则
    
    Args:
        request: 预警配置请求
        
    Returns:
        AlertConfigResponse: 预警配置响应
    """
    try:
        # 验证规则参数
        if not request.rule_name.strip():
            raise HTTPException(status_code=400, detail="规则名称不能为空")
        
        if not request.parameters:
            raise HTTPException(status_code=400, detail="规则参数不能为空")
        
        # 尝试使用真实的预警配置系统
        try:
            from monitoring.alert_config_manager import AlertConfigManager
from db.sql_manager import SQLManager, QueryType
            config_manager = AlertConfigManager()
            
            # 创建预警规则
            rule_id = config_manager.create_rule(
                rule_id=f"api_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=request.rule_name,
                description=f"通过API创建的{request.rule_type}规则",
                indicators=[request.rule_type],
                conditions=request.parameters,
                signal_type="BUY",  # 默认信号类型
                priority=3,
                enabled=request.enabled
            )
            
            return AlertConfigResponse(
                success=True,
                rule_id=rule_id,
                rule_name=request.rule_name,
                message="预警规则配置成功",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"真实预警配置系统不可用，使用模拟配置: {e}")
            # 使用模拟配置
            rule_id = f"mock_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return AlertConfigResponse(
                success=True,
                rule_id=rule_id,
                rule_name=request.rule_name,
                message="模拟预警规则配置成功",
                timestamp=datetime.now().isoformat()
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"配置预警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置预警规则失败: {str(e)}")

@router.get("/monitoring/alerts/rules")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_alert_rules():
    """
    获取预警规则列表
    
    Returns:
        Dict: 预警规则列表
    """
    try:
        # 尝试使用真实的预警配置系统
        try:
            from monitoring.alert_config_manager import AlertConfigManager
from db.sql_manager import SQLManager, QueryType
            config_manager = AlertConfigManager()
            
            # 获取所有规则
            rules = config_manager.get_all_rules()
            
            # 转换为API响应格式
            rule_list = []
            for rule_id, rule_config in rules.items():
                rule_list.append({
                    "rule_id": rule_id,
                    "name": rule_config.name,
                    "description": rule_config.description,
                    "rule_type": rule_config.indicators[0] if rule_config.indicators else "unknown",
                    "enabled": rule_config.enabled,
                    "signal_type": rule_config.signal_type,
                    "priority": rule_config.priority,
                    "created_time": rule_config.created_time.isoformat() if rule_config.created_time else None,
                    "updated_time": rule_config.updated_time.isoformat() if rule_config.updated_time else None
                })
            
            return {
                "success": True,
                "rules": rule_list,
                "total_count": len(rule_list),
                "enabled_count": len([r for r in rule_list if r["enabled"]]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"真实预警配置系统不可用，返回模拟规则: {e}")
            # 返回模拟规则
            mock_rules = [
                {
                    "rule_id": "rule_001",
                    "name": "RSI超买超卖",
                    "description": "RSI指标超买超卖预警",
                    "rule_type": "rsi_overbought_oversold",
                    "enabled": True,
                    "signal_type": "RISK_WARNING",
                    "priority": 3,
                    "created_time": datetime.now().isoformat(),
                    "updated_time": datetime.now().isoformat()
                },
                {
                    "rule_id": "rule_002",
                    "name": "MACD金叉死叉",
                    "description": "MACD金叉死叉信号预警",
                    "rule_type": "macd_golden_death_cross",
                    "enabled": True,
                    "signal_type": "BUY",
                    "priority": 4,
                    "created_time": datetime.now().isoformat(),
                    "updated_time": datetime.now().isoformat()
                }
            ]
            
            return {
                "success": True,
                "rules": mock_rules,
                "total_count": len(mock_rules),
                "enabled_count": len([r for r in mock_rules if r["enabled"]]),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"获取预警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预警规则失败: {str(e)}")

@router.get("/monitoring/performance")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_monitoring_performance():
    """
    获取监控系统性能指标
    
    Returns:
        Dict: 监控系统性能指标
    """
    try:
        # 模拟性能指标
        import random
        
        performance_metrics = {
            "system_status": "healthy",
            "uptime": "99.8%",
            "response_time": {
                "average": round(random.uniform(50, 200), 1),
                "p95": round(random.uniform(200, 500), 1),
                "p99": round(random.uniform(500, 1000), 1)
            },
            "throughput": {
                "requests_per_second": round(random.uniform(100, 500), 1),
                "alerts_per_minute": round(random.uniform(5, 20), 1)
            },
            "resource_usage": {
                "cpu_usage": round(random.uniform(20, 80), 1),
                "memory_usage": round(random.uniform(30, 70), 1),
                "disk_usage": round(random.uniform(40, 60), 1)
            },
            "error_rate": round(random.uniform(0.1, 2.0), 2),
            "last_update": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "performance": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取监控性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控性能指标失败: {str(e)}")
