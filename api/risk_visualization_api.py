#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风险指标可视化API接口

专业量化交易系统的风险指标可视化API模块，提供完整的RESTful API服务，
用于风险指标查询、实时监控数据获取和风险面板展示。

核心功能：
1. 风险指标查询API - 获取各类风险指标和历史数据
2. 实时监控API - 提供实时风险监控数据流
3. 风险面板API - 支持风险管理面板的数据需求
4. 预警管理API - 预警规则配置和预警事件管理
5. 报告生成API - 风险报告和分析报告生成

技术特性：
- FastAPI框架，高性能异步API
- 自动API文档生成（OpenAPI/Swagger）
- 数据验证和序列化（Pydantic）
- WebSocket实时数据推送
- 缓存优化和性能监控
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import io
import csv
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container

# 导入风控系统组件
from risk.unified_risk_management_system import (
    get_unified_risk_management_system,
    RiskMetricsSnapshot,
    RiskControlConfig,
    RiskControlMode
)
from risk.fast_pre_trade_risk_engine import (
    get_fast_pre_trade_risk_engine,
    RiskCheckInput,
    RiskCheckOutput,
    FastRiskCheckConfig
)
from risk.post_trade_analysis_system import (
    get_post_trade_analysis_system,
    PostTradeAnalysisResult,
    AnalysisPeriod
)
from risk.intelligent_risk_alert_system import (
from db.sql_manager import SQLManager, QueryType
    get_intelligent_alert_system,
    AlertEvent,
    AlertRule,
    AlertLevel,
    AlertType,
    NotificationChannel
)

logger = get_logger(__name__)

# 安全认证
security = HTTPBearer()

# 创建FastAPI应用
app = FastAPI(
    title="风险管理系统API",
    description="专业量化交易系统风险管理API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"WebSocket广播失败: {e}")
                self.disconnect(connection)

manager = get_connection_pool()

# Pydantic模型定义

class RiskMetricsResponse(BaseModel):
    """风险指标响应模型"""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    cash_balance: float
    portfolio_var_1d: float
    portfolio_var_5d: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    total_positions: int
    high_risk_positions: int
    concentration_risk: float
    sector_exposure: Dict[str, float]
    response_time_ms: float
    risk_score: float
    system_health: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PreTradeCheckRequest(BaseModel):
    """事前风控检查请求模型"""
    stock_code: str = Field(..., description="股票代码")
    trade_direction: str = Field(..., description="交易方向: BUY/SELL")
    quantity: int = Field(..., gt=0, description="交易数量")
    price: float = Field(..., gt=0, description="交易价格")
    order_type: str = Field(default="LIMIT", description="订单类型")
    account_id: Optional[str] = Field(None, description="账户ID")
    strategy_id: Optional[str] = Field(None, description="策略ID")


class PreTradeCheckResponse(BaseModel):
    """事前风控检查响应模型"""
    approved: bool
    risk_score: float
    status: str
    rule_results: Dict[str, Any]
    warnings: List[str]
    rejection_reasons: List[str]
    response_time_ms: float
    cache_hit: bool
    check_time: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PostTradeAnalysisRequest(BaseModel):
    """事后分析请求模型"""
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")
    analysis_period: str = Field(default="DAILY", description="分析周期")
    benchmark_code: Optional[str] = Field(None, description="基准代码")


class AlertRuleRequest(BaseModel):
    """预警规则请求模型"""
    rule_id: str
    rule_name: str
    alert_type: str
    alert_level: str
    enabled: bool = True
    condition_expression: str = ""
    threshold_value: float = 0.0
    duration_seconds: int = 0
    notification_channels: List[str] = []
    notification_template: str = ""
    suppression_duration: int = 3600
    max_alerts_per_hour: int = 10
    recipients: List[str] = []


class AlertEventResponse(BaseModel):
    """预警事件响应模型"""
    alert_id: str
    rule_id: str
    alert_type: str
    alert_level: str
    title: str
    message: str
    trigger_value: float
    threshold_value: float
    related_data: Dict[str, Any]
    trigger_time: datetime
    status: str
    notification_sent: List[str]
    notification_count: int

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    status: str
    mode: str
    start_time: datetime
    uptime_hours: float
    last_health_check: datetime
    avg_response_time_ms: float
    subsystems: Dict[str, bool]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# 依赖注入

async def get_risk_management_system():
    """获取风险管理系统"""
    return get_unified_risk_management_system()


async def get_pre_trade_engine():
    """获取事前风控引擎"""
    return get_fast_pre_trade_risk_engine()


async def get_post_trade_system():
    """获取事后分析系统"""
    return get_post_trade_analysis_system()


async def get_alert_system():
    """获取预警系统"""
    return get_intelligent_alert_system()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证API令牌（简化实现）"""
    # 实际应用中应该验证JWT或其他认证方式
    if credentials.credentials != "demo_token":
        raise HTTPException(status_code=401, detail="无效的认证令牌")
    return credentials.credentials


# API路由定义

@app.get("/", response_model=dict)
async def root():
    """API根路径"""
    return {
        "message": "风险管理系统API服务",
        "version": "1.0.0",
        "docs_url": "/docs",
        "status": "running"
    }


@app.get("/health", response_model=dict)
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "风险管理API"
    }


# 实时风险指标API
@app.get("/api/v1/risk/metrics/realtime", response_model=RiskMetricsResponse)
@performance_monitor
async def get_realtime_risk_metrics(
    risk_system=Depends(get_risk_management_system),
    token: str = Depends(verify_token)
):
    """获取实时风险指标"""
    try:
        metrics = risk_system.get_realtime_risk_metrics()

        return RiskMetricsResponse(
            timestamp=metrics.timestamp,
            portfolio_value=metrics.portfolio_value,
            total_pnl=metrics.total_pnl,
            unrealized_pnl=metrics.unrealized_pnl,
            realized_pnl=metrics.realized_pnl,
            cash_balance=metrics.cash_balance,
            portfolio_var_1d=metrics.portfolio_var_1d,
            portfolio_var_5d=metrics.portfolio_var_5d,
            max_drawdown=metrics.max_drawdown,
            current_drawdown=metrics.current_drawdown,
            volatility=metrics.volatility,
            sharpe_ratio=metrics.sharpe_ratio,
            total_positions=metrics.total_positions,
            high_risk_positions=metrics.high_risk_positions,
            concentration_risk=metrics.concentration_risk,
            sector_exposure=metrics.sector_exposure,
            response_time_ms=metrics.response_time_ms,
            risk_score=metrics.risk_score,
            system_health=metrics.system_health
        )

    except Exception as e:
        logger.error(f"获取实时风险指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取风险指标失败: {str(e)}")


@app.get("/api/v1/risk/metrics/history")
async def get_risk_metrics_history(
    hours: int = 24,
    risk_system=Depends(get_risk_management_system),
    token: str = Depends(verify_token)
):
    """获取历史风险指标"""
    try:
        # 获取历史数据（模拟实现）
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # 生成模拟历史数据
        timestamps = []
        risk_scores = []
        portfolio_values = []

        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time.isoformat())
            # 模拟数据
            risk_scores.append(np.random.normal(45.0, 10.0))
            portfolio_values.append(np.random.normal(1000000.0, 50000.0))
            current_time += timedelta(hours=1)

        return {
            "period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "hours": hours
            },
            "data": {
                "timestamps": timestamps,
                "risk_scores": risk_scores,
                "portfolio_values": portfolio_values
            }
        }

    except Exception as e:
        logger.error(f"获取历史风险指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史数据失败: {str(e)}")


# 事前风控API
@app.post("/api/v1/risk/precheck", response_model=PreTradeCheckResponse)
@performance_monitor
async def pre_trade_risk_check(
    request: PreTradeCheckRequest,
    pre_trade_engine=Depends(get_pre_trade_engine),
    token: str = Depends(verify_token)
):
    """事前风控检查"""
    try:
        # 构建风控检查输入
        check_input = RiskCheckInput(
            stock_code=request.stock_code,
            trade_direction=request.trade_direction,
            quantity=request.quantity,
            price=request.price,
            order_type=request.order_type,
            account_id=request.account_id,
            strategy_id=request.strategy_id
        )

        # 执行风控检查
        result = pre_trade_engine.check_risk_sync(check_input)

        return PreTradeCheckResponse(
            approved=result.approved,
            risk_score=result.risk_score,
            status=result.status.value,
            rule_results=result.rule_results,
            warnings=result.warnings,
            rejection_reasons=result.rejection_reasons,
            response_time_ms=result.response_time_ms,
            cache_hit=result.cache_hit,
            check_time=result.check_time
        )

    except Exception as e:
        logger.error(f"事前风控检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"风控检查失败: {str(e)}")


@app.get("/api/v1/risk/precheck/performance")
async def get_precheck_performance(
    pre_trade_engine=Depends(get_pre_trade_engine),
    token: str = Depends(verify_token)
):
    """获取事前风控性能统计"""
    try:
        stats = pre_trade_engine.get_performance_stats()
        return {
            "performance_metrics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")


# 事后分析API
@app.post("/api/v1/risk/analysis/post-trade")
async def post_trade_analysis(
    request: PostTradeAnalysisRequest,
    post_trade_system=Depends(get_post_trade_system),
    token: str = Depends(verify_token)
):
    """事后风险分析"""
    try:
        # 转换分析周期
        analysis_period = AnalysisPeriod(request.analysis_period)

        # 执行事后分析
        result = post_trade_system.analyze_post_trade_performance(
            start_date=request.start_date,
            end_date=request.end_date,
            analysis_period=analysis_period,
            benchmark_code=request.benchmark_code
        )

        # 转换为API响应格式
        return {
            "analysis_info": {
                "analysis_date": result.analysis_date.isoformat(),
                "period_start": result.period_start.isoformat(),
                "period_end": result.period_end.isoformat(),
                "analysis_period": result.analysis_period.value
            },
            "risk_metrics": {
                "var_1d": result.risk_metrics.var_1d,
                "var_5d": result.risk_metrics.var_5d,
                "max_drawdown": result.risk_metrics.max_drawdown,
                "current_drawdown": result.risk_metrics.current_drawdown,
                "volatility_annualized": result.risk_metrics.volatility_annualized,
                "systematic_risk": result.risk_metrics.systematic_risk,
                "specific_risk": result.risk_metrics.specific_risk
            },
            "performance_metrics": {
                "total_return": result.performance_metrics.total_return,
                "annualized_return": result.performance_metrics.annualized_return,
                "sharpe_ratio": result.performance_metrics.sharpe_ratio,
                "sortino_ratio": result.performance_metrics.sortino_ratio,
                "calmar_ratio": result.performance_metrics.calmar_ratio,
                "win_rate": result.performance_metrics.win_rate,
                "profit_loss_ratio": result.performance_metrics.profit_loss_ratio,
                "alpha": result.performance_metrics.alpha,
                "beta": result.performance_metrics.beta
            },
            "attribution_analysis": {
                "stock_selection_effect": result.attribution_analysis.stock_selection_effect,
                "sector_allocation_effect": result.attribution_analysis.sector_allocation_effect,
                "timing_effect": result.attribution_analysis.timing_effect,
                "market_risk_contribution": result.attribution_analysis.market_risk_contribution,
                "sector_risk_contribution": result.attribution_analysis.sector_risk_contribution,
                "stock_risk_contribution": result.attribution_analysis.stock_risk_contribution
            },
            "anomalies_detected": result.anomalies_detected,
            "optimization_suggestions": result.optimization_suggestions,
            "risk_warnings": result.risk_warnings
        }

    except Exception as e:
        logger.error(f"事后分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"事后分析失败: {str(e)}")


@app.get("/api/v1/risk/analysis/report/{analysis_id}")
async def get_analysis_report(
    analysis_id: str,
    format: str = "json",
    post_trade_system=Depends(get_post_trade_system),
    token: str = Depends(verify_token)
):
    """获取分析报告"""
    # 简化实现，实际应该从数据库获取
    if format == "csv":
        # 返回CSV格式报告
        def generate_csv():
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["指标", "数值"])
            writer.writerow(["总收益率", "8.5%"])
            writer.writerow(["夏普比率", "1.25"])
            writer.writerow(["最大回撤", "5.2%"])
            output.seek(0)
            return output.getvalue()

        return StreamingResponse(
            io.StringIO(generate_csv()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=risk_analysis_{analysis_id}.csv"}
        )
    else:
        return {"report_id": analysis_id, "format": format, "status": "generated"}


# 预警系统API
@app.get("/api/v1/alerts/active", response_model=List[AlertEventResponse])
async def get_active_alerts(
    alert_system=Depends(get_alert_system),
    token: str = Depends(verify_token)
):
    """获取活动预警"""
    try:
        active_alerts = alert_system.get_active_alerts()

        return [
            AlertEventResponse(
                alert_id=alert.alert_id,
                rule_id=alert.rule_id,
                alert_type=alert.alert_type.value,
                alert_level=alert.alert_level.value,
                title=alert.title,
                message=alert.message,
                trigger_value=alert.trigger_value,
                threshold_value=alert.threshold_value,
                related_data=alert.related_data,
                trigger_time=alert.trigger_time,
                status=alert.status.value,
                notification_sent=[ch.value for ch in alert.notification_sent],
                notification_count=alert.notification_count
            )
            for alert in active_alerts
        ]

    except Exception as e:
        logger.error(f"获取活动预警失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取活动预警失败: {str(e)}")


@app.post("/api/v1/alerts/rules")
async def create_alert_rule(
    request: AlertRuleRequest,
    alert_system=Depends(get_alert_system),
    token: str = Depends(verify_token)
):
    """创建预警规则"""
    try:
        # 构建预警规则
        alert_rule = AlertRule(
            rule_id=request.rule_id,
            rule_name=request.rule_name,
            alert_type=AlertType(request.alert_type),
            alert_level=AlertLevel(request.alert_level),
            enabled=request.enabled,
            condition_expression=request.condition_expression,
            threshold_value=request.threshold_value,
            duration_seconds=request.duration_seconds,
            notification_channels=[NotificationChannel(ch) for ch in request.notification_channels],
            notification_template=request.notification_template,
            suppression_duration=request.suppression_duration,
            max_alerts_per_hour=request.max_alerts_per_hour,
            recipients=request.recipients
        )

        # 添加规则
        alert_system.add_alert_rule(alert_rule)

        return {
            "status": "success",
            "message": f"预警规则 {request.rule_id} 创建成功",
            "rule_id": request.rule_id
        }

    except Exception as e:
        logger.error(f"创建预警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建预警规则失败: {str(e)}")


@app.put("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str,
    alert_system=Depends(get_alert_system),
    token: str = Depends(verify_token)
):
    """确认预警"""
    try:
        success = alert_system.acknowledge_alert(alert_id, acknowledged_by)

        if success:
            return {
                "status": "success",
                "message": f"预警 {alert_id} 已确认",
                "acknowledged_by": acknowledged_by,
                "acknowledged_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="预警不存在或已处理")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"确认预警失败: {e}")
        raise HTTPException(status_code=500, detail=f"确认预警失败: {str(e)}")


@app.get("/api/v1/alerts/statistics")
async def get_alert_statistics(
    alert_system=Depends(get_alert_system),
    token: str = Depends(verify_token)
):
    """获取预警统计"""
    try:
        stats = alert_system.get_alert_statistics()
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"获取预警统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取预警统计失败: {str(e)}")


# 系统状态API
@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    risk_system=Depends(get_risk_management_system),
    token: str = Depends(verify_token)
):
    """获取系统状态"""
    try:
        status = risk_system.get_system_status()

        return SystemStatusResponse(
            status=status['status'],
            mode=status['mode'],
            start_time=datetime.fromisoformat(status['start_time']),
            uptime_hours=status['uptime_hours'],
            last_health_check=datetime.fromisoformat(status['last_health_check']),
            avg_response_time_ms=status['avg_response_time_ms'],
            subsystems=status['subsystems']
        )

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


# WebSocket实时数据推送
@app.websocket("/ws/risk/realtime")
async def websocket_realtime_risk(websocket: WebSocket):
    """WebSocket实时风险数据推送"""
    await manager.connect(websocket)

    try:
        # 获取风险系统
        risk_system = get_unified_risk_management_system()

        while True:
            try:
                # 获取实时风险指标
                metrics = risk_system.get_realtime_risk_metrics()

                # 构建消息
                message = {
                    "type": "risk_metrics",
                    "timestamp": metrics.timestamp.isoformat(),
                    "data": {
                        "portfolio_value": metrics.portfolio_value,
                        "risk_score": metrics.risk_score,
                        "max_drawdown": metrics.max_drawdown,
                        "current_drawdown": metrics.current_drawdown,
                        "volatility": metrics.volatility,
                        "system_health": metrics.system_health,
                        "response_time_ms": metrics.response_time_ms
                    }
                }

                # 发送消息
                await websocket.send_text(json.dumps(message, ensure_ascii=False))

                # 等待下次推送
                await asyncio.sleep(5)  # 每5秒推送一次

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket推送数据失败: {e}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket连接异常: {e}")
    finally:
        manager.disconnect(websocket)


# 批量操作API
@app.post("/api/v1/risk/batch/precheck")
async def batch_pre_trade_check(
    requests: List[PreTradeCheckRequest],
    pre_trade_engine=Depends(get_pre_trade_engine),
    token: str = Depends(verify_token)
):
    """批量事前风控检查"""
    try:
        results = []

        for request in requests:
            check_input = RiskCheckInput(
                stock_code=request.stock_code,
                trade_direction=request.trade_direction,
                quantity=request.quantity,
                price=request.price,
                order_type=request.order_type,
                account_id=request.account_id,
                strategy_id=request.strategy_id
            )

            result = pre_trade_engine.check_risk_sync(check_input)

            results.append({
                "stock_code": request.stock_code,
                "approved": result.approved,
                "risk_score": result.risk_score,
                "response_time_ms": result.response_time_ms
            })

        return {
            "batch_size": len(requests),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"批量风控检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量风控检查失败: {str(e)}")


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"API异常: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "内部服务器错误",
            "timestamp": datetime.now().isoformat(),
            "status_code": 500
        }
    )


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("风险管理API服务启动")

    # 启动风险管理系统
    risk_system = get_unified_risk_management_system()
    risk_system.start_realtime_monitoring()

    # 启动预警系统
    alert_system = get_intelligent_alert_system()
    await alert_system.start_notification_processing()

    logger.info("所有系统组件已启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("风险管理API服务关闭")

    # 关闭系统组件
    try:
        risk_system = get_unified_risk_management_system()
        risk_system.stop_realtime_monitoring()

        alert_system = get_intelligent_alert_system()
        await alert_system.stop_notification_processing()

    except Exception as e:
        logger.error(f"关闭系统组件异常: {e}")

    logger.info("风险管理API服务已关闭")


if __name__ == "__main__":
    import uvicorn

    # 运行API服务
    uvicorn.run(
        "risk_visualization_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )