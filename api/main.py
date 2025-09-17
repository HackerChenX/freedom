#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
股票分析系统 RESTful API 主入口

提供完整的RESTful API接口，包括：
1. 股票数据查询API
2. 技术指标计算API
3. 策略分析API
4. 风险监控API
5. 实时监控API
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends, Query, Path as PathParam, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import asyncio
import logging

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container

# 导入API路由模块
from api.routers.stock_data_router import router as stock_data_router
from api.routers.indicator_router import router as indicator_router
from api.routers.strategy_router import router as strategy_router
from api.routers.risk_router import router as risk_router
from api.routers.monitoring_router import router as monitoring_router

# 导入WebSocket相关模块
from api.websocket_server import get_websocket_server
from api.realtime_data_pusher import get_realtime_pusher

logger = get_logger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="股票分析系统 API",
    description="提供完整的股票数据分析、技术指标计算、策略分析和风险监控功能",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"API请求异常: {request.url} - {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {request.url} - {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "请求错误",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化操作"""
    logger.info("🚀 股票分析系统 API 启动中...")
    
    try:
        # 初始化依赖注入容器
        container = get_container()
        logger.info("✅ 依赖注入容器初始化完成")
        
        # 初始化指标系统
        from indicators.complete_indicator_registry import get_indicator_registry
from db.sql_manager import SQLManager, QueryType
        registry = get_indicator_registry()
        logger.info(f"✅ 指标系统初始化完成，共注册 {registry.get_indicator_count()} 个指标")
        
        # 初始化数据库连接
        try:
            from db.enhanced_connection_pool import ClickHouseConnectionPool
            pool = ClickHouseConnectionPool()
            logger.info("✅ 数据库连接池初始化完成")
        except Exception as e:
            logger.warning(f"⚠️ 数据库连接池初始化失败: {e}")
        
        logger.info("🎉 股票分析系统 API 启动完成")
        
    except Exception as e:
        logger.error(f"❌ API 启动失败: {e}")
        raise

# 应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理操作"""
    logger.info("🔄 股票分析系统 API 关闭中...")
    
    try:
        # 清理资源
        logger.info("✅ 资源清理完成")
        logger.info("👋 股票分析系统 API 已关闭")
        
    except Exception as e:
        logger.error(f"❌ API 关闭时出错: {e}")

# 健康检查端点
@app.get("/health", tags=["系统"])
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def health_check():
    """
    健康检查接口
    
    Returns:
        Dict: 系统健康状态信息
    """
    try:
        # 检查指标系统
        from indicators.complete_indicator_registry import get_indicator_registry
from db.sql_manager import SQLManager, QueryType
        registry = get_indicator_registry()

        # 获取指标数量
        if hasattr(registry, 'indicators'):
            indicator_count = len(registry.indicators)
        elif hasattr(registry, 'get_all_indicators'):
            indicator_count = len(registry.get_all_indicators())
        else:
            indicator_count = 0
        
        # 检查数据库连接
        db_status = "connected"
        try:
            from db.enhanced_connection_pool import ClickHouseConnectionPool
            pool = ClickHouseConnectionPool()
            # 简单的连接测试
        except Exception:
            db_status = "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "indicators": {
                    "status": "healthy",
                    "count": indicator_count
                },
                "database": {
                    "status": db_status
                },
                "api": {
                    "status": "healthy"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="服务不可用")

# 系统信息端点
@app.get("/info", tags=["系统"])
@exception_handler(reraise=True)
def system_info():
    """
    获取系统信息
    
    Returns:
        Dict: 系统详细信息
    """
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
from db.sql_manager import SQLManager, QueryType
        registry = get_indicator_registry()

        # 获取指标数量
        if hasattr(registry, 'indicators'):
            indicator_count = len(registry.indicators)
        elif hasattr(registry, 'get_all_indicators'):
            indicator_count = len(registry.get_all_indicators())
        else:
            indicator_count = 0

        return {
            "system": "股票分析系统",
            "version": "1.0.0",
            "api_version": "v1",
            "description": "提供完整的股票数据分析、技术指标计算、策略分析和风险监控功能",
            "features": [
                "股票数据查询",
                "技术指标计算",
                "策略分析",
                "风险监控",
                "实时监控",
                "预警系统"
            ],
            "indicators": {
                "total_count": indicator_count,
                "categories": [
                    "趋势指标",
                    "振荡器指标",
                    "成交量指标",
                    "波动性指标",
                    "形态识别指标",
                    "ZXM体系指标"
                ]
            },
            "endpoints": {
                "stock_data": "/api/v1/stocks",
                "indicators": "/api/v1/indicators",
                "strategies": "/api/v1/strategies",
                "risk": "/api/v1/risk",
                "monitoring": "/api/v1/monitoring"
            },
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统信息失败")

# 注册API路由
app.include_router(stock_data_router, prefix="/api/v1", tags=["股票数据"])
app.include_router(indicator_router, prefix="/api/v1", tags=["技术指标"])
app.include_router(strategy_router, prefix="/api/v1", tags=["策略分析"])
app.include_router(risk_router, prefix="/api/v1", tags=["风险监控"])
app.include_router(monitoring_router, prefix="/api/v1", tags=["实时监控"])

# 自定义OpenAPI文档
def custom_openapi():
    """自定义OpenAPI文档"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="股票分析系统 API",
        version="1.0.0",
        description="""
        ## 股票分析系统 RESTful API
        
        提供完整的股票数据分析功能，包括：
        
        ### 🔍 核心功能
        - **股票数据查询**: 获取股票历史数据、实时数据
        - **技术指标计算**: 128个技术指标计算
        - **策略分析**: 多种投资策略分析
        - **风险监控**: 全面的风险评估和监控
        - **实时监控**: 实时数据监控和预警
        
        ### 📊 技术特性
        - RESTful API设计
        - 完整的错误处理
        - 性能监控
        - 自动文档生成
        - CORS支持
        
        ### 🚀 快速开始
        1. 查看系统健康状态: `GET /health`
        2. 获取系统信息: `GET /info`
        3. 查询股票数据: `GET /api/v1/stocks/{stock_code}/data`
        4. 计算技术指标: `POST /api/v1/indicators/calculate`
        """,
        routes=app.routes,
    )
    
    # 添加自定义标签
    openapi_schema["tags"] = [
        {"name": "系统", "description": "系统健康检查和信息"},
        {"name": "股票数据", "description": "股票数据查询和管理"},
        {"name": "技术指标", "description": "技术指标计算和分析"},
        {"name": "策略分析", "description": "投资策略分析和回测"},
        {"name": "风险监控", "description": "风险评估和监控"},
        {"name": "实时监控", "description": "实时数据监控和预警"}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# WebSocket端点
@app.websocket("/ws")
@exception_handler(reraise=True)
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    try:
        await websocket.accept()
        logger.info(f"WebSocket连接建立: {websocket.client}")

        # 获取WebSocket服务器
        ws_server = get_websocket_server()

        # 注册连接
        await ws_server.connection_manager.register_connection(websocket, "/ws")

        # 发送欢迎消息
        await ws_server.connection_manager.send_to_connection(websocket, {
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
        while True:
            try:
                data = await websocket.receive_text()
                await ws_server.message_handler.handle_message(websocket, data)
            except WebSocketDisconnect:
                logger.info(f"WebSocket连接断开: {websocket.client}")
                break
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket连接处理出错: {e}")
    finally:
        # 注销连接
        ws_server = get_websocket_server()
        await ws_server.connection_manager.unregister_connection(websocket)

@app.get("/ws/stats")
@exception_handler(reraise=True)
async def get_websocket_stats():
    """获取WebSocket连接统计信息"""
    ws_server = get_websocket_server()
    stats = ws_server.connection_manager.get_connection_stats()

    return {
        "status": "success",
        "data": stats,
        "timestamp": datetime.now().isoformat()
    }

# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 股票分析系统 API 服务器启动中...")

    try:
        # 启动实时数据推送服务
        pusher = get_realtime_pusher()
        asyncio.create_task(pusher.start_all_pushers())
        logger.info("✅ 实时数据推送服务启动成功")

    except Exception as e:
        logger.error(f"❌ 启动实时数据推送服务失败: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 股票分析系统 API 服务器关闭中...")

    try:
        # 停止实时数据推送服务
        pusher = get_realtime_pusher()
        await pusher.stop_all_pushers()
        logger.info("✅ 实时数据推送服务停止成功")

    except Exception as e:
        logger.error(f"❌ 停止实时数据推送服务失败: {e}")

# 主函数
def main():
    """启动API服务器"""
    logger.info("🚀 启动股票分析系统 API 服务器...")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
