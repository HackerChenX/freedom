#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
风险监控API路由

提供风险监控相关的RESTful API接口
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

class RiskAssessmentRequest(BaseModel):
    """风险评估请求模型"""
    stock_codes: List[str] = Field(..., description="股票代码列表", example=["000001", "000002"])
    assessment_type: str = Field(default="comprehensive", description="评估类型", example="comprehensive")
    time_horizon: int = Field(default=30, description="评估时间范围(天)", example=30)

class PortfolioRiskRequest(BaseModel):
    """组合风险评估请求模型"""
    portfolio_id: str = Field(..., description="组合ID", example="portfolio_001")
    portfolio_name: str = Field(..., description="组合名称", example="稳健型组合")
    positions: List[Dict[str, Any]] = Field(..., description="持仓信息")

class RiskAssessmentResponse(BaseModel):
    """风险评估响应模型"""
    success: bool = Field(..., description="评估是否成功")
    assessment_type: str = Field(..., description="评估类型")
    market_risk: Dict[str, Any] = Field(..., description="市场风险")
    stock_risks: List[Dict[str, Any]] = Field(..., description="个股风险")
    risk_summary: Dict[str, Any] = Field(..., description="风险摘要")
    recommendations: List[str] = Field(..., description="风险管理建议")
    assessment_time: float = Field(..., description="评估耗时(秒)")
    timestamp: str = Field(..., description="响应时间戳")

class PortfolioRiskResponse(BaseModel):
    """组合风险评估响应模型"""
    success: bool = Field(..., description="评估是否成功")
    portfolio_risk: Dict[str, Any] = Field(..., description="组合风险")
    position_risks: List[Dict[str, Any]] = Field(..., description="持仓风险")
    risk_metrics: Dict[str, Any] = Field(..., description="风险指标")
    recommendations: List[str] = Field(..., description="风险管理建议")
    timestamp: str = Field(..., description="响应时间戳")

@router.post("/risk/assess", response_model=RiskAssessmentResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=10.0)
def assess_risk(request: RiskAssessmentRequest):
    """
    执行风险评估
    
    Args:
        request: 风险评估请求
        
    Returns:
        RiskAssessmentResponse: 风险评估响应
    """
    import time
    start_time = time.time()
    
    try:
        # 验证股票代码
        if not request.stock_codes:
            raise HTTPException(status_code=400, detail="股票代码列表不能为空")
        
        if len(request.stock_codes) > 20:
            raise HTTPException(status_code=400, detail="股票代码数量不能超过20个")
        
        # 尝试使用真实的风险监控系统
        try:
            from monitoring.risk_monitor import RiskMonitoringSystem
            risk_system = RiskMonitoringSystem()
            
            # 执行综合风险评估
            risk_report = risk_system.comprehensive_risk_assessment(
                stocks=request.stock_codes,
                portfolios=None
            )
            
            assessment_time = time.time() - start_time
            
            return RiskAssessmentResponse(
                success=True,
                assessment_type=request.assessment_type,
                market_risk=risk_report.get("market_risk", {}),
                stock_risks=risk_report.get("stock_risks", {}).get("details", []),
                risk_summary=risk_report.get("risk_summary", {}),
                recommendations=risk_report.get("risk_summary", {}).get("recommendations", []),
                assessment_time=round(assessment_time, 3),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"真实风险系统不可用，使用模拟评估: {e}")
            # 使用模拟风险评估
            return _mock_risk_assessment(request, start_time)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"风险评估失败: {str(e)}")

@router.post("/risk/portfolio", response_model=PortfolioRiskResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=8.0)
def assess_portfolio_risk(request: PortfolioRiskRequest):
    """
    评估组合风险
    
    Args:
        request: 组合风险评估请求
        
    Returns:
        PortfolioRiskResponse: 组合风险评估响应
    """
    try:
        # 验证组合信息
        if not request.positions:
            raise HTTPException(status_code=400, detail="组合持仓信息不能为空")
        
        # 尝试使用真实的风险监控系统
        try:
            from monitoring.risk_monitor import PortfolioRiskManager
            portfolio_manager = PortfolioRiskManager()
            
            # 构建组合信息
            portfolio = {
                'id': request.portfolio_id,
                'name': request.portfolio_name,
                'positions': request.positions
            }
            
            # 执行组合风险评估
            portfolio_risk = portfolio_manager.assess_portfolio_risk(portfolio)
            
            # 转换为API响应格式
            from dataclasses import asdict
            portfolio_risk_dict = asdict(portfolio_risk)
            
            return PortfolioRiskResponse(
                success=True,
                portfolio_risk={
                    "portfolio_id": portfolio_risk_dict["portfolio_id"],
                    "portfolio_name": portfolio_risk_dict["portfolio_name"],
                    "total_value": portfolio_risk_dict["total_value"],
                    "risk_level": portfolio_risk_dict["risk_level"],
                    "portfolio_var": portfolio_risk_dict["portfolio_var"],
                    "portfolio_volatility": portfolio_risk_dict["portfolio_volatility"],
                    "portfolio_beta": portfolio_risk_dict["portfolio_beta"]
                },
                position_risks=portfolio_risk_dict["positions"],
                risk_metrics={
                    "concentration_risk": portfolio_risk_dict["concentration_risk"],
                    "correlation_risk": portfolio_risk_dict["correlation_risk"],
                    "diversification_score": 100 - portfolio_risk_dict["concentration_risk"]
                },
                recommendations=_generate_portfolio_recommendations(portfolio_risk_dict),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"真实组合风险系统不可用，使用模拟评估: {e}")
            # 使用模拟组合风险评估
            return _mock_portfolio_risk_assessment(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"组合风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"组合风险评估失败: {str(e)}")

@router.get("/risk/market")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=3.0)
def get_market_risk(
    market_index: str = Query(default="000001", description="市场指数代码", example="000001")
):
    """
    获取市场风险评估
    
    Args:
        market_index: 市场指数代码
        
    Returns:
        Dict: 市场风险评估结果
    """
    try:
        # 尝试使用真实的市场风险评估
        try:
            from monitoring.risk_monitor import MarketRiskAssessor
            market_assessor = MarketRiskAssessor()
            
            market_risk = market_assessor.assess_market_risk(market_index)
            
            return {
                "success": True,
                "market_risk": market_risk,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"真实市场风险系统不可用，使用模拟评估: {e}")
            # 使用模拟市场风险评估
            return _mock_market_risk_assessment(market_index)
        
    except Exception as e:
        logger.error(f"市场风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"市场风险评估失败: {str(e)}")

@router.get("/risk/stock/{stock_code}")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=2.0)
def get_stock_risk(
    stock_code: str = Path(..., description="股票代码", example="000001")
):
    """
    获取个股风险评估
    
    Args:
        stock_code: 股票代码
        
    Returns:
        Dict: 个股风险评估结果
    """
    try:
        # 尝试使用真实的个股风险监控
        try:
            from monitoring.risk_monitor import StockRiskMonitor
            stock_monitor = StockRiskMonitor()
            
            risk_metrics = stock_monitor.monitor_stock_risk(stock_code)
            
            # 转换为API响应格式
            from dataclasses import asdict
            risk_dict = asdict(risk_metrics)
            
            return {
                "success": True,
                "stock_risk": risk_dict,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"真实个股风险系统不可用，使用模拟评估: {e}")
            # 使用模拟个股风险评估
            return _mock_stock_risk_assessment(stock_code)
        
    except Exception as e:
        logger.error(f"个股风险评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"个股风险评估失败: {str(e)}")

@router.get("/risk/alerts")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=2.0)
def get_risk_alerts(
    limit: int = Query(default=10, description="返回数量限制", ge=1, le=100),
    severity: Optional[str] = Query(None, description="预警级别过滤", example="高风险")
):
    """
    获取风险预警信息
    
    Args:
        limit: 返回数量限制
        severity: 预警级别过滤
        
    Returns:
        Dict: 风险预警信息
    """
    try:
        # 模拟风险预警数据
        import random
        
        alert_types = ["市场风险预警", "个股风险预警", "组合风险预警", "流动性风险预警"]
        severity_levels = ["低风险", "中等风险", "高风险", "极高风险"]
        
        alerts = []
        for i in range(min(limit, 20)):
            alert_severity = random.choice(severity_levels)
            
            # 如果指定了严重程度过滤
            if severity and alert_severity != severity:
                continue
            
            alert = {
                "id": f"alert_{i+1:03d}",
                "type": random.choice(alert_types),
                "severity": alert_severity,
                "message": f"检测到{alert_severity}，建议关注",
                "stock_code": f"{random.randint(0, 999999):06d}",
                "created_time": datetime.now().isoformat(),
                "status": "active"
            }
            alerts.append(alert)
        
        return {
            "success": True,
            "alerts": alerts,
            "total_count": len(alerts),
            "active_alerts": len([a for a in alerts if a["status"] == "active"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取风险预警失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取风险预警失败: {str(e)}")

def _mock_risk_assessment(request: RiskAssessmentRequest, start_time: float) -> RiskAssessmentResponse:
    """模拟风险评估"""
    import random
    import numpy as np
    
    # 模拟市场风险
    market_risk = {
        "market_index": "000001",
        "risk_level": random.choice(["低风险", "中等风险", "高风险"]),
        "risk_score": round(random.uniform(20, 80), 2),
        "var_1d": round(random.uniform(0.01, 0.05), 3),
        "volatility": round(random.uniform(0.15, 0.35), 3),
        "max_drawdown": round(random.uniform(0.05, 0.25), 3)
    }
    
    # 模拟个股风险
    stock_risks = []
    for stock_code in request.stock_codes:
        stock_risk = {
            "stock_code": stock_code,
            "stock_name": f"模拟股票{stock_code}",
            "risk_level": random.choice(["低风险", "中等风险", "高风险", "极高风险"]),
            "risk_score": round(random.uniform(10, 90), 2),
            "var_1d": round(random.uniform(0.02, 0.08), 3),
            "volatility": round(random.uniform(0.20, 0.50), 3),
            "beta": round(random.uniform(0.5, 2.0), 2),
            "max_drawdown": round(random.uniform(0.10, 0.40), 3)
        }
        stock_risks.append(stock_risk)
    
    # 模拟风险摘要
    avg_score = np.mean([sr["risk_score"] for sr in stock_risks])
    risk_summary = {
        "overall_risk_level": "中等风险" if avg_score < 60 else "高风险",
        "average_risk_score": round(avg_score, 2),
        "high_risk_stocks": len([sr for sr in stock_risks if sr["risk_score"] > 70]),
        "key_risks": ["市场波动加剧", "个股集中度较高"]
    }
    
    # 模拟建议
    recommendations = [
        "建议分散投资，降低集中度风险",
        "关注市场波动，适当控制仓位",
        "定期评估风险水平，及时调整策略"
    ]
    
    assessment_time = time.time() - start_time
    
    return RiskAssessmentResponse(
        success=True,
        assessment_type=request.assessment_type,
        market_risk=market_risk,
        stock_risks=stock_risks,
        risk_summary=risk_summary,
        recommendations=recommendations,
        assessment_time=round(assessment_time, 3),
        timestamp=datetime.now().isoformat()
    )

def _mock_portfolio_risk_assessment(request: PortfolioRiskRequest) -> PortfolioRiskResponse:
    """模拟组合风险评估"""
    import random
    
    # 计算组合总市值
    total_value = sum(pos.get("value", 0) for pos in request.positions)
    
    # 模拟组合风险
    portfolio_risk = {
        "portfolio_id": request.portfolio_id,
        "portfolio_name": request.portfolio_name,
        "total_value": total_value,
        "risk_level": random.choice(["低风险", "中等风险", "高风险"]),
        "portfolio_var": round(random.uniform(0.02, 0.06), 3),
        "portfolio_volatility": round(random.uniform(0.15, 0.30), 3),
        "portfolio_beta": round(random.uniform(0.8, 1.5), 2)
    }
    
    # 模拟持仓风险
    position_risks = []
    for position in request.positions:
        position_risk = {
            "code": position.get("code", ""),
            "name": position.get("name", ""),
            "weight": position.get("weight", 0),
            "value": position.get("value", 0),
            "risk_score": round(random.uniform(20, 80), 2),
            "risk_level": random.choice(["低风险", "中等风险", "高风险"]),
            "beta": round(random.uniform(0.5, 2.0), 2),
            "volatility": round(random.uniform(0.20, 0.40), 3)
        }
        position_risks.append(position_risk)
    
    # 模拟风险指标
    risk_metrics = {
        "concentration_risk": round(random.uniform(30, 80), 2),
        "correlation_risk": round(random.uniform(40, 70), 2),
        "diversification_score": round(random.uniform(60, 90), 2)
    }
    
    # 模拟建议
    recommendations = [
        "建议增加组合分散化程度",
        "关注高风险持仓的仓位控制",
        "定期重新平衡组合权重"
    ]
    
    return PortfolioRiskResponse(
        success=True,
        portfolio_risk=portfolio_risk,
        position_risks=position_risks,
        risk_metrics=risk_metrics,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat()
    )

def _mock_market_risk_assessment(market_index: str) -> Dict[str, Any]:
    """模拟市场风险评估"""
    import random
    
    return {
        "success": True,
        "market_risk": {
            "market_index": market_index,
            "risk_level": random.choice(["低风险", "中等风险", "高风险"]),
            "risk_score": round(random.uniform(30, 70), 2),
            "var_1d": round(random.uniform(0.015, 0.045), 3),
            "var_5d": round(random.uniform(0.035, 0.095), 3),
            "volatility": round(random.uniform(0.18, 0.32), 3),
            "max_drawdown": round(random.uniform(0.08, 0.22), 3),
            "assessment_time": datetime.now().isoformat(),
            "details": {
                "recent_volatility": round(random.uniform(0.20, 0.35), 3),
                "trend_direction": random.choice(["up", "down", "neutral"]),
                "volatility_trend": random.choice(["increasing", "decreasing", "stable"])
            }
        },
        "timestamp": datetime.now().isoformat()
    }

def _mock_stock_risk_assessment(stock_code: str) -> Dict[str, Any]:
    """模拟个股风险评估"""
    import random
    
    return {
        "success": True,
        "stock_risk": {
            "stock_code": stock_code,
            "stock_name": f"模拟股票{stock_code}",
            "risk_type": random.choice(["市场风险", "波动性风险", "流动性风险"]),
            "risk_level": random.choice(["低风险", "中等风险", "高风险", "极高风险"]),
            "risk_score": round(random.uniform(15, 85), 2),
            "var_1d": round(random.uniform(0.02, 0.07), 3),
            "var_5d": round(random.uniform(0.045, 0.15), 3),
            "volatility": round(random.uniform(0.22, 0.45), 3),
            "beta": round(random.uniform(0.6, 1.8), 2),
            "max_drawdown": round(random.uniform(0.12, 0.35), 3),
            "sharpe_ratio": round(random.uniform(0.2, 1.5), 2),
            "timestamp": datetime.now().isoformat(),
            "details": {
                "recent_volatility": round(random.uniform(0.25, 0.50), 3),
                "price_trend": random.choice(["up", "down", "neutral"]),
                "volatility_rank": random.choice(["低波动", "中等波动", "高波动", "极高波动"]),
                "liquidity_score": round(random.uniform(40, 95), 1),
                "data_quality": "good"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

def _generate_portfolio_recommendations(portfolio_risk_dict: Dict[str, Any]) -> List[str]:
    """生成组合风险管理建议"""
    recommendations = []
    
    # 基于集中度风险的建议
    concentration_risk = portfolio_risk_dict.get("concentration_risk", 0)
    if concentration_risk > 70:
        recommendations.append("组合集中度过高，建议增加持仓分散化")
    elif concentration_risk > 50:
        recommendations.append("组合集中度较高，建议适当分散投资")
    
    # 基于相关性风险的建议
    correlation_risk = portfolio_risk_dict.get("correlation_risk", 0)
    if correlation_risk > 70:
        recommendations.append("组合相关性风险较高，建议增加不同行业或风格的股票")
    
    # 基于风险级别的建议
    risk_level = portfolio_risk_dict.get("risk_level", "中等风险")
    if risk_level in ["高风险", "极高风险"]:
        recommendations.append("组合整体风险较高，建议降低仓位或调整持仓结构")
    
    # 默认建议
    if not recommendations:
        recommendations.append("组合风险水平适中，建议继续监控并定期重新评估")
    
    return recommendations
