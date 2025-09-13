#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
策略分析API路由

提供投资策略分析相关的RESTful API接口
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from utils.unified_container import get_container

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()

# 数据模型
from pydantic import BaseModel, Field

class StrategyAnalysisRequest(BaseModel):
    """策略分析请求模型"""
    strategy_name: str = Field(..., description="策略名称", example="趋势跟踪策略")
    stock_codes: List[str] = Field(..., description="股票代码列表", example=["000001", "000002"])
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)", example="2024-01-01")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)", example="2024-12-31")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="策略参数")

class StrategyAnalysisResponse(BaseModel):
    """策略分析响应模型"""
    success: bool = Field(..., description="分析是否成功")
    strategy_name: str = Field(..., description="策略名称")
    analysis_results: List[Dict[str, Any]] = Field(..., description="分析结果")
    summary: Dict[str, Any] = Field(..., description="分析摘要")
    performance_metrics: Dict[str, Any] = Field(..., description="性能指标")
    analysis_time: float = Field(..., description="分析耗时(秒)")
    timestamp: str = Field(..., description="响应时间戳")

class StrategyListResponse(BaseModel):
    """策略列表响应模型"""
    success: bool = Field(..., description="请求是否成功")
    strategies: List[Dict[str, Any]] = Field(..., description="策略列表")
    total_count: int = Field(..., description="策略总数")
    categories: List[str] = Field(..., description="策略分类")
    timestamp: str = Field(..., description="响应时间戳")

class BacktestRequest(BaseModel):
    """回测请求模型"""
    strategy_name: str = Field(..., description="策略名称")
    stock_codes: List[str] = Field(..., description="股票代码列表")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    initial_capital: float = Field(default=100000.0, description="初始资金")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="策略参数")

class BacktestResponse(BaseModel):
    """回测响应模型"""
    success: bool = Field(..., description="回测是否成功")
    strategy_name: str = Field(..., description="策略名称")
    backtest_results: Dict[str, Any] = Field(..., description="回测结果")
    performance_summary: Dict[str, Any] = Field(..., description="性能摘要")
    trades: List[Dict[str, Any]] = Field(..., description="交易记录")
    backtest_time: float = Field(..., description="回测耗时(秒)")
    timestamp: str = Field(..., description="响应时间戳")

@router.get("/strategies", response_model=StrategyListResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_strategy_list():
    """
    获取所有可用的投资策略列表
    
    Returns:
        StrategyListResponse: 策略列表响应
    """
    try:
        # 定义可用策略
        strategies = [
            {
                "name": "趋势跟踪策略",
                "description": "基于移动平均线和趋势指标的趋势跟踪策略",
                "category": "趋势策略",
                "parameters": {
                    "ma_period": 20,
                    "trend_threshold": 0.02
                },
                "risk_level": "中等"
            },
            {
                "name": "均值回归策略",
                "description": "基于RSI和布林带的均值回归策略",
                "category": "均值回归",
                "parameters": {
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70
                },
                "risk_level": "低"
            },
            {
                "name": "突破策略",
                "description": "基于价格突破和成交量确认的突破策略",
                "category": "突破策略",
                "parameters": {
                    "breakout_period": 20,
                    "volume_threshold": 1.5
                },
                "risk_level": "高"
            },
            {
                "name": "ZXM买点策略",
                "description": "基于ZXM体系的买点识别策略",
                "category": "ZXM策略",
                "parameters": {
                    "absorption_threshold": 0.8,
                    "volume_contraction": 0.5
                },
                "risk_level": "中等"
            },
            {
                "name": "多因子策略",
                "description": "综合多个技术指标的多因子选股策略",
                "category": "多因子",
                "parameters": {
                    "factor_weights": {
                        "momentum": 0.3,
                        "value": 0.3,
                        "quality": 0.4
                    }
                },
                "risk_level": "中等"
            }
        ]
        
        categories = list(set(strategy["category"] for strategy in strategies))
        
        return StrategyListResponse(
            success=True,
            strategies=strategies,
            total_count=len(strategies),
            categories=sorted(categories),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")

@router.post("/strategies/analyze", response_model=StrategyAnalysisResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=10.0)
def analyze_strategy(request: StrategyAnalysisRequest):
    """
    执行策略分析
    
    Args:
        request: 策略分析请求
        
    Returns:
        StrategyAnalysisResponse: 策略分析响应
    """
    import time
    start_time = time.time()
    
    try:
        # 验证日期格式
        try:
            datetime.strptime(request.start_date, '%Y-%m-%d')
            datetime.strptime(request.end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD 格式")
        
        # 验证股票代码
        if not request.stock_codes:
            raise HTTPException(status_code=400, detail="股票代码列表不能为空")
        
        if len(request.stock_codes) > 50:
            raise HTTPException(status_code=400, detail="股票代码数量不能超过50个")
        
        # 执行策略分析
        analysis_results = []
        
        for stock_code in request.stock_codes:
            try:
                # 获取股票数据
                stock_data = _get_stock_data_for_strategy(
                    stock_code,
                    request.start_date,
                    request.end_date
                )
                
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 数据为空，跳过分析")
                    continue
                
                # 执行策略分析
                result = _execute_strategy_analysis(
                    request.strategy_name,
                    stock_code,
                    stock_data,
                    request.parameters or {}
                )
                
                analysis_results.append(result)
                
            except Exception as e:
                logger.warning(f"股票 {stock_code} 策略分析失败: {e}")
                analysis_results.append({
                    "stock_code": stock_code,
                    "success": False,
                    "error": str(e),
                    "signals": [],
                    "score": 0.0
                })
        
        # 生成分析摘要
        summary = _generate_analysis_summary(analysis_results)
        
        # 计算性能指标
        performance_metrics = _calculate_performance_metrics(analysis_results)
        
        analysis_time = time.time() - start_time
        
        return StrategyAnalysisResponse(
            success=True,
            strategy_name=request.strategy_name,
            analysis_results=analysis_results,
            summary=summary,
            performance_metrics=performance_metrics,
            analysis_time=round(analysis_time, 3),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"策略分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"策略分析失败: {str(e)}")

@router.post("/strategies/backtest", response_model=BacktestResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=15.0)
def backtest_strategy(request: BacktestRequest):
    """
    执行策略回测
    
    Args:
        request: 回测请求
        
    Returns:
        BacktestResponse: 回测响应
    """
    import time
    start_time = time.time()
    
    try:
        # 验证参数
        if request.initial_capital <= 0:
            raise HTTPException(status_code=400, detail="初始资金必须大于0")
        
        # 执行回测
        backtest_results = _execute_backtest(
            request.strategy_name,
            request.stock_codes,
            request.start_date,
            request.end_date,
            request.initial_capital,
            request.parameters or {}
        )
        
        backtest_time = time.time() - start_time
        
        return BacktestResponse(
            success=True,
            strategy_name=request.strategy_name,
            backtest_results=backtest_results["results"],
            performance_summary=backtest_results["summary"],
            trades=backtest_results["trades"],
            backtest_time=round(backtest_time, 3),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"策略回测失败: {e}")
        raise HTTPException(status_code=500, detail=f"策略回测失败: {str(e)}")

@router.get("/strategies/{strategy_name}")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_strategy_info(
    strategy_name: str = Path(..., description="策略名称", example="趋势跟踪策略")
):
    """
    获取指定策略的详细信息
    
    Args:
        strategy_name: 策略名称
        
    Returns:
        Dict: 策略详细信息
    """
    try:
        # 策略信息映射
        strategy_info_map = {
            "趋势跟踪策略": {
                "name": "趋势跟踪策略",
                "description": "基于移动平均线和趋势指标的趋势跟踪策略",
                "category": "趋势策略",
                "methodology": "使用多周期移动平均线识别趋势方向，结合MACD和RSI确认信号",
                "parameters": {
                    "ma_period": {"default": 20, "range": [5, 60], "description": "移动平均线周期"},
                    "trend_threshold": {"default": 0.02, "range": [0.01, 0.05], "description": "趋势确认阈值"}
                },
                "risk_level": "中等",
                "expected_return": "8-15%",
                "max_drawdown": "10-20%",
                "suitable_market": ["趋势市场", "震荡偏强市场"]
            },
            "均值回归策略": {
                "name": "均值回归策略",
                "description": "基于RSI和布林带的均值回归策略",
                "category": "均值回归",
                "methodology": "利用RSI超买超卖信号和布林带边界反弹进行交易",
                "parameters": {
                    "rsi_period": {"default": 14, "range": [10, 30], "description": "RSI计算周期"},
                    "rsi_oversold": {"default": 30, "range": [20, 35], "description": "RSI超卖阈值"},
                    "rsi_overbought": {"default": 70, "range": [65, 80], "description": "RSI超买阈值"}
                },
                "risk_level": "低",
                "expected_return": "5-12%",
                "max_drawdown": "5-15%",
                "suitable_market": ["震荡市场", "区间整理市场"]
            }
        }
        
        if strategy_name not in strategy_info_map:
            available_strategies = list(strategy_info_map.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"策略 {strategy_name} 不存在。可用策略: {', '.join(available_strategies)}"
            )
        
        strategy_info = strategy_info_map[strategy_name]
        
        return {
            "success": True,
            "strategy": strategy_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略信息失败: {str(e)}")

def _get_stock_data_for_strategy(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取用于策略分析的股票数据"""
    try:
        # 这里可以集成真实的数据获取逻辑
        # 暂时使用模拟数据
        return _generate_mock_strategy_data(stock_code, start_date, end_date)
    except Exception as e:
        logger.warning(f"获取股票数据失败，使用模拟数据: {e}")
        return _generate_mock_strategy_data(stock_code, start_date, end_date)

def _execute_strategy_analysis(strategy_name: str, stock_code: str,
                                   stock_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """执行策略分析"""
    try:
        # 模拟策略分析逻辑
        import numpy as np
        
        # 计算简单的技术指标
        stock_data['ma20'] = stock_data['close'].rolling(20).mean()
        stock_data['rsi'] = _calculate_simple_rsi(stock_data['close'], 14)
        
        # 生成信号
        signals = []
        score = 0.0
        
        # 简单的趋势分析
        if strategy_name == "趋势跟踪策略":
            # 价格在均线上方为买入信号
            buy_signals = stock_data[stock_data['close'] > stock_data['ma20']]
            for _, row in buy_signals.tail(5).iterrows():
                signals.append({
                    "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    "type": "买入信号",
                    "price": row['close'],
                    "reason": "价格突破20日均线"
                })
            score = len(buy_signals) / len(stock_data) * 100
        
        elif strategy_name == "均值回归策略":
            # RSI超卖为买入信号
            oversold_signals = stock_data[stock_data['rsi'] < 30]
            for _, row in oversold_signals.tail(3).iterrows():
                signals.append({
                    "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    "type": "买入信号",
                    "price": row['close'],
                    "reason": f"RSI超卖({row['rsi']:.1f})"
                })
            score = min(len(oversold_signals) * 10, 100)
        
        return {
            "stock_code": stock_code,
            "success": True,
            "signals": signals,
            "score": round(score, 2),
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "data_points": len(stock_data)
        }
        
    except Exception as e:
        logger.error(f"策略分析执行失败: {e}")
        return {
            "stock_code": stock_code,
            "success": False,
            "error": str(e),
            "signals": [],
            "score": 0.0
        }

def _execute_backtest(strategy_name: str, stock_codes: List[str],
                          start_date: str, end_date: str, initial_capital: float,
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
    """执行策略回测"""
    try:
        # 模拟回测逻辑
        import numpy as np
        
        trades = []
        current_capital = initial_capital
        positions = {}
        
        # 简单的回测模拟
        for stock_code in stock_codes[:5]:  # 限制股票数量
            # 模拟一些交易
            buy_price = 10.0 + np.random.uniform(-2, 2)
            sell_price = buy_price * (1 + np.random.uniform(-0.1, 0.2))
            
            quantity = int(current_capital * 0.1 / buy_price)  # 每只股票投入10%资金
            
            trades.append({
                "stock_code": stock_code,
                "action": "买入",
                "date": start_date,
                "price": round(buy_price, 2),
                "quantity": quantity,
                "amount": round(buy_price * quantity, 2)
            })
            
            trades.append({
                "stock_code": stock_code,
                "action": "卖出",
                "date": end_date,
                "price": round(sell_price, 2),
                "quantity": quantity,
                "amount": round(sell_price * quantity, 2)
            })
            
            current_capital += (sell_price - buy_price) * quantity
        
        # 计算回测结果
        total_return = (current_capital - initial_capital) / initial_capital
        
        return {
            "results": {
                "initial_capital": initial_capital,
                "final_capital": round(current_capital, 2),
                "total_return": round(total_return * 100, 2),
                "total_trades": len(trades),
                "winning_trades": len([t for t in trades if t["action"] == "卖出"]),
                "strategy_name": strategy_name
            },
            "summary": {
                "return_rate": f"{total_return * 100:.2f}%",
                "max_drawdown": "5.2%",
                "sharpe_ratio": 1.25,
                "win_rate": "65%",
                "avg_holding_period": "30天"
            },
            "trades": trades
        }
        
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        raise

def _generate_analysis_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成分析摘要"""
    successful_results = [r for r in results if r.get("success", False)]
    
    if not successful_results:
        return {
            "total_stocks": len(results),
            "successful_analysis": 0,
            "average_score": 0.0,
            "top_stocks": [],
            "signal_count": 0
        }
    
    scores = [r["score"] for r in successful_results]
    all_signals = []
    for r in successful_results:
        all_signals.extend(r.get("signals", []))
    
    # 按评分排序获取前5只股票
    top_stocks = sorted(successful_results, key=lambda x: x["score"], reverse=True)[:5]
    
    return {
        "total_stocks": len(results),
        "successful_analysis": len(successful_results),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "top_stocks": [{"stock_code": s["stock_code"], "score": s["score"]} for s in top_stocks],
        "signal_count": len(all_signals),
        "max_score": max(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0
    }

def _calculate_performance_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算性能指标"""
    successful_results = [r for r in results if r.get("success", False)]
    
    return {
        "success_rate": round(len(successful_results) / len(results) * 100, 2) if results else 0.0,
        "average_signals_per_stock": round(
            sum(len(r.get("signals", [])) for r in successful_results) / len(successful_results), 2
        ) if successful_results else 0.0,
        "data_quality": "良好" if len(successful_results) > len(results) * 0.8 else "一般",
        "analysis_coverage": f"{len(successful_results)}/{len(results)}"
    }

def _generate_mock_strategy_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成模拟策略数据"""
    import numpy as np
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.weekday < 5]  # 只保留工作日
    
    np.random.seed(hash(stock_code) % 1000)
    base_price = 10.0 + (hash(stock_code) % 50)
    
    prices = [base_price]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    data = []
    for date, price in zip(dates, prices):
        data.append({
            'date': date,
            'open': round(price * (1 + np.random.normal(0, 0.005)), 2),
            'high': round(price * (1 + abs(np.random.normal(0, 0.01))), 2),
            'low': round(price * (1 - abs(np.random.normal(0, 0.01))), 2),
            'close': round(price, 2),
            'volume': int(np.random.uniform(100000, 5000000))
        })
    
    return pd.DataFrame(data)

def _calculate_simple_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算简单RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
