#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
股票数据API路由

提供股票数据查询相关的RESTful API接口
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date
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

class StockDataRequest(BaseModel):
    """股票数据请求模型"""
    stock_code: str = Field(..., description="股票代码", example="000001")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)", example="2024-01-01")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)", example="2024-12-31")
    level: str = Field(default="日线", description="数据级别", example="日线")

class StockDataResponse(BaseModel):
    """股票数据响应模型"""
    success: bool = Field(..., description="请求是否成功")
    data: List[Dict[str, Any]] = Field(..., description="股票数据列表")
    total_count: int = Field(..., description="数据总数")
    stock_code: str = Field(..., description="股票代码")
    stock_name: Optional[str] = Field(None, description="股票名称")
    period: str = Field(..., description="查询时间范围")
    timestamp: str = Field(..., description="响应时间戳")

class StockListResponse(BaseModel):
    """股票列表响应模型"""
    success: bool = Field(..., description="请求是否成功")
    stocks: List[Dict[str, str]] = Field(..., description="股票列表")
    total_count: int = Field(..., description="股票总数")
    timestamp: str = Field(..., description="响应时间戳")

# 获取数据访问接口
def get_data_access():
    """获取数据访问接口"""
    try:
        container = get_container()
        # 尝试多种方式获取数据访问接口
        try:
            from db.interfaces.data_access_interface import DataAccessInterface
            return container.resolve(DataAccessInterface)
        except:
            # 尝试直接获取数据管理器
            try:
                return container.resolve("DataAccessManager")
            except:
                # 尝试获取数据库管理器
                return container.resolve("DBManager")
    except Exception as e:
        logger.warning(f"无法获取数据访问接口: {e}")
        return None

@router.get("/stocks/{stock_code}/data", response_model=StockDataResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=2.0)
def get_stock_data(
    stock_code: str = Path(..., description="股票代码", example="000001"),
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)", example="2024-01-01"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)", example="2024-12-31"),
    level: str = Query(default="日线", description="数据级别", example="日线")
):
    """
    获取股票历史数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        level: 数据级别
        
    Returns:
        StockDataResponse: 股票数据响应
    """
    try:
        # 验证日期格式
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD 格式")
        
        # 获取数据访问接口
        data_access = get_data_access()
        
        if data_access is None:
            # 使用模拟数据
            logger.warning("使用模拟数据响应")
            mock_data = _generate_mock_stock_data(stock_code, start_date, end_date)
            
            return StockDataResponse(
                success=True,
                data=mock_data,
                total_count=len(mock_data),
                stock_code=stock_code,
                stock_name=f"模拟股票{stock_code}",
                period=f"{start_date} 至 {end_date}",
                timestamp=datetime.now().isoformat()
            )
        
        # 构建查询SQL
        query = f"""
        SELECT code, name, date, open, high, low, close, volume, turnover_rate
        FROM stock_info 
        WHERE code = '{stock_code}'
        AND level = '{level}'
        AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date ASC
        """
        
        # 执行查询
        df = data_access.query_dataframe(query)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {stock_code} 的数据")
        
        # 转换为字典列表
        data_list = df.to_dict('records')
        
        # 处理日期格式
        for record in data_list:
            if 'date' in record and hasattr(record['date'], 'strftime'):
                record['date'] = record['date'].strftime('%Y-%m-%d')
        
        stock_name = data_list[0].get('name', stock_code) if data_list else stock_code
        
        return StockDataResponse(
            success=True,
            data=data_list,
            total_count=len(data_list),
            stock_code=stock_code,
            stock_name=stock_name,
            period=f"{start_date} 至 {end_date}",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票数据失败: {str(e)}")

@router.get("/stocks", response_model=StockListResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=3.0)
def get_stock_list(
    limit: int = Query(default=100, description="返回数量限制", ge=1, le=1000),
    offset: int = Query(default=0, description="偏移量", ge=0),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """
    获取股票列表
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
        search: 搜索关键词
        
    Returns:
        StockListResponse: 股票列表响应
    """
    try:
        # 获取数据访问接口
        data_access = get_data_access()
        
        if data_access is None:
            # 使用模拟数据
            logger.warning("使用模拟股票列表数据")
            mock_stocks = _generate_mock_stock_list(limit, offset, search)
            
            return StockListResponse(
                success=True,
                stocks=mock_stocks,
                total_count=len(mock_stocks),
                timestamp=datetime.now().isoformat()
            )
        
        # 构建查询SQL
        where_clause = ""
        if search:
            where_clause = f"WHERE code LIKE '%{search}%' OR name LIKE '%{search}%'"
        
        query = f"""
        SELECT DISTINCT code, name
        FROM stock_info 
        {where_clause}
        ORDER BY code
        LIMIT {limit} OFFSET {offset}
        """
        
        # 执行查询
        df = data_access.query_dataframe(query)
        
        # 转换为字典列表
        stocks = [{"code": row["code"], "name": row["name"]} for _, row in df.iterrows()]
        
        return StockListResponse(
            success=True,
            stocks=stocks,
            total_count=len(stocks),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票列表失败: {str(e)}")

@router.get("/stocks/{stock_code}/latest")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_latest_stock_data(
    stock_code: str = Path(..., description="股票代码", example="000001")
):
    """
    获取股票最新数据
    
    Args:
        stock_code: 股票代码
        
    Returns:
        Dict: 最新股票数据
    """
    try:
        # 获取数据访问接口
        data_access = get_data_access()
        
        if data_access is None:
            # 使用模拟数据
            logger.warning("使用模拟最新数据")
            return {
                "success": True,
                "data": {
                    "code": stock_code,
                    "name": f"模拟股票{stock_code}",
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "open": 10.50,
                    "high": 11.20,
                    "low": 10.30,
                    "close": 10.95,
                    "volume": 1500000,
                    "turnover_rate": 2.5
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # 构建查询SQL
        query = f"""
        SELECT code, name, date, open, high, low, close, volume, turnover_rate
        FROM stock_info 
        WHERE code = '{stock_code}'
        AND level = '日线'
        ORDER BY date DESC
        LIMIT 1
        """
        
        # 执行查询
        df = data_access.query_dataframe(query)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {stock_code} 的最新数据")
        
        # 转换为字典
        latest_data = df.iloc[0].to_dict()
        
        # 处理日期格式
        if 'date' in latest_data and hasattr(latest_data['date'], 'strftime'):
            latest_data['date'] = latest_data['date'].strftime('%Y-%m-%d')
        
        return {
            "success": True,
            "data": latest_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取最新股票数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取最新股票数据失败: {str(e)}")

def _generate_mock_stock_data(stock_code: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """生成模拟股票数据"""
    import numpy as np
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    while current <= end:
        # 只包含工作日
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    
    # 生成模拟价格数据
    np.random.seed(hash(stock_code) % 1000)
    base_price = 10.0 + (hash(stock_code) % 50)
    
    data = []
    for i, date in enumerate(dates):
        # 生成随机价格变动
        change = np.random.normal(0, 0.02)
        if i == 0:
            open_price = base_price
        else:
            open_price = data[-1]['close']
        
        high = open_price * (1 + abs(np.random.normal(0, 0.01)))
        low = open_price * (1 - abs(np.random.normal(0, 0.01)))
        close = open_price * (1 + change)
        
        # 确保价格逻辑正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'code': stock_code,
            'name': f'模拟股票{stock_code}',
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(np.random.uniform(100000, 5000000)),
            'turnover_rate': round(np.random.uniform(0.5, 8.0), 2)
        })
    
    return data

def _generate_mock_stock_list(limit: int, offset: int, search: Optional[str]) -> List[Dict[str, str]]:
    """生成模拟股票列表"""
    stocks = []
    
    # 生成一些常见的股票代码
    stock_codes = [
        ("000001", "平安银行"),
        ("000002", "万科A"),
        ("600000", "浦发银行"),
        ("600036", "招商银行"),
        ("000858", "五粮液"),
        ("600519", "贵州茅台"),
        ("000725", "京东方A"),
        ("002415", "海康威视"),
        ("600276", "恒瑞医药"),
        ("000063", "中兴通讯")
    ]
    
    # 扩展股票列表
    for i in range(100):
        code = f"{i:06d}"
        name = f"模拟股票{code}"
        stock_codes.append((code, name))
    
    # 应用搜索过滤
    if search:
        filtered_stocks = [
            {"code": code, "name": name}
            for code, name in stock_codes
            if search.lower() in code.lower() or search.lower() in name.lower()
        ]
    else:
        filtered_stocks = [
            {"code": code, "name": name}
            for code, name in stock_codes
        ]
    
    # 应用分页
    start_idx = offset
    end_idx = offset + limit
    
    return filtered_stocks[start_idx:end_idx]
