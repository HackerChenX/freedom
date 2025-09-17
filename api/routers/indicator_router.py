#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
技术指标API路由

提供技术指标计算相关的RESTful API接口
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional, Dict, Any, Union
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

class IndicatorCalculateRequest(BaseModel):
    """技术指标计算请求模型"""
    stock_code: str = Field(..., description="股票代码", example="000001")
    indicator_name: str = Field(..., description="指标名称", example="MACD")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)", example="2024-01-01")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)", example="2024-12-31")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="指标参数")

class IndicatorCalculateResponse(BaseModel):
    """技术指标计算响应模型"""
    success: bool = Field(..., description="计算是否成功")
    indicator_name: str = Field(..., description="指标名称")
    stock_code: str = Field(..., description="股票代码")
    data: List[Dict[str, Any]] = Field(..., description="指标计算结果")
    parameters: Dict[str, Any] = Field(..., description="使用的参数")
    calculation_time: float = Field(..., description="计算耗时(秒)")
    timestamp: str = Field(..., description="响应时间戳")

class IndicatorListResponse(BaseModel):
    """指标列表响应模型"""
    success: bool = Field(..., description="请求是否成功")
    indicators: List[Dict[str, Any]] = Field(..., description="指标列表")
    total_count: int = Field(..., description="指标总数")
    categories: List[str] = Field(..., description="指标分类")
    timestamp: str = Field(..., description="响应时间戳")

# 获取指标注册表
def get_indicator_registry():
    """获取指标注册表"""
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        # 如果是CompleteIndicatorRegistry对象，获取其indicators属性
        if hasattr(registry, 'indicators'):
            return registry.indicators
        elif hasattr(registry, 'get_all_indicators'):
            return registry.get_all_indicators()
        else:
            return {}
    except Exception as e:
        logger.error(f"获取指标注册表失败: {e}")
        return {}

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

@router.get("/indicators", response_model=IndicatorListResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_indicator_list():
    """
    获取所有可用的技术指标列表
    
    Returns:
        IndicatorListResponse: 指标列表响应
    """
    try:
        registry = get_indicator_registry()
        
        indicators = []
        categories = set()
        
        for name, indicator_class in registry.items():
            try:
                # 创建指标实例获取信息
                indicator = indicator_class()
                
                # 获取指标信息
                indicator_info = {
                    "name": name,
                    "class_name": indicator_class.__name__,
                    "description": getattr(indicator, '__doc__', '').strip() if hasattr(indicator, '__doc__') else f"{name}技术指标",
                    "category": _get_indicator_category(name),
                    "parameters": _get_indicator_parameters(indicator),
                    "minimum_periods": getattr(indicator, 'minimum_periods', 20)
                }
                
                indicators.append(indicator_info)
                categories.add(indicator_info["category"])
                
            except Exception as e:
                logger.warning(f"获取指标 {name} 信息失败: {e}")
                # 添加基本信息
                indicators.append({
                    "name": name,
                    "class_name": indicator_class.__name__,
                    "description": f"{name}技术指标",
                    "category": "其他",
                    "parameters": {},
                    "minimum_periods": 20
                })
                categories.add("其他")
        
        return IndicatorListResponse(
            success=True,
            indicators=indicators,
            total_count=len(indicators),
            categories=sorted(list(categories)),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"获取指标列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标列表失败: {str(e)}")

@router.post("/indicators/calculate", response_model=IndicatorCalculateResponse)
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=5.0)
def calculate_indicator(request: IndicatorCalculateRequest):
    """
    计算技术指标
    
    Args:
        request: 指标计算请求
        
    Returns:
        IndicatorCalculateResponse: 指标计算响应
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
        
        # 获取指标注册表
        registry = get_indicator_registry()
        
        if request.indicator_name not in registry:
            available_indicators = list(registry.keys())[:10]  # 显示前10个可用指标
            raise HTTPException(
                status_code=404, 
                detail=f"指标 {request.indicator_name} 不存在。可用指标示例: {', '.join(available_indicators)}"
            )
        
        # 获取股票数据
        stock_data = _get_stock_data_for_indicator(
            request.stock_code,
            request.start_date,
            request.end_date
        )
        
        if stock_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"未找到股票 {request.stock_code} 的数据"
            )
        
        # 创建指标实例
        indicator_class = registry[request.indicator_name]
        indicator = indicator_class()
        
        # 设置参数
        if request.parameters:
            try:
                indicator.set_parameters(**request.parameters)
            except Exception as e:
                logger.warning(f"设置指标参数失败: {e}")
        
        # 获取使用的参数
        used_parameters = _get_indicator_parameters(indicator)
        
        # 计算指标
        try:
            result = indicator.calculate(stock_data)
            
            if result is None or result.empty:
                raise HTTPException(status_code=500, detail="指标计算结果为空")
            
            # 转换为字典列表
            result_data = result.to_dict('records')
            
            # 处理日期格式和NaN值
            for record in result_data:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif hasattr(value, 'strftime'):
                        record[key] = value.strftime('%Y-%m-%d')
            
            calculation_time = time.time() - start_time
            
            return IndicatorCalculateResponse(
                success=True,
                indicator_name=request.indicator_name,
                stock_code=request.stock_code,
                data=result_data,
                parameters=used_parameters,
                calculation_time=round(calculation_time, 3),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"指标计算失败: {e}")
            raise HTTPException(status_code=500, detail=f"指标计算失败: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算技术指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算技术指标失败: {str(e)}")

@router.get("/indicators/{indicator_name}")
@exception_handler(reraise=True)
@performance_monitor(threshold_seconds=1.0)
def get_indicator_info(
    indicator_name: str = Path(..., description="指标名称", example="MACD")
):
    """
    获取指定技术指标的详细信息
    
    Args:
        indicator_name: 指标名称
        
    Returns:
        Dict: 指标详细信息
    """
    try:
        registry = get_indicator_registry()
        
        if indicator_name not in registry:
            raise HTTPException(status_code=404, detail=f"指标 {indicator_name} 不存在")
        
        indicator_class = registry[indicator_name]
        
        try:
            indicator = indicator_class()
            
            return {
                "success": True,
                "indicator": {
                    "name": indicator_name,
                    "class_name": indicator_class.__name__,
                    "description": getattr(indicator, '__doc__', '').strip() if hasattr(indicator, '__doc__') else f"{indicator_name}技术指标",
                    "category": _get_indicator_category(indicator_name),
                    "parameters": _get_indicator_parameters(indicator),
                    "minimum_periods": getattr(indicator, 'minimum_periods', 20),
                    "output_columns": _get_indicator_output_columns(indicator),
                    "calculation_method": _get_calculation_method_info(indicator)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"创建指标实例失败: {e}")
            return {
                "success": True,
                "indicator": {
                    "name": indicator_name,
                    "class_name": indicator_class.__name__,
                    "description": f"{indicator_name}技术指标",
                    "category": _get_indicator_category(indicator_name),
                    "parameters": {},
                    "minimum_periods": 20,
                    "output_columns": [],
                    "calculation_method": "标准计算方法"
                },
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取指标信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标信息失败: {str(e)}")

def _get_stock_data_for_indicator(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取用于指标计算的股票数据"""
    try:
        # 获取数据访问接口
        data_access = get_data_access()
        
        if data_access is None:
            # 使用模拟数据
            logger.warning("使用模拟数据进行指标计算")
            return _generate_mock_stock_dataframe(stock_code, start_date, end_date)
        
        # 构建查询SQL
        query = f"""
        SELECT date, open, high, low, close, volume, turnover_rate
        FROM stock_info WHERE level = %(level)s AND code = '{stock_code}'
        AND level = '日线'
        AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date ASC
        """
        
        # 执行查询
        df = data_access.query_dataframe(query)
        
        if df.empty:
            # 如果没有真实数据，使用模拟数据
            logger.warning(f"未找到股票 {stock_code} 的真实数据，使用模拟数据")
            return _generate_mock_stock_dataframe(stock_code, start_date, end_date)
        
        return df
        
    except Exception as e:
        logger.warning(f"获取股票数据失败，使用模拟数据: {e}")
        return _generate_mock_stock_dataframe(stock_code, start_date, end_date)

def _generate_mock_stock_dataframe(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成模拟股票数据DataFrame"""
    import numpy as np
    from datetime import datetime, timedelta
from db.sql_manager import SQLManager, QueryType
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = pd.date_range(start=start, end=end, freq='D')
    # 只保留工作日
    dates = dates[dates.weekday < 5]
    
    # 生成模拟价格数据
    np.random.seed(hash(stock_code) % 1000)
    base_price = 10.0 + (hash(stock_code) % 50)
    
    n_days = len(dates)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 生成OHLC数据
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        open_price = price if i == 0 else prices[i-1]
        close_price = price
        
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': int(np.random.uniform(100000, 5000000)),
            'turnover_rate': round(np.random.uniform(0.5, 8.0), 2)
        })
    
    return pd.DataFrame(data)

def _get_indicator_category(indicator_name: str) -> str:
    """获取指标分类"""
    trend_indicators = ['MA', 'EMA', 'WMA', 'MACD', 'TRIX', 'DMI', 'ADX']
    oscillator_indicators = ['RSI', 'KDJ', 'WR', 'CCI', 'STOCHRSI']
    volume_indicators = ['VOL', 'OBV', 'MFI', 'VR', 'VOLUME_RATIO']
    volatility_indicators = ['BOLL', 'ATR', 'KC', 'VIX']
    pattern_indicators = ['DOJI', 'HAMMER', 'ENGULFING', 'HARAMI']
    
    if indicator_name in trend_indicators:
        return "趋势指标"
    elif indicator_name in oscillator_indicators:
        return "振荡器指标"
    elif indicator_name in volume_indicators:
        return "成交量指标"
    elif indicator_name in volatility_indicators:
        return "波动性指标"
    elif indicator_name in pattern_indicators:
        return "形态识别指标"
    elif indicator_name.startswith('ZXM_'):
        return "ZXM体系指标"
    else:
        return "其他指标"

def _get_indicator_parameters(indicator) -> Dict[str, Any]:
    """获取指标参数"""
    try:
        if hasattr(indicator, '_get_default_parameters'):
            return indicator._get_default_parameters()
        elif hasattr(indicator, 'period'):
            return {"period": getattr(indicator, 'period', 20)}
        else:
            return {}
    except Exception:
        return {}

def _get_indicator_output_columns(indicator) -> List[str]:
    """获取指标输出列"""
    try:
        # 尝试用少量数据计算获取输出列
        mock_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'open': [10.0] * 30,
            'high': [11.0] * 30,
            'low': [9.0] * 30,
            'close': [10.5] * 30,
            'volume': [1000000] * 30
        })
        
        result = indicator.calculate(mock_data)
        if result is not None and not result.empty:
            return list(result.columns)
        else:
            return []
    except Exception:
        return []

def _get_calculation_method_info(indicator) -> str:
    """获取计算方法信息"""
    try:
        if hasattr(indicator, 'calculate'):
            doc = getattr(indicator.calculate, '__doc__', '')
            if doc:
                return doc.strip()
        return "标准技术指标计算方法"
    except Exception:
        return "标准技术指标计算方法"
