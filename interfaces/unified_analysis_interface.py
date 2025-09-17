"""
统一分析接口规范
解决analyze_buypoint等方法名不一致问题
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class AnalysisType(Enum):
    """分析类型枚举"""
    BUYPOINT = "buypoint"
    SELLPOINT = "sellpoint"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MULTI_PERIOD = "multi_period"
    STRATEGY = "strategy"


class TimeFrame(Enum):
    """时间周期枚举"""
    MINUTE_1 = "1分钟"
    MINUTE_5 = "5分钟"
    MINUTE_15 = "15分钟"
    MINUTE_30 = "30分钟"
    MINUTE_60 = "60分钟"
    DAILY = "日线"
    WEEKLY = "周线"
    MONTHLY = "月线"


@dataclass
class AnalysisRequest:
    """分析请求标准格式"""
    stock_code: str
    target_date: str
    analysis_type: AnalysisType
    timeframes: Optional[List[TimeFrame]] = None
    indicators: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """分析结果标准格式"""
    success: bool
    stock_code: str
    target_date: str
    analysis_type: AnalysisType
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class IUnifiedAnalyzer(ABC):
    """统一分析器接口"""
    
    @abstractmethod
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        统一分析方法 - 所有分析器的标准入口
        
        Args:
            request: 分析请求
            
        Returns:
            AnalysisResult: 分析结果
        """
        pass
    
    @abstractmethod
    def get_supported_analysis_types(self) -> List[AnalysisType]:
        """获取支持的分析类型"""
        pass
    
    @abstractmethod
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """获取支持的时间周期"""
        pass
    
    @abstractmethod
    def validate_request(self, request: AnalysisRequest) -> bool:
        """验证分析请求"""
        pass


class IBuypointAnalyzer(IUnifiedAnalyzer):
    """买点分析器接口"""
    
    def analyze_buypoint(self, stock_code: str, target_date: str, 
                        timeframes: Optional[List[str]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        买点分析方法 - 兼容性接口
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            timeframes: 时间周期列表
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 转换为标准请求格式
        tf_enums = []
        if timeframes:
            for tf in timeframes:
                try:
                    tf_enums.append(TimeFrame(tf))
                except ValueError:
                    continue
        
        request = AnalysisRequest(
            stock_code=stock_code,
            target_date=target_date,
            analysis_type=AnalysisType.BUYPOINT,
            timeframes=tf_enums,
            parameters=kwargs
        )
        
        result = self.analyze(request)
        
        # 转换为兼容格式
        if result.success:
            return result.data or {}
        else:
            return {
                'success': False,
                'error_message': result.error_message
            }
    
    def analyze_multi_period_buypoint(self, stock_code: str, target_date: str,
                                    periods: Optional[List[Any]] = None,
                                    indicator_names: Optional[List[str]] = None,
                                    enable_strategy_analysis: bool = True,
                                    **kwargs) -> Dict[str, Any]:
        """
        多周期买点分析方法 - 兼容性接口
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            periods: 分析周期列表
            indicator_names: 指标名称列表
            enable_strategy_analysis: 是否启用策略分析
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 转换周期格式
        timeframes = []
        if periods:
            for period in periods:
                if hasattr(period, 'value'):
                    timeframes.append(period.value)
                else:
                    timeframes.append(str(period))
        
        # 构建参数
        parameters = {
            'indicators': indicator_names,
            'enable_strategy_analysis': enable_strategy_analysis,
            **kwargs
        }
        
        request = AnalysisRequest(
            stock_code=stock_code,
            target_date=target_date,
            analysis_type=AnalysisType.MULTI_PERIOD,
            timeframes=[TimeFrame(tf) for tf in timeframes if tf in [e.value for e in TimeFrame]],
            parameters=parameters
        )
        
        result = self.analyze(request)
        
        # 转换为兼容格式
        if result.success:
            return {
                'status': 'SUCCESS',
                'result': result.data,
                'analysis_time': result.execution_time
            }
        else:
            return {
                'status': 'ERROR',
                'error': result.error_message,
                'analysis_time': result.execution_time
            }


class IStrategyAnalyzer(IUnifiedAnalyzer):
    """策略分析器接口"""
    
    def analyze_strategy(self, stock_code: str, target_date: str,
                        strategy_names: Optional[List[str]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        策略分析方法 - 兼容性接口
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            strategy_names: 策略名称列表
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        request = AnalysisRequest(
            stock_code=stock_code,
            target_date=target_date,
            analysis_type=AnalysisType.STRATEGY,
            strategies=strategy_names,
            parameters=kwargs
        )
        
        result = self.analyze(request)
        
        if result.success:
            return result.data or {}
        else:
            return {
                'success': False,
                'error_message': result.error_message
            }


class ITechnicalAnalyzer(IUnifiedAnalyzer):
    """技术分析器接口"""
    
    def analyze_technical(self, stock_code: str, target_date: str,
                         indicators: Optional[List[str]] = None,
                         timeframes: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        技术分析方法 - 兼容性接口
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            indicators: 指标列表
            timeframes: 时间周期列表
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        tf_enums = []
        if timeframes:
            for tf in timeframes:
                try:
                    tf_enums.append(TimeFrame(tf))
                except ValueError:
                    continue
        
        request = AnalysisRequest(
            stock_code=stock_code,
            target_date=target_date,
            analysis_type=AnalysisType.TECHNICAL,
            timeframes=tf_enums,
            indicators=indicators,
            parameters=kwargs
        )
        
        result = self.analyze(request)
        
        if result.success:
            return result.data or {}
        else:
            return {
                'success': False,
                'error_message': result.error_message
            }


class UnifiedAnalysisRouter:
    """统一分析路由器"""
    
    def __init__(self):
        self._analyzers: Dict[AnalysisType, IUnifiedAnalyzer] = {}
    
    def register_analyzer(self, analysis_type: AnalysisType, 
                         analyzer: IUnifiedAnalyzer):
        """注册分析器"""
        self._analyzers[analysis_type] = analyzer
    
    def route_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """路由分析请求"""
        if request.analysis_type not in self._analyzers:
            return AnalysisResult(
                success=False,
                stock_code=request.stock_code,
                target_date=request.target_date,
                analysis_type=request.analysis_type,
                error_message=f"不支持的分析类型: {request.analysis_type}"
            )
        
        analyzer = self._analyzers[request.analysis_type]
        return analyzer.analyze(request)
    
    def get_available_analyzers(self) -> List[AnalysisType]:
        """获取可用的分析器类型"""
        return list(self._analyzers.keys())


# 全局路由器实例
unified_analysis_router = UnifiedAnalysisRouter()


def get_analysis_router() -> UnifiedAnalysisRouter:
    """获取统一分析路由器实例"""
    return unified_analysis_router
