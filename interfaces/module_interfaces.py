"""
系统四大模块标准接口定义

定义技术指标分析、买点回测分析、策略选股分析、市场监控模块的标准接口
遵循六层架构设计，确保模块间协同和向后兼容性
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from db.sql_manager import SQLManager, QueryType


@dataclass
class StandardResponse:
    """标准响应格式"""
    success: bool
    data: Any
    message: str = ""
    error_code: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class IndicatorRequest:
    """指标计算请求"""
    indicator_name: str
    data: pd.DataFrame
    period: str
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class PatternRequest:
    """形态识别请求"""
    pattern_name: str
    indicator_name: str
    data: pd.DataFrame
    period: str
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class StrategyRequest:
    """策略执行请求"""
    strategy_config: Dict[str, Any]
    stock_pool: List[str]
    start_date: str
    end_date: str
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class TechnicalIndicatorInterface(ABC):
    """
    技术指标分析模块标准接口
    
    负责103个技术指标的计算、形态识别、信号生成
    为其他模块提供技术分析基础设施
    """
    
    @abstractmethod
    def calculate_indicator(self, request: IndicatorRequest) -> StandardResponse:
        """
        计算技术指标
        
        Args:
            request: 指标计算请求
            
        Returns:
            StandardResponse: 包含计算结果的标准响应
        """
        pass
    
    @abstractmethod
    def detect_pattern(self, request: PatternRequest) -> StandardResponse:
        """
        识别技术形态
        
        Args:
            request: 形态识别请求
            
        Returns:
            StandardResponse: 包含形态识别结果的标准响应
        """
        pass
    
    @abstractmethod
    def generate_signals(self, indicator_name: str, data: pd.DataFrame, 
                        period: str, **params) -> StandardResponse:
        """
        生成交易信号
        
        Args:
            indicator_name: 指标名称
            data: 股票数据
            period: 时间周期
            **params: 其他参数
            
        Returns:
            StandardResponse: 包含信号的标准响应
        """
        pass
    
    @abstractmethod
    def batch_calculate(self, requests: List[IndicatorRequest]) -> StandardResponse:
        """
        批量计算指标
        
        Args:
            requests: 指标计算请求列表
            
        Returns:
            StandardResponse: 包含批量计算结果的标准响应
        """
        pass
    
    @abstractmethod
    def get_supported_indicators(self) -> List[str]:
        """
        获取支持的指标列表
        
        Returns:
            List[str]: 支持的指标名称列表
        """
        pass
    
    @abstractmethod
    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """
        获取指标详细信息
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            Dict[str, Any]: 指标信息
        """
        pass


class BuyPointAnalysisInterface(ABC):
    """
    买点回测分析模块标准接口
    
    负责历史买点的深度分析、形态统计、策略规则提取
    为策略选股模块提供历史验证数据
    """
    
    @abstractmethod
    def analyze_buypoint(self, stock_code: str, buypoint_date: str, 
                        stock_name: str = "") -> StandardResponse:
        """
        分析单个买点
        
        Args:
            stock_code: 股票代码
            buypoint_date: 买点日期
            stock_name: 股票名称
            
        Returns:
            StandardResponse: 包含买点分析结果的标准响应
        """
        pass
    
    @abstractmethod
    def batch_analyze_buypoints(self, buypoints: List[Dict[str, str]]) -> StandardResponse:
        """
        批量分析买点
        
        Args:
            buypoints: 买点列表，每个元素包含stock_code和buypoint_date
            
        Returns:
            StandardResponse: 包含批量分析结果的标准响应
        """
        pass
    
    @abstractmethod
    def extract_strategy_rules(self, analysis_results: List[Dict[str, Any]]) -> StandardResponse:
        """
        从分析结果中提取策略规则
        
        Args:
            analysis_results: 买点分析结果列表
            
        Returns:
            StandardResponse: 包含提取的策略规则
        """
        pass
    
    @abstractmethod
    def validate_strategy(self, strategy_config: Dict[str, Any], 
                         validation_data: pd.DataFrame) -> StandardResponse:
        """
        验证策略有效性
        
        Args:
            strategy_config: 策略配置
            validation_data: 验证数据
            
        Returns:
            StandardResponse: 包含验证结果的标准响应
        """
        pass
    
    @abstractmethod
    def generate_analysis_report(self, analysis_results: List[Dict[str, Any]]) -> StandardResponse:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            StandardResponse: 包含分析报告的标准响应
        """
        pass


class StrategySelectionInterface(ABC):
    """
    策略选股分析模块标准接口
    
    系统核心模块，负责策略配置、高性能选股执行、结果处理
    将技术分析转化为实际可执行的选股决策
    """
    
    @abstractmethod
    def parse_strategy_formula(self, formula: str) -> StandardResponse:
        """
        解析通达信风格策略公式
        
        Args:
            formula: 策略公式字符串
            
        Returns:
            StandardResponse: 包含解析结果的标准响应
        """
        pass
    
    @abstractmethod
    def execute_selection(self, request: StrategyRequest) -> StandardResponse:
        """
        执行选股策略
        
        Args:
            request: 策略执行请求
            
        Returns:
            StandardResponse: 包含选股结果的标准响应
        """
        pass
    
    @abstractmethod
    def evaluate_and_rank(self, selection_results: List[Dict[str, Any]]) -> StandardResponse:
        """
        评估和排序选股结果
        
        Args:
            selection_results: 选股结果列表
            
        Returns:
            StandardResponse: 包含评估排序后的结果
        """
        pass
    
    @abstractmethod
    def track_strategy_performance(self, strategy_id: str, 
                                  start_date: str, end_date: str) -> StandardResponse:
        """
        跟踪策略绩效
        
        Args:
            strategy_id: 策略ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            StandardResponse: 包含绩效数据的标准响应
        """
        pass
    
    @abstractmethod
    def optimize_strategy_parameters(self, strategy_config: Dict[str, Any], 
                                   optimization_data: pd.DataFrame) -> StandardResponse:
        """
        优化策略参数
        
        Args:
            strategy_config: 策略配置
            optimization_data: 优化数据
            
        Returns:
            StandardResponse: 包含优化后的策略配置
        """
        pass


class MarketMonitoringInterface(ABC):
    """
    市场监控模块标准接口
    
    负责实时数据监控、智能预警、趋势跟踪、风险监控
    为选股执行提供实时数据支撑和风险预警
    """
    
    @abstractmethod
    def start_real_time_monitoring(self, stock_codes: List[str], 
                                  indicators: List[str]) -> StandardResponse:
        """
        启动实时监控
        
        Args:
            stock_codes: 监控的股票代码列表
            indicators: 监控的指标列表
            
        Returns:
            StandardResponse: 监控启动结果
        """
        pass
    
    @abstractmethod
    def setup_alerts(self, alert_configs: List[Dict[str, Any]]) -> StandardResponse:
        """
        设置预警规则
        
        Args:
            alert_configs: 预警配置列表
            
        Returns:
            StandardResponse: 预警设置结果
        """
        pass
    
    @abstractmethod
    def get_market_status(self) -> StandardResponse:
        """
        获取市场状态
        
        Returns:
            StandardResponse: 包含市场状态信息
        """
        pass
    
    @abstractmethod
    def detect_anomalies(self, stock_codes: List[str], 
                        detection_window: int = 20) -> StandardResponse:
        """
        检测异常波动
        
        Args:
            stock_codes: 股票代码列表
            detection_window: 检测时间窗口
            
        Returns:
            StandardResponse: 包含异常检测结果
        """
        pass
    
    @abstractmethod
    def track_trends(self, stock_codes: List[str], 
                    periods: List[str]) -> StandardResponse:
        """
        跟踪趋势变化
        
        Args:
            stock_codes: 股票代码列表
            periods: 时间周期列表
            
        Returns:
            StandardResponse: 包含趋势跟踪结果
        """
        pass
    
    @abstractmethod
    def assess_portfolio_risk(self, portfolio: Dict[str, float]) -> StandardResponse:
        """
        评估投资组合风险
        
        Args:
            portfolio: 投资组合，股票代码到权重的映射
            
        Returns:
            StandardResponse: 包含风险评估结果
        """
        pass
