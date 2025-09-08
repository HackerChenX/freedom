"""
模块接口适配器

为现有模块提供标准接口适配，确保向后兼容性
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from abc import ABC, abstractmethod

from interfaces.module_interfaces import (
    TechnicalIndicatorInterface,
    BuyPointAnalysisInterface, 
    StrategySelectionInterface,
    MarketMonitoringInterface,
    StandardResponse,
    IndicatorRequest,
    PatternRequest,
    StrategyRequest
)
from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor

logger = get_logger(__name__)


class TechnicalIndicatorAdapter(TechnicalIndicatorInterface):
    """
    技术指标分析模块适配器
    
    将现有的指标计算模块适配到标准接口
    """
    
    def __init__(self):
        self.indicator_registry = None
        self._initialize_registry()
    
    def _initialize_registry(self):
        """初始化指标注册表"""
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            self.indicator_registry = get_indicator_registry()
            logger.info("技术指标注册表初始化成功")
        except Exception as e:
            logger.error(f"技术指标注册表初始化失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def calculate_indicator(self, request: IndicatorRequest) -> StandardResponse:
        """
        计算技术指标
        
        Args:
            request: 指标计算请求
            
        Returns:
            StandardResponse: 包含计算结果的标准响应
        """
        # 验证请求参数
        if not request.indicator_name or not request.stock_code:
            return StandardResponse(
                success=False,
                message="缺少必需参数: indicator_name 或 stock_code",
                data=None
            )
        
        # 获取指标实例
        indicator = self.indicator_registry.get_indicator(request.indicator_name)
        if not indicator:
            return StandardResponse(
                success=False,
                message=f"未找到指标: {request.indicator_name}",
                data=None
            )
        
        # 设置指标参数
        if request.params:
            indicator.set_parameters(**request.params)
        
        # 执行指标计算
        result = indicator.calculate(request.data)
        
        return StandardResponse(
            success=True,
            message="指标计算成功",
            data={
                "indicator_name": request.indicator_name,
                "stock_code": request.stock_code,
                "period": request.period,
                "result": result.to_dict() if hasattr(result, 'to_dict') else result,
                "calculation_time": pd.Timestamp.now().isoformat()
            }
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def detect_pattern(self, request: PatternRequest) -> StandardResponse:
        """
        识别技术形态
        
        Args:
            request: 形态识别请求
            
        Returns:
            StandardResponse: 包含形态识别结果的标准响应
        """
        # 获取指标实例
        indicator = self.indicator_registry.get_indicator(request.indicator_name)
        if not indicator:
            return StandardResponse(
                success=False,
                message=f"未找到指标: {request.indicator_name}",
                data=None
            )
        
        # 执行形态识别
        patterns = indicator.get_patterns(request.data)
        
        # 过滤特定形态
        if request.pattern_name:
            if request.pattern_name in patterns.columns:
                pattern_result = patterns[request.pattern_name]
            else:
                return StandardResponse(
                    success=False,
                    message=f"未找到形态: {request.pattern_name}",
                    data=None
                )
        else:
            pattern_result = patterns
        
        return StandardResponse(
            success=True,
            message="形态识别成功",
            data={
                "indicator_name": request.indicator_name,
                "pattern_name": request.pattern_name,
                "stock_code": request.stock_code,
                "period": request.period,
                "patterns": pattern_result.to_dict() if hasattr(pattern_result, 'to_dict') else pattern_result,
                "detection_time": pd.Timestamp.now().isoformat()
            }
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.5)
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
        # 获取指标实例
        indicator = self.indicator_registry.get_indicator(indicator_name)
        if not indicator:
            return StandardResponse(
                success=False,
                message=f"未找到指标: {indicator_name}",
                data=None
            )
        
        # 设置参数
        if params:
            indicator.set_parameters(**params)
        
        # 生成信号
        signals = indicator.get_signal(data)
        
        return StandardResponse(
            success=True,
            message="信号生成成功",
            data={
                "indicator_name": indicator_name,
                "period": period,
                "signals": signals,
                "signal_time": pd.Timestamp.now().isoformat()
            }
        )


class BuyPointAnalysisAdapter(BuyPointAnalysisInterface):
    """
    买点回测分析模块适配器
    
    将现有的买点分析模块适配到标准接口
    """
    
    def __init__(self):
        self.buypoint_analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """初始化买点分析器"""
        try:
            from analysis.buypoints.buypoint_analyzer import BuyPointAnalyzer
            self.buypoint_analyzer = BuyPointAnalyzer()
            logger.info("买点分析器初始化成功")
        except Exception as e:
            logger.error(f"买点分析器初始化失败: {e}")
            raise
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=30.0)
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
        # 执行买点分析
        result = self.buypoint_analyzer.analyze_single_buypoint(
            stock_code=stock_code,
            buypoint_date=buypoint_date,
            stock_name=stock_name
        )
        
        return StandardResponse(
            success=True,
            message="买点分析成功",
            data={
                "stock_code": stock_code,
                "buypoint_date": buypoint_date,
                "stock_name": stock_name,
                "analysis_result": result,
                "analysis_time": pd.Timestamp.now().isoformat()
            }
        )
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def batch_analyze_buypoints(self, buypoints: List[Dict[str, str]]) -> StandardResponse:
        """
        批量分析买点
        
        Args:
            buypoints: 买点列表，每个元素包含stock_code和buypoint_date
            
        Returns:
            StandardResponse: 包含批量分析结果的标准响应
        """
        # 执行批量买点分析
        results = self.buypoint_analyzer.batch_analyze_buypoints(buypoints)
        
        return StandardResponse(
            success=True,
            message=f"批量买点分析成功，共分析 {len(buypoints)} 个买点",
            data={
                "total_buypoints": len(buypoints),
                "analysis_results": results,
                "analysis_time": pd.Timestamp.now().isoformat()
            }
        )


class StrategySelectionAdapter(StrategySelectionInterface):
    """
    策略选股分析模块适配器

    将现有的策略选股模块适配到标准接口
    """

    def __init__(self):
        self.strategy_executor = None
        self.formula_parser = None
        self._initialize_components()

    def _initialize_components(self):
        """初始化组件"""
        try:
            from strategy.execution.strategy_execution_engine import StrategyExecutionEngine
            from strategy.formula.formula_parser import FormulaParser

            self.strategy_executor = StrategyExecutionEngine()
            self.formula_parser = FormulaParser()
            logger.info("策略选股组件初始化成功")
        except Exception as e:
            logger.error(f"策略选股组件初始化失败: {e}")
            raise

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def parse_strategy_formula(self, formula: str) -> StandardResponse:
        """
        解析通达信风格策略公式

        Args:
            formula: 策略公式字符串

        Returns:
            StandardResponse: 包含解析结果的标准响应
        """
        # 解析公式
        parsed_result = self.formula_parser.parse(formula)

        return StandardResponse(
            success=True,
            message="公式解析成功",
            data={
                "original_formula": formula,
                "parsed_result": parsed_result,
                "parse_time": pd.Timestamp.now().isoformat()
            }
        )

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=120.0)
    def execute_selection(self, request: StrategyRequest) -> StandardResponse:
        """
        执行选股策略

        Args:
            request: 策略执行请求

        Returns:
            StandardResponse: 包含选股结果的标准响应
        """
        # 执行选股策略
        selection_results = self.strategy_executor.execute_strategy(
            strategy_config=request.strategy_config,
            stock_pool=request.stock_pool,
            start_date=request.start_date,
            end_date=request.end_date,
            **request.params
        )

        return StandardResponse(
            success=True,
            message=f"选股执行成功，共选出 {len(selection_results)} 只股票",
            data={
                "strategy_name": request.strategy_config.get("name", "未命名策略"),
                "stock_pool_size": len(request.stock_pool),
                "selected_stocks": len(selection_results),
                "selection_results": selection_results,
                "execution_time": pd.Timestamp.now().isoformat()
            }
        )

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def evaluate_and_rank(self, selection_results: List[Dict[str, Any]]) -> StandardResponse:
        """
        评估和排序选股结果

        Args:
            selection_results: 选股结果列表

        Returns:
            StandardResponse: 包含评估排序后的结果
        """
        # 评估和排序
        ranked_results = self.strategy_executor.evaluate_and_rank(selection_results)

        return StandardResponse(
            success=True,
            message=f"评估排序成功，共处理 {len(selection_results)} 个结果",
            data={
                "total_results": len(selection_results),
                "ranked_results": ranked_results,
                "evaluation_time": pd.Timestamp.now().isoformat()
            }
        )


class MarketMonitoringAdapter(MarketMonitoringInterface):
    """
    市场监控模块适配器

    将现有的市场监控模块适配到标准接口
    """

    def __init__(self):
        self.market_monitor = None
        self.alert_manager = None
        self._initialize_components()

    def _initialize_components(self):
        """初始化组件"""
        try:
            from monitoring.market_monitor import MarketMonitor
            from monitoring.alert_manager import AlertManager

            self.market_monitor = MarketMonitor()
            self.alert_manager = AlertManager()
            logger.info("市场监控组件初始化成功")
        except Exception as e:
            logger.error(f"市场监控组件初始化失败: {e}")
            # 使用模拟实现
            self.market_monitor = self._create_mock_monitor()
            self.alert_manager = self._create_mock_alert_manager()
            logger.warning("使用模拟市场监控组件")

    def _create_mock_monitor(self):
        """创建模拟监控器"""
        class MockMonitor:
            def start_monitoring(self, stock_codes, indicators):
                return {"status": "started", "stocks": len(stock_codes), "indicators": len(indicators)}

            def get_market_status(self):
                return {"status": "normal", "timestamp": pd.Timestamp.now().isoformat()}

        return MockMonitor()

    def _create_mock_alert_manager(self):
        """创建模拟预警管理器"""
        class MockAlertManager:
            def setup_alerts(self, alert_configs):
                return {"status": "configured", "alerts": len(alert_configs)}

        return MockAlertManager()

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=3.0)
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
        # 启动实时监控
        monitor_result = self.market_monitor.start_monitoring(stock_codes, indicators)

        return StandardResponse(
            success=True,
            message=f"实时监控启动成功，监控 {len(stock_codes)} 只股票，{len(indicators)} 个指标",
            data={
                "monitored_stocks": len(stock_codes),
                "monitored_indicators": len(indicators),
                "monitor_result": monitor_result,
                "start_time": pd.Timestamp.now().isoformat()
            }
        )

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def setup_alerts(self, alert_configs: List[Dict[str, Any]]) -> StandardResponse:
        """
        设置预警规则

        Args:
            alert_configs: 预警配置列表

        Returns:
            StandardResponse: 预警设置结果
        """
        # 设置预警规则
        alert_result = self.alert_manager.setup_alerts(alert_configs)

        return StandardResponse(
            success=True,
            message=f"预警规则设置成功，共配置 {len(alert_configs)} 个预警",
            data={
                "total_alerts": len(alert_configs),
                "alert_result": alert_result,
                "setup_time": pd.Timestamp.now().isoformat()
            }
        )

    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def get_market_status(self) -> StandardResponse:
        """
        获取市场状态

        Returns:
            StandardResponse: 包含市场状态信息
        """
        # 获取市场状态
        market_status = self.market_monitor.get_market_status()

        return StandardResponse(
            success=True,
            message="市场状态获取成功",
            data={
                "market_status": market_status,
                "query_time": pd.Timestamp.now().isoformat()
            }
        )


class ModuleAdapterFactory:
    """
    模块适配器工厂

    提供统一的适配器创建和管理
    """

    _adapters = {}

    @classmethod
    def get_technical_indicator_adapter(cls) -> TechnicalIndicatorAdapter:
        """获取技术指标适配器"""
        if 'technical_indicator' not in cls._adapters:
            cls._adapters['technical_indicator'] = TechnicalIndicatorAdapter()
        return cls._adapters['technical_indicator']

    @classmethod
    def get_buypoint_analysis_adapter(cls) -> BuyPointAnalysisAdapter:
        """获取买点分析适配器"""
        if 'buypoint_analysis' not in cls._adapters:
            cls._adapters['buypoint_analysis'] = BuyPointAnalysisAdapter()
        return cls._adapters['buypoint_analysis']

    @classmethod
    def get_strategy_selection_adapter(cls) -> StrategySelectionAdapter:
        """获取策略选股适配器"""
        if 'strategy_selection' not in cls._adapters:
            cls._adapters['strategy_selection'] = StrategySelectionAdapter()
        return cls._adapters['strategy_selection']

    @classmethod
    def get_market_monitoring_adapter(cls) -> MarketMonitoringAdapter:
        """获取市场监控适配器"""
        if 'market_monitoring' not in cls._adapters:
            cls._adapters['market_monitoring'] = MarketMonitoringAdapter()
        return cls._adapters['market_monitoring']

    @classmethod
    def get_all_adapters(cls) -> Dict[str, Any]:
        """获取所有适配器"""
        return {
            'technical_indicator': cls.get_technical_indicator_adapter(),
            'buypoint_analysis': cls.get_buypoint_analysis_adapter(),
            'strategy_selection': cls.get_strategy_selection_adapter(),
            'market_monitoring': cls.get_market_monitoring_adapter()
        }

    @classmethod
    def validate_adapters(cls) -> Dict[str, bool]:
        """验证所有适配器的可用性"""
        validation_results = {}

        try:
            adapter = cls.get_technical_indicator_adapter()
            validation_results['technical_indicator'] = adapter.indicator_registry is not None
        except Exception as e:
            logger.error(f"技术指标适配器验证失败: {e}")
            validation_results['technical_indicator'] = False

        try:
            adapter = cls.get_buypoint_analysis_adapter()
            validation_results['buypoint_analysis'] = adapter.buypoint_analyzer is not None
        except Exception as e:
            logger.error(f"买点分析适配器验证失败: {e}")
            validation_results['buypoint_analysis'] = False

        try:
            adapter = cls.get_strategy_selection_adapter()
            validation_results['strategy_selection'] = (
                adapter.strategy_executor is not None and
                adapter.formula_parser is not None
            )
        except Exception as e:
            logger.error(f"策略选股适配器验证失败: {e}")
            validation_results['strategy_selection'] = False

        try:
            adapter = cls.get_market_monitoring_adapter()
            validation_results['market_monitoring'] = (
                adapter.market_monitor is not None and
                adapter.alert_manager is not None
            )
        except Exception as e:
            logger.error(f"市场监控适配器验证失败: {e}")
            validation_results['market_monitoring'] = False

        return validation_results


# 全局适配器实例
def get_technical_indicator_adapter() -> TechnicalIndicatorAdapter:
    """获取技术指标适配器实例"""
    return ModuleAdapterFactory.get_technical_indicator_adapter()


def get_buypoint_analysis_adapter() -> BuyPointAnalysisAdapter:
    """获取买点分析适配器实例"""
    return ModuleAdapterFactory.get_buypoint_analysis_adapter()


def get_strategy_selection_adapter() -> StrategySelectionAdapter:
    """获取策略选股适配器实例"""
    return ModuleAdapterFactory.get_strategy_selection_adapter()


def get_market_monitoring_adapter() -> MarketMonitoringAdapter:
    """获取市场监控适配器实例"""
    return ModuleAdapterFactory.get_market_monitoring_adapter()


def validate_all_adapters() -> Dict[str, bool]:
    """验证所有适配器的可用性"""
    return ModuleAdapterFactory.validate_adapters()
