"""
模块协同管理器

负责协调四大模块间的协同工作，确保数据流向正确和接口调用规范
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from interfaces.module_adapters import (
    get_technical_indicator_adapter,
    get_buypoint_analysis_adapter,
    get_strategy_selection_adapter,
    get_market_monitoring_adapter,
    validate_all_adapters
)
from interfaces.module_interfaces import (
    StandardResponse,
    IndicatorRequest,
    PatternRequest,
    StrategyRequest
)
from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class ModuleCoordinator:
    """
    模块协同管理器
    
    负责协调四大模块间的协同工作：
    1. 技术指标分析模块 → 策略选股分析模块
    2. 买点回测分析模块 → 策略选股分析模块  
    3. 市场监控模块 → 策略选股分析模块
    4. 策略选股分析模块 → 市场监控模块
    """
    
    def __init__(self):
        self.technical_adapter = get_technical_indicator_adapter()
        self.buypoint_adapter = get_buypoint_analysis_adapter()
        self.strategy_adapter = get_strategy_selection_adapter()
        self.monitoring_adapter = get_market_monitoring_adapter()
        
        self._validate_initialization()
    
    def _validate_initialization(self):
        """验证初始化状态"""
        validation_results = validate_all_adapters()
        
        failed_modules = [module for module, status in validation_results.items() if not status]
        if failed_modules:
            logger.warning(f"以下模块初始化失败: {failed_modules}")
        else:
            logger.info("所有模块适配器初始化成功")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=300.0)
    def execute_comprehensive_analysis(self, stock_code: str, analysis_date: str, 
                                     strategy_config: Dict[str, Any]) -> StandardResponse:
        """
        执行综合分析工作流
        
        整合四大模块的能力，提供完整的股票分析服务
        
        Args:
            stock_code: 股票代码
            analysis_date: 分析日期
            strategy_config: 策略配置
            
        Returns:
            StandardResponse: 综合分析结果
        """
        logger.info(f"开始执行股票 {stock_code} 的综合分析")
        
        analysis_results = {
            "stock_code": stock_code,
            "analysis_date": analysis_date,
            "technical_indicators": {},
            "pattern_analysis": {},
            "buypoint_analysis": {},
            "strategy_signals": {},
            "market_status": {}
        }
        
        # 1. 技术指标分析
        try:
            # 获取核心技术指标
            core_indicators = ['RSI', 'MACD', 'KDJ', 'BOLL', 'MA']
            for indicator in core_indicators:
                indicator_request = IndicatorRequest(
                    indicator_name=indicator,
                    stock_code=stock_code,
                    period='日线',
                    data=None,  # 实际使用时需要提供数据
                    params={}
                )
                
                result = self.technical_adapter.calculate_indicator(indicator_request)
                if result.success:
                    analysis_results["technical_indicators"][indicator] = result.data
                else:
                    logger.warning(f"指标 {indicator} 计算失败: {result.message}")
            
            logger.info(f"技术指标分析完成，成功计算 {len(analysis_results['technical_indicators'])} 个指标")
            
        except Exception as e:
            logger.error(f"技术指标分析失败: {e}")
            analysis_results["technical_indicators"]["error"] = str(e)
        
        # 2. 形态识别分析
        try:
            # 识别关键形态
            for indicator in core_indicators:
                pattern_request = PatternRequest(
                    indicator_name=indicator,
                    stock_code=stock_code,
                    period='日线',
                    data=None,  # 实际使用时需要提供数据
                    pattern_name=None  # 获取所有形态
                )
                
                result = self.technical_adapter.detect_pattern(pattern_request)
                if result.success:
                    analysis_results["pattern_analysis"][indicator] = result.data
                else:
                    logger.warning(f"指标 {indicator} 形态识别失败: {result.message}")
            
            logger.info(f"形态识别分析完成，成功识别 {len(analysis_results['pattern_analysis'])} 个指标的形态")
            
        except Exception as e:
            logger.error(f"形态识别分析失败: {e}")
            analysis_results["pattern_analysis"]["error"] = str(e)
        
        # 3. 买点分析（如果是历史买点）
        try:
            buypoint_result = self.buypoint_adapter.analyze_buypoint(
                stock_code=stock_code,
                buypoint_date=analysis_date
            )
            
            if buypoint_result.success:
                analysis_results["buypoint_analysis"] = buypoint_result.data
                logger.info("买点分析完成")
            else:
                logger.warning(f"买点分析失败: {buypoint_result.message}")
                analysis_results["buypoint_analysis"]["error"] = buypoint_result.message
                
        except Exception as e:
            logger.error(f"买点分析失败: {e}")
            analysis_results["buypoint_analysis"]["error"] = str(e)
        
        # 4. 策略信号生成
        try:
            strategy_request = StrategyRequest(
                strategy_config=strategy_config,
                stock_pool=[stock_code],
                start_date=analysis_date,
                end_date=analysis_date,
                params={}
            )
            
            strategy_result = self.strategy_adapter.execute_selection(strategy_request)
            
            if strategy_result.success:
                analysis_results["strategy_signals"] = strategy_result.data
                logger.info("策略信号生成完成")
            else:
                logger.warning(f"策略信号生成失败: {strategy_result.message}")
                analysis_results["strategy_signals"]["error"] = strategy_result.message
                
        except Exception as e:
            logger.error(f"策略信号生成失败: {e}")
            analysis_results["strategy_signals"]["error"] = str(e)
        
        # 5. 市场状态监控
        try:
            market_result = self.monitoring_adapter.get_market_status()
            
            if market_result.success:
                analysis_results["market_status"] = market_result.data
                logger.info("市场状态获取完成")
            else:
                logger.warning(f"市场状态获取失败: {market_result.message}")
                analysis_results["market_status"]["error"] = market_result.message
                
        except Exception as e:
            logger.error(f"市场状态获取失败: {e}")
            analysis_results["market_status"]["error"] = str(e)
        
        # 6. 综合评估
        comprehensive_score = self._calculate_comprehensive_score(analysis_results)
        analysis_results["comprehensive_score"] = comprehensive_score
        
        logger.info(f"股票 {stock_code} 综合分析完成，综合评分: {comprehensive_score}")
        
        return StandardResponse(
            success=True,
            message=f"股票 {stock_code} 综合分析完成",
            data=analysis_results
        )
    
    def _calculate_comprehensive_score(self, analysis_results: Dict[str, Any]) -> float:
        """
        计算综合评分
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            float: 综合评分 (0-100)
        """
        score = 0.0
        weight_sum = 0.0
        
        # 技术指标权重 30%
        if analysis_results["technical_indicators"] and "error" not in analysis_results["technical_indicators"]:
            score += 30.0
            weight_sum += 30.0
        
        # 形态分析权重 25%
        if analysis_results["pattern_analysis"] and "error" not in analysis_results["pattern_analysis"]:
            score += 25.0
            weight_sum += 25.0
        
        # 买点分析权重 20%
        if analysis_results["buypoint_analysis"] and "error" not in analysis_results["buypoint_analysis"]:
            score += 20.0
            weight_sum += 20.0
        
        # 策略信号权重 20%
        if analysis_results["strategy_signals"] and "error" not in analysis_results["strategy_signals"]:
            score += 20.0
            weight_sum += 20.0
        
        # 市场状态权重 5%
        if analysis_results["market_status"] and "error" not in analysis_results["market_status"]:
            score += 5.0
            weight_sum += 5.0
        
        # 计算加权平均分
        if weight_sum > 0:
            return (score / weight_sum) * 100
        else:
            return 0.0
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=600.0)
    def execute_batch_stock_selection(self, stock_pool: List[str], 
                                    strategy_config: Dict[str, Any],
                                    analysis_date: str) -> StandardResponse:
        """
        执行批量选股分析
        
        Args:
            stock_pool: 股票池
            strategy_config: 策略配置
            analysis_date: 分析日期
            
        Returns:
            StandardResponse: 批量选股结果
        """
        logger.info(f"开始执行批量选股，股票池大小: {len(stock_pool)}")
        
        # 1. 启动市场监控
        monitoring_result = self.monitoring_adapter.start_real_time_monitoring(
            stock_codes=stock_pool,
            indicators=['RSI', 'MACD', 'KDJ']
        )
        
        if not monitoring_result.success:
            logger.warning(f"市场监控启动失败: {monitoring_result.message}")
        
        # 2. 执行策略选股
        strategy_request = StrategyRequest(
            strategy_config=strategy_config,
            stock_pool=stock_pool,
            start_date=analysis_date,
            end_date=analysis_date,
            params={}
        )
        
        selection_result = self.strategy_adapter.execute_selection(strategy_request)
        
        if not selection_result.success:
            return StandardResponse(
                success=False,
                message=f"批量选股失败: {selection_result.message}",
                data=None
            )
        
        # 3. 评估和排序结果
        ranking_result = self.strategy_adapter.evaluate_and_rank(
            selection_result.data.get("selection_results", [])
        )
        
        if not ranking_result.success:
            logger.warning(f"结果排序失败: {ranking_result.message}")
        
        logger.info(f"批量选股完成，共选出 {len(selection_result.data.get('selection_results', []))} 只股票")
        
        return StandardResponse(
            success=True,
            message=f"批量选股完成，共选出 {len(selection_result.data.get('selection_results', []))} 只股票",
            data={
                "stock_pool_size": len(stock_pool),
                "selection_results": selection_result.data,
                "ranking_results": ranking_result.data if ranking_result.success else None,
                "monitoring_status": monitoring_result.data if monitoring_result.success else None,
                "analysis_date": analysis_date,
                "execution_time": datetime.now().isoformat()
            }
        )


# 全局协调器实例
_coordinator_instance = None


def get_module_coordinator() -> ModuleCoordinator:
    """获取模块协调器实例"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = ModuleCoordinator()
    return _coordinator_instance
