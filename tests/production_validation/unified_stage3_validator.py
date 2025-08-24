#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一阶段3真实数据验证器

集成现有的生产系统进行真实数据双向验证：
1. 使用strategy_executor框架进行选股
2. 使用buypoint_analyzer进行买点分析
3. 使用ClickHouse数据管道获取真实数据
4. 实现端到端的生产级验证

这个统一框架可以验证任何技术指标，避免为每个指标创建单独的验证脚本
"""

import sys
import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

# 导入现有的生产系统组件
from strategy.strategy_executor import StrategyExecutor
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from analysis.enhanced_closed_loop_validator import EnhancedClosedLoopValidator
from db.unified_data_manager import get_unified_data_manager
from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.decorators import performance_monitor, exception_handler
from utils.logger import get_logger

logger = get_logger(__name__)

class UnifiedStage3Validator:
    """
    统一阶段3真实数据验证器
    
    集成现有生产系统，实现任何技术指标的真实数据双向验证
    """
    
    def __init__(self):
        """初始化统一验证器"""
        logger.info("🌐 初始化统一阶段3验证器")
        
        # 初始化现有生产系统组件
        self.strategy_executor = StrategyExecutor()
        self.buypoint_analyzer = BuyPointAnalyzer()
        self.closed_loop_validator = EnhancedClosedLoopValidator()
        self.data_manager = get_unified_data_manager()
        self.data_access = get_service(DataAccessInterface)
        
        # 指标注册表
        self.indicator_registry = complete_registry
        
        # 严格的生产级验证配置
        self.validation_config = {
            'min_stocks_required': 1,           # 每个形态最少选出股票数
            'max_stocks_to_analyze': 10,        # 最多分析股票数
            'lookback_days': 90,                # 扩展回看天数以增加找到形态的概率
            'consistency_threshold': 1.0,       # 100%一致性要求
            'simulated_accuracy_threshold': 1.0, # 100%模拟数据准确率要求
            'false_positive_tolerance': 0,      # 0容忍假阳性
            'false_negative_tolerance': 0,      # 0容忍假阴性
            'timeout_seconds': 600              # 增加超时时间（10分钟）
        }
        
        logger.info("✅ 统一阶段3验证器初始化完成")
    
    @performance_monitor(threshold=300.0)
    @exception_handler(reraise=True)
    def validate_indicator_stage3(self, indicator_name: str) -> Dict[str, Any]:
        """
        验证指定技术指标的阶段3真实数据双向验证
        
        Args:
            indicator_name: 技术指标名称 (如: 'MACD', 'RSI', 'KDJ')
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        start_time = time.time()
        
        validation_result = {
            'indicator_name': indicator_name,
            'validation_timestamp': datetime.now().isoformat(),
            'stage3_success': False,
            'database_connection': False,
            'stock_selection_result': None,
            'buypoint_analysis_result': None,
            'bidirectional_verification': None,
            'performance_metrics': {},
            'issues_found': [],
            'selected_stocks': []
        }
        
        try:
            logger.info(f"🚀 开始 {indicator_name} 阶段3真实数据验证")
            
            # 步骤1: 验证数据库连接和数据可用性
            logger.info("📊 步骤1: 验证数据库连接")
            db_status = self._verify_database_connection()
            validation_result['database_connection'] = db_status['success']
            
            if not db_status['success']:
                validation_result['issues_found'].append(f"数据库连接失败: {db_status['message']}")
                return validation_result
            
            logger.info(f"✅ 数据库连接成功: {db_status['message']}")
            
            # 步骤2: 使用strategy_executor进行选股
            logger.info("🔍 步骤2: 执行生产级选股")
            selection_result = self._execute_production_stock_selection(indicator_name)
            validation_result['stock_selection_result'] = selection_result
            
            if not selection_result['success']:
                validation_result['issues_found'].extend(selection_result['issues'])
                return validation_result
            
            selected_stocks = selection_result['selected_stocks']
            validation_result['selected_stocks'] = selected_stocks
            logger.info(f"✅ 选股成功: 选出 {len(selected_stocks)} 支股票")
            
            # 步骤3: 使用buypoint_analyzer进行买点分析
            logger.info("📈 步骤3: 执行生产级买点分析")
            buypoint_result = self._execute_production_buypoint_analysis(selected_stocks, indicator_name)
            validation_result['buypoint_analysis_result'] = buypoint_result
            
            if not buypoint_result['success']:
                validation_result['issues_found'].extend(buypoint_result['issues'])
                return validation_result
            
            logger.info(f"✅ 买点分析成功")
            
            # 步骤4: 使用closed_loop_validator进行双向验证
            logger.info("🔄 步骤4: 执行生产级双向验证")
            verification_result = self._execute_production_bidirectional_verification(
                selected_stocks, buypoint_result, indicator_name
            )
            validation_result['bidirectional_verification'] = verification_result
            
            if verification_result['success']:
                validation_result['stage3_success'] = True
                logger.info(f"🎉 {indicator_name} 阶段3验证成功！")
            else:
                validation_result['issues_found'].extend(verification_result['issues'])
                logger.warning(f"⚠️ {indicator_name} 阶段3验证部分失败")
            
            # 记录性能指标
            validation_result['performance_metrics'] = {
                'total_execution_time': time.time() - start_time,
                'database_query_time': db_status.get('query_time', 0),
                'selection_time': selection_result.get('execution_time', 0),
                'buypoint_analysis_time': buypoint_result.get('execution_time', 0),
                'verification_time': verification_result.get('execution_time', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ {indicator_name} 阶段3验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _verify_database_connection(self) -> Dict[str, Any]:
        """验证数据库连接和数据可用性"""
        
        result = {
            'success': False,
            'message': '',
            'stock_count': 0,
            'query_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # 测试基本连接
            if not self.data_manager.test_connection():
                result['message'] = "数据管理器连接测试失败"
                return result
            
            # 查询股票数量和最新数据
            query = """
            SELECT 
                COUNT(DISTINCT code) as stock_count,
                MAX(date) as latest_date
            FROM stock_info 
            WHERE date >= today() - 30
            AND level = '日线'
            """
            
            query_result = self.data_manager.query_Manager_Unified_Data_Manager(query)
            
            if query_result is not None and len(query_result) > 0:
                stock_count = query_result.iloc[0]['stock_count']
                latest_date = query_result.iloc[0]['latest_date']
                
                result['stock_count'] = stock_count
                result['query_time'] = time.time() - start_time
                
                if stock_count >= 1000:  # 至少1000只股票
                    result['success'] = True
                    result['message'] = f"数据库正常，共 {stock_count} 只股票，最新数据: {latest_date}"
                else:
                    result['message'] = f"股票数量不足: {stock_count} < 1000"
            else:
                result['message'] = "查询结果为空"
        
        except Exception as e:
            result['message'] = f"数据库验证异常: {str(e)}"
            result['query_time'] = time.time() - start_time
        
        return result
    
    def _execute_production_stock_selection(self, indicator_name: str) -> Dict[str, Any]:
        """使用生产级strategy_executor执行选股"""
        
        result = {
            'success': False,
            'selected_stocks': [],
            'issues': [],
            'execution_time': 0.0,
            'strategy_config': {}
        }
        
        start_time = time.time()
        
        try:
            # 构建基于指标的选股策略配置
            strategy_config = self._build_indicator_strategy_config(indicator_name)
            result['strategy_config'] = strategy_config
            
            # 使用strategy_executor执行选股
            execution_result = self.strategy_executor.execute_unified_strategy(
                strategy_config=strategy_config,
                enable_validation=True,
                enable_closed_loop=False  # 我们自己做闭环验证
            )
            
            if execution_result and execution_result.get('selection_result'):
                selected_stocks = execution_result['selection_result']
                
                # 限制分析的股票数量
                max_stocks = self.validation_config['max_stocks_to_analyze']
                if len(selected_stocks) > max_stocks:
                    selected_stocks = selected_stocks[:max_stocks]
                
                if len(selected_stocks) >= self.validation_config['min_stocks_required']:
                    result['success'] = True
                    result['selected_stocks'] = selected_stocks
                    logger.info(f"   📊 选股策略: {strategy_config['strategy']['id']}")
                    logger.info(f"   📈 选出股票: {[stock.get('code', stock.get('stock_code', 'N/A')) for stock in selected_stocks[:5]]}")
                else:
                    result['issues'].append(f"选出股票数量不足: {len(selected_stocks)} < {self.validation_config['min_stocks_required']}")
            else:
                result['issues'].append("strategy_executor未返回有效选股结果")
        
        except Exception as e:
            result['issues'].append(f"选股执行异常: {str(e)}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _build_indicator_strategy_config(self, indicator_name: str) -> Dict[str, Any]:
        """构建基于指标的选股策略配置"""
        
        # 基础策略配置模板
        strategy_config = {
            'strategy': {
                'id': f'{indicator_name.lower()}_validation_strategy',
                'name': f'{indicator_name}指标验证策略',
                'description': f'用于{indicator_name}指标阶段3验证的选股策略'
            },
            'conditions': {
                'primary': {
                    f'{indicator_name.lower()}_pattern': {
                        'indicator': indicator_name,
                        'operator': 'has_pattern',
                        'value': 'any'  # 任何形态
                    }
                }
            },
            'filters': {
                'market_cap': {'min': 1000000000},  # 最小市值10亿
                'volume': {'min': 1000000},         # 最小成交量100万
                'price': {'min': 5.0, 'max': 200.0}  # 价格范围
            },
            'output': {
                'max_stocks': self.validation_config['max_stocks_to_analyze'],
                'sort_by': 'volume',
                'sort_order': 'desc'
            },
            'validation': {
                'validation_method': 'indicator_consistency'
            }
        }
        
        return strategy_config
    
    def _execute_production_buypoint_analysis(self, selected_stocks: List[Dict[str, Any]], 
                                            indicator_name: str) -> Dict[str, Any]:
        """使用生产级buypoint_analyzer执行买点分析"""
        
        result = {
            'success': False,
            'analysis_results': {},
            'issues': [],
            'execution_time': 0.0,
            'analyzed_stocks': 0
        }
        
        start_time = time.time()
        
        try:
            analysis_results = {}
            analyzed_count = 0
            
            # 获取最近的交易日期
            latest_date = self._get_latest_trading_date()
            
            for stock_info in selected_stocks:
                stock_code = stock_info.get('code', stock_info.get('stock_code'))
                stock_name = stock_info.get('name', stock_info.get('stock_name', ''))
                
                if not stock_code:
                    continue
                
                try:
                    # 使用buypoint_analyzer分析
                    buypoint_result = self.buypoint_analyzer.analyze_stock(
                        stock_code=stock_code,
                        buy_date=latest_date.strftime('%Y%m%d'),
                        stock_name=stock_name
                    )
                    
                    if buypoint_result:
                        analysis_results[stock_code] = buypoint_result
                        analyzed_count += 1
                        
                except Exception as e:
                    logger.warning(f"⚠️ 分析股票 {stock_code} 失败: {e}")
                    continue
            
            if analyzed_count > 0:
                result['success'] = True
                result['analysis_results'] = analysis_results
                result['analyzed_stocks'] = analyzed_count
                
                # 打印分析摘要
                for stock_code, analysis in list(analysis_results.items())[:3]:  # 显示前3只
                    score = analysis.get('score', 0)
                    logger.info(f"   📊 {stock_code}: 买点评分 {score}")
            else:
                result['issues'].append("买点分析未产生有效结果")
        
        except Exception as e:
            result['issues'].append(f"买点分析异常: {str(e)}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _execute_production_bidirectional_verification(self, selected_stocks: List[Dict[str, Any]], 
                                                     buypoint_result: Dict[str, Any], 
                                                     indicator_name: str) -> Dict[str, Any]:
        """使用生产级closed_loop_validator执行双向验证"""
        
        result = {
            'success': False,
            'verification_results': {},
            'issues': [],
            'execution_time': 0.0,
            'consistency_rate': 0.0
        }
        
        start_time = time.time()
        
        try:
            # 构建验证用的策略配置
            strategy_config = self._build_indicator_strategy_config(indicator_name)
            
            # 使用enhanced_closed_loop_validator进行验证
            validation_result = self.closed_loop_validator.validate_strategy_selection(
                selection_results=selected_stocks,
                strategy_config=strategy_config,
                validation_method='indicator_consistency'
            )
            
            if validation_result and validation_result.get('validation_success'):
                consistency_rate = validation_result.get('consistency_rate', 0.0)
                
                if consistency_rate >= self.validation_config['consistency_threshold']:
                    result['success'] = True
                    result['consistency_rate'] = consistency_rate
                    result['verification_results'] = validation_result
                    
                    logger.info(f"   ✅ 双向验证成功率: {consistency_rate:.1%}")
                else:
                    result['issues'].append(f"一致性率过低: {consistency_rate:.1%} < {self.validation_config['consistency_threshold']:.1%}")
            else:
                result['issues'].append("闭环验证器未返回有效结果")
        
        except Exception as e:
            result['issues'].append(f"双向验证异常: {str(e)}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _get_latest_trading_date(self) -> datetime:
        """获取最新交易日期"""
        try:
            query = "SELECT MAX(date) as latest_date FROM stock_info WHERE level = '日线'"
            result = self.data_manager.query_Manager_Unified_Data_Manager(query)
            
            if result is not None and len(result) > 0:
                latest_date = pd.to_datetime(result.iloc[0]['latest_date'])
                return latest_date
            else:
                # 如果查询失败，返回最近的工作日
                today = datetime.now()
                while today.weekday() >= 5:  # 周末
                    today -= timedelta(days=1)
                return today
        except Exception as e:
            logger.warning(f"⚠️ 获取最新交易日期失败: {e}")
            # 返回最近的工作日
            today = datetime.now()
            while today.weekday() >= 5:  # 周末
                today -= timedelta(days=1)
            return today

def main():
    """主函数 - 测试统一验证器"""
    validator = UnifiedStage3Validator()
    
    # 测试MACD指标的阶段3验证
    print("🚀 测试MACD指标阶段3验证")
    results = validator.validate_indicator_stage3('MACD')
    
    print("\n" + "="*80)
    print("🎯 MACD阶段3验证结果汇总")
    print("="*80)
    
    print(f"🌐 整体成功: {results['stage3_success']}")
    print(f"📊 数据库连接: {results['database_connection']}")
    
    if results['selected_stocks']:
        print(f"📈 选出股票: {len(results['selected_stocks'])} 支")
    
    if results['bidirectional_verification']:
        verification = results['bidirectional_verification']
        consistency_rate = verification.get('consistency_rate', 0)
        print(f"🔄 双向验证一致性: {consistency_rate:.1%}")
    
    if results['performance_metrics']:
        metrics = results['performance_metrics']
        print(f"⏱️ 总执行时间: {metrics['total_execution_time']:.2f}秒")
    
    if results['issues_found']:
        print(f"\n⚠️ 发现问题:")
        for issue in results['issues_found']:
            print(f"   - {issue}")
    
    if results['stage3_success']:
        print(f"\n🎉 MACD指标阶段3真实数据验证成功！")
        print(f"✅ 已完成MACD指标的完整生产级验证（阶段1+2+3）")
    else:
        print(f"\n❌ MACD指标阶段3验证需要进一步完善")

if __name__ == "__main__":
    main()
