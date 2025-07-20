#!/usr/bin/env python3
"""
统一分析引擎

整合买点分析和策略选股系统，提供统一的分析接口和结果处理。
实现系统集成重构的核心组件。
"""

import sys
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.dependency_injection import get_logger
from analysis.integration.unified_data_adapter import get_unified_data_adapter
from analysis.buypoints.buypoint_batch_analyzer import Buy_point_batch_analyzer
from strategy.strategy_factory import Strategy_factory
from utils.decorators import safe_run, timing_decorator
from utils.path_utils import ensure_dir_exists
from utils.dependency_injection import get_service

logger = getLogger(__name__)


class UnifiedAnalysisEngine:
    """统一分析引擎"""
    
    def __init___109(self, max_workers: int = 4):
        """
        初始化统一分析引擎
        
        Args:
            max_workers: 最大并发工作线程数
        """
        self.data_adapter = get_unified_data_adapter()
        self.buypoint_analyzer = Buy_point_batch_analyzer()
        self.strategy_factory = Strategy_factory()
        self.max_workers = max_workers
        
        # 分析配置
        self.analysis_config = {
            'enable_buypoint_analysis': True,
            'enable_strategy_selection': True,
            'enable_result_merge': True,
            'min_score_threshold': 50.0,
            'max_results_per_analysis': 100,
            'analysis_timeout': 300,  # 5分钟超时
            'enable_parallel_processing': True
        }
        
        # 性能统计
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0,
            'last_analysis_time': None
        }
        
        logger.info("统一分析引擎初始化完成")
    
    @timing_decorator
    @safe_run
    def analyze_stocks(self, 
                      stock_codes: List[str], 
                      analysis_date: Optional[str] = None,
                      analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        分析股票列表
        
        Args:
            stock_codes: 股票代码列表
            analysis_date: 分析日期，默认为当前日期
            analysis_type: 分析类型 ('buypoint', 'strategy', 'comprehensive')
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            if not stock_codes:
                logger.warning("股票代码列表为空")
                return self._create_empty_result()
            
            # 设置分析日期
            if not analysis_date:
                analysis_date = datetime.now().strftime('%Y%m%d')
            
            logger.info(f"开始分析 {len(stock_codes)} 只股票，分析类型: {analysis_type}")
            
            # 根据分析类型执行不同的分析流程
            if analysis_type == "buypoint":
                return self._analyze_buypoints_only(stock_codes, analysis_date)
            elif analysis_type == "strategy":
                return self._analyze_strategies_only(stock_codes, analysis_date)
            elif analysis_type == "comprehensive":
                return self._analyze_comprehensive(stock_codes, analysis_date)
            else:
                logger.error(f"不支持的分析类型: {analysis_type}")
                return self._create_empty_result()
                
        except Exception as e:
            logger.error(f"分析股票时出错: {e}")
            self.performance_stats['failed_analyses'] += 1
            return self._create_empty_result()
    
    @timing_decorator
    @safe_run
    def analyze_from_csv(self, 
                        csv_file: str, 
                        analysis_type: str = "comprehensive",
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        从CSV文件分析股票
        
        Args:
            csv_file: CSV文件路径
            analysis_type: 分析类型
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 读取CSV文件
            if not os.path.exists(csv_file):
                logger.error(f"CSV文件不存在: {csv_file}")
                return self._create_empty_result()
            
            df = pd.read_csv(csv_file)
            
            # 提取股票代码
            stock_codes = []
            if 'stock_code' in df.columns:
                stock_codes = df['stock_code'].astype(str).str.zfill(6).tolist()
            elif 'code' in df.columns:
                stock_codes = df['code'].astype(str).str.zfill(6).tolist()
            else:
                logger.error("CSV文件中未找到股票代码列")
                return self._create_empty_result()
            
            # 去重
            stock_codes = list(set(stock_codes))
            
            # 分析日期
            analysis_date = None
            if 'date' in df.columns and not df['date'].empty:
                analysis_date = str(df['date'].iloc[0])
            elif 'buypoint_date' in df.columns and not df['buypoint_date'].empty:
                analysis_date = str(df['buypoint_date'].iloc[0])
            
            logger.info(f"从CSV文件读取到 {len(stock_codes)} 只股票")
            
            # 执行分析
            result = self.analyze_stocks(stock_codes, analysis_date, analysis_type)
            
            # 保存结果
            if output_dir and result['success']:
                self._save_analysis_results(result, output_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"从CSV文件分析时出错: {e}")
            return self._create_empty_result()
    
    @timing_decorator
    @safe_run
    def analyze_market_selection(self, 
                               strategy_name: str,
                               selection_count: int = 50,
                               analysis_date: Optional[str] = None) -> Dict[str, Any]:
        """
        市场选股分析
        
        Args:
            strategy_name: 策略名称
            selection_count: 选股数量
            analysis_date: 分析日期
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            if not analysis_date:
                analysis_date = datetime.now().strftime('%Y%m%d')
            
            logger.info(f"开始市场选股分析，策略: {strategy_name}，目标数量: {selection_count}")
            
            # 获取策略实例
            strategy = self.strategy_factory.create_strategy(strategy_name)
            if not strategy:
                logger.error(f"无法创建策略: {strategy_name}")
                return self._create_empty_result()
            
            # 执行选股
            selected_stocks = strategy.select_stocks(
                max_count=selection_count,
                date=analysis_date
            )
            
            if not selected_stocks:
                logger.warning("策略选股结果为空")
                return self._create_empty_result()
            
            # 提取股票代码
            stock_codes = [stock['stock_code'] for stock in selected_stocks]
            
            # 执行综合分析
            result = self.analyze_stocks(stock_codes, analysis_date, "comprehensive")
            
            # 添加策略信息
            if result['success']:
                result['strategy_info'] = {
                    'strategy_name': strategy_name,
                    'selection_count': len(selected_stocks),
                    'selection_date': analysis_date
                }
            
            return result
            
        except Exception as e:
            logger.error(f"市场选股分析时出错: {e}")
            return self._create_empty_result()
    
    def _analyze_buypoints_only(self, stock_codes: List[str], analysis_date: str) -> Dict[str, Any]:
        """仅执行买点分析"""
        try:
            logger.info("执行买点分析")
            
            # 创建临时CSV文件
            temp_csv = f"temp_buypoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df = pd.DataFrame({
                'stock_code': stock_codes,
                'buypoint_date': [analysis_date] * len(stock_codes)
            })
            df.to_csv(temp_csv, index=False)
            
            try:
                # 执行买点分析
                buypoint_results = self.buypoint_analyzer.analyze_batch_buypoints(df)
                
                # 转换为标准格式
                standard_results = self.data_adapter.batch_convert_to_standard(
                    buypoint_results, 'buypoint'
                )
                
                # 过滤低分结果
                filtered_results = [
                    result for result in standard_results 
                    if result.get('score', 0) >= self.analysis_config['min_score_threshold']
                ]
                
                # 限制结果数量
                max_results = self.analysis_config['max_results_per_analysis']
                if len(filtered_results) > max_results:
                    filtered_results = filtered_results[:max_results]
                
                return self._create_success_result(filtered_results, "buypoint")
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                    
        except Exception as e:
            logger.error(f"买点分析失败: {e}")
            return self._create_empty_result()
    
    def _analyze_strategies_only(self, stock_codes: List[str], analysis_date: str) -> Dict[str, Any]:
        """仅执行策略分析"""
        try:
            logger.info("执行策略分析")
            
            # 这里应该调用策略选股系统
            # 由于策略选股系统可能需要不同的输入格式，这里做简化处理
            strategy_results = []
            
            # 获取默认策略
            default_strategy = self.strategy_factory.create_strategy("default")
            if default_strategy:
                # 对每只股票执行策略评估
                for stock_code in stock_codes:
                    try:
                        # 这里应该调用策略的evaluate_stock方法
                        # 简化处理，创建模拟结果
                        strategy_result = {
                            'stock_code': stock_code,
                            'selection_date': analysis_date,
                            'score': 60.0,  # 模拟评分
                            'recommendation': '观望',
                            'match_details': {
                                'strategy_name': 'default',
                                'analysis_date': analysis_date
                            }
                        }
                        strategy_results.append(strategy_result)
                        
                    except Exception as e:
                        logger.error(f"策略评估股票 {stock_code} 失败: {e}")
                        continue
            
            # 转换为标准格式
            standard_results = self.data_adapter.batch_convert_to_standard(
                strategy_results, 'strategy'
            )
            
            # 过滤和限制结果
            filtered_results = [
                result for result in standard_results 
                if result.get('score', 0) >= self.analysis_config['min_score_threshold']
            ]
            
            max_results = self.analysis_config['max_results_per_analysis']
            if len(filtered_results) > max_results:
                filtered_results = filtered_results[:max_results]
            
            return self._create_success_result(filtered_results, "strategy")
            
        except Exception as e:
            logger.error(f"策略分析失败: {e}")
            return self._create_empty_result()
    
    def _analyze_comprehensive(self, stock_codes: List[str], analysis_date: str) -> Dict[str, Any]:
        """执行综合分析"""
        try:
            logger.info("执行综合分析")
            
            if self.analysis_config['enable_parallel_processing']:
                return self._analyze_comprehensive_parallel(stock_codes, analysis_date)
            else:
                return self._analyze_comprehensive_sequential(stock_codes, analysis_date)
                
        except Exception as e:
            logger.error(f"综合分析失败: {e}")
            return self._create_empty_result()
    
    def _analyze_comprehensive_parallel(self, stock_codes: List[str], analysis_date: str) -> Dict[str, Any]:
        """并行执行综合分析"""
        try:
            futures = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交买点分析任务
                if self.analysis_config['enable_buypoint_analysis']:
                    future_buypoint = executor.submit(
                        self._analyze_buypoints_only, stock_codes, analysis_date
                    )
                    futures.append(('buypoint', future_buypoint))
                
                # 提交策略分析任务
                if self.analysis_config['enable_strategy_selection']:
                    future_strategy = executor.submit(
                        self._analyze_strategies_only, stock_codes, analysis_date
                    )
                    futures.append(('strategy', future_strategy))
                
                # 收集结果
                results = {}
                for analysis_type, future in futures:
                    try:
                        results[analysis_type] = future.result(timeout=self.analysis_config['analysis_timeout'])
                    except Exception as e:
                        logger.error(f"{analysis_type} 分析超时或失败: {e}")
                        results[analysis_type] = self._create_empty_result()
            
            # 合并结果
            if self.analysis_config['enable_result_merge']:
                return self._merge_analysis_results(results)
            else:
                return results
                
        except Exception as e:
            logger.error(f"并行综合分析失败: {e}")
            return self._create_empty_result()
    
    def _analyze_comprehensive_sequential(self, stock_codes: List[str], analysis_date: str) -> Dict[str, Any]:
        """顺序执行综合分析"""
        try:
            results = {}
            
            # 执行买点分析
            if self.analysis_config['enable_buypoint_analysis']:
                results['buypoint'] = self._analyze_buypoints_only(stock_codes, analysis_date)
            
            # 执行策略分析
            if self.analysis_config['enable_strategy_selection']:
                results['strategy'] = self._analyze_strategies_only(stock_codes, analysis_date)
            
            # 合并结果
            if self.analysis_config['enable_result_merge']:
                return self._merge_analysis_results(results)
            else:
                return results
                
        except Exception as e:
            logger.error(f"顺序综合分析失败: {e}")
            return self._create_empty_result()
    
    def _merge_analysis_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """合并分析结果"""
        try:
            buypoint_results = results.get('buypoint', {}).get('data', [])
            strategy_results = results.get('strategy', {}).get('data', [])
            
            # 使用数据适配器合并结果
            merged_results = self.data_adapter.merge_analysis_results(
                buypoint_results, strategy_results
            )
            
            # 创建合并后的结果
            return {
                'success': True,
                'analysis_type': 'comprehensive',
                'analysis_date': datetime.now().strftime('%Y%m%d %H:%M:%S'),
                'total_stocks': len(merged_results),
                'data': merged_results,
                'summary': {
                    'buypoint_count': len(buypoint_results),
                    'strategy_count': len(strategy_results),
                    'merged_count': len(merged_results),
                    'average_score': sum(r.get('score', 0) for r in merged_results) / len(merged_results) if merged_results else 0,
                    'top_recommendations': [r for r in merged_results if r.get('score', 0) >= 80][:10]
                }
            }
            
        except Exception as e:
            logger.error(f"合并分析结果失败: {e}")
            return self._create_empty_result()
    
    def _create_success_result(self, data: List[Dict[str, Any]], analysis_type: str) -> Dict[str, Any]:
        """创建成功结果"""
        self.performance_stats['successful_analyses'] += 1
        self.performance_stats['total_analyses'] += 1
        self.performance_stats['last_analysis_time'] = datetime.now()
        
        return {
            'success': True,
            'analysis_type': analysis_type,
            'analysis_date': datetime.now().strftime('%Y%m%d %H:%M:%S'),
            'total_stocks': len(data),
            'data': data,
            'summary': {
                'average_score': sum(r.get('score', 0) for r in data) / len(data) if data else 0,
                'max_score': max(r.get('score', 0) for r in data) if data else 0,
                'min_score': min(r.get('score', 0) for r in data) if data else 0,
                'recommendations': {
                    '强烈买入': len([r for r in data if r.get('recommendation') == '强烈买入']),
                    '买入': len([r for r in data if r.get('recommendation') == '买入']),
                    '谨慎买入': len([r for r in data if r.get('recommendation') == '谨慎买入']),
                    '观望': len([r for r in data if r.get('recommendation') == '观望']),
                    '其他': len([r for r in data if r.get('recommendation') not in ['强烈买入', '买入', '谨慎买入', '观望']])
                }
            }
        }
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        self.performance_stats['failed_analyses'] += 1
        self.performance_stats['total_analyses'] += 1
        
        return {
            'success': False,
            'analysis_type': 'unknown',
            'analysis_date': datetime.now().strftime('%Y%m%d %H:%M:%S'),
            'total_stocks': 0,
            'data': [],
            'summary': {},
            'error': '分析失败或无结果'
        }
    
    def _save_analysis_results(self, result: Dict[str, Any], output_dir: str) -> None:
        """保存分析结果"""
        try:
            ensure_dir_exists(output_dir)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_type = result.get('analysis_type', 'unknown')
            filename = f"unified_analysis_{analysis_type}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # 保存JSON文件
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已保存到: {filepath}")
            
            # 如果有数据，也保存CSV文件
            if result.get('data'):
                csv_filename = f"unified_analysis_{analysis_type}_{timestamp}.csv"
                csv_filepath = os.path.join(output_dir, csv_filename)
                
                df = pd.DataFrame(result['data'])
                df.to_csv(csv_filepath, index=False, encoding='utf-8')
                
                logger.info(f"分析结果CSV已保存到: {csv_filepath}")
                
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        self.analysis_config.update(config)
        logger.info(f"配置已更新: {config}")


# ===== 依赖注入和兼容性接口 =====

def create_unified_analysis_engine(max_workers: int = 4) -> UnifiedAnalysisEngine:
    """创建统一分析引擎实例（兼容性方法）"""
    return UnifiedAnalysisEngine(max_workers)


def get_unified_analysis_engine_unified_analysis_engine() -> UnifiedAnalysisEngine:
    """
    获取统一分析引擎实例（依赖注入方式）
    
    Returns:
        UnifiedAnalysisEngine: 统一分析引擎实例
    """
    try:
        from utils.dependency_injection import get_container
        container = get_container()
        return container.resolve(UnifiedAnalysisEngine)
    except Exception as e:
        logger.warning(f"从依赖注入容器获取UnifiedAnalysisEngine失败，创建新实例: {e}")
        return UnifiedAnalysisEngine()


def get_legacy_unified_analysis_engine() -> UnifiedAnalysisEngine:
    """获取统一分析引擎实例（向后兼容）"""
    return get_unified_analysis_engine()


# 注册到依赖注入容器
try:
    from utils.dependency_injection import get_container
    container = get_container()
    if not container.is_registered(UnifiedAnalysisEngine):
        container.register_singleton(UnifiedAnalysisEngine, UnifiedAnalysisEngine)
        logger.info("UnifiedAnalysisEngine已注册到依赖注入容器")
except Exception as e:
    logger.warning(f"注册UnifiedAnalysisEngine到依赖注入容器失败: {e}")
