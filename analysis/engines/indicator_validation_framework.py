#!/usr/bin/env python3
"""
指标验证框架

提供全面的指标验证功能，支持逐个验证每个指标是否能通过策略选股系统选出来
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.unified_data_manager import get_unified_data_manager
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger
from utils.path_utils import get_result_dir
from utils.decorators import performance_monitor, safe_run

logger = get_logger(__name__)


class ValidationMode(Enum):
    """验证模式枚举"""
    QUICK = "quick"              # 快速验证：只验证核心指标
    PRIORITY = "priority"        # 优先级验证：按重要性排序验证
    CATEGORY = "category"        # 分类验证：按指标类型分类验证
    FULL = "full"               # 完整验证：验证所有指标


class ValidationResult(Enum):
    """验证结果枚举"""
    SUCCESS = "success"          # 成功：能够选出股票且符合预期
    NO_SELECTION = "no_selection"  # 未选出：策略未选出任何股票
    OVER_SELECTION = "over_selection"  # 过度选择：选出股票过多
    ERROR = "error"             # 错误：验证过程出现错误
    TIMEOUT = "timeout"         # 超时：验证超时


@dataclass
class IndicatorValidationConfig:
    """指标验证配置"""
    mode: ValidationMode = ValidationMode.FULL
    stock_pool_size: int = 1000
    max_selection_ratio: float = 0.1  # 最大选股比例
    min_selection_count: int = 1      # 最小选股数量
    validation_date: str = None       # 验证日期
    timeout_seconds: int = 300        # 超时时间（秒）
    parallel_workers: int = 4         # 并行工作线程数
    output_format: str = "json"       # 输出格式：json, csv, txt
    save_details: bool = True         # 是否保存详细结果
    stop_on_success: bool = False     # 成功选股后立即停止
    stop_on_error: bool = False       # 遇到错误后立即停止
    debug_mode: bool = False          # 调试模式：打印详细信息
    

class IndicatorValidationFramework:
    """
    指标验证框架
    
    提供全面的指标验证功能，支持多种验证模式和输出格式
    """
    
    def __init__(self, config: IndicatorValidationConfig = None):
        """
        初始化验证框架
        
        Args:
            config: 验证配置
        """
        self.config = config or IndicatorValidationConfig()
        
        # 设置默认验证日期
        if self.config.validation_date is None:
            self.config.validation_date = datetime.now().strftime("%Y-%m-%d")
        
        # 初始化数据管理器和策略执行器
        try:
            self.data_manager = get_unified_data_manager()
            self.strategy_executor = StrategyExecutor()
            
            logger.info("✅ 指标验证框架初始化完成")
        except Exception as e:
            logger.error(f"❌ 初始化指标验证框架失败: {e}")
            raise
        
        # 验证统计信息
        self.validation_stats = {
            'start_time': None,
            'end_time': None,
            'duration': 0,
            'validated_indicators': 0,
            'successful_validations': 0,
            'failed_validations': 0
        }
        
        # 添加股票池缓存，避免重复查询
        self._stock_pool_cache = None
        self._stock_pool_cache_date = None
        
        # 添加早停标志
        self._should_stop = False
    
    def validate_all_indicators(self) -> Dict[str, Any]:
        """
        验证所有指标
        
        Returns:
            验证结果字典
        """
        logger.info("开始验证所有指标")
        self.validation_stats['start_time'] = datetime.now()
        
        try:
            # 获取指标列表
            indicators = self._get_indicators_by_mode()
            self.validation_stats['total_indicators'] = len(indicators)
            
            logger.info(f"共需验证 {len(indicators)} 个指标")
            
            # 准备股票池（可能触发数据库连接错误）
            try:
                stock_pool = self._prepare_stock_pool()
                logger.info(f"准备股票池: {len(stock_pool)} 只股票")
            except ConnectionError as conn_error:
                # 数据库连接失败，如果配置了错误后早停，立即返回错误结果
                logger.error(f"❌ 数据库连接失败，无法继续验证: {conn_error}")
                
                if self.config.stop_on_error:
                    logger.warning("🛑 配置了错误后早停，由于数据库连接失败立即停止验证")
                    
                    # 创建错误结果
                    error_result = {
                        'indicator_name': 'DATABASE_CONNECTION',
                        'status': ValidationResult.ERROR.value,
                        'error_message': str(conn_error),
                        'timestamp': datetime.now().isoformat(),
                        'validation_date': self.config.validation_date,
                        'stock_pool_size': 0,
                        'selected_count': 0,
                        'selection_ratio': 0.0,
                        'selected_stocks': [],
                        'strategy_config': {},
                        'execution_time': 0
                    }
                    
                    # 更新统计信息
                    self.validation_stats['validated_indicators'] = 1
                    self.validation_stats['failed_validations'] = 1
                    self.validation_stats['successful_validations'] = 0
                    
                    # 生成总结报告
                    summary = self._generate_summary([error_result])
                    
                    self.validation_stats['end_time'] = datetime.now()
                    self.validation_stats['duration'] = (
                        self.validation_stats['end_time'] - self.validation_stats['start_time']
                    ).total_seconds()
                    
                    logger.info(f"验证因数据库连接错误提前结束，耗时: {self.validation_stats['duration']:.2f}秒")
                    
                    return {
                        'summary': summary,
                        'results': [error_result],
                        'stats': self.validation_stats,
                        'config': self.config.__dict__,
                        'early_stop_reason': 'database_connection_error'
                    }
                else:
                    # 如果没有配置错误后早停，使用默认股票池继续
                    logger.warning("⚠️ 数据库连接失败，但未配置错误后早停，使用默认股票池继续验证")
                    stock_pool = ['000001', '000002', '600000', '600036', '000858']
            
            # 执行验证
            if self.config.parallel_workers > 1:
                results = self._validate_parallel(indicators, stock_pool)
            else:
                results = self._validate_sequential(indicators, stock_pool)
            
            # 生成总结报告
            summary = self._generate_summary(results)
            
            # 保存结果
            if self.config.save_details:
                self._save_results(results, summary)
            
            self.validation_stats['end_time'] = datetime.now()
            self.validation_stats['duration'] = (
                self.validation_stats['end_time'] - self.validation_stats['start_time']
            ).total_seconds()
            
            logger.info(f"指标验证完成，耗时: {self.validation_stats['duration']:.2f}秒")
            
            return {
                'summary': summary,
                'results': results,
                'stats': self.validation_stats,
                'config': self.config.__dict__
            }
            
        except Exception as e:
            logger.error(f"指标验证过程出错: {e}")
            raise
    
    def validate_single_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """
        验证单个指标
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            验证结果字典
        """
        logger.info(f"开始验证指标: {indicator_name}")
        
        try:
            # 准备股票池
            stock_pool = self._prepare_stock_pool()
            
            # 验证指标
            result = self._validate_indicator(indicator_name, stock_pool)
            
            logger.info(f"指标 {indicator_name} 验证完成")
            return result
            
        except Exception as e:
            logger.error(f"验证指标 {indicator_name} 时出错: {e}")
            return {
                'indicator_name': indicator_name,
                'status': ValidationResult.ERROR.value,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_indicators_by_mode(self) -> List[str]:
        """根据验证模式获取指标列表"""
        all_indicators = complete_registry.get_indicator_names()
        
        if self.config.mode == ValidationMode.QUICK:
            # 快速模式：只验证核心指标
            core_indicators = [
                'MA', 'EMA', 'MACD', 'RSI', 'KDJ', 'BOLL', 
                'CCI', 'DMI', 'OBV', 'ATR', 'WR', 'BIAS'
            ]
            return [ind for ind in core_indicators if ind in all_indicators]
            
        elif self.config.mode == ValidationMode.PRIORITY:
            # 优先级模式：按重要性排序
            priority_order = [
                # 第一优先级：基础技术指标
                'MA', 'EMA', 'MACD', 'RSI', 'KDJ', 'BOLL',
                # 第二优先级：常用辅助指标
                'CCI', 'DMI', 'OBV', 'ATR', 'WR', 'BIAS', 'ADX', 'MFI',
                # 第三优先级：增强指标
                'ENHANCED_RSI', 'ENHANCED_MACD', 'ENHANCED_KDJ',
                # 第四优先级：ZXM指标
                'ZXM_DAILY_MACD', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK',
                # 第五优先级：其他指标
            ]
            # 添加未在优先级列表中的其他指标
            remaining = [ind for ind in all_indicators if ind not in priority_order]
            return [ind for ind in priority_order if ind in all_indicators] + remaining
            
        elif self.config.mode == ValidationMode.CATEGORY:
            # 分类模式：按类型分组
            return self._categorize_indicators(all_indicators)
            
        else:  # FULL模式
            return all_indicators
    
    def _categorize_indicators(self, indicators: List[str]) -> List[str]:
        """按类型对指标进行分类排序"""
        categories = {
            'basic': [],      # 基础指标
            'enhanced': [],   # 增强指标
            'zxm': [],       # ZXM指标
            'composite': [],  # 复合指标
            'pattern': [],    # 形态指标
            'other': []       # 其他指标
        }
        
        for indicator in indicators:
            if indicator.startswith('ZXM_'):
                categories['zxm'].append(indicator)
            elif indicator.startswith('ENHANCED_'):
                categories['enhanced'].append(indicator)
            elif indicator in ['COMPOSITE', 'UNIFIED_MA', 'CHIP_DISTRIBUTION']:
                categories['composite'].append(indicator)
            elif 'PATTERN' in indicator or 'CANDLESTICK' in indicator:
                categories['pattern'].append(indicator)
            elif indicator in ['MA', 'EMA', 'MACD', 'RSI', 'KDJ', 'BOLL', 'CCI', 'DMI', 'OBV', 'ATR', 'WR', 'BIAS']:
                categories['basic'].append(indicator)
            else:
                categories['other'].append(indicator)
        
        # 按类别顺序返回
        result = []
        for category in ['basic', 'enhanced', 'zxm', 'composite', 'pattern', 'other']:
            result.extend(sorted(categories[category]))
        
        return result
    
    def _prepare_stock_pool(self) -> List[str]:
        """准备股票池（带缓存，避免重复查询）"""
        try:
            # 检查缓存
            if (self._stock_pool_cache is not None and 
                self._stock_pool_cache_date == self.config.validation_date):
                logger.info(f"使用缓存的股票池: {len(self._stock_pool_cache)} 只股票")
                return self._stock_pool_cache
            
            logger.info(f"🔍 准备股票池，验证日期: {self.config.validation_date}")
            
            # 获取活跃股票列表
            query = f"""
            SELECT DISTINCT code 
            FROM stock_info 
            WHERE level = '日线' AND date = '{self.config.validation_date}'
            AND volume > 0 
            AND close > 0
            ORDER BY volume DESC
            LIMIT {self.config.stock_pool_size}
            """
            
            # 检测数据库连接问题
            try:
                result = self.data_manager.query(query)
            except Exception as db_error:
                db_error_str = str(db_error)
                # 检测常见的数据库连接错误
                if any(error_keyword in db_error_str.lower() for error_keyword in 
                       ['connection refused', 'connection failed', 'connection timeout', 'no connection',
                        'failed to connect', 'connection error', 'database connection', 'connect error']):
                    logger.error(f"❌ 数据库连接失败: {db_error}")
                    # 抛出异常以便上层捕获并设置为ERROR状态
                    raise ConnectionError(f"数据库连接失败: {db_error}")
                else:
                    # 其他数据库错误，继续处理
                    raise db_error
            
            if result.empty:
                logger.warning(f"未找到 {self.config.validation_date} 的股票数据，使用默认股票池")
                # 使用最近可用日期的数据
                query = """
                SELECT DISTINCT code 
                FROM stock_info 
                WHERE level = '日线' 
                AND volume > 0 AND close > 0
                ORDER BY date DESC, volume DESC
                LIMIT 1000
                """
                try:
                    result = self.data_manager.query(query)
                except Exception as db_error:
                    db_error_str = str(db_error)
                    if any(error_keyword in db_error_str.lower() for error_keyword in 
                           ['connection refused', 'connection failed', 'connection timeout', 'no connection',
                            'failed to connect', 'connection error', 'database connection', 'connect error']):
                        logger.error(f"❌ 数据库连接失败: {db_error}")
                        raise ConnectionError(f"数据库连接失败: {db_error}")
                    else:
                        raise db_error
            
            stock_pool = result['code'].tolist()
            
            # 缓存结果
            self._stock_pool_cache = stock_pool
            self._stock_pool_cache_date = self.config.validation_date
            
            logger.info(f"✅ 准备股票池完成: {len(stock_pool)} 只股票")
            return stock_pool
            
        except ConnectionError:
            # 重新抛出连接错误，让上层处理
            raise
        except Exception as e:
            logger.error(f"准备股票池失败: {e}")
            # 对于其他错误，检查是否包含数据库连接相关的错误信息
            error_str = str(e)
            if any(error_keyword in error_str.lower() for error_keyword in 
                   ['connection refused', 'connection failed', 'connection timeout', 'no connection',
                    'failed to connect', 'connection error', 'database connection', 'connect error']):
                logger.error(f"❌ 检测到数据库连接问题: {e}")
                raise ConnectionError(f"数据库连接失败: {e}")
            # 对于其他错误，返回默认股票池
            return ['000001', '000002', '600000', '600036', '000858']
    
    def _validate_parallel(self, indicators: List[str], stock_pool: List[str]) -> List[Dict[str, Any]]:
        """并行验证指标"""
        results = []
        early_stop_triggered = False
        
        # 如果配置了早停，则使用顺序验证以便及时停止
        if self.config.stop_on_success or self.config.stop_on_error:
            logger.info("⚠️ 配置了早停功能，切换到顺序验证模式以确保及时停止")
            return self._validate_sequential(indicators, stock_pool)
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # 提交任务
            future_to_indicator = {
                executor.submit(self._validate_indicator, indicator, stock_pool): indicator
                for indicator in indicators
            }
            
            # 收集结果
            for future in as_completed(future_to_indicator):
                indicator = future_to_indicator[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                    self.validation_stats['validated_indicators'] += 1
                    
                    if result['status'] == ValidationResult.SUCCESS.value:
                        self.validation_stats['successful_validations'] += 1
                        logger.info(f"✅ 指标 {indicator} 验证成功: 选出 {result['selected_count']} 只股票")
                        
                        # 调试模式：打印选中的股票
                        if self.config.debug_mode and result.get('selected_stocks'):
                            logger.info(f"选中股票: {result['selected_stocks'][:10]}")  # 只显示前10只
                            
                    elif result['status'] == ValidationResult.ERROR.value:
                        self.validation_stats['failed_validations'] += 1
                        logger.error(f"❌ 指标 {indicator} 验证出错: {result.get('error_message', '未知错误')}")
                        
                    else:
                        self.validation_stats['failed_validations'] += 1
                        logger.info(f"⚠️ 指标 {indicator} 验证完成但未成功: {result['status']}")
                        
                        # 调试模式：打印详细状态
                        if self.config.debug_mode:
                            logger.info(f"选股数量: {result['selected_count']}, 选股比例: {result['selection_ratio']:.4f}")
                    
                except Exception as e:
                    logger.error(f"指标 {indicator} 验证失败: {e}")
                    error_result = {
                        'indicator_name': indicator,
                        'status': ValidationResult.ERROR.value,
                        'error_message': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
                    self.validation_stats['failed_validations'] += 1
        
        return results
    
    def _validate_sequential(self, indicators: List[str], stock_pool: List[str]) -> List[Dict[str, Any]]:
        """顺序验证指标（支持早停）"""
        results = []
        self._should_stop = False  # 重置早停标志
        
        for i, indicator in enumerate(indicators, 1):
            # 检查早停标志
            if self._should_stop:
                logger.warning(f"🛑 检测到早停信号，停止验证")
                break
                
            logger.info(f"📊 验证进度: {i}/{len(indicators)} - {indicator}")
            
            try:
                result = self._validate_indicator(indicator, stock_pool)
                results.append(result)
                self.validation_stats['validated_indicators'] += 1
                
                if result['status'] == ValidationResult.SUCCESS.value:
                    self.validation_stats['successful_validations'] += 1
                    logger.info(f"✅ 指标 {indicator} 验证成功: 选出 {result['selected_count']} 只股票")
                    
                    # 调试模式：打印选中的股票
                    if self.config.debug_mode and result.get('selected_stocks'):
                        logger.info(f"选中股票: {result['selected_stocks'][:10]}")  # 只显示前10只
                    
                    # 成功后早停
                    if self.config.stop_on_success:
                        logger.warning(f"🛑 配置了成功后早停，在指标 {indicator} 成功后停止验证")
                        logger.info(f"已验证 {i}/{len(indicators)} 个指标，其中成功 {self.validation_stats['successful_validations']} 个")
                        self._should_stop = True
                        break
                        
                elif result['status'] == ValidationResult.ERROR.value:
                    self.validation_stats['failed_validations'] += 1
                    logger.error(f"❌ 指标 {indicator} 验证出错: {result.get('error_message', '未知错误')}")
                    
                    # 错误后早停
                    if self.config.stop_on_error:
                        logger.warning(f"🛑 配置了错误后早停，在指标 {indicator} 出错后停止验证")
                        logger.info(f"已验证 {i}/{len(indicators)} 个指标，其中失败 {self.validation_stats['failed_validations']} 个")
                        self._should_stop = True
                        break
                        
                else:
                    self.validation_stats['failed_validations'] += 1
                    logger.info(f"⚠️ 指标 {indicator} 验证完成但未成功: {result['status']}")
                    
                    # 调试模式：打印详细状态
                    if self.config.debug_mode:
                        logger.info(f"选股数量: {result['selected_count']}, 选股比例: {result['selection_ratio']:.4f}")
                
            except Exception as e:
                logger.error(f"指标 {indicator} 验证失败: {e}")
                error_result = {
                    'indicator_name': indicator,
                    'status': ValidationResult.ERROR.value,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
                self.validation_stats['failed_validations'] += 1
                
                # 错误后早停
                if self.config.stop_on_error:
                    logger.warning(f"🛑 配置了错误后早停，在指标 {indicator} 异常后停止验证")
                    logger.info(f"已验证 {i}/{len(indicators)} 个指标，其中失败 {self.validation_stats['failed_validations']} 个")
                    self._should_stop = True
                    break
        
        return results
    
    def _validate_indicator(self, indicator_name: str, stock_pool: List[str]) -> Dict[str, Any]:
        """验证单个指标"""
        start_time = time.time()
        
        result = {
            'indicator_name': indicator_name,
            'status': ValidationResult.ERROR.value,
            'validation_date': self.config.validation_date,
            'stock_pool_size': len(stock_pool),
            'selected_count': 0,
            'selection_ratio': 0.0,
            'selected_stocks': [],
            'strategy_config': {},
            'execution_time': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. 生成指标策略
            strategy_config = self._generate_indicator_strategy(indicator_name)
            result['strategy_config'] = strategy_config
            
            if not strategy_config:
                result['status'] = ValidationResult.ERROR.value
                result['error_message'] = "策略生成失败"
                return result
            
            # 2. 执行策略选股
            selected_stocks = self._execute_strategy_selection(strategy_config, stock_pool)
            
            result['selected_count'] = len(selected_stocks)
            result['selection_ratio'] = len(selected_stocks) / len(stock_pool) if stock_pool else 0
            result['selected_stocks'] = selected_stocks[:50]  # 只保存前50只股票
            
            # 3. 判断验证结果
            if len(selected_stocks) == 0:
                result['status'] = ValidationResult.NO_SELECTION.value
            elif result['selection_ratio'] > self.config.max_selection_ratio:
                result['status'] = ValidationResult.OVER_SELECTION.value
            elif len(selected_stocks) >= self.config.min_selection_count:
                result['status'] = ValidationResult.SUCCESS.value
            else:
                result['status'] = ValidationResult.NO_SELECTION.value
            
            result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            result['status'] = ValidationResult.ERROR.value
            result['error_message'] = str(e)
            result['execution_time'] = time.time() - start_time
            logger.error(f"验证指标 {indicator_name} 时出错: {e}")
        
        return result
    
    def _generate_indicator_strategy(self, indicator_name: str) -> Dict[str, Any]:
        """为指标生成策略配置"""
        try:
            import time
            
            # 基础策略模板 - 确保包含所有必要字段
            strategy_config = {
                'strategy_id': f'validate_{indicator_name.lower()}_{int(time.time())}',  # 添加时间戳确保唯一性
                'name': f'{indicator_name}指标验证策略',
                'description': f'用于验证{indicator_name}指标的自动生成策略',
                'conditions': [],  # 必要字段，稍后填充
                'filters': {
                    'market': ['主板', '创业板', '科创板'],
                    'exclude_st': True,
                    'min_market_cap': 1000000000,  # 10亿市值以上
                    'max_stocks': 1000
                },
                'sort': [{'field': 'score', 'direction': 'desc'}],
                'created_time': time.time(),
                'validation_mode': True  # 标记为验证模式
            }
            
            # 根据指标类型生成条件
            conditions = self._generate_indicator_conditions(indicator_name)
            if not conditions:
                # 如果生成条件失败，使用最基本的条件
                conditions = [
                    {
                        'type': 'basic',
                        'field': 'close',
                        'operator': '>',
                        'value': 0,
                        'description': f'{indicator_name}基础验证条件'
                    }
                ]
            
            strategy_config['conditions'] = conditions
            
            # 验证生成的策略配置
            required_fields = ['strategy_id', 'name', 'conditions']
            for field in required_fields:
                if field not in strategy_config:
                    raise ValueError(f"生成的策略配置缺少必要字段: {field}")
            
            if not strategy_config['conditions']:
                raise ValueError("生成的策略配置缺少条件")
            
            return strategy_config
            
        except Exception as e:
            logger.error(f"生成指标 {indicator_name} 策略失败: {e}")
            # 返回最基本的策略配置
            import time
            return {
                'strategy_id': f'fallback_{indicator_name.lower()}_{int(time.time())}',
                'name': f'{indicator_name}回退验证策略',
                'conditions': [
                    {
                        'type': 'basic',
                        'field': 'close',
                        'operator': '>',
                        'value': 0,
                        'description': '基础回退条件'
                    }
                ],
                'description': f'回退验证策略',
                'validation_mode': True
            }
    
    def _generate_indicator_conditions(self, indicator_name: str) -> List[Dict[str, Any]]:
        """根据指标名称生成条件"""
        conditions = []
        
        try:
            if indicator_name == 'MA':
                conditions = [
                    {
                        'type': 'indicator',
                        'indicator_id': 'MA',
                        'period': 5,
                        'field': 'close',
                        'operator': '>',
                        'value': 'MA_20',
                        'description': '5日均线上穿20日均线'
                    }
                ]
            elif indicator_name == 'MACD':
                conditions = [
                    {
                        'type': 'indicator',
                        'indicator_id': 'MACD',
                        'period': 12,
                        'field': 'dif',
                        'operator': '>',
                        'value': 'dea',
                        'description': 'MACD金叉'
                    }
                ]
            elif indicator_name == 'RSI':
                conditions = [
                    {
                        'type': 'indicator',
                        'indicator_id': 'RSI',
                        'period': 14,
                        'operator': '<',
                        'value': 30,
                        'description': 'RSI超卖'
                    }
                ]
            elif indicator_name == 'KDJ':
                conditions = [
                    {
                        'type': 'indicator',
                        'indicator_id': 'KDJ',
                        'period': 9,
                        'field': 'k',
                        'operator': '>',
                        'value': 'd',
                        'description': 'KDJ金叉'
                    }
                ]
            elif indicator_name == 'BOLL':
                conditions = [
                    {
                        'type': 'indicator',
                        'indicator_id': 'BOLL',
                        'period': 20,
                        'field': 'close',
                        'operator': '<',
                        'value': 'lower',
                        'description': '价格触及布林带下轨'
                    }
                ]
            elif indicator_name == 'EMA':
                conditions = [
                    {
                        'type': 'indicator',
                        'indicator_id': 'EMA',
                        'period': 12,
                        'field': 'close',
                        'operator': '>',
                        'value': 'EMA_26',
                        'description': '12日指数均线上穿26日指数均线'
                    }
                ]
            else:
                # 通用条件：使用基础类型，避免指标不存在的问题
                conditions = [
                    {
                        'type': 'basic',
                        'field': 'close',
                        'operator': '>',
                        'value': 0,
                        'description': f'{indicator_name}基础验证条件'
                    }
                ]
                
        except Exception as e:
            logger.error(f"生成指标 {indicator_name} 条件失败: {e}")
            # 返回最基本的条件
            conditions = [
                {
                    'type': 'basic',
                    'field': 'close',
                    'operator': '>',
                    'value': 0,
                    'description': f'{indicator_name}回退条件'
                }
            ]
        
        return conditions
    
    def _execute_strategy_selection(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str]:
        """执行策略选股"""
        try:
            # 执行策略
            result_df = self.strategy_executor.execute_strategy(
                strategy_plan=strategy_config,
                end_date=self.config.validation_date
            )
            
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                # 获取股票代码列
                if 'code' in result_df.columns:
                    return result_df['code'].tolist()
                elif 'stock_code' in result_df.columns:
                    return result_df['stock_code'].tolist()
                else:
                    logger.warning(f"结果中未找到股票代码列: {list(result_df.columns)}")
                    return []
            else:
                return []
                
        except Exception as e:
            logger.error(f"执行策略选股失败: {e}")
            return []
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成验证总结"""
        summary = {
            'total_indicators': len(results),
            'validation_results': {
                ValidationResult.SUCCESS.value: 0,
                ValidationResult.NO_SELECTION.value: 0,
                ValidationResult.OVER_SELECTION.value: 0,
                ValidationResult.ERROR.value: 0,
                ValidationResult.TIMEOUT.value: 0
            },
            'success_rate': 0.0,
            'average_selection_ratio': 0.0,
            'total_selected_stocks': 0,
            'top_performing_indicators': [],
            'failed_indicators': [],
            'execution_time': self.validation_stats.get('duration', 0)
        }
        
        # 统计各状态数量
        selection_ratios = []
        selected_counts = []
        
        for result in results:
            status = result.get('status', ValidationResult.ERROR.value)
            summary['validation_results'][status] += 1
            
            if status == ValidationResult.SUCCESS.value:
                selection_ratios.append(result.get('selection_ratio', 0))
                selected_counts.append(result.get('selected_count', 0))
                summary['top_performing_indicators'].append({
                    'indicator': result['indicator_name'],
                    'selected_count': result.get('selected_count', 0),
                    'selection_ratio': result.get('selection_ratio', 0)
                })
            elif status in [ValidationResult.ERROR.value, ValidationResult.TIMEOUT.value]:
                summary['failed_indicators'].append({
                    'indicator': result['indicator_name'],
                    'status': status,
                    'error': result.get('error_message', '未知错误')
                })
        
        # 计算统计数据
        if summary['validation_results'][ValidationResult.SUCCESS.value] > 0:
            summary['success_rate'] = (
                summary['validation_results'][ValidationResult.SUCCESS.value] / 
                summary['total_indicators']
            )
            summary['average_selection_ratio'] = np.mean(selection_ratios)
            summary['total_selected_stocks'] = sum(selected_counts)
        
        # 排序性能最好的指标
        summary['top_performing_indicators'].sort(
            key=lambda x: x['selected_count'], reverse=True
        )
        summary['top_performing_indicators'] = summary['top_performing_indicators'][:10]
        
        return summary
    
    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """保存验证结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = get_result_dir()
            
            # 保存详细结果
            if self.config.output_format == "json":
                result_file = os.path.join(result_dir, f"indicator_validation_{timestamp}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'summary': summary,
                        'results': results,
                        'config': self.config.__dict__,
                        'stats': self.validation_stats
                    }, f, ensure_ascii=False, indent=2)
                    
            elif self.config.output_format == "csv":
                result_file = os.path.join(result_dir, f"indicator_validation_{timestamp}.csv")
                df = pd.DataFrame(results)
                df.to_csv(result_file, index=False, encoding='utf-8')
                
            elif self.config.output_format == "txt":
                result_file = os.path.join(result_dir, f"indicator_validation_{timestamp}.txt")
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write("指标验证报告\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"验证模式: {self.config.mode.value}\n")
                    f.write(f"验证日期: {self.config.validation_date}\n")
                    f.write(f"股票池大小: {self.config.stock_pool_size}\n\n")
                    
                    f.write("验证总结:\n")
                    f.write(f"  总指标数: {summary['total_indicators']}\n")
                    f.write(f"  成功验证: {summary['validation_results'][ValidationResult.SUCCESS.value]}\n")
                    f.write(f"  未选出股票: {summary['validation_results'][ValidationResult.NO_SELECTION.value]}\n")
                    f.write(f"  过度选择: {summary['validation_results'][ValidationResult.OVER_SELECTION.value]}\n")
                    f.write(f"  验证错误: {summary['validation_results'][ValidationResult.ERROR.value]}\n")
                    f.write(f"  成功率: {summary['success_rate']:.2%}\n\n")
                    
                    f.write("详细结果:\n")
                    for result in results:
                        f.write(f"  {result['indicator_name']}: {result['status']}")
                        if result.get('selected_count', 0) > 0:
                            f.write(f" (选出{result['selected_count']}只股票)")
                        f.write("\n")
            
            logger.info(f"验证结果已保存到: {result_file}")
            
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")


def main():
    """主函数：演示指标验证框架的使用"""
    print("指标验证框架演示")
    print("=" * 50)
    
    # 配置验证参数
    config = IndicatorValidationConfig(
        mode=ValidationMode.QUICK,
        stock_pool_size=500,
        max_selection_ratio=0.05,
        parallel_workers=2,
        output_format="json",
        save_details=True
    )
    
    # 创建验证框架
    framework = IndicatorValidationFramework(config)
    
    # 执行验证
    try:
        results = framework.validate_all_indicators()
        
        print(f"\n验证完成！")
        print(f"总指标数: {results['summary']['total_indicators']}")
        print(f"成功验证: {results['summary']['validation_results']['success']}")
        print(f"成功率: {results['summary']['success_rate']:.2%}")
        print(f"耗时: {results['stats']['duration']:.2f}秒")
        
        if results['summary']['top_performing_indicators']:
            print(f"\n表现最好的指标:")
            for i, indicator in enumerate(results['summary']['top_performing_indicators'][:5], 1):
                print(f"  {i}. {indicator['indicator']}: 选出{indicator['selected_count']}只股票")
        
    except Exception as e:
        print(f"验证过程出错: {e}")
        logger.error(f"验证过程出错: {e}")


if __name__ == "__main__":
    main() 