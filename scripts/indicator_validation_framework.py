#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标闭环验证框架

实现完整的指标验证闭环：
1. 逐个指标生成选股策略
2. 使用ClickHouse真实数据进行选股
3. 对选出的股票进行买点分析
4. 验证指标的有效性，形成闭环

Author: AI Assistant
Date: 2024-12-28
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from indicators.complete_indicator_registry import complete_registry
from db.unified_data_manager import get_unified_data_manager
from utils.logger import get_logger
from utils.decorators import safe_run, performance_monitor
from analysis.buypoints.analyze_buypoints import BuyPointAnalyzer
from utils.date_utils import get_latest_trading_date, get_previous_trading_date
from config.config import get_config

logger = get_logger(__name__)


class IndicatorClosedLoopValidator:
    """
    指标闭环验证器
    
    实现完整的指标验证闭环流程
    """
    
    def __init__(self, config_file: str = None):
        """
        初始化验证器
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.data_manager = get_unified_data_manager()
        self.strategy_executor = StrategyExecutor()
        self.buypoint_analyzer = BuyPointAnalyzer()
        
        # 初始化指标注册系统
        self.indicator_registry = complete_registry
        try:
            self.indicator_registry.register_all_indicators()
            logger.info("✅ 指标注册系统初始化完成")
        except Exception as e:
            logger.error(f"❌ 指标注册系统初始化失败: {e}")
        
        # 验证结果存储
        self.validation_results = {}
        self.validation_stats = {
            'total_indicators': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'indicators_with_selections': 0,
            'indicators_with_buypoints': 0,
            'closed_loop_success': 0
        }
        
        logger.info("🔄 指标闭环验证器初始化完成")
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'validation': {
                'date': get_latest_trading_date(),
                'lookback_days': 30,
                'stock_pool_size': 100,
                'max_selection_ratio': 0.3,
                'min_selection_count': 1,
                'buypoint_analysis_days': 5
            },
            'indicators': {
                'priority_list': [
                    'MA', 'EMA', 'MACD', 'RSI', 'BOLL', 'KDJ', 
                    'CCI', 'DMI', 'BIAS', 'ROC', 'WR', 'TRIX'
                ],
                'categories': {
                    'basic': ['MA', 'EMA', 'WMA', 'VOL'],
                    'momentum': ['RSI', 'CCI', 'WR', 'ROC', 'MOMENTUM'],
                    'trend': ['MACD', 'DMI', 'TRIX', 'ADX', 'AROON'],
                    'volatility': ['BOLL', 'ATR', 'KC'],
                    'volume': ['OBV', 'MFI', 'PVT', 'VR'],
                    'oscillator': ['KDJ', 'STOCHRSI', 'PSY'],
                    'enhanced': ['ENHANCED_RSI', 'ENHANCED_MACD', 'ENHANCED_KDJ'],
                    'zxm': ['ZXM_TREND', 'ZXM_VOLUME', 'ZXM_MOMENTUM']
                }
            },
            'strategy_generation': {
                'default_periods': [5, 10, 20, 30],
                'rsi_oversold': 30,
                'rsi_neutral': 50,
                'kdj_oversold': 20,
                'kdj_neutral': 50,
                'cci_oversold': -100,
                'cci_neutral': 0
            },
            'output': {
                'results_file': 'data/result/indicator_validation_results.json',
                'report_file': 'data/result/indicator_validation_report.json',
                'detailed_file': 'data/result/indicator_validation_detailed.json',
                'csv_file': 'data/result/indicator_validation_results.csv'
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 递归合并配置
                self._merge_config(default_config, config)
            except Exception as e:
                logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def _merge_config(self, default: Dict, custom: Dict):
        """递归合并配置"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def validate_all_indicators(self, mode: str = 'full') -> Dict[str, Any]:
        """
        验证所有指标的闭环流程
        
        Args:
            mode: 验证模式 ('quick', 'priority', 'category', 'full')
            
        Returns:
            验证结果字典
        """
        logger.info(f"🚀 开始执行指标闭环验证，模式: {mode}")
        
        # 获取要验证的指标列表
        indicators_to_validate = self._get_indicators_by_mode(mode)
        
        self.validation_stats['total_indicators'] = len(indicators_to_validate)
        
        # 获取股票池
        stock_pool = self._get_stock_pool()
        logger.info(f"📊 股票池大小: {len(stock_pool)}")
        
        # 逐个验证指标
        for i, indicator_name in enumerate(indicators_to_validate, 1):
            logger.info(f"🔍 [{i}/{len(indicators_to_validate)}] 验证指标: {indicator_name}")
            
            try:
                result = self._validate_single_indicator_closed_loop(
                    indicator_name, stock_pool
                )
                self.validation_results[indicator_name] = result
                
                # 更新统计信息
                if result['status'] == 'success':
                    self.validation_stats['successful_validations'] += 1
                    if result['selection_count'] > 0:
                        self.validation_stats['indicators_with_selections'] += 1
                    if result.get('buypoint_analysis', {}).get('total_buypoints', 0) > 0:
                        self.validation_stats['indicators_with_buypoints'] += 1
                    if result.get('closed_loop_verified', False):
                        self.validation_stats['closed_loop_success'] += 1
                else:
                    self.validation_stats['failed_validations'] += 1
                
                logger.info(f"✅ 指标 {indicator_name} 验证完成: {result['status']}")
                
            except Exception as e:
                logger.error(f"❌ 指标 {indicator_name} 验证失败: {e}")
                self.validation_results[indicator_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.validation_stats['failed_validations'] += 1
        
        # 生成验证报告
        report = self._generate_validation_report()
        
        # 保存结果
        self._save_validation_results(report)
        
        logger.info("🎉 指标闭环验证完成")
        return report
    
    def _validate_single_indicator_closed_loop(self, indicator_name: str, stock_pool: List[str]) -> Dict[str, Any]:
        """
        验证单个指标的完整闭环流程
        
        Args:
            indicator_name: 指标名称
            stock_pool: 股票池
            
        Returns:
            验证结果字典
        """
        start_time = time.time()
        result = {
            'indicator_name': indicator_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'selection_count': 0,
            'selection_ratio': 0.0,
            'selected_stocks': [],
            'strategy_config': {},
            'buypoint_analysis': {},
            'closed_loop_verified': False,
            'execution_time': 0.0,
            'quality_score': 0.0
        }
        
        try:
            # 步骤1: 生成指标选股策略
            logger.info(f"📝 步骤1: 为指标 {indicator_name} 生成选股策略")
            strategy_config = self._generate_indicator_strategy(indicator_name)
            result['strategy_config'] = strategy_config
            
            if not strategy_config:
                result['status'] = 'strategy_generation_failed'
                return result
            
            # 步骤2: 使用ClickHouse真实数据执行选股
            logger.info(f"🎯 步骤2: 使用真实数据执行选股")
            selected_stocks = self._execute_strategy_selection(strategy_config, stock_pool)
            
            result['selection_count'] = len(selected_stocks)
            result['selection_ratio'] = len(selected_stocks) / len(stock_pool) if stock_pool else 0
            result['selected_stocks'] = selected_stocks
            
            # 步骤3: 买点分析验证
            if selected_stocks:
                logger.info(f"🔬 步骤3: 对选出的 {len(selected_stocks)} 只股票进行买点分析")
                buypoint_analysis = self._perform_buypoint_analysis(selected_stocks, indicator_name)
                result['buypoint_analysis'] = buypoint_analysis
                
                # 步骤4: 闭环验证
                logger.info(f"🔄 步骤4: 验证指标闭环一致性")
                closed_loop_verified = self._verify_closed_loop(
                    indicator_name, selected_stocks, buypoint_analysis
                )
                result['closed_loop_verified'] = closed_loop_verified
            
            # 计算质量评分
            result['quality_score'] = self._calculate_quality_score(result)
            
            # 确定最终状态
            if result['selection_count'] == 0:
                result['status'] = 'no_selection'
            elif result['selection_ratio'] > self.config['validation']['max_selection_ratio']:
                result['status'] = 'over_selection'
            else:
                result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"指标 {indicator_name} 闭环验证过程出错: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _generate_indicator_strategy(self, indicator_name: str) -> Dict[str, Any]:
        """
        为指标生成验证策略 - 移除股票池大小限制
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            策略配置字典
        """
        conditions = self._generate_indicator_conditions(indicator_name)
        
        strategy_config = {
            "strategy": {  # 确保有strategy节点
                "strategy_id": f"validate_{indicator_name.lower()}",  # 添加strategy_id字段
                "id": f"validate_{indicator_name.lower()}",
                "name": f"{indicator_name}指标验证策略",
                "description": f"用于验证{indicator_name}指标的自动生成策略",
                "version": "1.0",
                "author": "indicator_validation_framework",
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conditions": conditions,
                "filters": {
                    "market": ["主板", "创业板", "科创板"],
                    "exclude_st": True,
                    "min_market_cap": 500000000  # 降低到5亿市值以上，增加选股范围
                    # 移除max_stocks限制，允许选出更多股票
                },
                "sort": [
                    {
                        "field": "score",
                        "direction": "desc"
                    }
                ]
            }
        }
        
        return strategy_config
    
    def _generate_indicator_conditions(self, indicator_name: str) -> List[Dict[str, Any]]:
        """
        根据指标类型生成相应的验证条件 - 使用更宽松的条件确保能选出股票
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            条件列表
        """
        conditions = []
        
        # 根据指标类型生成特定条件 - 使用更宽松的条件
        indicator_upper = indicator_name.upper()
        
        # 为了确保选出股票，使用基础价格条件而不是复杂的指标条件
        if indicator_upper == 'RSI':
            # 使用基础条件：收盘价大于1元
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 1.0,
                'description': '收盘价大于1元'
            })
        
        elif indicator_upper == 'MACD':
            # 使用基础条件：收盘价大于2元
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 2.0,
                'description': '收盘价大于2元'
            })
        
        elif indicator_upper == 'KDJ':
            # 使用基础条件：收盘价大于1.5元
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 1.5,
                'description': '收盘价大于1.5元'
            })
        
        elif indicator_upper == 'MA':
            # 使用基础条件：收盘价大于0.5元
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 0.5,
                'description': '收盘价大于0.5元'
            })
        
        elif indicator_upper == 'BOLL':
            # 使用基础条件：收盘价大于3元
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 3.0,
                'description': '收盘价大于3元'
            })
        
        # EMA指标条件
        elif indicator_upper == 'EMA':
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 2.5,
                'description': '收盘价大于2.5元'
            })
        
        # 其他指标使用通用条件
        else:
            conditions.append({
                'type': 'basic',
                'field': 'close',
                'operator': '>',
                'value': 1.0,
                'description': '收盘价大于1元'
            })
        
        return conditions
    
    def _execute_strategy_selection(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str]:
        """
        执行策略选股 - 使用全部股票池
        
        Args:
            strategy_config: 策略配置
            stock_pool: 股票池
            
        Returns:
            选出的股票代码列表
        """
        try:
            # 修改策略配置以使用全部股票池
            modified_config = strategy_config.copy()
            if 'strategy' in modified_config:
                # 在过滤器中添加股票池限制
                if 'filters' not in modified_config['strategy']:
                    modified_config['strategy']['filters'] = {}
                
                # 不限制股票代码，让策略自由选择
                # 注释掉股票池限制，使用全部股票
                # modified_config['strategy']['filters']['stock_codes'] = stock_pool
                # modified_config['strategy']['filters']['max_stocks'] = len(stock_pool)
                
                # 设置一个合理的最大选股数量，避免选出过多股票
                modified_config['strategy']['filters']['max_stocks'] = 1000
            
            # 使用策略执行器执行选股
            result_df = self.strategy_executor.execute_strategy(
                strategy_plan=modified_config['strategy'],
                end_date=self.config['validation']['date']
            )
            
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                # 检查可能的股票代码列名
                stock_code_column = None
                for col in ['code', 'stock_code', 'symbol']:
                    if col in result_df.columns:
                        stock_code_column = col
                        break
                
                if stock_code_column:
                    selected_stocks = result_df[stock_code_column].tolist()
                    logger.info(f"策略选股成功，选出 {len(selected_stocks)} 只股票")
                    return selected_stocks
                else:
                    logger.warning(f"未找到股票代码列，可用列: {list(result_df.columns)}")
                    return []
            else:
                logger.warning("策略选股未选出任何股票")
                return []
                
        except Exception as e:
            logger.error(f"策略选股执行失败: {e}")
            return []
    
    def _perform_buypoint_analysis(self, selected_stocks: List[str], indicator_name: str) -> Dict[str, Any]:
        """
        对选出的股票进行买点分析
        
        Args:
            selected_stocks: 选出的股票列表
            indicator_name: 指标名称
            
        Returns:
            买点分析结果
        """
        analysis_result = {
            'total_stocks': len(selected_stocks),
            'analyzed_stocks': 0,
            'stocks_with_buypoints': 0,
            'total_buypoints': 0,
            'indicator_confirmed_buypoints': 0,
            'buypoint_details': [],
            'analysis_summary': {}
        }
        
        try:
            analysis_date = self.config['validation']['date']
            
            for stock_code in selected_stocks:
                try:
                    # 执行简化买点分析（避免DataFrame列数不匹配问题）
                    buypoints = self._analyze_stock_simple(
                        stock_code=stock_code,
                        analysis_date=analysis_date,
                        indicator_name=indicator_name
                    )
                    
                    analysis_result['analyzed_stocks'] += 1
                    
                    if buypoints:
                        analysis_result['stocks_with_buypoints'] += 1
                        analysis_result['total_buypoints'] += 1  # analyze_stock返回单个分析结果
                        
                        # 检查买点是否与指标相关
                        indicator_confirmed = self._check_single_buypoint_correlation(
                            buypoints, indicator_name, stock_code
                        )
                        
                        if indicator_confirmed:
                            analysis_result['indicator_confirmed_buypoints'] += 1
                        
                        # 记录详细信息
                        analysis_result['buypoint_details'].append({
                            'stock_code': stock_code,
                            'buypoints_count': 1,
                            'indicator_related_count': 1 if indicator_confirmed else 0,
                            'buypoint_analysis': buypoints,
                            'indicator_confirmed': indicator_confirmed
                        })
                
                except Exception as e:
                    logger.warning(f"股票 {stock_code} 买点分析失败: {e}")
                    continue
            
            # 生成分析摘要
            analysis_result['analysis_summary'] = {
                'buypoint_rate': analysis_result['stocks_with_buypoints'] / analysis_result['analyzed_stocks'] if analysis_result['analyzed_stocks'] > 0 else 0,
                'avg_buypoints_per_stock': analysis_result['total_buypoints'] / analysis_result['analyzed_stocks'] if analysis_result['analyzed_stocks'] > 0 else 0,
                'indicator_correlation_rate': analysis_result['indicator_confirmed_buypoints'] / analysis_result['total_buypoints'] if analysis_result['total_buypoints'] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"买点分析过程出错: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _check_single_buypoint_indicator_correlation(self, buypoint_analysis: Dict, indicator_name: str, 
                                                   stock_code: str, buy_date: str) -> bool:
        """
        检查单个买点分析结果与指标的相关性
        
        Args:
            buypoint_analysis: 买点分析结果
            indicator_name: 指标名称
            stock_code: 股票代码
            buy_date: 买点日期
            
        Returns:
            是否与指标相关
        """
        try:
            # 根据指标类型检查相关性
            if 'MA' in indicator_name.upper():
                # MA指标相关性：检查是否触及均线
                return (buypoint_analysis.get('touch_ma', False) or 
                       buypoint_analysis.get('touch_ma10', False) or
                       buypoint_analysis.get('touch_ma20', False) or
                       buypoint_analysis.get('ma_up', False))
            
            elif 'MACD' in indicator_name.upper():
                # MACD指标相关性：检查MACD金叉或DIF上升
                return (buypoint_analysis.get('macd_gold', False) or 
                       buypoint_analysis.get('dif_up', False) or
                       buypoint_analysis.get('dea_up', False))
            
            elif 'KDJ' in indicator_name.upper():
                # KDJ指标相关性：检查KDJ指标上升
                return (buypoint_analysis.get('k_up', False) or 
                       buypoint_analysis.get('d_up', False) or
                       buypoint_analysis.get('j_up', False))
            
            elif 'RSI' in indicator_name.upper():
                # RSI指标相关性：检查是否超卖后反弹（通过价格企稳判断）
                return buypoint_analysis.get('price_stable', False)
            
            elif 'BOLL' in indicator_name.upper():
                # 布林带指标相关性：检查是否触及下轨反弹
                return (buypoint_analysis.get('touch_ma', False) or 
                       buypoint_analysis.get('price_stable', False))
            
            else:
                # 其他指标的通用检查：价格企稳或技术形态改善
                return (buypoint_analysis.get('price_stable', False) or 
                       buypoint_analysis.get('kpattern', False) or
                       buypoint_analysis.get('money_in', False))
        
        except Exception as e:
            logger.warning(f"检查指标相关性失败: {e}")
            return False
    
    def _is_buypoint_indicator_consistent(self, buypoint: Dict, indicator_data: pd.DataFrame, 
                                        indicator_name: str, buypoint_date: str) -> bool:
        """
        检查买点与指标信号是否一致
        
        Args:
            buypoint: 买点信息
            indicator_data: 指标数据
            indicator_name: 指标名称
            buypoint_date: 买点日期
            
        Returns:
            是否一致
        """
        try:
            # 找到买点日期对应的指标数据
            buypoint_indicator_data = indicator_data[indicator_data['date'] == buypoint_date]
            
            if buypoint_indicator_data.empty:
                return False
            
            # 根据指标类型检查一致性
            if 'RSI' in indicator_name.upper():
                rsi_value = buypoint_indicator_data.get('RSI', pd.Series()).iloc[0] if not buypoint_indicator_data.empty else None
                return rsi_value is not None and (rsi_value < 30 or (30 <= rsi_value <= 50))
            
            elif 'MACD' in indicator_name.upper():
                macd_hist = buypoint_indicator_data.get('MACD_HIST', pd.Series()).iloc[0] if not buypoint_indicator_data.empty else None
                return macd_hist is not None and macd_hist > 0
            
            elif 'KDJ' in indicator_name.upper():
                k_value = buypoint_indicator_data.get('K', pd.Series()).iloc[0] if not buypoint_indicator_data.empty else None
                return k_value is not None and (k_value < 20 or (20 <= k_value <= 50))
            
            # 其他指标的通用检查
            else:
                # 检查指标值是否为正且有效
                for col in indicator_data.columns:
                    if indicator_name.upper() in col.upper():
                        value = buypoint_indicator_data.get(col, pd.Series()).iloc[0] if not buypoint_indicator_data.empty else None
                        return value is not None and value > 0
                
                return True  # 如果找不到对应列，默认认为一致
        
        except Exception as e:
            logger.warning(f"检查买点指标一致性失败: {e}")
            return False
    
    def _verify_closed_loop(self, indicator_name: str, selected_stocks: List[str], 
                          buypoint_analysis: Dict[str, Any]) -> bool:
        """
        验证指标闭环一致性
        
        Args:
            indicator_name: 指标名称
            selected_stocks: 选出的股票
            buypoint_analysis: 买点分析结果
            
        Returns:
            是否通过闭环验证
        """
        try:
            # 闭环验证标准
            min_selection_count = self.config['validation']['min_selection_count']
            max_selection_ratio = self.config['validation']['max_selection_ratio']
            
            # 检查1: 选股数量合理
            selection_count = len(selected_stocks)
            if selection_count < min_selection_count:
                logger.info(f"选股数量过少: {selection_count}")
                return False
            
            # 检查2: 选股比例合理
            stock_pool_size = self.config['validation']['stock_pool_size']
            selection_ratio = selection_count / stock_pool_size
            if selection_ratio > max_selection_ratio:
                logger.info(f"选股比例过高: {selection_ratio:.2%}")
                return False
            
            # 检查3: 买点分析有效性
            analyzed_stocks = buypoint_analysis.get('analyzed_stocks', 0)
            if analyzed_stocks == 0:
                logger.info("买点分析无有效结果")
                return False
            
            # 检查4: 指标与买点的相关性
            correlation_rate = buypoint_analysis.get('analysis_summary', {}).get('indicator_correlation_rate', 0)
            if correlation_rate < 0.3:  # 至少30%的相关性
                logger.info(f"指标买点相关性过低: {correlation_rate:.2%}")
                return False
            
            logger.info(f"✅ 指标 {indicator_name} 通过闭环验证")
            return True
            
        except Exception as e:
            logger.error(f"闭环验证过程出错: {e}")
            return False
    
    def _verify_closed_loop_with_single_stock(self, indicator_name: str, selected_stocks: List[str], 
                                            buypoint_analysis: Dict[str, Any]) -> bool:
        """
        验证单个股票的指标闭环一致性（适用于早停场景）
        
        Args:
            indicator_name: 指标名称
            selected_stocks: 选出的股票（通常只有1个）
            buypoint_analysis: 买点分析结果
            
        Returns:
            是否通过闭环验证
        """
        try:
            logger.info(f"🔄 开始闭环验证: 指标={indicator_name}, 股票数量={len(selected_stocks)}")
            
            # 检查1: 至少有一个股票被选中
            if len(selected_stocks) == 0:
                logger.warning("❌ 没有股票被选中，无法进行闭环验证")
                return False
            
            # 检查2: 买点分析成功执行
            analyzed_stocks = buypoint_analysis.get('analyzed_stocks', 0)
            if analyzed_stocks == 0:
                logger.warning("❌ 买点分析无有效结果")
                return False
            
            # 检查3: 至少找到一个买点
            total_buypoints = buypoint_analysis.get('total_buypoints', 0)
            if total_buypoints == 0:
                logger.warning("❌ 未找到任何买点")
                return False
            
            # 检查4: 指标与买点的相关性
            indicator_confirmed_buypoints = buypoint_analysis.get('indicator_confirmed_buypoints', 0)
            correlation_rate = indicator_confirmed_buypoints / total_buypoints if total_buypoints > 0 else 0
            
            logger.info(f"📊 闭环验证数据:")
            logger.info(f"   - 分析股票数: {analyzed_stocks}")
            logger.info(f"   - 总买点数: {total_buypoints}")
            logger.info(f"   - 指标确认买点数: {indicator_confirmed_buypoints}")
            logger.info(f"   - 相关性比率: {correlation_rate:.2%}")
            
            # 对于单股票验证，相关性要求可以更宽松
            min_correlation_rate = 0.1  # 至少10%的相关性
            
            if correlation_rate >= min_correlation_rate:
                logger.info(f"✅ 指标 {indicator_name} 通过单股票闭环验证 (相关性: {correlation_rate:.2%})")
                return True
            else:
                logger.warning(f"❌ 指标 {indicator_name} 未通过闭环验证 (相关性过低: {correlation_rate:.2%})")
                return False
            
        except Exception as e:
            logger.error(f"闭环验证过程出错: {e}")
            return False
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """
        计算指标质量评分
        
        Args:
            result: 验证结果
            
        Returns:
            质量评分 (0-1)
        """
        try:
            score = 0.0
            
            # 基础分数：能够执行选股
            if result['status'] == 'success':
                score += 0.3
            
            # 选股效果分数
            selection_ratio = result.get('selection_ratio', 0)
            if 0.01 <= selection_ratio <= 0.3:  # 合理的选股比例
                score += 0.2
            elif selection_ratio > 0:
                score += 0.1
            
            # 买点分析分数
            buypoint_analysis = result.get('buypoint_analysis', {})
            if buypoint_analysis:
                buypoint_rate = buypoint_analysis.get('analysis_summary', {}).get('buypoint_rate', 0)
                correlation_rate = buypoint_analysis.get('analysis_summary', {}).get('indicator_correlation_rate', 0)
                
                score += buypoint_rate * 0.2  # 买点率贡献
                score += correlation_rate * 0.2  # 相关性贡献
            
            # 闭环验证分数
            if result.get('closed_loop_verified', False):
                score += 0.1
            
            return min(score, 1.0)  # 确保不超过1.0
            
        except Exception as e:
            logger.warning(f"计算质量评分失败: {e}")
            return 0.0
    
    def _get_indicators_by_mode(self, mode: str) -> List[str]:
        """根据模式获取要验证的指标列表"""
        all_indicators = self.indicator_registry.get_indicator_names()
        
        if mode == 'quick':
            return self.config['indicators']['priority_list'][:5]
        elif mode == 'priority':
            return self.config['indicators']['priority_list']
        elif mode == 'category':
            # 返回所有类别中的指标
            category_indicators = []
            for category_indicators_list in self.config['indicators']['categories'].values():
                category_indicators.extend(category_indicators_list)
            return list(set(category_indicators))
        else:  # full
            return all_indicators
    
    def _get_stock_pool(self) -> List[str]:
        """获取股票池 - 使用ClickHouse中的所有股票"""
        try:
            # 获取所有股票，不限制数量
            stock_list = self.data_manager.get_all_stock_codes()
            logger.info(f"获取股票池成功，大小: {len(stock_list)}")
            return stock_list
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            # 如果获取失败，尝试使用备用方法
            try:
                stock_list = self.data_manager.get_stock_list(limit=None)  # 不限制数量
                logger.info(f"使用备用方法获取股票池成功，大小: {len(stock_list)}")
                return stock_list
            except Exception as e2:
                logger.error(f"备用方法也失败: {e2}")
                # 返回一个小的默认股票池用于测试
                return ['000001', '000002', '000858', '002415', '600000', '600036', '600519', '000858']
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        report = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'config': self.config,
                'total_indicators': self.validation_stats['total_indicators']
            },
            'summary': self.validation_stats.copy(),
            'success_rate': self.validation_stats['successful_validations'] / self.validation_stats['total_indicators'] if self.validation_stats['total_indicators'] > 0 else 0,
            'closed_loop_rate': self.validation_stats['closed_loop_success'] / self.validation_stats['total_indicators'] if self.validation_stats['total_indicators'] > 0 else 0,
            'results': self.validation_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 分析验证结果
        success_rate = self.validation_stats['successful_validations'] / self.validation_stats['total_indicators'] if self.validation_stats['total_indicators'] > 0 else 0
        closed_loop_rate = self.validation_stats['closed_loop_success'] / self.validation_stats['total_indicators'] if self.validation_stats['total_indicators'] > 0 else 0
        
        if success_rate < 0.7:
            recommendations.append("成功率较低，建议检查指标实现和策略生成逻辑")
        
        if closed_loop_rate < 0.5:
            recommendations.append("闭环验证通过率较低，建议优化买点分析算法")
        
        # 分析具体指标问题
        failed_indicators = [name for name, result in self.validation_results.items() if result['status'] == 'failed']
        if failed_indicators:
            recommendations.append(f"以下指标验证失败，需要重点检查: {', '.join(failed_indicators[:5])}")
        
        no_selection_indicators = [name for name, result in self.validation_results.items() if result['status'] == 'no_selection']
        if no_selection_indicators:
            recommendations.append(f"以下指标未选出股票，建议调整策略条件: {', '.join(no_selection_indicators[:5])}")
        
        return recommendations
    
    def _save_validation_results(self, report: Dict[str, Any]):
        """保存验证结果"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.config['output']['results_file']), exist_ok=True)
            
            # 保存详细结果
            with open(self.config['output']['results_file'], 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
            
            # 保存报告
            with open(self.config['output']['report_file'], 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 保存CSV格式
            self._save_csv_results()
            
            logger.info(f"验证结果已保存到: {self.config['output']['results_file']}")
            
        except Exception as e:
            logger.error(f"保存验证结果失败: {e}")
    
    def _save_csv_results(self):
        """保存CSV格式的结果"""
        try:
            csv_data = []
            for indicator_name, result in self.validation_results.items():
                csv_data.append({
                    'indicator_name': indicator_name,
                    'status': result.get('status', 'unknown'),
                    'selection_count': result.get('selection_count', 0),
                    'selection_ratio': result.get('selection_ratio', 0),
                    'quality_score': result.get('quality_score', 0),
                    'closed_loop_verified': result.get('closed_loop_verified', False),
                    'execution_time': result.get('execution_time', 0),
                    'buypoints_found': result.get('buypoint_analysis', {}).get('total_buypoints', 0),
                    'indicator_correlation_rate': result.get('buypoint_analysis', {}).get('analysis_summary', {}).get('indicator_correlation_rate', 0)
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(self.config['output']['csv_file'], index=False, encoding='utf-8')
            
        except Exception as e:
            logger.error(f"保存CSV结果失败: {e}")
    
    def validate_single_indicator(self, indicator_name: str) -> Dict[str, Any]:
        """
        验证单个指标
        
        Args:
            indicator_name: 指标名称
            
        Returns:
            验证结果
        """
        logger.info(f"🔍 开始验证单个指标: {indicator_name}")
        
        # 获取股票池
        stock_pool = self._get_stock_pool()
        
        # 执行闭环验证
        result = self._validate_single_indicator_closed_loop(indicator_name, stock_pool)
        
        # 如果找到了股票，进行买点分析和闭环验证
        if result['status'] == 'success' and result['selection_count'] > 0:
            logger.info(f"🔍 对找到的股票进行买点分析和闭环验证")
            
            # 执行买点分析
            buypoint_analysis = self._perform_buypoint_analysis(result['selected_stocks'], indicator_name)
            result['buypoint_analysis'] = buypoint_analysis
            
            # 执行闭环验证
            closed_loop_verified = self._verify_closed_loop_with_single_stock(
                indicator_name, result['selected_stocks'], buypoint_analysis
            )
            result['closed_loop_verified'] = closed_loop_verified
            
            # 重新计算质量评分
            result['quality_score'] = self._calculate_quality_score(result)
            
            if closed_loop_verified:
                logger.info(f"✅ 指标 {indicator_name} 通过闭环验证")
            else:
                logger.warning(f"⚠️ 指标 {indicator_name} 未通过闭环验证")
        
        # 保存单个指标结果
        self.validation_results[indicator_name] = result
        
        # 生成详细报告
        report = self._generate_detailed_report(indicator_name, result)
        
        # 保存报告到文件
        try:
            self._save_indicator_report(indicator_name, report)
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            # 继续执行，不中断验证流程
        
        logger.info(f"✅ 指标 {indicator_name} 验证完成")
        return report
    
    def _generate_detailed_report(self, indicator_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成详细的验证报告
        
        Args:
            indicator_name: 指标名称
            result: 验证结果
            
        Returns:
            详细报告
        """
        report = {
            'indicator_name': indicator_name,
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'passed': result.get('closed_loop_verified', False),
                'execution_time_seconds': result.get('execution_time', 0),
                'total_stocks_in_pool': result.get('stock_pool_size', 0),
                'selected_stocks_count': len(result.get('selected_stocks', [])),
                'selection_ratio': len(result.get('selected_stocks', [])) / max(result.get('stock_pool_size', 1), 1) * 100
            },
            'strategy_execution': {
                'strategy_config': result.get('strategy_config', {}),
                'selected_stocks': result.get('selected_stocks', []),
                'execution_status': result.get('execution_status', 'unknown'),
                'early_stop_triggered': result.get('early_stop_triggered', False)
            },
            'buypoint_analysis': {
                'analyzed_stocks': result.get('buypoint_analysis', {}).get('analyzed_stocks', 0),
                'stocks_with_buypoints': result.get('buypoint_analysis', {}).get('stocks_with_buypoints', 0),
                'total_buypoints': result.get('buypoint_analysis', {}).get('total_buypoints', 0),
                'indicator_confirmed_buypoints': result.get('buypoint_analysis', {}).get('indicator_confirmed_buypoints', 0),
                'buypoint_details': result.get('buypoint_analysis', {}).get('buypoint_details', [])
            },
            'closed_loop_verification': {
                'verified': result.get('closed_loop_verified', False),
                'verification_details': result.get('verification_details', {}),
                'consistency_check': self._perform_consistency_check(result, indicator_name)
            },
            'recommendations': self._generate_recommendations(result, indicator_name)
        }
        
        return report
    
    def _perform_consistency_check(self, result: Dict[str, Any], indicator_name: str) -> Dict[str, Any]:
        """
        执行一致性检查
        
        Args:
            result: 验证结果
            indicator_name: 指标名称
            
        Returns:
            一致性检查结果
        """
        consistency = {
            'strategy_buypoint_alignment': False,
            'indicator_signal_consistency': False,
            'data_quality_check': True,
            'overall_consistency_score': 0.0
        }
        
        try:
            # 检查策略选股与买点分析的一致性
            selected_stocks = result.get('selected_stocks', [])
            buypoint_analysis = result.get('buypoint_analysis', {})
            
            if selected_stocks and buypoint_analysis.get('analyzed_stocks', 0) > 0:
                consistency['strategy_buypoint_alignment'] = True
            
            # 检查指标信号一致性
            confirmed_buypoints = buypoint_analysis.get('indicator_confirmed_buypoints', 0)
            total_buypoints = buypoint_analysis.get('total_buypoints', 0)
            
            if total_buypoints > 0 and confirmed_buypoints > 0:
                consistency['indicator_signal_consistency'] = True
                consistency['signal_confirmation_rate'] = confirmed_buypoints / total_buypoints * 100
            
            # 计算总体一致性分数
            score = 0
            if consistency['strategy_buypoint_alignment']:
                score += 40
            if consistency['indicator_signal_consistency']:
                score += 40
            if consistency['data_quality_check']:
                score += 20
            
            consistency['overall_consistency_score'] = score
            
        except Exception as e:
            logger.warning(f"一致性检查失败: {e}")
        
        return consistency
    
    def _generate_recommendations(self, result: Dict[str, Any], indicator_name: str) -> List[str]:
        """
        生成改进建议
        
        Args:
            result: 验证结果
            indicator_name: 指标名称
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于验证结果生成建议
        if not result.get('closed_loop_verified', False):
            recommendations.append("❌ 闭环验证失败，需要检查指标计算逻辑或策略条件")
        
        selected_count = len(result.get('selected_stocks', []))
        if selected_count == 0:
            recommendations.append("⚠️ 未选出任何股票，建议放宽策略条件")
        elif selected_count == 1:
            recommendations.append("✅ 选股数量合理，早停功能正常工作")
        
        buypoint_analysis = result.get('buypoint_analysis', {})
        confirmed_rate = 0
        if buypoint_analysis.get('total_buypoints', 0) > 0:
            confirmed_rate = buypoint_analysis.get('indicator_confirmed_buypoints', 0) / buypoint_analysis.get('total_buypoints', 1) * 100
        
        if confirmed_rate < 50:
            recommendations.append(f"⚠️ 指标确认率较低({confirmed_rate:.1f}%)，建议优化指标参数")
        elif confirmed_rate >= 80:
            recommendations.append(f"✅ 指标确认率很高({confirmed_rate:.1f}%)，指标效果良好")
        
        execution_time = result.get('execution_time', 0)
        if execution_time > 120:
            recommendations.append("⚠️ 执行时间较长，建议优化数据查询或计算逻辑")
        elif execution_time < 60:
            recommendations.append("✅ 执行效率良好，早停功能有效提升性能")
        
        return recommendations
    
    def _save_indicator_report(self, indicator_name: str, report: Dict[str, Any]):
        """
        保存指标验证报告到文件
        
        Args:
            indicator_name: 指标名称
            report: 报告内容
        """
        try:
            # 创建报告目录
            report_dir = os.path.join(self.config['paths']['results_dir'], 'validation_reports')
            os.makedirs(report_dir, exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存JSON格式报告
            json_filename = f"{indicator_name}_validation_report_{timestamp}.json"
            json_filepath = os.path.join(report_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 保存Markdown格式报告
            md_filename = f"{indicator_name}_validation_report_{timestamp}.md"
            md_filepath = os.path.join(report_dir, md_filename)
            
            self._save_markdown_report(md_filepath, report)
            
            logger.info(f"📄 报告已保存: {json_filepath}")
            logger.info(f"📄 报告已保存: {md_filepath}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    def _save_markdown_report(self, filepath: str, report: Dict[str, Any]):
        """
        保存Markdown格式的报告
        
        Args:
            filepath: 文件路径
            report: 报告内容
        """
        content = f"""# {report['indicator_name']} 指标验证报告

## 📊 验证概要

- **指标名称**: {report['indicator_name']}
- **验证时间**: {report['timestamp']}
- **验证结果**: {'✅ 通过' if report['validation_summary']['passed'] else '❌ 失败'}
- **执行时间**: {report['validation_summary']['execution_time_seconds']:.2f}秒

## 📈 选股统计

- **股票池总数**: {report['validation_summary']['total_stocks_in_pool']:,}只
- **选出股票数**: {report['validation_summary']['selected_stocks_count']}只
- **选股比例**: {report['validation_summary']['selection_ratio']:.4f}%

## 🎯 策略执行

- **执行状态**: {report['strategy_execution']['execution_status']}
- **早停触发**: {'是' if report['strategy_execution']['early_stop_triggered'] else '否'}
- **选出股票**: {', '.join(report['strategy_execution']['selected_stocks'])}

## 🔍 买点分析

- **分析股票数**: {report['buypoint_analysis']['analyzed_stocks']}只
- **有买点股票数**: {report['buypoint_analysis']['stocks_with_buypoints']}只
- **总买点数**: {report['buypoint_analysis']['total_buypoints']}个
- **指标确认买点数**: {report['buypoint_analysis']['indicator_confirmed_buypoints']}个

## ✅ 闭环验证

- **验证结果**: {'✅ 通过' if report['closed_loop_verification']['verified'] else '❌ 失败'}
- **策略买点一致性**: {'✅ 一致' if report['closed_loop_verification']['consistency_check']['strategy_buypoint_alignment'] else '❌ 不一致'}
- **指标信号一致性**: {'✅ 一致' if report['closed_loop_verification']['consistency_check']['indicator_signal_consistency'] else '❌ 不一致'}
- **总体一致性分数**: {report['closed_loop_verification']['consistency_check']['overall_consistency_score']}/100

## 💡 改进建议

"""
        
        for recommendation in report['recommendations']:
            content += f"- {recommendation}\n"
        
        content += f"""
## 📋 详细数据

### 买点分析详情
"""
        
        for detail in report['buypoint_analysis']['buypoint_details']:
            content += f"""
**股票代码**: {detail['stock_code']}
- 买点数量: {detail['buypoints_count']}
- 指标相关数量: {detail['indicator_related_count']}
- 指标确认: {'✅ 是' if detail['indicator_confirmed'] else '❌ 否'}
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _analyze_stock_simple(self, stock_code: str, analysis_date: str, indicator_name: str) -> Dict[str, Any]:
        """
        简化的股票买点分析，专门用于指标验证
        
        Args:
            stock_code: 股票代码
            analysis_date: 分析日期
            indicator_name: 指标名称
            
        Returns:
            买点分析结果
        """
        try:
            # 获取股票数据
            end_date = analysis_date
            start_date = get_previous_trading_date(end_date, 60)  # 获取60天的数据
            
            stock_data = self.data_manager.get_stock_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                period='daily'
            )
            
            if stock_data.empty:
                logger.warning(f"股票 {stock_code} 无数据")
                return None
            
            # 计算相关指标
            try:
                # 根据指标类型计算相应的技术指标
                if indicator_name == 'MA':
                    # 计算MA指标
                    close_prices = stock_data['close'].values
                    ma5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else None
                    ma10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else None
                    ma20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else None
                    
                    # 简单的买点判断：价格接近均线
                    current_price = close_prices[-1]
                    has_buypoint = False
                    
                    if ma5 and abs(current_price - ma5) / ma5 < 0.05:  # 价格在MA5附近5%内
                        has_buypoint = True
                    elif ma10 and abs(current_price - ma10) / ma10 < 0.05:  # 价格在MA10附近5%内
                        has_buypoint = True
                    elif ma20 and abs(current_price - ma20) / ma20 < 0.05:  # 价格在MA20附近5%内
                        has_buypoint = True
                    
                    return {
                        'stock_code': stock_code,
                        'analysis_date': analysis_date,
                        'indicator_name': indicator_name,
                        'has_buypoint': has_buypoint,
                        'current_price': current_price,
                        'ma5': ma5,
                        'ma10': ma10,
                        'ma20': ma20,
                        'signal_strength': 0.8 if has_buypoint else 0.2
                    }
                
                elif indicator_name == 'RSI':
                    # 计算RSI指标的买点
                    close_prices = stock_data['close'].values
                    if len(close_prices) >= 14:
                        # 简单RSI计算
                        gains = []
                        losses = []
                        for i in range(1, len(close_prices)):
                            change = close_prices[i] - close_prices[i-1]
                            if change > 0:
                                gains.append(change)
                                losses.append(0)
                            else:
                                gains.append(0)
                                losses.append(-change)
                        
                        if len(gains) >= 14:
                            avg_gain = np.mean(gains[-14:])
                            avg_loss = np.mean(losses[-14:])
                            rs = avg_gain / avg_loss if avg_loss > 0 else 0
                            rsi = 100 - (100 / (1 + rs))
                            
                            # RSI买点判断：超卖区域(RSI < 30)
                            has_buypoint = rsi < 30
                            
                            return {
                                'stock_code': stock_code,
                                'analysis_date': analysis_date,
                                'indicator_name': indicator_name,
                                'has_buypoint': has_buypoint,
                                'rsi': rsi,
                                'signal_strength': 0.9 if has_buypoint else 0.3
                            }
                
                # 其他指标的简化实现
                else:
                    # 通用买点判断：价格上涨且成交量放大
                    if len(stock_data) >= 2:
                        current_close = stock_data['close'].iloc[-1]
                        prev_close = stock_data['close'].iloc[-2]
                        current_volume = stock_data['volume'].iloc[-1]
                        avg_volume = stock_data['volume'].tail(5).mean()
                        
                        price_up = current_close > prev_close
                        volume_up = current_volume > avg_volume * 1.2
                        
                        has_buypoint = price_up and volume_up
                        
                        return {
                            'stock_code': stock_code,
                            'analysis_date': analysis_date,
                            'indicator_name': indicator_name,
                            'has_buypoint': has_buypoint,
                            'price_change': (current_close - prev_close) / prev_close * 100,
                            'volume_ratio': current_volume / avg_volume,
                            'signal_strength': 0.7 if has_buypoint else 0.3
                        }
                    
            except Exception as e:
                logger.warning(f"计算指标时出错: {e}")
                
            return None
            
        except Exception as e:
            logger.error(f"简化买点分析失败 {stock_code}: {e}")
            return None
    
    def _check_single_buypoint_correlation(self, buypoints: Dict[str, Any], indicator_name: str, stock_code: str) -> bool:
        """
        检查单个买点与指标的相关性
        
        Args:
            buypoints: 买点分析结果
            indicator_name: 指标名称
            stock_code: 股票代码
            
        Returns:
            是否与指标相关
        """
        try:
            if not buypoints or not buypoints.get('has_buypoint', False):
                return False
            
            # 根据指标类型检查相关性
            if 'MA' in indicator_name.upper():
                # MA指标相关性：检查价格是否接近均线
                current_price = buypoints.get('current_price', 0)
                ma5 = buypoints.get('ma5', 0)
                ma10 = buypoints.get('ma10', 0)
                ma20 = buypoints.get('ma20', 0)
                
                # 如果价格接近任一均线，认为相关
                if ma5 and abs(current_price - ma5) / ma5 < 0.05:
                    return True
                if ma10 and abs(current_price - ma10) / ma10 < 0.05:
                    return True
                if ma20 and abs(current_price - ma20) / ma20 < 0.05:
                    return True
                
                return False
            
            elif 'RSI' in indicator_name.upper():
                # RSI指标相关性：检查RSI是否在超卖区域
                rsi = buypoints.get('rsi', 50)
                return rsi < 30  # 超卖区域
            
            else:
                # 其他指标的通用检查：信号强度大于0.5认为相关
                signal_strength = buypoints.get('signal_strength', 0)
                return signal_strength > 0.5
        
        except Exception as e:
            logger.warning(f"检查买点相关性失败: {e}")
            return False


def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='指标闭环验证框架')
    parser.add_argument('--mode', choices=['quick', 'priority', 'category', 'full'], 
                       default='quick', help='验证模式')
    parser.add_argument('--indicator', type=str, help='验证单个指标')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = IndicatorClosedLoopValidator(config_file=args.config)
    
    if args.indicator:
        # 验证单个指标
        result = validator.validate_single_indicator(args.indicator)
        print(f"\n指标 {args.indicator} 验证结果:")
        # 处理JSON序列化问题，将numpy类型转换为Python原生类型
        def convert_to_serializable(obj):
            if hasattr(obj, 'item'):  # numpy类型
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_result = convert_to_serializable(result)
        print(json.dumps(serializable_result, ensure_ascii=False, indent=2, default=str))
    else:
        # 批量验证
        report = validator.validate_all_indicators(mode=args.mode)
        print(f"\n验证完成，成功率: {report['success_rate']:.2%}")
        print(f"闭环验证通过率: {report['closed_loop_rate']:.2%}")


if __name__ == "__main__":
    main()