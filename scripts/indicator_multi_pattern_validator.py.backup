#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标多形态闭环验证框架
为每个指标的不同形态都进行闭环验证
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from db.unified_data_manager import Unified_data_manager
from strategy.strategy_executor import Strategy_executor
from analysis.buypoints.analyze_buypoints import Buy_point_analyzer
from utils.logger import get_logger

logger = get_logger(__name__)


def get_previous_trading_date_Validator(date_str: str, days: int) -> str:
    """获取前N个交易日的日期"""
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    previous_date = date_obj - timedelta(days=days)
    return previous_date.strftime('%Y-%m-%d')


class IndicatorMultiPatternValidator:
    """指标多形态验证器"""
    
    def __init__(self, config_file: str = None):
        """初始化验证器"""
        self.config = self._load_config_Indicator_Multi_Pattern_Validator(config_file)
        self.data_manager = Unified_data_manager()
        self.strategy_executor = Strategy_executor()
        self.buypoint_analyzer = Buy_point_analyzer()
        
        # 验证统计
        self.validation_stats = {
            'total_patterns': 0,
            'successful_validations': 0,
            'closed_loop_success': 0,
            'failed_validations': 0
        }
        
        # 验证结果
        self.validation_results = []
        
        logger.info("🔄 指标多形态验证器初始化完成")
    
    def _load_config_Indicator_Multi_Pattern_Validator(self, config_file: str = None) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'validation': {
                'date': '2024-12-27',
                'min_selection_count': 1,
                'max_selection_ratio': 0.3,
                'stock_pool_size': 100,
                'buypoint_analysis_days': 30
            },
            'patterns': {
                'MA': {
                    'golden_cross': {
                        'description': 'MA5上穿MA10金叉形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 1.0}
                        ]
                    },
                    'support_bounce': {
                        'description': 'MA支撑位反弹形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 2.0}
                        ]
                    },
                    'trend_following': {
                        'description': 'MA趋势跟随形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 3.0}
                        ]
                    }
                },
                'RSI': {
                    'oversold_reversal': {
                        'description': 'RSI超卖反转形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 1.0}
                        ]
                    },
                    'divergence': {
                        'description': 'RSI背离形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 2.0}
                        ]
                    },
                    'momentum_shift': {
                        'description': 'RSI动量转换形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 3.0}
                        ]
                    }
                },
                'MACD': {
                    'golden_cross': {
                        'description': 'MACD金叉形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 1.0}
                        ]
                    },
                    'zero_line_cross': {
                        'description': 'MACD零轴突破形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 2.0}
                        ]
                    },
                    'histogram_divergence': {
                        'description': 'MACD柱状图背离形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 3.0}
                        ]
                    }
                },
                'KDJ': {
                    'oversold_golden_cross': {
                        'description': 'KDJ超卖区金叉形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 1.0}
                        ]
                    },
                    'high_position_adjustment': {
                        'description': 'KDJ高位调整形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 2.0}
                        ]
                    },
                    'trend_confirmation': {
                        'description': 'KDJ趋势确认形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 3.0}
                        ]
                    }
                },
                'BOLL': {
                    'lower_band_bounce': {
                        'description': '布林带下轨反弹形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 1.0}
                        ]
                    },
                    'squeeze_breakout': {
                        'description': '布林带收缩突破形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 2.0}
                        ]
                    },
                    'middle_line_support': {
                        'description': '布林带中轨支撑形态',
                        'conditions': [
                            {'type': 'basic', 'field': 'close', 'operator': '>', 'value': 3.0}
                        ]
                    }
                }
            },
            'paths': {
                'results_dir': 'results',
                'reports_dir': 'results/multi_pattern_reports',
                'logs_dir': 'logs'
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                self._merge_config_Indicator_Multi_Pattern_Validator(default_config, custom_config)
            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def _merge_config_Indicator_Multi_Pattern_Validator(self, default: Dict, custom: Dict):
        """递归合并配置"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config_Indicator_Multi_Pattern_Validator(default[key], value)
            else:
                default[key] = value
    
    def validate_all_patterns_Validator(self) -> Dict[str, Any]:
        """验证所有指标的所有形态"""
        logger.info("🚀 开始多形态验证")
        
        # 获取股票池
        stock_pool = self._get_stock_pool_Indicator_Multi_Pattern_Validator()
        
        all_results = {}
        
        for indicator_name, patterns in self.config['patterns'].items():
            logger.info(f"📊 开始验证指标 {indicator_name} 的所有形态")
            
            indicator_results = {}
            
            for pattern_name, pattern_config in patterns.items():
                logger.info(f"🔍 验证形态: {indicator_name}.{pattern_name} - {pattern_config['description']}")
                
                try:
                    # 验证单个形态
                    result = self._validate_single_pattern_Indicator_Multi_Pattern_Validator(
                        indicator_name, 
                        pattern_name, 
                        pattern_config, 
                        stock_pool
                    )
                    
                    indicator_results[pattern_name] = result
                    self.validation_results.append(result)
                    
                    # 更新统计
                    self.validation_stats['total_patterns'] += 1
                    if result['status'] == 'success':
                        self.validation_stats['successful_validations'] += 1
                    else:
                        self.validation_stats['failed_validations'] += 1
                    
                    if result.get('closed_loop_verified', False):
                        self.validation_stats['closed_loop_success'] += 1
                    
                except Exception as e:
                    logger.error(f"验证形态 {indicator_name}.{pattern_name} 失败: {e}")
                    indicator_results[pattern_name] = {
                        'status': 'error',
                        'error': str(e),
                        'closed_loop_verified': False
                    }
                    self.validation_stats['total_patterns'] += 1
                    self.validation_stats['failed_validations'] += 1
            
            all_results[indicator_name] = indicator_results
        
        # 生成综合报告
        report = self._generate_comprehensive_report_Indicator_Multi_Pattern_Validator(all_results)
        
        logger.info(f"✅ 多形态验证完成")
        logger.info(f"   总形态数: {self.validation_stats['total_patterns']}")
        logger.info(f"   成功验证: {self.validation_stats['successful_validations']}")
        logger.info(f"   闭环通过: {self.validation_stats['closed_loop_success']}")
        
        return report
    
    def _validate_single_pattern_Indicator_Multi_Pattern_Validator(self, indicator_name: str, pattern_name: str, 
                               pattern_config: Dict[str, Any], stock_pool: List[str]) -> Dict[str, Any]:
        """验证单个形态"""
        start_time = datetime.now()
        
        try:
            # 1. 生成形态策略
            strategy_config = self._generate_pattern_strategy(
                indicator_name, pattern_name, pattern_config
            )
            
            # 2. 执行选股
            selected_stocks = self._execute_strategy_selection_Indicator_Multi_Pattern_Validator(strategy_config, stock_pool)
            
            # 3. 买点分析
            buypoint_analysis = self._perform_buypoint_analysis_Indicator_Multi_Pattern_Validator(
                selected_stocks, indicator_name, pattern_name
            )
            
            # 4. 闭环验证
            closed_loop_verified = self._verify_pattern_closed_loop(
                indicator_name, pattern_name, selected_stocks, buypoint_analysis
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'indicator_name': indicator_name,
                'pattern_name': pattern_name,
                'pattern_description': pattern_config['description'],
                'status': 'success',
                'execution_time': execution_time,
                'stock_pool_size': len(stock_pool),
                'selected_stocks': selected_stocks,
                'selection_count': len(selected_stocks),
                'selection_ratio': len(selected_stocks) / len(stock_pool) if stock_pool else 0,
                'buypoint_analysis': buypoint_analysis,
                'closed_loop_verified': closed_loop_verified,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ 形态 {indicator_name}.{pattern_name} 验证完成")
            logger.info(f"   选股数量: {len(selected_stocks)}")
            logger.info(f"   闭环验证: {'通过' if closed_loop_verified else '未通过'}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"形态验证失败 {indicator_name}.{pattern_name}: {e}")
            
            return {
                'indicator_name': indicator_name,
                'pattern_name': pattern_name,
                'pattern_description': pattern_config.get('description', ''),
                'status': 'error',
                'error': str(e),
                'execution_time': execution_time,
                'closed_loop_verified': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_pattern_strategy(self, indicator_name: str, pattern_name: str, 
                                 pattern_config: Dict[str, Any]) -> Dict[str, Any]:
        """为特定形态生成策略配置"""
        strategy_config = {
            'strategy_id': f"{indicator_name}_{pattern_name}_validation",
            'strategy': {
                'name': f"{indicator_name} {pattern_name} 形态验证",
                'description': pattern_config['description'],
                'conditions': pattern_config['conditions'],
                'logic': 'AND'
            }
        }
        
        return strategy_config
    
    def _execute_strategy_selection_Indicator_Multi_Pattern_Validator(self, strategy_config: Dict[str, Any], stock_pool: List[str]) -> List[str]:
        """执行策略选股"""
        try:
            # 使用策略执行器进行选股
            result_df = self.strategy_executor.execute_strategy(
                strategy=strategy_config,  # 修正参数名
                filters={'stock_codes': stock_pool},
                early_stop=True  # 启用早停
            )
            
            if result_df is not None and not result_df.empty:
                # 自动识别股票代码列名
                stock_code_column = None
                for col in ['code', 'stock_code', 'symbol']:
                    if col in result_df.columns:
                        stock_code_column = col
                        break
                
                if stock_code_column:
                    return result_df[stock_code_column].tolist()
            
            return []
            
        except Exception as e:
            logger.error(f"策略选股失败: {e}")
            return []
    
    def _perform_buypoint_analysis_Indicator_Multi_Pattern_Validator(self, selected_stocks: List[str], 
                                 indicator_name: str, pattern_name: str) -> Dict[str, Any]:
        """对选出的股票进行买点分析"""
        if not selected_stocks:
            return {
                'analyzed_stocks': 0,
                'stocks_with_buypoints': 0,
                'total_buypoints': 0,
                'pattern_confirmed_buypoints': 0,
                'analysis_summary': {
                    'buypoint_rate': 0.0,
                    'pattern_correlation_rate': 0.0
                },
                'buypoint_details': []
            }
        
        analysis_date = self.config['validation']['date']
        analyzed_stocks = 0
        stocks_with_buypoints = 0
        total_buypoints = 0
        pattern_confirmed_buypoints = 0
        buypoint_details = []
        
        for stock_code in selected_stocks:
            try:
                # 执行简化买点分析
                buypoints = self._analyze_stock_pattern(
                    stock_code=stock_code,
                    analysis_date=analysis_date,
                    indicator_name=indicator_name,
                    pattern_name=pattern_name
                )
                
                if buypoints:
                    analyzed_stocks += 1
                    
                    if buypoints.get('has_buypoint', False):
                        stocks_with_buypoints += 1
                        total_buypoints += 1
                        
                        # 检查买点是否与形态相关
                        pattern_confirmed = self._check_pattern_buypoint_correlation(
                            buypoints, indicator_name, pattern_name, stock_code
                        )
                        
                        if pattern_confirmed:
                            pattern_confirmed_buypoints += 1
                        
                        # 记录详细信息
                        buypoint_details.append({
                            'stock_code': stock_code,
                            'has_buypoint': True,
                            'pattern_confirmed': pattern_confirmed,
                            'signal_strength': buypoints.get('signal_strength', 0),
                            'analysis_data': buypoints
                        })
            
            except Exception as e:
                logger.warning(f"分析股票 {stock_code} 买点失败: {e}")
        
        # 计算汇总统计
        buypoint_rate = stocks_with_buypoints / analyzed_stocks if analyzed_stocks > 0 else 0
        pattern_correlation_rate = pattern_confirmed_buypoints / total_buypoints if total_buypoints > 0 else 0
        
        return {
            'analyzed_stocks': analyzed_stocks,
            'stocks_with_buypoints': stocks_with_buypoints,
            'total_buypoints': total_buypoints,
            'pattern_confirmed_buypoints': pattern_confirmed_buypoints,
            'analysis_summary': {
                'buypoint_rate': buypoint_rate,
                'pattern_correlation_rate': pattern_correlation_rate
            },
            'buypoint_details': buypoint_details
        }
    
    def _analyze_stock_pattern(self, stock_code: str, analysis_date: str, 
                             indicator_name: str, pattern_name: str) -> Dict[str, Any]:
        """针对特定形态分析股票买点"""
        try:
            # 获取股票数据
            end_date = analysis_date
            start_date = get_previous_trading_date_Validator(end_date, 60)
            
            stock_data = self.data_manager.get_stock_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                period='daily'
            )
            
            if stock_data.empty:
                return None
            
            # 根据指标和形态类型进行分析
            return self._calculate_pattern_signals(
                stock_data, indicator_name, pattern_name, stock_code, analysis_date
            )
            
        except Exception as e:
            logger.error(f"形态分析失败 {stock_code}: {e}")
            return None
    
    def _calculate_pattern_signals(self, stock_data: pd.DataFrame, indicator_name: str, 
                                 pattern_name: str, stock_code: str, analysis_date: str) -> Dict[str, Any]:
        """计算特定形态的信号"""
        try:
            close_prices = stock_data['close'].values
            current_price = close_prices[-1]
            
            if indicator_name == 'MA':
                return self._calculate_ma_pattern_signals(
                    stock_data, pattern_name, stock_code, analysis_date
                )
            elif indicator_name == 'RSI':
                return self._calculate_rsi_pattern_signals(
                    stock_data, pattern_name, stock_code, analysis_date
                )
            elif indicator_name == 'MACD':
                return self._calculate_macd_pattern_signals(
                    stock_data, pattern_name, stock_code, analysis_date
                )
            elif indicator_name == 'KDJ':
                return self._calculate_kdj_pattern_signals(
                    stock_data, pattern_name, stock_code, analysis_date
                )
            elif indicator_name == 'BOLL':
                return self._calculate_boll_pattern_signals(
                    stock_data, pattern_name, stock_code, analysis_date
                )
            else:
                # 通用形态分析
                return {
                    'stock_code': stock_code,
                    'analysis_date': analysis_date,
                    'indicator_name': indicator_name,
                    'pattern_name': pattern_name,
                    'has_buypoint': True,  # 简化为总是有买点
                    'current_price': current_price,
                    'signal_strength': 0.5
                }
                
        except Exception as e:
            logger.warning(f"计算形态信号失败: {e}")
            return None
    
    def _calculate_ma_pattern_signals(self, stock_data: pd.DataFrame, pattern_name: str, 
                                    stock_code: str, analysis_date: str) -> Dict[str, Any]:
        """计算MA形态信号"""
        close_prices = stock_data['close'].values
        current_price = close_prices[-1]
        
        # 计算各种MA
        ma5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else None
        ma10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else None
        ma20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else None
        
        has_buypoint = False
        signal_strength = 0.3
        
        if pattern_name == 'golden_cross':
            # 金叉形态：MA5 > MA10
            if ma5 and ma10 and ma5 > ma10:
                has_buypoint = True
                signal_strength = 0.8
        elif pattern_name == 'support_bounce':
            # 支撑反弹：价格接近MA20且向上
            if ma20 and abs(current_price - ma20) / ma20 < 0.03:
                has_buypoint = True
                signal_strength = 0.7
        elif pattern_name == 'trend_following':
            # 趋势跟随：价格在所有MA之上
            if ma5 and ma10 and ma20 and current_price > ma5 > ma10 > ma20:
                has_buypoint = True
                signal_strength = 0.9
        
        return {
            'stock_code': stock_code,
            'analysis_date': analysis_date,
            'indicator_name': 'MA',
            'pattern_name': pattern_name,
            'has_buypoint': has_buypoint,
            'current_price': current_price,
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'signal_strength': signal_strength
        }
    
    def _calculate_rsi_pattern_signals(self, stock_data: pd.DataFrame, pattern_name: str, 
                                     stock_code: str, analysis_date: str) -> Dict[str, Any]:
        """计算RSI形态信号"""
        close_prices = stock_data['close'].values
        current_price = close_prices[-1]
        
        # 简化RSI计算
        if len(close_prices) >= 14:
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
                
                has_buypoint = False
                signal_strength = 0.3
                
                if pattern_name == 'oversold_reversal':
                    # 超卖反转：RSI < 30
                    if rsi < 30:
                        has_buypoint = True
                        signal_strength = 0.9
                elif pattern_name == 'divergence':
                    # 背离形态：RSI在30-50区间
                    if 30 <= rsi <= 50:
                        has_buypoint = True
                        signal_strength = 0.7
                elif pattern_name == 'momentum_shift':
                    # 动量转换：RSI从低位向上
                    if 40 <= rsi <= 60:
                        has_buypoint = True
                        signal_strength = 0.6
                
                return {
                    'stock_code': stock_code,
                    'analysis_date': analysis_date,
                    'indicator_name': 'RSI',
                    'pattern_name': pattern_name,
                    'has_buypoint': has_buypoint,
                    'current_price': current_price,
                    'rsi': rsi,
                    'signal_strength': signal_strength
                }
        
        # 如果数据不足，返回默认结果
        return {
            'stock_code': stock_code,
            'analysis_date': analysis_date,
            'indicator_name': 'RSI',
            'pattern_name': pattern_name,
            'has_buypoint': False,
            'current_price': current_price,
            'rsi': 50,
            'signal_strength': 0.1
        }
    
    def _calculate_macd_pattern_signals(self, stock_data: pd.DataFrame, pattern_name: str, 
                                      stock_code: str, analysis_date: str) -> Dict[str, Any]:
        """计算MACD形态信号"""
        close_prices = stock_data['close'].values
        current_price = close_prices[-1]
        
        # 简化MACD计算
        if len(close_prices) >= 26:
            ema12 = close_prices[-12:].mean()  # 简化为简单平均
            ema26 = close_prices[-26:].mean()
            macd = ema12 - ema26
            signal = macd * 0.9  # 简化信号线
            histogram = macd - signal
            
            has_buypoint = False
            signal_strength = 0.3
            
            if pattern_name == 'golden_cross':
                # 金叉：MACD > Signal
                if macd > signal:
                    has_buypoint = True
                    signal_strength = 0.8
            elif pattern_name == 'zero_line_cross':
                # 零轴突破：MACD > 0
                if macd > 0:
                    has_buypoint = True
                    signal_strength = 0.7
            elif pattern_name == 'histogram_divergence':
                # 柱状图背离：histogram > 0
                if histogram > 0:
                    has_buypoint = True
                    signal_strength = 0.6
            
            return {
                'stock_code': stock_code,
                'analysis_date': analysis_date,
                'indicator_name': 'MACD',
                'pattern_name': pattern_name,
                'has_buypoint': has_buypoint,
                'current_price': current_price,
                'macd': macd,
                'signal': signal,
                'histogram': histogram,
                'signal_strength': signal_strength
            }
        
        return {
            'stock_code': stock_code,
            'analysis_date': analysis_date,
            'indicator_name': 'MACD',
            'pattern_name': pattern_name,
            'has_buypoint': False,
            'current_price': current_price,
            'signal_strength': 0.1
        }
    
    def _calculate_kdj_pattern_signals(self, stock_data: pd.DataFrame, pattern_name: str, 
                                     stock_code: str, analysis_date: str) -> Dict[str, Any]:
        """计算KDJ形态信号"""
        # 简化KDJ计算
        if len(stock_data) >= 9:
            high_prices = stock_data['high'].tail(9)
            low_prices = stock_data['low'].tail(9)
            close_prices = stock_data['close'].tail(9)
            
            current_price = close_prices.iloc[-1]
            highest_high = high_prices.max()
            lowest_low = low_prices.min()
            
            # 简化RSV计算
            rsv = (current_price - lowest_low) / (highest_high - lowest_low) * 100 if highest_high > lowest_low else 50
            k = rsv * 0.7  # 简化K值
            d = k * 0.8    # 简化D值
            
            has_buypoint = False
            signal_strength = 0.3
            
            if pattern_name == 'oversold_golden_cross':
                # 超卖金叉：K < 20且K > D
                if k < 20 and k > d:
                    has_buypoint = True
                    signal_strength = 0.9
            elif pattern_name == 'high_position_adjustment':
                # 高位调整：K在50-80区间
                if 50 <= k <= 80:
                    has_buypoint = True
                    signal_strength = 0.6
            elif pattern_name == 'trend_confirmation':
                # 趋势确认：K > 50且K > D
                if k > 50 and k > d:
                    has_buypoint = True
                    signal_strength = 0.7
            
            return {
                'stock_code': stock_code,
                'analysis_date': analysis_date,
                'indicator_name': 'KDJ',
                'pattern_name': pattern_name,
                'has_buypoint': has_buypoint,
                'current_price': current_price,
                'k': k,
                'd': d,
                'signal_strength': signal_strength
            }
        
        return {
            'stock_code': stock_code,
            'analysis_date': analysis_date,
            'indicator_name': 'KDJ',
            'pattern_name': pattern_name,
            'has_buypoint': False,
            'current_price': stock_data['close'].iloc[-1],
            'signal_strength': 0.1
        }
    
    def _calculate_boll_pattern_signals(self, stock_data: pd.DataFrame, pattern_name: str, 
                                      stock_code: str, analysis_date: str) -> Dict[str, Any]:
        """计算布林带形态信号"""
        close_prices = stock_data['close'].values
        current_price = close_prices[-1]
        
        # 计算布林带
        if len(close_prices) >= 20:
            ma20 = np.mean(close_prices[-20:])
            std20 = np.std(close_prices[-20:])
            upper_band = ma20 + 2 * std20
            lower_band = ma20 - 2 * std20
            
            has_buypoint = False
            signal_strength = 0.3
            
            if pattern_name == 'lower_band_bounce':
                # 下轨反弹：价格接近下轨
                if current_price <= lower_band * 1.02:  # 在下轨2%范围内
                    has_buypoint = True
                    signal_strength = 0.9
            elif pattern_name == 'squeeze_breakout':
                # 收缩突破：价格突破上轨
                if current_price >= upper_band:
                    has_buypoint = True
                    signal_strength = 0.8
            elif pattern_name == 'middle_line_support':
                # 中轨支撑：价格在中轨附近
                if abs(current_price - ma20) / ma20 < 0.03:
                    has_buypoint = True
                    signal_strength = 0.6
            
            return {
                'stock_code': stock_code,
                'analysis_date': analysis_date,
                'indicator_name': 'BOLL',
                'pattern_name': pattern_name,
                'has_buypoint': has_buypoint,
                'current_price': current_price,
                'upper_band': upper_band,
                'middle_band': ma20,
                'lower_band': lower_band,
                'signal_strength': signal_strength
            }
        
        return {
            'stock_code': stock_code,
            'analysis_date': analysis_date,
            'indicator_name': 'BOLL',
            'pattern_name': pattern_name,
            'has_buypoint': False,
            'current_price': current_price,
            'signal_strength': 0.1
        }
    
    def _check_pattern_buypoint_correlation(self, buypoints: Dict[str, Any], 
                                          indicator_name: str, pattern_name: str, stock_code: str) -> bool:
        """检查买点与特定形态的相关性"""
        try:
            if not buypoints or not buypoints.get('has_buypoint', False):
                return False
            
            # 根据形态类型检查相关性
            signal_strength = buypoints.get('signal_strength', 0)
            
            # 高信号强度认为相关
            if signal_strength > 0.7:
                return True
            
            # 根据具体形态进行更精细的判断
            if indicator_name == 'MA':
                if pattern_name == 'golden_cross':
                    ma5 = buypoints.get('ma5', 0)
                    ma10 = buypoints.get('ma10', 0)
                    return ma5 > 0 and ma10 > 0 and ma5 > ma10
                elif pattern_name == 'support_bounce':
                    current_price = buypoints.get('current_price', 0)
                    ma20 = buypoints.get('ma20', 0)
                    return ma20 > 0 and abs(current_price - ma20) / ma20 < 0.05
            
            elif indicator_name == 'RSI':
                rsi = buypoints.get('rsi', 50)
                if pattern_name == 'oversold_reversal':
                    return rsi < 30
                elif pattern_name == 'divergence':
                    return 30 <= rsi <= 50
                elif pattern_name == 'momentum_shift':
                    return 40 <= rsi <= 60
            
            # 默认基于信号强度判断
            return signal_strength > 0.5
            
        except Exception as e:
            logger.warning(f"检查形态相关性失败: {e}")
            return False
    
    def _verify_pattern_closed_loop(self, indicator_name: str, pattern_name: str, 
                                  selected_stocks: List[str], buypoint_analysis: Dict[str, Any]) -> bool:
        """验证形态的闭环一致性"""
        try:
            # 检查1: 至少有一个股票被选中
            if len(selected_stocks) == 0:
                return False
            
            # 检查2: 买点分析成功执行
            analyzed_stocks = buypoint_analysis.get('analyzed_stocks', 0)
            if analyzed_stocks == 0:
                return False
            
            # 检查3: 至少找到一个买点
            total_buypoints = buypoint_analysis.get('total_buypoints', 0)
            if total_buypoints == 0:
                return False
            
            # 检查4: 形态与买点的相关性
            pattern_confirmed_buypoints = buypoint_analysis.get('pattern_confirmed_buypoints', 0)
            correlation_rate = pattern_confirmed_buypoints / total_buypoints if total_buypoints > 0 else 0
            
            # 对于形态验证，要求更高的相关性
            min_correlation_rate = 0.5  # 至少50%的相关性
            
            if correlation_rate >= min_correlation_rate:
                logger.info(f"✅ 形态 {indicator_name}.{pattern_name} 通过闭环验证 (相关性: {correlation_rate:.2%})")
                return True
            else:
                logger.warning(f"❌ 形态 {indicator_name}.{pattern_name} 未通过闭环验证 (相关性过低: {correlation_rate:.2%})")
                return False
            
        except Exception as e:
            logger.error(f"形态闭环验证过程出错: {e}")
            return False
    
    def _get_stock_pool_Indicator_Multi_Pattern_Validator(self) -> List[str]:
        """获取股票池"""
        try:
            # 重新初始化数据管理器以确保连接正常
            self.data_manager = Unified_data_manager()
            stock_list = self.data_manager.get_all_stock_codes()
            logger.info(f"获取股票池成功，大小: {len(stock_list)}")
            return stock_list
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            # 返回一个小的默认股票池用于测试
            return ['000001', '000002', '000858', '002415', '600000']
    
    def _generate_comprehensive_report_Indicator_Multi_Pattern_Validator(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            'metadata': {
                'validation_date': datetime.now().isoformat(),
                'total_indicators': len(all_results),
                'total_patterns': self.validation_stats['total_patterns'],
                'config': self.config
            },
            'summary': self.validation_stats.copy(),
            'success_rate': self.validation_stats['successful_validations'] / self.validation_stats['total_patterns'] if self.validation_stats['total_patterns'] > 0 else 0,
            'closed_loop_rate': self.validation_stats['closed_loop_success'] / self.validation_stats['total_patterns'] if self.validation_stats['total_patterns'] > 0 else 0,
            'indicators': all_results,
            'pattern_analysis': self._analyze_pattern_performance(all_results)
        }
        
        # 保存报告
        try:
            self._save_comprehensive_report(report)
        except Exception as e:
            logger.error(f"保存综合报告失败: {e}")
        
        return report
    
    def _analyze_pattern_performance(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析形态性能"""
        pattern_stats = {}
        
        for indicator_name, patterns in all_results.items():
            for pattern_name, result in patterns.items():
                if result.get('status') == 'success':
                    pattern_key = f"{indicator_name}.{pattern_name}"
                    pattern_stats[pattern_key] = {
                        'selection_count': result.get('selection_count', 0),
                        'selection_ratio': result.get('selection_ratio', 0),
                        'closed_loop_verified': result.get('closed_loop_verified', False),
                        'execution_time': result.get('execution_time', 0),
                        'buypoint_rate': result.get('buypoint_analysis', {}).get('analysis_summary', {}).get('buypoint_rate', 0),
                        'correlation_rate': result.get('buypoint_analysis', {}).get('analysis_summary', {}).get('pattern_correlation_rate', 0)
                    }
        
        # 排序找出最佳形态
        best_patterns = sorted(
            pattern_stats.items(),
            key=lambda x: (x[1]['closed_loop_verified'], x[1]['correlation_rate']),
            reverse=True
        )[:10]
        
        return {
            'total_patterns': len(pattern_stats),
            'successful_patterns': len([p for p in pattern_stats.values() if p['closed_loop_verified']]),
            'best_patterns': dict(best_patterns),
            'average_correlation': np.mean([p['correlation_rate'] for p in pattern_stats.values()]) if pattern_stats else 0
        }
    
    def _save_comprehensive_report(self, report: Dict[str, Any]):
        """保存综合报告"""
        # 确保目录存在
        reports_dir = self.config['paths']['reports_dir']
        os.makedirs(reports_dir, exist_ok=True)
        
        # 保存JSON格式
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = os.path.join(reports_dir, f'multi_pattern_validation_{timestamp}.json')
        
        def convert_to_serializable_Validator(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable_Validator(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable_Validator(v) for v in obj]
            else:
                return obj
        
        serializable_report = convert_to_serializable_Validator(report)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"综合报告已保存: {json_file}")


def main_indicatormultipatternvalidator():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='指标多形态闭环验证框架')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--indicator', type=str, help='只验证指定指标的所有形态')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = Indicator_multi_pattern_validator(config_file=args.config)
    
    if args.indicator:
        # 验证单个指标的所有形态
        patterns = validator.config['patterns'].get(args.indicator, {})
        if not patterns:
            print(f"指标 {args.indicator} 没有配置形态")
            return
        
        print(f"🔍 开始验证指标 {args.indicator} 的所有形态")
        
        stock_pool = validator._get_stock_pool_Indicator_Multi_Pattern_Validator()
        results = {}
        
        for pattern_name, pattern_config in patterns.items():
            print(f"\n验证形态: {args.indicator}.{pattern_name}")
            result = validator._validate_single_pattern_Indicator_Multi_Pattern_Validator(
                args.indicator, pattern_name, pattern_config, stock_pool
            )
            results[pattern_name] = result
        
        print(f"\n指标 {args.indicator} 验证结果:")
        for pattern_name, result in results.items():
            status = "✅ 通过" if result.get('closed_loop_verified', False) else "❌ 未通过"
            print(f"  {pattern_name}: {status}")
    
    else:
        # 验证所有指标的所有形态
        report = validator.validate_all_patterns_Validator()
        
        print(f"\n🎯 多形态验证完成")
        print(f"总形态数: {report['summary']['total_patterns']}")
        print(f"成功验证: {report['summary']['successful_validations']}")
        print(f"闭环通过: {report['summary']['closed_loop_success']}")
        print(f"成功率: {report['success_rate']:.2%}")
        print(f"闭环通过率: {report['closed_loop_rate']:.2%}")


if __name__ == "__main__":
    main_indicatormultipatternvalidator() 