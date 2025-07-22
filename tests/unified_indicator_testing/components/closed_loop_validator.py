#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
闭环验证器

实现闭环验证逻辑，验证选股结果与买点分析的一致性
支持反向验证：从选股结果回溯验证买点形态识别
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import time

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

# 首先导入logger
from utils.logger import getLogger
logger = getLogger(__name__)


class ClosedLoopValidator:
    """
    闭环验证器
    
    验证选股结果和买点识别结果的一致性
    
    架构原则：
    - 数据层面：接受模拟数据和真实数据
    - 计算层面：统一使用真实指标计算引擎
    - 验证层面：使用通用的闭环验证逻辑
    """
    
    def __init__(self):
        """初始化闭环验证器"""
        
        # 初始化真实计算引擎
        self.real_indicator_engine = None
        try:
            from analysis.engines.unified_indicator_engine import UnifiedIndicatorEngine
            self.real_indicator_engine = UnifiedIndicatorEngine(enable_cache=True)
            logger.info("✅ 闭环验证器：成功初始化统一指标引擎")
        except Exception as e:
            logger.error(f"❌ 无法初始化统一指标引擎: {e}")
            logger.error("请确保统一指标引擎已正确配置")
            raise RuntimeError(f"统一指标引擎初始化失败: {e}")
        
        # 初始化支持的指标类型
        self.supported_indicators = {
            'MACD', 'RSI', 'KDJ', 'BOLL', 'VOL', 'CCI', 'WR', 'BIAS', 'EMA', 'MA', 'DMI', 'ADX'
        }
        
        logger.info("闭环验证器初始化完成")
    
    def _initialize_buypoint_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化买点形态定义"""
        return {
            'MACD': {
                'GOLDEN_CROSS': {
                    'entry_conditions': [
                        {'field': 'macd_line', 'operator': '>', 'reference': 'signal_line'},
                        {'field': 'histogram', 'operator': '>', 'value': 0},
                        {'field': 'macd_line', 'operator': 'cross_up', 'reference': 'signal_line', 'lookback': 3}
                    ],
                    'confirmation_signals': [
                        {'field': 'volume', 'operator': '>', 'reference': 'volume_ma'},
                        {'field': 'close', 'operator': '>', 'reference': 'open'}
                    ],
                    'risk_controls': [
                        {'field': 'rsi', 'operator': '<', 'value': 80},
                        {'field': 'price_change', 'operator': '>', 'value': -0.05}
                    ]
                },
                'DEATH_CROSS': {
                    'entry_conditions': [
                        {'field': 'macd_line', 'operator': '<', 'reference': 'signal_line'},
                        {'field': 'histogram', 'operator': '<', 'value': 0}
                    ],
                    'confirmation_signals': [],
                    'risk_controls': []
                }
            },
            'RSI': {
                'OVERSOLD': {
                    'entry_conditions': [
                        {'field': 'rsi', 'operator': '<', 'value': 30},
                        {'field': 'rsi', 'operator': 'trend_up', 'lookback': 2}
                    ],
                    'confirmation_signals': [
                        {'field': 'volume', 'operator': '>', 'reference': 'volume_ma'},
                        {'field': 'close', 'operator': '>', 'reference': 'low'}
                    ],
                    'risk_controls': [
                        {'field': 'price_change', 'operator': '>', 'value': -0.1}
                    ]
                },
                'OVERBOUGHT': {
                    'entry_conditions': [
                        {'field': 'rsi', 'operator': '>', 'value': 70}
                    ],
                    'confirmation_signals': [],
                    'risk_controls': []
                }
            },
            'KDJ': {
                'GOLDEN_CROSS': {
                    'entry_conditions': [
                        {'field': 'k', 'operator': 'cross_up', 'reference': 'd', 'lookback': 2},
                        {'field': 'k', 'operator': '<', 'value': 80},
                        {'field': 'j', 'operator': '>', 'reference': 'k'}
                    ],
                    'confirmation_signals': [
                        {'field': 'volume', 'operator': '>', 'reference': 'volume_ma'}
                    ],
                    'risk_controls': [
                        {'field': 'k', 'operator': '<', 'value': 90}
                    ]
                },
                'OVERSOLD': {
                    'entry_conditions': [
                        {'field': 'k', 'operator': '<', 'value': 20},
                        {'field': 'd', 'operator': '<', 'value': 20}
                    ],
                    'confirmation_signals': [],
                    'risk_controls': []
                }
            },
            'BOLL': {
                'LOWER_BREAKOUT': {
                    'entry_conditions': [
                        {'field': 'close', 'operator': '<', 'reference': 'lower_band'},
                        {'field': 'close', 'operator': 'bounce_up', 'reference': 'lower_band', 'lookback': 2}
                    ],
                    'confirmation_signals': [
                        {'field': 'volume', 'operator': '>', 'reference': 'volume_ma'},
                        {'field': 'rsi', 'operator': '<', 'value': 40}
                    ],
                    'risk_controls': [
                        {'field': 'band_width', 'operator': '>', 'value': 0.05}
                    ]
                },
                'SQUEEZE': {
                    'entry_conditions': [
                        {'field': 'band_width', 'operator': '<', 'value': 0.1},
                        {'field': 'close', 'operator': 'near', 'reference': 'middle_band', 'tolerance': 0.02}
                    ],
                    'confirmation_signals': [],
                    'risk_controls': []
                }
            },
            'VOL': {
                'VOLUME_SPIKE': {
                    'entry_conditions': [
                        {'field': 'volume_ratio', 'operator': '>', 'value': 2.0},
                        {'field': 'close', 'operator': '>', 'reference': 'open'}
                    ],
                    'confirmation_signals': [
                        {'field': 'price_change', 'operator': '>', 'value': 0.02}
                    ],
                    'risk_controls': [
                        {'field': 'volume_ratio', 'operator': '<', 'value': 10.0}
                    ]
                }
            }
        }
    
    def validate_selection_results(self, 
                                 selection_results: Dict[str, Any],
                                 stock_data_pool: List[pd.DataFrame],
                                 indicator_name: str,
                                 pattern_type: str) -> Dict[str, Any]:
        """
        验证选股结果
        
        Args:
            selection_results: 选股结果
            stock_data_pool: 股票数据池
            indicator_name: 指标名称
            pattern_type: 形态类型
            
        Returns:
            Dict: 验证结果
        """
        try:
            start_time = time.time()
            
            logger.debug(f"开始闭环验证: {indicator_name}.{pattern_type}")
            
            # 获取选中的股票
            selected_stocks = selection_results.get('selected_stocks', [])
            
            if not selected_stocks:
                return self._create_validation_result(
                    indicator_name, pattern_type, [], 0, 0, 
                    "没有选中的股票需要验证", time.time() - start_time
                )
            
            # 限制验证数量
            max_stocks = self.validation_config['max_validation_stocks']
            if len(selected_stocks) > max_stocks:
                selected_stocks = selected_stocks[:max_stocks]
                logger.info(f"限制验证股票数量为 {max_stocks}")
            
            # 创建股票代码到数据的映射
            stock_data_map = self._create_stock_data_map(stock_data_pool)
            
            # 执行逐个验证
            validation_results = []
            successful_validations = 0
            
            for stock in selected_stocks:
                stock_code = stock.get('code', '')
                
                if stock_code in stock_data_map:
                    validation_result = self._validate_single_stock(
                        stock, stock_data_map[stock_code], indicator_name, pattern_type
                    )
                    validation_results.append(validation_result)
                    
                    if validation_result['is_valid']:
                        successful_validations += 1
                else:
                    # 股票数据不存在
                    validation_results.append({
                        'stock_code': stock_code,
                        'is_valid': False,
                        'confidence_score': 0.0,
                        'validation_details': {'error': '股票数据不存在'},
                        'entry_points': []
                    })
            
            # 计算总体验证结果
            total_validations = len(validation_results)
            validation_rate = successful_validations / total_validations if total_validations > 0 else 0.0
            
            # 更新统计信息
            self._update_validation_stats(indicator_name, pattern_type, validation_results)
            
            execution_time = time.time() - start_time
            
            return self._create_validation_result(
                indicator_name, pattern_type, validation_results,
                successful_validations, total_validations,
                f"验证完成，成功率: {validation_rate:.2%}",
                execution_time
            )
            
        except Exception as e:
            logger.error(f"闭环验证失败: {e}")
            return self._create_validation_result(
                indicator_name, pattern_type, [], 0, 0,
                f"验证过程出错: {e}", time.time() - start_time
            )
    
    def _create_stock_data_map(self, stock_data_pool: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """创建股票代码到数据的映射"""
        stock_map = {}
        
        for data in stock_data_pool:
            if not data.empty and 'code' in data.columns:
                stock_code = data['code'].iloc[0]
                stock_map[stock_code] = data
        
        return stock_map

    def _validate_single_stock(self,
                              stock: Dict[str, Any],
                              stock_data: pd.DataFrame,
                              indicator_name: str,
                              pattern_type: str) -> Dict[str, Any]:
        """验证单个股票"""
        try:
            stock_code = stock.get('code', '')

            # 获取买点形态定义
            pattern_def = self._get_pattern_definition(indicator_name, pattern_type)
            if not pattern_def:
                return {
                    'stock_code': stock_code,
                    'is_valid': False,
                    'confidence_score': 0.0,
                    'validation_details': {'error': f'未找到形态定义: {indicator_name}.{pattern_type}'},
                    'entry_points': []
                }

            # 分析入口点
            entry_points = self._analyze_entry_points(stock_data, pattern_def, indicator_name, pattern_type)

            # 计算验证分数
            validation_score = self._calculate_validation_score(entry_points, pattern_def, stock_data)

            # 判断是否有效
            is_valid = validation_score >= self.validation_config['min_confidence_score']

            # 生成验证详情
            validation_details = self._generate_validation_details(
                entry_points, pattern_def, validation_score, stock_data
            )

            return {
                'stock_code': stock_code,
                'is_valid': is_valid,
                'confidence_score': validation_score,
                'validation_details': validation_details,
                'entry_points': entry_points
            }

        except Exception as e:
            logger.error(f"验证股票 {stock.get('code', 'UNKNOWN')} 失败: {e}")
            return {
                'stock_code': stock.get('code', ''),
                'is_valid': False,
                'confidence_score': 0.0,
                'validation_details': {'error': str(e)},
                'entry_points': []
            }

    def _get_pattern_definition(self, indicator_name: str, pattern_type: str) -> Optional[Dict[str, Any]]:
        """获取形态定义"""
        return self.buypoint_patterns.get(indicator_name, {}).get(pattern_type)

    def _analyze_entry_points(self,
                            stock_data: pd.DataFrame,
                            pattern_def: Dict[str, Any],
                            indicator_name: str,
                            pattern_type: str) -> List[Dict[str, Any]]:
        """分析入口点"""
        try:
            entry_points = []

            if len(stock_data) < 5:  # 数据太少
                return entry_points

            # 计算技术指标（模拟）
            enhanced_data = self._calculate_technical_indicators(stock_data, indicator_name)

            # 检查每个时间点
            for i in range(len(enhanced_data) - 1):  # 不检查最后一天
                current_row = enhanced_data.iloc[i]

                # 检查入口条件
                entry_score = self._check_entry_conditions(
                    enhanced_data, i, pattern_def.get('entry_conditions', [])
                )

                if entry_score > 0.5:  # 满足基本入口条件
                    # 检查确认信号
                    confirmation_score = self._check_confirmation_signals(
                        enhanced_data, i, pattern_def.get('confirmation_signals', [])
                    )

                    # 检查风险控制
                    risk_score = self._check_risk_controls(
                        enhanced_data, i, pattern_def.get('risk_controls', [])
                    )

                    # 计算综合分数
                    total_score = (entry_score * 0.6 + confirmation_score * 0.3 + risk_score * 0.1)

                    if total_score >= 0.6:  # 达到入口点阈值
                        entry_point = {
                            'date': current_row.get('date', ''),
                            'index': i,
                            'entry_price': current_row.get('close', 0),
                            'entry_score': entry_score,
                            'confirmation_score': confirmation_score,
                            'risk_score': risk_score,
                            'total_score': total_score,
                            'pattern_match': pattern_type
                        }
                        entry_points.append(entry_point)

            return entry_points

        except Exception as e:
            logger.error(f"分析入口点失败: {e}")
            return []

    def _calculate_technical_indicators(self, stock_data: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
        """
        计算技术指标
        
        架构原则：统一使用真实计算引擎，不区分数据来源
        数据来源（模拟/真实）的区别在数据本身的标识中体现
        """
        try:
            # 统一使用真实引擎计算
            if not self.real_indicator_engine:
                raise RuntimeError(f"无法计算技术指标{indicator_name}：统一指标引擎未初始化")
            
            return self._calculate_via_real_engine(stock_data, indicator_name)

        except Exception as e:
            logger.error(f"计算技术指标{indicator_name}失败: {e}")
            raise
    
    def _calculate_via_real_engine(self, stock_data: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
        """使用真实指标引擎计算"""
        if not self.real_indicator_engine:
            raise RuntimeError("真实指标引擎未初始化")
        
        # 根据指标类型调用对应的计算方法
        if indicator_name == 'MACD':
            result = self.real_indicator_engine.calculate_macd(stock_data)
            return self._merge_indicator_results(stock_data, result)
        elif indicator_name == 'RSI':
            result = self.real_indicator_engine.calculate_rsi_Engine(stock_data)
            stock_data['rsi'] = result
            return stock_data
        elif indicator_name == 'KDJ':
            result = self.real_indicator_engine.calculate_kdj(stock_data)
            return self._merge_indicator_results(stock_data, result)
        elif indicator_name == 'BOLL':
            result = self.real_indicator_engine.calculate_bollinger_bands(stock_data)
            return self._merge_indicator_results(stock_data, result)
        else:
            # 通用计算
            result = self.real_indicator_engine.calculate_all_indicators(stock_data, [indicator_name])
            return self._merge_indicator_results(stock_data, result)
    
    def _merge_indicator_results(self, original_data: pd.DataFrame, indicator_results: Dict) -> pd.DataFrame:
        """合并指标计算结果到原始数据"""
        enhanced_data = original_data.copy()
        
        for key, values in indicator_results.items():
            if isinstance(values, (pd.Series, list, np.ndarray)):
                enhanced_data[key.lower()] = values
        
        return enhanced_data
    


    def _calculate_macd_simplified(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标（简化实现，仅测试模式）"""
        close = data['close']

        # 计算EMA
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()

        # MACD线
        data['macd_line'] = ema12 - ema26

        # 信号线
        data['signal_line'] = data['macd_line'].ewm(span=9).mean()

        # 柱状图
        data['histogram'] = data['macd_line'] - data['signal_line']

        return data

    def _calculate_rsi_simplified(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标（简化实现，仅测试模式）"""
        close = data['close']
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi'] = data['rsi'].fillna(50)  # 填充NaN值

        return data

    def _calculate_kdj_simplified(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算KDJ指标（简化实现，仅测试模式）"""
        high = data.get('high', data['close'])
        low = data.get('low', data['close'])
        close = data['close']

        # 计算RSV
        lowest_low = low.rolling(window=9).min()
        highest_high = high.rolling(window=9).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)

        # 计算K、D、J
        data['k'] = rsv.ewm(alpha=1/3).mean()
        data['d'] = data['k'].ewm(alpha=1/3).mean()
        data['j'] = 3 * data['k'] - 2 * data['d']

        return data

    def _calculate_bollinger_simplified(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标（简化实现，仅测试模式）"""
        close = data['close']

        # 中轨（移动平均线）
        data['middle_band'] = close.rolling(window=20).mean()

        # 标准差
        std = close.rolling(window=20).std()

        # 上轨和下轨
        data['upper_band'] = data['middle_band'] + (std * 2)
        data['lower_band'] = data['middle_band'] - (std * 2)

        # 带宽
        data['band_width'] = (data['upper_band'] - data['lower_band']) / data['middle_band']
        data['band_width'] = data['band_width'].fillna(0.1)

        return data

    def _check_entry_conditions(self,
                               data: pd.DataFrame,
                               index: int,
                               conditions: List[Dict[str, Any]]) -> float:
        """检查入口条件"""
        if not conditions:
            return 0.5  # 没有条件时返回中性分数

        satisfied_conditions = 0
        total_conditions = len(conditions)

        for condition in conditions:
            if self._evaluate_condition(data, index, condition):
                satisfied_conditions += 1

        return satisfied_conditions / total_conditions

    def _check_confirmation_signals(self,
                                  data: pd.DataFrame,
                                  index: int,
                                  signals: List[Dict[str, Any]]) -> float:
        """检查确认信号"""
        if not signals:
            return 1.0  # 没有确认信号时返回满分

        satisfied_signals = 0
        total_signals = len(signals)

        for signal in signals:
            if self._evaluate_condition(data, index, signal):
                satisfied_signals += 1

        return satisfied_signals / total_signals

    def _check_risk_controls(self,
                           data: pd.DataFrame,
                           index: int,
                           controls: List[Dict[str, Any]]) -> float:
        """检查风险控制"""
        if not controls:
            return 1.0  # 没有风险控制时返回满分

        satisfied_controls = 0
        total_controls = len(controls)

        for control in controls:
            if self._evaluate_condition(data, index, control):
                satisfied_controls += 1

        return satisfied_controls / total_controls

    def _evaluate_condition(self,
                          data: pd.DataFrame,
                          index: int,
                          condition: Dict[str, Any]) -> bool:
        """评估单个条件"""
        try:
            field = condition.get('field', '')
            operator = condition.get('operator', '=')
            value = condition.get('value')
            reference = condition.get('reference')
            lookback = condition.get('lookback', 1)
            tolerance = condition.get('tolerance', 0.02)

            if field not in data.columns:
                return False

            current_value = data.iloc[index][field]

            # 处理不同的操作符
            if operator == '>':
                if reference:
                    if reference in data.columns:
                        reference_value = data.iloc[index][reference]
                        return bool(current_value > reference_value)
                else:
                    return bool(current_value > value)

            elif operator == '<':
                if reference:
                    if reference in data.columns:
                        reference_value = data.iloc[index][reference]
                        return bool(current_value < reference_value)
                else:
                    return bool(current_value < value)

            elif operator == '=':
                if reference:
                    if reference in data.columns:
                        reference_value = data.iloc[index][reference]
                        return bool(abs(current_value - reference_value) < tolerance)
                else:
                    return bool(abs(current_value - value) < tolerance)

            elif operator == 'cross_up':
                return self._check_cross_up(data, index, field, reference, lookback)

            elif operator == 'cross_down':
                return self._check_cross_down(data, index, field, reference, lookback)

            elif operator == 'trend_up':
                return self._check_trend_up(data, index, field, lookback)

            elif operator == 'trend_down':
                return self._check_trend_down(data, index, field, lookback)

            elif operator == 'bounce_up':
                return self._check_bounce_up(data, index, field, reference, lookback)

            elif operator == 'near':
                if reference and reference in data.columns:
                    reference_value = data.iloc[index][reference]
                    return bool(abs(current_value - reference_value) / reference_value < tolerance)

            elif operator == 'between':
                if isinstance(value, list) and len(value) == 2:
                    return bool(value[0] <= current_value <= value[1])

            return False

        except Exception as e:
            logger.debug(f"评估条件失败: {e}")
            return False

    def _check_cross_up(self, data: pd.DataFrame, index: int, field: str, reference: str, lookback: int) -> bool:
        """检查向上穿越"""
        try:
            if index < lookback or reference not in data.columns:
                return False

            # 检查当前值是否大于参考值
            current_field = data.iloc[index][field]
            current_ref = data.iloc[index][reference]

            if current_field <= current_ref:
                return False

            # 检查之前是否小于参考值
            for i in range(1, lookback + 1):
                prev_field = data.iloc[index - i][field]
                prev_ref = data.iloc[index - i][reference]
                if prev_field < prev_ref:
                    return True

            return False

        except Exception:
            return False

    def _check_cross_down(self, data: pd.DataFrame, index: int, field: str, reference: str, lookback: int) -> bool:
        """检查向下穿越"""
        try:
            if index < lookback or reference not in data.columns:
                return False

            # 检查当前值是否小于参考值
            current_field = data.iloc[index][field]
            current_ref = data.iloc[index][reference]

            if current_field >= current_ref:
                return False

            # 检查之前是否大于参考值
            for i in range(1, lookback + 1):
                prev_field = data.iloc[index - i][field]
                prev_ref = data.iloc[index - i][reference]
                if prev_field > prev_ref:
                    return True

            return False

        except Exception:
            return False

    def _check_trend_up(self, data: pd.DataFrame, index: int, field: str, lookback: int) -> bool:
        """检查上升趋势"""
        try:
            if index < lookback:
                return False

            current_value = data.iloc[index][field]
            prev_value = data.iloc[index - lookback][field]

            return current_value > prev_value

        except Exception:
            return False

    def _check_trend_down(self, data: pd.DataFrame, index: int, field: str, lookback: int) -> bool:
        """检查下降趋势"""
        try:
            if index < lookback:
                return False

            current_value = data.iloc[index][field]
            prev_value = data.iloc[index - lookback][field]

            return current_value < prev_value

        except Exception:
            return False

    def _check_bounce_up(self, data: pd.DataFrame, index: int, field: str, reference: str, lookback: int) -> bool:
        """检查反弹向上"""
        try:
            if index < lookback or reference not in data.columns:
                return False

            current_field = data.iloc[index][field]
            current_ref = data.iloc[index][reference]

            # 当前值应该高于参考值
            if current_field <= current_ref:
                return False

            # 检查之前是否低于参考值
            for i in range(1, lookback + 1):
                prev_field = data.iloc[index - i][field]
                prev_ref = data.iloc[index - i][reference]
                if prev_field < prev_ref:
                    return True

            return False

        except Exception:
            return False

    def _calculate_validation_score(self,
                                  entry_points: List[Dict[str, Any]],
                                  pattern_def: Dict[str, Any],
                                  stock_data: pd.DataFrame) -> float:
        """计算验证分数"""
        try:
            if not entry_points:
                return 0.0

            # 基础分数：有入口点就有基础分
            base_score = 0.3

            # 入口点质量分数
            avg_entry_score = sum(ep.get('total_score', 0) for ep in entry_points) / len(entry_points)
            entry_quality_score = avg_entry_score * 0.4

            # 入口点数量分数（适中的数量更好）
            num_points = len(entry_points)
            if num_points == 1:
                quantity_score = 0.2
            elif 2 <= num_points <= 3:
                quantity_score = 0.3
            elif 4 <= num_points <= 5:
                quantity_score = 0.2
            else:
                quantity_score = 0.1  # 太多或太少都不好

            total_score = base_score + entry_quality_score + quantity_score

            return min(1.0, total_score)  # 确保不超过1.0

        except Exception as e:
            logger.error(f"计算验证分数失败: {e}")
            return 0.0

    def _generate_validation_details(self,
                                   entry_points: List[Dict[str, Any]],
                                   pattern_def: Dict[str, Any],
                                   validation_score: float,
                                   stock_data: pd.DataFrame) -> Dict[str, Any]:
        """生成验证详情"""
        try:
            details = {
                'validation_score': validation_score,
                'entry_points_count': len(entry_points),
                'pattern_definition': {
                    'entry_conditions_count': len(pattern_def.get('entry_conditions', [])),
                    'confirmation_signals_count': len(pattern_def.get('confirmation_signals', [])),
                    'risk_controls_count': len(pattern_def.get('risk_controls', []))
                },
                'data_quality': {
                    'data_points': len(stock_data),
                    'has_required_fields': self._check_required_fields(stock_data),
                    'data_completeness': self._calculate_data_completeness(stock_data)
                },
                'entry_points_summary': []
            }

            # 添加入口点摘要
            for i, ep in enumerate(entry_points):
                summary = {
                    'index': i + 1,
                    'date': ep.get('date', ''),
                    'entry_price': ep.get('entry_price', 0),
                    'total_score': ep.get('total_score', 0),
                    'pattern_match': ep.get('pattern_match', '')
                }
                details['entry_points_summary'].append(summary)

            return details

        except Exception as e:
            logger.error(f"生成验证详情失败: {e}")
            return {'error': str(e)}

    def _check_required_fields(self, data: pd.DataFrame) -> bool:
        """检查必需字段"""
        required_fields = ['date', 'code', 'close']
        return all(field in data.columns for field in required_fields)

    def _calculate_data_completeness(self, data: pd.DataFrame) -> float:
        """计算数据完整性"""
        try:
            if data.empty:
                return 0.0

            total_cells = data.size
            non_null_cells = data.count().sum()

            return non_null_cells / total_cells

        except Exception:
            return 0.0

    def _create_validation_result(self,
                                indicator_name: str,
                                pattern_type: str,
                                validation_results: List[Dict[str, Any]],
                                successful_validations: int,
                                total_validations: int,
                                message: str,
                                execution_time: float) -> Dict[str, Any]:
        """创建验证结果"""
        validation_rate = successful_validations / total_validations if total_validations > 0 else 0.0

        return {
            'indicator_name': indicator_name,
            'pattern_type': pattern_type,
            'validation_rate': validation_rate,
            'successful_validations': successful_validations,
            'total_validations': total_validations,
            'execution_time': execution_time,
            'message': message,
            'validation_results': validation_results,
            'summary': {
                'avg_confidence_score': self._calculate_avg_confidence(validation_results),
                'entry_points_distribution': self._calculate_entry_points_distribution(validation_results),
                'validation_quality': self._assess_validation_quality(validation_rate, validation_results)
            },
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_avg_confidence(self, validation_results: List[Dict[str, Any]]) -> float:
        """计算平均置信度"""
        if not validation_results:
            return 0.0

        total_confidence = sum(result.get('confidence_score', 0) for result in validation_results)
        return total_confidence / len(validation_results)

    def _calculate_entry_points_distribution(self, validation_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算入口点分布"""
        distribution = {'0': 0, '1': 0, '2-3': 0, '4+': 0}

        for result in validation_results:
            entry_points_count = len(result.get('entry_points', []))

            if entry_points_count == 0:
                distribution['0'] += 1
            elif entry_points_count == 1:
                distribution['1'] += 1
            elif 2 <= entry_points_count <= 3:
                distribution['2-3'] += 1
            else:
                distribution['4+'] += 1

        return distribution

    def _assess_validation_quality(self, validation_rate: float, validation_results: List[Dict[str, Any]]) -> str:
        """评估验证质量"""
        if validation_rate >= 0.8:
            return "优秀"
        elif validation_rate >= 0.6:
            return "良好"
        elif validation_rate >= 0.4:
            return "一般"
        elif validation_rate >= 0.2:
            return "较差"
        else:
            return "很差"

    def _update_validation_stats(self,
                               indicator_name: str,
                               pattern_type: str,
                               validation_results: List[Dict[str, Any]]):
        """更新验证统计"""
        try:
            pattern_key = f"{indicator_name}_{pattern_type}"

            # 更新总体统计
            self.validation_stats['total_validated'] += len(validation_results)
            self.validation_stats['successful_validations'] += sum(
                1 for result in validation_results if result.get('is_valid', False)
            )
            self.validation_stats['failed_validations'] += sum(
                1 for result in validation_results if not result.get('is_valid', False)
            )

            # 更新形态匹配统计
            if pattern_key not in self.validation_stats['pattern_matches']:
                self.validation_stats['pattern_matches'][pattern_key] = {
                    'total': 0,
                    'successful': 0,
                    'avg_confidence': 0.0
                }

            pattern_stats = self.validation_stats['pattern_matches'][pattern_key]
            pattern_stats['total'] += len(validation_results)
            pattern_stats['successful'] += sum(
                1 for result in validation_results if result.get('is_valid', False)
            )

            # 计算平均置信度
            if validation_results:
                avg_confidence = sum(
                    result.get('confidence_score', 0) for result in validation_results
                ) / len(validation_results)
                pattern_stats['avg_confidence'] = avg_confidence

            # 更新入口点准确性
            for result in validation_results:
                confidence = result.get('confidence_score', 0)
                self.validation_stats['entry_point_accuracy'].append(confidence)

        except Exception as e:
            logger.error(f"更新验证统计失败: {e}")

    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        try:
            stats = self.validation_stats.copy()

            # 计算总体成功率
            if stats['total_validated'] > 0:
                stats['overall_success_rate'] = stats['successful_validations'] / stats['total_validated']
            else:
                stats['overall_success_rate'] = 0.0

            # 计算平均入口点准确性
            if stats['entry_point_accuracy']:
                stats['avg_entry_point_accuracy'] = sum(stats['entry_point_accuracy']) / len(stats['entry_point_accuracy'])
            else:
                stats['avg_entry_point_accuracy'] = 0.0

            return stats

        except Exception as e:
            logger.error(f"获取验证统计失败: {e}")
            return {}

    def cleanup(self):
        """清理资源"""
        try:
            # 重置统计信息
            self.validation_stats = {
                'total_validated': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'pattern_matches': {},
                'entry_point_accuracy': []
            }
            logger.debug("闭环验证器资源清理完成")
        except Exception as e:
            logger.warning(f"清理资源时出现警告: {e}")

    def get_supported_patterns(self) -> Dict[str, List[str]]:
        """获取支持的形态列表"""
        supported = {}
        for indicator, patterns in self.buypoint_patterns.items():
            supported[indicator] = list(patterns.keys())
        return supported
