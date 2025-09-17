"""
策略条件评估器模块

负责评估策略条件，计算股票是否满足策略要求
"""

import pandas as pd
import numpy as np
import operator
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

from utils.dependency_injection import get_config
from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, cache_result, exception_handler
from indicators.complete_indicator_registry import complete_registry
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import DataAccessInterface
from utils.parameter_standardizer import ParameterStandardizer
from utils.indicator_parameter_validator import IndicatorParameterValidator
from utils.exceptions import (
    StrategyEvaluationError,
    IndicatorExecutionError,
    DataValidationError
)

logger = get_logger(__name__)

class StrategyConditionEvaluator:
    """策略条件评估器，用于高效评估选股策略条件"""
    
    def __init__(self):
        """初始化条件评估器"""
        try:
            self.data_manager = get_service(DataAccessInterface)
        except Exception as e:
            logger.warning(f"依赖注入获取DataAccessInterface失败: {e}, 使用临时实现")
            # 临时实现，直接导入
            from db.managers.data_access_manager import DataAccessManager
            self.data_manager = DataAccessManager()
            
        self.indicator_registry = complete_registry
        self.condition_cache = {}

        # 初始化参数标准化器和验证器
        try:
            self.parameter_standardizer = ParameterStandardizer()
            self.parameter_validator = IndicatorParameterValidator()
        except Exception as e:
            logger.warning(f"参数标准化器/验证器初始化失败: {e}")
            self.parameter_standardizer = None
            self.parameter_validator = None

        logger.info("策略条件评估器已初始化，支持参数标准化和验证")
        
        # 操作符映射
        self.operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_,
        }

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def evaluate_conditions(self, stock_code: str, conditions: List[Dict], start_date: str, end_date: str) -> bool:
        """
        评估股票是否满足策略条件
        
        Args:
            stock_code: 股票代码
            conditions: 条件列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 是否满足所有条件
        """
        try:
            # 获取股票数据
            stock_data = self._get_stock_data(stock_code, start_date, end_date)
            if stock_data.empty:
                logger.warning(f"股票 {stock_code} 数据为空")
                return False
            
            # 计算所有需要的指标
            indicator_values = self._calculate_indicators(stock_data, conditions)
            
            # 评估所有条件
            for condition in conditions:
                if not self._evaluate_single_condition(condition, indicator_values, stock_data):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"评估股票 {stock_code} 条件失败: {e}")
            return False
            
    def _get_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据"""
        try:
            return self.data_manager.get_stock_data(stock_code, start_date, end_date)
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()
            
    def _calculate_indicators(self, stock_data: pd.DataFrame, conditions: List[Dict]) -> Dict[str, Any]:
        """计算指标值"""
        indicator_values = {}
        
        for condition in conditions:
            indicator_name = condition.get('indicator')
            if not indicator_name or indicator_name in indicator_values:
                continue
                
            try:
                # 获取指标计算器
                indicator_calculator = self.indicator_registry.get_indicator(indicator_name)
                if not indicator_calculator:
                    logger.warning(f"未找到指标: {indicator_name}")
                    continue
                
                # 计算指标值
                params = condition.get('parameters', {})
                result = indicator_calculator.calculate(stock_data, **params)
                indicator_values[indicator_name] = result
                
            except Exception as e:
                logger.error(f"计算指标 {indicator_name} 失败: {e}")
                
        return indicator_values
        
    def _evaluate_single_condition(self, condition: Dict, indicator_values: Dict, stock_data: pd.DataFrame) -> bool:
        """评估单个条件"""
        try:
            indicator_name = condition.get('indicator')
            operator_str = condition.get('operator', '>')
            threshold = condition.get('threshold', 0)

            if indicator_name not in indicator_values:
                return False

            indicator_value = indicator_values[indicator_name]

            # 获取最新值
            if isinstance(indicator_value, pd.Series):
                latest_value = indicator_value.iloc[-1]
            elif isinstance(indicator_value, (list, np.ndarray)):
                latest_value = indicator_value[-1]
            else:
                latest_value = indicator_value

            # 应用操作符
            operator_func = self.operators.get(operator_str)
            if not operator_func:
                logger.warning(f"不支持的操作符: {operator_str}")
                return False

            return operator_func(latest_value, threshold)

        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return False

    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=True)
    def evaluate_condition(self, condition: Dict, data: pd.DataFrame, end_date: str = None) -> bool:
        """
        评估单个条件（兼容策略执行器调用）

        Args:
            condition: 条件配置
            data: 股票数据
            end_date: 结束日期（可选）

        Returns:
            bool: 条件是否满足
        """
        try:
            # 兼容不同的条件格式
            condition_type = condition.get('type', 'indicator')

            if condition_type == 'indicator':
                return self._evaluate_indicator_condition(condition, data)

            elif condition_type == 'price':
                return self._evaluate_price_condition(condition, data)

            elif condition_type == 'volume':
                return self._evaluate_volume_condition(condition, data)

            elif condition_type == 'technical':
                return self._evaluate_technical_condition(condition, data)

            elif condition_type == 'logic':
                # 逻辑操作符，不参与评估
                return True

            else:
                logger.warning(f"不支持的条件类型: {condition_type}")
                return False

        except Exception as e:
            logger.error(f"评估条件失败: {e}")
            return False

    def _evaluate_indicator_condition(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估指标条件"""
        try:
            indicator_id = condition.get('indicator_id') or condition.get('indicator')
            if not indicator_id:
                logger.warning("条件中缺少indicator_id或indicator字段")
                return False

            # 获取指标计算器
            indicator_calculator = self.indicator_registry.get_indicator(indicator_id)
            if not indicator_calculator:
                logger.warning(f"未找到指标: {indicator_id}")
                return False

            # 计算指标值
            params = condition.get('parameters', {})
            try:
                indicator_result = indicator_calculator.calculate(data, **params)
                if indicator_result is None or (hasattr(indicator_result, 'empty') and indicator_result.empty):
                    return False
            except Exception as e:
                logger.error(f"计算指标 {indicator_id} 失败: {e}")
                return False

            # 评估条件
            operator_str = condition.get('operator', '>')
            threshold = condition.get('threshold', condition.get('value', 0))
            parameter = condition.get('parameter', 'value')

            # 获取指标值
            if isinstance(indicator_result, pd.DataFrame):
                if parameter in indicator_result.columns:
                    latest_value = indicator_result[parameter].iloc[-1]
                else:
                    # 如果参数不存在，尝试使用最后一列
                    latest_value = indicator_result.iloc[-1, -1]
            elif isinstance(indicator_result, pd.Series):
                latest_value = indicator_result.iloc[-1]
            elif isinstance(indicator_result, (list, np.ndarray)):
                latest_value = indicator_result[-1]
            else:
                latest_value = indicator_result

            # 应用操作符
            operator_func = self.operators.get(operator_str)
            if not operator_func:
                logger.warning(f"不支持的操作符: {operator_str}")
                return False

            return operator_func(latest_value, threshold)

        except Exception as e:
            logger.error(f"评估指标条件失败: {e}")
            return False

    def _evaluate_price_condition(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估价格条件"""
        try:
            field = condition.get('field', 'close')
            operator_str = condition.get('operator', '>')
            threshold = condition.get('value', condition.get('threshold', 0))

            # 验证字段是否存在
            if field not in data.columns:
                logger.warning(f"数据中不存在字段: {field}")
                return False

            # 获取最新价格
            latest_value = data[field].iloc[-1]

            # 应用操作符
            operator_func = self.operators.get(operator_str)
            if not operator_func:
                logger.warning(f"不支持的操作符: {operator_str}")
                return False

            result = operator_func(latest_value, threshold)
            logger.debug(f"价格条件评估: {field}={latest_value} {operator_str} {threshold} = {result}")
            return result

        except Exception as e:
            logger.error(f"评估价格条件失败: {e}")
            return False

    def _evaluate_volume_condition(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估成交量条件"""
        try:
            operator_str = condition.get('operator', '>')
            threshold = condition.get('value', condition.get('threshold', 0))

            # 验证成交量字段是否存在
            volume_field = 'volume'
            if volume_field not in data.columns:
                logger.warning(f"数据中不存在成交量字段: {volume_field}")
                return False

            # 获取最新成交量
            latest_volume = data[volume_field].iloc[-1]

            # 应用操作符
            operator_func = self.operators.get(operator_str)
            if not operator_func:
                logger.warning(f"不支持的操作符: {operator_str}")
                return False

            result = operator_func(latest_volume, threshold)
            logger.debug(f"成交量条件评估: volume={latest_volume} {operator_str} {threshold} = {result}")
            return result

        except Exception as e:
            logger.error(f"评估成交量条件失败: {e}")
            return False

    def _evaluate_technical_condition(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估技术分析条件（如均线、突破等）"""
        try:
            condition_name = condition.get('name', condition.get('condition'))
            if not condition_name:
                logger.warning("技术条件中缺少条件名称")
                return False

            # 根据条件名称执行相应的技术分析
            if condition_name == 'ma_cross_up':
                return self._evaluate_ma_cross_up(condition, data)
            elif condition_name == 'price_breakout':
                return self._evaluate_price_breakout(condition, data)
            elif condition_name == 'volume_surge':
                return self._evaluate_volume_surge(condition, data)
            elif condition_name == 'amplitude_condition':
                return self._evaluate_amplitude_condition(condition, data)
            elif condition_name == 'gain_condition':
                return self._evaluate_gain_condition(condition, data)
            elif condition_name == 'ma_retracement':
                return self._evaluate_ma_retracement(condition, data)
            elif condition_name == 'ma_rising':
                return self._evaluate_ma_rising(condition, data)
            elif condition_name == 'no_volume_decline':
                return self._evaluate_no_volume_decline(condition, data)
            elif condition_name == 'no_volume_bearish':
                return self._evaluate_no_volume_bearish(condition, data)
            else:
                logger.warning(f"不支持的技术条件: {condition_name}")
                return False

        except Exception as e:
            logger.error(f"评估技术条件失败: {e}")
            return False

    def _evaluate_ma_cross_up(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估均线上穿条件"""
        try:
            short_period = condition.get('short_period', 5)
            long_period = condition.get('long_period', 20)
            field = condition.get('field', 'close')

            if field not in data.columns:
                return False

            # 计算移动平均线
            short_ma = data[field].rolling(window=short_period).mean()
            long_ma = data[field].rolling(window=long_period).mean()

            # 检查是否有足够的数据
            if len(short_ma) < 2 or len(long_ma) < 2:
                return False

            # 检查上穿：当前短期均线 > 长期均线，且前一日短期均线 <= 长期均线
            current_cross = short_ma.iloc[-1] > long_ma.iloc[-1]
            previous_cross = short_ma.iloc[-2] <= long_ma.iloc[-2]

            return current_cross and previous_cross

        except Exception as e:
            logger.error(f"评估均线上穿条件失败: {e}")
            return False

    def _evaluate_price_breakout(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估价格突破条件"""
        try:
            period = condition.get('period', 20)
            field = condition.get('field', 'high')

            if field not in data.columns:
                return False

            # 计算历史最高价
            historical_high = data[field].rolling(window=period).max()

            # 检查是否有足够的数据
            if len(historical_high) < 2:
                return False

            # 检查突破：当前价格 > 历史最高价（排除当前价格）
            current_price = data['close'].iloc[-1]
            previous_high = historical_high.iloc[-2]  # 排除当前价格的历史最高价

            return current_price > previous_high

        except Exception as e:
            logger.error(f"评估价格突破条件失败: {e}")
            return False

    def _evaluate_volume_surge(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估成交量放量条件"""
        try:
            period = condition.get('period', 5)
            multiplier = condition.get('multiplier', 2.0)

            if 'volume' not in data.columns:
                return False

            # 计算平均成交量
            avg_volume = data['volume'].rolling(window=period).mean()

            # 检查是否有足够的数据
            if len(avg_volume) < 2:
                return False

            # 检查放量：当前成交量 > 平均成交量 * 倍数
            current_volume = data['volume'].iloc[-1]
            average_volume = avg_volume.iloc[-2]  # 使用前期平均成交量

            return current_volume > average_volume * multiplier

        except Exception as e:
            logger.error(f"评估成交量放量条件失败: {e}")
            return False

    def evaluate_complex_conditions(self, conditions: List[Dict], data: pd.DataFrame, end_date: str = None) -> bool:
        """
        评估复杂条件组合（支持AND/OR逻辑）

        Args:
            conditions: 条件列表
            data: 股票数据
            end_date: 结束日期

        Returns:
            bool: 条件组合是否满足
        """
        try:
            if not conditions:
                return True

            results = []
            current_logic = 'AND'  # 默认逻辑

            for condition in conditions:
                if 'logic' in condition:
                    # 逻辑操作符
                    current_logic = condition['logic'].upper()
                    continue

                # 评估单个条件
                result = self.evaluate_condition(condition, data, end_date)
                results.append(result)

                # 如果是AND逻辑且当前条件为False，可以提前返回
                if current_logic == 'AND' and not result:
                    return False
                # 如果是OR逻辑且当前条件为True，可以提前返回
                elif current_logic == 'OR' and result:
                    return True

            # 根据最后的逻辑操作符决定最终结果
            if current_logic == 'AND':
                return all(results)
            elif current_logic == 'OR':
                return any(results)
            else:
                return all(results)  # 默认AND逻辑

        except Exception as e:
            logger.error(f"评估复杂条件组合失败: {e}")
            return False

    def _evaluate_amplitude_condition(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估振幅条件：110日振幅>8.1至少两次"""
        try:
            params = condition.get('parameters', {})
            period = params.get('period', 110)
            amplitude_threshold = params.get('amplitude_threshold', 8.1)
            min_occurrences = params.get('min_occurrences', 2)

            if len(data) < period:
                return False

            # 计算每日振幅：(最高价-最低价)/最低价 * 100
            amplitude = ((data['high'] - data['low']) / data['low'] * 100).fillna(0)

            # 统计指定期间内振幅超过阈值的次数
            recent_data = amplitude.tail(period)
            count = (recent_data > amplitude_threshold).sum()

            result = count >= min_occurrences
            logger.debug(f"振幅条件: {period}日内振幅>{amplitude_threshold}的次数={count}, 要求>={min_occurrences}, 结果={result}")
            return result

        except Exception as e:
            logger.error(f"评估振幅条件失败: {e}")
            return False

    def _evaluate_gain_condition(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估涨幅条件：近60日至少一次涨幅大于7%"""
        try:
            params = condition.get('parameters', {})
            period = params.get('period', 60)
            gain_threshold = params.get('gain_threshold', 7.0)
            min_occurrences = params.get('min_occurrences', 1)

            if len(data) < period + 1:
                return False

            # 计算每日涨幅：(今日收盘价/昨日收盘价 - 1) * 100
            daily_return = (data['close'].pct_change() * 100).fillna(0)

            # 统计指定期间内涨幅超过阈值的次数
            recent_data = daily_return.tail(period)
            count = (recent_data > gain_threshold).sum()

            result = count >= min_occurrences
            logger.debug(f"涨幅条件: {period}日内涨幅>{gain_threshold}%的次数={count}, 要求>={min_occurrences}, 结果={result}")
            return result

        except Exception as e:
            logger.error(f"评估涨幅条件失败: {e}")
            return False

    def _evaluate_ma_retracement(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估均线回踩条件：回踩10/20/30均线（误差5%以内）"""
        try:
            params = condition.get('parameters', {})
            ma_periods = params.get('ma_periods', [10, 20, 30])
            tolerance = params.get('tolerance', 5.0)  # 5%误差

            current_price = data['close'].iloc[-1]

            # 检查是否回踩任一均线
            for period in ma_periods:
                if len(data) < period:
                    continue

                ma_value = data['close'].rolling(window=period).mean().iloc[-1]
                deviation = abs((current_price / ma_value - 1) * 100)

                if deviation <= tolerance:
                    logger.debug(f"均线回踩: 价格{current_price}接近{period}日均线{ma_value:.2f}, 偏差{deviation:.2f}%")
                    return True

            return False

        except Exception as e:
            logger.error(f"评估均线回踩条件失败: {e}")
            return False

    def _evaluate_ma_rising(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估均线上移条件：至少3条均线上移"""
        try:
            params = condition.get('parameters', {})
            ma_periods = params.get('ma_periods', [5, 10, 20, 30])
            min_rising_count = params.get('min_rising_count', 3)

            rising_count = 0

            for period in ma_periods:
                if len(data) < period + 1:
                    continue

                ma_series = data['close'].rolling(window=period).mean()
                if len(ma_series) < 2:
                    continue

                # 检查均线是否上移
                if ma_series.iloc[-1] > ma_series.iloc[-2]:
                    rising_count += 1

            result = rising_count >= min_rising_count
            logger.debug(f"均线上移: {rising_count}条均线上移, 要求>={min_rising_count}, 结果={result}")
            return result

        except Exception as e:
            logger.error(f"评估均线上移条件失败: {e}")
            return False

    def _evaluate_no_volume_decline(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估无放量大跌条件：排除过去N日内出现过放量下跌的股票"""
        try:
            params = condition.get('parameters', {})
            check_period = params.get('check_period', 5)
            decline_threshold = params.get('decline_threshold', -3.0)  # 跌幅超过3%
            volume_multiplier = params.get('volume_multiplier', 1.5)   # 成交量放大1.5倍

            if len(data) < check_period + 1:
                return True  # 数据不足，默认通过

            # 检查最近N日是否有放量大跌
            recent_data = data.tail(check_period + 1)

            for i in range(1, len(recent_data)):
                current = recent_data.iloc[i]
                previous = recent_data.iloc[i-1]

                # 计算跌幅
                price_change = (current['close'] / previous['close'] - 1) * 100

                # 检查是否放量下跌
                volume_surge = current['volume'] > previous['volume'] * volume_multiplier

                if price_change <= decline_threshold and volume_surge:
                    logger.debug(f"发现放量大跌: 跌幅{price_change:.2f}%, 成交量放大{current['volume']/previous['volume']:.2f}倍")
                    return False

            return True

        except Exception as e:
            logger.error(f"评估无放量大跌条件失败: {e}")
            return True  # 出错时默认通过

    def _evaluate_no_volume_bearish(self, condition: Dict, data: pd.DataFrame) -> bool:
        """评估无放量大阴线条件：排除N日内存在放量大阴线的股票"""
        try:
            params = condition.get('parameters', {})
            check_period = params.get('check_period', 5)
            bearish_threshold = params.get('bearish_threshold', -4.0)  # 阴线实体跌幅>4%
            volume_multiplier = params.get('volume_multiplier', 1.8)   # 成交量超过5日均量1.8倍

            if len(data) < check_period + 5:  # 需要额外5日计算均量
                return True  # 数据不足，默认通过

            # 计算5日平均成交量
            volume_ma5 = data['volume'].rolling(window=5).mean()

            # 检查最近N日是否有放量大阴线
            recent_data = data.tail(check_period)
            recent_volume_ma5 = volume_ma5.tail(check_period)

            for i in range(len(recent_data)):
                current = recent_data.iloc[i]
                current_volume_ma5 = recent_volume_ma5.iloc[i]

                # 计算阴线实体跌幅
                body_change = (current['close'] / current['open'] - 1) * 100

                # 检查是否放量大阴线
                volume_surge = current['volume'] > current_volume_ma5 * volume_multiplier

                if body_change <= bearish_threshold and volume_surge:
                    logger.debug(f"发现放量大阴线: 实体跌幅{body_change:.2f}%, 成交量超过5日均量{current['volume']/current_volume_ma5:.2f}倍")
                    return False

            return True

        except Exception as e:
            logger.error(f"评估无放量大阴线条件失败: {e}")
            return True  # 出错时默认通过