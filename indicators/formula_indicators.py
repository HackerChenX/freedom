"""
通达信公式专用指标模块

提供通达信公式转换为选股策略时使用的特定指标实现
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import re

from indicators.base_indicator import BaseIndicator
from indicators.technical_indicators import KDJ, MACD, MA
from utils.logger import get_logger
from indicators.factory import IndicatorFactory

logger = get_logger(__name__)


class CrossOver(BaseIndicator):
    """
    交叉指标，检测快线是否上穿慢线
    """
    
    def __init__(self, fast_line: str = "", slow_line: str = "", **kwargs):
        """
        初始化交叉指标
        
        Args:
            fast_line: 快线表达式
            slow_line: 慢线表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.fast_line = fast_line
        self.slow_line = slow_line
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 解析快线和慢线
        fast_values = self._parse_line_expression(self.fast_line, data)
        slow_values = self._parse_line_expression(self.slow_line, data)
        
        if fast_values is not None and slow_values is not None:
            # 计算交叉信号
            # 交叉条件: 前一个时刻快线低于慢线，当前时刻快线高于慢线
            cross_signal = np.zeros(len(data))
            
            for i in range(1, len(data)):
                if fast_values[i-1] < slow_values[i-1] and fast_values[i] >= slow_values[i]:
                    cross_signal[i] = 1
            
            result['cross_signal'] = cross_signal
            
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算交叉指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
            self._result = result
        else:
            result = self._result
            
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 检查结果是否有效
        if result.empty or 'cross_signal' not in result.columns:
            return score
            
        # 1. 基于交叉信号的评分
        # 发生交叉时给予高分，表示买入信号
        score[result['cross_signal'] > 0] = 90
        
        # 2. 基于快慢线差值的评分
        try:
            fast_values = self._parse_line_expression(self.fast_line, data)
            slow_values = self._parse_line_expression(self.slow_line, data)
            
            if fast_values is not None and slow_values is not None:
                # 计算快慢线的差值
                diff = fast_values - slow_values
                
                # 差值为正且较大时加分，为负且绝对值较大时减分
                # 将差值映射到-25到+25的范围内
                max_diff = np.percentile(np.abs(diff), 95)  # 使用95百分位数作为标准化参考
                if max_diff > 0:
                    normalized_diff = diff / max_diff * 25
                    normalized_diff = np.clip(normalized_diff, -25, 25)
                    
                    # 将归一化差值加到基础分上
                    for i in range(len(score)):
                        score.iloc[i] += normalized_diff[i]
        except Exception as e:
            logger.warning(f"计算快慢线差值评分时出错: {e}")
        
        # 3. 平滑处理：考虑最近几个周期的交叉情况
        window_size = min(5, len(score))
        if window_size > 0:
            for i in range(window_size, len(score)):
                # 交叉信号影响会随时间衰减
                if result['cross_signal'].iloc[i] <= 0:  # 当前周期没有交叉
                    # 检查过去几个周期是否有交叉，有的话会给予一定的分数
                    for j in range(1, window_size + 1):
                        if i - j >= 0 and result['cross_signal'].iloc[i - j] > 0:
                            # 分数随时间衰减
                            decay_factor = (window_size - j + 1) / window_size
                            score.iloc[i] = max(score.iloc[i], 50 + 40 * decay_factor)
                            break
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'cross_signal' in result.columns:
            result['signal'] = 0
            result.loc[result['cross_signal'] > 0, 'signal'] = 1
            
        return result
    
    def _parse_line_expression(self, expression: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        解析线表达式
        
        Args:
            expression: 线表达式
            data: 输入数据
            
        Returns:
            解析后的数值数组
        """
        try:
            # 简单的价格引用替换
            expression = expression.replace('C', 'close').replace('O', 'open') \
                                   .replace('H', 'high').replace('L', 'low') \
                                   .replace('V', 'volume')
            
            # 检查表达式中的函数调用
            ma_match = re.search(r'MA\s*\(\s*(\w+)\s*,\s*(\d+)\s*\)', expression)
            if ma_match:
                series_name = ma_match.group(1).lower()
                period = int(ma_match.group(2))
                
                if series_name in data.columns:
                    ma_indicator = MA(periods=[period])
                    result = ma_indicator.calculate(data)
                    return result[f'MA{period}'].values
            
            # 简单的列引用
            if expression.lower() in data.columns:
                return data[expression.lower()].values
                
            # 其他复杂表达式解析
            # 这里可以扩展更多的解析逻辑
            
            logger.warning(f"无法解析表达式: {expression}")
            return None
        except Exception as e:
            logger.error(f"解析表达式 {expression} 出错: {e}")
            return None


class KDJCondition(BaseIndicator):
    """
    KDJ条件指标，检测KDJ指标线的条件
    """
    
    def __init__(self, line: str = "K", operator: str = ">", value: float = 0, **kwargs):
        """
        初始化KDJ条件指标
        
        Args:
            line: KDJ指标线，可选K/D/J
            operator: 比较操作符，如> < >= <= =
            value: 比较值
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.line = line.upper()
        self.operator = operator
        self.value = value
        
        # KDJ默认参数
        self.n = kwargs.get('n', 9)
        self.m1 = kwargs.get('m1', 3)
        self.m2 = kwargs.get('m2', 3)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 计算KDJ指标
        kdj = KDJ(n=self.n, m1=self.m1, m2=self.m2)
        kdj_result = kdj.calculate(data)
        
        # 合并结果
        result = pd.concat([result, kdj_result], axis=1)
        
        # 检查条件
        if self.line in ['K', 'D', 'J']:
            line_value = result[self.line.lower()].values
            condition = self._evaluate_condition(line_value, self.operator, self.value)
            result['condition_met'] = condition
            
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算KDJ条件指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
            self._result = result
        else:
            result = self._result
            
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 检查结果是否有效
        if result.empty or 'condition_met' not in result.columns or self.line.lower() not in result.columns:
            return score
            
        # 1. 基于条件满足的评分
        # 条件满足时给予高分或低分，取决于条件性质
        condition_score = 90 if self._is_bullish_condition() else 10
        score[result['condition_met'] > 0] = condition_score
        
        # 2. 基于指标线值与阈值的差距评分
        try:
            line_values = result[self.line.lower()].values
            
            # 计算与阈值的差距
            diff = line_values - self.value
            
            # 差距评分，根据条件类型不同而不同
            if self._is_bullish_condition():
                # 对于看涨条件，差距越大越好
                max_diff = 20  # 假设最大有效差距为20
                normalized_diff = np.clip(diff / max_diff * 25, -25, 25)
            else:
                # 对于看跌条件，差距越大（负值）越好
                max_diff = 20  # 假设最大有效差距为20
                normalized_diff = np.clip(-diff / max_diff * 25, -25, 25)
            
            # 将归一化差距加到基础分上
            for i in range(len(score)):
                # 只有在没有满足条件的情况下才调整基础分
                if result['condition_met'].iloc[i] <= 0:
                    score.iloc[i] = 50 + normalized_diff[i]
        except Exception as e:
            logger.warning(f"计算KDJ条件差距评分时出错: {e}")
        
        # 3. 考虑KDJ指标自身的特性
        # KDJ典型的超买超卖区间在20-80之间
        try:
            k_values = result['k'].values if 'k' in result.columns else None
            d_values = result['d'].values if 'd' in result.columns else None
            j_values = result['j'].values if 'j' in result.columns else None
            
            for i in range(len(score)):
                # KDJ超买超卖调整
                if k_values is not None and k_values[i] > 80:  # 超买
                    score.iloc[i] = min(score.iloc[i], 80)  # 限制最高分
                elif k_values is not None and k_values[i] < 20:  # 超卖
                    score.iloc[i] = max(score.iloc[i], 20)  # 限制最低分
        except Exception as e:
            logger.warning(f"应用KDJ超买超卖调整时出错: {e}")
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def _is_bullish_condition(self) -> bool:
        """
        判断条件是否是看涨信号
        
        Returns:
            bool: 如果是看涨信号返回True，否则返回False
        """
        # 看涨条件类型，这些条件满足时通常是买入信号
        bullish_conditions = [
            (self.operator == ">" and self.line in ["K", "D", "J"]),  # K>x, D>x, J>x
            (self.operator == ">=" and self.line in ["K", "D", "J"]),  # K>=x, D>=x, J>=x
            (self.operator == "<" and self.value <= 20),  # K<20, D<20, J<20 (超卖反弹)
            (self.operator == "<=" and self.value <= 20)   # K<=20, D<=20, J<=20 (超卖反弹)
        ]
        
        return any(bullish_conditions)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result
    
    def _evaluate_condition(self, values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
        """
        评估条件
        
        Args:
            values: 值数组
            operator: 操作符
            threshold: 阈值
            
        Returns:
            条件评估结果数组
        """
        condition = np.zeros(len(values))
        
        if operator == '>':
            condition = (values > threshold).astype(int)
        elif operator == '>=':
            condition = (values >= threshold).astype(int)
        elif operator == '<':
            condition = (values < threshold).astype(int)
        elif operator == '<=':
            condition = (values <= threshold).astype(int)
        elif operator == '=' or operator == '==':
            condition = (values == threshold).astype(int)
        else:
            logger.warning(f"不支持的操作符: {operator}")
        
        return condition


class MACDCondition(BaseIndicator):
    """
    MACD条件指标，检测MACD指标线的条件
    """
    
    def __init__(self, line: str = "MACD", operator: str = ">", value: float = 0, **kwargs):
        """
        初始化MACD条件指标
        
        Args:
            line: MACD指标线，可选MACD/DIF/DEA
            operator: 比较操作符，如> < >= <= =
            value: 比较值
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.line = line.upper()
        self.operator = operator
        self.value = value
        
        # MACD默认参数
        self.fast_period = kwargs.get('fast_period', 12)
        self.slow_period = kwargs.get('slow_period', 26)
        self.signal_period = kwargs.get('signal_period', 9)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 计算MACD指标
        macd = MACD(fast_period=self.fast_period, 
                   slow_period=self.slow_period, 
                   signal_period=self.signal_period)
        macd_result = macd.calculate(data)
        
        # 合并结果
        result = pd.concat([result, macd_result], axis=1)
        
        # 标准化MACD线名称
        line_map = {
            'MACD': 'macd',
            'DIF': 'dif',
            'DEA': 'dea',
            'DIFF': 'dif',
            'SIGNAL': 'dea'
        }
        
        # 检查条件
        line_name = line_map.get(self.line, self.line.lower())
        if line_name in result.columns:
            line_value = result[line_name].values
            condition = self._evaluate_condition(line_value, self.operator, self.value)
            result['condition_met'] = condition
            
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD条件指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
            self._result = result
        else:
            result = self._result
            
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 标准化MACD线名称
        line_map = {
            'MACD': 'macd',
            'DIF': 'dif',
            'DEA': 'dea',
            'DIFF': 'dif',
            'SIGNAL': 'dea'
        }
        line_name = line_map.get(self.line, self.line.lower())
        
        # 检查结果是否有效
        if result.empty or 'condition_met' not in result.columns or line_name not in result.columns:
            return score
            
        # 1. 基于条件满足的评分
        # 条件满足时给予高分或低分，取决于条件性质
        condition_score = 90 if self._is_bullish_condition() else 10
        score[result['condition_met'] > 0] = condition_score
        
        # 2. 基于指标线值与阈值的差距评分
        try:
            line_values = result[line_name].values
            
            # 计算与阈值的差距
            diff = line_values - self.value
            
            # 差距评分，根据条件类型不同而不同
            if self._is_bullish_condition():
                # 对于看涨条件，差距越大越好
                max_diff = abs(self.value) * 0.5 if self.value != 0 else 0.5  # 动态设置最大有效差距
                max_diff = max(0.2, max_diff)  # 确保至少有一个最小值
                normalized_diff = np.clip(diff / max_diff * 25, -25, 25)
            else:
                # 对于看跌条件，差距越大（负值）越好
                max_diff = abs(self.value) * 0.5 if self.value != 0 else 0.5
                max_diff = max(0.2, max_diff)
                normalized_diff = np.clip(-diff / max_diff * 25, -25, 25)
            
            # 将归一化差距加到基础分上
            for i in range(len(score)):
                # 只有在没有满足条件的情况下才调整基础分
                if result['condition_met'].iloc[i] <= 0:
                    score.iloc[i] = 50 + normalized_diff[i]
        except Exception as e:
            logger.warning(f"计算MACD条件差距评分时出错: {e}")
        
        # 3. 考虑MACD指标自身的特性
        try:
            if 'dif' in result.columns and 'dea' in result.columns:
                dif_values = result['dif'].values
                dea_values = result['dea'].values
                
                for i in range(len(score)):
                    # MACD金叉/死叉调整
                    if i > 0:
                        # 金叉形成(DIF上穿DEA)
                        if dif_values[i-1] < dea_values[i-1] and dif_values[i] >= dea_values[i]:
                            score.iloc[i] = max(score.iloc[i], 75)  # 加分
                        # 死叉形成(DIF下穿DEA)
                        elif dif_values[i-1] > dea_values[i-1] and dif_values[i] <= dea_values[i]:
                            score.iloc[i] = min(score.iloc[i], 25)  # 减分
        except Exception as e:
            logger.warning(f"应用MACD金叉/死叉调整时出错: {e}")
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def _is_bullish_condition(self) -> bool:
        """
        判断条件是否是看涨信号
        
        Returns:
            bool: 如果是看涨信号返回True，否则返回False
        """
        # 看涨条件类型，这些条件满足时通常是买入信号
        bullish_conditions = [
            (self.operator == ">" and self.line in ["DIF", "DIFF"]),  # DIF>0 或 DIF>DEA
            (self.operator == ">" and self.line == "MACD" and self.value >= 0),  # MACD>0
            (self.operator == ">=" and self.line in ["DIF", "DIFF"]),  # DIF>=0 或 DIF>=DEA
            (self.operator == ">=" and self.line == "MACD" and self.value >= 0),  # MACD>=0
            (self.operator == "<" and self.line == "DEA" and self.value <= 0),  # DEA<0 (低位死叉反转)
            (self.operator == "<=" and self.line == "DEA" and self.value <= 0)  # DEA<=0 (低位死叉反转)
        ]
        
        return any(bullish_conditions)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result
    
    def _evaluate_condition(self, values: np.ndarray, operator: str, threshold: float) -> np.ndarray:
        """
        评估条件
        
        Args:
            values: 值数组
            operator: 操作符
            threshold: 阈值
            
        Returns:
            条件评估结果数组
        """
        condition = np.zeros(len(values))
        
        if operator == '>':
            condition = (values > threshold).astype(int)
        elif operator == '>=':
            condition = (values >= threshold).astype(int)
        elif operator == '<':
            condition = (values < threshold).astype(int)
        elif operator == '<=':
            condition = (values <= threshold).astype(int)
        elif operator == '=' or operator == '==':
            condition = (values == threshold).astype(int)
        else:
            logger.warning(f"不支持的操作符: {operator}")
        
        return condition


class MACondition(BaseIndicator):
    """
    均线条件指标，检测均线与价格或其他均线的关系
    """
    
    def __init__(self, ma_type: str = "MA", ma_period: int = 5, 
               operator: str = ">", compare_value: str = "CLOSE", **kwargs):
        """
        初始化均线条件指标
        
        Args:
            ma_type: 均线类型，MA/EMA/WMA
            ma_period: 均线周期
            operator: 比较操作符，如> < >= <= =
            compare_value: 比较值，可以是CLOSE/OPEN/HIGH/LOW或另一个均线的表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.ma_type = ma_type.upper()
        self.ma_period = ma_period
        self.operator = operator
        self.compare_value = compare_value
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 计算均线
        ma = MA(periods=[self.ma_period])
        ma_result = ma.calculate(data)
        
        # 合并结果
        for col in ma_result.columns:
            result[col] = ma_result[col]
        
        # 解析比较值
        compare_values = self._parse_compare_value(self.compare_value, data)
        
        # 检查条件
        if compare_values is not None:
            ma_values = result[f'MA{self.ma_period}'].values
            condition = self._evaluate_condition(ma_values, self.operator, compare_values)
            result['condition_met'] = condition
            
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算均线条件指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
            self._result = result
        else:
            result = self._result
            
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 获取均线名称
        ma_name = f"{self.ma_type}{self.ma_period}"
        
        # 检查结果是否有效
        if result.empty or 'condition_met' not in result.columns or ma_name not in result.columns:
            return score
            
        # 1. 基于条件满足的评分
        # 条件满足时给予高分或低分，取决于条件性质
        condition_score = 90 if self._is_bullish_condition() else 10
        score[result['condition_met'] > 0] = condition_score
        
        # 2. 基于均线与比较值的差距评分
        try:
            ma_values = result[ma_name].values
            compare_values = self._parse_compare_value(self.compare_value, data)
            
            if compare_values is not None:
                # 计算与比较值的差距的百分比
                price_avg = np.nanmean(data['close']) if 'close' in data.columns else 1  # 用于标准化
                diff_percent = (ma_values - compare_values) / price_avg * 100  # 转换为百分比
                
                # 差距评分，根据条件类型不同而不同
                if self._is_bullish_condition():
                    # 对于看涨条件，差距越大越好（例如：MA5 > MA10）
                    max_diff_percent = 5  # 假设最大差距为5%
                    normalized_diff = np.clip(diff_percent / max_diff_percent * 25, -25, 25)
                else:
                    # 对于看跌条件，差距越大（负值）越好
                    max_diff_percent = 5
                    normalized_diff = np.clip(-diff_percent / max_diff_percent * 25, -25, 25)
                
                # 将归一化差距加到基础分上
                for i in range(len(score)):
                    # 只有在没有满足条件的情况下才调整基础分
                    if result['condition_met'].iloc[i] <= 0:
                        score.iloc[i] = 50 + normalized_diff[i]
        except Exception as e:
            logger.warning(f"计算均线条件差距评分时出错: {e}")
        
        # 3. 考虑均线系统的整体状态
        try:
            # 查找其他周期的均线
            ma_columns = [col for col in result.columns if col.startswith('MA') and col != ma_name]
            
            if ma_columns and len(ma_columns) >= 2:
                # 检查多均线系统的排列顺序
                for i in range(len(score)):
                    ma_values_at_i = {}
                    
                    # 收集当前时间点的所有均线值
                    for ma_col in ma_columns:
                        try:
                            period = int(ma_col[2:])  # 提取周期数值
                            ma_values_at_i[period] = result[ma_col].iloc[i]
                        except (ValueError, IndexError):
                            continue
                    
                    if len(ma_values_at_i) >= 2:
                        # 检查均线多头排列（短期均线在上，长期均线在下）
                        periods = sorted(ma_values_at_i.keys())
                        is_bullish_alignment = True
                        
                        for j in range(1, len(periods)):
                            if ma_values_at_i[periods[j-1]] < ma_values_at_i[periods[j]]:
                                is_bullish_alignment = False
                                break
                        
                        # 根据均线排列加减分
                        if is_bullish_alignment:
                            score.iloc[i] += 10  # 多头排列加分
                        else:
                            score.iloc[i] -= 10  # 空头排列减分
        except Exception as e:
            logger.warning(f"评估均线系统排列状态时出错: {e}")
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def _is_bullish_condition(self) -> bool:
        """
        判断条件是否是看涨信号
        
        Returns:
            bool: 如果是看涨信号返回True，否则返回False
        """
        # 看涨条件类型，这些条件满足时通常是买入信号
        ma_name = f"{self.ma_type}{self.ma_period}"
        
        # 处理各种均线和价格的关系
        bullish_conditions = [
            # 均线上穿均线
            (self.operator == ">" and ma_name.startswith("MA") and 
             isinstance(self.compare_value, str) and self.compare_value.startswith("MA") and 
             int(ma_name[2:]) < int(self.compare_value[2:])),  # 短期均线上穿长期均线
            
            # 价格上穿均线
            (self.operator == ">" and self.compare_value.upper() == "CLOSE"),  # 收盘价上穿均线
            
            # 均线上涨
            (self.operator == ">" and isinstance(self.compare_value, (int, float)) and self.compare_value <= 0)  # 均线>0
        ]
        
        return any(bullish_conditions)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result
    
    def _parse_compare_value(self, compare_value: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        解析比较值
        
        Args:
            compare_value: 比较值表达式
            data: 输入数据
            
        Returns:
            解析后的比较值数组
        """
        try:
            # 尝试将比较值解析为数字
            try:
                value = float(compare_value)
                return np.full(len(data), value)
            except ValueError:
                pass
            
            # 如果是字符串，尝试解析引用
            if isinstance(compare_value, str):
                # 标准化为大写
                compare_value = compare_value.upper()
                
                # 处理常见的价格列引用
                if compare_value == 'CLOSE':
                    return data['close'].values
                elif compare_value == 'OPEN':
                    return data['open'].values
                elif compare_value == 'HIGH':
                    return data['high'].values
                elif compare_value == 'LOW':
                    return data['low'].values
                
                # 处理均线引用，如MA10
                ma_match = re.match(r'(MA|EMA|SMA|WMA)(\d+)', compare_value)
                if ma_match:
                    ma_type = ma_match.group(1)
                    period = int(ma_match.group(2))
                    ma_name = f"{ma_type}{period}"
                    
                    if ma_name in data.columns:
                        return data[ma_name].values
                    else:
                        # 尝试计算所需的均线
                        if ma_type == "MA":
                            ma_indicator = MA(periods=[period])
                            ma_result = ma_indicator.calculate(data)
                            # 将结果添加到输入数据中以便后续使用
                            data[ma_name] = ma_result[ma_name]
                            return data[ma_name].values
            
            logger.warning(f"无法解析比较值: {compare_value}")
            return None
        except Exception as e:
            logger.error(f"解析比较值 {compare_value} 出错: {e}")
            return None
    
    def _evaluate_condition(self, values: np.ndarray, operator: str, compare_values: np.ndarray) -> np.ndarray:
        """
        评估条件
        
        Args:
            values: 均线值数组
            operator: 操作符
            compare_values: 比较值数组
            
        Returns:
            条件评估结果数组
        """
        condition = np.zeros(len(values))
        
        if operator == '>':
            condition = (values > compare_values).astype(int)
        elif operator == '>=':
            condition = (values >= compare_values).astype(int)
        elif operator == '<':
            condition = (values < compare_values).astype(int)
        elif operator == '<=':
            condition = (values <= compare_values).astype(int)
        elif operator == '=' or operator == '==':
            condition = (values == compare_values).astype(int)
        else:
            logger.warning(f"不支持的操作符: {operator}")
        
        return condition


class GenericCondition(BaseIndicator):
    """
    通用条件指标，支持通达信公式自定义条件
    """
    
    def __init__(self, condition: str = "", **kwargs):
        """
        初始化通用条件指标
        
        Args:
            condition: 条件表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.condition = condition
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算指标值
        
        Args:
            data: 输入数据
            
        Returns:
            包含指标值的DataFrame
        """
        result = data.copy()
        
        # 解析条件表达式
        # 这里需要实现通达信条件表达式的解析逻辑
        # 为了简化，我们暂时只实现一个示例
        
        # 示例：解析 "MA5>MA10"
        if ">" in self.condition:
            parts = self.condition.split(">")
            left = parts[0].strip()
            right = parts[1].strip()
            
            # 解析左右两边的表达式
            # 这里需要完善以支持更复杂的表达式
            
            # 示例实现
            if left.startswith("MA") and right.startswith("MA"):
                try:
                    left_period = int(left[2:])
                    right_period = int(right[2:])
                    
                    ma_indicator = MA(periods=[left_period, right_period])
                    ma_result = ma_indicator.calculate(data)
                    
                    result = pd.concat([result, ma_result], axis=1)
                    result['condition_met'] = (result[f'MA{left_period}'] > result[f'MA{right_period}']).astype(int)
                except Exception as e:
                    logger.error(f"解析条件 {self.condition} 出错: {e}")
            
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算通用条件指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 确保已计算指标
        if not self.has_result():
            result = self.calculate(data)
            self._result = result
        else:
            result = self._result
            
        # 初始化评分，默认为50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 检查结果是否有效
        if result.empty or 'condition_met' not in result.columns:
            return score
            
        # 1. 基于条件满足的评分
        # 由于是通用条件，无法确定是看涨还是看跌条件
        # 所以我们根据条件的特征来推断
        is_bullish = self._infer_condition_bullishness()
        
        condition_score = 90 if is_bullish else 10
        score[result['condition_met'] > 0] = condition_score
        
        # 2. 计算差距分数
        # 由于是通用条件，具体实现取决于条件类型
        # 这里我们提供一个简单的实现，可以根据需要扩展
        try:
            # 解析条件表达式，尝试获取差距信息
            if ">" in self.condition:
                parts = self.condition.split(">")
                left = parts[0].strip()
                right = parts[1].strip()
                
                # 检查是否是均线交叉条件
                if left.startswith("MA") and right.startswith("MA"):
                    try:
                        left_period = int(left[2:])
                        right_period = int(right[2:])
                        
                        if f'MA{left_period}' in result.columns and f'MA{right_period}' in result.columns:
                            # 计算均线差距的百分比
                            ma_left = result[f'MA{left_period}']
                            ma_right = result[f'MA{right_period}']
                            price_avg = np.nanmean(data['close']) if 'close' in data.columns else 1
                            diff_percent = (ma_left - ma_right) / price_avg * 100
                            
                            # 将差距映射到评分
                            max_diff_percent = 5  # 假设最大差距为5%
                            
                            for i in range(len(score)):
                                # 只调整未满足条件的情况
                                if result['condition_met'].iloc[i] <= 0:
                                    # 如果是看涨条件，差距越接近阈值越高分
                                    if is_bullish:
                                        # 对于 MA5>MA10，diff_percent 在接近 0 时评分应该接近 50
                                        normalized_diff = np.clip(diff_percent.iloc[i] / max_diff_percent * 25, -25, 25)
                                    else:
                                        # 对于看跌条件，反之
                                        normalized_diff = np.clip(-diff_percent.iloc[i] / max_diff_percent * 25, -25, 25)
                                    
                                    score.iloc[i] = 50 + normalized_diff
                    except Exception as e:
                        logger.warning(f"计算均线差距评分时出错: {e}")
        except Exception as e:
            logger.warning(f"计算通用条件差距评分时出错: {e}")
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def _infer_condition_bullishness(self) -> bool:
        """
        推断条件是否是看涨信号
        
        Returns:
            bool: 如果推断为看涨信号返回True，否则返回False
        """
        # 根据条件表达式推断是否是看涨条件
        # 这里提供一些常见的看涨条件模式
        
        # 1. 均线金叉类
        if ">" in self.condition:
            parts = self.condition.split(">")
            left = parts[0].strip()
            right = parts[1].strip()
            
            # 短期均线上穿长期均线
            if left.startswith("MA") and right.startswith("MA"):
                try:
                    left_period = int(left[2:])
                    right_period = int(right[2:])
                    return left_period < right_period  # 短期均线在左侧，看涨
                except:
                    pass
            
            # 价格上穿均线
            if left.upper() in ["CLOSE", "C"] and right.startswith("MA"):
                return True  # 价格上穿均线，看涨
        
        # 2. MACD相关条件
        if "MACD" in self.condition:
            # MACD > 0 或 DIF > DEA 通常是看涨
            if "MACD>0" in self.condition.replace(" ", "") or "DIF>DEA" in self.condition.replace(" ", ""):
                return True
            # MACD < 0 或 DIF < DEA 通常是看跌
            if "MACD<0" in self.condition.replace(" ", "") or "DIF<DEA" in self.condition.replace(" ", ""):
                return False
        
        # 3. KDJ相关条件
        if "KDJ" in self.condition or "K" in self.condition or "D" in self.condition or "J" in self.condition:
            # K > D 通常是看涨
            if "K>D" in self.condition.replace(" ", ""):
                return True
            # K < D 通常是看跌
            if "K<D" in self.condition.replace(" ", ""):
                return False
            # K > 80 或 D > 80 通常是超买区域，看跌
            if "K>80" in self.condition.replace(" ", "") or "D>80" in self.condition.replace(" ", ""):
                return False
            # K < 20 或 D < 20 通常是超卖区域，看涨
            if "K<20" in self.condition.replace(" ", "") or "D<20" in self.condition.replace(" ", ""):
                return True
        
        # 默认情况，我们假设条件是中性的
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 包含指标值的DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        result = data.copy()
        
        if 'condition_met' in result.columns:
            result['signal'] = 0
            result.loc[result['condition_met'] > 0, 'signal'] = 1
            
        return result


# 注册指标到工厂
IndicatorFactory.register_indicator("CROSS_OVER", CrossOver)
IndicatorFactory.register_indicator("KDJ_CONDITION", KDJCondition)
IndicatorFactory.register_indicator("MACD_CONDITION", MACDCondition)
IndicatorFactory.register_indicator("MA_CONDITION", MACondition)
IndicatorFactory.register_indicator("GENERIC_CONDITION", GenericCondition) 