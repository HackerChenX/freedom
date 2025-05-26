"""
高级K线形态识别模块

实现更复杂的组合K线形态和复合形态识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from indicators.pattern.candlestick_patterns import PatternType, CandlestickPatterns
from utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedPatternType(Enum):
    """高级K线形态类型枚举"""
    # 三星形态
    THREE_WHITE_SOLDIERS = "三白兵"          # 连续三根阳线，每根都收于接近最高点
    THREE_BLACK_CROWS = "三黑鸦"             # 连续三根阴线，每根都收于接近最低点
    THREE_INSIDE_UP = "三内涨"               # 大阴线+小阳线在阴线实体内+突破阴线收盘价的阳线
    THREE_INSIDE_DOWN = "三内跌"             # 大阳线+小阴线在阳线实体内+突破阳线收盘价的阴线
    THREE_OUTSIDE_UP = "三外涨"              # 阴线+包含前一天阴线的阳线+更高收盘的阳线
    THREE_OUTSIDE_DOWN = "三外跌"            # 阳线+包含前一天阳线的阴线+更低收盘的阴线
    
    # 高级复合形态
    RISING_THREE_METHODS = "上升三法"        # 大阳线后三根小K线在大阳线范围内整理，然后一根突破的阳线
    FALLING_THREE_METHODS = "下降三法"        # 大阴线后三根小K线在大阴线范围内整理，然后一根突破的阴线
    MAT_HOLD = "铺垫形态"                    # 大阳线后2-3根小阴线在大阳线上部整理，然后一根大阳线
    STICK_SANDWICH = "棍心三明治"            # 阳线+阴线+与第一根收盘价相同的阳线
    
    # 其他复合形态
    LADDER_BOTTOM = "梯底形态"               # 连续下跌后出现的底部形态
    TOWER_TOP = "塔顶形态"                  # 连续上涨后出现的顶部形态
    BREAKAWAY = "脱离形态"                  # 五根K线组成的反转形态
    KICKING = "反冲形态"                    # 两根相反方向的光头光脚K线
    UNIQUE_THREE_RIVER = "奇特三河"          # 三根K线组成的底部反转形态


class AdvancedCandlestickPatterns(BaseIndicator):
    """
    高级K线形态识别指标
    
    实现更复杂的组合K线形态和复合形态识别功能，包括三星形态、高级复合形态和其他复合形态。
    提供信号强度评估、趋势确认和复合信号分析功能。
    """
    
    def __init__(self):
        """初始化高级K线形态识别指标"""
        super().__init__(name="AdvancedCandlestickPatterns", description="高级K线形态识别指标，识别更复杂的组合K线形态和复合形态")
        self.basic_patterns = CandlestickPatterns()
    
    def ensure_columns(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """
        确保数据包含必要的列
        
        Args:
            data: 输入数据
            required_columns: 必需的列名列表
            
        Raises:
            ValueError: 如果缺少必需的列
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别高级K线形态
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含各种高级K线形态的标记
            
        Raises:
            ValueError: 如果输入数据无效或缺少必要的列
        """
        # 验证输入数据
        if data is None or len(data) == 0:
            logger.warning("输入数据为空，无法识别K线形态")
            return pd.DataFrame(index=data.index if data is not None else [])
        
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close"])
        
        # 数据量不足以识别复杂形态时提前返回
        if len(data) < 5:
            logger.warning("数据量不足，无法识别复杂K线形态，至少需要5根K线")
            return pd.DataFrame(index=data.index)
        
        # 计算基础K线形态
        basic_patterns = self.basic_patterns.calculate(data)
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算三星形态
        result = self._calculate_three_star_patterns(data, result)
        
        # 计算高级复合形态（需要至少5根K线）
        if len(data) >= 5:
            result = self._calculate_advanced_compound_patterns(data, result)
        
        # 计算其他复合形态
        result = self._calculate_other_compound_patterns(data, result)
        
        # 合并基础形态和高级形态
        for column in basic_patterns.columns:
            result[column] = basic_patterns[column]
        
        return result
    
    def generate_signals(self, indicator_values: pd.DataFrame, **params) -> pd.DataFrame:
        """
        根据识别到的形态生成交易信号
        
        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记
            **params: 信号生成的参数
            
        Returns:
            pd.DataFrame: 信号DataFrame
            
        Raises:
            ValueError: 如果输入数据无效
        """
        # 验证输入数据
        if indicator_values is None or len(indicator_values) == 0:
            logger.warning("指标值为空，无法生成信号")
            return pd.DataFrame(index=indicator_values.index if indicator_values is not None else [])
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=indicator_values.index)
        
        # 分类定义各种形态的交易信号
        bullish_patterns = [
            AdvancedPatternType.THREE_WHITE_SOLDIERS.value,
            AdvancedPatternType.THREE_INSIDE_UP.value,
            AdvancedPatternType.THREE_OUTSIDE_UP.value,
            AdvancedPatternType.RISING_THREE_METHODS.value,
            AdvancedPatternType.MAT_HOLD.value,
            AdvancedPatternType.LADDER_BOTTOM.value,
            AdvancedPatternType.BREAKAWAY.value
        ]
        
        bearish_patterns = [
            AdvancedPatternType.THREE_BLACK_CROWS.value,
            AdvancedPatternType.THREE_INSIDE_DOWN.value,
            AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
            AdvancedPatternType.FALLING_THREE_METHODS.value,
            AdvancedPatternType.TOWER_TOP.value
        ]
        
        neutral_patterns = [
            AdvancedPatternType.STICK_SANDWICH.value,
            AdvancedPatternType.KICKING.value,
            AdvancedPatternType.UNIQUE_THREE_RIVER.value
        ]
        
        # 创建买入信号
        signals['buy_signal'] = False
        for pattern in bullish_patterns:
            if pattern in indicator_values.columns:
                signals['buy_signal'] |= indicator_values[pattern]
        
        # 创建卖出信号
        signals['sell_signal'] = False
        for pattern in bearish_patterns:
            if pattern in indicator_values.columns:
                signals['sell_signal'] |= indicator_values[pattern]
        
        # 创建观察信号
        signals['watch_signal'] = False
        for pattern in neutral_patterns:
            if pattern in indicator_values.columns:
                signals['watch_signal'] |= indicator_values[pattern]
        
        # 添加信号强度
        signals['signal_strength'] = self._calculate_signal_strength(indicator_values)
        
        # 添加趋势确认信号
        signals['trend_confirmed'] = self._calculate_trend_confirmation(indicator_values)
        
        # 添加复合信号（多种形态同时出现）
        signals['compound_signal'] = self._calculate_compound_signal(indicator_values)
        
        return signals
    
    def _calculate_three_star_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算三星形态
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 验证数据量是否足够
        if len(data) < 3:
            logger.warning("数据量不足，无法识别三星形态，至少需要3根K线")
            return result
        
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices
        
        # 计算实体大小
        body_size = np.abs(close_prices - open_prices)
        
        # 初始化结果数组
        n = len(data)
        three_white_soldiers = np.zeros(n, dtype=bool)
        three_black_crows = np.zeros(n, dtype=bool)
        three_inside_up = np.zeros(n, dtype=bool)
        three_inside_down = np.zeros(n, dtype=bool)
        three_outside_up = np.zeros(n, dtype=bool)
        three_outside_down = np.zeros(n, dtype=bool)
        
        # 计算三星形态
        for i in range(3, n):
            # 三白兵：连续三根阳线，每根都收于接近最高点，开盘价在前一根实体内
            if (bullish[i-2] and bullish[i-1] and bullish[i] and
                close_prices[i-2] > open_prices[i-2] * 1.01 and  # 第一根实体足够大
                close_prices[i-1] > open_prices[i-1] * 1.01 and  # 第二根实体足够大
                close_prices[i] > open_prices[i] * 1.01 and      # 第三根实体足够大
                open_prices[i-1] > open_prices[i-2] and        # 每根的开盘价高于前一根
                open_prices[i] > open_prices[i-1] and
                close_prices[i-1] > close_prices[i-2] and      # 每根的收盘价高于前一根
                close_prices[i] > close_prices[i-1] and
                (high_prices[i-2] - close_prices[i-2]) < body_size[i-2] * 0.3 and  # 上影线短
                (high_prices[i-1] - close_prices[i-1]) < body_size[i-1] * 0.3 and
                (high_prices[i] - close_prices[i]) < body_size[i] * 0.3):
                three_white_soldiers[i] = True
            
            # 三黑鸦：连续三根阴线，每根都收于接近最低点，开盘价在前一根实体内
            if (bearish[i-2] and bearish[i-1] and bearish[i] and
                open_prices[i-2] > close_prices[i-2] * 1.01 and  # 第一根实体足够大
                open_prices[i-1] > close_prices[i-1] * 1.01 and  # 第二根实体足够大
                open_prices[i] > close_prices[i] * 1.01 and      # 第三根实体足够大
                open_prices[i-1] < open_prices[i-2] and        # 每根的开盘价低于前一根
                open_prices[i] < open_prices[i-1] and
                close_prices[i-1] < close_prices[i-2] and      # 每根的收盘价低于前一根
                close_prices[i] < close_prices[i-1] and
                (close_prices[i-2] - low_prices[i-2]) < body_size[i-2] * 0.3 and  # 下影线短
                (close_prices[i-1] - low_prices[i-1]) < body_size[i-1] * 0.3 and
                (close_prices[i] - low_prices[i]) < body_size[i] * 0.3):
                three_black_crows[i] = True
            
            # 三内涨：大阴线+小阳线在阴线实体内+突破阴线收盘价的阳线
            if (bearish[i-2] and bullish[i-1] and bullish[i] and
                body_size[i-2] > body_size[i-1] and              # 第一根阴线实体大于第二根阳线
                open_prices[i-1] > close_prices[i-2] and       # 第二根开盘价高于第一根收盘价
                close_prices[i-1] < open_prices[i-2] and       # 第二根收盘价低于第一根开盘价
                close_prices[i] > open_prices[i-2]):           # 第三根收盘价高于第一根开盘价
                three_inside_up[i] = True
            
            # 三内跌：大阳线+小阴线在阳线实体内+突破阳线收盘价的阴线
            if (bullish[i-2] and bearish[i-1] and bearish[i] and
                body_size[i-2] > body_size[i-1] and              # 第一根阳线实体大于第二根阴线
                open_prices[i-1] < close_prices[i-2] and       # 第二根开盘价低于第一根收盘价
                close_prices[i-1] > open_prices[i-2] and       # 第二根收盘价高于第一根开盘价
                close_prices[i] < open_prices[i-2]):           # 第三根收盘价低于第一根开盘价
                three_inside_down[i] = True
            
            # 三外涨：阴线+包含前一天阴线的阳线+更高收盘的阳线
            if (bearish[i-2] and bullish[i-1] and bullish[i] and
                open_prices[i-1] <= close_prices[i-2] and      # 第二根阳线完全包含第一根阴线
                close_prices[i-1] >= open_prices[i-2] and
                close_prices[i] > close_prices[i-1]):          # 第三根收盘价高于第二根
                three_outside_up[i] = True
            
            # 三外跌：阳线+包含前一天阳线的阴线+更低收盘的阴线
            if (bullish[i-2] and bearish[i-1] and bearish[i] and
                open_prices[i-1] >= close_prices[i-2] and      # 第二根阴线完全包含第一根阳线
                close_prices[i-1] <= open_prices[i-2] and
                close_prices[i] < close_prices[i-1]):          # 第三根收盘价低于第二根
                three_outside_down[i] = True
        
        # 添加到结果
        result[AdvancedPatternType.THREE_WHITE_SOLDIERS.value] = three_white_soldiers
        result[AdvancedPatternType.THREE_BLACK_CROWS.value] = three_black_crows
        result[AdvancedPatternType.THREE_INSIDE_UP.value] = three_inside_up
        result[AdvancedPatternType.THREE_INSIDE_DOWN.value] = three_inside_down
        result[AdvancedPatternType.THREE_OUTSIDE_UP.value] = three_outside_up
        result[AdvancedPatternType.THREE_OUTSIDE_DOWN.value] = three_outside_down
        
        return result
    
    def _calculate_advanced_compound_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级复合形态
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 验证数据量是否足够
        if len(data) < 5:
            logger.warning("数据量不足，无法识别高级复合形态，至少需要5根K线")
            return result
        
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices
        
        # 计算实体大小
        body_size = np.abs(close_prices - open_prices)
        
        # 初始化结果数组
        n = len(data)
        rising_three_methods = np.zeros(n, dtype=bool)
        falling_three_methods = np.zeros(n, dtype=bool)
        mat_hold = np.zeros(n, dtype=bool)
        stick_sandwich = np.zeros(n, dtype=bool)
        
        # 计算高级复合形态
        for i in range(5, n):
            # 使用安全的数组切片，避免索引错误
            if i-4 < 0 or i >= n:
                continue
                
            # 上升三法：大阳线后三根小K线在大阳线范围内整理，然后一根突破的阳线
            try:
                if (bullish[i-4] and 
                    body_size[i-4] > np.mean(body_size[max(0, i-3):i]) and  # 第一根实体大于后续整理的平均实体
                    max(high_prices[max(0, i-3):i]) < high_prices[i-4] and  # 整理阶段的最高点低于第一根最高点
                    min(low_prices[max(0, i-3):i]) > low_prices[i-4] and    # 整理阶段的最低点高于第一根最低点
                    bullish[i] and                                 # 最后一根是阳线
                    close_prices[i] > high_prices[i-4]):           # 最后一根收盘价突破第一根最高点
                    rising_three_methods[i] = True
            except Exception as e:
                logger.debug(f"计算上升三法时出错: {e}")
            
            # 下降三法：大阴线后三根小K线在大阴线范围内整理，然后一根突破的阴线
            try:
                if (bearish[i-4] and 
                    body_size[i-4] > np.mean(body_size[max(0, i-3):i]) and  # 第一根实体大于后续整理的平均实体
                    max(high_prices[max(0, i-3):i]) < high_prices[i-4] and  # 整理阶段的最高点低于第一根最高点
                    min(low_prices[max(0, i-3):i]) > low_prices[i-4] and    # 整理阶段的最低点高于第一根最低点
                    bearish[i] and                                 # 最后一根是阴线
                    close_prices[i] < low_prices[i-4]):            # 最后一根收盘价突破第一根最低点
                    falling_three_methods[i] = True
            except Exception as e:
                logger.debug(f"计算下降三法时出错: {e}")
            
            # 铺垫形态：大阳线后2-3根小阴线在大阳线上部整理，然后一根大阳线
            try:
                if (bullish[i-4] and 
                    body_size[i-4] > np.mean(body_size[max(0, i-3):i-1]) and # 第一根实体大于中间整理的平均实体
                    all(bearish[max(0, i-3):i-1]) and                       # 中间整理是阴线
                    max(close_prices[max(0, i-3):i-1]) < close_prices[i-4] and # 整理阶段的收盘价低于第一根收盘价
                    min(open_prices[max(0, i-3):i-1]) > (open_prices[i-4] + close_prices[i-4]) / 2 and # 整理阶段的开盘价高于第一根中点
                    bullish[i] and                                  # 最后一根是阳线
                    body_size[i] > np.mean(body_size[max(0, i-3):i-1]) and  # 最后一根实体大于整理阶段的平均实体
                    close_prices[i] > close_prices[i-4]):           # 最后一根收盘价高于第一根收盘价
                    mat_hold[i] = True
            except Exception as e:
                logger.debug(f"计算铺垫形态时出错: {e}")
            
            # 棍心三明治：阳线+阴线+与第一根收盘价相同的阳线
            try:
                if (i-2 >= 0 and i < n and
                    bullish[i-2] and bearish[i-1] and bullish[i] and
                    abs(close_prices[i] - close_prices[i-2]) / close_prices[i-2] < 0.01 and  # 第三根收盘价接近第一根收盘价
                    close_prices[i-1] < open_prices[i-1] and                            # 第二根是阴线
                    close_prices[i-1] < min(open_prices[i-2], close_prices[i-2]) and    # 第二根收盘价低于第一根的最低点
                    open_prices[i] < open_prices[i-2]):                                # 第三根开盘价低于第一根开盘价
                    stick_sandwich[i] = True
            except Exception as e:
                logger.debug(f"计算棍心三明治时出错: {e}")
        
        # 添加到结果
        result[AdvancedPatternType.RISING_THREE_METHODS.value] = rising_three_methods
        result[AdvancedPatternType.FALLING_THREE_METHODS.value] = falling_three_methods
        result[AdvancedPatternType.MAT_HOLD.value] = mat_hold
        result[AdvancedPatternType.STICK_SANDWICH.value] = stick_sandwich
        
        return result
    
    def _calculate_other_compound_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算其他复合形态
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 验证数据量是否足够
        if len(data) < 5:
            logger.warning("数据量不足，无法识别其他复合形态，至少需要5根K线")
            return result
        
        # 提取数据
        open_prices = data["open"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 计算K线涨跌
        bullish = close_prices > open_prices
        bearish = close_prices < open_prices
        
        # 计算实体大小
        body_size = np.abs(close_prices - open_prices)
        
        # 初始化结果数组
        n = len(data)
        ladder_bottom = np.zeros(n, dtype=bool)
        tower_top = np.zeros(n, dtype=bool)
        breakaway = np.zeros(n, dtype=bool)
        kicking = np.zeros(n, dtype=bool)
        unique_three_river = np.zeros(n, dtype=bool)
        
        # 使用向量化操作进行预计算
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)  # 上影线
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices   # 下影线
        avg_body_size = np.mean(body_size)                                  # 平均实体大小
        
        # 计算其他复合形态
        for i in range(5, n):
            # 梯底形态：连续下跌后的底部反转形态，三根K线组成
            if i-2 >= 0:
                try:
                    if (all(bearish[max(0, i-4):i-2]) and        # 之前是连续下跌
                        low_prices[i-2] < low_prices[i-3] and    # 第一根创新低
                        bearish[i-2] and bearish[i-1] and        # 第一、二根是阴线
                        low_prices[i-1] > low_prices[i-2] and    # 第二根最低点高于第一根
                        bullish[i] and                          # 第三根是阳线
                        close_prices[i] > open_prices[i-1] and  # 第三根收盘价高于第二根开盘价
                        lower_shadow[i-2] > body_size[i-2]):     # 第一根有长下影线
                        ladder_bottom[i] = True
                except Exception as e:
                    logger.debug(f"计算梯底形态时出错: {e}")
            
            # 塔顶形态：连续上涨后的顶部反转形态，三根K线组成
            if i-2 >= 0:
                try:
                    if (all(bullish[max(0, i-4):i-2]) and        # 之前是连续上涨
                        high_prices[i-2] > high_prices[i-3] and  # 第一根创新高
                        bullish[i-2] and bearish[i-1] and        # 第一根是阳线，第二根是阴线
                        high_prices[i-1] < high_prices[i-2] and  # 第二根最高点低于第一根
                        bearish[i] and                          # 第三根是阴线
                        close_prices[i] < open_prices[i-1] and  # 第三根收盘价低于第二根开盘价
                        upper_shadow[i-2] > body_size[i-2]):     # 第一根有长上影线
                        tower_top[i] = True
                except Exception as e:
                    logger.debug(f"计算塔顶形态时出错: {e}")
            
            # 脱离形态：五根K线组成的反转形态
            if i-4 >= 0:
                # 看涨脱离形态
                try:
                    if (all(bearish[i-4:i-2]) and               # 前三根是阴线
                        open_prices[i-4] > close_prices[i-4] and # 第一根是阴线
                        open_prices[i-3] > close_prices[i-3] and # 第二根是阴线
                        open_prices[i-2] > close_prices[i-2] and # 第三根是阴线
                        low_prices[i-2] < low_prices[i-3] and    # 第三根创新低
                        bearish[i-1] and                        # 第四根是阴线
                        open_prices[i-1] < close_prices[i-2] and # 第四根跳空向下开盘
                        bullish[i] and                          # 第五根是阳线
                        close_prices[i] > open_prices[i-3]):     # 第五根收盘价高于第二根开盘价
                        breakaway[i] = True
                except Exception as e:
                    logger.debug(f"计算看涨脱离形态时出错: {e}")
                
                # 看跌脱离形态
                try:
                    if (all(bullish[i-4:i-2]) and               # 前三根是阳线
                        open_prices[i-4] < close_prices[i-4] and # 第一根是阳线
                        open_prices[i-3] < close_prices[i-3] and # 第二根是阳线
                        open_prices[i-2] < close_prices[i-2] and # 第三根是阳线
                        high_prices[i-2] > high_prices[i-3] and  # 第三根创新高
                        bullish[i-1] and                        # 第四根是阳线
                        open_prices[i-1] > close_prices[i-2] and # 第四根跳空向上开盘
                        bearish[i] and                          # 第五根是阴线
                        close_prices[i] < open_prices[i-3]):     # 第五根收盘价低于第二根开盘价
                        breakaway[i] = True
                except Exception as e:
                    logger.debug(f"计算看跌脱离形态时出错: {e}")
            
            # 反冲形态：两根相反方向的光头光脚K线
            if i-1 >= 0:
                try:
                    # 光头光脚K线：上下影线很短
                    bullish_marubozu_i = (bullish[i] and 
                                        upper_shadow[i] < body_size[i] * 0.1 and 
                                        lower_shadow[i] < body_size[i] * 0.1 and
                                        body_size[i] > avg_body_size * 1.5)
                    
                    bearish_marubozu_i = (bearish[i] and 
                                        upper_shadow[i] < body_size[i] * 0.1 and 
                                        lower_shadow[i] < body_size[i] * 0.1 and
                                        body_size[i] > avg_body_size * 1.5)
                    
                    bullish_marubozu_i_1 = (bullish[i-1] and 
                                        upper_shadow[i-1] < body_size[i-1] * 0.1 and 
                                        lower_shadow[i-1] < body_size[i-1] * 0.1 and
                                        body_size[i-1] > avg_body_size * 1.5)
                    
                    bearish_marubozu_i_1 = (bearish[i-1] and 
                                            upper_shadow[i-1] < body_size[i-1] * 0.1 and 
                                            lower_shadow[i-1] < body_size[i-1] * 0.1 and
                                            body_size[i-1] > avg_body_size * 1.5)
                    
                    # 两根反方向的光头光脚K线，中间有跳空
                    if ((bullish_marubozu_i_1 and bearish_marubozu_i and low_prices[i] > high_prices[i-1]) or
                        (bearish_marubozu_i_1 and bullish_marubozu_i and high_prices[i] < low_prices[i-1])):
                        kicking[i] = True
                except Exception as e:
                    logger.debug(f"计算反冲形态时出错: {e}")
            
            # 奇特三河：三根K线组成的底部反转形态
            if i-2 >= 0:
                try:
                    if (bearish[i-2] and                         # 第一根是阴线
                        bearish[i-1] and                         # 第二根是阴线
                        body_size[i-1] < body_size[i-2] * 0.5 and # 第二根实体小于第一根的一半
                        lower_shadow[i-1] > body_size[i-1] * 2 and # 第二根有长下影线
                        low_prices[i-1] < low_prices[i-2] and    # 第二根最低点低于第一根
                        bullish[i] and                           # 第三根是阳线
                        open_prices[i] < close_prices[i-1] and   # 第三根开盘价低于第二根收盘价
                        close_prices[i] < open_prices[i-2]):      # 第三根收盘价低于第一根开盘价
                        unique_three_river[i] = True
                except Exception as e:
                    logger.debug(f"计算奇特三河时出错: {e}")
        
        # 添加到结果
        result[AdvancedPatternType.LADDER_BOTTOM.value] = ladder_bottom
        result[AdvancedPatternType.TOWER_TOP.value] = tower_top
        result[AdvancedPatternType.BREAKAWAY.value] = breakaway
        result[AdvancedPatternType.KICKING.value] = kicking
        result[AdvancedPatternType.UNIQUE_THREE_RIVER.value] = unique_three_river
        
        return result
    
    def _calculate_signal_strength(self, indicator_values: pd.DataFrame) -> pd.Series:
        """
        计算信号强度
        
        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记
            
        Returns:
            pd.Series: 信号强度序列，取值范围[0, 100]
        """
        # 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(index=indicator_values.index if indicator_values is not None else [])
        
        # 初始化信号强度序列
        signal_strength = pd.Series(0, index=indicator_values.index)
        
        try:
            # 形态权重定义
            pattern_weights = {
                # 三星形态
                AdvancedPatternType.THREE_WHITE_SOLDIERS.value: 80,   # 三白兵
                AdvancedPatternType.THREE_BLACK_CROWS.value: 80,     # 三黑鸦
                AdvancedPatternType.THREE_INSIDE_UP.value: 70,       # 三内涨
                AdvancedPatternType.THREE_INSIDE_DOWN.value: 70,     # 三内跌
                AdvancedPatternType.THREE_OUTSIDE_UP.value: 75,      # 三外涨
                AdvancedPatternType.THREE_OUTSIDE_DOWN.value: 75,    # 三外跌
                
                # 高级复合形态
                AdvancedPatternType.RISING_THREE_METHODS.value: 85,  # 上升三法
                AdvancedPatternType.FALLING_THREE_METHODS.value: 85, # 下降三法
                AdvancedPatternType.MAT_HOLD.value: 82,             # 铺垫形态
                AdvancedPatternType.STICK_SANDWICH.value: 60,       # 棍心三明治
                
                # 其他复合形态
                AdvancedPatternType.LADDER_BOTTOM.value: 75,        # 梯底形态
                AdvancedPatternType.TOWER_TOP.value: 75,            # 塔顶形态
                AdvancedPatternType.BREAKAWAY.value: 78,            # 脱离形态
                AdvancedPatternType.KICKING.value: 83,              # 反冲形态
                AdvancedPatternType.UNIQUE_THREE_RIVER.value: 72    # 奇特三河
            }
            
            # 添加基础K线形态的权重
            from indicators.pattern.candlestick_patterns import PatternType
            basic_pattern_weights = {
                PatternType.HAMMER.value: 65,                     # 锤子
                PatternType.HANGING_MAN.value: 65,                # 上吊线
                PatternType.SHOOTING_STAR.value: 65,              # 流星
                PatternType.INVERTED_HAMMER.value: 65,            # 倒锤子
                PatternType.DOJI.value: 50,                       # 十字星
                PatternType.DRAGONFLY_DOJI.value: 60,             # 蜻蜓十字星
                PatternType.GRAVESTONE_DOJI.value: 60,            # 墓碑十字星
                PatternType.BULLISH_ENGULFING.value: 70,          # 看涨吞没
                PatternType.BEARISH_ENGULFING.value: 70,          # 看跌吞没
                PatternType.DARK_CLOUD_COVER.value: 65,           # 乌云盖顶
                PatternType.PIERCING_LINE.value: 65,              # 刺透形态
                PatternType.BULLISH_HARAMI.value: 60,             # 看涨母子线
                PatternType.BEARISH_HARAMI.value: 60,             # 看跌母子线
                PatternType.BULLISH_HARAMI_CROSS.value: 62,       # 看涨母子十字线
                PatternType.BEARISH_HARAMI_CROSS.value: 62,       # 看跌母子十字线
                PatternType.MORNING_STAR.value: 75,               # 晨星
                PatternType.EVENING_STAR.value: 75,               # 暮星
                PatternType.MORNING_DOJI_STAR.value: 78,          # 晨星十字星
                PatternType.EVENING_DOJI_STAR.value: 78,          # 暮星十字星
                PatternType.BULLISH_MARUBOZU.value: 68,           # 看涨光头光脚
                PatternType.BEARISH_MARUBOZU.value: 68            # 看跌光头光脚
            }
            
            # 合并权重字典
            pattern_weights.update(basic_pattern_weights)
            
            # 计算信号强度
            for pattern, weight in pattern_weights.items():
                if pattern in indicator_values.columns:
                    # 使用矢量化操作更新信号强度
                    signal_strength = signal_strength.mask(
                        indicator_values[pattern],
                        signal_strength + weight
                    )
            
            # 同时存在多个形态时，取最大信号强度的80%，再加上其他信号的20%
            # 这样可以避免多个弱信号叠加导致的虚假强信号
            pattern_count = indicator_values[list(pattern_weights.keys()) & set(indicator_values.columns)].sum(axis=1)
            
            # 当存在多个形态时进行调整
            multi_pattern_mask = pattern_count > 1
            if multi_pattern_mask.any():
                # 复制原始信号强度
                adjusted_strength = signal_strength.copy()
                
                # 对存在多个形态的位置进行调整
                for idx in signal_strength[multi_pattern_mask].index:
                    try:
                        # 计算平均信号强度
                        strength = signal_strength[idx]
                        count = pattern_count[idx]
                        if count > 0:
                            adjusted_strength[idx] = min(100, strength / count * 0.8 + strength * 0.2)
                    except Exception as e:
                        logger.debug(f"调整信号强度时出错: {e}")
                
                # 更新信号强度
                signal_strength = adjusted_strength
            
            # 确保信号强度在[0, 100]范围内
            signal_strength = signal_strength.clip(0, 100)
            
        except Exception as e:
            logger.error(f"计算信号强度时出错: {e}")
            # 出错时返回零信号强度
            signal_strength = pd.Series(0, index=indicator_values.index)
        
        return signal_strength
    
    def _calculate_trend_confirmation(self, indicator_values: pd.DataFrame) -> pd.Series:
        """
        计算趋势确认信号
        
        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记
            
        Returns:
            pd.Series: 趋势确认信号序列，True表示信号被趋势确认，False表示未确认
        """
        # 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(False, index=indicator_values.index if indicator_values is not None else [])
        
        # 初始化趋势确认序列
        trend_confirmed = pd.Series(False, index=indicator_values.index)
        
        try:
            # 定义看涨形态和看跌形态
            bullish_patterns = [
                AdvancedPatternType.THREE_WHITE_SOLDIERS.value,
                AdvancedPatternType.THREE_INSIDE_UP.value,
                AdvancedPatternType.THREE_OUTSIDE_UP.value,
                AdvancedPatternType.RISING_THREE_METHODS.value,
                AdvancedPatternType.MAT_HOLD.value,
                AdvancedPatternType.LADDER_BOTTOM.value,
                AdvancedPatternType.BREAKAWAY.value,
                'HAMMER', 'INVERTED_HAMMER', 'BULLISH_ENGULFING', 
                'PIERCING_LINE', 'BULLISH_HARAMI', 'BULLISH_HARAMI_CROSS',
                'MORNING_STAR', 'MORNING_DOJI_STAR', 'BULLISH_MARUBOZU'
            ]
            
            bearish_patterns = [
                AdvancedPatternType.THREE_BLACK_CROWS.value,
                AdvancedPatternType.THREE_INSIDE_DOWN.value,
                AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
                AdvancedPatternType.FALLING_THREE_METHODS.value,
                AdvancedPatternType.TOWER_TOP.value,
                'HANGING_MAN', 'SHOOTING_STAR', 'BEARISH_ENGULFING',
                'DARK_CLOUD_COVER', 'BEARISH_HARAMI', 'BEARISH_HARAMI_CROSS',
                'EVENING_STAR', 'EVENING_DOJI_STAR', 'BEARISH_MARUBOZU'
            ]
            
            # 创建看涨和看跌信号序列
            bullish_signal = pd.Series(False, index=indicator_values.index)
            for pattern in bullish_patterns:
                if pattern in indicator_values.columns:
                    bullish_signal |= indicator_values[pattern]
            
            bearish_signal = pd.Series(False, index=indicator_values.index)
            for pattern in bearish_patterns:
                if pattern in indicator_values.columns:
                    bearish_signal |= indicator_values[pattern]
            
            # 假设indicator_values中包含移动平均线等趋势指标
            # 这里可以添加与其他指标的集成逻辑
            
            # 简单规则：如果形态发生在合适的价格位置，则认为趋势确认
            # 实际应用中，可以与移动平均线、趋势线等结合使用
            
            # 当前仅使用信号强度作为趋势确认的简单方法
            if 'signal_strength' in indicator_values.columns:
                signal_strength = indicator_values['signal_strength']
                
                # 信号强度大于70时认为趋势确认
                trend_confirmed = (
                    (bullish_signal & (signal_strength > 70)) | 
                    (bearish_signal & (signal_strength > 70))
                )
            else:
                # 没有信号强度时，使用形态本身的存在作为确认
                trend_confirmed = bullish_signal | bearish_signal
                
        except Exception as e:
            logger.error(f"计算趋势确认信号时出错: {e}")
            # 出错时返回未确认
            trend_confirmed = pd.Series(False, index=indicator_values.index)
        
        return trend_confirmed
    
    def _calculate_compound_signal(self, indicator_values: pd.DataFrame) -> pd.Series:
        """
        计算复合信号，识别多种形态同时出现的情况
        
        Args:
            indicator_values: 指标计算结果，包含各种K线形态的标记
            
        Returns:
            pd.Series: 复合信号序列，值越大表示信号越强
        """
        # 防止数据为空
        if indicator_values is None or len(indicator_values) == 0:
            return pd.Series(0, index=indicator_values.index if indicator_values is not None else [])
        
        # 初始化复合信号序列
        compound_signal = pd.Series(0, index=indicator_values.index)
        
        try:
            # 定义所有形态列表
            all_patterns = [
                # 高级形态
                AdvancedPatternType.THREE_WHITE_SOLDIERS.value,
                AdvancedPatternType.THREE_BLACK_CROWS.value,
                AdvancedPatternType.THREE_INSIDE_UP.value,
                AdvancedPatternType.THREE_INSIDE_DOWN.value,
                AdvancedPatternType.THREE_OUTSIDE_UP.value,
                AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
                AdvancedPatternType.RISING_THREE_METHODS.value,
                AdvancedPatternType.FALLING_THREE_METHODS.value,
                AdvancedPatternType.MAT_HOLD.value,
                AdvancedPatternType.STICK_SANDWICH.value,
                AdvancedPatternType.LADDER_BOTTOM.value,
                AdvancedPatternType.TOWER_TOP.value,
                AdvancedPatternType.BREAKAWAY.value,
                AdvancedPatternType.KICKING.value,
                AdvancedPatternType.UNIQUE_THREE_RIVER.value,
                
                # 基础形态
                'HAMMER', 'HANGING_MAN', 'SHOOTING_STAR', 'INVERTED_HAMMER',
                'DOJI', 'DRAGONFLY_DOJI', 'GRAVESTONE_DOJI',
                'BULLISH_ENGULFING', 'BEARISH_ENGULFING',
                'DARK_CLOUD_COVER', 'PIERCING_LINE',
                'BULLISH_HARAMI', 'BEARISH_HARAMI',
                'BULLISH_HARAMI_CROSS', 'BEARISH_HARAMI_CROSS',
                'MORNING_STAR', 'EVENING_STAR',
                'MORNING_DOJI_STAR', 'EVENING_DOJI_STAR',
                'BULLISH_MARUBOZU', 'BEARISH_MARUBOZU'
            ]
            
            # 计算每行有多少个形态同时出现
            patterns_count = pd.Series(0, index=indicator_values.index)
            
            for pattern in all_patterns:
                if pattern in indicator_values.columns:
                    patterns_count += indicator_values[pattern].astype(int)
            
            # 设置复合信号的强度
            # 1个形态：信号强度为1
            # 2个形态：信号强度为3
            # 3个或更多形态：信号强度为5
            compound_signal = pd.Series(0, index=indicator_values.index)
            compound_signal = compound_signal.mask(patterns_count == 1, 1)
            compound_signal = compound_signal.mask(patterns_count == 2, 3)
            compound_signal = compound_signal.mask(patterns_count >= 3, 5)
            
            # 查看是否有冲突信号（同时出现看涨和看跌形态）
            bullish_patterns = [
                AdvancedPatternType.THREE_WHITE_SOLDIERS.value,
                AdvancedPatternType.THREE_INSIDE_UP.value,
                AdvancedPatternType.THREE_OUTSIDE_UP.value,
                AdvancedPatternType.RISING_THREE_METHODS.value,
                AdvancedPatternType.MAT_HOLD.value,
                AdvancedPatternType.LADDER_BOTTOM.value,
                AdvancedPatternType.BREAKAWAY.value,
                'HAMMER', 'INVERTED_HAMMER', 'BULLISH_ENGULFING', 
                'PIERCING_LINE', 'BULLISH_HARAMI', 'BULLISH_HARAMI_CROSS',
                'MORNING_STAR', 'MORNING_DOJI_STAR', 'BULLISH_MARUBOZU'
            ]
            
            bearish_patterns = [
                AdvancedPatternType.THREE_BLACK_CROWS.value,
                AdvancedPatternType.THREE_INSIDE_DOWN.value,
                AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
                AdvancedPatternType.FALLING_THREE_METHODS.value,
                AdvancedPatternType.TOWER_TOP.value,
                'HANGING_MAN', 'SHOOTING_STAR', 'BEARISH_ENGULFING',
                'DARK_CLOUD_COVER', 'BEARISH_HARAMI', 'BEARISH_HARAMI_CROSS',
                'EVENING_STAR', 'EVENING_DOJI_STAR', 'BEARISH_MARUBOZU'
            ]
            
            # 统计看涨形态数量
            bullish_count = pd.Series(0, index=indicator_values.index)
            for pattern in bullish_patterns:
                if pattern in indicator_values.columns:
                    bullish_count += indicator_values[pattern].astype(int)
            
            # 统计看跌形态数量
            bearish_count = pd.Series(0, index=indicator_values.index)
            for pattern in bearish_patterns:
                if pattern in indicator_values.columns:
                    bearish_count += indicator_values[pattern].astype(int)
            
            # 当看涨和看跌形态同时出现时，降低复合信号强度
            conflict_mask = (bullish_count > 0) & (bearish_count > 0)
            compound_signal = compound_signal.mask(conflict_mask, compound_signal / 2)
            
            # 考虑信号强度因素
            if 'signal_strength' in indicator_values.columns:
                # 信号强度高的位置，提升复合信号强度
                signal_strength = indicator_values['signal_strength']
                compound_signal = compound_signal * (1 + signal_strength / 100)
                
            # 规范化到[0, 10]范围
            compound_signal = compound_signal.clip(0, 10)
            
        except Exception as e:
            logger.error(f"计算复合信号时出错: {e}")
            # 出错时返回零信号
            compound_signal = pd.Series(0, index=indicator_values.index)
        
        return compound_signal 