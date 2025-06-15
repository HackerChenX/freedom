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
    
    # 复杂形态
    HEAD_SHOULDERS_TOP = "头肩顶"           # 左肩+头部+右肩的顶部反转形态
    HEAD_SHOULDERS_BOTTOM = "头肩底"        # 左肩+头部+右肩的底部反转形态
    DOUBLE_TOP = "双顶"                    # 两个相近高点的顶部反转形态
    DOUBLE_BOTTOM = "双底"                 # 两个相近低点的底部反转形态
    TRIPLE_TOP = "三重顶"                  # 三个相近高点的顶部反转形态
    TRIPLE_BOTTOM = "三重底"               # 三个相近低点的底部反转形态
    TRIANGLE_ASCENDING = "上升三角形"       # 水平上轨+上升下轨的整理形态
    TRIANGLE_DESCENDING = "下降三角形"      # 下降上轨+水平下轨的整理形态
    TRIANGLE_SYMMETRICAL = "对称三角形"     # 上轨下降+下轨上升的整理形态
    RECTANGLE = "矩形整理"                 # 价格在水平支撑压力间震荡
    DIAMOND_TOP = "钻石顶"                # 菱形的顶部反转形态
    DIAMOND_BOTTOM = "钻石底"             # 菱形的底部反转形态
    CUP_WITH_HANDLE = "杯柄形态"          # U形底部+小幅回调形成柄部


class AdvancedCandlestickPatterns(BaseIndicator):
    """
    高级K线形态识别指标
    
    实现更复杂的组合K线形态和复合形态识别功能，包括三星形态、高级复合形态和其他复合形态。
    提供信号强度评估、趋势确认和复合信号分析功能。
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化高级K线形态识别指标"""
        super().__init__(name="AdvancedCandlestickPatterns", description="高级K线形态识别指标，识别更复杂的组合K线形态和复合形态")
        self.basic_patterns = CandlestickPatterns()
    
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        """
        # 高级K线形态识别通常没有可变参数，但为了符合接口要求，提供此方法
        pass
    
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
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
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
        result = data.copy()
        
        # 计算三星形态
        result = self._calculate_three_star_patterns(data, result)
        
        # 计算高级复合形态（需要至少5根K线）
        if len(data) >= 5:
            result = self._calculate_advanced_compound_patterns(data, result)
        
        # 计算其他复合形态
        result = self._calculate_other_compound_patterns(data, result)
        
        # 计算复杂形态（需要更多数据，至少20根K线）
        if len(data) >= 20:
            result = self._calculate_complex_patterns(data, result)

        # 确保所有高级形态列都存在（即使数据不足）
        all_advanced_pattern_names = [pattern.value for pattern in AdvancedPatternType]
        for pattern_name in all_advanced_pattern_names:
            if pattern_name not in result.columns:
                result[pattern_name] = False

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
            AdvancedPatternType.BREAKAWAY.value,
            AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value,
            AdvancedPatternType.DOUBLE_BOTTOM.value,
            AdvancedPatternType.TRIPLE_BOTTOM.value,
            AdvancedPatternType.DIAMOND_BOTTOM.value,
            AdvancedPatternType.CUP_WITH_HANDLE.value
        ]
        
        bearish_patterns = [
            AdvancedPatternType.THREE_BLACK_CROWS.value,
            AdvancedPatternType.THREE_INSIDE_DOWN.value,
            AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
            AdvancedPatternType.FALLING_THREE_METHODS.value,
            AdvancedPatternType.TOWER_TOP.value,
            AdvancedPatternType.HEAD_SHOULDERS_TOP.value,
            AdvancedPatternType.DOUBLE_TOP.value,
            AdvancedPatternType.TRIPLE_TOP.value,
            AdvancedPatternType.DIAMOND_TOP.value
        ]
        
        neutral_patterns = [
            AdvancedPatternType.STICK_SANDWICH.value,
            AdvancedPatternType.KICKING.value,
            AdvancedPatternType.UNIQUE_THREE_RIVER.value,
            AdvancedPatternType.TRIANGLE_ASCENDING.value,
            AdvancedPatternType.TRIANGLE_DESCENDING.value,
            AdvancedPatternType.TRIANGLE_SYMMETRICAL.value,
            AdvancedPatternType.RECTANGLE.value
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
    
    def _calculate_complex_patterns(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算复杂形态（头肩顶/底、双顶/底、三角形等）
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        high_prices = data["high"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        
        # 计算移动平均线（用于帮助识别形态）
        ma20 = np.convolve(close_prices, np.ones(20)/20, mode='valid')
        
        # 初始化结果数组
        n = len(data)
        head_shoulders_top = np.zeros(n, dtype=bool)
        head_shoulders_bottom = np.zeros(n, dtype=bool)
        double_top = np.zeros(n, dtype=bool)
        double_bottom = np.zeros(n, dtype=bool)
        triple_top = np.zeros(n, dtype=bool)
        triple_bottom = np.zeros(n, dtype=bool)
        triangle_ascending = np.zeros(n, dtype=bool)
        triangle_descending = np.zeros(n, dtype=bool)
        triangle_symmetrical = np.zeros(n, dtype=bool)
        rectangle = np.zeros(n, dtype=bool)
        diamond_top = np.zeros(n, dtype=bool)
        diamond_bottom = np.zeros(n, dtype=bool)
        cup_with_handle = np.zeros(n, dtype=bool)
        
        # 局部极值查找窗口大小
        window = 5
        
        # 查找局部高点和低点
        peaks = np.zeros(n, dtype=bool)
        troughs = np.zeros(n, dtype=bool)
        
        for i in range(window, n - window):
            # 局部高点：当前高点高于前后window个点的高点
            if all(high_prices[i] > high_prices[i-window:i]) and all(high_prices[i] > high_prices[i+1:i+window+1]):
                peaks[i] = True
            
            # 局部低点：当前低点低于前后window个点的低点
            if all(low_prices[i] < low_prices[i-window:i]) and all(low_prices[i] < low_prices[i+1:i+window+1]):
                troughs[i] = True
        
        # 获取所有峰值和谷值的索引
        peak_indices = np.where(peaks)[0]
        trough_indices = np.where(troughs)[0]
        
        # 头肩顶识别
        for i in range(len(peak_indices) - 2):
            # 取三个连续的峰值
            p1 = peak_indices[i]
            p2 = peak_indices[i+1]
            p3 = peak_indices[i+2]
            
            # 确保峰值之间有足够的距离
            if p2 - p1 >= window * 2 and p3 - p2 >= window * 2:
                # 头部（中间峰值）高于两侧肩部
                if high_prices[p2] > high_prices[p1] and high_prices[p2] > high_prices[p3]:
                    # 两肩高度相近（差异不超过20%）
                    shoulder_diff = abs(high_prices[p1] - high_prices[p3]) / high_prices[p1]
                    if shoulder_diff < 0.2:
                        # 找到两个峰值之间的谷值
                        t1 = trough_indices[np.logical_and(trough_indices > p1, trough_indices < p2)]
                        t2 = trough_indices[np.logical_and(trough_indices > p2, trough_indices < p3)]
                        
                        if len(t1) > 0 and len(t2) > 0:
                            neckline_level1 = low_prices[t1[0]]
                            neckline_level2 = low_prices[t2[0]]
                            
                            # 颈线水平（差异不超过10%）
                            neckline_diff = abs(neckline_level1 - neckline_level2) / neckline_level1
                            if neckline_diff < 0.1:
                                # 标记头肩顶形态
                                head_shoulders_top[p3] = True
        
        # 头肩底识别
        for i in range(len(trough_indices) - 2):
            # 取三个连续的谷值
            t1 = trough_indices[i]
            t2 = trough_indices[i+1]
            t3 = trough_indices[i+2]
            
            # 确保谷值之间有足够的距离
            if t2 - t1 >= window * 2 and t3 - t2 >= window * 2:
                # 头部（中间谷值）低于两侧肩部
                if low_prices[t2] < low_prices[t1] and low_prices[t2] < low_prices[t3]:
                    # 两肩高度相近（差异不超过20%）
                    shoulder_diff = abs(low_prices[t1] - low_prices[t3]) / low_prices[t1]
                    if shoulder_diff < 0.2:
                        # 找到两个谷值之间的峰值
                        p1 = peak_indices[np.logical_and(peak_indices > t1, peak_indices < t2)]
                        p2 = peak_indices[np.logical_and(peak_indices > t2, peak_indices < t3)]
                        
                        if len(p1) > 0 and len(p2) > 0:
                            neckline_level1 = high_prices[p1[0]]
                            neckline_level2 = high_prices[p2[0]]
                            
                            # 颈线水平（差异不超过10%）
                            neckline_diff = abs(neckline_level1 - neckline_level2) / neckline_level1
                            if neckline_diff < 0.1:
                                # 标记头肩底形态
                                head_shoulders_bottom[t3] = True
        
        # 双顶识别
        for i in range(len(peak_indices) - 1):
            p1 = peak_indices[i]
            p2 = peak_indices[i+1]
            
            # 确保两个峰值之间有足够的距离
            if p2 - p1 >= window * 3:
                # 两个峰值高度相近（差异不超过5%）
                peak_diff = abs(high_prices[p1] - high_prices[p2]) / high_prices[p1]
                if peak_diff < 0.05:
                    # 找到两个峰值之间的谷值
                    mid_troughs = trough_indices[np.logical_and(trough_indices > p1, trough_indices < p2)]
                    
                    if len(mid_troughs) > 0:
                        # 谷值显著低于峰值（至少10%）
                        trough_depth = (high_prices[p1] - low_prices[mid_troughs[0]]) / high_prices[p1]
                        if trough_depth > 0.1:
                            # 标记双顶形态
                            double_top[p2] = True
        
        # 双底识别
        for i in range(len(trough_indices) - 1):
            t1 = trough_indices[i]
            t2 = trough_indices[i+1]
            
            # 确保两个谷值之间有足够的距离
            if t2 - t1 >= window * 3:
                # 两个谷值高度相近（差异不超过5%）
                trough_diff = abs(low_prices[t1] - low_prices[t2]) / low_prices[t1]
                if trough_diff < 0.05:
                    # 找到两个谷值之间的峰值
                    mid_peaks = peak_indices[np.logical_and(peak_indices > t1, peak_indices < t2)]
                    
                    if len(mid_peaks) > 0:
                        # 峰值显著高于谷值（至少10%）
                        peak_height = (high_prices[mid_peaks[0]] - low_prices[t1]) / low_prices[t1]
                        if peak_height > 0.1:
                            # 标记双底形态
                            double_bottom[t2] = True
        
        # 三角形识别（这里只实现对称三角形识别，其他三角形类似）
        for i in range(n - 20):
            # 至少需要3个峰值和3个谷值来形成三角形
            window_peaks = peak_indices[np.logical_and(peak_indices >= i, peak_indices < i + 20)]
            window_troughs = trough_indices[np.logical_and(trough_indices >= i, trough_indices < i + 20)]
            
            if len(window_peaks) >= 3 and len(window_troughs) >= 3:
                # 检查高点是否递减
                descending_tops = all(high_prices[window_peaks[j]] > high_prices[window_peaks[j+1]] for j in range(len(window_peaks)-1))
                
                # 检查低点是否递增
                ascending_bottoms = all(low_prices[window_troughs[j]] < low_prices[window_troughs[j+1]] for j in range(len(window_troughs)-1))
                
                if descending_tops and ascending_bottoms:
                    # 标记对称三角形
                    triangle_symmetrical[i+19] = True
        
        # 将识别结果添加到结果数据框
        result[AdvancedPatternType.HEAD_SHOULDERS_TOP.value] = head_shoulders_top
        result[AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value] = head_shoulders_bottom
        result[AdvancedPatternType.DOUBLE_TOP.value] = double_top
        result[AdvancedPatternType.DOUBLE_BOTTOM.value] = double_bottom
        result[AdvancedPatternType.TRIPLE_TOP.value] = triple_top
        result[AdvancedPatternType.TRIPLE_BOTTOM.value] = triple_bottom
        result[AdvancedPatternType.TRIANGLE_ASCENDING.value] = triangle_ascending
        result[AdvancedPatternType.TRIANGLE_DESCENDING.value] = triangle_descending
        result[AdvancedPatternType.TRIANGLE_SYMMETRICAL.value] = triangle_symmetrical
        result[AdvancedPatternType.RECTANGLE.value] = rectangle
        result[AdvancedPatternType.DIAMOND_TOP.value] = diamond_top
        result[AdvancedPatternType.DIAMOND_BOTTOM.value] = diamond_bottom
        result[AdvancedPatternType.CUP_WITH_HANDLE.value] = cup_with_handle
        
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
                AdvancedPatternType.UNIQUE_THREE_RIVER.value: 72,    # 奇特三河
                
                # 复杂形态
                AdvancedPatternType.HEAD_SHOULDERS_TOP.value: 80,    # 头肩顶
                AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value: 80,   # 头肩底
                AdvancedPatternType.DOUBLE_TOP.value: 75,                # 双顶
                AdvancedPatternType.DOUBLE_BOTTOM.value: 75,               # 双底
                AdvancedPatternType.TRIPLE_TOP.value: 70,                  # 三重顶
                AdvancedPatternType.TRIPLE_BOTTOM.value: 70,               # 三重底
                AdvancedPatternType.TRIANGLE_ASCENDING.value: 65,          # 上升三角形
                AdvancedPatternType.TRIANGLE_DESCENDING.value: 65,         # 下降三角形
                AdvancedPatternType.TRIANGLE_SYMMETRICAL.value: 60,         # 对称三角形
                AdvancedPatternType.RECTANGLE.value: 55,                    # 矩形整理
                AdvancedPatternType.DIAMOND_TOP.value: 50,                   # 钻石顶
                AdvancedPatternType.DIAMOND_BOTTOM.value: 50,                # 钻石底
                AdvancedPatternType.CUP_WITH_HANDLE.value: 45                 # 杯柄形态
            }
            
            # 添加基础K线形态的权重
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
                AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value,
                AdvancedPatternType.DOUBLE_BOTTOM.value,
                AdvancedPatternType.TRIPLE_BOTTOM.value,
                AdvancedPatternType.DIAMOND_BOTTOM.value,
                AdvancedPatternType.CUP_WITH_HANDLE.value
            ]
            
            bearish_patterns = [
                AdvancedPatternType.THREE_BLACK_CROWS.value,
                AdvancedPatternType.THREE_INSIDE_DOWN.value,
                AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
                AdvancedPatternType.FALLING_THREE_METHODS.value,
                AdvancedPatternType.TOWER_TOP.value,
                AdvancedPatternType.HEAD_SHOULDERS_TOP.value,
                AdvancedPatternType.DOUBLE_TOP.value,
                AdvancedPatternType.TRIPLE_TOP.value,
                AdvancedPatternType.DIAMOND_TOP.value
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
                
                # 复杂形态
                AdvancedPatternType.HEAD_SHOULDERS_TOP.value,
                AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value,
                AdvancedPatternType.DOUBLE_TOP.value,
                AdvancedPatternType.DOUBLE_BOTTOM.value,
                AdvancedPatternType.TRIPLE_TOP.value,
                AdvancedPatternType.TRIPLE_BOTTOM.value,
                AdvancedPatternType.TRIANGLE_ASCENDING.value,
                AdvancedPatternType.TRIANGLE_DESCENDING.value,
                AdvancedPatternType.TRIANGLE_SYMMETRICAL.value,
                AdvancedPatternType.RECTANGLE.value,
                AdvancedPatternType.DIAMOND_TOP.value,
                AdvancedPatternType.DIAMOND_BOTTOM.value,
                AdvancedPatternType.CUP_WITH_HANDLE.value
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
                AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value,
                AdvancedPatternType.DOUBLE_BOTTOM.value,
                AdvancedPatternType.TRIPLE_BOTTOM.value,
                AdvancedPatternType.DIAMOND_BOTTOM.value,
                AdvancedPatternType.CUP_WITH_HANDLE.value
            ]
            
            bearish_patterns = [
                AdvancedPatternType.THREE_BLACK_CROWS.value,
                AdvancedPatternType.THREE_INSIDE_DOWN.value,
                AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
                AdvancedPatternType.FALLING_THREE_METHODS.value,
                AdvancedPatternType.TOWER_TOP.value,
                AdvancedPatternType.HEAD_SHOULDERS_TOP.value,
                AdvancedPatternType.DOUBLE_TOP.value,
                AdvancedPatternType.TRIPLE_TOP.value,
                AdvancedPatternType.DIAMOND_TOP.value
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
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级K线形态识别指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 强烈看涨形态评分（+25到+40分）
        # 三星看涨形态
        if AdvancedPatternType.THREE_WHITE_SOLDIERS.value in indicator_data.columns:
            three_white_soldiers_mask = indicator_data[AdvancedPatternType.THREE_WHITE_SOLDIERS.value]
            score.loc[three_white_soldiers_mask] += 35
        
        if AdvancedPatternType.THREE_INSIDE_UP.value in indicator_data.columns:
            three_inside_up_mask = indicator_data[AdvancedPatternType.THREE_INSIDE_UP.value]
            score.loc[three_inside_up_mask] += 30
        
        if AdvancedPatternType.THREE_OUTSIDE_UP.value in indicator_data.columns:
            three_outside_up_mask = indicator_data[AdvancedPatternType.THREE_OUTSIDE_UP.value]
            score.loc[three_outside_up_mask] += 32
        
        # 高级复合看涨形态
        if AdvancedPatternType.RISING_THREE_METHODS.value in indicator_data.columns:
            rising_three_methods_mask = indicator_data[AdvancedPatternType.RISING_THREE_METHODS.value]
            score.loc[rising_three_methods_mask] += 28
        
        if AdvancedPatternType.MAT_HOLD.value in indicator_data.columns:
            mat_hold_mask = indicator_data[AdvancedPatternType.MAT_HOLD.value]
            score.loc[mat_hold_mask] += 25
        
        if AdvancedPatternType.LADDER_BOTTOM.value in indicator_data.columns:
            ladder_bottom_mask = indicator_data[AdvancedPatternType.LADDER_BOTTOM.value]
            score.loc[ladder_bottom_mask] += 30
        
        if AdvancedPatternType.BREAKAWAY.value in indicator_data.columns:
            breakaway_mask = indicator_data[AdvancedPatternType.BREAKAWAY.value]
            # 需要判断突破方向，这里假设是看涨突破
            score.loc[breakaway_mask] += 25
        
        # 复杂看涨形态
        if AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value in indicator_data.columns:
            head_shoulders_bottom_mask = indicator_data[AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value]
            score.loc[head_shoulders_bottom_mask] += 40
        
        if AdvancedPatternType.DOUBLE_BOTTOM.value in indicator_data.columns:
            double_bottom_mask = indicator_data[AdvancedPatternType.DOUBLE_BOTTOM.value]
            score.loc[double_bottom_mask] += 35
        
        if AdvancedPatternType.TRIPLE_BOTTOM.value in indicator_data.columns:
            triple_bottom_mask = indicator_data[AdvancedPatternType.TRIPLE_BOTTOM.value]
            score.loc[triple_bottom_mask] += 38
        
        if AdvancedPatternType.DIAMOND_BOTTOM.value in indicator_data.columns:
            diamond_bottom_mask = indicator_data[AdvancedPatternType.DIAMOND_BOTTOM.value]
            score.loc[diamond_bottom_mask] += 35
        
        if AdvancedPatternType.CUP_WITH_HANDLE.value in indicator_data.columns:
            cup_handle_mask = indicator_data[AdvancedPatternType.CUP_WITH_HANDLE.value]
            score.loc[cup_handle_mask] += 32
        
        # 2. 强烈看跌形态评分（-25到-40分）
        # 三星看跌形态
        if AdvancedPatternType.THREE_BLACK_CROWS.value in indicator_data.columns:
            three_black_crows_mask = indicator_data[AdvancedPatternType.THREE_BLACK_CROWS.value]
            score.loc[three_black_crows_mask] -= 35
        
        if AdvancedPatternType.THREE_INSIDE_DOWN.value in indicator_data.columns:
            three_inside_down_mask = indicator_data[AdvancedPatternType.THREE_INSIDE_DOWN.value]
            score.loc[three_inside_down_mask] -= 30
        
        if AdvancedPatternType.THREE_OUTSIDE_DOWN.value in indicator_data.columns:
            three_outside_down_mask = indicator_data[AdvancedPatternType.THREE_OUTSIDE_DOWN.value]
            score.loc[three_outside_down_mask] -= 32
        
        # 高级复合看跌形态
        if AdvancedPatternType.FALLING_THREE_METHODS.value in indicator_data.columns:
            falling_three_methods_mask = indicator_data[AdvancedPatternType.FALLING_THREE_METHODS.value]
            score.loc[falling_three_methods_mask] -= 28
        
        if AdvancedPatternType.TOWER_TOP.value in indicator_data.columns:
            tower_top_mask = indicator_data[AdvancedPatternType.TOWER_TOP.value]
            score.loc[tower_top_mask] -= 30
        
        # 复杂看跌形态
        if AdvancedPatternType.HEAD_SHOULDERS_TOP.value in indicator_data.columns:
            head_shoulders_top_mask = indicator_data[AdvancedPatternType.HEAD_SHOULDERS_TOP.value]
            score.loc[head_shoulders_top_mask] -= 40
        
        if AdvancedPatternType.DOUBLE_TOP.value in indicator_data.columns:
            double_top_mask = indicator_data[AdvancedPatternType.DOUBLE_TOP.value]
            score.loc[double_top_mask] -= 35
        
        if AdvancedPatternType.TRIPLE_TOP.value in indicator_data.columns:
            triple_top_mask = indicator_data[AdvancedPatternType.TRIPLE_TOP.value]
            score.loc[triple_top_mask] -= 38
        
        if AdvancedPatternType.DIAMOND_TOP.value in indicator_data.columns:
            diamond_top_mask = indicator_data[AdvancedPatternType.DIAMOND_TOP.value]
            score.loc[diamond_top_mask] -= 35
        
        # 3. 中性/整理形态评分（-10到+10分）
        if AdvancedPatternType.STICK_SANDWICH.value in indicator_data.columns:
            stick_sandwich_mask = indicator_data[AdvancedPatternType.STICK_SANDWICH.value]
            score.loc[stick_sandwich_mask] += 5  # 轻微看涨倾向
        
        if AdvancedPatternType.KICKING.value in indicator_data.columns:
            kicking_mask = indicator_data[AdvancedPatternType.KICKING.value]
            # 反冲形态需要判断方向，这里给中性评分
            score.loc[kicking_mask] += 0
        
        if AdvancedPatternType.UNIQUE_THREE_RIVER.value in indicator_data.columns:
            unique_three_river_mask = indicator_data[AdvancedPatternType.UNIQUE_THREE_RIVER.value]
            score.loc[unique_three_river_mask] += 15  # 底部反转形态
        
        # 三角形整理形态
        if AdvancedPatternType.TRIANGLE_ASCENDING.value in indicator_data.columns:
            triangle_ascending_mask = indicator_data[AdvancedPatternType.TRIANGLE_ASCENDING.value]
            score.loc[triangle_ascending_mask] += 8  # 轻微看涨倾向
        
        if AdvancedPatternType.TRIANGLE_DESCENDING.value in indicator_data.columns:
            triangle_descending_mask = indicator_data[AdvancedPatternType.TRIANGLE_DESCENDING.value]
            score.loc[triangle_descending_mask] -= 8  # 轻微看跌倾向
        
        if AdvancedPatternType.TRIANGLE_SYMMETRICAL.value in indicator_data.columns:
            triangle_symmetrical_mask = indicator_data[AdvancedPatternType.TRIANGLE_SYMMETRICAL.value]
            score.loc[triangle_symmetrical_mask] += 0  # 中性
        
        if AdvancedPatternType.RECTANGLE.value in indicator_data.columns:
            rectangle_mask = indicator_data[AdvancedPatternType.RECTANGLE.value]
            score.loc[rectangle_mask] += 0  # 中性整理
        
        # 4. 形态强度调整（±15分）
        # 根据成交量确认形态强度
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 任何形态如果伴随放量，增强信号强度
            high_volume_mask = vol_ratio > 1.5
            
            # 看涨形态+放量
            bullish_patterns = (
                indicator_data.get(AdvancedPatternType.THREE_WHITE_SOLDIERS.value, False) |
                indicator_data.get(AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value, False) |
                indicator_data.get(AdvancedPatternType.DOUBLE_BOTTOM.value, False) |
                indicator_data.get(AdvancedPatternType.CUP_WITH_HANDLE.value, False)
            )
            if isinstance(bullish_patterns, pd.Series):
                bullish_volume_confirm = bullish_patterns & high_volume_mask
                score.loc[bullish_volume_confirm] += 15
            
            # 看跌形态+放量
            bearish_patterns = (
                indicator_data.get(AdvancedPatternType.THREE_BLACK_CROWS.value, False) |
                indicator_data.get(AdvancedPatternType.HEAD_SHOULDERS_TOP.value, False) |
                indicator_data.get(AdvancedPatternType.DOUBLE_TOP.value, False) |
                indicator_data.get(AdvancedPatternType.TOWER_TOP.value, False)
            )
            if isinstance(bearish_patterns, pd.Series):
                bearish_volume_confirm = bearish_patterns & high_volume_mask
                score.loc[bearish_volume_confirm] -= 15
        
        # 5. 形态完整性调整（±10分）
        # 检查形态的完整性和质量
        # 这里可以添加更复杂的形态质量评估逻辑
        
        # 6. 多重形态确认（±20分）
        # 检查是否有多个形态同时出现
        pattern_count = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0  # 添加neutral_count初始化
        
        # 统计当前时点的形态数量
        for pattern_type in AdvancedPatternType:
            pattern_name = pattern_type.value
            if pattern_name in indicator_data.columns:
                current_pattern = indicator_data[pattern_name]
                if isinstance(current_pattern, pd.Series):
                    pattern_count += current_pattern.astype(int)
                    
                    # 分类统计
                    if pattern_type in [AdvancedPatternType.THREE_WHITE_SOLDIERS, 
                                      AdvancedPatternType.THREE_INSIDE_UP,
                                      AdvancedPatternType.THREE_OUTSIDE_UP,
                                      AdvancedPatternType.RISING_THREE_METHODS,
                                      AdvancedPatternType.MAT_HOLD,
                                      AdvancedPatternType.LADDER_BOTTOM,
                                      AdvancedPatternType.HEAD_SHOULDERS_BOTTOM,
                                      AdvancedPatternType.DOUBLE_BOTTOM,
                                      AdvancedPatternType.TRIPLE_BOTTOM,
                                      AdvancedPatternType.DIAMOND_BOTTOM,
                                      AdvancedPatternType.CUP_WITH_HANDLE,
                                      AdvancedPatternType.UNIQUE_THREE_RIVER]:
                        bullish_count += 1
                    elif pattern_type in [AdvancedPatternType.THREE_BLACK_CROWS,
                                        AdvancedPatternType.THREE_INSIDE_DOWN,
                                        AdvancedPatternType.THREE_OUTSIDE_DOWN,
                                        AdvancedPatternType.FALLING_THREE_METHODS,
                                        AdvancedPatternType.TOWER_TOP,
                                        AdvancedPatternType.HEAD_SHOULDERS_TOP,
                                        AdvancedPatternType.DOUBLE_TOP,
                                        AdvancedPatternType.TRIPLE_TOP,
                                        AdvancedPatternType.DIAMOND_TOP]:
                        bearish_count += 1
                    else:
                        neutral_count += 1
        
        # 多重看涨形态确认
        if isinstance(bullish_count, pd.Series):
            multiple_bullish = bullish_count >= 2
            score.loc[multiple_bullish] += 20
        
        # 多重看跌形态确认
        if isinstance(bearish_count, pd.Series):
            multiple_bearish = bearish_count >= 2
            score.loc[multiple_bearish] -= 20
        
        # 形态冲突（同时出现看涨看跌形态）
        if isinstance(bullish_count, pd.Series) and isinstance(bearish_count, pd.Series):
            conflict_patterns = (bullish_count > 0) & (bearish_count > 0)
            score.loc[conflict_patterns] -= 10  # 冲突信号减分
        
        # 7. 形态位置调整（±15分）
        # 在关键技术位置的形态更重要
        if 'close' in data.columns and len(data) >= 60:
            close_price = data['close']
            
            # 计算支撑阻力位
            high_60 = close_price.rolling(window=60).max()
            low_60 = close_price.rolling(window=60).min()
            
            # 在阻力位附近的看跌形态
            near_resistance = close_price > high_60 * 0.95
            bearish_at_resistance = (
                (indicator_data.get(AdvancedPatternType.THREE_BLACK_CROWS.value, False) |
                 indicator_data.get(AdvancedPatternType.HEAD_SHOULDERS_TOP.value, False) |
                 indicator_data.get(AdvancedPatternType.DOUBLE_TOP.value, False)) &
                near_resistance
            )
            if isinstance(bearish_at_resistance, pd.Series):
                score.loc[bearish_at_resistance] -= 15
            
            # 在支撑位附近的看涨形态
            near_support = close_price < low_60 * 1.05
            bullish_at_support = (
                (indicator_data.get(AdvancedPatternType.THREE_WHITE_SOLDIERS.value, False) |
                 indicator_data.get(AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value, False) |
                 indicator_data.get(AdvancedPatternType.DOUBLE_BOTTOM.value, False)) &
                near_support
            )
            if isinstance(bullish_at_support, pd.Series):
                score.loc[bullish_at_support] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别高级K线形态相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 5:
            return patterns
        
        # 检查最近5天的形态
        recent_data = indicator_data.tail(5)
        
        # 1. 三星形态
        three_star_patterns = [
            AdvancedPatternType.THREE_WHITE_SOLDIERS,
            AdvancedPatternType.THREE_BLACK_CROWS,
            AdvancedPatternType.THREE_INSIDE_UP,
            AdvancedPatternType.THREE_INSIDE_DOWN,
            AdvancedPatternType.THREE_OUTSIDE_UP,
            AdvancedPatternType.THREE_OUTSIDE_DOWN
        ]
        
        for pattern_type in three_star_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"三星形态-{pattern_name}")
        
        # 2. 高级复合形态
        advanced_compound_patterns = [
            AdvancedPatternType.RISING_THREE_METHODS,
            AdvancedPatternType.FALLING_THREE_METHODS,
            AdvancedPatternType.MAT_HOLD,
            AdvancedPatternType.STICK_SANDWICH
        ]
        
        for pattern_type in advanced_compound_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"高级复合形态-{pattern_name}")
        
        # 3. 其他复合形态
        other_compound_patterns = [
            AdvancedPatternType.LADDER_BOTTOM,
            AdvancedPatternType.TOWER_TOP,
            AdvancedPatternType.BREAKAWAY,
            AdvancedPatternType.KICKING,
            AdvancedPatternType.UNIQUE_THREE_RIVER
        ]
        
        for pattern_type in other_compound_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"其他复合形态-{pattern_name}")
        
        # 4. 复杂形态
        complex_patterns = [
            AdvancedPatternType.HEAD_SHOULDERS_TOP,
            AdvancedPatternType.HEAD_SHOULDERS_BOTTOM,
            AdvancedPatternType.DOUBLE_TOP,
            AdvancedPatternType.DOUBLE_BOTTOM,
            AdvancedPatternType.TRIPLE_TOP,
            AdvancedPatternType.TRIPLE_BOTTOM,
            AdvancedPatternType.TRIANGLE_ASCENDING,
            AdvancedPatternType.TRIANGLE_DESCENDING,
            AdvancedPatternType.TRIANGLE_SYMMETRICAL,
            AdvancedPatternType.RECTANGLE,
            AdvancedPatternType.DIAMOND_TOP,
            AdvancedPatternType.DIAMOND_BOTTOM,
            AdvancedPatternType.CUP_WITH_HANDLE
        ]
        
        for pattern_type in complex_patterns:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                patterns.append(f"复杂形态-{pattern_name}")
        
        # 5. 形态强度分析
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]
            
            if pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 2.0:
                    patterns.append("形态确认-巨量配合")
                elif latest_vol_ratio > 1.5:
                    patterns.append("形态确认-放量配合")
                elif latest_vol_ratio < 0.7:
                    patterns.append("形态确认-缩量形成")
        
        # 6. 形态组合分析
        # 统计不同类型形态的数量
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_patterns = 0
        
        for pattern_type in AdvancedPatternType:
            pattern_name = pattern_type.value
            if pattern_name in recent_data.columns and recent_data[pattern_name].any():
                total_patterns += 1
                
                # 分类统计
                if pattern_type in [AdvancedPatternType.THREE_WHITE_SOLDIERS, 
                                  AdvancedPatternType.THREE_INSIDE_UP,
                                  AdvancedPatternType.THREE_OUTSIDE_UP,
                                  AdvancedPatternType.RISING_THREE_METHODS,
                                  AdvancedPatternType.MAT_HOLD,
                                  AdvancedPatternType.LADDER_BOTTOM,
                                  AdvancedPatternType.HEAD_SHOULDERS_BOTTOM,
                                  AdvancedPatternType.DOUBLE_BOTTOM,
                                  AdvancedPatternType.TRIPLE_BOTTOM,
                                  AdvancedPatternType.DIAMOND_BOTTOM,
                                  AdvancedPatternType.CUP_WITH_HANDLE,
                                  AdvancedPatternType.UNIQUE_THREE_RIVER]:
                    bullish_count += 1
                elif pattern_type in [AdvancedPatternType.THREE_BLACK_CROWS,
                                    AdvancedPatternType.THREE_INSIDE_DOWN,
                                    AdvancedPatternType.THREE_OUTSIDE_DOWN,
                                    AdvancedPatternType.FALLING_THREE_METHODS,
                                    AdvancedPatternType.TOWER_TOP,
                                    AdvancedPatternType.HEAD_SHOULDERS_TOP,
                                    AdvancedPatternType.DOUBLE_TOP,
                                    AdvancedPatternType.TRIPLE_TOP,
                                    AdvancedPatternType.DIAMOND_TOP]:
                    bearish_count += 1
                else:
                    neutral_count += 1
        
        # 形态组合描述
        if total_patterns > 1:
            patterns.append(f"形态组合-{total_patterns}个高级形态同现")
        
        if bullish_count > bearish_count and bullish_count >= 2:
            patterns.append("高级形态共振-多重看涨信号")
        elif bearish_count > bullish_count and bearish_count >= 2:
            patterns.append("高级形态共振-多重看跌信号")
        elif bullish_count > 0 and bearish_count > 0:
            patterns.append("高级形态冲突-多空信号混杂")
        
        # 7. 形态复杂度分析
        if total_patterns >= 3:
            patterns.append("高复杂度形态组合")
        elif total_patterns == 2:
            patterns.append("中等复杂度形态组合")
        elif total_patterns == 1:
            patterns.append("单一高级形态")
        
        # 8. 形态时效性分析
        # 检查形态是否在最近1-2天内形成
        very_recent_data = indicator_data.tail(2)
        recent_pattern_count = 0
        
        for pattern_type in AdvancedPatternType:
            pattern_name = pattern_type.value
            if pattern_name in very_recent_data.columns and very_recent_data[pattern_name].any():
                recent_pattern_count += 1
        
        if recent_pattern_count > 0:
            patterns.append(f"新形成形态-{recent_pattern_count}个")
        
        # 9. 形态位置分析
        if 'close' in data.columns and len(data) >= 60:
            close_price = data['close']
            high_60 = close_price.rolling(window=60).max().iloc[-1]
            low_60 = close_price.rolling(window=60).min().iloc[-1]
            latest_close = close_price.iloc[-1]
            
            if pd.notna(latest_close) and pd.notna(high_60) and pd.notna(low_60):
                if latest_close > high_60 * 0.95:
                    patterns.append("高级形态位置-接近阻力位")
                elif latest_close < low_60 * 1.05:
                    patterns.append("高级形态位置-接近支撑位")
                else:
                    price_position = (latest_close - low_60) / (high_60 - low_60)
                    if price_position > 0.7:
                        patterns.append("高级形态位置-相对高位")
                    elif price_position < 0.3:
                        patterns.append("高级形态位置-相对低位")
                    else:
                        patterns.append("高级形态位置-中性区域")
        
        return patterns

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取AdvancedCandlestickPatterns相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return patterns

        # 直接返回计算结果，因为_calculate已经包含了所有形态
        return self._result

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算AdvancedCandlestickPatterns指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有高级形态数据
            advanced_pattern_columns = [pattern.value for pattern in AdvancedPatternType]
            available_patterns = [col for col in advanced_pattern_columns if col in self._result.columns]
            if available_patterns:
                # 高级形态数据越完整，置信度越高
                data_completeness = len(available_patterns) / len(AdvancedPatternType)
                confidence += data_completeness * 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查AdvancedCandlestickPatterns形态（只计算布尔列）
            bool_columns = patterns.select_dtypes(include=[bool]).columns
            if len(bool_columns) > 0:
                pattern_count = patterns[bool_columns].sum().sum()
                if pattern_count > 0:
                    confidence += min(pattern_count * 0.02, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于数据长度的置信度
        if len(score) >= 60:  # 两个月数据
            confidence += 0.1
        elif len(score) >= 30:  # 一个月数据
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册AdvancedCandlestickPatterns指标的形态到全局形态注册表
        """
        # 注册三星形态
        self.register_pattern_to_registry(
            pattern_id="THREE_WHITE_SOLDIERS",
            display_name="三白兵",
            description="连续三根阳线，每根都收于接近最高点，强烈的上涨信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0
        )

        self.register_pattern_to_registry(
            pattern_id="THREE_BLACK_CROWS",
            display_name="三黑鸦",
            description="连续三根阴线，每根都收于接近最低点，强烈的下跌信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0
        )

        self.register_pattern_to_registry(
            pattern_id="THREE_INSIDE_UP",
            display_name="三内涨",
            description="大阴线+小阳线在阴线实体内+突破阴线收盘价的阳线",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="THREE_INSIDE_DOWN",
            display_name="三内跌",
            description="大阳线+小阴线在阳线实体内+突破阳线收盘价的阴线",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0
        )

        # 注册高级复合形态
        self.register_pattern_to_registry(
            pattern_id="RISING_THREE_METHODS",
            display_name="上升三法",
            description="大阳线后三根小K线在大阳线范围内整理，然后一根突破的阳线",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=28.0
        )

        self.register_pattern_to_registry(
            pattern_id="FALLING_THREE_METHODS",
            display_name="下降三法",
            description="大阴线后三根小K线在大阴线范围内整理，然后一根突破的阴线",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-28.0
        )

        self.register_pattern_to_registry(
            pattern_id="MAT_HOLD",
            display_name="铺垫形态",
            description="大阳线后2-3根小阴线在大阳线上部整理，然后一根大阳线",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        # 注册复杂形态
        self.register_pattern_to_registry(
            pattern_id="HEAD_SHOULDERS_TOP",
            display_name="头肩顶",
            description="三个波峰，中间高于两侧，强烈的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-40.0
        )

        self.register_pattern_to_registry(
            pattern_id="HEAD_SHOULDERS_BOTTOM",
            display_name="头肩底",
            description="三个波谷，中间低于两侧，强烈的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=40.0
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_TOP",
            display_name="双顶",
            description="两个相近高点的顶部反转形态",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-35.0
        )

        self.register_pattern_to_registry(
            pattern_id="DOUBLE_BOTTOM",
            display_name="双底",
            description="两个相近低点的底部反转形态",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=35.0
        )

        # 注册其他重要形态
        self.register_pattern_to_registry(
            pattern_id="BREAKAWAY",
            display_name="脱离形态",
            description="五根K线组成的反转形态，突破性强",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="KICKING",
            display_name="反冲形态",
            description="两根相反方向的光头光脚K线，反转信号强烈",
            pattern_type="NEUTRAL",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="TRIANGLE_ASCENDING",
            display_name="上升三角形",
            description="水平上轨+上升下轨的整理形态，通常向上突破",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=8.0
        )

        self.register_pattern_to_registry(
            pattern_id="TRIANGLE_DESCENDING",
            display_name="下降三角形",
            description="下降上轨+水平下轨的整理形态，通常向下突破",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-8.0
        )

        self.register_pattern_to_registry(
            pattern_id="CUP_WITH_HANDLE",
            display_name="杯柄形态",
            description="U形底部+小幅回调形成柄部，长期看涨形态",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=32.0
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        生成AdvancedCandlestickPatterns交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            dict: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        if self._result is None or self._result.empty:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        # 初始化信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 定义看涨形态
        bullish_patterns = [
            AdvancedPatternType.THREE_WHITE_SOLDIERS.value,
            AdvancedPatternType.THREE_INSIDE_UP.value,
            AdvancedPatternType.THREE_OUTSIDE_UP.value,
            AdvancedPatternType.RISING_THREE_METHODS.value,
            AdvancedPatternType.MAT_HOLD.value,
            AdvancedPatternType.LADDER_BOTTOM.value,
            AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value,
            AdvancedPatternType.DOUBLE_BOTTOM.value,
            AdvancedPatternType.TRIPLE_BOTTOM.value,
            AdvancedPatternType.DIAMOND_BOTTOM.value,
            AdvancedPatternType.CUP_WITH_HANDLE.value,
            AdvancedPatternType.UNIQUE_THREE_RIVER.value
        ]

        # 定义看跌形态
        bearish_patterns = [
            AdvancedPatternType.THREE_BLACK_CROWS.value,
            AdvancedPatternType.THREE_INSIDE_DOWN.value,
            AdvancedPatternType.THREE_OUTSIDE_DOWN.value,
            AdvancedPatternType.FALLING_THREE_METHODS.value,
            AdvancedPatternType.TOWER_TOP.value,
            AdvancedPatternType.HEAD_SHOULDERS_TOP.value,
            AdvancedPatternType.DOUBLE_TOP.value,
            AdvancedPatternType.TRIPLE_TOP.value,
            AdvancedPatternType.DIAMOND_TOP.value
        ]

        # 强形态权重
        strong_patterns = {
            AdvancedPatternType.THREE_WHITE_SOLDIERS.value: 0.9,
            AdvancedPatternType.THREE_BLACK_CROWS.value: -0.9,
            AdvancedPatternType.RISING_THREE_METHODS.value: 0.85,
            AdvancedPatternType.FALLING_THREE_METHODS.value: -0.85,
            AdvancedPatternType.HEAD_SHOULDERS_BOTTOM.value: 0.9,
            AdvancedPatternType.HEAD_SHOULDERS_TOP.value: -0.9,
            AdvancedPatternType.DOUBLE_BOTTOM.value: 0.8,
            AdvancedPatternType.DOUBLE_TOP.value: -0.8,
            AdvancedPatternType.KICKING.value: 0.85
        }

        # 生成买入信号
        for pattern in bullish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                buy_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = 0.7

        # 生成卖出信号
        for pattern in bearish_patterns:
            if pattern in self._result.columns:
                pattern_mask = self._result[pattern]
                sell_signal |= pattern_mask

                # 设置信号强度
                if pattern in strong_patterns:
                    signal_strength[pattern_mask] = strong_patterns[pattern]
                else:
                    signal_strength[pattern_mask] = -0.7

        # 处理特殊形态
        if AdvancedPatternType.BREAKAWAY.value in self._result.columns:
            breakaway_mask = self._result[AdvancedPatternType.BREAKAWAY.value]
            if breakaway_mask.any() and len(data) >= 5:
                # 简单趋势判断
                price_change_5d = data['close'].pct_change(5)

                # 在下降趋势后的脱离形态（看涨）
                bullish_breakaway = breakaway_mask & (price_change_5d < -0.05)
                buy_signal |= bullish_breakaway
                signal_strength[bullish_breakaway] = 0.8

                # 在上升趋势后的脱离形态（看跌）
                bearish_breakaway = breakaway_mask & (price_change_5d > 0.05)
                sell_signal |= bearish_breakaway
                signal_strength[bearish_breakaway] = -0.8

        # 标准化信号强度
        signal_strength = signal_strength.clip(-1, 1)

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "ADVANCEDCANDLESTICKPATTERNS"

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取形态信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态信息字典
        """
        # 默认形态信息映射
        pattern_info_map = {
            # 基础形态
            'bullish': {'name': '看涨形态', 'description': '指标显示看涨信号', 'type': 'BULLISH'},
            'bearish': {'name': '看跌形态', 'description': '指标显示看跌信号', 'type': 'BEARISH'},
            'neutral': {'name': '中性形态', 'description': '指标显示中性信号', 'type': 'NEUTRAL'},
            
            # 通用形态
            'strong_signal': {'name': '强信号', 'description': '强烈的技术信号', 'type': 'STRONG'},
            'weak_signal': {'name': '弱信号', 'description': '较弱的技术信号', 'type': 'WEAK'},
            'trend_up': {'name': '上升趋势', 'description': '价格呈上升趋势', 'type': 'BULLISH'},
            'trend_down': {'name': '下降趋势', 'description': '价格呈下降趋势', 'type': 'BEARISH'},
        }
        
        # 默认形态信息
        default_pattern = {
            'name': pattern_id.replace('_', ' ').title(),
            'description': f'{pattern_id}形态',
            'type': 'UNKNOWN'
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)
