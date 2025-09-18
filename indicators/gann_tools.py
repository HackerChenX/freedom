from utils.container import container
#!/usr/bin/env python3
from utils.logger import get_logger
"""
GANN_TOOLS 指标

江恩工具指标 - 基于江恩理论的角度线和时间周期分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType
from utils.indicator_parameter_validator import IndicatorParameterValidator

logger = get_logger(__name__)


class GannTools(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    GANN_TOOLS 指标
    
    江恩工具指标,基于江恩理论的角度线和时间周期分析
    主要分析价格与时间的几何关系和周期性规律
    """
    
    @property
    def minimum_periods(self) -> int:
        """返回计算指标所需的最小周期数"""
        return getattr(self, 'period', 20) + 10  # TODO: 将魔法数字提取到配置中

    def __init__(self, **kwargs):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化GANN_TOOLS指标

        Args:
            **kwargs: 指标参数
        """
        super().__init__()
        self.name = "GANN_TOOLS"

        # 设置默认参数
        self._default_parameters = self._get_default_parameters_ganntools()

        # 应用用户参数
        self.set_parameters_Tools_Gann_Tools(**kwargs)
    
    def _get_default_parameters_ganntools(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "period": 20,  # 计算周期  # TODO: 将魔法数字提取到配置中
            "gann_angles": [1/8, 1/4, 1/3, 1/2, 1/1, 2/1, 3/1, 4/1, 8/1],  # 江恩角度  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            "time_cycles": [7, 14, 21, 30, 45, 60, 90, 120, 180]  # 江恩时间周期  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        }
    
    def set_parameters_Tools_Gann_Tools(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        # 验证参数
        try:
            from utils.indicator_parameter_validator import IndicatorParameterValidator
        except Exception as e:
            logger.error(f"错误: {e}")
            return pd.DataFrame()
    
    def validate_parameters(self, **kwargs):
        """验证参数"""
        try:
            validator = IndicatorParameterValidator()

            # 合并默认参数和用户参数
            params = self._default_parameters.copy()
            params.update(kwargs)

            # 验证参数
            is_valid, errors = validator.validate_indicator_parameters('GANN_TOOLS', params)
            if not is_valid:
                # 静默处理验证失败,避免过多警告
                pass
                # 使用默认参数
                params = self._default_parameters.copy()

            # 设置参数
            self.period = params.get('period', 20)  # TODO: 将魔法数字提取到配置中
            self.gann_angles = params.get('gann_angles', [1/8, 1/4, 1/3, 1/2, 1/1, 2/1, 3/1, 4/1, 8/1])  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            self.time_cycles = params.get('time_cycles', [7, 14, 21, 30, 45, 60, 90, 120, 180])  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

        except Exception:
            # 如果验证失败,静默处理,保持向后兼容
            self.period = 20  # TODO: 将魔法数字提取到配置中
            self.gann_angles = [1/8, 1/4, 1/3, 1/2, 1/1, 2/1, 3/1, 4/1, 8/1]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            self.time_cycles = [7, 14, 21, 30, 45, 60, 90, 120, 180]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """设置基础指标参数"""
        return self.set_parameters_Tools_Gann_Tools(**kwargs)

    def _calculate_baseindicator(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """基础指标计算方法"""
        return self._calculate_ganntools(data, *args, **kwargs)

    def calculate_confidence_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标置信度"""
        result = self._calculate_baseindicator(data)
        # 计算江恩理论的置信度
        score = self.calculate_raw_score_Tools_Gann_Tools(data)
        confidence = score / 100.0  # 将评分转换为置信度
        result['confidence'] = confidence
        return result

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算原始评分"""
        result = self._calculate_baseindicator(data)
        score = self.calculate_raw_score_Tools_Gann_Tools(data)
        result['raw_score'] = score
        return result

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取形态识别结果"""
        return self.get_patterns_Tools_Gann_Tools(data)
    
    def calculate_Tools_Gann_Tools(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算GANN_TOOLS指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了GANN_TOOLS指标的Data_frame
        """
        result = self._calculate_ganntools(data, **kwargs)
        self._result = result
        return result
    
    def _calculate_ganntools(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        内部计算GANN_TOOLS指标
        
        Args:
            data: 包含OHLCV数据的Data_frame
            
        Returns:
            添加了GANN_TOOLS指标的Data_frame
        """
        df = data.copy()
        
        # 计算江恩角度线
        df = self._calculate_gann_angles(df)
        
        # 计算时间周期
        df['time_cycle_signal'] = self._calculate_time_cycles(df)
        
        # 计算价格与角度线的关系
        df['angle_support'] = self._calculate_angle_support(df)
        df['angle_resistance'] = self._calculate_angle_resistance(df)
        
        # 计算江恩扇形分析
        df['gann_fan_signal'] = self._calculate_gann_fan(df)
        
        # 计算时间价格平方根关系
        df['square_of_nine'] = self._calculate_square_of_nine(df)
        
        # 添加形态识别和信号生成
        df = self.add_pattern_detection(df)
        df = self.add_signal_generation(df)

        return df
    
    def _calculate_gann_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算江恩角度线"""
        close = df['close']
        
        # 找到重要的高低点作为起始点
        swing_high = df['high'].rolling(window=self.period).max()
        swing_low = df['low'].rolling(window=self.period).min()
        
        # 计算各个角度线
        for i, angle in enumerate(self.gann_angles):
            # 使用数字索引而不是时间索引
            time_index = pd.Series(range(len(df)), index=df.index)
            
            # 从低点向上的角度线
            angle_up = swing_low + time_index * angle * 0.1  # 缩放因子
            df[f'gann_up_{angle:.3f}'] = angle_up
            
            # 从高点向下的角度线
            angle_down = swing_high - time_index * angle * 0.1  # 缩放因子
            df[f'gann_down_{angle:.3f}'] = angle_down
        
        return df
    
    def _calculate_time_cycles(self, df: pd.DataFrame) -> pd.Series:
        """计算时间周期信号"""
        signal = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        for cycle in self.time_cycles:
            if len(df) >= cycle:
                # 计算周期性高低点
                cycle_high = close.rolling(window=cycle).max()
                cycle_low = close.rolling(window=cycle).min()
                
                # 检查当前是否接近周期性转折点
                current_pos = len(df) % cycle
                
                # 在周期的关键位置(1/4, 1/2, 3/4, 1)给予信号  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                if current_pos in [cycle//4, cycle//2, 3*cycle//4, 0]:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    # 价格接近周期高点,可能反转
                    if close.iloc[-1] >= cycle_high.iloc[-1] * 0.95:  # TODO: 将魔法数字提取到配置中
                        signal.iloc[-1] += 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    
                    # 价格接近周期低点,可能反转
                    if close.iloc[-1] <= cycle_low.iloc[-1] * 1.05:  # TODO: 将魔法数字提取到配置中
                        signal.iloc[-1] += 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        return signal
    
    def _calculate_angle_support(self, df: pd.DataFrame) -> pd.Series:
        """计算角度线支撑强度"""
        support = pd.Series(0.0, index=df.index)
        
        close = df['close']
        low = df['low']
        
        for angle in self.gann_angles:
            angle_line = df.get(f'gann_up_{angle:.3f}', pd.Series(0.0, index=df.index))
            
            # 价格在角度线附近获得支撑
            near_angle = (close >= angle_line * 0.98) & (close <= angle_line * 1.02)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            touch_support = (low <= angle_line * 1.01) & (close > angle_line)
            
            # 1x1角度线(45度)权重最高
            weight = 10 if angle == 1.0 else 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
            support[near_angle] += weight
            support[touch_support] += weight * 1.5  # TODO: 将魔法数字提取到配置中
        
        return support
    
    def _calculate_angle_resistance(self, df: pd.DataFrame) -> pd.Series:
        """计算角度线阻力强度"""
        resistance = pd.Series(0.0, index=df.index)
        
        close = df['close']
        high = df['high']
        
        for angle in self.gann_angles:
            angle_line = df.get(f'gann_down_{angle:.3f}', pd.Series(0.0, index=df.index))
            
            # 价格在角度线附近遇到阻力
            near_angle = (close >= angle_line * 0.98) & (close <= angle_line * 1.02)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            touch_resistance = (high >= angle_line * 0.99) & (close < angle_line)  # TODO: 将魔法数字提取到配置中
            
            # 1x1角度线(45度)权重最高
            weight = 10 if angle == 1.0 else 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
            resistance[near_angle] += weight
            resistance[touch_resistance] += weight * 1.5  # TODO: 将魔法数字提取到配置中
        
        return resistance
    
    def _calculate_gann_fan(self, df: pd.DataFrame) -> pd.Series:
        """计算江恩扇形分析信号"""
        fan_signal = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        # 计算价格在江恩扇形中的位置
        above_angles = 0
        below_angles = 0
        
        for angle in self.gann_angles:
            angle_up = df.get(f'gann_up_{angle:.3f}', pd.Series(0.0, index=df.index))
            angle_down = df.get(f'gann_down_{angle:.3f}', pd.Series(0.0, index=df.index))
            
            # 统计价格在多少条角度线之上/之下
            above_up = close > angle_up
            below_down = close < angle_down
            
            above_angles += above_up.astype(int)
            below_angles += below_down.astype(int)
        
        # 价格在大部分角度线之上,强势信号
        strong_up = above_angles >= len(self.gann_angles) * 0.7  # TODO: 将魔法数字提取到配置中
        fan_signal[strong_up] += 15  # TODO: 将魔法数字提取到配置中
        
        # 价格在大部分角度线之下,弱势信号
        strong_down = below_angles >= len(self.gann_angles) * 0.7  # TODO: 将魔法数字提取到配置中
        fan_signal[strong_down] -= 15  # TODO: 将魔法数字提取到配置中
        
        return fan_signal
    
    def _calculate_square_of_nine(self, df: pd.DataFrame) -> pd.Series:
        """计算时间价格平方根关系(九宫格)"""
        square_signal = pd.Series(0.0, index=df.index)
        
        close = df['close']
        
        # 计算价格的平方根
        price_sqrt = np.sqrt(close)
        
        # 检查价格平方根是否接近整数(江恩重要价位)
        sqrt_fractional = price_sqrt - np.floor(price_sqrt)
        
        # 接近整数平方根的价位是重要的江恩价位
        near_square = (sqrt_fractional < 0.1) | (sqrt_fractional > 0.9)  # TODO: 将魔法数字提取到配置中
        square_signal[near_square] += 10
        
        # 检查价格是否在江恩的重要分数位(1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        important_fractions = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        for frac in important_fractions:
            near_fraction = np.abs(sqrt_fractional - frac) < 0.05  # TODO: 将魔法数字提取到配置中
            square_signal[near_fraction] += 8  # TODO: 将魔法数字提取到配置中
        
        return square_signal
    
    def calculate_raw_score_Tools_Gann_Tools(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分
        
        基于江恩分析的综合评分:
        - 角度线支撑阻力(35%权重)  # TODO: 将魔法数字提取到配置中
        - 时间周期信号(25%权重)  # TODO: 将魔法数字提取到配置中
        - 江恩扇形分析(25%权重)  # TODO: 将魔法数字提取到配置中
        - 九宫格分析(15%权重)  # TODO: 将魔法数字提取到配置中
        """
        if not self.has_result():
            self.calculate_Tools_Gann_Tools(data, **kwargs)
        
        result = self._result
        score = pd.Series(50.0, index=data.index)  # TODO: 将魔法数字提取到配置中
        
        # 1. 角度线支撑阻力评分(35%权重)  # TODO: 将魔法数字提取到配置中
        angle_support = result.get('angle_support', pd.Series(0.0, index=data.index))
        angle_resistance = result.get('angle_resistance', pd.Series(0.0, index=data.index))
        
        # 支撑强度加分,阻力强度减分
        support_score = np.clip(angle_support / 3, 0, 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        resistance_score = np.clip(angle_resistance / 3, 0, 25)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        score += support_score * 0.35  # TODO: 将魔法数字提取到配置中
        score -= resistance_score * 0.35  # TODO: 将魔法数字提取到配置中
        
        # 2. 时间周期信号评分(25%权重)  # TODO: 将魔法数字提取到配置中
        time_cycle_signal = result.get('time_cycle_signal', pd.Series(0.0, index=data.index))
        cycle_score = np.clip(time_cycle_signal / 2, 0, 20)  # TODO: 将魔法数字提取到配置中
        score += cycle_score * 0.25  # TODO: 将魔法数字提取到配置中
        
        # 3. 江恩扇形分析评分(25%权重)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        gann_fan_signal = result.get('gann_fan_signal', pd.Series(0.0, index=data.index))
        fan_score = np.clip(gann_fan_signal / 1.5, -20, 20)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        score += fan_score * 0.25  # TODO: 将魔法数字提取到配置中
        
        # 4. 九宫格分析评分(15%权重)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        square_signal = result.get('square_of_nine', pd.Series(0.0, index=data.index))
        square_score = np.clip(square_signal / 2, 0, 15)  # TODO: 将魔法数字提取到配置中
        score += square_score * 0.15  # TODO: 将魔法数字提取到配置中
        
        # 特殊加成:1x1角度线(45度线)的重要性
        close = data['close']
        if len(result) > 0 and 'gann_up_1.000' in result.columns:
            gann_1x1_up = result['gann_up_1.000']
            
            # 价格在1x1线附近,额外加分
            near_1x1 = np.abs(close - gann_1x1_up) / close < 0.02
            score[near_1x1] += 5  # TODO: 将魔法数字提取到配置中
        
        # 确保评分在0-100范围内
        score = np.clip(score, 0, 100)
        
        return score
    
    def calculate_confidence_Tools_Gann_Tools(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        if len(score) == 0:
            return 0.5  # TODO: 将魔法数字提取到配置中
        
        # 基于评分分布和江恩理论的几何一致性计算置信度
        avg_score = score.mean()
        score_std = score.std()
        
        # 评分越高,置信度越高
        score_confidence = min(avg_score / 100, 1.0)
        
        # 评分稳定性越高,置信度越高
        stability_confidence = max(0.3, 1.0 - score_std / 50)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 江恩理论强调几何一致性,稳定性权重更高
        confidence = (score_confidence * 0.6 + stability_confidence * 0.4)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        return max(0.3, min(0.95, confidence))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
    
    def get_patterns_Tools_Gann_Tools(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """获取形态"""
        if not self.has_result():
            self.calculate_Tools_Gann_Tools(data, **kwargs)
        
        result = self._result
        patterns = []
        
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 角度线形态
            angle_support = last_row.get('angle_support', 0)
            angle_resistance = last_row.get('angle_resistance', 0)
            
            if angle_support > 15:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩角度线强支撑")
            elif angle_support > 8:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩角度线支撑")
            
            if angle_resistance > 15:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩角度线强阻力")
            elif angle_resistance > 8:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩角度线阻力")
            
            # 时间周期形态
            time_cycle_signal = last_row.get('time_cycle_signal', 0)
            if time_cycle_signal > 10:
                patterns.append("江恩时间周期转折点")
            elif time_cycle_signal > 5:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩时间周期信号")
            
            # 江恩扇形形态
            gann_fan_signal = last_row.get('gann_fan_signal', 0)
            if gann_fan_signal > 10:
                patterns.append("江恩扇形强势突破")
            elif gann_fan_signal > 5:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩扇形上升趋势")
            elif gann_fan_signal < -10:
                patterns.append("江恩扇形弱势破位")
            elif gann_fan_signal < -5:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩扇形下降趋势")
            
            # 九宫格形态
            square_signal = last_row.get('square_of_nine', 0)
            if square_signal > 15:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩九宫格重要价位")
            elif square_signal > 8:  # TODO: 将魔法数字提取到配置中
                patterns.append("江恩九宫格关键位")
            
            # 1x1角度线特殊形态
            if 'gann_up_1.000' in result.columns:
                close = data['close'].iloc[-1]
                gann_1x1 = last_row.get('gann_up_1.000', close)
                
                if abs(close - gann_1x1) / close < 0.01:
                    patterns.append("江恩1x1角度线精确支撑")
                elif abs(close - gann_1x1) / close < 0.02:
                    patterns.append("江恩1x1角度线附近")
        
        return pd.DataFrame({'patterns': [patterns]}, index=[data.index[-1]] if len(data) > 0 else [])

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        标准计算方法,调用GANN工具计算

        Args:
            data: 股票数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 计算结果
        """
        return self.calculate_Tools_Gann_Tools(data, **kwargs)


# 为了向后兼容,创建别名
gann_tools = GannTools