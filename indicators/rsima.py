#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSI均线系统指标(RSIMA)

使用RSI的移动平均线系统来确认RSI趋势，增强趋势判断
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from utils.container import container
from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.common import crossover, crossunder
from utils.logger import get_logger
from utils.decorators import performance_monitor, exception_handler

logger = get_logger(__name__)


class Rsima(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    RSI均线系统(RSIMA)
    
    分类：趋势类指标
    描述：计算RSI的移动平均线系统，用于确认RSI趋势
    """
    
    def __init__(self, rsi_period: int = 14, ma_periods: List[int] = None, **kwargs):
        """
        初始化RSI均线系统(RSIMA)指标
        
        Args:
            rsi_period: RSI计算周期，默认为14
            ma_periods: RSI均线周期列表，默认为[3, 5, 10]
            **kwargs: 其他参数
        """
        super().__init__(name="RSIMA", **kwargs)
        
        # 依赖注入
        self.data_access = container.resolve("DataAccessInterface")
        self.cache_service = container.resolve("ICacheService")
        
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.rsi_period = rsi_period
        # 使用较小的默认周期，以便在数据量较少时也能计算
        self.ma_periods = ma_periods if ma_periods is not None else [3, 5, 10]
        self.description = "RSI均线系统指标,用于确认RSI趋势"
    
    def set_parameters_Rsima_Rsima_Rsima_rsima(self, rsi_period: int = None, ma_periods: List[int] = None):
        """
        设置指标参数
        """
        if rsi_period is not None:
            self.rsi_period = rsi_period
        if ma_periods is not None:
            self.ma_periods = ma_periods
    
    def get_patterns_Rsima(self):
        patterns = {
            "description": "RSI线下穿其移动平均线，可能预示下跌趋势。",
        }
        return patterns

    def register_patterns_Rsima(self):
        """
        注册RSIMA指标的形态到全局形态注册表
        """
        # 注册RSI均线交叉形态
        self.register_pattern_to_registry(
            pattern_id="RSI_MA_GOLDEN_CROSS",
            # display_name="RSI均线金叉",
            # description="RSI短期均线上穿长期均线，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="RSI_MA_DEATH_CROSS",
            # display_name="RSI均线死叉",
            # description="RSI短期均线下穿长期均线，看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册RSI中轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="RSI_CROSS_50_UP",
            # display_name="RSI上穿50",
            # description="RSI上穿50中轴，表明多头力量增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="RSI_CROSS_50_DOWN",
            # display_name="RSI下穿50",
            # description="RSI下穿50中轴，表明空头力量增强",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册RSI超买超卖形态
        self.register_pattern_to_registry(
            pattern_id="RSI_OVERBOUGHT",
            # display_name="RSI超买",
            # description="RSI进入超买区域(>70)，可能回调",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-12.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="RSI_OVERSOLD",
            # display_name="RSI超卖",
            # description="RSI进入超卖区域(<30)，可能反弹",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=12.0,
            polarity="POSITIVE"
        )

        # 注册RSI均线趋势形态
        self.register_pattern_to_registry(
            pattern_id="RSI_MA_UPTREND",
            # display_name="RSI均线上升趋势",
            # description="RSI均线呈上升趋势，多头占优",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="RSI_MA_DOWNTREND",
            # display_name="RSI均线下降趋势",
            # description="RSI均线呈下降趋势，空头占优",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        # 注册RSI背离形态
        self.register_pattern_to_registry(
            pattern_id="RSI_BULLISH_DIVERGENCE",
            # display_name="RSI看涨背离",
            # description="价格创新低但RSI未创新低，看涨背离",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="RSI_BEARISH_DIVERGENCE",
            # display_name="RSI看跌背离",
            # description="价格创新高但RSI未创新高，看跌背离",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

    def _validate_dataframe_rsima(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证Data_frame是否包含所需的列
        
        Args:
            df: 包含价格数据的Data_frame
            required_columns: 所需的列名列表
        
        Raises:
            ValueError: 如果Data_frame不包含所需的列，或者行数少于所需的最小行数
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
        # 数据行数至少要能计算RSI值
        min_rows = self.rsi_period + 1
        if len(df) < min_rows:
            raise ValueError(f"DataFrame至少需要 {min_rows} 行数据才能计算RSI，但只有 {len(df)} 行")
    
    def _calculateRsima(self, df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        """
        计算RSI均线系统
        
        Args:
            df: 包含价格数据的Data_frame
            price_column: 用于计算的价格列名，默认为'close'
        
        Returns:
            包含RSI均线系统结果的Data_frame
        """
        required_columns = [price_column]
        self._validate_dataframe_rsima(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算价格变化
        delta = df_copy[price_column].diff()
        
        # 计算上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # 计算平均上涨和平均下跌
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # 计算相对强度(RS)
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        df_copy['rsima_rsi'] = rsi
        
        # 计算可用的均线周期
        available_periods = []
        for period in self.ma_periods:
            # 如果数据行数足够计算该周期的均线，则添加到可用周期列表
            if len(df_copy) >= period + self.rsi_period:
                available_periods.append(period)
                df_copy[f'rsima_ma_{period}'] = df_copy['rsima_rsi'].rolling(window=period).mean()
        
        # 如果没有可用的均线周期，至少计算一个3日均线
        if not available_periods and len(df_copy) >= self.rsi_period + 3:
            df_copy['rsima_ma_3'] = df_copy['rsima_rsi'].rolling(window=3).mean()
            available_periods.append(3)
        
        # 记录可用的均线周期，供信号生成时使用
        self._available_periods = available_periods
        
        
        # 添加形态识别和信号生成
        df_copy = self.add_pattern_detection(df_copy)
        df_copy = self.add_signal_generation(df_copy)

        return df_copy
    
    def compute_Rsima(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI均线系统指标
        
        Args:
            df: 包含OHLCV数据的Data_frame
                
        Returns:
            包含RSI均线系统指标的Data_frame
        """
        try:
            result = self.calculate(df)
            result = self.get_signals_Rsima(result)
            return result
        except Exception as e:
            logger.error(f"计算指标 {self.name} 时出错: {str(e)}")
            raise
    
    def get_signals_Rsima(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成RSI均线系统指标交易信号
        
        Args:
            df: 包含价格数据和RSIMA指标的Data_frame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的Data_frame:
            # - rsima_buy_signal: 1=买入信号, 0=无信号
            # - rsima_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['rsima_rsi']
        self._validate_dataframe_rsima(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['rsima_buy_signal'] = 0
        df_copy['rsima_sell_signal'] = 0
        
        # 使用可用的均线周期生成信号
        available_periods = getattr(self, '_available_periods', [])
        
        # 如果有多个均线，使用短期均线上穿/下穿长期均线作为信号
        if len(available_periods) >= 2:
            # 按周期排序
            periods = sorted(available_periods)
            short_period = periods[0]
            long_period = periods[-1]
            
            # RSI短期均线上穿长期均线买入
            for i in range(1, len(df_copy)):
                if df_copy[f'rsima_ma_{short_period}'].iloc[i-1] < df_copy[f'rsima_ma_{long_period}'].iloc[i-1] and \
                   df_copy[f'rsima_ma_{short_period}'].iloc[i] > df_copy[f'rsima_ma_{long_period}'].iloc[i]:
                    df_copy.iloc[i, df_copy.columns.get_loc('rsima_buy_signal')] = 1
                
                # RSI短期均线下穿长期均线卖出
                elif df_copy[f'rsima_ma_{short_period}'].iloc[i-1] > df_copy[f'rsima_ma_{long_period}'].iloc[i-1] and \
                     df_copy[f'rsima_ma_{short_period}'].iloc[i] < df_copy[f'rsima_ma_{long_period}'].iloc[i]:
                    df_copy.iloc[i, df_copy.columns.get_loc('rsima_sell_signal')] = 1
        
        # RSI上穿50买入
        for i in range(1, len(df_copy)):
            if df_copy['rsima_rsi'].iloc[i-1] < 50 and df_copy['rsima_rsi'].iloc[i] > 50:
                df_copy.iloc[i, df_copy.columns.get_loc('rsima_buy_signal')] = 1
            
            # RSI下穿50卖出
            elif df_copy['rsima_rsi'].iloc[i-1] > 50 and df_copy['rsima_rsi'].iloc[i] < 50:
                df_copy.iloc[i, df_copy.columns.get_loc('rsima_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score_Rsima(self, data: pd.DataFrame) -> pd.Series:
        """
        计算RSIMA原始评分
        """
        if self._result is None:
            self.calculate(data)
        
        # 评分逻辑...
        score = pd.Series(50.0, index=data.index)
        return score

    def plot_Rsima(self, df: pd.DataFrame, ax=None, **kwargs):
        """
        绘制RSI均线系统指标图表
        
        Args:
            df: 包含RSIMA指标的Data_frame
            ax: matplotlib轴对象，如果为None则创建新的
            **kwargs: 额外绘图参数
            
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        # 检查必要的指标列是否存在
        required_columns = ['rsima_rsi']
        self._validate_dataframe_rsima(df, required_columns)
        
        # 创建新的轴对象（如果未提供）
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        # 绘制RSI线
        ax.plot(df.index, df['rsima_rsi'], label=f'RSI({self.rsi_period})')
        
        # 绘制RSI均线
        available_periods = getattr(self, '_available_periods', [])
        for period in available_periods:
            if f'rsima_ma_{period}' in df.columns:
                ax.plot(df.index, df[f'rsima_ma_{period}'], label=f'RSI MA{period}', linestyle='--')
        
        # 添加参考线
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=50, color='k', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('RSI均线系统(RSIMA)')
        ax.set_ylim([0, 100])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax 

    def get_pattern_info_Rsima(self, pattern_id: str) -> dict:
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

    def _get_default_parameters_rsima(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {}
    

    def calculate_confidence_Rsima(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """计算置信度"""
        return 0.5

    @property
    def minimum_periods(self) -> int:
        """
        Rsima指标所需的最少数据周期数

        计算逻辑：基于参数 rsi_period(14) 计算

        Returns:
            int: 最少需要的数据周期数
        """
        # 确保_parameters存在，如果不存在则使用实例属性或默认值
        if hasattr(self, '_parameters') and self._parameters:
            rsi_period = self._parameters.get('rsi_period', 14)
        else:
            rsi_period = getattr(self, 'rsi_period', 14)

        return rsi_period + max(10, rsi_period // 2)

    @performance_monitor(threshold=2.0)
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算RSIMA指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含RSIMA指标的DataFrame
        """
        return self._calculateRsima(data)
    
    @performance_monitor(threshold=1.0)
    @exception_handler(reraise=False, default_return=None)
    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        获取RSIMA指标信号

        Args:
            data: 包含价格数据的DataFrame
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 包含signal, score, confidence的字典
        """
        try:
            # 计算RSIMA指标
            result = self.calculate(data, **kwargs)
            
            if result.empty or len(result) == 0:
                return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}
            
            # 获取最新的RSIMA值
            latest = result.iloc[-1]
            rsima_rsi = latest.get('rsima_rsi', 50.0)
            
            # 初始化信号
            signal = "HOLD"
            score = 50.0
            confidence = 0.5
            
            # RSIMA信号逻辑 - 基于RSI值和均线交叉
            # 1. RSI超买超卖判断
            if rsima_rsi > 70:
                signal = "SELL"  # RSI超买，卖出信号
                score = min(80.0, 50.0 + (rsima_rsi - 70) * 1.5)
                confidence = 0.7
            elif rsima_rsi < 30:
                signal = "BUY"   # RSI超卖，买入信号
                score = max(20.0, 50.0 - (30 - rsima_rsi) * 1.5)
                confidence = 0.7
            elif rsima_rsi > 50:
                signal = "BUY"   # RSI在50以上，偏多头
                score = 50.0 + (rsima_rsi - 50) * 0.5
                confidence = 0.6
            elif rsima_rsi < 50:
                signal = "SELL"  # RSI在50以下，偏空头
                score = 50.0 - (50 - rsima_rsi) * 0.5
                confidence = 0.6
            
            # 2. 检查RSI均线交叉信号增强置信度
            available_periods = getattr(self, '_available_periods', [])
            if len(available_periods) >= 2 and len(result) >= 2:
                periods = sorted(available_periods)
                short_period = periods[0]
                long_period = periods[-1]
                
                short_ma_col = f'rsima_ma_{short_period}'
                long_ma_col = f'rsima_ma_{long_period}'
                
                if short_ma_col in result.columns and long_ma_col in result.columns:
                    current_short_ma = latest.get(short_ma_col, 50.0)
                    current_long_ma = latest.get(long_ma_col, 50.0)
                    prev_short_ma = result.iloc[-2].get(short_ma_col, 50.0) if len(result) >= 2 else current_short_ma
                    prev_long_ma = result.iloc[-2].get(long_ma_col, 50.0) if len(result) >= 2 else current_long_ma
                    
                    # 金叉：短期均线上穿长期均线
                    if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
                        signal = "BUY"
                        score = min(85.0, score + 20)
                        confidence = min(0.9, confidence + 0.2)
                    # 死叉：短期均线下穿长期均线
                    elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
                        signal = "SELL"
                        score = max(15.0, score - 20)
                        confidence = min(0.9, confidence + 0.2)
            
            # 3. 检查RSI穿越50中轴信号
            if len(result) >= 2:
                prev_rsi = result.iloc[-2].get('rsima_rsi', 50.0) if len(result) >= 2 else rsima_rsi
                
                # RSI上穿50
                if prev_rsi <= 50 and rsima_rsi > 50:
                    signal = "BUY"
                    score = min(75.0, score + 10)
                    confidence = min(0.8, confidence + 0.1)
                # RSI下穿50
                elif prev_rsi >= 50 and rsima_rsi < 50:
                    signal = "SELL"
                    score = max(25.0, score - 10)
                    confidence = min(0.8, confidence + 0.1)
            
            return {
                'signal': signal,
                'score': float(score),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.warning(f"RSIMA信号获取失败: {e}")
            return {'signal': 'HOLD', 'score': 50.0, 'confidence': 0.5}

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含RSIMA指标的DataFrame
        """
        return self._calculateRsima(data)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        BaseIndicator要求的置信度计算方法

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        return self.calculate_confidence_Rsima(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列
        """
        return self.calculate_raw_score_Rsima(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态DataFrame
        """
        return self.get_patterns_Rsima(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Rsima_Rsima_Rsima_rsima(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        BaseIndicator要求的默认参数获取方法

        Returns:
            dict: 默认参数字典
        """
        return self._get_default_parameters_rsima()

    def set_parameters(self, **kwargs):
        """
        标准参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Rsima_Rsima_Rsima_rsima(**kwargs)