#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一目均衡表指标(Ichimoku)

一目均衡表是一个综合性的技术分析指标，包含转换线、基准线、先行带A、先行带B和滞后线
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import crossover, crossunder
from utils.logger import get_logger

logger = get_logger(__name__)


class Ichimoku(BaseIndicator):
    """
    一目均衡表指标(Ichimoku) (Ichimoku)
    
    分类：趋势类指标
    描述：一目均衡表是一个综合性的技术分析指标，包含转换线、基准线、先行带A、先行带B和滞后线
    """
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52, chikou_period: int = 26):
        self.REQUIRED_COLUMNS = ['high', 'low', 'close']
        """
        初始化一目均衡表指标(Ichimoku)
        
        Args:
            tenkan_period: 转换线周期，默认为9
            kijun_period: 基准线周期，默认为26
            senkou_b_period: 先行带B周期，默认为52
            chikou_period: 滞后线周期，默认为26
        """
        super().__init__(name="Ichimoku", description="一目均衡表指标，综合性的技术分析指标")
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.chikou_period = chikou_period

    def set_parameters(self, tenkan_period: int = None, kijun_period: int = None,
                      senkou_b_period: int = None, chikou_period: int = None):
        """
        设置指标参数

        Args:
            tenkan_period: 转换线周期
            kijun_period: 基准线周期
            senkou_b_period: 先行带B周期
            chikou_period: 滞后线周期
        """
        if tenkan_period is not None:
            self.tenkan_period = tenkan_period
        if kijun_period is not None:
            self.kijun_period = kijun_period
        if senkou_b_period is not None:
            self.senkou_b_period = senkou_b_period
        if chikou_period is not None:
            self.chikou_period = chikou_period
        
        # 注册Ichimoku形态
        # self._register_ichimoku_patterns()
        
    def _register_ichimoku_patterns(self):
        """
        注册一目均衡表指标形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType, PatternStrength
        
        # 获取PatternRegistry实例
        registry = PatternRegistry()
        
        # 转换线和基准线相关形态
        registry.register(
            pattern_id="ICHIMOKU_TK_CROSS_BULLISH",
            display_name="一目均衡表金叉",
            description="转换线(Tenkan-sen)上穿基准线(Kijun-sen)，看涨信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=15.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_TK_CROSS_BEARISH",
            display_name="一目均衡表死叉",
            description="转换线(Tenkan-sen)下穿基准线(Kijun-sen)，看跌信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-15.0
        )
        
        # 价格与云图关系形态
        registry.register(
            pattern_id="ICHIMOKU_PRICE_ABOVE_KUMO",
            display_name="价格位于云层之上",
            description="价格位于云层上方，看涨信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=12.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_PRICE_BELOW_KUMO",
            display_name="价格位于云层之下",
            description="价格位于云层下方，看跌信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-12.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_PRICE_IN_KUMO",
            display_name="价格位于云层之中",
            description="价格位于云层中，表明市场处于盘整状态",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.WEAK,
            score_impact=0.0
        )
        
        # 价格突破云层形态
        registry.register(
            pattern_id="ICHIMOKU_PRICE_BREAK_KUMO_UP",
            display_name="价格向上突破云层",
            description="价格从下方突破云层，强烈看涨信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=20.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_PRICE_BREAK_KUMO_DOWN",
            display_name="价格向下突破云层",
            description="价格从上方突破云层，强烈看跌信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-20.0
        )
        
        # 滞后线相关形态
        registry.register(
            pattern_id="ICHIMOKU_CHIKOU_ABOVE_PRICE",
            display_name="滞后线位于价格之上",
            description="滞后线(Chikou Span)位于价格上方，看涨信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=10.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_CHIKOU_BELOW_PRICE",
            display_name="滞后线位于价格之下",
            description="滞后线(Chikou Span)位于价格下方，看跌信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.MEDIUM,
            score_impact=-10.0
        )
        
        # 云层形态
        registry.register(
            pattern_id="ICHIMOKU_KUMO_TWIST_BULLISH",
            display_name="云层看涨翻转",
            description="先行带A(Senkou Span A)上穿先行带B(Senkou Span B)，云层由红变绿，看涨信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.STRONG,
            score_impact=18.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_KUMO_TWIST_BEARISH",
            display_name="云层看跌翻转",
            description="先行带A(Senkou Span A)下穿先行带B(Senkou Span B)，云层由绿变红，看跌信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.STRONG,
            score_impact=-18.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_KUMO_THICK",
            display_name="云层加厚",
            description="云层厚度增加，表明趋势强度增强",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.MEDIUM,
            score_impact=5.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_KUMO_THIN",
            display_name="云层变薄",
            description="云层厚度减少，表明趋势强度减弱",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.NEUTRAL,
            default_strength=PatternStrength.WEAK,
            score_impact=-5.0
        )
        
        # 组合形态
        registry.register(
            pattern_id="ICHIMOKU_STRONG_BULLISH",
            display_name="一目均衡表强烈看涨",
            description="价格位于云层上方，转换线上穿基准线，滞后线位于价格上方，三重看涨信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BULLISH,
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=25.0
        )
        
        registry.register(
            pattern_id="ICHIMOKU_STRONG_BEARISH",
            display_name="一目均衡表强烈看跌",
            description="价格位于云层下方，转换线下穿基准线，滞后线位于价格下方，三重看跌信号",
            indicator_id="ICHIMOKU",
            pattern_type=PatternType.BEARISH,
            default_strength=PatternStrength.VERY_STRONG,
            score_impact=-25.0
        )
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 输入数据
            required_columns: 所需的列名列表
            
        Raises:
            ValueError: 如果缺少必要的列
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {', '.join(missing_columns)}")
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算Ichimoku指标
        
        Args:
            df: 包含OHLC数据的DataFrame
                
        Returns:
            包含Ichimoku指标的DataFrame
        """
        return self.calculate(df)
        
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算一目均衡表指标(Ichimoku)
        
        Args:
            df: 包含OHLC数据的DataFrame
                必须包含以下列：
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                
        Returns:
            添加了Ichimoku指标列的DataFrame
        """
        if df.empty:
            return df
            
        # 确保数据包含必要的列
        required_columns = ['high', 'low', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 计算转换线 (Tenkan-sen): (9日最高价 + 9日最低价) / 2
        tenkan_high = df_copy['high'].rolling(window=self.tenkan_period).max()
        tenkan_low = df_copy['low'].rolling(window=self.tenkan_period).min()
        df_copy['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # 计算基准线 (Kijun-sen): (26日最高价 + 26日最低价) / 2
        kijun_high = df_copy['high'].rolling(window=self.kijun_period).max()
        kijun_low = df_copy['low'].rolling(window=self.kijun_period).min()
        df_copy['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # 计算先行带A (Senkou Span A): (转换线 + 基准线) / 2，向前移动26期
        senkou_a = (df_copy['tenkan_sen'] + df_copy['kijun_sen']) / 2
        df_copy['senkou_span_a'] = senkou_a.shift(self.kijun_period)
        
        # 计算先行带B (Senkou Span B): (52日最高价 + 52日最低价) / 2，向前移动26期
        senkou_b_high = df_copy['high'].rolling(window=self.senkou_b_period).max()
        senkou_b_low = df_copy['low'].rolling(window=self.senkou_b_period).min()
        senkou_b = (senkou_b_high + senkou_b_low) / 2
        df_copy['senkou_span_b'] = senkou_b.shift(self.kijun_period)
        
        # 计算滞后线 (Chikou Span): 收盘价向后移动26期
        df_copy['chikou_span'] = df_copy['close'].shift(-self.chikou_period)
        
        # 计算云图上下边界
        df_copy['kumo_top'] = np.maximum(df_copy['senkou_span_a'], df_copy['senkou_span_b'])
        df_copy['kumo_bottom'] = np.minimum(df_copy['senkou_span_a'], df_copy['senkou_span_b'])
        
        # 计算云图厚度
        df_copy['kumo_thickness'] = df_copy['kumo_top'] - df_copy['kumo_bottom']
        
        # 存储结果
        self._result = df_copy[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                               'chikou_span', 'kumo_top', 'kumo_bottom', 'kumo_thickness']]
        
        return df_copy
        
    def get_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成一目均衡表指标(Ichimoku)交易信号
        
        Args:
            df: 包含价格数据和Ichimoku指标的DataFrame
            **kwargs: 额外参数
                
        Returns:
            添加了信号列的DataFrame:
            - ichimoku_buy_signal: 1=买入信号, 0=无信号
            - ichimoku_sell_signal: 1=卖出信号, 0=无信号
        """
        if df.empty:
            return df
            
        # 检查必要的指标列是否存在
        required_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'close']
        self._validate_dataframe(df, required_columns)
        
        df_copy = df.copy()
        
        # 初始化信号列
        df_copy['ichimoku_buy_signal'] = 0
        df_copy['ichimoku_sell_signal'] = 0
        
        # 生成交易信号
        for i in range(1, len(df_copy)):
            buy_signals = 0
            sell_signals = 0
            
            # 1. 转换线上穿基准线
            if (df_copy['tenkan_sen'].iloc[i-1] <= df_copy['kijun_sen'].iloc[i-1] and 
                df_copy['tenkan_sen'].iloc[i] > df_copy['kijun_sen'].iloc[i]):
                buy_signals += 1
            elif (df_copy['tenkan_sen'].iloc[i-1] >= df_copy['kijun_sen'].iloc[i-1] and 
                  df_copy['tenkan_sen'].iloc[i] < df_copy['kijun_sen'].iloc[i]):
                sell_signals += 1
            
            # 2. 价格突破云图
            kumo_top = df_copy['kumo_top'].iloc[i]
            kumo_bottom = df_copy['kumo_bottom'].iloc[i]
            current_price = df_copy['close'].iloc[i]
            prev_price = df_copy['close'].iloc[i-1]
            
            if not pd.isna(kumo_top) and not pd.isna(kumo_bottom):
                # 价格上穿云图上边界
                if prev_price <= kumo_top and current_price > kumo_top:
                    buy_signals += 1
                # 价格下穿云图下边界
                elif prev_price >= kumo_bottom and current_price < kumo_bottom:
                    sell_signals += 1
            
            # 3. 滞后线确认
            chikou_span = df_copy['chikou_span'].iloc[i]
            if not pd.isna(chikou_span):
                # 滞后线在价格上方
                if chikou_span > current_price:
                    buy_signals += 1
                # 滞后线在价格下方
                elif chikou_span < current_price:
                    sell_signals += 1
            
            # 设置信号
            if buy_signals >= 2:  # 至少2个买入条件
                df_copy.iloc[i, df_copy.columns.get_loc('ichimoku_buy_signal')] = 1
            elif sell_signals >= 2:  # 至少2个卖出条件
                df_copy.iloc[i, df_copy.columns.get_loc('ichimoku_sell_signal')] = 1
        
        return df_copy
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算Ichimoku原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算Ichimoku
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 云图评分
        kumo_score = self._calculate_kumo_score(data)
        score += kumo_score
        
        # 2. 转换线基准线评分
        tenkan_kijun_score = self._calculate_tenkan_kijun_score()
        score += tenkan_kijun_score
        
        # 3. 价格位置评分
        price_position_score = self._calculate_price_position_score(data)
        score += price_position_score
        
        # 4. 滞后线评分
        chikou_score = self._calculate_chikou_score(data)
        score += chikou_score
        
        # 5. 突破评分
        breakout_score = self._calculate_breakout_score(data)
        score += breakout_score
        
        return np.clip(score, 0, 100)

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Ichimoku指标的置信度

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

        # 2. 基于形态的置信度
        if isinstance(patterns, pd.DataFrame) and not patterns.empty:
            try:
                # 统计最近几个周期的形态数量
                numeric_cols = patterns.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    recent_data = patterns[numeric_cols].iloc[-5:] if len(patterns) >= 5 else patterns[numeric_cols]
                    recent_patterns = recent_data.sum().sum()
                    if recent_patterns > 0:
                        confidence += min(recent_patterns * 0.05, 0.2)
            except:
                pass

        # 3. 基于Ichimoku多重确认的置信度
        if hasattr(self, '_result') and self._result is not None:
            try:
                # 检查多重确认信号
                confirmations = 0

                # 转换线基准线关系
                if 'tenkan_sen' in self._result.columns and 'kijun_sen' in self._result.columns:
                    tenkan = self._result['tenkan_sen'].dropna()
                    kijun = self._result['kijun_sen'].dropna()
                    if len(tenkan) > 0 and len(kijun) > 0:
                        if tenkan.iloc[-1] > kijun.iloc[-1]:
                            confirmations += 1

                # 云图状态
                if 'senkou_span_a' in self._result.columns and 'senkou_span_b' in self._result.columns:
                    senkou_a = self._result['senkou_span_a'].dropna()
                    senkou_b = self._result['senkou_span_b'].dropna()
                    if len(senkou_a) > 0 and len(senkou_b) > 0:
                        if senkou_a.iloc[-1] > senkou_b.iloc[-1]:
                            confirmations += 1

                # 多重确认增加置信度
                if confirmations >= 2:
                    confidence += 0.15
                elif confirmations == 1:
                    confidence += 0.1

            except:
                pass

        # 4. 基于评分稳定性的置信度
        if len(score) >= 5:
            recent_scores = score.iloc[-5:]
            score_stability = 1.0 - (recent_scores.std() / 50.0)
            confidence += score_stability * 0.1

        return min(confidence, 1.0)

    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别Ichimoku技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算Ichimoku
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测云图形态
        kumo_patterns = self._detect_kumo_patterns()
        patterns.extend(kumo_patterns)
        
        # 2. 检测转换线基准线形态
        tenkan_kijun_patterns = self._detect_tenkan_kijun_patterns()
        patterns.extend(tenkan_kijun_patterns)
        
        # 3. 检测价格位置形态
        price_position_patterns = self._detect_price_position_patterns(data)
        patterns.extend(price_position_patterns)
        
        # 4. 检测滞后线形态
        chikou_patterns = self._detect_chikou_patterns(data)
        patterns.extend(chikou_patterns)
        
        # 5. 检测突破形态
        breakout_patterns = self._detect_breakout_patterns(data)
        patterns.extend(breakout_patterns)
        
        return patterns
    
    def _calculate_kumo_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算云图评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 云图评分
        """
        kumo_score = pd.Series(0.0, index=self._result.index)
        
        senkou_a = self._result['senkou_span_a']
        senkou_b = self._result['senkou_span_b']
        kumo_thickness = self._result['kumo_thickness']
        
        # 先行带A在先行带B上方（看涨云图）+15分
        bullish_kumo = senkou_a > senkou_b
        kumo_score += bullish_kumo * 15
        
        # 先行带A在先行带B下方（看跌云图）-15分
        bearish_kumo = senkou_a < senkou_b
        kumo_score -= bearish_kumo * 15
        
        # 云图厚度评分（厚度越大支撑阻力越强）
        if len(kumo_thickness) > 20:
            thickness_percentile = kumo_thickness.rolling(20).quantile(0.8)
            thick_kumo = kumo_thickness > thickness_percentile
            kumo_score += thick_kumo * bullish_kumo * 10  # 厚的看涨云图+10分
            kumo_score -= thick_kumo * bearish_kumo * 10  # 厚的看跌云图-10分
        
        return kumo_score
    
    def _calculate_tenkan_kijun_score(self) -> pd.Series:
        """
        计算转换线基准线评分
        
        Returns:
            pd.Series: 转换线基准线评分
        """
        tenkan_kijun_score = pd.Series(0.0, index=self._result.index)
        
        tenkan_sen = self._result['tenkan_sen']
        kijun_sen = self._result['kijun_sen']
        
        # 转换线上穿基准线+20分
        tenkan_cross_up_kijun = crossover(tenkan_sen, kijun_sen)
        tenkan_kijun_score += tenkan_cross_up_kijun * 20
        
        # 转换线下穿基准线-20分
        tenkan_cross_down_kijun = crossunder(tenkan_sen, kijun_sen)
        tenkan_kijun_score -= tenkan_cross_down_kijun * 20
        
        # 转换线在基准线上方+8分
        tenkan_above_kijun = tenkan_sen > kijun_sen
        tenkan_kijun_score += tenkan_above_kijun * 8
        
        # 转换线在基准线下方-8分
        tenkan_below_kijun = tenkan_sen < kijun_sen
        tenkan_kijun_score -= tenkan_below_kijun * 8
        
        return tenkan_kijun_score
    
    def _calculate_price_position_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算价格位置评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 价格位置评分
        """
        price_position_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return price_position_score
        
        close_price = data['close']
        kumo_top = self._result['kumo_top']
        kumo_bottom = self._result['kumo_bottom']
        tenkan_sen = self._result['tenkan_sen']
        kijun_sen = self._result['kijun_sen']
        
        # 价格在云图上方+15分
        price_above_kumo = close_price > kumo_top
        price_position_score += price_above_kumo * 15
        
        # 价格在云图下方-15分
        price_below_kumo = close_price < kumo_bottom
        price_position_score -= price_below_kumo * 15
        
        # 价格在云图内部（震荡）0分
        price_in_kumo = (close_price >= kumo_bottom) & (close_price <= kumo_top)
        # 云图内部不加分也不减分
        
        # 价格在转换线和基准线上方+10分
        price_above_lines = (close_price > tenkan_sen) & (close_price > kijun_sen)
        price_position_score += price_above_lines * 10
        
        # 价格在转换线和基准线下方-10分
        price_below_lines = (close_price < tenkan_sen) & (close_price < kijun_sen)
        price_position_score -= price_below_lines * 10
        
        return price_position_score
    
    def _calculate_chikou_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算滞后线评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 滞后线评分
        """
        chikou_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return chikou_score
        
        close_price = data['close']
        chikou_span = self._result['chikou_span']
        
        # 滞后线在当前价格上方+12分
        chikou_above_price = chikou_span > close_price
        chikou_score += chikou_above_price * 12
        
        # 滞后线在当前价格下方-12分
        chikou_below_price = chikou_span < close_price
        chikou_score -= chikou_below_price * 12
        
        # 滞后线穿越历史价格
        if len(close_price) > self.chikou_period:
            historical_price = close_price.shift(self.chikou_period)
            
            # 滞后线上穿历史价格+15分
            chikou_cross_up_historical = crossover(chikou_span, historical_price)
            chikou_score += chikou_cross_up_historical * 15
            
            # 滞后线下穿历史价格-15分
            chikou_cross_down_historical = crossunder(chikou_span, historical_price)
            chikou_score -= chikou_cross_down_historical * 15
        
        return chikou_score
    
    def _calculate_breakout_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算突破评分
        
        Args:
            data: 价格数据
            
        Returns:
            pd.Series: 突破评分
        """
        breakout_score = pd.Series(0.0, index=self._result.index)
        
        if 'close' not in data.columns:
            return breakout_score
        
        close_price = data['close']
        kumo_top = self._result['kumo_top']
        kumo_bottom = self._result['kumo_bottom']
        
        # 价格上穿云图上边界+25分
        price_breakout_kumo_top = crossover(close_price, kumo_top)
        breakout_score += price_breakout_kumo_top * 25
        
        # 价格下穿云图下边界-25分
        price_breakout_kumo_bottom = crossunder(close_price, kumo_bottom)
        breakout_score -= price_breakout_kumo_bottom * 25
        
        return breakout_score
    
    def _detect_kumo_patterns(self) -> List[str]:
        """
        检测云图形态
        
        Returns:
            List[str]: 云图形态列表
        """
        patterns = []
        
        senkou_a = self._result['senkou_span_a']
        senkou_b = self._result['senkou_span_b']
        kumo_thickness = self._result['kumo_thickness']
        
        if len(senkou_a) > 0 and len(senkou_b) > 0:
            current_a = senkou_a.iloc[-1]
            current_b = senkou_b.iloc[-1]
            
            if not pd.isna(current_a) and not pd.isna(current_b):
                if current_a > current_b:
                    patterns.append("看涨云图")
                elif current_a < current_b:
                    patterns.append("看跌云图")
                else:
                    patterns.append("云图交汇")
        
        # 检查云图厚度
        if len(kumo_thickness) >= 20:
            current_thickness = kumo_thickness.iloc[-1]
            avg_thickness = kumo_thickness.tail(20).mean()
            
            if not pd.isna(current_thickness) and not pd.isna(avg_thickness):
                if current_thickness > avg_thickness * 1.5:
                    patterns.append("厚云图")
                elif current_thickness < avg_thickness * 0.5:
                    patterns.append("薄云图")
        
        return patterns
    
    def _detect_tenkan_kijun_patterns(self) -> List[str]:
        """
        检测转换线基准线形态
        
        Returns:
            List[str]: 转换线基准线形态列表
        """
        patterns = []
        
        tenkan_sen = self._result['tenkan_sen']
        kijun_sen = self._result['kijun_sen']
        
        # 检查最近的交叉
        recent_periods = min(5, len(tenkan_sen))
        recent_tenkan = tenkan_sen.tail(recent_periods)
        recent_kijun = kijun_sen.tail(recent_periods)
        
        if crossover(recent_tenkan, recent_kijun).any():
            patterns.append("转换线上穿基准线")
        
        if crossunder(recent_tenkan, recent_kijun).any():
            patterns.append("转换线下穿基准线")
        
        # 检查当前位置关系
        if len(tenkan_sen) > 0 and len(kijun_sen) > 0:
            current_tenkan = tenkan_sen.iloc[-1]
            current_kijun = kijun_sen.iloc[-1]
            
            if not pd.isna(current_tenkan) and not pd.isna(current_kijun):
                if current_tenkan > current_kijun:
                    patterns.append("转换线基准线上方")
                elif current_tenkan < current_kijun:
                    patterns.append("转换线基准线下方")
                else:
                    patterns.append("转换线基准线重合")
        
        return patterns
    
    def _detect_price_position_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测价格位置形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 价格位置形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        kumo_top = self._result['kumo_top']
        kumo_bottom = self._result['kumo_bottom']
        
        if len(close_price) > 0:
            current_price = close_price.iloc[-1]
            current_kumo_top = kumo_top.iloc[-1]
            current_kumo_bottom = kumo_bottom.iloc[-1]
            
            if not pd.isna(current_price) and not pd.isna(current_kumo_top) and not pd.isna(current_kumo_bottom):
                if current_price > current_kumo_top:
                    patterns.append("价格云图上方")
                elif current_price < current_kumo_bottom:
                    patterns.append("价格云图下方")
                else:
                    patterns.append("价格云图内部")
        
        return patterns
    
    def _detect_chikou_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测滞后线形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 滞后线形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        chikou_span = self._result['chikou_span']
        
        if len(close_price) > 0 and len(chikou_span) > 0:
            current_price = close_price.iloc[-1]
            current_chikou = chikou_span.iloc[-1]
            
            if not pd.isna(current_price) and not pd.isna(current_chikou):
                if current_chikou > current_price:
                    patterns.append("滞后线价格上方")
                elif current_chikou < current_price:
                    patterns.append("滞后线价格下方")
                else:
                    patterns.append("滞后线价格重合")
        
        return patterns
    
    def _detect_breakout_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        检测突破形态
        
        Args:
            data: 价格数据
            
        Returns:
            List[str]: 突破形态列表
        """
        patterns = []
        
        if 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        kumo_top = self._result['kumo_top']
        kumo_bottom = self._result['kumo_bottom']
        
        # 检查最近的突破
        recent_periods = min(5, len(close_price))
        recent_price = close_price.tail(recent_periods)
        recent_kumo_top = kumo_top.tail(recent_periods)
        recent_kumo_bottom = kumo_bottom.tail(recent_periods)
        
        if crossover(recent_price, recent_kumo_top).any():
            patterns.append("价格突破云图上边界")
        
        if crossunder(recent_price, recent_kumo_bottom).any():
            patterns.append("价格突破云图下边界")
        
        return patterns
        
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Ichimoku指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算Ichimoku
        if not self.has_result():
            self.calculate(data, **kwargs)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        tenkan_sen = self._result['tenkan_sen']
        kijun_sen = self._result['kijun_sen']
        senkou_a = self._result['senkou_span_a']
        senkou_b = self._result['senkou_span_b']
        chikou_span = self._result['chikou_span']
        kumo_top = self._result['kumo_top']
        kumo_bottom = self._result['kumo_bottom']

        patterns_df = pd.DataFrame(index=data.index)

        # 1. 转换线基准线交叉形态
        patterns_df['ICHIMOKU_TK_GOLDEN_CROSS'] = crossover(tenkan_sen, kijun_sen)
        patterns_df['ICHIMOKU_TK_DEATH_CROSS'] = crossunder(tenkan_sen, kijun_sen)
        patterns_df['ICHIMOKU_TK_BULLISH'] = tenkan_sen > kijun_sen
        patterns_df['ICHIMOKU_TK_BEARISH'] = tenkan_sen < kijun_sen

        # 2. 价格与云图关系形态
        patterns_df['ICHIMOKU_PRICE_ABOVE_KUMO'] = data['close'] > kumo_top
        patterns_df['ICHIMOKU_PRICE_BELOW_KUMO'] = data['close'] < kumo_bottom
        patterns_df['ICHIMOKU_PRICE_IN_KUMO'] = (data['close'] >= kumo_bottom) & (data['close'] <= kumo_top)

        # 3. 价格突破云图形态
        patterns_df['ICHIMOKU_PRICE_BREAK_KUMO_UP'] = crossover(data['close'], kumo_top)
        patterns_df['ICHIMOKU_PRICE_BREAK_KUMO_DOWN'] = crossunder(data['close'], kumo_bottom)

        # 4. 云图形态
        patterns_df['ICHIMOKU_KUMO_BULLISH'] = senkou_a > senkou_b
        patterns_df['ICHIMOKU_KUMO_BEARISH'] = senkou_a < senkou_b
        patterns_df['ICHIMOKU_KUMO_TWIST_BULLISH'] = crossover(senkou_a, senkou_b)
        patterns_df['ICHIMOKU_KUMO_TWIST_BEARISH'] = crossunder(senkou_a, senkou_b)

        # 5. 滞后线形态
        if len(data) >= self.chikou_period:
            # 滞后线与价格比较（需要考虑时间偏移）
            price_shifted = data['close'].shift(self.chikou_period)
            patterns_df['ICHIMOKU_CHIKOU_ABOVE_PRICE'] = chikou_span > price_shifted
            patterns_df['ICHIMOKU_CHIKOU_BELOW_PRICE'] = chikou_span < price_shifted

        # 6. 云图厚度形态
        kumo_thickness = self._result['kumo_thickness']
        if len(kumo_thickness) >= 20:
            thickness_ma = kumo_thickness.rolling(20).mean()
            patterns_df['ICHIMOKU_KUMO_THICK'] = kumo_thickness > thickness_ma * 1.5
            patterns_df['ICHIMOKU_KUMO_THIN'] = kumo_thickness < thickness_ma * 0.5

        # 7. 综合形态
        patterns_df['ICHIMOKU_STRONG_BULLISH'] = (
            patterns_df['ICHIMOKU_PRICE_ABOVE_KUMO'] &
            patterns_df['ICHIMOKU_TK_BULLISH'] &
            patterns_df['ICHIMOKU_KUMO_BULLISH']
        )

        patterns_df['ICHIMOKU_STRONG_BEARISH'] = (
            patterns_df['ICHIMOKU_PRICE_BELOW_KUMO'] &
            patterns_df['ICHIMOKU_TK_BEARISH'] &
            patterns_df['ICHIMOKU_KUMO_BEARISH']
        )

        return patterns_df

    def register_patterns(self):
        """
        注册Ichimoku指标的技术形态
        """
        # 注册转换线基准线交叉形态
        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_TK_GOLDEN_CROSS",
            display_name="一目均衡表金叉",
            description="转换线上穿基准线，看涨信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0
        )

        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_TK_DEATH_CROSS",
            display_name="一目均衡表死叉",
            description="转换线下穿基准线，看跌信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0
        )

        # 注册价格与云图关系形态
        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_PRICE_ABOVE_KUMO",
            display_name="价格位于云层之上",
            description="价格位于云层上方，看涨信号",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0
        )

        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_PRICE_BELOW_KUMO",
            display_name="价格位于云层之下",
            description="价格位于云层下方，看跌信号",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0
        )

        # 注册价格突破云图形态
        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_PRICE_BREAK_KUMO_UP",
            display_name="价格向上突破云层",
            description="价格从下方突破云层，强烈看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0
        )

        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_PRICE_BREAK_KUMO_DOWN",
            display_name="价格向下突破云层",
            description="价格从上方突破云层，强烈看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0
        )

        # 注册云图翻转形态
        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_KUMO_TWIST_BULLISH",
            display_name="云层看涨翻转",
            description="先行带A上穿先行带B，云层由红变绿",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=18.0
        )

        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_KUMO_TWIST_BEARISH",
            display_name="云层看跌翻转",
            description="先行带A下穿先行带B，云层由绿变红",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-18.0
        )

        # 注册综合形态
        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_STRONG_BULLISH",
            display_name="一目均衡表强烈看涨",
            description="价格位于云层上方，转换线上穿基准线，云层看涨",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0
        )

        self.register_pattern_to_registry(
            pattern_id="ICHIMOKU_STRONG_BEARISH",
            display_name="一目均衡表强烈看跌",
            description="价格位于云层下方，转换线下穿基准线，云层看跌",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0
        )

    def calculate_score(self, data):
        """
        计算Ichimoku指标的综合评分

        Args:
            data: 输入数据

        Returns:
            dict: 包含评分和置信度的字典
        """
        # 计算原始评分
        raw_score = self.calculate_raw_score(data)

        # 获取形态
        patterns = self.get_patterns(data)

        # 计算置信度
        confidence = self.calculate_confidence(raw_score, patterns, {})

        # 计算最终评分
        final_score = raw_score.iloc[-1] if not raw_score.empty else 50.0

        return {
            'score': final_score,
            'confidence': confidence,
            'patterns': patterns,
            'raw_score': raw_score
        }