#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from indicators.base_indicator import BaseIndicator

class ZXMMarketBreadth(BaseIndicator):
    """
    ZXM市场宽度指标
    
    用于分析市场整体状况、板块轮动和市场强度的综合指标，
    通过计算涨跌家数比例、主要指数相对强度等方式评估市场健康度。
    """
    
    def __init__(self):
        """初始化ZXM市场宽度指标"""
        super().__init__()
        self.name = "ZXM市场宽度指标"
        self.description = "分析市场整体状况和板块轮动的综合指标"
        
        # 市场宽度评估指标权重
        self.breadth_weights = {
            'advance_decline': 0.20,      # 涨跌家数比率
            'new_highs_lows': 0.15,       # 新高新低比率
            'percentage_above_ma': 0.20,  # 站上均线比例
            'sector_strength': 0.15,      # 板块强度
            'volume_breadth': 0.15,       # 成交量宽度
            'momentum_breadth': 0.15      # 动量宽度
        }
        
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算市场宽度指标
        
        Args:
            data: DataFrame，包含市场数据，需要有多个股票的数据
                data必须是一个多层索引DataFrame，第一级是日期，第二级是股票代码
                必须包含的列：['open', 'high', 'low', 'close', 'volume']
            *args: 位置参数
            **kwargs: 关键字参数
                lookback_period: 回溯分析周期，默认60个交易日
                ma_periods: 均线周期列表，默认[20, 50, 200]
                index_code: 大盘指数代码，默认None
                
        Returns:
            DataFrame: 包含市场宽度指标的DataFrame
        """
        if data.empty:
            return pd.DataFrame()
            
        # 获取参数
        lookback_period = kwargs.get('lookback_period', 60)
        ma_periods = kwargs.get('ma_periods', [20, 50, 200])
        index_code = kwargs.get('index_code', None)
        
        # 检查数据结构
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("数据必须是多层索引DataFrame，第一级是日期，第二级是股票代码")
            
        # 获取唯一的日期列表
        dates = data.index.get_level_values(0).unique()
        
        # 初始化结果DataFrame
        result = pd.DataFrame(index=dates)
        
        # 1. 计算涨跌家数比率
        advance_decline = self._calculate_advance_decline_ratio(data)
        result['ad_ratio'] = advance_decline['ad_ratio']
        result['ad_line'] = advance_decline['ad_line']
        
        # 2. 计算新高新低比率
        highs_lows = self._calculate_new_highs_lows_ratio(data, lookback_period)
        result['new_highs_ratio'] = highs_lows['new_highs_ratio']
        result['new_lows_ratio'] = highs_lows['new_lows_ratio']
        result['hl_ratio'] = highs_lows['hl_ratio']
        
        # 3. 计算站上各均线的股票比例
        for period in ma_periods:
            above_ma = self._calculate_percentage_above_ma(data, period)
            result[f'above_ma{period}'] = above_ma
        
        # 4. 计算板块强度
        if 'sector' in data.columns or 'industry' in data.columns:
            sector_strength = self._calculate_sector_strength(data)
            result['strongest_sector'] = sector_strength['strongest_sector']
            result['weakest_sector'] = sector_strength['weakest_sector']
            result['sector_rotation'] = sector_strength['sector_rotation']
        
        # 5. 计算成交量宽度
        volume_breadth = self._calculate_volume_breadth(data)
        result['volume_surge_ratio'] = volume_breadth['volume_surge_ratio']
        result['volume_decline_ratio'] = volume_breadth['volume_decline_ratio']
        
        # 6. 计算动量宽度
        momentum_breadth = self._calculate_momentum_breadth(data)
        result['momentum_positive_ratio'] = momentum_breadth['positive_ratio']
        result['momentum_negative_ratio'] = momentum_breadth['negative_ratio']
        
        # 7. 如果有大盘指数，计算市场与指数的相对强度
        if index_code is not None and index_code in data.index.get_level_values(1):
            relative_strength = self._calculate_market_relative_strength(data, index_code)
            result['market_relative_strength'] = relative_strength
        
        # 8. 计算综合市场宽度指标 (0-100)
        result['market_breadth_indicator'] = self._calculate_breadth_indicator(result)
        
        # 9. 市场状态分类
        result['market_state'] = self._classify_market_state(result)
        
        return result
        
    def calculate_raw_score(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算市场宽度原始评分 (0-100分)
        
        Args:
            data: DataFrame，包含市场数据，需要有多个股票的数据
            *args: 位置参数
            **kwargs: 关键字参数
                
        Returns:
            DataFrame: 包含原始评分的DataFrame
        """
        # 初始化评分DataFrame
        dates = data.index.get_level_values(0).unique()
        scores = pd.DataFrame(index=dates)
        scores['raw_score'] = 50.0  # 默认评分50分（中性）
        
        # 计算市场宽度指标
        breadth_result = self.calculate(data, *args, **kwargs)
        
        if breadth_result.empty:
            return scores
        
        # 如果有市场宽度指标，直接使用
        if 'market_breadth_indicator' in breadth_result.columns:
            scores['raw_score'] = breadth_result['market_breadth_indicator']
            return scores
        
        # 否则通过涨跌比例和站上均线比例合成
        if 'ad_ratio' in breadth_result.columns and 'above_ma50' in breadth_result.columns:
            for i in range(len(scores)):
                # 涨跌比例评分（归一化到0-100）
                ad_score = min(100, max(0, (breadth_result['ad_ratio'].iloc[i] * 50) + 50))
                
                # 站上50日均线比例评分
                ma_score = min(100, max(0, breadth_result['above_ma50'].iloc[i] * 100))
                
                # 新高新低比例评分
                hl_score = 50
                if 'hl_ratio' in breadth_result.columns:
                    hl_ratio = breadth_result['hl_ratio'].iloc[i]
                    if not pd.isna(hl_ratio) and hl_ratio != 0:
                        if hl_ratio > 1:  # 新高多于新低
                            hl_score = min(100, 50 + hl_ratio * 10)
                        else:  # 新低多于新高
                            hl_score = max(0, 50 - (1/hl_ratio) * 10)
                
                # 综合评分
                final_score = ad_score * 0.4 + ma_score * 0.4 + hl_score * 0.2
                scores.loc[scores.index[i], 'raw_score'] = final_score
        
        return scores
        
    def identify_patterns(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别市场宽度中的关键形态
        
        Args:
            data: DataFrame，包含市场数据，需要有多个股票的数据
            *args: 位置参数
            **kwargs: 关键字参数
                min_pattern_strength: 最小形态强度阈值，默认0.6
                
        Returns:
            DataFrame: 包含识别出的形态的DataFrame
        """
        # 获取参数
        min_pattern_strength = kwargs.get('min_pattern_strength', 0.6)
        
        # 初始化形态DataFrame
        dates = data.index.get_level_values(0).unique()
        patterns = pd.DataFrame(index=dates)
        patterns['pattern'] = None
        patterns['pattern_strength'] = 0.0
        patterns['pattern_desc'] = None
        
        # 计算市场宽度指标
        breadth_result = self.calculate(data, *args, **kwargs)
        
        if breadth_result.empty:
            return patterns
        
        # 识别各种市场形态
        for i in range(5, len(patterns)):
            # 1. 市场顶部钝化形态
            if (self._is_breadth_divergence(breadth_result, i, bearish=True) and
                self._is_breadth_extreme(breadth_result, i, high=True)):
                patterns.loc[patterns.index[i], 'pattern'] = '市场顶部钝化'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.8
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场出现顶部钝化，涨跌比率下降但指数仍在上涨，警惕市场即将回调'
            
            # 2. 市场底部形态
            elif (self._is_breadth_divergence(breadth_result, i, bearish=False) and
                  self._is_breadth_extreme(breadth_result, i, high=False)):
                patterns.loc[patterns.index[i], 'pattern'] = '市场底部形成'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.8
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场出现底部形态，涨跌比率改善但指数仍在下跌，可能是市场即将反弹的信号'
            
            # 3. 市场宽度扩展形态
            elif self._is_breadth_expansion(breadth_result, i):
                patterns.loc[patterns.index[i], 'pattern'] = '市场宽度扩展'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.7
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场宽度指标快速扩展，表明市场强势上涨，多数股票参与'
            
            # 4. 市场宽度收缩形态
            elif self._is_breadth_contraction(breadth_result, i):
                patterns.loc[patterns.index[i], 'pattern'] = '市场宽度收缩'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.7
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场宽度指标快速收缩，表明市场走势变窄，可能是行情转变的前兆'
            
            # 5. 板块轮动形态
            elif 'sector_rotation' in breadth_result.columns and breadth_result['sector_rotation'].iloc[i] > 0.5:
                patterns.loc[patterns.index[i], 'pattern'] = '板块轮动'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.6
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场出现明显的板块轮动，热点切换，关注强势板块'
            
            # 6. 市场过热形态
            elif self._is_market_overheated(breadth_result, i):
                patterns.loc[patterns.index[i], 'pattern'] = '市场过热'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.75
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场各项指标过热，可能面临短期调整压力'
            
            # 7. 市场超跌形态
            elif self._is_market_oversold(breadth_result, i):
                patterns.loc[patterns.index[i], 'pattern'] = '市场超跌'
                patterns.loc[patterns.index[i], 'pattern_strength'] = 0.75
                patterns.loc[patterns.index[i], 'pattern_desc'] = '市场各项指标超跌，具备反弹条件'
        
        return patterns
        
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成市场宽度信号
        
        Args:
            data: DataFrame，包含市场数据
            *args: 位置参数
            **kwargs: 关键字参数
                signal_threshold: 信号阈值，默认70
                
        Returns:
            DataFrame: 包含标准化信号的DataFrame
        """
        # 获取参数
        signal_threshold = kwargs.get('signal_threshold', 70)
        
        # 初始化信号DataFrame
        dates = data.index.get_level_values(0).unique()
        signals = pd.DataFrame(index=dates)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认为中性信号
        signals['trend'] = 0  # 0表示中性
        signals['score'] = 50.0  # 默认评分50分
        signals['signal_type'] = None
        signals['signal_desc'] = None
        signals['confidence'] = 50.0
        signals['risk_level'] = '中'
        signals['position_size'] = 0.0
        signals['stop_loss'] = None
        signals['market_env'] = 'sideways_market'
        signals['volume_confirmation'] = False
        
        # 计算市场宽度指标
        breadth_result = self.calculate(data, *args, **kwargs)
        
        if breadth_result.empty:
            return signals
        
        # 识别形态
        patterns = self.identify_patterns(data, *args, **kwargs)
        
        # 基于市场宽度指标生成信号
        for i in range(5, len(signals)):
            # 跳过没有足够数据的早期记录
            if i < 5:
                continue
            
            # 设置基本分数
            if 'market_breadth_indicator' in breadth_result.columns:
                signals.loc[signals.index[i], 'score'] = breadth_result['market_breadth_indicator'].iloc[i]
            
            # 设置市场环境
            if 'market_state' in breadth_result.columns:
                market_state = breadth_result['market_state'].iloc[i]
                if market_state == 'bull':
                    signals.loc[signals.index[i], 'market_env'] = 'bull_market'
                    signals.loc[signals.index[i], 'trend'] = 1
                elif market_state == 'bear':
                    signals.loc[signals.index[i], 'market_env'] = 'bear_market'
                    signals.loc[signals.index[i], 'trend'] = -1
                elif market_state == 'top':
                    signals.loc[signals.index[i], 'market_env'] = 'top_market'
                    signals.loc[signals.index[i], 'trend'] = 0
                elif market_state == 'bottom':
                    signals.loc[signals.index[i], 'market_env'] = 'bottom_market'
                    signals.loc[signals.index[i], 'trend'] = 0
                else:
                    signals.loc[signals.index[i], 'market_env'] = 'sideways_market'
                    signals.loc[signals.index[i], 'trend'] = 0
            
            # 买入信号条件
            buy_condition = False
            
            if 'market_breadth_indicator' in breadth_result.columns:
                # 市场宽度指标由低位快速上升
                if (breadth_result['market_breadth_indicator'].iloc[i] > 
                    breadth_result['market_breadth_indicator'].iloc[i-1] + 10 and
                    breadth_result['market_breadth_indicator'].iloc[i-1] < 40):
                    buy_condition = True
                    signals.loc[signals.index[i], 'signal_type'] = '市场宽度改善'
                    signals.loc[signals.index[i], 'signal_desc'] = '市场宽度指标快速改善，市场可能开始走强'
                    signals.loc[signals.index[i], 'confidence'] = 70
            
            # 通过形态强化信号
            if patterns['pattern'].iloc[i] in ['市场底部形成', '市场超跌']:
                buy_condition = True
                signals.loc[signals.index[i], 'signal_type'] = patterns['pattern'].iloc[i]
                signals.loc[signals.index[i], 'signal_desc'] = patterns['pattern_desc'].iloc[i]
                signals.loc[signals.index[i], 'confidence'] = patterns['pattern_strength'].iloc[i] * 100
            
            # 卖出信号条件
            sell_condition = False
            
            if 'market_breadth_indicator' in breadth_result.columns:
                # 市场宽度指标由高位快速下降
                if (breadth_result['market_breadth_indicator'].iloc[i] < 
                    breadth_result['market_breadth_indicator'].iloc[i-1] - 10 and
                    breadth_result['market_breadth_indicator'].iloc[i-1] > 70):
                    sell_condition = True
                    signals.loc[signals.index[i], 'signal_type'] = '市场宽度恶化'
                    signals.loc[signals.index[i], 'signal_desc'] = '市场宽度指标快速恶化，市场可能开始走弱'
                    signals.loc[signals.index[i], 'confidence'] = 70
            
            # 通过形态强化信号
            if patterns['pattern'].iloc[i] in ['市场顶部钝化', '市场过热']:
                sell_condition = True
                signals.loc[signals.index[i], 'signal_type'] = patterns['pattern'].iloc[i]
                signals.loc[signals.index[i], 'signal_desc'] = patterns['pattern_desc'].iloc[i]
                signals.loc[signals.index[i], 'confidence'] = patterns['pattern_strength'].iloc[i] * 100
            
            # 应用买入信号
            if buy_condition:
                signals.loc[signals.index[i], 'buy_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = 1
                
                # 设置风险级别
                if signals['confidence'].iloc[i] > 80:
                    signals.loc[signals.index[i], 'risk_level'] = '低'
                
                # 建议仓位比例（根据信号强度和置信度）
                confidence_factor = signals['confidence'].iloc[i] / 100
                signals.loc[signals.index[i], 'position_size'] = min(0.8, 0.3 + confidence_factor * 0.5)
            
            # 应用卖出信号
            elif sell_condition:
                signals.loc[signals.index[i], 'sell_signal'] = True
                signals.loc[signals.index[i], 'neutral_signal'] = False
                signals.loc[signals.index[i], 'trend'] = -1
                
                # 设置风险级别
                if signals['confidence'].iloc[i] > 80:
                    signals.loc[signals.index[i], 'risk_level'] = '高'
                
                # 建议仓位比例（减仓或清仓）
                confidence_factor = signals['confidence'].iloc[i] / 100
                signals.loc[signals.index[i], 'position_size'] = max(0, 0.5 - confidence_factor * 0.5)
            
            # 成交量确认
            if 'volume_surge_ratio' in breadth_result.columns and 'volume_decline_ratio' in breadth_result.columns:
                if buy_condition and breadth_result['volume_surge_ratio'].iloc[i] > 0.4:
                    signals.loc[signals.index[i], 'volume_confirmation'] = True
                    signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
                
                if sell_condition and breadth_result['volume_decline_ratio'].iloc[i] > 0.4:
                    signals.loc[signals.index[i], 'volume_confirmation'] = True
                    signals.loc[signals.index[i], 'confidence'] = min(90, signals['confidence'].iloc[i] + 10)
        
        return signals 