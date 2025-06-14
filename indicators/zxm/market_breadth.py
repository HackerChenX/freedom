#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

from indicators.base_indicator import BaseIndicator

logger = logging.getLogger(__name__)

class ZXMMarketBreadth(BaseIndicator):
    """
    ZXM市场宽度指标
    
    用于分析市场整体状况、板块轮动和市场强度的综合指标，
    通过计算涨跌家数比例、主要指数相对强度等方式评估市场健康度。
    """
    
    def __init__(self, name: str = "ZXMMarketBreadth", description: str = "ZXM市场宽度指标"):
        """初始化ZXM市场宽度指标"""
        super().__init__(name, description)
        self.indicator_type = "ZXM_MARKET_BREADTH"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        # 市场宽度评估指标权重
        self.breadth_weights = {
            'advance_decline': 0.20,      # 涨跌家数比率
            'new_highs_lows': 0.15,       # 新高新低比率
            'percentage_above_ma': 0.20,  # 站上均线比例
            'sector_strength': 0.15,      # 板块强度
            'volume_breadth': 0.15,       # 成交量宽度
            'momentum_breadth': 0.15      # 动量宽度
        }
        
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
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
        
        # 检查数据结构，如果不是多层索引，则无法计算，返回空DataFrame
        if not isinstance(data.index, pd.MultiIndex):
            logger.warning(f"{self.name}: 输入数据不是多层索引DataFrame，无法计算市场宽度指标。")
            return pd.DataFrame(index=data.index)
            
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

    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算置信度

        Args:
            score: 评分序列
            patterns: 形态列表
            signals: 信号字典

        Returns:
            float: 置信度值，0-1之间
        """
        if score.empty:
            return 0.5

        latest_score = score.iloc[-1] if hasattr(score, 'iloc') else score

        # 基础置信度基于评分
        base_confidence = min(0.9, max(0.1, latest_score / 100))

        # 根据形态调整置信度
        pattern_boost = 0.0
        if "市场底部形成" in patterns:
            pattern_boost += 0.2
        elif "市场顶部钝化" in patterns:
            pattern_boost += 0.2
        elif "市场宽度扩展" in patterns:
            pattern_boost += 0.15
        elif "市场宽度收缩" in patterns:
            pattern_boost += 0.15
        elif "市场过热" in patterns:
            pattern_boost += 0.1
        elif "市场超跌" in patterns:
            pattern_boost += 0.1

        # 最终置信度
        final_confidence = min(1.0, max(0.0, base_confidence + pattern_boost))
        return final_confidence

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信号的DataFrame
        """
        return self.identify_patterns(data, **kwargs)

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典，可包含：
                - lookback_period: 回溯分析周期，默认60
                - ma_periods: 均线周期列表，默认[20, 50, 200]
                - signal_threshold: 信号阈值，默认70
        """
        self.lookback_period = kwargs.get('lookback_period', 60)
        self.ma_periods = kwargs.get('ma_periods', [20, 50, 200])
        self.signal_threshold = kwargs.get('signal_threshold', 70)

    def _calculate_advance_decline_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算涨跌家数比率"""
        dates = data.index.get_level_values(0).unique()
        result = pd.DataFrame(index=dates)

        for date in dates:
            try:
                day_data = data.loc[date]
                if 'close' in day_data.columns and len(day_data) > 1:
                    # 计算当日涨跌情况
                    if len(day_data) > 1:
                        prev_close = day_data['close'].shift(1)
                        advances = (day_data['close'] > prev_close).sum()
                        declines = (day_data['close'] < prev_close).sum()
                        unchanged = (day_data['close'] == prev_close).sum()

                        total = advances + declines
                        if total > 0:
                            ad_ratio = (advances - declines) / total
                        else:
                            ad_ratio = 0
                    else:
                        ad_ratio = 0
                else:
                    ad_ratio = 0

                result.loc[date, 'ad_ratio'] = ad_ratio
            except:
                result.loc[date, 'ad_ratio'] = 0

        # 计算AD线（累积）
        result['ad_line'] = result['ad_ratio'].cumsum()
        return result

    def _calculate_new_highs_lows_ratio(self, data: pd.DataFrame, lookback_period: int) -> pd.DataFrame:
        """计算新高新低比率"""
        dates = data.index.get_level_values(0).unique()
        result = pd.DataFrame(index=dates)

        for i, date in enumerate(dates):
            if i < lookback_period:
                result.loc[date, 'new_highs_ratio'] = 0
                result.loc[date, 'new_lows_ratio'] = 0
                result.loc[date, 'hl_ratio'] = 1
                continue

            try:
                # 获取当前日期和历史数据
                current_data = data.loc[date]
                historical_dates = dates[max(0, i-lookback_period):i]

                new_highs = 0
                new_lows = 0
                total_stocks = len(current_data)

                for stock in current_data.index:
                    current_high = current_data.loc[stock, 'high']
                    current_low = current_data.loc[stock, 'low']

                    # 检查是否创新高或新低
                    historical_highs = []
                    historical_lows = []

                    for hist_date in historical_dates:
                        try:
                            hist_data = data.loc[hist_date]
                            if stock in hist_data.index:
                                historical_highs.append(hist_data.loc[stock, 'high'])
                                historical_lows.append(hist_data.loc[stock, 'low'])
                        except:
                            continue

                    if historical_highs and current_high > max(historical_highs):
                        new_highs += 1
                    if historical_lows and current_low < min(historical_lows):
                        new_lows += 1

                if total_stocks > 0:
                    result.loc[date, 'new_highs_ratio'] = new_highs / total_stocks
                    result.loc[date, 'new_lows_ratio'] = new_lows / total_stocks

                    if new_lows > 0:
                        result.loc[date, 'hl_ratio'] = new_highs / new_lows
                    else:
                        result.loc[date, 'hl_ratio'] = new_highs if new_highs > 0 else 1
                else:
                    result.loc[date, 'new_highs_ratio'] = 0
                    result.loc[date, 'new_lows_ratio'] = 0
                    result.loc[date, 'hl_ratio'] = 1

            except:
                result.loc[date, 'new_highs_ratio'] = 0
                result.loc[date, 'new_lows_ratio'] = 0
                result.loc[date, 'hl_ratio'] = 1

        return result

    def _calculate_percentage_above_ma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """计算站上均线的股票比例"""
        dates = data.index.get_level_values(0).unique()
        result = pd.Series(index=dates, dtype=float)

        for date in dates:
            try:
                day_data = data.loc[date]
                if 'close' in day_data.columns and len(day_data) > 0:
                    above_ma_count = 0
                    total_count = 0

                    for stock in day_data.index:
                        # 获取该股票的历史数据计算均线
                        stock_data = []
                        for hist_date in dates:
                            try:
                                hist_day_data = data.loc[hist_date]
                                if stock in hist_day_data.index:
                                    stock_data.append(hist_day_data.loc[stock, 'close'])
                            except:
                                continue

                        if len(stock_data) >= period:
                            ma_value = np.mean(stock_data[-period:])
                            current_price = day_data.loc[stock, 'close']

                            if current_price > ma_value:
                                above_ma_count += 1
                            total_count += 1

                    if total_count > 0:
                        result[date] = above_ma_count / total_count
                    else:
                        result[date] = 0
                else:
                    result[date] = 0
            except:
                result[date] = 0

        return result

    def _calculate_sector_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算板块强度"""
        dates = data.index.get_level_values(0).unique()
        result = pd.DataFrame(index=dates)

        # 简化实现，返回默认值
        result['strongest_sector'] = 'Technology'
        result['weakest_sector'] = 'Energy'
        result['sector_rotation'] = 0.5

        return result

    def _calculate_volume_breadth(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量宽度"""
        dates = data.index.get_level_values(0).unique()
        result = pd.DataFrame(index=dates)

        for date in dates:
            try:
                day_data = data.loc[date]
                if 'volume' in day_data.columns and len(day_data) > 0:
                    # 计算成交量变化
                    volume_surge_count = 0
                    volume_decline_count = 0
                    total_count = len(day_data)

                    # 简化计算：假设成交量放大和缩小的比例
                    avg_volume = day_data['volume'].mean()
                    volume_surge_count = (day_data['volume'] > avg_volume * 1.5).sum()
                    volume_decline_count = (day_data['volume'] < avg_volume * 0.5).sum()

                    if total_count > 0:
                        result.loc[date, 'volume_surge_ratio'] = volume_surge_count / total_count
                        result.loc[date, 'volume_decline_ratio'] = volume_decline_count / total_count
                    else:
                        result.loc[date, 'volume_surge_ratio'] = 0
                        result.loc[date, 'volume_decline_ratio'] = 0
                else:
                    result.loc[date, 'volume_surge_ratio'] = 0
                    result.loc[date, 'volume_decline_ratio'] = 0
            except:
                result.loc[date, 'volume_surge_ratio'] = 0
                result.loc[date, 'volume_decline_ratio'] = 0

        return result

    def _calculate_momentum_breadth(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算动量宽度"""
        dates = data.index.get_level_values(0).unique()
        result = pd.DataFrame(index=dates)

        for date in dates:
            try:
                day_data = data.loc[date]
                if 'close' in day_data.columns and len(day_data) > 0:
                    # 简化计算：基于价格变化计算动量
                    positive_momentum = 0
                    negative_momentum = 0
                    total_count = len(day_data)

                    # 假设有前一日数据进行比较
                    for stock in day_data.index:
                        current_price = day_data.loc[stock, 'close']
                        # 简化：使用开盘价作为参考
                        open_price = day_data.loc[stock, 'open']

                        if current_price > open_price:
                            positive_momentum += 1
                        elif current_price < open_price:
                            negative_momentum += 1

                    if total_count > 0:
                        result.loc[date, 'positive_ratio'] = positive_momentum / total_count
                        result.loc[date, 'negative_ratio'] = negative_momentum / total_count
                    else:
                        result.loc[date, 'positive_ratio'] = 0
                        result.loc[date, 'negative_ratio'] = 0
                else:
                    result.loc[date, 'positive_ratio'] = 0
                    result.loc[date, 'negative_ratio'] = 0
            except:
                result.loc[date, 'positive_ratio'] = 0
                result.loc[date, 'negative_ratio'] = 0

        return result

    def _calculate_market_relative_strength(self, data: pd.DataFrame, index_code: str) -> pd.Series:
        """计算市场相对强度"""
        dates = data.index.get_level_values(0).unique()
        result = pd.Series(index=dates, dtype=float)

        # 简化实现，返回默认值
        result[:] = 1.0

        return result

    def _calculate_breadth_indicator(self, result: pd.DataFrame) -> pd.Series:
        """计算综合市场宽度指标"""
        breadth_indicator = pd.Series(index=result.index, dtype=float)

        for i in range(len(result)):
            score = 50.0  # 基础分数

            # AD比率贡献
            if 'ad_ratio' in result.columns:
                ad_ratio = result['ad_ratio'].iloc[i]
                score += ad_ratio * 25  # -1到1映射到25-75

            # 站上均线比例贡献
            if 'above_ma50' in result.columns:
                above_ma50 = result['above_ma50'].iloc[i]
                score += (above_ma50 - 0.5) * 50  # 0-1映射到25-75

            # 新高新低比率贡献
            if 'hl_ratio' in result.columns:
                hl_ratio = result['hl_ratio'].iloc[i]
                if hl_ratio > 1:
                    score += min(25, (hl_ratio - 1) * 10)
                else:
                    score -= min(25, (1 - hl_ratio) * 10)

            breadth_indicator.iloc[i] = max(0, min(100, score))

        return breadth_indicator

    def _classify_market_state(self, result: pd.DataFrame) -> pd.Series:
        """分类市场状态"""
        market_state = pd.Series(index=result.index, dtype=str)

        for i in range(len(result)):
            if 'market_breadth_indicator' in result.columns:
                breadth_score = result['market_breadth_indicator'].iloc[i]

                if breadth_score > 75:
                    market_state.iloc[i] = 'bull'
                elif breadth_score < 25:
                    market_state.iloc[i] = 'bear'
                elif breadth_score > 85:
                    market_state.iloc[i] = 'top'
                elif breadth_score < 15:
                    market_state.iloc[i] = 'bottom'
                else:
                    market_state.iloc[i] = 'sideways'
            else:
                market_state.iloc[i] = 'sideways'

        return market_state

    def _is_breadth_divergence(self, result: pd.DataFrame, index: int, bearish: bool = True) -> bool:
        """检测宽度背离"""
        if index < 5 or 'market_breadth_indicator' not in result.columns:
            return False

        # 简化实现
        current_breadth = result['market_breadth_indicator'].iloc[index]
        prev_breadth = result['market_breadth_indicator'].iloc[index-1]

        if bearish:
            return current_breadth < prev_breadth - 5
        else:
            return current_breadth > prev_breadth + 5

    def _is_breadth_extreme(self, result: pd.DataFrame, index: int, high: bool = True) -> bool:
        """检测宽度极值"""
        if 'market_breadth_indicator' not in result.columns:
            return False

        current_breadth = result['market_breadth_indicator'].iloc[index]

        if high:
            return current_breadth > 85
        else:
            return current_breadth < 15

    def _is_breadth_expansion(self, result: pd.DataFrame, index: int) -> bool:
        """检测宽度扩展"""
        if index < 3 or 'market_breadth_indicator' not in result.columns:
            return False

        current_breadth = result['market_breadth_indicator'].iloc[index]
        prev_breadth = result['market_breadth_indicator'].iloc[index-3]

        return current_breadth > prev_breadth + 15

    def _is_breadth_contraction(self, result: pd.DataFrame, index: int) -> bool:
        """检测宽度收缩"""
        if index < 3 or 'market_breadth_indicator' not in result.columns:
            return False

        current_breadth = result['market_breadth_indicator'].iloc[index]
        prev_breadth = result['market_breadth_indicator'].iloc[index-3]

        return current_breadth < prev_breadth - 15

    def _is_market_overheated(self, result: pd.DataFrame, index: int) -> bool:
        """检测市场过热"""
        if 'market_breadth_indicator' not in result.columns:
            return False

        current_breadth = result['market_breadth_indicator'].iloc[index]
        return current_breadth > 90

    def _is_market_oversold(self, result: pd.DataFrame, index: int) -> bool:
        """检测市场超跌"""
        if 'market_breadth_indicator' not in result.columns:
            return False

        current_breadth = result['market_breadth_indicator'].iloc[index]
        return current_breadth < 10