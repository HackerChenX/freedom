#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
V形反转指标模块

实现V形反转形态识别功能，用于识别急速下跌后快速反弹的形态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class VShapedReversal(BaseIndicator):
    """
    V形反转指标
    
    识别急速下跌后快速反弹的价格形态，用于短期超卖后的反弹信号
    """
    
    def __init__(self, decline_period: int = 5, rebound_period: int = 5,
               decline_threshold: float = 0.05, rebound_threshold: float = 0.05):
        """
        初始化V形反转指标
        
        Args:
            decline_period: 下跌周期，默认为5日
            rebound_period: 反弹周期，默认为5日
            decline_threshold: 下跌阈值，默认为5%
            rebound_threshold: 反弹阈值，默认为5%
        """
        super().__init__(name="VShapedReversal", description="V形反转指标，识别急速下跌后快速反弹的形态")
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        self.decline_period = decline_period
        self.rebound_period = rebound_period
        self.decline_threshold = decline_threshold
        self.rebound_threshold = rebound_threshold
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算V形反转指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
                
        Returns:
            包含V形反转指标的DataFrame
        """
        return self.calculate(df)
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算V形反转指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含V形反转信号
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算各周期的价格变化率
        result["decline_rate"] = data["close"].pct_change(periods=self.decline_period)
        result["rebound_rate"] = data["close"].pct_change(periods=self.rebound_period)
        
        # 初始化V形反转信号
        result["v_reversal"] = False
        
        # 最小索引值，确保有足够的数据计算
        min_idx = self.decline_period + self.rebound_period
        
        if len(data) > min_idx:
            close_values = data["close"].values
            
            # 使用滑动窗口计算下跌和反弹
            for i in range(min_idx, len(data)):
                # 计算下跌区间的起始和结束价格
                decline_start_idx = i - self.decline_period - self.rebound_period
                decline_end_idx = i - self.rebound_period
                
                if decline_start_idx >= 0:
                    decline_start_price = close_values[decline_start_idx]
                    decline_end_price = close_values[decline_end_idx]
                    
                    # 计算反弹区间的起始和结束价格
                    rebound_start_idx = i - self.rebound_period
                    rebound_end_idx = i
                    
                    rebound_start_price = close_values[rebound_start_idx]
                    rebound_end_price = close_values[rebound_end_idx]
                    
                    # 计算下跌和反弹幅度
                    if decline_start_price > 0 and rebound_start_price > 0:
                        decline_rate = (decline_end_price - decline_start_price) / decline_start_price
                        rebound_rate = (rebound_end_price - rebound_start_price) / rebound_start_price
                        
                        # 判断是否满足V形反转条件
                        if decline_rate <= -self.decline_threshold and rebound_rate >= self.rebound_threshold:
                            result.iloc[i, result.columns.get_loc("v_reversal")] = True
        
        # 计算V形底部位置
        result["v_bottom"] = False
        
        if len(data) > self.decline_period + self.rebound_period:
            close_values = data["close"].values
            
            # 使用滑动窗口检测V形底部
            for i in range(self.decline_period, len(data) - self.rebound_period):
                pre_window_start = i - self.decline_period
                pre_window_end = i + 1
                post_window_start = i
                post_window_end = i + self.rebound_period + 1
                
                if pre_window_start >= 0 and post_window_end <= len(close_values):
                    pre_window = close_values[pre_window_start:pre_window_end]
                    post_window = close_values[post_window_start:post_window_end]
                    
                    # 如果当前价格是前后窗口的最低点
                    if (close_values[i] <= np.min(pre_window) and
                        close_values[i] <= np.min(post_window)):
                        result.iloc[i, result.columns.get_loc("v_bottom")] = True
        
        return result

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取V形反转指标的技术形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算V形反转
        if not self.has_result():
            self.calculate(data, **kwargs)

        result = self._result.copy()
        patterns_df = pd.DataFrame(index=data.index)

        # 基本V形反转形态
        patterns_df['V_SHAPED_REVERSAL'] = result.get('v_reversal', False)
        patterns_df['V_BOTTOM'] = result.get('v_bottom', False)

        # 根据反转方向分类
        if 'decline_rate' in result.columns and 'rebound_rate' in result.columns:
            decline_rate = result['decline_rate']
            rebound_rate = result['rebound_rate']

            # 看涨V形反转（下跌后反弹）
            patterns_df['V_BULLISH_REVERSAL'] = (
                patterns_df['V_SHAPED_REVERSAL'] &
                (decline_rate < 0) & (rebound_rate > 0)
            )

            # 看跌倒V形反转（上涨后回落）
            patterns_df['V_BEARISH_REVERSAL'] = (
                patterns_df['V_SHAPED_REVERSAL'] &
                (decline_rate > 0) & (rebound_rate < 0)
            )
        else:
            patterns_df['V_BULLISH_REVERSAL'] = False
            patterns_df['V_BEARISH_REVERSAL'] = False

        return patterns_df

    def register_patterns(self):
        """
        注册VShapedReversal指标的形态到全局形态注册表
        """
        # 注册V形反转基本形态
        self.register_pattern_to_registry(
            pattern_id="V_SHAPED_REVERSAL",
            display_name="V形反转",
            description="价格形成V形反转形态，表明趋势可能发生转折",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        # 注册看涨V形反转
        self.register_pattern_to_registry(
            pattern_id="V_BULLISH_REVERSAL",
            display_name="看涨V形反转",
            description="价格急速下跌后快速反弹，形成V形底部，强烈看涨信号",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        # 注册看跌倒V形反转
        self.register_pattern_to_registry(
            pattern_id="V_BEARISH_REVERSAL",
            display_name="看跌倒V形反转",
            description="价格急速上涨后快速回落，形成倒V形顶部，强烈看跌信号",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

        # 注册V形底部
        self.register_pattern_to_registry(
            pattern_id="V_BOTTOM",
            display_name="V形底部",
            description="价格形成V形底部，表明下跌趋势结束，看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

    def get_signals(self, data: pd.DataFrame, confirmation_days: int = 2) -> pd.DataFrame:
        """
        生成V形反转信号
        
        Args:
            data: 输入数据，包含V形反转指标
            confirmation_days: 确认天数，默认为2日
            
        Returns:
            pd.DataFrame: 包含V形反转信号的数据框
        """
        if "v_reversal" not in data.columns:
            data = self.calculate(data)
        
        # 创建结果DataFrame的副本
        result = data.copy()
        
        # 初始化买入信号列
        result["v_buy_signal"] = False
        
        # 使用滑动窗口检测前N天是否有V形反转
        if len(data) > confirmation_days:
            # 获取v_reversal列的值
            v_reversal_values = data["v_reversal"].values
            close_values = data["close"].values
            
            # 滑动窗口检测
            for i in range(confirmation_days, len(data)):
                # 检查前confirmation_days天内是否有V形反转
                has_reversal = np.any(v_reversal_values[i-confirmation_days:i])
                
                # 检查价格是否上涨
                if i >= confirmation_days:
                    price_rising = close_values[i] > close_values[i-confirmation_days]
                    
                    # 生成买入信号
                    if has_reversal and price_rising:
                        result.iloc[i, result.columns.get_loc("v_buy_signal")] = True
        
        return result
    
    def get_reversal_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算V形反转强度
        
        Args:
            data: 输入数据，包含V形反转指标
            
        Returns:
            pd.DataFrame: 包含V形反转强度的数据框
        """
        if "v_reversal" not in data.columns:
            data = self.calculate(data)
        
        # 创建结果DataFrame的副本
        result = data.copy()
        
        # 初始化反转强度列
        result["reversal_strength"] = 0.0
        
        # 获取需要的列的值
        v_reversal_values = data["v_reversal"].values
        decline_rate_values = data["decline_rate"].values
        rebound_rate_values = data["rebound_rate"].values
        
        # 计算反转强度
        for i in range(len(data)):
            if v_reversal_values[i]:
                # 反转强度 = 下跌幅度与反弹幅度的乘积
                result.iloc[i, result.columns.get_loc("reversal_strength")] = (
                    abs(decline_rate_values[i]) * rebound_rate_values[i]
                ) * 100
        
        # 初始化反转分类列
        result["reversal_category"] = None
        
        # 根据强度阈值分类
        for i in range(len(result)):
            strength = result.iloc[i, result.columns.get_loc("reversal_strength")]
            
            if strength > 5:
                result.iloc[i, result.columns.get_loc("reversal_category")] = "强烈反转"
            elif strength > 2:
                result.iloc[i, result.columns.get_loc("reversal_category")] = "明显反转"
            elif strength > 0:
                result.iloc[i, result.columns.get_loc("reversal_category")] = "弱反转"
        
        return result
    
    def find_v_patterns(self, data: pd.DataFrame, window: int = 20) -> List[Tuple[int, str]]:
        """
        查找数据中的V形反转模式
        
        Args:
            data: 输入数据，包含价格数据
            window: 搜索窗口大小，默认为20日
            
        Returns:
            List[Tuple[int, str]]: V形反转位置及其类别的列表
        """
        if "v_reversal" not in data.columns:
            data = self.calculate(data)
        
        patterns = []
        
        # 获取所有强V形反转的位置
        v_indices = data.index[data["v_reversal"]].tolist()
        
        for idx in v_indices:
            # 获取对应的数据位置
            pos = data.index.get_loc(idx)
            
            # 检查是否有强度分类
            if "reversal_category" in data.columns and pd.notna(data.iloc[pos]["reversal_category"]):
                category = data.iloc[pos]["reversal_category"]
            else:
                category = "V形反转"
            
            # 添加到模式列表
            patterns.append((pos, category))
        
        return patterns
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算V形反转指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        if 'close' not in data.columns:
            return pd.DataFrame({'score': score}, index=data.index)
        
        close_price = data['close']
        
        # 1. V形反转确认评分（±40分）
        if 'v_reversal' in indicator_data.columns:
            v_reversal_mask = indicator_data['v_reversal']
            
            # 需要判断V形反转的方向
            if 'decline_rate' in indicator_data.columns and 'rebound_rate' in indicator_data.columns:
                decline_rate = indicator_data['decline_rate']
                rebound_rate = indicator_data['rebound_rate']
                
                # 看涨V形反转（下跌后反弹）
                bullish_v = v_reversal_mask & (decline_rate < 0) & (rebound_rate > 0)
                score.loc[bullish_v] += 40
                
                # 看跌倒V形反转（上涨后回落）
                bearish_v = v_reversal_mask & (decline_rate > 0) & (rebound_rate < 0)
                score.loc[bearish_v] -= 40
        
        # 2. V形反转深度评分（±10分）
        if 'decline_rate' in indicator_data.columns and 'rebound_rate' in indicator_data.columns:
            decline_rate = indicator_data['decline_rate']
            rebound_rate = indicator_data['rebound_rate']
            
            # 计算反转深度（下跌和反弹的幅度）
            reversal_depth = abs(decline_rate) + abs(rebound_rate)
            
            # 深度越大，评分调整越大
            for i in range(len(data)):
                if pd.notna(reversal_depth.iloc[i]) and reversal_depth.iloc[i] > 0:
                    depth_score = min(10, reversal_depth.iloc[i] * 100)  # 最多±10分
                    
                    # 根据反转方向调整评分
                    if (pd.notna(decline_rate.iloc[i]) and pd.notna(rebound_rate.iloc[i]) and
                        decline_rate.iloc[i] < 0 and rebound_rate.iloc[i] > 0):
                        score.iloc[i] += depth_score  # 看涨V形
                    elif (pd.notna(decline_rate.iloc[i]) and pd.notna(rebound_rate.iloc[i]) and
                          decline_rate.iloc[i] > 0 and rebound_rate.iloc[i] < 0):
                        score.iloc[i] -= depth_score  # 看跌倒V形
        
        # 3. V形反转速度评分（±15分）
        # 反转速度越快，信号越强
        if len(data) >= self.decline_period + self.rebound_period:
            for i in range(self.decline_period + self.rebound_period, len(data)):
                # 计算反转速度（时间越短，速度越快）
                total_period = self.decline_period + self.rebound_period
                
                if total_period <= 5:  # 快速反转（5天内）
                    if 'v_reversal' in indicator_data.columns and indicator_data['v_reversal'].iloc[i]:
                        # 判断反转方向
                        if (i >= self.decline_period and 
                            close_price.iloc[i] > close_price.iloc[i - self.rebound_period]):
                            score.iloc[i] += 15  # 快速看涨反转
                        elif (i >= self.decline_period and 
                              close_price.iloc[i] < close_price.iloc[i - self.rebound_period]):
                            score.iloc[i] -= 15  # 快速看跌反转
                
                elif total_period <= 10:  # 中速反转（6-10天）
                    if 'v_reversal' in indicator_data.columns and indicator_data['v_reversal'].iloc[i]:
                        if (i >= self.decline_period and 
                            close_price.iloc[i] > close_price.iloc[i - self.rebound_period]):
                            score.iloc[i] += 10  # 中速看涨反转
                        elif (i >= self.decline_period and 
                              close_price.iloc[i] < close_price.iloc[i - self.rebound_period]):
                            score.iloc[i] -= 10  # 中速看跌反转
        
        # 4. 成交量确认评分（±15分）
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 放量确认V形反转
            high_volume = vol_ratio > 1.5
            
            if 'v_reversal' in indicator_data.columns:
                v_reversal_mask = indicator_data['v_reversal']
                
                # 放量确认的V形反转更可靠
                volume_confirmed_v = v_reversal_mask & high_volume
                
                # 需要判断反转方向来决定加分还是减分
                if 'decline_rate' in indicator_data.columns and 'rebound_rate' in indicator_data.columns:
                    decline_rate = indicator_data['decline_rate']
                    rebound_rate = indicator_data['rebound_rate']
                    
                    # 放量确认的看涨V形反转
                    bullish_volume_v = (volume_confirmed_v & 
                                      (decline_rate < 0) & (rebound_rate > 0))
                    score.loc[bullish_volume_v] += 15
                    
                    # 放量确认的看跌倒V形反转
                    bearish_volume_v = (volume_confirmed_v & 
                                      (decline_rate > 0) & (rebound_rate < 0))
                    score.loc[bearish_volume_v] -= 15
        
        # 5. V形反转持续性评分（±20分）
        # 检查反转后的价格走势是否持续
        if 'v_reversal' in indicator_data.columns and len(data) >= 10:
            v_reversal_indices = indicator_data.index[indicator_data['v_reversal']].tolist()
            
            for v_idx in v_reversal_indices:
                idx_pos = data.index.get_loc(v_idx)
                
                # 检查反转后5天的走势
                if idx_pos + 5 < len(data):
                    reversal_price = close_price.iloc[idx_pos]
                    future_prices = close_price.iloc[idx_pos+1:idx_pos+6]
                    
                    if len(future_prices) >= 3:
                        # 计算反转后的价格趋势
                        price_trend = (future_prices.iloc[-1] - reversal_price) / reversal_price
                        
                        # 判断原始反转方向
                        if (idx_pos >= self.decline_period and 
                            reversal_price > close_price.iloc[idx_pos - self.decline_period]):
                            # 原本是看涨V形反转
                            if price_trend > 0.02:  # 持续上涨超过2%
                                score.iloc[idx_pos] += 20
                            elif price_trend < -0.02:  # 反转失败
                                score.iloc[idx_pos] -= 10
                        
                        elif (idx_pos >= self.decline_period and 
                              reversal_price < close_price.iloc[idx_pos - self.decline_period]):
                            # 原本是看跌倒V形反转
                            if price_trend < -0.02:  # 持续下跌超过2%
                                score.iloc[idx_pos] -= 20
                            elif price_trend > 0.02:  # 反转失败
                                score.iloc[idx_pos] += 10
        
        # 6. V形底部确认评分（±25分）
        if 'v_bottom' in indicator_data.columns:
            v_bottom_mask = indicator_data['v_bottom']
            
            # V形底部通常是看涨信号
            score.loc[v_bottom_mask] += 25
            
            # 如果V形底部伴随放量，额外加分
            if 'volume' in data.columns:
                volume = data['volume']
                vol_ma5 = volume.rolling(window=5).mean()
                vol_ratio = volume / vol_ma5
                
                high_volume = vol_ratio > 1.5
                volume_confirmed_bottom = v_bottom_mask & high_volume
                score.loc[volume_confirmed_bottom] += 10
        
        # 7. 反转强度评分（±15分）
        # 使用get_reversal_strength方法计算的强度
        try:
            strength_data = self.get_reversal_strength(data)
            
            if 'reversal_strength' in strength_data.columns:
                reversal_strength = strength_data['reversal_strength']
                
                # 根据反转强度调整评分
                for i in range(len(data)):
                    strength = reversal_strength.iloc[i]
                    
                    if pd.notna(strength) and strength > 0:
                        if strength > 5:  # 强烈反转
                            strength_score = 15
                        elif strength > 2:  # 明显反转
                            strength_score = 10
                        else:  # 弱反转
                            strength_score = 5
                        
                        # 判断反转方向
                        if ('decline_rate' in indicator_data.columns and 
                            'rebound_rate' in indicator_data.columns):
                            decline_rate = indicator_data['decline_rate'].iloc[i]
                            rebound_rate = indicator_data['rebound_rate'].iloc[i]
                            
                            if (pd.notna(decline_rate) and pd.notna(rebound_rate) and
                                decline_rate < 0 and rebound_rate > 0):
                                score.iloc[i] += strength_score  # 看涨反转
                            elif (pd.notna(decline_rate) and pd.notna(rebound_rate) and
                                  decline_rate > 0 and rebound_rate < 0):
                                score.iloc[i] -= strength_score  # 看跌反转
        
        except Exception as e:
            logger.warning(f"计算反转强度评分时出错: {e}")
        
        # 8. 多重V形反转评分（±10分）
        # 如果短期内出现多个V形反转，可能表示市场不稳定
        if 'v_reversal' in indicator_data.columns and len(data) >= 20:
            v_reversal_mask = indicator_data['v_reversal']
            
            # 检查最近20天内的V形反转次数
            for i in range(20, len(data)):
                recent_reversals = v_reversal_mask.iloc[i-20:i+1].sum()
                
                if recent_reversals >= 3:  # 频繁反转，市场不稳定
                    score.iloc[i] -= 10
                elif recent_reversals == 2:  # 适度反转
                    score.iloc[i] += 5
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别V形反转相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 10 or 'close' not in data.columns:
            return patterns
        
        close_price = data['close']
        
        # 1. V形反转基本形态识别
        if 'v_reversal' in indicator_data.columns:
            v_reversal_mask = indicator_data['v_reversal']
            recent_reversals = v_reversal_mask.tail(10)
            
            if recent_reversals.any():
                # 统计最近的V形反转
                reversal_count = recent_reversals.sum()
                patterns.append(f"近期V形反转-{reversal_count}次")
                
                # 分析反转方向
                if ('decline_rate' in indicator_data.columns and 
                    'rebound_rate' in indicator_data.columns):
                    
                    recent_decline = indicator_data['decline_rate'].tail(10)
                    recent_rebound = indicator_data['rebound_rate'].tail(10)
                    
                    # 统计看涨和看跌反转
                    bullish_reversals = ((recent_decline < 0) & (recent_rebound > 0) & recent_reversals).sum()
                    bearish_reversals = ((recent_decline > 0) & (recent_rebound < 0) & recent_reversals).sum()
                    
                    if bullish_reversals > 0:
                        patterns.append(f"看涨V形反转-{bullish_reversals}次")
                    if bearish_reversals > 0:
                        patterns.append(f"看跌倒V形反转-{bearish_reversals}次")
        
        # 2. V形底部形态识别
        if 'v_bottom' in indicator_data.columns:
            v_bottom_mask = indicator_data['v_bottom']
            recent_bottoms = v_bottom_mask.tail(10)
            
            if recent_bottoms.any():
                bottom_count = recent_bottoms.sum()
                patterns.append(f"V形底部-{bottom_count}次")
                
                # 检查最新的V形底部
                if v_bottom_mask.iloc[-1]:
                    patterns.append("当前V形底部")
                elif recent_bottoms.tail(5).any():
                    patterns.append("近期V形底部")
        
        # 3. 反转强度分析
        try:
            strength_data = self.get_reversal_strength(data)
            
            if 'reversal_category' in strength_data.columns:
                recent_categories = strength_data['reversal_category'].tail(10)
                
                # 统计不同强度的反转
                strong_count = (recent_categories == '强烈反转').sum()
                obvious_count = (recent_categories == '明显反转').sum()
                weak_count = (recent_categories == '弱反转').sum()
                
                if strong_count > 0:
                    patterns.append(f"强烈V形反转-{strong_count}次")
                if obvious_count > 0:
                    patterns.append(f"明显V形反转-{obvious_count}次")
                if weak_count > 0:
                    patterns.append(f"弱V形反转-{weak_count}次")
                
                # 最新反转强度
                latest_category = strength_data['reversal_category'].iloc[-1]
                if pd.notna(latest_category):
                    patterns.append(f"最新反转强度-{latest_category}")
        
        except Exception as e:
            logger.warning(f"分析反转强度时出错: {e}")
        
        # 4. 反转速度分析
        total_period = self.decline_period + self.rebound_period
        
        if total_period <= 5:
            patterns.append("快速V形反转设置")
        elif total_period <= 10:
            patterns.append("中速V形反转设置")
        else:
            patterns.append("慢速V形反转设置")
        
        # 5. 成交量配合分析
        if 'volume' in data.columns and 'v_reversal' in indicator_data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            latest_vol_ratio = (volume / vol_ma5).iloc[-1]
            
            if pd.notna(latest_vol_ratio):
                if latest_vol_ratio > 2.0:
                    patterns.append("V形反转-巨量配合")
                elif latest_vol_ratio > 1.5:
                    patterns.append("V形反转-放量配合")
                elif latest_vol_ratio < 0.7:
                    patterns.append("V形反转-缩量形成")
        
        # 6. 反转持续性分析
        if 'v_reversal' in indicator_data.columns and len(data) >= 15:
            # 检查最近的V形反转后的走势
            v_reversal_indices = indicator_data.index[indicator_data['v_reversal']].tolist()
            
            if v_reversal_indices:
                latest_reversal_idx = v_reversal_indices[-1]
                idx_pos = data.index.get_loc(latest_reversal_idx)
                
                # 检查反转后的走势
                if idx_pos + 5 < len(data):
                    reversal_price = close_price.iloc[idx_pos]
                    current_price = close_price.iloc[-1]
                    price_change = (current_price - reversal_price) / reversal_price
                    
                    if price_change > 0.05:
                        patterns.append("V形反转后持续上涨")
                    elif price_change < -0.05:
                        patterns.append("V形反转后持续下跌")
                    elif abs(price_change) < 0.02:
                        patterns.append("V形反转后横盘整理")
                    else:
                        patterns.append("V形反转后小幅波动")
        
        # 7. 反转频率分析
        if 'v_reversal' in indicator_data.columns and len(data) >= 30:
            # 分析不同时间段的反转频率
            recent_30d = indicator_data['v_reversal'].tail(30).sum()
            recent_20d = indicator_data['v_reversal'].tail(20).sum()
            recent_10d = indicator_data['v_reversal'].tail(10).sum()
            
            if recent_10d >= 2:
                patterns.append("高频V形反转")
            elif recent_20d >= 2:
                patterns.append("中频V形反转")
            elif recent_30d >= 1:
                patterns.append("低频V形反转")
            else:
                patterns.append("无V形反转")
        
        # 8. 反转幅度分析
        if ('decline_rate' in indicator_data.columns and 
            'rebound_rate' in indicator_data.columns):
            
            recent_decline = indicator_data['decline_rate'].tail(10)
            recent_rebound = indicator_data['rebound_rate'].tail(10)
            
            # 计算平均反转幅度
            avg_decline = abs(recent_decline[recent_decline < 0]).mean()
            avg_rebound = recent_rebound[recent_rebound > 0].mean()
            
            if pd.notna(avg_decline) and avg_decline > 0.1:
                patterns.append("大幅度下跌反转")
            elif pd.notna(avg_decline) and avg_decline > 0.05:
                patterns.append("中幅度下跌反转")
            elif pd.notna(avg_decline) and avg_decline > 0:
                patterns.append("小幅度下跌反转")
            
            if pd.notna(avg_rebound) and avg_rebound > 0.1:
                patterns.append("大幅度反弹")
            elif pd.notna(avg_rebound) and avg_rebound > 0.05:
                patterns.append("中幅度反弹")
            elif pd.notna(avg_rebound) and avg_rebound > 0:
                patterns.append("小幅度反弹")
        
        # 9. 市场环境分析
        if len(data) >= 30:
            # 分析当前市场环境对V形反转的影响
            ma20 = close_price.rolling(window=20).mean()
            latest_price = close_price.iloc[-1]
            latest_ma20 = ma20.iloc[-1]
            
            if pd.notna(latest_ma20):
                if latest_price > latest_ma20 * 1.05:
                    patterns.append("V形反转-上升趋势环境")
                elif latest_price < latest_ma20 * 0.95:
                    patterns.append("V形反转-下降趋势环境")
                else:
                    patterns.append("V形反转-横盘环境")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成V形反转指标信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
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
        
        # 计算评分
        if kwargs.get('use_raw_score', True):
            score_data = self.calculate_raw_score(data)
            if 'score' in score_data.columns:
                signals['score'] = score_data['score']
        
        # 基于V形反转信号生成买卖信号
        if 'v_reversal' in indicator_data.columns and 'decline_rate' in indicator_data.columns and 'rebound_rate' in indicator_data.columns:
            v_reversal_mask = indicator_data['v_reversal']
            decline_rate = indicator_data['decline_rate']
            rebound_rate = indicator_data['rebound_rate']
            
            # 看涨V形反转（下跌后反弹）
            bullish_v = v_reversal_mask & (decline_rate < 0) & (rebound_rate > 0)
            signals.loc[bullish_v, 'buy_signal'] = True
            signals.loc[bullish_v, 'neutral_signal'] = False
            signals.loc[bullish_v, 'trend'] = 1
            signals.loc[bullish_v, 'signal_type'] = '看涨V形反转'
            signals.loc[bullish_v, 'signal_desc'] = '价格经历明显下跌后出现V形反弹'
            
            # 看跌倒V形反转（上涨后回落）
            bearish_v = v_reversal_mask & (decline_rate > 0) & (rebound_rate < 0)
            signals.loc[bearish_v, 'sell_signal'] = True
            signals.loc[bearish_v, 'neutral_signal'] = False
            signals.loc[bearish_v, 'trend'] = -1
            signals.loc[bearish_v, 'signal_type'] = '看跌倒V形反转'
            signals.loc[bearish_v, 'signal_desc'] = '价格经历明显上涨后出现V形回落'
        
        # 计算信号强度并更新置信度
        for i in range(len(signals)):
            score = signals['score'].iloc[i]
            
            # 基于评分确定置信度
            if score >= 80 or score <= 20:
                signals.loc[signals.index[i], 'confidence'] = 80  # 极端区域的信号置信度高
            elif score >= 70 or score <= 30:
                signals.loc[signals.index[i], 'confidence'] = 70  # 明显区域的信号置信度较高
            elif score >= 60 or score <= 40:
                signals.loc[signals.index[i], 'confidence'] = 60  # 轻微区域的信号置信度一般
            
            # 基于评分确定风险等级
            if score >= 80 or score <= 20:
                signals.loc[signals.index[i], 'risk_level'] = '低'  # 极强信号风险较低
            elif score >= 70 or score <= 30:
                signals.loc[signals.index[i], 'risk_level'] = '中'  # 明显信号风险一般
            else:
                signals.loc[signals.index[i], 'risk_level'] = '高'  # 弱信号风险较高
            
            # 基于评分确定建议仓位
            if signals['buy_signal'].iloc[i]:
                if score >= 80:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif score >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif score >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
            elif signals['sell_signal'].iloc[i]:
                if score <= 20:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif score <= 30:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif score <= 40:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
        
        # 计算动态止损
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # 计算ATR
            atr = self.atr(data['high'], data['low'], data['close'], 14)
            
            for i in range(len(signals)):
                if signals['buy_signal'].iloc[i] and i < len(data):
                    # 买入信号的止损：当前价格 - ATR倍数
                    signals.loc[signals.index[i], 'stop_loss'] = data['close'].iloc[i] - 2 * atr.iloc[i]
                
                elif signals['sell_signal'].iloc[i] and i < len(data):
                    # 卖出信号的止损：当前价格 + ATR倍数
                    signals.loc[signals.index[i], 'stop_loss'] = data['close'].iloc[i] + 2 * atr.iloc[i]
        
        # 成交量确认
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma5 = volume.rolling(window=5).mean()
            vol_ratio = volume / vol_ma5
            
            # 成交量放大确认
            high_volume = vol_ratio > 1.5
            signals.loc[high_volume, 'volume_confirmation'] = True
        
        # 检测市场环境
        try:
            market_env = self.detect_market_environment(data)
            signals['market_env'] = market_env.value
        except Exception as e:
            logger.warning(f"检测市场环境时出错: {e}")
        
        return signals 

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
