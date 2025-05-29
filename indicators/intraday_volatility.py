#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日内波动率指标模块

实现日内波动率计算功能，用于评估价格日内波动幅度与开盘价的关系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class IntradayVolatility(BaseIndicator):
    """
    日内波动率指标
    
    计算日内波动范围与开盘价的比值，评估市场波动性
    """
    
    def __init__(self, smooth_period: int = 5):
        """
        初始化日内波动率指标
        
        Args:
            smooth_period: 平滑周期，默认为5日
        """
        super().__init__(name="IntradayVolatility", description="日内波动率指标，评估价格日内波动幅度")
        self.smooth_period = smooth_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算日内波动率指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含日内波动率指标值
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算日内波动范围
        intraday_range = data["high"] - data["low"]
        
        # 计算相对于开盘价的波动率
        result["volatility"] = intraday_range / data["open"] * 100
        
        # 计算波动率的移动平均
        result["volatility_ma"] = result["volatility"].rolling(window=self.smooth_period).mean()
        
        # 计算相对波动率（当前波动率与平均波动率的比值）
        result["relative_volatility"] = result["volatility"] / result["volatility_ma"]
        
        # 计算波动率变化率
        result["volatility_change"] = result["volatility"].pct_change()
        
        return result
    
    def get_signals(self, data: pd.DataFrame, high_threshold: float = 1.5, 
                   low_threshold: float = 0.5) -> pd.DataFrame:
        """
        生成波动率信号
        
        Args:
            data: 输入数据，包含日内波动率指标
            high_threshold: 高波动率阈值，默认为1.5
            low_threshold: 低波动率阈值，默认为0.5
            
        Returns:
            pd.DataFrame: 包含波动率信号的数据框
        """
        if "volatility" not in data.columns:
            data = self.calculate(data)
        
        # 初始化波动率状态列
        data["volatility_state"] = "正常"
        
        # 高波动状态
        high_volatility = data["relative_volatility"] > high_threshold
        data.loc[high_volatility, "volatility_state"] = "高波动"
        
        # 低波动状态
        low_volatility = data["relative_volatility"] < low_threshold
        data.loc[low_volatility, "volatility_state"] = "低波动"
        
        # 初始化信号列
        data["volatility_signal"] = 0
        
        # 使用向量化操作计算信号
        # 计算当前和前一天的高波动条件
        current_high = data["volatility"] > data["volatility_ma"] * high_threshold
        prev_high = current_high.shift(1).fillna(False)
        
        # 计算当前和前一天的低波动条件
        current_low = data["volatility"] < data["volatility_ma"] * low_threshold
        prev_low = current_low.shift(1).fillna(False)
        
        # 波动率突然上升：当前高波动但前一天不是
        up_signal = current_high & ~prev_high
        
        # 波动率突然下降：当前低波动但前一天不是
        down_signal = current_low & ~prev_low
        
        # 应用信号
        data.loc[up_signal, "volatility_signal"] = 1
        data.loc[down_signal, "volatility_signal"] = -1
        
        return data
    
    def get_volatility_trend(self, data: pd.DataFrame, trend_period: int = 10) -> pd.DataFrame:
        """
        分析波动率趋势
        
        Args:
            data: 输入数据，包含日内波动率指标
            trend_period: 趋势分析周期，默认为10日
            
        Returns:
            pd.DataFrame: 包含波动率趋势的数据框
        """
        if "volatility" not in data.columns:
            data = self.calculate(data)
        
        # 计算波动率趋势 - 使用rolling应用线性回归
        def calc_slope(y):
            if len(y) < trend_period or y.isna().any():
                return np.nan
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # 使用向量化操作计算趋势斜率
        data["volatility_trend"] = data["volatility"].rolling(window=trend_period).apply(
            calc_slope, raw=True
        )
        
        # 使用向量化操作对趋势进行分类
        data["trend_category"] = pd.NA
        
        # 根据斜率值分类趋势
        trend = data["volatility_trend"]
        data.loc[trend > 0.1, "trend_category"] = "强上升"
        data.loc[(trend > 0.01) & (trend <= 0.1), "trend_category"] = "上升"
        data.loc[(trend >= -0.01) & (trend <= 0.01), "trend_category"] = "平稳"
        data.loc[(trend < -0.01) & (trend >= -0.1), "trend_category"] = "下降"
        data.loc[trend < -0.1, "trend_category"] = "强下降"
        
        return data
    
    def get_market_phase(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        分析市场阶段
        
        Args:
            data: 输入数据，包含日内波动率指标
            
        Returns:
            pd.DataFrame: 包含市场阶段的数据框
        """
        if "volatility" not in data.columns:
            data = self.calculate(data)
        
        # 初始化市场阶段列
        data["market_phase"] = np.nan
        
        # 结合波动率和价格趋势分析市场阶段
        # 计算价格短期趋势（20日移动平均线方向）
        if "close" in data.columns:
            data["price_ma20"] = data["close"].rolling(window=20).mean()
            data["price_trend"] = np.nan
            
            for i in range(20, len(data)):
                if data["close"].iloc[i] > data["price_ma20"].iloc[i]:
                    data.iloc[i, data.columns.get_loc("price_trend")] = 1  # 上升趋势
                else:
                    data.iloc[i, data.columns.get_loc("price_trend")] = -1  # 下降趋势
            
            # 基于波动率和价格趋势确定市场阶段
            for i in range(20, len(data)):
                if pd.notna(data["price_trend"].iloc[i]) and pd.notna(data["relative_volatility"].iloc[i]):
                    price_trend = data["price_trend"].iloc[i]
                    rel_vol = data["relative_volatility"].iloc[i]
                    
                    if price_trend > 0 and rel_vol > 1.2:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "强势上涨"
                    elif price_trend > 0 and rel_vol < 0.8:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "稳步上涨"
                    elif price_trend < 0 and rel_vol > 1.2:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "恐慌下跌"
                    elif price_trend < 0 and rel_vol < 0.8:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "缓慢下跌"
                    else:
                        data.iloc[i, data.columns.get_loc("market_phase")] = "盘整"
        
        return data 
    
    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算日内波动率指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 波动率水平评分（-20到+20分）
        volatility = indicator_data['volatility']
        volatility_ma = indicator_data['volatility_ma']
        
        # 高波动率减分，低波动率加分
        high_vol_mask = volatility > volatility_ma * 1.5
        score.loc[high_vol_mask] -= 15
        
        very_high_vol_mask = volatility > volatility_ma * 2.0
        score.loc[very_high_vol_mask] -= 20
        
        low_vol_mask = volatility < volatility_ma * 0.7
        score.loc[low_vol_mask] += 15
        
        very_low_vol_mask = volatility < volatility_ma * 0.5
        score.loc[very_low_vol_mask] += 20
        
        # 2. 波动率趋势评分（-15到+15分）
        volatility_change = indicator_data['volatility_change'].fillna(0)
        
        # 波动率上升减分，下降加分
        vol_rising_mask = volatility_change > 0.1
        score.loc[vol_rising_mask] -= 10
        
        vol_falling_mask = volatility_change < -0.1
        score.loc[vol_falling_mask] += 10
        
        # 急剧变化额外评分
        sharp_rise_mask = volatility_change > 0.3
        score.loc[sharp_rise_mask] -= 15
        
        sharp_fall_mask = volatility_change < -0.3
        score.loc[sharp_fall_mask] += 15
        
        # 3. 相对波动率评分（-15到+15分）
        relative_volatility = indicator_data['relative_volatility'].fillna(1.0)
        
        # 相对历史水平的评分
        high_rel_vol_mask = relative_volatility > 1.3
        score.loc[high_rel_vol_mask] -= 12
        
        low_rel_vol_mask = relative_volatility < 0.7
        score.loc[low_rel_vol_mask] += 12
        
        # 极值情况
        extreme_high_mask = relative_volatility > 2.0
        score.loc[extreme_high_mask] -= 15
        
        extreme_low_mask = relative_volatility < 0.3
        score.loc[extreme_low_mask] += 15
        
        # 4. 波动率稳定性评分（-10到+10分）
        vol_std = volatility.rolling(window=10).std().fillna(0)
        vol_mean = volatility.rolling(window=10).mean().fillna(volatility)
        
        # 计算变异系数
        cv = vol_std / vol_mean
        cv = cv.fillna(0)
        
        # 稳定性好加分，不稳定减分
        stable_mask = cv < 0.3
        score.loc[stable_mask] += 8
        
        unstable_mask = cv > 0.7
        score.loc[unstable_mask] -= 8
        
        very_unstable_mask = cv > 1.0
        score.loc[very_unstable_mask] -= 10
        
        # 5. 市场阶段评分（-10到+10分）
        # 获取市场阶段信息
        try:
            phase_data = self.get_market_phase(data.copy())
            if 'market_phase' in phase_data.columns:
                market_phase = phase_data['market_phase']
                
                # 稳步上涨阶段加分
                stable_rise_mask = market_phase == "稳步上涨"
                score.loc[stable_rise_mask] += 10
                
                # 恐慌下跌阶段减分
                panic_fall_mask = market_phase == "恐慌下跌"
                score.loc[panic_fall_mask] -= 10
                
                # 强势上涨阶段中性（波动率高但趋势好）
                strong_rise_mask = market_phase == "强势上涨"
                score.loc[strong_rise_mask] += 5
        except Exception as e:
            # 如果市场阶段计算失败，跳过这部分评分
            pass
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index)
    
    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别日内波动率相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 20:
            return patterns
        
        # 获取最新数据
        latest_vol = indicator_data['volatility'].iloc[-1]
        latest_vol_ma = indicator_data['volatility_ma'].iloc[-1]
        latest_rel_vol = indicator_data['relative_volatility'].iloc[-1]
        latest_vol_change = indicator_data['volatility_change'].iloc[-1]
        
        # 1. 高波动形态
        if pd.notna(latest_rel_vol) and latest_rel_vol > 1.5:
            if latest_rel_vol > 2.0:
                patterns.append("极高波动率")
            else:
                patterns.append("高波动率")
        
        # 2. 低波动形态
        if pd.notna(latest_rel_vol) and latest_rel_vol < 0.7:
            if latest_rel_vol < 0.3:
                patterns.append("极低波动率")
            else:
                patterns.append("低波动率")
        
        # 3. 波动率趋势形态
        if pd.notna(latest_vol_change):
            if latest_vol_change > 0.3:
                patterns.append("波动率急剧上升")
            elif latest_vol_change > 0.1:
                patterns.append("波动率上升")
            elif latest_vol_change < -0.3:
                patterns.append("波动率急剧下降")
            elif latest_vol_change < -0.1:
                patterns.append("波动率下降")
        
        # 4. 波动率变化形态
        recent_vol = indicator_data['volatility'].tail(5)
        if len(recent_vol) >= 5:
            vol_trend = recent_vol.iloc[-1] - recent_vol.iloc[0]
            if vol_trend > recent_vol.mean() * 0.2:
                patterns.append("波动率持续上升")
            elif vol_trend < -recent_vol.mean() * 0.2:
                patterns.append("波动率持续下降")
        
        # 5. 市场阶段形态
        phase_data = self.get_market_phase(data.copy())
        if 'market_phase' in phase_data.columns:
            latest_phase = phase_data['market_phase'].iloc[-1]
            if pd.notna(latest_phase):
                if latest_phase == "稳步上涨":
                    patterns.append("稳步上涨阶段")
                elif latest_phase == "强势上涨":
                    patterns.append("强势上涨阶段")
                elif latest_phase == "恐慌下跌":
                    patterns.append("恐慌下跌阶段")
                elif latest_phase == "缓慢下跌":
                    patterns.append("缓慢下跌阶段")
        
        # 6. 波动率极值形态
        vol_20_max = indicator_data['volatility'].tail(20).max()
        vol_20_min = indicator_data['volatility'].tail(20).min()
        
        if pd.notna(latest_vol) and pd.notna(vol_20_max) and latest_vol >= vol_20_max:
            patterns.append("20日波动率新高")
        
        if pd.notna(latest_vol) and pd.notna(vol_20_min) and latest_vol <= vol_20_min:
            patterns.append("20日波动率新低")
        
        # 7. 波动率收敛/发散形态
        if len(indicator_data) >= 10:
            recent_vol_std = indicator_data['volatility'].tail(10).std()
            prev_vol_std = indicator_data['volatility'].tail(20).head(10).std()
            
            if pd.notna(recent_vol_std) and pd.notna(prev_vol_std):
                if recent_vol_std < prev_vol_std * 0.7:
                    patterns.append("波动率收敛")
                elif recent_vol_std > prev_vol_std * 1.3:
                    patterns.append("波动率发散")
        
        return patterns 