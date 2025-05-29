"""
平台突破指标模块

实现平台整理后突破的识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class BreakoutDirection(Enum):
    """突破方向枚举"""
    UP = "向上突破"      # 向上突破
    DOWN = "向下突破"    # 向下突破
    NONE = "无突破"      # 无突破


class PlatformBreakout(BaseIndicator):
    """
    平台突破指标
    
    识别价格在一定区间整理后的突破行为
    """
    
    def __init__(self, platform_period: int = 20, max_volatility: float = 0.05):
        """
        初始化平台突破指标
        
        Args:
            platform_period: 平台检测周期，默认为20天
            max_volatility: 平台最大波动率，默认为5%
        """
        super().__init__(name="PlatformBreakout", description="平台突破指标，识别价格在一定区间整理后的突破行为")
        self.platform_period = platform_period
        self.max_volatility = max_volatility
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算平台突破指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含平台和突破标记
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=data.index)
        
        # 计算平台特征
        result = self._detect_platforms(data, result)
        
        # 计算突破特征
        result = self._detect_breakouts(data, result)
        
        return result
    
    def _detect_platforms(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        检测平台
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 价格数据
        close_prices = data["close"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        
        # 初始化平台标记
        is_platform = np.zeros(len(data), dtype=bool)
        
        # 平台检测
        for i in range(self.platform_period, len(data)):
            # 计算平台区间
            period_high = np.max(high_prices[i-self.platform_period:i])
            period_low = np.min(low_prices[i-self.platform_period:i])
            
            # 计算波动率
            platform_volatility = (period_high - period_low) / period_low
            
            # 判断是否为平台
            is_platform[i] = platform_volatility <= self.max_volatility
        
        # 添加到结果
        result["is_platform"] = is_platform
        
        # 计算平台上下边界
        result["platform_upper"] = np.nan
        result["platform_lower"] = np.nan
        
        for i in range(self.platform_period, len(data)):
            if result["is_platform"].iloc[i]:
                period_high = np.max(high_prices[i-self.platform_period:i])
                period_low = np.min(low_prices[i-self.platform_period:i])
                
                result.iloc[i, result.columns.get_loc("platform_upper")] = period_high
                result.iloc[i, result.columns.get_loc("platform_lower")] = period_low
        
        # 计算平台持续天数
        result["platform_days"] = 0
        
        for i in range(1, len(data)):
            if result["is_platform"].iloc[i]:
                result.iloc[i, result.columns.get_loc("platform_days")] = result["platform_days"].iloc[i-1] + 1
        
        return result
    
    def _detect_breakouts(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        检测突破
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 价格数据
        close_prices = data["close"].values
        high_prices = data["high"].values
        low_prices = data["low"].values
        volumes = data["volume"].values
        
        # 初始化突破标记
        up_breakout = np.zeros(len(data), dtype=bool)
        down_breakout = np.zeros(len(data), dtype=bool)
        
        # 突破检测
        for i in range(self.platform_period + 1, len(data)):
            # 只在平台形成后检测突破
            if result["platform_days"].iloc[i-1] >= self.platform_period / 2:
                # 向上突破：收盘价突破平台上边界，且成交量放大
                if (close_prices[i] > result["platform_upper"].iloc[i-1] and 
                    volumes[i] > np.mean(volumes[i-self.platform_period:i])):
                    up_breakout[i] = True
                
                # 向下突破：收盘价跌破平台下边界，且成交量放大
                elif (close_prices[i] < result["platform_lower"].iloc[i-1] and 
                      volumes[i] > np.mean(volumes[i-self.platform_period:i])):
                    down_breakout[i] = True
        
        # 添加到结果
        result["up_breakout"] = up_breakout
        result["down_breakout"] = down_breakout
        
        # 突破方向
        breakout_direction = np.array([BreakoutDirection.NONE.value] * len(data), dtype=object)
        breakout_direction[up_breakout] = BreakoutDirection.UP.value
        breakout_direction[down_breakout] = BreakoutDirection.DOWN.value
        
        result["breakout_direction"] = breakout_direction
        
        # 计算突破强度
        result["breakout_strength"] = 0.0
        
        for i in range(self.platform_period + 1, len(data)):
            if up_breakout[i]:
                # 向上突破强度：(收盘价 - 平台上边界) / 平台上边界
                strength = (close_prices[i] - result["platform_upper"].iloc[i-1]) / result["platform_upper"].iloc[i-1]
                result.iloc[i, result.columns.get_loc("breakout_strength")] = strength
            
            elif down_breakout[i]:
                # 向下突破强度：(平台下边界 - 收盘价) / 平台下边界
                strength = (result["platform_lower"].iloc[i-1] - close_prices[i]) / result["platform_lower"].iloc[i-1]
                result.iloc[i, result.columns.get_loc("breakout_strength")] = strength
        
        return result
    
    def get_signals(self, data: pd.DataFrame, min_platform_days: int = 10, 
                  min_breakout_strength: float = 0.02) -> pd.DataFrame:
        """
        生成平台突破信号
        
        Args:
            data: 输入数据，包含平台突破指标
            min_platform_days: 最小平台天数，默认为10
            min_breakout_strength: 最小突破强度，默认为2%
            
        Returns:
            pd.DataFrame: 包含突破信号的数据框
        """
        if "breakout_direction" not in data.columns:
            data = self.calculate(data)
        
        # 初始化信号列
        data["valid_up_breakout"] = False
        data["valid_down_breakout"] = False
        
        # 生成有效突破信号
        for i in range(len(data)):
            # 向上有效突破：平台天数充分 + 向上突破 + 突破强度充分
            if (data["platform_days"].iloc[i] >= min_platform_days and 
                data["breakout_direction"].iloc[i] == BreakoutDirection.UP.value and 
                data["breakout_strength"].iloc[i] >= min_breakout_strength):
                data.iloc[i, data.columns.get_loc("valid_up_breakout")] = True
            
            # 向下有效突破：平台天数充分 + 向下突破 + 突破强度充分
            elif (data["platform_days"].iloc[i] >= min_platform_days and 
                  data["breakout_direction"].iloc[i] == BreakoutDirection.DOWN.value and 
                  data["breakout_strength"].iloc[i] >= min_breakout_strength):
                data.iloc[i, data.columns.get_loc("valid_down_breakout")] = True
        
        return data 

    def calculate_raw_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算平台突破指标的原始评分
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含原始评分的DataFrame
        """
        # 计算指标值
        indicator_data = self.calculate(data)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 平台突破确认评分（±35分）
        if 'up_breakout' in indicator_data.columns and 'down_breakout' in indicator_data.columns:
            up_breakout = indicator_data['up_breakout']
            down_breakout = indicator_data['down_breakout']
            
            # 向上突破为看涨信号
            score.loc[up_breakout] += 35
            
            # 向下突破为看跌信号
            score.loc[down_breakout] -= 35
        
        # 2. 平台持续时间评分（±15分）
        if 'platform_days' in indicator_data.columns:
            platform_days = indicator_data['platform_days']
            
            for i in range(len(data)):
                if pd.notna(platform_days.iloc[i]):
                    # 每增加5天，分数增加5分，最多15分
                    days_score = min(15, int(platform_days.iloc[i] / 5) * 5)
                    
                    # 根据突破方向加减分
                    if up_breakout.iloc[i]:
                        score.iloc[i] += days_score
                    elif down_breakout.iloc[i]:
                        score.iloc[i] -= days_score
        
        # 3. 突破幅度评分（±10分）
        if 'breakout_strength' in indicator_data.columns:
            breakout_strength = indicator_data['breakout_strength']
            
            for i in range(len(data)):
                if pd.notna(breakout_strength.iloc[i]) and breakout_strength.iloc[i] > 0:
                    # 突破幅度越大，评分越高，最多10分
                    strength_score = min(10, breakout_strength.iloc[i] * 100)
                    
                    # 根据突破方向加减分
                    if up_breakout.iloc[i]:
                        score.iloc[i] += strength_score
                    elif down_breakout.iloc[i]:
                        score.iloc[i] -= strength_score
        
        # 4. 成交量配合评分（±15分）
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma10 = volume.rolling(window=10).mean()
            vol_ratio = volume / vol_ma10
            
            for i in range(len(data)):
                if pd.notna(vol_ratio.iloc[i]) and vol_ratio.iloc[i] > 1.5:
                    # 成交量放大1.5倍以上，算作有效确认
                    if up_breakout.iloc[i]:
                        score.iloc[i] += 15
                    elif down_breakout.iloc[i]:
                        score.iloc[i] -= 15
                elif pd.notna(vol_ratio.iloc[i]) and vol_ratio.iloc[i] > 1.2:
                    # 成交量放大1.2-1.5倍，算作中等确认
                    if up_breakout.iloc[i]:
                        score.iloc[i] += 10
                    elif down_breakout.iloc[i]:
                        score.iloc[i] -= 10
        
        # 5. 回测确认评分（±10分）
        if len(data) >= 3:
            # 检查突破后的价格回测情况
            for i in range(3, len(data)):
                if up_breakout.iloc[i-3]:  # 3天前发生向上突破
                    # 获取突破前的平台上边界
                    if pd.notna(indicator_data['platform_upper'].iloc[i-3]):
                        upper_bound = indicator_data['platform_upper'].iloc[i-3]
                        
                        # 回测但未跌破平台上边界，确认突破有效
                        recent_low = min(data['low'].iloc[i-2:i+1])
                        if recent_low >= upper_bound * 0.98:  # 允许2%的误差
                            score.iloc[i] += 10
                
                elif down_breakout.iloc[i-3]:  # 3天前发生向下突破
                    # 获取突破前的平台下边界
                    if pd.notna(indicator_data['platform_lower'].iloc[i-3]):
                        lower_bound = indicator_data['platform_lower'].iloc[i-3]
                        
                        # 回测但未突破平台下边界，确认突破有效
                        recent_high = max(data['high'].iloc[i-2:i+1])
                        if recent_high <= lower_bound * 1.02:  # 允许2%的误差
                            score.iloc[i] -= 10
        
        # 6. 平台突破后持续性评分（±15分）
        if len(data) >= 5:
            # 检查突破后的价格走势是否持续
            for i in range(5, len(data)):
                if up_breakout.iloc[i-5]:  # 5天前发生向上突破
                    # 检查后续趋势是否持续上涨
                    if data['close'].iloc[i] > data['close'].iloc[i-5] * 1.03:  # 上涨超过3%
                        score.iloc[i] += 15
                    elif data['close'].iloc[i] < data['close'].iloc[i-5]:  # 实际下跌
                        score.iloc[i] -= 5  # 突破失败，降低评分
                
                elif down_breakout.iloc[i-5]:  # 5天前发生向下突破
                    # 检查后续趋势是否持续下跌
                    if data['close'].iloc[i] < data['close'].iloc[i-5] * 0.97:  # 下跌超过3%
                        score.iloc[i] -= 15
                    elif data['close'].iloc[i] > data['close'].iloc[i-5]:  # 实际上涨
                        score.iloc[i] += 5  # 突破失败，提高评分
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return pd.DataFrame({'score': score}, index=data.index) 

    def identify_patterns(self, data: pd.DataFrame) -> List[str]:
        """
        识别平台突破相关的技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 计算指标值
        indicator_data = self.calculate(data)
        
        if len(indicator_data) < 5 or 'close' not in data.columns:
            return patterns
        
        # 1. 平台形态识别
        if 'is_platform' in indicator_data.columns:
            # 检查当前是否处于平台期
            if indicator_data['is_platform'].iloc[-1]:
                # 获取平台持续天数
                if 'platform_days' in indicator_data.columns:
                    days = indicator_data['platform_days'].iloc[-1]
                    if days >= 30:
                        patterns.append(f"长期平台期-{days}天")
                    elif days >= 15:
                        patterns.append(f"中期平台期-{days}天")
                    else:
                        patterns.append(f"短期平台期-{days}天")
                
                # 平台区间大小
                if ('platform_upper' in indicator_data.columns and 
                    'platform_lower' in indicator_data.columns):
                    upper = indicator_data['platform_upper'].iloc[-1]
                    lower = indicator_data['platform_lower'].iloc[-1]
                    
                    if pd.notna(upper) and pd.notna(lower) and lower > 0:
                        range_percent = (upper - lower) / lower * 100
                        patterns.append(f"平台波动幅度-{range_percent:.2f}%")
        
        # 2. 突破形态识别
        breakout_detected = False
        
        if 'up_breakout' in indicator_data.columns:
            # 检查最近5天内是否有向上突破
            recent_up_breakouts = indicator_data['up_breakout'].tail(5).sum()
            if recent_up_breakouts > 0:
                breakout_detected = True
                patterns.append("向上突破形态")
                
                # 突破强度分析
                if 'breakout_strength' in indicator_data.columns:
                    for i in range(1, 6):
                        if i <= len(indicator_data) and indicator_data['up_breakout'].iloc[-i]:
                            strength = indicator_data['breakout_strength'].iloc[-i]
                            if pd.notna(strength):
                                if strength >= 0.05:
                                    patterns.append("强势向上突破")
                                elif strength >= 0.02:
                                    patterns.append("中等向上突破")
                                else:
                                    patterns.append("弱势向上突破")
                            break
        
        if 'down_breakout' in indicator_data.columns:
            # 检查最近5天内是否有向下突破
            recent_down_breakouts = indicator_data['down_breakout'].tail(5).sum()
            if recent_down_breakouts > 0:
                breakout_detected = True
                patterns.append("向下突破形态")
                
                # 突破强度分析
                if 'breakout_strength' in indicator_data.columns:
                    for i in range(1, 6):
                        if i <= len(indicator_data) and indicator_data['down_breakout'].iloc[-i]:
                            strength = indicator_data['breakout_strength'].iloc[-i]
                            if pd.notna(strength):
                                if strength >= 0.05:
                                    patterns.append("强势向下突破")
                                elif strength >= 0.02:
                                    patterns.append("中等向下突破")
                                else:
                                    patterns.append("弱势向下突破")
                            break
        
        # 如果没有检测到突破，分析是否接近突破
        if not breakout_detected and 'is_platform' in indicator_data.columns:
            if indicator_data['is_platform'].iloc[-1]:
                if ('platform_upper' in indicator_data.columns and 
                    'platform_lower' in indicator_data.columns and
                    'close' in data.columns):
                    
                    upper = indicator_data['platform_upper'].iloc[-1]
                    lower = indicator_data['platform_lower'].iloc[-1]
                    latest_close = data['close'].iloc[-1]
                    
                    if pd.notna(upper) and pd.notna(lower) and pd.notna(latest_close):
                        upper_distance = (upper - latest_close) / latest_close
                        lower_distance = (latest_close - lower) / latest_close
                        
                        if upper_distance <= 0.01:  # 接近上边界1%以内
                            patterns.append("接近上边界突破")
                        elif lower_distance <= 0.01:  # 接近下边界1%以内
                            patterns.append("接近下边界突破")
        
        # 3. 成交量分析
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma10 = volume.rolling(window=10).mean()
            latest_vol_ratio = (volume / vol_ma10).iloc[-1]
            
            if pd.notna(latest_vol_ratio):
                if latest_vol_ratio >= 2.0:
                    patterns.append("突破成交量异常放大")
                elif latest_vol_ratio >= 1.5:
                    patterns.append("突破成交量明显放大")
                elif latest_vol_ratio <= 0.5:
                    patterns.append("成交量明显萎缩")
        
        # 4. 平台突破后表现分析
        if len(data) >= 5:
            for i in range(5, 0, -1):
                if i < len(indicator_data):
                    if indicator_data['up_breakout'].iloc[-i]:
                        # 分析向上突破后的表现
                        post_breakout_change = (data['close'].iloc[-1] - data['close'].iloc[-i]) / data['close'].iloc[-i]
                        
                        if post_breakout_change >= 0.05:
                            patterns.append("向上突破后强势上涨")
                        elif post_breakout_change >= 0.02:
                            patterns.append("向上突破后温和上涨")
                        elif post_breakout_change <= -0.02:
                            patterns.append("向上突破后反向下跌")
                        else:
                            patterns.append("向上突破后盘整")
                        break
                    
                    elif indicator_data['down_breakout'].iloc[-i]:
                        # 分析向下突破后的表现
                        post_breakout_change = (data['close'].iloc[-1] - data['close'].iloc[-i]) / data['close'].iloc[-i]
                        
                        if post_breakout_change <= -0.05:
                            patterns.append("向下突破后加速下跌")
                        elif post_breakout_change <= -0.02:
                            patterns.append("向下突破后温和下跌")
                        elif post_breakout_change >= 0.02:
                            patterns.append("向下突破后反向上涨")
                        else:
                            patterns.append("向下突破后盘整")
                        break
        
        # 5. 平台整理形态分类
        if 'is_platform' in indicator_data.columns and indicator_data['is_platform'].sum() > 0:
            # 分析平台与趋势的关系
            if len(data) >= 30:
                ma20 = data['close'].rolling(window=20).mean()
                ma50 = data['close'].rolling(window=50).mean()
                
                if pd.notna(ma20.iloc[-1]) and pd.notna(ma50.iloc[-1]):
                    if ma20.iloc[-1] > ma50.iloc[-1] * 1.05:
                        patterns.append("上升趋势中的平台整理")
                    elif ma20.iloc[-1] < ma50.iloc[-1] * 0.95:
                        patterns.append("下降趋势中的平台整理")
                    else:
                        patterns.append("横盘趋势中的平台整理")
        
        # 6. 平台形态的常见演变
        if 'platform_days' in indicator_data.columns and len(data) >= 20:
            max_platform_days = indicator_data['platform_days'].max()
            if max_platform_days >= 20:
                patterns.append("持久平台整理")
                
                if 'up_breakout' in indicator_data.columns and indicator_data['up_breakout'].sum() > 0:
                    patterns.append("长期平台上行突破")
                elif 'down_breakout' in indicator_data.columns and indicator_data['down_breakout'].sum() > 0:
                    patterns.append("长期平台下行突破")
        
        return patterns 

    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成平台突破指标信号
        
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
        
        # 基于平台突破信号生成买卖信号
        if 'up_breakout' in indicator_data.columns and 'down_breakout' in indicator_data.columns:
            up_breakout = indicator_data['up_breakout']
            down_breakout = indicator_data['down_breakout']
            
            # 向上突破生成买入信号
            signals.loc[up_breakout, 'buy_signal'] = True
            signals.loc[up_breakout, 'neutral_signal'] = False
            signals.loc[up_breakout, 'trend'] = 1
            signals.loc[up_breakout, 'signal_type'] = '平台向上突破'
            signals.loc[up_breakout, 'signal_desc'] = '价格突破平台上边界'
            
            # 向下突破生成卖出信号
            signals.loc[down_breakout, 'sell_signal'] = True
            signals.loc[down_breakout, 'neutral_signal'] = False
            signals.loc[down_breakout, 'trend'] = -1
            signals.loc[down_breakout, 'signal_type'] = '平台向下突破'
            signals.loc[down_breakout, 'signal_desc'] = '价格跌破平台下边界'
        
        # 增强信号的可靠性判断
        if 'platform_days' in indicator_data.columns:
            platform_days = indicator_data['platform_days']
            
            for i in range(len(signals)):
                # 平台持续时间越长，信号越可靠
                if signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]:
                    days = platform_days.iloc[i]
                    
                    if pd.notna(days):
                        if days >= 30:  # 长期平台突破
                            signals.loc[signals.index[i], 'confidence'] = 80
                        elif days >= 20:  # 中期平台突破
                            signals.loc[signals.index[i], 'confidence'] = 70
                        elif days >= 10:  # 短期平台突破
                            signals.loc[signals.index[i], 'confidence'] = 60
        
        # 成交量确认
        if 'volume' in data.columns:
            volume = data['volume']
            vol_ma10 = volume.rolling(window=10).mean()
            vol_ratio = volume / vol_ma10
            
            # 成交量放大确认
            high_volume = vol_ratio > 1.5
            signals.loc[high_volume, 'volume_confirmation'] = True
            
            # 成交量确认增强信号可靠性
            for i in range(len(signals)):
                if (signals['buy_signal'].iloc[i] or signals['sell_signal'].iloc[i]) and signals['volume_confirmation'].iloc[i]:
                    current_confidence = signals['confidence'].iloc[i]
                    signals.loc[signals.index[i], 'confidence'] = min(90, current_confidence + 10)
        
        # 更新风险等级和仓位建议
        for i in range(len(signals)):
            score = signals['score'].iloc[i]
            confidence = signals['confidence'].iloc[i]
            
            # 根据信号强度和置信度设置风险等级
            if (score >= 80 or score <= 20) and confidence >= 70:
                signals.loc[signals.index[i], 'risk_level'] = '低'
            elif (score >= 70 or score <= 30) and confidence >= 60:
                signals.loc[signals.index[i], 'risk_level'] = '中'
            else:
                signals.loc[signals.index[i], 'risk_level'] = '高'
            
            # 设置建议仓位
            if signals['buy_signal'].iloc[i]:
                if score >= 80 and confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif score >= 70 and confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif score >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
            elif signals['sell_signal'].iloc[i]:
                if score <= 20 and confidence >= 70:
                    signals.loc[signals.index[i], 'position_size'] = 0.1  # 10%仓位
                elif score <= 30 and confidence >= 60:
                    signals.loc[signals.index[i], 'position_size'] = 0.07  # 7%仓位
                elif score <= 40:
                    signals.loc[signals.index[i], 'position_size'] = 0.05  # 5%仓位
        
        # 计算动态止损
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            # 动态止损计算
            for i in range(len(signals)):
                if signals['buy_signal'].iloc[i] and i < len(data):
                    # 买入信号的止损：取平台下边界或当前价格-2ATR的较小值
                    if 'platform_lower' in indicator_data.columns and pd.notna(indicator_data['platform_lower'].iloc[i]):
                        platform_lower = indicator_data['platform_lower'].iloc[i]
                        signals.loc[signals.index[i], 'stop_loss'] = platform_lower * 0.98  # 平台下边界下方2%
                
                elif signals['sell_signal'].iloc[i] and i < len(data):
                    # 卖出信号的止损：取平台上边界或当前价格+2ATR的较大值
                    if 'platform_upper' in indicator_data.columns and pd.notna(indicator_data['platform_upper'].iloc[i]):
                        platform_upper = indicator_data['platform_upper'].iloc[i]
                        signals.loc[signals.index[i], 'stop_loss'] = platform_upper * 1.02  # 平台上边界上方2%
        
        # 市场环境检测
        if len(data) >= 50:
            ma20 = data['close'].rolling(window=20).mean()
            ma50 = data['close'].rolling(window=50).mean()
            
            if pd.notna(ma20.iloc[-1]) and pd.notna(ma50.iloc[-1]):
                if ma20.iloc[-1] > ma50.iloc[-1] * 1.05:
                    signals['market_env'] = 'bull_market'
                elif ma20.iloc[-1] < ma50.iloc[-1] * 0.95:
                    signals['market_env'] = 'bear_market'
                else:
                    signals['market_env'] = 'sideways_market'
        
        return signals
    
    def atr(self, high, low, close, period=14):
        """计算ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr 