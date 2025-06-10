"""
ZXM洗盘形态识别模块

实现ZXM体系中的洗盘形态识别功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from enum import Enum

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class WashPlateType(Enum):
    """洗盘形态类型枚举"""
    SHOCK_WASH = "横盘震荡洗盘"           # 一定区间内的来回震荡，成交量忽大忽小
    PULLBACK_WASH = "回调洗盘"            # 短期快速回调后在重要支撑位止跌，成交量逐步萎缩
    FALSE_BREAK_WASH = "假突破洗盘"        # 向下突破重要支撑位后快速收复，突破时量能放大，收复时量能更大
    TIME_WASH = "时间洗盘"                # 价格小幅波动，但周期较长，整体呈萎缩趋势
    CONTINUOUS_YIN_WASH = "连续阴线洗盘"   # 连续3-5根中小阴线，实体不断缩小，下影线增多，量能逐步萎缩


class ZXMWashPlate(BaseIndicator):
    """
    ZXM洗盘形态识别指标
    
    识别ZXM体系中各种洗盘形态
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM洗盘形态识别指标"""
        super().__init__(name="ZXMWashPlate", description="ZXM洗盘形态识别指标，识别ZXM体系中各种洗盘形态")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        识别ZXM洗盘形态
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含各种洗盘形态的标记
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["open", "high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算横盘震荡洗盘
        result = self._calculate_shock_wash(data, result)
        
        # 计算回调洗盘
        result = self._calculate_pullback_wash(data, result)
        
        # 计算假突破洗盘
        result = self._calculate_false_break_wash(data, result)
        
        # 计算时间洗盘
        result = self._calculate_time_wash(data, result)
        
        # 计算连续阴线洗盘
        result = self._calculate_continuous_yin_wash(data, result)
        
        return result
    
    def _calculate_shock_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算横盘震荡洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_SHOCK_WASH:=(HHV(CLOSE,10)-LLV(CLOSE,10))/LLV(CLOSE,10)<0.07 AND HHV(VOL,10)/LLV(VOL,10)>2;
        """
        # 提取数据
        close = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        shock_wash = np.zeros(n, dtype=bool)
        
        # 计算横盘震荡洗盘
        window = 10
        for i in range(window, n):
            # 价格区间相对波动小于7%
            price_range = np.max(close[i-window:i]) - np.min(close[i-window:i])
            price_range_ratio = price_range / np.min(close[i-window:i])
            
            # 成交量波动大于2倍
            vol_range_ratio = np.max(volume[i-window:i]) / np.min(volume[i-window:i])
            
            if price_range_ratio < 0.07 and vol_range_ratio > 2:
                shock_wash[i] = True
        
        # 添加到结果
        result[WashPlateType.SHOCK_WASH.value] = shock_wash
        
        return result
    
    def _calculate_pullback_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算回调洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_PULLBACK_WASH:=REF(CLOSE>OPEN,1) AND CLOSE<OPEN AND LOW>MA(CLOSE,20)*0.97 AND VOL<REF(VOL,1);
        """
        # 提取数据
        open_prices = data["open"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        pullback_wash = np.zeros(n, dtype=bool)
        
        # 计算20日均线
        ma20 = np.zeros(n)
        for i in range(20, n):
            ma20[i] = np.mean(close_prices[i-20:i])
        
        # 计算回调洗盘
        for i in range(1, n):
            # 前一日阳线，当日阴线
            prev_bullish = close_prices[i-1] > open_prices[i-1]
            current_bearish = close_prices[i] < open_prices[i]
            
            # 当日最低价高于20日均线的97%
            above_support = i >= 20 and low_prices[i] > ma20[i] * 0.97
            
            # 当日成交量小于前日
            vol_decreasing = volume[i] < volume[i-1]
            
            if prev_bullish and current_bearish and above_support and vol_decreasing:
                pullback_wash[i] = True
        
        # 添加到结果
        result[WashPlateType.PULLBACK_WASH.value] = pullback_wash
        
        return result
    
    def _calculate_false_break_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算假突破洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_FALSE_BREAK:=REF(LOW<LLV(LOW,20),1) AND CLOSE>REF(CLOSE,1) AND CLOSE>REF(LOW,1)*1.02 AND VOL>REF(VOL,1);
        """
        # 提取数据
        low_prices = data["low"].values
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        false_break = np.zeros(n, dtype=bool)
        
        # 计算假突破洗盘
        window = 20
        for i in range(window+1, n):
            # 前一日最低价低于20日最低价
            prev_low_break = low_prices[i-1] < np.min(low_prices[i-window-1:i-1])
            
            # 当日收盘价高于前日收盘价
            price_recovery = close_prices[i] > close_prices[i-1]
            
            # 当日收盘价高于前日最低价的1.02倍
            strong_recovery = close_prices[i] > low_prices[i-1] * 1.02
            
            # 当日成交量大于前日
            vol_increasing = volume[i] > volume[i-1]
            
            if prev_low_break and price_recovery and strong_recovery and vol_increasing:
                false_break[i] = True
        
        # 添加到结果
        result[WashPlateType.FALSE_BREAK_WASH.value] = false_break
        
        return result
    
    def _calculate_time_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
        """
        # 提取数据
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        time_wash = np.zeros(n, dtype=bool)
        
        # 计算时间洗盘
        # 特征：价格小幅波动，成交量整体萎缩，周期较长（15天以上）
        window = 15
        for i in range(2*window, n):
            # 当前区间价格波动小
            current_range = np.max(close_prices[i-window:i]) - np.min(close_prices[i-window:i])
            current_range_ratio = current_range / np.min(close_prices[i-window:i])
            
            # 前一区间价格波动相对较大
            prev_range = np.max(close_prices[i-2*window:i-window]) - np.min(close_prices[i-2*window:i-window])
            prev_range_ratio = prev_range / np.min(close_prices[i-2*window:i-window])
            
            # 当前区间成交量整体萎缩
            current_vol_avg = np.mean(volume[i-window:i])
            prev_vol_avg = np.mean(volume[i-2*window:i-window])
            vol_shrinking = current_vol_avg < 0.8 * prev_vol_avg
            
            if current_range_ratio < 0.05 and current_range_ratio < prev_range_ratio and vol_shrinking:
                time_wash[i] = True
        
        # 添加到结果
        result[WashPlateType.TIME_WASH.value] = time_wash
        
        return result
    
    def _calculate_continuous_yin_wash(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        计算连续阴线洗盘
        
        Args:
            data: 输入数据
            result: 结果数据框
            
        Returns:
            pd.DataFrame: 更新后的结果数据框
            
        公式说明：
        ZXM_CONTINUOUS_YIN:=COUNT(CLOSE<OPEN,5)>=3 AND COUNT(MIN(OPEN,CLOSE)-LOW>ABS(CLOSE-OPEN),5)>=2 AND VOL/REF(VOL,5)<0.8;
        """
        # 提取数据
        open_prices = data["open"].values
        low_prices = data["low"].values
        close_prices = data["close"].values
        volume = data["volume"].values
        
        # 初始化结果数组
        n = len(data)
        continuous_yin = np.zeros(n, dtype=bool)
        
        # 计算连续阴线洗盘
        window = 5
        for i in range(window, n):
            # 最近5天中至少有3天是阴线
            bearish_count = np.sum(close_prices[i-window:i] < open_prices[i-window:i])
            
            # 最近5天中至少有2天下影线长于实体
            lower_shadow_count = 0
            for j in range(i-window, i):
                body_size = abs(close_prices[j] - open_prices[j])
                lower_shadow = min(open_prices[j], close_prices[j]) - low_prices[j]
                if lower_shadow > body_size:
                    lower_shadow_count += 1
            
            # 当日成交量低于5日前的80%
            vol_shrinking = volume[i] < 0.8 * volume[i-window]
            
            if bearish_count >= 3 and lower_shadow_count >= 2 and vol_shrinking:
                continuous_yin[i] = True
        
        # 添加到结果
        result[WashPlateType.CONTINUOUS_YIN_WASH.value] = continuous_yin
        
        return result
    
    def get_recent_wash_plates(self, data: pd.DataFrame, lookback: int = 10) -> Dict[str, bool]:
        """
        获取最近的洗盘形态
        
        Args:
            data: 输入数据
            lookback: 回溯天数
            
        Returns:
            Dict[str, bool]: 最近的洗盘形态
        """
        # 计算所有洗盘形态
        result = self.calculate(data)
        
        # 截取最近的数据
        recent_result = result.iloc[-lookback:]
        
        # 提取每种形态的最新状态
        recent_wash_plates = {}
        for wash_type in WashPlateType:
            wash_name = wash_type.value
            if wash_name in recent_result.columns:
                recent_wash_plates[wash_name] = recent_result[wash_name].any()
        
        return recent_wash_plates
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM洗盘指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 检查是否有任何类型的洗盘形态
        any_wash_plate = pd.Series(False, index=data.index)
        for wash_type in WashPlateType:
            if wash_type.value in result.columns:
                any_wash_plate = any_wash_plate | result[wash_type.value]
        
        # 任何洗盘形态基础加分30分
        score[any_wash_plate] += 30
        
        # 各洗盘形态的具体评分
        # 1. 横盘震荡洗盘 - 中等强度洗盘
        if WashPlateType.SHOCK_WASH.value in result.columns:
            score[result[WashPlateType.SHOCK_WASH.value]] += 10
        
        # 2. 回调洗盘 - 较弱洗盘
        if WashPlateType.PULLBACK_WASH.value in result.columns:
            score[result[WashPlateType.PULLBACK_WASH.value]] += 5
        
        # 3. 假突破洗盘 - 最强洗盘
        if WashPlateType.FALSE_BREAK_WASH.value in result.columns:
            score[result[WashPlateType.FALSE_BREAK_WASH.value]] += 15
        
        # 4. 时间洗盘 - 强度洗盘
        if WashPlateType.TIME_WASH.value in result.columns:
            score[result[WashPlateType.TIME_WASH.value]] += 12
        
        # 5. 连续阴线洗盘 - 较弱洗盘
        if WashPlateType.CONTINUOUS_YIN_WASH.value in result.columns:
            score[result[WashPlateType.CONTINUOUS_YIN_WASH.value]] += 8
        
        # 多重洗盘形态加分
        # 计算每天满足的洗盘形态数量
        wash_count = pd.Series(0, index=data.index)
        for i in range(len(data)):
            count = 0
            for wash_type in WashPlateType:
                if wash_type.value in result.columns and result[wash_type.value].iloc[i]:
                    count += 1
            wash_count.iloc[i] = count
        
        # 多重洗盘形态加分
        score[wash_count == 2] += 10  # 两种洗盘形态
        score[wash_count >= 3] += 15  # 三种及以上洗盘形态
        
        # 洗盘后上涨确认 - 为了实现这个，我们需要检查洗盘后是否有明显上涨
        # 我们定义：如果洗盘形成后5天内有任何一天收盘价上涨超过3%，就视为洗盘后上涨确认
        for i in range(len(data) - 5):
            if any_wash_plate.iloc[i]:
                # 检查未来5天内是否有明显上涨
                future_data = data.iloc[i:i+6]
                if len(future_data) > 1:
                    current_close = future_data['close'].iloc[0]
                    future_closes = future_data['close'].iloc[1:]
                    max_gain = ((future_closes / current_close) - 1).max() * 100
                    if max_gain > 3:
                        # 洗盘后有明显上涨，额外加分
                        score.iloc[i] += 15
        
        # 技术指标配合加分
        # 1. 计算均线系统
        ma5 = data['close'].rolling(window=5).mean()
        ma10 = data['close'].rolling(window=10).mean()
        ma20 = data['close'].rolling(window=20).mean()
        
        # 价格站上短期均线加分 - 表明洗盘后开始上攻
        price_above_ma5 = data['close'] > ma5
        price_above_ma10 = data['close'] > ma10
        
        score[any_wash_plate & price_above_ma5] += 5
        score[any_wash_plate & price_above_ma10] += 5
        
        # 2. 成交量变化配合
        # 计算成交量变化率（当日成交量/5日平均量）
        volume_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
        
        # 洗盘后放量加分 - 表明洗盘完成准备上攻
        score[any_wash_plate & (volume_ratio > 1.5)] += 10
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM洗盘相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易日的形态
        patterns = []
        last_row = result.iloc[-1] if not result.empty else None
        
        if last_row is not None:
            # 检查各种洗盘形态
            for wash_type in WashPlateType:
                if wash_type.value in result.columns and last_row[wash_type.value]:
                    patterns.append(f"{wash_type.value}")
            
            # 如果没有任何洗盘形态
            if not patterns:
                patterns.append("无洗盘形态")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成标准化的信号输出
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含标准化信号的DataFrame
        """
        # 计算指标和评分
        result = self.calculate(data)
        score = self.calculate_raw_score(data, **kwargs)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 任何洗盘形态都视为买入信号
        buy_signal = pd.Series(False, index=data.index)
        for wash_type in WashPlateType:
            if wash_type.value in result.columns:
                buy_signal = buy_signal | result[wash_type.value]
        
        # 设置标准化信号
        signals['buy_signal'] = buy_signal
        signals['sell_signal'] = False  # 洗盘指标没有卖出信号
        signals['neutral_signal'] = ~buy_signal
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[buy_signal, 'trend'] = 1  # 洗盘后看涨
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[buy_signal, 'signal_type'] = 'wash_plate_buy'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 根据不同洗盘类型设置详细描述
        for wash_type in WashPlateType:
            if wash_type.value in result.columns:
                mask = result[wash_type.value]
                signals.loc[mask, 'signal_desc'] = f"{wash_type.value}形态，可考虑买入"
        
        # 置信度设置 - 根据评分高低确定
        signals['confidence'] = 60  # 基础置信度
        signals.loc[score > 70, 'confidence'] = 80
        signals.loc[score > 85, 'confidence'] = 90
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位 - 根据评分确定
        signals['position_size'] = 0.0
        signals.loc[buy_signal, 'position_size'] = 0.3  # 基础仓位
        signals.loc[(buy_signal) & (score > 70), 'position_size'] = 0.5  # 高分仓位
        signals.loc[(buy_signal) & (score > 85), 'position_size'] = 0.7  # 极高分仓位
        
        # 止损位 - 使用洗盘期间的最低价
        signals['stop_loss'] = 0.0
        for wash_type in WashPlateType:
            if wash_type.value in result.columns:
                for idx in data.index[result[wash_type.value]]:
                    # 向前查看10个交易日的最低价作为止损位
                    try:
                        day_idx = data.index.get_loc(idx)
                        if day_idx >= 10:
                            low_price = data.iloc[day_idx-10:day_idx+1]['low'].min()
                            signals.loc[idx, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
                    except:
                        continue
        
        # 市场环境和成交量确认
        signals['market_env'] = 'normal'
        signals['volume_confirmation'] = False
        
        # 检查成交量确认
        for idx in data.index[buy_signal]:
            try:
                day_idx = data.index.get_loc(idx)
                if day_idx >= 5:
                    # 当前成交量是否大于前5日平均成交量
                    current_vol = data.iloc[day_idx]['volume']
                    avg_vol = data.iloc[day_idx-5:day_idx]['volume'].mean()
                    if current_vol > avg_vol * 1.5:
                        signals.loc[idx, 'volume_confirmation'] = True
            except:
                continue
        
        return signals 