"""
ZXM体系趋势识别指标模块

实现ZXM体系的7个趋势识别指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from scipy.stats import linregress

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger
from indicators.score_manager import IndicatorScoreManager

logger = get_logger(__name__)


class ZXMDailyTrendUp(BaseIndicator):
    """
    ZXM趋势-日线上移指标
    
    判断60日或120日均线是否向上移动
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-日线上移指标"""
        super().__init__(name="ZXMDailyTrendUp", description="ZXM趋势-日线上移指标，判断日线均线是否向上")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-日线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        j1:MA(C,60)>=REF(MA(C,60),1);
        j2:MA(C,120)>=REF(MA(C,120),1);
        xg:j1 OR j2;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算60日均线和120日均线
        ma60 = data["close"].rolling(window=60).mean()
        ma120 = data["close"].rolling(window=120).mean()
        
        # 计算均线是否上移
        j1 = ma60 >= ma60.shift(1)
        j2 = ma120 >= ma120.shift(1)
        
        # 计算趋势信号
        xg = j1 | j2
        
        # 添加计算结果到数据框
        result["MA60"] = ma60
        result["MA120"] = ma120
        result["J1"] = j1
        result["J2"] = j2
        result["XG"] = xg
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算日线上移指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. 均线上移信号(XG)：趋势向好，+25分
        score[result["XG"]] += 25
        
        # 2. 均线趋势强度评分
        # 如果价格位于均线上方，额外加分
        price_above_ma60 = data["close"] > result["MA60"]
        price_above_ma120 = data["close"] > result["MA120"]
        
        # 价格位于60日均线上方加分
        score[price_above_ma60] += 5
        
        # 价格位于120日均线上方加分
        score[price_above_ma120] += 10
        
        # 3. 连续性评分
        # 连续多日均线上移，表示趋势较强
        continuous_up_days = pd.Series(0, index=data.index)
        for i in range(5, len(data)):
            if all(result["XG"].iloc[i-5:i+1]):
                continuous_up_days.iloc[i] = 5
            elif all(result["XG"].iloc[i-3:i+1]):
                continuous_up_days.iloc[i] = 3
            elif all(result["XG"].iloc[i-1:i+1]):
                continuous_up_days.iloc[i] = 1
        
        # 根据连续上移天数加分
        score[continuous_up_days == 1] += 5
        score[continuous_up_days == 3] += 10
        score[continuous_up_days == 5] += 15
        
        # 4. 双均线共振评分
        # 60日均线和120日均线同时上移，信号更强
        both_up = result["J1"] & result["J2"]
        score[both_up] += 10
        
        # 5. 均线角度评分
        # 计算均线斜率
        ma60_slope = result["MA60"].diff(5) / result["MA60"].shift(5)
        ma120_slope = result["MA120"].diff(5) / result["MA120"].shift(5)
        
        # 根据斜率大小加分，斜率越大上升越快
        score[ma60_slope > 0.02] += 5  # 周涨幅>2%
        score[ma60_slope > 0.05] += 5  # 周涨幅>5%
        
        score[ma120_slope > 0.01] += 5  # 周涨幅>1%
        score[ma120_slope > 0.03] += 5  # 周涨幅>3%
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别日线上移指标相关的技术形态
        
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
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["XG"]:
                if last_row["J1"] and last_row["J2"]:
                    patterns.append("双均线上移")
                elif last_row["J1"]:
                    patterns.append("60日均线上移")
                elif last_row["J2"]:
                    patterns.append("120日均线上移")
            else:
                patterns.append("均线趋势走平或下移")
            
            # 均线位置关系
            if data["close"].iloc[-1] > last_row["MA60"] > last_row["MA120"]:
                patterns.append("价格站上双均线，多头排列")
            elif last_row["MA60"] > last_row["MA120"] and data["close"].iloc[-1] < last_row["MA60"]:
                patterns.append("均线多头排列，价格回踩60日线")
            elif last_row["MA60"] < last_row["MA120"] and data["close"].iloc[-1] < last_row["MA60"]:
                patterns.append("均线空头排列，价格在均线下方")
            
            # 近期趋势变化
            if len(result) >= 10:
                if not result["XG"].iloc[-10:-5].any() and result["XG"].iloc[-5:].all():
                    patterns.append("趋势由弱转强")
                elif result["XG"].iloc[-10:-5].all() and not result["XG"].iloc[-5:].any():
                    patterns.append("趋势由强转弱")
        
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
        
        # 设置买卖信号
        signals['buy_signal'] = result["XG"]
        signals['sell_signal'] = ~result["XG"]
        signals['neutral_signal'] = False
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["XG"], 'signal_type'] = 'daily_ma_up'
        signals.loc[~result["XG"], 'signal_type'] = 'daily_ma_down'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        signals.loc[result["XG"] & result["J1"] & result["J2"], 'signal_desc'] = '双均线上移，趋势强劲'
        signals.loc[result["XG"] & result["J1"] & ~result["J2"], 'signal_desc'] = '60日均线上移，中期趋势向好'
        signals.loc[result["XG"] & ~result["J1"] & result["J2"], 'signal_desc'] = '120日均线上移，长期趋势向好'
        signals.loc[~result["XG"], 'signal_desc'] = '均线不上移，趋势走弱'
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        # 双均线上移，置信度更高
        signals.loc[result["XG"] & result["J1"] & result["J2"], 'confidence'] = 80
        # 评分高的信号，置信度更高
        signals.loc[score > 70, 'confidence'] = 75
        signals.loc[score > 85, 'confidence'] = 90
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[result["XG"], 'position_size'] = 0.3  # 基础仓位
        signals.loc[(result["XG"]) & (score > 70), 'position_size'] = 0.5  # 高分仓位
        signals.loc[(result["XG"]) & (score > 85), 'position_size'] = 0.7  # 极高分仓位
        
        # 止损位 - 使用均线作为参考
        signals['stop_loss'] = 0.0
        mask = result["XG"]
        for i in data.index[mask]:
            ma60_val = result.loc[i, "MA60"]
            close_val = data.loc[i, "close"]
            # 如果价格在均线上方，则以均线为止损位
            if close_val > ma60_val:
                signals.loc[i, 'stop_loss'] = ma60_val * 0.98  # 均线下方2%
            else:
                # 否则使用近期低点
                try:
                    idx = data.index.get_loc(i)
                    if idx >= 10:
                        low_price = data.iloc[idx-10:idx+1]['low'].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.97  # 最低点下方3%
                except:
                    continue
        
        # 市场环境和成交量确认
        signals['market_env'] = 'normal'
        signals['volume_confirmation'] = False
        
        return signals
    

class ZXMWeeklyTrendUp(BaseIndicator):
    """
    ZXM趋势-周线上移指标
    
    判断周线10周、20周或30周均线是否向上移动
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-周线上移指标"""
        super().__init__(name="ZXMWeeklyTrendUp", description="ZXM趋势-周线上移指标，判断周线均线是否向上")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周线上移指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        a1:MA(C,10)>=REF(MA(C,10),1);
        b1:MA(C,20)>=REF(MA(C,20),1);
        c1:MA(C,30)>=REF(MA(C,30),1);
        xg:a1 OR b1 OR c1;
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算10周、20周和30周均线
        ma10 = data["close"].rolling(window=10).mean()
        ma20 = data["close"].rolling(window=20).mean()
        ma30 = data["close"].rolling(window=30).mean()
        
        # 计算均线是否上移
        a1 = ma10 >= ma10.shift(1)
        b1 = ma20 >= ma20.shift(1)
        c1 = ma30 >= ma30.shift(1)
        
        # 计算趋势信号
        xg = a1 | b1 | c1
        
        # 添加计算结果到数据框
        result["MA10"] = ma10
        result["MA20"] = ma20
        result["MA30"] = ma30
        result["A1"] = a1
        result["B1"] = b1
        result["C1"] = c1
        result["XG"] = xg
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算周线上移指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. 均线上移信号(XG)：周线级别上看涨，+30分
        score[result["XG"]] += 30
        
        # 2. 不同周期均线上移组合评分
        # 三均线同时上移是最强信号
        all_up = result["A1"] & result["B1"] & result["C1"]
        score[all_up] += 15
        
        # 只有一条均线上移是较弱信号，但不同均线有不同权重
        only_ma10_up = result["A1"] & ~result["B1"] & ~result["C1"]
        only_ma20_up = ~result["A1"] & result["B1"] & ~result["C1"]
        only_ma30_up = ~result["A1"] & ~result["B1"] & result["C1"]
        
        score[only_ma10_up] += 5   # 短期均线上移
        score[only_ma20_up] += 10  # 中期均线上移
        score[only_ma30_up] += 15  # 长期均线上移
        
        # 3. 价格与均线位置关系评分
        price_above_ma10 = data["close"] > result["MA10"]
        price_above_ma20 = data["close"] > result["MA20"]
        price_above_ma30 = data["close"] > result["MA30"]
        
        # 价格在均线上方，趋势更强
        score[price_above_ma10] += 5
        score[price_above_ma20] += 8
        score[price_above_ma30] += 12
        
        # 4. 均线多头排列评分
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        ma_bear_alignment = (result["MA10"] < result["MA20"]) & (result["MA20"] < result["MA30"])
        
        score[ma_bull_alignment] += 15  # 多头排列加分
        score[ma_bear_alignment] -= 15  # 空头排列减分
        
        # 5. 连续上移评分
        # 连续多周均线上移，表示趋势较强
        continuous_up_weeks = pd.Series(0, index=data.index)
        for i in range(4, len(data)):
            if all(result["XG"].iloc[i-4:i+1]):
                continuous_up_weeks.iloc[i] = 4
            elif all(result["XG"].iloc[i-2:i+1]):
                continuous_up_weeks.iloc[i] = 2
        
        # 根据连续上移周数加分
        score[continuous_up_weeks == 2] += 5
        score[continuous_up_weeks == 4] += 15
        
        # 6. 均线角度评分
        # 计算均线斜率
        ma10_slope = result["MA10"].diff(4) / result["MA10"].shift(4)
        ma20_slope = result["MA20"].diff(4) / result["MA20"].shift(4)
        ma30_slope = result["MA30"].diff(4) / result["MA30"].shift(4)
        
        # 根据斜率大小加分，斜率越大上升越快
        score[ma10_slope > 0.03] += 5  # 月涨幅>3%
        score[ma10_slope > 0.06] += 5  # 月涨幅>6%
        
        score[ma20_slope > 0.02] += 5  # 月涨幅>2%
        score[ma20_slope > 0.04] += 5  # 月涨幅>4%
        
        score[ma30_slope > 0.01] += 5  # 月涨幅>1%
        score[ma30_slope > 0.02] += 5  # 月涨幅>2%
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别周线上移指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易周的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["XG"]:
                if last_row["A1"] and last_row["B1"] and last_row["C1"]:
                    patterns.append("三均线同时上移")
                elif last_row["A1"] and last_row["B1"]:
                    patterns.append("10周和20周均线上移")
                elif last_row["A1"] and last_row["C1"]:
                    patterns.append("10周和30周均线上移")
                elif last_row["B1"] and last_row["C1"]:
                    patterns.append("20周和30周均线上移")
                elif last_row["A1"]:
                    patterns.append("10周均线上移")
                elif last_row["B1"]:
                    patterns.append("20周均线上移")
                elif last_row["C1"]:
                    patterns.append("30周均线上移")
            else:
                patterns.append("周均线趋势走平或下移")
            
            # 均线位置关系
            close_value = data["close"].iloc[-1]
            if close_value > last_row["MA10"] > last_row["MA20"] > last_row["MA30"]:
                patterns.append("价格站上三均线，多头排列")
            elif last_row["MA10"] > last_row["MA20"] > last_row["MA30"]:
                if close_value < last_row["MA10"]:
                    patterns.append("均线多头排列，价格回踩10周线")
            elif last_row["MA10"] < last_row["MA20"] < last_row["MA30"]:
                if close_value < last_row["MA10"]:
                    patterns.append("均线空头排列，价格在均线下方")
                else:
                    patterns.append("均线空头排列，价格反弹站上10周线")
            
            # 近期趋势变化
            if len(result) >= 8:
                if not result["XG"].iloc[-8:-4].any() and result["XG"].iloc[-4:].all():
                    patterns.append("周线趋势由弱转强")
                elif result["XG"].iloc[-8:-4].all() and not result["XG"].iloc[-4:].any():
                    patterns.append("周线趋势由强转弱")
        
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
        
        # 设置买卖信号
        signals['buy_signal'] = result["XG"]
        signals['sell_signal'] = ~result["XG"]
        signals['neutral_signal'] = False
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["XG"] & result["A1"] & result["B1"] & result["C1"], 'signal_type'] = 'weekly_three_ma_up'
        signals.loc[result["XG"] & ~(result["A1"] & result["B1"] & result["C1"]), 'signal_type'] = 'weekly_some_ma_up'
        signals.loc[~result["XG"], 'signal_type'] = 'weekly_ma_down'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "A1"] and result.loc[i, "B1"] and result.loc[i, "C1"]:
                signals.loc[i, 'signal_desc'] = "三周期均线同时上移，趋势强劲"
            elif result.loc[i, "XG"]:
                up_mas = []
                if result.loc[i, "A1"]: up_mas.append("10周")
                if result.loc[i, "B1"]: up_mas.append("20周")
                if result.loc[i, "C1"]: up_mas.append("30周")
                up_mas_str = "、".join(up_mas)
                signals.loc[i, 'signal_desc'] = f"{up_mas_str}均线上移，趋势向好"
            else:
                signals.loc[i, 'signal_desc'] = "周均线不上移，趋势走弱"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据上移均线数量调整置信度
        up_ma_count = result["A1"].astype(int) + result["B1"].astype(int) + result["C1"].astype(int)
        for i in range(len(signals)):
            signals['confidence'].iloc[i] += up_ma_count.iloc[i] * 10
        
        # 均线排列影响置信度
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        signals.loc[ma_bull_alignment, 'confidence'] += 10
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        signals.loc[score >= 75, 'risk_level'] = '低'
        signals.loc[score <= 30, 'risk_level'] = '高'
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据上移均线数量和排列调整仓位
        three_ma_up = result["A1"] & result["B1"] & result["C1"]
        ma_bull_alignment = (result["MA10"] > result["MA20"]) & (result["MA20"] > result["MA30"])
        
        signals.loc[three_ma_up, 'position_size'] = 0.5  # 三均线上移，加大仓位
        signals.loc[three_ma_up & ma_bull_alignment, 'position_size'] = 0.7  # 三均线上移且多头排列，大仓位
        
        # 止损位 - 使用均线作为参考
        signals['stop_loss'] = 0.0
        for i in signals.index[signals['buy_signal']]:
            ma10_val = result.loc[i, "MA10"]
            ma20_val = result.loc[i, "MA20"]
            close_val = data.loc[i, "close"]
            
            # 如果价格在均线上方，则以均线为止损位
            if close_val > ma10_val:
                signals.loc[i, 'stop_loss'] = ma10_val * 0.95  # 10周均线下方5%
            elif close_val > ma20_val:
                signals.loc[i, 'stop_loss'] = ma20_val * 0.95  # 20周均线下方5%
            else:
                # 否则使用近期低点
                try:
                    idx = data.index.get_loc(i)
                    if idx >= 8:
                        low_price = data.iloc[idx-8:idx+1]['low'].min()
                        signals.loc[i, 'stop_loss'] = low_price * 0.95  # 最低点下方5%
                except:
                    continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        bull_market = result["XG"] & ma_bull_alignment
        bear_market = ~result["XG"] & (result["MA10"] < result["MA20"]) & (result["MA20"] < result["MA30"])
        
        signals.loc[bull_market, 'market_env'] = 'bull_market'
        signals.loc[bear_market, 'market_env'] = 'bear_market'
        signals.loc[~bull_market & ~bear_market, 'market_env'] = 'sideways_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals


class ZXMMonthlyKDJTrendUp(BaseIndicator):
    """
    ZXM趋势-月KDJ·D及K上移指标
    
    判断月线KDJ指标的D值和K值是否同时向上移动
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-月KDJ·D及K上移指标"""
        super().__init__(name="ZXMMonthlyKDJTrendUp", description="ZXM趋势-月KDJ·D及K上移指标，判断月线KDJ·D和K值是否向上")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-月KDJ·D及K上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为月线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        xg:D>=REF(D,1) AND K>=REF(K,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算趋势信号
        xg = (d >= d.shift(1)) & (k >= k.shift(1))
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """计算移动平均线，类似于通达信中的SMA函数"""
        result = pd.Series(index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算月KDJ上移指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. KDJ上移信号(XG)：月线级别上看涨，+30分
        score[result["XG"]] += 30
        
        # 2. KDJ具体数值区间评分
        # KDJ值在不同区间有不同意义
        k_value = result["K"]
        d_value = result["D"]
        
        # 低位上移更有意义
        score[(result["XG"]) & (k_value < 30)] += 15  # 低位上移，更强烈买入信号
        score[(result["XG"]) & (k_value >= 30) & (k_value < 70)] += 10  # 中位上移，中等买入信号
        score[(result["XG"]) & (k_value >= 70)] += 5  # 高位上移，弱买入信号
        
        # 超买超卖区域评分
        score[k_value < 20] += 15  # 严重超卖，可能即将反弹
        score[k_value > 80] -= 15  # 严重超买，可能即将回调
        
        # 3. KD线金叉死叉评分
        k_cross_above_d = (k_value > d_value) & (k_value.shift(1) <= d_value.shift(1))
        k_cross_below_d = (k_value < d_value) & (k_value.shift(1) >= d_value.shift(1))
        
        score[k_cross_above_d] += 20  # KD金叉，买入信号
        score[k_cross_below_d] -= 20  # KD死叉，卖出信号
        
        # 4. J线极值评分
        j_value = result["J"]
        
        score[j_value < 0] += 10  # J值低于0，超卖信号
        score[j_value > 100] -= 10  # J值高于100，超买信号
        
        # 5. 连续上移评分
        # 连续多月K值上移，表示趋势较强
        continuous_up_months = pd.Series(0, index=data.index)
        for i in range(3, len(data)):
            if all(result["XG"].iloc[i-3:i+1]):
                continuous_up_months.iloc[i] = 3
            elif all(result["XG"].iloc[i-1:i+1]):
                continuous_up_months.iloc[i] = 1
        
        # 根据连续上移月数加分
        score[continuous_up_months == 1] += 5
        score[continuous_up_months == 3] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别月KDJ上移指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易月的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["XG"]:
                patterns.append("月KDJ指标K值上移")
            else:
                patterns.append("月KDJ指标K值不上移")
            
            # KDJ值形态判断
            k_value = last_row["K"]
            d_value = last_row["D"]
            j_value = last_row["J"]
            
            # K值位置判断
            if k_value < 20:
                patterns.append("月KDJ严重超卖区域")
            elif k_value < 40:
                patterns.append("月KDJ超卖区域")
            elif k_value > 80:
                patterns.append("月KDJ严重超买区域")
            elif k_value > 60:
                patterns.append("月KDJ超买区域")
                
            # KD关系判断
            if k_value > d_value:
                # 检查是否为金叉
                if len(result) > 1:
                    if result["K"].iloc[-2] <= result["D"].iloc[-2]:
                        patterns.append("月线KDJ金叉形成")
                    else:
                        patterns.append("月线KDJ金叉后持续上行")
            else:
                # 检查是否为死叉
                if len(result) > 1:
                    if result["K"].iloc[-2] >= result["D"].iloc[-2]:
                        patterns.append("月线KDJ死叉形成")
                    else:
                        patterns.append("月线KDJ死叉后持续下行")
            
            # J值极值判断
            if j_value < 0:
                patterns.append("月KDJ-J值低于0，超卖严重")
            elif j_value > 100:
                patterns.append("月KDJ-J值高于100，超买严重")
                
            # 趋势转变判断
            if len(result) >= 6:
                if not result["XG"].iloc[-6:-3].any() and result["XG"].iloc[-3:].all():
                    patterns.append("月KDJ趋势由弱转强")
                elif result["XG"].iloc[-6:-3].all() and not result["XG"].iloc[-3:].any():
                    patterns.append("月KDJ趋势由强转弱")
        
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
        
        # 识别形态
        patterns = []
        for i in range(len(data)):
            patterns.append(self.identify_patterns(data.iloc[:i+1], **kwargs))
        
        # 设置基本信号列
        signals["buy_signal"] = False
        signals["sell_signal"] = False 
        signals["neutral_signal"] = True
        signals["trend"] = 0
        signals["score"] = score
        signals["signal_type"] = "中性"
        signals["signal_desc"] = ""
        signals["confidence"] = 0
        signals["risk_level"] = "中"
        signals["position_size"] = 0.0
        signals["stop_loss"] = 0.0
        signals["market_env"] = "未知"
        signals["volume_confirmation"] = False
        
        # 填充信号描述和置信度
        for i in range(len(signals)):
            pattern_list = patterns[i] if i < len(patterns) else []
            signals.loc[signals.index[i], "signal_desc"] = "，".join(pattern_list) if pattern_list else "无明显形态"
        
        # 生成趋势方向
        # 当指标表明K值上移时，设置为1
        signals.loc[result["XG"], "trend"] = 1
        
        # KD金叉是重要买入信号
        k_cross_above_d = (result["K"] > result["D"]) & (result["K"].shift(1) <= result["D"].shift(1))
        signals.loc[k_cross_above_d, "trend"] = 1
        
        # KD死叉是重要卖出信号
        k_cross_below_d = (result["K"] < result["D"]) & (result["K"].shift(1) >= result["D"].shift(1))
        signals.loc[k_cross_below_d, "trend"] = -1
        
        # 设置买入信号
        # 当评分大于70且KDJ有看涨形态时，产生买入信号
        buy_condition1 = (score > 70) & (result["XG"])  # K值上移
        buy_condition2 = k_cross_above_d  # KD金叉
        buy_condition3 = (result["K"] < 30) & (result["XG"])  # 低位K值上移
        
        buy_condition = buy_condition1 | buy_condition2 | buy_condition3
        signals.loc[buy_condition, "buy_signal"] = True
        signals.loc[buy_condition, "neutral_signal"] = False
        
        # 根据不同条件设置不同的信号类型
        signals.loc[buy_condition1, "signal_type"] = "买入-月KDJ K值上移"
        signals.loc[buy_condition2, "signal_type"] = "买入-月KDJ金叉"
        signals.loc[buy_condition3, "signal_type"] = "买入-月KDJ低位回升"
        
        # 设置卖出信号
        # 当评分低于30且KDJ有看跌形态时，产生卖出信号
        sell_condition1 = (score < 30) & (~result["XG"])  # K值下移
        sell_condition2 = k_cross_below_d  # KD死叉
        sell_condition3 = (result["K"] > 70) & (~result["XG"])  # 高位K值下移
        
        sell_condition = sell_condition1 | sell_condition2 | sell_condition3
        signals.loc[sell_condition, "sell_signal"] = True
        signals.loc[sell_condition, "neutral_signal"] = False
        
        # 根据不同条件设置不同的信号类型
        signals.loc[sell_condition1, "signal_type"] = "卖出-月KDJ K值下移"
        signals.loc[sell_condition2, "signal_type"] = "卖出-月KDJ死叉"
        signals.loc[sell_condition3, "signal_type"] = "卖出-月KDJ高位回落"
        
        # 计算置信度
        # 置信度基于评分、KDJ值和KD关系
        signals["confidence"] = signals["score"].apply(lambda x: min(100, x + 20) if x > 50 else max(0, x - 20))
        
        # KD关系影响置信度
        kd_relation = (result["K"] - result["D"]) / 100 * 40  # 将差值归一化，影响置信度±20
        for i in range(len(signals)):
            if pd.notna(kd_relation.iloc[i]):
                current_confidence = signals["confidence"].iloc[i]
                new_confidence = max(0, min(100, current_confidence + kd_relation.iloc[i]))
                signals["confidence"].iloc[i] = new_confidence
        
        # 月线KDJ信号更重要，置信度整体提高
        signals["confidence"] = signals["confidence"] + 10
        signals["confidence"] = signals["confidence"].clip(0, 100)
        
        # 设置风险等级
        signals.loc[signals["score"] >= 70, "risk_level"] = "低"
        signals.loc[(signals["score"] < 70) & (signals["score"] >= 40), "risk_level"] = "中"
        signals.loc[signals["score"] < 40, "risk_level"] = "高"
        
        # 设置仓位建议
        # 基于评分和风险设置仓位大小，月线信号更重要，整体仓位更大
        signals["position_size"] = signals["score"].apply(lambda x: min(1.0, max(0.0, (x - 40) / 60 * 0.8)))
        
        # 设置止损位
        # 月线级别止损应更宽松
        for i in range(len(signals)):
            if i < 6:  # 数据不足，使用默认值
                signals.loc[signals.index[i], "stop_loss"] = data["close"].iloc[i] * 0.85
            else:
                if result["XG"].iloc[i]:  # K值上移
                    # 使用前6个月最低点作为止损
                    recent_low = data["low"].iloc[i-6:i+1].min()
                    signals.loc[signals.index[i], "stop_loss"] = recent_low * 0.95
                else:  # K值不上移
                    # 使用当前价格的85%作为止损
                    signals.loc[signals.index[i], "stop_loss"] = data["close"].iloc[i] * 0.85
        
        # 设置市场环境
        # 根据KDJ的值和方向判断市场环境
        bull_market = (result["K"] > result["D"]) & (result["K"] > 50) & (result["XG"])
        bear_market = (result["K"] < result["D"]) & (result["K"] < 50) & (~result["XG"])
        
        signals.loc[bull_market, "market_env"] = "牛市"
        signals.loc[bear_market, "market_env"] = "熊市"
        signals.loc[(~bull_market) & (~bear_market), "market_env"] = "震荡市"
        
        # 设置成交量确认
        # 月线级别可能不太依赖成交量确认
        signals["volume_confirmation"] = False
        
        return signals


class ZXMWeeklyKDJDOrDEATrendUp(BaseIndicator):
    """
    ZXM趋势-周KDJ·D/DEA上移指标
    
    判断周线KDJ指标的D值或MACD的DEA值是否有一个向上移动
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-周KDJ·D/DEA上移指标"""
        super().__init__(name="ZXMWeeklyKDJDOrDEATrendUp", description="ZXM趋势-周KDJ·D/DEA上移指标，判断周线KDJ·D或DEA值是否向上")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周KDJ·D/DEA上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        DIFF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIFF,9);
        xg:D>=REF(D,1) OR DEA>=REF(DEA,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        diff = ema12 - ema26
        dea = diff.ewm(span=9, adjust=False).mean()
        
        # 计算趋势信号
        xg = (d >= d.shift(1)) | (dea >= dea.shift(1))
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["DIFF"] = diff
        result["DEA"] = dea
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算SMA(简单移动平均)
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            pd.Series: SMA结果
        """
        result = pd.Series(np.zeros(len(series)), index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算周KDJ的D或DEA上移指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. D或DEA上移信号(XG)：周线级别上看涨，+30分
        score[result["XG"]] += 30
        
        # 2. 具体条件评分
        # D/DEA值在不同区间有不同意义
        d_value = result["D"]
        macd_dea = result.get("DEA", pd.Series(np.nan, index=data.index))  # 如果没有DEA列，使用NaN值
        
        # 使用D值的评分规则
        if "D" in result.columns:
            # 低位上移更有意义
            score[(result["XG"]) & (d_value < 30)] += 15  # 低位上移，更强烈买入信号
            score[(result["XG"]) & (d_value >= 30) & (d_value < 70)] += 10  # 中位上移，中等买入信号
            score[(result["XG"]) & (d_value >= 70)] += 5  # 高位上移，弱买入信号
            
            # 超买超卖区域评分
            score[d_value < 20] += 15  # 严重超卖，可能即将反弹
            score[d_value > 80] -= 15  # 严重超买，可能即将回调
        
        # 使用DEA值的评分规则
        if "DEA" in result.columns:
            # DEA从负值向上穿越0轴是强烈信号
            zero_cross_up = (macd_dea > 0) & (macd_dea.shift(1) <= 0)
            zero_cross_down = (macd_dea < 0) & (macd_dea.shift(1) >= 0)
            
            score[zero_cross_up] += 25  # 向上穿越0轴，强烈买入信号
            score[zero_cross_down] -= 25  # 向下穿越0轴，强烈卖出信号
            
            # DEA上移幅度评分
            dea_chg_pct = (macd_dea - macd_dea.shift(1)) / macd_dea.shift(1).abs().clip(lower=0.01)
            score[dea_chg_pct > 0.05] += 10  # DEA快速上移
            score[dea_chg_pct < -0.05] -= 10  # DEA快速下移
        
        # 3. 金叉死叉评分
        if all(col in result.columns for col in ["K", "D"]):
            k_value = result["K"]
            k_cross_above_d = (k_value > d_value) & (k_value.shift(1) <= d_value.shift(1))
            k_cross_below_d = (k_value < d_value) & (k_value.shift(1) >= d_value.shift(1))
            
            score[k_cross_above_d] += 20  # KD金叉，买入信号
            score[k_cross_below_d] -= 20  # KD死叉，卖出信号
        
        if all(col in result.columns for col in ["DIF", "DEA"]):
            dif_value = result["DIF"]
            dea_value = result["DEA"]
            dif_cross_above_dea = (dif_value > dea_value) & (dif_value.shift(1) <= dea_value.shift(1))
            dif_cross_below_dea = (dif_value < dea_value) & (dif_value.shift(1) >= dea_value.shift(1))
            
            score[dif_cross_above_dea] += 20  # MACD金叉，买入信号
            score[dif_cross_below_dea] -= 20  # MACD死叉，卖出信号
        
        # 4. J线极值评分（如果使用KDJ）
        if "J" in result.columns:
            j_value = result["J"]
            
            score[j_value < 0] += 10  # J值低于0，超卖信号
            score[j_value > 100] -= 10  # J值高于100，超买信号
        
        # 5. 连续上移评分
        # 连续多周D/DEA上移，表示趋势较强
        continuous_up_weeks = pd.Series(0, index=data.index)
        for i in range(3, len(data)):
            if all(result["XG"].iloc[i-3:i+1]):
                continuous_up_weeks.iloc[i] = 3
            elif all(result["XG"].iloc[i-1:i+1]):
                continuous_up_weeks.iloc[i] = 1
        
        # 根据连续上移周数加分
        score[continuous_up_weeks == 1] += 5
        score[continuous_up_weeks == 3] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别周KDJ·D/DEA上移指标相关的技术形态
        
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
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 趋势类型
            if last_row["XG"]:
                patterns.append("周KDJ·D/DEA上移")
            else:
                patterns.append("周KDJ·D/DEA不上移")
            
            # D/DEA值形态判断
            d_value = last_row["D"]
            macd_dea = last_row.get("DEA", pd.Series(np.nan, index=data.index))  # 如果没有DEA列，使用NaN值
            
            # D值位置判断
            if d_value < 20:
                patterns.append("周KDJ严重超卖区域")
            elif d_value < 40:
                patterns.append("周KDJ超卖区域")
            elif d_value > 80:
                patterns.append("周KDJ严重超买区域")
            elif d_value > 60:
                patterns.append("周KDJ超买区域")
                
            # KD关系判断
            if d_value > macd_dea:
                # 检查是否为金叉
                if len(result) > 1:
                    if result["D"].iloc[-2] <= macd_dea.iloc[-2]:
                        patterns.append("周线KDJ金叉形成")
                    else:
                        patterns.append("周线KDJ金叉后持续上行")
            else:
                # 检查是否为死叉
                if len(result) > 1:
                    if result["D"].iloc[-2] >= macd_dea.iloc[-2]:
                        patterns.append("周线KDJ死叉形成")
                    else:
                        patterns.append("周线KDJ死叉后持续下行")
            
            # DEA值极值判断
            if macd_dea < 0:
                patterns.append("周KDJ-DEA值低于0，超卖严重")
            elif macd_dea > 100:
                patterns.append("周KDJ-DEA值高于100，超买严重")
                
            # 趋势转变判断
            if len(result) >= 6:
                if not result["XG"].iloc[-6:-3].any() and result["XG"].iloc[-3:].all():
                    patterns.append("周KDJ趋势由弱转强")
                elif result["XG"].iloc[-6:-3].all() and not result["XG"].iloc[-3:].any():
                    patterns.append("周KDJ趋势由强转弱")
        
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
        
        # 设置买卖信号
        signals['buy_signal'] = result["XG"]
        signals['sell_signal'] = ~result["XG"]
        signals['neutral_signal'] = False
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["XG"], 'signal_type'] = 'trend_reversal_bullish'
        signals.loc[~result["XG"], 'signal_type'] = 'trend_reversal_bearish'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "XG"]:
                signals.loc[i, 'signal_desc'] = "周KDJ·D/DEA上移，趋势强劲"
            else:
                signals.loc[i, 'signal_desc'] = "周KDJ·D/DEA不上移，趋势走弱"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势强度调整置信度
        signals['confidence'] = signals['confidence'] + (score / 10).clip(0, 10)
        
        # 趋势反转信号有更高的置信度
        signals.loc[~result["XG"], 'confidence'] = 70
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        signals.loc[score >= 70, 'risk_level'] = '低'  # 强趋势风险较低
        signals.loc[score < 40, 'risk_level'] = '高'  # 弱趋势风险较高
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据趋势强度调整仓位
        strong_trend = result["XG"]
        signals.loc[strong_trend, 'position_size'] = 0.5  # 强趋势，加大仓位
        
        reversal_signal = ~result["XG"]
        signals.loc[reversal_signal, 'position_size'] = 0.4  # 反转信号，中等偏大仓位
        
        # 止损位 - 使用支撑位或移动平均线
        signals['stop_loss'] = 0.0
        ma_long = data["close"].rolling(window=30).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 30:
                    # 使用长期均线作为止损位
                    signals.loc[i, 'stop_loss'] = ma_long.iloc[idx] * 0.95  # 长期均线下方5%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        signals.loc[result["XG"], 'market_env'] = 'bull_market'
        signals.loc[~result["XG"], 'market_env'] = 'bear_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals


class ZXMWeeklyKDJDTrendUp(BaseIndicator):
    """
    ZXM趋势-周KDJ·D上移指标
    
    判断周线KDJ指标的D值是否向上移动
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-周KDJ·D上移指标"""
        super().__init__(name="ZXMWeeklyKDJDTrendUp", description="ZXM趋势-周KDJ·D上移指标，判断周线KDJ·D值是否向上")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-周KDJ·D上移指标
        
        Args:
            data: 输入数据，包含OHLC数据，需为周线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        RSV:=(CLOSE-LLV(LOW,9))/(HHV(HIGH,9)-LLV(LOW,9))*100;
        K:=SMA(RSV,3,1);
        D:=SMA(K,3,1);
        J:=3*K-2*D;
        xg:D>=REF(D,1);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close", "high", "low"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算KDJ指标
        low_9 = data["low"].rolling(window=9).min()
        high_9 = data["high"].rolling(window=9).max()
        
        # 计算RSV，处理除零情况
        rsv = pd.Series(np.zeros(len(data)), index=data.index)
        divisor = high_9 - low_9
        valid_idx = divisor > 0
        rsv[valid_idx] = ((data["close"] - low_9) / divisor * 100)[valid_idx]
        
        # 计算K、D、J值
        k = self._sma(rsv, 3, 1)
        d = self._sma(k, 3, 1)
        j = 3 * k - 2 * d
        
        # 计算趋势信号
        xg = d >= d.shift(1)
        
        # 添加计算结果到数据框
        result["RSV"] = rsv
        result["K"] = k
        result["D"] = d
        result["J"] = j
        result["XG"] = xg
        
        return result
    
    def _sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """计算移动平均线，类似于通达信中的SMA函数"""
        result = pd.Series(index=series.index)
        result.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算周KDJ·D上移指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. KDJ上移信号(XG)：周线级别上看涨，+30分
        score[result["XG"]] += 30
        
        # 2. KDJ具体数值区间评分
        # KDJ值在不同区间有不同意义
        k_value = result["K"]
        d_value = result["D"]
        
        # 低位上移更有意义
        score[(result["XG"]) & (k_value < 30)] += 15  # 低位上移，更强烈买入信号
        score[(result["XG"]) & (k_value >= 30) & (k_value < 70)] += 10  # 中位上移，中等买入信号
        score[(result["XG"]) & (k_value >= 70)] += 5  # 高位上移，弱买入信号
        
        # 超买超卖区域评分
        score[k_value < 20] += 15  # 严重超卖，可能即将反弹
        score[k_value > 80] -= 15  # 严重超买，可能即将回调
        
        # 3. KD线金叉死叉评分
        k_cross_above_d = (k_value > d_value) & (k_value.shift(1) <= d_value.shift(1))
        k_cross_below_d = (k_value < d_value) & (k_value.shift(1) >= d_value.shift(1))
        
        score[k_cross_above_d] += 20  # KD金叉，买入信号
        score[k_cross_below_d] -= 20  # KD死叉，卖出信号
        
        # 4. J线极值评分
        j_value = result["J"]
        
        score[j_value < 0] += 10  # J值低于0，超卖信号
        score[j_value > 100] -= 10  # J值高于100，超买信号
        
        # 5. 连续上移评分
        # 连续多周K值上移，表示趋势较强
        continuous_up_weeks = pd.Series(0, index=data.index)
        for i in range(3, len(data)):
            if all(result["XG"].iloc[i-3:i+1]):
                continuous_up_weeks.iloc[i] = 3
            elif all(result["XG"].iloc[i-1:i+1]):
                continuous_up_weeks.iloc[i] = 1
        
        # 根据连续上移周数加分
        score[continuous_up_weeks == 1] += 5
        score[continuous_up_weeks == 3] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别周KDJ·D上移指标相关的技术形态
        
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
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 基础形态判断
            if last_row["XG"]:
                patterns.append("周KDJ·D上移")
            else:
                patterns.append("周KDJ·D不上移")
            
            # KDJ值形态判断
            k_value = last_row["K"]
            d_value = last_row["D"]
            j_value = last_row["J"]
            
            # K值位置判断
            if k_value < 20:
                patterns.append("周KDJ严重超卖区域")
            elif k_value < 40:
                patterns.append("周KDJ超卖区域")
            elif k_value > 80:
                patterns.append("周KDJ严重超买区域")
            elif k_value > 60:
                patterns.append("周KDJ超买区域")
                
            # KD关系判断
            if d_value > k_value:
                # 检查是否为金叉
                if len(result) > 1:
                    if result["D"].iloc[-2] <= result["K"].iloc[-2]:
                        patterns.append("周线KDJ金叉形成")
                    else:
                        patterns.append("周线KDJ金叉后持续上行")
            else:
                # 检查是否为死叉
                if len(result) > 1:
                    if result["D"].iloc[-2] >= result["K"].iloc[-2]:
                        patterns.append("周线KDJ死叉形成")
                    else:
                        patterns.append("周线KDJ死叉后持续下行")
            
            # J值极值判断
            if j_value < 0:
                patterns.append("周KDJ-J值低于0，超卖严重")
            elif j_value > 100:
                patterns.append("周KDJ-J值高于100，超买严重")
                
            # 趋势转变判断
            if len(result) >= 6:
                if not result["XG"].iloc[-6:-3].any() and result["XG"].iloc[-3:].all():
                    patterns.append("周KDJ趋势由弱转强")
                elif result["XG"].iloc[-6:-3].all() and not result["XG"].iloc[-3:].any():
                    patterns.append("周KDJ趋势由强转弱")
        
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
        
        # 设置买卖信号
        signals['buy_signal'] = result["XG"]
        signals['sell_signal'] = ~result["XG"]
        signals['neutral_signal'] = False
        
        # 设置趋势
        signals['trend'] = 0  # 默认中性
        signals.loc[result["XG"], 'trend'] = 1  # 均线上移看涨
        signals.loc[~result["XG"], 'trend'] = -1  # 均线不上移看跌
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[result["XG"], 'signal_type'] = 'trend_reversal_bullish'
        signals.loc[~result["XG"], 'signal_type'] = 'trend_reversal_bearish'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if result.loc[i, "XG"]:
                signals.loc[i, 'signal_desc'] = "周KDJ·D上移，趋势强劲"
            else:
                signals.loc[i, 'signal_desc'] = "周KDJ·D不上移，趋势走弱"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势强度调整置信度
        signals['confidence'] = signals['confidence'] + (score / 10).clip(0, 10)
        
        # 趋势反转信号有更高的置信度
        signals.loc[~result["XG"], 'confidence'] = 70
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        signals.loc[score >= 70, 'risk_level'] = '低'  # 强趋势风险较低
        signals.loc[score < 40, 'risk_level'] = '高'  # 弱趋势风险较高
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据趋势强度调整仓位
        strong_trend = result["XG"]
        signals.loc[strong_trend, 'position_size'] = 0.5  # 强趋势，加大仓位
        
        reversal_signal = ~result["XG"]
        signals.loc[reversal_signal, 'position_size'] = 0.4  # 反转信号，中等偏大仓位
        
        # 止损位 - 使用支撑位或移动平均线
        signals['stop_loss'] = 0.0
        ma_long = data["close"].rolling(window=30).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 30:
                    # 使用长期均线作为止损位
                    signals.loc[i, 'stop_loss'] = ma_long.iloc[idx] * 0.95  # 长期均线下方5%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        signals.loc[result["XG"], 'market_env'] = 'bull_market'
        signals.loc[~result["XG"], 'market_env'] = 'bear_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals


class ZXMMonthlyMACD(BaseIndicator):
    """
    ZXM趋势-月MACD指标
    
    判断月线MACD金叉
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势-月MACD指标"""
        super().__init__(name="ZXMMonthlyMACD", description="ZXM趋势-月MACD指标，判断月线MACD金叉")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算ZXM趋势-月MACD指标
        
        Args:
            data: 输入数据，包含收盘价数据，需为月线数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势信号
            
        公式说明：
        DIF:EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:EMA(DIF,9);
        MACD:(DIF-DEA)*2;
        xg:CROSS(DIF,DEA);
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 初始化结果数据框
        result = data.copy()
        
        # 计算MACD指标
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        
        # 计算金叉信号
        xg = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        
        # 添加计算结果到数据框
        result["EMA12"] = ema12
        result["EMA26"] = ema26
        result["DIF"] = dif
        result["DEA"] = dea
        result["MACD"] = macd
        result["XG"] = xg
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算月线MACD指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 主要信号评分规则
        # 1. DIF上穿DEA的金叉信号(XG)：强烈买入信号，+35分
        dif_cross_above_dea = (result["DIF"] > result["DEA"]) & (result["DIF"].shift(1) <= result["DEA"].shift(1))
        score[dif_cross_above_dea] += 35
        
        # 2. DIF下穿DEA的死叉信号：强烈卖出信号，-35分
        dif_cross_below_dea = (result["DIF"] < result["DEA"]) & (result["DIF"].shift(1) >= result["DEA"].shift(1))
        score[dif_cross_below_dea] -= 35
        
        # 3. MACD柱状图评分
        macd_value = result["MACD"]
        
        # MACD柱状由负转正，买入信号
        macd_turn_positive = (macd_value > 0) & (macd_value.shift(1) <= 0)
        score[macd_turn_positive] += 30
        
        # MACD柱状由正转负，卖出信号
        macd_turn_negative = (macd_value < 0) & (macd_value.shift(1) >= 0)
        score[macd_turn_negative] -= 30
        
        # 4. MACD柱状连续变化评分
        # 连续增加表示趋势增强
        macd_increase = macd_value > macd_value.shift(1)
        
        # 连续3个月MACD柱状增加
        continuous_increase = pd.Series(False, index=data.index)
        for i in range(3, len(data)):
            if all(macd_increase.iloc[i-3:i+1]):
                continuous_increase.iloc[i] = True
        
        score[continuous_increase] += 15
        
        # 连续3个月MACD柱状减少
        continuous_decrease = pd.Series(False, index=data.index)
        for i in range(3, len(data)):
            if all(~macd_increase.iloc[i-3:i+1]):
                continuous_decrease.iloc[i] = True
        
        score[continuous_decrease] -= 15
        
        # 5. 零轴位置评分
        # DIF和DEA位于零轴上方，多头市场
        above_zero = (result["DIF"] > 0) & (result["DEA"] > 0)
        score[above_zero] += 10
        
        # DIF和DEA位于零轴下方，空头市场
        below_zero = (result["DIF"] < 0) & (result["DEA"] < 0)
        score[below_zero] -= 10
        
        # 6. DIF与DEA背离评分
        # DIF向上但股价向下，顶背离
        price_down = data["close"] < data["close"].shift(1)
        dif_up = result["DIF"] > result["DIF"].shift(1)
        top_divergence = price_down & dif_up
        score[top_divergence] -= 15
        
        # DIF向下但股价向上，底背离
        price_up = data["close"] > data["close"].shift(1)
        dif_down = result["DIF"] < result["DIF"].shift(1)
        bottom_divergence = price_up & dif_down
        score[bottom_divergence] += 15
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别月线MACD指标相关的技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别的形态列表
        """
        # 计算指标
        result = self.calculate(data)
        
        # 只关注最后一个交易月的形态
        patterns = []
        if len(result) > 0:
            last_row = result.iloc[-1]
            prev_row = result.iloc[-2] if len(result) > 1 else None
            
            # MACD金叉死叉
            if prev_row is not None:
                if (last_row["DIF"] > last_row["DEA"]) and (prev_row["DIF"] <= prev_row["DEA"]):
                    patterns.append("月线MACD金叉形成")
                elif (last_row["DIF"] < last_row["DEA"]) and (prev_row["DIF"] >= prev_row["DEA"]):
                    patterns.append("月线MACD死叉形成")
            
            # DIF和DEA相对位置
            if last_row["DIF"] > last_row["DEA"]:
                patterns.append("月线MACD多头排列")
            else:
                patterns.append("月线MACD空头排列")
            
            # MACD柱状图形态
            macd_value = last_row["MACD"]
            prev_macd = prev_row["MACD"] if prev_row is not None else 0
            
            if macd_value > 0:
                if macd_value > prev_macd:
                    patterns.append("月线MACD柱状图扩大")
                elif macd_value < prev_macd:
                    patterns.append("月线MACD柱状图收缩")
                
                if prev_macd <= 0:
                    patterns.append("月线MACD由负转正")
            else:
                if macd_value < prev_macd:
                    patterns.append("月线MACD柱状图负向扩大")
                elif macd_value > prev_macd:
                    patterns.append("月线MACD柱状图负向收缩")
                
                if prev_macd >= 0:
                    patterns.append("月线MACD由正转负")
            
            # 零轴上下方
            if last_row["DIF"] > 0 and last_row["DEA"] > 0:
                patterns.append("月线MACD双线位于零轴上方")
            elif last_row["DIF"] < 0 and last_row["DEA"] < 0:
                patterns.append("月线MACD双线位于零轴下方")
            elif last_row["DIF"] > 0 and last_row["DEA"] < 0:
                patterns.append("月线MACD-DIF位于零轴上方，DEA位于零轴下方")
            elif last_row["DIF"] < 0 and last_row["DEA"] > 0:
                patterns.append("月线MACD-DIF位于零轴下方，DEA位于零轴上方")
            
            # 判断背离
            if len(data) >= 6:
                # 取最近6个月数据判断背离
                last_6m_data = data.iloc[-6:]
                last_6m_result = result.iloc[-6:]
                
                # 价格创新高，但DIF未创新高，顶背离
                if last_6m_data["close"].iloc[-1] > last_6m_data["close"].iloc[:-1].max() and \
                   last_6m_result["DIF"].iloc[-1] < last_6m_result["DIF"].iloc[:-1].max():
                    patterns.append("月线MACD可能顶背离")
                
                # 价格创新低，但DIF未创新低，底背离
                if last_6m_data["close"].iloc[-1] < last_6m_data["close"].iloc[:-1].min() and \
                   last_6m_result["DIF"].iloc[-1] > last_6m_result["DIF"].iloc[:-1].min():
                    patterns.append("月线MACD可能底背离")
        
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
        
        # 识别形态
        patterns = []
        for i in range(len(data)):
            patterns.append(self.identify_patterns(data.iloc[:i+1], **kwargs))
        
        # 设置基本信号列
        signals["buy_signal"] = False
        signals["sell_signal"] = False 
        signals["neutral_signal"] = True
        signals["trend"] = 0
        signals["score"] = score
        signals["signal_type"] = "中性"
        signals["signal_desc"] = ""
        signals["confidence"] = 0
        signals["risk_level"] = "中"
        signals["position_size"] = 0.0
        signals["stop_loss"] = 0.0
        signals["market_env"] = "未知"
        signals["volume_confirmation"] = False
        
        # 填充信号描述和置信度
        for i in range(len(signals)):
            pattern_list = patterns[i] if i < len(patterns) else []
            signals.loc[signals.index[i], "signal_desc"] = "，".join(pattern_list) if pattern_list else "无明显形态"
        
        # 生成趋势方向
        # DIF > DEA 为多头趋势
        signals.loc[result["DIF"] > result["DEA"], "trend"] = 1
        # DIF < DEA 为空头趋势
        signals.loc[result["DIF"] < result["DEA"], "trend"] = -1
        
        # 设置买入信号
        # MACD金叉是强烈买入信号
        dif_cross_above_dea = (result["DIF"] > result["DEA"]) & (result["DIF"].shift(1) <= result["DEA"].shift(1))
        
        # MACD柱状由负转正也是买入信号
        macd_turn_positive = (result["MACD"] > 0) & (result["MACD"].shift(1) <= 0)
        
        # 结合评分设置买入信号
        buy_condition1 = dif_cross_above_dea  # 金叉
        buy_condition2 = macd_turn_positive  # 柱状转正
        buy_condition3 = (score > 70) & (result["DIF"] > result["DEA"])  # 高评分多头排列
        
        buy_condition = buy_condition1 | buy_condition2 | buy_condition3
        signals.loc[buy_condition, "buy_signal"] = True
        signals.loc[buy_condition, "neutral_signal"] = False
        
        # 根据不同条件设置不同的信号类型
        signals.loc[buy_condition1, "signal_type"] = "买入-月线MACD金叉"
        signals.loc[buy_condition2, "signal_type"] = "买入-月线MACD柱状转正"
        signals.loc[buy_condition3 & ~buy_condition1 & ~buy_condition2, "signal_type"] = "买入-月线MACD多头排列"
        
        # 设置卖出信号
        # MACD死叉是强烈卖出信号
        dif_cross_below_dea = (result["DIF"] < result["DEA"]) & (result["DIF"].shift(1) >= result["DEA"].shift(1))
        
        # MACD柱状由正转负也是卖出信号
        macd_turn_negative = (result["MACD"] < 0) & (result["MACD"].shift(1) >= 0)
        
        # 结合评分设置卖出信号
        sell_condition1 = dif_cross_below_dea  # 死叉
        sell_condition2 = macd_turn_negative  # 柱状转负
        sell_condition3 = (score < 30) & (result["DIF"] < result["DEA"])  # 低评分空头排列
        
        sell_condition = sell_condition1 | sell_condition2 | sell_condition3
        signals.loc[sell_condition, "sell_signal"] = True
        signals.loc[sell_condition, "neutral_signal"] = False
        
        # 根据不同条件设置不同的信号类型
        signals.loc[sell_condition1, "signal_type"] = "卖出-月线MACD死叉"
        signals.loc[sell_condition2, "signal_type"] = "卖出-月线MACD柱状转负"
        signals.loc[sell_condition3 & ~sell_condition1 & ~sell_condition2, "signal_type"] = "卖出-月线MACD空头排列"
        
        # 计算置信度
        # 置信度基于评分、DIF/DEA差值和零轴位置
        signals["confidence"] = signals["score"].apply(lambda x: min(100, x + 10) if x > 50 else max(0, x - 10))
        
        # MACD金叉/死叉提高置信度
        signals.loc[dif_cross_above_dea, "confidence"] = signals.loc[dif_cross_above_dea, "confidence"].apply(lambda x: min(100, x + 20))
        signals.loc[dif_cross_below_dea, "confidence"] = signals.loc[dif_cross_below_dea, "confidence"].apply(lambda x: min(100, x + 20))
        
        # DIF/DEA同向且远离零轴提高置信度
        strong_bullish = (result["DIF"] > 0) & (result["DEA"] > 0) & (result["DIF"] > result["DEA"])
        strong_bearish = (result["DIF"] < 0) & (result["DEA"] < 0) & (result["DIF"] < result["DEA"])
        
        signals.loc[strong_bullish, "confidence"] = signals.loc[strong_bullish, "confidence"].apply(lambda x: min(100, x + 15))
        signals.loc[strong_bearish, "confidence"] = signals.loc[strong_bearish, "confidence"].apply(lambda x: min(100, x + 15))
        
        # 设置风险等级
        signals.loc[signals["score"] >= 70, "risk_level"] = "低"
        signals.loc[(signals["score"] < 70) & (signals["score"] >= 40), "risk_level"] = "中"
        signals.loc[signals["score"] < 40, "risk_level"] = "高"
        
        # 设置仓位建议
        # 基于评分和风险设置仓位大小，月线信号更重要，整体仓位更大
        signals["position_size"] = signals["score"].apply(lambda x: min(1.0, max(0.0, (x - 40) / 60 * 0.8)))
        
        # 设置止损位
        # 月线级别止损应更宽松
        for i in range(len(signals)):
            if i < 6:  # 数据不足，使用默认值
                signals.loc[signals.index[i], "stop_loss"] = data["close"].iloc[i] * 0.85
            else:
                if signals["trend"].iloc[i] > 0:  # 多头趋势
                    # 使用前6个月最低点作为止损
                    recent_low = data["low"].iloc[i-6:i+1].min()
                    signals.loc[signals.index[i], "stop_loss"] = recent_low * 0.95
                else:  # 空头趋势
                    # 使用当前价格的85%作为止损
                    signals.loc[signals.index[i], "stop_loss"] = data["close"].iloc[i] * 0.85
        
        # 设置市场环境
        # 根据MACD指标判断市场环境
        bull_market = (result["DIF"] > 0) & (result["DEA"] > 0) & (result["DIF"] > result["DEA"])
        bear_market = (result["DIF"] < 0) & (result["DEA"] < 0) & (result["DIF"] < result["DEA"])
        
        signals.loc[bull_market, "market_env"] = "牛市"
        signals.loc[bear_market, "market_env"] = "熊市"
        signals.loc[(~bull_market) & (~bear_market), "market_env"] = "震荡市"
        
        # 设置成交量确认
        # 月线级别可能不太依赖成交量确认
        signals["volume_confirmation"] = False
        
        return signals
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM趋势持续性指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 根据趋势状态和成熟度调整评分
        for i in range(len(data)):
            trend_state = result["TrendState"].iloc[i]
            maturity = result["TrendMaturity"].iloc[i]
            health = result["TrendHealth"].iloc[i]
            
            # 基础调整 - 趋势状态
            if trend_state == 1:  # 上升趋势
                state_score = 20
            elif trend_state == -1:  # 下降趋势
                state_score = -20
            else:  # 震荡趋势
                state_score = 0
            
            # 成熟度调整
            # 成熟度较低(0-30)：趋势初期，加分
            # 成熟度中等(30-70)：趋势中期，保持
            # 成熟度较高(70-100)：趋势后期，减分
            if maturity < 30:
                maturity_score = 10
            elif maturity > 70:
                maturity_score = -10
            else:
                maturity_score = 0
            
            # 健康度调整
            health_score = (health - 50) / 5  # -10到+10的范围
            
            # 综合评分调整
            score.iloc[i] = 50 + state_score + maturity_score + health_score
        
        # 趋势变化点额外调整
        trend_change = result["TrendChange"]
        trend_state = result["TrendState"]
        
        # 由下降转为上升：强烈买入信号
        up_change = trend_change & (trend_state == 1)
        score[up_change] = 80
        
        # 由上升转为下降：强烈卖出信号
        down_change = trend_change & (trend_state == -1)
        score[down_change] = 20
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势持续性指标相关的技术形态
        
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
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 趋势状态
            if last_row["TrendState"] == 1:
                patterns.append("上升趋势")
                
                # 趋势成熟度
                maturity = last_row["TrendMaturity"]
                if maturity < 30:
                    patterns.append("上升趋势初期")
                elif maturity < 70:
                    patterns.append("上升趋势中期")
                else:
                    patterns.append("上升趋势后期")
                
                # 趋势健康度
                health = last_row["TrendHealth"]
                if health >= 80:
                    patterns.append("健康上升趋势")
                elif health >= 60:
                    patterns.append("稳定上升趋势")
                else:
                    patterns.append("虚弱上升趋势")
                
            elif last_row["TrendState"] == -1:
                patterns.append("下降趋势")
                
                # 趋势成熟度
                maturity = last_row["TrendMaturity"]
                if maturity < 30:
                    patterns.append("下降趋势初期")
                elif maturity < 70:
                    patterns.append("下降趋势中期")
                else:
                    patterns.append("下降趋势后期")
                
                # 趋势健康度
                health = last_row["TrendHealth"]
                if health >= 80:
                    patterns.append("健康下降趋势")
                elif health >= 60:
                    patterns.append("稳定下降趋势")
                else:
                    patterns.append("虚弱下降趋势")
                
            else:
                patterns.append("震荡/无趋势")
            
            # 趋势变化点
            if last_row["TrendChange"]:
                if last_row["TrendState"] == 1:
                    patterns.append("趋势转折：由空转多")
                elif last_row["TrendState"] == -1:
                    patterns.append("趋势转折：由多转空")
            
            # 趋势持续性
            duration = last_row["TrendDuration"]
            if duration >= 60:
                patterns.append("超长期趋势")
            elif duration >= 30:
                patterns.append("长期趋势")
            elif duration >= 10:
                patterns.append("中期趋势")
            else:
                patterns.append("短期趋势")
            
            # 周期特性
            if last_row["CycleRegularity"] < 0.3:
                patterns.append("规律性强的周期")
            elif last_row["CycleRegularity"] > 0.7:
                patterns.append("不规律的周期")
        
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
        
        # 设置买卖信号
        # 买入信号：
        # 1. 趋势由下降转为上升
        # 2. 上升趋势初期（成熟度<30）且健康度高（>=70）
        buy_condition1 = result["TrendChange"] & (result["TrendState"] == 1)
        buy_condition2 = (result["TrendState"] == 1) & (result["TrendMaturity"] < 30) & (result["TrendHealth"] >= 70)
        signals['buy_signal'] = buy_condition1 | buy_condition2
        
        # 卖出信号：
        # 1. 趋势由上升转为下降
        # 2. 上升趋势后期（成熟度>80）且健康度低（<60）
        sell_condition1 = result["TrendChange"] & (result["TrendState"] == -1)
        sell_condition2 = (result["TrendState"] == 1) & (result["TrendMaturity"] > 80) & (result["TrendHealth"] < 60)
        signals['sell_signal'] = sell_condition1 | sell_condition2
        
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result["TrendState"]
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[signals['buy_signal'], 'signal_type'] = 'trend_reversal_bullish'
        signals.loc[signals['sell_signal'], 'signal_type'] = 'trend_reversal_bearish'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            trend_state = result.loc[i, "TrendState"]
            duration = result.loc[i, "TrendDuration"]
            maturity = result.loc[i, "TrendMaturity"]
            health = result.loc[i, "TrendHealth"]
            
            if result.loc[i, "TrendChange"]:
                if trend_state == 1:
                    signals.loc[i, 'signal_desc'] = f"趋势反转：由空转多，健康度{health:.1f}"
                elif trend_state == -1:
                    signals.loc[i, 'signal_desc'] = f"趋势反转：由多转空，健康度{health:.1f}"
            elif trend_state == 1:
                maturity_desc = "初期" if maturity < 30 else "中期" if maturity < 70 else "后期"
                signals.loc[i, 'signal_desc'] = f"上升趋势{maturity_desc}，持续{duration:.0f}天，健康度{health:.1f}"
            elif trend_state == -1:
                maturity_desc = "初期" if maturity < 30 else "中期" if maturity < 70 else "后期"
                signals.loc[i, 'signal_desc'] = f"下降趋势{maturity_desc}，持续{duration:.0f}天，健康度{health:.1f}"
            else:
                signals.loc[i, 'signal_desc'] = "震荡/无趋势"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势健康度调整置信度
        health_confidence_adj = ((result["TrendHealth"] - 50) / 50 * 20).clip(-20, 20)
        signals['confidence'] = signals['confidence'] + health_confidence_adj
        
        # 趋势变化点有更高的置信度
        signals.loc[result["TrendChange"], 'confidence'] += 10
        
        # 周期规律性高的趋势有更高的置信度
        signals.loc[result["CycleRegularity"] < 0.3, 'confidence'] += 10
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 风险等级
        signals['risk_level'] = '中'  # 默认中等风险
        
        # 趋势初期或变化点风险较高
        early_trend = (result["TrendMaturity"] < 30) & ((result["TrendState"] != 0))
        signals.loc[early_trend | result["TrendChange"], 'risk_level'] = '高'
        
        # 健康稳定的中期趋势风险较低
        stable_trend = (result["TrendMaturity"] >= 30) & (result["TrendMaturity"] <= 70) & (result["TrendHealth"] >= 70)
        signals.loc[stable_trend, 'risk_level'] = '低'
        
        # 建议仓位
        signals['position_size'] = 0.0
        signals.loc[signals['buy_signal'], 'position_size'] = 0.3  # 基础仓位
        
        # 根据趋势健康度调整仓位
        signals.loc[(signals['buy_signal']) & (result["TrendHealth"] >= 80), 'position_size'] = 0.5  # 健康趋势，加大仓位
        
        # 止损位
        signals['stop_loss'] = 0.0
        ma20 = data["close"].rolling(window=20).mean()
        
        for i in signals.index[signals['buy_signal']]:
            try:
                idx = data.index.get_loc(i)
                if idx >= 20:
                    # 使用最近20天的最低价作为止损参考
                    low_price = data.iloc[idx-20:idx+1]['low'].min()
                    signals.loc[i, 'stop_loss'] = low_price * 0.95  # 最低价下方5%
            except:
                continue
        
        # 市场环境
        signals['market_env'] = 'normal'
        signals.loc[result["TrendState"] == 1, 'market_env'] = 'bull_market'
        signals.loc[result["TrendState"] == -1, 'market_env'] = 'bear_market'
        signals.loc[result["TrendState"] == 0, 'market_env'] = 'sideways_market'
        
        # 成交量确认 - 简单设为True，实际应结合成交量指标
        signals['volume_confirmation'] = True
        
        return signals 


class TrendDetector(BaseIndicator):
    """
    ZXM趋势检测器
    
    识别价格趋势的方向和强度
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势检测器"""
        super().__init__(name="TrendDetector", description="ZXM趋势检测器，识别价格趋势的方向和强度")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算趋势检测结果
        
        Args:
            data: 输入数据，包含OHLC价格数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势方向、强度、持续时间等
        """
        self.ensure_columns(data, ["close"])
        
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 计算短期和长期移动平均线
        result['MA20'] = data['close'].rolling(window=20).mean()
        result['MA60'] = data['close'].rolling(window=60).mean()
        
        # 计算趋势方向（基于MA线的斜率）
        result['MA20_Slope'] = result['MA20'].diff(5) / 5
        result['MA60_Slope'] = result['MA60'].diff(10) / 10
        
        # 确定趋势状态
        # 1: 上升趋势, -1: 下降趋势, 0: 横盘/无趋势
        result['TrendState'] = 0
        
        # 上升趋势条件: 短期均线向上 且 短期均线高于长期均线
        up_trend = (result['MA20_Slope'] > 0) & (result['MA20'] > result['MA60'])
        result.loc[up_trend, 'TrendState'] = 1
        
        # 下降趋势条件: 短期均线向下 且 短期均线低于长期均线
        down_trend = (result['MA20_Slope'] < 0) & (result['MA20'] < result['MA60'])
        result.loc[down_trend, 'TrendState'] = -1
        
        # 计算趋势变化点
        result['TrendChange'] = result['TrendState'].diff().abs() > 0
        
        # 计算趋势持续时间
        result['TrendDuration'] = 0
        
        # 遍历计算趋势持续时间
        current_trend = 0
        duration = 0
        
        for i in range(len(result)):
            if i == 0 or result['TrendChange'].iloc[i]:
                # 趋势变化或首个数据点
                current_trend = result['TrendState'].iloc[i]
                duration = 1
            else:
                # 趋势持续
                if result['TrendState'].iloc[i] == current_trend:
                    duration += 1
                else:
                    # 状态变化但未触发TrendChange（可能是从震荡到有趋势）
                    current_trend = result['TrendState'].iloc[i]
                    duration = 1
            
            result['TrendDuration'].iloc[i] = duration
        
        # 计算趋势强度
        result['TrendStrength'] = 0
        
        # 上升趋势强度: 基于价格与MA20的距离和MA20斜率
        up_strength = ((data['close'] - result['MA20']) / result['MA20'] * 100 + 
                      result['MA20_Slope'] * 20).clip(0, 100)
        
        # 下降趋势强度: 基于MA20与价格的距离和MA20斜率绝对值
        down_strength = ((result['MA20'] - data['close']) / result['MA20'] * 100 + 
                        abs(result['MA20_Slope']) * 20).clip(0, 100)
        
        # 根据趋势方向选择强度
        result.loc[result['TrendState'] == 1, 'TrendStrength'] = up_strength.loc[result['TrendState'] == 1]
        result.loc[result['TrendState'] == -1, 'TrendStrength'] = down_strength.loc[result['TrendState'] == -1]
        
        # 计算趋势成熟度（基于持续时间的相对值）
        max_duration = 60  # 最大参考持续时间
        result['TrendMaturity'] = (result['TrendDuration'] / max_duration * 100).clip(0, 100)
        
        # 计算趋势健康度（基于强度和价格波动）
        result['PriceVolatility'] = data['close'].pct_change().rolling(window=20).std() * 100
        
        # 健康度计算：考虑趋势强度和稳定性
        volatility_factor = 1 - (result['PriceVolatility'] / 5).clip(0, 1)  # 波动率对健康度的影响
        result['TrendHealth'] = (result['TrendStrength'] * volatility_factor).clip(0, 100)
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算趋势检测器的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 0-100的评分，>50表示看涨，<50表示看跌
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分序列
        score = pd.Series(50, index=data.index)  # 默认中性分数
        
        # 遍历每个时间点进行评分
        for i in range(len(score)):
            # 趋势状态评分
            state = result["TrendState"].iloc[i]
            
            # 根据趋势状态基础评分
            if state == 1:  # 上升趋势
                state_score = 15  # 看涨加分
            elif state == -1:  # 下降趋势
                state_score = -15  # 看跌减分
            else:  # 无趋势
                state_score = 0
            
            # 趋势强度和健康度
            strength = result["TrendStrength"].iloc[i]
            health = result["TrendHealth"].iloc[i]
            maturity = result["TrendMaturity"].iloc[i]
            
            # 成熟度调整
            # 成熟度较低(0-30)：趋势初期，加分
            # 成熟度中等(30-70)：趋势中期，保持
            # 成熟度较高(70-100)：趋势后期，减分
            if maturity < 30:
                maturity_score = 10
            elif maturity > 70:
                maturity_score = -10
            else:
                maturity_score = 0
            
            # 健康度调整
            health_score = (health - 50) / 5  # -10到+10的范围
            
            # 综合评分调整
            score.iloc[i] = 50 + state_score + maturity_score + health_score
        
        # 趋势变化点额外调整
        trend_change = result["TrendChange"]
        trend_state = result["TrendState"]
        
        # 由下降转为上升：强烈买入信号
        up_change = trend_change & (trend_state == 1)
        score[up_change] = 80
        
        # 由上升转为下降：强烈卖出信号
        down_change = trend_change & (trend_state == -1)
        score[down_change] = 20
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势检测器相关的技术形态
        
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
        if len(result) > 0:
            last_row = result.iloc[-1]
            
            # 趋势状态
            if last_row["TrendState"] == 1:
                patterns.append("上升趋势")
                
                # 趋势成熟度
                maturity = last_row["TrendMaturity"]
                if maturity < 30:
                    patterns.append("上升趋势初期")
                elif maturity < 70:
                    patterns.append("上升趋势中期")
                else:
                    patterns.append("上升趋势后期")
                
                # 趋势健康度
                health = last_row["TrendHealth"]
                if health >= 80:
                    patterns.append("健康上升趋势")
                elif health >= 60:
                    patterns.append("稳定上升趋势")
                else:
                    patterns.append("虚弱上升趋势")
                
            elif last_row["TrendState"] == -1:
                patterns.append("下降趋势")
                
                # 趋势成熟度
                maturity = last_row["TrendMaturity"]
                if maturity < 30:
                    patterns.append("下降趋势初期")
                elif maturity < 70:
                    patterns.append("下降趋势中期")
                else:
                    patterns.append("下降趋势后期")
                
                # 趋势健康度
                health = last_row["TrendHealth"]
                if health >= 80:
                    patterns.append("健康下降趋势")
                elif health >= 60:
                    patterns.append("稳定下降趋势")
                else:
                    patterns.append("虚弱下降趋势")
                
            else:
                patterns.append("震荡/无趋势")
            
            # 趋势变化点
            if last_row["TrendChange"]:
                if last_row["TrendState"] == 1:
                    patterns.append("趋势转折：由空转多")
                elif last_row["TrendState"] == -1:
                    patterns.append("趋势转折：由多转空")
            
            # 趋势持续性
            duration = last_row["TrendDuration"]
            if duration >= 60:
                patterns.append("超长期趋势")
            elif duration >= 30:
                patterns.append("长期趋势")
            elif duration >= 10:
                patterns.append("中期趋势")
            else:
                patterns.append("短期趋势")
        
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
        
        # 设置买卖信号
        # 买入信号：
        # 1. 趋势由下降转为上升
        # 2. 上升趋势初期（成熟度<30）且健康度高（>=70）
        buy_condition1 = result["TrendChange"] & (result["TrendState"] == 1)
        buy_condition2 = (result["TrendState"] == 1) & (result["TrendMaturity"] < 30) & (result["TrendHealth"] >= 70)
        signals['buy_signal'] = buy_condition1 | buy_condition2
        
        # 卖出信号：
        # 1. 趋势由上升转为下降
        # 2. 上升趋势后期（成熟度>80）且健康度低（<60）
        sell_condition1 = result["TrendChange"] & (result["TrendState"] == -1)
        sell_condition2 = (result["TrendState"] == 1) & (result["TrendMaturity"] > 80) & (result["TrendHealth"] < 60)
        signals['sell_signal'] = sell_condition1 | sell_condition2
        
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result["TrendState"]
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[signals['buy_signal'], 'signal_type'] = 'trend_reversal_bullish'
        signals.loc[signals['sell_signal'], 'signal_type'] = 'trend_reversal_bearish'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in signals.index:
            if i >= len(result):
                continue
                
            trend_state = result.loc[i, "TrendState"]
            duration = result.loc[i, "TrendDuration"]
            maturity = result.loc[i, "TrendMaturity"]
            health = result.loc[i, "TrendHealth"]
            
            if result.loc[i, "TrendChange"]:
                if trend_state == 1:
                    signals.loc[i, 'signal_desc'] = f"趋势反转：由空转多，健康度{health:.1f}"
                elif trend_state == -1:
                    signals.loc[i, 'signal_desc'] = f"趋势反转：由多转空，健康度{health:.1f}"
            elif trend_state == 1:
                maturity_desc = "初期" if maturity < 30 else "中期" if maturity < 70 else "后期"
                signals.loc[i, 'signal_desc'] = f"上升趋势{maturity_desc}，持续{duration:.0f}天，健康度{health:.1f}"
            elif trend_state == -1:
                maturity_desc = "初期" if maturity < 30 else "中期" if maturity < 70 else "后期"
                signals.loc[i, 'signal_desc'] = f"下降趋势{maturity_desc}，持续{duration:.0f}天，健康度{health:.1f}"
            else:
                signals.loc[i, 'signal_desc'] = "震荡/无趋势"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势健康度调整置信度
        signals['confidence'] = signals['confidence'] + ((result["TrendHealth"] - 50) / 50 * 20).clip(-20, 20)
        
        # 根据趋势持续时间调整置信度
        duration_confidence = ((result["TrendDuration"].clip(0, 30)) / 30 * 10)
        signals['confidence'] = signals['confidence'] + duration_confidence
        
        # 趋势变化点的置信度根据前期趋势强度调整
        for i in range(1, len(signals)):
            if result.loc[signals.index[i], "TrendChange"]:
                prev_strength = result.loc[signals.index[i-1], "TrendStrength"]
                signals.loc[signals.index[i], 'confidence'] = signals.loc[signals.index[i], 'confidence'] + (prev_strength / 10)
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 添加风险等级
        signals['risk_level'] = '中'
        signals.loc[signals['confidence'] >= 80, 'risk_level'] = '低'
        signals.loc[signals['confidence'] <= 40, 'risk_level'] = '高'
        
        # 设置仓位建议
        signals['position_size'] = signals['confidence'] / 200  # 0-0.5范围
        
        # 设置止损建议 (基于ATR的简单实现)
        if 'high' in data.columns and 'low' in data.columns:
            # 计算ATR
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # 根据趋势设置止损
            signals['stop_loss'] = data['close']
            signals.loc[signals['buy_signal'], 'stop_loss'] = data.loc[signals['buy_signal'], 'close'] - (atr.loc[signals['buy_signal']] * 2)
            signals.loc[signals['sell_signal'], 'stop_loss'] = data.loc[signals['sell_signal'], 'close'] + (atr.loc[signals['sell_signal']] * 2)
        else:
            # 如果没有高低价数据，使用收盘价的波动率作为替代
            volatility = data['close'].pct_change().rolling(window=20).std() * data['close']
            signals['stop_loss'] = data['close']
            signals.loc[signals['buy_signal'], 'stop_loss'] = data.loc[signals['buy_signal'], 'close'] - (volatility.loc[signals['buy_signal']] * 2)
            signals.loc[signals['sell_signal'], 'stop_loss'] = data.loc[signals['sell_signal'], 'close'] + (volatility.loc[signals['sell_signal']] * 2)
        
        return signals


class TrendStrength(BaseIndicator):
    """
    ZXM趋势强度指标
    
    分析价格趋势的强度和持续性
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势强度指标"""
        super().__init__(name="TrendStrength", description="ZXM趋势强度指标，分析价格趋势的强度和持续性")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算趋势强度指标
        
        Args:
            data: 输入数据，包含OHLC价格数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势强度相关指标
        """
        self.ensure_columns(data, ["close"])
        
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 计算多个周期的移动平均线
        ma_periods = [5, 10, 20, 60]
        for period in ma_periods:
            result[f'MA{period}'] = data['close'].rolling(window=period).mean()
        
        # 计算各均线斜率
        for period in ma_periods:
            result[f'MA{period}_Slope'] = result[f'MA{period}'].diff(5) / (5 * result[f'MA{period}'].shift(5))
        
        # 计算均线排列得分
        result['MAAlignmentScore'] = 0
        
        # 对于每个时间点计算均线排列情况
        for i in range(len(result)):
            # 跳过没有足够数据的点
            if i < 60:
                continue
                
            # 提取当前时间点的各均线值
            ma_values = [result[f'MA{period}'].iloc[i] for period in ma_periods]
            
            # 判断是否为多头排列（短期均线在上，长期均线在下）
            is_bullish_alignment = all(ma_values[j] > ma_values[j+1] for j in range(len(ma_values)-1))
            
            # 判断是否为空头排列（短期均线在下，长期均线在上）
            is_bearish_alignment = all(ma_values[j] < ma_values[j+1] for j in range(len(ma_values)-1))
            
            # 计算排列分数
            if is_bullish_alignment:
                # 多头排列得分：检查斜率是否也是正的
                slope_positive = sum(1 for period in ma_periods if result[f'MA{period}_Slope'].iloc[i] > 0)
                result['MAAlignmentScore'].iloc[i] = 50 + (slope_positive / len(ma_periods) * 50)
            elif is_bearish_alignment:
                # 空头排列得分：检查斜率是否也是负的
                slope_negative = sum(1 for period in ma_periods if result[f'MA{period}_Slope'].iloc[i] < 0)
                result['MAAlignmentScore'].iloc[i] = 50 - (slope_negative / len(ma_periods) * 50)
            else:
                # 混合排列：基于短期均线位置计算分数
                if result['MA5'].iloc[i] > result['MA60'].iloc[i]:
                    result['MAAlignmentScore'].iloc[i] = 60  # 短期均线在长期均线上方，轻微看涨
                elif result['MA5'].iloc[i] < result['MA60'].iloc[i]:
                    result['MAAlignmentScore'].iloc[i] = 40  # 短期均线在长期均线下方，轻微看跌
                else:
                    result['MAAlignmentScore'].iloc[i] = 50  # 中性
        
        # 计算价格动量
        result['Momentum5'] = data['close'].pct_change(5)
        result['Momentum20'] = data['close'].pct_change(20)
        
        # 计算动量强度
        result['MomentumStrength'] = (
            (result['Momentum5'] * 100).clip(-10, 10) + 
            (result['Momentum20'] * 100).clip(-10, 10)
        ) / 2 + 50
        
        # 计算价格波动率（波动性）
        result['Volatility'] = data['close'].pct_change().rolling(window=20).std() * 100
        
        # 计算趋势强度（综合考虑均线排列、动量和波动率）
        result['TrendStrength'] = 0
        
        for i in range(len(result)):
            if i < 60:
                continue
                
            # 获取均线排列分数
            alignment_score = result['MAAlignmentScore'].iloc[i]
            
            # 获取动量强度
            momentum_strength = result['MomentumStrength'].iloc[i]
            
            # 获取波动率
            volatility = result['Volatility'].iloc[i]
            
            # 波动率调整因子：中等波动率最优，过高或过低都不利于趋势强度
            volatility_factor = 1.0
            if volatility < 1:  # 波动率过低
                volatility_factor = 0.7
            elif volatility > 5:  # 波动率过高
                volatility_factor = 0.8
            
            # 计算综合趋势强度
            trend_direction = 1 if alignment_score > 50 else -1 if alignment_score < 50 else 0
            
            # 基础强度：均线排列得分与50的偏离度
            base_strength = abs(alignment_score - 50) * 2
            
            # 动量调整：动量与趋势方向一致时加强，不一致时减弱
            momentum_direction = 1 if momentum_strength > 50 else -1 if momentum_strength < 50 else 0
            momentum_adjustment = 10 if momentum_direction == trend_direction else -10
            
            # 综合计算
            raw_strength = base_strength + momentum_adjustment
            
            # 应用波动率调整
            adjusted_strength = raw_strength * volatility_factor
            
            # 确保在0-100范围内
            result['TrendStrength'].iloc[i] = min(100, max(0, adjusted_strength))
        
        # 计算趋势方向
        result['TrendDirection'] = 0
        result.loc[result['MAAlignmentScore'] > 60, 'TrendDirection'] = 1  # 上升趋势
        result.loc[result['MAAlignmentScore'] < 40, 'TrendDirection'] = -1  # 下降趋势
        
        # 计算趋势稳定性（基于均线斜率的一致性）
        result['TrendStability'] = 0
        
        for i in range(len(result)):
            if i < 60:
                continue
                
            # 获取各均线斜率
            slopes = [result[f'MA{period}_Slope'].iloc[i] for period in ma_periods]
            
            # 计算斜率方向的一致性
            positive_slopes = sum(1 for slope in slopes if slope > 0)
            negative_slopes = sum(1 for slope in slopes if slope < 0)
            
            # 一致性分数：完全一致为100，完全不一致为0
            if positive_slopes == len(slopes) or negative_slopes == len(slopes):
                consistency = 100  # 所有斜率方向一致
            else:
                # 计算主导方向的斜率比例
                dominant_slope_count = max(positive_slopes, negative_slopes)
                consistency = (dominant_slope_count / len(slopes)) * 100
            
            result['TrendStability'].iloc[i] = consistency
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算趋势强度指标的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 0-100的评分，>50表示看涨，<50表示看跌
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分序列
        score = pd.Series(50, index=data.index)  # 默认中性分数
        
        # 遍历每个时间点进行评分
        for i in range(len(score)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            # 趋势方向
            trend_direction = result['TrendDirection'].iloc[i]
            
            # 趋势强度
            trend_strength = result['TrendStrength'].iloc[i]
            
            # 趋势稳定性
            trend_stability = result['TrendStability'].iloc[i]
            
            # 基础分数：基于趋势方向
            base_score = 50 + (trend_direction * 15)  # 上升+15，下降-15，中性+0
            
            # 强度调整：强度越高，分数偏离中性值越多
            strength_adjustment = (trend_strength / 100) * 20 * trend_direction
            
            # 稳定性奖励：稳定性高的趋势更可靠
            stability_bonus = (trend_stability - 50) / 50 * 10 if trend_stability > 50 else 0
            stability_bonus = stability_bonus * trend_direction  # 根据趋势方向调整奖励值
            
            # 综合评分
            score.iloc[i] = base_score + strength_adjustment + stability_bonus
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势强度指标相关的技术形态
        
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
        if len(result) > 60:  # 确保有足够的数据
            last_row = result.iloc[-1]
            
            # 趋势方向
            trend_direction = last_row['TrendDirection']
            trend_strength = last_row['TrendStrength']
            trend_stability = last_row['TrendStability']
            
            # 强趋势
            if trend_strength >= 80:
                if trend_direction == 1:
                    patterns.append("强劲上升趋势")
                elif trend_direction == -1:
                    patterns.append("强劲下降趋势")
            
            # 中等趋势
            elif trend_strength >= 50:
                if trend_direction == 1:
                    patterns.append("中等上升趋势")
                elif trend_direction == -1:
                    patterns.append("中等下降趋势")
            
            # 弱趋势
            elif trend_strength >= 30:
                if trend_direction == 1:
                    patterns.append("弱上升趋势")
                elif trend_direction == -1:
                    patterns.append("弱下降趋势")
            
            # 无明显趋势
            else:
                patterns.append("无明显趋势")
            
            # 趋势稳定性
            if trend_stability >= 80:
                patterns.append("高度稳定趋势")
            elif trend_stability >= 60:
                patterns.append("稳定趋势")
            else:
                patterns.append("不稳定趋势")
            
            # 均线排列
            ma_alignment = last_row['MAAlignmentScore']
            if ma_alignment >= 80:
                patterns.append("完美多头排列")
            elif ma_alignment >= 60:
                patterns.append("多头排列")
            elif ma_alignment <= 20:
                patterns.append("完美空头排列")
            elif ma_alignment <= 40:
                patterns.append("空头排列")
            else:
                patterns.append("混合均线排列")
            
            # 检查趋势变化
            if len(result) > 5:
                prev_direction = result['TrendDirection'].iloc[-6]
                if trend_direction == 1 and prev_direction <= 0:
                    patterns.append("趋势转向上升")
                elif trend_direction == -1 and prev_direction >= 0:
                    patterns.append("趋势转向下降")
                elif trend_direction == 0 and prev_direction != 0:
                    patterns.append("趋势进入盘整")
        
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
        
        # 买入信号：
        # 1. 趋势方向由中性或下降转为上升，且强度>40
        # 2. 上升趋势强度突破70，且稳定性>60
        buy_condition1 = (result['TrendDirection'].shift(1) <= 0) & (result['TrendDirection'] == 1) & (result['TrendStrength'] > 40)
        buy_condition2 = (result['TrendDirection'] == 1) & (result['TrendStrength'] > 70) & (result['TrendStability'] > 60)
        signals['buy_signal'] = buy_condition1 | buy_condition2
        
        # 卖出信号：
        # 1. 趋势方向由中性或上升转为下降，且强度>40
        # 2. 下降趋势强度突破70，且稳定性>60
        sell_condition1 = (result['TrendDirection'].shift(1) >= 0) & (result['TrendDirection'] == -1) & (result['TrendStrength'] > 40)
        sell_condition2 = (result['TrendDirection'] == -1) & (result['TrendStrength'] > 70) & (result['TrendStability'] > 60)
        signals['sell_signal'] = sell_condition1 | sell_condition2
        
        # 中性信号
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result['TrendDirection']
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        buy_signal_transition = (result['TrendDirection'].shift(1) <= 0) & (result['TrendDirection'] == 1)
        sell_signal_transition = (result['TrendDirection'].shift(1) >= 0) & (result['TrendDirection'] == -1)
        
        signals.loc[buy_signal_transition, 'signal_type'] = 'trend_transition_bullish'
        signals.loc[sell_signal_transition, 'signal_type'] = 'trend_transition_bearish'
        signals.loc[buy_condition2 & ~buy_signal_transition, 'signal_type'] = 'strong_bullish_trend'
        signals.loc[sell_condition2 & ~sell_signal_transition, 'signal_type'] = 'strong_bearish_trend'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in range(len(signals)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            if buy_signal_transition.iloc[i]:
                signals['signal_desc'].iloc[i] = f"趋势由{'下降' if result['TrendDirection'].shift(1).iloc[i] == -1 else '震荡'}转为上升，强度{result['TrendStrength'].iloc[i]:.1f}"
            elif sell_signal_transition.iloc[i]:
                signals['signal_desc'].iloc[i] = f"趋势由{'上升' if result['TrendDirection'].shift(1).iloc[i] == 1 else '震荡'}转为下降，强度{result['TrendStrength'].iloc[i]:.1f}"
            elif result['TrendDirection'].iloc[i] == 1:
                signals['signal_desc'].iloc[i] = f"上升趋势，强度{result['TrendStrength'].iloc[i]:.1f}，稳定性{result['TrendStability'].iloc[i]:.1f}"
            elif result['TrendDirection'].iloc[i] == -1:
                signals['signal_desc'].iloc[i] = f"下降趋势，强度{result['TrendStrength'].iloc[i]:.1f}，稳定性{result['TrendStability'].iloc[i]:.1f}"
            else:
                signals['signal_desc'].iloc[i] = "无明显趋势"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据趋势强度调整置信度
        strength_confidence_adj = ((result['TrendStrength'] - 50) / 50 * 20).clip(-20, 20)
        signals['confidence'] = signals['confidence'] + strength_confidence_adj
        
        # 根据趋势稳定性调整置信度
        stability_confidence_adj = ((result['TrendStability'] - 50) / 50 * 20).clip(-20, 20)
        signals['confidence'] = signals['confidence'] + stability_confidence_adj
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 添加风险等级
        signals['risk_level'] = '中'
        signals.loc[signals['confidence'] >= 80, 'risk_level'] = '低'
        signals.loc[signals['confidence'] <= 40, 'risk_level'] = '高'
        
        # 设置仓位建议
        signals['position_size'] = signals['confidence'] / 200  # 0-0.5范围
        
        # 设置止损建议 (基于价格和波动率)
        volatility = result['Volatility'] / 100 * data['close']
        signals['stop_loss'] = data['close']
        
        for i in range(len(signals)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            if signals['buy_signal'].iloc[i]:
                # 买入信号的止损：根据趋势强度设置不同的止损距离
                if result['TrendStrength'].iloc[i] > 70:
                    # 强趋势设置宽松止损
                    signals['stop_loss'].iloc[i] = data['close'].iloc[i] - (volatility.iloc[i] * 2.5)
                else:
                    # 一般趋势设置适中止损
                    signals['stop_loss'].iloc[i] = data['close'].iloc[i] - (volatility.iloc[i] * 2.0)
            elif signals['sell_signal'].iloc[i]:
                # 卖出信号的止损：同样根据趋势强度调整
                if result['TrendStrength'].iloc[i] > 70:
                    signals['stop_loss'].iloc[i] = data['close'].iloc[i] + (volatility.iloc[i] * 2.5)
                else:
                    signals['stop_loss'].iloc[i] = data['close'].iloc[i] + (volatility.iloc[i] * 2.0)
        
        return signals


class TrendDuration(BaseIndicator):
    """
    ZXM趋势持续性指标
    
    分析价格趋势的持续时间和生命周期特征
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """初始化ZXM趋势持续性指标"""
        super().__init__(name="TrendDuration", description="ZXM趋势持续性指标，分析价格趋势的持续时间和生命周期特征")
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算趋势持续性指标
        
        Args:
            data: 输入数据，包含OHLC价格数据
            
        Returns:
            pd.DataFrame: 计算结果，包含趋势持续时间相关指标
        """
        self.ensure_columns(data, ["close"])
        
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 计算多个周期的移动平均线
        result['MA10'] = data['close'].rolling(window=10).mean()
        result['MA20'] = data['close'].rolling(window=20).mean()
        result['MA60'] = data['close'].rolling(window=60).mean()
        
        # 计算均线斜率
        result['MA10_Slope'] = result['MA10'].diff(5) / 5
        result['MA20_Slope'] = result['MA20'].diff(5) / 5
        result['MA60_Slope'] = result['MA60'].diff(10) / 10
        
        # 确定趋势状态
        # 1: 上升趋势, -1: 下降趋势, 0: 横盘/无趋势
        result['TrendState'] = 0
        
        # 上升趋势条件: 短期均线和中期均线向上 且 短期均线高于中期均线
        up_trend = (result['MA10_Slope'] > 0) & (result['MA20_Slope'] > 0) & (result['MA10'] > result['MA20'])
        result.loc[up_trend, 'TrendState'] = 1
        
        # 下降趋势条件: 短期均线和中期均线向下 且 短期均线低于中期均线
        down_trend = (result['MA10_Slope'] < 0) & (result['MA20_Slope'] < 0) & (result['MA10'] < result['MA20'])
        result.loc[down_trend, 'TrendState'] = -1
        
        # 计算趋势变化点
        result['TrendChange'] = result['TrendState'].diff().abs() > 0
        
        # 计算趋势持续时间
        result['TrendDuration'] = 0
        
        # 遍历计算趋势持续时间
        current_trend = 0
        duration = 0
        
        for i in range(len(result)):
            if i == 0 or result['TrendChange'].iloc[i]:
                # 趋势变化或首个数据点
                current_trend = result['TrendState'].iloc[i]
                duration = 1
            else:
                # 趋势持续
                if result['TrendState'].iloc[i] == current_trend:
                    duration += 1
                else:
                    # 状态变化但未触发TrendChange（可能是从震荡到有趋势）
                    current_trend = result['TrendState'].iloc[i]
                    duration = 1
            
            result['TrendDuration'].iloc[i] = duration
        
        # 计算典型趋势持续期的统计特性
        # 获取历史趋势的持续时间数据
        up_trend_durations = []
        down_trend_durations = []
        
        trend_start = 0
        current_trend = 0
        
        for i in range(1, len(result)):
            if result['TrendChange'].iloc[i]:
                if current_trend == 1 and i - trend_start > 5:  # 仅考虑持续至少5天的趋势
                    up_trend_durations.append(i - trend_start)
                elif current_trend == -1 and i - trend_start > 5:
                    down_trend_durations.append(i - trend_start)
                
                trend_start = i
                current_trend = result['TrendState'].iloc[i]
        
        # 计算平均趋势持续时间
        avg_up_duration = np.mean(up_trend_durations) if up_trend_durations else 0
        avg_down_duration = np.mean(down_trend_durations) if down_trend_durations else 0
        
        # 计算趋势生命周期阶段
        # 0: 初始阶段, 1: 发展阶段, 2: 成熟阶段, 3: 衰退阶段
        result['TrendLifecyclePhase'] = 0
        
        for i in range(len(result)):
            if result['TrendState'].iloc[i] == 0:
                # 无趋势
                result['TrendLifecyclePhase'].iloc[i] = 0
            else:
                # 有趋势
                duration = result['TrendDuration'].iloc[i]
                avg_duration = avg_up_duration if result['TrendState'].iloc[i] == 1 else avg_down_duration
                
                if avg_duration == 0:
                    # 没有历史数据，使用默认值
                    avg_duration = 30 if result['TrendState'].iloc[i] == 1 else 20
                
                # 根据当前持续时间与平均持续时间的比例确定生命周期阶段
                ratio = duration / avg_duration
                
                if ratio < 0.3:
                    # 初始阶段
                    result['TrendLifecyclePhase'].iloc[i] = 0
                elif ratio < 0.7:
                    # 发展阶段
                    result['TrendLifecyclePhase'].iloc[i] = 1
                elif ratio < 1.1:
                    # 成熟阶段
                    result['TrendLifecyclePhase'].iloc[i] = 2
                else:
                    # 衰退阶段
                    result['TrendLifecyclePhase'].iloc[i] = 3
        
        # 计算趋势成熟度（0-100）
        result['TrendMaturity'] = 0
        
        for i in range(len(result)):
            if result['TrendState'].iloc[i] == 0:
                # 无趋势
                result['TrendMaturity'].iloc[i] = 0
            else:
                # 有趋势
                duration = result['TrendDuration'].iloc[i]
                avg_duration = avg_up_duration if result['TrendState'].iloc[i] == 1 else avg_down_duration
                
                if avg_duration == 0:
                    # 没有历史数据，使用默认值
                    avg_duration = 30 if result['TrendState'].iloc[i] == 1 else 20
                
                # 计算成熟度百分比
                maturity = min(100, (duration / avg_duration) * 100)
                result['TrendMaturity'].iloc[i] = maturity
        
        # 计算趋势剩余寿命估计
        result['EstimatedRemainingDuration'] = 0
        
        for i in range(len(result)):
            if result['TrendState'].iloc[i] == 0:
                # 无趋势
                result['EstimatedRemainingDuration'].iloc[i] = 0
            else:
                # 有趋势
                duration = result['TrendDuration'].iloc[i]
                avg_duration = avg_up_duration if result['TrendState'].iloc[i] == 1 else avg_down_duration
                
                if avg_duration == 0:
                    # 没有历史数据，使用默认值
                    avg_duration = 30 if result['TrendState'].iloc[i] == 1 else 20
                
                # 估计剩余寿命
                remaining = max(0, avg_duration - duration)
                result['EstimatedRemainingDuration'].iloc[i] = remaining
        
        # 计算周期规律性（低值表示更规律）
        # 计算趋势持续时间的标准差与平均值的比率
        up_std = np.std(up_trend_durations) if up_trend_durations else 0
        down_std = np.std(down_trend_durations) if down_trend_durations else 0
        
        up_cv = up_std / avg_up_duration if avg_up_duration else 0
        down_cv = down_std / avg_down_duration if avg_down_duration else 0
        
        # 存储周期规律性指标
        result['CycleRegularity'] = 0
        
        for i in range(len(result)):
            if result['TrendState'].iloc[i] == 1:
                result['CycleRegularity'].iloc[i] = min(1, up_cv)
            elif result['TrendState'].iloc[i] == -1:
                result['CycleRegularity'].iloc[i] = min(1, down_cv)
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算趋势持续性指标的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 0-100的评分，>50表示看涨，<50表示看跌
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分序列
        score = pd.Series(50, index=data.index)  # 默认中性分数
        
        # 遍历每个时间点进行评分
        for i in range(len(score)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            # 基础评分：基于趋势状态
            trend_state = result['TrendState'].iloc[i]
            if trend_state == 1:  # 上升趋势
                base_score = 60
            elif trend_state == -1:  # 下降趋势
                base_score = 40
            else:  # 无趋势
                base_score = 50
            
            # 生命周期阶段评分
            lifecycle_phase = result['TrendLifecyclePhase'].iloc[i]
            if trend_state == 1:  # 上升趋势
                if lifecycle_phase == 0:  # 初始阶段
                    lifecycle_score = 15  # 初始阶段最看好
                elif lifecycle_phase == 1:  # 发展阶段
                    lifecycle_score = 10
                elif lifecycle_phase == 2:  # 成熟阶段
                    lifecycle_score = 5
                else:  # 衰退阶段
                    lifecycle_score = -10  # 衰退阶段看跌
            elif trend_state == -1:  # 下降趋势
                if lifecycle_phase == 0:  # 初始阶段
                    lifecycle_score = -15  # 初始下降趋势最看空
                elif lifecycle_phase == 1:  # 发展阶段
                    lifecycle_score = -10
                elif lifecycle_phase == 2:  # 成熟阶段
                    lifecycle_score = -5
                else:  # 衰退阶段
                    lifecycle_score = 10  # 下降趋势的衰退阶段可能即将反转
            else:  # 无趋势
                lifecycle_score = 0
            
            # 趋势持续时间评分
            # 持续时间长的趋势更可能接近尾声
            duration = result['TrendDuration'].iloc[i]
            if duration > 60:  # 极长趋势
                duration_score = -10 * np.sign(trend_state)  # 反向打分
            elif duration > 30:  # 长趋势
                duration_score = -5 * np.sign(trend_state)  # 反向打分
            else:
                duration_score = 0
            
            # 周期规律性评分
            # 规律性高的趋势可能更可预测
            regularity = result['CycleRegularity'].iloc[i]
            if regularity < 0.3:  # 高规律性
                regularity_score = 5
            else:
                regularity_score = 0
            
            # 计算最终评分
            score.iloc[i] = base_score + lifecycle_score + duration_score + regularity_score
        
        # 趋势变化点额外调整
        trend_change = result["TrendChange"]
        trend_state = result["TrendState"]
        
        # 趋势刚开始变化时给予更明确的信号
        up_change = trend_change & (trend_state == 1)
        score[up_change] = 75  # 刚转为上升趋势
        
        down_change = trend_change & (trend_state == -1)
        score[down_change] = 25  # 刚转为下降趋势
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ZXM趋势持续性指标相关的技术形态
        
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
        if len(result) > 60:  # 确保有足够的数据
            last_row = result.iloc[-1]
            
            # 趋势状态和持续时间
            trend_state = last_row['TrendState']
            duration = last_row['TrendDuration']
            lifecycle_phase = last_row['TrendLifecyclePhase']
            maturity = last_row['TrendMaturity']
            regularity = last_row['CycleRegularity']
            
            # 趋势状态
            if trend_state == 1:
                patterns.append("上升趋势")
                
                # 趋势持续时间
                if duration >= 60:
                    patterns.append("超长期上升趋势")
                elif duration >= 30:
                    patterns.append("长期上升趋势")
                elif duration >= 10:
                    patterns.append("中期上升趋势")
                else:
                    patterns.append("短期上升趋势")
                
                # 生命周期阶段
                if lifecycle_phase == 0:
                    patterns.append("上升趋势初始阶段")
                elif lifecycle_phase == 1:
                    patterns.append("上升趋势发展阶段")
                elif lifecycle_phase == 2:
                    patterns.append("上升趋势成熟阶段")
                else:
                    patterns.append("上升趋势衰退阶段")
                
            elif trend_state == -1:
                patterns.append("下降趋势")
                
                # 趋势持续时间
                if duration >= 60:
                    patterns.append("超长期下降趋势")
                elif duration >= 30:
                    patterns.append("长期下降趋势")
                elif duration >= 10:
                    patterns.append("中期下降趋势")
                else:
                    patterns.append("短期下降趋势")
                
                # 生命周期阶段
                if lifecycle_phase == 0:
                    patterns.append("下降趋势初始阶段")
                elif lifecycle_phase == 1:
                    patterns.append("下降趋势发展阶段")
                elif lifecycle_phase == 2:
                    patterns.append("下降趋势成熟阶段")
                else:
                    patterns.append("下降趋势衰退阶段")
                
            else:
                patterns.append("震荡/无趋势")
            
            # 趋势变化
            if last_row['TrendChange']:
                if trend_state == 1:
                    patterns.append("趋势刚转为上升")
                elif trend_state == -1:
                    patterns.append("趋势刚转为下降")
            
            # 趋势成熟度
            if maturity > 90:
                patterns.append("趋势接近尾声")
            elif maturity > 70:
                patterns.append("趋势成熟期")
            elif maturity < 30:
                patterns.append("趋势初期")
            
            # 周期规律性
            if regularity < 0.3:
                patterns.append("高规律性周期")
            elif regularity > 0.7:
                patterns.append("低规律性周期")
        
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
        
        # 买入信号：
        # 1. 趋势刚转为上升
        # 2. 上升趋势初期（生命周期阶段为0或1）
        buy_condition1 = result['TrendChange'] & (result['TrendState'] == 1)
        buy_condition2 = (result['TrendState'] == 1) & (result['TrendLifecyclePhase'].isin([0, 1]))
        signals['buy_signal'] = buy_condition1 | buy_condition2
        
        # 卖出信号：
        # 1. 趋势刚转为下降
        # 2. 上升趋势接近尾声（成熟度>90）
        # 3. 下降趋势初期
        sell_condition1 = result['TrendChange'] & (result['TrendState'] == -1)
        sell_condition2 = (result['TrendState'] == 1) & (result['TrendMaturity'] > 90)
        sell_condition3 = (result['TrendState'] == -1) & (result['TrendLifecyclePhase'] == 0)
        signals['sell_signal'] = sell_condition1 | sell_condition2 | sell_condition3
        
        # 中性信号
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 设置趋势
        signals['trend'] = result['TrendState']
        
        # 设置评分
        signals['score'] = score
        
        # 设置信号类型
        signals['signal_type'] = 'neutral'
        signals.loc[buy_condition1, 'signal_type'] = 'trend_start_bullish'
        signals.loc[buy_condition2 & ~buy_condition1, 'signal_type'] = 'early_trend_bullish'
        signals.loc[sell_condition1, 'signal_type'] = 'trend_start_bearish'
        signals.loc[sell_condition2, 'signal_type'] = 'trend_exhaustion_bearish'
        signals.loc[sell_condition3 & ~sell_condition1, 'signal_type'] = 'early_trend_bearish'
        
        # 设置信号描述
        signals['signal_desc'] = ''
        
        # 为每个信号设置详细描述
        for i in range(len(signals)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            if buy_condition1.iloc[i]:
                signals['signal_desc'].iloc[i] = "趋势刚转为上升，买入时机"
            elif buy_condition2.iloc[i] and not buy_condition1.iloc[i]:
                lifecycle = "初始" if result['TrendLifecyclePhase'].iloc[i] == 0 else "发展"
                signals['signal_desc'].iloc[i] = f"上升趋势{lifecycle}阶段，持续{result['TrendDuration'].iloc[i]:.0f}天"
            elif sell_condition1.iloc[i]:
                signals['signal_desc'].iloc[i] = "趋势刚转为下降，卖出时机"
            elif sell_condition2.iloc[i]:
                signals['signal_desc'].iloc[i] = f"上升趋势接近尾声，成熟度{result['TrendMaturity'].iloc[i]:.1f}%"
            elif sell_condition3.iloc[i] and not sell_condition1.iloc[i]:
                signals['signal_desc'].iloc[i] = f"下降趋势初期，持续{result['TrendDuration'].iloc[i]:.0f}天"
            elif result['TrendState'].iloc[i] == 1:
                phase = ["初始", "发展", "成熟", "衰退"][int(result['TrendLifecyclePhase'].iloc[i])]
                signals['signal_desc'].iloc[i] = f"上升趋势{phase}阶段，持续{result['TrendDuration'].iloc[i]:.0f}天，成熟度{result['TrendMaturity'].iloc[i]:.1f}%"
            elif result['TrendState'].iloc[i] == -1:
                phase = ["初始", "发展", "成熟", "衰退"][int(result['TrendLifecyclePhase'].iloc[i])]
                signals['signal_desc'].iloc[i] = f"下降趋势{phase}阶段，持续{result['TrendDuration'].iloc[i]:.0f}天，成熟度{result['TrendMaturity'].iloc[i]:.1f}%"
            else:
                signals['signal_desc'].iloc[i] = "震荡/无趋势"
        
        # 置信度设置
        signals['confidence'] = 60  # 基础置信度
        
        # 根据生命周期阶段和规律性调整置信度
        for i in range(len(signals)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            # 生命周期阶段调整
            phase = result['TrendLifecyclePhase'].iloc[i]
            if result['TrendState'].iloc[i] != 0:  # 有明确趋势
                if phase == 0:  # 初始阶段
                    signals['confidence'].iloc[i] += 10
                elif phase == 3:  # 衰退阶段
                    signals['confidence'].iloc[i] -= 5
            
            # 规律性调整
            regularity = result['CycleRegularity'].iloc[i]
            if regularity < 0.3:  # 高规律性
                signals['confidence'].iloc[i] += 10
            elif regularity > 0.7:  # 低规律性
                signals['confidence'].iloc[i] -= 10
            
            # 趋势变化点调整
            if result['TrendChange'].iloc[i]:
                # 变化点的置信度略低
                signals['confidence'].iloc[i] -= 5
        
        # 确保置信度在0-100范围内
        signals['confidence'] = signals['confidence'].clip(0, 100)
        
        # 添加风险等级
        signals['risk_level'] = '中'
        signals.loc[signals['confidence'] >= 80, 'risk_level'] = '低'
        signals.loc[signals['confidence'] <= 40, 'risk_level'] = '高'
        
        # 设置仓位建议
        signals['position_size'] = signals['confidence'] / 200  # 0-0.5范围
        
        # 设置止损建议
        # 对于上升趋势，使用预计趋势剩余寿命来调整止损位置
        for i in range(len(signals)):
            if i < 60:  # 跳过没有足够数据的点
                continue
                
            signals['stop_loss'].iloc[i] = data['close'].iloc[i]  # 默认值
            
            if signals['buy_signal'].iloc[i]:
                # 计算止损位：基于波动率和趋势剩余寿命
                volatility = data['close'].iloc[i] * data['close'].pct_change().rolling(window=20).std().iloc[i]
                remaining = result['EstimatedRemainingDuration'].iloc[i]
                
                # 根据剩余寿命调整止损距离
                if remaining > 20:  # 预计还有较长时间
                    stop_distance = 2.5 * volatility
                elif remaining > 10:
                    stop_distance = 2.0 * volatility
                else:
                    stop_distance = 1.5 * volatility
                
                signals['stop_loss'].iloc[i] = data['close'].iloc[i] - stop_distance
                
            elif signals['sell_signal'].iloc[i]:
                # 类似逻辑用于卖出信号
                volatility = data['close'].iloc[i] * data['close'].pct_change().rolling(window=20).std().iloc[i]
                remaining = result['EstimatedRemainingDuration'].iloc[i]
                
                if remaining > 20:
                    stop_distance = 2.5 * volatility
                elif remaining > 10:
                    stop_distance = 2.0 * volatility
                else:
                    stop_distance = 1.5 * volatility
                
                signals['stop_loss'].iloc[i] = data['close'].iloc[i] + stop_distance
        
        # 确保所有必要的列都存在
        if 'market_env' not in signals.columns:
            signals['market_env'] = 'normal'
        
        if 'volume_confirmation' not in signals.columns:
            signals['volume_confirmation'] = False
        
        return signals


class ZXMWeeklyMACD(BaseIndicator):
    """
    ZXM系统周线MACD指标
    用于检测中期趋势的变化和买卖信号
    特点是对中期趋势敏感，可靠性高于日线MACD
    """
    
    def __init__(self):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        super().__init__()
        self.name = "ZXM周线MACD指标"
        self.description = "基于周线数据的MACD指标，用于检测中期趋势变化"
        self.score_manager = IndicatorScoreManager()
    
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算周线MACD指标
        
        Args:
            data: DataFrame，必须包含'close'列
            
        Returns:
            DataFrame: 包含计算结果的DataFrame
        """
        if len(data) < 40:  # 确保数据量足够
            return data
        
        result_data = data.copy()
        
        # 计算MACD指标 (12,26,9)
        close = result_data['close']
        
        # 计算EMA
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        # 计算DIF、DEA和MACD
        result_data['DIF'] = ema12 - ema26
        result_data['DEA'] = result_data['DIF'].ewm(span=9, adjust=False).mean()
        result_data['MACD'] = 2 * (result_data['DIF'] - result_data['DEA'])
        
        # 计算DIF和DEA的斜率（趋势方向）
        result_data['DIF_slope'] = result_data['DIF'].diff(3)
        result_data['DEA_slope'] = result_data['DEA'].diff(3)
        
        # 判断金叉和死叉
        result_data['golden_cross'] = (result_data['DIF'] > result_data['DEA']) & (result_data['DIF'].shift(1) <= result_data['DEA'].shift(1))
        result_data['death_cross'] = (result_data['DIF'] < result_data['DEA']) & (result_data['DIF'].shift(1) >= result_data['DEA'].shift(1))
        
        # 判断零轴上下方
        result_data['above_zero'] = result_data['DIF'] > 0
        result_data['below_zero'] = result_data['DIF'] < 0
        
        # 计算柱状图变化
        result_data['hist_increase'] = result_data['MACD'] > result_data['MACD'].shift(1)
        result_data['hist_decrease'] = result_data['MACD'] < result_data['MACD'].shift(1)
        
        # 计算背离
        self._calculate_divergence(result_data)
        
        return result_data
    
    def _calculate_divergence(self, data: pd.DataFrame) -> None:
        """
        计算MACD与价格的背离情况
        
        Args:
            data: DataFrame，包含MACD指标和价格数据
        """
        # 初始化背离列
        data['bullish_divergence'] = False
        data['bearish_divergence'] = False
        
        # 寻找价格和DIF的低点和高点
        window = 8  # 用于寻找局部极值的窗口大小
        
        # 确保数据量足够
        if len(data) < window * 2:
            return
        
        # 寻找局部低点（价格创新低但DIF不创新低）
        for i in range(window, len(data) - window):
            # 检查价格是否为局部低点
            if (data['close'].iloc[i] < data['close'].iloc[i-window:i].min()) and \
               (data['close'].iloc[i] < data['close'].iloc[i+1:i+window+1].min()):
                
                # 找前一个低点
                prev_low_idx = None
                for j in range(i-window, i):
                    if (data['close'].iloc[j] < data['close'].iloc[max(0, j-window):j].min()) and \
                       (data['close'].iloc[j] < data['close'].iloc[j+1:j+window+1].min()):
                        prev_low_idx = j
                        break
                
                if prev_low_idx is not None and data['close'].iloc[i] < data['close'].iloc[prev_low_idx]:
                    # 价格创新低
                    if data['DIF'].iloc[i] > data['DIF'].iloc[prev_low_idx]:
                        # DIF不创新低，形成底背离
                        data.loc[data.index[i], 'bullish_divergence'] = True
        
        # 寻找局部高点（价格创新高但DIF不创新高）
        for i in range(window, len(data) - window):
            # 检查价格是否为局部高点
            if (data['close'].iloc[i] > data['close'].iloc[i-window:i].max()) and \
               (data['close'].iloc[i] > data['close'].iloc[i+1:i+window+1].max()):
                
                # 找前一个高点
                prev_high_idx = None
                for j in range(i-window, i):
                    if (data['close'].iloc[j] > data['close'].iloc[max(0, j-window):j].max()) and \
                       (data['close'].iloc[j] > data['close'].iloc[j+1:j+window+1].max()):
                        prev_high_idx = j
                        break
                
                if prev_high_idx is not None and data['close'].iloc[i] > data['close'].iloc[prev_high_idx]:
                    # 价格创新高
                    if data['DIF'].iloc[i] < data['DIF'].iloc[prev_high_idx]:
                        # DIF不创新高，形成顶背离
                        data.loc[data.index[i], 'bearish_divergence'] = True
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算原始评分（0-100分）
        
        Args:
            data: DataFrame，必须包含MACD计算结果
            
        Returns:
            Series: 包含每个时间点评分的Series
        """
        if 'DIF' not in data.columns or 'DEA' not in data.columns:
            data = self.calculate(data)
        
        # 初始化评分为50（中性）
        scores = pd.Series(50, index=data.index)
        
        # 1. 基础MACD评分
        # DIF和DEA位置关系评分
        for i in range(len(data)):
            score = 50
            
            # 金叉加分
            if data['golden_cross'].iloc[i]:
                score += 25
            
            # 死叉减分
            if data['death_cross'].iloc[i]:
                score -= 25
            
            # DIF和DEA在零轴上方加分
            if data['above_zero'].iloc[i]:
                if data['DIF'].iloc[i] > data['DEA'].iloc[i]:
                    score += 15  # 零轴上方且金叉
                else:
                    score += 5   # 零轴上方但死叉
            
            # DIF和DEA在零轴下方减分
            if data['below_zero'].iloc[i]:
                if data['DIF'].iloc[i] < data['DEA'].iloc[i]:
                    score -= 15  # 零轴下方且死叉
                else:
                    score -= 5   # 零轴下方但金叉
            
            # DIF穿越零轴评分
            if i > 0:
                if data['DIF'].iloc[i] > 0 and data['DIF'].iloc[i-1] <= 0:
                    score += 20  # DIF上穿零轴
                if data['DIF'].iloc[i] < 0 and data['DIF'].iloc[i-1] >= 0:
                    score -= 20  # DIF下穿零轴
            
            # 柱状图变化评分
            if data['hist_increase'].iloc[i]:
                score += 10  # 柱状图增加
            if data['hist_decrease'].iloc[i]:
                score -= 10  # 柱状图减少
            
            # 背离评分
            if data['bullish_divergence'].iloc[i]:
                score += 30  # 底背离，强烈买入信号
            if data['bearish_divergence'].iloc[i]:
                score -= 30  # 顶背离，强烈卖出信号
            
            # 趋势强度评分 - 根据DIF和DEA的斜率
            if i > 3:
                if data['DIF_slope'].iloc[i] > 0:
                    score += 10 * min(1, data['DIF_slope'].iloc[i])  # DIF向上倾斜
                else:
                    score -= 10 * min(1, abs(data['DIF_slope'].iloc[i]))  # DIF向下倾斜
                
                if data['DEA_slope'].iloc[i] > 0:
                    score += 5 * min(1, data['DEA_slope'].iloc[i])  # DEA向上倾斜
                else:
                    score -= 5 * min(1, abs(data['DEA_slope'].iloc[i]))  # DEA向下倾斜
            
            # 限制评分范围在0-100之间
            scores.iloc[i] = max(0, min(100, score))
        
        return scores
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MACD指标中的技术形态
        
        Args:
            data: DataFrame，包含MACD计算结果
            
        Returns:
            List[str]: 识别出的技术形态列表
        """
        if 'DIF' not in data.columns or 'DEA' not in data.columns:
            data = self.calculate(data)
        
        patterns = []
        
        # 只关注最近的数据
        recent_data = data.iloc[-10:]
        
        # 1. 金叉形态
        if recent_data['golden_cross'].any():
            if recent_data['above_zero'].iloc[-1]:
                patterns.append("周线MACD零轴上方金叉：强势上涨信号")
            else:
                patterns.append("周线MACD零轴下方金叉：反转向上信号")
        
        # 2. 死叉形态
        if recent_data['death_cross'].any():
            if recent_data['below_zero'].iloc[-1]:
                patterns.append("周线MACD零轴下方死叉：强势下跌信号")
            else:
                patterns.append("周线MACD零轴上方死叉：反转向下信号")
        
        # 3. 零轴穿越
        if any(recent_data['DIF'].diff().iloc[1:] > 0) and recent_data['DIF'].iloc[-1] > 0:
            patterns.append("周线MACD-DIF上穿零轴：中期趋势转为向上")
        if any(recent_data['DIF'].diff().iloc[1:] < 0) and recent_data['DIF'].iloc[-1] < 0:
            patterns.append("周线MACD-DIF下穿零轴：中期趋势转为向下")
        
        # 4. 背离形态
        if recent_data['bullish_divergence'].any():
            patterns.append("周线MACD底背离：强烈反转向上信号")
        if recent_data['bearish_divergence'].any():
            patterns.append("周线MACD顶背离：强烈反转向下信号")
        
        # 5. 柱状图形态
        if len(recent_data) >= 3:
            # 连续3周柱状图增加
            if all(recent_data['MACD'].diff().iloc[-3:] > 0):
                patterns.append("周线MACD柱状图连续增加：动能增强")
            # 连续3周柱状图减少
            if all(recent_data['MACD'].diff().iloc[-3:] < 0):
                patterns.append("周线MACD柱状图连续减少：动能减弱")
        
        # 6. 0轴附近徘徊
        if all(abs(recent_data['DIF'].iloc[-3:]) < 0.1):
            patterns.append("周线MACD在零轴附近徘徊：趋势不明确")
        
        # 7. 高位钝化
        if (recent_data['DIF'].iloc[-1] > 0) and (recent_data['DIF'].diff().iloc[-3:].mean() < 0) and (recent_data['DIF'].iloc[-1] > 2 * recent_data['DIF'].std()):
            patterns.append("周线MACD高位钝化：上涨动能减弱")
        
        # 8. 低位钝化
        if (recent_data['DIF'].iloc[-1] < 0) and (recent_data['DIF'].diff().iloc[-3:].mean() > 0) and (abs(recent_data['DIF'].iloc[-1]) > 2 * recent_data['DIF'].std()):
            patterns.append("周线MACD低位钝化：下跌动能减弱")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: DataFrame，包含MACD计算结果
            
        Returns:
            DataFrame: 包含交易信号的DataFrame
        """
        if 'DIF' not in data.columns or 'DEA' not in data.columns:
            data = self.calculate(data)
        
        # 计算评分
        scores = self.calculate_raw_score(data)
        
        # 识别形态
        patterns = self.identify_patterns(data)
        
        # 初始化信号DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True  # 默认中性信号
        signals['score'] = scores
        signals['trend'] = 0  # 默认中性趋势
        signals['signal_type'] = 'neutral'
        signals['signal_desc'] = ''
        signals['confidence'] = 70.0  # 默认置信度
        signals['risk_level'] = 'medium'  # 默认风险等级
        signals['position_size'] = 0.0  # 默认仓位
        signals['stop_loss'] = np.nan  # 默认止损
        signals['market_env'] = 'normal'  # 默认市场环境
        signals['volume_confirmation'] = False  # 默认成交量确认
        
        # 生成买入信号
        buy_conditions = (
            data['golden_cross'] |  # 金叉
            data['bullish_divergence'] |  # 底背离
            ((data['DIF'] > 0) & (data['DIF'].shift(1) <= 0))  # DIF上穿零轴
        )
        signals.loc[buy_conditions, 'buy_signal'] = True
        signals.loc[buy_conditions, 'neutral_signal'] = False
        signals.loc[buy_conditions, 'trend'] = 1
        
        # 生成卖出信号
        sell_conditions = (
            data['death_cross'] |  # 死叉
            data['bearish_divergence'] |  # 顶背离
            ((data['DIF'] < 0) & (data['DIF'].shift(1) >= 0))  # DIF下穿零轴
        )
        signals.loc[sell_conditions, 'sell_signal'] = True
        signals.loc[sell_conditions, 'neutral_signal'] = False
        signals.loc[sell_conditions, 'trend'] = -1
        
        # 根据不同条件设置信号类型和描述
        for i in range(len(signals)):
            # 跳过开头几个不完整的数据点
            if i < 30:
                signals['signal_type'].iloc[i] = 'insufficient_data'
                signals['signal_desc'].iloc[i] = '数据不足'
                continue
                
            if signals['buy_signal'].iloc[i]:
                if data['golden_cross'].iloc[i]:
                    signals.loc[signals.index[i], 'signal_type'] = 'macd_golden_cross'
                    if data['above_zero'].iloc[i]:
                        signals.loc[signals.index[i], 'signal_desc'] = '周线MACD零轴上方金叉，强势上涨信号'
                        signals.loc[signals.index[i], 'confidence'] = 85.0
                        signals.loc[signals.index[i], 'risk_level'] = 'low'
                        signals.loc[signals.index[i], 'position_size'] = 0.7
                    else:
                        signals.loc[signals.index[i], 'signal_desc'] = '周线MACD零轴下方金叉，反转向上信号'
                        signals.loc[signals.index[i], 'confidence'] = 75.0
                        signals.loc[signals.index[i], 'risk_level'] = 'medium'
                        signals.loc[signals.index[i], 'position_size'] = 0.5
                elif data['bullish_divergence'].iloc[i]:
                    signals.loc[signals.index[i], 'signal_type'] = 'macd_bullish_divergence'
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD底背离，强烈反转向上信号'
                    signals.loc[signals.index[i], 'confidence'] = 90.0
                    signals.loc[signals.index[i], 'risk_level'] = 'low'
                    signals.loc[signals.index[i], 'position_size'] = 0.8
                elif (data['DIF'].iloc[i] > 0) and (data['DIF'].iloc[i-1] <= 0):
                    signals.loc[signals.index[i], 'signal_type'] = 'macd_zero_line_cross_up'
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD-DIF上穿零轴，中期趋势转为向上'
                    signals.loc[signals.index[i], 'confidence'] = 80.0
                    signals.loc[signals.index[i], 'risk_level'] = 'medium'
                    signals.loc[signals.index[i], 'position_size'] = 0.6
            
            elif signals['sell_signal'].iloc[i]:
                if data['death_cross'].iloc[i]:
                    signals.loc[signals.index[i], 'signal_type'] = 'macd_death_cross'
                    if data['below_zero'].iloc[i]:
                        signals.loc[signals.index[i], 'signal_desc'] = '周线MACD零轴下方死叉，强势下跌信号'
                        signals.loc[signals.index[i], 'confidence'] = 85.0
                        signals.loc[signals.index[i], 'risk_level'] = 'high'
                        signals.loc[signals.index[i], 'position_size'] = 0.0
                    else:
                        signals.loc[signals.index[i], 'signal_desc'] = '周线MACD零轴上方死叉，反转向下信号'
                        signals.loc[signals.index[i], 'confidence'] = 75.0
                        signals.loc[signals.index[i], 'risk_level'] = 'medium'
                        signals.loc[signals.index[i], 'position_size'] = 0.3
                elif data['bearish_divergence'].iloc[i]:
                    signals.loc[signals.index[i], 'signal_type'] = 'macd_bearish_divergence'
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD顶背离，强烈反转向下信号'
                    signals.loc[signals.index[i], 'confidence'] = 90.0
                    signals.loc[signals.index[i], 'risk_level'] = 'high'
                    signals.loc[signals.index[i], 'position_size'] = 0.0
                elif (data['DIF'].iloc[i] < 0) and (data['DIF'].iloc[i-1] >= 0):
                    signals.loc[signals.index[i], 'signal_type'] = 'macd_zero_line_cross_down'
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD-DIF下穿零轴，中期趋势转为向下'
                    signals.loc[signals.index[i], 'confidence'] = 80.0
                    signals.loc[signals.index[i], 'risk_level'] = 'medium'
                    signals.loc[signals.index[i], 'position_size'] = 0.2
            
            else:
                # 中性信号
                signals.loc[signals.index[i], 'signal_type'] = 'neutral'
                if data['DIF'].iloc[i] > data['DEA'].iloc[i] and data['DIF'].iloc[i] > 0:
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD处于多头排列，但无明显买入信号'
                    signals.loc[signals.index[i], 'confidence'] = 60.0
                    signals.loc[signals.index[i], 'position_size'] = 0.5
                elif data['DIF'].iloc[i] < data['DEA'].iloc[i] and data['DIF'].iloc[i] < 0:
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD处于空头排列，但无明显卖出信号'
                    signals.loc[signals.index[i], 'confidence'] = 60.0
                    signals.loc[signals.index[i], 'position_size'] = 0.3
                else:
                    signals.loc[signals.index[i], 'signal_desc'] = '周线MACD无明显信号'
                    signals.loc[signals.index[i], 'confidence'] = 50.0
                    signals.loc[signals.index[i], 'position_size'] = 0.4
            
            # 设置市场环境
            if data['DIF'].iloc[i] > 0 and data['DEA'].iloc[i] > 0:
                signals.loc[signals.index[i], 'market_env'] = 'bull_market'
            elif data['DIF'].iloc[i] < 0 and data['DEA'].iloc[i] < 0:
                signals.loc[signals.index[i], 'market_env'] = 'bear_market'
            else:
                signals.loc[signals.index[i], 'market_env'] = 'sideways_market'
                
            # 计算止损位
            if signals['buy_signal'].iloc[i]:
                # 买入信号的止损设置在近期低点
                stop_window = 10
                if i >= stop_window:
                    signals.loc[signals.index[i], 'stop_loss'] = data['close'].iloc[i-stop_window:i+1].min() * 0.95
            elif signals['sell_signal'].iloc[i]:
                # 卖出信号不需要设置止损
                pass
        
        return signals

    def _calculate_trend_stability(self, result: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算趋势稳定性
        
        Args:
            result: 包含趋势方向的DataFrame
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 添加了趋势稳定性的DataFrame
        """
        try:
            # 初始化趋势稳定性列
            result = result.copy()
            result['TrendStability'] = 0.0
            
            if 'TrendDirection' not in result.columns:
                logger.warning("没有TrendDirection列，无法计算趋势稳定性")
                return result
                
            # 获取趋势方向列
            trend_direction = result['TrendDirection']
            
            # 计算趋势稳定性
            lookback = 5  # 回看窗口
            
            for i in range(lookback, len(result)):
                # 获取过去n天的趋势方向
                past_trend = trend_direction.values[i-lookback:i+1]
                
                # 计算一致性
                # 如果全部一致，返回1.0；如果完全相反，返回0.0
                if np.all(past_trend == 1) or np.all(past_trend == -1):
                    consistency = 1.0
                else:
                    # 计算上升趋势的比例
                    up_ratio = np.sum(past_trend == 1) / lookback
                    # 计算下降趋势的比例
                    down_ratio = np.sum(past_trend == -1) / lookback
                    # 取主导趋势的比例作为一致性
                    consistency = max(up_ratio, down_ratio)
                
                # 使用loc而不是链式索引，避免SettingWithCopyWarning
                result.loc[result.index[i], 'TrendStability'] = consistency
            
            return result
        except Exception as e:
            logger.error(f"计算趋势稳定性时出错: {e}")
            return result