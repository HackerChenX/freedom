"""
MACD指标模块

实现MACD指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.common import macd as calc_macd, cross
from enums.indicator_types import CrossType


class MACD(BaseIndicator):
    """
    MACD(Moving Average Convergence Divergence)指标
    
    MACD是一种趋势跟踪的动量指标，通过计算两条不同周期的指数移动平均线之差，
    以及该差值的移动平均线来判断市场趋势和动量。
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
               adapt_to_volatility: bool = False):
        """
        初始化MACD指标
        
        Args:
            fast_period: 快线周期，默认为12
            slow_period: 慢线周期，默认为26
            signal_period: 信号线周期，默认为9
            adapt_to_volatility: 是否根据波动率自适应调整参数，默认为False
        """
        super().__init__(name="MACD", description="移动平均线收敛散度指标")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.adapt_to_volatility = adapt_to_volatility
        self.market_environment = "normal"  # 默认市场环境
    
    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment: 市场环境，可选值为 "bull_market", "bear_market", "sideways_market", "volatile_market", "normal"
        """
        valid_environments = ["bull_market", "bear_market", "sideways_market", "volatile_market", "normal"]
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境，有效值为: {', '.join(valid_environments)}")
            
        self.market_environment = environment
    
    def get_market_environment(self) -> str:
        """
        获取当前市场环境
        
        Returns:
            str: 当前市场环境
        """
        return self.market_environment
    
    def detect_market_environment(self, data: pd.DataFrame) -> str:
        """
        根据价格数据检测市场环境
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            str: 检测到的市场环境
        """
        if 'close' not in data.columns:
            raise ValueError("输入数据必须包含'close'列")
            
        price = data['close']
        
        # 计算短期和长期趋势
        ma20 = price.rolling(window=20).mean()
        ma60 = price.rolling(window=60).mean()
        
        # 计算波动率
        returns = price.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 年化波动率
        
        # 检查是否有足够的数据
        if len(price) < 60:
            return "normal"
            
        # 获取最新值
        latest_price = price.iloc[-1]
        latest_ma20 = ma20.iloc[-1]
        latest_ma60 = ma60.iloc[-1]
        latest_volatility = volatility.iloc[-1]
        
        # 计算长期波动率均值
        long_term_volatility = volatility.iloc[-60:].mean() if len(volatility) >= 60 else volatility.mean()
        
        # 判断市场环境
        if latest_volatility > long_term_volatility * 1.5:
            # 高波动率市场
            return "volatile_market"
        elif latest_price > latest_ma20 and latest_ma20 > latest_ma60 and latest_price > price.iloc[-20:].min() * 1.1:
            # 牛市条件: 价格高于20日均线，20日均线高于60日均线，且价格比近期最低点高10%以上
            return "bull_market"
        elif latest_price < latest_ma20 and latest_ma20 < latest_ma60 and latest_price < price.iloc[-20:].max() * 0.9:
            # 熊市条件: 价格低于20日均线，20日均线低于60日均线，且价格比近期最高点低10%以上
            return "bear_market"
        elif abs((latest_price / latest_ma60) - 1) < 0.05:
            # 盘整市场: 价格在长期均线附近波动不超过5%
            return "sideways_market"
        else:
            # 默认为正常市场
            return "normal"
    
    def calculate(self, data: pd.DataFrame, price_col: str = 'close', 
                  add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            price_col: 价格列名，默认为'close'
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含MACD指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含价格列
        """
        # 确保数据包含价格列
        self.ensure_columns(data, [price_col])
        
        # 复制输入数据
        result = data.copy()
        
        # 如果开启波动率自适应，调整MACD参数
        fast_period = self.fast_period
        slow_period = self.slow_period
        signal_period = self.signal_period
        
        if self.adapt_to_volatility:
            # 计算价格波动率
            returns = result[price_col].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 年化波动率
            
            # 确保有足够的数据
            if len(volatility) >= 20 and not volatility.iloc[-1] != volatility.iloc[-1]:  # 检查NaN
                # 计算相对波动率（相对于过去120天）
                long_term_vol = volatility.iloc[-120:].mean() if len(volatility) >= 120 else volatility.mean()
                relative_vol = volatility.iloc[-1] / long_term_vol
                
                # 根据相对波动率调整参数
                if relative_vol > 1.5:  # 高波动
                    fast_period = int(fast_period * 1.2)  # 增加快周期
                    slow_period = int(slow_period * 1.2)  # 增加慢周期
                elif relative_vol < 0.7:  # 低波动
                    fast_period = max(6, int(fast_period * 0.8))  # 减少快周期，但不低于6
                    slow_period = max(16, int(slow_period * 0.8))  # 减少慢周期，但不低于16
        
        # 使用统一的公共函数计算MACD
        dif, dea, macd_hist = calc_macd(
            result[price_col].values,
            fast_period,
            slow_period,
            signal_period
        )
        
        # 确保前N个值为NaN，其中N = max(fast_period, slow_period) - 1
        min_periods = max(fast_period, slow_period) - 1
        dif[:min_periods] = np.nan
        dea[:min_periods + signal_period - 1] = np.nan
        macd_hist[:min_periods + signal_period - 1] = np.nan
        
        # 设置列名
        if add_prefix:
            macd_col = self.get_column_name('DIF')
            signal_col = self.get_column_name('DEA')
            hist_col = self.get_column_name('MACD')
        else:
            macd_col = 'DIF'
            signal_col = 'DEA'
            hist_col = 'MACD'
        
        # 添加结果列
        result[macd_col] = dif
        result[signal_col] = dea
        result[hist_col] = macd_hist
        
        # 添加MACD背离信号
        result = self._add_divergence_signals(result, macd_col, price_col=price_col)
        
        # 保存结果
        self._result = result
        
        return result
    
    def add_signals(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                   signal_col: str = 'DEA', hist_col: str = 'MACD') -> pd.DataFrame:
        """
        添加MACD交易信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            hist_col: 柱状图列名(MACD)
            
        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        result = data.copy()
        
        # 计算金叉和死叉信号
        result['macd_buy_signal'] = self.get_buy_signal(result, macd_col, signal_col)
        result['macd_sell_signal'] = self.get_sell_signal(result, macd_col, signal_col)
        
        # 计算零轴穿越信号
        result['macd_zero_cross_up'] = (result[macd_col] > 0) & (result[macd_col].shift(1) <= 0)
        result['macd_zero_cross_down'] = (result[macd_col] < 0) & (result[macd_col].shift(1) >= 0)
        
        # 计算柱状图趋势
        result['macd_hist_increasing'] = result[hist_col] > result[hist_col].shift(1)
        result['macd_hist_decreasing'] = result[hist_col] < result[hist_col].shift(1)
        
        # 计算背离指标
        result = self._add_divergence_signals(result, macd_col, price_col='close')
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                      signal_col: str = 'DEA') -> pd.Series:
        """
        获取MACD买入信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.Series: 买入信号序列（布尔值）
        """
        # 使用公共cross函数检测金叉
        return pd.Series(
            cross(data[macd_col].values, data[signal_col].values),
            index=data.index
        )
    
    def get_sell_signal(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                       signal_col: str = 'DEA') -> pd.Series:
        """
        获取MACD卖出信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.Series: 卖出信号序列（布尔值）
        """
        # 使用公共cross函数检测死叉
        return pd.Series(
            cross(data[signal_col].values, data[macd_col].values),
            index=data.index
        )
    
    def _add_divergence_signals(self, data: pd.DataFrame, macd_col: str = 'DIF', 
                              price_col: str = 'close', window: int = 20) -> pd.DataFrame:
        """
        添加MACD背离信号
        
        Args:
            data: 包含MACD指标的DataFrame
            macd_col: MACD线列名(DIF)
            price_col: 价格列名
            window: 寻找背离的窗口大小
            
        Returns:
            pd.DataFrame: 添加了背离信号的DataFrame
        """
        result = data.copy()
        
        # 初始化背离信号列
        result['macd_bullish_divergence'] = False
        result['macd_bearish_divergence'] = False
        
        # 循环检测背离
        for i in range(window, len(result)):
            # 只检查窗口内的数据
            window_data = result.iloc[i-window:i+1]
            
            # 计算价格的局部最低点
            price_lows = window_data[window_data[price_col] == window_data[price_col].min()]
            
            # 计算价格的局部最高点
            price_highs = window_data[window_data[price_col] == window_data[price_col].max()]
            
            # 计算MACD的局部最低点
            macd_lows = window_data[window_data[macd_col] == window_data[macd_col].min()]
            
            # 计算MACD的局部最高点
            macd_highs = window_data[window_data[macd_col] == window_data[macd_col].max()]
            
            # 检查是否有足够的点来比较
            if len(price_lows) > 1 and len(macd_lows) > 1:
                # 检查看涨背离：价格创新低但MACD没有创新低
                if (price_lows.iloc[-1][price_col] < price_lows.iloc[0][price_col] and 
                    macd_lows.iloc[-1][macd_col] > macd_lows.iloc[0][macd_col]):
                    result.loc[result.index[i], 'macd_bullish_divergence'] = True
            
            # 检查是否有足够的点来比较
            if len(price_highs) > 1 and len(macd_highs) > 1:
                # 检查看跌背离：价格创新高但MACD没有创新高
                if (price_highs.iloc[-1][price_col] > price_highs.iloc[0][price_col] and 
                    macd_highs.iloc[-1][macd_col] < macd_highs.iloc[0][macd_col]):
                    result.loc[result.index[i], 'macd_bearish_divergence'] = True
        
        return result
    
    def ensure_columns(self, data: pd.DataFrame, columns: List[str]) -> None:
        """
        确保DataFrame包含所需的列
        
        Args:
            data: 输入数据
            columns: 所需的列名列表
            
        Raises:
            ValueError: 如果数据不包含所需的列
        """
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"输入数据缺少所需的列: {', '.join(missing_columns)}")
    
    def get_column_name(self, suffix: str = "") -> str:
        """
        获取指标列名
        
        Args:
            suffix: 列名后缀
            
        Returns:
            str: 指标列名
        """
        if suffix:
            return f"{self.name.lower()}_{suffix}"
        return self.name.lower()
    
    def get_cross_points(self, data: pd.DataFrame, cross_type: CrossType = CrossType.GOLDEN_CROSS,
                        macd_col: str = 'DIF', signal_col: str = 'DEA') -> pd.DataFrame:
        """
        获取MACD交叉点
        
        Args:
            data: 包含MACD指标的DataFrame
            cross_type: 交叉类型，金叉或死叉
            macd_col: MACD线列名(DIF)
            signal_col: 信号线列名(DEA)
            
        Returns:
            pd.DataFrame: 交叉点DataFrame
        """
        if cross_type == CrossType.GOLDEN_CROSS:
            # 金叉：MACD从下方穿过信号线
            cross_points = data[self.get_buy_signal(data, macd_col, signal_col)]
        else:
            # 死叉：MACD从上方穿过信号线
            cross_points = data[self.get_sell_signal(data, macd_col, signal_col)]
        
        return cross_points

    def to_dict(self) -> Dict:
        """
        将指标转换为字典表示
        
        Returns:
            Dict: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period
            },
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        }
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成MACD交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 信号DataFrame
        """
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data)
            
        result = self.calculate(data)
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=result.index)
        signals['dif'] = result['DIF']
        signals['dea'] = result['DEA']
        signals['macd'] = result['MACD']
        
        # 计算MACD评分
        macd_score = self.calculate_raw_score(data)
        signals['score'] = macd_score
        
        # 计算传统的金叉/死叉信号
        golden_cross = self.get_buy_signal(result)
        death_cross = self.get_sell_signal(result)
        
        # 计算零轴穿越信号
        zero_cross_up = (result['DIF'] > 0) & (result['DIF'].shift(1) <= 0)
        zero_cross_down = (result['DIF'] < 0) & (result['DIF'].shift(1) >= 0)
        
        # 计算柱状图转向信号
        hist_turn_positive = (result['MACD'] > 0) & (result['MACD'].shift(1) <= 0)
        hist_turn_negative = (result['MACD'] < 0) & (result['MACD'].shift(1) >= 0)
        
        # 背离信号
        bullish_divergence = result['macd_bullish_divergence']
        bearish_divergence = result['macd_bearish_divergence']
        
        # 1. 基于传统信号的初步买入/卖出信号
        buy_signal = (
            golden_cross |  # 金叉
            zero_cross_up |  # DIF上穿零轴
            hist_turn_positive |  # 柱状图由负转正
            bullish_divergence  # 看涨背离
        )
        
        sell_signal = (
            death_cross |  # 死叉
            zero_cross_down |  # DIF下穿零轴
            hist_turn_negative |  # 柱状图由正转负
            bearish_divergence  # 看跌背离
        )
        
        # 2. 基于评分的买入/卖出信号增强
        # 评分高于70，增加买入信号
        buy_signal = buy_signal | (macd_score > 70)
        
        # 评分低于30，增加卖出信号
        sell_signal = sell_signal | (macd_score < 30)
        
        # 3. 信号质量评估与过滤
        # 计算DIF与DEA的距离，用于评估交叉质量
        dif_dea_distance = abs(signals['dif'] - signals['dea'])
        dif_dea_std = dif_dea_distance.rolling(20).std().fillna(dif_dea_distance)
        distance_ratio = dif_dea_distance / dif_dea_std
        
        # 过滤弱交叉信号（距离太小的交叉）
        weak_cross = distance_ratio < 0.5  # 交叉时DIF与DEA距离小于平均的一半
        buy_signal = buy_signal & (~(golden_cross & weak_cross))  # 排除弱金叉
        sell_signal = sell_signal & (~(death_cross & weak_cross))  # 排除弱死叉
        
        # 4. 连续信号抑制（避免短期内重复信号）
        # 计算前10个周期内的信号数量
        prev_buy_signals = buy_signal.rolling(10).sum().fillna(0)
        prev_sell_signals = sell_signal.rolling(10).sum().fillna(0)
        
        # 如果前10个周期已有相同类型的信号，且评分变化不大，则抑制当前信号
        score_change = abs(macd_score - macd_score.shift(10))
        
        buy_signal = buy_signal & ((prev_buy_signals < 1) | (score_change > 15))
        sell_signal = sell_signal & ((prev_sell_signals < 1) | (score_change > 15))
        
        # 5. 市场环境适应性调整（如果有市场环境信息）
        if hasattr(self, 'market_environment'):
            market_env = self.market_environment
            
            # 根据市场环境调整信号
            if market_env == 'bull_market':
                # 牛市环境：降低买入门槛，提高卖出门槛
                buy_signal = buy_signal | (macd_score > 65)
                sell_signal = sell_signal & (macd_score < 25)
            elif market_env == 'bear_market':
                # 熊市环境：提高买入门槛，降低卖出门槛
                buy_signal = buy_signal & (macd_score > 75)
                sell_signal = sell_signal | (macd_score < 35)
            elif market_env == 'volatile_market':
                # 高波动环境：要求更强的信号确认
                buy_signal = buy_signal & (golden_cross | (macd_score > 75))
                sell_signal = sell_signal & (death_cross | (macd_score < 25))
        
        # 保存最终信号
        signals['buy_signal'] = buy_signal
        signals['sell_signal'] = sell_signal
        
        # 计算信号强度（1-5，数字越大表示信号越强）
        signals['buy_strength'] = 1
        signals['sell_strength'] = 1
        
        # 根据评分调整信号强度
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                score_val = signals['score'].iloc[i]
                if score_val > 90:
                    signals.loc[signals.index[i], 'buy_strength'] = 5
                elif score_val > 80:
                    signals.loc[signals.index[i], 'buy_strength'] = 4
                elif score_val > 70:
                    signals.loc[signals.index[i], 'buy_strength'] = 3
                elif score_val > 60:
                    signals.loc[signals.index[i], 'buy_strength'] = 2
                    
            if signals['sell_signal'].iloc[i]:
                score_val = signals['score'].iloc[i]
                if score_val < 10:
                    signals.loc[signals.index[i], 'sell_strength'] = 5
                elif score_val < 20:
                    signals.loc[signals.index[i], 'sell_strength'] = 4
                elif score_val < 30:
                    signals.loc[signals.index[i], 'sell_strength'] = 3
                elif score_val < 40:
                    signals.loc[signals.index[i], 'sell_strength'] = 2
        
        # 添加信号描述
        signals['signal_desc'] = ""
        for i in range(len(signals)):
            desc = []
            if golden_cross.iloc[i]:
                desc.append("金叉")
            if death_cross.iloc[i]:
                desc.append("死叉")
            if zero_cross_up.iloc[i]:
                desc.append("DIF上穿零轴")
            if zero_cross_down.iloc[i]:
                desc.append("DIF下穿零轴")
            if hist_turn_positive.iloc[i]:
                desc.append("柱状图转正")
            if hist_turn_negative.iloc[i]:
                desc.append("柱状图转负")
            if bullish_divergence.iloc[i]:
                desc.append("看涨背离")
            if bearish_divergence.iloc[i]:
                desc.append("看跌背离")
                
            signals.loc[signals.index[i], 'signal_desc'] = ", ".join(desc) if desc else ""
        
        return signals

    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD的原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分序列（0-100）
        """
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data)
        
        # 获取MACD指标数据
        result = self.calculate(data)
        dif = result['DIF']
        dea = result['DEA']
        macd_hist = result['MACD']
        
        # 初始化评分（50为中性）
        score = pd.Series(50.0, index=result.index)
        
        # 1. 基础金叉/死叉评分
        golden_cross = self.get_buy_signal(result)
        death_cross = self.get_sell_signal(result)
        
        # 计算DIF与零轴的距离系数（距离越远权重越低）
        zero_distance = abs(dif) / dif.rolling(120).std().fillna(0.01)
        zero_distance_coef = np.clip(1 - zero_distance * 0.2, 0.5, 1.0)
        
        # 金叉/死叉评分（考虑零轴距离）
        cross_score = golden_cross * 20 - death_cross * 20
        cross_score = cross_score * zero_distance_coef
        score += cross_score
        
        # 2. DIF和DEA位置评分
        score += ((dif > 0) & (dea > 0)) * 10  # 同时位于零轴上方，看涨
        score -= ((dif < 0) & (dea < 0)) * 10  # 同时位于零轴下方，看跌
        
        # 3. 零轴穿越评分
        zero_cross_up = (dif > 0) & (dif.shift(1) <= 0)
        zero_cross_down = (dif < 0) & (dif.shift(1) >= 0)
        score += zero_cross_up * 15
        score -= zero_cross_down * 15
        
        # 4. 柱状图变化评分
        hist_up = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
        hist_down = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
        score += hist_up * 12
        score -= hist_down * 12
        
        # 5. 柱状图能量因子评分（新增）
        # 计算最近10个周期与前10个周期的柱状图能量比率
        hist_window = 10
        for i in range(hist_window, len(macd_hist)):
            recent_hist = abs(macd_hist.iloc[i-hist_window:i])
            prev_hist = abs(macd_hist.iloc[i-hist_window*2:i-hist_window])
            
            if len(recent_hist) == hist_window and len(prev_hist) == hist_window:
                # 计算能量比率（当前能量/前期能量）
                recent_energy = recent_hist.sum()
                prev_energy = prev_hist.sum()
                
                if prev_energy > 0:
                    energy_ratio = recent_energy / prev_energy
                    
                    # 能量增加时增强信号得分
                    if energy_ratio > 1.2:  # 柱状图能量明显增加
                        energy_score = min(15, (energy_ratio - 1) * 30)
                        
                        # 根据柱状图方向确定加减分
                        if macd_hist.iloc[i] > 0:
                            score.iloc[i] += energy_score  # 正柱状图能量增加，加分
                        else:
                            score.iloc[i] -= energy_score  # 负柱状图能量增加，减分
        
        # 6. MACD背离评分（增强）
        bearish_div = result['macd_bearish_divergence']
        bullish_div = result['macd_bullish_divergence']
        
        score += bullish_div * 25  # 看涨背离强度更高
        score -= bearish_div * 25  # 看跌背离强度更高
        
        # 7. DIF与DEA的离散度评分（新增）
        # 计算DIF与DEA的绝对距离相对于近期标准差的比例
        dif_dea_distance = abs(dif - dea)
        dif_dea_std = dif_dea_distance.rolling(20).std().fillna(dif_dea_distance)
        distance_ratio = dif_dea_distance / dif_dea_std
        
        # 距离显著增大（离散度增加），表示趋势加速
        accel_score = np.clip(distance_ratio - 1, 0, 2) * 5
        
        # 根据DIF和DEA的相对位置确定加减分
        score += ((dif > dea) & (distance_ratio > 1)) * accel_score  # DIF > DEA且离散度增加，加分
        score -= ((dif < dea) & (distance_ratio > 1)) * accel_score  # DIF < DEA且离散度增加，减分
        
        # 8. 柱状图连续增长/下降评分（新增）
        hist_consecutive_up = 0
        hist_consecutive_down = 0
        
        for i in range(1, len(macd_hist)):
            if macd_hist.iloc[i] > macd_hist.iloc[i-1]:
                hist_consecutive_up += 1
                hist_consecutive_down = 0
            elif macd_hist.iloc[i] < macd_hist.iloc[i-1]:
                hist_consecutive_down += 1
                hist_consecutive_up = 0
            else:
                hist_consecutive_up = 0
                hist_consecutive_down = 0
                
            # 柱状图连续5周期增长，强势加分
            if hist_consecutive_up >= 5:
                score.iloc[i] += min(10, hist_consecutive_up)
                
            # 柱状图连续5周期下降，强势减分
            if hist_consecutive_down >= 5:
                score.iloc[i] -= min(10, hist_consecutive_down)
        
        # 限制评分范围在0-100之间
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MACD指标形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data)
            
        result = self.calculate(data)
        patterns = []
        
        dif = result['DIF']
        dea = result['DEA']
        macd_hist = result['MACD']
        
        # 1. 基础交叉形态识别
        if self.get_buy_signal(result).iloc[-1]:
            # 判断金叉质量
            dif_dea_dist = abs(dif.iloc[-1] - dea.iloc[-1])
            avg_dist = abs(dif - dea).rolling(20).mean().iloc[-1]
            
            if dif_dea_dist > avg_dist * 1.5:
                patterns.append("高质量金叉")
            else:
                patterns.append("金叉")
                
        if self.get_sell_signal(result).iloc[-1]:
            # 判断死叉质量
            dif_dea_dist = abs(dif.iloc[-1] - dea.iloc[-1])
            avg_dist = abs(dif - dea).rolling(20).mean().iloc[-1]
            
            if dif_dea_dist > avg_dist * 1.5:
                patterns.append("高质量死叉")
            else:
                patterns.append("死叉")
        
        # 2. 零轴相关形态
        if dif.iloc[-1] > 0 and dif.iloc[-2] <= 0:
            patterns.append("DIF上穿零轴")
        elif dif.iloc[-1] < 0 and dif.iloc[-2] >= 0:
            patterns.append("DIF下穿零轴")
            
        if dif.iloc[-1] > 0 and dea.iloc[-1] > 0:
            patterns.append("DIF和DEA均在零轴上方")
        elif dif.iloc[-1] < 0 and dea.iloc[-1] < 0:
            patterns.append("DIF和DEA均在零轴下方")
        
        # 3. 柱状图形态识别
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
            patterns.append("柱状图由负转正")
        elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
            patterns.append("柱状图由正转负")
            
        # 4. 柱状图能量形态（新增）
        if len(macd_hist) >= 10:
            # 计算近期柱状图能量
            recent_energy = abs(macd_hist.iloc[-5:]).sum()
            prev_energy = abs(macd_hist.iloc[-10:-5]).sum()
            
            if prev_energy > 0:
                energy_ratio = recent_energy / prev_energy
                
                if energy_ratio > 1.5 and macd_hist.iloc[-1] > 0:
                    patterns.append("柱状图能量强劲增加(多)")
                elif energy_ratio > 1.5 and macd_hist.iloc[-1] < 0:
                    patterns.append("柱状图能量强劲增加(空)")
                elif energy_ratio < 0.5:
                    patterns.append("柱状图能量明显衰减")
        
        # 5. 连续柱状图形态（新增）
        if len(macd_hist) >= 6:
            consecutive_up = True
            consecutive_down = True
            
            for i in range(1, 6):
                if macd_hist.iloc[-i] <= macd_hist.iloc[-(i+1)]:
                    consecutive_up = False
                if macd_hist.iloc[-i] >= macd_hist.iloc[-(i+1)]:
                    consecutive_down = False
            
            if consecutive_up:
                patterns.append("柱状图连续5周期增长")
            if consecutive_down:
                patterns.append("柱状图连续5周期减少")
        
        # 6. 背离形态（增强）
        # 检查结果是否包含背离信号
        if 'macd_bullish_divergence' in result.columns and result['macd_bullish_divergence'].iloc[-1]:
            patterns.append("MACD正背离")
            
        if 'macd_bearish_divergence' in result.columns and result['macd_bearish_divergence'].iloc[-1]:
            patterns.append("MACD负背离")
            
        # 7. 隐藏背离形态（新增）
        hidden_div = self._detect_hidden_divergence(data['close'], dif)
        if hidden_div == "bullish":
            patterns.append("MACD隐藏正背离")
        elif hidden_div == "bearish":
            patterns.append("MACD隐藏负背离")
        
        # 8. MACD钩子形态（新增）
        if len(dif) >= 5 and len(dea) >= 5:
            # 判断MACD顶部钩子形态（看跌）
            if (dif.iloc[-3] > dif.iloc[-4] and 
                dif.iloc[-2] > dif.iloc[-3] and 
                dif.iloc[-1] < dif.iloc[-2] and 
                dif.iloc[-1] > dea.iloc[-1] and
                dif.iloc[-2] - dea.iloc[-2] > dif.iloc[-1] - dea.iloc[-1]):
                patterns.append("MACD顶部钩子")
                
            # 判断MACD底部钩子形态（看涨）
            if (dif.iloc[-3] < dif.iloc[-4] and 
                dif.iloc[-2] < dif.iloc[-3] and 
                dif.iloc[-1] > dif.iloc[-2] and 
                dif.iloc[-1] < dea.iloc[-1] and
                dea.iloc[-2] - dif.iloc[-2] > dea.iloc[-1] - dif.iloc[-1]):
                patterns.append("MACD底部钩子")
        
        # 9. 零轴徘徊形态（新增）
        if len(dif) >= 10:
            # 计算DIF与零轴的距离
            zero_distance = abs(dif.iloc[-10:])
            avg_distance = zero_distance.mean()
            
            # 如果平均距离小于DIF标准差的一半，判定为零轴徘徊
            if avg_distance < dif.iloc[-60:].std() * 0.5:
                patterns.append("DIF零轴徘徊")
        
        return patterns
        
    def _detect_hidden_divergence(self, price: pd.Series, indicator: pd.Series) -> Optional[str]:
        """
        检测隐藏背离
        
        Args:
            price: 价格序列
            indicator: 指标序列
            
        Returns:
            Optional[str]: 背离类型 ("bullish", "bearish" 或 None)
        """
        if len(price) < 20 or len(indicator) < 20:
            return None
            
        # 获取最近20个周期的数据
        recent_price = price.iloc[-20:]
        recent_indicator = indicator.iloc[-20:]
        
        # 寻找价格高点和低点
        price_highs = []
        price_lows = []
        
        for i in range(1, len(recent_price) - 1):
            # 价格高点
            if recent_price.iloc[i] > recent_price.iloc[i-1] and recent_price.iloc[i] > recent_price.iloc[i+1]:
                price_highs.append((i, recent_price.iloc[i]))
            # 价格低点
            if recent_price.iloc[i] < recent_price.iloc[i-1] and recent_price.iloc[i] < recent_price.iloc[i+1]:
                price_lows.append((i, recent_price.iloc[i]))
        
        # 寻找指标高点和低点
        indicator_highs = []
        indicator_lows = []
        
        for i in range(1, len(recent_indicator) - 1):
            # 指标高点
            if recent_indicator.iloc[i] > recent_indicator.iloc[i-1] and recent_indicator.iloc[i] > recent_indicator.iloc[i+1]:
                indicator_highs.append((i, recent_indicator.iloc[i]))
            # 指标低点
            if recent_indicator.iloc[i] < recent_indicator.iloc[i-1] and recent_indicator.iloc[i] < recent_indicator.iloc[i+1]:
                indicator_lows.append((i, recent_indicator.iloc[i]))
        
        # 检查是否有足够的点来比较
        if len(price_highs) >= 2 and len(indicator_highs) >= 2:
            # 取最近的两个高点
            recent_price_highs = sorted(price_highs, key=lambda x: x[0])[-2:]
            recent_indicator_highs = sorted(indicator_highs, key=lambda x: x[0])[-2:]
            
            # 检查隐藏负背离: 价格高点下降，但指标高点上升
            if (recent_price_highs[1][1] < recent_price_highs[0][1] and 
                recent_indicator_highs[1][1] > recent_indicator_highs[0][1]):
                return "bearish"
        
        # 检查是否有足够的点来比较
        if len(price_lows) >= 2 and len(indicator_lows) >= 2:
            # 取最近的两个低点
            recent_price_lows = sorted(price_lows, key=lambda x: x[0])[-2:]
            recent_indicator_lows = sorted(indicator_lows, key=lambda x: x[0])[-2:]
            
            # 检查隐藏正背离: 价格低点上升，但指标低点下降
            if (recent_price_lows[1][1] > recent_price_lows[0][1] and 
                recent_indicator_lows[1][1] < recent_indicator_lows[0][1]):
                return "bullish"
        
        return None 

    def _calculate_histogram_trend_strength(self, hist: pd.Series, window: int = 10) -> pd.Series:
        """
        计算MACD柱状图趋势强度
        
        Args:
            hist: MACD柱状图序列
            window: 计算窗口大小
            
        Returns:
            pd.Series: 趋势强度序列
        """
        # 初始化趋势强度序列
        strength = pd.Series(0.0, index=hist.index)
        
        # 计算柱状图符号（正负）
        hist_sign = np.sign(hist)
        
        # 计算连续相同符号的柱状图数量
        consecutive_count = 0
        for i in range(1, len(hist)):
            if hist_sign.iloc[i] == hist_sign.iloc[i-1]:
                consecutive_count += 1
            else:
                consecutive_count = 0
                
            # 计算窗口内的柱状图能量
            if i >= window:
                window_hist = hist.iloc[i-window+1:i+1]
                pos_energy = window_hist[window_hist > 0].sum()
                neg_energy = abs(window_hist[window_hist < 0].sum())
                
                # 计算能量差值与总能量的比值作为强度
                total_energy = pos_energy + neg_energy
                if total_energy > 0:
                    energy_ratio = (pos_energy - neg_energy) / total_energy
                    
                    # 考虑连续柱状图增强趋势强度
                    consecutive_factor = min(1.0, consecutive_count / window)
                    strength.iloc[i] = energy_ratio * (1 + consecutive_factor)
        
        return strength
    
    def _calculate_cross_angle(self, line1: pd.Series, line2: pd.Series, window: int = 5) -> pd.Series:
        """
        计算两条线交叉的角度
        
        Args:
            line1: 第一条线
            line2: 第二条线
            window: 计算窗口大小
            
        Returns:
            pd.Series: 交叉角度序列
        """
        # 计算两条线的斜率
        line1_slope = (line1 - line1.shift(window)) / window
        line2_slope = (line2 - line2.shift(window)) / window
        
        # 计算斜率差值作为角度近似值
        angle = abs(line1_slope - line2_slope)
        
        # 归一化角度
        max_angle = angle.rolling(60).max()
        angle = angle / max_angle.replace(0, 1)  # 避免除以零
        
        return angle
    
    def _calculate_zero_axis_interaction(self, dif: pd.Series, window: int = 10) -> pd.Series:
        """
        计算DIF与零轴的交互强度
        
        Args:
            dif: DIF序列
            window: 计算窗口大小
            
        Returns:
            pd.Series: 零轴交互强度序列
        """
        # 初始化交互强度序列
        interaction = pd.Series(0.0, index=dif.index)
        
        # 计算DIF穿越零轴
        zero_cross_up = (dif > 0) & (dif.shift(1) <= 0)
        zero_cross_down = (dif < 0) & (dif.shift(1) >= 0)
        
        # 计算DIF与零轴的距离
        zero_distance = abs(dif)
        
        # 归一化距离
        avg_distance = zero_distance.rolling(window=60).mean()
        norm_distance = zero_distance / avg_distance.replace(0, 1)  # 避免除以零
        
        # 计算交互强度
        for i in range(window, len(dif)):
            # 如果最近有穿越零轴
            if any(zero_cross_up.iloc[i-window:i+1]) or any(zero_cross_down.iloc[i-window:i+1]):
                # 穿越后的运行距离
                max_distance = zero_distance.iloc[i-window:i+1].max()
                
                # 当前距离相对于最大距离的比例
                if max_distance > 0:
                    relative_distance = zero_distance.iloc[i] / max_distance
                    interaction.iloc[i] = 1 - relative_distance  # 距离越小，交互越强
            else:
                # 如果没有穿越，则根据距离计算交互强度
                # 距离越小，交互越强
                interaction.iloc[i] = 1 / (1 + norm_distance.iloc[i])
        
        return interaction
    
    def has_result(self) -> bool:
        """
        检查是否已计算过指标
        
        Returns:
            bool: 是否已计算过指标
        """
        return hasattr(self, '_result') and self._result is not None 