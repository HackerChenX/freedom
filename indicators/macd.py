"""
MACD指标模块

实现MACD指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
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
        self._market_environment = MarketEnvironment.SIDEWAYS_MARKET
        
        # 注册MACD指标形态
        self._register_macd_patterns()
    
    def _register_macd_patterns(self):
        """注册MACD指标的所有形态"""
        # 注册金叉形态
        self.register_pattern(
            pattern_id="golden_cross",
            display_name="MACD金叉",
            detection_func=self._detect_golden_cross,
            score_impact=20.0
        )
        
        # 注册死叉形态
        self.register_pattern(
            pattern_id="death_cross",
            display_name="MACD死叉",
            detection_func=self._detect_death_cross,
            score_impact=-20.0
        )
        
        # 注册零轴上穿形态
        self.register_pattern(
            pattern_id="zero_axis_crossover",
            display_name="MACD零轴上穿",
            detection_func=self._detect_zero_axis_crossover,
            score_impact=15.0
        )
        
        # 注册零轴下穿形态
        self.register_pattern(
            pattern_id="zero_axis_crossunder",
            display_name="MACD零轴下穿",
            detection_func=self._detect_zero_axis_crossunder,
            score_impact=-15.0
        )
        
        # 注册底背离形态
        self.register_pattern(
            pattern_id="bullish_divergence",
            display_name="MACD底背离",
            detection_func=self._detect_bullish_divergence,
            score_impact=25.0
        )
        
        # 注册顶背离形态
        self.register_pattern(
            pattern_id="bearish_divergence",
            display_name="MACD顶背离",
            detection_func=self._detect_bearish_divergence,
            score_impact=-25.0
        )
        
        # 注册双底形态
        self.register_pattern(
            pattern_id="double_bottom",
            display_name="MACD双底",
            detection_func=self._detect_double_bottom,
            score_impact=18.0
        )
        
        # 注册连续三阳上穿形态
        self.register_pattern(
            pattern_id="three_line_strike",
            display_name="MACD三线上穿",
            detection_func=self._detect_three_line_strike,
            score_impact=12.0
        )
        
        # 注册连续三阴下穿形态
        self.register_pattern(
            pattern_id="three_line_fall",
            display_name="MACD三线下穿",
            detection_func=self._detect_three_line_fall,
            score_impact=-12.0
        )
    
    def _detect_golden_cross(self, data: pd.DataFrame) -> bool:
        """检测MACD金叉形态"""
        if not self.has_result() or 'DIF' not in data.columns or 'DEA' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最近两个周期的MACD值
        dif = data['DIF'].iloc[-2:].values
        dea = data['DEA'].iloc[-2:].values
        
        # 金叉条件：当前DIF在DEA上方，且前一周期DIF在DEA下方
        golden_cross = dif[-1] > dea[-1] and dif[-2] <= dea[-2]
        
        return golden_cross
    
    def _detect_death_cross(self, data: pd.DataFrame) -> bool:
        """检测MACD死叉形态"""
        if not self.has_result() or 'DIF' not in data.columns or 'DEA' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最近两个周期的MACD值
        dif = data['DIF'].iloc[-2:].values
        dea = data['DEA'].iloc[-2:].values
        
        # 死叉条件：当前DIF在DEA下方，且前一周期DIF在DEA上方
        death_cross = dif[-1] < dea[-1] and dif[-2] >= dea[-2]
        
        return death_cross
    
    def _detect_zero_axis_crossover(self, data: pd.DataFrame) -> bool:
        """检测MACD零轴上穿形态"""
        if not self.has_result() or 'DIF' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最近两个周期的DIF值
        dif = data['DIF'].iloc[-2:].values
        
        # 零轴上穿条件：当前DIF大于0，且前一周期DIF小于等于0
        zero_axis_crossover = dif[-1] > 0 and dif[-2] <= 0
        
        return zero_axis_crossover
    
    def _detect_zero_axis_crossunder(self, data: pd.DataFrame) -> bool:
        """检测MACD零轴下穿形态"""
        if not self.has_result() or 'DIF' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 2:
            return False
        
        # 获取最近两个周期的DIF值
        dif = data['DIF'].iloc[-2:].values
        
        # 零轴下穿条件：当前DIF小于0，且前一周期DIF大于等于0
        zero_axis_crossunder = dif[-1] < 0 and dif[-2] >= 0
        
        return zero_axis_crossunder
    
    def _detect_bullish_divergence(self, data: pd.DataFrame) -> bool:
        """检测MACD底背离形态"""
        if not self.has_result() or 'DIF' not in data.columns or 'MACD' not in data.columns or 'close' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 20:
            return False
        
        # 底背离：价格创新低，但MACD指标未创新低
        try:
            # 获取最近20个周期的数据
            close = data['close'].iloc[-20:].values
            macd = data['MACD'].iloc[-20:].values
            
            # 查找局部最小值的位置
            close_lows = []
            macd_lows = []
            
            for i in range(1, len(close) - 1):
                if close[i] < close[i-1] and close[i] < close[i+1]:
                    close_lows.append((i, close[i]))
                
                if macd[i] < macd[i-1] and macd[i] < macd[i+1]:
                    macd_lows.append((i, macd[i]))
            
            # 至少需要2个低点才能形成背离
            if len(close_lows) < 2 or len(macd_lows) < 2:
                return False
            
            # 排序找出最低的两个点
            close_lows.sort(key=lambda x: x[1])
            
            # 获取价格的两个最低点位置
            idx1, val1 = close_lows[0]
            idx2, val2 = close_lows[1]
            
            # 确保两个低点之间的距离足够
            if abs(idx1 - idx2) < 3:
                return False
            
            # 获取这两个时间点对应的MACD值
            macd_val1 = macd[idx1]
            macd_val2 = macd[idx2]
            
            # 如果价格第二个低点低于第一个低点，但MACD第二个低点高于第一个低点，则形成底背离
            return val2 < val1 and macd_val2 > macd_val1
        except Exception as e:
            logger.error(f"检测MACD底背离形态出错: {e}")
            return False
    
    def _detect_bearish_divergence(self, data: pd.DataFrame) -> bool:
        """检测MACD顶背离形态"""
        if not self.has_result() or 'DIF' not in data.columns or 'MACD' not in data.columns or 'close' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 20:
            return False
        
        # 顶背离：价格创新高，但MACD指标未创新高
        try:
            # 获取最近20个周期的数据
            close = data['close'].iloc[-20:].values
            macd = data['MACD'].iloc[-20:].values
            
            # 查找局部最大值的位置
            close_highs = []
            macd_highs = []
            
            for i in range(1, len(close) - 1):
                if close[i] > close[i-1] and close[i] > close[i+1]:
                    close_highs.append((i, close[i]))
                
                if macd[i] > macd[i-1] and macd[i] > macd[i+1]:
                    macd_highs.append((i, macd[i]))
            
            # 至少需要2个高点才能形成背离
            if len(close_highs) < 2 or len(macd_highs) < 2:
                return False
            
            # 排序找出最高的两个点
            close_highs.sort(key=lambda x: x[1], reverse=True)
            
            # 获取价格的两个最高点位置
            idx1, val1 = close_highs[0]
            idx2, val2 = close_highs[1]
            
            # 确保两个高点之间的距离足够
            if abs(idx1 - idx2) < 3:
                return False
            
            # 获取这两个时间点对应的MACD值
            macd_val1 = macd[idx1]
            macd_val2 = macd[idx2]
            
            # 如果价格第二个高点高于第一个高点，但MACD第二个高点低于第一个高点，则形成顶背离
            return val2 > val1 and macd_val2 < macd_val1
        except Exception as e:
            logger.error(f"检测MACD顶背离形态出错: {e}")
            return False
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> bool:
        """检测MACD双底形态"""
        if not self.has_result() or 'MACD' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 30:
            return False
        
        try:
            # 获取最近30个周期的MACD数据
            macd = data['MACD'].iloc[-30:].values
            
            # 查找局部最小值
            troughs = []
            for i in range(2, len(macd) - 2):
                if (macd[i] < macd[i-1] and 
                    macd[i] < macd[i-2] and 
                    macd[i] < macd[i+1] and 
                    macd[i] < macd[i+2]):
                    troughs.append((i, macd[i]))
            
            # 至少需要2个低点才能形成双底
            if len(troughs) < 2:
                return False
            
            # 按时间排序
            troughs.sort(key=lambda x: x[0])
            
            # 检查最近的两个低点
            if len(troughs) >= 2:
                idx1, val1 = troughs[-2]
                idx2, val2 = troughs[-1]
                
                # 确保两个低点之间的距离足够
                if abs(idx2 - idx1) < 5 or abs(idx2 - idx1) > 20:
                    return False
                
                # 确保两个低点的深度接近
                depth_diff = abs(val1 - val2)
                if depth_diff > abs(val1) * 0.3:  # 深度差异不超过30%
                    return False
                
                # 确保中间有一个小的反弹
                middle_idx = (idx1 + idx2) // 2
                middle_val = macd[middle_idx]
                
                if middle_val <= min(val1, val2):
                    return False
                
                # 确保第二个低点之后有上升趋势
                if idx2 < len(macd) - 2 and macd[idx2 + 1] > macd[idx2] and macd[idx2 + 2] > macd[idx2 + 1]:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"检测MACD双底形态出错: {e}")
            return False
    
    def _detect_three_line_strike(self, data: pd.DataFrame) -> bool:
        """检测MACD三连阳形态"""
        if not self.has_result() or 'MACD' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 4:
            return False
        
        try:
            # 获取最近4个周期的MACD柱状值
            macd = data['MACD'].iloc[-4:].values
            
            # 三连阳条件：连续三个周期MACD柱状值为正且依次增大
            three_positive = (
                macd[0] > 0 and 
                macd[1] > 0 and 
                macd[2] > 0 and 
                macd[1] > macd[0] and 
                macd[2] > macd[1]
            )
            
            return three_positive
        except Exception as e:
            logger.error(f"检测MACD三连阳形态出错: {e}")
            return False
    
    def _detect_three_line_fall(self, data: pd.DataFrame) -> bool:
        """检测MACD三连阴形态"""
        if not self.has_result() or 'MACD' not in data.columns:
            return False
        
        # 确保数据量足够
        if len(data) < 4:
            return False
        
        try:
            # 获取最近4个周期的MACD柱状值
            macd = data['MACD'].iloc[-4:].values
            
            # 三连阴条件：连续三个周期MACD柱状值为负且依次减小
            three_negative = (
                macd[0] < 0 and 
                macd[1] < 0 and 
                macd[2] < 0 and 
                macd[1] < macd[0] and 
                macd[2] < macd[1]
            )
            
            return three_negative
        except Exception as e:
            logger.error(f"检测MACD三连阴形态出错: {e}")
            return False
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算MACD指标的原始评分（0-100分制）
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列，取值范围0-100
        """
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 获取MACD指标值
        if 'DIF' not in data.columns or 'DEA' not in data.columns or 'MACD' not in data.columns:
            raise ValueError("数据中缺少MACD指标列")
        
        dif = data['DIF']
        dea = data['DEA']
        macd_hist = data['MACD']
        
        # 基础评分计算
        # 1. DIF位置分：基于DIF相对零轴的位置，贡献30分权重
        # DIF > 0 时分数高于50，DIF < 0 时分数低于50
        dif_score = 50 + dif * 100  # 调整系数，使得MACD常见值域能映射到0-100
        # 限制在30-70范围内
        dif_score = np.clip(dif_score, 30, 70)
        
        # 2. DIF-DEA关系分：基于DIF和DEA的关系，贡献25分权重
        diff_score = 50 + (dif - dea) * 200  # DIF高于DEA得分高，低于则得分低
        # 限制在20-80范围内
        diff_score = np.clip(diff_score, 20, 80)
        
        # 3. 柱状图分：基于MACD柱状图，贡献25分权重
        hist_score = 50 + macd_hist * 300  # 正柱得分高，负柱得分低
        # 限制在20-80范围内
        hist_score = np.clip(hist_score, 20, 80)
        
        # 4. 趋势分：基于DIF和DEA的趋势，贡献20分权重
        dif_trend = dif - dif.shift(3)
        dea_trend = dea - dea.shift(3)
        trend_score = 50 + (dif_trend + dea_trend) * 200  # 上升趋势得分高，下降趋势得分低
        # 限制在20-80范围内
        trend_score = np.clip(trend_score, 20, 80)
        
        # 合并各部分得分，按权重加权平均
        raw_score = (
            dif_score * 0.3 +     # DIF位置分权重30%
            diff_score * 0.25 +   # DIF-DEA关系分权重25%
            hist_score * 0.25 +   # 柱状图分权重25%
            trend_score * 0.2     # 趋势分权重20%
        )
        
        # 检测形态对评分的影响
        patterns = self.get_patterns(data, **kwargs)
        
        # 形态影响分数：最多调整±20分
        pattern_adjustment = pd.Series(0, index=data.index)
        for pattern in patterns:
            # 找到对应的注册形态信息
            if pattern.pattern_id in self._registered_patterns:
                score_impact = self._registered_patterns[pattern.pattern_id]['score_impact']
                # 应用形态影响
                pattern_adjustment += score_impact
        
        # 限制形态调整范围
        pattern_adjustment = np.clip(pattern_adjustment, -20, 20)
        raw_score += pattern_adjustment
        
        # 确保最终分数在0-100范围内
        final_score = np.clip(raw_score, 0, 100)
        
        return pd.Series(final_score, index=data.index)
    
    def set_market_environment(self, environment: Union[str, MarketEnvironment]) -> None:
        """
        设置市场环境
        
        Args:
            environment: 市场环境，可以是MarketEnvironment枚举或字符串
        """
        if isinstance(environment, MarketEnvironment):
            self._market_environment = environment
            return
            
        # 兼容旧版接口
        env_mapping = {
            "bull_market": MarketEnvironment.BULL_MARKET,
            "bear_market": MarketEnvironment.BEAR_MARKET,
            "sideways_market": MarketEnvironment.SIDEWAYS_MARKET,
            "volatile_market": MarketEnvironment.VOLATILE_MARKET,
            "normal": MarketEnvironment.SIDEWAYS_MARKET,
        }
        
        if environment not in env_mapping:
            valid_values = list(env_mapping.keys())
            raise ValueError(f"无效的市场环境，有效值为: {', '.join(valid_values)}")
            
        self._market_environment = env_mapping[environment]
    
    def get_market_environment(self) -> MarketEnvironment:
        """
        获取当前市场环境
        
        Returns:
            MarketEnvironment: 当前市场环境
        """
        return self._market_environment
    
    def detect_market_environment(self, data: pd.DataFrame) -> MarketEnvironment:
        """
        根据价格数据检测市场环境
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            MarketEnvironment: 检测到的市场环境
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
            return MarketEnvironment.SIDEWAYS_MARKET
            
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
            return MarketEnvironment.VOLATILE_MARKET
        elif latest_price > latest_ma20 and latest_ma20 > latest_ma60 and latest_price > price.iloc[-20:].min() * 1.1:
            # 牛市条件: 价格高于20日均线，20日均线高于60日均线，且价格比近期最低点高10%以上
            return MarketEnvironment.BULL_MARKET
        elif latest_price < latest_ma20 and latest_ma20 < latest_ma60 and latest_price < price.iloc[-20:].max() * 0.9:
            # 熊市条件: 价格低于20日均线，20日均线低于60日均线，且价格比近期最高点低10%以上
            return MarketEnvironment.BEAR_MARKET
        elif abs((latest_price / latest_ma60) - 1) < 0.05:
            # 盘整市场: 价格在长期均线附近波动不超过5%
            return MarketEnvironment.SIDEWAYS_MARKET
        else:
            # 默认为盘整市场
            return MarketEnvironment.SIDEWAYS_MARKET
    
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
        try:
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
                if len(volatility) >= 20 and not np.isnan(volatility.iloc[-1]):  # 检查NaN
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
            
        except Exception as e:
            self._error = e
            raise e
    
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
        
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成MACD交易信号
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, pd.Series]: 包含各类信号的字典
        """
        # 确保已计算MACD
        if not self.has_result():
            self.calculate(data)
            
        result = self._result
        
        # 初始化信号字典
        signals = {}
        
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
        bullish_divergence = result['macd_bullish_divergence'] if 'macd_bullish_divergence' in result.columns else pd.Series(False, index=result.index)
        bearish_divergence = result['macd_bearish_divergence'] if 'macd_bearish_divergence' in result.columns else pd.Series(False, index=result.index)
        
        # 计算原始评分
        score = self.calculate_raw_score(data, **kwargs)
        
        # 构建买入信号
        buy_signal = (
            golden_cross |  # 金叉
            zero_cross_up |  # DIF上穿零轴
            hist_turn_positive |  # 柱状图由负转正
            bullish_divergence |  # 看涨背离
            (score > 70)  # 评分高于70
        )
        
        # 构建卖出信号
        sell_signal = (
            death_cross |  # 死叉
            zero_cross_down |  # DIF下穿零轴
            hist_turn_negative |  # 柱状图由正转负
            bearish_divergence |  # 看跌背离
            (score < 30)  # 评分低于30
        )
        
        # 计算信号强度
        buy_strength = pd.Series(0, index=result.index)
        sell_strength = pd.Series(0, index=result.index)
        
        # 根据评分设置信号强度
        for i in range(len(score)):
            if buy_signal.iloc[i]:
                score_val = score.iloc[i]
                if score_val > 90:
                    buy_strength.iloc[i] = 5  # 非常强
                elif score_val > 80:
                    buy_strength.iloc[i] = 4  # 强
                elif score_val > 70:
                    buy_strength.iloc[i] = 3  # 中等
                elif score_val > 60:
                    buy_strength.iloc[i] = 2  # 弱
                else:
                    buy_strength.iloc[i] = 1  # 非常弱
                    
            if sell_signal.iloc[i]:
                score_val = score.iloc[i]
                if score_val < 10:
                    sell_strength.iloc[i] = 5  # 非常强
                elif score_val < 20:
                    sell_strength.iloc[i] = 4  # 强
                elif score_val < 30:
                    sell_strength.iloc[i] = 3  # 中等
                elif score_val < 40:
                    sell_strength.iloc[i] = 2  # 弱
                else:
                    sell_strength.iloc[i] = 1  # 非常弱
        
        # 添加所有信号到字典
        signals['buy_signal'] = buy_signal
        signals['sell_signal'] = sell_signal
        signals['buy_strength'] = buy_strength
        signals['sell_strength'] = sell_strength
        signals['golden_cross'] = golden_cross
        signals['death_cross'] = death_cross
        signals['zero_cross_up'] = zero_cross_up
        signals['zero_cross_down'] = zero_cross_down
        signals['hist_turn_positive'] = hist_turn_positive
        signals['hist_turn_negative'] = hist_turn_negative
        signals['bullish_divergence'] = bullish_divergence
        signals['bearish_divergence'] = bearish_divergence
        signals['score'] = score
        
        return signals

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
            
        result = self._result
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

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        获取MACD指标的技术形态
        
        Args:
            data: 输入数据，通常是K线数据
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 形态列表
        """
        # 确保已计算MACD指标
        if self._result is None:
            self.calculate(data)
        
        # 调用父类方法获取基础形态
        patterns = super().get_patterns(data, **kwargs)
        
        # MACD特有的形态处理逻辑
        for pattern in patterns:
            # 添加MACD特有的详细信息
            if self._result is not None and not self._result.empty:
                latest_idx = self._result.index[-1]
                pattern['details'].update({
                    'dif': float(self._result['DIF'].iloc[-1]),
                    'dea': float(self._result['DEA'].iloc[-1]),
                    'macd': float(self._result['MACD'].iloc[-1]),
                    'is_above_zero': bool(self._result['DIF'].iloc[-1] > 0),
                    'hist_direction': 'up' if self._result['MACD'].iloc[-1] > self._result['MACD'].iloc[-2] else 'down'
                })
                
                # 根据形态类型增强强度计算
                if pattern['pattern_id'] == 'golden_cross':
                    # 金叉强度与角度和DIF值相关
                    angle = self._calculate_cross_angle(self._result['DIF'], self._result['DEA'])
                    pattern['strength'] = min(100, 50 + angle.iloc[-1] * 10)
                    
                elif pattern['pattern_id'] == 'death_cross':
                    # 死叉强度与角度和DIF值相关
                    angle = self._calculate_cross_angle(self._result['DEA'], self._result['DIF'])
                    pattern['strength'] = min(100, 50 + angle.iloc[-1] * 10)
                    
                elif pattern['pattern_id'] in ['bullish_divergence', 'bearish_divergence']:
                    # 背离强度与价格和MACD的差异程度相关
                    pattern['strength'] = self._calculate_divergence_strength(
                        data['close'], 
                        self._result['DIF'],
                        is_bullish=(pattern['pattern_id'] == 'bullish_divergence')
                    )
        
        return patterns

    def _calculate_divergence_strength(self, price: pd.Series, indicator: pd.Series, 
                                     is_bullish: bool = True, lookback: int = 20) -> float:
        """
        计算背离强度
        
        Args:
            price: 价格序列
            indicator: 指标序列
            is_bullish: 是否为底背离
            lookback: 回溯周期数
            
        Returns:
            float: 背离强度(0-100)
        """
        if len(price) < lookback or len(indicator) < lookback:
            return 50.0
        
        # 截取回溯窗口的数据
        price_window = price.iloc[-lookback:]
        indicator_window = indicator.iloc[-lookback:]
        
        if is_bullish:
            # 底背离: 寻找价格和指标的低点
            price_min_idx = price_window.idxmin()
            indicator_min_idx = indicator_window.idxmin()
            
            # 如果最低点不一致，说明存在背离
            if price_min_idx != indicator_min_idx:
                # 计算价格新低的程度
                price_latest_min = price_window.iloc[-5:].min()
                price_prev_min = price_window.iloc[:-5].min()
                price_decline = (price_prev_min - price_latest_min) / price_prev_min if price_prev_min > 0 else 0
                
                # 计算指标背离的程度
                indicator_latest_min = indicator_window.iloc[-5:].min()
                indicator_prev_min = indicator_window.iloc[:-5].min()
                indicator_improve = max(0, indicator_latest_min - indicator_prev_min)
                
                # 综合评分: 价格下跌越多，指标改善越明显，评分越高
                return min(100, 50 + price_decline * 100 + indicator_improve * 20)
        else:
            # 顶背离: 寻找价格和指标的高点
            price_max_idx = price_window.idxmax()
            indicator_max_idx = indicator_window.idxmax()
            
            # 如果最高点不一致，说明存在背离
            if price_max_idx != indicator_max_idx:
                # 计算价格新高的程度
                price_latest_max = price_window.iloc[-5:].max()
                price_prev_max = price_window.iloc[:-5].max()
                price_rise = (price_latest_max - price_prev_max) / price_prev_max if price_prev_max > 0 else 0
                
                # 计算指标背离的程度
                indicator_latest_max = indicator_window.iloc[-5:].max()
                indicator_prev_max = indicator_window.iloc[:-5].max()
                indicator_weaken = max(0, indicator_prev_max - indicator_latest_max)
                
                # 综合评分: 价格上涨越多，指标减弱越明显，评分越高
                return min(100, 50 + price_rise * 100 + indicator_weaken * 20)
        
        return 50.0

    def calculate_score(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        计算指标评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 评分结果，包含：
                - raw_score: 原始评分序列
                - final_score: 最终评分序列
                - market_environment: 市场环境
                - patterns: 识别的形态
                - signals: 生成的信号
                - confidence: 置信度
        """
        try:
            # 确保已计算MACD
            if not self.has_result():
                self.calculate(data)
                
            # 检测市场环境
            market_env = self.detect_market_environment(data)
            self.set_market_environment(market_env)
            
            # 计算原始评分
            raw_score = self.calculate_raw_score(data, **kwargs)
            
            # 应用市场环境调整
            final_score = self.apply_market_environment_adjustment(raw_score, market_env)
            
            # 识别形态
            patterns = self.get_patterns(data, **kwargs)
            
            # 生成信号
            signals = self.generate_trading_signals(data, **kwargs)
            
            # 计算置信度
            confidence = self.calculate_confidence(raw_score, patterns, signals)
            
            # 构建返回结果
            result = {
                'raw_score': raw_score,
                'final_score': final_score,
                'market_environment': market_env,
                'patterns': patterns,
                'signals': signals,
                'confidence': confidence
            }
            
            return result
            
        except Exception as e:
            self._error = e
            import traceback
            traceback.print_exc()
            return {
                'raw_score': None,
                'final_score': None,
                'market_environment': None,
                'patterns': [],
                'signals': {},
                'confidence': 0.0,
                'error': str(e)
            }
            
    def apply_market_environment_adjustment(self, score: pd.Series, market_env: MarketEnvironment) -> pd.Series:
        """
        根据市场环境调整评分
        
        Args:
            score: 原始评分序列
            market_env: 市场环境
            
        Returns:
            pd.Series: 调整后的评分序列
        """
        adjusted_score = score.copy()
        
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市环境下，上涨信号得分提高，下跌信号得分降低
            adjusted_score = np.where(score > 50, 
                                    score + (score - 50) * 0.2,  # 多头信号增强
                                    score + (score - 50) * 0.1)  # 空头信号减弱
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市环境下，下跌信号得分提高，上涨信号得分降低
            adjusted_score = np.where(score < 50, 
                                    score - (50 - score) * 0.2,  # 空头信号增强
                                    score - (score - 50) * 0.1)  # 多头信号减弱
        elif market_env == MarketEnvironment.VOLATILE_MARKET:
            # 高波动市场，极端信号得分更加极端，中性信号得分更加中性
            adjusted_score = np.where(
                (score > 60) | (score < 40),  # 极端信号
                score + (score - 50) * 0.15,  # 极端更极端
                50 + (score - 50) * 0.8  # 中性更中性
            )
            
        # 限制评分范围在0-100之间
        return pd.Series(np.clip(adjusted_score, 0, 100), index=score.index)
        
    def calculate_confidence(self, score: pd.Series, patterns: List[str], signals: Dict[str, pd.Series]) -> float:
        """
        计算指标结果的置信度
        
        Args:
            score: 评分序列
            patterns: 识别的形态
            signals: 生成的信号
            
        Returns:
            float: 置信度（0-1）
        """
        # 基础置信度：0.5
        confidence = 0.5
        
        # 根据形态数量调整置信度
        if len(patterns) >= 3:
            confidence += 0.1  # 多种形态同时出现，置信度提高
        
        # 根据评分极端程度调整置信度
        latest_score = score.iloc[-1]
        if latest_score > 80 or latest_score < 20:
            confidence += 0.15  # 评分极端，置信度提高
        elif 40 <= latest_score <= 60:
            confidence -= 0.1  # 评分中性，置信度降低
        
        # 根据信号强度调整置信度
        if 'buy_strength' in signals and signals['buy_strength'].iloc[-1] >= 4:
            confidence += 0.1  # 强买入信号，置信度提高
        if 'sell_strength' in signals and signals['sell_strength'].iloc[-1] >= 4:
            confidence += 0.1  # 强卖出信号，置信度提高
        
        # 根据高质量形态调整置信度
        high_quality_patterns = ["高质量金叉", "高质量死叉", "MACD正背离", "MACD负背离"]
        for pattern in high_quality_patterns:
            if pattern in patterns:
                confidence += 0.05  # 高质量形态，置信度提高
                
        # 限制置信度范围在0-1之间
        return max(0.0, min(1.0, confidence)) 