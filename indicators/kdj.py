"""
KDJ指标模块

实现KDJ随机指标的计算和相关功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any

from indicators.base_indicator import BaseIndicator
from indicators.common import kdj as calc_kdj, highest, lowest
from utils.decorators import exception_handler, validate_dataframe, log_calls
from utils.logger import get_logger

logger = get_logger(__name__)


class KDJ(BaseIndicator):
    """
    KDJ随机指标
    
    KDJ指标是RSI和随机指标的结合体，是一种超买超卖指标，用于判断股价走势的超买超卖状态。
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        """
        初始化KDJ指标
        
        Args:
            n: RSV周期，默认为9
            m1: K值平滑因子，默认为3
            m2: D值平滑因子，默认为3
        """
        super().__init__(name="KDJ", description="随机指标")
        self.n = n
        self.m1 = m1
        self.m2 = m2
    
    @validate_dataframe(required_columns=['high', 'low', 'close'], min_rows=9)
    @log_calls(level='debug')
    @exception_handler(reraise=True)
    def calculate(self, data: pd.DataFrame, add_prefix: bool = False, **kwargs) -> pd.DataFrame:
        """
        计算KDJ指标
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            add_prefix: 是否在输出列名前添加指标名称前缀
            kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 包含KDJ指标的DataFrame
            
        Raises:
            ValueError: 如果输入数据不包含必需的列
        """
        # 复制输入数据
        result = data.copy()
        
        # 使用统一的公共函数计算KDJ
        k, d, j = calc_kdj(
            result['close'].values,
            result['high'].values,
            result['low'].values,
            self.n,
            self.m1,
            self.m2
        )
        
        # 设置列名（使用大写字母）
        if add_prefix:
            k_col = self.get_column_name('K')
            d_col = self.get_column_name('D')
            j_col = self.get_column_name('J')
        else:
            k_col = 'K'
            d_col = 'D'
            j_col = 'J'
        
        # 添加结果列
        result[k_col] = k
        result[d_col] = d
        result[j_col] = j
        
        # 添加信号
        result = self.add_signals(result, k_col, d_col, j_col)
        
        # 保存结果
        self._result = result
        
        return result
    
    def add_signals(self, data: pd.DataFrame, k_col: str = 'K', 
                   d_col: str = 'D', j_col: str = 'J') -> pd.DataFrame:
        """
        添加KDJ交易信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            j_col: J值列名
            
        Returns:
            pd.DataFrame: 添加了信号的DataFrame
        """
        result = data.copy()
        
        # 计算超买超卖信号
        result['kdj_overbought'] = (result[k_col] > 80) & (result[d_col] > 80)
        result['kdj_oversold'] = (result[k_col] < 20) & (result[d_col] < 20)
        
        # 计算金叉和死叉信号
        result['kdj_buy_signal'] = self.get_buy_signal(result, k_col, d_col)
        result['kdj_sell_signal'] = self.get_sell_signal(result, k_col, d_col)
        
        # 计算J值超买超卖
        result['kdj_j_overbought'] = result[j_col] > 100
        result['kdj_j_oversold'] = result[j_col] < 0
        
        # 计算KDJ三线同向（顺势信号）
        result['kdj_uptrend'] = (result[j_col] > result[k_col]) & (result[k_col] > result[d_col])
        result['kdj_downtrend'] = (result[j_col] < result[k_col]) & (result[k_col] < result[d_col])
        
        return result
    
    def get_buy_signal(self, data: pd.DataFrame, k_col: str = 'K', d_col: str = 'D') -> pd.Series:
        """
        获取KDJ买入信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            
        Returns:
            pd.Series: 买入信号序列（布尔值）
        """
        # KDJ金叉：K线从下方穿过D线，且处于低位（<30）
        golden_cross = (data[k_col] > data[d_col]) & (data[k_col].shift(1) <= data[d_col].shift(1))
        low_position = data[k_col] < 30
        
        return golden_cross & low_position
    
    def get_sell_signal(self, data: pd.DataFrame, k_col: str = 'K', d_col: str = 'D') -> pd.Series:
        """
        获取KDJ卖出信号
        
        Args:
            data: 包含KDJ指标的DataFrame
            k_col: K值列名
            d_col: D值列名
            
        Returns:
            pd.Series: 卖出信号序列（布尔值）
        """
        # KDJ死叉：K线从上方穿过D线，且处于高位（>70）
        death_cross = (data[k_col] < data[d_col]) & (data[k_col].shift(1) >= data[d_col].shift(1))
        high_position = data[k_col] > 70
        
        return death_cross & high_position
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将指标转换为字典表示
        
        Returns:
            Dict[str, Any]: 指标的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'n': self.n,
                'm1': self.m1,
                'm2': self.m2
            },
            'has_result': self.has_result(),
            'has_error': self.has_error(),
            'error': str(self._error) if self._error else None
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成KDJ信号
        
        Args:
            data: 输入数据，包含价格数据的DataFrame
            
        Returns:
            pd.DataFrame: 包含信号的DataFrame
        """
        # 计算KDJ指标
        if not self.has_result():
            self.compute(data)
            
        if not self.has_result():
            return pd.DataFrame()
            
        # 获取K、D和J值
        k_values = self._result['K']
        d_values = self._result['D']
        j_values = self._result['J']
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index)
        
        # 添加买入信号
        signals['buy_signal'] = self.get_buy_signal(self._result)
        
        # 添加卖出信号
        signals['sell_signal'] = self.get_sell_signal(self._result)
        
        # 添加超买超卖信号
        signals['overbought'] = (k_values > 80) & (d_values > 80)
        signals['oversold'] = (k_values < 20) & (d_values < 20)
        
        # 添加J值超买超卖
        signals['j_overbought'] = j_values > 100
        signals['j_oversold'] = j_values < 0
        
        # 添加KDJ三线同向（顺势信号）
        signals['uptrend'] = (j_values > k_values) & (k_values > d_values)
        signals['downtrend'] = (j_values < k_values) & (k_values < d_values)
        
        # 计算信号强度
        # 范围是0-100，0表示最弱，100表示最强
        strength = 50.0  # 默认中性
        
        # 如果出现金叉，信号强度增加
        if signals['buy_signal'].iloc[-1]:
            strength += 25.0
            
        # 如果处于超卖区域，信号强度增加
        if signals['oversold'].iloc[-1]:
            strength += 15.0
            
        # 如果J值在超卖区域，信号强度增加
        if signals['j_oversold'].iloc[-1]:
            strength += 10.0
            
        # 如果三线同向看涨，信号强度增加
        if signals['uptrend'].iloc[-1]:
            strength += 10.0
            
        # 如果处于超买区域，信号强度减少
        if signals['overbought'].iloc[-1]:
            strength -= 15.0
            
        # 如果出现死叉，信号强度减少
        if signals['sell_signal'].iloc[-1]:
            strength -= 25.0
            
        # 确保强度在0-100范围内
        strength = max(0.0, min(100.0, strength))
        
        # 添加信号强度
        signals['signal_strength'] = 0.0
        signals.loc[signals.index[-1], 'signal_strength'] = strength
        
        return signals
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算KDJ原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算KDJ
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        k = self._result['K']
        d = self._result['D']
        j = self._result['J']
        
        # 1. K和D线交叉评分
        golden_cross = self.crossover(k, d)
        death_cross = self.crossunder(k, d)
        
        # 优化点：KD线夹角系数
        if len(k) >= 3 and len(d) >= 3:
            # 计算K与D线夹角（使用差值占平均值的百分比来近似）
            kd_angle = abs(k - d) / ((k + d) / 2) * 100
            
            # 夹角越大，交叉信号越强
            kd_angle_coef = np.clip(1 + kd_angle * 0.01, 1.0, 1.5)  # 系数范围：1.0-1.5
            
            # 使用夹角系数调整交叉信号分数
            golden_cross_score = golden_cross * 20 * np.where(golden_cross, kd_angle_coef, 1.0)
            death_cross_score = death_cross * 20 * np.where(death_cross, kd_angle_coef, 1.0)
            
            score += golden_cross_score  # K上穿D+20分（带夹角系数）
            score -= death_cross_score   # K下穿D-20分（带夹角系数）
        else:
            # 如果历史数据不足，使用默认加分
            score += golden_cross * 20  # K上穿D+20分
            score -= death_cross * 20   # K下穿D-20分
        
        # 2. 超买超卖区域评分
        oversold_area = (k < 20) & (d < 20) & (j < 20)
        overbought_area = (k > 80) & (d > 80) & (j > 80)
        score += oversold_area * 25   # 三线超卖+25分
        score -= overbought_area * 25 # 三线超买-25分
        
        # 3. K线区域变化评分
        k_leaving_oversold = (k > 20) & (k.shift(1) <= 20)
        k_leaving_overbought = (k < 80) & (k.shift(1) >= 80)
        score += k_leaving_oversold * 15   # K线离开超卖区+15分
        score -= k_leaving_overbought * 15 # K线离开超买区-15分
        
        # 4. J线极端区域评分 - 优化点：历史极值归一化
        if len(j) >= 60:
            # 计算J值的60周期历史极值
            j_rolling_max = j.rolling(60).max()
            j_rolling_min = j.rolling(60).min()
            
            # 归一化J值到历史极值范围内（0-100）
            j_normalized = (j - j_rolling_min) / (j_rolling_max - j_rolling_min) * 100
            j_normalized = j_normalized.fillna(50)  # 处理可能的NaN值
            
            # 使用归一化的J值进行评分
            j_extreme_oversold = j_normalized < 10    # 历史10%分位
            j_extreme_overbought = j_normalized > 90  # 历史90%分位
            
            score += j_extreme_oversold * 30   # J线历史极低+30分
            score -= j_extreme_overbought * 30 # J线历史极高-30分
        else:
            # 如果历史数据不足，使用绝对值判断
            j_extreme_oversold = j < 0
            j_extreme_overbought = j > 100
            score += j_extreme_oversold * 30   # J线极端超卖+30分
            score -= j_extreme_overbought * 30 # J线极端超买-30分
        
        # 5. 钝化形态评分
        # 低位钝化（连续多个周期在超卖区）
        low_stagnation = self._detect_stagnation(k, d, j, low_threshold=20, periods=5)
        # 高位钝化（连续多个周期在超买区）
        high_stagnation = self._detect_stagnation(k, d, j, high_threshold=80, periods=5)
        
        score += low_stagnation * 20   # 低位钝化+20分
        score -= high_stagnation * 20  # 高位钝化-20分
        
        # 6. J线加速度评估 - 优化点：加速度因子
        if len(j) >= 3:
            # 计算J线的一阶导数（速度）和二阶导数（加速度）
            j_velocity = j - j.shift(1)
            j_acceleration = j_velocity - j_velocity.shift(1)
            
            # J线正加速（加速上升）
            j_positive_acceleration = j_acceleration > 0
            
            # J线负加速（加速下降）
            j_negative_acceleration = j_acceleration < 0
            
            # 加速度分值：加速度越大，分值越高
            accel_magnitude = np.clip(abs(j_acceleration) * 0.5, 0, 15)
            
            # 上升加速加分，下降加速减分
            accel_score = np.where(j_positive_acceleration, accel_magnitude, -accel_magnitude)
            score += accel_score
        
        # 7. 背离评分
        if len(data) >= 20:
            divergence_score = self._calculate_kdj_divergence(data['close'], k, d)
            score += divergence_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别KDJ技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        # 确保已计算KDJ
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return []
        
        patterns = []
        
        k = self._result['K']
        d = self._result['D']
        j = self._result['J']
        
        # 1. 检测金叉/死叉
        golden_cross = self.crossover(k, d)
        death_cross = self.crossunder(k, d)
        
        if golden_cross.any():
            # 优化：检测KD夹角
            if len(k) >= 3 and len(d) >= 3:
                last_cross_idx = golden_cross.to_numpy().nonzero()[0][-1]
                kd_angle = abs(k.iloc[last_cross_idx] - d.iloc[last_cross_idx]) / ((k.iloc[last_cross_idx] + d.iloc[last_cross_idx]) / 2) * 100
                
                if kd_angle > 10:
                    patterns.append("KDJ大角度金叉")
                else:
                    patterns.append("KDJ金叉")
            else:
                patterns.append("KDJ金叉")
        
        if death_cross.any():
            # 优化：检测KD夹角
            if len(k) >= 3 and len(d) >= 3:
                last_cross_idx = death_cross.to_numpy().nonzero()[0][-1]
                kd_angle = abs(k.iloc[last_cross_idx] - d.iloc[last_cross_idx]) / ((k.iloc[last_cross_idx] + d.iloc[last_cross_idx]) / 2) * 100
                
                if kd_angle > 10:
                    patterns.append("KDJ大角度死叉")
                else:
                    patterns.append("KDJ死叉")
            else:
                patterns.append("KDJ死叉")
        
        # 2. 检测超买超卖区域
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        current_j = j.iloc[-1]
        
        if current_k < 20 and current_d < 20:
            patterns.append("KDJ超卖区")
        elif current_k > 80 and current_d > 80:
            patterns.append("KDJ超买区")
        
        # 3. J线极值
        if current_j < 0:
            patterns.append("J值极端低位")
        elif current_j > 100:
            patterns.append("J值极端高位")
        
        # 4. 三线同向
        recent_period = min(5, len(k))
        if recent_period >= 3:
            k_trend = k.iloc[-1] > k.iloc[-3]
            d_trend = d.iloc[-1] > d.iloc[-3]
            j_trend = j.iloc[-1] > j.iloc[-3]
            
            if k_trend and d_trend and j_trend:
                patterns.append("KDJ三线同向上")
            elif not k_trend and not d_trend and not j_trend:
                patterns.append("KDJ三线同向下")
        
        # 5. J线加速度形态 - 优化：检测加速度变化
        if len(j) >= 3:
            j_velocity = j - j.shift(1)
            j_acceleration = j_velocity - j_velocity.shift(1)
            
            recent_accel = j_acceleration.iloc[-1]
            
            if recent_accel > 2:
                patterns.append("J线加速上升")
            elif recent_accel < -2:
                patterns.append("J线加速下降")
        
        # 6. KD线排列
        if current_j > current_k > current_d:
            patterns.append("KDJ多头排列")
        elif current_j < current_k < current_d:
            patterns.append("KDJ空头排列")
        
        # 7. 背离检测
        if len(data) >= 20 and 'close' in data.columns:
            divergence_type = self._detect_divergence(data['close'], k)
            if divergence_type:
                patterns.append(f"KDJ{divergence_type}")
        
        return patterns
    
    def _detect_stagnation(self, k: pd.Series, d: pd.Series, j: pd.Series, 
                          low_threshold: float = None, high_threshold: float = None, 
                          periods: int = 5) -> pd.Series:
        """
        检测钝化形态
        
        Args:
            k, d, j: KDJ三线
            low_threshold: 低位阈值
            high_threshold: 高位阈值
            periods: 检测周期数
            
        Returns:
            pd.Series: 钝化信号
        """
        stagnation = pd.Series(False, index=k.index)
        
        if low_threshold is not None:
            # 低位钝化：连续periods个周期都在低位
            low_condition = (k < low_threshold) & (d < low_threshold)
            low_stagnation = low_condition.rolling(window=periods).sum() >= periods
            stagnation |= low_stagnation
        
        if high_threshold is not None:
            # 高位钝化：连续periods个周期都在高位
            high_condition = (k > high_threshold) & (d > high_threshold)
            high_stagnation = high_condition.rolling(window=periods).sum() >= periods
            stagnation |= high_stagnation
        
        return stagnation
    
    def _calculate_kdj_divergence(self, price: pd.Series, k: pd.Series, d: pd.Series) -> pd.Series:
        """
        计算KDJ背离评分
        
        Args:
            price: 价格序列
            k, d: KDJ的K线和D线
            
        Returns:
            pd.Series: 背离评分序列
        """
        divergence_score = pd.Series(0.0, index=price.index)
        
        if len(price) < 20:
            return divergence_score
        
        # 寻找价格和KDJ的峰值谷值
        window = 5
        for i in range(window, len(price) - window):
            # 检查是否为价格峰值/谷值
            price_window = price.iloc[i-window:i+window+1]
            k_window = k.iloc[i-window:i+window+1]
            
            if price.iloc[i] == price_window.max():  # 价格峰值
                if k.iloc[i] != k_window.max():  # KDJ未创新高
                    divergence_score.iloc[i:i+10] -= 25  # 负背离
            elif price.iloc[i] == price_window.min():  # 价格谷值
                if k.iloc[i] != k_window.min():  # KDJ未创新低
                    divergence_score.iloc[i:i+10] += 25  # 正背离
        
        return divergence_score
    
    def _detect_divergence(self, price: pd.Series, k: pd.Series) -> Optional[str]:
        """
        检测KDJ背离形态
        
        Args:
            price: 价格序列
            k: KDJ的K线
            
        Returns:
            Optional[str]: 背离类型，如果没有检测到则返回None
        """
        if len(price) < 20 or len(k) < 20:
            return None
        
        # 查找局部极值点
        window = 5
        price_highs = []
        price_lows = []
        k_highs = []
        k_lows = []
        
        for i in range(window, len(price) - window):
            price_window = price.iloc[i-window:i+window+1]
            k_window = k.iloc[i-window:i+window+1]
            
            # 价格高点和低点
            if price.iloc[i] == price_window.max():  # 价格高点
                price_highs.append((i, price.iloc[i]))
                k_highs.append((i, k.iloc[i]))
            
            if price.iloc[i] == price_window.min():  # 价格低点
                price_lows.append((i, price.iloc[i]))
                k_lows.append((i, k.iloc[i]))
        
        # 至少需要两个高点或低点来判断背离
        if len(price_highs) >= 2 and len(k_highs) >= 2:
            # 检查顶背离：价格创新高，但KDJ未创新高
            latest_price_high = price_highs[-1][1]
            previous_price_high = price_highs[-2][1]
            latest_k_high = k_highs[-1][1]
            previous_k_high = k_highs[-2][1]
            
            if latest_price_high > previous_price_high and latest_k_high < previous_k_high:
                return "顶背离"
        
        if len(price_lows) >= 2 and len(k_lows) >= 2:
            # 检查底背离：价格创新低，但KDJ未创新低
            latest_price_low = price_lows[-1][1]
            previous_price_low = price_lows[-2][1]
            latest_k_low = k_lows[-1][1]
            previous_k_low = k_lows[-2][1]
            
            if latest_price_low < previous_price_low and latest_k_low > previous_k_low:
                return "底背离"
        
        return None 