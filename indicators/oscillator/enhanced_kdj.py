"""
增强型随机指标(KDJ)模块

实现改进版的KDJ指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.kdj import KDJ
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedKDJ(KDJ):
    """
    增强型随机指标(KDJ)
    
    在标准KDJ基础上增加了K线与D线交叉质量评估、J线加速度分析、信号过滤和多周期确认等功能
    """
    
    def __init__(self, 
                n: int = 9, 
                m1: int = 3, 
                m2: int = 3,
                sensitivity: float = 1.0,
                multi_periods: List[int] = None,
                j_weight: float = 1.0,
                smoothing_period_d: int = 3,
                adaptive_params: bool = True,
                volatility_lookback: int = 20,
                secondary_n: int = 6,
                use_smoothed_kdj: bool = True,
                smoothing_period: int = 3):
        """
        初始化增强型KDJ指标
        
        Args:
            n: RSV计算周期，默认为9
            m1: K值平滑周期，默认为3
            m2: D值平滑周期，默认为3
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            multi_periods: 多周期分析参数，默认为[5, 9, 14]
            j_weight: J线权重，用于调整J线在信号生成中的重要性，默认为1.0
            smoothing_period_d: D值平滑周期，默认为3
            adaptive_params: 是否使用自适应参数，默认为True
            volatility_lookback: 波动性回看周期，默认为20
            secondary_n: 次要周期，默认为6
            use_smoothed_kdj: 是否使用平滑后的KDJ，默认为True
            smoothing_period: 平滑周期，默认为3
        """
        super().__init__(n=n, m1=m1, m2=m2)
        self.name = "EnhancedKDJ"
        self.indicator_type = "ENHANCEDKDJ"
        self.description = "增强型随机指标，优化计算方法和信号质量，增加多周期适应和市场环境感知"
        self.sensitivity = sensitivity
        self.multi_periods = multi_periods or [5, 9, 14]
        self.j_weight = j_weight
        self.smoothing_period_d = smoothing_period_d
        self.adaptive_params = adaptive_params
        self.volatility_lookback = volatility_lookback
        self.secondary_n = secondary_n
        self.use_smoothed_kdj = use_smoothed_kdj
        self.smoothing_period = smoothing_period
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型KDJ指标
        
        Args:
            data: 输入数据，包含OHLC数据
            
        Returns:
            pd.DataFrame: 计算结果，包含KDJ及其多周期指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "close"])
        
        # 复制输入数据
        result = data.copy()
        
        # 计算标准KDJ
        self._calculate_kdj(result, self.n, self.m1, self.m2)
        
        # 计算多周期KDJ
        for n in self.multi_periods:
            if n != self.n:  # 避免重复计算
                self._calculate_multi_period_kdj(result, n, self.m1, self.m2)
        
        # 计算J线加速度
        result["j_acceleration"] = self._calculate_j_acceleration(result["j"])
        
        # 计算KD交叉角度
        result["kd_cross_angle"] = self._calculate_kd_cross_angle(result["k"], result["d"])
        
        # 计算KD距离
        result["kd_distance"] = result["k"] - result["d"]
        
        # 计算历史极值归一化J值
        result["j_normalized"] = self._normalize_j_values(result["j"])
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_kdj(self, data: pd.DataFrame, n: int, m1: int, m2: int) -> None:
        """
        计算KDJ指标
        
        Args:
            data: 输入数据
            n: RSV计算周期
            m1: K值平滑周期
            m2: D值平滑周期
        """
        # 计算N日内的最高价和最低价
        high_n = data["high"].rolling(window=n).max()
        low_n = data["low"].rolling(window=n).min()
        
        # 计算RSV值
        rsv = 100 * (data["close"] - low_n) / (high_n - low_n + 1e-10)
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            # 调整RSV值的响应度
            rsv = 50 + (rsv - 50) * self.sensitivity
            rsv = rsv.clip(0, 100)  # 确保RSV在0-100范围内
        
        # 计算K值，初始值为50
        k = pd.Series(50.0, index=data.index)
        for i in range(1, len(data)):
            k.iloc[i] = (m1 - 1) / m1 * k.iloc[i-1] + 1 / m1 * rsv.iloc[i]
        
        # 计算D值，初始值为50
        d = pd.Series(50.0, index=data.index)
        for i in range(1, len(data)):
            d.iloc[i] = (m2 - 1) / m2 * d.iloc[i-1] + 1 / m2 * k.iloc[i]
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 保存结果
        data["rsv"] = rsv
        data["k"] = k
        data["d"] = d
        data["j"] = j
    
    def _calculate_multi_period_kdj(self, data: pd.DataFrame, n: int, m1: int, m2: int) -> None:
        """
        计算多周期KDJ指标
        
        Args:
            data: 输入数据
            n: RSV计算周期
            m1: K值平滑周期
            m2: D值平滑周期
        """
        # 计算N日内的最高价和最低价
        high_n = data["high"].rolling(window=n).max()
        low_n = data["low"].rolling(window=n).min()
        
        # 计算RSV值
        rsv = 100 * (data["close"] - low_n) / (high_n - low_n + 1e-10)
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            rsv = 50 + (rsv - 50) * self.sensitivity
            rsv = rsv.clip(0, 100)
        
        # 计算K值，初始值为50
        k = pd.Series(50.0, index=data.index)
        for i in range(1, len(data)):
            k.iloc[i] = (m1 - 1) / m1 * k.iloc[i-1] + 1 / m1 * rsv.iloc[i]
        
        # 计算D值，初始值为50
        d = pd.Series(50.0, index=data.index)
        for i in range(1, len(data)):
            d.iloc[i] = (m2 - 1) / m2 * d.iloc[i-1] + 1 / m2 * k.iloc[i]
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 保存结果
        data[f"rsv_{n}"] = rsv
        data[f"k_{n}"] = k
        data[f"d_{n}"] = d
        data[f"j_{n}"] = j
    
    def _calculate_j_acceleration(self, j: pd.Series, window: int = 3) -> pd.Series:
        """
        计算J线加速度
        
        Args:
            j: J线序列
            window: 计算窗口
            
        Returns:
            pd.Series: J线加速度
        """
        # 计算J线一阶差分（速度）
        j_velocity = j.diff(periods=1)
        
        # 计算J线二阶差分（加速度）
        j_acceleration = j_velocity.diff(periods=1)
        
        # 使用移动平均平滑加速度
        j_acceleration_smooth = j_acceleration.rolling(window=window).mean()
        
        return j_acceleration_smooth
    
    def _calculate_kd_cross_angle(self, k: pd.Series, d: pd.Series) -> pd.Series:
        """
        计算K线与D线交叉角度
        
        Args:
            k: K线序列
            d: D线序列
            
        Returns:
            pd.Series: KD交叉角度
        """
        # 计算K线和D线的斜率
        k_slope = k.diff(periods=1)
        d_slope = d.diff(periods=1)
        
        # 计算角度（近似值，用斜率差表示）
        angle = k_slope - d_slope
        
        return angle
    
    def _normalize_j_values(self, j: pd.Series, window: int = 100) -> pd.Series:
        """
        对J值进行历史极值归一化
        
        Args:
            j: J线序列
            window: 历史窗口大小
            
        Returns:
            pd.Series: 归一化后的J值
        """
        # 使用滚动窗口计算历史最高和最低J值
        j_max = j.rolling(window=window).max()
        j_min = j.rolling(window=window).min()
        
        # 归一化到0-100范围
        j_normalized = (j - j_min) / (j_max - j_min + 1e-10) * 100
        
        # 填充前window个值
        j_normalized.fillna(50, inplace=True)
        
        return j_normalized
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别KDJ指标形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        if not self.has_result():
            self.calculate(data)
            
        patterns = []
        k = self._result["k"]
        d = self._result["d"]
        j = self._result["j"]
        
        # 1. 超买超卖区域判断
        if k.iloc[-1] < 20 and d.iloc[-1] < 20:
            patterns.append("KD超卖区域")
        elif k.iloc[-1] > 80 and d.iloc[-1] > 80:
            patterns.append("KD超买区域")
            
        # 2. J线极值判断
        if j.iloc[-1] < 0:
            patterns.append("J线超卖区域")
        elif j.iloc[-1] > 100:
            patterns.append("J线超买区域")
            
        # 3. 金叉/死叉判断
        if self.crossover(k, d).iloc[-1]:
            # 检查金叉质量
            cross_angle = self._result["kd_cross_angle"].iloc[-1]
            j_acceleration = self._result["j_acceleration"].iloc[-1]
            
            if cross_angle > 2 and j_acceleration > 0:
                patterns.append("KD高质量金叉")
            else:
                patterns.append("KD金叉")
                
        elif self.crossunder(k, d).iloc[-1]:
            # 检查死叉质量
            cross_angle = self._result["kd_cross_angle"].iloc[-1]
            j_acceleration = self._result["j_acceleration"].iloc[-1]
            
            if cross_angle < -2 and j_acceleration < 0:
                patterns.append("KD高质量死叉")
            else:
                patterns.append("KD死叉")
                
        # 4. 三重底/三重顶形态
        if len(j) >= 20:
            if self._detect_triple_bottom(j.iloc[-20:]):
                patterns.append("KDJ三重底")
            elif self._detect_triple_top(j.iloc[-20:]):
                patterns.append("KDJ三重顶")
                
        # 5. 背离检测
        if len(data) >= 30:
            divergence = self._detect_divergence(data["close"], j)
            if divergence == "bullish":
                patterns.append("KDJ正背离")
            elif divergence == "bearish":
                patterns.append("KDJ负背离")
        
        return patterns
    
    def _detect_triple_bottom(self, j_series: pd.Series) -> bool:
        """
        检测三重底形态
        
        Args:
            j_series: J线序列
            
        Returns:
            bool: 是否形成三重底形态
        """
        if len(j_series) < 15:
            return False
            
        # 寻找局部最低点
        valleys = []
        for i in range(1, len(j_series) - 1):
            if j_series.iloc[i] < j_series.iloc[i-1] and j_series.iloc[i] < j_series.iloc[i+1]:
                valleys.append((i, j_series.iloc[i]))
                
        # 至少需要三个谷点
        if len(valleys) < 3:
            return False
            
        # 取最近的三个谷点
        recent_valleys = sorted(valleys, key=lambda x: x[0])[-3:]
        
        # 条件1: 三个谷点都在超卖区域或接近超卖区域
        if not all(v[1] < 20 for v in recent_valleys):
            return False
            
        # 条件2: 谷点之间有足够的距离
        if recent_valleys[1][0] - recent_valleys[0][0] < 3 or recent_valleys[2][0] - recent_valleys[1][0] < 3:
            return False
            
        # 条件3: 第三个谷点不低于第一和第二个谷点
        if recent_valleys[2][1] < min(recent_valleys[0][1], recent_valleys[1][1]):
            return False
            
        return True
    
    def _detect_triple_top(self, j_series: pd.Series) -> bool:
        """
        检测三重顶形态
        
        Args:
            j_series: J线序列
            
        Returns:
            bool: 是否形成三重顶形态
        """
        if len(j_series) < 15:
            return False
            
        # 寻找局部最高点
        peaks = []
        for i in range(1, len(j_series) - 1):
            if j_series.iloc[i] > j_series.iloc[i-1] and j_series.iloc[i] > j_series.iloc[i+1]:
                peaks.append((i, j_series.iloc[i]))
                
        # 至少需要三个峰点
        if len(peaks) < 3:
            return False
            
        # 取最近的三个峰点
        recent_peaks = sorted(peaks, key=lambda x: x[0])[-3:]
        
        # 条件1: 三个峰点都在超买区域或接近超买区域
        if not all(p[1] > 80 for p in recent_peaks):
            return False
            
        # 条件2: 峰点之间有足够的距离
        if recent_peaks[1][0] - recent_peaks[0][0] < 3 or recent_peaks[2][0] - recent_peaks[1][0] < 3:
            return False
            
        # 条件3: 第三个峰点不高于第一和第二个峰点
        if recent_peaks[2][1] > max(recent_peaks[0][1], recent_peaks[1][1]):
            return False
            
        return True
    
    def _detect_divergence(self, price: pd.Series, j: pd.Series) -> Optional[str]:
        """
        检测KDJ背离
        
        Args:
            price: 价格序列
            j: J线序列
            
        Returns:
            Optional[str]: 背离类型 ("bullish", "bearish" 或 None)
        """
        if len(price) < 30 or len(j) < 30:
            return None
            
        # 寻找价格高点和低点
        price_highs = []
        price_lows = []
        
        for i in range(5, len(price) - 5):
            # 价格高点
            if price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1] and \
               price.iloc[i] == price.iloc[i-5:i+6].max():
                price_highs.append((i, price.iloc[i]))
            # 价格低点
            if price.iloc[i] < price.iloc[i-1] and price.iloc[i] < price.iloc[i+1] and \
               price.iloc[i] == price.iloc[i-5:i+6].min():
                price_lows.append((i, price.iloc[i]))
        
        # 寻找J线高点和低点
        j_highs = []
        j_lows = []
        
        for i in range(5, len(j) - 5):
            # J线高点
            if j.iloc[i] > j.iloc[i-1] and j.iloc[i] > j.iloc[i+1] and \
               j.iloc[i] == j.iloc[i-5:i+6].max():
                j_highs.append((i, j.iloc[i]))
            # J线低点
            if j.iloc[i] < j.iloc[i-1] and j.iloc[i] < j.iloc[i+1] and \
               j.iloc[i] == j.iloc[i-5:i+6].min():
                j_lows.append((i, j.iloc[i]))
        
        # 检查顶背离（价格创新高，J线未创新高）
        if len(price_highs) >= 2 and len(j_highs) >= 2:
            recent_price_highs = sorted(price_highs, key=lambda x: x[0])[-2:]
            recent_j_highs = sorted(j_highs, key=lambda x: x[0])[-2:]
            
            # 确保最近的高点在相似的时间范围内
            if abs(recent_price_highs[1][0] - recent_j_highs[1][0]) <= 3:
                # 价格创新高但J线未创新高
                if recent_price_highs[1][1] > recent_price_highs[0][1] and \
                   recent_j_highs[1][1] < recent_j_highs[0][1]:
                    return "bearish"
        
        # 检查底背离（价格创新低，J线未创新低）
        if len(price_lows) >= 2 and len(j_lows) >= 2:
            recent_price_lows = sorted(price_lows, key=lambda x: x[0])[-2:]
            recent_j_lows = sorted(j_lows, key=lambda x: x[0])[-2:]
            
            # 确保最近的低点在相似的时间范围内
            if abs(recent_price_lows[1][0] - recent_j_lows[1][0]) <= 3:
                # 价格创新低但J线未创新低
                if recent_price_lows[1][1] < recent_price_lows[0][1] and \
                   recent_j_lows[1][1] > recent_j_lows[0][1]:
                    return "bullish"
        
        return None
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型KDJ原始评分
        
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
        
        k = self._result["k"]
        d = self._result["d"]
        j = self._result["j"]
        j_norm = self._result["j_normalized"]
        kd_angle = self._result["kd_cross_angle"]
        j_accel = self._result["j_acceleration"]
        
        # 1. 基础KDJ评分 - 超买超卖区域
        # KD在超卖区域，看涨
        oversold_score = ((20 - k) / 20 * 15 + (20 - d) / 20 * 15).clip(0, 30)
        score += ((k < 20) & (d < 20)) * oversold_score
        
        # KD在超买区域，看跌
        overbought_score = ((k - 80) / 20 * 15 + (d - 80) / 20 * 15).clip(0, 30)
        score -= ((k > 80) & (d > 80)) * overbought_score
        
        # 2. J线评分 - 考虑J线权重
        # J线超卖，看涨
        j_oversold_score = ((0 - j) / 50 * 20 * self.j_weight).clip(0, 20)
        score += (j < 0) * j_oversold_score
        
        # J线超买，看跌
        j_overbought_score = ((j - 100) / 50 * 20 * self.j_weight).clip(0, 20)
        score -= (j > 100) * j_overbought_score
        
        # 3. KD交叉信号评分
        # 金叉质量评分
        golden_cross = self.crossover(k, d)
        cross_quality = kd_angle.abs().clip(0, 5)
        score += golden_cross * (15 + cross_quality)  # 基础15分 + 最多5分质量加成
        
        # 死叉质量评分
        death_cross = self.crossunder(k, d)
        score -= death_cross * (15 + cross_quality)  # 基础15分 + 最多5分质量加成
        
        # 4. J线加速度评分
        # J线加速度为正，看涨
        score += (j_accel > 0.5) * 10
        
        # J线加速度为负，看跌
        score -= (j_accel < -0.5) * 10
        
        # 5. KD位置评分
        # KD均值位置（估算当前位置相对于历史范围）
        kd_position = (k + d) / 2
        score += ((kd_position - 50) / 50 * 10)  # KD均值偏高，加分；偏低，减分
        
        # 6. 多周期KDJ一致性评分
        multi_period_score = self._calculate_multi_period_consistency()
        score += multi_period_score
        
        # 7. 背离评分
        divergence = [self._detect_divergence(data["close"].iloc[:i+1], j.iloc[:i+1]) 
                     for i in range(len(data))]
        
        # 正背离（看涨）
        score += pd.Series([25 if d == "bullish" else 0 for d in divergence], index=score.index)
        
        # 负背离（看跌）
        score -= pd.Series([25 if d == "bearish" else 0 for d in divergence], index=score.index)
        
        # 8. KD交叉位置评分（低位金叉和高位死叉更有效）
        # 低位金叉（在20以下金叉）更看涨
        low_golden_cross = golden_cross & (k < 20) & (d < 20)
        score += low_golden_cross * 10
        
        # 高位死叉（在80以上死叉）更看跌
        high_death_cross = death_cross & (k > 80) & (d > 80)
        score -= high_death_cross * 10
        
        # 限制得分范围
        return np.clip(score, 0, 100)
    
    def _calculate_multi_period_consistency(self) -> pd.Series:
        """
        计算多周期KDJ一致性评分
        
        Returns:
            pd.Series: 多周期一致性评分
        """
        if not self.has_result():
            return pd.Series(0.0, index=self._result.index)
        
        score = pd.Series(0.0, index=self._result.index)
        
        # 获取所有计算的KDJ周期
        k_columns = [col for col in self._result.columns if col.startswith("k_")]
        d_columns = [col for col in self._result.columns if col.startswith("d_")]
        
        if not k_columns or not d_columns:
            return score
        
        # 添加主周期的K和D
        k_columns.append("k")
        d_columns.append("d")
        
        # 计算每个时间点的趋势一致性
        for i in range(len(self._result)):
            bullish_count = 0
            bearish_count = 0
            
            for k_col, d_col in zip(k_columns, d_columns):
                if i < 1:  # 数据不足，跳过
                    continue
                    
                k_val = self._result[k_col].iloc[i]
                d_val = self._result[d_col].iloc[i]
                
                # 金叉形成
                if k_val > d_val and self._result[k_col].iloc[i-1] <= self._result[d_col].iloc[i-1]:
                    bullish_count += 1
                # 死叉形成
                elif k_val < d_val and self._result[k_col].iloc[i-1] >= self._result[d_col].iloc[i-1]:
                    bearish_count += 1
                # K>D看涨，K<D看跌
                elif k_val > d_val:
                    bullish_count += 0.5
                elif k_val < d_val:
                    bearish_count += 0.5
            
            # 计算一致性得分
            total_count = bullish_count + bearish_count
            if total_count > 0:
                if bullish_count > bearish_count:
                    # 多数看涨
                    consistency = bullish_count / total_count
                    score.iloc[i] = consistency * 20  # 最高20分
                elif bearish_count > bullish_count:
                    # 多数看跌
                    consistency = bearish_count / total_count
                    score.iloc[i] = -consistency * 20  # 最低-20分
        
        return score
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成KDJ交易信号
        
        Args:
            data: 输入数据
            *args, **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 信号DataFrame
        """
        # 确保已计算KDJ
        if not self.has_result():
            self.calculate(data, *args, **kwargs)
            
        result = self._result.copy()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=result.index)
        signals['k'] = result["k"]
        signals['d'] = result["d"]
        signals['j'] = result["j"]
        
        # 计算KDJ评分
        kdj_score = self.calculate_raw_score(data)
        signals['score'] = kdj_score
        
        # 生成买入信号
        buy_signal = (
            (self.crossover(signals['k'], signals['d'])) |  # KD金叉
            ((signals['k'] < 20) & (signals['d'] < 20)) |  # KD处于超卖区
            (signals['j'] < 0) |  # J线处于超卖区
            (signals['score'] > 70)  # 评分高于70
        )
        
        # 生成卖出信号
        sell_signal = (
            (self.crossunder(signals['k'], signals['d'])) |  # KD死叉
            ((signals['k'] > 80) & (signals['d'] > 80)) |  # KD处于超买区
            (signals['j'] > 100) |  # J线处于超买区
            (signals['score'] < 30)  # 评分低于30
        )
        
        # 应用信号过滤 - 只保留高质量信号
        j_accel = result["j_acceleration"]
        kd_angle = result["kd_cross_angle"]
        
        # 买入信号过滤 - 要求J加速度为正或KD交叉角度较大
        buy_signal = buy_signal & ((j_accel > 0) | (kd_angle > 1))
        
        # 卖出信号过滤 - 要求J加速度为负或KD交叉角度较大（负向）
        sell_signal = sell_signal & ((j_accel < 0) | (kd_angle < -1))
        
        # 应用市场环境调整
        market_env = self.get_market_environment()
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中降低买入门槛，提高卖出门槛
            buy_signal = buy_signal | (signals['score'] > 65)
            sell_signal = sell_signal & (signals['score'] < 25)
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市中提高买入门槛，降低卖出门槛
            buy_signal = buy_signal & (signals['score'] > 75)
            sell_signal = sell_signal | (signals['score'] < 35)
        
        signals['buy_signal'] = buy_signal
        signals['sell_signal'] = sell_signal
        
        # 计算指标多空趋势
        signals['bull_trend'] = signals['score'] > 60
        signals['bear_trend'] = signals['score'] < 40
        
        # 添加信号强度
        signals['signal_strength'] = self._calculate_signal_strength(result, signals)
        
        return signals
    
    def _calculate_signal_strength(self, result: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        计算信号强度
        
        Args:
            result: 指标计算结果
            signals: 信号DataFrame
            
        Returns:
            pd.Series: 信号强度序列
        """
        strength = pd.Series(SignalStrength.NEUTRAL.value, index=result.index)
        
        # 买入信号强度
        for i in range(len(result)):
            if signals['buy_signal'].iloc[i]:
                # 根据评分、J线和KD交叉角度确定信号强度
                score = signals['score'].iloc[i]
                j_val = result['j'].iloc[i]
                kd_angle = result['kd_cross_angle'].iloc[i] if i > 0 else 0
                
                if score > 85 and j_val < -10 and kd_angle > 2:
                    strength.iloc[i] = SignalStrength.VERY_STRONG.value
                elif score > 75 and j_val < 0:
                    strength.iloc[i] = SignalStrength.STRONG.value
                elif score > 65:
                    strength.iloc[i] = SignalStrength.MODERATE.value
                else:
                    strength.iloc[i] = SignalStrength.WEAK.value
            
            elif signals['sell_signal'].iloc[i]:
                # 根据评分、J线和KD交叉角度确定信号强度
                score = signals['score'].iloc[i]
                j_val = result['j'].iloc[i]
                kd_angle = result['kd_cross_angle'].iloc[i] if i > 0 else 0
                
                if score < 15 and j_val > 110 and kd_angle < -2:
                    strength.iloc[i] = SignalStrength.VERY_STRONG_NEGATIVE.value
                elif score < 25 and j_val > 100:
                    strength.iloc[i] = SignalStrength.STRONG_NEGATIVE.value
                elif score < 35:
                    strength.iloc[i] = SignalStrength.MODERATE_NEGATIVE.value
                else:
                    strength.iloc[i] = SignalStrength.WEAK_NEGATIVE.value
        
        return strength 