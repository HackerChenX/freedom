"""
增强型随机指标(KDJ)模块

实现改进版的KDJ指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.kdj import KDJ
from utils.logger import get_logger
from utils.indicator_utils import crossover, crossunder

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
        # 先设置indicator_type，因为父类初始化时会调用get_indicator_type
        self.indicator_type = "ENHANCEDKDJ"
        super().__init__(n=n, m1=m1, m2=m2)
        self.name = "EnhancedKDJ"
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
        # 调用父类的calculate方法，获取包含K, D, J基础计算的DataFrame
        result = super().calculate(data, *args, **kwargs)

        # 确保数据包含必需的列
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in result.columns:
                raise ValueError(f"数据必须包含'{col}'列")

        # 确保基础KDJ列存在
        if 'K' not in result.columns or 'D' not in result.columns or 'J' not in result.columns:
            # 如果基础KDJ列不存在，计算它们
            self._calculate_kdj(result, self.n, self.m1, self.m2)
        
        # 计算多周期KDJ
        for n in self.multi_periods:
            if n != self.n:  # 避免重复计算
                self._calculate_multi_period_kdj(result, n, self.m1, self.m2)
        
        # 计算J线加速度
        if "J" in result.columns:
            result["j_acceleration"] = self._calculate_j_acceleration(result["J"])
        
        # 计算KD交叉角度
        if "K" in result.columns and "D" in result.columns:
            result["kd_cross_angle"] = self._calculate_kd_cross_angle(result["K"], result["D"])
        
        # 计算KD距离
        if "K" in result.columns and "D" in result.columns:
            result["kd_distance"] = result["K"] - result["D"]

        # 计算历史极值归一化J值
        if "J" in result.columns:
            result["j_normalized"] = self._normalize_j_values(result["J"])
        
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
        data["K"] = k
        data["D"] = d
        data["J"] = j
    
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
        data[f"K_{n}"] = k
        data[f"D_{n}"] = d
        data[f"J_{n}"] = j
    
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
        k = self._result["K"]
        d = self._result["D"]
        j = self._result["J"]
        
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
        if crossover(k, d).iloc[-1]:
            # 检查金叉质量
            cross_angle = self._result["kd_cross_angle"].iloc[-1]
            j_acceleration = self._result["j_acceleration"].iloc[-1]

            if cross_angle > 2 and j_acceleration > 0:
                patterns.append("KD高质量金叉")
            else:
                patterns.append("KD金叉")

        elif crossunder(k, d).iloc[-1]:
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
        
        k = self._result["K"]
        d = self._result["D"]
        j = self._result["J"]
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
        golden_cross = crossover(k, d)
        cross_quality = kd_angle.abs().clip(0, 5)
        score += golden_cross * (15 + cross_quality)  # 基础15分 + 最多5分质量加成

        # 死叉质量评分
        death_cross = crossunder(k, d)
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
        k_columns = [col for col in self._result.columns if col.startswith("K_")]
        d_columns = [col for col in self._result.columns if col.startswith("D_")]
        
        if not k_columns or not d_columns:
            return score
        
        # 添加主周期的K和D
        k_columns.append("K")
        d_columns.append("D")
        
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
        signals['K'] = result["K"]
        signals['D'] = result["D"]
        signals['J'] = result["J"]
        
        # 计算KDJ评分
        kdj_score = self.calculate_raw_score(data)
        signals['score'] = kdj_score
        
        # 生成买入信号
        buy_signal = (
            (crossover(signals['K'], signals['D'])) |  # KD金叉
            ((signals['K'] < 20) & (signals['D'] < 20)) |  # KD处于超卖区
            (signals['J'] < 0) |  # J线处于超卖区
            (signals['score'] > 70)  # 评分高于70
        )

        # 生成卖出信号
        sell_signal = (
            (crossunder(signals['K'], signals['D'])) |  # KD死叉
            ((signals['K'] > 80) & (signals['D'] > 80)) |  # KD处于超买区
            (signals['J'] > 100) |  # J线处于超买区
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
        market_env = getattr(self, 'market_environment', 'normal')
        if market_env == 'bull_market':
            # 牛市中降低买入门槛，提高卖出门槛
            buy_signal = buy_signal | (signals['score'] > 65)
            sell_signal = sell_signal & (signals['score'] < 25)
        elif market_env == 'bear_market':
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
        strength = pd.Series(0.0, index=result.index)  # 中性强度为0
        
        # 买入信号强度
        for i in range(len(result)):
            if signals['buy_signal'].iloc[i]:
                # 根据评分、J线和KD交叉角度确定信号强度
                score = signals['score'].iloc[i]
                j_val = result['J'].iloc[i]
                kd_angle = result['kd_cross_angle'].iloc[i] if i > 0 else 0
                
                if score > 85 and j_val < -10 and kd_angle > 2:
                    strength.iloc[i] = 1.0  # 非常强
                elif score > 75 and j_val < 0:
                    strength.iloc[i] = 0.8  # 强
                elif score > 65:
                    strength.iloc[i] = 0.6  # 中等
                else:
                    strength.iloc[i] = 0.4  # 弱
            
            elif signals['sell_signal'].iloc[i]:
                # 根据评分、J线和KD交叉角度确定信号强度
                score = signals['score'].iloc[i]
                j_val = result['J'].iloc[i]
                kd_angle = result['kd_cross_angle'].iloc[i] if i > 0 else 0
                
                if score < 15 and j_val > 110 and kd_angle < -2:
                    strength.iloc[i] = -1.0  # 非常强（负向）
                elif score < 25 and j_val > 100:
                    strength.iloc[i] = -0.8  # 强（负向）
                elif score < 35:
                    strength.iloc[i] = -0.6  # 中等（负向）
                else:
                    strength.iloc[i] = -0.4  # 弱（负向）
        
        return strength

    def set_parameters(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        if 'n' in kwargs:
            self.n = kwargs['n']
        if 'm1' in kwargs:
            self.m1 = kwargs['m1']
        if 'm2' in kwargs:
            self.m2 = kwargs['m2']
        if 'sensitivity' in kwargs:
            self.sensitivity = kwargs['sensitivity']
        if 'j_weight' in kwargs:
            self.j_weight = kwargs['j_weight']
        if 'multi_periods' in kwargs:
            self.multi_periods = kwargs['multi_periods']

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算EnhancedKDJ指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5

        # 基础置信度
        confidence = 0.5

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:
            confidence += 0.25
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:
            confidence += 0.1
        else:
            confidence += 0.15

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查EnhancedKDJ形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)

        # 3. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)

        # 4. 基于评分趋势的置信度
        if len(score) >= 3:
            recent_scores = score.iloc[-3:]
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取EnhancedKDJ相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        patterns = pd.DataFrame(index=data.index)

        # 获取KDJ数据
        k = self._result['K']
        d = self._result['D']
        j = self._result['J']

        # 基本形态
        patterns['KDJ_GOLDEN_CROSS'] = crossover(k, d)
        patterns['KDJ_DEATH_CROSS'] = crossunder(k, d)
        patterns['KDJ_OVERSOLD'] = (k < 20) & (d < 20)
        patterns['KDJ_OVERBOUGHT'] = (k > 80) & (d > 80)
        patterns['KDJ_J_OVERSOLD'] = j < 0
        patterns['KDJ_J_OVERBOUGHT'] = j > 100

        # 趋势形态
        patterns['KDJ_K_RISING'] = k > k.shift(1)
        patterns['KDJ_K_FALLING'] = k < k.shift(1)
        patterns['KDJ_D_RISING'] = d > d.shift(1)
        patterns['KDJ_D_FALLING'] = d < d.shift(1)

        # 强度形态
        if 'kd_cross_angle' in self._result.columns:
            kd_angle = self._result['kd_cross_angle']
            patterns['KDJ_STRONG_GOLDEN_CROSS'] = patterns['KDJ_GOLDEN_CROSS'] & (kd_angle > 2)
            patterns['KDJ_STRONG_DEATH_CROSS'] = patterns['KDJ_DEATH_CROSS'] & (kd_angle < -2)

        return patterns

    def register_patterns(self):
        """
        注册EnhancedKDJ指标的形态到全局形态注册表
        """
        # 注册KDJ交叉形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_GOLDEN_CROSS",
            display_name="KDJ金叉",
            description="K线上穿D线，表明上升趋势开始",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="KDJ_DEATH_CROSS",
            display_name="KDJ死叉",
            description="K线下穿D线，表明下降趋势开始",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,
            polarity="NEGATIVE"
        )

        # 注册KDJ超买超卖形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_OVERSOLD",
            display_name="KDJ超卖",
            description="K线和D线均低于20，表明市场超卖",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="KDJ_OVERBOUGHT",
            display_name="KDJ超买",
            description="K线和D线均高于80，表明市场超买",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,
            polarity="NEGATIVE"
        )

        # 注册J线极值形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_J_OVERSOLD",
            display_name="J线超卖",
            description="J线低于0，表明极度超卖",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="KDJ_J_OVERBOUGHT",
            display_name="J线超买",
            description="J线高于100，表明极度超买",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,
            polarity="NEGATIVE"
        )

        # 注册强势交叉形态
        self.register_pattern_to_registry(
            pattern_id="KDJ_STRONG_GOLDEN_CROSS",
            display_name="KDJ强势金叉",
            description="K线以大角度上穿D线，表明强势上升趋势",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="KDJ_STRONG_DEATH_CROSS",
            display_name="KDJ强势死叉",
            description="K线以大角度下穿D线，表明强势下降趋势",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,
            polarity="NEGATIVE"
        )

    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成EnhancedKDJ交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含买卖信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data)

        if self._result is None:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        k = self._result['K']
        d = self._result['D']
        j = self._result['J']

        # 生成信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 1. KDJ金叉死叉信号
        golden_cross = crossover(k, d)
        death_cross = crossunder(k, d)

        buy_signal |= golden_cross
        sell_signal |= death_cross
        signal_strength += golden_cross * 0.7
        signal_strength += death_cross * 0.7

        # 2. KDJ超买超卖信号
        oversold = (k < 20) & (d < 20)
        overbought = (k > 80) & (d > 80)

        buy_signal |= oversold
        sell_signal |= overbought
        signal_strength += oversold * 0.6
        signal_strength += overbought * 0.6

        # 3. J线极值信号
        j_oversold = j < 0
        j_overbought = j > 100

        buy_signal |= j_oversold
        sell_signal |= j_overbought
        signal_strength += j_oversold * 0.8
        signal_strength += j_overbought * 0.8

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return self.indicator_type

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态信息字典，包含name, description, strength等
        """
        pattern_info_map = {
            'KDJ_GOLDEN_CROSS': {
                'name': 'KDJ金叉',
                'description': 'K线上穿D线，表明上升趋势开始',
                'strength': 'medium',
                'type': 'bullish'
            },
            'KDJ_DEATH_CROSS': {
                'name': 'KDJ死叉',
                'description': 'K线下穿D线，表明下降趋势开始',
                'strength': 'medium',
                'type': 'bearish'
            },
            'KDJ_OVERSOLD': {
                'name': 'KDJ超卖',
                'description': 'K线和D线均低于20，表明市场超卖',
                'strength': 'medium',
                'type': 'bullish'
            },
            'KDJ_OVERBOUGHT': {
                'name': 'KDJ超买',
                'description': 'K线和D线均高于80，表明市场超买',
                'strength': 'medium',
                'type': 'bearish'
            },
            'KDJ_J_OVERSOLD': {
                'name': 'J线超卖',
                'description': 'J线低于0，表明极度超卖',
                'strength': 'strong',
                'type': 'bullish'
            },
            'KDJ_J_OVERBOUGHT': {
                'name': 'J线超买',
                'description': 'J线高于100，表明极度超买',
                'strength': 'strong',
                'type': 'bearish'
            },
            'KDJ_STRONG_GOLDEN_CROSS': {
                'name': 'KDJ强势金叉',
                'description': 'K线以大角度上穿D线，表明强势上升趋势',
                'strength': 'strong',
                'type': 'bullish'
            },
            'KDJ_STRONG_DEATH_CROSS': {
                'name': 'KDJ强势死叉',
                'description': 'K线以大角度下穿D线，表明强势下降趋势',
                'strength': 'strong',
                'type': 'bearish'
            },
            'KDJ_K_RISING': {
                'name': 'K线上升',
                'description': 'K线呈上升趋势',
                'strength': 'weak',
                'type': 'bullish'
            },
            'KDJ_K_FALLING': {
                'name': 'K线下降',
                'description': 'K线呈下降趋势',
                'strength': 'weak',
                'type': 'bearish'
            },
            'KDJ_D_RISING': {
                'name': 'D线上升',
                'description': 'D线呈上升趋势',
                'strength': 'weak',
                'type': 'bullish'
            },
            'KDJ_D_FALLING': {
                'name': 'D线下降',
                'description': 'D线呈下降趋势',
                'strength': 'weak',
                'type': 'bearish'
            },
            'KDJ_BULLISH_DIVERGENCE': {
                'name': 'KDJ牛市背离',
                'description': '价格创新低而KDJ不创新低，表示看涨信号',
                'strength': 'very_strong',
                'type': 'bullish'
            },
            'KDJ_BEARISH_DIVERGENCE': {
                'name': 'KDJ熊市背离',
                'description': '价格创新高而KDJ不创新高，表示看跌信号',
                'strength': 'very_strong',
                'type': 'bearish'
            }
        }

        return pattern_info_map.get(pattern_id, {
            'name': pattern_id,
            'description': f'EnhancedKDJ形态: {pattern_id}',
            'strength': 'medium',
            'type': 'neutral'
        })