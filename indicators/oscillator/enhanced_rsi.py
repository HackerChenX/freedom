"""
增强型相对强弱指标(RSI)模块

实现改进版的RSI指标，优化计算方法和信号质量，增加多周期适应能力和市场环境感知
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment, SignalStrength
from indicators.rsi import RSI
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRSI(RSI):
    """
    增强型相对强弱指标(RSI)
    
    在标准RSI基础上增加了动态超买超卖阈值、多周期分析、形态识别和背离检测等功能
    """
    
    def __init__(self, 
                period: int = 14, 
                sensitivity: float = 1.0,
                multi_periods: List[int] = None):
        """
        初始化增强型RSI指标
        
        Args:
            period: RSI计算周期，默认为14
            sensitivity: 灵敏度参数，控制对价格变化的响应程度，默认为1.0
            multi_periods: 多周期分析参数，默认为[9, 14, 21]
        """
        super().__init__(period=period)
        self.name = "EnhancedRSI"
        self.description = "增强型相对强弱指标，优化计算方法和信号质量，增加多周期适应和市场环境感知"
        self.sensitivity = sensitivity
        self.multi_periods = multi_periods or [9, 14, 21]
        self.indicator_type = "oscillator"  # 指标类型：震荡类
    
    def get_indicator_type(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.indicator_type
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型RSI指标
        
        Args:
            data: 输入数据，包含价格数据
            
        Returns:
            pd.DataFrame: 计算结果，包含RSI及其多周期指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["close"])
        
        # 复制输入数据
        result = data.copy()
        
        # 计算标准RSI
        close = result["close"]
        rsi = self._calculate_rsi(close, self.period)
        result[f"rsi_{self.period}"] = rsi
        
        # 计算多周期RSI
        for period in self.multi_periods:
            if period != self.period:  # 避免重复计算
                result[f"rsi_{period}"] = self._calculate_rsi(close, period)
        
        # 计算RSI动量 - 衡量RSI的变化速率
        result["rsi_momentum"] = self._calculate_rsi_momentum(rsi)
        
        # 计算RSI变化率 - 相对变化百分比
        result["rsi_rate"] = rsi.diff(periods=1).fillna(0)
        
        # 计算动态超买超卖阈值
        thresholds = self._calculate_dynamic_thresholds(data)
        result["oversold_threshold"] = thresholds["oversold"]
        result["overbought_threshold"] = thresholds["overbought"]
        
        # 保存结果
        self._result = result
        
        return result
    
    def _calculate_rsi(self, price_series: pd.Series, period: int) -> pd.Series:
        """
        计算RSI
        
        Args:
            price_series: 价格序列
            period: 计算周期
            
        Returns:
            pd.Series: RSI序列
        """
        # 计算价格变化
        delta = price_series.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # 应用灵敏度调整
        if self.sensitivity != 1.0:
            gain = gain * self.sensitivity
            loss = loss * self.sensitivity
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 处理零除情况
        rs = pd.Series(0.0, index=price_series.index)
        valid_idx = avg_loss > 0
        rs[valid_idx] = avg_gain[valid_idx] / avg_loss[valid_idx]
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_momentum(self, rsi: pd.Series, period: int = 5) -> pd.Series:
        """
        计算RSI动量
        
        Args:
            rsi: RSI序列
            period: 计算周期
            
        Returns:
            pd.Series: RSI动量序列
        """
        # 计算RSI变化率
        rsi_change = rsi - rsi.shift(period)
        
        # 标准化动量
        rsi_momentum = rsi_change / period
        
        return rsi_momentum
    
    def _calculate_dynamic_thresholds(self, data: pd.DataFrame, window: int = 60) -> Dict[str, pd.Series]:
        """
        计算动态超买超卖阈值
        
        Args:
            data: 输入数据
            window: 计算窗口
            
        Returns:
            Dict[str, pd.Series]: 包含超买超卖阈值的字典
        """
        # 计算价格波动率
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 年化波动率
        
        # 创建阈值序列
        oversold = pd.Series(30.0, index=data.index)  # 默认超卖阈值
        overbought = pd.Series(70.0, index=data.index)  # 默认超买阈值
        
        # 根据波动率调整阈值
        for i in range(len(data)):
            if i >= 20:  # 确保有足够数据计算波动率
                current_volatility = volatility.iloc[i]
                
                if current_volatility > 0.3:  # 高波动率
                    oversold.iloc[i] = 25.0
                    overbought.iloc[i] = 75.0
                elif current_volatility < 0.15:  # 低波动率
                    oversold.iloc[i] = 35.0
                    overbought.iloc[i] = 65.0
                # 中等波动率使用默认值
        
        # 考虑市场环境调整阈值
        market_env = self.get_market_environment()
        
        if market_env == MarketEnvironment.BULL_MARKET:
            # 牛市中提高超买阈值，降低超卖阈值
            overbought = overbought + 5
            oversold = oversold - 2
        elif market_env == MarketEnvironment.BEAR_MARKET:
            # 熊市中降低超买阈值，提高超卖阈值
            overbought = overbought - 5
            oversold = oversold + 2
        
        return {"oversold": oversold, "overbought": overbought}
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别RSI指标形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        if not self.has_result():
            self.calculate(data)
            
        patterns = []
        rsi = self._result[f"rsi_{self.period}"]
        
        # 1. 超买超卖区域判断
        oversold = self._result["oversold_threshold"]
        overbought = self._result["overbought_threshold"]
        
        if rsi.iloc[-1] < oversold.iloc[-1]:
            patterns.append("超卖区域")
        elif rsi.iloc[-1] > overbought.iloc[-1]:
            patterns.append("超买区域")
            
        # 2. W底/M顶形态检测
        if len(rsi) >= 20:
            if self._detect_w_bottom(rsi.iloc[-20:]):
                patterns.append("W底形态")
            if self._detect_m_top(rsi.iloc[-20:]):
                patterns.append("M顶形态")
                
        # 3. 背离检测
        if len(data) >= 30:
            divergence = self._detect_divergence(data["close"], rsi)
            if divergence == "bullish":
                patterns.append("RSI正背离")
            elif divergence == "bearish":
                patterns.append("RSI负背离")
            
        # 4. 失败摆动形态
        if len(rsi) >= 15:
            if self._detect_failure_swing(rsi.iloc[-15:], "bullish"):
                patterns.append("看涨失败摆动")
            elif self._detect_failure_swing(rsi.iloc[-15:], "bearish"):
                patterns.append("看跌失败摆动")
        
        return patterns
    
    def _detect_w_bottom(self, rsi_series: pd.Series) -> bool:
        """
        检测W底形态
        
        Args:
            rsi_series: RSI序列
            
        Returns:
            bool: 是否形成W底形态
        """
        if len(rsi_series) < 10:
            return False
            
        # 寻找局部最低点
        valleys = []
        for i in range(1, len(rsi_series) - 1):
            if rsi_series.iloc[i] < rsi_series.iloc[i-1] and rsi_series.iloc[i] < rsi_series.iloc[i+1]:
                valleys.append((i, rsi_series.iloc[i]))
                
        # 至少需要两个谷点
        if len(valleys) < 2:
            return False
            
        # 检查最近的两个谷点是否形成W底
        # 条件1: 两个谷点都在超卖区域或接近超卖区域
        # 条件2: 两个谷点之间有一个明显的峰值
        # 条件3: 第二个谷点不低于第一个谷点
        
        # 取最近的两个谷点
        recent_valleys = sorted(valleys, key=lambda x: x[0])[-2:]
        
        # 条件1: 谷点在超卖区域或接近超卖区域
        if not (recent_valleys[0][1] < 35 and recent_valleys[1][1] < 35):
            return False
            
        # 条件2: 两个谷点之间有一个明显的峰值
        mid_point = (recent_valleys[0][0] + recent_valleys[1][0]) // 2
        mid_range = rsi_series.iloc[recent_valleys[0][0]:recent_valleys[1][0]]
        
        if len(mid_range) == 0 or mid_range.max() < 40:
            return False
            
        # 条件3: 第二个谷点不低于第一个谷点 (可以相等)
        if recent_valleys[1][1] < recent_valleys[0][1]:
            return False
            
        return True
    
    def _detect_m_top(self, rsi_series: pd.Series) -> bool:
        """
        检测M顶形态
        
        Args:
            rsi_series: RSI序列
            
        Returns:
            bool: 是否形成M顶形态
        """
        if len(rsi_series) < 10:
            return False
            
        # 寻找局部最高点
        peaks = []
        for i in range(1, len(rsi_series) - 1):
            if rsi_series.iloc[i] > rsi_series.iloc[i-1] and rsi_series.iloc[i] > rsi_series.iloc[i+1]:
                peaks.append((i, rsi_series.iloc[i]))
                
        # 至少需要两个峰点
        if len(peaks) < 2:
            return False
            
        # 检查最近的两个峰点是否形成M顶
        # 条件1: 两个峰点都在超买区域或接近超买区域
        # 条件2: 两个峰点之间有一个明显的谷值
        # 条件3: 第二个峰点不高于第一个峰点
        
        # 取最近的两个峰点
        recent_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
        
        # 条件1: 峰点在超买区域或接近超买区域
        if not (recent_peaks[0][1] > 65 and recent_peaks[1][1] > 65):
            return False
            
        # 条件2: 两个峰点之间有一个明显的谷值
        mid_point = (recent_peaks[0][0] + recent_peaks[1][0]) // 2
        mid_range = rsi_series.iloc[recent_peaks[0][0]:recent_peaks[1][0]]
        
        if len(mid_range) == 0 or mid_range.min() > 60:
            return False
            
        # 条件3: 第二个峰点不高于第一个峰点 (可以相等)
        if recent_peaks[1][1] > recent_peaks[0][1]:
            return False
            
        return True
    
    def _detect_failure_swing(self, rsi_series: pd.Series, direction: str) -> bool:
        """
        检测失败摆动形态
        
        Args:
            rsi_series: RSI序列
            direction: 方向 ("bullish" 或 "bearish")
            
        Returns:
            bool: 是否形成失败摆动形态
        """
        if len(rsi_series) < 10:
            return False
            
        if direction == "bullish":
            # 看涨失败摆动: RSI跌破前低后反弹不创新低
            # 寻找局部最低点
            valleys = []
            for i in range(1, len(rsi_series) - 1):
                if rsi_series.iloc[i] < rsi_series.iloc[i-1] and rsi_series.iloc[i] < rsi_series.iloc[i+1]:
                    valleys.append((i, rsi_series.iloc[i]))
                    
            if len(valleys) < 2:
                return False
                
            # 检查最近的两个谷点
            recent_valleys = sorted(valleys, key=lambda x: x[0])[-2:]
            
            # 第一个谷点应该在超卖区域
            if recent_valleys[0][1] > 35:
                return False
                
            # 第二个谷点不低于第一个谷点
            if recent_valleys[1][1] < recent_valleys[0][1]:
                return False
                
            # 两个谷点之间有明显反弹
            mid_range = rsi_series.iloc[recent_valleys[0][0]:recent_valleys[1][0]]
            if len(mid_range) == 0 or mid_range.max() < recent_valleys[0][1] + 10:
                return False
                
            return True
            
        elif direction == "bearish":
            # 看跌失败摆动: RSI突破前高后回落不创新高
            # 寻找局部最高点
            peaks = []
            for i in range(1, len(rsi_series) - 1):
                if rsi_series.iloc[i] > rsi_series.iloc[i-1] and rsi_series.iloc[i] > rsi_series.iloc[i+1]:
                    peaks.append((i, rsi_series.iloc[i]))
                    
            if len(peaks) < 2:
                return False
                
            # 检查最近的两个峰点
            recent_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
            
            # 第一个峰点应该在超买区域
            if recent_peaks[0][1] < 65:
                return False
                
            # 第二个峰点不高于第一个峰点
            if recent_peaks[1][1] > recent_peaks[0][1]:
                return False
                
            # 两个峰点之间有明显回落
            mid_range = rsi_series.iloc[recent_peaks[0][0]:recent_peaks[1][0]]
            if len(mid_range) == 0 or mid_range.min() > recent_peaks[0][1] - 10:
                return False
                
            return True
            
        return False
    
    def _detect_divergence(self, price: pd.Series, rsi: pd.Series) -> Optional[str]:
        """
        检测RSI背离
        
        Args:
            price: 价格序列
            rsi: RSI序列
            
        Returns:
            Optional[str]: 背离类型 ("bullish", "bearish" 或 None)
        """
        if len(price) < 30 or len(rsi) < 30:
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
        
        # 寻找RSI高点和低点
        rsi_highs = []
        rsi_lows = []
        
        for i in range(5, len(rsi) - 5):
            # RSI高点
            if rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i] > rsi.iloc[i+1] and \
               rsi.iloc[i] == rsi.iloc[i-5:i+6].max():
                rsi_highs.append((i, rsi.iloc[i]))
            # RSI低点
            if rsi.iloc[i] < rsi.iloc[i-1] and rsi.iloc[i] < rsi.iloc[i+1] and \
               rsi.iloc[i] == rsi.iloc[i-5:i+6].min():
                rsi_lows.append((i, rsi.iloc[i]))
        
        # 检查顶背离（价格创新高，RSI未创新高）
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            recent_price_highs = sorted(price_highs, key=lambda x: x[0])[-2:]
            recent_rsi_highs = sorted(rsi_highs, key=lambda x: x[0])[-2:]
            
            # 确保最近的高点在相似的时间范围内
            if abs(recent_price_highs[1][0] - recent_rsi_highs[1][0]) <= 3:
                # 价格创新高但RSI未创新高
                if recent_price_highs[1][1] > recent_price_highs[0][1] and \
                   recent_rsi_highs[1][1] < recent_rsi_highs[0][1]:
                    return "bearish"
        
        # 检查底背离（价格创新低，RSI未创新低）
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            recent_price_lows = sorted(price_lows, key=lambda x: x[0])[-2:]
            recent_rsi_lows = sorted(rsi_lows, key=lambda x: x[0])[-2:]
            
            # 确保最近的低点在相似的时间范围内
            if abs(recent_price_lows[1][0] - recent_rsi_lows[1][0]) <= 3:
                # 价格创新低但RSI未创新低
                if recent_price_lows[1][1] < recent_price_lows[0][1] and \
                   recent_rsi_lows[1][1] > recent_rsi_lows[0][1]:
                    return "bullish"
        
        return None
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型RSI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算RSI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        rsi = self._result[f"rsi_{self.period}"]
        oversold = self._result["oversold_threshold"]
        overbought = self._result["overbought_threshold"]
        
        # 1. 基础RSI评分 - 超买超卖区域
        # RSI在超卖区域，看涨
        oversold_score = ((oversold - rsi) / oversold * 30).clip(0, 30)
        score += (rsi < oversold) * oversold_score
        
        # RSI在超买区域，看跌
        overbought_score = ((rsi - overbought) / (100 - overbought) * 30).clip(0, 30)
        score -= (rsi > overbought) * overbought_score
        
        # 2. RSI穿越信号评分
        # RSI上穿超卖阈值，买入信号
        score += self.crossover(rsi, oversold) * 20
        
        # RSI下穿超买阈值，卖出信号
        score -= self.crossunder(rsi, overbought) * 20
        
        # 3. RSI中线穿越评分
        # RSI上穿50，看涨
        score += self.crossover(rsi, 50) * 15
        
        # RSI下穿50，看跌
        score -= self.crossunder(rsi, 50) * 15
        
        # 4. 多周期RSI一致性评分
        multi_period_score = self._calculate_multi_period_consistency()
        score += multi_period_score
        
        # 5. RSI形态评分
        pattern_score = self._calculate_pattern_score(data)
        score += pattern_score
        
        # 6. RSI动量评分
        momentum = self._result["rsi_momentum"]
        
        # RSI动量为正，加分
        score += (momentum > 0.5) * 10
        
        # RSI动量为负，减分
        score -= (momentum < -0.5) * 10
        
        # 7. 背离评分
        divergence = [self._detect_divergence(data["close"].iloc[:i+1], rsi.iloc[:i+1]) 
                     for i in range(len(data))]
        
        # 正背离（看涨）
        score += pd.Series([25 if d == "bullish" else 0 for d in divergence], index=score.index)
        
        # 负背离（看跌）
        score -= pd.Series([25 if d == "bearish" else 0 for d in divergence], index=score.index)
        
        # 限制得分范围
        return np.clip(score, 0, 100)
    
    def _calculate_multi_period_consistency(self) -> pd.Series:
        """
        计算多周期RSI一致性评分
        
        Returns:
            pd.Series: 多周期一致性评分
        """
        if not self.has_result():
            return pd.Series(0.0, index=self._result.index)
        
        score = pd.Series(0.0, index=self._result.index)
        
        # 获取所有计算的RSI周期
        rsi_columns = [col for col in self._result.columns if col.startswith("rsi_")]
        
        if len(rsi_columns) < 2:
            return score
        
        # 计算每个时间点的趋势一致性
        for i in range(len(self._result)):
            bullish_count = 0
            bearish_count = 0
            
            for col in rsi_columns:
                if i < 5:  # 数据不足，跳过
                    continue
                    
                rsi_val = self._result[col].iloc[i]
                rsi_prev = self._result[col].iloc[i-5]
                
                if rsi_val > rsi_prev:  # RSI上升
                    bullish_count += 1
                elif rsi_val < rsi_prev:  # RSI下降
                    bearish_count += 1
            
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
    
    def _calculate_pattern_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算RSI形态评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 形态评分
        """
        score = pd.Series(0.0, index=data.index)
        
        if not self.has_result() or len(data) < 20:
            return score
            
        rsi = self._result[f"rsi_{self.period}"]
        
        # 逐点检查形态
        for i in range(20, len(data)):
            # W底形态 - 强烈看涨
            if self._detect_w_bottom(rsi.iloc[i-20:i+1]):
                score.iloc[i:i+10] = 25  # 影响后续10个周期
                
            # M顶形态 - 强烈看跌
            elif self._detect_m_top(rsi.iloc[i-20:i+1]):
                score.iloc[i:i+10] = -25  # 影响后续10个周期
                
            # 看涨失败摆动
            elif i >= 15 and self._detect_failure_swing(rsi.iloc[i-15:i+1], "bullish"):
                score.iloc[i:i+5] = 20  # 影响后续5个周期
                
            # 看跌失败摆动
            elif i >= 15 and self._detect_failure_swing(rsi.iloc[i-15:i+1], "bearish"):
                score.iloc[i:i+5] = -20  # 影响后续5个周期
        
        return score
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成RSI交易信号
        
        Args:
            data: 输入数据
            *args, **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 信号DataFrame
        """
        # 确保已计算RSI
        if not self.has_result():
            self.calculate(data, *args, **kwargs)
            
        result = self._result.copy()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=result.index)
        signals['rsi'] = result[f"rsi_{self.period}"]
        signals['oversold'] = result["oversold_threshold"]
        signals['overbought'] = result["overbought_threshold"]
        
        # 计算RSI评分
        rsi_score = self.calculate_raw_score(data)
        signals['score'] = rsi_score
        
        # 生成买入信号
        buy_signal = (
            (signals['rsi'] < signals['oversold']) |  # RSI进入超卖区
            (self.crossover(signals['rsi'], signals['oversold'])) |  # RSI上穿超卖阈值
            (signals['score'] > 70)  # 评分高于70
        )
        
        # 生成卖出信号
        sell_signal = (
            (signals['rsi'] > signals['overbought']) |  # RSI进入超买区
            (self.crossunder(signals['rsi'], signals['overbought'])) |  # RSI下穿超买阈值
            (signals['score'] < 30)  # 评分低于30
        )
        
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
                # 根据评分、RSI动量和多周期一致性确定信号强度
                score = signals['score'].iloc[i]
                
                if score > 85:
                    strength.iloc[i] = SignalStrength.VERY_STRONG.value
                elif score > 75:
                    strength.iloc[i] = SignalStrength.STRONG.value
                elif score > 65:
                    strength.iloc[i] = SignalStrength.MODERATE.value
                else:
                    strength.iloc[i] = SignalStrength.WEAK.value
            
            elif signals['sell_signal'].iloc[i]:
                # 根据评分、RSI动量和多周期一致性确定信号强度
                score = signals['score'].iloc[i]
                
                if score < 15:
                    strength.iloc[i] = SignalStrength.VERY_STRONG_NEGATIVE.value
                elif score < 25:
                    strength.iloc[i] = SignalStrength.STRONG_NEGATIVE.value
                elif score < 35:
                    strength.iloc[i] = SignalStrength.MODERATE_NEGATIVE.value
                else:
                    strength.iloc[i] = SignalStrength.WEAK_NEGATIVE.value
        
        return strength 