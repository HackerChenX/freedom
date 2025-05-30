"""
增强型资金流向指标(EnhancedMFI)模块

实现增强型MFI指标计算，提供自适应阈值、异常成交量滤波、价格结构协同分析和市场环境适应功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator, MarketEnvironment
from indicators.mfi import MFI
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedMFI(MFI):
    """
    增强型资金流向指标(Enhanced Money Flow Index)
    
    具有以下增强特性:
    1. 动态阈值调整：根据市场波动率动态调整MFI超买超卖阈值
    2. 异常成交量滤波：识别并平滑异常成交量数据，减少对MFI计算的干扰
    3. 价格结构协同分析：分析MFI在价格关键结构点的表现
    4. 市场环境适应：根据不同市场环境动态调整MFI的解释框架
    """
    
    def __init__(self, 
                 period: int = 14,
                 volatility_lookback: int = 20,
                 enable_volume_filter: bool = True,
                 volume_filter_threshold: float = 3.0):
        """
        初始化增强型MFI指标
        
        Args:
            period: 计算周期，默认为14日
            volatility_lookback: 波动率计算回溯期，默认为20
            enable_volume_filter: 是否启用成交量过滤，默认为True
            volume_filter_threshold: 成交量过滤阈值，默认为3.0倍标准差
        """
        super().__init__(period=period)
        self.name = "EnhancedMFI"
        self.description = "增强型资金流向指标，提供自适应阈值、异常成交量滤波和市场环境适应功能"
        self.volatility_lookback = volatility_lookback
        self.enable_volume_filter = enable_volume_filter
        self.volume_filter_threshold = volume_filter_threshold
        self.market_environment = "normal"
        
        # 动态阈值默认值
        self._dynamic_overbought = 80
        self._dynamic_oversold = 20
        
        # 存储价格结构关键点
        self._price_key_levels = {}
    
    def set_market_environment(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment
    
    def calculate(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        计算增强型MFI指标
        
        Args:
            data: 输入数据，包含OHLCV数据
            
        Returns:
            pd.DataFrame: 计算结果，包含MFI及其相关指标
        """
        # 确保数据包含必需的列
        self.ensure_columns(data, ["high", "low", "close", "volume"])
        
        # 处理异常成交量
        processed_data = data.copy()
        if self.enable_volume_filter:
            processed_data["volume"] = self._smooth_abnormal_volume(data["volume"])
        
        # 使用父类方法计算MFI
        result = super().calculate(processed_data, *args, **kwargs)
        
        # 计算动态阈值
        self._calculate_dynamic_thresholds(data)
        
        # 添加动态阈值到结果
        result["mfi_overbought"] = self._dynamic_overbought
        result["mfi_oversold"] = self._dynamic_oversold
        
        # 计算MFI与价格的相对变化率
        result["mfi_price_ratio"] = self._calculate_mfi_price_ratio(data["close"], result["mfi"])
        
        # 分析价格结构关键点
        self._analyze_price_structure(data)
        
        # 计算MFI动态特性
        result["mfi_momentum"] = result["mfi"] - result["mfi"].shift(3)
        result["mfi_slope"] = result["mfi"].diff(3)
        result["mfi_accel"] = result["mfi_slope"].diff()
        
        # 根据市场环境调整MFI解释
        result["mfi_adjusted"] = self._adjust_mfi_by_environment(result["mfi"])
        
        # 存储结果
        self._result = result
        
        return result
    
    def _smooth_abnormal_volume(self, volume: pd.Series) -> pd.Series:
        """
        识别并平滑异常成交量数据
        
        Args:
            volume: 成交量序列
            
        Returns:
            pd.Series: 平滑后的成交量序列
        """
        # 计算成交量的移动平均和标准差
        vol_ma = volume.rolling(window=20).mean()
        vol_std = volume.rolling(window=20).std()
        
        # 初始化结果为原始成交量
        smoothed_volume = volume.copy()
        
        # 填充初始的NA值
        vol_ma = vol_ma.fillna(volume)
        vol_std = vol_std.fillna(volume * 0.1)  # 假设初始标准差为10%
        
        # 识别异常成交量
        for i in range(len(volume)):
            if volume.iloc[i] > vol_ma.iloc[i] + self.volume_filter_threshold * vol_std.iloc[i]:
                # 异常高成交量，使用上限值替代
                smoothed_volume.iloc[i] = vol_ma.iloc[i] + self.volume_filter_threshold * vol_std.iloc[i]
        
        return smoothed_volume
    
    def _calculate_dynamic_thresholds(self, data: pd.DataFrame) -> None:
        """
        根据市场波动率动态调整MFI超买超卖阈值
        
        Args:
            data: 包含价格数据的DataFrame
        """
        # 计算价格波动率
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        # 如果波动率数据不足，则使用默认阈值
        if pd.isna(volatility):
            self._dynamic_overbought = 80
            self._dynamic_oversold = 20
            return
        
        # 计算历史波动率
        historical_volatility = returns.rolling(window=self.volatility_lookback*3).std().iloc[-1]
        
        # 如果历史波动率数据不足，则使用默认阈值
        if pd.isna(historical_volatility) or historical_volatility == 0:
            self._dynamic_overbought = 80
            self._dynamic_oversold = 20
            return
        
        # 计算相对波动率
        relative_volatility = volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # 根据相对波动率调整阈值
        if relative_volatility > 1.5:  # 高波动市场
            # 放宽阈值
            self._dynamic_overbought = 85
            self._dynamic_oversold = 15
        elif relative_volatility < 0.7:  # 低波动市场
            # 收紧阈值
            self._dynamic_overbought = 75
            self._dynamic_oversold = 25
        else:  # 正常波动市场
            # 使用默认阈值
            self._dynamic_overbought = 80
            self._dynamic_oversold = 20
        
        # 根据市场环境进一步调整
        if self.market_environment == 'bull_market':
            # 牛市中提高超买阈值，降低超卖阈值
            self._dynamic_overbought += 5
            self._dynamic_oversold += 5
        elif self.market_environment == 'bear_market':
            # 熊市中降低超买阈值，提高超卖阈值
            self._dynamic_overbought -= 5
            self._dynamic_oversold -= 5
        elif self.market_environment == 'volatile_market':
            # 高波动市场进一步放宽阈值
            self._dynamic_overbought += 3
            self._dynamic_oversold -= 3
        
        logger.debug(f"调整MFI阈值: 超买={self._dynamic_overbought}, 超卖={self._dynamic_oversold}, "
                    f"相对波动率={relative_volatility:.2f}, 市场环境={self.market_environment}")
    
    def _calculate_mfi_price_ratio(self, price: pd.Series, mfi: pd.Series) -> pd.Series:
        """
        计算MFI与价格的相对变化率
        
        Args:
            price: 价格序列
            mfi: MFI序列
            
        Returns:
            pd.Series: MFI与价格的相对变化率
        """
        # 计算价格变化率
        price_change = price.pct_change(5)
        
        # 计算MFI变化率
        mfi_change = mfi.diff(5) / 100
        
        # 计算相对变化率
        ratio = pd.Series(np.nan, index=price.index)
        mask = (price_change != 0) & (~pd.isna(price_change)) & (~pd.isna(mfi_change))
        ratio[mask] = mfi_change[mask] / price_change[mask]
        
        return ratio
    
    def _analyze_price_structure(self, data: pd.DataFrame) -> None:
        """
        分析价格结构关键点
        
        Args:
            data: 包含OHLCV数据的DataFrame
        """
        # 查找最近的支撑位和阻力位
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # 使用简单的方法找出最近的高点和低点
        window = 20
        
        # 寻找局部高点
        local_highs = []
        for i in range(window, len(high) - window):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                local_highs.append((i, high.iloc[i]))
        
        # 寻找局部低点
        local_lows = []
        for i in range(window, len(low) - window):
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                local_lows.append((i, low.iloc[i]))
        
        # 保存到字典
        self._price_key_levels = {
            "local_highs": local_highs[-3:] if len(local_highs) > 0 else [],  # 最近的3个高点
            "local_lows": local_lows[-3:] if len(local_lows) > 0 else [],     # 最近的3个低点
        }
    
    def _adjust_mfi_by_environment(self, mfi: pd.Series) -> pd.Series:
        """
        根据市场环境调整MFI解释
        
        Args:
            mfi: MFI序列
            
        Returns:
            pd.Series: 调整后的MFI序列
        """
        adjusted_mfi = mfi.copy()
        
        if self.market_environment == 'bull_market':
            # 牛市中MFI表现更强，上涨更容易
            adjusted_mfi = adjusted_mfi * 1.1
            adjusted_mfi = adjusted_mfi.clip(0, 100)
        elif self.market_environment == 'bear_market':
            # 熊市中MFI表现更弱，下跌更容易
            adjusted_mfi = adjusted_mfi * 0.9
        elif self.market_environment == 'volatile_market':
            # 高波动市场MFI波动更大，需要平滑处理
            adjusted_mfi = adjusted_mfi.rolling(window=3).mean()
        
        return adjusted_mfi
    
    def analyze_price_structure_synergy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        分析MFI在价格关键结构点的表现
        
        Args:
            data: 包含OHLCV和MFI数据的DataFrame
            
        Returns:
            pd.DataFrame: 价格结构协同分析结果
        """
        if not self.has_result():
            self.calculate(data)
        
        # 初始化结果DataFrame
        result = pd.DataFrame(index=data.index)
        result["synergy_score"] = 0
        
        # 获取MFI数据
        mfi = self._result["mfi"]
        
        # 分析局部高点处的MFI表现
        for idx, price in self._price_key_levels["local_highs"]:
            if 0 <= idx < len(data):
                # 价格创新高但MFI未创新高（顶背离）
                if idx > 20:
                    mfi_window = mfi.iloc[idx-20:idx+1]
                    if mfi.iloc[idx] < mfi_window.max() * 0.95:  # MFI低于区间最大值的95%
                        result.iloc[idx, result.columns.get_loc("synergy_score")] -= 30
                        logger.debug(f"检测到价格高点({price:.2f})处的MFI顶背离，位置={idx}")
        
        # 分析局部低点处的MFI表现
        for idx, price in self._price_key_levels["local_lows"]:
            if 0 <= idx < len(data):
                # 价格创新低但MFI未创新低（底背离）
                if idx > 20:
                    mfi_window = mfi.iloc[idx-20:idx+1]
                    if mfi.iloc[idx] > mfi_window.min() * 1.05:  # MFI高于区间最小值的105%
                        result.iloc[idx, result.columns.get_loc("synergy_score")] += 30
                        logger.debug(f"检测到价格低点({price:.2f})处的MFI底背离，位置={idx}")
        
        # 平滑评分
        result["synergy_score"] = result["synergy_score"].rolling(window=5, min_periods=1).mean()
        
        return result
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型MFI原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算MFI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        # 1. 使用动态阈值的超买超卖评分
        ob_os_score = self._calculate_mfi_dynamic_threshold_score()
        score += ob_os_score
        
        # 2. MFI背离评分
        divergence_score = self._calculate_mfi_divergence_score(data)
        score += divergence_score
        
        # 3. 价格结构协同评分
        structure_score = self.analyze_price_structure_synergy(data)["synergy_score"]
        score += structure_score
        
        # 4. MFI趋势评分
        trend_score = self._calculate_mfi_trend_score()
        score += trend_score
        
        # 5. 市场环境调整
        score = self._adjust_score_by_environment(score)
        
        return np.clip(score, 0, 100)
    
    def _calculate_mfi_dynamic_threshold_score(self) -> pd.Series:
        """
        使用动态阈值计算MFI超买超卖评分
        
        Returns:
            pd.Series: 超买超卖评分
        """
        score = pd.Series(0.0, index=self._result.index)
        
        mfi = self._result["mfi"]
        
        # 使用动态阈值
        overbought = self._dynamic_overbought
        oversold = self._dynamic_oversold
        
        # 超买区域 (得分随MFI增加而降低)
        overbought_score = -1 * np.maximum(0, (mfi - overbought)) * 1.0
        score += overbought_score
        
        # 超卖区域 (得分随MFI降低而增加)
        oversold_score = np.maximum(0, (oversold - mfi)) * 1.0
        score += oversold_score
        
        # 中性区域上方 (小幅加分)
        neutral_high = (mfi > 50) & (mfi < overbought)
        score.loc[neutral_high] += (mfi.loc[neutral_high] - 50) * 0.3
        
        # 中性区域下方 (小幅减分)
        neutral_low = (mfi < 50) & (mfi > oversold)
        score.loc[neutral_low] -= (50 - mfi.loc[neutral_low]) * 0.3
        
        return score
    
    def _adjust_score_by_environment(self, score: pd.Series) -> pd.Series:
        """
        根据市场环境调整评分
        
        Args:
            score: 原始评分序列
            
        Returns:
            pd.Series: 调整后的评分序列
        """
        adjusted_score = score.copy()
        
        if self.market_environment == 'bull_market':
            # 牛市中看涨信号更重要
            above_50 = score > 50
            adjusted_score[above_50] = 50 + (score[above_50] - 50) * 1.2  # 增强看涨信号
            adjusted_score[~above_50] = 50 - (50 - score[~above_50]) * 0.8  # 减弱看跌信号
        
        elif self.market_environment == 'bear_market':
            # 熊市中看跌信号更重要
            below_50 = score < 50
            adjusted_score[below_50] = 50 - (50 - score[below_50]) * 1.2  # 增强看跌信号
            adjusted_score[~below_50] = 50 + (score[~below_50] - 50) * 0.8  # 减弱看涨信号
        
        elif self.market_environment == 'volatile_market':
            # 高波动市场需要更强的信号确认
            adjusted_score = 50 + (score - 50) * 0.7  # 减弱所有信号
            # 但极端信号保持不变
            extreme_signal = (score < 20) | (score > 80)
            adjusted_score[extreme_signal] = score[extreme_signal]
        
        return adjusted_score
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别MFI技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算MFI
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        # 1. 检测MFI超买超卖形态
        ob_os_patterns = self._detect_mfi_dynamic_threshold_patterns()
        patterns.extend(ob_os_patterns)
        
        # 2. 检测MFI背离形态
        divergence_patterns = self._detect_mfi_divergence_patterns(data)
        patterns.extend(divergence_patterns)
        
        # 3. 检测MFI趋势形态
        trend_patterns = self._detect_mfi_trend_patterns()
        patterns.extend(trend_patterns)
        
        # 4. 检测MFI结构形态
        structure_patterns = self._detect_mfi_structure_patterns()
        patterns.extend(structure_patterns)
        
        return patterns
    
    def _detect_mfi_dynamic_threshold_patterns(self) -> List[str]:
        """
        使用动态阈值检测MFI超买超卖形态
        
        Returns:
            List[str]: 超买超卖形态列表
        """
        patterns = []
        
        mfi = self._result["mfi"]
        
        # 获取最近的MFI值
        if len(mfi) < 5:
            return patterns
        
        recent_mfi = mfi.iloc[-5:]
        current_mfi = recent_mfi.iloc[-1]
        
        # 使用动态阈值
        overbought = self._dynamic_overbought
        oversold = self._dynamic_oversold
        
        # 超买区形态
        if current_mfi > overbought:
            patterns.append(f"MFI超买区域(>{overbought:.0f})")
            
            # 检测是否继续走高
            if current_mfi > recent_mfi.iloc[-2] and recent_mfi.iloc[-2] > recent_mfi.iloc[-3]:
                patterns.append("MFI超买区域继续走高")
            # 检测是否从超买区回落
            elif current_mfi < recent_mfi.iloc[-2]:
                patterns.append("MFI从超买区回落")
        
        # 超卖区形态
        elif current_mfi < oversold:
            patterns.append(f"MFI超卖区域(<{oversold:.0f})")
            
            # 检测是否继续走低
            if current_mfi < recent_mfi.iloc[-2] and recent_mfi.iloc[-2] < recent_mfi.iloc[-3]:
                patterns.append("MFI超卖区域继续走低")
            # 检测是否从超卖区回升
            elif current_mfi > recent_mfi.iloc[-2]:
                patterns.append("MFI从超卖区回升")
        
        # 中性区域形态
        else:
            # 中性区域偏多
            if current_mfi > 50:
                patterns.append("MFI中性区域偏多")
            # 中性区域偏空
            else:
                patterns.append("MFI中性区域偏空")
            
            # 检测穿越动态阈值
            if current_mfi < overbought and recent_mfi.iloc[-2] >= overbought:
                patterns.append(f"MFI下穿超买阈值({overbought:.0f})")
            elif current_mfi > oversold and recent_mfi.iloc[-2] <= oversold:
                patterns.append(f"MFI上穿超卖阈值({oversold:.0f})")
        
        return patterns
    
    def _detect_mfi_structure_patterns(self) -> List[str]:
        """
        检测MFI结构形态
        
        Returns:
            List[str]: 结构形态列表
        """
        patterns = []
        
        # 获取局部高点和低点数据
        local_highs = self._price_key_levels.get("local_highs", [])
        local_lows = self._price_key_levels.get("local_lows", [])
        
        if not local_highs and not local_lows:
            return patterns
        
        # 检测最近的价格高点和低点
        if local_highs:
            patterns.append(f"价格关键阻力位: {local_highs[-1][1]:.2f}")
        
        if local_lows:
            patterns.append(f"价格关键支撑位: {local_lows[-1][1]:.2f}")
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        生成增强型MFI指标标准化交易信号
        
        Args:
            data: 输入数据，包含OHLCV数据
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 信号结果DataFrame，包含标准化信号
        """
        # 确保已计算增强型MFI指标
        if not self.has_result():
            self.calculate(data)
        
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
        signals['market_env'] = self.market_environment
        signals['volume_confirmation'] = False
        
        # 计算评分
        score = self.calculate_raw_score(data, **kwargs)
        signals['score'] = score
        
        # 检测形态
        patterns = self.identify_patterns(data, **kwargs)
        
        # 获取MFI数据
        mfi = self._result['mfi']
        
        # 1. MFI从超卖区上穿动态阈值，买入信号
        mfi_cross_oversold = (mfi > self._dynamic_oversold) & (mfi.shift(1) <= self._dynamic_oversold)
        signals.loc[mfi_cross_oversold, 'buy_signal'] = True
        signals.loc[mfi_cross_oversold, 'neutral_signal'] = False
        signals.loc[mfi_cross_oversold, 'trend'] = 1
        signals.loc[mfi_cross_oversold, 'signal_type'] = 'MFI超卖反弹'
        signals.loc[mfi_cross_oversold, 'signal_desc'] = f'MFI从超卖区上穿{self._dynamic_oversold}，买入信号'
        signals.loc[mfi_cross_oversold, 'confidence'] = 70.0
        signals.loc[mfi_cross_oversold, 'position_size'] = 0.4
        signals.loc[mfi_cross_oversold, 'risk_level'] = '中'
        
        # 2. MFI从超买区下穿动态阈值，卖出信号
        mfi_cross_overbought = (mfi < self._dynamic_overbought) & (mfi.shift(1) >= self._dynamic_overbought)
        signals.loc[mfi_cross_overbought, 'sell_signal'] = True
        signals.loc[mfi_cross_overbought, 'neutral_signal'] = False
        signals.loc[mfi_cross_overbought, 'trend'] = -1
        signals.loc[mfi_cross_overbought, 'signal_type'] = 'MFI超买回落'
        signals.loc[mfi_cross_overbought, 'signal_desc'] = f'MFI从超买区下穿{self._dynamic_overbought}，卖出信号'
        signals.loc[mfi_cross_overbought, 'confidence'] = 70.0
        signals.loc[mfi_cross_overbought, 'position_size'] = 0.4
        signals.loc[mfi_cross_overbought, 'risk_level'] = '中'
        
        # 3. MFI底背离，买入信号
        mfi_bullish_divergence = (self._result['mfi_price_ratio'] > 2) & (mfi < 30)
        signals.loc[mfi_bullish_divergence, 'buy_signal'] = True
        signals.loc[mfi_bullish_divergence, 'neutral_signal'] = False
        signals.loc[mfi_bullish_divergence, 'trend'] = 1
        signals.loc[mfi_bullish_divergence, 'signal_type'] = 'MFI底背离'
        signals.loc[mfi_bullish_divergence, 'signal_desc'] = 'MFI与价格形成底背离，买入信号'
        signals.loc[mfi_bullish_divergence, 'confidence'] = 80.0
        signals.loc[mfi_bullish_divergence, 'position_size'] = 0.6
        signals.loc[mfi_bullish_divergence, 'risk_level'] = '低'
        
        # 4. MFI顶背离，卖出信号
        mfi_bearish_divergence = (self._result['mfi_price_ratio'] < -2) & (mfi > 70)
        signals.loc[mfi_bearish_divergence, 'sell_signal'] = True
        signals.loc[mfi_bearish_divergence, 'neutral_signal'] = False
        signals.loc[mfi_bearish_divergence, 'trend'] = -1
        signals.loc[mfi_bearish_divergence, 'signal_type'] = 'MFI顶背离'
        signals.loc[mfi_bearish_divergence, 'signal_desc'] = 'MFI与价格形成顶背离，卖出信号'
        signals.loc[mfi_bearish_divergence, 'confidence'] = 80.0
        signals.loc[mfi_bearish_divergence, 'position_size'] = 0.6
        signals.loc[mfi_bearish_divergence, 'risk_level'] = '低'
        
        # 5. MFI上穿50中线，买入信号
        mfi_cross_50_up = (mfi > 50) & (mfi.shift(1) <= 50)
        signals.loc[mfi_cross_50_up, 'buy_signal'] = True
        signals.loc[mfi_cross_50_up, 'neutral_signal'] = False
        signals.loc[mfi_cross_50_up, 'trend'] = 1
        signals.loc[mfi_cross_50_up, 'signal_type'] = 'MFI上穿中线'
        signals.loc[mfi_cross_50_up, 'signal_desc'] = 'MFI上穿50中线，买入信号'
        signals.loc[mfi_cross_50_up, 'confidence'] = 60.0
        signals.loc[mfi_cross_50_up, 'position_size'] = 0.3
        signals.loc[mfi_cross_50_up, 'risk_level'] = '中'
        
        # 6. MFI下穿50中线，卖出信号
        mfi_cross_50_down = (mfi < 50) & (mfi.shift(1) >= 50)
        signals.loc[mfi_cross_50_down, 'sell_signal'] = True
        signals.loc[mfi_cross_50_down, 'neutral_signal'] = False
        signals.loc[mfi_cross_50_down, 'trend'] = -1
        signals.loc[mfi_cross_50_down, 'signal_type'] = 'MFI下穿中线'
        signals.loc[mfi_cross_50_down, 'signal_desc'] = 'MFI下穿50中线，卖出信号'
        signals.loc[mfi_cross_50_down, 'confidence'] = 60.0
        signals.loc[mfi_cross_50_down, 'position_size'] = 0.3
        signals.loc[mfi_cross_50_down, 'risk_level'] = '中'
        
        # 7. 根据得分产生强弱信号
        strong_buy = score > 80
        signals.loc[strong_buy, 'buy_signal'] = True
        signals.loc[strong_buy, 'neutral_signal'] = False
        signals.loc[strong_buy, 'trend'] = 1
        signals.loc[strong_buy, 'signal_type'] = 'MFI强烈买入'
        signals.loc[strong_buy, 'signal_desc'] = 'MFI综合评分超过80，强烈买入信号'
        signals.loc[strong_buy, 'confidence'] = 85.0
        signals.loc[strong_buy, 'position_size'] = 0.7
        signals.loc[strong_buy, 'risk_level'] = '低'
        
        strong_sell = score < 20
        signals.loc[strong_sell, 'sell_signal'] = True
        signals.loc[strong_sell, 'neutral_signal'] = False
        signals.loc[strong_sell, 'trend'] = -1
        signals.loc[strong_sell, 'signal_type'] = 'MFI强烈卖出'
        signals.loc[strong_sell, 'signal_desc'] = 'MFI综合评分低于20，强烈卖出信号'
        signals.loc[strong_sell, 'confidence'] = 85.0
        signals.loc[strong_sell, 'position_size'] = 0.7
        signals.loc[strong_sell, 'risk_level'] = '低'
        
        # 根据市场环境调整信号
        if self.market_environment == 'bull_market':
            # 牛市中提高买入信号的信心度和仓位
            buy_signals = signals['buy_signal']
            signals.loc[buy_signals, 'confidence'] = signals.loc[buy_signals, 'confidence'] * 1.1
            signals.loc[buy_signals, 'confidence'] = signals.loc[buy_signals, 'confidence'].clip(0, 100)
            signals.loc[buy_signals, 'position_size'] = signals.loc[buy_signals, 'position_size'] * 1.2
            signals.loc[buy_signals, 'position_size'] = signals.loc[buy_signals, 'position_size'].clip(0, 1)
        
        elif self.market_environment == 'bear_market':
            # 熊市中提高卖出信号的信心度和仓位
            sell_signals = signals['sell_signal']
            signals.loc[sell_signals, 'confidence'] = signals.loc[sell_signals, 'confidence'] * 1.1
            signals.loc[sell_signals, 'confidence'] = signals.loc[sell_signals, 'confidence'].clip(0, 100)
            signals.loc[sell_signals, 'position_size'] = signals.loc[sell_signals, 'position_size'] * 1.2
            signals.loc[sell_signals, 'position_size'] = signals.loc[sell_signals, 'position_size'].clip(0, 1)
        
        return signals 