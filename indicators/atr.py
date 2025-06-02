#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ATR - 平均真实波幅

波动性指标，衡量价格波动的平均幅度
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from enum import Enum
import warnings

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_STRONG = 2.0      # 极强信号
    STRONG = 1.5           # 强信号 
    MODERATE = 1.0         # 中等信号
    WEAK = 0.5            # 弱信号
    VERY_WEAK = 0.25      # 极弱信号
    
    # 方向性信号强度（带方向）
    STRONG_POSITIVE = 1.5  # 强正向
    STRONG_NEGATIVE = -1.5 # 强负向
    
logger = get_logger(__name__)


class ATR(BaseIndicator):
    """
    平均真实波幅(ATR)
    
    衡量价格波动的幅度，常用于止损设置和波动性分析
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化ATR指标
        
        Args:
            params: 参数字典，可包含：
                - period: ATR计算周期，默认为14
                - high_volatility_threshold: 高波动性阈值倍数，默认为2.0
        """
        super().__init__(name="ATR", description="平均真实波幅指标")
        
        # 设置默认参数
        self.params = {
            "period": 14,
            "high_volatility_threshold": 2.0
        }
        
        # 更新自定义参数
        if params:
            self.params.update(params)
        
        # 注册ATR形态
        self._register_atr_patterns()
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算ATR指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外的参数
            
        Returns:
            添加了ATR指标的DataFrame
        """
        df = data.copy()
        
        # 提取参数
        period = self.params["period"]
        high_volatility = self.params["high_volatility_threshold"]
        
        # 确保数据有足够的长度
        if len(df) < period + 1:
            logger.warning(f"数据长度({len(df)})小于所需的回溯周期({period + 1})，返回原始数据")
            df[f'ATR{period}'] = np.nan
            df[f'TR'] = np.nan
            df[f'high_volatility_{period}'] = False
            return df
        
        # 计算真实波幅(TR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR - TR的period周期平均值
        df[f'ATR{period}'] = df['TR'].rolling(window=period).mean()
        
        # 计算相对ATR (占收盘价的百分比)
        df[f'ATR_pct{period}'] = df[f'ATR{period}'] / df['close'] * 100
        
        # 相对于历史的波动性水平
        avg_atr_pct = df[f'ATR_pct{period}'].rolling(window=period*3).mean()
        df[f'volatility_ratio_{period}'] = df[f'ATR_pct{period}'] / avg_atr_pct
        
        # 标记高波动性
        df[f'high_volatility_{period}'] = df[f'volatility_ratio_{period}'] > high_volatility
        
        # 清理中间计算列
        df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return df
    
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ATR原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 原始评分序列（0-100分）
        """
        # 确保已计算ATR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        score = pd.Series(50.0, index=data.index)  # 基础分50分
        
        atr = self._result[f'ATR{self.params["period"]}']
        atr_percent = self._result[f'ATR_pct{self.params["period"]}']
        
        # 1. ATR相对水平评分
        volatility_score = self._calculate_volatility_score(atr, atr_percent)
        score += volatility_score
        
        # 2. ATR趋势评分
        trend_score = self._calculate_atr_trend_score(atr)
        score += trend_score
        
        # 3. ATR突破评分
        breakout_score = self._calculate_atr_breakout_score(atr)
        score += breakout_score
        
        # 4. ATR收敛发散评分
        convergence_score = self._calculate_atr_convergence_score(atr)
        score += convergence_score
        
        # 5. ATR市场状态评分
        market_state_score = self._calculate_market_state_score(atr_percent)
        score += market_state_score
        
        return np.clip(score, 0, 100)
    
    def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """
        识别ATR技术形态
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            List[str]: 识别出的形态列表
        """
        patterns = []
        
        # 确保已计算ATR
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return patterns
        
        atr = self._result[f'ATR{self.params["period"]}']
        atr_percent = self._result[f'ATR_pct{self.params["period"]}']
        
        # 检查最近的信号
        recent_periods = min(20, len(atr))
        if recent_periods == 0:
            return patterns
        
        recent_atr = atr.tail(recent_periods)
        recent_atr_percent = atr_percent.tail(recent_periods)
        current_atr_percent = recent_atr_percent.iloc[-1]
        
        # 1. 波动性水平形态
        volatility_level = self._classify_volatility_level(recent_atr_percent)
        patterns.append(f"ATR{volatility_level}")
        
        # 2. ATR趋势形态
        atr_trend = self._detect_atr_trend(recent_atr)
        if atr_trend:
            patterns.append(f"ATR{atr_trend}")
        
        # 3. ATR突破形态
        if self._detect_atr_breakout(recent_atr, direction='up'):
            patterns.append("ATR向上突破")
        if self._detect_atr_breakout(recent_atr, direction='down'):
            patterns.append("ATR向下突破")
        
        # 4. ATR收敛发散形态
        convergence_pattern = self._detect_atr_convergence_pattern(recent_atr)
        if convergence_pattern:
            patterns.append(f"ATR{convergence_pattern}")
        
        # 5. 市场状态形态
        market_state = self._detect_market_state(recent_atr_percent)
        if market_state:
            patterns.append(f"市场{market_state}")
        
        # 6. ATR极值形态
        if current_atr_percent > 5:
            patterns.append("ATR极高波动")
        elif current_atr_percent < 1:
            patterns.append("ATR极低波动")
        
        return patterns
    
    def _calculate_volatility_score(self, atr: pd.Series, atr_percent: pd.Series) -> pd.Series:
        """
        计算波动性评分
        
        Args:
            atr: ATR序列
            atr_percent: ATR百分比序列
            
        Returns:
            pd.Series: 波动性评分
        """
        volatility_score = pd.Series(0.0, index=atr.index)
        
        if len(atr_percent) < 20:
            return volatility_score
        
        # 计算ATR百分比的历史分位数
        rolling_quantile_80 = atr_percent.rolling(window=60).quantile(0.8)
        rolling_quantile_20 = atr_percent.rolling(window=60).quantile(0.2)
        rolling_median = atr_percent.rolling(window=60).median()
        
        # 高波动性评分
        high_volatility = atr_percent > rolling_quantile_80
        volatility_score += high_volatility * 15  # 高波动+15分
        
        # 极高波动性评分
        extreme_high_volatility = atr_percent > rolling_quantile_80 * 1.5
        volatility_score += extreme_high_volatility * 20  # 极高波动+20分
        
        # 低波动性评分
        low_volatility = atr_percent < rolling_quantile_20
        volatility_score -= low_volatility * 10  # 低波动-10分
        
        # 极低波动性评分（可能预示变盘）
        extreme_low_volatility = atr_percent < rolling_quantile_20 * 0.5
        volatility_score += extreme_low_volatility * 25  # 极低波动+25分（变盘信号）
        
        return volatility_score
    
    def _calculate_atr_trend_score(self, atr: pd.Series) -> pd.Series:
        """
        计算ATR趋势评分
        
        Args:
            atr: ATR序列
            
        Returns:
            pd.Series: 趋势评分
        """
        trend_score = pd.Series(0.0, index=atr.index)
        
        if len(atr) < 10:
            return trend_score
        
        # 计算ATR的短期和长期趋势
        atr_ma5 = atr.rolling(window=5).mean()
        atr_ma20 = atr.rolling(window=20).mean()
        
        # ATR上升趋势
        atr_rising = atr > atr.shift(5)
        trend_score += atr_rising * 10
        
        # ATR强势上升
        strong_rising = (atr_ma5 > atr_ma20) & (atr > atr_ma5)
        trend_score += strong_rising * 15
        
        # ATR下降趋势
        atr_falling = atr < atr.shift(5)
        trend_score -= atr_falling * 5
        
        return trend_score
    
    def _calculate_atr_breakout_score(self, atr: pd.Series) -> pd.Series:
        """
        计算ATR突破评分
        
        Args:
            atr: ATR序列
            
        Returns:
            pd.Series: 突破评分
        """
        breakout_score = pd.Series(0.0, index=atr.index)
        
        if len(atr) < 20:
            return breakout_score
        
        # 计算ATR的历史高点和低点
        rolling_max = atr.rolling(window=20).max()
        rolling_min = atr.rolling(window=20).min()
        
        # ATR突破历史高点
        atr_breakout_high = atr > rolling_max.shift(1)
        breakout_score += atr_breakout_high * 20
        
        # ATR跌破历史低点
        atr_breakout_low = atr < rolling_min.shift(1)
        breakout_score += atr_breakout_low * 15  # 低波动可能预示变盘
        
        return breakout_score
    
    def _calculate_atr_convergence_score(self, atr: pd.Series) -> pd.Series:
        """
        计算ATR收敛发散评分
        
        Args:
            atr: ATR序列
            
        Returns:
            pd.Series: 收敛发散评分
        """
        convergence_score = pd.Series(0.0, index=atr.index)
        
        if len(atr) < 20:
            return convergence_score
        
        # 计算ATR的波动性
        atr_volatility = atr.rolling(window=10).std()
        atr_mean_volatility = atr_volatility.rolling(window=20).mean()
        
        # ATR收敛（波动性降低）
        convergence = atr_volatility < atr_mean_volatility * 0.7
        convergence_score += convergence * 15  # 收敛可能预示突破
        
        # ATR发散（波动性增加）
        divergence = atr_volatility > atr_mean_volatility * 1.3
        convergence_score += divergence * 10  # 发散表示市场活跃
        
        return convergence_score
    
    def _calculate_market_state_score(self, atr_percent: pd.Series) -> pd.Series:
        """
        计算市场状态评分
        
        Args:
            atr_percent: ATR百分比序列
            
        Returns:
            pd.Series: 市场状态评分
        """
        market_score = pd.Series(0.0, index=atr_percent.index)
        
        if len(atr_percent) < 20:
            return market_score
        
        # 根据ATR百分比判断市场状态
        # 高波动市场（>3%）
        high_volatility_market = atr_percent > 3
        market_score += high_volatility_market * 10
        
        # 中等波动市场（1.5%-3%）
        medium_volatility_market = (atr_percent >= 1.5) & (atr_percent <= 3)
        market_score += medium_volatility_market * 5
        
        # 低波动市场（<1.5%）
        low_volatility_market = atr_percent < 1.5
        market_score += low_volatility_market * 15  # 低波动可能预示变盘
        
        return market_score
    
    def _classify_volatility_level(self, atr_percent: pd.Series) -> str:
        """
        分类波动性水平
        
        Args:
            atr_percent: ATR百分比序列
            
        Returns:
            str: 波动性水平
        """
        current_atr_percent = atr_percent.iloc[-1]
        
        if current_atr_percent > 5:
            return "极高波动"
        elif current_atr_percent > 3:
            return "高波动"
        elif current_atr_percent > 1.5:
            return "中等波动"
        elif current_atr_percent > 1:
            return "低波动"
        else:
            return "极低波动"
    
    def _detect_atr_trend(self, atr: pd.Series) -> Optional[str]:
        """
        检测ATR趋势
        
        Args:
            atr: ATR序列
            
        Returns:
            Optional[str]: 趋势类型或None
        """
        if len(atr) < 10:
            return None
        
        # 计算ATR的趋势
        recent_atr = atr.tail(10)
        
        # 连续上升
        if all(recent_atr.iloc[i] >= recent_atr.iloc[i-1] for i in range(1, len(recent_atr))):
            return "持续上升"
        
        # 连续下降
        if all(recent_atr.iloc[i] <= recent_atr.iloc[i-1] for i in range(1, len(recent_atr))):
            return "持续下降"
        
        # 整体上升趋势
        if recent_atr.iloc[-1] > recent_atr.iloc[0] * 1.1:
            return "上升趋势"
        
        # 整体下降趋势
        if recent_atr.iloc[-1] < recent_atr.iloc[0] * 0.9:
            return "下降趋势"
        
        return "横盘整理"
    
    def _detect_atr_breakout(self, atr: pd.Series, direction: str) -> bool:
        """
        检测ATR突破
        
        Args:
            atr: ATR序列
            direction: 突破方向 ('up' 或 'down')
            
        Returns:
            bool: 是否突破
        """
        if len(atr) < 20:
            return False
        
        current_atr = atr.iloc[-1]
        historical_atr = atr.iloc[:-5]  # 排除最近5天
        
        if direction == 'up':
            # 突破历史高点
            return current_atr > historical_atr.max()
        elif direction == 'down':
            # 跌破历史低点
            return current_atr < historical_atr.min()
        
        return False
    
    def _detect_atr_convergence_pattern(self, atr: pd.Series) -> Optional[str]:
        """
        检测ATR收敛发散形态
        
        Args:
            atr: ATR序列
            
        Returns:
            Optional[str]: 收敛发散类型或None
        """
        if len(atr) < 15:
            return None
        
        # 计算ATR的波动性
        recent_atr = atr.tail(15)
        early_volatility = recent_atr.iloc[:5].std()
        late_volatility = recent_atr.iloc[-5:].std()
        
        # 收敛形态
        if late_volatility < early_volatility * 0.6:
            return "收敛形态"
        
        # 发散形态
        if late_volatility > early_volatility * 1.4:
            return "发散形态"
        
        return None
    
    def _detect_market_state(self, atr_percent: pd.Series) -> Optional[str]:
        """
        检测市场状态
        
        Args:
            atr_percent: ATR百分比序列
            
        Returns:
            Optional[str]: 市场状态或None
        """
        if len(atr_percent) < 10:
            return None
        
        recent_atr_percent = atr_percent.tail(10)
        avg_atr_percent = recent_atr_percent.mean()
        
        if avg_atr_percent > 4:
            return "高度活跃"
        elif avg_atr_percent > 2.5:
            return "活跃"
        elif avg_atr_percent > 1.5:
            return "正常"
        elif avg_atr_percent > 1:
            return "平静"
        else:
            return "极度平静"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含ATR指标的DataFrame
        
        Returns:
            添加了交易信号的DataFrame
        """
        result = df.copy()
        
        # 确保包含ATR列
        if f'ATR{self.params["period"]}' not in result.columns:
            result = self.calculate(df)
        
        # 初始化信号列
        result['volatility_high'] = 0
        result['volatility_low'] = 0
        result['atr_rising'] = 0
        result['atr_falling'] = 0
        
        # 填充NaN值，避免计算问题
        result = result.fillna(0)
        
        # 计算ATR的历史统计特征
        atr = result[f'ATR{self.params["period"]}'].values
        atr_percent = result[f'ATR_pct{self.params["period"]}'].values
        
        # 使用20天窗口计算均值和标准差
        window_size = 20
        for i in range(window_size, len(atr)):
            # 计算过去20天的ATR均值和标准差
            window = atr[i-window_size:i]
            atr_mean = np.mean(window)
            atr_std = np.std(window)
            
            # 如果标准差为0，则设置为一个很小的值以避免除以0
            if atr_std == 0:
                atr_std = 0.0001
                
            # 计算当前ATR在历史分布中的位置 (z-score)
            atr_z_score = (atr[i] - atr_mean) / atr_std
            
            # 高波动性信号
            if atr_z_score > 1.5:
                result.iloc[i, result.columns.get_loc('volatility_high')] = 1
                
            # 低波动性信号
            if atr_z_score < -1.5:
                result.iloc[i, result.columns.get_loc('volatility_low')] = 1
                
            # ATR上升信号
            if i > 4 and all(atr[i-j] > atr[i-j-1] for j in range(4)):
                result.iloc[i, result.columns.get_loc('atr_rising')] = 1
                
            # ATR下降信号
            if i > 4 and all(atr[i-j] < atr[i-j-1] for j in range(4)):
                result.iloc[i, result.columns.get_loc('atr_falling')] = 1
                
        return result
    
    def plot(self, df: pd.DataFrame, result: pd.DataFrame, ax=None):
        """
        绘制ATR指标图表
        
        Args:
            df: 原始数据DataFrame
            result: 计算指标后的DataFrame
            ax: matplotlib轴对象，如果为None则创建新的
        
        Returns:
            matplotlib轴对象
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制ATR
        ax.plot(result[f'ATR{self.params["period"]}'], label=f'ATR({self.params["period"]})')
        
        # 标记高波动性和低波动性区域
        high_volatility = result[result['volatility_high'] == 1].index
        low_volatility = result[result['volatility_low'] == 1].index
        
        ax.scatter(high_volatility, result.loc[high_volatility, f'ATR{self.params["period"]}'], color='red', marker='^', s=100, label='高波动性')
        ax.scatter(low_volatility, result.loc[low_volatility, f'ATR{self.params["period"]}'], color='green', marker='v', s=100, label='低波动性')
        
        ax.set_title(f'平均真实波幅(ATR) - 周期:{self.params["period"]}')
        ax.set_ylabel('ATR值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ATR指标并生成交易信号
        
        Args:
            df: 包含价格数据的DataFrame
        
        Returns:
            包含ATR指标和交易信号的DataFrame
        """
        try:
            result = self.calculate(df)
            result = self.generate_signals(df)
            self._result = result
            return result
        except Exception as e:
            logger.error(f"计算ATR指标时出错: {e}")
            return df
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        识别ATR技术形态
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            List[Dict[str, Any]]: 识别出的形态列表
        """
        from indicators.base_indicator import PatternResult
        
        patterns = []
        
        # 确保已计算ATR
        if not self.has_result():
            self.calculate(data)
        
        if self._result is None or len(self._result) < 5:
            return patterns
        
        # 提取参数
        period = self.params["period"]
        high_volatility = self.params["high_volatility_threshold"]
        
        # 获取ATR数据
        atr = self._result[f'ATR{period}']
        atr_pct = self._result[f'ATR_pct{period}']
        vol_ratio = self._result[f'volatility_ratio_{period}']
        high_vol_signal = self._result[f'high_volatility_{period}']
        close = data['close']
        
        # 1. 波动性水平形态
        current_atr_pct = atr_pct.iloc[-1]
        avg_atr_pct = atr_pct.rolling(window=20).mean().iloc[-1]
        
        # 获取当前波动性水平
        current_vol_ratio = vol_ratio.iloc[-1]
        
        if pd.notna(current_vol_ratio) and current_vol_ratio > high_volatility:
            # 高波动性
            strength = min(90, 50 + (current_vol_ratio - high_volatility) * 20)
            patterns.append(PatternResult(
                pattern_id="ATR_HIGH_VOLATILITY",
                display_name="ATR高波动性",
                strength=strength,
                duration=self._detect_pattern_duration(high_vol_signal),
                details={"volatility_ratio": current_vol_ratio, "atr_pct": current_atr_pct}
            ).to_dict())
        elif pd.notna(current_vol_ratio) and current_vol_ratio < 0.5:
            # 低波动性
            strength = min(80, 40 + (0.5 - current_vol_ratio) * 30)
            patterns.append(PatternResult(
                pattern_id="ATR_LOW_VOLATILITY",
                display_name="ATR低波动性",
                strength=strength,
                duration=self._detect_pattern_duration(vol_ratio < 0.5),
                details={"volatility_ratio": current_vol_ratio, "atr_pct": current_atr_pct}
            ).to_dict())
        else:
            # 正常波动性
            patterns.append(PatternResult(
                pattern_id="ATR_NORMAL_VOLATILITY",
                display_name="ATR正常波动性",
                strength=50,
                duration=self._detect_pattern_duration((vol_ratio >= 0.5) & (vol_ratio <= high_volatility)),
                details={"volatility_ratio": current_vol_ratio, "atr_pct": current_atr_pct}
            ).to_dict())
        
        # 2. ATR趋势形态
        atr_trend = self._detect_atr_trend(atr)
        if atr_trend == "RISING_STRONG":
            patterns.append(PatternResult(
                pattern_id="ATR_RISING_STRONG",
                display_name="ATR快速上升",
                strength=85,
                duration=3,
                details={"atr_change_pct": (atr.iloc[-1] / atr.iloc[-5] - 1) * 100}
            ).to_dict())
        elif atr_trend == "RISING":
            patterns.append(PatternResult(
                pattern_id="ATR_RISING",
                display_name="ATR上升",
                strength=75,
                duration=3,
                details={"atr_change_pct": (atr.iloc[-1] / atr.iloc[-5] - 1) * 100}
            ).to_dict())
        elif atr_trend == "FALLING_STRONG":
            patterns.append(PatternResult(
                pattern_id="ATR_FALLING_STRONG",
                display_name="ATR快速下降",
                strength=85,
                duration=3,
                details={"atr_change_pct": (atr.iloc[-5] / atr.iloc[-1] - 1) * 100}
            ).to_dict())
        elif atr_trend == "FALLING":
            patterns.append(PatternResult(
                pattern_id="ATR_FALLING",
                display_name="ATR下降",
                strength=75,
                duration=3,
                details={"atr_change_pct": (atr.iloc[-5] / atr.iloc[-1] - 1) * 100}
            ).to_dict())
        elif atr_trend == "FLAT":
            patterns.append(PatternResult(
                pattern_id="ATR_FLAT",
                display_name="ATR平稳",
                strength=50,
                duration=3,
                details={"atr_change_pct": (atr.iloc[-1] / atr.iloc[-5] - 1) * 100}
            ).to_dict())
        
        # 3. ATR突破形态
        atr_breakout = self._detect_atr_breakout(atr)
        if atr_breakout:
            patterns.append(PatternResult(
                pattern_id="ATR_BREAKOUT",
                display_name="ATR突破",
                strength=90,
                duration=1,
                details={"breakout_pct": (atr.iloc[-1] / atr.iloc[-2] - 1) * 100}
            ).to_dict())
        
        # 4. ATR与价格的关系
        if len(close) >= 20:
            # 计算价格波动和ATR的比率
            price_range_20d = close.rolling(window=20).max() - close.rolling(window=20).min()
            current_price_range = price_range_20d.iloc[-1]
            current_atr = atr.iloc[-1]
            
            if pd.notna(current_price_range) and pd.notna(current_atr) and current_price_range > 0:
                price_atr_ratio = current_price_range / (current_atr * 20)
                
                if price_atr_ratio > 1.5:
                    # 价格波动大于ATR预期
                    patterns.append(PatternResult(
                        pattern_id="ATR_PRICE_VOLATILE",
                        display_name="价格波动超ATR预期",
                        strength=75,
                        duration=1,
                        details={"price_atr_ratio": price_atr_ratio}
                    ).to_dict())
                elif price_atr_ratio < 0.5:
                    # 价格波动小于ATR预期
                    patterns.append(PatternResult(
                        pattern_id="ATR_PRICE_STABLE",
                        display_name="价格波动低于ATR预期",
                        strength=75,
                        duration=1,
                        details={"price_atr_ratio": price_atr_ratio}
                    ).to_dict())
        
        # 5. ATR收敛/发散模式
        convergence_pattern = self._detect_atr_convergence_pattern(atr)
        if convergence_pattern == "CONVERGENCE":
            patterns.append(PatternResult(
                pattern_id="ATR_CONVERGENCE",
                display_name="ATR收敛",
                strength=80,
                duration=5,
                details={"convergence_pct": (1 - atr.iloc[-1] / atr.iloc[-10]) * 100 if len(atr) >= 10 else 0}
            ).to_dict())
        elif convergence_pattern == "DIVERGENCE":
            patterns.append(PatternResult(
                pattern_id="ATR_DIVERGENCE",
                display_name="ATR发散",
                strength=80,
                duration=5,
                details={"divergence_pct": (atr.iloc[-1] / atr.iloc[-10] - 1) * 100 if len(atr) >= 10 else 0}
            ).to_dict())
        
        # 6. 市场状态形态
        market_state = self._detect_market_state(atr_pct)
        if market_state == "VOLATILITY_EXPLOSION":
            patterns.append(PatternResult(
                pattern_id="ATR_VOLATILITY_EXPLOSION",
                display_name="ATR波动爆发",
                strength=95,
                duration=1,
                details={"atr_pct": current_atr_pct, "avg_atr_pct": avg_atr_pct}
            ).to_dict())
        elif market_state == "VOLATILITY_COLLAPSE":
            patterns.append(PatternResult(
                pattern_id="ATR_VOLATILITY_COLLAPSE",
                display_name="ATR波动崩塌",
                strength=95,
                duration=1,
                details={"atr_pct": current_atr_pct, "avg_atr_pct": avg_atr_pct}
            ).to_dict())
        elif market_state == "HIGH_VOLATILITY":
            patterns.append(PatternResult(
                pattern_id="ATR_MARKET_VOLATILE",
                display_name="ATR高波动市场",
                strength=85,
                duration=self._detect_pattern_duration(atr_pct > avg_atr_pct * 1.5),
                details={"atr_pct": current_atr_pct, "avg_atr_pct": avg_atr_pct}
            ).to_dict())
        elif market_state == "LOW_VOLATILITY":
            patterns.append(PatternResult(
                pattern_id="ATR_MARKET_QUIET",
                display_name="ATR低波动市场",
                strength=85,
                duration=self._detect_pattern_duration(atr_pct < avg_atr_pct * 0.5),
                details={"atr_pct": current_atr_pct, "avg_atr_pct": avg_atr_pct}
            ).to_dict())
        
        return patterns
    
    def _detect_pattern_duration(self, condition_series: pd.Series) -> int:
        """
        检测形态持续的天数
        
        Args:
            condition_series: 条件序列
        
        Returns:
            int: 持续天数
        """
        if len(condition_series) == 0:
            return 0
        
        # 获取连续满足条件的天数
        reverse_cond = condition_series.iloc[::-1]
        duration = 0
        
        for val in reverse_cond:
            if val:
                duration += 1
            else:
                break
        
        return duration
    
    def generate_trading_signals(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Dict[str, pd.Series]: 包含交易信号的字典
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        # 初始化信号
        signals = {}
        signals['buy_signal'] = pd.Series(False, index=data.index)
        signals['sell_signal'] = pd.Series(False, index=data.index)
        signals['signal_strength'] = pd.Series(0, index=data.index)
    
        # 在这里实现指标特定的信号生成逻辑
        # 此处提供默认实现
    
        return signals
    
    def _register_atr_patterns(self):
        """
        注册ATR形态
        """
        from indicators.pattern_registry import PatternRegistry, PatternType
        
        # 波动性水平形态
        PatternRegistry.register(
            pattern_id="ATR_HIGH_VOLATILITY",
            display_name="ATR高波动性",
            description="ATR相对值高于阈值，表示市场波动性大",
            indicator_types=["ATR", "波动"],
            score_impact=10.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_LOW_VOLATILITY",
            display_name="ATR低波动性",
            description="ATR相对值低于0.5，表示市场波动性小",
            indicator_types=["ATR", "波动"],
            score_impact=8.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_NORMAL_VOLATILITY",
            display_name="ATR正常波动性",
            description="ATR相对值处于正常范围，表示市场波动性正常",
            indicator_types=["ATR", "波动"],
            score_impact=0.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        # ATR趋势形态
        PatternRegistry.register(
            pattern_id="ATR_RISING_STRONG",
            display_name="ATR快速上升",
            description="ATR快速上升，表示波动性显著增加",
            indicator_types=["ATR", "波动"],
            score_impact=15.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_RISING",
            display_name="ATR上升",
            description="ATR上升，表示波动性增加",
            indicator_types=["ATR", "波动"],
            score_impact=10.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_FALLING_STRONG",
            display_name="ATR快速下降",
            description="ATR快速下降，表示波动性显著减少",
            indicator_types=["ATR", "波动"],
            score_impact=-10.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_FALLING",
            display_name="ATR下降",
            description="ATR下降，表示波动性减少",
            indicator_types=["ATR", "波动"],
            score_impact=-5.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_FLAT",
            display_name="ATR平稳",
            description="ATR保持平稳，表示波动性稳定",
            indicator_types=["ATR", "波动"],
            score_impact=0.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        # ATR突破形态
        PatternRegistry.register(
            pattern_id="ATR_BREAKOUT",
            display_name="ATR突破",
            description="ATR突破上升，表示波动性快速增加",
            indicator_types=["ATR", "波动"],
            score_impact=20.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        # ATR与价格关系
        PatternRegistry.register(
            pattern_id="ATR_PRICE_VOLATILE",
            display_name="价格波动超ATR预期",
            description="价格波动范围大于ATR预期，表示市场可能超预期变化",
            indicator_types=["ATR", "波动"],
            score_impact=15.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_PRICE_STABLE",
            display_name="价格波动低于ATR预期",
            description="价格波动范围小于ATR预期，表示市场可能被压制",
            indicator_types=["ATR", "波动"],
            score_impact=-15.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        # ATR收敛/发散形态
        PatternRegistry.register(
            pattern_id="ATR_CONVERGENCE",
            display_name="ATR收敛",
            description="ATR逐渐收敛，表示波动性减弱",
            indicator_types=["ATR", "波动"],
            score_impact=-10.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_DIVERGENCE",
            display_name="ATR发散",
            description="ATR逐渐发散，表示波动性增强",
            indicator_types=["ATR", "波动"],
            score_impact=10.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        # 市场状态形态
        PatternRegistry.register(
            pattern_id="ATR_VOLATILITY_EXPLOSION",
            display_name="ATR波动爆发",
            description="ATR波动性爆发式增长，表示市场可能剧烈波动",
            indicator_types=["ATR", "波动"],
            score_impact=25.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_VOLATILITY_COLLAPSE",
            display_name="ATR波动崩塌",
            description="ATR波动性急剧下降，表示市场可能进入平静期",
            indicator_types=["ATR", "波动"],
            score_impact=-25.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_MARKET_VOLATILE",
            display_name="ATR高波动市场",
            description="ATR表明市场处于高波动状态",
            indicator_types=["ATR", "波动"],
            score_impact=15.0,
            pattern_type="volatility",
            signal_type="neutral"
        )
        
        PatternRegistry.register(
            pattern_id="ATR_MARKET_QUIET",
            display_name="ATR低波动市场",
            description="ATR表明市场处于低波动状态",
            indicator_types=["ATR", "波动"],
            score_impact=-15.0,
            pattern_type="volatility",
            signal_type="neutral"
        ) 