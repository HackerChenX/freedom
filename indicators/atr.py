#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
平均真实波幅(Average True Range，ATR)

衡量价格波动的幅度，是一种波动性指标
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple

from indicators.base_indicator import BaseIndicator
from utils.logger import get_logger

logger = get_logger(__name__)


class ATR(BaseIndicator):
    """
    平均真实波幅(Average True Range，ATR)
    
    分类：波动性指标
    描述：衡量价格波动的幅度，反映市场波动性的大小
    """
    
    def __init__(self, period: int = 14):
        """
        初始化平均真实波幅(ATR)
        
        Args:
            period: 计算周期，默认为14
        """
        super().__init__(name="ATR", description="平均真实波幅，衡量价格波动幅度")
        self.period = period
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        验证DataFrame是否包含所需的列
        
        Args:
            df: 包含价格数据的DataFrame
        
        Raises:
            ValueError: 如果DataFrame不包含所需的列，或者行数少于所需的最小行数
        """
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame缺少必要的列: {', '.join(missing_columns)}")
        
        if len(df) < self.period + 1:  # 需要前一天的收盘价，因此+1
            raise ValueError(f"DataFrame至少需要 {self.period + 1} 行数据，但只有 {len(df)} 行")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算平均真实波幅
        
        Args:
            df: 包含价格数据的DataFrame，必须包含high、low、close列
        
        Returns:
            包含ATR指标结果的DataFrame
        """
        self._validate_dataframe(df)
        
        result = df.copy()
        
        # 计算真实波幅(True Range)
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        prev_close = np.roll(close, 1)
        
        # 第一个值没有前一天的收盘价，设为NaN
        prev_close[0] = np.nan
        
        # 计算三种情况的最大值
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        
        # 将真实波幅添加到结果DataFrame
        result['TR'] = tr
        
        # 计算ATR (使用简单移动平均)
        result['ATR'] = result['TR'].rolling(window=self.period).mean()
        
        # 计算相对ATR (ATR/收盘价的百分比)
        result['ATR_Percent'] = (result['ATR'] / df['close']) * 100
        
        # 保存结果
        self._result = result
        
        return result
    
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
        
        atr = self._result['ATR']
        atr_percent = self._result['ATR_Percent']
        
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
        
        atr = self._result['ATR']
        atr_percent = self._result['ATR_Percent']
        
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
        if 'ATR' not in result.columns:
            result = self.calculate(df)
        
        # 初始化信号列
        result['volatility_high'] = 0
        result['volatility_low'] = 0
        result['atr_rising'] = 0
        result['atr_falling'] = 0
        
        # 填充NaN值，避免计算问题
        result = result.fillna(0)
        
        # 计算ATR的历史统计特征
        atr = result['ATR'].values
        atr_percent = result['ATR_Percent'].values
        
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
        ax.plot(result['ATR'], label=f'ATR({self.period})')
        
        # 标记高波动性和低波动性区域
        high_volatility = result[result['volatility_high'] == 1].index
        low_volatility = result[result['volatility_low'] == 1].index
        
        ax.scatter(high_volatility, result.loc[high_volatility, 'ATR'], color='red', marker='^', s=100, label='高波动性')
        ax.scatter(low_volatility, result.loc[low_volatility, 'ATR'], color='green', marker='v', s=100, label='低波动性')
        
        ax.set_title(f'平均真实波幅(ATR) - 周期:{self.period}')
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