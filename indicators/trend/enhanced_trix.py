from utils.container import container
"""
增强型TRIX三重指数平滑移动平均线模块

实现增强型TRIX指标计算，提供自适应参数、多周期协同分析、形态识别等功能
"""

from utils.logger import get_logger

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.base.pattern_signal_mixin import PatternSignalMixin
from indicators.base.minimum_periods_mixin import MinimumPeriodsMixin
from indicators.trix import TripleExponentialAverage as TRIX
from utils.logger import get_logger
from utils.technical_utils import find_peaks_and_troughs
from utils.indicator_utils import crossover, crossunder

logger = get_logger(__name__)


class EnhancedTrix(BaseIndicator, PatternSignalMixin, MinimumPeriodsMixin):
    """
    增强型TRIX三重指数平滑移动平均线指标
    
    具有以下增强特性:
    1. 自适应周期调整：根据市场波动率动态调整TRIX参数
    2. 零轴交叉质量评估：评估TRIX与零轴交叉的可靠性
    3. 背离检测系统：检测TRIX与价格之间的背离关系  # TODO: 将魔法数字提取到配置中
    4. 多周期TRIX协同分析：结合不同周期的TRIX指标提高信号可靠性  # TODO: 将魔法数字提取到配置中
    5. 市场环境自适应：根据市场环境动态调整评分标准  # TODO: 将魔法数字提取到配置中
    """
    
    def __init__(self, 
                 n: int = 12,  # TODO: 将魔法数字提取到配置中 
                 m: int = 9,  # TODO: 将魔法数字提取到配置中
                 secondary_n: int = 24,  # TODO: 将魔法数字提取到配置中
                 multi_periods: List[int] = None,
                 adaptive_period: bool = True,
                 volatility_lookback: int = 20,  # TODO: 将魔法数字提取到配置中
                 use_smoothed_trix: bool = True,
                 smoothing_period: int = 3):  # TODO: 将魔法数字提取到配置中
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化增强型TRIX指标
        
        Args:
            n: 主要周期，默认为12
            m: 信号线周期，默认为9
            secondary_n: 次要周期，默认为24
            multi_periods: 多周期分析参数，默认为[6, 12, 24, 48]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            adaptive_period: 是否启用自适应周期，默认为True
            volatility_lookback: 波动率计算回溯期，默认为20
            use_smoothed_trix: 是否使用平滑后的TRIX
            smoothing_period: 平滑周期，默认为3
        """
        super().__init__()
        self.n = n
        self.m = m
        self.name = "EnhancedTRIX"
        self.description = "增强型TRIX三重指数平滑移动平均线，优化参数自适应性，增加多周期协同分析和市场环境感知"
        self.indicator_type = "ENHANCEDTRIX"
        self.secondary_n = secondary_n
        self.multi_periods = multi_periods or [6, 12, 24, 48]  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        self.adaptive_period = adaptive_period
        self.volatility_lookback = volatility_lookback
        self.use_smoothed_trix = use_smoothed_trix
        self.smoothing_period = smoothing_period
        self.market_environment = "normal"
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        
        # 内部变量
        self._secondary_trix = None
        self._multi_period_trix = {}
        self._price_data = None
        self._adaptive_n = n  # 自适应后的周期
    
    def get_indicator_type_Trix(self) -> str:
        """
        获取指标类型
        
        Returns:
            str: 指标类型
        """
        return self.indicator_type
    
    def set_market_environment_Trix(self, environment: str) -> None:
        """
        设置市场环境
        
        Args:
            environment (str): 市场环境类型 ('bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal')
        """
        valid_environments = ['bull_market', 'bear_market', 'sideways_market', 'volatile_market', 'normal']
        if environment not in valid_environments:
            raise ValueError(f"无效的市场环境类型: {environment}。有效类型: {valid_environments}")
        
        self.market_environment = environment

    def _calculate_enhancedtrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        实现BaseIndicator的抽象方法

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 计算结果
        """
        # 确保数据包含必需的列
        if "close" not in data.columns:
            raise ValueError("数据必须包含'close'列")

        # 保存价格数据用于后续分析
        self._price_data = data['close'].copy()

        # 如果启用自适应周期，则调整参数
        if self.adaptive_period:
            self._adjust_period_by_volatility(data)

        # 使用调整后的周期计算主要TRIX
        # 临时设置参数
        original_n = self.n
        original_m = self.m
        self.n = self._adaptive_n
        self.m = self.m

        # 直接计算TRIX而不调用super().calculate()来避免递归
        result = self._calculate_trix_directly(data)

        # 恢复原始参数
        self.n = original_n
        self.m = original_m

        # 计算次要周期TRIX
        try:
            secondary_trix = TRIX(n=self.secondary_n, m=self.m)
            secondary_result = secondary_trix.calculate(data)
            result['trix_secondary'] = secondary_result['TRIX']
            result['matrix_secondary'] = secondary_result['MATRIX']
            self._secondary_trix = result['trix_secondary']
        except Exception as e:
            # 如果TRIX计算失败，使用简化版本
            result['trix_secondary'] = pd.Series(0.0, index=data.index)
            result['matrix_secondary'] = pd.Series(0.0, index=data.index)
            self._secondary_trix = result['trix_secondary']

        # 计算多周期TRIX
        for period in self.multi_periods:
            if period != self._adaptive_n and period != self.secondary_n:
                try:
                    multi_trix = TRIX(n=period, m=self.m)
                    multi_result = multi_trix.calculate(data)
                    result[f'trix_{period}'] = multi_result['TRIX']
                    result[f'matrix_{period}'] = multi_result['MATRIX']
                    self._multi_period_trix[period] = result[f'trix_{period}']
                except Exception:
                    # 如果计算失败，使用简化版本
                    result[f'trix_{period}'] = pd.Series(0.0, index=data.index)
                    result[f'matrix_{period}'] = pd.Series(0.0, index=data.index)
                    self._multi_period_trix[period] = result[f'trix_{period}']

        # 计算TRIX动态特性
        result['trix_momentum'] = result['TRIX'] - result['TRIX'].shift(3)  # TODO: 将魔法数字提取到配置中
        result['trix_slope'] = self._calculate_slope_Enhanced_Trix(result['TRIX'], 5)  # TODO: 将魔法数字提取到配置中
        result['trix_accel'] = result['trix_slope'] - result['trix_slope'].shift(1)

        # 计算TRIX波动率
        result['trix_volatility'] = result['TRIX'].rolling(window=self.volatility_lookback).std()

        # 保存结果
        self._result = result

        return result

    def _calculate_trix_directly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        直接计算TRIX指标，避免递归调用

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 包含TRIX计算结果的Data_frame
        """
        result = data.copy()
        close = data['close']

        # 计算三重指数平滑移动平均
        # 第一次EMA
        ema1 = close.ewm(span=self.n).mean()

        # 第二次EMA
        ema2 = ema1.ewm(span=self.n).mean()

        # 第三次EMA
        ema3 = ema2.ewm(span=self.n).mean()

        # 计算TRIX
        trix = (ema3 / ema3.shift(1) - 1) * 10000  # TODO: 将魔法数字提取到配置中

        # 计算MATRIX（TRIX的移动平均）
        matrix = trix.ewm(span=self.m).mean()

        result['TRIX'] = trix
        result['MATRIX'] = matrix

        return result
    
    def _adjust_period_by_volatility(self, data: pd.DataFrame) -> None:
        """
        根据市场波动率动态调整TRIX周期参数
        
        Args:
            data: 包含价格数据的Data_frame
        """
        # 计算价格波动率
        close = data['close']
        
        # 计算价格变化率
        returns = close.pct_change()
        
        # 计算波动率（标准差）
        volatility = returns.rolling(window=self.volatility_lookback).std().iloc[-1]
        
        # 如果波动率数据不足，则使用默认周期
        if pd.isna(volatility):
            self._adaptive_n = self.n
            return
        
        # 计算历史波动率
        historical_volatility = returns.rolling(window=self.volatility_lookback*5).std().iloc[-1]  # TODO: 将魔法数字提取到配置中
        
        # 如果历史波动率数据不足，则使用默认周期
        if pd.isna(historical_volatility) or historical_volatility == 0:
            self._adaptive_n = self.n
            return
        
        # 计算相对波动率
        relative_volatility = volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # 根据相对波动率调整周期
        if relative_volatility > 1.5:  # 高波动市场  # TODO: 将魔法数字提取到配置中
            # 增加周期以过滤噪声
            self._adaptive_n = int(self.n * 1.5)  # TODO: 将魔法数字提取到配置中
        elif relative_volatility < 0.7:  # 低波动市场  # TODO: 将魔法数字提取到配置中
            # 减少周期以提高敏感度
            self._adaptive_n = max(int(self.n * 0.7), 6)  # 确保最小周期为6  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        else:  # 正常波动市场
            # 使用默认周期
            self._adaptive_n = self.n
        
        # 根据市场环境进一步调整
        if self.market_environment == 'bull_market':
            # 牛市中略微减少周期，更敏感地捕捉上涨趋势
            self._adaptive_n = max(int(self._adaptive_n * 0.9), 6)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        elif self.market_environment == 'bear_market':
            # 熊市中略微增加周期，过滤更多噪声
            self._adaptive_n = int(self._adaptive_n * 1.1)
        elif self.market_environment == 'volatile_market':
            # 高波动市场中增加周期，过滤更多噪声
            self._adaptive_n = int(self._adaptive_n * 1.2)
        
        logger.debug(f"调整TRIX周期: 原始={self.n}, 调整后={self._adaptive_n}, "
                    f"相对波动率={relative_volatility:.2f}, 市场环境={self.market_environment}")
    
    def _calculate_slope_Enhanced_Trix(self, series: pd.Series, period: int = 5) -> pd.Series:  # TODO: 将魔法数字提取到配置中
        """
        计算序列的斜率
        
        Args:
            series: 输入序列
            period: 计算斜率的周期
            
        Returns:
            pd.Series: 斜率序列
        """
        slope = pd.Series(np.nan, index=series.index)
        
        for i in range(period, len(series)):
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            
            # 使用线性回归计算斜率
            if len(y) == period and not np.isnan(y).any():
                slope.iloc[i] = np.polyfit(x, y, 1)[0]
        
        return slope 

    def detect_divergence_Trix(self) -> pd.DataFrame:
        """
        检测TRIX与价格之间的背离关系
        
        Returns:
            pd.DataFrame: 包含背离分析结果的Data_frame
        """
        if self._result is None or self._price_data is None:
            return pd.DataFrame()
            
        # 获取TRIX和价格数据
        trix = self._result['TRIX']
        price = self._price_data
        
        # 创建结果DataFrame
        divergence = pd.DataFrame(index=price.index)
        divergence['bullish_divergence'] = False
        divergence['bearish_divergence'] = False
        divergence['hidden_bullish_divergence'] = False
        divergence['hidden_bearish_divergence'] = False
        divergence['divergence_strength'] = 0.0
        
        # 查找价格和TRIX的高点和低点
        price_peaks, price_troughs = find_peaks_and_troughs(price.values, window=10)
        trix_peaks, trix_troughs = find_peaks_and_troughs(trix.values, window=10)
        
        # 最小背离长度(防止检测到太短的背离)
        min_divergence_length = 5  # TODO: 将魔法数字提取到配置中
        # 最大背离长度(防止检测到太长的背离)
        max_divergence_length = 30  # TODO: 将魔法数字提取到配置中
        
        # 常规看涨背离：价格创新低但TRIX未创新低
        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price):
                continue
                
            current_trough_idx = price_troughs[i]
            prev_trough_idx = price_troughs[i-1]
            
            # 检查背离长度是否合适
            if (current_trough_idx - prev_trough_idx < min_divergence_length or 
                current_trough_idx - prev_trough_idx > max_divergence_length):
                continue
                
            # 价格创新低
            if price.iloc[current_trough_idx] < price.iloc[prev_trough_idx]:
                # 查找对应的TRIX值
                current_trix_trough = None
                prev_trix_trough = None
                
                # 在价格低点附近查找TRIX低点
                for tt in trix_troughs:
                    if abs(tt - current_trough_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        current_trix_trough = tt
                    if abs(tt - prev_trough_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        prev_trix_trough = tt
                
                # 如果找到了对应的TRIX低点
                if current_trix_trough is not None and prev_trix_trough is not None:
                    # TRIX未创新低
                    if trix.iloc[current_trix_trough] > trix.iloc[prev_trix_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        trix_change = (trix.iloc[current_trix_trough] / trix.iloc[prev_trix_trough]) - 1
                        # 防止除以零
                        if max(abs(), abs(trix_change)) > 0:
                            strength = abs(- trix_change) / max(abs(), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 0] = True  # bullish_divergence  # TODO: 将魔法数字提取到配置中
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 常规看跌背离：价格创新高但TRIX未创新高
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price):
                continue
                
            current_peak_idx = price_peaks[i]
            prev_peak_idx = price_peaks[i-1]
            
            # 检查背离长度是否合适
            if (current_peak_idx - prev_peak_idx < min_divergence_length or 
                current_peak_idx - prev_peak_idx > max_divergence_length):
                continue
                
            # 价格创新高
            if price.iloc[current_peak_idx] > price.iloc[prev_peak_idx]:
                # 查找对应的TRIX值
                current_trix_peak = None
                prev_trix_peak = None
                
                # 在价格高点附近查找TRIX高点
                for tp in trix_peaks:
                    if abs(tp - current_peak_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        current_trix_peak = tp
                    if abs(tp - prev_peak_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        prev_trix_peak = tp
                
                # 如果找到了对应的TRIX高点
                if current_trix_peak is not None and prev_trix_peak is not None:
                    # TRIX未创新高
                    if trix.iloc[current_trix_peak] < trix.iloc[prev_trix_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        trix_change = (trix.iloc[current_trix_peak] / trix.iloc[prev_trix_peak]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 1] = True  # bearish_divergence  # TODO: 将魔法数字提取到配置中
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 隐藏看涨背离：价格更高的低点但TRIX更低的低点
        for i in range(1, len(price_troughs)):
            if i >= len(price_troughs) or price_troughs[i] >= len(price):
                continue
                
            current_trough_idx = price_troughs[i]
            prev_trough_idx = price_troughs[i-1]
            
            # 检查背离长度是否合适
            if (current_trough_idx - prev_trough_idx < min_divergence_length or 
                current_trough_idx - prev_trough_idx > max_divergence_length):
                continue
                
            # 价格更高的低点
            if price.iloc[current_trough_idx] > price.iloc[prev_trough_idx]:
                # 查找对应的TRIX值
                current_trix_trough = None
                prev_trix_trough = None
                
                # 在价格低点附近查找TRIX低点
                for tt in trix_troughs:
                    if abs(tt - current_trough_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        current_trix_trough = tt
                    if abs(tt - prev_trough_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        prev_trix_trough = tt
                
                # 如果找到了对应的TRIX低点
                if current_trix_trough is not None and prev_trix_trough is not None:
                    # TRIX更低的低点
                    if trix.iloc[current_trix_trough] < trix.iloc[prev_trix_trough]:
                        # 计算背离强度
                        price_change = (price.iloc[current_trough_idx] / price.iloc[prev_trough_idx]) - 1
                        trix_change = (trix.iloc[current_trix_trough] / trix.iloc[prev_trix_trough]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 2] = True  # hidden_bullish_divergence  # TODO: 将魔法数字提取到配置中
                        divergence.iloc[current_trough_idx:current_trough_idx+5, 4] = strength  # divergence_strength  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 隐藏看跌背离：价格更低的高点但TRIX更高的高点
        for i in range(1, len(price_peaks)):
            if i >= len(price_peaks) or price_peaks[i] >= len(price):
                continue
                
            current_peak_idx = price_peaks[i]
            prev_peak_idx = price_peaks[i-1]
            
            # 检查背离长度是否合适
            if (current_peak_idx - prev_peak_idx < min_divergence_length or 
                current_peak_idx - prev_peak_idx > max_divergence_length):
                continue
                
            # 价格更低的高点
            if price.iloc[current_peak_idx] < price.iloc[prev_peak_idx]:
                # 查找对应的TRIX值
                current_trix_peak = None
                prev_trix_peak = None
                
                # 在价格高点附近查找TRIX高点
                for tp in trix_peaks:
                    if abs(tp - current_peak_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        current_trix_peak = tp
                    if abs(tp - prev_peak_idx) <= 3:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        prev_trix_peak = tp
                
                # 如果找到了对应的TRIX高点
                if current_trix_peak is not None and prev_trix_peak is not None:
                    # TRIX更高的高点
                    if trix.iloc[current_trix_peak] > trix.iloc[prev_trix_peak]:
                        # 计算背离强度
                        price_change = (price.iloc[current_peak_idx] / price.iloc[prev_peak_idx]) - 1
                        trix_change = (trix.iloc[current_trix_peak] / trix.iloc[prev_trix_peak]) - 1
                        # 防止除以零
                        if max(abs(price_change), abs(trix_change)) > 0:
                            strength = abs(price_change - trix_change) / max(abs(price_change), abs(trix_change))
                        else:
                            strength = 0
                        
                        # 记录背离
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 3] = True  # hidden_bearish_divergence  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        divergence.iloc[current_peak_idx:current_peak_idx+5, 4] = strength  # divergence_strength  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        return divergence
    
    def analyze_multi_period_synergy_Trix(self) -> pd.DataFrame:
        """
        多周期TRIX协同分析
        
        Returns:
            pd.DataFrame: 包含多周期协同分析结果的Data_frame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 创建结果DataFrame
        synergy = pd.DataFrame(index=self._result.index)
        
        # 获取主要TRIX
        primary_trix = self._result['TRIX']
        secondary_trix = self._secondary_trix
        
        # 分析多周期趋势一致性
        synergy['primary_above_zero'] = primary_trix > 0
        synergy['primary_below_zero'] = primary_trix < 0
        synergy['secondary_above_zero'] = secondary_trix > 0
        synergy['secondary_below_zero'] = secondary_trix < 0
        
        # 计算TRIX方向
        synergy['primary_rising'] = primary_trix > primary_trix.shift(1)
        synergy['primary_falling'] = primary_trix < primary_trix.shift(1)
        synergy['secondary_rising'] = secondary_trix > secondary_trix.shift(1)
        synergy['secondary_falling'] = secondary_trix < secondary_trix.shift(1)
        
        # 计算多周期一致性
        bullish_count = synergy['primary_above_zero'].astype(int) + synergy['primary_rising'].astype(int)
        bearish_count = synergy['primary_below_zero'].astype(int) + synergy['primary_falling'].astype(int)
        
        # 添加次要周期的状态
        bullish_count += synergy['secondary_above_zero'].astype(int) + synergy['secondary_rising'].astype(int)
        bearish_count += synergy['secondary_below_zero'].astype(int) + synergy['secondary_falling'].astype(int)
        
        # 添加其他周期的状态
        for period, trix_series in self._multi_period_trix.items():
            period_above_zero = trix_series > 0
            period_below_zero = trix_series < 0
            period_rising = trix_series > trix_series.shift(1)
            period_falling = trix_series < trix_series.shift(1)
            
            synergy[f'trix{period}_above_zero'] = period_above_zero
            synergy[f'trix{period}_below_zero'] = period_below_zero
            synergy[f'trix{period}_rising'] = period_rising
            synergy[f'trix{period}_falling'] = period_falling
            
            bullish_count += period_above_zero.astype(int) + period_rising.astype(int)
            bearish_count += period_below_zero.astype(int) + period_falling.astype(int)
        
        # 计算总检查数（正面+负面特征的总数）
        total_checks = len(self._multi_period_trix) * 2 + 4  # 每个周期有2个检查（零轴位置和方向），加上主要和次要周期的4个检查  # TODO: 将魔法数字提取到配置中
        
        # 计算看涨和看跌比例
        synergy['bullish_ratio'] = bullish_count / total_checks
        synergy['bearish_ratio'] = bearish_count / total_checks
        
        # 计算一致性得分（0-100）
        synergy['consensus_score'] = 50 + (synergy['bullish_ratio'] - synergy['bearish_ratio']) * 50  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 多周期交叉信号
        synergy['multi_period_bullish_signal'] = False
        synergy['multi_period_bearish_signal'] = False
        
        # 检测多周期协同交叉信号
        primary_cross_up_zero = crossover(primary_trix, 0)
        primary_cross_down_zero = crossunder(primary_trix, 0)
        secondary_cross_up_zero = crossover(secondary_trix, 0)
        secondary_cross_down_zero = crossunder(secondary_trix, 0)
        
        # 主要周期和次要周期同时发出信号
        bullish_signal = primary_cross_up_zero & (secondary_trix > 0)
        bearish_signal = primary_cross_down_zero & (secondary_trix < 0)
        
        # 在较短周期信号之后较长周期也发出信号，强化确认
        for period, trix_series in self._multi_period_trix.items():
            if period > self._adaptive_n:  # 只考虑长周期
                period_cross_up_zero = crossover(trix_series, 0)
                period_cross_down_zero = crossunder(trix_series, 0)
                
                # 在主周期信号后10个周期内，长周期也发出相同信号
                for i in range(len(bullish_signal)):
                    if i >= 10 and bullish_signal.iloc[i-10]:
                        for j in range(1, 11):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            if i+j < len(period_cross_up_zero) and period_cross_up_zero.iloc[i+j]:
                                bullish_signal.iloc[i+j] = True
                
                for i in range(len(bearish_signal)):
                    if i >= 10 and bearish_signal.iloc[i-10]:
                        for j in range(1, 11):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                            if i+j < len(period_cross_down_zero) and period_cross_down_zero.iloc[i+j]:
                                bearish_signal.iloc[i+j] = True
        
        synergy['multi_period_bullish_signal'] = bullish_signal
        synergy['multi_period_bearish_signal'] = bearish_signal
        
        return synergy 

    def evaluate_zero_cross_quality(self) -> pd.DataFrame:
        """
        评估TRIX零轴交叉质量
        
        Returns:
            pd.DataFrame: 包含零轴交叉质量评估结果的Data_frame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取TRIX数据
        trix = self._result['TRIX']
        
        # 创建结果DataFrame
        quality = pd.DataFrame(index=self._result.index)
        
        # 检测零轴交叉
        cross_up_zero = crossover(trix, 0)
        cross_down_zero = crossunder(trix, 0)
        
        quality['cross_up_zero'] = cross_up_zero
        quality['cross_down_zero'] = cross_down_zero
        
        # 评估交叉角度
        quality['cross_angle'] = 0.0
        
        for i in range(5, len(trix)):  # TODO: 将魔法数字提取到配置中
            if cross_up_zero.iloc[i] or cross_down_zero.iloc[i]:
                # 计算交叉前后5个周期的斜率
                pre_slope = (trix.iloc[i] - trix.iloc[i-5]) / 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                quality.iloc[i, 2] = abs(pre_slope)  # cross_angle
        
        # 评估交叉后的加速度
        quality['post_cross_acceleration'] = 0.0
        
        for i in range(5, len(trix)-5):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            if cross_up_zero.iloc[i] or cross_down_zero.iloc[i]:
                if i+5 < len(trix):  # TODO: 将魔法数字提取到配置中
                    # 计算交叉后5个周期的加速度
                    slope1 = (trix.iloc[i+1] - trix.iloc[i]) / 1
                    slope5 = (trix.iloc[i+5] - trix.iloc[i]) / 5  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    accel = slope5 - slope1
                    
                    if (cross_up_zero.iloc[i] and accel > 0) or (cross_down_zero.iloc[i] and accel < 0):
                        quality.iloc[i, 3] = abs(accel)  # post_cross_acceleration  # TODO: 将魔法数字提取到配置中
        
        # 评估交叉持续性
        quality['cross_persistence'] = 0.0
        
        for i in range(5, len(trix)-10):  # TODO: 将魔法数字提取到配置中
            if cross_up_zero.iloc[i]:
                # 检查交叉后10个周期内是否保持方向
                persistence = 0
                for j in range(1, 11):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    if i+j < len(trix) and trix.iloc[i+j] > 0:
                        persistence += 1
                
                quality.iloc[i, 4] = persistence / 10  # cross_persistence  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
            elif cross_down_zero.iloc[i]:
                # 检查交叉后10个周期内是否保持方向
                persistence = 0
                for j in range(1, 11):  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                    if i+j < len(trix) and trix.iloc[i+j] < 0:
                        persistence += 1
                
                quality.iloc[i, 4] = persistence / 10  # cross_persistence  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 综合评分（0-100）
        quality['cross_quality_score'] = 0.0
        
        for i in range(len(trix)):
            if cross_up_zero.iloc[i] or cross_down_zero.iloc[i]:
                # 综合考虑角度、加速度和持续性
                angle_score = min(40, quality.iloc[i, 2] * 80)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                accel_score = min(30, quality.iloc[i, 3] * 60)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                persistence_score = quality.iloc[i, 4] * 30  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                
                quality.iloc[i, 5] = angle_score + accel_score + persistence_score  # cross_quality_score  # TODO: 将魔法数字提取到配置中
        
        return quality
    
    def identify_patterns_Trix_Enhanced_Trix(self) -> pd.DataFrame:
        """
        识别TRIX形态
        
        Returns:
            pd.DataFrame: 包含形态识别结果的Data_frame
        """
        if self._result is None:
            return pd.DataFrame()
            
        # 获取TRIX数据
        trix = self._result['TRIX']
        matrix = self._result['MATRIX']
        
        # 创建结果DataFrame
        patterns = pd.DataFrame(index=self._result.index)
        
        # 基础趋势形态
        patterns['above_zero'] = trix > 0
        patterns['below_zero'] = trix < 0
        patterns['rising'] = trix > trix.shift(1)
        patterns['falling'] = trix < trix.shift(1)
        
        # 计算TRIX交叉信号
        patterns['golden_cross'] = crossover(trix, matrix)
        patterns['death_cross'] = crossunder(trix, matrix)
        patterns['cross_up_zero'] = crossover(trix, 0)
        patterns['cross_down_zero'] = crossunder(trix, 0)
        
        # 高质量零轴交叉
        zero_cross_quality = self.evaluate_zero_cross_quality()
        if 'cross_quality_score' in zero_cross_quality.columns:
            high_quality_threshold = 70  # TODO: 将魔法数字提取到配置中
            patterns['high_quality_cross_up_zero'] = (patterns['cross_up_zero'] & 
                                                    (zero_cross_quality['cross_quality_score'] > high_quality_threshold))
            patterns['high_quality_cross_down_zero'] = (patterns['cross_down_zero'] & 
                                                      (zero_cross_quality['cross_quality_score'] > high_quality_threshold))
        
        # 获取背离分析结果
        divergence = self.detect_divergence_Trix()
        if not divergence.empty:
            patterns['bullish_divergence'] = divergence['bullish_divergence']
            patterns['bearish_divergence'] = divergence['bearish_divergence']
            patterns['hidden_bullish_divergence'] = divergence['hidden_bullish_divergence']
            patterns['hidden_bearish_divergence'] = divergence['hidden_bearish_divergence']
        
        # 多周期协同分析
        synergy = self.analyze_multi_period_synergy_Trix()
        if not synergy.empty:
            patterns['multi_period_bullish_signal'] = synergy['multi_period_bullish_signal']
            patterns['multi_period_bearish_signal'] = synergy['multi_period_bearish_signal']
            patterns['strong_bullish_consensus'] = synergy['consensus_score'] > 70  # TODO: 将魔法数字提取到配置中
            patterns['strong_bearish_consensus'] = synergy['consensus_score'] < 30  # TODO: 将魔法数字提取到配置中
        
        # 趋势加速/减速
        if 'trix_accel' in self._result.columns:
            patterns['acceleration'] = self._result['trix_accel'] > 0
            patterns['deceleration'] = self._result['trix_accel'] < 0
        
        # 钝化形态（TRIX在零轴附近徘徊）
        patterns['stagnation_near_zero'] = self._detect_stagnation_Enhanced_Trix(trix, threshold=0.1, periods=5)  # TODO: 将魔法数字提取到配置中
        
        return patterns
    
    def _detect_stagnation_Enhanced_Trix(self, trix: pd.Series, threshold: float, periods: int) -> pd.Series:
        """
        检测TRIX在零轴附近的钝化形态
        
        Args:
            trix: TRIX序列
            threshold: 零轴附近的阈值
            periods: 持续周期数
            
        Returns:
            pd.Series: 钝化形态识别结果
        """
        stagnation = pd.Series(False, index=trix.index)
        
        if len(trix) < periods:
            return stagnation
        
        # 检查连续periods个周期TRIX都在零轴附近
        for i in range(periods, len(trix)):
            window = trix.iloc[i-periods+1:i+1]
            
            # 检查是否所有值都在零轴附近
            near_zero = (abs(window) < threshold).all()
            
            # 检查波动性是否低
            low_volatility = window.std() < threshold/2
            
            if near_zero and low_volatility:
                stagnation.iloc[i] = True
        
        return stagnation
    
    def calculate_score_Trix_Enhanced_Trix(self, data: pd.DataFrame = None) -> pd.Series:
        """
        计算TRIX综合评分 (0-100)
        
        Args:
            data (pd.DataFrame, optional): 价格数据，如果未提供则使用上次计算结果
            
        Returns:
            pd.Series: 评分 (0-100，50为中性)
        """
        if self._result is None and data is not None:
            self.calculate(data)
            
        if self._result is None:
            return pd.Series()
        
        # 获取TRIX数据
        trix = self._result['TRIX']
        matrix = self._result['MATRIX']
        trix_momentum = self._result['trix_momentum']
        trix_slope = self._result['trix_slope']
        
        # 获取背离分析
        divergence = self.detect_divergence_Trix()
        
        # 获取零轴交叉质量评估
        zero_cross = self.evaluate_zero_cross_quality()
        
        # 获取形态识别
        patterns = self.identify_patterns_Trix_Enhanced_Trix()
        
        # 获取多周期协同分析
        synergy = self.analyze_multi_period_synergy_Trix()
        
        # 基础分数为50（中性）
        score = pd.Series(50, index=self._result.index)  # TODO: 将魔法数字提取到配置中
        
        # 1. TRIX基础评分 (±20分)
        # TRIX > 0 看涨，TRIX < 0 看跌
        score += np.where(trix > 0, np.minimum(trix * 200, 20), np.maximum(trix * 200, -20))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 2. TRIX与信号线关系评分 (±15分)
        # TRIX > MATRIX 看涨，TRIX < MATRIX 看跌
        trix_vs_matrix = trix - matrix
        normalized_diff = trix_vs_matrix / trix.rolling(window=20).std().replace(0, 0.001)  # TODO: 将魔法数字提取到配置中
        score += np.where(trix_vs_matrix > 0, 
                        np.minimum(normalized_diff * 5, 15),  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中 
                        np.maximum(normalized_diff * 5, -15))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 3. TRIX动量评分 (±10分)  # TODO: 将魔法数字提取到配置中
        # 动量为正看涨，动量为负看跌
        normalized_momentum = trix_momentum / trix_momentum.rolling(window=20).std().replace(0, 0.001)  # TODO: 将魔法数字提取到配置中
        score += np.where(trix_momentum > 0, 
                        np.minimum(normalized_momentum * 3, 10),  # TODO: 将魔法数字提取到配置中 
                        np.maximum(normalized_momentum * 3, -10))  # TODO: 将魔法数字提取到配置中
        
        # 4. TRIX斜率评分 (±10分)  # TODO: 将魔法数字提取到配置中
        # 斜率为正看涨，斜率为负看跌
        score += np.where(trix_slope > 0, 
                        np.minimum(trix_slope * 50, 10),  # TODO: 将魔法数字提取到配置中 
                        np.maximum(trix_slope * 50, -10))  # TODO: 将魔法数字提取到配置中
        
        # 5. 零轴交叉评分 (±15分)  # TODO: 将魔法数字提取到配置中
        if not zero_cross.empty:
            # 向上交叉零轴
            upward_cross = zero_cross.get('zero_cross_up', pd.Series(False, index=score.index))
            if isinstance(upward_cross, pd.Series) and not upward_cross.empty:
                cross_quality_up = zero_cross.get('cross_quality', pd.Series(50, index=score.index))  # TODO: 将魔法数字提取到配置中
                score.loc[upward_cross] += np.minimum((cross_quality_up.loc[upward_cross] - 50) / 10 * 15, 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
            # 向下交叉零轴
            downward_cross = zero_cross.get('zero_cross_down', pd.Series(False, index=score.index))
            if isinstance(downward_cross, pd.Series) and not downward_cross.empty:
                cross_quality_down = zero_cross.get('cross_quality', pd.Series(50, index=score.index))  # TODO: 将魔法数字提取到配置中
                score.loc[downward_cross] -= np.minimum((cross_quality_down.loc[downward_cross] - 50) / 10 * 15, 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 6. 背离评分 (±15分)  # TODO: 将魔法数字提取到配置中
        if not divergence.empty:
            # 牛市背离
            bullish_div = divergence.get('bullish_divergence', pd.Series(False, index=score.index))
            if isinstance(bullish_div, pd.Series) and not bullish_div.empty:
                div_strength = divergence.get('divergence_strength', pd.Series(0.5, index=score.index))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                score.loc[bullish_div] += np.minimum(div_strength.loc[bullish_div] * 30, 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
            # 熊市背离
            bearish_div = divergence.get('bearish_divergence', pd.Series(False, index=score.index))
            if isinstance(bearish_div, pd.Series) and not bearish_div.empty:
                div_strength = divergence.get('divergence_strength', pd.Series(0.5, index=score.index))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                score.loc[bearish_div] -= np.minimum(div_strength.loc[bearish_div] * 30, 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 7. 多周期协同评分 (±15分)  # TODO: 将魔法数字提取到配置中
        if not synergy.empty:
            bull_synergy = synergy.get('bullish_agreement', pd.Series(False, index=score.index))
            if isinstance(bull_synergy, pd.Series) and not bull_synergy.empty:
                synergy_strength = synergy.get('synergy_strength', pd.Series(0.5, index=score.index))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                score.loc[bull_synergy] += np.minimum(synergy_strength.loc[bull_synergy] * 30, 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            
            bear_synergy = synergy.get('bearish_agreement', pd.Series(False, index=score.index))
            if isinstance(bear_synergy, pd.Series) and not bear_synergy.empty:
                synergy_strength = synergy.get('synergy_strength', pd.Series(0.5, index=score.index))  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                score.loc[bear_synergy] -= np.minimum(synergy_strength.loc[bear_synergy] * 30, 15)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 8. 特殊形态评分 (±10分)  # TODO: 将魔法数字提取到配置中
        if not patterns.empty:
            # 看涨形态
            for pattern in ['hook_bottom', 'bottom_reversal', 'breakout_up']:
                if pattern in patterns.columns:
                    pattern_signal = patterns[pattern]
                    if isinstance(pattern_signal, pd.Series) and not pattern_signal.empty:
                        score.loc[pattern_signal] += 10
            
            # 看跌形态
            for pattern in ['hook_top', 'top_reversal', 'breakout_down']:
                if pattern in patterns.columns:
                    pattern_signal = patterns[pattern]
                    if isinstance(pattern_signal, pd.Series) and not pattern_signal.empty:
                        score.loc[pattern_signal] -= 10
        
        # 9. 市场环境调整  # TODO: 将魔法数字提取到配置中
        if self.market_environment == "bull_market":
            # 牛市中增强多头信号，弱化空头信号
            bull_adjustment = np.where(score > 50, (score - 50) * 0.2, (score - 50) * 0.1)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score += bull_adjustment
        elif self.market_environment == "bear_market":
            # 熊市中增强空头信号，弱化多头信号
            bear_adjustment = np.where(score < 50, (50 - score) * 0.2, (50 - score) * 0.1)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score -= bear_adjustment
        elif self.market_environment == "volatile_market":
            # 高波动市场需要更强的信号
            vol_adjustment = (score - 50).abs() * 0.3  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            score = np.where(score > 50, 50 + vol_adjustment, 50 - vol_adjustment)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        
        # 限制分数范围在0-100之间
        score = score.clip(0, 100)
        
        return score
    
    def calculate_raw_score_Trix_Enhanced_Trix(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算增强型TRIX指标原始评分 (0-100分)
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            pd.Series: 评分序列，取值范围0-100
        """
        # 直接使用现有的calculate_score方法
        if not self.has_result():
            self.calculate(data)
        
        return self.calculate_score_Trix_Enhanced_Trix()
    
    def generate_signals_Trix_Enhanced_Trix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含交易信号的Data_frame
        """
        # 确保已计算TRIX
        if self._result is None:
            self.calculate(data)
            
        if self._result is None:
            return pd.DataFrame()
            
        # 获取TRIX数据
        trix = self._result['TRIX']
        
        # 计算TRIX综合评分
        score = self.calculate_score_Trix_Enhanced_Trix()
        
        # 识别形态
        patterns = self.identify_patterns_Trix_Enhanced_Trix()
        
        # 创建信号DataFrame
        signals = pd.DataFrame(index=self._result.index)
        signals['trix'] = trix
        signals['score'] = score
        
        # 生成基础信号
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['neutral_signal'] = True
        
        # 买入信号条件
        buy_conditions = [
            (score > 70),  # 评分高于70  # TODO: 将魔法数字提取到配置中
            patterns.get('cross_up_zero', pd.Series(False, index=self._result.index)),  # 零轴上穿
            patterns.get('golden_cross', pd.Series(False, index=self._result.index)) & (trix > 0),  # 金叉且在零轴上方
            patterns.get('bullish_divergence', pd.Series(False, index=self._result.index)),  # 正背离
            patterns.get('multi_period_bullish_signal', pd.Series(False, index=self._result.index))  # 多周期看涨信号
        ]
        
        # 卖出信号条件
        sell_conditions = [
            (score < 30),  # 评分低于30  # TODO: 将魔法数字提取到配置中
            patterns.get('cross_down_zero', pd.Series(False, index=self._result.index)),  # 零轴下穿
            patterns.get('death_cross', pd.Series(False, index=self._result.index)) & (trix < 0),  # 死叉且在零轴下方
            patterns.get('bearish_divergence', pd.Series(False, index=self._result.index)),  # 负背离
            patterns.get('multi_period_bearish_signal', pd.Series(False, index=self._result.index))  # 多周期看跌信号
        ]
        
        # 合并买入信号
        for condition in buy_conditions:
            signals['buy_signal'] = signals['buy_signal'] | condition
        
        # 合并卖出信号
        for condition in sell_conditions:
            signals['sell_signal'] = signals['sell_signal'] | condition
        
        # 处理买卖信号冲突
        conflict = signals['buy_signal'] & signals['sell_signal']
        if conflict.any():
            # 使用评分解决冲突
            signals.loc[conflict & (score >= 50), 'sell_signal'] = False  # TODO: 将魔法数字提取到配置中
            signals.loc[conflict & (score < 50), 'buy_signal'] = False  # TODO: 将魔法数字提取到配置中
        
        # 更新中性信号
        signals['neutral_signal'] = ~(signals['buy_signal'] | signals['sell_signal'])
        
        # 添加信号描述和类型
        signals['signal_type'] = ''
        signals['signal_desc'] = ''
        signals['trend'] = 0  # 默认中性
        
        # 买入信号类型和描述
        buy_signals = signals['buy_signal']
        signals.loc[buy_signals & patterns.get('cross_up_zero', False), 'signal_type'] = 'TRIX零轴上穿'
        signals.loc[buy_signals & patterns.get('cross_up_zero', False), 'signal_desc'] = 'TRIX上穿零轴，显示转入上升趋势'
        
        signals.loc[buy_signals & patterns.get('golden_cross', False), 'signal_type'] = 'TRIX金叉'
        signals.loc[buy_signals & patterns.get('golden_cross', False), 'signal_desc'] = 'TRIX上穿信号线，显示短期上升动能增强'
        
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_type'] = 'TRIX正背离'
        signals.loc[buy_signals & patterns.get('bullish_divergence', False), 'signal_desc'] = '价格创新低但TRIX未创新低，显示下跌动能减弱'
        
        signals.loc[buy_signals & patterns.get('multi_period_bullish_signal', False), 'signal_type'] = 'TRIX多周期看涨'
        signals.loc[buy_signals & patterns.get('multi_period_bullish_signal', False), 'signal_desc'] = '多周期TRIX共振发出看涨信号'
        
        # 卖出信号类型和描述
        sell_signals = signals['sell_signal']
        signals.loc[sell_signals & patterns.get('cross_down_zero', False), 'signal_type'] = 'TRIX零轴下穿'
        signals.loc[sell_signals & patterns.get('cross_down_zero', False), 'signal_desc'] = 'TRIX下穿零轴，显示转入下降趋势'
        
        signals.loc[sell_signals & patterns.get('death_cross', False), 'signal_type'] = 'TRIX死叉'
        signals.loc[sell_signals & patterns.get('death_cross', False), 'signal_desc'] = 'TRIX下穿信号线，显示短期下降动能增强'
        
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_type'] = 'TRIX负背离'
        signals.loc[sell_signals & patterns.get('bearish_divergence', False), 'signal_desc'] = '价格创新高但TRIX未创新高，显示上涨动能减弱'
        
        signals.loc[sell_signals & patterns.get('multi_period_bearish_signal', False), 'signal_type'] = 'TRIX多周期看跌'
        signals.loc[sell_signals & patterns.get('multi_period_bearish_signal', False), 'signal_desc'] = '多周期TRIX共振发出看跌信号'
        
        # 设置趋势方向
        signals['trend'] = 0  # 默认中性
        signals.loc[buy_signals, 'trend'] = 1  # 上升趋势
        signals.loc[sell_signals, 'trend'] = -1  # 下降趋势
        
        # 计算信号置信度
        signals['confidence'] = self._calculate_signal_confidence(signals, patterns)
        
        # 计算建议止损价
        if 'close' in data.columns:
            signals['stop_loss'] = self._calculate_stop_loss(data, signals)
        
        # 添加市场环境信息
        signals['market_env'] = self.market_environment
        
        return signals
    
    def _calculate_signal_confidence(self, signals: pd.DataFrame, patterns: pd.DataFrame) -> pd.Series:
        """
        计算信号置信度
        
        Args:
            signals: 信号Data_frame
            patterns: 形态Data_frame
            
        Returns:
            pd.Series: 信号置信度 (0-100)
        """
        confidence = pd.Series(50, index=signals.index)  # TODO: 将魔法数字提取到配置中
        
        # 根据评分计算基础置信度
        score = signals['score']
        
        # 高评分对应高置信度
        confidence_from_score = np.where(score > 50, 50 + (score - 50) * 0.8, 50 - (50 - score) * 0.8)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        confidence = confidence_from_score
        
        # 增强型形态提高置信度
        for pattern, boost in [
            ('bullish_divergence', 15),  # TODO: 将魔法数字提取到配置中
            ('bearish_divergence', 15),  # TODO: 将魔法数字提取到配置中
            ('hidden_bullish_divergence', 10),
            ('hidden_bearish_divergence', 10),
            ('high_quality_cross_up_zero', 20),  # TODO: 将魔法数字提取到配置中
            ('high_quality_cross_down_zero', 20),  # TODO: 将魔法数字提取到配置中
            ('multi_period_bullish_signal', 15),  # TODO: 将魔法数字提取到配置中
            ('multi_period_bearish_signal', 15),  # TODO: 将魔法数字提取到配置中
            ('strong_bullish_consensus', 10),
            ('strong_bearish_consensus', 10)
        ]:
            if pattern in patterns.columns:
                confidence = np.where(patterns[pattern], np.minimum(100, confidence + boost), confidence)
        
        # 零轴位置对确信度的影响
        trix_values = self._result['TRIX']
        
        # TRIX远离零轴时信号更可靠
        confidence += np.where(abs(trix_values) > 0.5, 5, 0)  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
        confidence += np.where(abs(trix_values) > 1.0, 5, 0)  # TODO: 将魔法数字提取到配置中
        
        return confidence
    
    def _calculate_stop_loss(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """
        计算建议止损价
        
        Args:
            data: 价格数据
            signals: 信号Data_frame
            
        Returns:
            pd.Series: 建议止损价
        """
        stop_loss = pd.Series(np.nan, index=signals.index)
        
        # 获取价格数据
        close = data['close']
        
        if 'low' in data.columns:
            low = data['low']
        else:
            low = close
        
        # 计算ATR (如果可能)
        atr = None
        if 'high' in data.columns and 'low' in data.columns:
            high = data['high']
            atr = self.atr_Trix(high, low, close, 14)  # TODO: 将魔法数字提取到配置中
        
        # 买入信号的止损
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                current_close = close.iloc[i]
                
                if atr is not None and i < len(atr) and not np.isnan(atr.iloc[i]):
                    # 使用ATR计算动态止损
                    atr_value = atr.iloc[i]
                    confidence = signals['confidence'].iloc[i]
                    
                    # 根据信号置信度调整ATR倍数
                    if confidence >= 80:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        atr_multiplier = 2.0  # 高置信度，较宽止损
                    elif confidence >= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        atr_multiplier = 1.5  # TODO: 将魔法数字提取到配置中  # 中等置信度，中等止损  # TODO: 将魔法数字提取到配置中
                    else:
                        atr_multiplier = 1.0  # 低置信度，紧止损
                    
                    stop_loss.iloc[i] = current_close - (atr_value * atr_multiplier)
                else:
                    # 使用最近低点作为止损
                    if i >= 5:  # TODO: 将魔法数字提取到配置中
                        recent_low = low.iloc[i-5:i+1].min()  # TODO: 将魔法数字提取到配置中
                        stop_loss.iloc[i] = recent_low * 0.99  # 微调1%  # TODO: 将魔法数字提取到配置中
        
        # 卖出信号的止损 (反向操作的止损位)
        for i in range(len(signals)):
            if signals['sell_signal'].iloc[i]:
                current_close = close.iloc[i]
                
                if atr is not None and i < len(atr) and not np.isnan(atr.iloc[i]):
                    # 使用ATR计算动态止损
                    atr_value = atr.iloc[i]
                    confidence = signals['confidence'].iloc[i]
                    
                    # 根据信号置信度调整ATR倍数
                    if confidence >= 80:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        atr_multiplier = 2.0
                    elif confidence >= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
                        atr_multiplier = 1.5  # TODO: 将魔法数字提取到配置中
                    else:
                        atr_multiplier = 1.0
                    
                    stop_loss.iloc[i] = current_close + (atr_value * atr_multiplier)
                else:
                    # 使用最近高点作为止损
                    if 'high' in data.columns and i >= 5:  # TODO: 将魔法数字提取到配置中
                        high = data['high']
                        recent_high = high.iloc[i-5:i+1].max()  # TODO: 将魔法数字提取到配置中
                        stop_loss.iloc[i] = recent_high * 1.01  # 微调1%
        
        return stop_loss

    def set_parameters_Trix_Enhanced_Trix(self, **kwargs):
        """
        设置指标参数

        Args:
            **kwargs: 参数字典
        """
        if 'n' in kwargs:
            self.n = kwargs['n']
        if 'm' in kwargs:
            self.m = kwargs['m']
        if 'secondary_n' in kwargs:
            self.secondary_n = kwargs['secondary_n']
        if 'adaptive_period' in kwargs:
            self.adaptive_period = kwargs['adaptive_period']
        if 'volatility_lookback' in kwargs:
            self.volatility_lookback = kwargs['volatility_lookback']

    def calculate_confidence_Trix_Enhanced_Trix(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算Enhanced_tRIX指标的置信度

        Args:
            score: 得分序列
            patterns: 检测到的形态Data_frame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        if score.empty:
            return 0.5  # TODO: 将魔法数字提取到配置中

        # 基础置信度
        confidence = 0.5  # TODO: 将魔法数字提取到配置中

        # 1. 基于评分的置信度
        last_score = score.iloc[-1]

        # 极端评分置信度较高
        if last_score > 80 or last_score < 20:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.25  # TODO: 将魔法数字提取到配置中
        # 中性评分置信度中等
        elif 40 <= last_score <= 60:  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            confidence += 0.1
        else:
            confidence += 0.15  # TODO: 将魔法数字提取到配置中

        # 2. 基于形态的置信度
        if not patterns.empty:
            # 检查EnhancedTRIX形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.05, 0.2)  # TODO: 将魔法数字提取到配置中

        # 3. 基于信号的置信度  # TODO: 将魔法数字提取到配置中
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.1, 0.15)  # TODO: 将魔法数字提取到配置中

        # 4. 基于评分趋势的置信度  # TODO: 将魔法数字提取到配置中
        if len(score) >= 3:  # TODO: 将魔法数字提取到配置中
            recent_scores = score.iloc[-3:]  # TODO: 将魔法数字提取到配置中
            trend = recent_scores.iloc[-1] - recent_scores.iloc[0]

            # 明确的趋势增加置信度
            if abs(trend) > 10:
                confidence += 0.05  # TODO: 将魔法数字提取到配置中

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def get_patterns_Trix_Enhanced_Trix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取Enhanced_tRIX相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的Data_frame
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None:
            return pd.DataFrame(index=data.index)

        # 使用现有的identify_patterns方法
        return self.identify_patterns_Trix_Enhanced_Trix()

    def register_patterns_Trix_Enhanced_Trix(self):
        """
        注册Enhanced_tRIX指标的形态到全局形态注册表
        """
        # 注册TRIX交叉形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_GOLDEN_CROSS",
            display_name="TRIX金叉",
            description="TRIX线上穿信号线，表明上升趋势开始",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_DEATH_CROSS",
            display_name="TRIX死叉",
            description="TRIX线下穿信号线，表明下降趋势开始",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-20.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册TRIX零轴穿越形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_ZERO_CROSS_UP",
            display_name="TRIX零轴上穿",
            description="TRIX从下方穿越零轴，表明趋势转为看涨",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_ZERO_CROSS_DOWN",
            display_name="TRIX零轴下穿",
            description="TRIX从上方穿越零轴，表明趋势转为看跌",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册TRIX背离形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_BULLISH_DIVERGENCE",
            display_name="TRIX看涨背离",
            description="价格创新低但TRIX未创新低，表明下跌动能减弱",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=25.0,  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_BEARISH_DIVERGENCE",
            display_name="TRIX看跌背离",
            description="价格创新高但TRIX未创新高，表明上涨动能减弱",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-25.0,  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册TRIX多周期协同形态
        self.register_pattern_to_registry(
            pattern_id="TRIX_MULTI_PERIOD_BULLISH",
            display_name="TRIX多周期看涨",
            description="多周期TRIX共振发出看涨信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_MULTI_PERIOD_BEARISH",
            display_name="TRIX多周期看跌",
            description="多周期TRIX共振发出看跌信号",
            pattern_type="BEARISH",
            default_strength="STRONG",
            score_impact=-30.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        # 注册TRIX状态形态（从centralized mapping迁移）
        self.register_pattern_to_registry(
            pattern_id="TRIX_ABOVE_ZERO",
            display_name="TRIX零轴上方",
            description="TRIX位于零轴上方，长期趋势偏多",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_BELOW_ZERO",
            display_name="TRIX零轴下方",
            description="TRIX位于零轴下方，长期趋势偏空",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_RISING",
            display_name="TRIX上升",
            description="TRIX指标上升，长期动量增强",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_FALLING",
            display_name="TRIX下降",
            description="TRIX指标下降，长期动量减弱",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-15.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_ACCELERATION",
            display_name="TRIX加速上升",
            description="TRIX指标加速上升，表明价格上涨动能不断增强",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=20.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_DECELERATION",
            display_name="TRIX减速",
            description="TRIX指标减速变化，动能转变",
            pattern_type="NEUTRAL",
            default_strength="MEDIUM",
            score_impact=0.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_STRONG_BULLISH_CONSENSUS",
            display_name="TRIX强烈看涨共振",
            description="TRIX多重信号共振，形成强烈看涨态势",
            pattern_type="BULLISH",
            default_strength="VERY_STRONG",
            score_impact=30.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="TRIX_STRONG_BEARISH_CONSENSUS",
            display_name="TRIX强烈看跌共振",
            description="TRIX多重信号共振，形成强烈看跌态势",
            pattern_type="BEARISH",
            default_strength="VERY_STRONG",
            score_impact=-30.0,  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            polarity="NEGATIVE"
        )

    def generate_trading_signals_Trix(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """
        生成Enhanced_tRIX交易信号

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            Dict[str, pd.Series]: 包含买卖信号的字典
        """
        # 确保已计算指标
        if self._result is None:
            self.calculate(data)

        if self._result is None:
            return {
                'buy_signal': pd.Series(False, index=data.index),
                'sell_signal': pd.Series(False, index=data.index),
                'signal_strength': pd.Series(0.0, index=data.index)
            }

        trix = self._result['TRIX']
        matrix = self._result['MATRIX']

        # 生成信号
        buy_signal = pd.Series(False, index=data.index)
        sell_signal = pd.Series(False, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

        # 1. TRIX金叉死叉信号
        golden_cross = crossover(trix, matrix)
        death_cross = crossunder(trix, matrix)

        buy_signal |= golden_cross
        sell_signal |= death_cross
        signal_strength += golden_cross * 0.7  # TODO: 将魔法数字提取到配置中
        signal_strength += death_cross * 0.7  # TODO: 将魔法数字提取到配置中

        # 2. TRIX零轴穿越信号
        zero_cross_up = crossover(trix, 0)
        zero_cross_down = crossunder(trix, 0)

        buy_signal |= zero_cross_up
        sell_signal |= zero_cross_down
        signal_strength += zero_cross_up * 0.8  # TODO: 将魔法数字提取到配置中
        signal_strength += zero_cross_down * 0.8  # TODO: 将魔法数字提取到配置中

        # 3. 高质量零轴交叉信号  # TODO: 将魔法数字提取到配置中
        zero_cross_quality = self.evaluate_zero_cross_quality()
        if not zero_cross_quality.empty and 'cross_quality_score' in zero_cross_quality.columns:
            high_quality_up = zero_cross_up & (zero_cross_quality['cross_quality_score'] > 70)  # TODO: 将魔法数字提取到配置中
            high_quality_down = zero_cross_down & (zero_cross_quality['cross_quality_score'] > 70)  # TODO: 将魔法数字提取到配置中

            buy_signal |= high_quality_up
            sell_signal |= high_quality_down
            signal_strength += high_quality_up * 1.0
            signal_strength += high_quality_down * 1.0

        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'signal_strength': signal_strength
        }

    def atr_Trix(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:  # TODO: 将魔法数字提取到配置中
        """
        计算平均真实范围(ATR)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期

        Returns:
            pd.Series: ATR序列
        """
        # 计算真实范围
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = tr.rolling(window=period).mean()

        return atr

    def get_pattern_info_Trix_Enhanced_Trix(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息

        Args:
            pattern_id: 形态ID

        Returns:
            dict: 形态详细信息
        """
        pattern_info_map = {
            "TRIX_GOLDEN_CROSS": {
                "id": "TRIX_GOLDEN_CROSS",
                "name": "TRIX金叉",
                "description": "TRIX线上穿信号线，表明上升趋势开始",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 20.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_DEATH_CROSS": {
                "id": "TRIX_DEATH_CROSS",
                "name": "TRIX死叉",
                "description": "TRIX线下穿信号线，表明下降趋势开始",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -20.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_ZERO_CROSS_UP": {
                "id": "TRIX_ZERO_CROSS_UP",
                "name": "TRIX零轴上穿",
                "description": "TRIX从下方穿越零轴，表明趋势转为看涨",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 15.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_ZERO_CROSS_DOWN": {
                "id": "TRIX_ZERO_CROSS_DOWN",
                "name": "TRIX零轴下穿",
                "description": "TRIX从上方穿越零轴，表明趋势转为看跌",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -15.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_BULLISH_DIVERGENCE": {
                "id": "TRIX_BULLISH_DIVERGENCE",
                "name": "TRIX看涨背离",
                "description": "价格创新低但TRIX未创新低，表明下跌动能减弱",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 25.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_BEARISH_DIVERGENCE": {
                "id": "TRIX_BEARISH_DIVERGENCE",
                "name": "TRIX看跌背离",
                "description": "价格创新高但TRIX未创新高，表明上涨动能减弱",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -25.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_MULTI_PERIOD_BULLISH": {
                "id": "TRIX_MULTI_PERIOD_BULLISH",
                "name": "TRIX多周期看涨",
                "description": "多周期TRIX共振发出看涨信号",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 30.0  # TODO: 将魔法数字提取到配置中
            },
            "TRIX_MULTI_PERIOD_BEARISH": {
                "id": "TRIX_MULTI_PERIOD_BEARISH",
                "name": "TRIX多周期看跌",
                "description": "多周期TRIX共振发出看跌信号",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -30.0  # TODO: 将魔法数字提取到配置中
            }
        }

        return pattern_info_map.get(pattern_id, {
            "id": pattern_id,
            "name": "TRIX趋势转折",
            "description": f"基于TRIX指标的趋势转折分析: {pattern_id}",
            "type": "NEUTRAL",
            "strength": "WEAK",
            "score_impact": 0.0
        })
    @property
    def minimum_periods(self) -> int:
        """
        EnhancedTrix指标所需的最少数据周期数

        计算逻辑：基于参数 smoothing_period(3) 计算  # TODO: 将魔法数字提取到配置中

        Returns:
            int: 最少需要的数据周期数
        """
        # 确保_parameters存在，如果不存在则使用默认值
        if not hasattr(self, '_parameters') or not self._parameters:
            return 40  # 返回默认的最小周期数  # TODO: 将魔法数字提取到配置中

        smoothing_period = self._parameters.get('smoothing_period', 3)  # TODO: 将魔法数字提取到配置中
        return smoothing_period + max(10, smoothing_period // 2)

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Enhanced TRIX指标的主要入口方法

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced TRIX指标的DataFrame
        """
        return self.calculate_Trix_Enhanced_Trix(data, **kwargs)

    def calculate_Trix_Enhanced_Trix(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算Enhanced TRIX指标

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced TRIX指标的DataFrame
        """
        return self._calculate_enhancedtrix(data)

    def _get_default_parameters_enhancedtrix(self) -> dict:
        """
        获取Enhanced TRIX指标的默认参数

        Returns:
            dict: 默认参数字典
        """
        return {
            'n': 12,  # TODO: 将魔法数字提取到配置中
            'm': 9,  # TODO: 将魔法数字提取到配置中
            'secondary_n': 24,  # TODO: 将魔法数字提取到配置中
            'multi_periods': [6, 12, 24],  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中  # TODO: 将魔法数字提取到配置中
            'adaptive_period': True,
            'volatility_lookback': 20,  # TODO: 将魔法数字提取到配置中
            'use_smoothed_trix': True,
            'smoothing_period': 3  # TODO: 将魔法数字提取到配置中
        }

    def _calculate_baseindicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的抽象方法实现

        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 其他参数

        Returns:
            包含Enhanced TRIX指标的DataFrame
        """
        return self._calculate_enhancedtrix(data)

    def calculate_confidence_Indicator_Base_Indicator(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        BaseIndicator要求的置信度计算方法

        Args:
            score: 得分序列
            patterns: 检测到的形态DataFrame
            signals: 生成的信号字典

        Returns:
            float: 置信度分数 (0-1)
        """
        return self.calculate_confidence_Trix_Enhanced_Trix(score, patterns, signals)

    def calculate_raw_score_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        BaseIndicator要求的原始评分计算方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.Series: 原始评分序列
        """
        return self.calculate_raw_score_Trix_Enhanced_Trix(data, **kwargs)

    def get_patterns_Indicator_Base_Indicator(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        BaseIndicator要求的形态获取方法

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 形态DataFrame
        """
        return self.get_patterns_Trix_Enhanced_Trix(data, **kwargs)

    def set_parameters_Indicator_Base_Indicator(self, **kwargs):
        """
        BaseIndicator要求的参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Trix_Enhanced_Trix(**kwargs)

    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        BaseIndicator要求的默认参数获取方法

        Returns:
            dict: 默认参数字典
        """
        return self._get_default_parameters_enhancedtrix()

    def set_parameters(self, **kwargs):
        """
        标准参数设置方法

        Args:
            **kwargs: 参数字典
        """
        self.set_parameters_Trix_Enhanced_Trix(**kwargs)

    def get_signal(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        【核心抽象方法2】基于Enhanced TRIX指标数值生成最新的交易信号
        
        Enhanced TRIX交易信号逻辑（融合多重增强特性）：
        - 基础TRIX信号：零轴穿越、金叉死叉信号
        - 自适应周期调整：根据市场波动率优化信号
        - 多周期协同：结合主周期和次要周期TRIX
        - 背离检测：价格与TRIX背离信号
        - 零轴交叉质量评估：评估穿越的可靠性
        - 趋势强度分析：基于TRIX斜率和动量
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 标准化交易信号格式
        """
        try:
            # 1. 数据验证
            if not self._validate_signal_data(data):
                return self._get_default_signal("数据验证失败")
            
            # 2. 确保已计算指标
            if not self.has_result():
                self.calculate(data, **kwargs)

            if self._result is None or len(self._result) == 0:
                return self._get_default_signal("Enhanced TRIX计算结果为空")

            # 3. 获取最新数据
            latest_close = data['close'].iloc[-1]
            
            # 4. 获取Enhanced TRIX相关值
            if len(self._result) < 2:
                return self._get_default_signal("Enhanced TRIX数据不足")
                
            # 检查必要的列是否存在
            required_columns = ['TRIX', 'MATRIX']
            if not all(col in self._result.columns for col in required_columns):
                return self._get_default_signal("Enhanced TRIX结果列不完整")
                
            latest_trix = self._result['TRIX'].iloc[-1]
            latest_matrix = self._result['MATRIX'].iloc[-1]
            prev_trix = self._result['TRIX'].iloc[-2]
            prev_matrix = self._result['MATRIX'].iloc[-2]
            
            # 获取增强特性数据
            latest_trix_secondary = self._result.get('trix_secondary', pd.Series([latest_trix])).iloc[-1] if 'trix_secondary' in self._result.columns else latest_trix
            latest_trix_momentum = self._result.get('trix_momentum', pd.Series([0])).iloc[-1] if 'trix_momentum' in self._result.columns else 0
            latest_trix_slope = self._result.get('trix_slope', pd.Series([0])).iloc[-1] if 'trix_slope' in self._result.columns else 0
            latest_trix_volatility = self._result.get('trix_volatility', pd.Series([1])).iloc[-1] if 'trix_volatility' in self._result.columns else 1
            
            # 5. Enhanced TRIX信号生成逻辑
            signal_type = "hold"
            strength = 0.0
            confidence = 0.5
            reason = "无明确信号"
            metadata = {}
            
            # 计算增强信号强度因子
            # 自适应周期因子：自适应周期偏离原始周期越多，信号越强
            adaptive_factor = 1.0
            if hasattr(self, '_adaptive_n') and hasattr(self, 'n'):
                period_deviation = abs(self._adaptive_n - self.n) / self.n
                adaptive_factor = min(1.2, max(0.8, 1 + period_deviation * 0.5))
            
            # 波动率因子：低波动率时信号更可靠
            volatility_factor = min(1.2, max(0.8, 2 / (1 + latest_trix_volatility)))
            
            # 斜率因子：斜率越大，趋势越强
            slope_factor = min(1.3, max(0.7, 1 + abs(latest_trix_slope) / 100))
            
            # 零轴穿越信号（最高优先级，Enhanced版本）
            if latest_trix > 0 and prev_trix <= 0:
                # Enhanced TRIX上穿零轴 - 强烈买入信号
                signal_type = "buy"
                base_strength = 0.9
                
                # 多周期确认
                if latest_trix_secondary > latest_trix_secondary - 0.1:  # 次要周期也在上升
                    base_strength += 0.05
                
                # 零轴交叉质量评估
                cross_quality = min(1.0, abs(latest_trix) / 0.5)  # TRIX离零轴越远，穿越质量越高
                base_strength *= (0.8 + 0.2 * cross_quality)
                
                strength = base_strength * adaptive_factor * volatility_factor
                confidence = 0.9
                reason = f"Enhanced TRIX上穿零轴({latest_trix:.4f})，多重确认强烈买入信号"
                
            elif latest_trix < 0 and prev_trix >= 0:
                # Enhanced TRIX下穿零轴 - 强烈卖出信号
                signal_type = "sell"
                base_strength = 0.9
                
                # 多周期确认
                if latest_trix_secondary < latest_trix_secondary + 0.1:  # 次要周期也在下降
                    base_strength += 0.05
                
                # 零轴交叉质量评估
                cross_quality = min(1.0, abs(latest_trix) / 0.5)
                base_strength *= (0.8 + 0.2 * cross_quality)
                
                strength = base_strength * adaptive_factor * volatility_factor
                confidence = 0.9
                reason = f"Enhanced TRIX下穿零轴({latest_trix:.4f})，多重确认强烈卖出信号"
                
            # 金叉死叉信号（增强版）
            elif latest_trix > latest_matrix and prev_trix <= prev_matrix:
                # Enhanced TRIX金叉MATRIX - 买入信号
                signal_type = "buy"
                base_strength = 0.8
                
                # 零轴位置确认：在零轴上方的金叉更强
                if latest_trix > 0:
                    base_strength += 0.1
                    reason_prefix = "Enhanced TRIX在零轴上方金叉"
                else:
                    reason_prefix = "Enhanced TRIX金叉"
                
                # 斜率确认：上升斜率增强信号
                if latest_trix_slope > 0.01:
                    base_strength += 0.05
                
                strength = base_strength * slope_factor * volatility_factor
                confidence = 0.85
                reason = f"{reason_prefix}MATRIX({latest_trix:.4f}>{latest_matrix:.4f})，斜率确认买入信号"
                
            elif latest_trix < latest_matrix and prev_trix >= prev_matrix:
                # Enhanced TRIX死叉MATRIX - 卖出信号
                signal_type = "sell"
                base_strength = 0.8
                
                # 零轴位置确认：在零轴下方的死叉更强
                if latest_trix < 0:
                    base_strength += 0.1
                    reason_prefix = "Enhanced TRIX在零轴下方死叉"
                else:
                    reason_prefix = "Enhanced TRIX死叉"
                
                # 斜率确认：下降斜率增强信号
                if latest_trix_slope < -0.01:
                    base_strength += 0.05
                
                strength = base_strength * slope_factor * volatility_factor
                confidence = 0.85
                reason = f"{reason_prefix}MATRIX({latest_trix:.4f}<{latest_matrix:.4f})，斜率确认卖出信号"
            
            # 趋势持续信号（增强版）
            if signal_type == "hold":
                trix_trend_strength = abs(latest_trix)
                
                if latest_trix > 0 and latest_trix > latest_matrix:
                    # Enhanced TRIX在零轴上方且高于信号线 - 持续买入
                    signal_type = "buy"
                    base_strength = 0.6
                    
                    # 趋势强度调整
                    if trix_trend_strength > 1.0:  # 强趋势
                        base_strength += min(0.2, trix_trend_strength / 10)
                    
                    # 多周期协同确认
                    if latest_trix_secondary > 0 and abs(latest_trix - latest_trix_secondary) < 0.5:
                        base_strength += 0.1
                    
                    strength = base_strength * adaptive_factor
                    confidence = 0.75
                    reason = f"Enhanced TRIX强势上升趋势({latest_trix:.4f})，多重确认持续买入"
                    
                elif latest_trix < 0 and latest_trix < latest_matrix:
                    # Enhanced TRIX在零轴下方且低于信号线 - 持续卖出
                    signal_type = "sell"
                    base_strength = 0.6
                    
                    # 趋势强度调整
                    if trix_trend_strength > 1.0:  # 强趋势
                        base_strength += min(0.2, trix_trend_strength / 10)
                    
                    # 多周期协同确认
                    if latest_trix_secondary < 0 and abs(latest_trix - latest_trix_secondary) < 0.5:
                        base_strength += 0.1
                    
                    strength = base_strength * adaptive_factor
                    confidence = 0.75
                    reason = f"Enhanced TRIX强势下降趋势({latest_trix:.4f})，多重确认持续卖出"
            
            # 计算Enhanced TRIX趋势变化
            trix_rising = latest_trix > prev_trix
            trix_momentum_change = latest_trix_momentum
            trix_matrix_spread = latest_trix - latest_matrix
            
            # 设置增强元数据
            metadata = {
                'trix_value': latest_trix,
                'matrix_value': latest_matrix,
                'trix_secondary': latest_trix_secondary,
                'trix_momentum': latest_trix_momentum,
                'trix_slope': latest_trix_slope,
                'trix_volatility': latest_trix_volatility,
                'trix_trend': 'rising' if trix_rising else 'falling',
                'trix_matrix_spread': trix_matrix_spread,
                'zero_position': 'above' if latest_trix > 0 else 'below',
                'adaptive_factor': adaptive_factor,
                'volatility_factor': volatility_factor,
                'slope_factor': slope_factor,
                'trend_strength': 'strong' if abs(latest_trix) > 1.0 else 'moderate' if abs(latest_trix) > 0.5 else 'weak',
                'multi_period_sync': abs(latest_trix - latest_trix_secondary) < 0.5,
                'cross_quality': min(1.0, abs(latest_trix) / 0.5)
            }
            
            # 检测背离模式（Enhanced特有）
            if len(self._result) >= 15:
                divergence_detected = self._detect_enhanced_trix_divergence(data)
                if divergence_detected:
                    metadata['divergence_detected'] = True
                    # 背离信号调整
                    if signal_type == "hold":
                        signal_type = "sell" if latest_trix > 0 else "buy"
                        strength = 0.75 * volatility_factor
                        confidence = 0.8
                        reason = "检测到Enhanced TRIX背离，多重确认反转信号"
                    else:
                        # 增强现有信号
                        strength = min(1.0, strength + 0.1)
                        confidence = min(1.0, confidence + 0.05)
                        reason += "（背离确认）"
            
            # 6. 标准化输出
            return {
                'signal_type': signal_type,
                'strength': max(0.0, min(1.0, strength)),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': pd.Timestamp.now(),
                'reason': reason,
                'metadata': {
                    'latest_close': latest_close,
                    **metadata
                }
            }

        except Exception as e:
            logger.warning(f"Enhanced TRIX信号生成失败: {e}")
            return self._get_default_signal(f"信号生成失败: {str(e)}")

    def _validate_signal_data(self, data: pd.DataFrame) -> bool:
        """
        验证信号生成所需的数据
        
        Args:
            data: 输入数据DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False
            
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # Enhanced TRIX需要更多数据（由于三重指数平滑和多周期特性）
        min_periods = max(
            getattr(self, 'n', 12) * 3,
            getattr(self, 'secondary_n', 24) * 3,
            getattr(self, 'volatility_lookback', 20)
        ) + 15
        if len(data) < min_periods:
            return False
            
        return True

    def _get_default_signal(self, reason: str = "数据不足") -> Dict[str, Any]:
        """
        生成默认信号（持有信号）
        
        Args:
            reason: 生成默认信号的原因
            
        Returns:
            Dict[str, Any]: 默认信号
        """
        return {
            'signal_type': 'hold',
            'strength': 0.0,
            'confidence': 0.0,
            'timestamp': pd.Timestamp.now(),
            'reason': reason,
            'metadata': {}
        }

    def has_result(self) -> bool:
        """
        检查是否已有计算结果
        
        Returns:
            bool: 是否已有计算结果
        """
        return (self._result is not None and 
                hasattr(self._result, 'empty') and 
                not self._result.empty and
                'TRIX' in self._result.columns and
                'MATRIX' in self._result.columns)

    def _detect_enhanced_trix_divergence(self, data: pd.DataFrame) -> bool:
        """
        检测Enhanced TRIX背离形态（智能多维度背离检测）
        
        Args:
            data: 价格数据
            
        Returns:
            bool: 是否检测到背离
        """
        try:
            if len(data) < 15 or len(self._result) < 15:
                return False
                
            # 获取最近15个周期的数据
            recent_prices = data['close'].iloc[-15:]
            recent_trix = self._result['TRIX'].iloc[-15:]
            
            # Enhanced背离检测：考虑TRIX的特殊性质（三重平滑）
            # 短期背离（5周期）- 对TRIX更敏感
            short_prices = recent_prices.iloc[-5:]
            short_trix = recent_trix.iloc[-5:]
            
            # 中期背离（10周期）- TRIX的主要分析周期
            mid_prices = recent_prices.iloc[-10:]
            mid_trix = recent_trix.iloc[-10:]
            
            # 长期背离（15周期）- 三重平滑的长期趋势
            long_prices = recent_prices.iloc[-15:]
            long_trix = recent_trix.iloc[-15:]
            
            # 检测短期背离
            short_top_divergence = (short_prices.iloc[-1] == short_prices.max() and 
                                  short_trix.iloc[-1] < short_trix.max())
            short_bottom_divergence = (short_prices.iloc[-1] == short_prices.min() and 
                                     short_trix.iloc[-1] > short_trix.min())
            
            # 检测中期背离
            mid_top_divergence = (mid_prices.iloc[-1] >= mid_prices.quantile(0.9) and 
                                mid_trix.iloc[-1] < mid_trix.quantile(0.9))
            mid_bottom_divergence = (mid_prices.iloc[-1] <= mid_prices.quantile(0.1) and 
                                   mid_trix.iloc[-1] > mid_trix.quantile(0.1))
            
            # 检测长期背离
            long_top_divergence = (long_prices.iloc[-1] >= long_prices.quantile(0.8) and 
                                 long_trix.iloc[-1] < long_trix.quantile(0.8))
            long_bottom_divergence = (long_prices.iloc[-1] <= long_prices.quantile(0.2) and 
                                    long_trix.iloc[-1] > long_trix.quantile(0.2))
            
            # Enhanced综合判断：至少两个时间框架出现背离
            top_divergence_count = sum([short_top_divergence, mid_top_divergence, long_top_divergence])
            bottom_divergence_count = sum([short_bottom_divergence, mid_bottom_divergence, long_bottom_divergence])
            
            return top_divergence_count >= 2 or bottom_divergence_count >= 2
            
        except Exception as e:
            logger.warning(f"Enhanced TRIX背离检测失败: {e}")
            return False