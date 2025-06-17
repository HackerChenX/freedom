#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
个股VIX指标

实现个股级别的波动率指数，类似于市场VIX波动率指数
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
import warnings
from scipy import stats
import math

from indicators.base_indicator import BaseIndicator
# from indicators.atr import ATR  # 暂时注释掉，避免talib依赖
from utils.logger import get_logger

# 静默警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger(__name__)


class StockVIX(BaseIndicator):
    """
    个股VIX波动率指数
    
    计算个股的隐含波动率指数，类似于市场VIX指数，用于评估个股的恐慌程度和波动风险
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
        """
        初始化个股VIX指标
        
        Args:
            params: 参数字典，可包含：
                - window: 计算波动率的窗口大小，默认为22（约一个月）
                - alpha: 短期波动权重，默认为0.94
                - normal_periods: 用于确定正常波动范围的历史周期数，默认为252（约一年）
                - parkinson_window: Parkinson波动率窗口，默认为10
                - garch_window: GARCH模型窗口，默认为30
        """
        default_params = {
            'window': 22,  # 约一个月
            'alpha': 0.94,  # 短期波动权重
            'normal_periods': 252,  # 约一年
            'parkinson_window': 10,
            'garch_window': 30
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name="StockVIX", description="个股VIX波动率指数")
        self._parameters = default_params
        # self.atr = ATR()  # 暂时注释掉，避免talib依赖
    
    def set_parameters(self, **kwargs):
        """
        设置指标参数
        
        Args:
            **kwargs: 参数键值对
        """
        for key, value in kwargs.items():
            if key in self._parameters:
                self._parameters[key] = value
    
    def get_patterns(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        获取StockVIX相关形态

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 包含形态信息的DataFrame
        """
        # 确保已计算指标
        if not self.has_result():
            self._calculate(data, **kwargs)

        patterns = pd.DataFrame(index=data.index)

        # 如果没有计算结果，返回空DataFrame
        if self._result is None or self._result.empty:
            return patterns

        # 1. 波动率区域形态
        if 'volatility_zone' in self._result.columns:
            zone = self._result['volatility_zone']

            # 各波动区域形态
            patterns['VIX_VERY_LOW_VOLATILITY'] = zone == '极低波动'
            patterns['VIX_LOW_VOLATILITY'] = zone == '低波动'
            patterns['VIX_NORMAL_VOLATILITY'] = zone == '正常波动'
            patterns['VIX_HIGH_VOLATILITY'] = zone == '高波动'
            patterns['VIX_VERY_HIGH_VOLATILITY'] = zone == '极高波动'

        # 2. 波动率趋势形态
        if 'volatility_trend' in self._result.columns:
            trend = self._result['volatility_trend']

            # 趋势形态
            patterns['VIX_UPTREND'] = trend == 1
            patterns['VIX_DOWNTREND'] = trend == -1
            patterns['VIX_SIDEWAYS'] = trend == 0

        # 3. 异常波动形态
        if 'volatility_anomaly' in self._result.columns:
            anomaly = self._result['volatility_anomaly']

            # 异常形态
            patterns['VIX_ANOMALY_SPIKE'] = anomaly == 1
            patterns['VIX_ANOMALY_DROP'] = anomaly == -1
            patterns['VIX_NORMAL'] = anomaly == 0

        # 4. 波动率百分位形态
        if 'vix_percentile' in self._result.columns:
            percentile = self._result['vix_percentile']

            # 百分位形态
            patterns['VIX_EXTREME_LOW'] = percentile < 10
            patterns['VIX_LOW_PERCENTILE'] = (percentile >= 10) & (percentile < 25)
            patterns['VIX_MEDIUM_PERCENTILE'] = (percentile >= 25) & (percentile < 75)
            patterns['VIX_HIGH_PERCENTILE'] = (percentile >= 75) & (percentile < 90)
            patterns['VIX_EXTREME_HIGH'] = percentile >= 90

        # 5. 波动率强度形态
        if 'volatility_strength' in self._result.columns:
            strength = self._result['volatility_strength']

            # 强度形态
            patterns['VIX_WEAK_STRENGTH'] = strength < 25
            patterns['VIX_MODERATE_STRENGTH'] = (strength >= 25) & (strength < 75)
            patterns['VIX_STRONG_STRENGTH'] = strength >= 75

        # 6. VIX值形态
        if 'stock_vix' in self._result.columns:
            vix = self._result['stock_vix']

            # VIX变化形态
            vix_change = vix.diff()
            patterns['VIX_RISING'] = vix_change > 0
            patterns['VIX_FALLING'] = vix_change < 0

            # VIX突破形态
            vix_ma20 = vix.rolling(20).mean()
            patterns['VIX_ABOVE_MA20'] = vix > vix_ma20
            patterns['VIX_BELOW_MA20'] = vix < vix_ma20

            # VIX极值形态
            vix_rolling_max = vix.rolling(60).max()
            vix_rolling_min = vix.rolling(60).min()
            patterns['VIX_NEAR_HIGH'] = vix > vix_rolling_max * 0.9
            patterns['VIX_NEAR_LOW'] = vix < vix_rolling_min * 1.1

        return patterns
    
    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算个股VIX指标
        
        Args:
            df: 输入数据，必须包含open, high, low, close列
            
        Returns:
            pd.DataFrame: 计算结果，包含VIX值和相关指标
        """
        # 验证输入数据
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"输入数据缺少必要的列: {col}")
        
        # 计算各种波动率指标
        result = pd.DataFrame(index=df.index)
        
        # 1. 收益率波动率 - 基于收盘价的对数收益率
        log_returns = np.log(df['close'] / df['close'].shift(1))
        result['returns_volatility'] = log_returns.rolling(
            window=self._parameters['window']
        ).std() * np.sqrt(252) * 100  # 年化并转为百分比
        
        # 2. Parkinson波动率 - 基于高低价范围的估计
        result['parkinson_volatility'] = self._calculate_parkinson_volatility(
            df['high'], df['low'], self._parameters['parkinson_window']
        ) * 100  # 转为百分比
        
        # 3. Garman-Klass波动率 - 使用开盘价、最高价、最低价和收盘价
        result['garman_klass_volatility'] = self._calculate_garman_klass_volatility(
            df['open'], df['high'], df['low'], df['close']
        ) * 100  # 转为百分比
        
        # 4. EWMA波动率 - 指数加权移动平均波动率
        result['ewma_volatility'] = self._calculate_ewma_volatility(
            log_returns, alpha=self._parameters['alpha']
        ) * 100  # 转为百分比
        
        # 5. 条件自回归波动率 (简化版GARCH)
        result['garch_volatility'] = self._calculate_simple_garch_volatility(
            log_returns, self._parameters['garch_window']
        ) * 100  # 转为百分比
        
        # 6. ATR相对波动率（简化实现）
        # atr_result = self.atr.calculate(df)
        # if 'ATR' in atr_result.columns:
        #     result['atr_volatility'] = (atr_result['ATR'] / df['close']) * 100  # 相对ATR，转为百分比

        # 简化的ATR计算
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        result['atr_volatility'] = (atr / df['close']) * 100  # 相对ATR，转为百分比
        
        # 7. 基于各种波动率指标的综合VIX指数
        result['stock_vix'] = self._calculate_composite_vix(result)
        
        # 8. 波动率百分位
        if len(df) >= self._parameters['normal_periods']:
            result['vix_percentile'] = self._calculate_percentile(result['stock_vix'])
        
        # 9. 波动率区域分类
        result['volatility_zone'] = self._classify_volatility_zone(result['stock_vix'])
        
        # 10. 波动率趋势
        result['volatility_trend'] = self._calculate_volatility_trend(result['stock_vix'])
        
        # 11. 波动率预测（短期趋势）
        result['predicted_volatility'] = self._predict_volatility(result['stock_vix'])
        
        # 12. 波动率相对强度
        if len(df) >= self._parameters['normal_periods']:
            result['volatility_strength'] = self._calculate_volatility_strength(result['stock_vix'])
        
        # 13. 异常波动标记
        result['volatility_anomaly'] = self._detect_volatility_anomaly(result['stock_vix'])
        
        return result
    
    def _calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """
        计算Parkinson波动率
        
        Args:
            high: 最高价序列
            low: 最低价序列
            window: 窗口大小
            
        Returns:
            pd.Series: Parkinson波动率
        """
        # Parkinson公式: σ² = 1/(4*ln(2)) * Σ(ln(H/L))²
        log_hl = np.log(high / low)
        log_hl_squared = log_hl ** 2
        
        parkinson_var = log_hl_squared.rolling(window=window).mean() / (4 * np.log(2))
        parkinson_vol = np.sqrt(252 * parkinson_var)  # 年化
        
        return parkinson_vol
    
    def _calculate_garman_klass_volatility(self, open_price: pd.Series, high: pd.Series, 
                                         low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算Garman-Klass波动率
        
        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            
        Returns:
            pd.Series: Garman-Klass波动率
        """
        # Garman-Klass公式的简化版本
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_price) ** 2
        
        # 计算日度方差
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # 使用22日移动平均并年化
        gk_vol = np.sqrt(252 * gk_var.rolling(window=22).mean())
        
        return gk_vol
    
    def _calculate_ewma_volatility(self, returns: pd.Series, alpha: float) -> pd.Series:
        """
        计算指数加权移动平均波动率
        
        Args:
            returns: 收益率序列
            alpha: 权重参数
            
        Returns:
            pd.Series: EWMA波动率
        """
        # 初始化方差
        variance = returns.var()
        
        # 初始化结果序列
        ewma_var = pd.Series(index=returns.index)
        ewma_var.iloc[0] = variance
        
        # 计算EWMA方差
        for i in range(1, len(returns)):
            if not np.isnan(returns.iloc[i-1]):
                ewma_var.iloc[i] = alpha * ewma_var.iloc[i-1] + (1 - alpha) * returns.iloc[i-1]**2
            else:
                ewma_var.iloc[i] = ewma_var.iloc[i-1]
        
        # 计算波动率并年化
        ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)
        
        return ewma_vol
    
    def _calculate_simple_garch_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """
        计算简化版的GARCH波动率（实际上是EWMA的一种变体）
        
        Args:
            returns: 收益率序列
            window: 窗口大小
            
        Returns:
            pd.Series: 简化GARCH波动率
        """
        # 初始化权重（模拟GARCH的时间衰减）
        weights = np.array([(0.94 ** i) for i in range(window)])
        weights = weights / weights.sum()
        
        # 计算加权波动率
        returns_squared = returns ** 2
        garch_var = returns_squared.rolling(window=window).apply(
            lambda x: np.sum(weights[:len(x)] * x) / weights[:len(x)].sum() if len(x) > 0 else np.nan,
            raw=True
        )
        
        # 年化波动率
        garch_vol = np.sqrt(252 * garch_var)
        
        return garch_vol
    
    def _calculate_composite_vix(self, volatilities: pd.DataFrame) -> pd.Series:
        """
        计算综合VIX指数
        
        Args:
            volatilities: 包含各种波动率指标的DataFrame
            
        Returns:
            pd.Series: 综合VIX指数
        """
        # 使用的波动率指标及其权重
        vol_indicators = {
            'returns_volatility': 0.2,
            'parkinson_volatility': 0.2,
            'garman_klass_volatility': 0.2,
            'ewma_volatility': 0.2,
            'garch_volatility': 0.2
        }
        
        # 检查哪些指标存在
        available_indicators = [col for col in vol_indicators if col in volatilities.columns]
        if not available_indicators:
            logger.warning("没有可用的波动率指标用于计算综合VIX")
            return pd.Series(np.nan, index=volatilities.index)
        
        # 重新归一化权重
        total_weight = sum(vol_indicators[col] for col in available_indicators)
        normalized_weights = {col: vol_indicators[col] / total_weight for col in available_indicators}
        
        # 计算加权平均
        composite_vix = sum(volatilities[col] * normalized_weights[col] for col in available_indicators)
        
        return composite_vix
    
    def _calculate_percentile(self, vix: pd.Series) -> pd.Series:
        """
        计算VIX的历史百分位
        
        Args:
            vix: VIX序列
            
        Returns:
            pd.Series: VIX百分位
        """
        percentile = pd.Series(index=vix.index)
        normal_periods = self._parameters['normal_periods']
        
        for i in range(len(vix)):
            if i < normal_periods:
                # 数据不足，使用所有可用历史数据
                history = vix.iloc[:i+1]
            else:
                # 使用过去N个周期的数据
                history = vix.iloc[i-normal_periods+1:i+1]
            
            if len(history) > 0 and not history.isna().all():
                # 计算当前值在历史分布中的百分位
                current_vix = vix.iloc[i]
                percentile.iloc[i] = (history <= current_vix).mean() * 100
            else:
                percentile.iloc[i] = np.nan
        
        return percentile
    
    def _classify_volatility_zone(self, vix: pd.Series) -> pd.Series:
        """
        将VIX分类为不同波动区域
        
        Args:
            vix: VIX序列
            
        Returns:
            pd.Series: 波动区域分类
        """
        # 定义分类阈值
        very_low_threshold = 10.0
        low_threshold = 15.0
        normal_threshold = 25.0
        high_threshold = 40.0
        
        # 初始化结果
        zone = pd.Series(index=vix.index, dtype='object')
        
        # 分类
        zone[vix < very_low_threshold] = '极低波动'
        zone[(vix >= very_low_threshold) & (vix < low_threshold)] = '低波动'
        zone[(vix >= low_threshold) & (vix < normal_threshold)] = '正常波动'
        zone[(vix >= normal_threshold) & (vix < high_threshold)] = '高波动'
        zone[vix >= high_threshold] = '极高波动'
        
        return zone
    
    def _calculate_volatility_trend(self, vix: pd.Series) -> pd.Series:
        """
        计算波动率趋势
        
        Args:
            vix: VIX序列
            
        Returns:
            pd.Series: 波动率趋势 (1=上升, -1=下降, 0=横盘)
        """
        # 使用5日和10日移动平均线判断趋势
        ma5 = vix.rolling(window=5).mean()
        ma10 = vix.rolling(window=10).mean()
        
        # 初始化结果
        trend = pd.Series(0, index=vix.index)
        
        # MA5 > MA10 且都上升，认为是上升趋势
        ma5_diff = ma5.diff()
        ma10_diff = ma10.diff()
        
        uptrend = (ma5 > ma10) & (ma5_diff > 0) & (ma10_diff > 0)
        downtrend = (ma5 < ma10) & (ma5_diff < 0) & (ma10_diff < 0)
        
        trend[uptrend] = 1
        trend[downtrend] = -1
        
        return trend
    
    def _predict_volatility(self, vix: pd.Series) -> pd.Series:
        """
        预测未来波动率（简单线性外推）
        
        Args:
            vix: VIX序列
            
        Returns:
            pd.Series: 预测的波动率
        """
        # 使用10日数据进行线性回归预测
        prediction = pd.Series(index=vix.index)
        
        for i in range(10, len(vix)):
            # 提取最近10个点
            y = vix.iloc[i-10:i].values
            x = np.arange(10)
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # 预测下一个点
            next_value = intercept + slope * 10
            prediction.iloc[i] = next_value
        
        return prediction
    
    def _calculate_volatility_strength(self, vix: pd.Series) -> pd.Series:
        """
        计算波动率相对强度
        
        Args:
            vix: VIX序列
            
        Returns:
            pd.Series: 波动率相对强度 (0-100)
        """
        # 使用Z-score计算相对强度
        normal_periods = self._parameters['normal_periods']
        strength = pd.Series(index=vix.index)
        
        for i in range(normal_periods, len(vix)):
            # 获取历史数据
            history = vix.iloc[i-normal_periods:i]
            
            # 计算Z-score
            mean = history.mean()
            std = history.std()
            if std > 0:
                z_score = (vix.iloc[i] - mean) / std
                
                # 转换为0-100的强度
                # 将z_score的[-3, 3]范围映射到[0, 100]
                strength.iloc[i] = min(max((z_score + 3) / 6 * 100, 0), 100)
            else:
                strength.iloc[i] = 50  # 默认中等强度
        
        return strength
    
    def _detect_volatility_anomaly(self, vix: pd.Series) -> pd.Series:
        """
        检测异常波动
        
        Args:
            vix: VIX序列
            
        Returns:
            pd.Series: 异常波动标记 (1=异常上升, -1=异常下降, 0=正常)
        """
        # 计算20日移动平均和标准差
        ma20 = vix.rolling(window=20).mean()
        std20 = vix.rolling(window=20).std()
        
        # 初始化结果
        anomaly = pd.Series(0, index=vix.index)
        
        # 如果VIX超过MA20 + 2*STD20，标记为异常上升
        anomaly[vix > ma20 + 2 * std20] = 1
        
        # 如果VIX低于MA20 - 2*STD20，标记为异常下降
        anomaly[vix < ma20 - 2 * std20] = -1
        
        return anomaly
    
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
        
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算指标原始评分
        
        Args:
            data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分(0-100)
        """
        # 确保已计算指标
        if not self.has_result():
            self.calculate(data, **kwargs)
        
        if self._result is None:
            return pd.Series(50.0, index=data.index)
        
        # 初始化评分
        score = pd.Series(50.0, index=data.index)
    
        # 在这里实现指标特定的评分逻辑
        # 此处提供默认实现
    
        return score

    def calculate_confidence(self, score: pd.Series, patterns: pd.DataFrame, signals: dict) -> float:
        """
        计算StockVIX指标的置信度

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

        # 2. 基于数据质量的置信度
        if hasattr(self, '_result') and self._result is not None:
            # 检查是否有VIX数据
            if 'stock_vix' in self._result.columns:
                vix_values = self._result['stock_vix'].dropna()
                if len(vix_values) > 0:
                    # VIX数据越完整，置信度越高
                    data_completeness = len(vix_values) / len(self._result)
                    confidence += data_completeness * 0.1

            # 检查波动率指标数量
            vol_indicators = ['returns_volatility', 'parkinson_volatility', 'garman_klass_volatility',
                             'ewma_volatility', 'garch_volatility']
            available_indicators = sum(1 for col in vol_indicators if col in self._result.columns)
            confidence += (available_indicators / len(vol_indicators)) * 0.1

        # 3. 基于形态的置信度
        if not patterns.empty:
            # 检查StockVIX形态
            pattern_count = patterns.sum().sum()
            if pattern_count > 0:
                confidence += min(pattern_count * 0.02, 0.15)

        # 4. 基于信号的置信度
        if signals:
            # 检查信号强度
            signal_count = sum(1 for signal in signals.values() if hasattr(signal, 'any') and signal.any())
            if signal_count > 0:
                confidence += min(signal_count * 0.05, 0.1)

        # 5. 基于数据长度的置信度
        if len(score) >= 252:  # 一年数据
            confidence += 0.1
        elif len(score) >= 60:  # 两个月数据
            confidence += 0.05

        # 确保置信度在0-1范围内
        return max(0.0, min(1.0, confidence))

    def register_patterns(self):
        """
        注册StockVIX指标的形态到全局形态注册表
        """
        # 注册波动率区域形态
        self.register_pattern_to_registry(
            pattern_id="VIX_VERY_LOW_VOLATILITY",
            display_name="极低波动",
            description="VIX处于极低水平，市场恐慌情绪极低，可能是买入机会",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VIX_VERY_HIGH_VOLATILITY",
            display_name="极高波动",
            description="VIX处于极高水平，市场恐慌情绪极高，存在超跌反弹机会",
            pattern_type="NEUTRAL",
            default_strength="STRONG",
            score_impact=0.0,
            polarity="NEGATIVE"
        )

        # 注册异常波动形态
        self.register_pattern_to_registry(
            pattern_id="VIX_ANOMALY_SPIKE",
            display_name="VIX异常飙升",
            description="VIX异常飙升，市场恐慌情绪爆发，可能是短期底部信号",
            pattern_type="BULLISH",
            default_strength="STRONG",
            score_impact=15.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="VIX_ANOMALY_DROP",
            display_name="VIX异常下跌",
            description="VIX异常下跌，市场过度乐观，需警惕风险",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEUTRAL"
        )

        # 注册趋势形态
        self.register_pattern_to_registry(
            pattern_id="VIX_UPTREND",
            display_name="VIX上升趋势",
            description="VIX处于上升趋势，市场恐慌情绪增加",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-10.0,
            polarity="NEGATIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VIX_DOWNTREND",
            display_name="VIX下降趋势",
            description="VIX处于下降趋势，市场恐慌情绪减少",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        # 注册百分位形态
        self.register_pattern_to_registry(
            pattern_id="VIX_EXTREME_LOW",
            display_name="VIX极低百分位",
            description="VIX处于历史极低百分位，市场可能过度乐观",
            pattern_type="BEARISH",
            default_strength="MEDIUM",
            score_impact=-5.0,
            polarity="NEUTRAL"
        )

        self.register_pattern_to_registry(
            pattern_id="VIX_EXTREME_HIGH",
            display_name="VIX极高百分位",
            description="VIX处于历史极高百分位，市场可能过度悲观",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=15.0,
            polarity="NEUTRAL"
        )

        # 注册VIX位置形态
        self.register_pattern_to_registry(
            pattern_id="VIX_NEAR_HIGH",
            display_name="VIX接近高点",
            description="VIX接近近期高点，恐慌情绪接近峰值",
            pattern_type="BULLISH",
            default_strength="MEDIUM",
            score_impact=10.0,
            polarity="POSITIVE"
        )

        self.register_pattern_to_registry(
            pattern_id="VIX_NEAR_LOW",
            display_name="VIX接近低点",
            description="VIX接近近期低点，市场情绪过于乐观",
            pattern_type="BEARISH",
            default_strength="WEAK",
            score_impact=-5.0,
            polarity="NEGATIVE"
        )

    def get_indicator_type(self) -> str:
        """
        获取指标类型

        Returns:
            str: 指标类型
        """
        return "STOCKVIX"

    def get_pattern_info(self, pattern_id: str) -> dict:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            
        Returns:
            dict: 形态详细信息
        """
        # 默认形态信息
        default_pattern = {
            "id": pattern_id,
            "name": pattern_id,
            "description": f"{pattern_id}形态",
            "type": "NEUTRAL",
            "strength": "MEDIUM",
            "score_impact": 0.0
        }
        
        # StockVIX指标特定的形态信息映射
        pattern_info_map = {
            # 基础形态
            "超买区域": {
                "id": "超买区域",
                "name": "超买区域",
                "description": "指标进入超买区域，可能面临回调压力",
                "type": "BEARISH",
                "strength": "MEDIUM",
                "score_impact": -10.0
            },
            "超卖区域": {
                "id": "超卖区域", 
                "name": "超卖区域",
                "description": "指标进入超卖区域，可能出现反弹机会",
                "type": "BULLISH",
                "strength": "MEDIUM",
                "score_impact": 10.0
            },
            "中性区域": {
                "id": "中性区域",
                "name": "中性区域", 
                "description": "指标处于中性区域，趋势不明确",
                "type": "NEUTRAL",
                "strength": "WEAK",
                "score_impact": 0.0
            },
            # 趋势形态
            "上升趋势": {
                "id": "上升趋势",
                "name": "上升趋势",
                "description": "指标显示上升趋势，看涨信号",
                "type": "BULLISH", 
                "strength": "STRONG",
                "score_impact": 15.0
            },
            "下降趋势": {
                "id": "下降趋势",
                "name": "下降趋势",
                "description": "指标显示下降趋势，看跌信号",
                "type": "BEARISH",
                "strength": "STRONG", 
                "score_impact": -15.0
            },
            # 信号形态
            "买入信号": {
                "id": "买入信号",
                "name": "买入信号",
                "description": "指标产生买入信号，建议关注",
                "type": "BULLISH",
                "strength": "STRONG",
                "score_impact": 20.0
            },
            "卖出信号": {
                "id": "卖出信号", 
                "name": "卖出信号",
                "description": "指标产生卖出信号，建议谨慎",
                "type": "BEARISH",
                "strength": "STRONG",
                "score_impact": -20.0
            }
        }
        
        return pattern_info_map.get(pattern_id, default_pattern)