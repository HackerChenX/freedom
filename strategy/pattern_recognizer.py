#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术形态模式识别器

专门用于识别技术指标组合模式的高性能分析引擎。
通过多维度技术分析识别买点前后的指标特征模式。
遵循六层架构规范。
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import hashlib
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from analysis.buypoints.enhanced_backtest_engine import BuyPointData

logger = get_logger(__name__)

@dataclass
class IndicatorSignal:
    """指标信号"""
    indicator_name: str
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: float  # 0-1之间
    confidence_level: float  # 0-1之间
    timestamp: str
    raw_value: float
    normalized_value: float

@dataclass
class PatternSignature:
    """模式特征签名"""
    pattern_id: str
    signature_hash: str
    indicator_signals: List[IndicatorSignal]
    pattern_strength: float
    coherence_score: float  # 各指标信号一致性得分
    temporal_consistency: float  # 时间序列一致性

@dataclass
class RecognitionResult:
    """识别结果"""
    buypoint_code: str
    buypoint_date: str
    recognized_patterns: List[PatternSignature]
    confidence_score: float
    analysis_summary: Dict[str, Any]
    execution_time: float

@dataclass
class PatternRecognitionConfig:
    """模式识别配置"""
    lookback_window: int = 10  # 买点前分析窗口
    lookforward_window: int = 5   # 买点后分析窗口
    min_signal_strength: float = 0.3  # 最小信号强度
    min_confidence_level: float = 0.5  # 最小置信水平
    enable_clustering: bool = True  # 启用聚类分析
    max_patterns_per_buypoint: int = 5  # 每个买点最多识别模式数
    parallel_workers: int = 4
    cache_enabled: bool = True

class PatternRecognizer:
    """
    技术形态模式识别器

    高性能技术分析模式识别引擎，专门用于：
    1. 多指标信号识别和强度评估
    2. 指标组合模式的自动发现
    3. 买点前后技术形态特征提取
    4. 模式一致性和可靠性评估
    """

    def __init__(self, config: Optional[PatternRecognitionConfig] = None):
        """
        初始化模式识别器

        Args:
            config: 识别配置参数
        """
        self.config = config or PatternRecognitionConfig()
        self.logger = get_logger(__name__)

        # 初始化依赖组件
        container = get_container()
        try:
            self.data_access = container.resolve("DataAccessInterface")
            self.indicator_registry = container.resolve("CompleteIndicatorRegistry")
        except:
            self.data_access = None
            self.indicator_registry = None
            self.logger.warning("未能获取依赖注入服务，将使用内置实现")

        # 内置指标计算器
        self.indicator_calculators = self._init_indicator_calculators()

        # 缓存和统计
        self._pattern_cache = {}
        self._signal_cache = {}
        self.recognition_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'total_patterns_found': 0,
            'average_confidence': 0.0,
            'average_recognition_time': 0.0
        }

        self.logger.info("技术形态模式识别器初始化完成")

    def _init_indicator_calculators(self) -> Dict[str, callable]:
        """初始化内置指标计算器"""
        return {
            'MA': self._calculate_ma,
            'EMA': self._calculate_ema,
            'MACD': self._calculate_macd,
            'RSI': self._calculate_rsi,
            'KDJ': self._calculate_kdj,
            'BOLL': self._calculate_boll,
            'CCI': self._calculate_cci,
            'ATR': self._calculate_atr,
            'OBV': self._calculate_obv,
            'WILLR': self._calculate_willr
        }

    @exception_handler(reraise=True)
    @performance_monitor(threshold=120.0)
    def recognize_patterns_batch(self, buypoints: List[BuyPointData]) -> List[RecognitionResult]:
        """
        批量识别买点模式

        Args:
            buypoints: 买点数据列表

        Returns:
            List[RecognitionResult]: 识别结果列表
        """
        start_time = time.time()
        self.logger.info(f"开始批量模式识别，买点数量: {len(buypoints)}")

        results = []

        # 并行处理买点
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_buypoint = {
                executor.submit(self._recognize_single_buypoint_pattern, bp): bp
                for bp in buypoints
            }

            for future in as_completed(future_to_buypoint):
                buypoint = future_to_buypoint[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"识别买点 {buypoint.stock_code} 模式失败: {e}")

        # 更新统计
        execution_time = time.time() - start_time
        self._update_recognition_stats(len(buypoints), len(results), execution_time)

        self.logger.info(f"批量模式识别完成，成功识别 {len(results)} 个买点，耗时 {execution_time:.2f}秒")
        return results

    @exception_handler(reraise=False, default_return=None)
    def _recognize_single_buypoint_pattern(self, buypoint: BuyPointData) -> Optional[RecognitionResult]:
        """
        识别单个买点的模式

        Args:
            buypoint: 买点数据

        Returns:
            Optional[RecognitionResult]: 识别结果
        """
        start_time = time.time()

        try:
            # 1. 获取买点前后的股票数据
            stock_data = self._get_buypoint_data(buypoint)
            if stock_data.empty:
                return None

            # 2. 计算技术指标
            indicators_data = self._calculate_all_indicators(stock_data)
            if not indicators_data:
                return None

            # 3. 提取买点时刻的指标信号
            buypoint_signals = self._extract_buypoint_signals(
                indicators_data, buypoint.buypoint_date
            )

            # 4. 识别模式特征签名
            pattern_signatures = self._identify_pattern_signatures(
                buypoint_signals, buypoint.stock_code
            )

            # 5. 计算整体置信度
            confidence_score = self._calculate_overall_confidence(pattern_signatures)

            # 6. 生成分析摘要
            analysis_summary = self._generate_analysis_summary(
                buypoint_signals, pattern_signatures
            )

            execution_time = time.time() - start_time

            result = RecognitionResult(
                buypoint_code=buypoint.stock_code,
                buypoint_date=buypoint.buypoint_date,
                recognized_patterns=pattern_signatures,
                confidence_score=confidence_score,
                analysis_summary=analysis_summary,
                execution_time=execution_time
            )

            return result

        except Exception as e:
            self.logger.warning(f"识别买点 {buypoint.stock_code} 模式异常: {e}")
            return None

    def _get_buypoint_data(self, buypoint: BuyPointData) -> pd.DataFrame:
        """
        获取买点前后的数据

        Args:
            buypoint: 买点数据

        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            buypoint_date = datetime.strptime(buypoint.buypoint_date, '%Y-%m-%d')

            start_date = (buypoint_date -
                         timedelta(days=self.config.lookback_window)).strftime('%Y-%m-%d')
            end_date = (buypoint_date +
                       timedelta(days=self.config.lookforward_window)).strftime('%Y-%m-%d')

            if self.data_access:
                return self.data_access.get_stock_data(
                    code=buypoint.stock_code,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                # 返回模拟数据用于测试
                return self._generate_mock_stock_data(buypoint.stock_code, start_date, end_date)

        except Exception as e:
            self.logger.warning(f"获取股票 {buypoint.stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def _generate_mock_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟股票数据用于测试"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(stock_code) % 2**32)  # 确保每个股票的数据一致

        n_days = len(dates)
        base_price = 10.0

        # 生成价格序列
        returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # 生成OHLC数据
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close
            volume = int(np.random.normal(1000000, 300000))

            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': max(open_price, 0.1),
                'high': max(high, close, open_price),
                'low': min(low, close, open_price),
                'close': max(close, 0.1),
                'volume': max(volume, 1000)
            })

        return pd.DataFrame(data)

    def _calculate_all_indicators(self, stock_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算所有技术指标

        Args:
            stock_data: 股票数据

        Returns:
            Dict[str, pd.Series]: 指标数据字典
        """
        indicators = {}

        try:
            # 确保数据有必需的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in stock_data.columns for col in required_cols):
                self.logger.warning("股票数据缺少必需列")
                return {}

            # 计算各类指标
            for indicator_name, calculator in self.indicator_calculators.items():
                try:
                    indicator_result = calculator(stock_data)
                    if isinstance(indicator_result, dict):
                        # 多个输出的指标（如MACD, KDJ）
                        for key, value in indicator_result.items():
                            indicators[f"{indicator_name}_{key}"] = value
                    else:
                        # 单个输出的指标
                        indicators[indicator_name] = indicator_result

                except Exception as e:
                    self.logger.debug(f"计算指标 {indicator_name} 失败: {e}")

        except Exception as e:
            self.logger.warning(f"计算技术指标失败: {e}")

        return indicators

    # 指标计算方法
    def _calculate_ma(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算移动平均线"""
        results = {}
        periods = [5, 10, 20, 30, 60]

        for period in periods:
            if len(data) >= period:
                results[f'MA{period}'] = data['close'].rolling(window=period).mean()

        return results

    def _calculate_ema(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算指数移动平均线"""
        results = {}
        periods = [12, 26]

        for period in periods:
            results[f'EMA{period}'] = data['close'].ewm(span=period, adjust=False).mean()

        return results

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()

        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'DIF': macd_line,
            'DEA': signal_line,
            'HISTOGRAM': histogram
        }

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI指标"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_kdj(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        low_min = data['low'].rolling(window=9).min()
        high_max = data['high'].rolling(window=9).max()

        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)  # 填充NaN值

        k_values = []
        d_values = []
        k = 50.0
        d = 50.0

        for rsv_val in rsv:
            if pd.notna(rsv_val):
                k = (2/3) * k + (1/3) * rsv_val
                d = (2/3) * d + (1/3) * k
            k_values.append(k)
            d_values.append(d)

        k_series = pd.Series(k_values, index=data.index)
        d_series = pd.Series(d_values, index=data.index)
        j_series = 3 * k_series - 2 * d_series

        return {
            'K': k_series,
            'D': d_series,
            'J': j_series
        }

    def _calculate_boll(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算布林带指标"""
        period = 20
        std_multiplier = 2

        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()

        return {
            'UPPER': sma + (std * std_multiplier),
            'MIDDLE': sma,
            'LOWER': sma - (std * std_multiplier)
        }

    def _calculate_cci(self, data: pd.DataFrame) -> pd.Series:
        """计算CCI指标"""
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)

        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """计算ATR指标"""
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        return atr

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """计算OBV指标"""
        obv = []
        obv_val = 0

        for i in range(len(data)):
            if i == 0:
                obv_val = data['volume'].iloc[i]
            else:
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv_val += data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv_val -= data['volume'].iloc[i]
                # 相等时OBV不变

            obv.append(obv_val)

        return pd.Series(obv, index=data.index)

    def _calculate_willr(self, data: pd.DataFrame) -> pd.Series:
        """计算威廉指标"""
        period = 14
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()

        willr = -100 * (high_max - data['close']) / (high_max - low_min)
        return willr

    def _extract_buypoint_signals(self, indicators_data: Dict[str, pd.Series],
                                buypoint_date: str) -> List[IndicatorSignal]:
        """
        提取买点时刻的指标信号

        Args:
            indicators_data: 指标数据
            buypoint_date: 买点日期

        Returns:
            List[IndicatorSignal]: 指标信号列表
        """
        signals = []
        target_date = pd.to_datetime(buypoint_date)

        for indicator_name, indicator_series in indicators_data.items():
            try:
                # 找到最接近买点日期的数据点
                if hasattr(indicator_series, 'index') and len(indicator_series) > 0:
                    # 查找最近的日期索引
                    closest_idx = indicator_series.index.get_indexer([target_date], method='nearest')[0]

                    if closest_idx >= 0 and closest_idx < len(indicator_series):
                        raw_value = indicator_series.iloc[closest_idx]

                        if pd.notna(raw_value):
                            # 分析信号类型和强度
                            signal_info = self._analyze_indicator_signal(
                                indicator_name, raw_value, indicator_series, closest_idx
                            )

                            if signal_info['signal_strength'] >= self.config.min_signal_strength:
                                signal = IndicatorSignal(
                                    indicator_name=indicator_name,
                                    signal_type=signal_info['signal_type'],
                                    signal_strength=signal_info['signal_strength'],
                                    confidence_level=signal_info['confidence_level'],
                                    timestamp=buypoint_date,
                                    raw_value=float(raw_value),
                                    normalized_value=signal_info['normalized_value']
                                )
                                signals.append(signal)

            except Exception as e:
                self.logger.debug(f"提取指标 {indicator_name} 信号失败: {e}")

        return signals

    def _analyze_indicator_signal(self, indicator_name: str, current_value: float,
                                indicator_series: pd.Series, current_idx: int) -> Dict[str, Any]:
        """
        分析指标信号类型和强度

        Args:
            indicator_name: 指标名称
            current_value: 当前值
            indicator_series: 指标序列
            current_idx: 当前索引

        Returns:
            Dict[str, Any]: 信号分析结果
        """
        signal_info = {
            'signal_type': 'neutral',
            'signal_strength': 0.0,
            'confidence_level': 0.5,
            'normalized_value': 0.0
        }

        try:
            # 获取历史数据用于对比
            lookback_data = indicator_series.iloc[max(0, current_idx-10):current_idx+1]

            if len(lookback_data) < 2:
                return signal_info

            # 基于指标类型的特定分析
            if 'RSI' in indicator_name:
                signal_info.update(self._analyze_rsi_signal(current_value, lookback_data))

            elif 'MACD' in indicator_name:
                signal_info.update(self._analyze_macd_signal(
                    indicator_name, current_value, lookback_data
                ))

            elif 'KDJ' in indicator_name:
                signal_info.update(self._analyze_kdj_signal(
                    indicator_name, current_value, lookback_data
                ))

            elif 'BOLL' in indicator_name:
                signal_info.update(self._analyze_boll_signal(
                    indicator_name, current_value, lookback_data
                ))

            elif 'MA' in indicator_name or 'EMA' in indicator_name:
                signal_info.update(self._analyze_ma_signal(current_value, lookback_data))

            elif 'CCI' in indicator_name:
                signal_info.update(self._analyze_cci_signal(current_value))

            elif 'WILLR' in indicator_name:
                signal_info.update(self._analyze_willr_signal(current_value))

            else:
                # 通用分析：基于趋势和相对位置
                signal_info.update(self._analyze_generic_signal(current_value, lookback_data))

        except Exception as e:
            self.logger.debug(f"分析指标 {indicator_name} 信号失败: {e}")

        return signal_info

    def _analyze_rsi_signal(self, current_value: float, lookback_data: pd.Series) -> Dict[str, Any]:
        """分析RSI信号"""
        if current_value < 30:
            return {
                'signal_type': 'bullish',
                'signal_strength': (30 - current_value) / 30 * 0.8,
                'confidence_level': 0.8,
                'normalized_value': current_value / 100
            }
        elif current_value > 70:
            return {
                'signal_type': 'bearish',
                'signal_strength': (current_value - 70) / 30 * 0.8,
                'confidence_level': 0.8,
                'normalized_value': current_value / 100
            }
        else:
            # 中性区域，看趋势
            if len(lookback_data) >= 3:
                trend = np.polyfit(range(len(lookback_data)), lookback_data.values, 1)[0]
                signal_strength = min(abs(trend) / 10, 0.5)

                return {
                    'signal_type': 'bullish' if trend > 0 else 'bearish' if trend < 0 else 'neutral',
                    'signal_strength': signal_strength,
                    'confidence_level': 0.6,
                    'normalized_value': current_value / 100
                }

        return {
            'signal_type': 'neutral',
            'signal_strength': 0.2,
            'confidence_level': 0.5,
            'normalized_value': current_value / 100
        }

    def _analyze_macd_signal(self, indicator_name: str, current_value: float,
                           lookback_data: pd.Series) -> Dict[str, Any]:
        """分析MACD信号"""
        if 'DIF' in indicator_name:
            # MACD线分析
            if current_value > 0:
                signal_strength = min(abs(current_value) / 0.1, 0.8)  # 假设0.1为强信号阈值
                return {
                    'signal_type': 'bullish',
                    'signal_strength': signal_strength,
                    'confidence_level': 0.7,
                    'normalized_value': np.tanh(current_value)  # tanh归一化
                }
            else:
                return {
                    'signal_type': 'bearish',
                    'signal_strength': min(abs(current_value) / 0.1, 0.8),
                    'confidence_level': 0.7,
                    'normalized_value': np.tanh(current_value)
                }

        elif 'HISTOGRAM' in indicator_name:
            # 柱状图分析（寻找零轴交叉）
            if len(lookback_data) >= 2:
                prev_value = lookback_data.iloc[-2] if len(lookback_data) > 1 else 0

                if prev_value < 0 and current_value > 0:
                    # 金叉
                    return {
                        'signal_type': 'bullish',
                        'signal_strength': 0.9,
                        'confidence_level': 0.9,
                        'normalized_value': np.tanh(current_value)
                    }
                elif prev_value > 0 and current_value < 0:
                    # 死叉
                    return {
                        'signal_type': 'bearish',
                        'signal_strength': 0.9,
                        'confidence_level': 0.9,
                        'normalized_value': np.tanh(current_value)
                    }

        return self._analyze_generic_signal(current_value, lookback_data)

    def _analyze_kdj_signal(self, indicator_name: str, current_value: float,
                          lookback_data: pd.Series) -> Dict[str, Any]:
        """分析KDJ信号"""
        if current_value < 20:
            return {
                'signal_type': 'bullish',
                'signal_strength': (20 - current_value) / 20 * 0.8,
                'confidence_level': 0.7,
                'normalized_value': current_value / 100
            }
        elif current_value > 80:
            return {
                'signal_type': 'bearish',
                'signal_strength': (current_value - 80) / 20 * 0.8,
                'confidence_level': 0.7,
                'normalized_value': current_value / 100
            }
        else:
            return self._analyze_generic_signal(current_value, lookback_data)

    def _analyze_boll_signal(self, indicator_name: str, current_value: float,
                           lookback_data: pd.Series) -> Dict[str, Any]:
        """分析布林带信号"""
        # 布林带信号需要价格与带的关系，这里简化处理
        if 'LOWER' in indicator_name:
            return {
                'signal_type': 'bullish',
                'signal_strength': 0.7,
                'confidence_level': 0.8,
                'normalized_value': 0.0  # 下轨归一化为0
            }
        elif 'UPPER' in indicator_name:
            return {
                'signal_type': 'bearish',
                'signal_strength': 0.7,
                'confidence_level': 0.8,
                'normalized_value': 1.0  # 上轨归一化为1
            }
        else:
            return self._analyze_generic_signal(current_value, lookback_data)

    def _analyze_ma_signal(self, current_value: float, lookback_data: pd.Series) -> Dict[str, Any]:
        """分析移动平均线信号"""
        if len(lookback_data) >= 3:
            # 分析均线趋势
            trend = np.polyfit(range(len(lookback_data)), lookback_data.values, 1)[0]
            signal_strength = min(abs(trend) / current_value * 100, 0.8) if current_value > 0 else 0.3

            return {
                'signal_type': 'bullish' if trend > 0 else 'bearish' if trend < 0 else 'neutral',
                'signal_strength': signal_strength,
                'confidence_level': 0.6,
                'normalized_value': trend / abs(trend) if trend != 0 else 0.0
            }

        return self._analyze_generic_signal(current_value, lookback_data)

    def _analyze_cci_signal(self, current_value: float) -> Dict[str, Any]:
        """分析CCI信号"""
        if current_value < -100:
            return {
                'signal_type': 'bullish',
                'signal_strength': min(abs(current_value + 100) / 100, 0.8),
                'confidence_level': 0.7,
                'normalized_value': np.tanh(current_value / 200)
            }
        elif current_value > 100:
            return {
                'signal_type': 'bearish',
                'signal_strength': min((current_value - 100) / 100, 0.8),
                'confidence_level': 0.7,
                'normalized_value': np.tanh(current_value / 200)
            }
        else:
            return {
                'signal_type': 'neutral',
                'signal_strength': 0.3,
                'confidence_level': 0.5,
                'normalized_value': np.tanh(current_value / 200)
            }

    def _analyze_willr_signal(self, current_value: float) -> Dict[str, Any]:
        """分析威廉指标信号"""
        if current_value < -80:
            return {
                'signal_type': 'bullish',
                'signal_strength': (80 + current_value) / (-20) * 0.8,
                'confidence_level': 0.7,
                'normalized_value': (current_value + 100) / 100
            }
        elif current_value > -20:
            return {
                'signal_type': 'bearish',
                'signal_strength': (-20 - current_value) / 80 * 0.8,
                'confidence_level': 0.7,
                'normalized_value': (current_value + 100) / 100
            }
        else:
            return {
                'signal_type': 'neutral',
                'signal_strength': 0.3,
                'confidence_level': 0.5,
                'normalized_value': (current_value + 100) / 100
            }

    def _analyze_generic_signal(self, current_value: float, lookback_data: pd.Series) -> Dict[str, Any]:
        """通用信号分析"""
        if len(lookback_data) < 2:
            return {
                'signal_type': 'neutral',
                'signal_strength': 0.2,
                'confidence_level': 0.4,
                'normalized_value': 0.0
            }

        # 计算相对位置（百分位数）
        percentile = stats.percentileofscore(lookback_data.values, current_value) / 100

        # 计算趋势
        if len(lookback_data) >= 3:
            trend = np.polyfit(range(len(lookback_data)), lookback_data.values, 1)[0]
            trend_strength = min(abs(trend) / (lookback_data.std() + 1e-8), 1.0)
        else:
            trend = lookback_data.iloc[-1] - lookback_data.iloc[0]
            trend_strength = min(abs(trend) / (lookback_data.std() + 1e-8), 1.0)

        # 综合分析
        if percentile > 0.7 and trend > 0:
            signal_type = 'bullish'
            signal_strength = min((percentile - 0.5) * 2 * trend_strength, 0.8)
        elif percentile < 0.3 and trend < 0:
            signal_type = 'bearish'
            signal_strength = min((0.5 - percentile) * 2 * trend_strength, 0.8)
        else:
            signal_type = 'neutral'
            signal_strength = 0.3

        return {
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'confidence_level': 0.5 + trend_strength * 0.3,
            'normalized_value': percentile
        }

    def _identify_pattern_signatures(self, signals: List[IndicatorSignal],
                                   stock_code: str) -> List[PatternSignature]:
        """
        识别模式特征签名

        Args:
            signals: 指标信号列表
            stock_code: 股票代码

        Returns:
            List[PatternSignature]: 模式签名列表
        """
        signatures = []

        if not signals:
            return signatures

        # 按信号类型分组
        bullish_signals = [s for s in signals if s.signal_type == 'bullish']
        bearish_signals = [s for s in signals if s.signal_type == 'bearish']
        neutral_signals = [s for s in signals if s.signal_type == 'neutral']

        # 生成看多模式签名
        if len(bullish_signals) >= 2:
            bullish_signature = self._create_pattern_signature(
                bullish_signals, 'BULLISH_PATTERN', stock_code
            )
            signatures.append(bullish_signature)

        # 生成看空模式签名
        if len(bearish_signals) >= 2:
            bearish_signature = self._create_pattern_signature(
                bearish_signals, 'BEARISH_PATTERN', stock_code
            )
            signatures.append(bearish_signature)

        # 生成混合模式签名（同时有多空信号时的复杂模式）
        if bullish_signals and bearish_signals:
            mixed_signature = self._create_mixed_pattern_signature(
                bullish_signals, bearish_signals, stock_code
            )
            signatures.append(mixed_signature)

        # 聚类分析识别特殊模式（如果启用）
        if self.config.enable_clustering and len(signals) >= 4:
            clustered_signatures = self._cluster_based_pattern_recognition(signals, stock_code)
            signatures.extend(clustered_signatures)

        # 限制返回的模式数量
        signatures.sort(key=lambda x: x.pattern_strength, reverse=True)
        return signatures[:self.config.max_patterns_per_buypoint]

    def _create_pattern_signature(self, signals: List[IndicatorSignal],
                                pattern_type: str, stock_code: str) -> PatternSignature:
        """
        创建模式签名

        Args:
            signals: 相同类型的信号列表
            pattern_type: 模式类型
            stock_code: 股票代码

        Returns:
            PatternSignature: 模式签名
        """
        # 计算模式强度（所有信号强度的加权平均）
        total_weight = sum(s.confidence_level for s in signals)
        if total_weight > 0:
            pattern_strength = sum(s.signal_strength * s.confidence_level for s in signals) / total_weight
        else:
            pattern_strength = sum(s.signal_strength for s in signals) / len(signals)

        # 计算一致性得分
        signal_strengths = [s.signal_strength for s in signals]
        coherence_score = 1.0 - np.std(signal_strengths) / (np.mean(signal_strengths) + 1e-8)
        coherence_score = max(0.0, min(1.0, coherence_score))

        # 时间一致性（所有信号来自同一时间点，所以为1.0）
        temporal_consistency = 1.0

        # 生成模式哈希
        signal_names = sorted([s.indicator_name for s in signals])
        signature_string = f"{pattern_type}_{','.join(signal_names)}_{stock_code}"
        signature_hash = hashlib.md5(signature_string.encode()).hexdigest()[:12]

        pattern_id = f"{pattern_type}_{signature_hash[:8]}"

        return PatternSignature(
            pattern_id=pattern_id,
            signature_hash=signature_hash,
            indicator_signals=signals,
            pattern_strength=pattern_strength,
            coherence_score=coherence_score,
            temporal_consistency=temporal_consistency
        )

    def _create_mixed_pattern_signature(self, bullish_signals: List[IndicatorSignal],
                                      bearish_signals: List[IndicatorSignal],
                                      stock_code: str) -> PatternSignature:
        """
        创建混合模式签名（多空并存）

        Args:
            bullish_signals: 看多信号
            bearish_signals: 看空信号
            stock_code: 股票代码

        Returns:
            PatternSignature: 混合模式签名
        """
        all_signals = bullish_signals + bearish_signals

        # 计算净强度（看多强度 - 看空强度）
        bullish_strength = sum(s.signal_strength * s.confidence_level for s in bullish_signals)
        bearish_strength = sum(s.signal_strength * s.confidence_level for s in bearish_signals)
        net_strength = abs(bullish_strength - bearish_strength) / (bullish_strength + bearish_strength + 1e-8)

        # 混合模式的一致性较低（因为有矛盾信号）
        coherence_score = 0.5 - (min(len(bullish_signals), len(bearish_signals)) / max(len(bullish_signals), len(bearish_signals))) * 0.3

        # 生成签名
        signal_names = sorted([s.indicator_name for s in all_signals])
        signature_string = f"MIXED_PATTERN_{','.join(signal_names)}_{stock_code}"
        signature_hash = hashlib.md5(signature_string.encode()).hexdigest()[:12]

        pattern_id = f"MIXED_{signature_hash[:8]}"

        return PatternSignature(
            pattern_id=pattern_id,
            signature_hash=signature_hash,
            indicator_signals=all_signals,
            pattern_strength=net_strength,
            coherence_score=coherence_score,
            temporal_consistency=1.0
        )

    def _cluster_based_pattern_recognition(self, signals: List[IndicatorSignal],
                                         stock_code: str) -> List[PatternSignature]:
        """
        基于聚类的模式识别

        Args:
            signals: 信号列表
            stock_code: 股票代码

        Returns:
            List[PatternSignature]: 聚类识别的模式
        """
        signatures = []

        try:
            # 构建特征矩阵
            features = []
            for signal in signals:
                feature_vector = [
                    signal.signal_strength,
                    signal.confidence_level,
                    signal.normalized_value,
                    1.0 if signal.signal_type == 'bullish' else -1.0 if signal.signal_type == 'bearish' else 0.0
                ]
                features.append(feature_vector)

            features = np.array(features)

            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # 聚类分析
            n_clusters = min(3, len(signals) // 2)  # 限制聚类数量
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)

                # 为每个聚类创建模式签名
                for cluster_id in range(n_clusters):
                    cluster_signals = [signals[i] for i, label in enumerate(cluster_labels) if label == cluster_id]

                    if len(cluster_signals) >= 2:
                        # 确定聚类的主要信号类型
                        signal_types = [s.signal_type for s in cluster_signals]
                        main_type = Counter(signal_types).most_common(1)[0][0]

                        cluster_signature = self._create_pattern_signature(
                            cluster_signals,
                            f"CLUSTER_{main_type.upper()}_{cluster_id}",
                            stock_code
                        )
                        signatures.append(cluster_signature)

        except Exception as e:
            self.logger.debug(f"聚类分析失败: {e}")

        return signatures

    def _calculate_overall_confidence(self, pattern_signatures: List[PatternSignature]) -> float:
        """
        计算整体置信度

        Args:
            pattern_signatures: 模式签名列表

        Returns:
            float: 整体置信度
        """
        if not pattern_signatures:
            return 0.0

        # 基于所有模式的综合评估
        pattern_strengths = [p.pattern_strength for p in pattern_signatures]
        coherence_scores = [p.coherence_score for p in pattern_signatures]

        # 加权平均
        avg_strength = np.mean(pattern_strengths)
        avg_coherence = np.mean(coherence_scores)

        # 模式数量对置信度的影响（更多一致的模式增加置信度）
        pattern_count_factor = min(len(pattern_signatures) / 3.0, 1.0)

        overall_confidence = (avg_strength * 0.4 + avg_coherence * 0.4 + pattern_count_factor * 0.2)

        return min(overall_confidence, 1.0)

    def _generate_analysis_summary(self, signals: List[IndicatorSignal],
                                 pattern_signatures: List[PatternSignature]) -> Dict[str, Any]:
        """
        生成分析摘要

        Args:
            signals: 指标信号列表
            pattern_signatures: 模式签名列表

        Returns:
            Dict[str, Any]: 分析摘要
        """
        summary = {
            'total_signals': len(signals),
            'total_patterns': len(pattern_signatures),
            'signal_distribution': {},
            'top_indicators': [],
            'pattern_summary': [],
            'overall_sentiment': 'neutral'
        }

        # 信号分布统计
        signal_types = [s.signal_type for s in signals]
        summary['signal_distribution'] = dict(Counter(signal_types))

        # 顶级指标（按信号强度排序）
        sorted_signals = sorted(signals, key=lambda x: x.signal_strength, reverse=True)
        summary['top_indicators'] = [
            {
                'indicator': s.indicator_name,
                'signal_type': s.signal_type,
                'strength': s.signal_strength,
                'confidence': s.confidence_level
            }
            for s in sorted_signals[:5]
        ]

        # 模式摘要
        for pattern in pattern_signatures:
            summary['pattern_summary'].append({
                'pattern_id': pattern.pattern_id,
                'strength': pattern.pattern_strength,
                'coherence': pattern.coherence_score,
                'indicators_count': len(pattern.indicator_signals)
            })

        # 整体情绪倾向
        bullish_count = summary['signal_distribution'].get('bullish', 0)
        bearish_count = summary['signal_distribution'].get('bearish', 0)

        if bullish_count > bearish_count * 1.5:
            summary['overall_sentiment'] = 'bullish'
        elif bearish_count > bullish_count * 1.5:
            summary['overall_sentiment'] = 'bearish'
        else:
            summary['overall_sentiment'] = 'neutral'

        return summary

    def _update_recognition_stats(self, total_buypoints: int, successful_recognitions: int,
                                execution_time: float):
        """更新识别统计信息"""
        self.recognition_stats['total_recognitions'] += 1
        if successful_recognitions > 0:
            self.recognition_stats['successful_recognitions'] += 1

        # 更新平均识别时间
        total_time = (self.recognition_stats['average_recognition_time'] *
                     (self.recognition_stats['total_recognitions'] - 1) + execution_time)
        self.recognition_stats['average_recognition_time'] = total_time / self.recognition_stats['total_recognitions']

    @exception_handler(reraise=False, default_return={})
    def get_recognition_statistics(self) -> Dict[str, Any]:
        """获取识别统计信息"""
        stats = self.recognition_stats.copy()

        if stats['total_recognitions'] > 0:
            stats['success_rate'] = stats['successful_recognitions'] / stats['total_recognitions']
        else:
            stats['success_rate'] = 0.0

        return stats

    @exception_handler(reraise=False, default_return={})
    def export_recognition_result(self, result: RecognitionResult) -> Dict[str, Any]:
        """
        导出识别结果为标准格式

        Args:
            result: 识别结果

        Returns:
            Dict[str, Any]: 标准格式的识别结果
        """
        export_data = {
            'buypoint_info': {
                'stock_code': result.buypoint_code,
                'buypoint_date': result.buypoint_date,
                'confidence_score': result.confidence_score
            },
            'patterns': [],
            'signals': [],
            'analysis_summary': result.analysis_summary,
            'execution_time': result.execution_time,
            'export_timestamp': datetime.now().isoformat()
        }

        # 导出模式信息
        for pattern in result.recognized_patterns:
            pattern_data = {
                'pattern_id': pattern.pattern_id,
                'signature_hash': pattern.signature_hash,
                'pattern_strength': pattern.pattern_strength,
                'coherence_score': pattern.coherence_score,
                'temporal_consistency': pattern.temporal_consistency,
                'indicators': [s.indicator_name for s in pattern.indicator_signals]
            }
            export_data['patterns'].append(pattern_data)

        # 导出信号信息
        all_signals = []
        for pattern in result.recognized_patterns:
            all_signals.extend(pattern.indicator_signals)

        for signal in all_signals:
            signal_data = asdict(signal)
            export_data['signals'].append(signal_data)

        return export_data