#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强买点识别系统

提供智能的买点识别功能，包括：
- 多维度买点检测
- 机器学习辅助识别
- 买点质量评估
- 实时买点监控
遵循六层架构规范
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from utils.dependency_injection import get_logger
from utils.decorators import performance_monitor, exception_handler
from utils.unified_container import get_container
from enums.signal_types import SignalType
from enums.pattern_types import Candle_pattern_type as PatternType

logger = get_logger(__name__)

class BuyPointType(Enum):
    """买点类型"""
    VOLUME_BREAKOUT = "放量突破"
    PULLBACK_SUPPORT = "回调支撑"
    TREND_REVERSAL = "趋势反转"
    PATTERN_BREAKOUT = "形态突破"
    MOMENTUM_ACCELERATION = "动量加速"
    OVERSOLD_REBOUND = "超卖反弹"
    MULTI_TIMEFRAME_CONFLUENCE = "多周期共振"

class BuyPointQuality(Enum):
    """买点质量"""
    EXCELLENT = "优秀"
    GOOD = "良好"
    AVERAGE = "一般"
    POOR = "较差"

@dataclass
class BuyPointSignal:
    """买点信号"""
    signal_id: str
    stock_code: str
    detection_date: str
    buypoint_type: BuyPointType
    quality: BuyPointQuality
    confidence: float  # 置信度 0-1
    score: float  # 评分 0-100
    price: float
    volume: int
    technical_indicators: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    
@dataclass
class BuyPointDetectionConfig:
    """买点检测配置"""
    min_confidence: float = 0.6
    min_score: float = 60.0
    lookback_periods: int = 20
    volume_threshold: float = 1.5  # 成交量倍数
    price_change_threshold: float = 0.02  # 价格变化阈值
    enable_ml_detection: bool = False
    detection_types: List[BuyPointType] = None
    
    def __post_init__(self):
        if self.detection_types is None:
            self.detection_types = list(BuyPointType)

class EnhancedBuyPointDetector:
    """
    增强买点识别系统
    
    提供全面的买点识别和质量评估功能
    """
    
    def __init__(self, config: Optional[BuyPointDetectionConfig] = None):
        """
        初始化买点识别系统
        
        Args:
            config: 检测配置
        """
        self.config = config or BuyPointDetectionConfig()
        self.logger = get_logger(__name__)
        
        # 从容器获取服务
        container = get_container()
        try:
            self.data_access = container.resolve("DataAccessInterface")
        except:
            self.data_access = self._create_mock_data_access()
        try:
            self.indicator_registry = container.resolve("IndicatorRegistry")
        except:
            self.indicator_registry = self._create_mock_indicator_registry()
        
        # 检测统计
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'detection_by_type': {},
            'quality_distribution': {},
            'average_confidence': 0.0
        }
        
        self.logger.info(f"增强买点识别系统初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=30.0)
    def detect_buypoints(self, stock_codes: List[str], 
                        detection_date: Optional[str] = None) -> List[BuyPointSignal]:
        """
        检测买点信号
        
        Args:
            stock_codes: 股票代码列表
            detection_date: 检测日期，默认为当前日期
            
        Returns:
            List[BuyPointSignal]: 检测到的买点信号列表
        """
        if detection_date is None:
            detection_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"开始检测买点，股票数量: {len(stock_codes)}, 检测日期: {detection_date}")
        
        detected_signals = []
        
        for stock_code in stock_codes:
            try:
                signals = self._detect_stock_buypoints(stock_code, detection_date)
                detected_signals.extend(signals)
                
                if signals:
                    self.logger.debug(f"✅ {stock_code} 检测到 {len(signals)} 个买点信号")
                
            except Exception as e:
                self.logger.warning(f"❌ {stock_code} 买点检测失败: {e}")
        
        # 过滤和排序
        filtered_signals = self._filter_and_rank_signals(detected_signals)
        
        self.logger.info(f"买点检测完成，共检测到 {len(filtered_signals)} 个有效信号")
        return filtered_signals
    
    @exception_handler(reraise=False, default_return=[])
    def _detect_stock_buypoints(self, stock_code: str, detection_date: str) -> List[BuyPointSignal]:
        """
        检测单只股票的买点
        
        Args:
            stock_code: 股票代码
            detection_date: 检测日期
            
        Returns:
            List[BuyPointSignal]: 买点信号列表
        """
        # 获取股票数据
        stock_data = self._get_stock_data(stock_code, detection_date)
        if stock_data is None or len(stock_data) < self.config.lookback_periods:
            return []
        
        # 计算技术指标
        technical_indicators = self._calculate_technical_indicators(stock_data)
        
        # 检测各类买点
        detected_signals = []
        
        for buypoint_type in self.config.detection_types:
            try:
                signal = self._detect_specific_buypoint(
                    stock_code, stock_data, technical_indicators, 
                    buypoint_type, detection_date
                )
                
                if signal and signal.confidence >= self.config.min_confidence:
                    detected_signals.append(signal)
                    
            except Exception as e:
                self.logger.debug(f"检测 {buypoint_type.value} 失败: {e}")
        
        return detected_signals
    
    def _get_stock_data(self, stock_code: str, detection_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 计算数据范围
            end_date = datetime.strptime(detection_date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=self.config.lookback_periods * 2)
            
            query = f"""
            SELECT code, name, date, open, high, low, close, volume, price_change
            FROM stock_info 
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date.strftime('%Y-%m-%d')}' 
            AND date <= '{detection_date}'
            ORDER BY date ASC
            """
            
            data = self.data_access.query_dataframe(query)
            return data if len(data) >= self.config.lookback_periods else None
            
        except Exception as e:
            self.logger.warning(f"获取 {stock_code} 数据失败: {e}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标"""
        indicators = {}
        
        # 基础指标列表
        indicator_names = ['MA', 'MACD', 'RSI', 'KDJ', 'VOL', 'BOLL']
        
        for indicator_name in indicator_names:
            try:
                indicator_class = self.indicator_registry.get_indicator(indicator_name)
                if indicator_class:
                    indicator = indicator_class()
                    result = indicator.calculate(data)
                    indicators[indicator_name] = result
                    
            except Exception as e:
                self.logger.debug(f"计算指标 {indicator_name} 失败: {e}")
        
        return indicators
    
    def _detect_specific_buypoint(self, stock_code: str, data: pd.DataFrame,
                                 indicators: Dict[str, Any], buypoint_type: BuyPointType,
                                 detection_date: str) -> Optional[BuyPointSignal]:
        """检测特定类型的买点"""
        
        if buypoint_type == BuyPointType.VOLUME_BREAKOUT:
            return self._detect_volume_breakout(stock_code, data, indicators, detection_date)
        elif buypoint_type == BuyPointType.PULLBACK_SUPPORT:
            return self._detect_pullback_support(stock_code, data, indicators, detection_date)
        elif buypoint_type == BuyPointType.TREND_REVERSAL:
            return self._detect_trend_reversal(stock_code, data, indicators, detection_date)
        elif buypoint_type == BuyPointType.PATTERN_BREAKOUT:
            return self._detect_pattern_breakout(stock_code, data, indicators, detection_date)
        elif buypoint_type == BuyPointType.MOMENTUM_ACCELERATION:
            return self._detect_momentum_acceleration(stock_code, data, indicators, detection_date)
        elif buypoint_type == BuyPointType.OVERSOLD_REBOUND:
            return self._detect_oversold_rebound(stock_code, data, indicators, detection_date)
        elif buypoint_type == BuyPointType.MULTI_TIMEFRAME_CONFLUENCE:
            return self._detect_multi_timeframe_confluence(stock_code, data, indicators, detection_date)
        
        return None
    
    def _detect_volume_breakout(self, stock_code: str, data: pd.DataFrame,
                               indicators: Dict[str, Any], detection_date: str) -> Optional[BuyPointSignal]:
        """检测放量突破买点"""
        if len(data) < 5:
            return None
        
        # 获取最近几天的数据
        recent_data = data.tail(5)
        latest = recent_data.iloc[-1]
        
        # 成交量条件：今日成交量 > 前5日平均成交量 * 阈值
        avg_volume = recent_data['volume'].iloc[:-1].mean()
        volume_condition = latest['volume'] > avg_volume * self.config.volume_threshold
        
        # 价格条件：收盘价创近期新高或接近新高
        price_condition = latest['close'] >= recent_data['high'].max() * 0.98
        
        # 趋势条件：短期均线向上
        ma_condition = False
        if 'MA' in indicators and indicators['MA'] is not None:
            ma_data = indicators['MA']
            if len(ma_data) >= 2:
                ma_condition = ma_data.iloc[-1] > ma_data.iloc[-2]
        
        if volume_condition and price_condition:
            confidence = 0.7 + (0.2 if ma_condition else 0.0)
            score = 70 + (10 if ma_condition else 0) + min((latest['volume'] / avg_volume - 1) * 10, 20)
            
            return BuyPointSignal(
                signal_id=f"{stock_code}_{detection_date}_volume_breakout",
                stock_code=stock_code,
                detection_date=detection_date,
                buypoint_type=BuyPointType.VOLUME_BREAKOUT,
                quality=self._determine_quality(score),
                confidence=min(confidence, 1.0),
                score=min(score, 100.0),
                price=latest['close'],
                volume=latest['volume'],
                technical_indicators={'volume_ratio': latest['volume'] / avg_volume},
                pattern_analysis={'breakout_strength': 'strong' if ma_condition else 'moderate'},
                risk_assessment={'risk_level': 'medium'},
                recommendations=['关注成交量持续性', '设置止损位']
            )
        
        return None

    def _detect_pullback_support(self, stock_code: str, data: pd.DataFrame,
                                indicators: Dict[str, Any], detection_date: str) -> Optional[BuyPointSignal]:
        """检测回调支撑买点"""
        if len(data) < 10:
            return None

        recent_data = data.tail(10)
        latest = recent_data.iloc[-1]

        # 支撑位条件：价格接近重要支撑位（如均线、前低等）
        support_condition = False
        support_level = 0.0

        if 'MA' in indicators and indicators['MA'] is not None:
            ma_data = indicators['MA'].tail(10)
            ma20 = ma_data.iloc[-1] if len(ma_data) > 0 else 0
            if ma20 > 0 and abs(latest['close'] - ma20) / ma20 < 0.02:  # 接近20日均线
                support_condition = True
                support_level = ma20

        # 回调条件：从高点回调但未破重要支撑
        high_point = recent_data['high'].max()
        pullback_ratio = (high_point - latest['close']) / high_point
        pullback_condition = 0.05 < pullback_ratio < 0.15  # 5%-15%的回调

        # 反弹信号：今日收盘价高于开盘价
        rebound_signal = latest['close'] > latest['open']

        if support_condition and pullback_condition and rebound_signal:
            confidence = 0.65 + (0.15 if pullback_ratio < 0.10 else 0.0)
            score = 65 + (pullback_ratio < 0.10) * 15 + rebound_signal * 10

            return BuyPointSignal(
                signal_id=f"{stock_code}_{detection_date}_pullback_support",
                stock_code=stock_code,
                detection_date=detection_date,
                buypoint_type=BuyPointType.PULLBACK_SUPPORT,
                quality=self._determine_quality(score),
                confidence=min(confidence, 1.0),
                score=min(score, 100.0),
                price=latest['close'],
                volume=latest['volume'],
                technical_indicators={'support_level': support_level, 'pullback_ratio': pullback_ratio},
                pattern_analysis={'support_type': 'MA20', 'rebound_strength': 'moderate'},
                risk_assessment={'risk_level': 'low'},
                recommendations=['关注支撑位有效性', '适合中长线持有']
            )

        return None

    def _detect_trend_reversal(self, stock_code: str, data: pd.DataFrame,
                              indicators: Dict[str, Any], detection_date: str) -> Optional[BuyPointSignal]:
        """检测趋势反转买点"""
        if len(data) < 15:
            return None

        recent_data = data.tail(15)
        latest = recent_data.iloc[-1]

        # MACD反转信号
        macd_reversal = False
        if 'MACD' in indicators and indicators['MACD'] is not None:
            macd_data = indicators['MACD'].tail(5)
            if len(macd_data) >= 2:
                # MACD从负转正或金叉
                macd_reversal = (macd_data.iloc[-1] > 0 and macd_data.iloc[-2] <= 0)

        # RSI超卖反弹
        rsi_oversold = False
        if 'RSI' in indicators and indicators['RSI'] is not None:
            rsi_data = indicators['RSI'].tail(5)
            if len(rsi_data) >= 2:
                rsi_oversold = (rsi_data.iloc[-2] < 30 and rsi_data.iloc[-1] > 30)

        # 价格形态：锤子线、十字星等反转形态
        reversal_pattern = self._detect_reversal_candlestick(latest)

        if macd_reversal or rsi_oversold or reversal_pattern:
            confidence = 0.6 + (macd_reversal * 0.15) + (rsi_oversold * 0.15) + (reversal_pattern * 0.1)
            score = 60 + (macd_reversal * 15) + (rsi_oversold * 15) + (reversal_pattern * 10)

            return BuyPointSignal(
                signal_id=f"{stock_code}_{detection_date}_trend_reversal",
                stock_code=stock_code,
                detection_date=detection_date,
                buypoint_type=BuyPointType.TREND_REVERSAL,
                quality=self._determine_quality(score),
                confidence=min(confidence, 1.0),
                score=min(score, 100.0),
                price=latest['close'],
                volume=latest['volume'],
                technical_indicators={'macd_reversal': macd_reversal, 'rsi_oversold': rsi_oversold},
                pattern_analysis={'reversal_signals': [macd_reversal, rsi_oversold, reversal_pattern]},
                risk_assessment={'risk_level': 'medium'},
                recommendations=['确认反转有效性', '分批建仓']
            )

        return None

    def _detect_pattern_breakout(self, stock_code: str, data: pd.DataFrame,
                                indicators: Dict[str, Any], detection_date: str) -> Optional[BuyPointSignal]:
        """检测形态突破买点"""
        if len(data) < 20:
            return None

        recent_data = data.tail(20)
        latest = recent_data.iloc[-1]

        # 检测三角形整理突破
        triangle_breakout = self._detect_triangle_breakout(recent_data)

        # 检测箱体突破
        box_breakout = self._detect_box_breakout(recent_data)

        # 检测双底突破
        double_bottom = self._detect_double_bottom(recent_data)

        if triangle_breakout or box_breakout or double_bottom:
            confidence = 0.7 + (triangle_breakout * 0.1) + (box_breakout * 0.1) + (double_bottom * 0.1)
            score = 70 + (triangle_breakout * 10) + (box_breakout * 10) + (double_bottom * 15)

            pattern_type = []
            if triangle_breakout:
                pattern_type.append('triangle')
            if box_breakout:
                pattern_type.append('box')
            if double_bottom:
                pattern_type.append('double_bottom')

            return BuyPointSignal(
                signal_id=f"{stock_code}_{detection_date}_pattern_breakout",
                stock_code=stock_code,
                detection_date=detection_date,
                buypoint_type=BuyPointType.PATTERN_BREAKOUT,
                quality=self._determine_quality(score),
                confidence=min(confidence, 1.0),
                score=min(score, 100.0),
                price=latest['close'],
                volume=latest['volume'],
                technical_indicators={'pattern_types': pattern_type},
                pattern_analysis={'breakout_patterns': pattern_type},
                risk_assessment={'risk_level': 'medium'},
                recommendations=['关注突破有效性', '设置合理止损']
            )

        return None

    def _detect_oversold_rebound(self, stock_code: str, data: pd.DataFrame,
                                indicators: Dict[str, Any], detection_date: str) -> Optional[BuyPointSignal]:
        """检测超卖反弹买点"""
        if len(data) < 10:
            return None

        latest = data.iloc[-1]

        # RSI超卖
        rsi_oversold = False
        rsi_value = 50
        if 'RSI' in indicators and indicators['RSI'] is not None:
            rsi_data = indicators['RSI']
            if len(rsi_data) > 0:
                rsi_value = rsi_data.iloc[-1]
                rsi_oversold = rsi_value < 30

        # KDJ超卖
        kdj_oversold = False
        if 'KDJ' in indicators and indicators['KDJ'] is not None:
            kdj_data = indicators['KDJ']
            if len(kdj_data) > 0 and 'K' in kdj_data.columns:
                k_value = kdj_data['K'].iloc[-1]
                kdj_oversold = k_value < 20

        # 价格超跌：跌幅较大
        recent_high = data.tail(10)['high'].max()
        decline_ratio = (recent_high - latest['close']) / recent_high
        price_oversold = decline_ratio > 0.15  # 跌幅超过15%

        if rsi_oversold or kdj_oversold or price_oversold:
            confidence = 0.55 + (rsi_oversold * 0.15) + (kdj_oversold * 0.15) + (price_oversold * 0.1)
            score = 55 + (rsi_oversold * 15) + (kdj_oversold * 15) + (price_oversold * 10)

            return BuyPointSignal(
                signal_id=f"{stock_code}_{detection_date}_oversold_rebound",
                stock_code=stock_code,
                detection_date=detection_date,
                buypoint_type=BuyPointType.OVERSOLD_REBOUND,
                quality=self._determine_quality(score),
                confidence=min(confidence, 1.0),
                score=min(score, 100.0),
                price=latest['close'],
                volume=latest['volume'],
                technical_indicators={'rsi': rsi_value, 'decline_ratio': decline_ratio},
                pattern_analysis={'oversold_signals': [rsi_oversold, kdj_oversold, price_oversold]},
                risk_assessment={'risk_level': 'high'},
                recommendations=['等待反弹确认', '控制仓位']
            )

        return None

    def _detect_multi_timeframe_confluence(self, stock_code: str, data: pd.DataFrame,
                                          indicators: Dict[str, Any], detection_date: str) -> Optional[BuyPointSignal]:
        """检测多周期共振买点"""
        # 这里简化实现，实际应该获取多个周期的数据
        # 当前只基于日线数据进行模拟多周期分析

        if len(data) < 20:
            return None

        latest = data.iloc[-1]

        # 短期信号（5日）
        short_term_signal = self._analyze_short_term_signal(data.tail(5))

        # 中期信号（10日）
        medium_term_signal = self._analyze_medium_term_signal(data.tail(10))

        # 长期信号（20日）
        long_term_signal = self._analyze_long_term_signal(data.tail(20))

        # 多周期共振条件
        confluence_score = short_term_signal + medium_term_signal + long_term_signal

        if confluence_score >= 2:  # 至少两个周期发出买入信号
            confidence = 0.8 + (confluence_score - 2) * 0.1
            score = 80 + (confluence_score - 2) * 10

            return BuyPointSignal(
                signal_id=f"{stock_code}_{detection_date}_multi_timeframe",
                stock_code=stock_code,
                detection_date=detection_date,
                buypoint_type=BuyPointType.MULTI_TIMEFRAME_CONFLUENCE,
                quality=self._determine_quality(score),
                confidence=min(confidence, 1.0),
                score=min(score, 100.0),
                price=latest['close'],
                volume=latest['volume'],
                technical_indicators={'confluence_score': confluence_score},
                pattern_analysis={'timeframe_signals': [short_term_signal, medium_term_signal, long_term_signal]},
                risk_assessment={'risk_level': 'low'},
                recommendations=['多周期共振信号', '适合中长线投资']
            )

        return None

    def _analyze_short_term_signal(self, data: pd.DataFrame) -> int:
        """分析短期信号"""
        if len(data) < 2:
            return 0

        # 价格上涨
        price_up = data['close'].iloc[-1] > data['close'].iloc[-2]

        # 成交量放大
        volume_up = data['volume'].iloc[-1] > data['volume'].mean()

        return int(price_up and volume_up)

    def _analyze_medium_term_signal(self, data: pd.DataFrame) -> int:
        """分析中期信号"""
        if len(data) < 5:
            return 0

        # 均线向上
        ma5 = data['close'].rolling(5).mean()
        ma_up = ma5.iloc[-1] > ma5.iloc[-2] if len(ma5) >= 2 else False

        # 价格在均线上方
        price_above_ma = data['close'].iloc[-1] > ma5.iloc[-1]

        return int(ma_up and price_above_ma)

    def _analyze_long_term_signal(self, data: pd.DataFrame) -> int:
        """分析长期信号"""
        if len(data) < 10:
            return 0

        # 长期趋势向上
        ma10 = data['close'].rolling(10).mean()
        trend_up = ma10.iloc[-1] > ma10.iloc[-5] if len(ma10) >= 5 else False

        # 价格创新高
        recent_high = data['high'].tail(10).max()
        new_high = data['close'].iloc[-1] >= recent_high * 0.95

        return int(trend_up and new_high)

    def _detect_reversal_candlestick(self, candle_data: pd.Series) -> bool:
        """检测反转K线形态"""
        open_price = candle_data['open']
        high_price = candle_data['high']
        low_price = candle_data['low']
        close_price = candle_data['close']

        # 锤子线：下影线长，实体小，上影线短
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        lower_shadow = min(open_price, close_price) - low_price
        upper_shadow = high_price - max(open_price, close_price)

        if total_range > 0:
            hammer = (lower_shadow > body_size * 2 and
                     upper_shadow < body_size and
                     body_size / total_range < 0.3)

            # 十字星：开盘价接近收盘价
            doji = body_size / total_range < 0.1 if total_range > 0 else False

            return hammer or doji

        return False

    def _detect_triangle_breakout(self, data: pd.DataFrame) -> bool:
        """检测三角形突破"""
        if len(data) < 10:
            return False

        # 简化的三角形检测：高点逐渐降低，低点逐渐升高
        highs = data['high'].values
        lows = data['low'].values

        # 检查最近的突破
        recent_high = data['high'].tail(5).max()
        previous_resistance = data['high'].iloc[:-5].tail(10).max()

        return recent_high > previous_resistance * 1.02  # 突破前期阻力

    def _detect_box_breakout(self, data: pd.DataFrame) -> bool:
        """检测箱体突破"""
        if len(data) < 15:
            return False

        # 检测价格是否在一个区间内震荡后突破
        box_data = data.iloc[:-3]  # 排除最近3天
        resistance = box_data['high'].max()
        support = box_data['low'].min()
        box_range = resistance - support

        # 箱体有效性：震荡幅度适中
        if box_range / support < 0.05 or box_range / support > 0.20:
            return False

        # 突破条件
        recent_close = data['close'].iloc[-1]
        return recent_close > resistance * 1.01

    def _detect_double_bottom(self, data: pd.DataFrame) -> bool:
        """检测双底形态"""
        if len(data) < 15:
            return False

        # 简化的双底检测
        lows = data['low'].values

        # 找到两个相对低点
        min_idx1 = np.argmin(lows[:len(lows)//2])
        min_idx2 = np.argmin(lows[len(lows)//2:]) + len(lows)//2

        if min_idx1 < min_idx2:
            low1 = lows[min_idx1]
            low2 = lows[min_idx2]

            # 两个低点接近
            if abs(low1 - low2) / min(low1, low2) < 0.03:
                # 当前价格高于低点
                current_price = data['close'].iloc[-1]
                return current_price > max(low1, low2) * 1.05

        return False

    def _determine_quality(self, score: float) -> BuyPointQuality:
        """确定买点质量"""
        if score >= 85:
            return BuyPointQuality.EXCELLENT
        elif score >= 75:
            return BuyPointQuality.GOOD
        elif score >= 60:
            return BuyPointQuality.AVERAGE
        else:
            return BuyPointQuality.POOR

    def _filter_and_rank_signals(self, signals: List[BuyPointSignal]) -> List[BuyPointSignal]:
        """过滤和排序信号"""
        # 过滤低质量信号
        filtered = [s for s in signals if s.score >= self.config.min_score]

        # 按评分排序
        filtered.sort(key=lambda x: x.score, reverse=True)

        # 更新统计
        self.detection_stats['total_detections'] = len(signals)
        self.detection_stats['successful_detections'] = len(filtered)

        return filtered

    def _create_mock_data_access(self):
        """创建模拟数据访问对象"""
        class MockDataAccess:
            def query_dataframe(self, query: str) -> pd.DataFrame:
                # 生成模拟股票数据
                dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
                data = []
                for i, date in enumerate(dates):
                    data.append({
                        'code': '000001',
                        'name': '平安银行',
                        'date': date.strftime('%Y-%m-%d'),
                        'open': 10.0 + np.random.normal(0, 0.5),
                        'high': 10.5 + np.random.normal(0, 0.5),
                        'low': 9.5 + np.random.normal(0, 0.5),
                        'close': 10.0 + np.random.normal(0, 0.5),
                        'volume': 1000000 + np.random.randint(0, 500000),
                        'turnover': np.random.uniform(0.5, 5.0)
                    })
                return pd.DataFrame(data)

        return MockDataAccess()

    def _create_mock_indicator_registry(self):
        """创建模拟指标注册表"""
        class MockIndicatorRegistry:
            def get_indicator(self, name: str):
                class MockIndicator:
                    def __init__(self):
                        self.name = name

                    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                        # 返回简单的模拟指标数据
                        result = data.copy()
                        if name == 'MA':
                            result['MA'] = data['close'].rolling(20).mean()
                        elif name == 'MACD':
                            result['MACD'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
                        elif name == 'RSI':
                            result['RSI'] = 50 + np.random.normal(0, 10, len(data))
                        return result

                return MockIndicator

        return MockIndicatorRegistry()
