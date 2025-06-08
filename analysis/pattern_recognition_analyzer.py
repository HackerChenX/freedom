#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pattern Recognition Analyzer - 形态识别分析器

识别和分析技术指标形态的专用分析模块，支持多指标联合分析、跨周期分析和统计评分
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime, timedelta
import json
import talib
from enum import Enum

from indicators.factory import IndicatorFactory
from indicators.base_indicator import BaseIndicator, MarketEnvironment
from indicators.pattern_registry import get_pattern_registry
from indicators.pattern_manager import PatternManager
from utils.logger import get_logger
from enums.kline_period import KlinePeriod
from utils.decorators import validate_dataframe, log_calls, exception_handler, performance_monitor, cache_result
from utils.scoring_validator import validate_score

logger = get_logger(__name__)

class PatternType(Enum):
    """K线形态类型枚举"""
    BULLISH = "bullish"  # 看涨形态
    BEARISH = "bearish"  # 看跌形态
    REVERSAL = "reversal"  # 反转形态
    CONTINUATION = "continuation"  # 持续形态
    
class PatternStrength(Enum):
    """形态强度枚举"""
    WEAK = 1  # 弱信号
    MEDIUM = 2  # 中等信号
    STRONG = 3  # 强信号

class PatternRecognitionAnalyzer:
    """
    形态识别分析器
    
    提供多指标形态识别、多周期形态分析和形态统计评分功能
    """
    
    def __init__(self, indicators: List[str] = None, periods: List[str] = None):
        """
        初始化形态识别分析器
        
        Args:
            indicators: 要分析的指标列表，如 ['MACD', 'KDJ', 'RSI']
            periods: 要分析的周期列表，如 ['DAILY', 'WEEKLY']
        """
        # 初始化指标列表和周期列表
        self.indicators = indicators or []
        self.periods = periods or ["DAILY"]
        
        # 初始化指标实例缓存
        self._indicator_instances: Dict[str, BaseIndicator] = {}
        
        # 初始化形态管理器
        self.pattern_manager = PatternManager.get_instance()
        
        # 初始化形态注册表
        self.pattern_registry = get_pattern_registry()
        
        # 当前分析的股票代码和名称
        self.current_stock_code = None
        self.current_stock_name = None
        
        # 初始化结果缓存
        self._results_cache = {}
        
        # TA-Lib 形态函数映射
        self.talib_patterns = {
            # 单日形态
            "doji": talib.CDLDOJI,  # 十字星
            "hammer": talib.CDLHAMMER,  # 锤子线
            "hanging_man": talib.CDLHANGINGMAN,  # 上吊线
            "inverted_hammer": talib.CDLINVERTEDHAMMER,  # 倒锤子线
            "shooting_star": talib.CDLSHOOTINGSTAR,  # 流星线
            "marubozu": talib.CDLMARUBOZU,  # 光头光脚阳线/阴线
            
            # 双日形态
            "engulfing": talib.CDLENGULFING,  # 吞噬形态
            "harami": talib.CDLHARAMI,  # 孕线形态
            "piercing": talib.CDLPIERCING,  # 刺透形态
            "dark_cloud_cover": talib.CDLDARKCLOUDCOVER,  # 乌云盖顶
            
            # 三日形态
            "morning_star": talib.CDLMORNINGSTAR,  # 晨星
            "evening_star": talib.CDLEVENINGSTAR,  # 暮星
            "three_white_soldiers": talib.CDL3WHITESOLDIERS,  # 三白兵
            "three_black_crows": talib.CDL3BLACKCROWS,  # 三黑鸦
            
            # 其他复杂形态
            "abandoned_baby": talib.CDLABANDONEDBABY,  # 弃婴
            "belt_hold": talib.CDLBELTHOLD,  # 捉腰带线
            "breakaway": talib.CDLBREAKAWAY,  # 脱离形态
            "closing_marubozu": talib.CDLCLOSINGMARUBOZU,  # 收盘光头光脚线
            "concealing_baby_swallow": talib.CDLCONCEALBABYSWALL,  # 藏婴吞没
            "counterattack": talib.CDLCOUNTERATTACK,  # 反击线
            "dragonfly_doji": talib.CDLDRAGONFLYDOJI,  # 蜻蜓十字星
            "gravestone_doji": talib.CDLGRAVESTONEDOJI,  # 墓碑十字星
            "harami_cross": talib.CDLHARAMICROSS,  # 十字孕线
            "high_wave": talib.CDLHIGHWAVE,  # 高浪线
            "hikkake": talib.CDLHIKKAKE,  # 陷阱
            "hikkake_mod": talib.CDLHIKKAKEMOD,  # 改良陷阱
            "identical_three_crows": talib.CDLIDENTICAL3CROWS,  # 三胞胎乌鸦
            "in_neck": talib.CDLINNECK,  # 颈内线
            "kicking": talib.CDLKICKING,  # 反冲形态
            "kicking_by_length": talib.CDLKICKINGBYLENGTH,  # 由长度决定的反冲形态
            "ladder_bottom": talib.CDLLADDERBOTTOM,  # 梯底
            "long_legged_doji": talib.CDLLONGLEGGEDDOJI,  # 长脚十字星
            "matching_low": talib.CDLMATCHINGLOW,  # 相同低点
            "mat_hold": talib.CDLMATHOLD,  # 铺垫形态
            "on_neck": talib.CDLONNECK,  # 颈上线
            "rickshaw_man": talib.CDLRICKSHAWMAN,  # 黄包车夫
            "rise_fall_three_methods": talib.CDLRISEFALL3METHODS,  # 上升/下降三法
            "separating_lines": talib.CDLSEPARATINGLINES,  # 分离线
            "short_line": talib.CDLSHORTLINE,  # 短线
            "spinning_top": talib.CDLSPINNINGTOP,  # 纺锤线
            "stalled_pattern": talib.CDLSTALLEDPATTERN,  # 停顿形态
            "stick_sandwich": talib.CDLSTICKSANDWICH,  # 条形三明治
            "takuri": talib.CDLTAKURI,  # 探水竿
            "tasuki_gap": talib.CDLTASUKIGAP,  # 跳空并列阴阳线
            "thrusting": talib.CDLTHRUSTING,  # 插入
            "tristar": talib.CDLTRISTAR,  # 三星
            "unique_three_river": talib.CDLUNIQUE3RIVER,  # 奇特三河床
            "upside_gap_two_crows": talib.CDLUPSIDEGAP2CROWS,  # 向上跳空的两只乌鸦
            "xside_gap_three_methods": talib.CDLXSIDEGAP3METHODS,  # 上升/下降跳空三法
        }
        
        # 形态类型映射
        self.pattern_types = {
            # 看涨形态
            "hammer": PatternType.BULLISH,
            "inverted_hammer": PatternType.BULLISH,
            "morning_star": PatternType.BULLISH,
            "three_white_soldiers": PatternType.BULLISH,
            "bullish_engulfing": PatternType.BULLISH,
            "piercing": PatternType.BULLISH,
            
            # 看跌形态
            "hanging_man": PatternType.BEARISH,
            "shooting_star": PatternType.BEARISH,
            "evening_star": PatternType.BEARISH,
            "three_black_crows": PatternType.BEARISH,
            "bearish_engulfing": PatternType.BEARISH,
            "dark_cloud_cover": PatternType.BEARISH,
            
            # 反转形态
            "doji": PatternType.REVERSAL,
            "dragonfly_doji": PatternType.REVERSAL,
            "gravestone_doji": PatternType.REVERSAL,
            "harami": PatternType.REVERSAL,
            "harami_cross": PatternType.REVERSAL,
            
            # 持续形态
            "rise_fall_three_methods": PatternType.CONTINUATION,
            "mat_hold": PatternType.CONTINUATION,
            "tasuki_gap": PatternType.CONTINUATION,
            "xside_gap_three_methods": PatternType.CONTINUATION,
        }
        
        # 形态强度映射
        self.pattern_strengths = {
            # 强信号
            "morning_star": PatternStrength.STRONG,
            "evening_star": PatternStrength.STRONG,
            "three_white_soldiers": PatternStrength.STRONG,
            "three_black_crows": PatternStrength.STRONG,
            "abandoned_baby": PatternStrength.STRONG,
            
            # 中等信号
            "engulfing": PatternStrength.MEDIUM,
            "hammer": PatternStrength.MEDIUM,
            "hanging_man": PatternStrength.MEDIUM,
            "piercing": PatternStrength.MEDIUM,
            "dark_cloud_cover": PatternStrength.MEDIUM,
            "harami": PatternStrength.MEDIUM,
            
            # 弱信号
            "doji": PatternStrength.WEAK,
            "spinning_top": PatternStrength.WEAK,
            "high_wave": PatternStrength.WEAK,
            "short_line": PatternStrength.WEAK,
        }
    
    def add_indicator(self, indicator_type: str, **params) -> None:
        """
        添加指标到分析器
        
        Args:
            indicator_type: 指标类型名称，如 'MACD', 'KDJ'
            **params: 指标参数
        """
        # 检查指标是否已存在
        if indicator_type in self.indicators:
            logger.warning(f"指标 {indicator_type} 已存在，不再重复添加")
            return
        
        # 创建指标实例
        indicator = IndicatorFactory.create(indicator_type, **params)
        
        if indicator is None:
            logger.error(f"无法创建指标 {indicator_type}")
            return
        
        # 添加到指标列表和缓存
        self.indicators.append(indicator_type)
        self._indicator_instances[indicator_type] = indicator
        
        logger.info(f"已添加指标 {indicator_type} 到分析器")
    
    def add_period(self, period: str) -> None:
        """
        添加周期到分析器
        
        Args:
            period: 周期名称，如 'DAILY', 'WEEKLY'
        """
        if period in self.periods:
            logger.warning(f"周期 {period} 已存在，不再重复添加")
            return
        
        self.periods.append(period)
        logger.info(f"已添加周期 {period} 到分析器")
    
    def identify_patterns(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        使用 TA-Lib 识别所有支持的K线形态

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            Dict[str, pd.DataFrame]: 包含所有已识别形态的字典，
                                     键是形态名称，值是包含形态信息的DataFrame
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            logger.error("输入数据缺少OHLC列，无法进行形态识别")
            return {}

        all_patterns = {}
        # 确保列是小写的，以匹配talib的要求
        df = data.rename(columns=str.lower)

        for pattern_name, pattern_func in self.talib_patterns.items():
            try:
                # 调用TA-Lib函数进行形态识别
                result = pattern_func(df['open'], df['high'], df['low'], df['close'])
                
                # 找出形态出现的位置
                pattern_dates = result[result != 0].index
                
                if not pattern_dates.empty:
                    # 获取形态出现日期的原始数据行
                    pattern_data = data.loc[pattern_dates].copy()
                    pattern_data['pattern_name'] = pattern_name
                    pattern_data['strength'] = result.loc[pattern_dates]
                    all_patterns[pattern_name] = pattern_data

            except Exception as e:
                logger.error(f"识别形态 {pattern_name} 时出错: {e}")
        
        return all_patterns
    
    def get_indicator_instance(self, indicator_type: str) -> Optional[BaseIndicator]:
        """
        获取指标实例
        
        Args:
            indicator_type: 指标类型名称
            
        Returns:
            Optional[BaseIndicator]: 指标实例，如果不存在则返回None
        """
        # 如果缓存中已有实例，直接返回
        if indicator_type in self._indicator_instances:
            return self._indicator_instances[indicator_type]
        
        # 否则创建新实例
        if indicator_type not in self.indicators:
            self.indicators.append(indicator_type)
            
        indicator = IndicatorFactory.create(indicator_type)
        
        if indicator is not None:
            self._indicator_instances[indicator_type] = indicator
            
        return indicator
    
    @validate_dataframe(required_columns=['open', 'high', 'low', 'close', 'volume'], min_rows=30)
    @log_calls(level='debug')
    @exception_handler(reraise=False)
    def analyze(self, data: Dict[str, pd.DataFrame], stock_code: str = None, 
               stock_name: str = None) -> Dict[str, Any]:
        """
        分析指定股票在多个周期上的形态
        
        Args:
            data: 按周期组织的数据字典，如 {'DAILY': daily_df, 'WEEKLY': weekly_df}
            stock_code: 股票代码
            stock_name: 股票名称
            
        Returns:
            Dict[str, Any]: 形态分析结果
        """
        self.current_stock_code = stock_code
        self.current_stock_name = stock_name
        
        # 清除形态管理器的历史记录
        self.pattern_manager.clear_occurrences()
        
        # 分析结果
        results = {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "periods": {},
            "common_patterns": [],
            "scores": {}
        }
        
        # 按周期分析
        for period, period_data in data.items():
            if period not in self.periods:
                logger.debug(f"跳过未配置的周期 {period}")
                continue
                
            if period_data.empty:
                logger.warning(f"周期 {period} 的数据为空")
                continue
                
            # 获取最新日期
            latest_date = period_data.index[-1] if isinstance(period_data.index, pd.DatetimeIndex) else period_data.iloc[-1].name
            latest_date_str = latest_date.strftime("%Y-%m-%d") if isinstance(latest_date, (pd.Timestamp, datetime)) else str(latest_date)
            
            # 按指标分析形态
            period_results = self._analyze_period(period, period_data, latest_date_str)
            
            # 添加到结果字典
            results["periods"][period] = period_results
        
        # 获取跨周期共同形态
        results["common_patterns"] = self.pattern_manager.get_common_patterns(
            periods=list(data.keys()),
            min_occurrence=2
        )
        
        # 计算综合评分
        # 获取最新日期的评分
        latest_date = max([
            max([pattern.date for pattern in self.pattern_manager.pattern_occurrences])
            if self.pattern_manager.pattern_occurrences else "1970-01-01"
        ])
        
        results["scores"] = self.pattern_manager.calculate_pattern_score(
            date=latest_date,
            weight_by_period=True
        )
        
        # 添加统计信息
        results["statistics"] = self.pattern_manager.get_pattern_statistics()
        
        # 缓存结果
        self._results_cache = results
        
        return results
    
    def _analyze_period(self, period: str, data: pd.DataFrame, 
                       date: str) -> Dict[str, Any]:
        """
        分析单个周期的形态
        
        Args:
            period: 周期名称
            data: 周期数据
            date: 日期
            
        Returns:
            Dict[str, Any]: 周期形态分析结果
        """
        period_results = {
            "patterns": [],
            "scores": {},
            "market_environment": MarketEnvironment.SIDEWAYS_MARKET.value
        }
        
        # 按指标识别形态
        for indicator_type in self.indicators:
            indicator = self.get_indicator_instance(indicator_type)
            
            if indicator is None:
                logger.warning(f"无法获取指标 {indicator_type} 的实例")
                continue
            
            # 计算指标
            try:
                indicator.calculate(data)
                
                # 检测市场环境
                market_env = indicator.detect_market_environment(data)
                period_results["market_environment"] = market_env.value
                
                # 识别形态
                patterns = indicator.get_patterns(data)
                
                if patterns:
                    # 为每个形态添加周期信息
                    for pattern in patterns:
                        pattern['period'] = period
                        # 添加信号类型
                        if 'score_impact' in pattern:
                            if pattern['score_impact'] > 0:
                                pattern['signal_type'] = 'bullish'
                            elif pattern['score_impact'] < 0:
                                pattern['signal_type'] = 'bearish'
                            else:
                                pattern['signal_type'] = 'neutral'
                    
                    # 将形态添加到结果
                    period_results["patterns"].extend(patterns)
                    
                    # 注册到形态管理器
                    self.pattern_manager.register_multiple_patterns(
                        patterns=patterns,
                        indicator_id=indicator_type,
                        period=period,
                        date=date
                    )
                
                # 计算指标评分
                score_result = indicator.calculate_score(data)
                
                # 处理不同的评分返回格式
                if isinstance(score_result, dict) and 'total_score' in score_result:
                    period_results["scores"][indicator_type] = score_result['total_score']
                elif isinstance(score_result, (int, float)):
                    period_results["scores"][indicator_type] = score_result
                else:
                    # 如果无法获取评分，给予默认中性评分
                    period_results["scores"][indicator_type] = 50.0
                
            except Exception as e:
                logger.error(f"分析指标 {indicator_type} 出错: {e}")
        
        # 计算综合评分
        if period_results["scores"]:
            # 使用加权平均计算综合评分
            total_score = 0.0
            total_weight = 0.0
            
            for indicator_type, score in period_results["scores"].items():
                indicator = self.get_indicator_instance(indicator_type)
                weight = getattr(indicator, 'weight', 1.0)
                total_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                period_results["scores"]["TOTAL"] = total_score / total_weight
            else:
                period_results["scores"]["TOTAL"] = 50.0  # 默认中性评分
        
        return period_results
    
    def get_pattern_details(self, pattern_id: str, indicator_id: str = None) -> Dict[str, Any]:
        """
        获取指定形态的详细信息
        
        Args:
            pattern_id: 形态ID
            indicator_id: 指标ID，如果为None则返回所有指标中的该形态
            
        Returns:
            Dict[str, Any]: 形态详细信息
        """
        pattern_info = self.pattern_registry.get_pattern_info(pattern_id)
        
        if not pattern_info:
            return {"error": f"未找到形态 {pattern_id}"}
        
        # 转换为字典
        result = {
            "pattern_id": pattern_id,
            "display_name": pattern_info.display_name,
            "indicator_type": pattern_info.indicator_type,
            "pattern_type": pattern_info.pattern_type.value,
            "strength": pattern_info.strength.value,
            "occurrences": []
        }
        
        # 获取形态出现记录
        occurrences = []
        for p in self.pattern_manager.pattern_occurrences:
            if p.pattern_id == pattern_id and (indicator_id is None or p.indicator_id == indicator_id):
                occurrences.append({
                    "indicator_id": p.indicator_id,
                    "period": p.period,
                    "date": p.date,
                    "strength": p.strength,
                    "details": p.details
                })
        
        result["occurrences"] = occurrences
        result["occurrence_count"] = len(occurrences)
        
        return result
    
    def get_latest_patterns(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最新的形态列表
        
        Args:
            top_n: 返回的形态数量
            
        Returns:
            List[Dict[str, Any]]: 形态列表
        """
        # 将形态管理器中的形态按日期排序
        sorted_patterns = sorted(
            self.pattern_manager.pattern_occurrences,
            key=lambda p: p.date,
            reverse=True
        )
        
        # 取前top_n个
        latest_patterns = sorted_patterns[:top_n]
        
        # 转换为字典列表
        result = []
        for p in latest_patterns:
            result.append({
                "pattern_id": p.pattern_id,
                "display_name": p.display_name,
                "indicator_id": p.indicator_id,
                "period": p.period,
                "date": p.date,
                "strength": p.strength,
                "signal_type": self.pattern_registry.get_signal_type(p.pattern_id),
                "details": p.details
            })
        
        return result
    
    def get_indicator_score(self, indicator_type: str) -> Dict[str, Any]:
        """
        获取指定指标的评分详情
        
        Args:
            indicator_type: 指标类型名称
            
        Returns:
            Dict[str, Any]: 评分详情
        """
        result = {
            "indicator_type": indicator_type,
            "scores": {},
            "patterns": []
        }
        
        # 从缓存中获取各周期的评分
        for period, period_data in self._results_cache.get("periods", {}).items():
            if indicator_type in period_data.get("scores", {}):
                result["scores"][period] = period_data["scores"][indicator_type]
        
        # 获取该指标的所有形态
        patterns = self.pattern_manager.get_patterns_by_indicator(indicator_type)
        
        # 转换为字典列表
        pattern_list = []
        for p in patterns:
            pattern_list.append({
                "pattern_id": p.pattern_id,
                "display_name": p.display_name,
                "period": p.period,
                "date": p.date,
                "strength": p.strength,
                "signal_type": self.pattern_registry.get_signal_type(p.pattern_id),
                "details": p.details
            })
        
        result["patterns"] = pattern_list
        
        # 计算平均评分
        if result["scores"]:
            result["average_score"] = sum(result["scores"].values()) / len(result["scores"])
        else:
            result["average_score"] = 50.0  # 默认中性评分
        
        return result
    
    def get_period_patterns(self, period: str) -> Dict[str, Any]:
        """
        获取指定周期的形态分析结果
        
        Args:
            period: 周期名称
            
        Returns:
            Dict[str, Any]: 周期形态分析结果
        """
        if period not in self._results_cache.get("periods", {}):
            return {"error": f"未找到周期 {period} 的分析结果"}
        
        return self._results_cache["periods"][period]
    
    def get_score_summary(self) -> Dict[str, Any]:
        """
        获取综合评分摘要
        
        Returns:
            Dict[str, Any]: 评分摘要
        """
        result = {
            "total_score": 50.0,  # 默认中性评分
            "by_indicator": {},
            "by_period": {},
            "recommendation": "中性",
            "confidence": 0.0,
            "score_components": {},
            "bullish_patterns": [],
            "bearish_patterns": []
        }
        
        # 从缓存中获取各指标的评分
        indicator_scores = {}
        period_scores = {}
        
        for period, period_data in self._results_cache.get("periods", {}).items():
            # 各周期的总评分
            if "TOTAL" in period_data.get("scores", {}):
                period_scores[period] = period_data["scores"]["TOTAL"]
            
            # 各指标的评分
            for indicator, score in period_data.get("scores", {}).items():
                if indicator != "TOTAL":
                    if indicator not in indicator_scores:
                        indicator_scores[indicator] = []
                    indicator_scores[indicator].append(score)
        
        # 计算各指标的平均评分
        for indicator, scores in indicator_scores.items():
            result["by_indicator"][indicator] = sum(scores) / len(scores)
        
        # 各周期的评分
        result["by_period"] = period_scores
        
        # 总评分
        if "scores" in self._results_cache and "total_score" in self._results_cache["scores"]:
            result["total_score"] = self._results_cache["scores"]["total_score"]
        elif period_scores:
            # 加权平均计算总评分
            weights = {period: self.pattern_manager.period_weights.get(period, 1.0) for period in period_scores.keys()}
            weighted_sum = sum(score * weights.get(period, 1.0) for period, score in period_scores.items())
            total_weight = sum(weights.values())
            result["total_score"] = weighted_sum / total_weight if total_weight > 0 else 50.0
        
        # 验证评分
        result["total_score"] = validate_score(result["total_score"])
        
        # 提取看涨和看跌形态
        bullish_patterns = []
        bearish_patterns = []
        
        # 按贡献度计算评分组成
        score_components = {}
        
        for period, period_data in self._results_cache.get("periods", {}).items():
            for pattern in period_data.get("patterns", []):
                signal_type = pattern.get("signal_type", "")
                pattern_with_period = {**pattern, "period": period}
                
                if signal_type == "bullish":
                    bullish_patterns.append(pattern_with_period)
                    component_key = f"{pattern['indicator_id']}_{pattern['pattern_id']}"
                    if component_key not in score_components:
                        score_components[component_key] = 0
                    score_components[component_key] += pattern.get('score_impact', 0) * self.pattern_manager.period_weights.get(period, 1.0)
                elif signal_type == "bearish":
                    bearish_patterns.append(pattern_with_period)
                    component_key = f"{pattern['indicator_id']}_{pattern['pattern_id']}"
                    if component_key not in score_components:
                        score_components[component_key] = 0
                    score_components[component_key] += pattern.get('score_impact', 0) * self.pattern_manager.period_weights.get(period, 1.0)
        
        # 按影响程度排序
        result["bullish_patterns"] = sorted(bullish_patterns, key=lambda x: x.get('score_impact', 0), reverse=True)[:5]
        result["bearish_patterns"] = sorted(bearish_patterns, key=lambda x: x.get('score_impact', 0))[:5]
        
        # 按绝对值排序评分组成
        sorted_components = sorted(score_components.items(), key=lambda x: abs(x[1]), reverse=True)
        result["score_components"] = {k: v for k, v in sorted_components[:10]}
        
        # 评分置信度计算
        pattern_count = self._results_cache.get("statistics", {}).get("total_patterns", 0)
        indicator_count = len(self.indicators)
        period_count = len(self.periods)
        
        # 置信度基于模式数量、指标数量和周期数量
        confidence_base = min(1.0, pattern_count / 10)  # 至少10个模式才有较高置信度
        confidence_multiplier = min(1.0, (indicator_count * period_count) / 15)  # 指标和周期的乘积至少15才有较高置信度
        
        result["confidence"] = confidence_base * confidence_multiplier
        
        # 推荐意见
        if result["total_score"] >= 70:
            result["recommendation"] = "强烈看涨"
        elif result["total_score"] >= 60:
            result["recommendation"] = "看涨"
        elif result["total_score"] <= 30:
            result["recommendation"] = "强烈看跌"
        elif result["total_score"] <= 40:
            result["recommendation"] = "看跌"
        else:
            result["recommendation"] = "中性"
        
        return result
    
    def to_json(self) -> str:
        """
        将分析结果转换为JSON字符串
        
        Returns:
            str: JSON字符串
        """
        # 直接返回缓存的结果
        return json.dumps(self._results_cache, indent=2, ensure_ascii=False)
    
    def save_results(self, filepath: str) -> None:
        """
        保存分析结果到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._results_cache, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存分析结果到 {filepath}")

    def clear(self) -> None:
        """清除所有分析结果和缓存"""
        self.pattern_manager.clear_occurrences()
        self._results_cache = {}
        self.current_stock_code = None
        self.current_stock_name = None
        logger.info("已清除所有分析结果和缓存")

    def get_pattern_combinations(self, min_occurrence: int = 2, 
                               min_strength: float = 60.0,
                               max_combinations: int = 20) -> List[Dict[str, Any]]:
        """
        获取出现频率最高的形态组合
        
        Args:
            min_occurrence: 最小出现次数
            min_strength: 最小形态强度
            max_combinations: 最多返回的组合数量
            
        Returns:
            List[Dict[str, Any]]: 形态组合列表
        """
        # 收集所有符合强度要求的形态
        all_patterns = []
        
        # 遍历周期和指标
        for period, period_data in self._results_cache.get("periods", {}).items():
            for indicator_id, indicator_data in period_data.get("indicators", {}).items():
                patterns = indicator_data.get("patterns", [])
                
                # 过滤出强度符合要求的形态
                strong_patterns = [
                    {
                        "period": period,
                        "indicator_id": indicator_id,
                        "pattern_id": p.get("pattern_id"),
                        "display_name": p.get("display_name"),
                        "strength": p.get("strength", 0),
                        "duration": p.get("duration", 1)
                    }
                    for p in patterns if p.get("strength", 0) >= min_strength
                ]
                
                all_patterns.extend(strong_patterns)
        
        # 如果没有符合条件的形态，返回空列表
        if not all_patterns:
            return []
        
        # 计算形态组合
        combinations = []
        
        # 1. 生成二元形态组合
        from itertools import combinations as iter_combinations
        for p1, p2 in iter_combinations(all_patterns, 2):
            # 创建组合的唯一标识符
            combo_id = f"{p1['period']}_{p1['indicator_id']}_{p1['pattern_id']}+{p2['period']}_{p2['indicator_id']}_{p2['pattern_id']}"
            
            # 计算组合评分
            combo_score = (p1['strength'] + p2['strength']) / 2
            
            combinations.append({
                "combo_id": combo_id,
                "patterns": [p1, p2],
                "score": combo_score,
                "occurrence": 1
            })
        
        # 2. 统计组合出现次数
        from collections import defaultdict
        combo_counter = defaultdict(int)
        for combo in combinations:
            combo_counter[combo["combo_id"]] += 1
        
        # 3. 过滤出频次满足要求的组合
        filtered_combinations = [
            combo for combo in combinations 
            if combo_counter[combo["combo_id"]] >= min_occurrence
        ]
        
        # 4. 去重并排序
        unique_combinations = {}
        for combo in filtered_combinations:
            if combo["combo_id"] not in unique_combinations:
                unique_combinations[combo["combo_id"]] = combo
        
        # 按评分和出现次数排序
        sorted_combinations = sorted(
            unique_combinations.values(),
            key=lambda x: (combo_counter[x["combo_id"]], x["score"]),
            reverse=True
        )
        
        # 返回前N个结果
        return sorted_combinations[:max_combinations]

    def validate_pattern_combination(self, combination: Dict[str, Any]) -> bool:
        """
        验证形态组合是否有效
        
        Args:
            combination: 形态组合
            
        Returns:
            bool: 是否是有效的形态组合
        """
        patterns = combination.get("patterns", [])
        if len(patterns) < 2:
            return False
        
        # 验证所有形态是否都在同一时间窗口内出现
        # 这里假设形态都是最近发现的
        return True

    def find_cross_period_patterns(self, periods: List[str] = None) -> List[Dict[str, Any]]:
        """
        查找跨周期形态组合
        
        Args:
            periods: 要分析的周期列表，如果为None则使用所有周期
            
        Returns:
            List[Dict[str, Any]]: 跨周期形态组合列表
        """
        if periods is None:
            periods = list(self._results_cache.get("periods", {}).keys())
        
        if len(periods) < 2:
            logger.warning("至少需要两个周期才能进行跨周期分析")
            return []
        
        # 收集各周期的形态
        period_patterns = {}
        for period in periods:
            period_patterns[period] = []
            
            period_data = self._results_cache.get("periods", {}).get(period, {})
            for indicator_id, indicator_data in period_data.get("indicators", {}).items():
                patterns = indicator_data.get("patterns", [])
                
                for pattern in patterns:
                    period_patterns[period].append({
                        "period": period,
                        "indicator_id": indicator_id,
                        "pattern_id": pattern.get("pattern_id"),
                        "display_name": pattern.get("display_name"),
                        "strength": pattern.get("strength", 0),
                        "duration": pattern.get("duration", 1)
                    })
        
        # 寻找跨周期的相同指标形态
        cross_period_patterns = []
        
        # 遍历所有指标
        all_indicators = set()
        for period in periods:
            for pattern in period_patterns[period]:
                all_indicators.add(pattern["indicator_id"])
        
        # 对每个指标，查找在多个周期中出现的形态
        for indicator_id in all_indicators:
            indicator_patterns = {}
            
            # 收集该指标在各周期的形态
            for period in periods:
                indicator_patterns[period] = [
                    p for p in period_patterns[period] 
                    if p["indicator_id"] == indicator_id
                ]
            
            # 检查是否有形态在多个周期中出现
            for pattern_type in ["golden_cross", "dead_cross", "overbought", "oversold", "bullish", "bearish"]:
                pattern_periods = []
                pattern_details = []
                
                for period in periods:
                    matching_patterns = [
                        p for p in indicator_patterns[period]
                        if pattern_type in p["pattern_id"].lower()
                    ]
                    
                    if matching_patterns:
                        pattern_periods.append(period)
                        pattern_details.extend(matching_patterns)
                
                # 如果形态在多个周期中出现
                if len(pattern_periods) >= 2:
                    # 计算组合评分
                    avg_strength = sum(p["strength"] for p in pattern_details) / len(pattern_details)
                    
                    cross_period_patterns.append({
                        "indicator_id": indicator_id,
                        "pattern_type": pattern_type,
                        "periods": pattern_periods,
                        "patterns": pattern_details,
                        "strength": avg_strength,
                        "is_cross_period": True
                    })
        
        # 按强度排序
        cross_period_patterns.sort(key=lambda x: x["strength"], reverse=True)
        
        return cross_period_patterns

    def calculate_combination_score(self, combination: Dict[str, Any]) -> float:
        """
        计算形态组合的评分
        
        Args:
            combination: 形态组合
            
        Returns:
            float: 组合评分 (0-100)
        """
        patterns = combination.get("patterns", [])
        if not patterns:
            return 0.0
        
        # 基础评分：形态强度的加权平均
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            # 获取形态的权重
            weight = self._get_pattern_weight(pattern)
            strength = pattern.get("strength", 50.0)
            
            weighted_sum += strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        base_score = weighted_sum / total_weight
        
        # 调整因素：形态数量
        pattern_count_factor = min(1.2, 1 + (len(patterns) - 1) * 0.1)  # 每多一个形态增加10%，最多增加20%
        
        # 调整因素：跨周期确认
        periods = set(pattern["period"] for pattern in patterns)
        cross_period_factor = min(1.3, 1 + (len(periods) - 1) * 0.15)  # 每多一个周期增加15%，最多增加30%
        
        # 调整因素：指标多样性
        indicators = set(pattern["indicator_id"] for pattern in patterns)
        diversity_factor = min(1.25, 1 + (len(indicators) - 1) * 0.125)  # 每多一种指标增加12.5%，最多增加25%
        
        # 计算最终评分
        final_score = base_score * pattern_count_factor * cross_period_factor * diversity_factor
        
        # 确保分数在0-100范围内
        final_score = max(0, min(100, final_score))
        
        return final_score

    def _get_pattern_weight(self, pattern: Dict[str, Any]) -> float:
        """
        获取形态的权重
        
        Args:
            pattern: 形态信息
            
        Returns:
            float: 形态权重
        """
        # 基础权重
        base_weight = 1.0
        
        # 根据指标类型调整权重
        indicator_id = pattern.get("indicator_id", "")
        
        if "MACD" in indicator_id:
            base_weight = 1.2
        elif "KDJ" in indicator_id:
            base_weight = 1.1
        elif "RSI" in indicator_id:
            base_weight = 1.0
        elif "BOLL" in indicator_id:
            base_weight = 1.1
        elif "MA" in indicator_id:
            base_weight = 0.9
        elif "VOL" in indicator_id:
            base_weight = 0.8
        
        # 根据周期调整权重
        period = pattern.get("period", "DAILY")
        
        if period == "DAILY":
            period_factor = 1.0
        elif period == "WEEKLY":
            period_factor = 1.2
        elif period == "MONTHLY":
            period_factor = 1.3
        elif "MIN" in period:
            # 分钟周期权重较低
            period_factor = 0.8
        else:
            period_factor = 1.0
        
        # 计算最终权重
        return base_weight * period_factor

    def get_bullish_combinations(self, min_score: float = 70.0, 
                               max_results: int = 10) -> List[Dict[str, Any]]:
        """
        获取看涨形态组合
        
        Args:
            min_score: 最小评分
            max_results: 最多返回的结果数量
            
        Returns:
            List[Dict[str, Any]]: 看涨形态组合列表
        """
        # 获取所有形态组合
        all_combinations = self.get_pattern_combinations(min_occurrence=1, min_strength=50.0)
        
        # 过滤出看涨形态组合
        bullish_patterns = ["golden_cross", "bullish", "oversold", "positive_divergence", "support", "breakout"]
        
        bullish_combinations = []
        for combo in all_combinations:
            patterns = combo.get("patterns", [])
            
            # 检查是否包含看涨形态
            is_bullish = False
            for pattern in patterns:
                pattern_id = pattern.get("pattern_id", "").lower()
                if any(bp in pattern_id for bp in bullish_patterns):
                    is_bullish = True
                    break
            
            if is_bullish:
                # 计算组合评分
                combo_score = self.calculate_combination_score(combo)
                
                if combo_score >= min_score:
                    bullish_combinations.append({
                        **combo,
                        "score": combo_score,
                        "type": "bullish"
                    })
        
        # 按评分排序
        bullish_combinations.sort(key=lambda x: x["score"], reverse=True)
        
        return bullish_combinations[:max_results]

    def get_bearish_combinations(self, min_score: float = 70.0,
                               max_results: int = 10) -> List[Dict[str, Any]]:
        """
        获取看跌形态组合
        
        Args:
            min_score: 最小评分
            max_results: 最多返回的结果数量
            
        Returns:
            List[Dict[str, Any]]: 看跌形态组合列表
        """
        # 获取所有形态组合
        all_combinations = self.get_pattern_combinations(min_occurrence=1, min_strength=50.0)
        
        # 过滤出看跌形态组合
        bearish_patterns = ["dead_cross", "bearish", "overbought", "negative_divergence", "resistance", "breakdown"]
        
        bearish_combinations = []
        for combo in all_combinations:
            patterns = combo.get("patterns", [])
            
            # 检查是否包含看跌形态
            is_bearish = False
            for pattern in patterns:
                pattern_id = pattern.get("pattern_id", "").lower()
                if any(bp in pattern_id for bp in bearish_patterns):
                    is_bearish = True
                    break
            
            if is_bearish:
                # 计算组合评分
                combo_score = self.calculate_combination_score(combo)
                
                if combo_score >= min_score:
                    bearish_combinations.append({
                        **combo,
                        "score": combo_score,
                        "type": "bearish"
                    })
        
        # 按评分排序
        bearish_combinations.sort(key=lambda x: x["score"], reverse=True)
        
        return bearish_combinations[:max_results]

    def get_strongest_pattern_combination(self) -> Optional[Dict[str, Any]]:
        """
        获取最强的形态组合
        
        Returns:
            Optional[Dict[str, Any]]: 最强的形态组合，如果没有则返回None
        """
        # 获取看涨和看跌形态组合
        bullish = self.get_bullish_combinations(min_score=0.0, max_results=1)
        bearish = self.get_bearish_combinations(min_score=0.0, max_results=1)
        
        # 选择评分最高的组合
        if not bullish and not bearish:
            return None
        
        if not bullish:
            return bearish[0]
        
        if not bearish:
            return bullish[0]
        
        return bullish[0] if bullish[0]["score"] >= bearish[0]["score"] else bearish[0]

    @performance_monitor()
    @cache_result(cache_size=100)
    def recognize_patterns(self, stock_data: pd.DataFrame, 
                         patterns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        识别K线形态
        
        Args:
            stock_data: 股票K线数据，必须包含open, high, low, close列
            patterns: 要识别的形态列表，如果为None则识别所有支持的形态
            
        Returns:
            pd.DataFrame: 包含形态识别结果的DataFrame
        """
        # 检查必要的列
        required_columns = ["open", "high", "low", "close"]
        for col in required_columns:
            if col not in stock_data.columns:
                logger.error(f"股票数据缺少必要的列: {col}")
                return pd.DataFrame()
        
        # 如果未指定形态，使用所有支持的形态
        if patterns is None:
            patterns = list(self.talib_patterns.keys())
        
        # 识别形态
        pattern_results = {}
        
        for pattern in patterns:
            if pattern in self.talib_patterns:
                try:
                    # 调用TA-Lib函数识别形态
                    pattern_func = self.talib_patterns[pattern]
                    result = pattern_func(
                        stock_data["open"].values, 
                        stock_data["high"].values,
                        stock_data["low"].values,
                        stock_data["close"].values
                    )
                    
                    # 将结果添加到结果字典
                    pattern_results[f"pattern_{pattern}"] = result
                    
                except Exception as e:
                    logger.error(f"识别形态 {pattern} 时出错: {e}")
            else:
                logger.warning(f"不支持的形态: {pattern}")
        
        # 如果没有识别出任何形态，返回空DataFrame
        if not pattern_results:
            logger.warning("未识别出任何形态")
            return pd.DataFrame()
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(pattern_results, index=stock_data.index)
        
        return result_df

    @performance_monitor()
    def recognize_pattern_on_date(self, stock_data: pd.DataFrame, 
                               pattern: str, date: str) -> bool:
        """
        识别指定日期的K线形态
        
        Args:
            stock_data: 股票K线数据
            pattern: 形态名称
            date: 日期
            
        Returns:
            bool: 是否识别出指定形态
        """
        # 识别所有形态
        pattern_results = self.recognize_patterns(stock_data, [pattern])
        
        if pattern_results.empty:
            return False
        
        # 获取指定日期的形态结果
        pattern_col = f"pattern_{pattern}"
        
        if pattern_col not in pattern_results.columns:
            return False
        
        # 找到日期位置
        if date in pattern_results.index:
            value = pattern_results.loc[date, pattern_col]
        else:
            # 尝试找到最接近的日期
            try:
                nearest_date = pattern_results.index[pattern_results.index <= date][-1]
                value = pattern_results.loc[nearest_date, pattern_col]
            except (IndexError, KeyError):
                logger.warning(f"未找到日期 {date} 或之前的数据")
                return False
        
        # 判断是否识别出形态（非零值表示识别出）
        return value != 0
    
    @performance_monitor()
    def get_pattern_type(self, pattern: str) -> PatternType:
        """
        获取形态类型
        
        Args:
            pattern: 形态名称
            
        Returns:
            PatternType: 形态类型
        """
        # 处理engulfing特殊情况
        if pattern == "engulfing":
            return PatternType.REVERSAL  # 吞噬形态可能是看涨或看跌的反转形态
        
        return self.pattern_types.get(pattern, PatternType.REVERSAL)
    
    @performance_monitor()
    def get_pattern_strength(self, pattern: str) -> PatternStrength:
        """
        获取形态强度
        
        Args:
            pattern: 形态名称
            
        Returns:
            PatternStrength: 形态强度
        """
        return self.pattern_strengths.get(pattern, PatternStrength.WEAK)
    
    @performance_monitor()
    def detect_pattern_sequence(self, stock_data: pd.DataFrame, 
                             pattern_sequence: List[str],
                             max_days_between: int = 3) -> pd.DataFrame:
        """
        检测形态序列
        
        Args:
            stock_data: 股票K线数据
            pattern_sequence: 形态序列列表
            max_days_between: 序列中形态之间的最大间隔天数
            
        Returns:
            pd.DataFrame: 序列检测结果
        """
        if len(pattern_sequence) < 2:
            logger.warning("形态序列至少需要两个形态")
            return pd.DataFrame()
        
        # 识别所有需要的形态
        pattern_results = self.recognize_patterns(stock_data, pattern_sequence)
        
        if pattern_results.empty:
            return pd.DataFrame()
        
        # 创建序列检测结果列
        sequence_col = f"pattern_sequence_{'_'.join(pattern_sequence)}"
        pattern_results[sequence_col] = 0
        
        # 遍历每一天
        for i in range(len(pattern_results) - max_days_between * (len(pattern_sequence) - 1)):
            sequence_found = True
            last_index = i
            
            # 检查每个形态是否按顺序出现
            for pattern in pattern_sequence:
                pattern_col = f"pattern_{pattern}"
                
                # 在允许的间隔内查找形态
                found = False
                for j in range(last_index, min(last_index + max_days_between + 1, len(pattern_results))):
                    if pattern_results.iloc[j][pattern_col] != 0:
                        last_index = j + 1
                        found = True
                        break
                
                if not found:
                    sequence_found = False
                    break
            
            # 如果找到完整序列，标记起始日期
            if sequence_found:
                pattern_results.iloc[i][sequence_col] = 1
        
        return pattern_results
    
    @performance_monitor()
    def get_dominant_patterns(self, stock_data: pd.DataFrame, 
                           lookback_days: int = 20) -> Dict[str, int]:
        """
        获取主导形态
        
        Args:
            stock_data: 股票K线数据
            lookback_days: 回看天数
            
        Returns:
            Dict[str, int]: 形态名称和出现次数的字典
        """
        # 识别所有形态
        pattern_results = self.recognize_patterns(stock_data)
        
        if pattern_results.empty:
            return {}
        
        # 获取最近N天的数据
        recent_data = pattern_results.iloc[-min(lookback_days, len(pattern_results)):]
        
        # 统计每种形态的出现次数
        pattern_counts = {}
        
        for col in recent_data.columns:
            if col.startswith("pattern_"):
                pattern_name = col[8:]  # 去掉"pattern_"前缀
                
                # 统计非零值的数量
                count = (recent_data[col] != 0).sum()
                
                if count > 0:
                    pattern_counts[pattern_name] = count
        
        # 按出现次数降序排序
        return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True))
    
    @performance_monitor()
    def analyze_pattern_reliability(self, stock_data: pd.DataFrame, 
                                 pattern: str, 
                                 forward_days: int = 5) -> Dict[str, float]:
        """
        分析形态可靠性
        
        Args:
            stock_data: 股票K线数据
            pattern: 形态名称
            forward_days: 向前看的天数
            
        Returns:
            Dict[str, float]: 可靠性分析结果
        """
        # 识别指定形态
        pattern_results = self.recognize_patterns(stock_data, [pattern])
        
        if pattern_results.empty:
            return {}
        
        pattern_col = f"pattern_{pattern}"
        
        if pattern_col not in pattern_results.columns:
            return {}
        
        # 合并K线数据和形态结果
        merged_data = stock_data.copy()
        merged_data[pattern_col] = pattern_results[pattern_col]
        
        # 找出形态出现的日期
        pattern_dates = merged_data[merged_data[pattern_col] != 0].index
        
        if len(pattern_dates) == 0:
            return {"occurrences": 0}
        
        # 分析形态后的价格变化
        success_count = 0
        avg_gain = 0.0
        max_gain = -float('inf')
        min_gain = float('inf')
        
        for date in pattern_dates:
            date_idx = merged_data.index.get_loc(date)
            
            # 检查是否有足够的后续数据
            if date_idx + forward_days >= len(merged_data):
                continue
            
            # 获取形态信号（正值表示看涨，负值表示看跌）
            signal = merged_data.iloc[date, pattern_col]
            
            # 获取后续价格变化
            current_price = merged_data.iloc[date_idx]["close"]
            future_price = merged_data.iloc[date_idx + forward_days]["close"]
            price_change = (future_price - current_price) / current_price * 100
            
            # 根据信号类型判断成功与否
            if (signal > 0 and price_change > 0) or (signal < 0 and price_change < 0):
                success_count += 1
            
            # 累计收益
            avg_gain += abs(price_change)
            max_gain = max(max_gain, price_change)
            min_gain = min(min_gain, price_change)
        
        # 计算成功率和平均收益
        success_rate = success_count / len(pattern_dates) * 100
        avg_gain /= len(pattern_dates)
        
        return {
            "occurrences": len(pattern_dates),
            "success_rate": success_rate,
            "avg_gain": avg_gain,
            "max_gain": max_gain,
            "min_gain": min_gain
        }
    
    def combine_patterns(self, stock_data: pd.DataFrame, 
                      pattern_list: List[str]) -> pd.DataFrame:
        """
        组合多个形态的识别结果
        
        Args:
            stock_data: 股票K线数据
            pattern_list: 形态列表
            
        Returns:
            pd.DataFrame: 组合后的结果
        """
        # 识别所有指定形态
        pattern_results = self.recognize_patterns(stock_data, pattern_list)
        
        if pattern_results.empty:
            return pd.DataFrame()
        
        # 创建组合结果列
        combined_col = f"pattern_combined_{'_'.join(pattern_list)}"
        pattern_results[combined_col] = 0
        
        # 组合形态结果（任一形态出现即标记为1）
        for pattern in pattern_list:
            pattern_col = f"pattern_{pattern}"
            if pattern_col in pattern_results.columns:
                # 非零值表示形态出现
                pattern_results[combined_col] = np.where(
                    (pattern_results[combined_col] != 0) | (pattern_results[pattern_col] != 0),
                    1, 0
                )
        
        return pattern_results
    
    def get_pattern_description(self, pattern: str) -> str:
        """
        获取形态描述
        
        Args:
            pattern: 形态名称
            
        Returns:
            str: 形态描述
        """
        descriptions = {
            "doji": "十字星，开盘价和收盘价接近，表示市场犹豫不决",
            "hammer": "锤子线，看涨反转形态，下影线长，上影线短或没有",
            "hanging_man": "上吊线，看跌反转形态，下影线长，上影线短或没有",
            "inverted_hammer": "倒锤子线，看涨反转形态，上影线长，下影线短或没有",
            "shooting_star": "流星线，看跌反转形态，上影线长，下影线短或没有",
            "marubozu": "光头光脚线，几乎没有上下影线的实体",
            "engulfing": "吞噬形态，第二天的实体完全覆盖前一天的实体",
            "harami": "孕线形态，第二天的实体完全在前一天实体之内",
            "piercing": "刺透形态，看涨反转形态，第二天的阳线收盘价高于前一天阴线的中点",
            "dark_cloud_cover": "乌云盖顶，看跌反转形态，第二天的阴线收盘价低于前一天阳线的中点",
            "morning_star": "晨星，看涨反转形态，由阴线、小实体和阳线组成的三日形态",
            "evening_star": "暮星，看跌反转形态，由阳线、小实体和阴线组成的三日形态",
            "three_white_soldiers": "三白兵，看涨形态，连续三个收盘价逐渐走高的阳线",
            "three_black_crows": "三黑鸦，看跌形态，连续三个收盘价逐渐走低的阴线"
        }
        
        return descriptions.get(pattern, f"未知形态: {pattern}")
    
    def clear_cache(self):
        """清除缓存"""
        self.pattern_cache.clear()
        logger.info("已清除形态识别缓存") 