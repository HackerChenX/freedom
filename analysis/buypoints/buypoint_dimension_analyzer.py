#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db, get_default_config
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_result_dir
from indicators.factory import IndicatorFactory

# 获取日志记录器
logger = get_logger(__name__)

class BuyPointDimensionAnalyzer:
    """
    买点维度分析器 - 对买点进行多维度分析
    
    支持对形态、趋势、时间特征等维度进行买点分析，发现买点共性特征
    """
    
    def __init__(self):
        """初始化买点维度分析器"""
        logger.info("初始化买点维度分析器")
        
        # 获取数据库连接
        config = get_default_config()
        self.ch_db = get_clickhouse_db(config=config)
        
        # 创建指标工厂
        self.indicator_factory = IndicatorFactory()
        
        # 存储分析结果
        self.analysis_results = {
            "pattern_analysis": {},
            "trend_analysis": {},
            "volume_analysis": {},
            "time_analysis": {},
            "indicator_analysis": {}
        }
        
        # 结果输出目录
        self.result_dir = get_result_dir()
        os.makedirs(self.result_dir, exist_ok=True)
        
        logger.info("买点维度分析器初始化完成") 

    def analyze_pattern_features(self, stock_codes: List[str], 
                               start_date: str, end_date: str,
                               period: str = 'DAILY') -> Dict[str, Any]:
        """
        分析买点K线形态特征
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            period: K线周期
            
        Returns:
            Dict: 形态特征分析结果
        """
        logger.info(f"开始分析买点K线形态特征: {len(stock_codes)}只股票, 周期={period}")
        
        try:
            # 形态特征统计
            pattern_stats = {
                "single_candle": {},   # 单根K线形态
                "combined_candle": {}, # 组合K线形态
                "complex_pattern": {}  # 复杂形态
            }
            
            # 获取K线数据
            for stock_code in stock_codes:
                # 获取股票K线数据
                sql = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM stock_{period.lower()}
                WHERE stock_code = '{stock_code}'
                  AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
                """
                kline_data = self.ch_db.query_df(sql)
                
                if kline_data.empty:
                    logger.warning(f"未找到股票 {stock_code} 的K线数据")
                    continue
                
                # 分析单根K线形态
                self._analyze_single_candle_patterns(kline_data, pattern_stats["single_candle"])
                
                # 分析组合K线形态
                self._analyze_combined_candle_patterns(kline_data, pattern_stats["combined_candle"])
                
                # 分析复杂形态
                self._analyze_complex_patterns(kline_data, pattern_stats["complex_pattern"])
            
            # 计算形态出现频率
            result = {
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "stock_count": len(stock_codes),
                "pattern_stats": self._calculate_pattern_frequency(pattern_stats, len(stock_codes))
            }
            
            # 识别高频特征
            result["common_features"] = self._identify_common_pattern_features(result["pattern_stats"])
            
            # 保存到分析结果
            self.analysis_results["pattern_analysis"] = result
            
            logger.info(f"买点K线形态特征分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析买点K线形态特征时出错: {e}")
            return {}
    
    def _analyze_single_candle_patterns(self, kline_data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析单根K线形态"""
        # 计算K线实体和影线
        kline_data['body'] = abs(kline_data['close'] - kline_data['open'])
        kline_data['upper_shadow'] = kline_data['high'] - kline_data[['open', 'close']].max(axis=1)
        kline_data['lower_shadow'] = kline_data[['open', 'close']].min(axis=1) - kline_data['low']
        
        # 计算前一日K线
        kline_data['prev_close'] = kline_data['close'].shift(1)
        kline_data['prev_body'] = kline_data['body'].shift(1)
        
        # 统计各种单K线形态
        # 十字星
        doji = kline_data[kline_data['body'] < kline_data['body'].mean() * 0.3]
        self._update_pattern_count(stats, "十字星", len(doji))
        
        # 锤子线
        hammer = kline_data[(kline_data['lower_shadow'] > kline_data['body'] * 2) & 
                          (kline_data['upper_shadow'] < kline_data['body'] * 0.5)]
        self._update_pattern_count(stats, "锤子线", len(hammer))
        
        # 吊颈线
        hanging_man = kline_data[(kline_data['lower_shadow'] > kline_data['body'] * 2) & 
                               (kline_data['upper_shadow'] < kline_data['body'] * 0.5) &
                               (kline_data['close'] < kline_data['open'])]
        self._update_pattern_count(stats, "吊颈线", len(hanging_man))
        
        # 长腿十字星
        long_legged_doji = kline_data[(kline_data['body'] < kline_data['body'].mean() * 0.3) &
                                    (kline_data['upper_shadow'] > kline_data['body']) &
                                    (kline_data['lower_shadow'] > kline_data['body'])]
        self._update_pattern_count(stats, "长腿十字星", len(long_legged_doji))
        
        # 射击之星
        shooting_star = kline_data[(kline_data['upper_shadow'] > kline_data['body'] * 2) & 
                                 (kline_data['lower_shadow'] < kline_data['body'] * 0.5)]
        self._update_pattern_count(stats, "射击之星", len(shooting_star))
        
        # 穿刺线
        piercing = kline_data[(kline_data['close'] > kline_data['open']) & 
                            (kline_data['open'] < kline_data['prev_close']) &
                            (kline_data['close'] > (kline_data['prev_close'] + kline_data['prev_close'].shift(1)) / 2)]
        self._update_pattern_count(stats, "穿刺线", len(piercing))
    
    def _analyze_combined_candle_patterns(self, kline_data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析组合K线形态"""
        # 获取必要的前置数据
        kline_data['prev_close'] = kline_data['close'].shift(1)
        kline_data['prev_open'] = kline_data['open'].shift(1)
        kline_data['prev_high'] = kline_data['high'].shift(1)
        kline_data['prev_low'] = kline_data['low'].shift(1)
        
        kline_data['prev2_close'] = kline_data['close'].shift(2)
        kline_data['prev2_open'] = kline_data['open'].shift(2)
        kline_data['prev2_high'] = kline_data['high'].shift(2)
        kline_data['prev2_low'] = kline_data['low'].shift(2)
        
        # 统计各种组合K线形态
        # 看涨吞没
        bullish_engulfing = kline_data[(kline_data['open'] < kline_data['prev_close']) & 
                                     (kline_data['close'] > kline_data['prev_open']) &
                                     (kline_data['prev_close'] < kline_data['prev_open'])]
        self._update_pattern_count(stats, "看涨吞没", len(bullish_engulfing))
        
        # 看跌吞没
        bearish_engulfing = kline_data[(kline_data['open'] > kline_data['prev_close']) & 
                                     (kline_data['close'] < kline_data['prev_open']) &
                                     (kline_data['prev_close'] > kline_data['prev_open'])]
        self._update_pattern_count(stats, "看跌吞没", len(bearish_engulfing))
        
        # 启明星
        morning_star = kline_data[(kline_data['prev2_close'] < kline_data['prev2_open']) &  # 第一天下跌
                                (abs(kline_data['prev_close'] - kline_data['prev_open']) < 
                                 abs(kline_data['prev2_close'] - kline_data['prev2_open']) * 0.3) &  # 第二天十字星
                                (kline_data['close'] > kline_data['open']) &  # 第三天上涨
                                (kline_data['close'] > (kline_data['prev2_open'] + kline_data['prev2_close']) / 2)]  # 收复部分跌幅
        self._update_pattern_count(stats, "启明星", len(morning_star))
        
        # 黄昏星
        evening_star = kline_data[(kline_data['prev2_close'] > kline_data['prev2_open']) &  # 第一天上涨
                                (abs(kline_data['prev_close'] - kline_data['prev_open']) < 
                                 abs(kline_data['prev2_close'] - kline_data['prev2_open']) * 0.3) &  # 第二天十字星
                                (kline_data['close'] < kline_data['open']) &  # 第三天下跌
                                (kline_data['close'] < (kline_data['prev2_open'] + kline_data['prev2_close']) / 2)]  # 回撤部分涨幅
        self._update_pattern_count(stats, "黄昏星", len(evening_star))
        
        # 三只乌鸦
        three_black_crows = []
        for i in range(3, len(kline_data)):
            if (kline_data.iloc[i-3]['close'] > kline_data.iloc[i-3]['open'] and
                kline_data.iloc[i-2]['close'] < kline_data.iloc[i-2]['open'] and
                kline_data.iloc[i-1]['close'] < kline_data.iloc[i-1]['open'] and
                kline_data.iloc[i]['close'] < kline_data.iloc[i]['open'] and
                kline_data.iloc[i-1]['close'] < kline_data.iloc[i-2]['close'] and
                kline_data.iloc[i]['close'] < kline_data.iloc[i-1]['close']):
                three_black_crows.append(i)
        self._update_pattern_count(stats, "三只乌鸦", len(three_black_crows))
        
        # 三白兵
        three_white_soldiers = []
        for i in range(3, len(kline_data)):
            if (kline_data.iloc[i-3]['close'] < kline_data.iloc[i-3]['open'] and
                kline_data.iloc[i-2]['close'] > kline_data.iloc[i-2]['open'] and
                kline_data.iloc[i-1]['close'] > kline_data.iloc[i-1]['open'] and
                kline_data.iloc[i]['close'] > kline_data.iloc[i]['open'] and
                kline_data.iloc[i-1]['close'] > kline_data.iloc[i-2]['close'] and
                kline_data.iloc[i]['close'] > kline_data.iloc[i-1]['close']):
                three_white_soldiers.append(i)
        self._update_pattern_count(stats, "三白兵", len(three_white_soldiers))
    
    def _analyze_complex_patterns(self, kline_data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析复杂形态"""
        # 移动平均线
        kline_data['ma5'] = kline_data['close'].rolling(5).mean()
        kline_data['ma10'] = kline_data['close'].rolling(10).mean()
        kline_data['ma20'] = kline_data['close'].rolling(20).mean()
        
        # 双底形态检测
        double_bottom = self._detect_double_bottom(kline_data)
        self._update_pattern_count(stats, "双底", len(double_bottom))
        
        # 双顶形态检测
        double_top = self._detect_double_top(kline_data)
        self._update_pattern_count(stats, "双顶", len(double_top))
        
        # 头肩底形态检测
        head_and_shoulders_bottom = self._detect_head_and_shoulders_bottom(kline_data)
        self._update_pattern_count(stats, "头肩底", len(head_and_shoulders_bottom))
        
        # 头肩顶形态检测
        head_and_shoulders_top = self._detect_head_and_shoulders_top(kline_data)
        self._update_pattern_count(stats, "头肩顶", len(head_and_shoulders_top))
        
        # 三角形整理形态
        triangle = self._detect_triangle(kline_data)
        self._update_pattern_count(stats, "三角形整理", len(triangle))
        
        # 旗形整理
        flag = self._detect_flag(kline_data)
        self._update_pattern_count(stats, "旗形整理", len(flag))
        
        # 杯柄形态
        cup_and_handle = self._detect_cup_and_handle(kline_data)
        self._update_pattern_count(stats, "杯柄形态", len(cup_and_handle))
        
    def _detect_double_bottom(self, data: pd.DataFrame) -> List[int]:
        """检测双底形态"""
        # 简化检测逻辑
        result = []
        window = 20
        
        for i in range(window, len(data) - window):
            # 获取窗口数据
            window_data = data.iloc[i-window:i+window]
            
            # 找到窗口内的最低点
            min_indices = window_data.loc[window_data['low'] == window_data['low'].min()].index
            
            if len(min_indices) < 2:
                continue
                
            # 检查两个最低点之间的距离和价格差异
            min_idx1 = min_indices[0]
            min_idx2 = min_indices[-1]
            
            if min_idx2 - min_idx1 < 10:
                continue
                
            min_price1 = data.loc[min_idx1, 'low']
            min_price2 = data.loc[min_idx2, 'low']
            
            # 底部高度差异不大
            if abs(min_price2 - min_price1) / min_price1 > 0.05:
                continue
                
            # 检查中间是否有明显的反弹
            mid_idx = (min_idx1 + min_idx2) // 2
            mid_price = data.loc[mid_idx:mid_idx+5, 'high'].max()
            
            if (mid_price - min_price1) / min_price1 < 0.03:
                continue
                
            result.append(i)
            
        return result
    
    def _detect_double_top(self, data: pd.DataFrame) -> List[int]:
        """检测双顶形态"""
        # 简化检测逻辑
        result = []
        window = 20
        
        for i in range(window, len(data) - window):
            # 获取窗口数据
            window_data = data.iloc[i-window:i+window]
            
            # 找到窗口内的最高点
            max_indices = window_data.loc[window_data['high'] == window_data['high'].max()].index
            
            if len(max_indices) < 2:
                continue
                
            # 检查两个最高点之间的距离和价格差异
            max_idx1 = max_indices[0]
            max_idx2 = max_indices[-1]
            
            if max_idx2 - max_idx1 < 10:
                continue
                
            max_price1 = data.loc[max_idx1, 'high']
            max_price2 = data.loc[max_idx2, 'high']
            
            # 顶部高度差异不大
            if abs(max_price2 - max_price1) / max_price1 > 0.05:
                continue
                
            # 检查中间是否有明显的回调
            mid_idx = (max_idx1 + max_idx2) // 2
            mid_price = data.loc[mid_idx:mid_idx+5, 'low'].min()
            
            if (max_price1 - mid_price) / max_price1 < 0.03:
                continue
                
            result.append(i)
            
        return result
    
    def _detect_head_and_shoulders_bottom(self, data: pd.DataFrame) -> List[int]:
        """检测头肩底形态"""
        # 简化检测逻辑
        result = []
        window = 30
        
        for i in range(window, len(data) - window):
            # 待实现
            pass
            
        return result
    
    def _detect_head_and_shoulders_top(self, data: pd.DataFrame) -> List[int]:
        """检测头肩顶形态"""
        # 简化检测逻辑
        result = []
        window = 30
        
        for i in range(window, len(data) - window):
            # 待实现
            pass
            
        return result
    
    def _detect_triangle(self, data: pd.DataFrame) -> List[int]:
        """检测三角形整理形态"""
        # 简化检测逻辑
        result = []
        window = 20
        
        for i in range(window, len(data) - window):
            # 待实现
            pass
            
        return result
    
    def _detect_flag(self, data: pd.DataFrame) -> List[int]:
        """检测旗形整理形态"""
        # 简化检测逻辑
        result = []
        window = 15
        
        for i in range(window, len(data) - window):
            # 待实现
            pass
            
        return result
    
    def _detect_cup_and_handle(self, data: pd.DataFrame) -> List[int]:
        """检测杯柄形态"""
        # 简化检测逻辑
        result = []
        window = 40
        
        for i in range(window, len(data) - window):
            # 待实现
            pass
            
        return result
    
    def _update_pattern_count(self, stats: Dict[str, int], pattern_name: str, count: int) -> None:
        """更新形态计数"""
        if pattern_name in stats:
            stats[pattern_name] += count
        else:
            stats[pattern_name] = count
    
    def _calculate_pattern_frequency(self, pattern_stats: Dict[str, Dict[str, int]], 
                                    stock_count: int) -> Dict[str, Any]:
        """计算形态出现频率"""
        result = {}
        
        for category, patterns in pattern_stats.items():
            category_result = {}
            
            for pattern, count in patterns.items():
                frequency = count / stock_count if stock_count > 0 else 0
                category_result[pattern] = {
                    "count": count,
                    "frequency": frequency
                }
            
            # 按频率排序
            sorted_patterns = {k: v for k, v in sorted(
                category_result.items(), 
                key=lambda item: item[1]["frequency"], 
                reverse=True
            )}
            
            result[category] = sorted_patterns
        
        return result
    
    def _identify_common_pattern_features(self, pattern_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别共性形态特征"""
        common_features = []
        
        # 设置频率阈值
        threshold = 0.3
        
        # 遍历各类形态
        for category, patterns in pattern_stats.items():
            for pattern, stats in patterns.items():
                if stats["frequency"] >= threshold:
                    common_features.append({
                        "category": category,
                        "pattern": pattern,
                        "frequency": stats["frequency"],
                        "count": stats["count"]
                    })
        
        # 按频率排序
        common_features.sort(key=lambda x: x["frequency"], reverse=True)
        
        return common_features

    def analyze_trend_features(self, stock_codes: List[str], 
                            start_date: str, end_date: str,
                            period: str = 'DAILY') -> Dict[str, Any]:
        """
        分析买点趋势特征
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            period: K线周期
            
        Returns:
            Dict: 趋势特征分析结果
        """
        logger.info(f"开始分析买点趋势特征: {len(stock_codes)}只股票, 周期={period}")
        
        try:
            # 趋势特征统计
            trend_stats = {
                "ma_trend": {},     # 均线趋势
                "price_trend": {},  # 价格趋势
                "trend_strength": {} # 趋势强度
            }
            
            # 获取K线数据
            for stock_code in stock_codes:
                # 获取股票K线数据，前推30天以计算均线
                start_date_extended = self._get_extended_date(start_date, days=30)
                
                sql = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM stock_{period.lower()}
                WHERE stock_code = '{stock_code}'
                  AND date >= '{start_date_extended}' AND date <= '{end_date}'
                ORDER BY date
                """
                kline_data = self.ch_db.query_df(sql)
                
                if kline_data.empty:
                    logger.warning(f"未找到股票 {stock_code} 的K线数据")
                    continue
                
                # 计算均线
                kline_data['ma5'] = kline_data['close'].rolling(5).mean()
                kline_data['ma10'] = kline_data['close'].rolling(10).mean()
                kline_data['ma20'] = kline_data['close'].rolling(20).mean()
                kline_data['ma30'] = kline_data['close'].rolling(30).mean()
                
                # 过滤掉扩展的日期数据
                valid_data = kline_data[kline_data['date'] >= start_date]
                
                if valid_data.empty:
                    continue
                
                # 分析均线趋势
                self._analyze_ma_trend(valid_data, trend_stats["ma_trend"])
                
                # 分析价格趋势
                self._analyze_price_trend(valid_data, trend_stats["price_trend"])
                
                # 分析趋势强度
                self._analyze_trend_strength(valid_data, trend_stats["trend_strength"])
            
            # 计算趋势特征频率
            result = {
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "stock_count": len(stock_codes),
                "trend_stats": self._calculate_trend_frequency(trend_stats, len(stock_codes))
            }
            
            # 识别高频特征
            result["common_features"] = self._identify_common_trend_features(result["trend_stats"])
            
            # 保存到分析结果
            self.analysis_results["trend_analysis"] = result
            
            logger.info(f"买点趋势特征分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析买点趋势特征时出错: {e}")
            return {}
    
    def _analyze_ma_trend(self, data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析均线趋势"""
        # MA5上穿MA10
        ma5_cross_above_ma10 = data[(data['ma5'] > data['ma10']) & (data['ma5'].shift(1) <= data['ma10'].shift(1))]
        self._update_pattern_count(stats, "MA5上穿MA10", len(ma5_cross_above_ma10))
        
        # MA5下穿MA10
        ma5_cross_below_ma10 = data[(data['ma5'] < data['ma10']) & (data['ma5'].shift(1) >= data['ma10'].shift(1))]
        self._update_pattern_count(stats, "MA5下穿MA10", len(ma5_cross_below_ma10))
        
        # MA10上穿MA20
        ma10_cross_above_ma20 = data[(data['ma10'] > data['ma20']) & (data['ma10'].shift(1) <= data['ma20'].shift(1))]
        self._update_pattern_count(stats, "MA10上穿MA20", len(ma10_cross_above_ma20))
        
        # MA10下穿MA20
        ma10_cross_below_ma20 = data[(data['ma10'] < data['ma20']) & (data['ma10'].shift(1) >= data['ma20'].shift(1))]
        self._update_pattern_count(stats, "MA10下穿MA20", len(ma10_cross_below_ma20))
        
        # 均线多头排列（MA5 > MA10 > MA20 > MA30）
        bullish_alignment = data[(data['ma5'] > data['ma10']) & (data['ma10'] > data['ma20']) & (data['ma20'] > data['ma30'])]
        self._update_pattern_count(stats, "均线多头排列", len(bullish_alignment))
        
        # 均线空头排列（MA5 < MA10 < MA20 < MA30）
        bearish_alignment = data[(data['ma5'] < data['ma10']) & (data['ma10'] < data['ma20']) & (data['ma20'] < data['ma30'])]
        self._update_pattern_count(stats, "均线空头排列", len(bearish_alignment))
        
        # 均线交叉密集
        ma_crosses = 0
        for i in range(1, len(data)):
            crosses = 0
            if (data.iloc[i]['ma5'] > data.iloc[i]['ma10'] and data.iloc[i-1]['ma5'] <= data.iloc[i-1]['ma10']) or \
               (data.iloc[i]['ma5'] < data.iloc[i]['ma10'] and data.iloc[i-1]['ma5'] >= data.iloc[i-1]['ma10']):
                crosses += 1
            if (data.iloc[i]['ma10'] > data.iloc[i]['ma20'] and data.iloc[i-1]['ma10'] <= data.iloc[i-1]['ma20']) or \
               (data.iloc[i]['ma10'] < data.iloc[i]['ma20'] and data.iloc[i-1]['ma10'] >= data.iloc[i-1]['ma20']):
                crosses += 1
            if crosses >= 2:
                ma_crosses += 1
        self._update_pattern_count(stats, "均线交叉密集", ma_crosses)
        
        # 均线收敛
        ma_converge = data[((data['ma5'] - data['ma30']).abs() < (data['ma5'].shift(5) - data['ma30'].shift(5)).abs() * 0.8)]
        self._update_pattern_count(stats, "均线收敛", len(ma_converge))
        
        # 均线发散
        ma_diverge = data[((data['ma5'] - data['ma30']).abs() > (data['ma5'].shift(5) - data['ma30'].shift(5)).abs() * 1.2)]
        self._update_pattern_count(stats, "均线发散", len(ma_diverge))
    
    def _analyze_price_trend(self, data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析价格趋势"""
        # 计算价格变动百分比
        data['price_change_pct'] = data['close'].pct_change()
        
        # 计算n日涨跌幅
        data['change_5d'] = data['close'] / data['close'].shift(5) - 1 if len(data) >= 5 else np.nan
        data['change_10d'] = data['close'] / data['close'].shift(10) - 1 if len(data) >= 10 else np.nan
        data['change_20d'] = data['close'] / data['close'].shift(20) - 1 if len(data) >= 20 else np.nan
        
        # 连续上涨
        consecutive_up = 0
        for i in range(1, len(data)):
            if data.iloc[i]['close'] > data.iloc[i-1]['close']:
                consecutive_up += 1
                if consecutive_up >= 3:  # 连续3日上涨
                    self._update_pattern_count(stats, "连续3日上涨", 1)
                if consecutive_up >= 5:  # 连续5日上涨
                    self._update_pattern_count(stats, "连续5日上涨", 1)
            else:
                consecutive_up = 0
                
        # 连续下跌
        consecutive_down = 0
        for i in range(1, len(data)):
            if data.iloc[i]['close'] < data.iloc[i-1]['close']:
                consecutive_down += 1
                if consecutive_down >= 3:  # 连续3日下跌
                    self._update_pattern_count(stats, "连续3日下跌", 1)
                if consecutive_down >= 5:  # 连续5日下跌
                    self._update_pattern_count(stats, "连续5日下跌", 1)
            else:
                consecutive_down = 0
        
        # 大涨
        big_up = data[data['price_change_pct'] > 0.05]  # 单日涨幅超过5%
        self._update_pattern_count(stats, "单日大涨(>5%)", len(big_up))
        
        # 大跌
        big_down = data[data['price_change_pct'] < -0.05]  # 单日跌幅超过5%
        self._update_pattern_count(stats, "单日大跌(>5%)", len(big_down))
        
        # 高位回调
        high_pullback = []
        window = 10
        for i in range(window, len(data)):
            # 前window天的最高价
            max_high = data.iloc[i-window:i]['high'].max()
            max_idx = data.iloc[i-window:i]['high'].idxmax()
            
            # 如果当前收盘价比最高价回调超过5%，且最高价出现在3天前以上
            if (max_high - data.iloc[i]['close']) / max_high > 0.05 and i - max_idx > 3:
                high_pullback.append(i)
        self._update_pattern_count(stats, "高位回调", len(high_pullback))
        
        # 低位反弹
        low_rebound = []
        for i in range(window, len(data)):
            # 前window天的最低价
            min_low = data.iloc[i-window:i]['low'].min()
            min_idx = data.iloc[i-window:i]['low'].idxmin()
            
            # 如果当前收盘价比最低价反弹超过5%，且最低价出现在3天前以上
            if (data.iloc[i]['close'] - min_low) / min_low > 0.05 and i - min_idx > 3:
                low_rebound.append(i)
        self._update_pattern_count(stats, "低位反弹", len(low_rebound))
        
        # 突破前高
        break_high = []
        for i in range(window, len(data)):
            prev_high = data.iloc[i-window:i-1]['high'].max()
            if data.iloc[i]['close'] > prev_high:
                break_high.append(i)
        self._update_pattern_count(stats, "突破前高", len(break_high))
        
        # 跌破前低
        break_low = []
        for i in range(window, len(data)):
            prev_low = data.iloc[i-window:i-1]['low'].min()
            if data.iloc[i]['close'] < prev_low:
                break_low.append(i)
        self._update_pattern_count(stats, "跌破前低", len(break_low))
    
    def _analyze_trend_strength(self, data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析趋势强度"""
        # 计算ADX (Average Directional Index)
        try:
            import talib
            data['adx'] = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            
            # 强趋势 (ADX > 25)
            strong_trend = data[data['adx'] > 25]
            self._update_pattern_count(stats, "强趋势(ADX>25)", len(strong_trend))
            
            # 极强趋势 (ADX > 50)
            very_strong_trend = data[data['adx'] > 50]
            self._update_pattern_count(stats, "极强趋势(ADX>50)", len(very_strong_trend))
            
            # 无趋势 (ADX < 20)
            no_trend = data[data['adx'] < 20]
            self._update_pattern_count(stats, "无趋势(ADX<20)", len(no_trend))
            
        except ImportError:
            # 如果没有talib，使用简化的趋势判断
            logger.warning("未安装talib库，使用简化的趋势判断")
            
            # 计算简单的趋势强度：连续上涨或下跌的幅度
            up_strength = []
            down_strength = []
            
            for i in range(5, len(data)):
                # 过去5天的涨跌幅
                change_5d = data.iloc[i]['close'] / data.iloc[i-5]['close'] - 1
                
                if change_5d > 0.1:  # 5天涨幅超过10%
                    up_strength.append(i)
                elif change_5d < -0.1:  # 5天跌幅超过10%
                    down_strength.append(i)
            
            self._update_pattern_count(stats, "强上涨趋势(5日涨幅>10%)", len(up_strength))
            self._update_pattern_count(stats, "强下跌趋势(5日跌幅>10%)", len(down_strength))
    
    def _get_extended_date(self, date_str: str, days: int) -> str:
        """获取向前推days天的日期"""
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        extended_date = date_obj - timedelta(days=days)
        return extended_date.strftime("%Y%m%d")
    
    def _calculate_trend_frequency(self, trend_stats: Dict[str, Dict[str, int]], 
                                  stock_count: int) -> Dict[str, Any]:
        """计算趋势特征频率"""
        return self._calculate_pattern_frequency(trend_stats, stock_count)
    
    def _identify_common_trend_features(self, trend_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别共性趋势特征"""
        return self._identify_common_pattern_features(trend_stats)

    def analyze_volume_features(self, stock_codes: List[str], 
                              start_date: str, end_date: str,
                              period: str = 'DAILY') -> Dict[str, Any]:
        """
        分析买点量能特征
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            period: K线周期
            
        Returns:
            Dict: 量能特征分析结果
        """
        logger.info(f"开始分析买点量能特征: {len(stock_codes)}只股票, 周期={period}")
        
        try:
            # 量能特征统计
            volume_stats = {
                "volume_change": {},   # 成交量变化
                "volume_pattern": {},  # 成交量形态
                "price_volume": {}     # 量价关系
            }
            
            # 获取K线数据
            for stock_code in stock_codes:
                # 获取股票K线数据，前推20天以计算均量
                start_date_extended = self._get_extended_date(start_date, days=20)
                
                sql = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM stock_{period.lower()}
                WHERE stock_code = '{stock_code}'
                  AND date >= '{start_date_extended}' AND date <= '{end_date}'
                ORDER BY date
                """
                kline_data = self.ch_db.query_df(sql)
                
                if kline_data.empty:
                    logger.warning(f"未找到股票 {stock_code} 的K线数据")
                    continue
                
                # 计算均量
                kline_data['volume_ma5'] = kline_data['volume'].rolling(5).mean()
                kline_data['volume_ma10'] = kline_data['volume'].rolling(10).mean()
                kline_data['volume_ma20'] = kline_data['volume'].rolling(20).mean()
                
                # 过滤掉扩展的日期数据
                valid_data = kline_data[kline_data['date'] >= start_date]
                
                if valid_data.empty:
                    continue
                
                # 分析成交量变化
                self._analyze_volume_change(valid_data, volume_stats["volume_change"])
                
                # 分析成交量形态
                self._analyze_volume_pattern(valid_data, volume_stats["volume_pattern"])
                
                # 分析量价关系
                self._analyze_price_volume_relation(valid_data, volume_stats["price_volume"])
            
            # 计算量能特征频率
            result = {
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "stock_count": len(stock_codes),
                "volume_stats": self._calculate_volume_frequency(volume_stats, len(stock_codes))
            }
            
            # 识别高频特征
            result["common_features"] = self._identify_common_volume_features(result["volume_stats"])
            
            # 保存到分析结果
            self.analysis_results["volume_analysis"] = result
            
            logger.info(f"买点量能特征分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析买点量能特征时出错: {e}")
            return {}
    
    def _analyze_volume_change(self, data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析成交量变化"""
        # 计算量比（当日成交量/5日均量）
        data['volume_ratio'] = data['volume'] / data['volume_ma5']
        
        # 放量（成交量大于5日均量的2倍）
        volume_surge = data[data['volume_ratio'] > 2.0]
        self._update_pattern_count(stats, "放量(>2倍均量)", len(volume_surge))
        
        # 大幅放量（成交量大于5日均量的3倍）
        large_volume_surge = data[data['volume_ratio'] > 3.0]
        self._update_pattern_count(stats, "大幅放量(>3倍均量)", len(large_volume_surge))
        
        # 极度放量（成交量大于5日均量的5倍）
        extreme_volume_surge = data[data['volume_ratio'] > 5.0]
        self._update_pattern_count(stats, "极度放量(>5倍均量)", len(extreme_volume_surge))
        
        # 缩量（成交量小于5日均量的0.5倍）
        volume_shrink = data[data['volume_ratio'] < 0.5]
        self._update_pattern_count(stats, "缩量(<0.5倍均量)", len(volume_shrink))
        
        # 极度缩量（成交量小于5日均量的0.3倍）
        extreme_volume_shrink = data[data['volume_ratio'] < 0.3]
        self._update_pattern_count(stats, "极度缩量(<0.3倍均量)", len(extreme_volume_shrink))
        
        # 连续放量
        consecutive_surge = 0
        for i in range(1, len(data)):
            if data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 1.5:
                consecutive_surge += 1
                if consecutive_surge >= 3:  # 连续3日放量
                    self._update_pattern_count(stats, "连续3日放量", 1)
            else:
                consecutive_surge = 0
        
        # 连续缩量
        consecutive_shrink = 0
        for i in range(1, len(data)):
            if data.iloc[i]['volume'] < data.iloc[i]['volume_ma5'] * 0.8:
                consecutive_shrink += 1
                if consecutive_shrink >= 3:  # 连续3日缩量
                    self._update_pattern_count(stats, "连续3日缩量", 1)
            else:
                consecutive_shrink = 0
        
        # 量能突变（今日成交量比昨日成交量增加200%以上）
        volume_change = data.copy()
        volume_change['volume_change_ratio'] = volume_change['volume'] / volume_change['volume'].shift(1)
        volume_spike = volume_change[volume_change['volume_change_ratio'] > 3.0]
        self._update_pattern_count(stats, "量能突变(日环比>3倍)", len(volume_spike))
    
    def _analyze_volume_pattern(self, data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析成交量形态"""
        # 梯量上升（连续3日以上量能逐步放大）
        volume_step_up = 0
        for i in range(3, len(data)):
            if (data.iloc[i]['volume'] > data.iloc[i-1]['volume'] > 
                data.iloc[i-2]['volume'] > data.iloc[i-3]['volume']):
                volume_step_up += 1
        self._update_pattern_count(stats, "梯量上升", volume_step_up)
        
        # 梯量下降（连续3日以上量能逐步萎缩）
        volume_step_down = 0
        for i in range(3, len(data)):
            if (data.iloc[i]['volume'] < data.iloc[i-1]['volume'] < 
                data.iloc[i-2]['volume'] < data.iloc[i-3]['volume']):
                volume_step_down += 1
        self._update_pattern_count(stats, "梯量下降", volume_step_down)
        
        # 量能分布（单日成交量在总成交量中的占比）
        if len(data) >= 5:
            for i in range(5, len(data)):
                # 计算5日内成交量总和
                total_volume = data.iloc[i-5:i]['volume'].sum()
                # 今日成交量占比
                today_ratio = data.iloc[i-1]['volume'] / total_volume if total_volume > 0 else 0
                
                # 单日占比超过40%，说明成交高度集中
                if today_ratio > 0.4:
                    self._update_pattern_count(stats, "量能高度集中(单日>40%)", 1)
        
        # 先量后价（先出现放量，后出现价格变动）
        volume_lead_price = 0
        for i in range(3, len(data)):
            # 前天放量（大于5日均量的1.5倍）但价格变动小
            if (data.iloc[i-2]['volume'] > data.iloc[i-2]['volume_ma5'] * 1.5 and
                abs(data.iloc[i-2]['close'] / data.iloc[i-3]['close'] - 1) < 0.02 and
                # 今天价格变动大
                abs(data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1) > 0.03):
                volume_lead_price += 1
        self._update_pattern_count(stats, "先量后价", volume_lead_price)
        
        # 先价后量（先出现价格变动，后出现放量）
        price_lead_volume = 0
        for i in range(3, len(data)):
            # 前天价格变动大但量能一般
            if (abs(data.iloc[i-2]['close'] / data.iloc[i-3]['close'] - 1) > 0.03 and
                data.iloc[i-2]['volume'] < data.iloc[i-2]['volume_ma5'] * 1.5 and
                # 今天放量
                data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 1.5):
                price_lead_volume += 1
        self._update_pattern_count(stats, "先价后量", price_lead_volume)
    
    def _analyze_price_volume_relation(self, data: pd.DataFrame, stats: Dict[str, int]) -> None:
        """分析量价关系"""
        # 量增价增（同向）
        volume_price_up = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] > data.iloc[i-1]['volume'] * 1.2 and
                data.iloc[i]['close'] > data.iloc[i-1]['close']):
                volume_price_up += 1
        self._update_pattern_count(stats, "量增价增", volume_price_up)
        
        # 量增价减（背离）
        volume_up_price_down = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] > data.iloc[i-1]['volume'] * 1.2 and
                data.iloc[i]['close'] < data.iloc[i-1]['close']):
                volume_up_price_down += 1
        self._update_pattern_count(stats, "量增价减", volume_up_price_down)
        
        # 量减价增（背离）
        volume_down_price_up = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] < data.iloc[i-1]['volume'] * 0.8 and
                data.iloc[i]['close'] > data.iloc[i-1]['close']):
                volume_down_price_up += 1
        self._update_pattern_count(stats, "量减价增", volume_down_price_up)
        
        # 量减价减（同向）
        volume_price_down = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['volume'] < data.iloc[i-1]['volume'] * 0.8 and
                data.iloc[i]['close'] < data.iloc[i-1]['close']):
                volume_price_down += 1
        self._update_pattern_count(stats, "量减价减", volume_price_down)
        
        # 大阳大量（单日上涨超过3%且放量超过1.5倍）
        big_up_big_volume = 0
        for i in range(1, len(data)):
            if (data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1 > 0.03 and
                data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 1.5):
                big_up_big_volume += 1
        self._update_pattern_count(stats, "大阳大量", big_up_big_volume)
        
        # 大阴大量（单日下跌超过3%且放量超过1.5倍）
        big_down_big_volume = 0
        for i in range(1, len(data)):
            if (data.iloc[i-1]['close'] / data.iloc[i]['close'] - 1 > 0.03 and
                data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 1.5):
                big_down_big_volume += 1
        self._update_pattern_count(stats, "大阴大量", big_down_big_volume)
        
        # 小阳大量（单日上涨小于2%但放量超过2倍）
        small_up_big_volume = 0
        for i in range(1, len(data)):
            if (0 < data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1 < 0.02 and
                data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 2.0):
                small_up_big_volume += 1
        self._update_pattern_count(stats, "小阳大量", small_up_big_volume)
        
        # 小阴大量（单日下跌小于2%但放量超过2倍）
        small_down_big_volume = 0
        for i in range(1, len(data)):
            if (0 < data.iloc[i-1]['close'] / data.iloc[i]['close'] - 1 < 0.02 and
                data.iloc[i]['volume'] > data.iloc[i]['volume_ma5'] * 2.0):
                small_down_big_volume += 1
        self._update_pattern_count(stats, "小阴大量", small_down_big_volume)
    
    def _calculate_volume_frequency(self, volume_stats: Dict[str, Dict[str, int]], 
                                  stock_count: int) -> Dict[str, Any]:
        """计算量能特征频率"""
        return self._calculate_pattern_frequency(volume_stats, stock_count)
    
    def _identify_common_volume_features(self, volume_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别共性量能特征"""
        return self._identify_common_pattern_features(volume_stats)

    def analyze_multi_period(self, stock_codes: List[str], date: str, 
                          periods: List[str] = ['DAILY', 'WEEKLY']) -> Dict[str, Any]:
        """
        多周期分析
        
        Args:
            stock_codes: 股票代码列表
            date: 分析日期，格式YYYYMMDD
            periods: 周期列表
            
        Returns:
            Dict: 多周期分析结果
        """
        logger.info(f"开始多周期分析: {len(stock_codes)}只股票, 周期={periods}")
        
        try:
            # 获取每个周期的特征分析
            period_results = {}
            
            for period in periods:
                # 获取对应周期的数据起止日期
                start_date, end_date = self._get_period_date_range(date, period)
                
                # 获取形态特征
                pattern_features = self.analyze_pattern_features(stock_codes, start_date, end_date, period)
                
                # 获取趋势特征
                trend_features = self.analyze_trend_features(stock_codes, start_date, end_date, period)
                
                # 获取量能特征
                volume_features = self.analyze_volume_features(stock_codes, start_date, end_date, period)
                
                # 合并结果
                period_results[period] = {
                    "pattern_features": pattern_features.get("common_features", []),
                    "trend_features": trend_features.get("common_features", []),
                    "volume_features": volume_features.get("common_features", [])
                }
            
            # 寻找多周期共性特征
            common_features = self._find_multi_period_common_features(period_results)
            
            # 构建结果
            result = {
                "date": date,
                "periods": periods,
                "stock_count": len(stock_codes),
                "period_results": period_results,
                "common_features": common_features
            }
            
            logger.info(f"多周期分析完成")
            return result
            
        except Exception as e:
            logger.error(f"多周期分析时出错: {e}")
            return {}
    
    def analyze_indicator_signals(self, stock_codes: List[str], 
                                start_date: str, end_date: str,
                                indicators: List[Dict[str, Any]],
                                period: str = 'DAILY') -> Dict[str, Any]:
        """
        分析指标信号
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            indicators: 指标列表，每个指标为Dict格式
                {
                    "name": "指标名称",
                    "id": "指标ID",
                    "parameters": {"参数1": 值1, ...}
                }
            period: K线周期
            
        Returns:
            Dict: 指标信号分析结果
        """
        logger.info(f"开始分析指标信号: {len(stock_codes)}只股票, {len(indicators)}个指标, 周期={period}")
        
        try:
            # 指标信号统计
            indicator_stats = {}
            
            # 对每个股票计算指标
            for stock_code in stock_codes:
                # 获取股票K线数据
                start_date_extended = self._get_extended_date(start_date, days=50)  # 预留足够数据计算指标
                
                sql = f"""
                SELECT date, open, high, low, close, volume, amount
                FROM stock_{period.lower()}
                WHERE stock_code = '{stock_code}'
                  AND date >= '{start_date_extended}' AND date <= '{end_date}'
                ORDER BY date
                """
                kline_data = self.ch_db.query_df(sql)
                
                if kline_data.empty:
                    logger.warning(f"未找到股票 {stock_code} 的K线数据")
                    continue
                
                # 过滤掉扩展的日期数据
                valid_data = kline_data[kline_data['date'] >= start_date]
                
                if valid_data.empty:
                    continue
                
                # 对每个指标计算信号
                for indicator_config in indicators:
                    indicator_id = indicator_config["id"]
                    indicator_name = indicator_config["name"]
                    parameters = indicator_config.get("parameters", {})
                    
                    # 统计字典初始化
                    if indicator_id not in indicator_stats:
                        indicator_stats[indicator_id] = {
                            "name": indicator_name,
                            "signals": {}
                        }
                    
                    try:
                        # 创建指标实例
                        indicator = self.indicator_factory.create(indicator_id, **parameters)
                        
                        # 计算指标值
                        indicator_result = indicator.calculate(kline_data)
                        
                        # 生成信号
                        signals = indicator.generate_signals(indicator_result)
                        
                        # 过滤有效期内的信号
                        valid_signals = signals[signals.index.isin(valid_data.index)]
                        
                        # 统计信号出现次数
                        for column in valid_signals.columns:
                            # 只统计布尔型信号列
                            if valid_signals[column].dtype == bool:
                                signal_count = valid_signals[column].sum()
                                
                                if column not in indicator_stats[indicator_id]["signals"]:
                                    indicator_stats[indicator_id]["signals"][column] = 0
                                
                                indicator_stats[indicator_id]["signals"][column] += signal_count
                                
                    except Exception as e:
                        logger.error(f"计算指标 {indicator_id} 时出错: {e}")
            
            # 计算信号频率
            result = {
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "stock_count": len(stock_codes),
                "indicator_stats": self._calculate_signal_frequency(indicator_stats, len(stock_codes))
            }
            
            # 识别高频信号
            result["common_signals"] = self._identify_common_signals(result["indicator_stats"])
            
            # 保存到分析结果
            self.analysis_results["indicator_analysis"] = result
            
            logger.info(f"指标信号分析完成")
            return result
            
        except Exception as e:
            logger.error(f"分析指标信号时出错: {e}")
            return {}
    
    def _get_period_date_range(self, date: str, period: str) -> Tuple[str, str]:
        """根据周期获取日期范围"""
        date_obj = datetime.strptime(date, "%Y%m%d")
        
        if period == 'DAILY':
            # 日线取最近30天
            start_date = (date_obj - timedelta(days=30)).strftime("%Y%m%d")
            end_date = date
        elif period == 'WEEKLY':
            # 周线取最近12周
            start_date = (date_obj - timedelta(days=12*7)).strftime("%Y%m%d")
            end_date = date
        elif period == 'MONTHLY':
            # 月线取最近6个月
            start_date = (date_obj - timedelta(days=6*30)).strftime("%Y%m%d")
            end_date = date
        else:
            # 默认取30天
            start_date = (date_obj - timedelta(days=30)).strftime("%Y%m%d")
            end_date = date
            
        return start_date, end_date
    
    def _find_multi_period_common_features(self, period_results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """寻找多周期共性特征"""
        # 将所有特征平铺并记录周期信息
        all_features = []
        
        for period, features in period_results.items():
            for feature_type, feature_list in features.items():
                for feature in feature_list:
                    all_features.append({
                        "period": period,
                        "type": feature_type,
                        "feature": feature["pattern"] if "pattern" in feature else feature.get("signal", ""),
                        "frequency": feature["frequency"],
                        "category": feature.get("category", ""),
                        "count": feature.get("count", 0)
                    })
        
        # 按特征名称分组
        feature_groups = {}
        for feature in all_features:
            key = f"{feature['type']}_{feature['feature']}"
            if key not in feature_groups:
                feature_groups[key] = []
            feature_groups[key].append(feature)
        
        # 寻找出现在多个周期的特征
        common_features = []
        for key, group in feature_groups.items():
            if len(group) > 1:  # 出现在多个周期
                periods = sorted(list(set([f["period"] for f in group])))
                avg_frequency = sum([f["frequency"] for f in group]) / len(group)
                
                common_features.append({
                    "feature": group[0]["feature"],
                    "type": group[0]["type"],
                    "category": group[0]["category"],
                    "periods": periods,
                    "avg_frequency": avg_frequency,
                    "details": group
                })
        
        # 按平均频率排序
        common_features.sort(key=lambda x: x["avg_frequency"], reverse=True)
        
        return common_features
    
    def _calculate_signal_frequency(self, indicator_stats: Dict[str, Dict[str, Any]], 
                                  stock_count: int) -> Dict[str, Any]:
        """计算信号频率"""
        result = {}
        
        for indicator_id, stats in indicator_stats.items():
            signals_result = {}
            
            for signal, count in stats["signals"].items():
                frequency = count / stock_count if stock_count > 0 else 0
                signals_result[signal] = {
                    "count": count,
                    "frequency": frequency
                }
            
            # 按频率排序
            sorted_signals = {k: v for k, v in sorted(
                signals_result.items(), 
                key=lambda item: item[1]["frequency"], 
                reverse=True
            )}
            
            result[indicator_id] = {
                "name": stats["name"],
                "signals": sorted_signals
            }
        
        return result
    
    def _identify_common_signals(self, indicator_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别共性信号特征"""
        common_signals = []
        
        # 设置频率阈值
        threshold = 0.3
        
        # 遍历各指标信号
        for indicator_id, stats in indicator_stats.items():
            for signal, signal_stats in stats["signals"].items():
                if signal_stats["frequency"] >= threshold:
                    common_signals.append({
                        "indicator": stats["name"],
                        "indicator_id": indicator_id,
                        "signal": signal,
                        "frequency": signal_stats["frequency"],
                        "count": signal_stats["count"]
                    })
        
        # 按频率排序
        common_signals.sort(key=lambda x: x["frequency"], reverse=True)
        
        return common_signals

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        获取买点分析综合报告
        
        Returns:
            Dict: 综合分析报告
        """
        logger.info("生成买点维度综合分析报告")
        
        try:
            # 整合各模块分析结果
            report = {
                "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pattern_analysis": {
                    "common_features": self.analysis_results.get("pattern_analysis", {}).get("common_features", [])
                },
                "trend_analysis": {
                    "common_features": self.analysis_results.get("trend_analysis", {}).get("common_features", [])
                },
                "volume_analysis": {
                    "common_features": self.analysis_results.get("volume_analysis", {}).get("common_features", [])
                },
                "indicator_analysis": {
                    "common_signals": self.analysis_results.get("indicator_analysis", {}).get("common_signals", [])
                }
            }
            
            # 构建买点关键特征
            key_features = []
            
            # 从各分析中提取高频特征
            for analysis_type in ["pattern_analysis", "trend_analysis", "volume_analysis"]:
                features = self.analysis_results.get(analysis_type, {}).get("common_features", [])
                for feature in features:
                    if isinstance(feature, dict) and "frequency" in feature and feature["frequency"] > 0.5:
                        key_features.append({
                            "type": analysis_type.replace("_analysis", ""),
                            "feature": feature.get("pattern", feature.get("signal", "")),
                            "frequency": feature["frequency"],
                            "count": feature.get("count", 0)
                        })
            
            # 添加指标信号
            signals = self.analysis_results.get("indicator_analysis", {}).get("common_signals", [])
            for signal in signals:
                if signal["frequency"] > 0.5:
                    key_features.append({
                        "type": "indicator",
                        "feature": f"{signal['indicator']}:{signal['signal']}",
                        "frequency": signal["frequency"],
                        "count": signal.get("count", 0)
                    })
            
            # 按频率排序
            key_features.sort(key=lambda x: x["frequency"], reverse=True)
            
            report["key_features"] = key_features
            
            # 买点描述生成
            description = self._generate_buypoint_description(key_features)
            report["description"] = description
            
            logger.info("买点维度综合分析报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成买点维度综合分析报告时出错: {e}")
            return {}
    
    def save_results(self, output_file: str, format_type: str = "json") -> None:
        """
        保存分析结果
        
        Args:
            output_file: 输出文件路径
            format_type: 输出格式类型，支持json和markdown
        """
        logger.info(f"保存买点分析结果到: {output_file}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format_type.lower() == "json":
                # 获取综合报告
                report = self.get_comprehensive_report()
                
                # 构建完整结果
                full_results = {
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "comprehensive_report": report,
                    "detailed_results": self.analysis_results
                }
                
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(full_results, f, indent=2, ensure_ascii=False)
                
            elif format_type.lower() == "markdown":
                # 获取综合报告
                report = self.get_comprehensive_report()
                
                # 构建Markdown内容
                markdown = "# 买点多维度分析报告\n\n"
                markdown += f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # 买点描述
                if "description" in report:
                    markdown += "## 买点特征描述\n\n"
                    markdown += f"{report['description']}\n\n"
                
                # 关键特征
                if "key_features" in report and report["key_features"]:
                    markdown += "## 关键特征\n\n"
                    markdown += "| 特征类型 | 特征 | 出现频率 | 出现次数 |\n"
                    markdown += "| -------- | ---- | -------- | -------- |\n"
                    
                    for feature in report["key_features"]:
                        markdown += f"| {feature['type']} | {feature['feature']} | "
                        markdown += f"{feature['frequency']*100:.2f}% | {feature['count']} |\n"
                    
                    markdown += "\n"
                
                # K线形态特征
                common_patterns = report.get("pattern_analysis", {}).get("common_features", [])
                if common_patterns:
                    markdown += "## K线形态特征\n\n"
                    markdown += "| 形态类别 | 形态 | 出现频率 | 出现次数 |\n"
                    markdown += "| -------- | ---- | -------- | -------- |\n"
                    
                    for pattern in common_patterns:
                        markdown += f"| {pattern.get('category', '-')} | {pattern.get('pattern', '-')} | "
                        markdown += f"{pattern.get('frequency', 0)*100:.2f}% | {pattern.get('count', 0)} |\n"
                    
                    markdown += "\n"
                
                # 趋势特征
                common_trends = report.get("trend_analysis", {}).get("common_features", [])
                if common_trends:
                    markdown += "## 趋势特征\n\n"
                    markdown += "| 趋势类别 | 趋势特征 | 出现频率 | 出现次数 |\n"
                    markdown += "| -------- | -------- | -------- | -------- |\n"
                    
                    for trend in common_trends:
                        markdown += f"| {trend.get('category', '-')} | {trend.get('pattern', '-')} | "
                        markdown += f"{trend.get('frequency', 0)*100:.2f}% | {trend.get('count', 0)} |\n"
                    
                    markdown += "\n"
                
                # 量能特征
                common_volumes = report.get("volume_analysis", {}).get("common_features", [])
                if common_volumes:
                    markdown += "## 量能特征\n\n"
                    markdown += "| 量能类别 | 量能特征 | 出现频率 | 出现次数 |\n"
                    markdown += "| -------- | -------- | -------- | -------- |\n"
                    
                    for volume in common_volumes:
                        markdown += f"| {volume.get('category', '-')} | {volume.get('pattern', '-')} | "
                        markdown += f"{volume.get('frequency', 0)*100:.2f}% | {volume.get('count', 0)} |\n"
                    
                    markdown += "\n"
                
                # 指标信号
                common_signals = report.get("indicator_analysis", {}).get("common_signals", [])
                if common_signals:
                    markdown += "## 指标信号\n\n"
                    markdown += "| 指标 | 信号 | 出现频率 | 出现次数 |\n"
                    markdown += "| ---- | ---- | -------- | -------- |\n"
                    
                    for signal in common_signals:
                        markdown += f"| {signal.get('indicator', '-')} | {signal.get('signal', '-')} | "
                        markdown += f"{signal.get('frequency', 0)*100:.2f}% | {signal.get('count', 0)} |\n"
                    
                    markdown += "\n"
                
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            
            else:
                logger.error(f"不支持的输出格式: {format_type}")
                return
                
            logger.info(f"买点分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存买点分析结果时出错: {e}")
    
    def export_to_excel(self, output_file: str) -> None:
        """
        导出分析结果到Excel
        
        Args:
            output_file: 输出Excel文件路径
        """
        logger.info(f"导出买点分析结果到Excel: {output_file}")
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 获取综合报告
            report = self.get_comprehensive_report()
            
            # 创建Excel写入器
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 关键特征
                if "key_features" in report and report["key_features"]:
                    key_features_df = pd.DataFrame(report["key_features"])
                    key_features_df.to_excel(writer, sheet_name="关键特征", index=False)
                
                # K线形态特征
                common_patterns = report.get("pattern_analysis", {}).get("common_features", [])
                if common_patterns:
                    patterns_df = pd.DataFrame(common_patterns)
                    patterns_df.to_excel(writer, sheet_name="K线形态特征", index=False)
                
                # 趋势特征
                common_trends = report.get("trend_analysis", {}).get("common_features", [])
                if common_trends:
                    trends_df = pd.DataFrame(common_trends)
                    trends_df.to_excel(writer, sheet_name="趋势特征", index=False)
                
                # 量能特征
                common_volumes = report.get("volume_analysis", {}).get("common_features", [])
                if common_volumes:
                    volumes_df = pd.DataFrame(common_volumes)
                    volumes_df.to_excel(writer, sheet_name="量能特征", index=False)
                
                # 指标信号
                common_signals = report.get("indicator_analysis", {}).get("common_signals", [])
                if common_signals:
                    signals_df = pd.DataFrame(common_signals)
                    signals_df.to_excel(writer, sheet_name="指标信号", index=False)
            
            logger.info(f"买点分析结果已导出到Excel: {output_file}")
            
        except Exception as e:
            logger.error(f"导出买点分析结果到Excel时出错: {e}")
    
    def _generate_buypoint_description(self, key_features: List[Dict[str, Any]]) -> str:
        """
        生成买点描述
        
        Args:
            key_features: 关键特征列表
            
        Returns:
            str: 买点描述文本
        """
        if not key_features:
            return "无法生成买点描述，关键特征不足。"
        
        # 按类型分组特征
        feature_by_type = {}
        for feature in key_features:
            feature_type = feature["type"]
            if feature_type not in feature_by_type:
                feature_by_type[feature_type] = []
            feature_by_type[feature_type].append(feature)
        
        description_parts = []
        
        # 形态描述
        if "pattern" in feature_by_type:
            pattern_features = feature_by_type["pattern"]
            if pattern_features:
                patterns = [f"{f['feature']}({f['frequency']*100:.0f}%)" for f in pattern_features[:3]]
                description_parts.append(f"K线形态呈现{', '.join(patterns)}")
        
        # 趋势描述
        if "trend" in feature_by_type:
            trend_features = feature_by_type["trend"]
            if trend_features:
                trends = [f"{f['feature']}({f['frequency']*100:.0f}%)" for f in trend_features[:3]]
                description_parts.append(f"趋势特征表现为{', '.join(trends)}")
        
        # 量能描述
        if "volume" in feature_by_type:
            volume_features = feature_by_type["volume"]
            if volume_features:
                volumes = [f"{f['feature']}({f['frequency']*100:.0f}%)" for f in volume_features[:3]]
                description_parts.append(f"量能表现为{', '.join(volumes)}")
        
        # 指标信号描述
        if "indicator" in feature_by_type:
            indicator_features = feature_by_type["indicator"]
            if indicator_features:
                indicators = [f"{f['feature']}({f['frequency']*100:.0f}%)" for f in indicator_features[:3]]
                description_parts.append(f"技术指标显示{', '.join(indicators)}")
        
        # 综合描述
        if description_parts:
            description = "该买点特征为：" + "；".join(description_parts) + "。"
            
            # 添加总结
            top_features = sorted(key_features, key=lambda x: x["frequency"], reverse=True)[:5]
            if top_features:
                top_feature_texts = [f"{f['feature']}({f['frequency']*100:.0f}%)" for f in top_features]
                description += f"\n\n综合来看，最显著的特征是：{', '.join(top_feature_texts)}。"
            
            return description
        else:
            return "无法生成买点描述，关键特征不足。" 