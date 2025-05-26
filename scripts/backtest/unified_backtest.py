#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import argparse
import datetime
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_backtest_result_dir, get_stock_result_file, get_strategies_dir
from db.db_manager import DBManager
from indicators.factory import IndicatorFactory

# 获取日志记录器
logger = get_logger(__name__)

# 添加JSON编码器处理numpy类型
class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None or obj == np.nan:
            return None
        return super(NumpyEncoder, self).default(obj)

class UnifiedBacktest:
    """
    统一回测系统
    
    结合了:
    1. comprehensive_backtest.py的多周期分析能力
    2. indicator_analysis.py的全面技术指标分析能力
    
    支持多周期、多股票、多买点分析，使用全部可用的技术指标
    """
    
    def __init__(self):
        """初始化统一回测系统"""
        self.db_manager = DBManager.get_instance()
        self.result_dir = get_backtest_result_dir()
        self.strategies_dir = get_strategies_dir()
        
        # 确保目录存在
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        # 存储分析结果
        self.analysis_results = []
        self.stocks_data = []
        self.pattern_stats = defaultdict(int)
        
    def analyze_stock(self, code: str, buy_date: str, pattern_type: str = "", 
                     days_before: int = 20, days_after: int = 10) -> Dict[str, Any]:
        """
        分析单个股票的买点，包括所有可用技术指标和多周期分析
        
        Args:
            code: 股票代码
            buy_date: 买点日期，格式为YYYYMMDD
            pattern_type: 买点类型描述，如"回踩反弹"、"横盘突破"等
            days_before: 分析买点前几天的数据
            days_after: 分析买点后几天的数据
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 日期转换
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d")
            end_date = (buy_date_obj + datetime.timedelta(days=days_after)).strftime("%Y%m%d")
            
            # 修改为不指定起始日期，只指定结束日期，获取所有历史数据
            # 默认从数据库获取股票的全部历史数据
            f = formula.Formula(code, end=end_date)
            
            # 检查数据有效性
            if not isinstance(f.dataDay.close, (list, np.ndarray)) or len(f.dataDay.close) == 0 or not np.any(f.dataDay.close):
                logger.warning(f"股票 {code} 数据为空")
                return {}
                
            # 买点索引
            buy_index = None
            for i, date in enumerate(f.dataDay.history['date']):
                date_str = pd.to_datetime(date).strftime("%Y%m%d")
                if date_str == buy_date:
                    buy_index = i
                    break
            
            if buy_index is None:
                logger.warning(f"未找到股票 {code} 买点日期 {buy_date} 的数据")
                return {}
            
            # 创建结果字典
            result = {
                'code': code,
                'name': f.name,
                'industry': f.industry,
                'buy_date': buy_date,
                'pattern_type': pattern_type,
                'buy_price': f.dataDay.close[buy_index],
                'periods': {},
                'patterns': []
            }
            
            # 分析各个周期
            periods = {
                'daily': KlinePeriod.DAILY,
                'min15': KlinePeriod.MIN_15,
                'min30': KlinePeriod.MIN_30,
                'min60': KlinePeriod.MIN_60,
                'weekly': KlinePeriod.WEEKLY,
                'monthly': KlinePeriod.MONTHLY
            }
            
            for period_name, period_type in periods.items():
                period_result = self._analyze_period(code, buy_date, period_type, buy_index, end_date=end_date)
                if period_result:
                    result['periods'][period_name] = period_result
            
            # 提取跨周期共性特征
            result['cross_period_patterns'] = self._extract_cross_period_patterns(result['periods'])
            
            # 合并所有周期的形态
            all_patterns = []
            for period_data in result['periods'].values():
                if period_data and 'patterns' in period_data:
                    all_patterns.extend(period_data['patterns'])
                    
            # 统计模式出现次数
            for pattern in all_patterns:
                self.pattern_stats[pattern] += 1
                
            # 统一存放的形态列表，去重
            result['patterns'] = list(set(all_patterns))
            
            # 添加到分析结果
            self.stocks_data.append(result)
            self.analysis_results.append(result)
            
            return result
                
        except Exception as e:
            logger.error(f"分析股票 {code} 时出错: {e}")
            return {}
            
    def _analyze_period(self, code: str, buy_date: str, period: KlinePeriod, daily_buy_index: int,
                       start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        分析指定周期的技术指标
        
        Args:
            code: 股票代码
            buy_date: 买点日期
            period: 周期类型
            daily_buy_index: 日线数据的买点索引
            start_date: 开始日期，默认为None，表示使用所有历史数据
            end_date: 结束日期
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 获取股票数据，不指定起始日期，只指定结束日期
            stock_data = formula.StockData(code, period, end=end_date)
            
            if not hasattr(stock_data, 'close') or len(stock_data.close) == 0:
                logger.warning(f"未获取到股票 {code} 周期 {period.name} 的数据")
                return {}
                
            # 准备数据DataFrame
            data = pd.DataFrame({
                'date': stock_data.history['date'],
                'open': stock_data.open,
                'high': stock_data.high,
                'low': stock_data.low,
                'close': stock_data.close,
                'volume': stock_data.volume if hasattr(stock_data, 'volume') else None
            })
            
            # 买点日期（转换为当前周期）
            buy_date_obj = datetime.datetime.strptime(buy_date, "%Y%m%d").date()
            
            # 在当前周期中找到对应的买点位置
            buy_index = None
            for i, date in enumerate(data['date']):
                if isinstance(date, str):
                    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                elif isinstance(date, pd.Timestamp):
                    date = date.date()
                if date >= buy_date_obj:
                    buy_index = i
                    break
                    
            if buy_index is None:
                logger.warning(f"未找到股票 {code} 周期 {period.name} 买点日期 {buy_date} 的数据")
                return {}
                
            # 计算所有技术指标
            result = self._calculate_all_indicators(data, buy_index)
            
            return result
                
        except Exception as e:
            logger.error(f"分析周期 {period.name} 时出错: {e}")
            return {}
            
    def _calculate_all_indicators(self, data: pd.DataFrame, buy_index: int, days_before: int = 10) -> Dict[str, Any]:
        """
        计算所有技术指标并进行形态识别
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            days_before: 分析买点前几天的数据
            
        Returns:
            Dict: 分析结果，包含各指标值和识别出的形态
        """
        result = {
            'indicators': {},
            'patterns': []
        }
        
        try:
            # 股票基础数据
            close = data['close'].values
            open_price = data['open'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values if 'volume' in data.columns else None
            
            # 基本指标计算
            self._calculate_ma_indicators(data, buy_index, result)
            self._calculate_macd_indicators(data, buy_index, result)
            self._calculate_kdj_indicators(data, buy_index, result)
            self._calculate_rsi_indicators(data, buy_index, result)
            self._calculate_boll_indicators(data, buy_index, result)
            self._calculate_vol_indicators(data, buy_index, result)
            
            # 进阶指标计算
            self._calculate_advanced_indicators(data, buy_index, result)
            
            # ZXM体系指标计算
            self._calculate_zxm_indicators(data, buy_index, result)
            
            # 特殊形态识别
            self._identify_patterns(data, buy_index, days_before, result)
            
            return result
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            return result
            
    def _calculate_ma_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算均线指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 计算MA指标
            ma_indicator = IndicatorFactory.create_indicator("MA", periods=[5, 10, 20, 30, 60])
            ma_result = ma_indicator.compute(data)
            
            close = data['close'].values
            ma5 = ma_result['MA5'].values
            ma10 = ma_result['MA10'].values
            ma20 = ma_result['MA20'].values
            ma30 = ma_result['MA30'].values
            ma60 = ma_result['MA60'].values
            
            result['indicators']['ma'] = {
                'ma5': ma5[buy_index],
                'ma10': ma10[buy_index],
                'ma20': ma20[buy_index],
                'ma30': ma30[buy_index],
                'ma60': ma60[buy_index],
                'close': close[buy_index],
                'ma5_diff': ma5[buy_index] - ma5[buy_index-1] if buy_index > 0 else None,
                'ma10_diff': ma10[buy_index] - ma10[buy_index-1] if buy_index > 0 else None,
                'ma20_diff': ma20[buy_index] - ma20[buy_index-1] if buy_index > 0 else None,
                'close_ma5_ratio': close[buy_index] / ma5[buy_index] if ma5[buy_index] > 0 else None,
                'close_ma10_ratio': close[buy_index] / ma10[buy_index] if ma10[buy_index] > 0 else None,
                'close_ma20_ratio': close[buy_index] / ma20[buy_index] if ma20[buy_index] > 0 else None,
                'ma5_ma10_ratio': ma5[buy_index] / ma10[buy_index] if ma10[buy_index] > 0 else None,
                'ma10_ma20_ratio': ma10[buy_index] / ma20[buy_index] if ma20[buy_index] > 0 else None
            }
            
            # 检测均线多头排列
            if ma5[buy_index] > ma10[buy_index] > ma20[buy_index]:
                result['patterns'].append('均线多头排列')
            
            # 检测均线交叉
            if buy_index > 0:
                # 5日均线上穿10日均线
                if ma5[buy_index-1] < ma10[buy_index-1] and ma5[buy_index] > ma10[buy_index]:
                    result['patterns'].append('MA5上穿MA10')
                # 10日均线上穿20日均线
                if ma10[buy_index-1] < ma20[buy_index-1] and ma10[buy_index] > ma20[buy_index]:
                    result['patterns'].append('MA10上穿MA20')
                    
            # 计算EMA指标
            ema_indicator = IndicatorFactory.create_indicator("EMA", periods=[5, 10, 20, 30, 60])
            ema_result = ema_indicator.compute(data)
            
            ema5 = ema_result['EMA5'].values
            ema10 = ema_result['EMA10'].values
            ema20 = ema_result['EMA20'].values
            ema30 = ema_result['EMA30'].values
            ema60 = ema_result['EMA60'].values
            
            result['indicators']['ema'] = {
                'ema5': ema5[buy_index],
                'ema10': ema10[buy_index],
                'ema20': ema20[buy_index],
                'ema30': ema30[buy_index],
                'ema60': ema60[buy_index],
                'close': close[buy_index],
                'ema5_diff': ema5[buy_index] - ema5[buy_index-1] if buy_index > 0 else None,
                'ema10_diff': ema10[buy_index] - ema10[buy_index-1] if buy_index > 0 else None,
                'ema20_diff': ema20[buy_index] - ema20[buy_index-1] if buy_index > 0 else None,
                'close_ema5_ratio': close[buy_index] / ema5[buy_index] if ema5[buy_index] > 0 else None,
                'close_ema10_ratio': close[buy_index] / ema10[buy_index] if ema10[buy_index] > 0 else None,
                'close_ema20_ratio': close[buy_index] / ema20[buy_index] if ema20[buy_index] > 0 else None
            }
            
            # 检测EMA多头排列
            if ema5[buy_index] > ema10[buy_index] > ema20[buy_index]:
                result['patterns'].append('EMA多头排列')
                
            # 计算WMA指标
            wma_indicator = IndicatorFactory.create_indicator("WMA", periods=[5, 10, 20, 30])
            wma_result = wma_indicator.compute(data)
            
            wma5 = wma_result['WMA5'].values
            wma10 = wma_result['WMA10'].values
            wma20 = wma_result['WMA20'].values
            wma30 = wma_result['WMA30'].values
            
            result['indicators']['wma'] = {
                'wma5': wma5[buy_index],
                'wma10': wma10[buy_index],
                'wma20': wma20[buy_index],
                'wma30': wma30[buy_index],
                'close': close[buy_index],
                'wma5_diff': wma5[buy_index] - wma5[buy_index-1] if buy_index > 0 else None,
                'wma10_diff': wma10[buy_index] - wma10[buy_index-1] if buy_index > 0 else None,
                'wma20_diff': wma20[buy_index] - wma20[buy_index-1] if buy_index > 0 else None
            }
            
            # 检测WMA多头排列
            if wma5[buy_index] > wma10[buy_index] > wma20[buy_index]:
                result['patterns'].append('WMA多头排列')
                
        except Exception as e:
            logger.error(f"计算均线指标时出错: {e}")
            
    def _calculate_macd_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算MACD指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 计算MACD指标
            macd_indicator = IndicatorFactory.create_indicator("MACD")
            macd_result = macd_indicator.compute(data)
            
            diff = macd_result['DIF'].values
            dea = macd_result['DEA'].values
            macd = macd_result['MACD'].values
            
            result['indicators']['macd'] = {
                'diff': diff[buy_index],
                'diff_prev_1': diff[buy_index-1] if buy_index > 0 else None,
                'diff_prev_2': diff[buy_index-2] if buy_index > 1 else None,
                'dea': dea[buy_index],
                'dea_prev_1': dea[buy_index-1] if buy_index > 0 else None,
                'dea_prev_2': dea[buy_index-2] if buy_index > 1 else None,
                'macd': macd[buy_index],
                'macd_prev_1': macd[buy_index-1] if buy_index > 0 else None,
                'macd_prev_2': macd[buy_index-2] if buy_index > 1 else None,
                'diff_diff': diff[buy_index] - diff[buy_index-1] if buy_index > 0 else None,
                'dea_diff': dea[buy_index] - dea[buy_index-1] if buy_index > 0 else None,
                'macd_diff': macd[buy_index] - macd[buy_index-1] if buy_index > 0 else None
            }
            
            # MACD金叉检测
            if buy_index > 0 and diff[buy_index-1] < dea[buy_index-1] and diff[buy_index] > dea[buy_index]:
                result['patterns'].append('MACD金叉')
            
            # MACD柱状图由负转正
            if buy_index > 0 and macd[buy_index-1] < 0 and macd[buy_index] > 0:
                result['patterns'].append('MACD由负转正')
                
            # MACD零轴上方金叉
            if (buy_index > 0 and diff[buy_index-1] < dea[buy_index-1] and diff[buy_index] > dea[buy_index] and
                diff[buy_index] > 0 and dea[buy_index] > 0):
                result['patterns'].append('MACD零轴上方金叉')
                
            # MACD底背离检测
            # 在前20个交易日内寻找收盘价的低点
            if buy_index >= 20:
                min_price_idx = np.argmin(data['close'].values[buy_index-20:buy_index]) + buy_index-20
                curr_price = data['close'].values[buy_index]
                min_price = data['close'].values[min_price_idx]
                
                # 如果当前价格高于低点，但MACD的DIF值低于低点时的DIF值，形成底背离
                if curr_price > min_price and diff[buy_index] > diff[min_price_idx]:
                    result['patterns'].append('MACD底背离')
                    
        except Exception as e:
            logger.error(f"计算MACD指标时出错: {e}")
            
    def _calculate_kdj_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算KDJ指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 计算KDJ指标
            kdj_indicator = IndicatorFactory.create_indicator("KDJ")
            kdj_result = kdj_indicator.compute(data)
            
            k = kdj_result['K'].values
            d = kdj_result['D'].values
            j = kdj_result['J'].values
            
            result['indicators']['kdj'] = {
                'k': k[buy_index],
                'k_prev_1': k[buy_index-1] if buy_index > 0 else None,
                'k_prev_2': k[buy_index-2] if buy_index > 1 else None,
                'd': d[buy_index],
                'd_prev_1': d[buy_index-1] if buy_index > 0 else None,
                'd_prev_2': d[buy_index-2] if buy_index > 1 else None,
                'j': j[buy_index],
                'j_prev_1': j[buy_index-1] if buy_index > 0 else None,
                'j_prev_2': j[buy_index-2] if buy_index > 1 else None,
                'k_diff': k[buy_index] - k[buy_index-1] if buy_index > 0 else None,
                'd_diff': d[buy_index] - d[buy_index-1] if buy_index > 0 else None,
                'j_diff': j[buy_index] - j[buy_index-1] if buy_index > 0 else None
            }
            
            # KDJ金叉检测
            if buy_index > 0 and k[buy_index-1] < d[buy_index-1] and k[buy_index] > d[buy_index]:
                result['patterns'].append('KDJ金叉')
            
            # KDJ超卖区金叉
            if (buy_index > 0 and k[buy_index-1] < d[buy_index-1] and k[buy_index] > d[buy_index] and
                k[buy_index-1] < 20 and d[buy_index-1] < 20):
                result['patterns'].append('KDJ超卖区金叉')
                
            # J值超卖反弹
            if buy_index > 0 and j[buy_index-1] < 0 and j[buy_index] > j[buy_index-1]:
                result['patterns'].append('J值超卖反弹')
                
            # KDJ三线向上发散
            if (buy_index > 1 and 
                k[buy_index] > k[buy_index-1] > k[buy_index-2] and
                d[buy_index] > d[buy_index-1] > d[buy_index-2] and
                j[buy_index] > j[buy_index-1] > j[buy_index-2] and
                j[buy_index] > d[buy_index] > k[buy_index]):
                result['patterns'].append('KDJ三线向上发散')
                
        except Exception as e:
            logger.error(f"计算KDJ指标时出错: {e}")
            
    def _calculate_rsi_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算RSI指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 计算RSI指标，周期为6、12、24
            rsi6_indicator = IndicatorFactory.create_indicator("RSI", period=6)
            rsi6_result = rsi6_indicator.compute(data)
            
            rsi12_indicator = IndicatorFactory.create_indicator("RSI", period=12)
            rsi12_result = rsi12_indicator.compute(data)
            
            rsi24_indicator = IndicatorFactory.create_indicator("RSI", period=24)
            rsi24_result = rsi24_indicator.compute(data)
            
            # 提取RSI值，处理不同的列名
            rsi6 = rsi6_result['RSI6'].values if 'RSI6' in rsi6_result.columns else rsi6_result['rsi'].values
            rsi12 = rsi12_result['RSI12'].values if 'RSI12' in rsi12_result.columns else rsi12_result['rsi'].values
            rsi24 = rsi24_result['RSI24'].values if 'RSI24' in rsi24_result.columns else rsi24_result['rsi'].values
            
            result['indicators']['rsi'] = {
                'rsi6': rsi6[buy_index],
                'rsi12': rsi12[buy_index],
                'rsi24': rsi24[buy_index],
                'rsi6_prev': rsi6[buy_index-1] if buy_index > 0 else None,
                'rsi12_prev': rsi12[buy_index-1] if buy_index > 0 else None,
                'rsi24_prev': rsi24[buy_index-1] if buy_index > 0 else None,
                'rsi6_diff': rsi6[buy_index] - rsi6[buy_index-1] if buy_index > 0 else None,
                'rsi12_diff': rsi12[buy_index] - rsi12[buy_index-1] if buy_index > 0 else None,
                'rsi24_diff': rsi24[buy_index] - rsi24[buy_index-1] if buy_index > 0 else None
            }
            
            # RSI超卖反弹
            if buy_index > 0 and rsi6[buy_index] > rsi6[buy_index-1] and rsi6[buy_index-1] < 30:
                result['patterns'].append('RSI超卖反弹')
                
            # RSI金叉
            if buy_index > 0 and rsi6[buy_index-1] < rsi12[buy_index-1] and rsi6[buy_index] > rsi12[buy_index]:
                result['patterns'].append('RSI金叉')
                
            # RSI底背离检测
            if buy_index >= 20:
                min_price_idx = np.argmin(data['close'].values[buy_index-20:buy_index]) + buy_index-20
                curr_price = data['close'].values[buy_index]
                min_price = data['close'].values[min_price_idx]
                
                if curr_price > min_price and rsi6[min_price_idx] > rsi6[buy_index]:
                    result['patterns'].append('RSI底背离')
                    
            # 使用RSIMA指标
            try:
                rsima_indicator = IndicatorFactory.create_indicator("RSIMA")
                rsima_result = rsima_indicator.compute(data)
                
                rsi_value = rsima_result['rsi'].values
                rsi_ma = rsima_result['rsi_ma'].values
                trend_strength = rsima_result['trend_strength'].values if 'trend_strength' in rsima_result.columns else np.zeros_like(rsi_value)
                
                result['indicators']['rsima'] = {
                    'rsi': rsi_value[buy_index],
                    'rsi_ma': rsi_ma[buy_index],
                    'trend_strength': trend_strength[buy_index],
                    'rsi_prev': rsi_value[buy_index-1] if buy_index > 0 else None,
                    'rsi_ma_prev': rsi_ma[buy_index-1] if buy_index > 0 else None
                }
                
                # RSI金叉均线
                if buy_index > 0 and rsi_value[buy_index-1] < rsi_ma[buy_index-1] and rsi_value[buy_index] > rsi_ma[buy_index]:
                    result['patterns'].append('RSI金叉均线')
                
                # 检测强势趋势
                if trend_strength[buy_index] > 80:
                    result['patterns'].append('RSIMA强势趋势')
                elif trend_strength[buy_index] > 50:
                    result['patterns'].append('RSIMA中等趋势')
            except Exception as e:
                logger.warning(f"计算RSIMA指标失败: {e}")
                
        except Exception as e:
            logger.error(f"计算RSI指标时出错: {e}")
            
    def _calculate_boll_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算BOLL布林带指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 计算BOLL指标
            boll_indicator = IndicatorFactory.create_indicator("BOLL")
            boll_result = boll_indicator.compute(data)
            
            upper = boll_result['upper'].values
            middle = boll_result['middle'].values
            lower = boll_result['lower'].values
            
            close = data['close'].values
            
            result['indicators']['boll'] = {
                'upper': upper[buy_index],
                'middle': middle[buy_index],
                'lower': lower[buy_index],
                'width': (upper[buy_index] - lower[buy_index]) / middle[buy_index] if middle[buy_index] > 0 else None,
                'close_position': (close[buy_index] - lower[buy_index]) / (upper[buy_index] - lower[buy_index]) if (upper[buy_index] - lower[buy_index]) > 0 else None
            }
            
            # BOLL突破检测
            if buy_index > 0 and close[buy_index] > middle[buy_index] and close[buy_index-1] < middle[buy_index-1]:
                result['patterns'].append('BOLL中轨突破')
            
            if close[buy_index] > upper[buy_index]:
                result['patterns'].append('BOLL上轨突破')
                
            if buy_index > 0 and close[buy_index-1] < lower[buy_index-1] and close[buy_index] > lower[buy_index]:
                result['patterns'].append('BOLL下轨支撑反弹')
                
            # BOLL带宽收窄后放大
            if buy_index > 5:
                width_now = (upper[buy_index] - lower[buy_index]) / middle[buy_index] if middle[buy_index] > 0 else 0
                width_prev = (upper[buy_index-1] - lower[buy_index-1]) / middle[buy_index-1] if middle[buy_index-1] > 0 else 0
                width_min = float('inf')
                
                for i in range(buy_index-5, buy_index):
                    width = (upper[i] - lower[i]) / middle[i] if middle[i] > 0 else 0
                    width_min = min(width_min, width)
                
                if width_now > width_prev and width_prev <= width_min:
                    result['patterns'].append('BOLL带宽收窄后放大')
                    
        except Exception as e:
            logger.error(f"计算BOLL指标时出错: {e}")
            
    def _calculate_vol_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算成交量相关指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            if 'volume' not in data.columns or data['volume'].isnull().all():
                logger.warning("缺少成交量数据，跳过成交量指标计算")
                return
                
            volume = data['volume'].values
            close = data['close'].values
            
            # 基本成交量统计
            result['indicators']['volume'] = {
                'volume': volume[buy_index],
                'volume_prev': volume[buy_index-1] if buy_index > 0 else None,
                'volume_change_ratio': volume[buy_index] / volume[buy_index-1] if buy_index > 0 and volume[buy_index-1] > 0 else None,
                'volume_ma5': np.mean(volume[max(0, buy_index-4):buy_index+1]) if buy_index >= 4 else None,
                'volume_ma10': np.mean(volume[max(0, buy_index-9):buy_index+1]) if buy_index >= 9 else None
            }
            
            # 计算VOL指标
            try:
                vol_indicator = IndicatorFactory.create_indicator("VOL")
                vol_result = vol_indicator.compute(data)
                
                vol = vol_result['volume'].values if 'volume' in vol_result.columns else volume
                vol_ma5 = vol_result['volume_ma5'].values if 'volume_ma5' in vol_result.columns else np.zeros_like(vol)
                vol_ma10 = vol_result['volume_ma10'].values if 'volume_ma10' in vol_result.columns else np.zeros_like(vol)
                
                result['indicators']['vol'] = {
                    'vol': vol[buy_index],
                    'vol_ma5': vol_ma5[buy_index],
                    'vol_ma10': vol_ma10[buy_index],
                    'vol_ratio': vol[buy_index] / vol_ma5[buy_index] if vol_ma5[buy_index] > 0 else None
                }
                
                # 检测放量上涨
                if (buy_index > 0 and 
                    volume[buy_index] > volume[buy_index-1] * 1.5 and 
                    close[buy_index] > close[buy_index-1]):
                    result['patterns'].append('放量上涨')
                
                # 检测缩量上涨
                if (buy_index > 0 and 
                    volume[buy_index] < volume[buy_index-1] * 0.8 and 
                    close[buy_index] > close[buy_index-1]):
                    result['patterns'].append('缩量上涨')
            except Exception as e:
                logger.warning(f"计算VOL指标失败: {e}")
                
            # 计算VR指标(成交量比率)
            try:
                vr_indicator = IndicatorFactory.create_indicator("VR")
                vr_result = vr_indicator.compute(data)
                
                vr = vr_result['vr'].values
                vr_ma = vr_result['vr_ma'].values
                
                result['indicators']['vr'] = {
                    'vr': vr[buy_index],
                    'vr_ma': vr_ma[buy_index],
                    'vr_prev': vr[buy_index-1] if buy_index > 0 else None,
                    'vr_diff': vr[buy_index] - vr[buy_index-1] if buy_index > 0 else None
                }
                
                # VR指标形态识别
                # VR底部区域反弹
                if buy_index > 0 and vr[buy_index] > vr[buy_index-1] and vr[buy_index-1] < 70:
                    result['patterns'].append('VR超卖反弹')
                
                # VR上穿均线
                if buy_index > 0 and vr[buy_index] > vr_ma[buy_index] and vr[buy_index-1] < vr_ma[buy_index-1]:
                    result['patterns'].append('VR上穿均线')
            except Exception as e:
                logger.warning(f"计算VR指标失败: {e}")
                
            # 计算OBV能量潮指标
            try:
                obv_indicator = IndicatorFactory.create_indicator("OBV")
                obv_result = obv_indicator.compute(data)
                
                obv = obv_result['obv'].values
                obv_ma = obv_result['obv_ma'].values
                
                result['indicators']['obv'] = {
                    'obv': obv[buy_index],
                    'obv_ma': obv_ma[buy_index],
                    'obv_prev': obv[buy_index-1] if buy_index > 0 else None,
                    'obv_diff': obv[buy_index] - obv[buy_index-1] if buy_index > 0 else None
                }
                
                # OBV上穿均线
                if buy_index > 0 and obv[buy_index] > obv_ma[buy_index] and obv[buy_index-1] < obv_ma[buy_index-1]:
                    result['patterns'].append('OBV上穿均线')
            except Exception as e:
                logger.warning(f"计算OBV指标失败: {e}")
                
        except Exception as e:
            logger.error(f"计算成交量指标时出错: {e}")
            
    def _calculate_advanced_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算高级技术指标
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 计算三重指数平滑移动平均线(TRIX)指标
            try:
                trix_indicator = IndicatorFactory.create_indicator("TRIX")
                trix_result = trix_indicator.compute(data)
                
                trix = trix_result['TRIX'].values
                matrix = trix_result['MATRIX'].values
                
                result['indicators']['trix'] = {
                    'trix': trix[buy_index],
                    'matrix': matrix[buy_index],
                    'trix_prev': trix[buy_index-1] if buy_index > 0 else None,
                    'matrix_prev': matrix[buy_index-1] if buy_index > 0 else None
                }
                
                # TRIX金叉
                if buy_index > 0 and trix[buy_index-1] < matrix[buy_index-1] and trix[buy_index] > matrix[buy_index]:
                    result['patterns'].append('TRIX金叉')
                    
                # TRIX零轴上穿
                if buy_index > 0 and trix[buy_index-1] < 0 and trix[buy_index] > 0:
                    result['patterns'].append('TRIX零轴上穿')
            except Exception as e:
                logger.warning(f"计算TRIX指标失败: {e}")
                
            # 计算动量指标(MTM)
            try:
                mtm_indicator = IndicatorFactory.create_indicator("MTM")
                mtm_result = mtm_indicator.compute(data)
                
                mtm = mtm_result['mtm'].values
                mtm_ma = mtm_result['mtm_ma'].values
                
                result['indicators']['mtm'] = {
                    'mtm': mtm[buy_index],
                    'mtm_ma': mtm_ma[buy_index],
                    'mtm_prev': mtm[buy_index-1] if buy_index > 0 else None,
                    'mtm_ma_prev': mtm_ma[buy_index-1] if buy_index > 0 else None
                }
                
                # MTM金叉
                if buy_index > 0 and mtm[buy_index-1] < mtm_ma[buy_index-1] and mtm[buy_index] > mtm_ma[buy_index]:
                    result['patterns'].append('MTM金叉')
                    
                # MTM零轴上穿
                if buy_index > 0 and mtm[buy_index-1] < 0 and mtm[buy_index] > 0:
                    result['patterns'].append('MTM零轴上穿')
            except Exception as e:
                logger.warning(f"计算MTM指标失败: {e}")
                
            # 计算恐慌指数(VIX)
            try:
                vix_indicator = IndicatorFactory.create_indicator("VIX")
                vix_result = vix_indicator.compute(data)
                
                vix = vix_result['vix'].values
                vix_ma = vix_result['vix_ma'].values
                
                result['indicators']['vix'] = {
                    'vix': vix[buy_index],
                    'vix_ma': vix_ma[buy_index],
                    'vix_prev': vix[buy_index-1] if buy_index > 0 else None,
                    'vix_ma_prev': vix_ma[buy_index-1] if buy_index > 0 else None
                }
                
                # VIX高位回落
                if buy_index > 0 and vix[buy_index] < vix[buy_index-1] and vix[buy_index-1] > 30:
                    result['patterns'].append('VIX高位回落')
                    
                # VIX低位上升
                if buy_index > 0 and vix[buy_index] > vix[buy_index-1] and vix[buy_index-1] < 15 and vix[buy_index] < 20:
                    result['patterns'].append('VIX低位上升')
            except Exception as e:
                logger.warning(f"计算VIX指标失败: {e}")
                
            # 计算量价背离(DIVERGENCE)
            try:
                divergence_indicator = IndicatorFactory.create_indicator("DIVERGENCE")
                divergence_result = divergence_indicator.compute(data)
                
                price_volume_bull_div = divergence_result['price_volume_bull_div'].values if 'price_volume_bull_div' in divergence_result.columns else np.zeros(len(data))
                price_volume_bear_div = divergence_result['price_volume_bear_div'].values if 'price_volume_bear_div' in divergence_result.columns else np.zeros(len(data))
                
                result['indicators']['divergence'] = {
                    'price_volume_bull_div': bool(price_volume_bull_div[buy_index]),
                    'price_volume_bear_div': bool(price_volume_bear_div[buy_index])
                }
                
                # 量价底背离
                if price_volume_bull_div[buy_index]:
                    result['patterns'].append('量价底背离')
            except Exception as e:
                logger.warning(f"计算DIVERGENCE指标失败: {e}")
                
            # 计算张新民吸筹(ZXM_ABSORB)
            try:
                zxm_absorb_indicator = IndicatorFactory.create_indicator("ZXM_ABSORB")
                zxm_absorb_result = zxm_absorb_indicator.compute(data)
                
                absorb_signal = zxm_absorb_result['absorb_signal'].values if 'absorb_signal' in zxm_absorb_result.columns else np.zeros(len(data))
                absorb_strength = zxm_absorb_result['absorb_strength'].values if 'absorb_strength' in zxm_absorb_result.columns else np.zeros(len(data))
                
                result['indicators']['zxm_absorb'] = {
                    'absorb_signal': bool(absorb_signal[buy_index]),
                    'absorb_strength': absorb_strength[buy_index]
                }
                
                # 张新民吸筹
                if absorb_signal[buy_index]:
                    result['patterns'].append('ZXM吸筹')
            except Exception as e:
                logger.warning(f"计算ZXM_ABSORB指标失败: {e}")
                
            # 计算多周期共振(MULTI_PERIOD_RESONANCE)
            try:
                mpr_indicator = IndicatorFactory.create_indicator("MULTI_PERIOD_RESONANCE")
                mpr_result = mpr_indicator.compute(data)
                
                buy_strength = mpr_result['buy_strength'].values if 'buy_strength' in mpr_result.columns else np.zeros(len(data))
                buy_signal = mpr_result['buy_signal'].values if 'buy_signal' in mpr_result.columns else np.zeros(len(data))
                
                result['indicators']['multi_period_resonance'] = {
                    'buy_strength': buy_strength[buy_index],
                    'buy_signal': bool(buy_signal[buy_index])
                }
                
                # 多周期共振买点
                if buy_signal[buy_index]:
                    result['patterns'].append('多周期共振买点')
            except Exception as e:
                logger.warning(f"计算MULTI_PERIOD_RESONANCE指标失败: {e}")
                
            # 计算V形反转(V_SHAPED_REVERSAL)
            try:
                vshape_indicator = IndicatorFactory.create_indicator("V_SHAPED_REVERSAL")
                vshape_result = vshape_indicator.compute(data)
                
                v_signal = vshape_result['v_signal'].values if 'v_signal' in vshape_result.columns else np.zeros(len(data))
                v_strength = vshape_result['v_strength'].values if 'v_strength' in vshape_result.columns else np.zeros(len(data))
                
                result['indicators']['v_shaped_reversal'] = {
                    'v_signal': bool(v_signal[buy_index]),
                    'v_strength': v_strength[buy_index]
                }
                
                # V形反转
                if v_signal[buy_index]:
                    result['patterns'].append('V形反转')
            except Exception as e:
                logger.warning(f"计算V_SHAPED_REVERSAL指标失败: {e}")
        
        except Exception as e:
            logger.error(f"计算高级指标时出错: {e}")
            
    def _calculate_zxm_indicators(self, data: pd.DataFrame, buy_index: int, result: Dict[str, Any]) -> None:
        """
        计算ZXM体系指标，各指标分别计算，不整合在一起
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            result: 结果字典，将被修改
        """
        try:
            # 初始化ZXM趋势指标结果字典
            result['indicators']['zxm_trend'] = {}
            
            # 计算ZXM趋势-日线上移指标
            try:
                zxm_daily_trend = IndicatorFactory.create_indicator("ZXM_DAILY_TREND_UP")
                zxm_daily_trend_result = zxm_daily_trend.compute(data)
                
                ma60 = zxm_daily_trend_result['MA60'].values if 'MA60' in zxm_daily_trend_result.columns else np.zeros(len(data))
                ma120 = zxm_daily_trend_result['MA120'].values if 'MA120' in zxm_daily_trend_result.columns else np.zeros(len(data))
                j1 = zxm_daily_trend_result['J1'].values if 'J1' in zxm_daily_trend_result.columns else np.zeros(len(data), dtype=bool)
                j2 = zxm_daily_trend_result['J2'].values if 'J2' in zxm_daily_trend_result.columns else np.zeros(len(data), dtype=bool)
                xg = zxm_daily_trend_result['XG'].values if 'XG' in zxm_daily_trend_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_trend']['daily_trend_up'] = {
                    'ma60': ma60[buy_index],
                    'ma120': ma120[buy_index],
                    'j1': bool(j1[buy_index]),
                    'j2': bool(j2[buy_index]),
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM日线上移趋势形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM日线上移趋势')
            except Exception as e:
                logger.warning(f"计算ZXM日线上移趋势指标失败: {e}")
            
            # 初始化ZXM弹性指标结果字典
            result['indicators']['zxm_elasticity'] = {}
            
            # 计算ZXM弹性-振幅指标
            try:
                zxm_amplitude = IndicatorFactory.create_indicator("ZXM_AMPLITUDE_ELASTICITY")
                zxm_amplitude_result = zxm_amplitude.compute(data)
                
                amplitude = zxm_amplitude_result['Amplitude'].values if 'Amplitude' in zxm_amplitude_result.columns else np.zeros(len(data))
                a1 = zxm_amplitude_result['A1'].values if 'A1' in zxm_amplitude_result.columns else np.zeros(len(data), dtype=bool)
                xg = zxm_amplitude_result['XG'].values if 'XG' in zxm_amplitude_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_elasticity']['amplitude_elasticity'] = {
                    'amplitude': amplitude[buy_index],
                    'a1': bool(a1[buy_index]),
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM振幅弹性形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM振幅弹性')
            except Exception as e:
                logger.warning(f"计算ZXM振幅弹性指标失败: {e}")
                
            # 计算ZXM弹性-涨幅指标
            try:
                zxm_rise = IndicatorFactory.create_indicator("ZXM_RISE_ELASTICITY")
                zxm_rise_result = zxm_rise.compute(data)
                
                rise_ratio = zxm_rise_result['RiseRatio'].values if 'RiseRatio' in zxm_rise_result.columns else np.zeros(len(data))
                a1 = zxm_rise_result['A1'].values if 'A1' in zxm_rise_result.columns else np.zeros(len(data), dtype=bool)
                xg = zxm_rise_result['XG'].values if 'XG' in zxm_rise_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_elasticity']['rise_elasticity'] = {
                    'rise_ratio': rise_ratio[buy_index],
                    'a1': bool(a1[buy_index]),
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM涨幅弹性形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM涨幅弹性')
            except Exception as e:
                logger.warning(f"计算ZXM涨幅弹性指标失败: {e}")
                
            # 初始化ZXM买点指标结果字典
            result['indicators']['zxm_buypoint'] = {}
                
            # 计算ZXM买点-日MACD指标
            try:
                zxm_daily_macd = IndicatorFactory.create_indicator("ZXM_DAILY_MACD")
                zxm_daily_macd_result = zxm_daily_macd.compute(data)
                
                macd = zxm_daily_macd_result['MACD'].values if 'MACD' in zxm_daily_macd_result.columns else np.zeros(len(data))
                xg = zxm_daily_macd_result['XG'].values if 'XG' in zxm_daily_macd_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_buypoint']['daily_macd'] = {
                    'macd': macd[buy_index],
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM日MACD买点形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM日MACD买点')
            except Exception as e:
                logger.warning(f"计算ZXM日MACD买点指标失败: {e}")
                
            # 计算ZXM买点-换手率指标
            try:
                # 检查是否有capital列，如果没有，尝试添加模拟值
                data_with_capital = data.copy()
                if 'capital' not in data.columns and 'volume' in data.columns:
                    data_with_capital['capital'] = data['volume'] * 100
                
                zxm_turnover = IndicatorFactory.create_indicator("ZXM_TURNOVER")
                zxm_turnover_result = zxm_turnover.compute(data_with_capital)
                
                turnover = zxm_turnover_result['Turnover'].values if 'Turnover' in zxm_turnover_result.columns else np.zeros(len(data))
                xg = zxm_turnover_result['XG'].values if 'XG' in zxm_turnover_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_buypoint']['turnover'] = {
                    'turnover': turnover[buy_index],
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM换手买点形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM换手买点')
            except Exception as e:
                logger.warning(f"计算ZXM换手买点指标失败: {e}")
                
            # 计算ZXM买点-缩量指标
            try:
                zxm_vol_shrink = IndicatorFactory.create_indicator("ZXM_VOLUME_SHRINK")
                zxm_vol_shrink_result = zxm_vol_shrink.compute(data)
                
                vol_ratio = zxm_vol_shrink_result['VOL_RATIO'].values if 'VOL_RATIO' in zxm_vol_shrink_result.columns else np.zeros(len(data))
                xg = zxm_vol_shrink_result['XG'].values if 'XG' in zxm_vol_shrink_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_buypoint']['volume_shrink'] = {
                    'vol_ratio': vol_ratio[buy_index],
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM缩量买点形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM缩量买点')
            except Exception as e:
                logger.warning(f"计算ZXM缩量买点指标失败: {e}")
                
            # 计算ZXM买点-回踩均线指标
            try:
                zxm_ma_callback = IndicatorFactory.create_indicator("ZXM_MA_CALLBACK", callback_percent=4.0)
                zxm_ma_callback_result = zxm_ma_callback.compute(data)
                
                a20 = zxm_ma_callback_result['A20'].values if 'A20' in zxm_ma_callback_result.columns else np.zeros(len(data), dtype=bool)
                a30 = zxm_ma_callback_result['A30'].values if 'A30' in zxm_ma_callback_result.columns else np.zeros(len(data), dtype=bool)
                a60 = zxm_ma_callback_result['A60'].values if 'A60' in zxm_ma_callback_result.columns else np.zeros(len(data), dtype=bool)
                a120 = zxm_ma_callback_result['A120'].values if 'A120' in zxm_ma_callback_result.columns else np.zeros(len(data), dtype=bool)
                xg = zxm_ma_callback_result['XG'].values if 'XG' in zxm_ma_callback_result.columns else np.zeros(len(data), dtype=bool)
                
                result['indicators']['zxm_buypoint']['ma_callback'] = {
                    'a20': bool(a20[buy_index]),
                    'a30': bool(a30[buy_index]),
                    'a60': bool(a60[buy_index]),
                    'a120': bool(a120[buy_index]),
                    'signal': bool(xg[buy_index])
                }
                
                # 添加ZXM回踩均线买点形态
                if xg[buy_index]:
                    result['patterns'].append('ZXM回踩均线买点')
            except Exception as e:
                logger.warning(f"计算ZXM回踩均线买点指标失败: {e}")
                
            # 计算各类指标满足情况 - 但不再合并为一个总得分
            try:
                # 统计满足的ZXM指标数量
                trend_indicators = result['indicators']['zxm_trend'].keys() if 'zxm_trend' in result['indicators'] else []
                elasticity_indicators = result['indicators']['zxm_elasticity'].keys() if 'zxm_elasticity' in result['indicators'] else []
                buypoint_indicators = result['indicators']['zxm_buypoint'].keys() if 'zxm_buypoint' in result['indicators'] else []
                
                trend_count = sum(1 for ind in trend_indicators if result['indicators']['zxm_trend'][ind]['signal'])
                elasticity_count = sum(1 for ind in elasticity_indicators if result['indicators']['zxm_elasticity'][ind]['signal'])
                buypoint_count = sum(1 for ind in buypoint_indicators if result['indicators']['zxm_buypoint'][ind]['signal'])
                
                # 计算各类指标得分
                trend_score = (trend_count / len(trend_indicators)) * 100 if trend_indicators else 0
                elasticity_score = (elasticity_count / len(elasticity_indicators)) * 100 if elasticity_indicators else 0
                buypoint_score = (buypoint_count / len(buypoint_indicators)) * 100 if buypoint_indicators else 0
                
                # 将各部分得分单独保存，不计算综合得分
                result['indicators']['zxm_trend_score'] = trend_score
                result['indicators']['zxm_elasticity_score'] = elasticity_score
                result['indicators']['zxm_buypoint_score'] = buypoint_score
                
                # 添加单独的买点形态
                if trend_score >= 50:
                    result['patterns'].append('ZXM趋势满足')
                if elasticity_score >= 50:
                    result['patterns'].append('ZXM弹性满足')
                if buypoint_score >= 50:
                    result['patterns'].append('ZXM买点满足')
                
            except Exception as e:
                logger.warning(f"计算ZXM各部分指标得分失败: {e}")
                
        except Exception as e:
            logger.error(f"计算ZXM体系指标时出错: {e}")
            
    def _identify_patterns(self, data: pd.DataFrame, buy_index: int, days_before: int, result: Dict[str, Any]) -> None:
        """
        识别特殊形态
        
        Args:
            data: 股票数据
            buy_index: 买点索引
            days_before: 分析买点前几天的数据
            result: 结果字典，将被修改
        """
        try:
            # 检查数据长度是否足够
            if buy_index < days_before:
                logger.warning(f"数据长度不足，无法进行形态识别，需要至少{days_before}天数据")
                return
            
            close = data['close'].values
            open_price = data['open'].values
            high = data['high'].values
            low = data['low'].values
            
            # 突破前期高点
            if buy_index >= days_before:
                prev_high = np.max(high[buy_index-days_before:buy_index])
                if close[buy_index] > prev_high:
                    result['patterns'].append('突破前期高点')
            
            # 回踩均线支撑
            if 'ma' in result['indicators'] and 'ma20' in result['indicators']['ma']:
                ma20 = result['indicators']['ma']['ma20']
                if buy_index > 0 and low[buy_index] <= ma20 <= close[buy_index]:
                    result['patterns'].append('回踩均线支撑')
            
            # 长上影线
            if high[buy_index] > close[buy_index]:
                upper_shadow = high[buy_index] - max(open_price[buy_index], close[buy_index])
                body = abs(close[buy_index] - open_price[buy_index])
                if body > 0 and upper_shadow > body * 2:
                    result['patterns'].append('长上影线')
            
            # 长下影线
            if low[buy_index] < close[buy_index]:
                lower_shadow = min(open_price[buy_index], close[buy_index]) - low[buy_index]
                body = abs(close[buy_index] - open_price[buy_index])
                if body > 0 and lower_shadow > body * 2:
                    result['patterns'].append('长下影线')
                    
            # 十字星
            body = abs(close[buy_index] - open_price[buy_index])
            candle_range = high[buy_index] - low[buy_index]
            if candle_range > 0 and body / candle_range < 0.1:
                result['patterns'].append('十字星')
                
            # 吞没形态
            if (buy_index > 0 and 
                ((close[buy_index] > open_price[buy_index] > close[buy_index-1] > open_price[buy_index-1]) or  # 看涨吞没
                 (close[buy_index] < open_price[buy_index] < close[buy_index-1] < open_price[buy_index-1]))):  # 看跌吞没
                if close[buy_index] > open_price[buy_index]:
                    result['patterns'].append('看涨吞没')
                else:
                    result['patterns'].append('看跌吞没')
                    
            # 岛型反转
            if buy_index > 1:
                gap_down = high[buy_index-1] < low[buy_index-2]  # 向下跳空
                gap_up = low[buy_index] > high[buy_index-1]     # 向上跳空
                if gap_down and gap_up:
                    result['patterns'].append('岛型反转')
                    
        except Exception as e:
            logger.error(f"识别特殊形态时出错: {e}")
            
    def _extract_cross_period_patterns(self, periods_data: Dict[str, Dict]) -> List[str]:
        """
        提取跨周期共性特征
        
        Args:
            periods_data: 各周期的分析结果
            
        Returns:
            List: 跨周期共性特征列表
        """
        try:
            # 如果只有一个周期或没有周期数据，返回空列表
            if len(periods_data) <= 1:
                return []
                
            # 按周期从低到高排序
            period_order = {
                'min15': 0,
                'min30': 1,
                'min60': 2,
                'daily': 3,
                'weekly': 4,
                'monthly': 5
            }
            
            # 周期中文名称
            period_names = {
                'min15': '15分钟',
                'min30': '30分钟',
                'min60': '60分钟',
                'daily': '日线',
                'weekly': '周线',
                'monthly': '月线'
            }
            
            sorted_periods = sorted(periods_data.keys(), key=lambda x: period_order.get(x, 999))
            
            # 收集所有周期的形态
            all_patterns = defaultdict(list)
            for period in sorted_periods:
                if periods_data[period] and 'patterns' in periods_data[period]:
                    for pattern in periods_data[period]['patterns']:
                        all_patterns[pattern].append(period)
            
            # 提取出现在多个周期的形态
            cross_period_patterns = []
            for pattern, periods in all_patterns.items():
                if len(periods) >= 2:
                    # 转换周期为中文名称
                    period_names_list = [period_names.get(p, p) for p in periods]
                    periods_str = '+'.join(period_names_list)
                    cross_period_patterns.append(f"{pattern}({periods_str})")
                    
                    # 重要的指标出现在日线及以上周期
                    if 'daily' in periods and ('weekly' in periods or 'monthly' in periods):
                        cross_period_patterns.append(f"强信号:{pattern}(日线+{'周线' if 'weekly' in periods else ''}{'月线' if 'monthly' in periods else ''})")
            
            return cross_period_patterns
            
        except Exception as e:
            logger.error(f"提取跨周期共性特征时出错: {e}")
            return []
    
    def batch_analyze(self, input_file: str, output_file: str, pattern_type: str = "") -> None:
        """
        批量分析多个股票
        
        Args:
            input_file: 输入文件路径，包含股票代码和买点日期
            output_file: 输出文件路径
            pattern_type: 买点类型描述
        """
        try:
            # 读取输入文件
            input_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        buy_date = parts[1].strip()
                        type_desc = parts[2].strip() if len(parts) > 2 else pattern_type
                        input_data.append((code, buy_date, type_desc))
            
            if not input_data:
                logger.warning(f"输入文件 {input_file} 中没有有效数据")
                return
                
            # 分析每个股票
            logger.info(f"开始批量分析 {len(input_data)} 个买点")
            for i, (code, buy_date, type_desc) in enumerate(input_data):
                logger.info(f"[{i+1}/{len(input_data)}] 分析 {code} {buy_date} {type_desc}")
                self.analyze_stock(code, buy_date, type_desc)
                
            # 保存结果
            self.save_results(output_file)
            
            # 打印模式统计
            self._print_pattern_stats()
            
        except Exception as e:
            logger.error(f"批量分析失败: {e}")
            
    def save_results(self, output_file: str) -> None:
        """
        保存分析结果
        
        Args:
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # 保存分析结果
            self.save_analysis_results(output_file)
            
            # 生成并保存Markdown格式的结果报告
            markdown_file = output_file.replace('.json', '.md')
            self.save_markdown_results(markdown_file)
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            
    def save_analysis_results(self, output_file):
        """
        保存分析结果到文件
        
        Args:
            output_file: 输出文件路径
        """
        try:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 首先进行深度拷贝，避免修改原始数据
            import copy
            clean_results = copy.deepcopy(self.analysis_results)
            
            # 递归处理所有嵌套数据，彻底移除百分号和处理特殊类型
            def clean_object(obj):
                if isinstance(obj, str):
                    return obj.replace('%', '')
                elif isinstance(obj, dict):
                    return {k: clean_object(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_object(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return [clean_object(x) for x in obj.tolist()]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif pd.isna(obj) or obj is None or obj == np.nan:
                    return None
                return obj
            
            # 清理结果
            cleaned_data = clean_object(clean_results)
            
            # 使用标准JSON编码器，不使用自定义编码器避免潜在问题
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
            # 额外验证步骤：读取文件并检查是否包含百分号
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 如果仍然包含百分号，使用字符串替换方法处理
            if '%' in content:
                logger.warning("JSON内容中仍存在百分号，使用字符串替换方法修复...")
                content = content.replace('%', '')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            logger.info(f"分析结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")
            # 最后的备用方案：使用系统命令移除所有百分号
            try:
                logger.info("尝试使用系统命令移除百分号...")
                temp_file = f"{output_file}.tmp"
                os.system(f"cat {output_file} | tr -d '%' > {temp_file}")
                os.system(f"mv {temp_file} {output_file}")
                logger.info("使用系统命令修复完成")
            except Exception as e2:
                logger.error(f"所有修复方法均失败: {str(e2)}")
            
    def _print_pattern_stats(self) -> None:
        """打印模式统计信息"""
        if not self.pattern_stats:
            logger.info("没有检测到任何模式")
            return
            
        logger.info("=" * 40)
        logger.info("模式统计信息:")
        logger.info("=" * 40)
        
        # 按出现次数排序
        sorted_patterns = sorted(self.pattern_stats.items(), key=lambda x: x[1], reverse=True)
        
        for pattern, count in sorted_patterns:
            percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
            logger.info(f"{pattern}: {count} 次 ({percentage:.2f}百分比)")
            
        logger.info("=" * 40)
        
    def generate_strategy(self, output_file: str) -> None:
        """
        根据分析结果生成策略
        
        Args:
            output_file: 输出策略文件路径
        """
        try:
            if not self.pattern_stats or not self.stocks_data:
                logger.warning("没有足够的数据生成策略")
                return
                
            # 按出现频率排序的模式
            top_patterns = sorted(self.pattern_stats.items(), key=lambda x: x[1], reverse=True)
            top_5_patterns = [p[0] for p in top_patterns[:5]]
            
            # 生成策略文本
            strategy_text = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
基于 {len(self.stocks_data)} 个买点的统一回测系统生成策略
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

最常见的5个形态:
{', '.join(top_5_patterns)}
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
"""
            
            # 添加通用函数定义
            strategy_text += """
def MA(close, period):
    \"\"\"计算移动平均线\"\"\"
    return pd.Series(close).rolling(period).mean().values

def EMA(close, period):
    \"\"\"计算指数移动平均线\"\"\"
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def REF(series, n):
    \"\"\"引用N个周期前的数据\"\"\"
    if n <= 0:
        return series
    series_pd = pd.Series(series)
    return series_pd.shift(n).values

def HHV(series, n):
    \"\"\"N个周期内的最高值\"\"\"
    return pd.Series(series).rolling(n).max().values

def LLV(series, n):
    \"\"\"N个周期内的最低值\"\"\"
    return pd.Series(series).rolling(n).min().values

def SMA(series, n, m):
    \"\"\"计算平滑移动平均\"\"\"
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = (m * series[i] + (n - m) * result[i-1]) / n
    return result

def MACD(close, fast=12, slow=26, signal=9):
    \"\"\"计算MACD指标\"\"\"
    ema_fast = EMA(close, fast)
    ema_slow = EMA(close, slow)
    dif = ema_fast - ema_slow
    dea = EMA(dif, signal)
    macd = (dif - dea) * 2
    return dif, dea, macd

def KDJ(close, high, low, n=9, m1=3, m2=3):
    \"\"\"计算KDJ指标\"\"\"
    high_n = pd.Series(high).rolling(n).max()
    low_n = pd.Series(low).rolling(n).min()
    rsv = (pd.Series(close) - low_n) / (high_n - low_n) * 100
    
    k = pd.Series(rsv).ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k.values, d.values, j.values

def RSI(close, period=14):
    \"\"\"计算RSI指标\"\"\"
    diff = pd.Series(close).diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi.values

def BOLL(close, period=20, dev=2):
    \"\"\"计算BOLL指标\"\"\"
    middle = pd.Series(close).rolling(period).mean()
    std = pd.Series(close).rolling(period).std()
    
    upper = middle + std * dev
    lower = middle - std * dev
    
    return upper.values, middle.values, lower.values
"""
            
            # 添加买入信号函数
            strategy_text += """
def is_buy_signal(context, stock):
    \"\"\"
    买入信号判断
    
    Args:
        context: 上下文环境
        stock: 股票数据对象
        
    Returns:
        bool: 是否为买入信号
    \"\"\"
    # 基本条件检查
    if not stock.is_trading:
        return False
        
    # 获取基本数据
    close = stock.close
    open_price = stock.open
    high = stock.high
    low = stock.low
    volume = stock.volume
    
    # 计算常用指标
    ma5 = MA(close, 5)
    ma10 = MA(close, 10)
    ma20 = MA(close, 20)
    ma60 = MA(close, 60)
    
    # MACD指标
    dif, dea, macd = MACD(close)
    
    # KDJ指标
    k, d, j = KDJ(close, high, low)
    
    # RSI指标
    rsi6 = RSI(close, 6)
    rsi12 = RSI(close, 12)
    
    # 策略条件 - 基于回测统计
    conditions = []
"""
            
            # 添加最常见的模式作为策略条件
            for pattern in top_5_patterns:
                if '均线多头排列' in pattern:
                    strategy_text += "    # 均线多头排列\n"
                    strategy_text += "    conditions.append(ma5[-1] > ma10[-1] and ma10[-1] > ma20[-1])\n"
                elif 'MACD金叉' in pattern:
                    strategy_text += "    # MACD金叉\n"
                    strategy_text += "    conditions.append(dif[-1] > dea[-1] and dif[-2] < dea[-2])\n"
                elif 'KDJ金叉' in pattern:
                    strategy_text += "    # KDJ金叉\n"
                    strategy_text += "    conditions.append(k[-1] > d[-1] and k[-2] < d[-2])\n"
                elif 'BOLL下轨支撑反弹' in pattern:
                    strategy_text += "    # BOLL下轨支撑反弹\n"
                    strategy_text += "    upper, middle, lower = BOLL(close)\n"
                    strategy_text += "    conditions.append(close[-1] > lower[-1] and close[-2] < lower[-2])\n"
                elif 'RSI超卖反弹' in pattern:
                    strategy_text += "    # RSI超卖反弹\n"
                    strategy_text += "    conditions.append(rsi6[-1] > rsi6[-2] and rsi6[-2] < 30)\n"
                elif '突破前期高点' in pattern:
                    strategy_text += "    # 突破前期高点\n"
                    strategy_text += "    conditions.append(close[-1] > max(high[-21:-1]))\n"
                elif '放量上涨' in pattern:
                    strategy_text += "    # 放量上涨\n"
                    strategy_text += "    conditions.append(volume[-1] > volume[-2] * 1.5 and close[-1] > close[-2])\n"
                else:
                    # 通用模式注释
                    strategy_text += f"    # {pattern} - 自定义条件\n"
                    strategy_text += "    # conditions.append(False)  # 请根据实际情况实现\n"
            
            # 完成策略
            strategy_text += """
    # 最终信号 - 满足至少2个条件
    return len([1 for c in conditions if c]) >= 2

# 策略测试
if __name__ == "__main__":
    print("策略测试")
    # 在这里添加测试代码
"""
            
            # 保存策略文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(strategy_text)
                
            logger.info(f"策略已生成并保存到 {output_file}")
            
        except Exception as e:
            logger.error(f"生成策略失败: {e}")
            
    def save_markdown_results(self, output_file):
        """
        保存分析结果到Markdown格式文件
        
        Args:
            output_file: 输出Markdown文件路径
        """
        try:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入标题
                f.write("# 回测分析报告\n\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"分析股票数量: {len(self.stocks_data)}\n\n")
                
                # 写入模式统计
                f.write("## 技术形态统计\n\n")
                f.write("| 技术形态 | 出现次数 | 出现比例 |\n")
                f.write("|---------|---------|--------|\n")
                
                # 按出现次数排序
                sorted_patterns = sorted(self.pattern_stats.items(), key=lambda x: x[1], reverse=True)
                
                for pattern, count in sorted_patterns:
                    percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                    f.write(f"| {pattern} | {count} | {percentage:.2f}% |\n")
                
                # 添加新的部分：根据共性技术形态总结选股策略
                f.write("\n## 共性技术形态选股策略\n\n")
                
                # 高频形态 (出现比例 > 50%)
                high_freq_patterns = [(p, c) for p, c in sorted_patterns if c/len(self.stocks_data)*100 > 50] if self.stocks_data else []
                # 中频形态 (出现比例 > 30%)
                medium_freq_patterns = [(p, c) for p, c in sorted_patterns if 30 < c/len(self.stocks_data)*100 <= 50] if self.stocks_data else []
                # 低频但重要形态 (与关键技术指标相关)
                key_patterns = [(p, c) for p, c in sorted_patterns 
                               if any(k in p.lower() for k in ['金叉', '底背离', '反弹', '突破', '上穿']) 
                               and 10 < c/len(self.stocks_data)*100 <= 30] if self.stocks_data else []
                
                # ZXM专属形态
                zxm_patterns = [(p, c) for p, c in sorted_patterns if p.startswith('ZXM')] if self.stocks_data else []
                zxm_trend_patterns = [(p, c) for p, c in zxm_patterns if '趋势' in p]
                zxm_elasticity_patterns = [(p, c) for p, c in zxm_patterns if '弹性' in p]
                zxm_buypoint_patterns = [(p, c) for p, c in zxm_patterns if '买点' in p]
                
                # 生成选股策略
                f.write("### 推荐选股策略\n\n")
                f.write("根据回测结果中的共性技术形态，建议采用以下选股策略：\n\n")
                
                # 基于ZXM体系的选股策略
                f.write("#### 1. ZXM体系选股策略\n\n")
                f.write("选股条件：同时满足以下条件的股票\n\n")
                
                # ZXM趋势条件
                if zxm_trend_patterns:
                    top_trend = sorted(zxm_trend_patterns, key=lambda x: x[1], reverse=True)[:2]
                    f.write("**趋势条件**（至少满足1个）：\n")
                    for pattern, count in top_trend:
                        percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                        f.write(f"- {pattern} ({percentage:.2f}%)\n")
                    f.write("\n")
                
                # ZXM弹性条件
                if zxm_elasticity_patterns:
                    top_elasticity = sorted(zxm_elasticity_patterns, key=lambda x: x[1], reverse=True)[:2]
                    f.write("**弹性条件**（至少满足1个）：\n")
                    for pattern, count in top_elasticity:
                        percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                        f.write(f"- {pattern} ({percentage:.2f}%)\n")
                    f.write("\n")
                
                # ZXM买点条件
                if zxm_buypoint_patterns:
                    top_buypoint = sorted(zxm_buypoint_patterns, key=lambda x: x[1], reverse=True)[:3]
                    f.write("**买点条件**（至少满足2个）：\n")
                    for pattern, count in top_buypoint:
                        percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                        f.write(f"- {pattern} ({percentage:.2f}%)\n")
                    f.write("\n")
                
                # 基于高频技术形态的选股策略
                f.write("#### 2. 通用技术形态选股策略\n\n")
                f.write("选股条件：同时满足以下条件的股票\n\n")
                
                # 高频形态条件
                if high_freq_patterns:
                    f.write("**必要条件**（必须全部满足）：\n")
                    for pattern, count in high_freq_patterns[:3]:  # 最多取前3个
                        percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                        f.write(f"- {pattern} ({percentage:.2f}%)\n")
                    f.write("\n")
                
                # 中频形态条件
                if medium_freq_patterns:
                    f.write("**重要条件**（至少满足2个）：\n")
                    for pattern, count in medium_freq_patterns[:5]:  # 最多取前5个
                        percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                        f.write(f"- {pattern} ({percentage:.2f}%)\n")
                    f.write("\n")
                
                # 关键形态条件
                if key_patterns:
                    f.write("**确认条件**（至少满足1个）：\n")
                    for pattern, count in key_patterns[:3]:  # 最多取前3个
                        percentage = count / len(self.stocks_data) * 100 if self.stocks_data else 0
                        f.write(f"- {pattern} ({percentage:.2f}%)\n")
                    f.write("\n")
                
                # 选股策略实现代码示例
                f.write("### 策略代码实现示例\n\n")
                
                # Python代码示例
                f.write("```python\n")
                f.write("# ZXM体系选股策略示例代码\n")
                f.write("def zxm_select_strategy(stock_data):\n")
                f.write("    \"\"\"ZXM体系选股策略\"\"\"\n")
                f.write("    # 趋势条件\n")
                f.write("    trend_conditions = [\n")
                
                # 添加趋势条件
                if zxm_trend_patterns:
                    for pattern, _ in sorted(zxm_trend_patterns, key=lambda x: x[1], reverse=True)[:2]:
                        if '日线上移' in pattern:
                            f.write("        ma60_up = stock_data['ma60'][-1] > stock_data['ma60'][-2],  # 60日均线上移\n")
                            f.write("        ma120_up = stock_data['ma120'][-1] > stock_data['ma120'][-2],  # 120日均线上移\n")
                        elif '周线上移' in pattern:
                            f.write("        weekly_ma10_up = stock_data['weekly_ma10'][-1] > stock_data['weekly_ma10'][-2],  # 周线10周均线上移\n")
                        elif '月KDJ' in pattern:
                            f.write("        monthly_kdj_up = stock_data['monthly_kdj_d'][-1] > stock_data['monthly_kdj_d'][-2] and stock_data['monthly_kdj_k'][-1] > stock_data['monthly_kdj_k'][-2],  # 月KDJ上移\n")
                
                f.write("    ]\n\n")
                f.write("    # 弹性条件\n")
                f.write("    elasticity_conditions = [\n")
                
                # 添加弹性条件
                if zxm_elasticity_patterns:
                    for pattern, _ in sorted(zxm_elasticity_patterns, key=lambda x: x[1], reverse=True)[:2]:
                        if '振幅弹性' in pattern:
                            f.write("        max_amplitude > 8.1,  # 近期最大振幅超过8.1%\n")
                        elif '涨幅弹性' in pattern:
                            f.write("        max_rise > 7.0,  # 近期最大涨幅超过7%\n")
                
                f.write("    ]\n\n")
                f.write("    # 买点条件\n")
                f.write("    buypoint_conditions = [\n")
                
                # 添加买点条件
                if zxm_buypoint_patterns:
                    for pattern, _ in sorted(zxm_buypoint_patterns, key=lambda x: x[1], reverse=True)[:3]:
                        if '日MACD买点' in pattern:
                            f.write("        stock_data['macd'][-1] < 0.9,  # 日MACD小于0.9\n")
                        elif '换手买点' in pattern:
                            f.write("        stock_data['turnover'][-1] > 0.7,  # 换手率大于0.7%\n")
                        elif '缩量买点' in pattern:
                            f.write("        stock_data['volume'][-1] < np.mean(stock_data['volume'][-3:-1]) * 0.9,  # 成交量缩减10%以上\n")
                        elif '回踩均线买点' in pattern:
                            f.write("        abs(stock_data['close'][-1] - stock_data['ma20'][-1]) / stock_data['ma20'][-1] < 0.04,  # 收盘价回踩至20日均线4%以内\n")
                
                f.write("    ]\n\n")
                f.write("    # 策略判断\n")
                f.write("    has_trend = any(trend_conditions)  # 至少满足1个趋势条件\n")
                f.write("    has_elasticity = any(elasticity_conditions)  # 至少满足1个弹性条件\n")
                f.write("    has_buypoint = sum(buypoint_conditions) >= 2  # 至少满足2个买点条件\n\n")
                f.write("    # 最终选股条件\n")
                f.write("    return has_trend and has_elasticity and has_buypoint\n")
                f.write("```\n\n")
                
                # 通用技术形态选股策略代码示例
                f.write("```python\n")
                f.write("# 通用技术形态选股策略示例代码\n")
                f.write("def technical_select_strategy(stock_data):\n")
                f.write("    \"\"\"通用技术形态选股策略\"\"\"\n")
                f.write("    # 必要条件（必须全部满足）\n")
                f.write("    necessary_conditions = [\n")
                
                # 添加必要条件
                if high_freq_patterns:
                    for pattern, _ in high_freq_patterns[:3]:
                        if '均线多头排列' in pattern:
                            f.write("        stock_data['ma5'][-1] > stock_data['ma10'][-1] > stock_data['ma20'][-1],  # 均线多头排列\n")
                        elif 'MACD金叉' in pattern:
                            f.write("        stock_data['dif'][-1] > stock_data['dea'][-1] and stock_data['dif'][-2] < stock_data['dea'][-2],  # MACD金叉\n")
                        elif 'KDJ金叉' in pattern:
                            f.write("        stock_data['k'][-1] > stock_data['d'][-1] and stock_data['k'][-2] < stock_data['d'][-2],  # KDJ金叉\n")
                        elif '突破' in pattern:
                            f.write("        stock_data['close'][-1] > max(stock_data['high'][-21:-1]),  # 突破前期高点\n")
                
                f.write("    ]\n\n")
                f.write("    # 重要条件（至少满足2个）\n")
                f.write("    important_conditions = [\n")
                
                # 添加重要条件
                if medium_freq_patterns:
                    for pattern, _ in medium_freq_patterns[:5]:
                        if 'RSI' in pattern:
                            f.write("        stock_data['rsi6'][-1] > stock_data['rsi6'][-2] and stock_data['rsi6'][-2] < 30,  # RSI超卖反弹\n")
                        elif 'BOLL' in pattern:
                            f.write("        stock_data['close'][-1] > stock_data['boll_lower'][-1] and stock_data['close'][-2] < stock_data['boll_lower'][-2],  # BOLL下轨支撑反弹\n")
                        elif '放量' in pattern:
                            f.write("        stock_data['volume'][-1] > stock_data['volume'][-2] * 1.5 and stock_data['close'][-1] > stock_data['close'][-2],  # 放量上涨\n")
                        elif '缩量' in pattern:
                            f.write("        stock_data['volume'][-1] < stock_data['volume'][-2] * 0.8 and stock_data['close'][-1] > stock_data['close'][-2],  # 缩量上涨\n")
                
                f.write("    ]\n\n")
                f.write("    # 确认条件（至少满足1个）\n")
                f.write("    confirmation_conditions = [\n")
                
                # 添加确认条件
                if key_patterns:
                    for pattern, _ in key_patterns[:3]:
                        if '底背离' in pattern:
                            f.write("        min_price_idx = np.argmin(stock_data['close'][-20:]),\n")
                            f.write("        curr_price = stock_data['close'][-1],\n")
                            f.write("        min_price = stock_data['close'][min_price_idx],\n")
                            f.write("        curr_price > min_price and stock_data['macd'][-1] > stock_data['macd'][min_price_idx],  # MACD底背离\n")
                        elif '上穿' in pattern:
                            f.write("        stock_data['k'][-1] > stock_data['d'][-1] and stock_data['k'][-2] < stock_data['d'][-2],  # KDJ金叉\n")
                        elif '零轴' in pattern:
                            f.write("        stock_data['dif'][-1] > 0 and stock_data['dif'][-2] < 0,  # DIF上穿零轴\n")
                
                f.write("    ]\n\n")
                f.write("    # 策略判断\n")
                f.write("    all_necessary = all(necessary_conditions)  # 必须全部满足\n")
                f.write("    enough_important = sum(important_conditions) >= 2  # 至少满足2个\n")
                f.write("    has_confirmation = any(confirmation_conditions)  # 至少满足1个\n\n")
                f.write("    # 最终选股条件\n")
                f.write("    return all_necessary and enough_important and has_confirmation\n")
                f.write("```\n")
                
                f.write("\n## 各股票分析结果\n\n")
                
                # 按股票和买点日期组织数据
                stocks_by_code = {}
                for result in self.analysis_results:
                    code = result.get('code', '')
                    if code not in stocks_by_code:
                        stocks_by_code[code] = []
                    stocks_by_code[code].append(result)
                
                # 写入每个股票的分析结果
                for code, results in stocks_by_code.items():
                    f.write(f"### {code}\n\n")
                    
                    for result in results:
                        buy_date = result.get('buy_date', '')
                        pattern_type = result.get('pattern_type', '')
                        f.write(f"#### 买点日期: {buy_date} {pattern_type}\n\n")
                        
                        # 写入识别到的形态
                        patterns = result.get('patterns', [])
                        if patterns:
                            f.write("**识别到的技术形态:**\n\n")
                            for pattern in patterns:
                                f.write(f"- {pattern}\n")
                            f.write("\n")
                            
                        # 写入跨周期共性特征
                        cross_period_patterns = result.get('cross_period_patterns', [])
                        if cross_period_patterns:
                            f.write("**跨周期共性特征:**\n\n")
                            # 分别列出普通跨周期特征和强信号特征
                            normal_patterns = [p for p in cross_period_patterns if not p.startswith("强信号:")]
                            strong_patterns = [p for p in cross_period_patterns if p.startswith("强信号:")]
                            
                            if normal_patterns:
                                f.write("跨周期形态:\n\n")
                                for pattern in normal_patterns:
                                    f.write(f"- {pattern}\n")
                                f.write("\n")
                                
                            if strong_patterns:
                                f.write("强信号形态:\n\n")
                                for pattern in strong_patterns:
                                    f.write(f"- {pattern}\n")
                                f.write("\n")
                        
                        # 写入各周期分析结果
                        for period_key in ['daily', 'min15', 'min30', 'min60', 'weekly', 'monthly']:
                            period_data = result.get('periods', {}).get(period_key)
                            if not period_data:
                                continue
                                
                            period_name = {
                                'daily': '日线',
                                'min15': '15分钟',
                                'min30': '30分钟',
                                'min60': '60分钟',
                                'weekly': '周线',
                                'monthly': '月线'
                            }.get(period_key, period_key)
                            
                            f.write(f"**{period_name}分析:**\n\n")
                            
                            # 写入该周期的形态
                            period_patterns = period_data.get('patterns', [])
                            if period_patterns:
                                f.write("形态:\n\n")
                                for pattern in period_patterns:
                                    f.write(f"- {pattern}\n")
                                f.write("\n")
                            
                            # 写入指标数据
                            indicators = period_data.get('indicators', {})
                            if indicators:
                                f.write("指标数据:\n\n")
                                for indicator_name, indicator_data in indicators.items():
                                    if isinstance(indicator_data, dict):
                                        # 格式化指标名称，包含周期信息
                                        formatted_name = f"{period_name}_{indicator_name}"
                                        f.write(f"- **{formatted_name}**:\n")
                                        for k, v in indicator_data.items():
                                            if isinstance(v, (list, np.ndarray)):
                                                continue
                                            f.write(f"  - {k}: {v}\n")
                                    else:
                                        f.write(f"- **{period_name}_{indicator_name}**: {indicator_data}\n")
                                f.write("\n")
                        
                        f.write("---\n\n")
                
                # 写入结论和建议
                f.write("## 结论和建议\n\n")
                
                # 提取最常见的3个形态
                top_3_patterns = [p[0] for p in sorted_patterns[:3]]
                f.write("### 最常见的技术形态\n\n")
                for i, pattern in enumerate(top_3_patterns):
                    f.write(f"{i+1}. {pattern}\n")
                f.write("\n")
                
                # 写入建议的交易策略
                f.write("### 建议的交易策略\n\n")
                f.write("基于上述分析，可以构建以下交易策略：\n\n")
                
                # 根据最常见的形态生成简单策略建议
                for pattern in top_3_patterns:
                    if '均线多头排列' in pattern:
                        f.write("- **均线多头排列策略**: 当短期均线在上，中期均线在中，长期均线在下，形成多头排列时考虑买入\n")
                    elif 'MACD' in pattern:
                        f.write("- **MACD策略**: 关注MACD指标的金叉和底背离信号\n")
                    elif 'KDJ' in pattern:
                        f.write("- **KDJ策略**: 当KDJ指标从超卖区金叉向上时考虑买入\n")
                    elif 'RSI' in pattern:
                        f.write("- **RSI策略**: 当RSI指标从超卖区反弹上行时考虑买入\n")
                    elif 'BOLL' in pattern:
                        f.write("- **布林带策略**: 当价格触及下轨后反弹，或突破中轨上行时考虑买入\n")
                    elif 'ZXM吸筹' in pattern:
                        f.write("- **ZXM吸筹策略**: 关注成交量与价格关系，识别主力吸筹行为\n")
                    elif '放量' in pattern:
                        f.write("- **量价策略**: 关注价格上涨伴随成交量放大的情况\n")
                
                f.write("\n### 风险提示\n\n")
                f.write("- 本报告基于历史数据分析，不构成投资建议\n")
                f.write("- 投资者应结合市场环境、个股基本面等多方面因素进行决策\n")
                f.write("- 任何交易策略都存在失效的可能，应严格控制仓位和止损\n")
            
            logger.info(f"Markdown格式分析报告已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存Markdown结果失败: {str(e)}")
            
def analyze_buypoint_indicators(input_file: str, pattern_type: str, output_file: str) -> None:
    """
    分析买点的技术指标
    
    Args:
        input_file: 输入文件路径，包含股票代码和买点日期
        pattern_type: 买点类型描述
        output_file: 输出文件路径
    """
    backtest = UnifiedBacktest()
    backtest.batch_analyze(input_file, output_file, pattern_type)
    
    # 生成策略文件
    strategy_file = os.path.join(get_strategies_dir(), f"strategy_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.py")
    backtest.generate_strategy(strategy_file)
    
def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="统一回测系统")
    parser.add_argument("--input", "-i", help="输入文件路径，包含股票代码和买点日期", required=True)
    parser.add_argument("--output", "-o", help="输出文件路径", required=True)
    parser.add_argument("--type", "-t", help="买点类型描述", default="")
    
    args = parser.parse_args()
    
    logger.info(f"开始分析，输入文件: {args.input}，输出文件: {args.output}")
    analyze_buypoint_indicators(args.input, args.type, args.output)
    logger.info("分析完成")
    
if __name__ == "__main__":
    main() 