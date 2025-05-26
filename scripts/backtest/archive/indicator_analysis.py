#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from utils.path_utils import get_stock_result_file, get_backtest_result_dir
from db.db_manager import DBManager
from indicators.factory import IndicatorFactory
from indicators.common import ma, rsi

# 获取日志记录器
logger = get_logger(__name__)

class IndicatorAnalyzer:
    """
    技术指标分析器
    分析买点时间附近的技术指标情况，找出共性
    """
    
    def __init__(self):
        """初始化分析器"""
        self.db_manager = DBManager.get_instance()
        self.result_dir = get_backtest_result_dir()
        
        # 保存分析结果
        self.indicator_stats = defaultdict(lambda: defaultdict(int))
        self.stocks_data = []
        self.pattern_stats = defaultdict(int)
        
    def analyze_stock(self, code: str, buy_date: str, pattern_type: str = "", 
                     days_before: int = 10, days_after: int = 5) -> Dict[str, Any]:
        """
        分析单个股票买点附近的技术指标
        
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
            start_date = (buy_date_obj - datetime.timedelta(days=days_before*2)).strftime("%Y%m%d")
            
            # 获取股票数据
            f = formula.Formula(code, start=start_date, end=end_date)
            
            # 修正：使用numpy的any()方法判断数组
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
                'indicators': {},
                'patterns': []
            }
            
            # 准备数据DataFrame，用于传递给指标类
            data = pd.DataFrame({
                'date': f.dataDay.history['date'],
                'open': f.dataDay.open,
                'high': f.dataDay.high,
                'low': f.dataDay.low,
                'close': f.dataDay.close,
                'volume': f.dataDay.volume
            })
            
            # 分析KDJ指标
            factory = IndicatorFactory()
            kdj_indicator = factory.create_indicator("KDJ")
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
            kdj_golden_cross = False
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0 and k[i-1] < d[i-1] and k[i] > d[i]:
                    kdj_golden_cross = True
                    result['patterns'].append('KDJ金叉')
                    break
            
            # 分析MACD指标
            factory = IndicatorFactory()
            macd_indicator = factory.create_indicator("MACD")
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
            macd_golden_cross = False
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0 and diff[i-1] < dea[i-1] and diff[i] > dea[i]:
                    macd_golden_cross = True
                    result['patterns'].append('MACD金叉')
                    break
            
            # 分析均线系统
            # 创建MA指标实例
            ma_indicator = IndicatorFactory.create_indicator("MA", periods=[5, 10, 20, 30, 60])
            ma_result = ma_indicator.compute(data)
            
            close = f.dataDay.close
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
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0:
                    # 5日均线上穿10日均线
                    if ma5[i-1] < ma10[i-1] and ma5[i] > ma10[i]:
                        result['patterns'].append('MA5上穿MA10')
                    # 10日均线上穿20日均线
                    if ma10[i-1] < ma20[i-1] and ma10[i] > ma20[i]:
                        result['patterns'].append('MA10上穿MA20')
            
            # 检测回踩均线反弹
            if close[buy_index] > ma5[buy_index]:
                found_touch = False
                for i in range(max(0, buy_index-days_before), buy_index):
                    # 最低价触及或接近均线，然后反弹
                    if f.dataDay.low[i] <= ma5[i] * 1.02 and close[i] > ma5[i]:
                        found_touch = True
                        break
                if found_touch:
                    result['patterns'].append('回踩5日均线反弹')
            
            # 计算RSI指标
            rsi6_indicator = IndicatorFactory.create_indicator("RSI", period=6)
            rsi6_result = rsi6_indicator.compute(data)
            
            rsi12_indicator = IndicatorFactory.create_indicator("RSI", period=12)
            rsi12_result = rsi12_indicator.compute(data)
            
            rsi24_indicator = IndicatorFactory.create_indicator("RSI", period=24)
            rsi24_result = rsi24_indicator.compute(data)
            
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
            
            # RSI底背离检测
            lowest_price_idx = None
            lowest_price = float('inf')
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if f.dataDay.low[i] < lowest_price:
                    lowest_price = f.dataDay.low[i]
                    lowest_price_idx = i
            
            if lowest_price_idx is not None and lowest_price_idx < buy_index:
                if rsi6[lowest_price_idx] > rsi6[buy_index-1] and close[lowest_price_idx] < close[buy_index-1]:
                    result['patterns'].append('RSI底背离')
            
            # 计算BOLL指标
            boll_indicator = IndicatorFactory.create_indicator("BOLL")
            boll_result = boll_indicator.compute(data)
            
            upper = boll_result['upper'].values
            middle = boll_result['middle'].values
            lower = boll_result['lower'].values
            
            result['indicators']['boll'] = {
                'upper': upper[buy_index],
                'middle': middle[buy_index],
                'lower': lower[buy_index],
                'width': (upper[buy_index] - lower[buy_index]) / middle[buy_index] if middle[buy_index] > 0 else None,
                'close_position': (close[buy_index] - lower[buy_index]) / (upper[buy_index] - lower[buy_index]) if (upper[buy_index] - lower[buy_index]) > 0 else None
            }
            
            # BOLL突破检测
            if close[buy_index] > middle[buy_index] and close[buy_index-1] < middle[buy_index-1]:
                result['patterns'].append('BOLL中轨突破')
            
            if close[buy_index] > upper[buy_index]:
                result['patterns'].append('BOLL上轨突破')
                
            if close[buy_index-1] < lower[buy_index-1] and close[buy_index] > lower[buy_index]:
                result['patterns'].append('BOLL下轨支撑反弹')
            
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
                'close_ema20_ratio': close[buy_index] / ema20[buy_index] if ema20[buy_index] > 0 else None,
                'ema5_ema10_ratio': ema5[buy_index] / ema10[buy_index] if ema10[buy_index] > 0 else None,
                'ema10_ema20_ratio': ema10[buy_index] / ema20[buy_index] if ema20[buy_index] > 0 else None
            }
            
            # 检测EMA多头排列
            if ema5[buy_index] > ema10[buy_index] > ema20[buy_index]:
                result['patterns'].append('EMA多头排列')
            
            # 检测EMA交叉
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0:
                    # 5日EMA上穿10日EMA
                    if ema5[i-1] < ema10[i-1] and ema5[i] > ema10[i]:
                        result['patterns'].append('EMA5上穿EMA10')
                    # 10日EMA上穿20日EMA
                    if ema10[i-1] < ema20[i-1] and ema10[i] > ema20[i]:
                        result['patterns'].append('EMA10上穿EMA20')
            
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
                'wma20_diff': wma20[buy_index] - wma20[buy_index-1] if buy_index > 0 else None,
                'close_wma5_ratio': close[buy_index] / wma5[buy_index] if wma5[buy_index] > 0 else None,
                'close_wma10_ratio': close[buy_index] / wma10[buy_index] if wma10[buy_index] > 0 else None,
                'close_wma20_ratio': close[buy_index] / wma20[buy_index] if wma20[buy_index] > 0 else None
            }
            
            # 检测WMA多头排列
            if wma5[buy_index] > wma10[buy_index] > wma20[buy_index]:
                result['patterns'].append('WMA多头排列')
            
            # 检测WMA交叉
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0:
                    # 5日WMA上穿10日WMA
                    if wma5[i-1] < wma10[i-1] and wma5[i] > wma10[i]:
                        result['patterns'].append('WMA5上穿WMA10')
                    # 10日WMA上穿20日WMA
                    if wma10[i-1] < wma20[i-1] and wma10[i] > wma20[i]:
                        result['patterns'].append('WMA10上穿WMA20')
            
            # 计算BIAS指标
            bias_indicator = IndicatorFactory.create_indicator("BIAS", periods=[6, 12, 24])
            bias_result = bias_indicator.compute(data)
            
            bias6 = bias_result['BIAS6'].values
            bias12 = bias_result['BIAS12'].values
            bias24 = bias_result['BIAS24'].values
            
            result['indicators']['bias'] = {
                'bias6': bias6[buy_index],
                'bias12': bias12[buy_index],
                'bias24': bias24[buy_index],
                'bias6_prev': bias6[buy_index-1] if buy_index > 0 else None,
                'bias12_prev': bias12[buy_index-1] if buy_index > 0 else None,
                'bias24_prev': bias24[buy_index-1] if buy_index > 0 else None,
                'bias6_diff': bias6[buy_index] - bias6[buy_index-1] if buy_index > 0 else None,
                'bias12_diff': bias12[buy_index] - bias12[buy_index-1] if buy_index > 0 else None,
                'bias24_diff': bias24[buy_index] - bias24[buy_index-1] if buy_index > 0 else None
            }
            
            # BIAS金叉检测
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0:
                    # BIAS6上穿0轴
                    if bias6[i-1] < 0 and bias6[i] > 0:
                        result['patterns'].append('BIAS6上穿0轴')
                    # BIAS12上穿0轴
                    if bias12[i-1] < 0 and bias12[i] > 0:
                        result['patterns'].append('BIAS12上穿0轴')
            
            # BIAS超卖反弹检测
            if bias6[buy_index] > bias6[buy_index-1] and bias6[buy_index-1] < -6:
                result['patterns'].append('BIAS6超卖反弹')
            
            # 计算DMI指标
            dmi_indicator = IndicatorFactory.create_indicator("DMI")
            dmi_result = dmi_indicator.compute(data)
            
            pdi = dmi_result['PDI'].values
            mdi = dmi_result['MDI'].values
            adx = dmi_result['ADX'].values
            adxr = dmi_result['ADXR'].values
            
            result['indicators']['dmi'] = {
                'pdi': pdi[buy_index],
                'mdi': mdi[buy_index],
                'adx': adx[buy_index],
                'adxr': adxr[buy_index],
                'pdi_prev': pdi[buy_index-1] if buy_index > 0 else None,
                'mdi_prev': mdi[buy_index-1] if buy_index > 0 else None,
                'adx_prev': adx[buy_index-1] if buy_index > 0 else None,
                'adxr_prev': adxr[buy_index-1] if buy_index > 0 else None
            }
            
            # DMI金叉检测
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0:
                    if pdi[i-1] < mdi[i-1] and pdi[i] > mdi[i]:
                        result['patterns'].append('DMI金叉')
                        break
            
            # ADX上升检测
            adx_rising = True
            for i in range(buy_index-2, buy_index+1):
                if i > 0 and adx[i] <= adx[i-1]:
                    adx_rising = False
                    break
            if adx_rising:
                result['patterns'].append('ADX连续上升')
            
            # 计算CCI指标
            cci_indicator = IndicatorFactory.create_indicator("CCI")
            cci_result = cci_indicator.compute(data)
            
            cci = cci_result['CCI'].values
            
            result['indicators']['cci'] = {
                'cci': cci[buy_index],
                'cci_prev': cci[buy_index-1] if buy_index > 0 else None,
                'cci_diff': cci[buy_index] - cci[buy_index-1] if buy_index > 0 else None
            }
            
            # 检测CCI超卖反弹检测
            if cci[buy_index] > cci[buy_index-1] and cci[buy_index-1] < -100:
                result['patterns'].append('CCI超卖反弹')
            
            # CCI上穿0轴检测
            for i in range(max(0, buy_index-days_before), buy_index+1):
                if i > 0:
                    if cci[i-1] < 0 and cci[i] > 0:
                        result['patterns'].append('CCI上穿0轴')
                        break
            
            # 计算成交量变化
            volume = f.dataDay.volume
            result['indicators']['volume'] = {
                'volume': volume[buy_index],
                'volume_prev': volume[buy_index-1] if buy_index > 0 else None,
                'volume_change_ratio': volume[buy_index] / volume[buy_index-1] if buy_index > 0 and volume[buy_index-1] > 0 else None,
                'volume_ma5': np.mean(volume[max(0, buy_index-4):buy_index+1]) if buy_index >= 4 else None,
                'volume_ma10': np.mean(volume[max(0, buy_index-9):buy_index+1]) if buy_index >= 9 else None
            }
            
            # 计算VR指标(成交量比率)
            vr_indicator = IndicatorFactory.create_indicator("VR")
            vr_result = vr_indicator.compute(data)
            
            vr = vr_result['vr'].values
            vr_ma = vr_result['vr_ma'].values
            
            result['indicators']['vr'] = {
                'vr': vr[buy_index],
                'vr_ma': vr_ma[buy_index],
                'vr_prev': vr[buy_index-1] if buy_index > 0 else None,
                'vr_diff': vr[buy_index] - vr[buy_index-1] if buy_index > 0 else None,
                'vr_ma_diff': vr_ma[buy_index] - vr_ma[buy_index-1] if buy_index > 0 else None
            }
            
            # VR指标形态识别
            # VR底部区域反弹
            if vr[buy_index] > vr[buy_index-1] and vr[buy_index-1] < 70:
                result['patterns'].append('VR超卖反弹')
            
            # VR上穿均线
            if vr[buy_index] > vr_ma[buy_index] and vr[buy_index-1] < vr_ma[buy_index-1]:
                result['patterns'].append('VR上穿均线')
            
            # VR持续放大
            vr_rising = True
            for i in range(buy_index-2, buy_index+1):
                if i > 0 and vr[i] <= vr[i-1]:
                    vr_rising = False
                    break
            if vr_rising:
                result['patterns'].append('VR持续放大')
            
            # 计算MFI指标(资金流量指标)
            mfi_indicator = IndicatorFactory.create_indicator("MFI")
            mfi_result = mfi_indicator.compute(data)
            
            mfi = mfi_result['mfi'].values
            mfi_change = mfi_result['mfi_change'].values
            
            result['indicators']['mfi'] = {
                'mfi': mfi[buy_index],
                'mfi_prev': mfi[buy_index-1] if buy_index > 0 else None,
                'mfi_change': mfi_change[buy_index],
                'mfi_diff': mfi[buy_index] - mfi[buy_index-1] if buy_index > 0 else None
            }
            
            # MFI指标形态识别
            # MFI超卖区域反弹
            if mfi[buy_index] > mfi[buy_index-1] and mfi[buy_index-1] < 20:
                result['patterns'].append('MFI超卖反弹')
            
            # MFI上穿50
            if mfi[buy_index] > 50 and mfi[buy_index-1] < 50:
                result['patterns'].append('MFI上穿50')
            
            # MFI底背离检测
            # 寻找前20个交易日的最低价
            price_window = 20
            if buy_index >= price_window:
                min_price_idx = np.argmin(close[buy_index-price_window:buy_index]) + buy_index-price_window
                if close[min_price_idx] < close[buy_index] and mfi[min_price_idx] > mfi[buy_index]:
                    result['patterns'].append('MFI底背离')
            
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
            
            # 计算OBV能量潮指标
            obv_indicator = IndicatorFactory.create_indicator("OBV")
            obv_result = obv_indicator.compute(data)
            
            obv = obv_result['obv'].values
            obv_ma = obv_result['obv_ma'].values
            
            result['indicators']['obv'] = {
                'obv': obv[buy_index],
                'obv_ma': obv_ma[buy_index],
                'obv_prev': obv[buy_index-1] if buy_index > 0 else None,
                'obv_diff': obv[buy_index] - obv[buy_index-1] if buy_index > 0 else None,
                'obv_ma_diff': obv_ma[buy_index] - obv_ma[buy_index-1] if buy_index > 0 else None
            }
            
            # OBV形态识别
            # OBV上穿均线
            if obv[buy_index] > obv_ma[buy_index] and obv[buy_index-1] < obv_ma[buy_index-1]:
                result['patterns'].append('OBV上穿均线')
                
            # OBV正背离
            if buy_index >= price_window:
                min_price_idx = np.argmin(close[buy_index-price_window:buy_index]) + buy_index-price_window
                if close[buy_index] > close[min_price_idx] and obv[buy_index] < obv[min_price_idx]:
                    result['patterns'].append('OBV正背离')
                    
            # OBV持续增长
            obv_rising = True
            for i in range(buy_index-3, buy_index+1):
                if i > 0 and obv[i] <= obv[i-1]:
                    obv_rising = False
                    break
            if obv_rising:
                result['patterns'].append('OBV持续增长')
            
            # 计算WR威廉指标
            wr_indicator = IndicatorFactory.create_indicator("WR")
            wr_result = wr_indicator.compute(data)
            
            wr = wr_result['wr'].values
            
            result['indicators']['wr'] = {
                'wr': wr[buy_index],
                'wr_prev': wr[buy_index-1] if buy_index > 0 else None,
                'wr_diff': wr[buy_index] - wr[buy_index-1] if buy_index > 0 else None
            }
            
            # WR形态识别
            # WR超卖反弹
            if wr[buy_index] > wr[buy_index-1] and wr[buy_index-1] < -80:
                result['patterns'].append('WR超卖反弹')
                
            # WR上穿-50
            if wr[buy_index] > -50 and wr[buy_index-1] < -50:
                result['patterns'].append('WR上穿-50线')
            
            # 计算VOSC成交量摆动指标
            vosc_indicator = IndicatorFactory.create_indicator("VOSC")
            vosc_result = vosc_indicator.compute(data)
            
            vosc = vosc_result['vosc'].values
            
            result['indicators']['vosc'] = {
                'vosc': vosc[buy_index],
                'vosc_prev': vosc[buy_index-1] if buy_index > 0 else None,
                'vosc_diff': vosc[buy_index] - vosc[buy_index-1] if buy_index > 0 else None
            }
            
            # VOSC形态识别
            # VOSC上穿0轴
            if vosc[buy_index] > 0 and vosc[buy_index-1] < 0:
                result['patterns'].append('VOSC上穿0轴')
                
            # VOSC持续放大
            vosc_rising = True
            for i in range(buy_index-2, buy_index+1):
                if i > 0 and vosc[i] <= vosc[i-1]:
                    vosc_rising = False
                    break
            if vosc_rising and vosc[buy_index] > 0:
                result['patterns'].append('VOSC持续放大')
            
            # 计算PVT价量趋势指标
            pvt_indicator = IndicatorFactory.create_indicator("PVT")
            pvt_result = pvt_indicator.compute(data)
            
            pvt = pvt_result['pvt'].values
            pvt_signal = pvt_result['pvt_signal'].values
            
            result['indicators']['pvt'] = {
                'pvt': pvt[buy_index],
                'pvt_signal': pvt_signal[buy_index],
                'pvt_prev': pvt[buy_index-1] if buy_index > 0 else None,
                'pvt_diff': pvt[buy_index] - pvt[buy_index-1] if buy_index > 0 else None
            }
            
            # PVT形态识别
            # PVT上穿均线
            if pvt[buy_index] > pvt_signal[buy_index] and pvt[buy_index-1] < pvt_signal[buy_index-1]:
                result['patterns'].append('PVT上穿均线')
                
            # 计算MULTI_PERIOD_RESONANCE多周期共振指标
            try:
                mpr_indicator = IndicatorFactory.create_indicator("MULTI_PERIOD_RESONANCE")
                mpr_result = mpr_indicator.compute(data)
                
                buy_strength = mpr_result['buy_strength'].values
                buy_signal = mpr_result['buy_signal'].values
                
                result['indicators']['mpr'] = {
                    'buy_strength': buy_strength[buy_index],
                    'buy_signal': bool(buy_signal[buy_index])
                }
                
                # 添加多周期共振形态
                if buy_signal[buy_index]:
                    result['patterns'].append('多周期共振买点')
                    
                # 强度分级
                if buy_strength[buy_index] >= 0.8:
                    result['patterns'].append('强多周期共振')
                elif buy_strength[buy_index] >= 0.5:
                    result['patterns'].append('中等多周期共振')
            except Exception as e:
                logger.warning(f"计算MULTI_PERIOD_RESONANCE指标失败: {str(e)}")
                
            # 计算ZXM_ABSORB张新民吸筹理论指标
            try:
                zxm_absorb_indicator = IndicatorFactory.create_indicator("ZXM_ABSORB")
                zxm_absorb_result = zxm_absorb_indicator.compute(data)
                
                absorb_signal = zxm_absorb_result['absorb_signal'].values
                absorb_strength = zxm_absorb_result['absorb_strength'].values
                
                result['indicators']['zxm_absorb'] = {
                    'absorb_signal': bool(absorb_signal[buy_index]),
                    'absorb_strength': absorb_strength[buy_index]
                }
                
                # 添加吸筹形态
                if absorb_signal[buy_index]:
                    result['patterns'].append('张新民吸筹')
                    
                # 强度分级
                if absorb_strength[buy_index] >= 80:
                    result['patterns'].append('强力吸筹')
                elif absorb_strength[buy_index] >= 50:
                    result['patterns'].append('中等吸筹')
            except Exception as e:
                logger.warning(f"计算ZXM_ABSORB指标失败: {str(e)}")
            
            # 计算V_SHAPED_REVERSAL V形反转指标
            try:
                v_shaped_reversal_indicator = IndicatorFactory.create_indicator("V_SHAPED_REVERSAL")
                v_shaped_reversal_result = v_shaped_reversal_indicator.compute(data)
                
                v_reversal = v_shaped_reversal_result['v_reversal'].values
                v_bottom = v_shaped_reversal_result['v_bottom'].values
                
                # 获取信号数据
                v_shaped_signals = v_shaped_reversal_indicator.get_signals(v_shaped_reversal_result)
                v_buy_signal = v_shaped_signals['v_buy_signal'].values if 'v_buy_signal' in v_shaped_signals.columns else np.zeros_like(v_reversal)
                
                # 获取反转强度
                v_shaped_strength = v_shaped_reversal_indicator.get_reversal_strength(v_shaped_reversal_result)
                reversal_strength = v_shaped_strength['reversal_strength'].values if 'reversal_strength' in v_shaped_strength.columns else np.zeros_like(v_reversal)
                reversal_category = v_shaped_strength['reversal_category'].values if 'reversal_category' in v_shaped_strength.columns else np.array([None] * len(v_reversal))
                
                result['indicators']['v_shaped_reversal'] = {
                    'v_reversal': bool(v_reversal[buy_index]),
                    'v_bottom': bool(v_bottom[buy_index]),
                    'v_buy_signal': bool(v_buy_signal[buy_index]),
                    'reversal_strength': reversal_strength[buy_index],
                    'reversal_category': reversal_category[buy_index]
                }
                
                # 添加V形反转形态
                if v_reversal[buy_index]:
                    result['patterns'].append('V形反转')
                
                if v_bottom[buy_index]:
                    result['patterns'].append('V形底部')
                
                if v_buy_signal[buy_index]:
                    result['patterns'].append('V形反转买入信号')
                
                # 添加反转强度分类
                if reversal_category[buy_index] == '强烈反转':
                    result['patterns'].append('强烈V形反转')
                elif reversal_category[buy_index] == '明显反转':
                    result['patterns'].append('明显V形反转')
            except Exception as e:
                logger.warning(f"计算V_SHAPED_REVERSAL指标失败: {str(e)}")
                
            # 计算ISLAND_REVERSAL岛型反转指标
            try:
                island_reversal_indicator = IndicatorFactory.create_indicator("ISLAND_REVERSAL")
                island_reversal_result = island_reversal_indicator.compute(data)
                
                top_island_reversal = island_reversal_result['top_island_reversal'].values
                bottom_island_reversal = island_reversal_result['bottom_island_reversal'].values
                
                result['indicators']['island_reversal'] = {
                    'top_island_reversal': bool(top_island_reversal[buy_index]),
                    'bottom_island_reversal': bool(bottom_island_reversal[buy_index])
                }
                
                # 添加岛型反转形态
                if top_island_reversal[buy_index]:
                    result['patterns'].append('顶部岛型反转')
                
                if bottom_island_reversal[buy_index]:
                    result['patterns'].append('底部岛型反转')
                    
                # 检查前N天是否有底部岛型反转
                for i in range(max(0, buy_index-days_before), buy_index):
                    if bottom_island_reversal[i]:
                        result['patterns'].append('近期出现底部岛型反转')
                        break
            except Exception as e:
                logger.warning(f"计算ISLAND_REVERSAL指标失败: {str(e)}")
            
            # 计算FIBONACCI_TOOLS斐波那契工具指标
            try:
                fibonacci_indicator = IndicatorFactory.create_indicator("FIBONACCI_TOOLS")
                fibonacci_result = fibonacci_indicator.compute(data)
                
                # 获取关键支撑位和阻力位
                fib_retracement_618 = fibonacci_result['fib_retracement_0_618'].values
                fib_retracement_382 = fibonacci_result['fib_retracement_0_382'].values
                fib_retracement_236 = fibonacci_result['fib_retracement_0_236'].values
                fib_retracement_0 = fibonacci_result['fib_retracement_0_000'].values
                fib_retracement_1 = fibonacci_result['fib_retracement_1_000'].values
                
                # 提取高点和低点
                swing_high = fibonacci_result['swing_high'].values
                swing_low = fibonacci_result['swing_low'].values
                
                result['indicators']['fibonacci'] = {
                    'fib_618': fib_retracement_618[buy_index],
                    'fib_382': fib_retracement_382[buy_index],
                    'fib_236': fib_retracement_236[buy_index],
                    'swing_high': swing_high[buy_index],
                    'swing_low': swing_low[buy_index],
                    'at_support': False,
                    'at_resistance': False
                }
                
                # 检测价格是否在关键支撑位附近
                close_price = close[buy_index]
                tolerance = 0.02  # 2%的容差
                
                # 在618支撑位附近
                if fib_retracement_618[buy_index] > 0 and abs(close_price - fib_retracement_618[buy_index]) / fib_retracement_618[buy_index] < tolerance:
                    result['patterns'].append('斐波那契618支撑')
                    result['indicators']['fibonacci']['at_support'] = True
                
                # 在382支撑位附近
                if fib_retracement_382[buy_index] > 0 and abs(close_price - fib_retracement_382[buy_index]) / fib_retracement_382[buy_index] < tolerance:
                    result['patterns'].append('斐波那契382支撑')
                    result['indicators']['fibonacci']['at_support'] = True
                
                # 价格在斐波那契支撑位之间反弹
                if close_price > fib_retracement_618[buy_index] and close_price < fib_retracement_382[buy_index]:
                    # 检查是否之前有触及618支撑位
                    touched_618 = False
                    for i in range(max(0, buy_index-5), buy_index):
                        if f.dataDay.low[i] <= fib_retracement_618[i] * (1 + tolerance):
                            touched_618 = True
                            break
                    
                    if touched_618 and close_price > close[buy_index-1]:
                        result['patterns'].append('斐波那契支撑区间反弹')
            except Exception as e:
                logger.warning(f"计算FIBONACCI_TOOLS指标失败: {str(e)}")
                
            # 计算ELLIOTT_WAVE艾略特波浪指标
            try:
                elliott_indicator = IndicatorFactory.create_indicator("ELLIOTT_WAVE")
                elliott_result = elliott_indicator.compute(data)
                
                wave_number = elliott_result['wave_number'].values
                wave_label = elliott_result['wave_label'].values
                wave_direction = elliott_result['wave_direction'].values
                wave_pattern = elliott_result['wave_pattern'].values
                next_wave_prediction = elliott_result['next_wave_prediction'].values
                
                result['indicators']['elliott_wave'] = {
                    'wave_number': wave_number[buy_index],
                    'wave_label': wave_label[buy_index],
                    'wave_direction': wave_direction[buy_index],
                    'wave_pattern': wave_pattern[buy_index],
                    'next_wave_prediction': next_wave_prediction[buy_index]
                }
                
                # 添加波浪形态
                if wave_pattern[buy_index] and wave_pattern[buy_index] != '':
                    result['patterns'].append(f'艾略特波浪{wave_pattern[buy_index]}')
                
                # 检测买点位置
                if wave_label[buy_index] in ['2', '4', 'B']:
                    result['patterns'].append(f'艾略特波浪{wave_label[buy_index]}浪买点')
                
                # 上升浪开始
                if wave_direction[buy_index] == 'up' and wave_label[buy_index] in ['1', '3', '5', 'A', 'C']:
                    result['patterns'].append(f'艾略特波浪上升{wave_label[buy_index]}浪')
                
                # 特殊买点：3浪启动
                if wave_label[buy_index] == '3' and wave_direction[buy_index] == 'up':
                    result['patterns'].append('艾略特3浪启动')
            except Exception as e:
                logger.warning(f"计算ELLIOTT_WAVE指标失败: {str(e)}")
            
            # 计算TRIX三重指数平滑移动平均线指标
            try:
                trix_indicator = IndicatorFactory.create_indicator("TRIX")
                trix_result = trix_indicator.compute(data)
                
                trix = trix_result['TRIX'].values
                matrix = trix_result['MATRIX'].values
                
                result['indicators']['trix'] = {
                    'trix': trix[buy_index],
                    'matrix': matrix[buy_index],
                    'trix_prev': trix[buy_index-1] if buy_index > 0 else None,
                    'matrix_prev': matrix[buy_index-1] if buy_index > 0 else None,
                    'diff': trix[buy_index] - matrix[buy_index]
                }
                
                # 添加TRIX形态
                # TRIX上穿0轴
                if buy_index > 0 and trix[buy_index-1] < 0 and trix[buy_index] > 0:
                    result['patterns'].append('TRIX上穿0轴')
                
                # TRIX金叉
                if buy_index > 0 and trix[buy_index-1] < matrix[buy_index-1] and trix[buy_index] > matrix[buy_index]:
                    result['patterns'].append('TRIX金叉')
                
                # TRIX死叉
                if buy_index > 0 and trix[buy_index-1] > matrix[buy_index-1] and trix[buy_index] < matrix[buy_index]:
                    result['patterns'].append('TRIX死叉')
                
                # TRIX底背离
                # 寻找前20个交易日的最低价
                if buy_index >= 20:
                    min_price_idx = np.argmin(close[buy_index-20:buy_index]) + buy_index-20
                    if close[min_price_idx] < close[buy_index] and trix[min_price_idx] > trix[buy_index]:
                        result['patterns'].append('TRIX底背离')
                
            except Exception as e:
                logger.warning(f"计算TRIX指标失败: {str(e)}")
                
            # 计算VIX恐慌指数指标
            try:
                vix_indicator = IndicatorFactory.create_indicator("VIX")
                vix_result = vix_indicator.compute(data)
                
                vix = vix_result['vix'].values
                vix_ma = vix_result['vix_ma'].values
                
                result['indicators']['vix'] = {
                    'vix': vix[buy_index],
                    'vix_ma': vix_ma[buy_index],
                    'vix_prev': vix[buy_index-1] if buy_index > 0 else None,
                    'vix_ma_prev': vix_ma[buy_index-1] if buy_index > 0 else None,
                    'vix_diff': vix[buy_index] - vix[buy_index-1] if buy_index > 0 else None
                }
                
                # 添加VIX形态
                # 判断VIX处于高位(超过30)
                if vix[buy_index] > 30:
                    result['patterns'].append('VIX高位(恐慌)')
                
                # 判断VIX处于低位(低于15)
                elif vix[buy_index] < 15:
                    result['patterns'].append('VIX低位(贪婪)')
                
                # VIX从高位回落买点
                if buy_index > 0 and vix[buy_index] < vix[buy_index-1] and vix[buy_index-1] > 30:
                    result['patterns'].append('VIX高位回落')
                
                # VIX金叉
                if buy_index > 0 and vix[buy_index-1] < vix_ma[buy_index-1] and vix[buy_index] > vix_ma[buy_index]:
                    result['patterns'].append('VIX金叉')
                    
                # VIX死叉
                if buy_index > 0 and vix[buy_index-1] > vix_ma[buy_index-1] and vix[buy_index] < vix_ma[buy_index]:
                    result['patterns'].append('VIX死叉')
                    
            except Exception as e:
                logger.warning(f"计算VIX指标失败: {str(e)}")
                
            # 计算DIVERGENCE量价背离指标
            try:
                divergence_indicator = IndicatorFactory.create_indicator("DIVERGENCE")
                divergence_result = divergence_indicator.compute(data)
                
                # 提取背离信号
                price_volume_bull_div = divergence_result['price_volume_bull_div'].values
                price_volume_bear_div = divergence_result['price_volume_bear_div'].values
                
                macd_bull_div = divergence_result['macd_bull_div'].values if 'macd_bull_div' in divergence_result.columns else np.zeros_like(price_volume_bull_div)
                macd_bear_div = divergence_result['macd_bear_div'].values if 'macd_bear_div' in divergence_result.columns else np.zeros_like(price_volume_bear_div)
                
                rsi_bull_div = divergence_result['rsi_bull_div'].values if 'rsi_bull_div' in divergence_result.columns else np.zeros_like(price_volume_bull_div)
                rsi_bear_div = divergence_result['rsi_bear_div'].values if 'rsi_bear_div' in divergence_result.columns else np.zeros_like(price_volume_bear_div)
                
                # 存储背离信号
                result['indicators']['divergence'] = {
                    'price_volume_bull_div': bool(price_volume_bull_div[buy_index]),
                    'price_volume_bear_div': bool(price_volume_bear_div[buy_index]),
                    'macd_bull_div': bool(macd_bull_div[buy_index]),
                    'macd_bear_div': bool(macd_bear_div[buy_index]),
                    'rsi_bull_div': bool(rsi_bull_div[buy_index]),
                    'rsi_bear_div': bool(rsi_bear_div[buy_index])
                }
                
                # 添加背离形态
                # 价量正背离
                if price_volume_bull_div[buy_index]:
                    result['patterns'].append('价量正背离')
                
                # 价量负背离
                if price_volume_bear_div[buy_index]:
                    result['patterns'].append('价量负背离')
                
                # MACD正背离
                if macd_bull_div[buy_index]:
                    result['patterns'].append('MACD正背离')
                    
                # MACD负背离
                if macd_bear_div[buy_index]:
                    result['patterns'].append('MACD负背离')
                
                # RSI正背离
                if rsi_bull_div[buy_index]:
                    result['patterns'].append('RSI正背离')
                    
                # RSI负背离
                if rsi_bear_div[buy_index]:
                    result['patterns'].append('RSI负背离')
                
                # 检查前N天是否有背离信号
                found_div = False
                for i in range(max(0, buy_index-10), buy_index):
                    if price_volume_bull_div[i] or macd_bull_div[i] or rsi_bull_div[i]:
                        result['patterns'].append('近期出现技术指标正背离')
                        found_div = True
                        break
                
                # 多重背离(更强信号)
                div_count = sum([
                    bool(price_volume_bull_div[buy_index]),
                    bool(macd_bull_div[buy_index]),
                    bool(rsi_bull_div[buy_index])
                ])
                
                if div_count >= 2:
                    result['patterns'].append('多重技术指标正背离')
                    
            except Exception as e:
                logger.warning(f"计算DIVERGENCE指标失败: {str(e)}")
                
            # 计算Momentum动量指标
            try:
                momentum_indicator = IndicatorFactory.create_indicator("MOMENTUM")
                momentum_result = momentum_indicator.compute(data)
                
                momentum = momentum_result['momentum'].values
                momentum_signal = momentum_result['momentum_signal'].values
                
                result['indicators']['momentum'] = {
                    'momentum': momentum[buy_index],
                    'momentum_signal': momentum_signal[buy_index],
                    'momentum_prev': momentum[buy_index-1] if buy_index > 0 else None,
                    'momentum_signal_prev': momentum_signal[buy_index-1] if buy_index > 0 else None
                }
                
                # 添加动量指标形态
                # 动量上穿0轴
                if buy_index > 0 and momentum[buy_index-1] < 0 and momentum[buy_index] > 0:
                    result['patterns'].append('Momentum上穿0轴')
                
                # 动量金叉
                if buy_index > 0 and momentum[buy_index-1] < momentum_signal[buy_index-1] and momentum[buy_index] > momentum_signal[buy_index]:
                    result['patterns'].append('Momentum金叉')
                
                # 检测动量增强
                if buy_index > 2:
                    momentum_increasing = True
                    for i in range(buy_index-2, buy_index+1):
                        if momentum[i] <= momentum[i-1]:
                            momentum_increasing = False
                            break
                    
                    if momentum_increasing:
                        result['patterns'].append('Momentum持续增强')
                
                # 检测动量从低位回升
                if buy_index > 5:
                    min_momentum = min(momentum[buy_index-5:buy_index])
                    if momentum[buy_index] > 0 and min_momentum < -5:
                        result['patterns'].append('Momentum低位回升')
                        
            except Exception as e:
                logger.warning(f"计算Momentum指标失败: {str(e)}")
            
            # 计算RSIMA指标
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
                
                # 添加RSIMA形态
                # RSI超买区域（大于70）
                if rsi_value[buy_index] > 70:
                    result['patterns'].append('RSI超买区域')
                
                # RSI超卖区域（小于30）
                if rsi_value[buy_index] < 30:
                    result['patterns'].append('RSI超卖区域')
                
                # RSI金叉均线
                if buy_index > 0 and rsi_value[buy_index-1] < rsi_ma[buy_index-1] and rsi_value[buy_index] > rsi_ma[buy_index]:
                    result['patterns'].append('RSI金叉均线')
                
                # RSI死叉均线
                if buy_index > 0 and rsi_value[buy_index-1] > rsi_ma[buy_index-1] and rsi_value[buy_index] < rsi_ma[buy_index]:
                    result['patterns'].append('RSI死叉均线')
                
                # RSI从超卖区反弹
                if buy_index > 0 and rsi_value[buy_index] > rsi_value[buy_index-1] and rsi_value[buy_index-1] < 30:
                    result['patterns'].append('RSI超卖反弹')
                
                # 检测强势趋势
                if trend_strength[buy_index] > 80:
                    result['patterns'].append('RSIMA强势趋势')
                elif trend_strength[buy_index] > 50:
                    result['patterns'].append('RSIMA中等趋势')
                    
            except Exception as e:
                logger.warning(f"计算RSIMA指标失败: {str(e)}")
            
            # 记录模式统计
            for pattern in result['patterns']:
                self.pattern_stats[pattern] += 1
            
            # 统计数据
            self.stocks_data.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"分析股票 {code} 买点指标时出错: {e}")
            return {}
    
    def analyze_multiple(self, stock_list, buy_dates, pattern_types=None):
        """
        批量分析多个股票买点的技术指标
        
        Args:
            stock_list: 股票代码列表
            buy_dates: 对应的买点日期列表
            pattern_types: 对应的买点类型列表，可选
            
        Returns:
            Dict: 分析结果
        """
        if pattern_types is None:
            pattern_types = [""] * len(stock_list)
            
        if len(stock_list) != len(buy_dates) or len(stock_list) != len(pattern_types):
            raise ValueError("股票列表、买点日期列表和类型列表长度必须一致")
            
        for i, code in enumerate(stock_list):
            self.analyze_stock(code, buy_dates[i], pattern_types[i])
            
        return self.get_analysis_result()
    
    def analyze_from_csv(self, csv_file):
        """
        从CSV文件加载股票买点数据并分析
        
        Args:
            csv_file: CSV文件路径，至少包含code和buy_date列
            
        Returns:
            Dict: 分析结果
        """
        try:
            df = pd.read_csv(csv_file)
            if 'code' not in df.columns or 'buy_date' not in df.columns:
                logger.error(f"CSV文件 {csv_file} 必须包含code和buy_date列")
                return {}
                
            pattern_types = df['pattern_type'].tolist() if 'pattern_type' in df.columns else [""] * len(df)
            
            return self.analyze_multiple(
                df['code'].astype(str).tolist(),
                df['buy_date'].astype(str).tolist(),
                pattern_types
            )
        except Exception as e:
            logger.error(f"从CSV文件 {csv_file} 分析买点指标时出错: {e}")
            return {}
    
    def get_common_patterns(self, threshold=0.5):
        """
        获取共性的技术形态
        
        Args:
            threshold: 出现比例阈值，超过该比例认为是共性
            
        Returns:
            List: 共性技术形态列表
        """
        if not self.stocks_data:
            return []
            
        total_stocks = len(self.stocks_data)
        common_patterns = []
        
        for pattern, count in self.pattern_stats.items():
            if count / total_stocks >= threshold:
                common_patterns.append({
                    'pattern': pattern,
                    'count': count,
                    'ratio': count / total_stocks
                })
                
        # 按比例降序排序
        common_patterns.sort(key=lambda x: x['ratio'], reverse=True)
        return common_patterns
    
    def get_analysis_result(self):
        """
        获取分析结果
        
        Returns:
            Dict: 分析结果
        """
        if not self.stocks_data:
            return {
                'stocks_count': 0,
                'common_patterns': [],
                'stocks_data': []
            }
            
        return {
            'stocks_count': len(self.stocks_data),
            'common_patterns': self.get_common_patterns(),
            'stocks_data': self.stocks_data
        }
    
    def save_to_json(self, output_file=None):
        """
        将分析结果保存为JSON文件
        
        Args:
            output_file: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if output_file is None:
            # 使用当前日期作为文件名
            today = datetime.datetime.now().strftime("%Y%m%d")
            output_file = f"{self.result_dir}/{today}_买点指标分析.json"
            
        # 创建目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        result = self.get_analysis_result()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"分析结果已保存到 {output_file}")
        return output_file


def analyze_buypoint_indicators(input_source, source_type="csv", output_file=None):
    """
    分析买点指标共性
    
    Args:
        input_source: 输入源，可以是CSV文件路径、股票代码列表等
        source_type: 输入源类型，支持csv、list、db等
        output_file: 输出文件路径，如果为None则使用默认路径
        
    Returns:
        Dict: 分析结果
    """
    analyzer = IndicatorAnalyzer()
    
    if source_type == "csv":
        result = analyzer.analyze_from_csv(input_source)
    elif source_type == "list":
        # 输入格式: [{"code": "000001", "buy_date": "20240101", "pattern_type": "回踩反弹"}, ...]
        stock_list = [item["code"] for item in input_source]
        buy_dates = [item["buy_date"] for item in input_source]
        pattern_types = [item.get("pattern_type", "") for item in input_source]
        result = analyzer.analyze_multiple(stock_list, buy_dates, pattern_types)
    elif source_type == "db":
        # 从数据库表加载数据
        db_manager = DBManager.get_instance()
        stocks = db_manager.db.client.execute(f"SELECT code, buy_date, pattern_type FROM {input_source}")
        if not stocks:
            logger.warning(f"表 {input_source} 中没有股票买点数据")
            return {}
            
        stock_list = [item[0] for item in stocks]
        buy_dates = [item[1].strftime("%Y%m%d") for item in stocks]
        pattern_types = [item[2] if len(item) > 2 else "" for item in stocks]
        result = analyzer.analyze_multiple(stock_list, buy_dates, pattern_types)
    else:
        logger.error(f"不支持的输入源类型: {source_type}")
        return {}
    
    # 保存结果
    if result and result['stocks_count'] > 0:
        analyzer.save_to_json(output_file)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='买点技术指标分析工具')
    parser.add_argument('--input', type=str, required=True, help='输入源，根据模式不同代表不同含义')
    parser.add_argument('--type', type=str, default="csv", choices=['csv', 'db'], help='输入源类型: csv-CSV文件, db-数据库表')
    parser.add_argument('--output', type=str, help='输出文件路径，默认为data/result/日期_买点指标分析.json')
    
    args = parser.parse_args()
    
    analyze_buypoint_indicators(args.input, args.type, args.output) 