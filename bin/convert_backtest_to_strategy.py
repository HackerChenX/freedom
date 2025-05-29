#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
回测结果转换选股策略工具

将回测结果中的有效信号和模式转换为可配置的选股策略
确保保留所有重要指标，特别是ZXM体系指标
"""

import os
import sys
import yaml
import json
import re
import pandas as pd
from datetime import datetime
import argparse

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import get_logger, init_logging
from utils.path_utils import get_result_dir
from indicators.indicator_registry import indicator_registry, IndicatorEnum, STANDARD_PARAMETER_MAPPING

logger = get_logger(__name__)

# 回测系统形态到指标类型的映射
PATTERN_TO_INDICATOR = {
    # MA指标形态
    "均线多头排列": IndicatorEnum.MA,
    "均线空头排列": IndicatorEnum.MA,
    "均线多头发散": IndicatorEnum.MA,
    "均线金叉": IndicatorEnum.MA,
    "均线死叉": IndicatorEnum.MA,
    "价格站上均线": IndicatorEnum.MA,
    "价格跌破均线": IndicatorEnum.MA,
    "MA5上穿MA10": IndicatorEnum.MA,
    "MA5上穿MA20": IndicatorEnum.MA,
    "MA5上穿MA30": IndicatorEnum.MA,
    "MA10上穿MA20": IndicatorEnum.MA,
    "MA10上穿MA30": IndicatorEnum.MA,
    "MA20上穿MA30": IndicatorEnum.MA,
    "MA5下穿MA10": IndicatorEnum.MA,
    "MA5下穿MA20": IndicatorEnum.MA,
    "MA5下穿MA30": IndicatorEnum.MA,
    "MA10下穿MA20": IndicatorEnum.MA,
    "MA10下穿MA30": IndicatorEnum.MA,
    "MA20下穿MA30": IndicatorEnum.MA,
    
    # MACD指标形态
    "MACD金叉": IndicatorEnum.MACD,
    "MACD死叉": IndicatorEnum.MACD,
    "MACD零轴上穿": IndicatorEnum.MACD,
    "MACD零轴下穿": IndicatorEnum.MACD,
    "MACD柱状图由负转正": IndicatorEnum.MACD,
    "MACD柱状图由正转负": IndicatorEnum.MACD,
    "MACD底背离": IndicatorEnum.MACD,
    "MACD顶背离": IndicatorEnum.MACD,
    "MACD柱状图放大": IndicatorEnum.MACD,
    "MACD柱状图缩小": IndicatorEnum.MACD,
    
    # KDJ指标形态
    "KDJ金叉": IndicatorEnum.KDJ,
    "KDJ死叉": IndicatorEnum.KDJ,
    "KDJ超卖": IndicatorEnum.KDJ,
    "KDJ超买": IndicatorEnum.KDJ,
    "KDJ超卖反弹": IndicatorEnum.KDJ,
    "KDJ超买回落": IndicatorEnum.KDJ,
    "KDJ中位看涨": IndicatorEnum.KDJ,
    "KDJ中位看跌": IndicatorEnum.KDJ,
    "KDJ顶背离": IndicatorEnum.KDJ,
    "KDJ底背离": IndicatorEnum.KDJ,
    
    # RSI指标形态
    "RSI超卖": IndicatorEnum.RSI,
    "RSI超买": IndicatorEnum.RSI,
    "RSI金叉": IndicatorEnum.RSI,
    "RSI死叉": IndicatorEnum.RSI,
    "RSI超卖反弹": IndicatorEnum.RSI,
    "RSI超买回落": IndicatorEnum.RSI,
    "RSI底背离": IndicatorEnum.RSI,
    "RSI顶背离": IndicatorEnum.RSI,
    "RSI6底背离": IndicatorEnum.RSI,
    "RSI12底背离": IndicatorEnum.RSI,
    "RSI24底背离": IndicatorEnum.RSI,
    "RSI6顶背离": IndicatorEnum.RSI,
    "RSI12顶背离": IndicatorEnum.RSI,
    "RSI24顶背离": IndicatorEnum.RSI,
    "RSI6金叉RSI12": IndicatorEnum.RSI,
    "RSI6死叉RSI12": IndicatorEnum.RSI,
    "RSI多头排列": IndicatorEnum.RSI,
    "RSI空头排列": IndicatorEnum.RSI,
    
    # BOLL指标形态
    "BOLL通道收窄": IndicatorEnum.BOLL,
    "BOLL通道扩大": IndicatorEnum.BOLL,
    "价格触及上轨": IndicatorEnum.BOLL,
    "价格触及下轨": IndicatorEnum.BOLL,
    "价格突破上轨": IndicatorEnum.BOLL,
    "价格跌破下轨": IndicatorEnum.BOLL,
    "价格在通道中运行": IndicatorEnum.BOLL,
    "BOLL多头排列": IndicatorEnum.BOLL,
    "BOLL空头排列": IndicatorEnum.BOLL,
    "BOLL下轨支撑反弹": IndicatorEnum.BOLL,
    "BOLL上轨压力回落": IndicatorEnum.BOLL,
    
    # 成交量指标形态
    "放量上涨": IndicatorEnum.VOL,
    "放量下跌": IndicatorEnum.VOL,
    "缩量上涨": IndicatorEnum.VOL,
    "缩量下跌": IndicatorEnum.VOL,
    "量价背离": IndicatorEnum.VOL,
    "量价共振": IndicatorEnum.VOL,
    "成交量逐渐萎缩": IndicatorEnum.VOL,
    "成交量逐渐放大": IndicatorEnum.VOL,
    "成交量创新高": IndicatorEnum.VOL,
    "成交量突然放大": IndicatorEnum.VOL,
    
    # K线形态
    "长上影线": IndicatorEnum.CUSTOM,
    "长下影线": IndicatorEnum.CUSTOM,
    "十字星": IndicatorEnum.CUSTOM,
    "看涨吞没": IndicatorEnum.CUSTOM,
    "看跌吞没": IndicatorEnum.CUSTOM,
    "锤子线": IndicatorEnum.CUSTOM,
    "倒锤子线": IndicatorEnum.CUSTOM,
    "吊颈线": IndicatorEnum.CUSTOM,
    "启明星": IndicatorEnum.CUSTOM,
    "黄昏星": IndicatorEnum.CUSTOM,
    "刺透形态": IndicatorEnum.CUSTOM,
    "乌云盖顶": IndicatorEnum.CUSTOM,
    "黄包车形态": IndicatorEnum.CUSTOM,
    "突破前期高点": IndicatorEnum.CUSTOM,
    "回踩均线支撑": IndicatorEnum.MA,
    "岛型反转": IndicatorEnum.ISLAND_REVERSAL,
    
    # BIAS指标形态
    "BIAS6超卖回升": IndicatorEnum.BIAS,
    "BIAS12超卖回升": IndicatorEnum.BIAS,
    "BIAS6超买回落": IndicatorEnum.BIAS,
    "BIAS6零轴上穿": IndicatorEnum.BIAS,
    "BIAS12零轴上穿": IndicatorEnum.BIAS,
    "BIAS多头排列": IndicatorEnum.BIAS,
    
    # SAR指标形态
    "SAR由空转多": IndicatorEnum.SAR,
    "SAR多头确认": IndicatorEnum.SAR,
    
    # OBV指标形态
    "OBV持续上升": IndicatorEnum.OBV,
    "OBV底背离": IndicatorEnum.OBV,
    "OBV上穿均线": IndicatorEnum.OBV,
    
    # DMI指标形态
    "DMI金叉": IndicatorEnum.DMI,
    "DMI多头趋势": IndicatorEnum.DMI,
    "ADX持续上升": IndicatorEnum.DMI,
    
    # WR指标形态
    "WR6超卖反弹": IndicatorEnum.WR,
    "WR14超卖反弹": IndicatorEnum.WR,
    "WR6上穿中轴线": IndicatorEnum.WR,
    "WR14上穿中轴线": IndicatorEnum.WR,
    "WR多头排列": IndicatorEnum.WR,
    
    # CCI指标形态
    "CCI14超卖反弹": IndicatorEnum.CCI,
    "CCI20超卖反弹": IndicatorEnum.CCI,
    "CCI14零轴上穿": IndicatorEnum.CCI,
    "CCI20零轴上穿": IndicatorEnum.CCI,
    "CCI14强势区间": IndicatorEnum.CCI,
    "CCI14超强信号": IndicatorEnum.CCI,
    
    # ROC指标形态
    "ROC6金叉": IndicatorEnum.ROC,
    "ROC12金叉": IndicatorEnum.ROC,
    "ROC6零轴上穿": IndicatorEnum.ROC,
    "ROC12零轴上穿": IndicatorEnum.ROC,
    "ROC多头排列": IndicatorEnum.ROC,
    "ROC6超卖反弹": IndicatorEnum.ROC,
    "ROC12超卖反弹": IndicatorEnum.ROC,
    "ROC6加速上涨": IndicatorEnum.ROC,
    
    # VOSC指标形态
    "VOSC零轴上穿": IndicatorEnum.VOSC,
    "VOSC金叉": IndicatorEnum.VOSC,
    "VOSC快速上升": IndicatorEnum.VOSC,
    "VOSC底背离": IndicatorEnum.VOSC,
    
    # MFI指标形态
    "MFI超卖区域": IndicatorEnum.MFI,
    "MFI超卖反弹": IndicatorEnum.MFI,
    "MFI上穿中轴线": IndicatorEnum.MFI,
    "MFI底背离": IndicatorEnum.MFI,
    "MFI钩子买入信号": IndicatorEnum.MFI,
    "MFI底部失败摆动": IndicatorEnum.MFI,
    
    # STOCHRSI指标形态
    "STOCHRSI金叉": IndicatorEnum.STOCHRSI,
    "STOCHRSI死叉": IndicatorEnum.STOCHRSI,
    "STOCHRSI超卖反弹": IndicatorEnum.STOCHRSI,
    "STOCHRSI超买回落": IndicatorEnum.STOCHRSI,
    "STOCHRSI超卖区域": IndicatorEnum.STOCHRSI,
    "STOCHRSI超买区域": IndicatorEnum.STOCHRSI,
    "STOCHRSI_K上穿中轴线": IndicatorEnum.STOCHRSI,
    "STOCHRSI_K下穿中轴线": IndicatorEnum.STOCHRSI,
    "STOCHRSI底背离": IndicatorEnum.STOCHRSI,
    "STOCHRSI顶背离": IndicatorEnum.STOCHRSI,
    
    # MOMENTUM指标形态
    "MTM金叉": IndicatorEnum.MOMENTUM,
    "MTM死叉": IndicatorEnum.MOMENTUM,
    "MTM零轴上穿": IndicatorEnum.MOMENTUM,
    "MTM零轴下穿": IndicatorEnum.MOMENTUM,
    "MTM加速上升": IndicatorEnum.MOMENTUM,
    "MTM顶背离": IndicatorEnum.MOMENTUM,
    "MTM底背离": IndicatorEnum.MOMENTUM,
    "MTM爆发性增长": IndicatorEnum.MOMENTUM,
    
    # RSIMA指标形态
    "RSI金叉均线": IndicatorEnum.RSIMA,
    "RSIMA金叉": IndicatorEnum.RSIMA,
    "RSIMA死叉": IndicatorEnum.RSIMA,
    "RSIMA多头排列": IndicatorEnum.RSIMA,
    "RSI上穿50线": IndicatorEnum.RSIMA,
    "RSI下穿50线": IndicatorEnum.RSIMA,
    "RSI超买区域": IndicatorEnum.RSIMA,
    "RSI超卖区域": IndicatorEnum.RSIMA,
    "RSI超卖反弹": IndicatorEnum.RSIMA,
    "RSIMA强势趋势": IndicatorEnum.RSIMA,
    
    # INTRADAY_VOLATILITY指标形态
    "波动率突然上升": IndicatorEnum.INTRADAY_VOLATILITY,
    "波动率突然下降": IndicatorEnum.INTRADAY_VOLATILITY,
    "低波动率区域": IndicatorEnum.INTRADAY_VOLATILITY,
    "高波动率区域": IndicatorEnum.INTRADAY_VOLATILITY,
    "波动率持续降低": IndicatorEnum.INTRADAY_VOLATILITY,
    "波动率极低后上升": IndicatorEnum.INTRADAY_VOLATILITY,
    "波动率筑底": IndicatorEnum.INTRADAY_VOLATILITY,
    "波动率盘整突破": IndicatorEnum.INTRADAY_VOLATILITY,
    "日内波动率异常": IndicatorEnum.INTRADAY_VOLATILITY,
    "日内波动率上升": IndicatorEnum.INTRADAY_VOLATILITY,
    "日内波动率下降": IndicatorEnum.INTRADAY_VOLATILITY,
    "日内波动率顶背离": IndicatorEnum.INTRADAY_VOLATILITY,
    "日内波动率底背离": IndicatorEnum.INTRADAY_VOLATILITY,
    
    # ATR指标形态
    "ATR异常放大": IndicatorEnum.ATR,
    "ATR异常收缩": IndicatorEnum.ATR,
    "ATR持续上升": IndicatorEnum.ATR,
    "ATR持续下降": IndicatorEnum.ATR,
    "ATR突破上行": IndicatorEnum.ATR,
    "ATR突破下行": IndicatorEnum.ATR,
    "高波动性股票": IndicatorEnum.ATR,
    "低波动性股票": IndicatorEnum.ATR,
    "ATR收缩后爆发": IndicatorEnum.ATR,
    "EMV由负转正": IndicatorEnum.EMV,
    "EMV多头": IndicatorEnum.EMV,
    "EMV持续上升": IndicatorEnum.EMV,
    "价格下跌但EMV上升": IndicatorEnum.EMV,
    "WR超卖反弹": IndicatorEnum.WR,
    "量价背离": IndicatorEnum.VOL,
    "背离": IndicatorEnum.CUSTOM,
    "BOLL中轨突破": IndicatorEnum.BOLL,
    "回踩均线支撑": IndicatorEnum.MA,
    "BIAS超卖反弹": IndicatorEnum.BIAS,
    "SAR由空转多": IndicatorEnum.SAR,
    "DMI金叉": IndicatorEnum.DMI,
    "CCI超卖反弹": IndicatorEnum.CCI,
    "ATR异常放大": IndicatorEnum.ATR,
    "ATR异常收缩": IndicatorEnum.ATR,
    "ATR持续上升": IndicatorEnum.ATR,
    "ATR持续下降": IndicatorEnum.ATR,
    "ATR突破上行": IndicatorEnum.ATR,
    "ATR突破下行": IndicatorEnum.ATR,
    "高波动性股票": IndicatorEnum.ATR,
    "低波动性股票": IndicatorEnum.ATR,
    "ATR收缩后爆发": IndicatorEnum.ATR
}

# 默认参数，使用标准参数映射替代
DEFAULT_PARAMETERS = {
    IndicatorEnum.MA: {
        "periods": [5, 10, 20, 30, 60]
    },
    IndicatorEnum.EMA: {
        "periods": [5, 10, 20, 30, 60]
    },
    IndicatorEnum.WMA: {
        "periods": [5, 10, 20, 30]
    },
    IndicatorEnum.MACD: {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    IndicatorEnum.KDJ: {
        "period": 9,
        "k_period": 3,
        "d_period": 3
    },
    IndicatorEnum.RSI: {
        "periods": [6, 12, 24]
    },
    IndicatorEnum.BOLL: {
        "period": 20,
        "dev_up": 2,
        "dev_down": 2
    },
    IndicatorEnum.VOL: {
        "period": 5
    },
    IndicatorEnum.TRIX: {
        "period": 12,
        "signal_period": 9
    },
    IndicatorEnum.MTM: {
        "period": 12,
        "signal_period": 6
    },
    IndicatorEnum.MOMENTUM: {
        "period": 14,
        "signal_period": 6
    },
    IndicatorEnum.BIAS: {
        "periods": [6, 12, 24]
    },
    IndicatorEnum.SAR: {
        "acceleration": 0.02,
        "maximum": 0.2
    },
    IndicatorEnum.OBV: {
        "ma_period": 30
    },
    IndicatorEnum.DMI: {
        "period": 14
    },
    IndicatorEnum.WR: {
        "periods": [6, 10, 14]
    },
    IndicatorEnum.CCI: {
        "periods": [14, 20]
    },
    IndicatorEnum.ROC: {
        "periods": [6, 12, 24],
        "signal_period": 6
    },
    IndicatorEnum.VOSC: {
        "short_period": 12, 
        "long_period": 26,
        "signal_period": 9
    },
    IndicatorEnum.MFI: {
        "period": 14,
        "overbought": 80,
        "oversold": 20
    },
    IndicatorEnum.STOCHRSI: {
        "rsi_period": 14,
        "stoch_period": 14,
        "k_period": 3,
        "d_period": 3
    },
    IndicatorEnum.RSIMA: {
        "rsi_period": 14,
        "ma_periods": [5, 10, 20]
    },
    IndicatorEnum.INTRADAY_VOLATILITY: {
        "smooth_period": 5
    },
    IndicatorEnum.ATR: {
        "period": 14
    },
    IndicatorEnum.EMV: {
        "period": 14,
        "volume_scale": 10000
    },
    IndicatorEnum.ZXM_ABSORB: {
        "threshold": 0.0
    },
    IndicatorEnum.ZXM_TURNOVER: {
        "threshold": 0.0
    },
    IndicatorEnum.ZXM_DAILY_MACD: {
        "threshold": 0.0
    },
    IndicatorEnum.ZXM_BUYPOINT_SCORE: {
        "threshold": 0.0
    }
}

# 周期映射
PERIOD_MAPPING = {
    "日线": "DAILY",
    "周线": "WEEKLY",
    "月线": "MONTHLY",
    "15分钟": "MIN_15",
    "30分钟": "MIN_30",
    "60分钟": "MIN_60"
}

def extract_indicators_from_backtest(backtest_file):
    """
    从回测结果中提取指标定义
    
    Args:
        backtest_file: 回测结果文件路径
        
    Returns:
        Dict: 指标定义字典
    """
    try:
        # 读取回测结果
        with open(backtest_file, 'r', encoding='utf-8') as f:
            backtest_data = json.load(f)
            
        # 初始化指标字典
        indicators = {
            'ma': {'periods': [5, 10, 20, 30, 60]},
            'ema': {'periods': [5, 10, 20, 30, 60]},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'kdj': {'period': 9, 'k_period': 3, 'd_period': 3},
            'rsi': {'periods': [6, 12, 24]},
            'boll': {'period': 20, 'std_dev': 2},
            'vol': {'periods': [5, 10]},
            'bias': {'periods': [6, 12, 24]},
            'sar': {'acceleration': 0.02, 'maximum': 0.2},
            'obv': {'period': 30},
            'dmi': {'period': 14},
            'wr': {'periods': [6, 14]},
            'cci': {'periods': [14, 20]},
            'roc': {'periods': [6, 12, 24], 'signal_period': 6},
            'vosc': {'short_period': 12, 'long_period': 26, 'signal_period': 9},
            'mfi': {'period': 14},
            'stochrsi': {'rsi_period': 14, 'stoch_period': 14, 'k_period': 3, 'd_period': 3},
            'momentum': {'period': 10, 'signal_period': 5},
            'rsima': {'rsi_period': 14, 'ma_periods': [5, 10, 20]},
            'intraday_volatility': {'period': 14, 'smooth_period': 5},
            'atr': {'period': 14},
            'emv': {'period': 14, 'volume_scale': 10000},
            # ZXM指标
            'zxm_elasticity': {},
            'zxm_buypoint': {},
            'zxm_trend': {}
        }
        
        # 分析回测结果中的形态
        patterns = []
        stocks_count = 0
        
        # 提取所有形态
        for stock_data in backtest_data:
            stocks_count += 1
            if 'patterns' in stock_data:
                patterns.extend(stock_data['patterns'])
                
        # 统计形态出现次数
        pattern_counts = {}
        for pattern in patterns:
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            else:
                pattern_counts[pattern] = 1
                
        # 排序形态，按出现次数降序
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 添加到指标定义
        indicators['patterns'] = [p[0] for p in sorted_patterns[:20]]  # 只取前20个最常见的形态
        indicators['pattern_counts'] = pattern_counts
        indicators['stocks_count'] = stocks_count
        
        return indicators
        
    except Exception as e:
        logger.error(f"从回测结果中提取指标定义失败: {e}")
        return {}

def extract_parameters_from_source(source_file):
    """从源策略文件中提取指标参数"""
    logger.info(f"从源策略文件中提取指标参数: {source_file}")
    
    parameters = {}
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取各种指标的参数
        # ZXM吸筹
        zxm_absorb_match = re.search(r'ZXM吸筹\([^\)]+\)', content)
        if zxm_absorb_match:
            param_match = re.search(r'threshold=(\d+(?:\.\d+)?)', zxm_absorb_match.group(0))
            if param_match:
                parameters[IndicatorEnum.ZXM_ABSORB] = {
                    'threshold': float(param_match.group(1))
                }
        
        # KDJ参数
        kdj_match = re.search(r'KDJ\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if kdj_match:
            parameters[IndicatorEnum.KDJ] = {
                'n': int(kdj_match.group(1)),
                'm1': int(kdj_match.group(2)),
                'm2': int(kdj_match.group(3)),
                'signal_type': 'GOLDEN_CROSS'
            }
        
        # MACD参数
        macd_match = re.search(r'MACD\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if macd_match:
            parameters[IndicatorEnum.DIVERGENCE] = {
                'indicator_type': 'MACD',
                'divergence_type': 'BULLISH',
                'fast_period': int(macd_match.group(1)),
                'slow_period': int(macd_match.group(2)),
                'signal_period': int(macd_match.group(3))
            }
        
        # MA参数
        ma_match = re.search(r'MA\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if ma_match:
            parameters[IndicatorEnum.MA] = {
                'periods': [int(ma_match.group(1)), int(ma_match.group(2)), int(ma_match.group(3))],
                'is_bullish': True
            }
        
        # 其他参数的提取...
        
        logger.info(f"从源文件提取到 {len(parameters)} 个指标的参数")
        
    except Exception as e:
        logger.error(f"提取参数时出错: {e}")
    
    return parameters

def create_strategy_from_backtest(backtest_data, parameters, output_file):
    """
    根据回测结果创建策略文件
    
    Args:
        backtest_data: 回测结果数据
        parameters: 指标参数字典
        output_file: 输出文件路径
    """
    try:
        # 从回测结果中提取形态和频率
        patterns = []
        pattern_counts = {}
        
        for stock_data in backtest_data:
            if 'patterns' in stock_data:
                for pattern in stock_data['patterns']:
                    if pattern in pattern_counts:
                        pattern_counts[pattern] += 1
                    else:
                        pattern_counts[pattern] = 1
                        patterns.append(pattern)
        
        # 按频率排序形态
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        top_patterns = [p[0] for p in sorted_patterns[:10]]  # 取前10个最常见的形态
        
        # 统计所有股票数量
        stocks_count = len(backtest_data)
        
        # 创建策略头部
        strategy_code = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
基于回测系统自动生成的策略
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
回测分析股票数量: {stocks_count}

最常见的形态:
{', '.join(top_patterns)}
'''

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

"""
        
        # 添加通用函数
        strategy_code += """
# 通用函数定义
def MA(close, period):
    """计算移动平均线"""
    return pd.Series(close).rolling(period).mean().values

def EMA(close, period):
    """计算指数移动平均线"""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def SMA(series, n, m):
    """计算平滑移动平均"""
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = (m * series[i] + (n - m) * result[i-1]) / n
    return result

def REF(series, n):
    """引用N个周期前的数据"""
    if n <= 0:
        return series
    series_pd = pd.Series(series)
    return series_pd.shift(n).values

def HHV(series, n):
    """N个周期内的最高值"""
    return pd.Series(series).rolling(n).max().values

def LLV(series, n):
    """N个周期内的最低值"""
    return pd.Series(series).rolling(n).min().values

def CROSS(series1, series2):
    """判断series1是否上穿series2"""
    cond1 = series1 > series2
    cond2 = REF(series1, 1) <= REF(series2, 1)
    return cond1 & cond2

def MACD(close, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = EMA(close, fast)
    ema_slow = EMA(close, slow)
    dif = ema_fast - ema_slow
    dea = EMA(dif, signal)
    macd = (dif - dea) * 2
    return dif, dea, macd

def KDJ(close, high, low, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    high_n = pd.Series(high).rolling(n).max()
    low_n = pd.Series(low).rolling(n).min()
    rsv = (pd.Series(close) - low_n) / (high_n - low_n) * 100
    
    k = pd.Series(rsv).ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k.values, d.values, j.values

def RSI(close, period=14):
    """计算RSI指标"""
    diff = pd.Series(close).diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi.values

def BOLL(close, period=20, dev=2):
    """计算BOLL指标"""
    middle = pd.Series(close).rolling(period).mean()
    std = pd.Series(close).rolling(period).std()
    
    upper = middle + std * dev
    lower = middle - std * dev
    
    return upper.values, middle.values, lower.values

def BIAS(close, period=6):
    """计算BIAS指标"""
    ma = MA(close, period)
    bias = (close - ma) / ma * 100
    return bias

def WR(high, low, close, period=14):
    """计算威廉指标"""
    highest = pd.Series(high).rolling(period).max()
    lowest = pd.Series(low).rolling(period).min()
    wr = (highest - close) / (highest - lowest) * 100
    return wr.values

def CCI(high, low, close, period=14):
    """计算CCI顺势指标"""
    tp = (high + low + close) / 3
    ma_tp = pd.Series(tp).rolling(period).mean()
    md_tp = pd.Series(tp).rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - ma_tp) / (0.015 * md_tp)
    return cci.values

def ATR(high, low, close, period=14):
    """计算ATR指标"""
    tr1 = high - low
    tr2 = np.abs(high - REF(close, 1))
    tr3 = np.abs(low - REF(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

def EMV(high, low, volume, period=14, volume_scale=10000):
    """计算EMV指标"""
    midpoint = (high + low) / 2
    midpoint_move = np.zeros_like(midpoint)
    midpoint_move[1:] = midpoint[1:] - midpoint[:-1]
    
    price_range = high - low
    price_range = np.where(price_range == 0, 0.001, price_range)
    
    box_ratio = volume / price_range / volume_scale
    box_ratio = np.where(box_ratio == 0, 0.001, box_ratio)
    
    daily_emv = midpoint_move / box_ratio
    emv = pd.Series(daily_emv).rolling(window=period).mean().values
    
    return emv
"""
        
        # 添加买入信号函数
        strategy_code += """
def is_buy_signal(data):
    """
    买入信号判断
    
    Args:
        data: 股票数据，包含OHLCV数据和计算的指标
        
    Returns:
        bool: 是否为买入信号
    """
    # 基本条件检查
    if len(data['close']) < 60:
        return False
        
    # 获取数据
    close = data['close']
    open_price = data['open']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # 计算指标
    ma5 = MA(close, 5)
    ma10 = MA(close, 10)
    ma20 = MA(close, 20)
    ma30 = MA(close, 30)
    ma60 = MA(close, 60)
    
    # MACD指标
    dif, dea, macd = MACD(close, fast=12, slow=26, signal=9)
    
    # KDJ指标
    k, d, j = KDJ(close, high, low, n=9, m1=3, m2=3)
    
    # RSI指标
    rsi6 = RSI(close, period=6)
    rsi12 = RSI(close, period=12)
    
    # BOLL指标
    upper, middle, lower = BOLL(close, period=20, dev=2)
    
    # BIAS指标
    bias6 = BIAS(close, period=6)
    bias12 = BIAS(close, period=12)
    
    # ATR指标
    atr14 = ATR(high, low, close, period=14)
    
    # EMV指标
    emv14 = EMV(high, low, volume, period=14, volume_scale=10000)
    
    # 构建信号条件列表
    signal_conditions = []
    
    # 基于回测结果添加条件
"""
        
        # 添加条件定义，基于回测结果中的常见形态
        for pattern, count in sorted_patterns[:20]:  # 使用前20个最常见的形态
            frequency = count / stocks_count * 100 if stocks_count > 0 else 0
            
            # 只添加频率超过10%的形态
            if frequency < 10:
                continue
                
            strategy_code += f"    # {pattern} (出现频率: {frequency:.2f}%)\n"
            
            if "均线多头排列" in pattern:
                strategy_code += "    signal_conditions.append(ma5[-1] > ma10[-1] > ma20[-1])\n"
            elif "MACD金叉" in pattern:
                strategy_code += "    signal_conditions.append(dif[-1] > dea[-1] and dif[-2] < dea[-2])\n"
            elif "MACD由负转正" in pattern:
                strategy_code += "    signal_conditions.append(macd[-1] > 0 and macd[-2] < 0)\n"
            elif "KDJ金叉" in pattern:
                strategy_code += "    signal_conditions.append(k[-1] > d[-1] and k[-2] < d[-2])\n"
            elif "RSI超卖反弹" in pattern:
                strategy_code += "    signal_conditions.append(rsi6[-1] > rsi6[-2] and rsi6[-2] < 30)\n"
            elif "RSI金叉" in pattern:
                strategy_code += "    signal_conditions.append(rsi6[-1] > rsi12[-1] and rsi6[-2] < rsi12[-2])\n"
            elif "BOLL下轨支撑反弹" in pattern:
                strategy_code += "    signal_conditions.append(close[-2] < lower[-2] and close[-1] > lower[-1])\n"
            elif "BOLL中轨突破" in pattern:
                strategy_code += "    signal_conditions.append(close[-2] < middle[-2] and close[-1] > middle[-1])\n"
            elif "回踩均线支撑" in pattern:
                strategy_code += "    signal_conditions.append(low[-1] <= ma20[-1] <= close[-1])\n"
            elif "BIAS超卖反弹" in pattern:
                strategy_code += "    signal_conditions.append(bias6[-1] > bias6[-2] and bias6[-2] < -6)\n"
            elif "SAR由空转多" in pattern:
                strategy_code += "    # SAR指标需要特殊计算，这里简化处理\n"
                strategy_code += "    signal_conditions.append(close[-1] > ma10[-1] and close[-2] < ma10[-2])\n"
            elif "DMI金叉" in pattern:
                strategy_code += "    # DMI指标需要特殊计算，这里简化处理\n"
                strategy_code += "    signal_conditions.append(close[-1] > ma20[-1] and close[-3] < ma20[-3])\n"
            elif "CCI超卖反弹" in pattern:
                strategy_code += "    # CCI指标需要特殊计算，这里简化处理\n"
                strategy_code += "    signal_conditions.append(close[-1] > close[-2] > close[-3] and close[-4] > close[-3])\n"
            elif "EMV由负转正" in pattern:
                strategy_code += "    signal_conditions.append(emv14[-1] > 0 and emv14[-2] <= 0)\n"
            elif "EMV多头" in pattern:
                strategy_code += "    signal_conditions.append(emv14[-1] > 0 and emv14[-1] > emv14[-2])\n"
            elif "EMV持续上升" in pattern:
                strategy_code += "    signal_conditions.append(emv14[-1] > emv14[-2] > emv14[-3])\n"
            elif "价格下跌但EMV上升" in pattern:
                strategy_code += "    signal_conditions.append(close[-1] < close[-2] and emv14[-1] > emv14[-2])\n"
            elif "WR超卖反弹" in pattern:
                strategy_code += "    # WR指标需要特殊计算，这里简化处理\n"
                strategy_code += "    signal_conditions.append(close[-1] > close[-2] and close[-3] < close[-4])\n"
            elif "量价背离" in pattern or "背离" in pattern:
                strategy_code += "    # 量价背离需要复杂计算，这里简化处理\n"
                strategy_code += "    signal_conditions.append(volume[-1] < volume[-2] and close[-1] > close[-2])\n"
            elif "放量上涨" in pattern:
                strategy_code += "    signal_conditions.append(close[-1] > close[-2] and volume[-1] > volume[-2] * 1.5)\n"
            elif "缩量上涨" in pattern:
                strategy_code += "    signal_conditions.append(close[-1] > close[-2] and volume[-1] < volume[-2] * 0.8)\n"
            elif "突破前期高点" in pattern:
                strategy_code += "    period_high = max(high[-20:-1]) if len(high) >= 20 else max(high[:-1])\n"
                strategy_code += "    signal_conditions.append(close[-1] > period_high)\n"
            elif "ATR异常放大" in pattern:
                strategy_code += "    avg_atr = np.mean(atr14[-5:]) if len(atr14) >= 5 else atr14[-1]\n"
                strategy_code += "    signal_conditions.append(atr14[-1] > avg_atr * 1.5)\n"
            else:
                # 其他无法直接转换的形态，添加注释
                strategy_code += f"    # 形态'{pattern}'需要特殊处理，此处略过\n"
        
        # 添加总结部分
        strategy_code += """
    # 确保至少有足够的数据
    if any(np.isnan(ma60)):
        return False
        
    # 计算满足的条件数量
    conditions_met = sum(1 for cond in signal_conditions if cond)
    
    # 如果满足条件数量大于阈值，返回买入信号
    threshold = max(1, len(signal_conditions) // 3)  # 至少满足1/3的条件
    return conditions_met >= threshold

# 策略测试
if __name__ == "__main__":
    # 示例数据
    example_data = {
        'open': np.random.rand(100) * 10 + 50,
        'high': np.random.rand(100) * 10 + 55,
        'low': np.random.rand(100) * 10 + 45,
        'close': np.random.rand(100) * 10 + 50,
        'volume': np.random.rand(100) * 10000 + 10000
    }
    
    # 测试买入信号
    is_buy = is_buy_signal(example_data)
    print(f"买入信号: {is_buy}")
"""
        
        # 保存策略文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(strategy_code)
            
        logger.info(f"策略文件已生成: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"生成策略文件失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将回测结果转换为选股策略")
    parser.add_argument("-b", "--backtest", required=True, help="回测结果文件路径")
    parser.add_argument("-s", "--source", help="源策略文件路径，用于提取参数")
    parser.add_argument("-o", "--output", help="输出策略配置文件路径")
    parser.add_argument("-i", "--indicator_ids", help="指标ID列表，用逗号分隔，直接指定要使用的指标")
    
    args = parser.parse_args()
    
    # 初始化日志
    init_logging(level="INFO")
    
    # 默认输出文件路径
    if not args.output:
        output_file = os.path.join(
            get_result_dir(), 
            f"backtest_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}.yaml"
        )
    else:
        output_file = args.output
    
    # 如果直接指定了指标ID列表，使用这些ID创建策略
    if args.indicator_ids:
        indicator_ids = [id.strip() for id in args.indicator_ids.split(',')]
        logger.info(f"使用指定的指标ID列表: {indicator_ids}")
        
        # 创建一个模拟的回测数据结构
        backtest_data = {
            'indicator_ids': [{'id': id, 'description': None} for id in indicator_ids],
            'common_patterns': [],
            'cross_period_patterns': [],
            'period_analysis': {}
        }
    else:
        # 提取回测数据
        backtest_data = extract_indicators_from_backtest(args.backtest)
    
    # 提取参数（如果有源文件）
    parameters = {}
    if args.source:
        parameters = extract_parameters_from_source(args.source)
    
    # 创建策略配置
    strategy_config = create_strategy_from_backtest(backtest_data, parameters, output_file)
    
    logger.info(f"策略生成完成，包含 {len(strategy_config['strategy']['conditions'])} 个条件")
    
    # 输出指标统计
    indicator_count = sum(1 for c in strategy_config['strategy']['conditions'] 
                         if isinstance(c, dict) and 'indicator_id' in c)
    zxm_count = sum(1 for c in strategy_config['strategy']['conditions'] 
                   if isinstance(c, dict) and 'indicator_id' in c and 'ZXM' in c.get('indicator_id', ''))
    
    logger.info(f"策略中包含 {indicator_count} 个指标，其中 {zxm_count} 个ZXM系列指标")
    
    # 验证是否包含必要的ZXM指标
    required_zxm = [IndicatorEnum.ZXM_ABSORB, IndicatorEnum.ZXM_TURNOVER, 
                    IndicatorEnum.ZXM_DAILY_MACD, IndicatorEnum.ZXM_BUYPOINT_SCORE]
    missing_zxm = [zxm for zxm in required_zxm if not any(
        isinstance(c, dict) and c.get('indicator_id') == zxm 
        for c in strategy_config['strategy']['conditions']
    )]
    
    if missing_zxm:
        logger.warning(f"策略缺少以下重要的ZXM指标: {', '.join(missing_zxm)}")
    else:
        logger.info("策略包含所有必要的ZXM指标")

if __name__ == "__main__":
    main() 