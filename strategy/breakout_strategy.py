"""
横盘突破买点策略模块

实现横盘整理后向上突破形态的选股策略
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from strategy.base_strategy import BaseStrategy
from formula import formula
from enums.kline_period import KlinePeriod
from utils.logger import get_logger
from indicators.factory import IndicatorFactory
from indicators.ma import MA
from indicators.boll import BOLL

logger = get_logger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    横盘突破买点策略
    
    识别横盘整理后向上突破的买点形态
    """
    
    def __init__(self, name: str = "横盘突破", description: str = "横盘整理后向上突破买点策略"):
        """
        初始化横盘突破买点策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self._parameters = {
            'consolidation_days': 10,  # 横盘整理的最短天数
            'price_range_pct': 0.05,   # 横盘整理期间的价格波动范围百分比
            'breakout_pct': 0.03,      # 突破确认的最小涨幅百分比
            'volume_ratio': 1.5,       # 突破时成交量放大的最小倍数
            'macd_up': True,           # 是否要求MACD上移
            'ma_support': True,        # 是否要求均线支撑
            'boll_use': True,          # 是否使用布林带确认
            'lookback_days': 20,       # 回溯天数
            'min_price': 5             # 最低股价要求
        }
    
    def select(self, universe: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        执行横盘突破选股策略
        
        Args:
            universe: 股票代码列表，表示选股范围
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 选股结果，包含股票代码、名称等信息
        """
        # 更新参数
        if kwargs:
            self.set_parameters(kwargs)
        
        result_list = []
        
        # 获取参数
        consolidation_days = self._parameters['consolidation_days']
        price_range_pct = self._parameters['price_range_pct']
        breakout_pct = self._parameters['breakout_pct']
        volume_ratio = self._parameters['volume_ratio']
        macd_up = self._parameters['macd_up']
        ma_support = self._parameters['ma_support']
        boll_use = self._parameters['boll_use']
        lookback_days = self._parameters['lookback_days']
        min_price = self._parameters['min_price']
        
        # 创建技术指标实例
        ma_indicator = IndicatorFactory.create_indicator("MA", periods=[20, 60])
        
        for code in universe:
            try:
                # 初始化公式计算对象
                f = formula.Formula(code)
                
                if not f.dataDay.close.any():
                    logger.debug(f"股票 {code} 数据为空，跳过")
                    continue
                
                # 获取股票数据
                close = f.dataDay.close
                high = f.dataDay.high
                low = f.dataDay.low
                volume = f.dataDay.volume
                
                # 至少需要60个交易日的数据
                if len(close) < 60:
                    logger.debug(f"股票 {code} 数据不足60个交易日，跳过")
                    continue
                
                # 检查最低价格要求
                if close[-1] < min_price:
                    continue
                
                # 准备数据DataFrame
                data = pd.DataFrame({
                    'open': f.dataDay.open,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
                
                # 计算移动平均线
                ma_result = ma_indicator.compute(data)
                ma20 = ma_result['MA20'].values
                ma60 = ma_result['MA60'].values
                
                # 计算布林带
                if boll_use:
                    boll_indicator = IndicatorFactory.create_indicator("BOLL")
                    boll_result = boll_indicator.compute(data)
                    upper = boll_result['upper'].values
                    middle = boll_result['middle'].values
                    lower = boll_result['lower'].values
                
                # 检查是否符合横盘突破形态
                is_match = False
                breakout_index = -1
                consolidation_start = -1
                
                # 从最近的K线往前回溯，寻找突破点
                for i in range(len(close)-1, max(0, len(close)-lookback_days-1), -1):
                    # 已经确认符合形态，无需继续检查
                    if is_match:
                        break
                    
                    # 检查是否是突破点
                    if i > consolidation_days and close[i] > close[i-1] * (1 + breakout_pct):
                        # 检查成交量是否放大
                        volume_ok = volume[i] > volume[i-1] * volume_ratio
                        
                        if not volume_ok:
                            continue
                        
                        # 检查前期是否有足够长的横盘整理期
                        # 计算前N天的价格波动范围
                        price_high = max(high[i-consolidation_days:i])
                        price_low = min(low[i-consolidation_days:i])
                        price_range = (price_high - price_low) / price_low
                        
                        # 判断价格波动是否在指定范围内
                        if price_range <= price_range_pct:
                            is_match = True
                            breakout_index = i
                            consolidation_start = i - consolidation_days
                    
                # 检查MACD条件
                macd_ok = True
                if macd_up and breakout_index > 0:
                    # 使用MACD指标类
                    macd_indicator = IndicatorFactory.create_indicator("MACD")
                    macd_result = macd_indicator.compute(data)
                    
                    diff = macd_result['DIF'].values
                    dea = macd_result['DEA'].values
                    
                    # MACD要求上移
                    if not (diff[breakout_index] > diff[breakout_index-1] and 
                            dea[breakout_index] > dea[breakout_index-1]):
                        macd_ok = False
                
                # 检查均线支撑条件
                ma_ok = True
                if ma_support and breakout_index > 0:
                    # 突破点上方有均线支撑
                    if not (close[breakout_index] > ma20[breakout_index] and 
                            ma20[breakout_index] > ma20[breakout_index-1]):
                        ma_ok = False
                
                # 检查布林带条件
                boll_ok = True
                if boll_use and breakout_index > 0:
                    # 股价突破布林带上轨
                    if not close[breakout_index] > upper[breakout_index-1]:
                        boll_ok = False
                
                # 所有条件都满足
                if is_match and macd_ok and ma_ok and boll_ok:
                    # 添加到结果
                    result_list.append({
                        'code': code,
                        'name': f.name,
                        'industry': f.industry,
                        'close': close[-1],
                        'breakout_price': close[breakout_index],
                        'consolidation_start': consolidation_start,
                        'breakout_index': breakout_index,
                        'price_change_pct': (close[-1] / close[breakout_index] - 1) * 100
                    })
            
            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")
        
        # 转换为DataFrame并排序
        if result_list:
            result_df = pd.DataFrame(result_list)
            result_df = result_df.sort_values('price_change_pct', ascending=False)
            return result_df
        else:
            return pd.DataFrame(columns=['code', 'name', 'industry', 'close', 'breakout_price', 
                                        'consolidation_start', 'breakout_index', 'price_change_pct'])
    
    def _is_consolidating(self, high, low, consolidation_days, price_range_pct):
        """判断是否处于横盘整理状态"""
        if len(high) < consolidation_days or len(low) < consolidation_days:
            return False
            
        # 计算价格波动范围
        highest_price = max(high[-consolidation_days:])
        lowest_price = min(low[-consolidation_days:])
        
        price_range = (highest_price - lowest_price) / lowest_price
        
        return price_range <= price_range_pct
    
    def _is_breaking_out(self, close, consolidation_days, breakout_pct):
        """判断是否突破"""
        if len(close) < consolidation_days + 1:
            return False
            
        # 获取整理期间的最高价
        consolidation_high = max(close[-(consolidation_days+1):-1])
        
        # 当前价格突破整理期间最高价
        return close[-1] > consolidation_high * (1 + breakout_pct) 