"""
回踩反弹买点策略模块

实现回踩均线反弹买点的选股策略
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
from indicators.kdj import KDJ

logger = get_logger(__name__)


class ReboundStrategy(BaseStrategy):
    """
    回踩反弹买点策略
    
    识别回踩均线然后反弹上行的买点形态
    """
    
    def __init__(self, name: str = "回踩反弹", description: str = "回踩均线反弹买点策略"):
        """
        初始化回踩反弹买点策略
        
        Args:
            name: 策略名称
            description: 策略描述
        """
        super().__init__(name, description)
        
        # 设置默认参数
        self._parameters = {
            'ma_period': 5,          # 均线周期
            'touch_threshold': 0.02, # 接触均线阈值，如0.02表示2%以内都算接触
            'bounce_min_pct': 0.01,  # 反弹最小涨幅百分比
            'volume_min_ratio': 0.8, # 成交量最小放大比例
            'kdj_up': True,          # 是否要求KDJ上移
            'lookback_days': 5,      # 回溯天数
            'rsi_bottom': 30,        # RSI超卖区域阈值
            'min_distance': 0.05     # 股价距离均线最小距离要求
        }
    
    def select(self, universe: List[str], *args, **kwargs) -> pd.DataFrame:
        """
        执行回踩反弹选股策略
        
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
        ma_period = self._parameters['ma_period']
        touch_threshold = self._parameters['touch_threshold']
        bounce_min_pct = self._parameters['bounce_min_pct']
        volume_min_ratio = self._parameters['volume_min_ratio']
        kdj_up = self._parameters['kdj_up']
        lookback_days = self._parameters['lookback_days']
        rsi_bottom = self._parameters['rsi_bottom']
        min_distance = self._parameters['min_distance']
        
        # 创建技术指标实例
        ma_indicator = IndicatorFactory.create_indicator("MA", periods=[ma_period])
        rsi_indicator = IndicatorFactory.create_indicator("RSI", periods=[6])
        
        for code in universe:
            try:
                # 初始化公式计算对象
                f = formula.Formula(code)
                
                if not f.dataDay.close.any():
                    logger.debug(f"股票 {code} 数据为空，跳过")
                    continue
                
                # 获取收盘价、最低价和成交量
                close = f.dataDay.close
                low = f.dataDay.low
                volume = f.dataDay.volume
                
                # 至少需要60个交易日的数据
                if len(close) < 60:
                    logger.debug(f"股票 {code} 数据不足60个交易日，跳过")
                    continue
                
                # 准备数据DataFrame
                data = pd.DataFrame({
                    'open': f.dataDay.open,
                    'high': f.dataDay.high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
                
                # 计算均线
                ma_result = ma_indicator.compute(data)
                ma_values = ma_result[f'MA{ma_period}'].values
                
                # 计算RSI
                if rsi_bottom > 0:
                    rsi_result = rsi_indicator.compute(data)
                    rsi_values = rsi_result['RSI6'].values
                
                # 检查是否符合回踩反弹形态
                is_match = False
                touch_index = -1
                
                # 从最近的K线往前回溯，寻找回踩点
                for i in range(len(close)-1, max(0, len(close)-lookback_days-1), -1):
                    # 已经确认符合形态，无需继续检查
                    if is_match:
                        break
                    
                    # 检查当前点是否高于均线
                    if close[i] > ma_values[i] * (1 + min_distance):
                        # 往前寻找最近的回踩点
                        for j in range(i-1, max(0, i-lookback_days), -1):
                            # 最低价接触或接近均线
                            if low[j] <= ma_values[j] * (1 + touch_threshold) and low[j] >= ma_values[j] * (1 - touch_threshold):
                                # 确认是反弹走势
                                if close[i] >= close[j] * (1 + bounce_min_pct):
                                    is_match = True
                                    touch_index = j
                                    break
                
                # 检查交易量条件
                volume_ok = True
                if volume_min_ratio > 0 and touch_index > 0:
                    # 反弹点的成交量要大于前一天的成交量*设置的比例
                    if volume[touch_index+1] < volume[touch_index] * volume_min_ratio:
                        volume_ok = False
                
                # 检查KDJ上移条件
                kdj_ok = True
                if kdj_up:
                    # 使用KDJ指标类
                    kdj_indicator = IndicatorFactory.create_indicator("KDJ")
                    kdj_result = kdj_indicator.compute(data)
                    
                    k = kdj_result['K'].values
                    d = kdj_result['D'].values
                    
                    if not (k[-1] > k[-2] and d[-1] > d[-2]):
                        kdj_ok = False
                
                # 检查RSI条件
                rsi_ok = True
                if rsi_bottom > 0 and touch_index > 0:
                    # 回踩点的RSI应该接近超卖区域
                    if rsi_values[touch_index] > rsi_bottom:
                        rsi_ok = False
                
                # 所有条件都满足
                if is_match and volume_ok and kdj_ok and rsi_ok:
                    # 添加到结果
                    result_list.append({
                        'code': code,
                        'name': f.name,
                        'industry': f.industry,
                        'close': close[-1],
                        'ma': ma_values[-1],
                        'touch_index': touch_index,
                        'bounce_pct': (close[-1] / close[touch_index] - 1) * 100 if touch_index >= 0 else 0
                    })
            
            except Exception as e:
                logger.error(f"处理股票 {code} 时出错: {e}")
        
        # 转换为DataFrame并排序
        if result_list:
            result_df = pd.DataFrame(result_list)
            result_df = result_df.sort_values('bounce_pct', ascending=False)
            return result_df
        else:
            return pd.DataFrame(columns=['code', 'name', 'industry', 'close', 'ma', 'touch_index', 'bounce_pct']) 