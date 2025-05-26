"""
ZXM体系通用选股模型模块

整合ZXM体系的所有指标，提供完整的选股功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple

from indicators.base_indicator import BaseIndicator
from indicators.zxm.trend_indicators import (
    ZXMDailyTrendUp, ZXMWeeklyTrendUp, ZXMMonthlyKDJTrendUp, 
    ZXMWeeklyKDJDOrDEATrendUp, ZXMWeeklyKDJDTrendUp,
    ZXMMonthlyMACD, ZXMWeeklyMACD
)
from indicators.zxm.elasticity_indicators import (
    ZXMAmplitudeElasticity, ZXMRiseElasticity
)
from indicators.zxm.buy_point_indicators import (
    ZXMDailyMACD, ZXMTurnover, ZXMVolumeShrink,
    ZXMMACallback, ZXMBSAbsorb
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ZXMSelectionModel(BaseIndicator):
    """
    ZXM体系通用选股模型
    
    整合ZXM体系的趋势、弹性和买点指标，提供完整的选股功能
    """
    
    def __init__(self, callback_percent: float = 4.0):
        """
        初始化ZXM体系通用选股模型
        
        Args:
            callback_percent: 回踩均线指标的回踩百分比，默认为4%
        """
        super().__init__(name="ZXMSelectionModel", description="ZXM体系通用选股模型，整合趋势、弹性和买点指标")
        
        # 初始化趋势指标
        self.daily_trend_up = ZXMDailyTrendUp()
        self.weekly_trend_up = ZXMWeeklyTrendUp()
        self.monthly_kdj_trend_up = ZXMMonthlyKDJTrendUp()
        self.weekly_kdj_dea_trend_up = ZXMWeeklyKDJDOrDEATrendUp()
        self.weekly_kdj_d_trend_up = ZXMWeeklyKDJDTrendUp()
        self.monthly_macd = ZXMMonthlyMACD()
        self.weekly_macd = ZXMWeeklyMACD()
        
        # 初始化弹性指标
        self.amplitude_elasticity = ZXMAmplitudeElasticity()
        self.rise_elasticity = ZXMRiseElasticity()
        
        # 初始化买点指标
        self.daily_macd = ZXMDailyMACD()
        self.turnover = ZXMTurnover()
        self.volume_shrink = ZXMVolumeShrink()
        self.ma_callback = ZXMMACallback(callback_percent=callback_percent)
        self.bs_absorb = ZXMBSAbsorb()
        
        # 记录各组指标权重
        self.trend_weight = 0.4  # 趋势指标权重
        self.elasticity_weight = 0.2  # 弹性指标权重
        self.buy_point_weight = 0.4  # 买点指标权重
    
    def calculate(self, daily_data: pd.DataFrame, weekly_data: Optional[pd.DataFrame] = None, 
                  monthly_data: Optional[pd.DataFrame] = None, 
                  hourly_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算ZXM体系通用选股模型
        
        Args:
            daily_data: 日线数据，包含OHLCV数据
            weekly_data: 周线数据，如果未提供则尝试从日线数据聚合
            monthly_data: 月线数据，如果未提供则尝试从日线数据聚合
            hourly_data: 60分钟数据，用于BS吸筹指标计算，如果未提供则不计算该指标
            
        Returns:
            pd.DataFrame: 计算结果，包含各指标结果和最终选股得分
        """
        # 确保数据包含必需的列
        self.ensure_columns(daily_data, ["open", "high", "low", "close", "volume"])
        
        # 初始化结果数据框
        result = pd.DataFrame(index=daily_data.index)
        
        # 计算趋势指标（共7个）
        trend_results = self._calculate_trend_indicators(daily_data, weekly_data, monthly_data)
        for key, value in trend_results.items():
            result[key] = value
        
        # 计算弹性指标（共2个）
        elasticity_results = self._calculate_elasticity_indicators(daily_data)
        for key, value in elasticity_results.items():
            result[key] = value
        
        # 计算买点指标（共5个）
        buy_point_results = self._calculate_buy_point_indicators(daily_data, hourly_data)
        for key, value in buy_point_results.items():
            result[key] = value
        
        # 计算各组指标的满足数量
        trend_count = sum(trend_results.values())
        elasticity_count = sum(elasticity_results.values())
        buy_point_count = sum(buy_point_results.values())
        
        # 计算各组指标的满足比例
        trend_ratio = trend_count / len(trend_results) if len(trend_results) > 0 else 0
        elasticity_ratio = elasticity_count / len(elasticity_results) if len(elasticity_results) > 0 else 0
        buy_point_ratio = buy_point_count / len(buy_point_results) if len(buy_point_results) > 0 else 0
        
        # 计算最终得分（0-100）
        score = 100 * (
            self.trend_weight * trend_ratio + 
            self.elasticity_weight * elasticity_ratio + 
            self.buy_point_weight * buy_point_ratio
        )
        
        # 添加统计结果到数据框
        result["趋势指标满足数"] = trend_count
        result["趋势指标总数"] = len(trend_results)
        result["趋势指标得分"] = trend_ratio * 100
        
        result["弹性指标满足数"] = elasticity_count
        result["弹性指标总数"] = len(elasticity_results)
        result["弹性指标得分"] = elasticity_ratio * 100
        
        result["买点指标满足数"] = buy_point_count
        result["买点指标总数"] = len(buy_point_results)
        result["买点指标得分"] = buy_point_ratio * 100
        
        result["ZXM选股总得分"] = score
        
        return result
    
    def _calculate_trend_indicators(self, daily_data: pd.DataFrame, 
                                   weekly_data: Optional[pd.DataFrame], 
                                   monthly_data: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算趋势指标
        
        Args:
            daily_data: 日线数据
            weekly_data: 周线数据
            monthly_data: 月线数据
            
        Returns:
            Dict[str, pd.Series]: 趋势指标结果
        """
        result = {}
        
        # 计算日线上移趋势
        try:
            daily_trend = self.daily_trend_up.calculate(daily_data)
            result["日线上移趋势"] = daily_trend["XG"]
        except Exception as e:
            logger.error(f"计算日线上移趋势出错: {e}")
            result["日线上移趋势"] = pd.Series(False, index=daily_data.index)
        
        # 如果未提供周线数据，则跳过周线指标
        if weekly_data is not None:
            # 计算周线上移趋势
            try:
                weekly_trend = self.weekly_trend_up.calculate(weekly_data)
                # 将周线结果扩展到日线
                result["周线上移趋势"] = self._expand_to_daily(weekly_trend["XG"], daily_data.index)
            except Exception as e:
                logger.error(f"计算周线上移趋势出错: {e}")
                result["周线上移趋势"] = pd.Series(False, index=daily_data.index)
            
            # 计算周KDJ·D或DEA上移趋势
            try:
                weekly_kdj_dea = self.weekly_kdj_dea_trend_up.calculate(weekly_data)
                result["周KDJ·D或DEA上移趋势"] = self._expand_to_daily(weekly_kdj_dea["XG"], daily_data.index)
            except Exception as e:
                logger.error(f"计算周KDJ·D或DEA上移趋势出错: {e}")
                result["周KDJ·D或DEA上移趋势"] = pd.Series(False, index=daily_data.index)
            
            # 计算周KDJ·D上移趋势
            try:
                weekly_kdj_d = self.weekly_kdj_d_trend_up.calculate(weekly_data)
                result["周KDJ·D上移趋势"] = self._expand_to_daily(weekly_kdj_d["XG"], daily_data.index)
            except Exception as e:
                logger.error(f"计算周KDJ·D上移趋势出错: {e}")
                result["周KDJ·D上移趋势"] = pd.Series(False, index=daily_data.index)
            
            # 计算周MACD<2趋势
            try:
                weekly_macd = self.weekly_macd.calculate(weekly_data)
                result["周MACD<2趋势"] = self._expand_to_daily(weekly_macd["XG"], daily_data.index)
            except Exception as e:
                logger.error(f"计算周MACD<2趋势出错: {e}")
                result["周MACD<2趋势"] = pd.Series(False, index=daily_data.index)
        
        # 如果未提供月线数据，则跳过月线指标
        if monthly_data is not None:
            # 计算月KDJ·D及K上移趋势
            try:
                monthly_kdj = self.monthly_kdj_trend_up.calculate(monthly_data)
                result["月KDJ·D及K上移趋势"] = self._expand_to_daily(monthly_kdj["XG"], daily_data.index)
            except Exception as e:
                logger.error(f"计算月KDJ·D及K上移趋势出错: {e}")
                result["月KDJ·D及K上移趋势"] = pd.Series(False, index=daily_data.index)
            
            # 计算月MACD<1.5趋势
            try:
                monthly_macd = self.monthly_macd.calculate(monthly_data)
                result["月MACD<1.5趋势"] = self._expand_to_daily(monthly_macd["XG"], daily_data.index)
            except Exception as e:
                logger.error(f"计算月MACD<1.5趋势出错: {e}")
                result["月MACD<1.5趋势"] = pd.Series(False, index=daily_data.index)
        
        return result
    
    def _calculate_elasticity_indicators(self, daily_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算弹性指标
        
        Args:
            daily_data: 日线数据
            
        Returns:
            Dict[str, pd.Series]: 弹性指标结果
        """
        result = {}
        
        # 计算振幅弹性
        try:
            amplitude = self.amplitude_elasticity.calculate(daily_data)
            result["振幅弹性"] = amplitude["XG"]
        except Exception as e:
            logger.error(f"计算振幅弹性出错: {e}")
            result["振幅弹性"] = pd.Series(False, index=daily_data.index)
        
        # 计算涨幅弹性
        try:
            rise = self.rise_elasticity.calculate(daily_data)
            result["涨幅弹性"] = rise["XG"]
        except Exception as e:
            logger.error(f"计算涨幅弹性出错: {e}")
            result["涨幅弹性"] = pd.Series(False, index=daily_data.index)
        
        return result
    
    def _calculate_buy_point_indicators(self, daily_data: pd.DataFrame, 
                                       hourly_data: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算买点指标
        
        Args:
            daily_data: 日线数据
            hourly_data: 60分钟数据
            
        Returns:
            Dict[str, pd.Series]: 买点指标结果
        """
        result = {}
        
        # 计算日MACD买点
        try:
            daily_macd = self.daily_macd.calculate(daily_data)
            result["日MACD买点"] = daily_macd["XG"]
        except Exception as e:
            logger.error(f"计算日MACD买点出错: {e}")
            result["日MACD买点"] = pd.Series(False, index=daily_data.index)
        
        # 计算换手买点
        try:
            # 检查是否有capital列，如果没有，尝试添加模拟值
            if "capital" not in daily_data.columns:
                logger.warning("数据中没有流通股本(capital)列，使用成交量的100倍作为模拟值")
                daily_data_with_capital = daily_data.copy()
                daily_data_with_capital["capital"] = daily_data["volume"] * 100
                turnover = self.turnover.calculate(daily_data_with_capital)
            else:
                turnover = self.turnover.calculate(daily_data)
            result["换手买点"] = turnover["XG"]
        except Exception as e:
            logger.error(f"计算换手买点出错: {e}")
            result["换手买点"] = pd.Series(False, index=daily_data.index)
        
        # 计算缩量买点
        try:
            volume_shrink = self.volume_shrink.calculate(daily_data)
            result["缩量买点"] = volume_shrink["XG"]
        except Exception as e:
            logger.error(f"计算缩量买点出错: {e}")
            result["缩量买点"] = pd.Series(False, index=daily_data.index)
        
        # 计算回踩均线买点
        try:
            ma_callback = self.ma_callback.calculate(daily_data)
            result["回踩均线买点"] = ma_callback["XG"]
        except Exception as e:
            logger.error(f"计算回踩均线买点出错: {e}")
            result["回踩均线买点"] = pd.Series(False, index=daily_data.index)
        
        # 如果未提供60分钟数据，则跳过BS吸筹买点
        if hourly_data is not None:
            try:
                bs_absorb = self.bs_absorb.calculate(hourly_data)
                # BS吸筹指标结果是计数，需要转换为布尔值
                bs_absorb_signal = bs_absorb["XG"] >= 1
                result["BS吸筹买点"] = self._expand_to_daily(bs_absorb_signal, daily_data.index)
            except Exception as e:
                logger.error(f"计算BS吸筹买点出错: {e}")
                result["BS吸筹买点"] = pd.Series(False, index=daily_data.index)
        
        return result
    
    def _expand_to_daily(self, series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
        """
        将周线或月线数据扩展到日线
        
        Args:
            series: 待扩展的序列
            daily_index: 日线索引
            
        Returns:
            pd.Series: 扩展后的序列
        """
        # 创建一个空的Series，以日线索引为基础
        result = pd.Series(False, index=daily_index)
        
        # 对于series中的每一个日期
        for date in series.index:
            # 找到下一个日期（如果有）
            try:
                next_date = series.index[series.index.get_loc(date) + 1]
            except (IndexError, KeyError):
                # 如果是最后一个日期，使用未来的某个日期
                next_date = date + pd.Timedelta(days=30)
            
            # 为日线索引中位于当前日期和下一个日期之间的所有日期赋值
            mask = (daily_index >= date) & (daily_index < next_date)
            result.loc[mask] = series.loc[date]
        
        return result 