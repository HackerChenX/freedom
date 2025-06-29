"""
统一指标计算引擎

提供统一的技术指标计算接口，作为买点分析和策略选股的共享计算核心
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

from utils.logger import get_logger
from utils.cache import MemoryCache

logger = get_logger(__name__)


class UnifiedIndicatorEngine:
    """
    统一指标计算引擎
    
    提供标准化的技术指标计算接口，支持缓存机制和性能优化
    """
    
    def __init__(self, enable_cache: bool = True):
        """
        初始化统一指标计算引擎
        
        Args:
            enable_cache: 是否启用缓存机制
        """
        self.enable_cache = enable_cache
        self.cache_manager = MemoryCache.get_instance() if enable_cache else None
        self.performance_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'calculation_times': []
        }
        logger.info("统一指标计算引擎初始化完成")
    
    def _get_cache_key(self, stock_code: str, indicator: str, params: Dict[str, Any], 
                      start_date: str, end_date: str) -> str:
        """生成缓存键"""
        params_str = "_".join([f"{k}_{v}" for k, v in sorted(params.items())])
        return f"indicator_{stock_code}_{indicator}_{params_str}_{start_date}_{end_date}"
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据的有效性"""
        if data is None or len(data) == 0:
            return False
        
        required_columns = ['close']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"数据缺少必需列: {col}")
                return False
            
        return True
    
    def calculate_ma(self, data: Union[pd.DataFrame, np.ndarray], period: int, 
                    field: str = 'close') -> Union[pd.Series, np.ndarray]:
        """
        计算移动平均线
        
        Args:
            data: 股票数据DataFrame或numpy数组
            period: 计算周期
            field: 计算字段，默认为收盘价（仅DataFrame时使用）
            
        Returns:
            移动平均线序列
        """
        try:
            # 处理numpy数组输入
            if isinstance(data, np.ndarray):
                if len(data) == 0:
                    return np.array([])
                
                result = np.zeros_like(data, dtype=float)
                for i in range(len(data)):
                    start_idx = max(0, i - period + 1)
                    result[i] = np.mean(data[start_idx:i+1])
                
                self.performance_stats['total_calculations'] += 1
                return result
            
            # 处理DataFrame输入
            if not self._validate_data(data):
                return pd.Series(dtype=float)
            
            if field not in data.columns:
                logger.error(f"数据中不存在字段: {field}")
                return pd.Series(dtype=float)
            
            result = data[field].rolling(window=period, min_periods=1).mean()
            self.performance_stats['total_calculations'] += 1
            return result
            
        except Exception as e:
            logger.error(f"计算MA指标失败: {e}")
            if isinstance(data, np.ndarray):
                return np.array([])
            else:
                return pd.Series(dtype=float)
    
    def calculate_ema(self, data: Union[pd.DataFrame, np.ndarray], period: int, 
                     field: str = 'close') -> Union[pd.Series, np.ndarray]:
        """
        计算指数移动平均线
        
        Args:
            data: 股票数据DataFrame或numpy数组
            period: 计算周期
            field: 计算字段，默认为收盘价（仅DataFrame时使用）
            
        Returns:
            指数移动平均线序列
        """
        try:
            # 处理numpy数组输入
            if isinstance(data, np.ndarray):
                if len(data) == 0:
                    return np.array([])
                
                alpha = 2.0 / (period + 1)
                result = np.zeros_like(data, dtype=float)
                result[0] = data[0]
                
                for i in range(1, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                
                self.performance_stats['total_calculations'] += 1
                return result
            
            # 处理DataFrame输入
            if not self._validate_data(data):
                return pd.Series(dtype=float)
            
            if field not in data.columns:
                logger.error(f"数据中不存在字段: {field}")
                return pd.Series(dtype=float)
            
            result = data[field].ewm(span=period, adjust=False).mean()
            self.performance_stats['total_calculations'] += 1
            return result
            
        except Exception as e:
            logger.error(f"计算EMA指标失败: {e}")
            if isinstance(data, np.ndarray):
                return np.array([])
            else:
                return pd.Series(dtype=float)
    
    def calculate_sma(self, series: pd.Series, n: int, m: int) -> pd.Series:
        """
        计算平滑移动平均
        
        Args:
            series: 输入序列
            n: 周期
            m: 权重
            
        Returns:
            平滑移动平均序列
        """
        try:
            result = pd.Series(index=series.index, dtype=float)
            result.iloc[0] = series.iloc[0]
            
            for i in range(1, len(series)):
                result.iloc[i] = (m * series.iloc[i] + (n - m) * result.iloc[i-1]) / n
                
            return result
        except Exception as e:
            logger.error(f"计算SMA指标失败: {e}")
            return pd.Series(dtype=float)
    
    def calculate_macd(self, data: Union[pd.DataFrame, np.ndarray], fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        计算MACD指标
        
        Args:
            data: 股票数据DataFrame或numpy数组
            fast: 快线周期，默认12
            slow: 慢线周期，默认26
            signal: 信号线周期，默认9
            
        Returns:
            包含DIF、DEA、MACD的字典
        """
        try:
            # 处理numpy数组输入
            if isinstance(data, np.ndarray):
                if len(data) == 0:
                    return {'dif': np.array([]), 'dea': np.array([]), 'macd': np.array([])}
                
                # 计算EMA
                ema_fast = self.calculate_ema(data, fast)
                ema_slow = self.calculate_ema(data, slow)
                
                # 计算DIF (快线-慢线)
                dif = ema_fast - ema_slow
                
                # 计算DEA (DIF的EMA)
                dea = self.calculate_ema(dif, signal)
                
                # 计算MACD柱状图
                macd = (dif - dea) * 2
                
                return {'dif': dif, 'dea': dea, 'macd': macd}
            
            # 处理DataFrame输入
            if not self._validate_data(data):
                return {'DIF': pd.Series(dtype=float), 'DEA': pd.Series(dtype=float), 'MACD': pd.Series(dtype=float)}
            
            close = data['close']
            
            # 计算EMA
            ema_fast = self.calculate_ema(data, fast)
            ema_slow = self.calculate_ema(data, slow)
            
            # 计算DIF (快线-慢线)
            dif = ema_fast - ema_slow
            
            # 计算DEA (DIF的EMA)
            dea = dif.ewm(span=signal, adjust=False).mean()
            
            # 计算MACD柱状图
            macd = (dif - dea) * 2
            
            return {'DIF': dif, 'DEA': dea, 'MACD': macd}
            
        except Exception as e:
            logger.error(f"计算MACD指标失败: {e}")
            if isinstance(data, np.ndarray):
                return {'dif': np.array([]), 'dea': np.array([]), 'macd': np.array([])}
            else:
                return {'DIF': pd.Series(dtype=float), 'DEA': pd.Series(dtype=float), 'MACD': pd.Series(dtype=float)}
    
    def calculate_kdj(self, high: Union[pd.DataFrame, np.ndarray], 
                     low: Union[pd.DataFrame, np.ndarray] = None, 
                     close: Union[pd.DataFrame, np.ndarray] = None,
                     k_period: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        计算KDJ指标
        
        Args:
            high: 最高价数据（DataFrame或numpy数组）
            low: 最低价数据（仅numpy数组时需要）
            close: 收盘价数据（仅numpy数组时需要）
            k_period: RSV周期，默认9
            k_smooth: K值平滑周期，默认3
            d_smooth: D值平滑周期，默认3
            
        Returns:
            包含K、D、J的字典
        """
        try:
            # 处理numpy数组输入
            if isinstance(high, np.ndarray):
                if low is None or close is None:
                    logger.error("numpy数组模式需要提供high、low、close三个数组")
                    return {'k': np.array([]), 'd': np.array([]), 'j': np.array([])}
                
                if len(high) == 0:
                    return {'k': np.array([]), 'd': np.array([]), 'j': np.array([])}
                
                # 计算RSV
                rsv = np.zeros_like(close, dtype=float)
                for i in range(len(close)):
                    start_idx = max(0, i - k_period + 1)
                    period_high = np.max(high[start_idx:i+1])
                    period_low = np.min(low[start_idx:i+1])
                    
                    if period_high != period_low:
                        rsv[i] = (close[i] - period_low) / (period_high - period_low) * 100
                    else:
                        rsv[i] = 50.0
                
                # 计算K值 (使用SMA)
                k = self._calculate_sma_array(rsv, k_smooth)
                
                # 计算D值 (K值的SMA)
                d = self._calculate_sma_array(k, d_smooth)
                
                # 计算J值
                j = 3 * k - 2 * d
                
                return {'k': k, 'd': d, 'j': j}
            
            # 处理DataFrame输入（原有逻辑）
            data = high  # 在DataFrame模式下，第一个参数是完整的data
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"计算KDJ需要{col}列")
                    return {'K': pd.Series(dtype=float), 'D': pd.Series(dtype=float), 'J': pd.Series(dtype=float)}
            
            high_series = data['high']
            low_series = data['low']
            close_series = data['close']
            
            # 计算RSV
            low_min = low_series.rolling(window=k_period, min_periods=1).min()
            high_max = high_series.rolling(window=k_period, min_periods=1).max()
            
            # 避免除零
            rsv = pd.Series(50.0, index=close_series.index)  # 默认值50
            valid_mask = (high_max != low_min)
            rsv[valid_mask] = (close_series[valid_mask] - low_min[valid_mask]) / (high_max[valid_mask] - low_min[valid_mask]) * 100
            
            # 计算K值 (使用SMA)
            k = self.calculate_sma(rsv, k_smooth, 1)
            
            # 计算D值 (K值的SMA)
            d = self.calculate_sma(k, d_smooth, 1)
            
            # 计算J值
            j = 3 * k - 2 * d
            
            return {'K': k, 'D': d, 'J': j}
            
        except Exception as e:
            logger.error(f"计算KDJ指标失败: {e}")
            if isinstance(high, np.ndarray):
                return {'k': np.array([]), 'd': np.array([]), 'j': np.array([])}
            else:
                return {'K': pd.Series(dtype=float), 'D': pd.Series(dtype=float), 'J': pd.Series(dtype=float)}
    
    def _calculate_sma_array(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算数组的简单移动平均"""
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = (data[i] + result[i-1] * (period - 1)) / period
            
        return result
    
    def calculate_rsi(self, data: Union[pd.DataFrame, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
        """
        计算RSI指标
        
        Args:
            data: 股票数据DataFrame或numpy数组
            period: 计算周期，默认14
            
        Returns:
            RSI值序列
        """
        try:
            # 处理numpy数组输入
            if isinstance(data, np.ndarray):
                if len(data) <= 1:
                    return np.array([50.0] * len(data)) if len(data) > 0 else np.array([])
                
                # 计算价格变化
                delta = np.diff(data, prepend=data[0])
                
                # 分离涨跌
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                # 计算平均涨跌
                avg_gain = np.zeros_like(data, dtype=float)
                avg_loss = np.zeros_like(data, dtype=float)
                
                for i in range(len(data)):
                    start_idx = max(0, i - period + 1)
                    avg_gain[i] = np.mean(gain[start_idx:i+1])
                    avg_loss[i] = np.mean(loss[start_idx:i+1])
                
                # 计算RSI
                rsi = np.zeros_like(data, dtype=float)
                for i in range(len(data)):
                    if avg_loss[i] == 0:
                        rsi[i] = 100.0
                    else:
                        rs = avg_gain[i] / avg_loss[i]
                        rsi[i] = 100 - (100 / (1 + rs))
                
                return rsi
            
            # 处理DataFrame输入
            if not self._validate_data(data):
                return pd.Series(dtype=float)
            
            close = data['close']
            delta = close.diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            # 避免除零
            rs = pd.Series(1.0, index=close.index)
            valid_mask = (avg_loss != 0)
            rs[valid_mask] = avg_gain[valid_mask] / avg_loss[valid_mask]
            
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"计算RSI指标失败: {e}")
            if isinstance(data, np.ndarray):
                return np.array([])
            else:
                return pd.Series(dtype=float)
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带指标
        
        Args:
            data: 股票数据DataFrame
            period: 计算周期，默认20
            std_dev: 标准差倍数，默认2.0
            
        Returns:
            包含UPPER、MIDDLE、LOWER的字典
        """
        if not self._validate_data(data):
            return {'UPPER': pd.Series(dtype=float), 'MIDDLE': pd.Series(dtype=float), 'LOWER': pd.Series(dtype=float)}
        
        try:
            close = data['close']
            
            # 计算中轨（移动平均线）
            middle = self.calculate_ma(data, period)
            
            # 计算标准差
            std = close.rolling(window=period, min_periods=1).std()
            
            # 计算上轨和下轨
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {'UPPER': upper, 'MIDDLE': middle, 'LOWER': lower}
            
        except Exception as e:
            logger.error(f"计算布林带指标失败: {e}")
            return {'UPPER': pd.Series(dtype=float), 'MIDDLE': pd.Series(dtype=float), 'LOWER': pd.Series(dtype=float)}
    
    def calculate_wvad(self, data: pd.DataFrame, period: int = 55) -> Dict[str, pd.Series]:
        """
        计算WVAD指标（威廉变异离散量）
        
        Args:
            data: 股票数据DataFrame，需包含high、low、close列
            period: 计算周期，默认55
            
        Returns:
            包含WVAD、WV_MA、WV_CHG的字典
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"计算WVAD需要{col}列")
                return {'WVAD': pd.Series(dtype=float), 'WV_MA': pd.Series(dtype=float), 'WV_CHG': pd.Series(dtype=float)}
        
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # 计算最低价和最高价
            l55 = low.rolling(window=period, min_periods=1).min()
            h55 = high.rolling(window=period, min_periods=1).max()
            
            # 计算差值
            diff = pd.Series(50.0, index=close.index)  # 默认值50
            valid_mask = (h55 != l55)
            diff[valid_mask] = (close[valid_mask] - l55[valid_mask]) / (h55[valid_mask] - l55[valid_mask]) * 100
            
            # 计算SMA
            s1 = self.calculate_sma(diff, 5, 1)
            s2 = self.calculate_sma(s1, 3, 1)
            
            # 计算WVAD
            wvad = 3 * s1 - 2 * s2
            
            # 计算WVAD的移动平均
            wv_ma = self.calculate_ma(pd.DataFrame({'close': wvad}), 3)
            
            # 计算变化率
            wv_chg = wv_ma.pct_change() * 100
            
            return {'WVAD': wvad, 'WV_MA': wv_ma, 'WV_CHG': wv_chg}
            
        except Exception as e:
            logger.error(f"计算WVAD指标失败: {e}")
            return {'WVAD': pd.Series(dtype=float), 'WV_MA': pd.Series(dtype=float), 'WV_CHG': pd.Series(dtype=float)}
    
    def calculate_all_indicators(self, data: pd.DataFrame, 
                               indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        计算所有或指定的技术指标
        
        Args:
            data: 股票数据DataFrame
            indicators: 指定要计算的指标列表，None表示计算所有指标
            
        Returns:
            包含所有指标结果的字典
        """
        if not self._validate_data(data):
            return {}
        
        # 默认计算所有指标
        if indicators is None:
            indicators = ['MA', 'EMA', 'MACD', 'KDJ', 'RSI', 'BOLL', 'WVAD']
        
        results = {}
        
        try:
            for indicator in indicators:
                indicator_upper = indicator.upper()
                
                if indicator_upper == 'MA':
                    # 计算多个周期的MA
                    for period in [5, 10, 20, 30, 60]:
                        results[f'MA{period}'] = self.calculate_ma(data, period)
                
                elif indicator_upper == 'EMA':
                    # 计算多个周期的EMA
                    for period in [5, 10, 20, 30, 60]:
                        results[f'EMA{period}'] = self.calculate_ema(data, period)
                
                elif indicator_upper == 'MACD':
                    macd_result = self.calculate_macd(data)
                    results.update(macd_result)
                
                elif indicator_upper == 'KDJ':
                    kdj_result = self.calculate_kdj(data)
                    results.update(kdj_result)
                
                elif indicator_upper == 'RSI':
                    results['RSI'] = self.calculate_rsi(data)
                
                elif indicator_upper == 'BOLL':
                    boll_result = self.calculate_bollinger_bands(data)
                    results.update(boll_result)
                
                elif indicator_upper == 'WVAD':
                    wvad_result = self.calculate_wvad(data)
                    results.update(wvad_result)
                
                else:
                    logger.warning(f"不支持的指标: {indicator}")
            
            self.performance_stats['total_calculations'] += 1
            logger.info(f"成功计算{len(results)}个技术指标")
            
            return results
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def get_indicator_value_at_date(self, data: pd.DataFrame, indicator: str, 
                                   date_idx: int, **kwargs) -> Union[float, Dict[str, float]]:
        """
        获取指定日期索引的指标值
        
        Args:
            data: 股票数据DataFrame
            indicator: 指标名称
            date_idx: 日期索引
            **kwargs: 指标参数
            
        Returns:
            指标值或指标值字典
        """
        try:
            indicator_upper = indicator.upper()
            
            if indicator_upper == 'MA':
                period = kwargs.get('period', 5)
                ma_values = self.calculate_ma(data, period)
                return ma_values.iloc[date_idx] if date_idx < len(ma_values) else np.nan
            
            elif indicator_upper == 'EMA':
                period = kwargs.get('period', 5)
                ema_values = self.calculate_ema(data, period)
                return ema_values.iloc[date_idx] if date_idx < len(ema_values) else np.nan
            
            elif indicator_upper == 'MACD':
                macd_result = self.calculate_macd(data, **kwargs)
                return {
                    'DIF': macd_result['DIF'].iloc[date_idx] if date_idx < len(macd_result['DIF']) else np.nan,
                    'DEA': macd_result['DEA'].iloc[date_idx] if date_idx < len(macd_result['DEA']) else np.nan,
                    'MACD': macd_result['MACD'].iloc[date_idx] if date_idx < len(macd_result['MACD']) else np.nan
                }
            
            elif indicator_upper == 'KDJ':
                kdj_result = self.calculate_kdj(data, **kwargs)
                return {
                    'K': kdj_result['K'].iloc[date_idx] if date_idx < len(kdj_result['K']) else np.nan,
                    'D': kdj_result['D'].iloc[date_idx] if date_idx < len(kdj_result['D']) else np.nan,
                    'J': kdj_result['J'].iloc[date_idx] if date_idx < len(kdj_result['J']) else np.nan
                }
            
            elif indicator_upper == 'RSI':
                rsi_values = self.calculate_rsi(data, **kwargs)
                return rsi_values.iloc[date_idx] if date_idx < len(rsi_values) else np.nan
            
            elif indicator_upper == 'BOLL':
                boll_result = self.calculate_bollinger_bands(data, **kwargs)
                return {
                    'UPPER': boll_result['UPPER'].iloc[date_idx] if date_idx < len(boll_result['UPPER']) else np.nan,
                    'MIDDLE': boll_result['MIDDLE'].iloc[date_idx] if date_idx < len(boll_result['MIDDLE']) else np.nan,
                    'LOWER': boll_result['LOWER'].iloc[date_idx] if date_idx < len(boll_result['LOWER']) else np.nan
                }
            
            elif indicator_upper == 'WVAD':
                wvad_result = self.calculate_wvad(data, **kwargs)
                return {
                    'WVAD': wvad_result['WVAD'].iloc[date_idx] if date_idx < len(wvad_result['WVAD']) else np.nan,
                    'WV_MA': wvad_result['WV_MA'].iloc[date_idx] if date_idx < len(wvad_result['WV_MA']) else np.nan,
                    'WV_CHG': wvad_result['WV_CHG'].iloc[date_idx] if date_idx < len(wvad_result['WV_CHG']) else np.nan
                }
            
            else:
                logger.error(f"不支持的指标: {indicator}")
                return np.nan
                
        except Exception as e:
            logger.error(f"获取指标值失败: {e}")
            return np.nan
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        if stats['total_calculations'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calculations']
        else:
            stats['cache_hit_rate'] = 0.0
        
        if stats['calculation_times']:
            stats['avg_calculation_time'] = np.mean(stats['calculation_times'])
            stats['max_calculation_time'] = np.max(stats['calculation_times'])
            stats['min_calculation_time'] = np.min(stats['calculation_times'])
        
        return stats
    
    def clear_cache(self):
        """清除缓存"""
        if self.cache_manager:
            self.cache_manager.clear()
            logger.info("缓存已清除")
    
    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'calculation_times': []
        }
        logger.info("性能统计已重置") 