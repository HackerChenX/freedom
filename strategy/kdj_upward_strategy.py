from utils.container import container
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KDJ上升策略
基于KDJ指标的K、D、J三线均上升的选股策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from strategy.unified_base_strategy import UnifiedBaseStrategy
from utils.logger import get_logger
from utils.decorators import exception_handler, performance_monitor
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class KDJUpwardStrategy(UnifiedBaseStrategy):
    """
    KDJ上升策略
    
    选择KDJ指标的K、D、J三线均呈上升趋势的股票
    """
    
    def __init__(self, k_period: int = 9, d_period: int = 3, j_multiplier: int = 3):
        # 依赖注入示例:
        # self.data_access = container.resolve("DataAccessInterface")
        # self.cache_service = container.resolve("ICacheService")
        """
        初始化KDJ上升策略
        
        Args:
            k_period: K值计算周期
            d_period: D值平滑周期  
            j_multiplier: J值计算倍数
        """
        super().__init__(
            name="KDJ上升策略",
            description="选择KDJ指标K、D、J三线均上升的股票",
            default_period='1d'
        )
        
        self.k_period = k_period
        self.d_period = d_period
        self.j_multiplier = j_multiplier
        
        logger.info(f"KDJ上升策略初始化完成: K周期={k_period}, D周期={d_period}, J倍数={j_multiplier}")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def select_stocks(self, stock_codes: List[str], target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        执行选股策略
        
        Args:
            stock_codes: 股票代码列表
            target_date: 目标日期，格式YYYY-MM-DD
            
        Returns:
            List[Dict]: 符合条件的股票列表
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"开始执行KDJ上升策略选股，股票数量: {len(stock_codes)}, 目标日期: {target_date}")
        
        selected_stocks = []
        
        for stock_code in stock_codes:
            try:
                result = self._analyze_single_stock(stock_code, target_date)
                if result and result.get('meets_condition', False):
                    selected_stocks.append(result)
                    
            except Exception as e:
                logger.warning(f"分析股票 {stock_code} 失败: {e}")
                continue
        
        # 按评分排序
        selected_stocks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"KDJ上升策略选股完成，选中股票数量: {len(selected_stocks)}")
        return selected_stocks
    
    def _analyze_single_stock(self, stock_code: str, target_date: str) -> Optional[Dict[str, Any]]:
        """
        分析单只股票
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 获取股票数据
            stock_data = self._get_stock_data(stock_code, target_date)
            if stock_data is None or len(stock_data) < self.k_period + 5:
                return None
            
            # 计算KDJ指标
            kdj_data = self._calculate_kdj(stock_data)
            if kdj_data is None:
                return None
            
            # 检查KDJ上升条件
            condition_result = self._check_kdj_upward_condition(kdj_data, target_date)
            
            if condition_result['meets_condition']:
                return {
                    'stock_code': stock_code,
                    'stock_name': stock_data.iloc[-1].get('name', f'股票{stock_code}'),
                    'target_date': target_date,
                    'strategy_name': self.name,
                    **condition_result
                }
            
            return None
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 失败: {e}")
            return None
    
    def _get_stock_data(self, stock_code: str, target_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据"""
        try:
            # 计算开始日期（需要足够的历史数据）
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_dt = target_dt - pd.Timedelta(days=60)  # 60天历史数据
            start_date = start_dt.strftime('%Y-%m-%d')
            
            # 使用数据访问接口获取数据
            data = self.data_access.get_stock_data(
                stock_code=stock_code,
                start_date=start_date,
                end_date=target_date,
                level='日线'
            )
            
            if data is None or data.empty:
                return None
            
            # 确保数据按日期排序
            if 'date' in data.columns:
                data = data.sort_values('date')
            
            return data
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据失败: {e}")
            return None
    
    def _calculate_kdj(self, data: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
        """计算KDJ指标"""
        try:
            if len(data) < self.k_period:
                return None
            
            high = pd.to_numeric(data['high'], errors='coerce')
            low = pd.to_numeric(data['low'], errors='coerce')
            close = pd.to_numeric(data['close'], errors='coerce')
            
            # 计算RSV (Raw Stochastic Value)
            lowest_low = low.rolling(window=self.k_period, min_periods=self.k_period).min()
            highest_high = high.rolling(window=self.k_period, min_periods=self.k_period).max()
            
            rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
            
            # 计算K值 (使用指数移动平均)
            k = rsv.ewm(alpha=1/self.d_period, adjust=False).mean()
            
            # 计算D值 (K值的指数移动平均)
            d = k.ewm(alpha=1/self.d_period, adjust=False).mean()
            
            # 计算J值
            j = self.j_multiplier * k - 2 * d
            
            return {'k': k, 'd': d, 'j': j}
            
        except Exception as e:
            logger.error(f"计算KDJ指标失败: {e}")
            return None
    
    def _check_kdj_upward_condition(self, kdj_data: Dict[str, pd.Series], target_date: str) -> Dict[str, Any]:
        """检查KDJ上升条件"""
        try:
            k_series = kdj_data['k']
            d_series = kdj_data['d']
            j_series = kdj_data['j']
            
            if len(k_series) < 2:
                return {'meets_condition': False, 'reason': 'insufficient_data'}
            
            # 获取最新和前一个值
            k_current = k_series.iloc[-1]
            k_previous = k_series.iloc[-2]
            d_current = d_series.iloc[-1]
            d_previous = d_series.iloc[-2]
            j_current = j_series.iloc[-1]
            j_previous = j_series.iloc[-2]
            
            # 检查是否有NaN值
            if pd.isna(k_current) or pd.isna(k_previous) or pd.isna(d_current) or pd.isna(d_previous) or pd.isna(j_current) or pd.isna(j_previous):
                return {'meets_condition': False, 'reason': 'invalid_data'}
            
            # 检查三线均上升条件
            k_upward = k_current > k_previous
            d_upward = d_current > d_previous
            j_upward = j_current > j_previous
            
            # 检查KDJ值范围（避免极端值）
            k_in_range = 20 <= k_current <= 80
            d_in_range = 20 <= d_current <= 80
            j_in_range = 0 <= j_current <= 100
            
            # 综合判断
            meets_condition = k_upward and d_upward and j_upward and k_in_range and d_in_range and j_in_range
            
            # 计算上升强度和评分
            k_strength = (k_current - k_previous) / k_previous if k_previous != 0 else 0
            d_strength = (d_current - d_previous) / d_previous if d_previous != 0 else 0
            j_strength = (j_current - j_previous) / abs(j_previous) if j_previous != 0 else 0
            
            # 计算综合评分
            score = (k_strength * 0.35 + d_strength * 0.35 + j_strength * 0.30) if meets_condition else 0
            
            return {
                'meets_condition': meets_condition,
                'k_current': k_current,
                'k_previous': k_previous,
                'k_upward': k_upward,
                'k_strength': k_strength,
                'd_current': d_current,
                'd_previous': d_previous,
                'd_upward': d_upward,
                'd_strength': d_strength,
                'j_current': j_current,
                'j_previous': j_previous,
                'j_upward': j_upward,
                'j_strength': j_strength,
                'score': score,
                'reason': 'kdj_all_upward' if meets_condition else 'condition_not_met'
            }
            
        except Exception as e:
            logger.error(f"检查KDJ上升条件失败: {e}")
            return {'meets_condition': False, 'reason': f'error: {e}'}
    
    def select_stocks_unified_base_strategy(self, universe: List[str],
                                           start_date: str, end_date: str,
                                           **kwargs) -> pd.DataFrame:
        """
        统一基类要求的选股方法实现

        Args:
            universe: 股票池
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 选股结果
        """
        # 使用现有的select_stocks方法
        results = self.select_stocks(universe, end_date)

        # 转换为DataFrame格式
        if not results:
            return pd.DataFrame()

        df_data = []
        for result in results:
            df_data.append({
                'stock_code': result.get('stock_code'),
                'stock_name': result.get('stock_name'),
                'score': result.get('score', 0),
                'k_current': result.get('k_current'),
                'd_current': result.get('d_current'),
                'j_current': result.get('j_current'),
                'strategy_name': self.name
            })

        return pd.DataFrame(df_data)

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'k_period': self.k_period,
                'd_period': self.d_period,
                'j_multiplier': self.j_multiplier
            },
            'type': 'technical_indicator',
            'category': 'momentum_oscillator'
        }
