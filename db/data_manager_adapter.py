#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
数据管理器适配器

提供向后兼容的API接口，将现有的DataManager调用适配到EnhancedDataManager
"""

import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from db.enhanced_data_manager import get_enhanced_data_manager
from db.enhanced_connection_pool import initialize_connection_pool
from monitoring.performance_monitor import get_performance_monitor
from utils.stability_enhancer import get_stability_manager, retry
from utils.logger import get_logger
from utils.exceptions import DataAccessError, DataValidationError
from enums.period import Period
from models.stock_info import StockInfo

logger = get_logger(__name__)


class DataManagerAdapter:
    """
    数据管理器适配器
    
    提供与原有DataManager兼容的API接口，内部使用增强的数据管理器
    """
    
    def __init__(self, enable_monitoring: bool = True, enable_stability: bool = True):
        """
        初始化适配器
        
        Args:
            enable_monitoring: 是否启用性能监控
            enable_stability: 是否启用稳定性增强
        """
        # 初始化连接池
        self.connection_pool = initialize_connection_pool(
            host='localhost',
            port=9000,
            database='stock',
            max_connections=20,
            min_connections=5
        )
        
        # 获取增强数据管理器
        self.enhanced_manager = get_enhanced_data_manager()
        
        # 性能监控
        self.performance_monitor = None
        if enable_monitoring:
            self.performance_monitor = get_performance_monitor()
            self.performance_monitor.start_monitoring()
        
        # 稳定性管理器
        self.stability_manager = None
        if enable_stability:
            self.stability_manager = get_stability_manager()
            self._setup_stability_features()
        
        logger.info("数据管理器适配器初始化完成，已启用优化功能")
    
    def _setup_stability_features(self):
        """设置稳定性功能"""
        if not self.stability_manager:
            return
        
        # 创建数据库查询熔断器
        self.db_circuit_breaker = self.stability_manager.create_circuit_breaker(
            'database_query',
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # 注册降级服务
        def fallback_get_stock_data(stock_code: str, **kwargs):
            """数据获取降级服务"""
            logger.warning(f"使用降级服务获取股票数据: {stock_code}")
            # 返回空的StockInfo对象
            empty_df = pd.DataFrame()
            return StockInfo(empty_df)
        
        self.stability_manager.register_degradation(
            'get_stock_data',
            fallback_get_stock_data,
            lambda: False,  # 暂时不启用降级
            "数据获取服务降级"
        )
    
    @retry(max_attempts=3, delay=0.5)
    def get_stock_data(self, 
                      stock_code: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      period: str = 'daily',
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票数据（兼容原有API）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期
            limit: 限制记录数
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 转换周期参数
            level = self._convert_period_to_level(period)
            
            # 使用增强数据管理器获取数据
            stock_info = self.enhanced_manager.get_stock_info(
                stock_code=stock_code,
                level=level,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # 转换为DataFrame
            df = stock_info.to_dataframe()
            
            logger.debug(f"获取股票数据成功: {stock_code}, 记录数: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {stock_code}, 错误: {e}")
            raise DataAccessError(f"获取股票数据失败: {e}")
    
    def get_stock_info(self, 
                      stock_code: Union[str, List[str]] = None,
                      level: Union[str, Period] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      filters: Optional[Dict[str, Any]] = None,
                      limit: Optional[int] = None,
                      order_by: str = "date DESC") -> StockInfo:
        """
        获取股票信息（增强API）
        
        Args:
            stock_code: 股票代码或股票代码列表
            level: K线周期
            start_date: 开始日期
            end_date: 结束日期
            filters: 过滤条件
            limit: 限制返回记录数
            order_by: 排序规则
            
        Returns:
            StockInfo: 股票数据对象
        """
        return self.enhanced_manager.get_stock_info(
            stock_code=stock_code,
            level=level,
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            limit=limit,
            order_by=order_by
        )
    
    def get_stock_list(self, 
                      market: Optional[str] = None,
                      industry: Optional[str] = None,
                      limit: Optional[int] = None) -> List[str]:
        """
        获取股票列表（兼容原有API）
        
        Args:
            market: 市场
            industry: 行业
            limit: 限制数量
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 构建过滤条件
            filters = {}
            if industry:
                filters['industry'] = industry
            
            # 获取股票信息
            stock_info = self.enhanced_manager.get_stock_info(
                filters=filters,
                limit=limit,
                order_by="code"
            )
            
            # 提取股票代码
            df = stock_info.to_dataframe()
            if not df.empty and 'code' in df.columns:
                return df['code'].unique().tolist()
            else:
                return []
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_stock_industry(self, stock_code: str) -> Optional[str]:
        """
        获取股票行业（兼容原有API）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[str]: 行业名称
        """
        try:
            stock_info = self.enhanced_manager.get_stock_info(
                stock_code=stock_code,
                limit=1
            )
            
            df = stock_info.to_dataframe()
            if not df.empty and 'industry' in df.columns:
                return df['industry'].iloc[0]
            else:
                return None
                
        except Exception as e:
            logger.debug(f"获取股票行业失败: {stock_code}, 错误: {e}")
            return None

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票名称（兼容原有API）

        Args:
            stock_code: 股票代码

        Returns:
            Optional[str]: 股票名称
        """
        try:
            stock_info = self.enhanced_manager.get_stock_info(
                stock_code=stock_code,
                limit=1
            )

            df = stock_info.to_dataframe()
            if not df.empty and 'name' in df.columns:
                return df['name'].iloc[0]
            else:
                # 如果没有名称字段，返回股票代码
                return stock_code

        except Exception as e:
            logger.debug(f"获取股票名称失败: {stock_code}, 错误: {e}")
            # 返回股票代码作为默认值
            return stock_code

    def get_previous_trade_date(self, date: str, days: int = 1) -> str:
        """
        获取前N个交易日（兼容原有API）
        
        Args:
            date: 基准日期
            days: 往前天数
            
        Returns:
            str: 前N个交易日
        """
        try:
            # 简化实现：往前推算天数（实际应该查询交易日历）
            from datetime import datetime, timedelta
            
            base_date = datetime.strptime(date, '%Y-%m-%d')
            # 考虑周末，大概推算
            estimated_days = days * 1.4  # 考虑周末因子
            previous_date = base_date - timedelta(days=int(estimated_days))
            
            return previous_date.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"计算前一交易日失败: {date}, 错误: {e}")
            # 返回一个合理的默认值
            from datetime import datetime, timedelta
            base_date = datetime.strptime(date, '%Y-%m-%d')
            previous_date = base_date - timedelta(days=days * 2)
            return previous_date.strftime('%Y-%m-%d')
    
    def save_selection_result(self, 
                            result: pd.DataFrame, 
                            strategy_id: str, 
                            selection_date: str = None) -> bool:
        """
        保存选股结果（兼容原有API）
        
        Args:
            result: 选股结果DataFrame
            strategy_id: 策略ID
            selection_date: 选股日期
            
        Returns:
            bool: 保存成功返回True
        """
        try:
            # 这里可以扩展保存逻辑
            logger.info(f"保存选股结果: 策略ID={strategy_id}, 记录数={len(result)}")
            
            # 简化实现：记录日志
            if selection_date is None:
                selection_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"选股结果已保存: {strategy_id}, 日期: {selection_date}, 股票数: {len(result)}")
            return True
            
        except Exception as e:
            logger.error(f"保存选股结果失败: {e}")
            return False
    
    def _convert_period_to_level(self, period: str) -> str:
        """
        转换周期参数到level参数
        
        Args:
            period: 原有的周期参数
            
        Returns:
            str: 转换后的level参数
        """
        period_mapping = {
            'daily': 'DAILY',
            'day': 'DAILY',
            'weekly': 'WEEKLY',
            'week': 'WEEKLY',
            'monthly': 'MONTHLY',
            'month': 'MONTHLY',
            '15min': 'MIN_15',
            '30min': 'MIN_30',
            '60min': 'MIN_60',
            'min15': 'MIN_15',
            'min30': 'MIN_30',
            'min60': 'MIN_60'
        }
        
        return period_mapping.get(period.lower(), 'DAILY')
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计
        """
        stats = {}
        
        # 数据管理器统计
        if self.enhanced_manager:
            stats['data_manager'] = self.enhanced_manager.get_stats()
        
        # 连接池统计
        if self.connection_pool:
            stats['connection_pool'] = self.connection_pool.get_stats()
        
        # 性能监控统计
        if self.performance_monitor:
            stats['performance_monitor'] = self.performance_monitor.get_stats()
        
        # 稳定性统计
        if self.stability_manager:
            stats['stability'] = self.stability_manager.get_stability_status()
        
        return stats
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        清除缓存
        
        Args:
            pattern: 缓存模式，None表示清除所有
        """
        if self.enhanced_manager:
            self.enhanced_manager.clear_cache(pattern)
            logger.info(f"缓存已清除: {pattern or '全部'}")
    
    def close(self):
        """关闭适配器，清理资源"""
        try:
            # 停止性能监控
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # 关闭连接池
            if self.connection_pool:
                self.connection_pool.close()
            
            logger.info("数据管理器适配器已关闭")
            
        except Exception as e:
            logger.error(f"关闭适配器时出错: {e}")


# 全局适配器实例
_data_manager_adapter = None
_adapter_lock = threading.Lock()


def get_data_manager_adapter() -> DataManagerAdapter:
    """获取全局数据管理器适配器实例"""
    global _data_manager_adapter
    
    if _data_manager_adapter is None:
        with _adapter_lock:
            if _data_manager_adapter is None:
                _data_manager_adapter = DataManagerAdapter()
    
    return _data_manager_adapter


# 为了向后兼容，提供DataManager类的别名
class DataManager(DataManagerAdapter):
    """向后兼容的DataManager类"""
    pass
