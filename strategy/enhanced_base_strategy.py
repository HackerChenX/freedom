"""
增强选股策略基类模块

提供多周期选股策略的通用接口和功能，支持：
1. 默认使用最新数据时间
2. 多周期数据支持（15分钟、30分钟、60分钟、日、周、月）
3. 周期+指标的唯一性配置
4. 智能时间管理
"""

import abc
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import get_logger
from db.clickhouse_db import get_clickhouse_db
from enums.kline_period import KlinePeriod

logger = get_logger(__name__)


@dataclass
class PeriodConfig:
    """周期配置类"""
    period: str  # 周期：'15m', '30m', '1h', '1d', '1w', '1M'
    period_name: str  # 周期名称：'15分钟', '30分钟', '1小时', '日线', '周线', '月线'
    table_name: str  # 数据表名
    
    @classmethod
    def get_all_periods(cls) -> List['PeriodConfig']:
        """获取所有支持的周期配置"""
        return [
            cls('15m', '15分钟', 'stock_kline_15m'),
            cls('30m', '30分钟', 'stock_kline_30m'), 
            cls('1h', '1小时', 'stock_kline_1h'),
            cls('1d', '日线', 'stock_info'),  # 使用现有的stock_info表
            cls('1w', '周线', 'stock_kline_1w'),
            cls('1M', '月线', 'stock_kline_1M')
        ]
    
    @classmethod
    def get_period_by_name(cls, period: str) -> Optional['PeriodConfig']:
        """根据周期名称获取配置"""
        for config in cls.get_all_periods():
            if config.period == period:
                return config
        return None


@dataclass
class IndicatorCondition:
    """指标条件配置类"""
    indicator_name: str  # 指标名称
    period: str  # 周期
    parameters: Dict[str, Any]  # 指标参数
    condition: str  # 条件表达式
    signal_type: str  # 信号类型：'BUY', 'SELL', 'HOLD'
    weight: float = 1.0  # 权重
    
    def get_unique_key(self) -> str:
        """获取唯一标识键（周期+指标）"""
        return f"{self.period}_{self.indicator_name}_{hash(str(self.parameters))}"


class EnhancedBaseStrategy(abc.ABC):
    """
    增强选股策略基类
    
    支持多周期数据和智能时间管理
    """
    
    def __init__(self, name: str, description: str = "", default_period: str = "1d"):
        """
        初始化增强选股策略
        
        Args:
            name: 策略名称
            description: 策略描述
            default_period: 默认周期
        """
        self.name = name
        self.description = description
        self.default_period = default_period
        self._result = None
        self._error = None
        self._parameters = {}
        self._conditions = []  # 指标条件列表
        self.db = get_clickhouse_db()
        
        # 初始化默认参数
        self._init_default_parameters()
        
        logger.info(f"增强选股策略 {name} 初始化完成，默认周期: {default_period}")
    
    def _init_default_parameters(self):
        """初始化默认参数"""
        self._parameters.update({
            'start_date': None,  # 开始日期，None表示自动计算
            'end_date': None,    # 结束日期，None表示使用最新数据
            'period': self.default_period,  # 数据周期
            'lookback_days': 30,  # 回看天数
            'min_volume': 100000,  # 最小成交量
            'min_market_cap': 1000000000,  # 最小市值（10亿）
            'exclude_st': True,   # 排除ST股票
            'exclude_new': True,  # 排除新股（上市不满30天）
            'max_results': 100    # 最大结果数量
        })
    
    def get_latest_data_date(self) -> str:
        """
        获取数据库中的最新数据日期
        
        Returns:
            str: 最新数据日期，格式YYYY-MM-DD
        """
        try:
            # 从stock_info表查询最新日期
            result = self.db.query('SELECT MAX(date) as max_date FROM stock_info LIMIT 1')
            if not result.empty:
                max_date = result.iloc[0]['max_date']
                return str(max_date)
            
            # 如果查询失败，返回默认日期
            return "2025-05-23"
        except Exception as e:
            logger.error(f"获取最新数据日期失败: {e}")
            return "2025-05-23"
    
    def get_effective_date_range(self) -> Tuple[str, str]:
        """
        获取有效的日期范围
        
        Returns:
            Tuple[str, str]: (开始日期, 结束日期)
        """
        # 获取最新数据日期
        latest_date = self.get_latest_data_date()
        
        # 如果没有指定结束日期，使用最新数据日期
        end_date = self._parameters.get('end_date') or latest_date
        
        # 如果没有指定开始日期，根据回看天数计算
        start_date = self._parameters.get('start_date')
        if not start_date:
            lookback_days = self._parameters.get('lookback_days', 30)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=lookback_days * 2)  # 预留更多数据用于计算
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"有效日期范围: {start_date} 到 {end_date}")
        return start_date, end_date
    
    def add_indicator_condition(self, condition: IndicatorCondition):
        """
        添加指标条件
        
        Args:
            condition: 指标条件
        """
        # 检查是否已存在相同的条件（周期+指标的唯一性）
        unique_key = condition.get_unique_key()
        existing_keys = [c.get_unique_key() for c in self._conditions]
        
        if unique_key in existing_keys:
            logger.warning(f"指标条件已存在: {unique_key}，将覆盖原有条件")
            # 移除旧条件
            self._conditions = [c for c in self._conditions if c.get_unique_key() != unique_key]
        
        self._conditions.append(condition)
        logger.info(f"添加指标条件: {condition.indicator_name} ({condition.period})")
    
    def get_conditions_by_period(self, period: str) -> List[IndicatorCondition]:
        """
        获取指定周期的所有条件
        
        Args:
            period: 周期
            
        Returns:
            List[IndicatorCondition]: 条件列表
        """
        return [c for c in self._conditions if c.period == period]
    
    def get_stock_data(self, stock_code: str, period: str, 
                       start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据
        
        Args:
            stock_code: 股票代码
            period: 周期
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Optional[pd.DataFrame]: 股票数据
        """
        try:
            period_config = PeriodConfig.get_period_by_name(period)
            if not period_config:
                logger.error(f"不支持的周期: {period}")
                return None
            
            # 构建查询SQL
            sql = f"""
            SELECT *
            FROM {period_config.table_name}
            WHERE code = %(stock_code)s
              AND date >= %(start_date)s
              AND date <= %(end_date)s
            ORDER BY date ASC
            """
            
            params = {
                'stock_code': stock_code,
                'start_date': start_date,
                'end_date': end_date
            }
            
            data = self.db.query(sql, params)
            
            if data.empty:
                logger.warning(f"股票 {stock_code} 在周期 {period} 的数据为空")
                return None
            
            # 确保日期列为datetime类型
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"获取股票数据失败 {stock_code} ({period}): {e}")
            return None
    
    def get_stock_universe(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        获取股票池
        
        Args:
            filters: 过滤条件
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 构建基础查询
            sql = """
            SELECT DISTINCT code
            FROM stock_info
            WHERE 1=1
            """
            
            params = {}
            
            # 应用过滤条件
            if self._parameters.get('exclude_st', True):
                sql += " AND name NOT LIKE '%ST%'"
            
            if self._parameters.get('min_volume', 0) > 0:
                sql += " AND volume >= %(min_volume)s"
                params['min_volume'] = self._parameters['min_volume']
            
            # 排除新股
            if self._parameters.get('exclude_new', True):
                latest_date = self.get_latest_data_date()
                cutoff_date = (datetime.strptime(latest_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
                sql += " AND code IN (SELECT DISTINCT code FROM stock_info WHERE date <= %(cutoff_date)s)"
                params['cutoff_date'] = cutoff_date
            
            # 限制结果数量
            max_results = self._parameters.get('max_results', 100)
            sql += f" LIMIT {max_results}"
            
            result = self.db.query(sql, params)
            
            if result.empty:
                logger.warning("未获取到任何股票代码")
                return []
            
            stock_codes = result['code'].tolist()
            logger.info(f"获取到 {len(stock_codes)} 只股票")
            
            return stock_codes
            
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return []
    
    @property
    def result(self) -> Optional[pd.DataFrame]:
        """获取选股结果"""
        return self._result
    
    @property
    def error(self) -> Optional[Exception]:
        """获取错误信息"""
        return self._error
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self._parameters.copy()
    
    @property
    def conditions(self) -> List[IndicatorCondition]:
        """获取指标条件列表"""
        return self._conditions.copy()
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        设置策略参数
        
        Args:
            key: 参数名
            value: 参数值
        """
        self._parameters[key] = value
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        批量设置策略参数
        
        Args:
            params: 参数字典
        """
        self._parameters.update(params)
    
    @abc.abstractmethod
    def select(self, universe: Optional[List[str]] = None, *args, **kwargs) -> pd.DataFrame:
        """
        执行选股策略
        
        Args:
            universe: 股票代码列表，None表示使用默认股票池
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 选股结果，包含股票代码、名称等信息
        """
        pass
    
    def run(self, universe: Optional[List[str]] = None, *args, **kwargs) -> pd.DataFrame:
        """
        运行选股策略并处理异常
        
        Args:
            universe: 股票代码列表
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            pd.DataFrame: 选股结果
        """
        try:
            # 如果没有指定股票池，使用默认股票池
            if universe is None:
                universe = self.get_stock_universe()
            
            self._result = self.select(universe, *args, **kwargs)
            self._error = None
            return self._result
        except Exception as e:
            logger.error(f"执行选股策略 {self.name} 时出错: {e}")
            self._error = e
            self._result = None
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将策略转换为字典表示
        
        Returns:
            Dict[str, Any]: 策略的字典表示
        """
        return {
            'name': self.name,
            'description': self.description,
            'default_period': self.default_period,
            'parameters': self._parameters,
            'conditions_count': len(self._conditions),
            'conditions': [
                {
                    'indicator': c.indicator_name,
                    'period': c.period,
                    'signal_type': c.signal_type,
                    'unique_key': c.get_unique_key()
                }
                for c in self._conditions
            ],
            'has_result': self._result is not None,
            'has_error': self._error is not None,
            'error': str(self._error) if self._error else None
        } 