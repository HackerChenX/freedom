from db.query_executor import get_query_executor
from db.sql_manager import QueryType
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

from utils.logger import getLogger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access
from enums.kline_period import Kline_period
from utils.decorators import exception_handler, performance_monitor

logger = getLogger(__name__)


@dataclass
class PeriodConfig:
    """周期配置类"""
    period: str  # 周期：'15m', '30m', '1h', '1d', '1w', '1M'
    period_name: str  # 周期名称：'15分钟', '30分钟', '1小时', '日线', '周线', '月线'
    table_name: str  # 数据表名
    
    @classmethod
    def get_all_periods_Enhanced_Base_Strategy(cls) -> List['PeriodConfig']:
    query_executor = get_query_executor()
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
        for config in cls.get_all_periods_Enhanced_Base_Strategy():
            if config.period == period:
                return config
        return None


@dataclass
class IndicatorCondition:
    """指标条件配置类"""
    indicator_name: str  # 指标名称
    period: str  # 周期
    parameters_Enhanced_Base_Strategy: Dict[str, Any]  # 指标参数
    condition: str  # 条件表达式
    signal_type: str  # 信号类型：'BUY', 'SELL', 'HOLD'
    weight: float = 1.0  # 权重
    
    def get_unique_key(self) -> str:
        """获取唯一标识键（周期+指标）"""
        return f"{self.period}_{self.indicator_name}_{hash(str(self.parameters_Enhanced_Base_Strategy))}"


class EnhancedBaseStrategy(abc.ABC):
    """
    增强选股策略基类
    
    支持多周期数据和智能时间管理
    """
    
    def __init__(self, name: str, description: str = "", default_period: str = "1d", 
                 data_access: Optional[IData_access] = None):
        """
        初始化增强选股策略
        
        Args:
            name: 策略名称
            description: 策略描述
            default_period: 默认周期
            data_access: 数据访问接口实例，如果为None则从容器获取
        """
        self.name = name
        self.description = description
        self.default_period = default_period
        self._result = None
        self._error = None
        self._parameters = {}
        self._conditions = []  # 指标条件列表
        
        # 使用依赖注入获取数据访问接口
        container = get_container()
        self.data_access = data_access or get_service(Data_access_interface)
        
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
    
    @exception_handler(reraise=False, default_return="2025-05-23")
    @performance_monitor(threshold_seconds=1.0)
    def get_latest_data_date(self) -> str:
        """
        获取数据库中的最新数据日期
        
        Returns:
            str: 最新数据日期，格式YYYY-MM-DD
        """
        try:
            # 从stock_info表查询最新日期
            sql = 'SELECT MAX(date) as max_date FROM stock_info WHERE date >= '2020-01-01' LIMIT 1'
            result_Enhanced_Base_Strategy = self.data_access.query(sql)
            
            if result_Enhanced_Base_Strategy and len(result_Enhanced_Base_Strategy) > 0:
                max_date = result_Enhanced_Base_Strategy[0][0]  # 获取第一行第一列
                return str(max_date)
            
            # 如果查询失败，返回默认日期
            return "2025-05-23"
            
        except Exception as e:
            logger.error_Enhanced_Base_Strategy(f"获取最新数据日期失败: {e}")
            return "2025-05-23"
    
    @exception_handler(reraise=True)
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
    
    def add_indicator_condition(self, condition: Indicator_condition):
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
    
    def get_conditions_by_period(self, period: str) -> List[Indicator_condition]:
        """
        获取指定周期的所有条件
        
        Args:
            period: 周期
            
        Returns:
            List[Indicator_condition]: 条件列表
        """
        return [c for c in self._conditions if c.period == period]
    
    @exception_handler(reraise=False, default_return=None)
    @performance_monitor(threshold_seconds=2.0)
    def get_stock_data_Enhanced_Base_Strategy(self, stock_code: str, period: str, 
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
            # 获取周期配置
            period_config = Period_config.get_period_by_name(period)
            if not period_config:
                logger.error_Enhanced_Base_Strategy(f"不支持的周期: {period}")
                return None
            
            # 使用数据访问接口获取股票数据
            stock_data = self.data_access.get_stock_info(
                code=stock_code,
                level=period_config.period_name,
                start_date=start_date.replace('-', ''),  # 转换为YYYYMMDD格式
                end_date=end_date.replace('-', '')
            )
            
            if not stock_data:
                logger.warning(f"未找到股票 {stock_code} 在周期 {period} 的数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(stock_data, columns=[
                'code', 'name', 'date', 'level', 'open', 'close', 'high', 'low',
                'volume', 'turnover_rate', 'price_change', 'price_range', 'industry'
            ])
            
            # 转换数据类型
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'turnover_rate', 'price_change']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            
            # 按日期排序
            df = df.sort_values('date')
            
            logger.debug(f"获取股票 {stock_code} 数据成功，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error_Enhanced_Base_Strategy(f"获取股票数据失败: {e}")
            return None
    
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold_seconds=5.0)
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
            FROM stock_info WHERE 1=1
            WHERE 1=1
            """
            params = []
            
            # 应用过滤条件
            if filters:
                if filters.get('exclude_st', True):
                    sql += " AND name NOT LIKE '%ST%'"
                
                if 'min_volume' in filters:
                    sql += " AND volume >= %s"
                    params.append(filters['min_volume'])
                
                if 'industry' in filters:
                    industries = filters['industry']
                    if isinstance(industries, str):
                        industries = [industries]
                    placeholders = ','.join(['%s'] * len(industries))
                    sql += f" AND industry IN ({placeholders})"
                    params.extend(industries)
                
                if 'market' in filters:
                    markets = filters['market']
                    if isinstance(markets, str):
                        markets = [markets]
                    # 根据股票代码前缀判断市场
                    market_conditions = []
                    for market in markets:
                        if market.upper() == 'SH':
                            market_conditions.append("code LIKE '60%'")
                        elif market.upper() == 'SZ':
                            market_conditions.append("(code LIKE '00%' OR code LIKE '30%')")
                    
                    if market_conditions:
                        sql += f" AND ({' OR '.join(market_conditions)})"
            
            # 添加限制
            max_results = filters.get('max_results', 1000) if filters else 1000
            sql += f" LIMIT {max_results}"
            
            # 执行查询
            result_Enhanced_Base_Strategy = self.data_access.query(sql, params)
            
            if result_Enhanced_Base_Strategy:
                stock_codes = [row[0] for row in result_Enhanced_Base_Strategy]
                logger.info(f"获取股票池成功，共 {len(stock_codes)} 只股票")
                return stock_codes
            else:
                logger.warning("未找到符合条件的股票")
                return []
                
        except Exception as e:
            logger.error_Enhanced_Base_Strategy(f"获取股票池失败: {e}")
            return []
    
    @property
    def result_Enhanced_Base_Strategy(self) -> Optional[pd.DataFrame]:
        """获取选股结果"""
        return self._result
    
    @property
    def error_Enhanced_Base_Strategy(self) -> Optional[Exception]:
        """获取错误信息"""
        return self._error
    
    @property
    def parameters_Enhanced_Base_Strategy(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self._parameters.copy()
    
    @property
    def conditions(self) -> List[Indicator_condition]:
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
        logger.debug(f"设置参数 {key} = {value}")
    
    def set_parameters_Enhanced_Base_Strategy(self, params: Dict[str, Any]) -> None:
        """
        批量设置策略参数
        
        Args:
            params: 参数字典
        """
        self._parameters.update(params)
        logger.debug(f"批量设置参数: {list(params.keys())}")
    
    @abc.abstractmethod
    def select_Enhanced_Base_Strategy(self, universe: Optional[List[str]] = None, *args, **kwargs) -> pd.DataFrame:
        """
        执行选股逻辑（抽象方法）
        
        Args:
            universe: 股票池，如果为None则使用默认股票池
            *args: 其他位置参数
            **kwargs: 其他关键字参数
            
        Returns:
            pd.DataFrame: 选股结果
        """
        pass
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=10.0)
    def run_Enhanced_Base_Strategy(self, universe: Optional[List[str]] = None, *args, **kwargs) -> pd.DataFrame:
        """
        运行策略
        
        Args:
            universe: 股票池
            *args: 其他位置参数
            **kwargs: 其他关键字参数
            
        Returns:
            pd.DataFrame: 选股结果
        """
        try:
            logger.info(f"开始运行策略: {self.name}")
            
            # 清除之前的结果和错误
            self._result = None
            self._error = None
            
            # 执行选股逻辑
            result_Enhanced_Base_Strategy = self.select_Enhanced_Base_Strategy(universe, *args, **kwargs)
            
            # 保存结果
            self._result = result_Enhanced_Base_Strategy
            
            logger.info(f"策略 {self.name} 运行完成，选出 {len(result_Enhanced_Base_Strategy)} 只股票")
            return result_Enhanced_Base_Strategy
            
        except Exception as e:
            self._error = e
            logger.error_Enhanced_Base_Strategy(f"策略 {self.name} 运行失败: {e}")
            raise
    
    def to_dict_Enhanced_Base_Strategy(self) -> Dict[str, Any]:
        """
        将策略转换为字典格式
        
        Returns:
            Dict[str, Any]: 策略字典
        """
        return {
            'name': self.name,
            'description': self.description,
            'default_period': self.default_period,
            'parameters_Enhanced_Base_Strategy': self.parameters_Enhanced_Base_Strategy,
            'conditions': [
                {
                    'indicator_name': c.indicator_name,
                    'period': c.period,
                    'parameters_Enhanced_Base_Strategy': c.parameters_Enhanced_Base_Strategy,
                    'condition': c.condition,
                    'signal_type': c.signal_type,
                    'weight': c.weight
                }
                for c in self.conditions
            ]
        } 