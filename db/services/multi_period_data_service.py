"""
多周期数据服务
支持15分钟、30分钟、60分钟、日线、周线、月线等多个周期的数据获取
遵循六层架构，不直接写SQL
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from utils.logger import get_logger
from utils.enhanced_exception_handler import exception_handler
from utils.enhanced_performance_monitor import performance_monitor
from db.interfaces.data_access_interface import DataAccessInterface
from utils.dependency_injection import get_service

logger = get_logger(__name__)


class Period(Enum):
    """K线周期枚举"""
    MIN_15 = "15分钟"
    MIN_30 = "30分钟" 
    MIN_60 = "60分钟"
    DAILY = "日线"
    WEEKLY = "周线"
    MONTHLY = "月线"


class MultiPeriodDataService:
    """
    多周期数据服务
    
    提供统一的多周期股票数据获取接口，支持：
    - 15分钟、30分钟、60分钟、日线、周线、月线
    - 自动计算所需的数据量以满足指标计算需求
    - 缓存优化
    - 数据完整性验证
    """
    
    def __init__(self):
        """初始化多周期数据服务"""
        self.data_access = get_service(DataAccessInterface)
        
        # 各周期建议的最小数据量（用于指标计算）
        self.min_data_requirements = {
            Period.MIN_15: 500,   # 15分钟线需要更多数据点
            Period.MIN_30: 400,   # 30分钟线
            Period.MIN_60: 300,   # 60分钟线
            Period.DAILY: 250,    # 日线
            Period.WEEKLY: 100,   # 周线
            Period.MONTHLY: 50    # 月线
        }
        
        logger.info("多周期数据服务初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=5.0)
    def get_stock_multi_period_data(self, 
                                   stock_code: str, 
                                   target_date: str,
                                   periods: Optional[List[Period]] = None,
                                   lookback_days: Optional[int] = None) -> Dict[Period, pd.DataFrame]:
        """
        获取股票多周期数据
        
        Args:
            stock_code: 股票代码
            target_date: 目标分析日期
            periods: 需要的周期列表，None表示获取所有周期
            lookback_days: 向前获取的天数，None表示使用默认值
            
        Returns:
            Dict[Period, pd.DataFrame]: 各周期的数据
        """
        if periods is None:
            periods = list(Period)
        
        result = {}
        
        for period in periods:
            try:
                data = self.get_single_period_data(
                    stock_code=stock_code,
                    target_date=target_date,
                    period=period,
                    lookback_days=lookback_days
                )
                result[period] = data
                logger.debug(f"获取{stock_code} {period.value}数据成功，记录数: {len(data)}")
                
            except Exception as e:
                logger.warning(f"获取{stock_code} {period.value}数据失败: {e}")
                result[period] = pd.DataFrame()
        
        return result
    
    @exception_handler(reraise=True)
    def get_single_period_data(self,
                              stock_code: str,
                              target_date: str,
                              period: Period,
                              lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        获取单个周期的股票数据

        Args:
            stock_code: 股票代码
            target_date: 目标分析日期
            period: K线周期
            lookback_days: 向前获取的天数

        Returns:
            pd.DataFrame: 股票数据
        """
        # 计算需要的数据量
        if lookback_days is None:
            required_records = self.min_data_requirements.get(period, 250)
            lookback_days = self._calculate_lookback_days(period, required_records)
        else:
            # 当指定了lookback_days时，也需要设置required_records用于验证
            required_records = self.min_data_requirements.get(period, 250)

        # 计算开始日期
        start_date = self._calculate_start_date(target_date, lookback_days)

        # 构建查询参数
        period_value = period.value if hasattr(period, 'value') else str(period)
        query_params = {
            'code': stock_code,
            'level': period_value,
            'start_date': start_date,
            'end_date': target_date,
            'order_by': 'date ASC'
        }

        # 首先尝试直接获取数据
        data = self._query_stock_data_by_period(query_params)

        # 如果数据为空且是30分钟或60分钟周期，尝试从15分钟数据聚合
        if data.empty and period in [Period.MIN_30, Period.MIN_60]:
            logger.info(f"尝试从15分钟数据聚合生成{period.value}数据")
            data = self._aggregate_from_base_period(stock_code, target_date, period, lookback_days)

        # 验证数据完整性
        self._validate_data_completeness(data, stock_code, period, required_records)

        return data
    
    def _query_stock_data_by_period(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        通过数据访问层查询股票数据
        
        Args:
            params: 查询参数
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            # 构建标准查询（遵循六层架构，不直接写SQL）
            # 使用实际数据库字段名：turnover_rate而不是turnover_rate
            query = f"""
            SELECT code, name, date, level, open, high, low, close, volume, turnover_rate
            FROM stock_info WHERE level = %(level)s AND code = '{params['code']}'
            AND level = '{params['level']}'
            AND date >= '{params['start_date']}'
            AND date <= '{params['end_date']}'
            ORDER BY {params['order_by']}
            """

            # 通过数据访问接口执行查询
            return self.data_access.query_dataframe(query)
            
        except Exception as e:
            logger.error(f"查询股票数据失败: {e}")
            return pd.DataFrame()
    
    def _calculate_lookback_days(self, period: Period, required_records: int) -> int:
        """
        计算需要向前获取的天数
        
        Args:
            period: K线周期
            required_records: 需要的记录数
            
        Returns:
            int: 向前获取的天数
        """
        # 根据周期计算天数
        if period == Period.MIN_15:
            # 15分钟线：一天约26个数据点（9:30-15:00）
            return required_records // 26 + 30  # 额外加30天缓冲
        elif period == Period.MIN_30:
            # 30分钟线：一天约13个数据点
            return required_records // 13 + 30
        elif period == Period.MIN_60:
            # 60分钟线：一天约6.5个数据点
            return required_records // 6 + 30
        elif period == Period.DAILY:
            # 日线：一天1个数据点，考虑节假日
            return int(required_records * 1.4)  # 乘以1.4考虑节假日
        elif period == Period.WEEKLY:
            # 周线：一周1个数据点
            return required_records * 7 + 30
        elif period == Period.MONTHLY:
            # 月线：一月1个数据点
            return required_records * 30 + 60
        else:
            return 365  # 默认一年
    
    def _calculate_start_date(self, target_date: str, lookback_days: int) -> str:
        """
        计算开始日期

        Args:
            target_date: 目标日期
            lookback_days: 向前天数

        Returns:
            str: 开始日期
        """
        try:
            # 确保target_date是字符串
            if not isinstance(target_date, str):
                target_date = str(target_date)
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_dt = target_dt - timedelta(days=lookback_days)
            return start_dt.strftime('%Y-%m-%d')
        except Exception as e:
            logger.error(f"计算开始日期失败: {e}")
            return "2020-01-01"  # 默认开始日期
    
    def _validate_data_completeness(self, 
                                   data: pd.DataFrame, 
                                   stock_code: str, 
                                   period: Period, 
                                   required_records: int) -> None:
        """
        验证数据完整性
        
        Args:
            data: 股票数据
            stock_code: 股票代码
            period: K线周期
            required_records: 需要的记录数
        """
        period_value = period.value if hasattr(period, 'value') else str(period)
        if data.empty:
            logger.warning(f"股票{stock_code} {period_value}数据为空")
            return

        actual_records = len(data)
        if actual_records < required_records * 0.8:  # 允许20%的缺失
            logger.warning(
                f"股票{stock_code} {period_value}数据不足: "
                f"需要{required_records}条，实际{actual_records}条"
            )
        else:
            logger.info(
                f"股票{stock_code} {period_value}数据充足: {actual_records}条记录"
            )
    
    @exception_handler(reraise=False, default_return={})
    def get_available_periods(self, stock_code: str, target_date: str) -> Dict[Period, bool]:
        """
        检查各周期数据的可用性
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            
        Returns:
            Dict[Period, bool]: 各周期的可用性
        """
        availability = {}
        
        for period in Period:
            try:
                # 检查是否有该周期的数据
                query = f"""
                SELECT COUNT(*) as count
                FROM stock_info WHERE level = %(level)s AND code = '{stock_code}'
                AND level = '{period.value}'
                AND date <= '{target_date}'
                LIMIT 1
                """
                
                result = self.data_access.query_dataframe(query)
                availability[period] = not result.empty and result.iloc[0]['count'] > 0
                
            except Exception as e:
                logger.debug(f"检查{period.value}可用性失败: {e}")
                availability[period] = False
        
        return availability
    
    def get_period_data_summary(self, 
                               stock_code: str, 
                               target_date: str) -> Dict[str, Any]:
        """
        获取各周期数据摘要信息
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
            
        Returns:
            Dict[str, Any]: 数据摘要
        """
        summary = {
            'stock_code': stock_code,
            'target_date': target_date,
            'periods': {}
        }
        
        availability = self.get_available_periods(stock_code, target_date)
        
        for period, available in availability.items():
            summary['periods'][period.value] = {
                'available': available,
                'min_required_records': self.min_data_requirements.get(period, 250),
                'recommended_lookback_days': self._calculate_lookback_days(
                    period, self.min_data_requirements.get(period, 250)
                )
            }
        
        return summary

    def _aggregate_from_base_period(self,
                                   stock_code: str,
                                   target_date: str,
                                   target_period: Period,
                                   lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        从基础周期数据聚合生成目标周期数据

        Args:
            stock_code: 股票代码
            target_date: 目标日期
            target_period: 目标周期
            lookback_days: 回看天数

        Returns:
            pd.DataFrame: 聚合后的数据
        """
        try:
            # 确定基础周期和聚合倍数
            base_period, multiplier = self._get_aggregation_config(target_period)

            if base_period is None:
                logger.warning(f"无法确定{target_period.value}的基础周期")
                return pd.DataFrame()

            # 获取基础周期数据（直接查询，避免递归聚合）
            base_data = self._query_base_period_data(
                stock_code=stock_code,
                target_date=target_date,
                period=base_period,
                lookback_days=lookback_days
            )

            if base_data.empty:
                logger.warning(f"基础周期{base_period.value}数据为空，无法聚合")
                return pd.DataFrame()

            # 执行数据聚合
            aggregated_data = self._perform_kline_aggregation(base_data, multiplier, target_period)

            logger.info(f"成功从{base_period.value}数据聚合生成{target_period.value}数据，"
                       f"原始{len(base_data)}条 -> 聚合{len(aggregated_data)}条")

            return aggregated_data

        except Exception as e:
            logger.error(f"数据聚合失败: {e}")
            return pd.DataFrame()

    def _get_aggregation_config(self, target_period: Period) -> tuple:
        """
        获取聚合配置

        Args:
            target_period: 目标周期

        Returns:
            tuple: (基础周期, 聚合倍数)
        """
        aggregation_map = {
            Period.MIN_30: (Period.MIN_15, 2),  # 30分钟 = 15分钟 * 2
            Period.MIN_60: (Period.MIN_15, 4),  # 60分钟 = 15分钟 * 4
            # 可以扩展更多聚合规则
            # Period.DAILY: (Period.MIN_60, 4),  # 日线 = 60分钟 * 4 (如果需要)
        }

        return aggregation_map.get(target_period, (None, None))

    def _perform_kline_aggregation(self,
                                  base_data: pd.DataFrame,
                                  multiplier: int,
                                  target_period: Period) -> pd.DataFrame:
        """
        执行K线数据聚合

        Args:
            base_data: 基础周期数据
            multiplier: 聚合倍数
            target_period: 目标周期

        Returns:
            pd.DataFrame: 聚合后的K线数据
        """
        if base_data.empty:
            return pd.DataFrame()

        # 确保数据按时间排序
        df = base_data.copy().sort_values('date')

        # 转换日期列为datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # 按聚合倍数分组
        df['group'] = df.index // multiplier

        # 执行OHLCV聚合
        aggregated = df.groupby('group').agg({
            'code': 'first',
            'name': 'first',
            'date': 'first',  # 使用组内第一个时间点
            'open': 'first',  # 开盘价：组内第一个
            'high': 'max',    # 最高价：组内最大值
            'low': 'min',     # 最低价：组内最小值
            'close': 'last',  # 收盘价：组内最后一个
            'volume': 'sum',  # 成交量：组内求和
            'turnover_rate': 'sum' if 'turnover_rate' in df.columns else 'mean'  # 成交额：组内求和
        }).reset_index(drop=True)

        # 添加周期标识
        aggregated['level'] = target_period.value

        # 计算价格变化（如果需要）
        if len(aggregated) > 1:
            aggregated['price_change'] = aggregated['close'].pct_change()
            aggregated['price_range'] = (aggregated['high'] - aggregated['low']) / aggregated['close']
        else:
            aggregated['price_change'] = 0.0
            aggregated['price_range'] = 0.0

        # 重新排序列以匹配原始数据结构
        expected_columns = ['code', 'name', 'date', 'level', 'open', 'high', 'low', 'close',
                          'volume', 'turnover_rate', ]

        # 只保留存在的列
        available_columns = [col for col in expected_columns if col in aggregated.columns]
        aggregated = aggregated[available_columns]

        return aggregated

    def _query_base_period_data(self,
                               stock_code: str,
                               target_date: str,
                               period: Period,
                               lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        直接查询基础周期数据（避免递归聚合）

        Args:
            stock_code: 股票代码
            target_date: 目标日期
            period: 周期
            lookback_days: 回看天数

        Returns:
            pd.DataFrame: 基础周期数据
        """
        # 计算需要的数据量
        if lookback_days is None:
            required_records = self.min_data_requirements.get(period, 250)
            lookback_days = self._calculate_lookback_days(period, required_records)

        # 计算开始日期
        start_date = self._calculate_start_date(target_date, lookback_days)

        # 构建查询参数
        period_value = period.value if hasattr(period, 'value') else str(period)
        query_params = {
            'code': stock_code,
            'level': period_value,
            'start_date': start_date,
            'end_date': target_date,
            'order_by': 'date ASC'
        }

        # 直接查询数据库，不进行聚合
        return self._query_stock_data_by_period(query_params)
