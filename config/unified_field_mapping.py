"""
统一字段映射配置中心
解决turnover_rate等字段分散化问题
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from db.sql_manager import SQLManager, QueryType


class DatabaseType(Enum):
    """数据库类型枚举"""
    CLICKHOUSE = "clickhouse"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


@dataclass
class FieldMapping:
    """字段映射配置"""
    standard_name: str          # 标准字段名
    database_fields: Dict[DatabaseType, str]  # 各数据库中的实际字段名
    aliases: List[str]          # 字段别名列表
    data_type: str             # 数据类型
    description: str           # 字段描述
    required: bool = True      # 是否必需字段


class UnifiedFieldMapper:
    """统一字段映射器"""
    
    def __init__(self):
        self._field_mappings = self._initialize_field_mappings()
        
    def _initialize_field_mappings(self) -> Dict[str, FieldMapping]:
        """初始化字段映射配置"""
        return {
            # 基础价格字段
            'open': FieldMapping(
                standard_name='open',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'open',
                    DatabaseType.MYSQL: 'open_price',
                    DatabaseType.POSTGRESQL: 'open'
                },
                aliases=['open_price', 'opening_price', 'o'],
                data_type='Float64',
                description='开盘价'
            ),
            
            'high': FieldMapping(
                standard_name='high',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'high',
                    DatabaseType.MYSQL: 'high_price',
                    DatabaseType.POSTGRESQL: 'high'
                },
                aliases=['high_price', 'highest_price', 'h'],
                data_type='Float64',
                description='最高价'
            ),
            
            'low': FieldMapping(
                standard_name='low',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'low',
                    DatabaseType.MYSQL: 'low_price',
                    DatabaseType.POSTGRESQL: 'low'
                },
                aliases=['low_price', 'lowest_price', 'l'],
                data_type='Float64',
                description='最低价'
            ),
            
            'close': FieldMapping(
                standard_name='close',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'close',
                    DatabaseType.MYSQL: 'close_price',
                    DatabaseType.POSTGRESQL: 'close'
                },
                aliases=['close_price', 'closing_price', 'c', 'price'],
                data_type='Float64',
                description='收盘价'
            ),
            
            # 成交量字段
            'volume': FieldMapping(
                standard_name='volume',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'volume',
                    DatabaseType.MYSQL: 'volume',
                    DatabaseType.POSTGRESQL: 'volume'
                },
                aliases=['vol', 'trading_volume', 'trade_volume', 'v'],
                data_type='UInt64',
                description='成交量'
            ),
            
            # 换手率字段 - 关键修复点
            'turnover_rate': FieldMapping(
                standard_name='turnover_rate',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'turnover_rate',  # 实际数据库中的字段名
                    DatabaseType.MYSQL: 'turnover_rate',
                    DatabaseType.POSTGRESQL: 'turnover_rate'
                },
                aliases=['turnover_rate', 'turnover_rate_ratio', 'tr'],
                data_type='Float64',
                description='换手率',
                required=False  # 不是所有数据源都有此字段
            ),
            
            # 基础信息字段
            'code': FieldMapping(
                standard_name='code',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'code',
                    DatabaseType.MYSQL: 'stock_code',
                    DatabaseType.POSTGRESQL: 'code'
                },
                aliases=['stock_code', 'symbol', 'ticker'],
                data_type='String',
                description='股票代码'
            ),
            
            'name': FieldMapping(
                standard_name='name',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'name',
                    DatabaseType.MYSQL: 'stock_name',
                    DatabaseType.POSTGRESQL: 'name'
                },
                aliases=['stock_name', 'company_name'],
                data_type='String',
                description='股票名称'
            ),
            
            'date': FieldMapping(
                standard_name='date',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'date',
                    DatabaseType.MYSQL: 'trade_date',
                    DatabaseType.POSTGRESQL: 'date'
                },
                aliases=['trade_date', 'trading_date', 'dt', 'datetime'],
                data_type='Date',
                description='交易日期'
            ),
            
            'level': FieldMapping(
                standard_name='level',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'level',
                    DatabaseType.MYSQL: 'period',
                    DatabaseType.POSTGRESQL: 'level'
                },
                aliases=['period', 'timeframe', 'interval'],
                data_type='String',
                description='数据级别（日线/分钟线等）'
            ),
            
            # 扩展字段
            'price_change': FieldMapping(
                standard_name='price_change',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'price_change',
                    DatabaseType.MYSQL: 'change_amount',
                    DatabaseType.POSTGRESQL: 'price_change'
                },
                aliases=['change_amount', 'change', 'pct_change'],
                data_type='Float64',
                description='价格变化',
                required=False
            ),
            
            'industry': FieldMapping(
                standard_name='industry',
                database_fields={
                    DatabaseType.CLICKHOUSE: 'industry',
                    DatabaseType.MYSQL: 'industry_name',
                    DatabaseType.POSTGRESQL: 'industry'
                },
                aliases=['industry_name', 'sector'],
                data_type='String',
                description='行业分类',
                required=False
            )
        }
    
    def get_database_field(self, standard_field: str, 
                          db_type: DatabaseType = DatabaseType.CLICKHOUSE) -> str:
        """获取指定数据库中的实际字段名"""
        if standard_field not in self._field_mappings:
            return standard_field  # 如果没有映射，返回原字段名
        
        mapping = self._field_mappings[standard_field]
        return mapping.database_fields.get(db_type, standard_field)
    
    def get_standard_field(self, database_field: str, 
                          db_type: DatabaseType = DatabaseType.CLICKHOUSE) -> Optional[str]:
        """根据数据库字段名获取标准字段名"""
        for standard_name, mapping in self._field_mappings.items():
            # 检查数据库字段映射
            if mapping.database_fields.get(db_type) == database_field:
                return standard_name
            # 检查别名
            if database_field in mapping.aliases:
                return standard_name
        return None
    
    def get_required_fields(self) -> List[str]:
        """获取必需字段列表"""
        return [name for name, mapping in self._field_mappings.items() 
                if mapping.required]
    
    def get_optional_fields(self) -> List[str]:
        """获取可选字段列表"""
        return [name for name, mapping in self._field_mappings.items() 
                if not mapping.required]
    
    def build_select_clause(self, fields: List[str], 
                           db_type: DatabaseType = DatabaseType.CLICKHOUSE) -> str:
        """构建SELECT子句"""
        db_fields = []
        for field in fields:
            db_field = self.get_database_field(field, db_type)
            if db_field != field:
                # 需要别名
                db_fields.append(f"{db_field} as {field}")
            else:
                db_fields.append(db_field)
        
        return ", ".join(db_fields)
    
    def validate_fields(self, available_fields: List[str]) -> Dict[str, bool]:
        """验证字段可用性"""
        result = {}
        for standard_field in self._field_mappings:
            mapping = self._field_mappings[standard_field]
            # 检查是否有对应的数据库字段或别名
            found = any(
                field in available_fields 
                for field in [mapping.database_fields.get(DatabaseType.CLICKHOUSE, '')] + mapping.aliases
            )
            result[standard_field] = found
        
        return result


# 全局实例
unified_field_mapper = UnifiedFieldMapper()


def get_field_mapper() -> UnifiedFieldMapper:
    """获取统一字段映射器实例"""
    return unified_field_mapper
