"""
统一查询构建器
基于统一字段映射，解决SQL查询分散化问题
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from config.unified_field_mapping import get_field_mapper, DatabaseType


class QueryType(Enum):
    """查询类型枚举"""
    STOCK_BASIC = "stock_basic"
    STOCK_RANGE = "stock_range"
    STOCK_BATCH = "stock_batch"
    STOCK_INDICATOR = "stock_indicator"
    INDUSTRY_LIST = "industry_list"
    STOCK_LIST = "stock_list"


@dataclass
class QueryCondition:
    """查询条件"""
    field: str
    operator: str  # =, >, <, >=, <=, IN, LIKE, BETWEEN
    value: Any
    logical_operator: str = "AND"  # AND, OR


@dataclass
class QueryRequest:
    """查询请求"""
    query_type: QueryType
    fields: List[str]
    conditions: List[QueryCondition]
    order_by: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class UnifiedQueryBuilder:
    """统一查询构建器"""
    
    def __init__(self, db_type: DatabaseType = DatabaseType.CLICKHOUSE):
        self.db_type = db_type
        self.field_mapper = get_field_mapper()
        self.table_name = "stock_info"  # 默认表名
    
    def build_stock_basic_query(self, stock_code: str, start_date: str, 
                               end_date: str, level: str = "日线",
                               fields: Optional[List[str]] = None) -> str:
        """
        构建股票基础数据查询
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            fields: 查询字段列表
            
        Returns:
            str: SQL查询语句
        """
        if fields is None:
            fields = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        # 添加turnover_rate字段（如果可用）
        if 'turnover_rate' not in fields:
            fields.append('turnover_rate')
        
        # 构建SELECT子句
        select_clause = self.field_mapper.build_select_clause(fields, self.db_type)
        
        # 构建WHERE条件
        conditions = [
            QueryCondition('code', '=', stock_code),
            QueryCondition('level', '=', level),
            QueryCondition('date', '>=', start_date),
            QueryCondition('date', '<=', end_date)
        ]
        
        where_clause = self._build_where_clause(conditions)
        
        # 构建完整查询
        query = f"""
        SELECT {select_clause}
        FROM {self.table_name}
        WHERE {where_clause}
        ORDER BY {self.field_mapper.get_database_field('date', self.db_type)} ASC
        """
        
        return self._clean_query(query)
    
    def build_stock_batch_query(self, stock_codes: List[str], start_date: str,
                               end_date: str, level: str = "日线",
                               fields: Optional[List[str]] = None,
                               batch_size: int = 100) -> List[str]:
        """
        构建批量股票数据查询
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            level: 数据级别
            fields: 查询字段列表
            batch_size: 批次大小
            
        Returns:
            List[str]: 批量查询语句列表
        """
        if fields is None:
            fields = ['code', 'name', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover_rate']
        
        queries = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i + batch_size]
            
            # 构建SELECT子句
            select_clause = self.field_mapper.build_select_clause(fields, self.db_type)
            
            # 构建IN条件
            codes_str = "', '".join(batch_codes)
            
            # 构建WHERE条件
            conditions = [
                QueryCondition('code', 'IN', f"('{codes_str}')"),
                QueryCondition('level', '=', level),
                QueryCondition('date', '>=', start_date),
                QueryCondition('date', '<=', end_date)
            ]
            
            where_clause = self._build_where_clause(conditions)
            
            query = f"""
            SELECT {select_clause}
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY {self.field_mapper.get_database_field('code', self.db_type)}, 
                     {self.field_mapper.get_database_field('date', self.db_type)} ASC
            """
            
            queries.append(self._clean_query(query))
        
        return queries
    
    def build_industry_list_query(self) -> str:
        """构建行业列表查询"""
        industry_field = self.field_mapper.get_database_field(self.db_type)
        
        query = f"""
        SELECT DISTINCT {industry_field} as industry
        FROM {self.table_name}
        WHERE {industry_field} IS NOT NULL 
        AND {industry_field} != ''
        """
        
        return self._clean_query(query)
    
    def build_stock_list_query(self: Optional[str] = None,
                              limit: Optional[int] = None) -> str:
        """
        构建股票列表查询
        
        Args:
            industry: 行业筛选
            limit: 限制数量
            
        Returns:
            str: SQL查询语句
        """
        code_field = self.field_mapper.get_database_field('code', self.db_type)
        name_field = self.field_mapper.get_database_field('name', self.db_type)
        industry_field = self.field_mapper.get_database_field(self.db_type)
        
        select_clause = f"DISTINCT {code_field} as code, {name_field} as name"
        if industry:
            select_clause += f", {industry_field} as industry"
        
        conditions = []
        if industry:
            conditions.append(QueryCondition('='))
        
        where_clause = self._build_where_clause(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT {select_clause}
        FROM {self.table_name}
        WHERE {where_clause}
        ORDER BY {code_field}
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self._clean_query(query)
    
    def build_custom_query(self, request: QueryRequest) -> str:
        """
        构建自定义查询
        
        Args:
            request: 查询请求
            
        Returns:
            str: SQL查询语句
        """
        # 构建SELECT子句
        select_clause = self.field_mapper.build_select_clause(request.fields, self.db_type)
        
        # 构建WHERE子句
        where_clause = self._build_where_clause(request.conditions)
        
        # 构建ORDER BY子句
        order_clause = ""
        if request.order_by:
            order_fields = []
            for field in request.order_by:
                db_field = self.field_mapper.get_database_field(field, self.db_type)
                order_fields.append(db_field)
            order_clause = f"ORDER BY {', '.join(order_fields)}"
        
        # 构建LIMIT子句
        limit_clause = ""
        if request.limit:
            limit_clause = f"LIMIT {request.limit}"
            if request.offset:
                limit_clause += f" OFFSET {request.offset}"
        
        # 组装完整查询
        query_parts = [
            f"SELECT {select_clause}",
            f"FROM {self.table_name}",
            f"WHERE {where_clause}" if where_clause else "",
            order_clause,
            limit_clause
        ]
        
        query = "\n".join(part for part in query_parts if part)
        return self._clean_query(query)
    
    def _build_where_clause(self, conditions: List[QueryCondition]) -> str:
        """构建WHERE子句"""
        if not conditions:
            return "1=1"
        
        where_parts = []
        for i, condition in enumerate(conditions):
            # 获取数据库字段名
            db_field = self.field_mapper.get_database_field(condition.field, self.db_type)
            
            # 构建条件
            if condition.operator == 'IN':
                condition_str = f"{db_field} IN {condition.value}"
            elif condition.operator == 'BETWEEN':
                if isinstance(condition.value, (list, tuple)) and len(condition.value) == 2:
                    condition_str = f"{db_field} BETWEEN '{condition.value[0]}' AND '{condition.value[1]}'"
                else:
                    raise ValueError(f"BETWEEN操作符需要两个值: {condition.value}")
            elif condition.operator == 'LIKE':
                condition_str = f"{db_field} LIKE '{condition.value}'"
            else:
                condition_str = f"{db_field} {condition.operator} '{condition.value}'"
            
            # 添加逻辑操作符
            if i > 0:
                where_parts.append(f" {condition.logical_operator} {condition_str}")
            else:
                where_parts.append(condition_str)
        
        return "".join(where_parts)
    
    def _clean_query(self, query: str) -> str:
        """清理查询语句"""
        # 移除多余的空白字符
        lines = [line.strip() for line in query.split('\n') if line.strip()]
        return ' '.join(lines)
    
    def get_available_fields(self) -> List[str]:
        """获取可用字段列表"""
        return list(self.field_mapper._field_mappings.keys())
    
    def validate_fields(self, fields: List[str]) -> Dict[str, bool]:
        """验证字段有效性"""
        return {field: field in self.field_mapper._field_mappings for field in fields}


# 全局查询构建器实例
unified_query_builder = UnifiedQueryBuilder()


def get_query_builder(db_type: DatabaseType = DatabaseType.CLICKHOUSE) -> UnifiedQueryBuilder:
    """获取统一查询构建器实例"""
    return UnifiedQueryBuilder(db_type)
