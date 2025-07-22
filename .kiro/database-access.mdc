# 数据库访问规范

## 强制性规则

### 1. 禁止直接数据库访问

**❌ 错误方式 - 禁止使用**:
```python
# 直接使用数据库连接
from db.clickhouse_db import get_clickhouse_db

conn = get_clickhouse_db()
data = conn.query_dataframe("SELECT * FROM stock_info WHERE code = '000001'")
```

**✅ 正确方式 - 必须使用**:
```python
# 使用统一查询执行器
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

query_executor = get_query_executor()
data = query_executor.execute_query(
    QueryType.STOCK_DATA,
    {'code': '000001', 'start_date': '2024-01-01'}
)
```

### 2. 统一查询接口

**核心接口**: [db/query_executor.py](mdc:db/query_executor.py)
```python
from db.query_executor import get_query_executor

# 获取查询执行器实例
query_executor = get_query_executor()

# 执行标准查询
data = query_executor.execute_query(QueryType.STOCK_DATA, params)

# 获取股票列表
stocks = query_executor.get_stock_list()

# 获取股票数据
stock_data = query_executor.get_stock_data({'code': '000001'})

# 获取股票数量
count = query_executor.get_stock_count()
```

### 3. 查询类型管理

**使用预定义查询类型**: [db/sql_manager.py](mdc:db/sql_manager.py)
```python
from db.sql_manager import QueryType

# 可用的查询类型
QueryType.STOCK_DATA          # 股票数据查询
QueryType.STOCK_LIST          # 股票列表查询
QueryType.STOCK_COUNT         # 股票数量查询
QueryType.INDICATOR_DATA      # 指标数据查询
QueryType.CUSTOM              # 自定义查询
```

## 数据访问模式

### 1. 标准查询模式

**股票数据查询**:
```python
def get_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取股票数据"""
    query_executor = get_query_executor()
    
    params = {
        'code': code,
        'start_date': start_date,
        'end_date': end_date,
        'level': '日线'
    }
    
    return query_executor.execute_query(QueryType.STOCK_DATA, params)
```

**指标数据查询**:
```python
def get_indicator_data(code: str, indicator_type: str) -> pd.DataFrame:
    """获取指标数据"""
    query_executor = get_query_executor()
    
    params = {
        'code': code,
        'indicator_type': indicator_type
    }
    
    return query_executor.execute_query(QueryType.INDICATOR_DATA, params)
```

### 2. 批量查询模式

**批量股票数据**:
```python
def get_batch_stock_data(codes: List[str], date: str) -> pd.DataFrame:
    """批量获取股票数据"""
    query_executor = get_query_executor()
    
    params = {
        'codes': codes,
        'date': date,
        'level': '日线'
    }
    
    return query_executor.execute_query(QueryType.BATCH_STOCK_DATA, params)
```

### 3. 分页查询模式

**大数据量查询**:
```python
def get_large_dataset(params: Dict[str, Any], page_size: int = 10000) -> Iterator[pd.DataFrame]:
    """分页获取大数据集"""
    query_executor = get_query_executor()
    offset = 0
    
    while True:
        page_params = {
            **params,
            'limit': page_size,
            'offset': offset
        }
        
        data = query_executor.execute_query(QueryType.PAGINATED_DATA, page_params)
        
        if data.empty:
            break
            
        yield data
        offset += page_size
```

## 参数验证规范

### 1. 必要参数检查

```python
def validate_stock_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """验证股票查询参数"""
    required_fields = ['code']
    
    for field in required_fields:
        if field not in params or not params[field]:
            raise ValueError(f"缺少必要参数: {field}")
    
    # 股票代码格式验证
    if not re.match(r'^\d{6}$', params['code']):
        raise ValueError(f"股票代码格式错误: {params['code']}")
    
    return params
```

### 2. 日期参数处理

```python
def validate_date_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """验证日期参数"""
    from utils.date_utils import parse_date, format_date
    
    # 处理日期格式
    if 'start_date' in params:
        params['start_date'] = format_date(parse_date(params['start_date']))
    
    if 'end_date' in params:
        params['end_date'] = format_date(parse_date(params['end_date']))
    
    # 验证日期范围
    if 'start_date' in params and 'end_date' in params:
        if params['start_date'] > params['end_date']:
            raise ValueError("开始日期不能大于结束日期")
    
    return params
```

## 错误处理规范

### 1. 数据库连接错误

```python
def handle_database_error(func):
    """数据库错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"数据库连接失败: {e}")
            # 尝试重连
            return retry_with_backoff(func, *args, **kwargs)
        except Exception as e:
            logger.error(f"数据库操作失败: {e}")
            raise
    return wrapper
```

### 2. 查询错误处理

```python
def safe_query(query_type: QueryType, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """安全查询执行"""
    try:
        query_executor = get_query_executor()
        result = query_executor.execute_query(query_type, params)
        
        if result.empty:
            logger.warning(f"查询无结果: {query_type}, {params}")
            return None
        
        return result
        
    except Exception as e:
        logger.error(f"查询执行失败: {query_type}, {params}, {e}")
        return None
```

## 性能优化规范

### 1. 查询缓存

```python
from functools import lru_cache
from utils.cache import cache_result

@cache_result(ttl=300)  # 5分钟缓存
def get_cached_stock_data(code: str, date: str) -> pd.DataFrame:
    """获取缓存的股票数据"""
    query_executor = get_query_executor()
    return query_executor.get_stock_data({'code': code, 'date': date})
```

### 2. 连接池管理

```python
def get_optimized_query_executor():
    """获取优化的查询执行器"""
    # 使用连接池配置
    config = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 3600
    }
    
    return get_query_executor(config)
```

### 3. 批量操作优化

```python
def batch_insert_data(data_list: List[Dict[str, Any]], batch_size: int = 1000):
    """批量插入数据"""
    query_executor = get_query_executor()
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        query_executor.batch_insert(batch)
```

## 数据质量保证

### 1. 数据验证

```python
def validate_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """验证股票数据质量"""
    # 检查必要列
    required_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"缺少必要列: {missing_columns}")
    
    # 检查数据类型
    if not pd.api.types.is_numeric_dtype(data['close']):
        raise ValueError("收盘价必须是数值类型")
    
    # 检查数据范围
    if (data['close'] <= 0).any():
        raise ValueError("收盘价必须大于0")
    
    return data
```

### 2. 数据清洗

```python
def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """清洗股票数据"""
    # 去除重复数据
    data = data.drop_duplicates(subset=['code', 'date'])
    
    # 处理缺失值
    data = data.dropna(subset=['close'])
    
    # 排序
    data = data.sort_values(['code', 'date'])
    
    return data
```

## 监控和日志

### 1. 查询监控

```python
def monitor_query_performance(func):
    """查询性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"查询执行成功: {func.__name__}, 耗时: {execution_time:.2f}s")
            
            # 记录性能指标
            if execution_time > 5.0:  # 超过5秒的慢查询
                logger.warning(f"慢查询告警: {func.__name__}, 耗时: {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"查询执行失败: {func.__name__}, 耗时: {execution_time:.2f}s, 错误: {e}")
            raise
    
    return wrapper
```

### 2. 数据访问日志

```python
def log_data_access(query_type: QueryType, params: Dict[str, Any], result_count: int):
    """记录数据访问日志"""
    logger.info(f"数据访问: {query_type}, 参数: {params}, 结果数量: {result_count}")
    
    # 记录到访问日志文件
    access_logger = get_logger('data_access')
    access_logger.info(f"{query_type}|{params}|{result_count}")
```

## 降级处理策略

### 1. 查询降级

```python
def query_with_fallback(query_type: QueryType, params: Dict[str, Any]) -> pd.DataFrame:
    """带降级处理的查询"""
    try:
        # 尝试使用统一查询接口
        query_executor = get_query_executor()
        return query_executor.execute_query(query_type, params)
        
    except Exception as e:
        logger.warning(f"统一查询接口失败，使用降级方案: {e}")
        
        # 降级到直接数据库连接
        return fallback_direct_query(query_type, params)
```

### 2. 数据降级

```python
def get_data_with_fallback(code: str, date: str) -> pd.DataFrame:
    """带降级的数据获取"""
    # 尝试获取实时数据
    try:
        return get_realtime_data(code, date)
    except Exception:
        logger.warning("实时数据获取失败，使用历史数据")
        return get_historical_data(code, date)
```
description:
globs:
alwaysApply: true
---
