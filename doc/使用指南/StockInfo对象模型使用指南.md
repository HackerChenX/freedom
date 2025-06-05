# StockInfo对象模型使用指南

## 简介

StockInfo是一个标准化的股票数据结构和相关操作方法的对象模型，作为数据库层和业务逻辑层之间的桥梁。它提供了统一的数据结构和访问接口，支持单条数据和批量数据的处理。

## 基本用法

### 导入模块

```python
from models.stock_info import StockInfo
from db.clickhouse_db import get_clickhouse_db
from enums.period import Period
```

### 创建空的StockInfo对象

```python
# 创建空对象
stock = StockInfo()
stock.code = '000001'
stock.name = '平安银行'
stock.level = '日线'
stock.close = 10.5
print(stock)  # 输出：StockInfo(code=000001, name=平安银行, date=None, close=10.5)
```

### 从字典创建StockInfo对象

```python
# 从字典创建
data_dict = {
    'code': '000001',
    'name': '平安银行',
    'date': '2023-01-01',
    'level': '日线',
    'open': 10.5,
    'high': 11.2,
    'low': 10.1,
    'close': 10.8,
    'volume': 123456,
    'turnover_rate': 2.5,
    'price_change': 0.3,
    'price_range': 1.1,
    'industry': '银行',
    'datetime': '2023-01-01 15:00:00',
    'seq': 1
}

stock = StockInfo(data_dict)
print(f"股票代码: {stock.code}")
print(f"股票名称: {stock.name}")
print(f"收盘价: {stock.close}")
print(f"日期时间: {stock.datetime_value}")
```

### 从数据库获取StockInfo对象

```python
# 获取数据库连接
db = get_clickhouse_db()

# 获取股票数据
stock_code = '000001'  # 平安银行
level = Period.DAILY
start_date = '20230101'
end_date = '20230131'

# 返回的是StockInfo对象，可能是单个对象或集合
stock_info = db.get_stock_info(stock_code, level, start_date, end_date)

# 判断是否为集合
if stock_info.is_collection:
    print(f"获取到 {len(stock_info)} 条数据")
    
    # 访问第一条数据
    first_item = stock_info[0]
    print(f"第一条数据: {first_item}")
    print(f"  收盘价: {first_item.close}")
    
    # 遍历所有数据
    for i, item in enumerate(stock_info):
        print(f"第 {i+1} 条数据: {item.date}, 收盘价: {item.close}")
else:
    # 单条数据的情况
    print(f"股票信息: {stock_info}")
    print(f"收盘价: {stock_info.close}")
```

### 将StockInfo对象转换为DataFrame

```python
# 获取股票数据并转换为DataFrame
stock_info = db.get_stock_info('000001', Period.DAILY, '20230101', '20230131')
df = stock_info.to_dataframe()

# 使用pandas进行后续操作
print(f"DataFrame形状: {df.shape}")
print(f"DataFrame头部数据:\n{df.head()}")

# 计算均值等统计量
print(f"收盘价均值: {df['close'].mean()}")
print(f"收盘价最大值: {df['close'].max()}")
print(f"收盘价最小值: {df['close'].min()}")
```

## 字段说明

StockInfo对象包含以下字段：

| 字段名 | 类型 | 说明 |
| ----- | ----- | ----- |
| code | str | 股票代码 |
| name | str | 股票名称 |
| date | datetime.date | 日期 |
| level | str | K线周期，如'日线'、'周线'等 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| turnover_rate | float | 换手率 |
| price_change | float | 价格变动 |
| price_range | float | 价格区间 |
| industry | str | 行业 |
| datetime_value | datetime.datetime | 日期时间 |
| seq | int | 序号 |
| is_collection | bool | 是否为集合 |

## 集合操作

StockInfo对象可以包含多条数据，形成一个集合。集合支持以下操作：

### 从DataFrame创建集合

```python
import pandas as pd

# 创建DataFrame
df = pd.DataFrame([
    {'code': '000001', 'name': '平安银行', 'date': '2023-01-01', 'close': 10.5},
    {'code': '000001', 'name': '平安银行', 'date': '2023-01-02', 'close': 10.8},
    {'code': '000001', 'name': '平安银行', 'date': '2023-01-03', 'close': 11.0}
])

# 创建StockInfo集合
stock_collection = StockInfo(df)
print(f"集合大小: {len(stock_collection)}")
```

### 访问集合中的元素

```python
# 通过索引访问
first_item = stock_collection[0]
print(f"第一条数据: {first_item}")

# 遍历集合
for item in stock_collection:
    print(f"数据: {item}")
```

### 将集合转换为字典列表

```python
# 转换为字典列表
dict_list = stock_collection.to_dicts()
print(f"字典列表第一项: {dict_list[0]}")
```

### 将集合转换为DataFrame

```python
# 转换为DataFrame
df = stock_collection.to_dataframe()
print(f"DataFrame形状: {df.shape}")
```

## 最佳实践

1. **统一数据结构**：使用StockInfo对象作为股票数据的标准结构，确保在各模块之间传递数据时保持一致。

2. **类型安全**：StockInfo类提供了类型安全的访问方式，避免直接操作DataFrame时可能出现的类型错误。

3. **批量数据处理**：对于大量数据，可以使用StockInfo集合模式进行批量处理，也可以方便地转换回DataFrame进行高性能计算。

4. **结构化操作**：利用StockInfo的属性和方法进行结构化操作，使代码更易于理解和维护。

5. **兼容性**：StockInfo支持从多种数据源构造，包括字典、DataFrame等，具有良好的兼容性。

## 注意事项

1. **字段名映射**：StockInfo内部实现了字段名映射，可以处理不同来源的数据字段名差异。

2. **集合与单条数据**：使用is_collection属性区分是集合还是单条数据。

3. **日期时间字段**：日期字段(date)和日期时间字段(datetime_value)分别存储日期和时间信息。

4. **数值类型转换**：所有数值字段会自动进行类型转换，处理可能的类型错误。

5. **空值处理**：StockInfo对象对空值进行了处理，确保不会因为空值导致错误。 