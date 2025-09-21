# VnPy ClickHouse数据库接口

## 简介

vnpy_clickhouse是VnPy量化交易平台的ClickHouse数据库接口模块。ClickHouse是一个高性能的列式数据库管理系统，特别适合OLAP场景和大数据量的时序数据分析。

## 特性

### 🚀 **高性能**
- 列式存储，查询速度快
- 优秀的数据压缩率
- 支持并行查询处理
- 适合大数据量场景

### 📊 **时序数据优化**
- 专为时序数据设计的表结构
- 高效的时间范围查询
- 支持数据分区和索引优化

### 🔧 **完整功能**
- 支持K线数据(BarData)存储和查询
- 支持Tick数据(TickData)存储和查询
- 提供数据概览功能
- 支持数据删除和管理

### 🛡️ **可靠性**
- 严格遵循VnPy数据库接口规范
- 完善的错误处理机制
- 支持事务和数据一致性

## 安装

### 1. 安装ClickHouse服务

#### Docker方式（推荐）
```bash
# 拉取ClickHouse镜像
docker pull clickhouse/clickhouse-server:23.8-alpine

# 启动ClickHouse容器
docker run -d \
  --name vnpy-clickhouse \
  -p 8123:8123 \
  -p 9000:9000 \
  -p 9004:9004 \
  clickhouse/clickhouse-server:23.8-alpine
```

#### 本地安装
请参考[ClickHouse官方文档](https://clickhouse.com/docs/en/install)

### 2. 安装Python依赖

```bash
pip install clickhouse-connect
```

### 3. 配置VnPy

在VnPy配置文件`vt_setting.json`中添加ClickHouse配置：

```json
{
    "database.name": "clickhouse",
    "database.host": "localhost",
    "database.port": 8123,
    "database.user": "default",
    "database.password": "123456",
    "database.database": "stock",
    "database.timezone": "Asia/Shanghai"
}
```

**重要**: 本模块使用您现有的`stock`数据库和`stock_info`表，无需创建新的数据库。

## 使用方法

### 基本使用

```python
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData, TickData
from vnpy.trader.constant import Exchange, Interval
from datetime import datetime

# 获取数据库实例
database = get_database()

# 保存K线数据
bars = [
    BarData(
        symbol="000001",
        exchange=Exchange.SSE,
        datetime=datetime.now(),
        interval=Interval.MINUTE,
        volume=1000,
        turnover=10000,
        open_price=10.0,
        high_price=10.5,
        low_price=9.8,
        close_price=10.2,
        open_interest=0,
        gateway_name="test"
    )
]
database.save_bar_data(bars)

# 查询K线数据
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
bars = database.load_bar_data("000001", Exchange.SSE, Interval.MINUTE, start, end)

# 获取数据概览
bar_overview = database.get_bar_overview()
tick_overview = database.get_tick_overview()
```

### 高级功能

#### 批量数据导入
```python
# 批量保存大量数据
large_bars = []  # 大量K线数据
database.save_bar_data(large_bars)
```

#### 数据管理
```python
# 删除指定数据
deleted_count = database.delete_bar_data("000001", Exchange.SSE, Interval.MINUTE)
print(f"删除了 {deleted_count} 条记录")

# 查看数据概览
overviews = database.get_bar_overview()
for overview in overviews:
    print(f"{overview.symbol}.{overview.exchange.value}: {overview.count} 条记录")
```

## 数据库表结构

### stock_info (现有K线数据表) ⭐
本模块直接使用您现有的`stock.stock_info`表，包含完整的股票历史数据：

| 字段 | 类型 | 说明 | VnPy映射 |
|------|------|------|----------|
| code | String | 股票代码 | symbol |
| name | String | 股票名称 | name |
| date | Date | 交易日期 | - |
| level | String | 时间周期 | interval |
| open | Float64 | 开盘价 | open_price |
| close | Float64 | 收盘价 | close_price |
| high | Float64 | 最高价 | high_price |
| low | Float64 | 最低价 | low_price |
| volume | Float64 | 成交量 | volume |
| turnover_rate | Float64 | 换手率 | - |
| price_change | Float64 | 价格变动 | - |
| price_range | Float64 | 涨跌幅 | - |
| industry | String | 行业 | - |
| datetime | DateTime | 时间戳 | datetime |
| seq | UInt32 | 序列号 | - |

**数据统计**:
- 📊 **总记录数**: 19,608,204 条
- 🏢 **股票数量**: 4,378 只
- 📅 **时间范围**: 1990-12-19 ~ 2025-05-23
- ⏰ **时间周期**: 日线(1382万)、周线(292万)、15分钟(215万)、月线(69万)
- 🏛️ **交易所分布**: 深交所(2294只)、上交所(1648只)

### 时间周期映射
| stock_info.level | VnPy.Interval | 说明 |
|------------------|---------------|------|
| 1分钟 | MINUTE | 1分钟K线 |
| 5分钟 | MINUTE_5 | 5分钟K线 |
| 15分钟 | MINUTE_15 | 15分钟K线 |
| 30分钟 | MINUTE_30 | 30分钟K线 |
| 1小时 | HOUR | 1小时K线 |
| 日线 | DAILY | 日K线 |
| 周线 | WEEKLY | 周K线 |
| 月线 | MONTHLY | 月K线 |

## 性能优化

### 1. 索引优化
- 主键索引：(symbol, exchange, interval, datetime)
- 支持高效的时间范围查询
- 自动优化查询计划

### 2. 数据分区
```sql
-- 可以按时间分区优化查询性能
PARTITION BY toYYYYMM(datetime)
```

### 3. 批量操作
- 建议批量插入数据以提高性能
- 单次插入建议不超过10万条记录

## 注意事项

### 1. 时区处理
- 所有时间数据统一转换为配置的时区
- 建议使用UTC或本地时区

### 2. 数据类型
- 所有价格和数量字段使用Float64类型
- 时间字段支持毫秒精度

### 3. 连接管理
- 自动管理数据库连接
- 支持连接池和重连机制

### 4. 错误处理
- 完善的异常处理机制
- 详细的错误日志记录

## 故障排除

### 常见问题

1. **连接失败**
   - 检查ClickHouse服务是否启动
   - 验证连接参数是否正确
   - 确认网络连接正常

2. **权限错误**
   - 检查用户名和密码
   - 确认用户有数据库操作权限

3. **性能问题**
   - 检查索引是否正确创建
   - 考虑数据分区策略
   - 优化查询条件

### 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 版本历史

- **v1.0.0**: 初始版本，支持基本的CRUD操作
- 严格遵循VnPy数据库接口规范
- 支持K线和Tick数据的完整生命周期管理

## 贡献

欢迎提交Issue和Pull Request来改进这个模块。

## 许可证

MIT License
