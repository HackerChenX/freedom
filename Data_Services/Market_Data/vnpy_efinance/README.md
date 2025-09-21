# VnPy EFinance数据源

基于[efinance](https://github.com/Micro-sheep/efinance)库的VnPy数据源适配器，提供免费、开源的金融数据获取服务。

## 特性

- ✅ **完全免费**：无需注册，无API限制
- ✅ **多市场支持**：A股、美股、港股、期货
- ✅ **多周期支持**：1分钟到月线的完整周期
- ✅ **数据质量高**：实时更新，数据准确
- ✅ **易于使用**：一键配置，开箱即用

## 支持的市场

### 股票市场
- **A股**：上海证券交易所(SSE)、深圳证券交易所(SZSE)、北京证券交易所(BSE)
- **美股**：纳斯达克(NASDAQ)、纽约证券交易所(NYSE)
- **港股**：香港证券交易所(SEHK)

### 期货市场
- **金融期货**：中国金融期货交易所(CFFEX)
- **商品期货**：上海期货交易所(SHFE)、郑州商品交易所(CZCE)、大连商品交易所(DCE)
- **能源期货**：上海国际能源交易中心(INE)
- **新兴期货**：广州期货交易所(GFEX)

## 支持的时间周期

| 周期 | EFinance标识 | 说明 |
|------|------------|------|
| 1分钟 | 1 | 分钟级数据 |
| 5分钟 | 5 | 5分钟K线 |
| 15分钟 | 15 | 15分钟K线 |
| 30分钟 | 30 | 30分钟K线 |
| 60分钟 | 60 | 小时线 |
| 日线 | 101 | 日K线 |
| 周线 | 102 | 周K线 |
| 月线 | 103 | 月K线 |

## 安装

### 方法1：使用pip安装

```bash
pip install vnpy_efinance
```

### 方法2：从源码安装

```bash
git clone https://github.com/vnpy/vnpy_efinance.git
cd vnpy_efinance
pip install -e .
```

### 依赖要求

- Python 3.8+
- vnpy
- efinance>=0.5.5
- pandas>=1.0.0

## 配置

在VnPy的`vt_setting.json`配置文件中添加：

```json
{
    "datafeed.name": "efinance",
    "datafeed.username": "",
    "datafeed.password": ""
}
```

> **注意**：EFinance完全免费，无需用户名和密码，但仍需在配置中声明数据源名称。

## 使用示例

### 在VnPy中使用

```python
from vnpy.trader.engine import MainEngine
from vnpy.event import EventEngine
from vnpy_efinance import EfinanceDatafeed

# 创建主引擎
event_engine = EventEngine()
main_engine = MainEngine(event_engine)

# 初始化EFinance数据源
datafeed = EfinanceDatafeed()
if datafeed.init():
    print("✅ EFinance数据源初始化成功")
else:
    print("❌ EFinance数据源初始化失败")
```

### 直接使用数据源

```python
from datetime import datetime
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest
from vnpy_efinance import EfinanceDatafeed

# 创建数据源实例
datafeed = EfinanceDatafeed()
datafeed.init()

# 创建历史数据请求
req = HistoryRequest(
    symbol="000001",
    exchange=Exchange.SZSE,
    interval=Interval.DAILY,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31)
)

# 获取历史数据
bars = datafeed.query_bar_history(req)
print(f"获取到 {len(bars)} 条数据")
```

### 策略回测中使用

在VnPy的策略回测中，EFinance数据源会自动为策略提供历史数据：

```python
from vnpy_ctabacktester import BacktestingEngine
from vnpy.trader.constant import Interval
from datetime import datetime

# 创建回测引擎
engine = BacktestingEngine()

# 设置回测参数
engine.set_parameters(
    vt_symbol="000001.SZSE",
    interval=Interval.DAILY,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
    rate=0.0003,
    slippage=0.01,
    size=100,
    pricetick=0.01,
    capital=1000000,
)

# 回测将自动使用EFinance数据源获取历史数据
```

## 数据质量控制

EFinance数据源包含完整的数据质量控制机制：

### 数据验证规则
- 价格合理性检查（最高价≥最低价等）
- 成交量非负检查
- 数据完整性验证
- 时间序列连续性检查

### 错误处理
- 自动重试机制
- 详细错误日志
- 优雅降级处理

## 性能优化

### 数据获取优化
- 批量数据请求
- 智能缓存机制
- 并发查询支持

### 内存管理
- 流式数据处理
- 内存使用监控
- 自动垃圾回收

## 故障排除

### 常见问题

**1. 数据获取失败**
```bash
# 检查网络连接
ping -c 3 efinance.readthedocs.io

# 更新efinance库
pip install efinance --upgrade
```

**2. 股票代码格式错误**
```python
# 正确的股票代码格式
"000001"  # A股，6位数字
"AAPL"    # 美股，字母代码
"00700"   # 港股，5位数字
```

**3. 时间周期不支持**
```python
# 检查支持的周期
datafeed = EfinanceDatafeed()
print(datafeed.get_supported_intervals())
```

### 调试模式

启用详细日志进行调试：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 初始化时会输出详细调试信息
datafeed = EfinanceDatafeed()
datafeed.init(output=print)
```

## 版本历史

### v1.0.0 (2024-12-22)
- 首次发布
- 支持A股、美股、港股数据获取
- 支持多种时间周期
- 完整的数据质量控制
- 详细的错误处理和日志记录

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 相关链接

- [EFinance官方文档](https://efinance.readthedocs.io/en/latest/)
- [EFinance GitHub](https://github.com/Micro-sheep/efinance)
- [VnPy官方网站](https://www.vnpy.com/)
- [VnPy GitHub](https://github.com/vnpy/vnpy)

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。使用本项目进行交易的风险由用户自行承担。
