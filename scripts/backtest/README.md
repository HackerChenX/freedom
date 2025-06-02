# 股票回测系统

本目录包含股票回测系统的实现，用于对股票进行技术分析、形态识别、策略生成和验证。

## 统一合并回测系统

`consolidated_backtest.py` 是合并了多个回测脚本功能后的统一回测系统，它具备以下核心功能：

1. **多周期分析**：解决了周期与指标混淆问题，支持多个周期的联合分析
2. **技术指标分析**：支持多种技术指标，包括MACD、KDJ、RSI等
3. **形态识别**：识别各种技术形态，并分析形态的有效性
4. **形态组合分析**：分析多个形态组合的有效性
5. **策略生成与验证**：基于回测结果生成选股策略，并验证策略有效性
6. **策略优化**：自动优化策略参数，提高策略成功率
7. **ZXM体系支持**：支持ZXM体系指标的专用分析
8. **高性能处理**：支持多进程并行处理，大幅提高回测速度
9. **多种输出格式**：支持JSON、CSV、Excel等多种输出格式

## 系统最新改进

### 1. 形态识别封装

最新版本实现了形态识别逻辑的完整封装，主要改进包括：

- 所有形态识别逻辑从回测脚本迁移至各指标类内
- 实现了统一的形态识别接口 `get_patterns()`
- 为每个指标实现特定的形态识别算法
- 新增 `PatternRegistry` 形态注册表，管理所有技术形态
- 形态结果包含标准化的强度、描述和详细信息

### 2. 周期与指标分离机制

解决了不同周期同名指标混淆的问题：

- 重新设计分析结果数据结构，确保周期隔离
- 实现 `PeriodManager` 周期管理器，处理不同周期的数据需求
- 按周期组织分析流程和结果存储
- 为不同周期设置合理的数据量需求
- 支持多周期结果的组合分析

### 3. 统一指标评分系统

实现了标准化的指标评分系统：

- 为所有指标提供0-100分制的统一评分标准
- 评分考虑当前指标值、历史统计和形态强度
- 实现形态权重和组合评分算法
- 支持评分历史记录和变化分析
- 增强策略决策依据

### 4. 增强的共性特征分析

提升了共性特征分析能力：

- 全面分析所有出现的形态，不限制数量
- 按周期和指标维度分组统计分析
- 实现特征之间的关联性分析
- 发现高频高准确率形态组合
- 支持多维度特征筛选

### 5. 可配置的策略自动生成

改进了策略自动生成功能：

- 设计统一的策略配置格式，支持JSON/YAML
- 精确描述"周期+指标+形态"组合
- 基于回测结果智能生成策略
- 自动分配指标和形态权重
- 支持策略参数调优和性能评估

## 安装依赖

```bash
pip install pandas numpy matplotlib tqdm pymongo clickhouse-driver joblib pyyaml
```

## 使用方法

### 基本用法

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101
```

### 使用股票列表文件

```bash
python consolidated_backtest.py --stock-list stocks.txt --start-date 20220101 --end-date 20230101
```

### 指定配置文件

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --config config.json
```

### 回测模式

系统支持多种回测模式：

1. **标准回测模式**（默认）

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --mode standard
```

2. **形态回测模式**

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --mode pattern --indicators MACD,KDJ,RSI --periods daily,weekly
```

3. **组合回测模式**

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --mode combination --combination-file combinations.json
```

4. **策略回测模式**

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --mode strategy --strategy-file strategy.json
```

5. **ZXM体系回测模式**

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --mode zxm
```

### 策略优化

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --mode strategy --strategy-file strategy.json --optimize --target-success-rate 0.7
```

### 输出格式

```bash
python consolidated_backtest.py --stock-code 000001 --start-date 20220101 --end-date 20230101 --output result.json --format json
```

支持的格式：json, csv, excel, html, text

## 编程接口使用示例

除了命令行接口，系统也提供了完整的编程API，可以在Python代码中直接使用。

### 基本使用示例

```python
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest

# 初始化回测系统
backtest = ConsolidatedBacktest()

# 分析单个股票的特定买点
result = backtest.analyze_stock(
    code="000001", 
    buy_date="20230101"
)

# 打印分析结果
print(f"分析结果：{result.stock_code}, {result.buy_date}")
for period_result in result.period_results.values():
    print(f"周期：{period_result.period}")
    for indicator_id, indicator_result in period_result.indicator_results.items():
        print(f"  指标：{indicator_id}, 得分：{indicator_result.score}")
        for pattern in indicator_result.patterns:
            print(f"    形态：{pattern['pattern_id']}, 强度：{pattern['strength']}")
```

### 批量回测示例

```python
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest

# 初始化回测系统
backtest = ConsolidatedBacktest()

# 批量回测
results = backtest.batch_analyze(
    input_file="data/stocks_to_test.csv",
    output_file="data/result/backtest_result.xlsx"
)

# 生成策略
strategy = backtest.generate_strategy(
    results=results,
    output_file="data/result/strategy.json",
    threshold=3  # 最少出现次数
)

# 验证策略
validation = backtest.validate_strategy(
    stock_codes=["000001", "000002", "000003"],
    strategy_config=strategy,
    start_date="20230101",
    end_date="20230601"
)

print(f"策略成功率：{validation['success_rate']}")
```

### 形态组合回测示例

```python
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest
from enums.period import Period

# 初始化回测系统
backtest = ConsolidatedBacktest()

# 定义形态组合
pattern_combinations = [
    {
        "indicator_id": "MACD",
        "pattern_id": "golden_cross",
        "period": Period.DAILY
    },
    {
        "indicator_id": "KDJ",
        "pattern_id": "oversold_rebound",
        "period": Period.WEEKLY
    }
]

# 形态组合回测
result = backtest.backtest_with_pattern_combination(
    stock_codes=["000001", "000002", "000003"],
    start_date="20230101",
    end_date="20230601",
    pattern_combinations=pattern_combinations,
    forward_days=5,
    threshold=0.02
)

# 打印回测结果
print(f"组合出现次数：{result['occurrence_count']}")
print(f"成功次数：{result['success_count']}")
print(f"成功率：{result['success_rate']}")
print(f"平均收益率：{result['avg_profit']}")
```

### 策略优化示例

```python
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest
import json

# 初始化回测系统
backtest = ConsolidatedBacktest()

# 加载策略
with open("data/strategy.json", "r") as f:
    strategy = json.load(f)
    
# 验证策略
validation = backtest.validate_strategy(
    stock_codes=["000001", "000002", "000003"],
    strategy_config=strategy,
    start_date="20230101",
    end_date="20230601"
)

# 优化策略
optimized_strategy = backtest.optimize_strategy(
    strategy_config=strategy,
    validation_result=validation,
    target_success_rate=0.7,
    max_iterations=5
)

# 保存优化后的策略
with open("data/optimized_strategy.json", "w") as f:
    json.dump(optimized_strategy, f, indent=2)
```

## 配置文件示例

### 基本配置

```json
{
  "periods": ["daily", "weekly", "monthly", "min_60", "min_30", "min_15"],
  "indicators": {
    "MACD": {"enabled": true},
    "KDJ": {"enabled": true},
    "RSI": {"enabled": true},
    "BOLL": {"enabled": true},
    "MA": {"enabled": true},
    "VOL": {"enabled": true}
  },
  "days_before": 60,
  "days_after": 20,
  "trading_rules": {
    "entry": {
      "min_score": 70,
      "max_spread": 0.02,
      "prefer_pattern": true
    },
    "exit": {
      "take_profit": 0.15,
      "stop_loss": 0.07,
      "trailing_stop": 0.05,
      "max_hold_days": 30
    }
  },
  "parallel": {
    "enabled": true,
    "chunk_size": 10
  },
  "output": {
    "save_interim": false,
    "detail_level": "medium",
    "formats": ["json", "excel"]
  }
}
```

### 组合配置示例

```json
[
  {
    "indicator_id": "MACD",
    "pattern_id": "golden_cross",
    "period": "daily",
    "min_strength": 0.7
  },
  {
    "indicator_id": "KDJ",
    "pattern_id": "oversold_rebound",
    "period": "weekly",
    "min_strength": 0.6
  }
]
```

### 策略配置示例

```json
{
  "strategy": {
    "name": "MACD金叉+KDJ超卖反弹策略",
    "description": "日线MACD金叉与周线KDJ超卖反弹组合策略",
    "id": "STRATEGY_A1B2C3D4",
    "create_time": "2024-05-20 10:00:00",
    "update_time": "2024-05-20 10:00:00",
    "author": "System",
    "version": "1.0",
    "tags": ["MACD", "KDJ", "金叉", "超卖反弹"],
    "conditions": [
      {
        "period": "daily",
        "indicator_id": "MACD",
        "pattern_id": "golden_cross",
        "min_strength": 0.7,
        "score_threshold": 60
      },
      {
        "logic": "AND"
      },
      {
        "period": "weekly",
        "indicator_id": "KDJ",
        "pattern_id": "oversold_rebound",
        "min_strength": 0.6,
        "score_threshold": 50
      }
    ],
    "trading_rules": {
      "entry": {
        "min_score": 70,
        "max_spread": 0.02
      },
      "exit": {
        "take_profit": 0.15,
        "stop_loss": 0.07
      }
    }
  }
}
```

## 进阶使用指南

### 1. 自定义指标和形态

要添加自定义指标和形态，需要按以下步骤操作：

1. 在 `indicators/` 目录中创建新的指标类，继承自 `BaseIndicator`
2. 实现必要的方法，包括 `calculate()` 和 `get_patterns()`
3. 将指标类添加到 `indicators/factory.py` 的注册表中

示例：
```python
from indicators.base_indicator import BaseIndicator
from indicators.pattern_recognition import PatternResult

class MyCustomIndicator(BaseIndicator):
    def __init__(self, param1=10, param2=20):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.name = "MyCustomIndicator"
        
    def calculate(self, data):
        # 实现指标计算逻辑
        # ...
        return result
        
    def get_patterns(self, data=None, result=None):
        # 如果没有提供result，则计算
        if result is None:
            if data is None:
                return []
            result = self.calculate(data)
            
        # 识别形态
        patterns = []
        # ...实现形态识别逻辑
        
        # 返回识别的形态
        return patterns
```

### 2. 自定义回测策略

您可以通过以下方式扩展回测系统：

1. 继承 `ConsolidatedBacktest` 类创建自定义回测类
2. 重写关键方法自定义行为
3. 添加新的分析功能

示例：
```python
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest

class MyCustomBacktest(ConsolidatedBacktest):
    def __init__(self, config=None, cpu_cores=None):
        super().__init__(config, cpu_cores)
        # 添加自定义初始化
        
    def custom_analyze_method(self, stock_code, date):
        # 实现自定义分析方法
        # ...
        return result
        
    # 重写方法以自定义行为
    def _analyze_period(self, code, buy_date, period, start_date, end_date, config):
        # 自定义周期分析行为
        # ...
        return super()._analyze_period(code, buy_date, period, start_date, end_date, config)
```

### 3. 性能优化

对于大规模回测，可以使用以下方法优化性能：

1. **增加并行处理**：设置更高的CPU核心数
   ```python
   backtest = ConsolidatedBacktest(cpu_cores=12)
   ```

2. **数据缓存**：使用内置的数据缓存机制
   ```python
   # config.json
   {
     "cache": {
       "enabled": true,
       "max_size": 1000,
       "ttl": 3600
     }
   }
   ```

3. **减少指标和周期**：针对特定分析需求选择必要的指标和周期
   ```python
   # config.json
   {
     "periods": ["daily", "weekly"],  # 仅使用日线和周线
     "indicators": {
       "MACD": {"enabled": true},
       "KDJ": {"enabled": true},
       # 其他指标设为false
     }
   }
   ```

4. **批量处理**：调整批处理大小，平衡内存使用和性能
   ```python
   # config.json
   {
     "parallel": {
       "enabled": true,
       "chunk_size": 20  # 增加批处理大小
     }
   }
   ```

## 常见问题

### Q: 如何添加自定义指标？

A: 需要在 `indicators` 目录下创建新的指标类，继承 `BaseIndicator`，并实现 `calculate()` 和 `get_patterns()` 方法。然后在 `indicators/factory.py` 中注册。

### Q: 如何提高回测速度？

A: 增加 `--workers` 参数的值可以提高并行处理能力，但会增加内存消耗。也可以减小回测的股票数量和日期范围。

### Q: 如何生成HTML格式的回测报告？

A: 使用 `--output report.html --format html` 参数可以生成HTML格式的回测报告。

### Q: 回测数据的来源是什么？

A: 系统从ClickHouse数据库获取历史行情数据，确保数据库中有相应的数据表和股票数据。

### Q: 如何解决内存不足问题？

A: 减少批处理大小(`chunk_size`)，减少分析的股票数量，或减少使用的指标和周期数量。也可以分批次运行回测，然后合并结果。

### Q: 如何验证生成的策略？

A: 使用 `validate_strategy()` 方法或命令行参数 `--validate` 可以验证策略在不同时间段的表现。

### Q: 形态识别结果不准确怎么办？

A: 检查指标参数设置，调整形态识别的阈值，或自定义形态识别逻辑。也可以增加形态强度的最低要求。 