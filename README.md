# 技术形态分析系统

一个专注于股票技术形态识别和分析的系统。系统通过分析股票数据，识别各种技术形态，并提供形态强度评估。

## 功能特点

### 1. 形态分析
- 支持多种技术形态识别（MACD、KDJ等）
- 提供形态强度评估
- 支持多周期分析（日线、周线、月线）
- 支持形态组合分析

### 2. 分析功能
- 单只股票分析
- 批量股票分析
- 多周期一致性分析
- 形态统计和报告生成

### 3. 系统特性
- 模块化设计，易于扩展
- 支持自定义形态和分析器
- 高效的数据处理和分析
- 完整的日志记录

## 系统架构

### 核心组件
1. PatternScanner
   - 负责扫描和分析股票数据
   - 管理分析流程
   - 生成分析报告

2. PatternMatcher
   - 负责形态匹配和分析
   - 提供形态强度计算
   - 支持多周期分析

3. PatternAnalyzer
   - 负责具体形态的分析
   - 管理形态定义
   - 提供形态注册机制

4. PatternManager
   - 管理形态匹配器
   - 协调分析流程
   - 整合分析结果

## 安装说明

### 环境要求
- Python 3.8+
- pandas
- numpy
- sqlalchemy (用于数据库连接)

### 安装步骤
1. 克隆仓库
```bash
git clone https://github.com/yourusername/technical-pattern-analysis.git
cd technical-pattern-analysis
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置数据库
- 复制 `config/database.example.yaml` 为 `config/database.yaml`
- 修改数据库连接配置

## 使用说明

### 1. 命令行使用

#### 形态识别分析
```bash
python scripts/backtest/backtest_runner.py pattern \
    --stock 000001 \
    --start-date 20230101 \
    --end-date 20231231 \
    --pattern-ids macd_golden_cross kdj_golden_cross \
    --min-strength 0.6
```

#### ZXM形态分析
```bash
python scripts/backtest/backtest_runner.py zxm \
    --stock 000001 \
    --start-date 20230101 \
    --end-date 20231231 \
    --periods daily weekly monthly \
    --threshold 60.0
```

#### 批量分析
```bash
python scripts/backtest/backtest_runner.py batch \
    --type pattern \
    --stock-file stock_list.txt \
    --start-date 20230101 \
    --end-date 20231231 \
    --output results.csv
```

### 2. 代码调用示例

```python
from scripts.backtest.backtest_runner import PatternScanner

# 初始化扫描器
scanner = PatternScanner()

# 扫描形态
result = scanner.scan_patterns(
    stock_code="000001",
    start_date="20230101",
    end_date="20231231",
    pattern_ids=["macd_golden_cross", "kdj_golden_cross"],
    min_strength=0.6
)

# 输出结果
print(f"形态分析结果: {result}")
```

## 形态定义

### 基础形态
1. MACD形态
   - MACD金叉：快线上穿慢线
   - MACD死叉：快线下穿慢线

2. KDJ形态
   - KDJ金叉：K线上穿D线
   - KDJ死叉：K线下穿D线

### 自定义形态
系统支持通过 `PatternAnalyzer.register_pattern()` 方法注册自定义形态。

## 注意事项

1. 数据质量
   - 确保数据完整性
   - 注意数据时效性
   - 处理异常数据

2. 形态识别
   - 理解形态定义
   - 注意形态的局限性
   - 避免过度依赖单一形态

3. 系统使用
   - 定期更新形态库
   - 及时调整参数
   - 保持系统维护

## 开发计划

1. 形态库扩展
   - 添加更多技术形态
   - 优化形态定义
   - 增加形态组合分析

2. 分析能力提升
   - 改进强度计算方法
   - 优化多周期分析
   - 增加形态演化分析

3. 系统优化
   - 提升分析效率
   - 优化内存使用
   - 改进错误处理

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 作者：Your Name
- 邮箱：your.email@example.com
- 项目地址：https://github.com/yourusername/technical-pattern-analysis 