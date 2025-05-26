# 股票分析系统

## 项目结构

```
.
├── analysis/            # 分析模块
│   ├── buypoints/       # 买点分析
│   └── market/          # 市场分析
├── api/                 # 外部API接口实现
├── bin/                 # 可执行脚本
│   ├── main.py          # 主程序入口
│   └── stock_select.py  # 选股功能入口
├── config/              # 配置管理模块
├── data/                # 数据目录
│   ├── reference/       # 参考数据
│   └── result/          # 结果数据
├── db/                  # 数据库接口模块
├── enums/               # 枚举定义模块
├── formula/             # 公式计算模块
├── indicators/          # 技术指标模块
├── logs/                # 日志目录
│   └── archive/         # 日志存档
├── scripts/             # 脚本工具
│   ├── backtest/        # 回测脚本
│   ├── sql/             # SQL脚本
│   └── utils/           # 工具脚本
├── sql/                 # SQL文件
├── strategy/            # 选股策略模块
├── tests/               # 测试模块
└── utils/               # 工具模块
```

## 主要功能模块

1. **选股功能** - `bin/stock_select.py`
   - 多线程选股
   - 基于策略选股
   - 按行业选股

2. **市场分析** - `analysis/market/a_stock_market_analysis.py`
   - 市场强度计算
   - 操作建议

3. **买点分析** - `analysis/buypoints/analyze_buypoints.py`
   - 技术形态识别
   - 买点评分

4. **回测功能** - `scripts/backtest/backtest.py`
   - KDJ金叉回测
   - 多种回测模式

5. **数据同步** - `scripts/utils/data_sync.py` 和 `scripts/utils/akshare_to_clickhouse.py`
   - 股票数据同步
   - 行业数据同步

## 技术指标

项目实现了多种常用技术指标，位于 `indicators/` 目录：
- MACD
- KDJ
- RSI
- 布林带
- 移动平均线 (MA, EMA, SMA)

## 数据库

使用ClickHouse作为后端数据库，存储股票历史数据和行业数据。数据库接口封装在 `db/` 目录下。

## 配置管理

配置项存放在 `config/` 目录，包括数据库连接配置等。

## 使用方法

1. **选股**：
   ```
   python bin/stock_select.py --mode strategy --strategy momentum_strategy --params '{"min_turnover_rate": 1.5}'
   ```

2. **市场分析**：
   ```
   python bin/main.py --date 20240520
   ```

3. **数据同步**：
   ```
   python scripts/utils/data_sync.py --mode all
   ```

4. **回测**：
   ```
   python scripts/backtest/backtest.py --mode batch --date 20240401
   ```

## 开发规范

遵循项目架构规范文档，保持代码质量和一致性。

## 技术栈

- Python 3.8+
- NumPy/Pandas：数据处理
- ClickHouse：数据存储
- Matplotlib/Plotly：数据可视化

## 优化改进

最近的优化改进包括：

### 1. 配置管理

- 实现了集中式配置管理
- 添加了敏感信息加密功能
- 支持环境变量和配置文件覆盖
- 增加了配置验证功能

### 2. 技术指标系统

- 重构了技术指标实现，使用统一的基类和接口
- 创建了指标工厂，支持动态创建指标
- 实现了指标的可组合性
- 添加了单元测试，确保计算准确性
- 修复了MACD指标的NaN值处理问题
- 创建了统一的公共函数库，提高代码复用性

### 3. 项目结构优化

- 按业务领域重新组织文件结构
- 将分析工具移至analysis目录
- 将可执行脚本移至bin目录
- 将SQL文件集中管理
- 规范化日志和数据目录

### 4. 异常处理和日志

- 增强了异常处理机制
- 实现了集中式日志管理
- 添加了日志分级和轮转功能
- 增加了装饰器，简化异常捕获和日志记录

### 5. 类设计优化

- 将类变量与实例变量分离，避免数据共享问题
- 使用工厂模式创建对象
- 添加了可扩展的策略模式
- 优化了代码结构，提高了可维护性

### 6. 安全性增强

- 加密敏感配置信息
- 提供环境变量读取功能
- 实现了安全的密码管理
- 权限控制和访问限制

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT 