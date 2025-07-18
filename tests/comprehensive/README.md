# 综合选股测试系统

这个测试框架用于全面测试选股系统，包括指标发现、形态注册、选股引擎、闭环验证、报告生成等功能。

## 目录结构

```
tests/
  ├── comprehensive/       # 综合测试系统
  │   ├── cli.py                # 命令行界面
  │   ├── test_config.yaml      # 测试配置文件
  │   ├── test_config_manager.py # 配置管理器
  │   ├── test_orchestrator.py  # 测试执行编排器
  │   ├── test_result_validator.py # 测试结果验证器
  │   ├── monitoring.py         # 监控系统
  │   ├── monitoring_dashboard.py # 监控仪表板
  │   ├── log_analyzer.py       # 日志分析器
  │   ├── system_manager.py     # 系统管理器
  │   ├── run_tests.py          # 测试运行脚本
  │   ├── fix_issues.py         # 问题修复脚本
  │   └── main.py               # 主脚本
  ├── unit/                # 单元测试
  │   ├── test_config_manager.py # 配置管理器测试
  │   └── test_result_validator.py # 测试结果验证器测试
  └── integration/         # 集成测试
      └── test_system_integration.py # 系统集成测试
```

## 使用方法

### 运行全面测试

```bash
python -m tests.comprehensive.main --workspace test_workspace
```

### 启用监控仪表板

```bash
python -m tests.comprehensive.main --workspace test_workspace --dashboard
```

### 只运行测试

```bash
python -m tests.comprehensive.run_tests --workspace test_workspace
```

### 只修复问题

```bash
python -m tests.comprehensive.fix_issues --workspace test_workspace
```

### 使用命令行界面

```bash
python -m tests.comprehensive.cli run --config tests/comprehensive/test_config.yaml
```

## 测试流程

1. **单元测试** - 测试各个组件的独立功能
2. **集成测试** - 测试组件之间的交互
3. **系统测试** - 测试整个系统的功能
4. **性能测试** - 测试系统在高负载下的表现
5. **错误处理测试** - 测试系统对异常情况的处理

## 修复流程

1. **分析问题** - 分析测试结果，找出问题
2. **修复问题** - 根据问题类型修复问题
3. **生成报告** - 生成修复报告

## 监控仪表板

监控仪表板提供了实时监控系统状态的功能，包括：

- 系统状态
- 资源使用情况
- 活跃会话
- 告警信息

访问地址：http://localhost:8080

## 日志分析

日志分析器提供了分析日志的功能，包括：

- 错误分析
- 性能分析
- 会话分析

## 配置文件

测试配置文件位于 `tests/comprehensive/test_config.yaml`，可以根据需要修改配置参数。

## 注意事项

1. 确保系统环境已经正确配置
2. 确保数据库连接可用
3. 确保有足够的系统资源（内存、CPU）
4. 测试可能需要较长时间，请耐心等待
5. 如果测试失败，可以查看日志文件了解详细信息