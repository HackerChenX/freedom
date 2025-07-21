# 统一指标测试框架

## 📖 概述

统一指标测试框架是一个综合性的测试系统，同时验证技术指标的买点形态识别能力和选股策略有效性。该框架确保修复的指标不仅在理论上正确，在实际生产环境中也能发挥预期作用。

## 🎯 核心特性

- **双重验证**：同时测试买点形态识别和选股策略
- **生产级测试**：使用正式选股脚本，确保生产环境兼容性
- **闭环验证**：选股结果能被买点分析识别对应形态
- **100%通过率**：严格的质量标准
- **全面覆盖**：支持所有指标的所有形态测试
- **性能保证**：支持4000+股票规模的性能测试

## 🏗️ 架构组件

### 核心组件
- `UnifiedIndicatorTester`: 统一测试控制器
- `StockInfoCompatibleDataGenerator`: 数据生成器
- `BuypointRecognitionTester`: 买点识别测试器
- `SelectionStrategyTester`: 选股策略测试器
- `ClosedLoopValidator`: 闭环验证器
- `PerformanceTester`: 性能测试器

### 配置文件
- `config.yaml`: 主配置文件
- 包含21个已修复指标的完整测试配置
- 性能要求和验证标准定义

## 🚀 快速开始

### 1. 环境准备
```bash
# 确保在项目根目录
cd /path/to/freedom

# 安装依赖（如果需要）
pip install -r requirements.txt
```

### 2. 基本使用

#### 测试所有已修复指标
```bash
python tests/unified_indicator_testing/run_unified_tests.py --mode all
```

#### 测试单个指标
```bash
python tests/unified_indicator_testing/run_unified_tests.py --mode single --indicator MACD
```

#### 测试多个指标
```bash
python tests/unified_indicator_testing/run_unified_tests.py --mode multiple --indicators MACD,RSI,KDJ
```

#### 快速测试（核心指标）
```bash
python tests/unified_indicator_testing/run_unified_tests.py --mode quick
```

#### 性能测试
```bash
python tests/unified_indicator_testing/run_unified_tests.py --mode performance --scale 4000
```

#### 严格模式（要求100%通过率）
```bash
python tests/unified_indicator_testing/run_unified_tests.py --mode all --strict
```

### 3. 高级用法

#### 自定义配置文件
```bash
python tests/unified_indicator_testing/run_unified_tests.py \
  --config custom_config.yaml \
  --mode all
```

#### 保存测试结果
```bash
python tests/unified_indicator_testing/run_unified_tests.py \
  --mode all \
  --output test_results/ \
  --verbose
```

#### 试运行（查看将要执行的测试）
```bash
python tests/unified_indicator_testing/run_unified_tests.py \
  --mode all \
  --dry-run
```

## 📊 测试流程详解

### 1. 数据生成阶段
- 生成符合stockInfo结构的模拟数据
- 确保足够的历史数据满足指标计算需求
- 创建包含目标形态和干扰数据的测试池

### 2. 买点识别测试
- 使用现有的买点分析器
- 验证能否正确识别目标形态
- 计算识别准确率和置信度

### 3. 选股策略测试
- 自动生成对应的选股策略配置
- 调用正式选股脚本（`bin/stock_select.py`）
- 使用模拟数据模式进行选股

### 4. 闭环验证
- 对选股结果进行买点形态分析
- 验证是否能识别出期望的形态
- 确保选股和识别的一致性

### 5. 性能测试
- 大规模数据处理能力测试
- 内存使用和执行时间监控
- 并发处理能力验证

## 📋 配置说明

### 指标配置示例
```yaml
MACD:
  patterns: ['GOLDEN_CROSS', 'DEATH_CROSS', 'DIVERGENCE']
  history_requirement: 60
  complex_conditions: ['MACD_RSI_COMBO']
  expected_accuracy: 1.0
  priority: "high"
```

### 性能要求
```yaml
performance_requirements:
  max_execution_time_per_indicator: 120  # 秒
  max_memory_usage: "2GB"
  min_throughput: 1000  # 股票/秒
```

### 验证标准
```yaml
validation_criteria:
  buypoint_recognition_accuracy: 1.0
  selection_precision: 0.95
  selection_recall: 0.90
  closed_loop_validation_rate: 1.0
```

## 📈 测试结果解读

### 测试输出示例
```
📋 测试结果摘要
==================================================
MACD            ✅ 通过 (评分: 1.00)
RSI             ✅ 通过 (评分: 1.00)
KDJ             ❌ 失败 (评分: 0.85)
--------------------------------------------------
总计: 3 个指标
通过: 2 个
失败: 1 个
通过率: 66.7%
```

### 详细报告
测试完成后会生成详细的Markdown报告，包含：
- 执行摘要
- 每个指标的详细测试结果
- 性能分析
- 闭环验证结果
- 改进建议

## 🔧 扩展和定制

### 添加新指标测试
1. 在`config.yaml`中添加指标配置
2. 确保指标已在系统中注册
3. 运行测试验证

### 自定义测试模式
1. 继承`UnifiedIndicatorTester`类
2. 实现自定义测试逻辑
3. 在配置文件中定义新的测试参数

### 集成CI/CD
```yaml
# .github/workflows/indicator-tests.yml
name: Indicator Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Unified Tests
        run: |
          python tests/unified_indicator_testing/run_unified_tests.py \
            --mode all --strict
```

## 🐛 故障排除

### 常见问题

#### 1. 模拟数据生成失败
```bash
# 检查指标是否正确注册
python -c "from indicators.complete_indicator_registry import complete_registry; print(complete_registry.indicators.keys())"
```

#### 2. 选股脚本调用失败
```bash
# 验证选股脚本是否可用
python bin/stock_select.py --help
```

#### 3. 内存不足
```bash
# 减少测试规模
python run_unified_tests.py --mode quick
```

### 调试模式
```bash
# 启用详细日志
python run_unified_tests.py --mode single --indicator MACD --verbose
```

## 📞 支持和贡献

### 获取帮助
- 查看详细日志文件：`logs/unified_indicator_test.log`
- 检查测试报告：`test_reports/`
- 参考配置文件注释

### 贡献指南
1. Fork项目
2. 创建功能分支
3. 添加测试用例
4. 提交Pull Request

## 📚 相关文档

- [统一指标测试架构设计方案](../../docs/统一指标测试架构设计方案.md)
- [技术指标系统修复工作总结](../../docs/技术指标系统修复工作总结与交接文档.md)
- [指标修复进度跟踪表](../../docs/指标修复进度跟踪表.md)

---

**版本**: 1.0.0  
**最后更新**: 2025-07-21  
**维护者**: AI Assistant
