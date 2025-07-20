# 统一分析引擎全面测试系统

## 🚀 快速开始

### 一键启动测试
```bash
# 交互式测试界面
python scripts/start_unified_engine_test.py

# 或者直接运行快速测试
python bin/run_unified_engine_test.py --test-type quick
```

## 📊 测试特性

- ✅ **88个指标全覆盖** - 包括基础、增强、ZXM、复合指标
- 🔗 **真实数据测试** - 连接ClickHouse数据库
- ⚡ **早停机制** - 发现错误立即停止，提高效率
- 🎯 **多层次验证** - 计算、逻辑、选股条件全面验证
- 📈 **性能优化** - 并发测试、批量处理、智能缓存

## 🎯 测试类型

| 类型 | 命令 | 时间 | 说明 |
|------|------|------|------|
| 快速测试 | `--test-type quick` | ~5分钟 | 测试核心指标 |
| 优先级测试 | `--test-type priority` | ~10分钟 | 测试重要指标 |
| 逻辑验证 | `--test-type logic` | ~15分钟 | 深度验证计算逻辑 |
| ZXM测试 | `--test-type zxm` | ~20分钟 | 测试ZXM体系指标 |
| 增强测试 | `--test-type enhanced` | ~15分钟 | 测试增强指标 |
| 全面测试 | `--test-type comprehensive` | ~60分钟 | 测试所有指标 |

## 🔧 配置文件

主要配置文件：`config/unified_engine_test_config.json`

```json
{
  "test_configuration": {
    "test_stock_count": 200,
    "enable_early_stop": true,
    "max_concurrent_tests": 3,
    "debug_mode": true
  }
}
```

## 📋 测试报告

测试结果保存在 `data/result/` 目录：
- JSON详细报告
- 文本摘要报告  
- 最终综合报告

## ⚡ 性能优化

系统已集成以下优化：
- 批量数据查询（减少99%数据库连接）
- 并发指标计算（提升56%处理能力）
- 智能内存管理
- 动态批次调整

## 🛠️ 故障排除

### 常见问题
1. **数据库连接失败** - 检查ClickHouse服务状态
2. **内存不足** - 减少测试股票数量或并发数
3. **指标计算错误** - 查看详细日志进行调试

### 调试命令
```bash
# 查看日志
tail -f logs/unified_engine_test.log

# 单独测试指标
python scripts/indicator_logic_validator.py

# 小数据集测试
python bin/run_unified_engine_test.py --test-type quick --sample-stocks 1
```

## 📚 详细文档

完整使用指南：[doc/使用指南/统一分析引擎测试系统使用指南.md](doc/使用指南/统一分析引擎测试系统使用指南.md)

## 🎯 项目目标

确保统一分析引擎的：
- 88个指标计算准确性
- 复杂选股条件正确性  
- 大规模数据处理性能
- 系统稳定性和可靠性

---

**立即开始测试：**
```bash
python scripts/start_unified_engine_test.py
``` 