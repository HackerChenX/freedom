# 选股系统端到端综合测试报告

## 📊 测试概要

- **测试时间**: 2025-06-22 14:35:56
- **测试时长**: 0.06 秒
- **数据源**: simulated
- **数据质量**: medium
- **测试策略数**: 5
- **成功策略数**: 0
- **失败策略数**: 5
- **整体成功率**: 0.00%
- **处理股票总数**: 100
- **选中股票总数**: 0
- **平均选股率**: 0.00%

## 📋 策略测试详情

### ❌ 趋势跟踪测试策略 (TEST_TREND_FOLLOWING)

- **执行状态**: 失败
- **执行时间**: 0.01 秒
- **处理股票数**: 0
- **选中股票数**: 0
- **选股率**: 0.00%
- **错误信息**: StrategyExecutor.execute_strategy() got an unexpected keyword argument 'strategy_file'

### ❌ 均值回归测试策略 (TEST_MEAN_REVERSION)

- **执行状态**: 失败
- **执行时间**: 0.00 秒
- **处理股票数**: 0
- **选中股票数**: 0
- **选股率**: 0.00%
- **错误信息**: StrategyExecutor.execute_strategy() got an unexpected keyword argument 'strategy_file'

### ❌ 突破测试策略 (TEST_BREAKOUT)

- **执行状态**: 失败
- **执行时间**: 0.00 秒
- **处理股票数**: 0
- **选中股票数**: 0
- **选股率**: 0.00%
- **错误信息**: StrategyExecutor.execute_strategy() got an unexpected keyword argument 'strategy_file'

### ❌ ZXM买点测试策略 (TEST_ZXM_BUYPOINT)

- **执行状态**: 失败
- **执行时间**: 0.00 秒
- **处理股票数**: 0
- **选中股票数**: 0
- **选股率**: 0.00%
- **错误信息**: StrategyExecutor.execute_strategy() got an unexpected keyword argument 'strategy_file'

### ❌ 多因子综合测试策略 (TEST_MULTI_FACTOR)

- **执行状态**: 失败
- **执行时间**: 0.01 秒
- **处理股票数**: 0
- **选中股票数**: 0
- **选股率**: 0.00%
- **错误信息**: StrategyExecutor.execute_strategy() got an unexpected keyword argument 'strategy_file'

## 🚀 性能分析

- **系统稳定性**: unstable
- **平均执行时间**: 0.01 秒
- **内存效率**: good
- **错误率**: 100.00%

## 🎯 质量评估

- **系统可靠性**: poor
- **结果一致性**: high
- **技术正确性**: verified
- **性能效率**: satisfactory
- **选股率合理性**: unknown

## 📈 技术指标覆盖情况

- **框架总指标数**: 82
- **测试指标数**: 15
- **覆盖率**: 18.29%
- **测试指标**: ATR, ZXM_BUYPOINT_SCORE, MA, ZXM_DAILY_MACD, ZXM_ELASTICITY_SCORE, TREND_STRENGTH_INDICATOR, RSI, WR, SYSTEM_PERFORMANCE_SCORE, VR, OBV, COMPOSITE_MOMENTUM_INDEX, MACD, BOLL, KDJ

## 🔧 系统集成状态

- **数据管理器集成**: functional
- **策略执行器集成**: functional
- **技术指标集成**: verified
- **配置管理**: operational
- **错误处理**: robust
- **整体集成健康度**: needs_improvement

## 💡 改进建议

1. 有 5 个策略执行失败，建议检查策略配置和数据完整性
2. 平均选股率过低，建议调整策略条件或阈值

---

*本报告由选股系统端到端综合测试框架自动生成*
