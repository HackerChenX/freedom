# 买点策略反向验证快速入门指南

## 🚀 5分钟快速开始

### 步骤1：准备数据
确保您有买点数据文件（CSV格式）：
```csv
stock_code,stock_name,date,price,volume,reason
000001,平安银行,2024-01-15,12.50,1000000,技术突破
600036,招商银行,2024-01-16,45.20,800000,均线金叉
```

### 步骤2：一键执行完整验证
```bash
python scripts/complete_validation_workflow.py \
    --buypoints data/buypoints.csv \
    --output results/quick_validation \
    --max-conditions 30
```

### 步骤3：查看结果
```bash
# 查看摘要报告
cat results/quick_validation/workflow_*/workflow_summary.txt

# 查看反向验证结果
cat results/quick_validation/workflow_*/04_reverse_validation/optimized/reverse_validation_report_*.txt
```

## 📊 结果解读

### 关键指标
- **匹配率**：策略能识别出多少原始买点股票（目标：≥80%）
- **精确率**：策略选中的股票中有多少是正确的（目标：≥70%）
- **F1分数**：综合性能指标（目标：≥0.7）
- **综合评级**：A（优秀）、B（良好）、C（需改进）

### 成功示例
```
匹配率: 100.0%
精确率: 100.0%
F1分数: 1.00
综合评级: A
```

## 🔧 常用命令

### 单独执行各步骤
```bash
# 1. 生成策略
python bin/buypoint_batch_analyzer.py --input data/buypoints.csv --output results/analysis

# 2. 优化策略
python scripts/optimize_strategy.py --strategy results/analysis/generated_strategy.json --output results/optimization

# 3. 反向验证
python scripts/reverse_validation.py --buypoints data/buypoints.csv --strategy results/optimization/optimized_strategy_*.json --output results/validation
```

### 参数调优
```bash
# 更激进的优化（更少条件）
python scripts/complete_validation_workflow.py \
    --buypoints data/buypoints.csv \
    --max-conditions 20 \
    --min-frequency 8

# 更保守的优化（更多条件）
python scripts/complete_validation_workflow.py \
    --buypoints data/buypoints.csv \
    --max-conditions 80 \
    --min-frequency 3
```

## ❓ 常见问题

**Q: 匹配率很低怎么办？**
A: 尝试放宽优化参数：`--max-conditions 100 --min-frequency 2`

**Q: 精确率很低怎么办？**
A: 尝试收紧优化参数：`--max-conditions 20 --min-frequency 10`

**Q: 文件路径错误怎么办？**
A: 检查文件是否存在：`ls -la data/buypoints.csv`

## 📚 更多信息

详细文档请参考：`docs/buypoint_strategy_reverse_validation_guide.md`

---

*快速入门指南 v1.0*
