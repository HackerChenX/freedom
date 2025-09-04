# 买点回测系统快速开始指南

## 🚀 5分钟快速体验

### 第一步：环境检查
```bash
# 检查Python版本 (需要3.7+)
python --version

# 检查必要模块
python -c "import pandas, numpy; print('✅ 基础模块OK')"
```

### 第二步：准备买点数据
```bash
# 创建数据目录
mkdir -p data

# 创建示例买点文件
cat > data/buypoints.csv << EOF
stock_code,buypoint_date
603359,20250512
000001,20250520
600036,20250515
000858,20250518
600000,20250522
EOF

echo "✅ 买点文件创建完成"
```

### 第三步：运行回测
```bash
# 运行买点回测系统
python bin/run_buypoint_backtest.py --verbose
```

### 第四步：查看结果
```bash
# 查看生成的报告
ls results/buypoint_backtest/

# 查看摘要报告
cat results/buypoint_backtest/backtest_summary_*.md
```

## 📊 预期输出示例

### 控制台输出
```
🚀 买点回测分析系统启动
================================================================================
📋 买点文件: data/buypoints.csv
🕐 开始时间: 2025-09-04 20:50:32

🔧 初始化买点回测引擎...
买点回测引擎初始化完成，支持103个指标

🎯 开始执行买点回测...
📋 加载买点数据: 5个买点

[1/5] 处理买点: 603359 @ 20250512
  📊 获取多周期数据...
  🔍 分析15min周期...
    ✅ 15min周期: 12/103 指标命中
  🔍 分析30min周期...
    ✅ 30min周期: 8/103 指标命中
  🔍 分析daily周期...
    ✅ daily周期: 15/103 指标命中

🎉 策略执行完成: 3/15 股票符合条件

📊 回测结果摘要
================================================================================
📋 处理买点数: 5
✅ 成功分析: 5
❌ 失败分析: 0

🏆 热门指标+周期组合排行 (前5名):
  1. daily周期MACD: 15次命中
  2. 30min周期RSI: 12次命中
  3. daily周期KDJ: 10次命中
  4. 60min周期BOLL: 8次命中
  5. weekly周期MA: 7次命中

📊 仅指标名统计 (参考，前3名):
  1. MACD: 28次命中 (跨所有周期)
  2. RSI: 24次命中 (跨所有周期)
  3. KDJ: 19次命中 (跨所有周期)

📈 周期效果分析:
  15min   : 平均8.2个指标命中, 成功率80.0%
  daily   : 平均12.4个指标命中, 成功率100.0%
  weekly  : 平均4.1个指标命中, 成功率60.0%

🎯 生成选股策略: 3个
✅ 双向验证成功! 选出股票: 3只
📄 详细结果已保存到 results/buypoint_backtest/ 目录
```

## 🎯 核心功能演示

### 1. 多周期数据处理
系统自动处理6个周期的数据：
- ✅ **15分钟**: 直接从数据库获取
- ✅ **30分钟**: 从15分钟数据自动合并生成
- ✅ **60分钟**: 从15分钟数据自动合并生成  
- ✅ **日线**: 直接从数据库获取
- ✅ **周线**: 直接从数据库获取
- ✅ **月线**: 直接从数据库获取

### 2. 103个指标测试
系统逐个测试所有已验证的指标：
- **基础指标**: MA, MACD, RSI, KDJ, BOLL等
- **ZXM指标**: ZXM_DAILY_MACD, ZXM_BS_ABSORB等
- **形态指标**: DOJI, HAMMER, V_SHAPED_REVERSAL等
- **评分指标**: MACD_SCORE, RSI_SCORE等

### 3. 指标与周期强制绑定 ⚠️ **核心原则**
严格确保指标分析永远不能脱离周期，指标和周期必须绑定在一起：

```
✅ 正确的指标+周期绑定:
- "MACD_daily_golden_cross"   (日线MACD金叉)
- "MACD_30min_golden_cross"   (30分钟MACD金叉)
- "MACD_weekly_golden_cross"  (周线MACD金叉)
- "KDJ_15min_oversold"        (15分钟KDJ超卖)
- "KDJ_monthly_oversold"      (月线KDJ超卖)

❌ 绝对禁止的模式:
- "MACD_golden_cross"         (缺少周期信息)
- "golden_cross"              (缺少指标和周期信息)

📊 统计和显示规则:
- 热门排行按"周期+指标"组合显示
- 日线MACD和30分钟MACD分别统计
- 策略生成时明确指定周期条件
```

## 📋 命令行选项

### 基本用法
```bash
# 使用默认买点文件
python bin/run_buypoint_backtest.py

# 指定买点文件
python bin/run_buypoint_backtest.py --buypoints data/my_buypoints.csv

# 详细输出模式
python bin/run_buypoint_backtest.py --verbose

# 查看帮助
python bin/run_buypoint_backtest.py --help
```

## 🔧 自定义配置

### 1. 修改买点文件格式
买点文件必须包含以下列：
- `stock_code`: 股票代码
- `buypoint_date`: 买点日期 (YYYYMMDD格式)

### 2. 调整周期配置
在 `buypoint_backtest_engine.py` 中修改：
```python
# 修改支持的周期
self.periods = ['15min', '30min', '60min', 'daily', 'weekly', 'monthly']

# 修改周期权重
period_weights = {
    'daily': 0.3,    # 日线权重最高
    '60min': 0.2,
    '30min': 0.15,
    'weekly': 0.15,
    'monthly': 0.1,
    '15min': 0.1
}
```

## ⚠️ 常见问题

### Q1: 买点文件格式错误
```
❌ 错误: 买点文件缺少必要列: ['buypoint_date']
💡 解决: 确保CSV文件包含 stock_code 和 buypoint_date 列
```

### Q2: 没有找到买点文件
```
❌ 错误: 买点文件不存在: data/buypoints.csv
💡 解决: 创建买点文件或使用 --buypoints 指定正确路径
```

### Q3: 指标计算失败
```
⚠️ 警告: 指标MACD计算失败: insufficient data
💡 说明: 这是正常现象，系统会跳过失败的指标继续处理
```

## 🎯 下一步

### 1. 查看详细结果
```bash
# 查看JSON详细结果
cat results/buypoint_backtest/backtest_detail_*.json

# 查看Markdown报告
cat results/buypoint_backtest/backtest_summary_*.md
```

### 2. 分析结果
- 关注热门指标排行
- 分析周期效果差异
- 研究生成的策略逻辑
- 验证双向验证结果

### 3. 优化策略
- 调整周期权重
- 筛选高效指标
- 优化评分阈值
- 改进形态组合

## 📞 技术支持

如果遇到问题，请检查：
1. **环境依赖**: 确保Python 3.7+和必要模块已安装
2. **文件路径**: 确认买点文件路径正确
3. **数据格式**: 验证买点文件格式符合要求
4. **系统资源**: 确保有足够的内存和磁盘空间

---

**文档类型**: 用户快速指南  
**适用对象**: 系统用户  
**更新时间**: 2025-09-04
