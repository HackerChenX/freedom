# 选股系统反向验证测试框架

## 概述

本框架通过构造符合特定技术形态的模拟数据来验证选股系统中技术指标的形态识别准确性。采用反向验证方法：先构造已知形态的数据，然后验证指标是否能正确识别出这些形态。

## 核心特性

- **数据格式标准化**：模拟数据与 stockInfo 数据格式完全一致
- **逐个指标测试**：支持对 P0 级别（核心指标）和 P1 级别（重要指标）的独立测试
- **形态模拟策略**：为每个指标的关键形态构造相应的价格/成交量时间序列数据
- **自动化验证**：自动运行指标分析并检查形态识别结果与预期的匹配度
- **详细报告**：生成准确率报告和修复建议

## 支持的指标和形态

### P0 级别核心指标

#### RSI (相对强弱指标)
- RSI_OVERBOUGHT: RSI超买形态（RSI > 70）
- RSI_OVERSOLD: RSI超卖形态（RSI < 30）
- RSI_GOLDEN_CROSS: RSI金叉形态（RSI从下方突破50）
- RSI_DEATH_CROSS: RSI死叉形态（RSI从上方跌破50）
- RSI_DIVERGENCE: RSI背离形态

#### MACD (指数平滑移动平均线)
- MACD_GOLDEN_CROSS: MACD金叉形态（DIF上穿DEA）
- MACD_DEATH_CROSS: MACD死叉形态（DIF下穿DEA）
- MACD_ABOVE_ZERO_GOLDEN: MACD零轴上金叉
- MACD_BELOW_ZERO_DEATH: MACD零轴下死叉
- MACD_HISTOGRAM_DIVERGENCE: MACD柱状图背离

#### KDJ (随机指标)
- KDJ_GOLDEN_CROSS: KDJ金叉形态（K线上穿D线）
- KDJ_DEATH_CROSS: KDJ死叉形态（K线下穿D线）
- KDJ_OVERBOUGHT: KDJ超买形态（K、D、J都大于80）
- KDJ_OVERSOLD: KDJ超卖形态（K、D、J都小于20）
- KDJ_BLUNT: KDJ钝化形态

#### BOLL (布林带)
- BOLL_UPPER_BREAKOUT: BOLL上轨突破
- BOLL_LOWER_BREAKOUT: BOLL下轨突破
- BOLL_SQUEEZE: BOLL收口形态
- BOLL_EXPANSION: BOLL开口形态
- BOLL_MIDDLE_SUPPORT: BOLL中轨支撑/阻力

#### MA (移动平均线)
- MA_GOLDEN_CROSS: MA金叉形态（短期MA上穿长期MA）
- MA_DEATH_CROSS: MA死叉形态（短期MA下穿长期MA）
- MA_BULLISH_ALIGNMENT: MA多头排列
- MA_BEARISH_ALIGNMENT: MA空头排列
- MA_SUPPORT: MA支撑形态

#### EMA (指数移动平均线)
- EMA_GOLDEN_CROSS: EMA金叉形态
- EMA_DEATH_CROSS: EMA死叉形态
- EMA_TREND_CONFIRMATION: EMA趋势确认
- EMA_DIVERGENCE: EMA背离形态
- EMA_SUPPORT_RESISTANCE: EMA支撑阻力

## 文件结构

```
tests/reverse_validation/
├── README.md                          # 本文档
├── pattern_data_generator.py          # 形态数据生成器
├── reverse_validation_framework.py    # 反向验证测试框架
├── run_reverse_validation.py          # 主执行脚本
├── test_framework.py                  # 框架功能测试脚本
└── results/                           # 测试结果输出目录
    ├── reverse_validation_report_*.md # 测试报告（Markdown格式）
    └── reverse_validation_results_*.json # 测试结果（JSON格式）
```

## 使用方法

### 1. 基本测试

测试所有核心指标：

```bash
cd /Users/hacker/PycharmProjects/freedom
python tests/reverse_validation/run_reverse_validation.py
```

### 2. 指定指标测试

测试特定指标：

```bash
python tests/reverse_validation/run_reverse_validation.py --indicators RSI MACD KDJ
```

### 3. 自定义输出

指定输出目录和报告格式：

```bash
python tests/reverse_validation/run_reverse_validation.py \
    --output-dir my_results \
    --report-format markdown \
    --verbose
```

### 4. 框架功能测试

在运行完整测试之前，可以先测试框架基本功能：

```bash
python tests/reverse_validation/test_framework.py
```

## 参数说明

### run_reverse_validation.py 参数

- `--indicators`: 要测试的指标列表，可选值：KDJ, RSI, MACD, BOLL, MA, EMA
- `--output-dir`: 输出目录，默认为 `tests/reverse_validation/results`
- `--report-format`: 报告格式，可选值：markdown, json, both（默认）
- `--verbose`: 详细输出模式

## 输出说明

### 测试报告 (Markdown)

包含以下内容：
- 总体统计（总测试数、成功率、平均匹配分）
- 指标详细结果（每个指标的成功率和形态识别详情）
- 最佳和最差指标分析
- 改进建议

### 测试结果 (JSON)

包含完整的测试数据，可用于进一步分析：
- 每个指标的详细测试结果
- 每个形态的匹配评分
- 预期形态 vs 识别形态的对比
- 错误信息和调试数据

## 评分标准

### 匹配评分算法

1. **形态匹配**: 检查识别出的形态是否包含预期形态的关键词
2. **模糊匹配**: 使用包含关系进行匹配（如 "RSI超买" 匹配 "RSI_OVERBOUGHT"）
3. **评分计算**: 匹配数量 / 预期形态数量
4. **成功标准**: 匹配评分 > 0.5 认为测试成功

### 整体评价标准

- **优秀**: 成功率 ≥ 80%
- **良好**: 成功率 ≥ 60%
- **需要改进**: 成功率 < 60%

## 快速开始

### 1. 环境准备

确保您的Python环境已安装必要的依赖：

```bash
pip install pandas numpy
```

### 2. 运行演示脚本

最简单的开始方式是运行演示脚本：

```bash
cd /Users/hacker/PycharmProjects/freedom
python3 tests/reverse_validation/demo_single_indicator.py RSI --save-results
```

这将：
- 生成RSI指标的5种形态数据
- 模拟形态识别过程
- 显示详细的验证结果
- 保存结果到JSON文件

### 3. 查看结果

演示脚本会输出类似以下的结果：

```
============================================================
单个指标反向验证演示
============================================================
测试指标: RSI
开始时间: 2025-06-22 11:12:57

开始验证指标: RSI
--------------------------------------------------
  测试形态: RSI_OVERBOUGHT
    数据点数: 50
    价格范围: 100.00 - 132.01
    结果: ✅ 成功 (匹配分: 0.667)
    预期形态: 超买, 高位, overbought
    识别形态: RSI超买, 高位
    价格趋势: 30.65%

============================================================
测试总结
============================================================
指标: RSI
总形态数: 5
成功识别: 3
识别失败: 2
成功率: 60.00%
平均匹配分: 0.367
建议: 指标形态识别表现良好，但仍有改进空间
```

## 详细使用指南

### 演示脚本使用

`demo_single_indicator.py` 是一个独立的演示脚本，不依赖复杂的系统环境：

```bash
# 基本用法
python3 tests/reverse_validation/demo_single_indicator.py <INDICATOR>

# 支持的指标
python3 tests/reverse_validation/demo_single_indicator.py RSI
python3 tests/reverse_validation/demo_single_indicator.py MACD
python3 tests/reverse_validation/demo_single_indicator.py KDJ
python3 tests/reverse_validation/demo_single_indicator.py BOLL
python3 tests/reverse_validation/demo_single_indicator.py MA
python3 tests/reverse_validation/demo_single_indicator.py EMA

# 保存详细结果
python3 tests/reverse_validation/demo_single_indicator.py RSI --save-results

# 指定输出文件
python3 tests/reverse_validation/demo_single_indicator.py RSI --save-results --output my_results.json

# 查看帮助
python3 tests/reverse_validation/demo_single_indicator.py --help
```

### 完整框架使用

如果您的环境支持完整的选股系统，可以使用完整的验证框架：

```bash
# 测试所有核心指标
python3 tests/reverse_validation/run_reverse_validation.py

# 测试特定指标
python3 tests/reverse_validation/run_reverse_validation.py --indicators RSI MACD

# 自定义输出
python3 tests/reverse_validation/run_reverse_validation.py \
    --indicators RSI \
    --output-dir my_results \
    --report-format markdown \
    --verbose
```

### 框架功能测试

在使用之前，建议先运行功能测试：

```bash
python3 tests/reverse_validation/test_framework.py
```

## 命令行参数详解

### demo_single_indicator.py 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `indicator` | 位置参数 | 是 | 要测试的指标名称 (RSI/MACD/KDJ/BOLL/MA/EMA) |
| `--save-results` | 标志 | 否 | 保存详细结果到JSON文件 |
| `--output`, `-o` | 字符串 | 否 | 指定输出文件路径 |
| `--help`, `-h` | 标志 | 否 | 显示帮助信息 |

### run_reverse_validation.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--indicators` | 列表 | 所有核心指标 | 要测试的指标列表 |
| `--output-dir` | 字符串 | `tests/reverse_validation/results` | 输出目录 |
| `--report-format` | 选择 | `both` | 报告格式 (markdown/json/both) |
| `--verbose`, `-v` | 标志 | 否 | 详细输出模式 |

## 输出文件说明

### JSON结果文件

演示脚本生成的JSON文件包含：

```json
{
  "indicator": "RSI",
  "total_patterns": 5,
  "successful_patterns": 3,
  "failed_patterns": 2,
  "success_rate": 0.6,
  "average_score": 0.367,
  "pattern_results": {
    "RSI_OVERBOUGHT": {
      "indicator": "RSI",
      "pattern_name": "RSI_OVERBOUGHT",
      "expected_patterns": ["超买", "高位", "overbought"],
      "identified_patterns": ["RSI超买", "高位"],
      "match_score": 0.667,
      "is_successful": true,
      "price_trend": "30.65%",
      "data_points": 50,
      "price_range": "100.00 - 132.01"
    }
  },
  "summary": {
    "success_rate": "60.00%",
    "average_score": "0.367",
    "recommendation": "指标形态识别表现良好，但仍有改进空间"
  }
}
```

### Markdown报告文件

完整框架生成的Markdown报告包含：
- 总体统计信息
- 每个指标的详细结果表格
- 最佳和最差指标分析
- 改进建议和下一步行动

## 故障排除指南

### 常见问题及解决方案

#### 1. 导入错误

**问题**: `ModuleNotFoundError: No module named 'tests.helper'`

**解决方案**:
```bash
# 确保在项目根目录运行
cd /Users/hacker/PycharmProjects/freedom
python3 tests/reverse_validation/demo_single_indicator.py RSI

# 或者设置PYTHONPATH
export PYTHONPATH=/Users/hacker/PycharmProjects/freedom:$PYTHONPATH
```

#### 2. 依赖缺失

**问题**: `ModuleNotFoundError: No module named 'pandas'`

**解决方案**:
```bash
# 安装必要依赖
pip install pandas numpy

# 或使用conda
conda install pandas numpy
```

#### 3. 权限错误

**问题**: `PermissionError: [Errno 13] Permission denied`

**解决方案**:
```bash
# 确保有写入权限
chmod +w tests/reverse_validation/

# 或指定其他输出目录
python3 tests/reverse_validation/demo_single_indicator.py RSI --save-results --output ~/my_results.json
```

#### 4. 数据格式错误

**问题**: 生成的数据不符合预期格式

**解决方案**:
```bash
# 运行框架测试检查数据格式
python3 tests/reverse_validation/test_framework.py

# 检查输出中的"数据格式标准化测试"部分
```

#### 5. 内存不足

**问题**: 大量数据生成时内存不足

**解决方案**:
- 减少测试的指标数量
- 使用演示脚本而非完整框架
- 增加系统内存或使用更强大的机器

### 调试技巧

#### 1. 启用详细输出

```bash
# 演示脚本默认已有详细输出
python3 tests/reverse_validation/demo_single_indicator.py RSI

# 完整框架使用verbose模式
python3 tests/reverse_validation/run_reverse_validation.py --verbose
```

#### 2. 检查生成的数据

```python
# 在Python中检查数据
from tests.reverse_validation.pattern_data_generator import PatternDataGenerator

generator = PatternDataGenerator()
rsi_patterns = generator.generate_rsi_patterns()

# 检查数据结构
for name, data in rsi_patterns.items():
    print(f"{name}: {len(data)} rows, columns: {list(data.columns)}")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
```

#### 3. 逐步测试

```bash
# 1. 先测试框架基本功能
python3 tests/reverse_validation/test_framework.py

# 2. 测试单个指标
python3 tests/reverse_validation/demo_single_indicator.py RSI

# 3. 测试其他指标
python3 tests/reverse_validation/demo_single_indicator.py MACD

# 4. 运行完整测试（如果环境支持）
python3 tests/reverse_validation/run_reverse_validation.py --indicators RSI
```

### 性能优化建议

#### 1. 减少数据量

如果测试运行缓慢，可以修改数据生成器中的periods参数：

```python
# 在pattern_data_generator.py中
sequence_specs = [
    {
        'type': 'trend',
        'periods': 20,  # 减少从30到20
        'start_price': 100,
        'end_price': 130,
        'volume_trend': 'follow_price'
    }
]
```

#### 2. 并行处理

对于多个指标的测试，可以考虑并行处理：

```bash
# 分别在不同终端运行
python3 tests/reverse_validation/demo_single_indicator.py RSI &
python3 tests/reverse_validation/demo_single_indicator.py MACD &
python3 tests/reverse_validation/demo_single_indicator.py KDJ &
```

### 环境兼容性

#### Python版本要求

- **最低要求**: Python 3.6+
- **推荐版本**: Python 3.8+
- **测试版本**: Python 3.9, 3.10, 3.11

#### 操作系统支持

- ✅ **macOS**: 完全支持
- ✅ **Linux**: 完全支持
- ⚠️ **Windows**: 基本支持（路径分隔符可能需要调整）

#### 依赖版本

```txt
pandas>=1.0.0
numpy>=1.18.0
```

### 获取帮助

如果遇到其他问题：

1. **查看日志**: 检查控制台输出中的错误信息
2. **检查文档**: 重新阅读本README文档
3. **运行测试**: 使用`test_framework.py`诊断问题
4. **简化测试**: 从演示脚本开始，逐步增加复杂度
5. **检查环境**: 确保Python版本和依赖满足要求

## 扩展开发

### 添加新指标支持

1. **在PatternDataGenerator中添加新方法**:

```python
def generate_new_indicator_patterns(self) -> Dict[str, pd.DataFrame]:
    """生成新指标的形态数据"""
    patterns = {}
    patterns['NEW_PATTERN'] = self._generate_new_pattern()
    return patterns

def _generate_new_pattern(self) -> pd.DataFrame:
    """生成具体形态数据"""
    # 实现形态生成逻辑
    pass
```

2. **在验证器中添加预期形态映射**:

```python
self.expected_patterns.update({
    'NEW_PATTERN': ['新形态', 'new_pattern', '特征描述']
})
```

3. **更新主要方法以支持新指标**:

```python
elif indicator.upper() == 'NEW_INDICATOR':
    patterns = self.pattern_generator.generate_new_indicator_patterns()
```

### 自定义验证逻辑

可以继承`SimplifiedIndicatorValidator`类并重写验证方法：

```python
class CustomValidator(SimplifiedIndicatorValidator):
    def simulate_indicator_analysis(self, indicator, pattern_name, pattern_data):
        # 实现自定义验证逻辑
        return super().simulate_indicator_analysis(indicator, pattern_name, pattern_data)
```

### 集成真实指标分析器

要集成真实的技术指标分析器，需要：

1. 替换`simulate_indicator_analysis`方法
2. 导入相应的指标分析模块
3. 处理真实分析结果的数据格式
4. 更新形态匹配逻辑