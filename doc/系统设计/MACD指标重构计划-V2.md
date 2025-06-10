# [模式: 计划 V2]

## 一、 问题背景与根本原因分析

在执行第一版`MACD`重构计划时，单元测试反复失败。失败的核心原因是标准`crossover`交叉检测函数对于`MACD`指标过于敏感。`MACD`的`DIF`线和`DEA`线均为平滑的指数移动平均线，在趋势转换时，它们会紧密缠绕并可能在小范围内多次穿越，导致标准检测逻辑报告多个交叉点，而实际上只发生了一次趋势性交叉。

旧计划试图通过优化测试数据来解决问题，但实践证明此路不通。根本解决方案必须从**改进交叉检测算法**本身入手。

## 二、 新重构方案：引入鲁棒交叉检测

本计划将放弃标准`crossover`函数，转而为`MACD`实现一个定制的、更鲁棒的交叉检测逻辑。

### 1. `indicators/macd.py` 重构方案

#### a. 新增 `_detect_robust_crossover` 方法
- **方法签名**: `_detect_robust_crossover(self, series1: pd.Series, series2: pd.Series, window: int = 3, cross_type: str = 'above') -> pd.Series`
- **参数说明**:
    - `series1`, `series2`: 待检测的两个序列（例如`DIF`和`DEA`）。
    - `window`: 稳定窗口期。一个交叉发生后，必须在该窗口期内保持状态，才被确认为有效交叉。
    - `cross_type`: `'above'` 用于检测金叉（上穿），`'below'` 用于检测死叉（下穿）。
- **实现逻辑**:
    1.  找到所有潜在的交叉点（使用标准`crossover`或`crossunder`）。
    2.  对于每一个潜在交叉点，检查其后的`window`个周期。
    3.  只有当`series1`和`series2`的关系在整个`window`内都保持在交叉后的新状态时，该交叉点才被确认为`True`。
    4.  例如，对于金叉（`cross_type='above'`），在交叉点`i`，必须满足`series1[i+1:i+1+window]`的所有值都大于`series2[i+1:i+1+window]`。

#### b. 更新 `get_patterns` 方法
- 将`patterns_df['MACD_GOLDEN_CROSS'] = self.crossover(dif, dea)`的调用替换为:
  `patterns_df['MACD_GOLDEN_CROSS'] = self._detect_robust_crossover(dif, dea, window=3, cross_type='above')`
- 将`patterns_df['MACD_DEATH_CROSS'] = self.crossunder(dif, dea)`的调用替换为:
  `patterns_df['MACD_DEATH_CROSS'] = self._detect_robust_crossover(dif, dea, window=3, cross_type='below')`

### 2. `tests/unit/test_macd.py` 调整方案

- **无需修改**。测试文件中的数据和断言（`assertEqual(sum, 1)`）在新的鲁棒检测逻辑下应该是正确的，可以直接用于验证新方法的有效性。

## 三、 新实施清单

1.  **文件: `indicators/macd.py`**: 创建一个新的私有方法 `_detect_robust_crossover(self, series1: pd.Series, series2: pd.Series, window: int = 3, cross_type: str = 'above') -> pd.Series`。
2.  **文件: `indicators/macd.py`**: 在 `_detect_robust_crossover` 中，实现鲁棒交叉检测逻辑，该逻辑要求交叉后的状态必须持续`window`个周期。
3.  **文件: `indicators/macd.py`**: 修改 `get_patterns` 方法，使其调用 `_detect_robust_crossover` 来识别金叉和死叉。
4.  **运行测试**: 执行 `tests/unit/test_macd.py` 中的单元测试，验证所有测试是否通过。
5.  **文件: `doc/任务安排/技术债-指标重构清单.md`**: 在所有测试通过后，将`MACD`指标的状态更新为"完成"。
6.  **清理**: 删除旧的计划文件`doc/系统设计/MACD指标重构计划.md`。 