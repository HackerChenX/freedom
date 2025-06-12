[模式: 研究]

# MACD 指标修复日志

## 1. 研究阶段

### 文件分析

-   **指标实现**: `indicators/macd.py`
    -   `MACD` 类继承自 `BaseIndicator`。
    -   功能包括：MACD 值计算、多种技术形态（金叉、死叉、背离等）的识别、形态注册 (`PatternRegistry`) 和信号评分。
    -   实现了一个私有的 `_calculate` 方法用于核心计算，并提供了多个公共接口如 `get_patterns`, `get_signals`, `calculate_raw_score` 等。
-   **单元测试**: `tests/unit/test_macd.py`
    -   使用 `unittest` 框架。
    -   通过 `TestDataGenerator` 生成特定价格序列来测试金叉、死叉和背离等形态。
    -   测试重点是 `get_patterns` 方法的输出。
-   **废弃脚本**: `test_macd.py` 和 `test_macd_simple.py`
    -   位于项目根目录。
    -   似乎是用于手动执行的旧测试脚本。
    -   测试的接口 (`calculate`) 和期望的列名 (`macd`, `signal`, `hist`) 与当前实现和单元测试不一致，表明其已过时。

### 代码结构与接口观察

-   `BaseIndicator` (`indicators/base_indicator.py`) 定义了一个公共的 `calculate` 方法，该方法调用由子类（如 `MACD`）实现的抽象方法 `_calculate`。
-   `MACD` 类同时实现了 `_calculate` 和 `get_patterns`。`get_patterns` 提供了更丰富的功能，包含了计算和形态分析。
-   接口不一致：单元测试 (`tests/unit/test_macd.py`) 调用 `get_patterns`，而废弃脚本调用 `calculate`。这反映了指标功能迭代后，测试脚本未同步更新。

### 初步结论

-   核心逻辑位于 `indicators/macd.py`。
-   单元测试 `tests/unit/test_macd.py` 是当前有效且相关的测试。
-   根目录的两个测试脚本 (`test_macd.py`, `test_macd_simple.py`) 已过时，应予清理。
-   修复工作应围绕 `indicators/macd.py` 和 `tests/unit/test_macd.py` 展开。

## 2. 计划阶段

### 目标
根据研究阶段的结论，全面修复并增强 `MACD` 指标的测试套件，确保其功能的健壮性、准确性和完整性，并清理过时的测试脚本。

### 详细计划

#### 第 1 阶段：环境清理与准备

1.  **清理过时脚本**：删除项目根目录下的两个废弃测试脚本，以消除混乱并确保唯一测试源。
    -   删除 `test_macd.py`
    -   删除 `test_macd_simple.py`
2.  **更新进度文档**：修改 `doc/任务安排/指标修复进度表.md`，将 `MACD` 指标的"主要测试脚本"字段更新为正确的路径 `tests/unit/test_macd.py`。

#### 第 2 阶段：核心计算验证

1.  **添加数值准确性测试**：在 `tests/unit/test_macd.py` 中新增一个测试方法 `test_calculation_correctness`。
    -   目的：验证 `_calculate` 方法生成的 `macd_line`, `macd_signal`, `macd_histogram` 数值的准确性。
    -   方法：使用一个公开的、已知结果的短序列数据集（例如，来自 `TA-Lib` 库的验证数据），断言计算结果与预期的精确值在一定误差范围内相等。

#### 第 3 阶段：扩展形态测试覆盖范围

在 `tests/unit/test_macd.py` 中，为 `MACD` 类中已注册但尚未测试的每一种重要形态添加专门的测试方法。

1.  **测试零轴穿越** (`test_zero_cross_patterns`):
    -   生成能触发DIF线上穿和下穿零轴的价格序列。
    -   验证 `MACD_ZERO_CROSS_ABOVE` 和 `MACD_ZERO_CROSS_BELOW` 形态能否被正确识别。
2.  **测试柱状图形态** (`test_histogram_patterns`):
    -   生成能分别导致柱状图持续扩张和收缩的价格序列。
    -   验证 `MACD_HISTOGRAM_EXPANDING` 和 `MACD_HISTOGRAM_CONTRACTING` 形态能否被正确识别。
3.  **测试双顶和双底形态** (`test_double_patterns`):
    -   生成能触发 MACD 指标形成双顶和双底形态的价格序列。
    -   验证 `MACD_DOUBLE_TOP` 和 `MACD_DOUBLE_BOTTOM` 形态能否被正确识别。

#### 第 4 阶段：其他公共接口测试

1.  **测试得分计算** (`test_calculate_raw_score`):
    -   基于一个包含多种形态的复杂价格序列。
    -   调用 `calculate_raw_score` 方法，验证返回的 `pd.Series` 不为空，且其数值的分布符合预期逻辑（例如，金叉点得分为正，死叉点得分为负）。
2.  **测试信号生成** (`test_get_signals`):
    -   调用 `get_signals` 方法。
    -   验证返回的是一个字典，且包含预期的键（如 `buy_signal`, `sell_signal` 等）。
    -   验证字典中的值为布尔型的 `pd.Series`。

#### 第 5 阶段：边缘场景测试

1.  **测试无效数据输入** (`test_edge_cases`):
    -   测试输入数据量过少（例如，少于 `slow_period`）的情况，确保系统能正常处理（例如，返回空的或充满 `NaN` 的 `DataFrame`），而不是崩溃。
    -   测试输入数据包含 `NaN` 值的情况。

#### 第 6 阶段：最终化与文档更新

1.  **更新修复日志**：在 `doc/任务安排/MACD指标修复日志.md` 中记录所有执行步骤和结果。
2.  **更新总体进度**：在 `doc/任务安排/指标修复进度表.md` 中，将 `MACD` 指标的状态更新为"已完成"，并将"是否可用"标记为"是"。

---

### IMPLEMENTATION CHECKLIST:

1.  [Delete file `test_macd.py`]
2.  [Delete file `test_macd_simple.py`]
3.  [Update `doc/任务安排/指标修复进度表.md` to change the test script path for MACD to `tests/unit/test_macd.py`]
4.  [In `tests/unit/test_macd.py`, add a new method `test_calculation_correctness` to verify the numerical accuracy of MACD values against a known dataset]
5.  [In `tests/unit/test_macd.py`, add a new method `test_zero_cross_patterns` to test for `MACD_ZERO_CROSS_ABOVE` and `MACD_ZERO_CROSS_BELOW`]
6.  [In `tests/unit/test_macd.py`, add a new method `test_histogram_patterns` to test for `MACD_HISTOGRAM_EXPANDING` and `MACD_HISTOGRAM_CONTRACTING`]
7.  [In `tests/unit/test_macd.py`, add a new method `test_double_patterns` to test for `MACD_DOUBLE_TOP` and `MACD_DOUBLE_BOTTOM`]
8.  [In `tests/unit/test_macd.py`, add a new method `test_calculate_raw_score` to verify the output of the scoring function]
9.  [In `tests/unit/test_macd.py`, add a new method `test_get_signals` to verify the structure and output of the signal generation function]
10. [In `tests/unit/test_macd.py`, add a new method `test_edge_cases` to handle insufficient data and NaN values]
11. [Append the execution summary to `doc/任务安排/MACD指标修复日志.md`]
12. [Update the status of MACD in `doc/任务安排/指标修复进度表.md` to '已完成' and '是'] 