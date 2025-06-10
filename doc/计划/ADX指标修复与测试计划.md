[模式：计划]

## ADX 指标修复与测试完善计划

本计划旨在详细说明如何修复 `ADX` 指标的初始化错误，并为其编写一套全面的单元测试，以确保其功能的正确性、健壮性和覆盖率。

### 1. 修复 `indicators/adx.py` 中的初始化错误

1.1. **修正 `__init__` 方法**:
    - **操作**: 编辑 `indicators/adx.py` 文件，将 `ADX` 类中 `__init__` 方法的文档字符串（docstring）移动到该方法体的第一行，即在 `self.REQUIRED_COLUMNS` 赋值语句之前。
    - **目的**: 遵循 Python 的语法规范，确保父类 `BaseIndicator` 的 `super().__init__` 方法被正确调用，从而使指标能够成功设置 `name` 属性并注册到指标工厂中。

### 2. 增强 `tests/unit/test_adx.py` 的测试用例

为达到全面的测试覆盖，需要在现有的 `test_adx.py` 文件中补充以下测试用例：

2.1. **添加形态识别测试 (`test_pattern_detection`)**:
    - **操作**: 在 `TestADX` 类中新增一个名为 `test_pattern_detection` 的测试方法。
    - **目的**: 验证 `get_patterns` 方法能否在不同的市场数据中正确识别出 ADX 相关的形态（如强趋势、弱趋势、DI交叉等）。
    - **实现细节**:
        - 创建一个包含明确趋势（上升、下降、盘整）的数据集。
        - 调用 `indicator.calculate()` 和 `indicator.get_patterns()`。
        - 断言返回的 `patterns` 是一个非空的 `pd.DataFrame`。
        - 断言返回的 DataFrame 包含预期的形态信息列，如 `pattern_id`, `display_name`, `strength`。

2.2. **添加评分逻辑测试 (`test_raw_score_calculation`)**:
    - **操作**: 在 `TestADX` 类中新增一个名为 `test_raw_score_calculation` 的测试方法。
    - **目的**: 验证 `calculate_raw_score` 方法的输出是否符合预期。
    - **实现细节**:
        - 调用 `indicator.calculate_raw_score()`。
        - 断言返回的结果是一个 `pd.Series`。
        - 断言该 Series 中的所有非 NaN 值都在 0 到 100 的有效范围内。

2.3. **添加信号生成测试 (`test_signal_generation`)**:
    - **操作**: 在 `TestADX` 类中新增一个名为 `test_signal_generation` 的测试方法。
    - **目的**: 验证 `generate_trading_signals` 方法能否在适当条件下正确生成交易信号。
    - **实现细节**:
        - 创建一个能够明确触发 DI 金叉/死叉形态的数据集。
        - 调用 `indicator.generate_trading_signals()`。
        - 断言返回的结果是一个字典。
        - 断言该字典中包含 `buy_signal` 和 `sell_signal` 键，并且它们的值是布尔型的 `pd.Series`。

2.4. **添加边界条件测试 (`test_edge_cases`)**:
    - **操作**: 在 `TestADX` 类中新增一个名为 `test_edge_cases` 的测试方法。
    - **目的**: 确保指标在处理异常数据时不会崩溃，并且其行为符合预期。
    - **实现细节**:
        - **数据不足**: 使用长度小于指标计算周期的数据进行测试，验证指标是否能优雅地处理（例如，返回一个包含 NaN 的 DataFrame）并且不抛出异常。
        - **包含 NaN 值的输入**: 使用在其基础列（open, high, low, close）中包含 `NaN` 的数据进行测试，验证计算结果的健壮性。

---
### IMPLEMENTATION CHECKLIST:

1.  编辑 `indicators/adx.py` 文件，将 `ADX` 类 `__init__` 方法的文档字符串移到方法体的最开始。
2.  在 `tests/unit/test_adx.py` 的 `TestADX` 类中，添加 `test_pattern_detection` 方法，用于测试形态识别功能。
3.  在 `tests/unit/test_adx.py` 的 `TestADX` 类中，添加 `test_raw_score_calculation` 方法，用于测试原始评分的计算。
4.  在 `tests/unit/test_adx.py` 的 `TestADX` 类中，添加 `test_signal_generation` 方法，用于测试交易信号的生成。
5.  在 `tests/unit/test_adx.py` 的 `TestADX` 类中，添加 `test_edge_cases` 方法，用于测试指标对数据不足和包含 NaN 等边界情况的处理能力。 