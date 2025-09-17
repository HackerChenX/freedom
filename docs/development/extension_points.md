# BaseIndicator扩展点设计文档

## 概述

BaseIndicator提供了多个扩展点，允许开发者在不修改核心逻辑的情况下定制指标行为。

## 扩展点列表

### 1. validate_data(self, data: pd.DataFrame) -> bool
**用途**: 验证输入数据的有效性
**默认行为**: 检查数据是否为空，是否包含必要的列
**扩展建议**:
- 添加特定的数据质量检查
- 验证数据的时间范围
- 检查数据的完整性

### 2. preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame
**用途**: 预处理输入数据
**默认行为**: 返回原始数据
**扩展建议**:
- 数据清洗和去噪
- 数据格式转换
- 缺失值处理

### 3. postprocess_result(self, result: pd.DataFrame) -> pd.DataFrame
**用途**: 后处理计算结果
**默认行为**: 返回原始结果
**扩展建议**:
- 结果平滑处理
- 异常值处理
- 结果格式化

## 使用示例

```python
class CustomIndicator(BaseIndicator):
    def validate_data(self, data: pd.DataFrame) -> bool:
        # 自定义数据验证逻辑
        if not super().validate_data(data):
            return False

        # 检查特定列是否存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # 自定义预处理逻辑
        processed_data = super().preprocess_data(data)

        # 填充缺失值
        processed_data = processed_data.fillna(method='forward')

        return processed_data

    def postprocess_result(self, result: pd.DataFrame) -> pd.DataFrame:
        # 自定义后处理逻辑
        processed_result = super().postprocess_result(result)

        # 平滑处理
        for col in processed_result.columns:
            if col.endswith('_value'):
                processed_result[col] = processed_result[col].rolling(window=3).mean()

        return processed_result
```

## 最佳实践

1. **总是调用父类方法**: 确保基础功能正常工作
2. **保持方法签名一致**: 不要改变方法的参数和返回类型
3. **添加适当的错误处理**: 确保扩展点的健壮性
4. **文档化自定义行为**: 清楚地说明扩展点的自定义逻辑
