[模式：计划]

## Volume 指标重构计划

本计划旨在通过重构 `Volume` 指标的目录结构，彻底解决其在项目中引发的循环导入问题，并优化项目结构使其更加清晰和健壮。

### 1. 读取 `indicators/vol.py` 内容

1.1. **读取源文件**:
    - **操作**: 读取并暂存 `indicators/vol.py` 的全部文件内容。
    - **目的**: 获取 `VOL` 类的完整实现代码，为下一步的迁移做准备。

### 2. 创建并写入新文件

2.1. **创建 `indicators/volume/vol.py`**:
    - **操作**: 在 `indicators/volume/` 目录下创建一个新的 `vol.py` 文件，并将上一步读取到的 `VOL` 类的代码完整地写入该新文件。
    - **目的**: 将 `VOL` 指标的物理实现迁移到其逻辑上正确的 `volume` 包中。

### 3. 更新 `indicators/volume/__init__.py`

3.1. **修改导入语句**:
    - **操作**: 编辑 `indicators/volume/__init__.py` 文件，将其中的导入语句 `from indicators.vol import VOL as Volume` 修改为本地相对导入 `from .vol import VOL as Volume`。
    - **目的**: 使 `volume` 包从其内部的 `vol.py` 模块导出 `Volume` 类，彻底解决跨级导入问题，使包的结构更加内聚。

### 4. 清理废弃文件

4.1. **删除 `indicators/vol.py`**:
    - **操作**: 删除项目顶层的 `indicators/vol.py` 文件。
    - **目的**: 完成重构，移除冗余且结构错误的旧文件，保持项目代码库的整洁性。

---
### IMPLEMENTATION CHECKLIST:

1.  读取 `indicators/vol.py` 的完整内容并暂存。
2.  在 `indicators/volume/` 目录下创建一个新文件 `vol.py`，并将第一步中读取的内容写入该文件。
3.  修改 `indicators/volume/__init__.py` 文件，将导入语句更改为 `from .vol import VOL as Volume`。
4.  删除 `indicators/vol.py` 文件。 