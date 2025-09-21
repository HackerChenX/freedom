# VnPy Freedom 本地开发环境

## 概述

本项目已配置为使用本地vnpy模块，便于二次开发和定制。所有模块都引用 `Core_Framework/vnpy` 中的本地vnpy代码，而不是系统安装的vnpy包。

## 开发环境配置

### 自动配置
运行以下命令自动配置开发环境：
```bash
python setup_local_development.py
```

### 手动配置
如果需要手动配置，请确保：
1. `Core_Framework/vnpy` 路径在 `sys.path` 的最前面
2. 所有模块路径都已添加到 `sys.path`
3. 启动脚本包含正确的路径配置

## 使用方式

### 启动系统
```bash
# 快速启动
python quick_start.py

# 完整启动器
python run_vnpy_freedom.py

# 检查模块状态
python check_modules.py
```

### 导入模块
```python
# 使用导入辅助器
from vnpy_freedom_importer import import_gateway, import_app, list_modules

# 导入网关
XtpGateway = import_gateway("xtp")

# 导入应用
CtaStrategyApp = import_app("ctastrategy")

# 列出可用模块
modules = list_modules()
```

### 直接导入
```python
# 由于路径已配置，可以直接导入
from vnpy.trader.engine import MainEngine
from vnpy_xtp import XtpGateway
from vnpy_ctastrategy import CtaStrategyApp
```

## 开发指南

### 修改核心框架
- 直接修改 `Core_Framework/vnpy` 中的代码
- 修改后重启应用即可生效
- 建议使用版本控制跟踪修改

### 开发新模块
1. 在对应分类目录下创建新模块
2. 遵循 `vnpy_xxx` 命名规范
3. 实现标准的VnPy接口
4. 添加到启动脚本的模块列表

### 调试技巧
- 使用 `vnpy_freedom_dev.json` 配置调试选项
- 启用 `debug_mode` 获取详细日志
- 使用 `auto_reload` 自动重载模块

## 文件说明

- `vnpy_freedom.pth`: Python路径配置文件
- `vnpy_freedom_dev.json`: 开发环境配置
- `vnpy_freedom_importer.py`: 导入辅助模块
- `setup_local_development.py`: 环境配置脚本

## 注意事项

1. **路径优先级**: 本地vnpy路径优先于系统安装的vnpy
2. **模块冲突**: 避免同时安装系统vnpy和使用本地vnpy
3. **依赖管理**: 确保所有依赖都已正确安装
4. **版本兼容**: 保持模块间的版本兼容性

## 故障排除

### 导入错误
- 检查 `sys.path` 配置
- 确认模块路径正确
- 验证 `__init__.py` 文件存在

### 模块未找到
- 运行 `python check_modules.py` 检查状态
- 确认模块目录结构正确
- 检查模块命名规范

### 性能问题
- 减少不必要的模块导入
- 使用延迟导入
- 优化路径配置
