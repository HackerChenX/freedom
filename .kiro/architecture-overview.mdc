# 股票分析系统架构总览

## 系统架构概述

本系统采用**六层架构模型（L1-L6）**，严格遵循分层设计原则：

### 架构分层结构

```
L6: 应用层 (Application Layer)
├── bin/                    # 可执行脚本和主程序入口
└── scripts/               # 工具脚本和批处理程序

L5: 业务逻辑层 (Business Logic Layer)  
├── analysis/              # 分析模块（市场分析、买点分析）
├── strategy/              # 选股策略模块
└── formula/               # 公式计算模块

L4: 服务层 (Service Layer)
├── indicators/            # 技术指标服务
├── api/                   # 外部API接口实现
└── monitoring/            # 监控服务

L3: 数据访问层 (Data Access Layer)
├── db/                    # 数据库接口和查询执行器
│   ├── query_executor.py  # 统一查询执行器
│   ├── sql_manager.py     # SQL查询管理器
│   └── clickhouse_db.py   # ClickHouse数据库连接
└── crawler/               # 数据爬取模块

L2: 基础设施层 (Infrastructure Layer)
├── config/                # 配置管理
├── utils/                 # 工具模块
├── enums/                 # 枚举定义
└── logs/                  # 日志管理

L1: 数据层 (Data Layer)
├── data/                  # 数据存储目录
└── sql/                   # SQL脚本文件
```

## 核心设计原则

### 1. 分层依赖原则
- **单向依赖**: 上层可以依赖下层，下层不能依赖上层
- **接口隔离**: 各层通过标准接口交互
- **依赖注入**: 使用依赖注入而非全局单例

### 2. 数据访问统一
- **统一查询接口**: 所有SQL查询通过 [query_executor.py](mdc:db/query_executor.py) 执行
- **查询管理**: 使用 [sql_manager.py](mdc:db/sql_manager.py) 管理查询模板
- **连接池管理**: 通过 [clickhouse_db.py](mdc:db/clickhouse_db.py) 管理数据库连接

### 3. 配置管理规范
- **分层配置**: 环境变量 > 配置文件 > 默认值
- **配置中心**: 统一通过 [config/](mdc:config/) 目录管理
- **敏感信息**: 加密存储，支持环境变量覆盖

### 4. 错误处理统一
- **异常处理**: 使用统一的异常处理机制
- **日志记录**: 通过 [utils/logger.py](mdc:utils/logger.py) 统一日志管理
- **降级处理**: 关键功能提供降级处理方案

## 关键模块说明

### 主程序入口
- [bin/main.py](mdc:bin/main.py) - 系统主入口
- [bin/stock_select.py](mdc:bin/stock_select.py) - 选股功能入口

### 数据访问层
- [db/query_executor.py](mdc:db/query_executor.py) - 统一查询执行器
- [db/sql_manager.py](mdc:db/sql_manager.py) - SQL查询管理器
- [db/clickhouse_db.py](mdc:db/clickhouse_db.py) - ClickHouse数据库接口

### 配置管理
- [config/database_config_manager.py](mdc:config/database_config_manager.py) - 数据库配置管理
- [config/docker_config.json](mdc:config/docker_config.json) - Docker配置文件

### 工具模块
- [utils/logger.py](mdc:utils/logger.py) - 日志管理
- [utils/path_utils.py](mdc:utils/path_utils.py) - 路径工具
- [utils/date_utils.py](mdc:utils/date_utils.py) - 日期工具

## 架构合规性要求

### 必须遵循的规则
1. **禁止直接数据库访问**: 必须通过查询执行器访问数据库
2. **禁止全局单例**: 使用依赖注入替代@singleton装饰器
3. **禁止硬编码配置**: 所有配置项必须可配置化
4. **禁止跨层依赖**: 严格遵循分层依赖原则
5. **禁止通配符导入**: 使用明确的导入语句

### 推荐的最佳实践
1. **使用类型提示**: 所有函数和方法使用类型提示
2. **文档字符串**: 使用Google风格的文档字符串
3. **异常处理**: 适当的异常处理和日志记录
4. **单元测试**: 为核心功能编写单元测试
5. **代码审查**: 遵循代码审查检查清单
alwaysApply: true
description: 股票分析系统架构总览和设计原则
---
