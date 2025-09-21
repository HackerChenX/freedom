# 🚀 VnPy Freedom - 量化交易平台

[![GitHub](https://img.shields.io/github/license/HackerChenX/freedom)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![VnPy](https://img.shields.io/badge/VnPy-3.0+-green.svg)](https://github.com/vnpy/vnpy)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-Database-orange.svg)](https://clickhouse.com/)

**专业级量化交易解决方案，集成完整的策略开发、回测分析、实盘交易生态系统**

---

## ✨ 核心特性

### 🎯 交易策略引擎
- **缩量回踩策略** - 严格按照通达信公式实现的9重条件筛选
- **二波企稳策略** - 多时间框架结合的趋势确认策略
- **决策分析系统** - 完整的策略决策过程追踪和可视化
- **T+1交易制度** - 针对A股市场的专业交易规则支持

### 📊 数据服务生态
- **ClickHouse时序数据库** - 高性能历史数据存储和查询
- **EFinance数据源** - 免费开源的A股实时数据接入
- **多时间框架支持** - 自动K线数据合成（15分钟→30分钟→1小时）
- **智能数据管理** - 数据下载、清洗、验证一体化

### 🔧 专业级回测系统
- **CTA策略回测** - 单品种策略完整回测分析
- **组合策略回测** - 多品种投资组合策略支持
- **风险管理集成** - 实时风险监控和仓位管理
- **绩效分析报告** - 详细的策略表现和风险指标

### 🖥️ 智能用户界面
- **一键式平台启动** - 智能模块发现和自动加载
- **实时K线图表** - 集成技术指标和交易信号显示
- **决策时间线** - 策略每日决策过程的完整记录
- **模块化设计** - 灵活的功能组件按需加载

---

## 🏗️ 系统架构

```
VnPy Freedom
├── 📁 Core_Framework/           # VnPy核心框架
│   ├── vnpy/                   # 主要VnPy库
│   └── vnag/                   # 图形分析工具
├── 📁 Strategy_Applications/    # 策略应用模块
│   ├── Trading_Strategies/     # 交易策略
│   ├── Backtesting_Analysis/   # 回测分析
│   ├── Data_Management/        # 数据管理
│   └── Risk_Management/        # 风险管理
├── 📁 Trading_Gateways/        # 交易网关
│   ├── Cryptocurrency_Exchanges/ # 数字货币交易所
│   ├── Domestic_Securities/    # 国内证券接口
│   └── International_Markets/  # 国际市场接口
├── 📁 Data_Services/           # 数据服务
│   ├── Market_Data/           # 市场数据
│   └── Special_Data/          # 特殊数据
└── 📁 Database_Interfaces/     # 数据库接口
    ├── Time_Series_Databases/ # 时序数据库
    └── Relational_Databases/ # 关系型数据库
```

---

## 🚀 快速开始

### 1. 环境要求
```bash
Python 3.11+
ClickHouse (可选，用于历史数据存储)
```

### 2. 安装依赖
```bash
# 克隆项目
git clone https://github.com/HackerChenX/freedom.git
cd freedom

# 创建虚拟环境
python3.11 -m venv vnpy_freedom_env
source vnpy_freedom_env/bin/activate  # Linux/macOS
# vnpy_freedom_env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置数据源
编辑 `vt_setting.json` 配置数据源：
```json
{
    "datafeed.name": "efinance",
    "database.name": "clickhouse",
    "database.host": "localhost",
    "database.port": 8123,
    "database.database": "vnpy",
    "database.user": "default",
    "database.password": "123456"
}
```

### 4. 启动平台
```bash
python3.11 run_vnpy_freedom.py
```

---

## 📈 策略展示

### 缩量回踩策略
**专业级选股算法，基于通达信公式实现**

🔍 **9重条件筛选**：
- ✅ 10/20日均线上移
- ✅ 60/120日均线上移  
- ✅ 回踩10/20/30日均线
- ✅ 近60日至少一次涨幅>7%
- ✅ 110日振幅>8.1%至少两次
- ✅ 15分钟吸筹信号
- ✅ 连续缩量阴线
- ✅ KDJ/DEA任一上移
- ✅ 无负面信号过滤

📊 **决策分析界面**：
```
📅 2025-01-24 交易日决策记录：
🕐 09:30:00 价格: 10.45
📊 条件评估：
  ✅ 10/20均线上移    ✅ 60/120均线上移
  ✅ 回踩均线        ✅ 60日涨幅>7%
  ❌ 110日振幅>8.1%  ✅ 15分钟吸筹
  ✅ 连续缩量        ✅ KDJ/DEA上移
  ✅ 无负面信号

📈 满足条件: 8/9
🚀 **执行买入操作**
💡 买入理由: 强烈关注信号触发：8个条件满足...
```

---

## 🛠️ 核心功能

### 智能策略引擎
- **多时间框架分析** - 日线+15分钟联合决策
- **技术指标集成** - KDJ、MACD、均线系统
- **风险信号过滤** - 智能排除不利市场环境
- **回测验证系统** - 历史数据验证策略有效性

### 专业数据管理
- **高频数据存储** - ClickHouse时序数据库
- **实时数据更新** - EFinance免费数据源
- **数据质量监控** - 自动数据验证和清洗
- **多源数据整合** - 支持TuShare、米筐等数据源

### 用户友好界面
- **可视化策略监控** - 实时策略状态展示
- **交互式图表** - 集成技术分析工具
- **决策过程追踪** - 完整的买卖决策记录
- **一键启动管理** - 简化的平台操作流程

---

## 📚 文档与教程

- 📖 [项目Wiki](docs/) - 详细的开发文档
- 🎯 [策略开发指南](docs/base/) - 策略开发最佳实践
- 🔧 [配置说明](配置文件说明.md) - 系统配置详解
- 📊 [使用教程](LOCAL_DEVELOPMENT.md) - 本地开发指南

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目基于 [MIT许可证](LICENSE) - 查看 LICENSE 文件了解详情

---

## 🙏 致谢

- [VnPy](https://github.com/vnpy/vnpy) - 优秀的量化交易框架
- [ClickHouse](https://clickhouse.com/) - 高性能时序数据库
- [EFinance](https://github.com/Micro-sheep/efinance) - 免费股票数据接口

---

## 📬 联系方式

- 项目主页: [https://github.com/HackerChenX/freedom](https://github.com/HackerChenX/freedom)
- 问题反馈: [Issues](https://github.com/HackerChenX/freedom/issues)
- 功能建议: [Discussions](https://github.com/HackerChenX/freedom/discussions)

---

<div align="center">

**🌟 如果这个项目对您有帮助，请点击 Star 支持我们！🌟**

![GitHub stars](https://img.shields.io/github/stars/HackerChenX/freedom?style=social)
![GitHub forks](https://img.shields.io/github/forks/HackerChenX/freedom?style=social)

*让量化交易更简单，让投资决策更智能* 💎

</div>