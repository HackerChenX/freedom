# ClickHouse 配置说明

本项目已配置为自动使用 `/Users/hacker/clickhouse_data` 作为 ClickHouse 数据目录。

## 快速启动

### 方法1：使用启动脚本（推荐）
```bash
# 启动 ClickHouse
./scripts/start_clickhouse.sh

# 停止 ClickHouse
./scripts/stop_clickhouse.sh
```

### 方法2：使用别名
在您的 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
source /Users/hacker/PycharmProjects/freedom/scripts/clickhouse_aliases.sh
```

然后您可以使用：
```bash
ch-start    # 启动 ClickHouse
ch-stop     # 停止 ClickHouse
ch-status   # 查看状态
ch-log      # 查看日志
ch-client   # 启动客户端
ch-dbs      # 显示数据库
ch-tables   # 显示表
```

## 配置文件

- **启动脚本**: `scripts/start_clickhouse.sh`
- **停止脚本**: `scripts/stop_clickhouse.sh`
- **配置文件**: `config/clickhouse-config.xml`
- **别名文件**: `scripts/clickhouse_aliases.sh`

## 连接信息

- **HTTP 接口**: http://127.0.0.1:8123
- **原生协议**: 127.0.0.1:9000
- **MySQL 兼容**: 127.0.0.1:9004
- **数据目录**: /Users/hacker/clickhouse_data
- **日志文件**: clickhouse.log

## 数据库信息

当前配置的数据库：
- `stock` - 包含股票数据
  - `stock_info` - 股票信息表 (19,608,204 行)
  - `industry_info` - 行业信息表

## 测试连接

运行测试脚本验证连接：
```bash
python scripts/simple_clickhouse_test.py
```

## 故障排除

1. **检查进程状态**:
   ```bash
   ps aux | grep clickhouse-server
   ```

2. **查看日志**:
   ```bash
   tail -f clickhouse.log
   ```

3. **测试连接**:
   ```bash
   curl 'http://127.0.0.1:8123/?query=SELECT%201'
   ```

4. **手动启动**（如果脚本失败）:
   ```bash
   cd /Users/hacker/clickhouse_data
   clickhouse-server
   ```
