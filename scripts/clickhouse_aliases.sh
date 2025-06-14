#!/bin/bash

# ClickHouse 便捷别名设置
# 在您的 ~/.bashrc 或 ~/.zshrc 中添加以下行来使用这些别名：
# source /Users/hacker/PycharmProjects/freedom/scripts/clickhouse_aliases.sh

PROJECT_DIR="/Users/hacker/PycharmProjects/freedom"

# ClickHouse 管理别名
alias ch-start="$PROJECT_DIR/scripts/start_clickhouse.sh"
alias ch-stop="$PROJECT_DIR/scripts/stop_clickhouse.sh"
alias ch-status="ps aux | grep clickhouse-server | grep -v grep"
alias ch-log="tail -f $PROJECT_DIR/clickhouse.log"
alias ch-client="clickhouse-client --host 127.0.0.1 --port 9000"

# 快速查询别名
alias ch-dbs="curl -s 'http://127.0.0.1:8123/?query=SHOW%20DATABASES'"
alias ch-tables="curl -s 'http://127.0.0.1:8123/?query=SHOW%20TABLES%20FROM%20stock'"

echo "ClickHouse 别名已加载："
echo "  ch-start   - 启动 ClickHouse"
echo "  ch-stop    - 停止 ClickHouse"
echo "  ch-status  - 查看 ClickHouse 进程状态"
echo "  ch-log     - 查看 ClickHouse 日志"
echo "  ch-client  - 启动 ClickHouse 客户端"
echo "  ch-dbs     - 显示所有数据库"
echo "  ch-tables  - 显示 stock 数据库中的表"
