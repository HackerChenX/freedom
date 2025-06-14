#!/bin/bash

# ClickHouse 启动脚本
# 使用指定的数据目录: /Users/hacker/clickhouse_data

CLICKHOUSE_DATA_DIR="/Users/hacker/clickhouse_data"
PROJECT_DIR="/Users/hacker/PycharmProjects/freedom"
LOG_FILE="$PROJECT_DIR/clickhouse.log"

echo "ClickHouse 启动脚本"
echo "数据目录: $CLICKHOUSE_DATA_DIR"
echo "日志文件: $LOG_FILE"

# 检查数据目录是否存在
if [ ! -d "$CLICKHOUSE_DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $CLICKHOUSE_DATA_DIR"
    exit 1
fi

# 检查是否已有 ClickHouse 进程在运行
if pgrep -f "clickhouse-server" > /dev/null; then
    echo "检测到 ClickHouse 进程正在运行，正在停止..."
    pkill clickhouse-server
    sleep 2
fi

# 配置文件路径
CONFIG_FILE="$PROJECT_DIR/config/clickhouse-config.xml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "警告: 配置文件不存在: $CONFIG_FILE"
    echo "使用默认配置启动..."
    # 切换到数据目录并启动 ClickHouse
    echo "正在启动 ClickHouse 服务器..."
    cd "$CLICKHOUSE_DATA_DIR"
    nohup clickhouse-server > "$LOG_FILE" 2>&1 &
else
    echo "使用配置文件: $CONFIG_FILE"
    echo "正在启动 ClickHouse 服务器..."
    nohup clickhouse-server --config-file="$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
fi
CLICKHOUSE_PID=$!

echo "ClickHouse 已启动，进程 ID: $CLICKHOUSE_PID"
echo "日志文件: $LOG_FILE"

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 检查服务是否成功启动
if curl -s 'http://127.0.0.1:8123/?query=SELECT%201' > /dev/null 2>&1; then
    echo "✅ ClickHouse 服务启动成功！"
    echo ""
    echo "连接信息:"
    echo "- HTTP 接口: http://127.0.0.1:8123"
    echo "- 原生协议: 127.0.0.1:9000"
    echo "- MySQL 兼容: 127.0.0.1:9004"
    echo ""
    echo "数据库列表:"
    curl -s 'http://127.0.0.1:8123/?query=SHOW%20DATABASES' | sed 's/^/  - /'
else
    echo "❌ ClickHouse 服务启动失败"
    echo "请检查日志文件: $LOG_FILE"
    exit 1
fi
