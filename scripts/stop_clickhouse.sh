#!/bin/bash

# ClickHouse 停止脚本

echo "正在停止 ClickHouse 服务器..."

# 检查是否有 ClickHouse 进程在运行
if pgrep -f "clickhouse-server" > /dev/null; then
    echo "找到 ClickHouse 进程，正在停止..."
    pkill clickhouse-server
    
    # 等待进程完全停止
    sleep 3
    
    # 再次检查
    if pgrep -f "clickhouse-server" > /dev/null; then
        echo "进程仍在运行，强制停止..."
        pkill -9 clickhouse-server
        sleep 1
    fi
    
    echo "✅ ClickHouse 服务已停止"
else
    echo "ℹ️  没有找到运行中的 ClickHouse 进程"
fi

# 显示当前进程状态
echo ""
echo "当前 ClickHouse 进程状态:"
ps aux | grep clickhouse | grep -v grep || echo "  无运行中的 ClickHouse 进程"
