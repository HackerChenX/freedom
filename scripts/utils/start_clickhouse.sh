#!/bin/bash
# 启动 ClickHouse 服务

nohup clickhouse-server --config-file=/usr/local/etc/clickhouse-server/config.xml > /tmp/clickhouse.log 2>&1 &
echo "ClickHouse 服务已启动，日志文件：/tmp/clickhouse.log" 