#!/bin/bash

# ClickHouse Docker 启动脚本
# 设置密码为 123456

echo "正在启动 ClickHouse Docker 容器..."

# 停止并删除现有容器（如果存在）
echo "清理现有容器..."
docker stop clickhouse-server 2>/dev/null || true
docker rm clickhouse-server 2>/dev/null || true

# 创建必要的目录
echo "创建配置目录..."
mkdir -p config/clickhouse
mkdir -p data/clickhouse
mkdir -p logs/clickhouse

# 启动 ClickHouse 容器
echo "启动新的 ClickHouse 容器..."
docker run -d \
  --name clickhouse-server \
  -p 8123:8123 \
  -p 9000:9000 \
  -p 9004:9004 \
  -e CLICKHOUSE_PASSWORD=123456 \
  -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
  -v $(pwd)/config/clickhouse/users.xml:/etc/clickhouse-server/users.d/custom-users.xml \
  -v $(pwd)/data/clickhouse:/var/lib/clickhouse \
  -v $(pwd)/logs/clickhouse:/var/log/clickhouse-server \
  --ulimit nofile=262144:262144 \
  clickhouse/clickhouse-server:latest

# 等待容器启动
echo "等待 ClickHouse 启动..."
sleep 10

# 检查容器状态
if docker ps | grep -q clickhouse-server; then
    echo "✅ ClickHouse 容器启动成功！"
    
    # 测试连接
    echo "测试数据库连接..."
    if curl -s "http://localhost:8123/?query=SELECT 1" > /dev/null; then
        echo "✅ ClickHouse 服务运行正常！"
        echo ""
        echo "连接信息："
        echo "- HTTP 接口: http://localhost:8123"
        echo "- 原生协议: localhost:9000"
        echo "- 用户名: default"
        echo "- 密码: 123456"
        echo ""
        echo "测试连接命令："
        echo "curl 'http://localhost:8123/?query=SELECT 1'"
        echo "或者："
        echo "docker exec -it clickhouse-server clickhouse-client --password=123456"
    else
        echo "❌ ClickHouse 服务启动失败"
        echo "查看日志："
        docker logs clickhouse-server
    fi
else
    echo "❌ ClickHouse 容器启动失败"
    echo "查看日志："
    docker logs clickhouse-server
fi
