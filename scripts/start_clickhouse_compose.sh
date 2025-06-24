#!/bin/bash

# 使用 docker-compose 启动 ClickHouse

echo "使用 docker-compose 启动 ClickHouse..."

# 检查 docker-compose.yml 是否存在
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml 文件不存在"
    exit 1
fi

# 停止现有服务
echo "停止现有服务..."
docker-compose down

# 创建必要的目录
echo "创建配置目录..."
mkdir -p config/clickhouse

# 启动服务
echo "启动 ClickHouse 服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 15

# 检查服务状态
if docker-compose ps | grep -q "Up"; then
    echo "✅ ClickHouse 服务启动成功！"
    
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
        echo "管理命令："
        echo "- 查看日志: docker-compose logs clickhouse"
        echo "- 进入容器: docker-compose exec clickhouse clickhouse-client --password=123456"
        echo "- 停止服务: docker-compose down"
    else
        echo "❌ ClickHouse 服务连接失败"
        echo "查看日志："
        docker-compose logs clickhouse
    fi
else
    echo "❌ ClickHouse 服务启动失败"
    echo "查看状态："
    docker-compose ps
    echo "查看日志："
    docker-compose logs clickhouse
fi
