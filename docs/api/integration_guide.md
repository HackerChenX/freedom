# 系统集成指南

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd stock-analysis-system

# 安装依赖
pip install -r requirements.txt

# 配置数据库
# 编辑 config/database_config.yaml
```

### 2. 启动服务

```bash
# 启动API服务器
python api/main.py

# 服务将在以下地址启动:
# HTTP API: http://localhost:8000
# WebSocket: ws://localhost:8000/ws
# API文档: http://localhost:8000/docs
```

### 3. 验证安装

```bash
# 检查API健康状态
curl http://localhost:8000/health

# 测试WebSocket连接
python tools/websocket_client_test.py
```

## 集成示例

### RESTful API集成

#### Python示例
```python
import requests

class StockAnalysisClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_stocks(self, page=1, size=20):
        response = requests.get(f"{self.base_url}/api/v1/stocks", 
                              params={"page": page, "size": size})
        return response.json()
    
    def calculate_indicator(self, stock_code, indicator, parameters):
        data = {
            "stock_code": stock_code,
            "indicator": indicator,
            "parameters": parameters
        }
        response = requests.post(f"{self.base_url}/api/v1/indicators/calculate", 
                               json=data)
        return response.json()

# 使用示例
client = StockAnalysisClient()
stocks = client.get_stocks()
ma_result = client.calculate_indicator("000001", "MA", {"period": 20})
```

### WebSocket集成

#### JavaScript示例
```javascript
class StockWebSocketClient {
    constructor(url = 'ws://localhost:8000/ws') {
        this.url = url;
        this.ws = null;
        this.subscriptions = new Set();
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('WebSocket连接已建立');
            this.onConnected();
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket连接已关闭');
            setTimeout(() => this.connect(), 5000); // 自动重连
        };
    }
    
    subscribe(topic) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                topic: topic
            }));
            this.subscriptions.add(topic);
        }
    }
    
    handleMessage(data) {
        switch(data.type) {
            case 'stock_prices':
                this.onStockPrices(data.data);
                break;
            case 'alerts':
                this.onAlerts(data.data);
                break;
            default:
                console.log('未知消息类型:', data.type);
        }
    }
    
    onStockPrices(prices) {
        // 处理股票价格数据
        prices.forEach(price => {
            console.log(`${price.name}: ${price.price} (${price.change_percent}%)`);
        });
    }
    
    onAlerts(alerts) {
        // 处理预警消息
        alerts.forEach(alert => {
            console.log(`预警: ${alert.title} - ${alert.message}`);
        });
    }
}

// 使用示例
const client = new StockWebSocketClient();
client.connect();
client.subscribe('stock_prices');
client.subscribe('alerts');
```

## 部署指南

### Docker部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api/main.py"]
```

### 生产环境配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=clickhouse://localhost:9000
    depends_on:
      - clickhouse
  
  clickhouse:
    image: clickhouse/clickhouse-server
    ports:
      - "9000:9000"
      - "8123:8123"
```

## 监控和日志

### 健康检查

```bash
# API健康检查
curl http://localhost:8000/health

# WebSocket统计
curl http://localhost:8000/ws/stats
```

### 日志配置

```python
import logging

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 故障排除

### 常见问题

1. **连接被拒绝**
   - 检查服务是否启动
   - 验证端口是否正确
   - 检查防火墙设置

2. **WebSocket连接失败**
   - 确认WebSocket URL正确
   - 检查网络连接
   - 验证服务器WebSocket支持

3. **API响应错误**
   - 检查请求参数格式
   - 验证数据库连接
   - 查看服务器日志

### 性能优化

1. **API调用优化**
   - 使用批量请求
   - 实现请求缓存
   - 控制并发数量

2. **WebSocket优化**
   - 实现连接池
   - 优化消息处理
   - 控制订阅数量

## 技术支持

如有问题，请查看：
1. API文档: http://localhost:8000/docs
2. 项目文档: docs/
3. 示例代码: examples/
