# WebSocket API 文档

## 连接信息

**WebSocket URL**: ws://localhost:8000/ws

## 连接示例

### JavaScript
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    console.log('WebSocket连接已建立');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('收到消息:', data);
};

ws.onclose = function(event) {
    console.log('WebSocket连接已关闭');
};
```

### Python
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"收到消息: {data}")

def on_open(ws):
    print("WebSocket连接已建立")
    # 订阅股票价格
    ws.send(json.dumps({
        "type": "subscribe",
        "topic": "stock_prices"
    }))

ws = websocket.WebSocketApp('ws://localhost:8000/ws',
                          on_message=on_message,
                          on_open=on_open)
ws.run_forever()
```

## 消息协议

### 客户端发送消息

#### 订阅主题
```json
{
    "type": "subscribe",
    "topic": "stock_prices"
}
```

#### 取消订阅
```json
{
    "type": "unsubscribe",
    "topic": "stock_prices"
}
```

#### 心跳
```json
{
    "type": "ping"
}
```

#### 获取统计信息
```json
{
    "type": "get_stats"
}
```

### 服务器推送消息

#### 欢迎消息
```json
{
    "type": "welcome",
    "message": "欢迎连接到股票分析系统WebSocket服务器",
    "server_time": "2025-09-13T15:00:00",
    "available_topics": [
        "stock_prices",
        "alerts", 
        "monitoring",
        "indicators",
        "risk_updates"
    ]
}
```

#### 股票价格推送
```json
{
    "type": "stock_prices",
    "data": [
        {
            "code": "000001",
            "name": "股票000001",
            "price": 10.70,
            "change": 0.20,
            "change_percent": 1.90,
            "volume": 1000000,
            "timestamp": "2025-09-13T15:00:00"
        }
    ],
    "timestamp": "2025-09-13T15:00:00"
}
```

## 可用主题

- **stock_prices**: 股票价格实时推送
- **alerts**: 预警消息推送
- **monitoring**: 系统监控状态推送
- **indicators**: 技术指标推送
- **risk_updates**: 风险更新推送

## 错误处理

```json
{
    "type": "error",
    "message": "错误描述",
    "timestamp": "2025-09-13T15:00:00"
}
```

## 最佳实践

1. **心跳保活**: 定期发送ping消息保持连接
2. **重连机制**: 实现自动重连逻辑
3. **消息缓冲**: 处理网络延迟和消息堆积
4. **错误处理**: 妥善处理连接错误和消息错误
