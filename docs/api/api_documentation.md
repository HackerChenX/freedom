# 股票分析系统 API 文档

**版本**: 1.0.0  
**生成时间**: 2025-09-13T15:58:51.313319  
**基础URL**: http://localhost:8000  
**WebSocket URL**: ws://localhost:8000/ws  

## 📋 目录

1. [概述](#概述)
2. [认证](#认证)
3. [RESTful API](#restful-api)
4. [WebSocket API](#websocket-api)
5. [错误处理](#错误处理)
6. [示例代码](#示例代码)
7. [集成指南](#集成指南)

## 📖 概述

完整的股票分析系统API接口文档，包括RESTful API和WebSocket API

### 主要功能
- 📊 股票数据查询API
- 📈 技术指标计算API
- 🎯 策略分析API
- ⚠️ 风险监控API
- 📡 实时监控API
- 🔌 WebSocket实时推送

### 技术特性
- RESTful API设计
- WebSocket实时通信
- 自动API文档生成
- 完整的错误处理
- 性能监控和日志

## 🔐 认证

当前版本暂不需要认证，所有API端点都可以直接访问。

## 🌐 RESTful API

### 系统信息

#### GET /health
健康检查端点

**响应示例:**
```json
{
    "status": "healthy",
    "timestamp": "2025-09-13T15:00:00",
    "version": "1.0.0"
}
```

#### GET /info
系统信息端点

**响应示例:**
```json
{
    "name": "股票分析系统",
    "version": "1.0.0",
    "description": "完整的股票分析和监控系统",
    "features": ["技术指标", "策略分析", "风险监控", "实时推送"]
}
```

### 股票数据API

#### GET /api/v1/stocks
获取股票列表

**查询参数:**
- `page` (int): 页码，默认1
- `size` (int): 每页大小，默认20
- `industry` (str): 行业筛选，可选

**响应示例:**
```json
{
    "status": "success",
    "data": [
        {
            "code": "000001",
            "name": "平安银行",
            "industry": "银行",
            "market": "深圳"
        }
    ],
    "pagination": {
        "page": 1,
        "size": 20,
        "total": 100
    }
}
```

#### GET /api/v1/stocks/{code}/data
获取股票历史数据

**路径参数:**
- `code` (str): 股票代码

**查询参数:**
- `start_date` (str): 开始日期，格式YYYY-MM-DD
- `end_date` (str): 结束日期，格式YYYY-MM-DD
- `period` (str): 数据周期，默认'日线'

**响应示例:**
```json
{
    "status": "success",
    "data": [
        {
            "date": "2025-09-13",
            "open": 10.50,
            "high": 10.80,
            "low": 10.30,
            "close": 10.70,
            "volume": 1000000,
            "turnover_rate": 2.5
        }
    ]
}
```

### 技术指标API

#### GET /api/v1/indicators
获取可用技术指标列表

**响应示例:**
```json
{
    "status": "success",
    "data": [
        {
            "name": "MA",
            "description": "移动平均线",
            "category": "趋势指标",
            "parameters": ["period"]
        }
    ]
}
```

#### POST /api/v1/indicators/calculate
计算技术指标

**请求体:**
```json
{
    "stock_code": "000001",
    "indicator": "MA",
    "parameters": {"period": 20},
    "start_date": "2025-01-01",
    "end_date": "2025-09-13"
}
```

**响应示例:**
```json
{
    "status": "success",
    "data": {
        "indicator": "MA",
        "parameters": {"period": 20},
        "values": [
            {"date": "2025-09-13", "value": 10.65}
        ]
    }
}
```

### 策略分析API

#### GET /api/v1/strategies
获取可用策略列表

**响应示例:**
```json
{
    "status": "success",
    "data": [
        {
            "name": "趋势跟踪策略",
            "description": "基于移动平均线的趋势跟踪",
            "type": "trend_following"
        }
    ]
}
```

#### POST /api/v1/strategies/analyze
执行策略分析

**请求体:**
```json
{
    "strategy": "trend_following",
    "stock_code": "000001",
    "parameters": {"ma_period": 20},
    "start_date": "2025-01-01",
    "end_date": "2025-09-13"
}
```

### 风险监控API

#### POST /api/v1/risk/assess
风险评估

**请求体:**
```json
{
    "type": "comprehensive",
    "stock_codes": ["000001", "000002"],
    "portfolio_weights": [0.6, 0.4]
}
```

### 实时监控API

#### GET /api/v1/monitoring/status
获取监控状态

**响应示例:**
```json
{
    "status": "success",
    "data": {
        "system_status": "running",
        "active_monitors": 5,
        "last_update": "2025-09-13T15:00:00"
    }
}
```

## 🔌 WebSocket API

### 连接

连接到WebSocket服务器：
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### 消息格式

所有WebSocket消息都使用JSON格式：

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

### 可用主题

- `stock_prices`: 股票价格推送
- `alerts`: 预警消息推送
- `monitoring`: 监控状态推送
- `indicators`: 技术指标推送
- `risk_updates`: 风险更新推送

### 推送消息示例

#### 股票价格推送
```json
{
    "type": "stock_prices",
    "data": [
        {
            "code": "000001",
            "name": "平安银行",
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

#### 预警消息推送
```json
{
    "type": "alerts",
    "data": [
        {
            "id": "alert_20250913_150000_1234",
            "type": "technical_alert",
            "stock_code": "000001",
            "title": "技术指标信号",
            "message": "MACD金叉信号出现，建议关注",
            "level": "WARNING",
            "timestamp": "2025-09-13T15:00:00"
        }
    ],
    "timestamp": "2025-09-13T15:00:00"
}
```

## ❌ 错误处理

### HTTP状态码

- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

### 错误响应格式

```json
{
    "status": "error",
    "error": {
        "code": "INVALID_PARAMETER",
        "message": "股票代码格式不正确",
        "details": "股票代码必须是6位数字"
    },
    "timestamp": "2025-09-13T15:00:00"
}
```

## 💻 示例代码

### Python示例

```python
import requests
import websocket
import json

# RESTful API调用
response = requests.get('http://localhost:8000/api/v1/stocks')
stocks = response.json()

# WebSocket连接
def on_message(ws, message):
    data = json.loads(message)
    print(f"收到消息: {data['type']}")

ws = websocket.WebSocketApp('ws://localhost:8000/ws',
                          on_message=on_message)
ws.run_forever()
```

### JavaScript示例

```javascript
// RESTful API调用
fetch('http://localhost:8000/api/v1/stocks')
    .then(response => response.json())
    .then(data => console.log(data));

// WebSocket连接
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('收到消息:', data.type);
};

// 订阅股票价格
ws.send(JSON.stringify({
    type: 'subscribe',
    topic: 'stock_prices'
}));
```

## 🔧 集成指南

### 快速开始

1. **启动API服务器**
   ```bash
   cd /path/to/project
   python api/main.py
   ```

2. **测试API连接**
   ```bash
   curl http://localhost:8000/health
   ```

3. **测试WebSocket连接**
   使用提供的测试工具：
   ```bash
   python tools/websocket_client_test.py
   ```

### 最佳实践

1. **错误处理**: 始终检查API响应的status字段
2. **重试机制**: 对于网络错误实现指数退避重试
3. **WebSocket重连**: 实现自动重连机制
4. **数据缓存**: 合理缓存API响应数据
5. **限流处理**: 注意API调用频率限制

### 性能优化

1. **批量请求**: 尽可能使用批量API减少请求次数
2. **WebSocket优先**: 对于实时数据优先使用WebSocket
3. **数据压缩**: 大量数据传输时启用压缩
4. **连接复用**: 复用HTTP连接和WebSocket连接

---

**更新时间**: 2025-09-13 15:58:51  
**文档版本**: 1.0.0
