#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
API文档生成器
自动生成完整的API文档，包括RESTful API和WebSocket API
"""

import os
import sys
import json
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.exception_handler import exception_handler
from utils.performance_monitor import performance_monitor
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)

class APIDocumentationGenerator:
    """API文档生成器"""
    
    def __init__(self, output_dir: str = "docs/api"):
        """初始化API文档生成器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_info = {
            'title': '股票分析系统 API 文档',
            'version': '1.0.0',
            'description': '完整的股票分析系统API接口文档，包括RESTful API和WebSocket API',
            'generated_at': datetime.now().isoformat(),
            'base_url': 'http://localhost:8000',
            'websocket_url': 'ws://localhost:8000/ws'
        }
        logger.info("API文档生成器初始化完成")
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=5.0)
    def generate_complete_documentation(self):
        """生成完整的API文档"""
        logger.info("开始生成完整的API文档")
        
        # 生成各种格式的文档
        self._generate_markdown_documentation()
        self._generate_openapi_specification()
        self._generate_postman_collection()
        self._generate_websocket_documentation()
        self._generate_integration_guide()
        
        logger.info("完整的API文档生成完成")
    
    @exception_handler(reraise=True)
    def _generate_markdown_documentation(self):
        """生成Markdown格式的API文档"""
        logger.info("生成Markdown格式的API文档")
        
        markdown_content = self._create_markdown_content()
        
        output_file = self.output_dir / "api_documentation.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown文档已生成: {output_file}")
    
    def _create_markdown_content(self) -> str:
        """创建Markdown文档内容"""
        return f"""# {self.api_info['title']}

**版本**: {self.api_info['version']}  
**生成时间**: {self.api_info['generated_at']}  
**基础URL**: {self.api_info['base_url']}  
**WebSocket URL**: {self.api_info['websocket_url']}  

## 📋 目录

1. [概述](#概述)
2. [认证](#认证)
3. [RESTful API](#restful-api)
4. [WebSocket API](#websocket-api)
5. [错误处理](#错误处理)
6. [示例代码](#示例代码)
7. [集成指南](#集成指南)

## 📖 概述

{self.api_info['description']}

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
{{
    "status": "healthy",
    "timestamp": "2025-09-13T15:00:00",
    "version": "1.0.0"
}}
```

#### GET /info
系统信息端点

**响应示例:**
```json
{{
    "name": "股票分析系统",
    "version": "1.0.0",
    "description": "完整的股票分析和监控系统",
    "features": ["技术指标", "策略分析", "风险监控", "实时推送"]
}}
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
{{
    "status": "success",
    "data": [
        {{
            "code": "000001",
            "name": "平安银行",
            "industry": "银行",
            "market": "深圳"
        }}
    ],
    "pagination": {{
        "page": 1,
        "size": 20,
        "total": 100
    }}
}}
```

#### GET /api/v1/stocks/{{code}}/data
获取股票历史数据

**路径参数:**
- `code` (str): 股票代码

**查询参数:**
- `start_date` (str): 开始日期，格式YYYY-MM-DD
- `end_date` (str): 结束日期，格式YYYY-MM-DD
- `period` (str): 数据周期，默认'日线'

**响应示例:**
```json
{{
    "status": "success",
    "data": [
        {{
            "date": "2025-09-13",
            "open": 10.50,
            "high": 10.80,
            "low": 10.30,
            "close": 10.70,
            "volume": 1000000,
            "turnover_rate": 2.5
        }}
    ]
}}
```

### 技术指标API

#### GET /api/v1/indicators
获取可用技术指标列表

**响应示例:**
```json
{{
    "status": "success",
    "data": [
        {{
            "name": "MA",
            "description": "移动平均线",
            "category": "趋势指标",
            "parameters": ["period"]
        }}
    ]
}}
```

#### POST /api/v1/indicators/calculate
计算技术指标

**请求体:**
```json
{{
    "stock_code": "000001",
    "indicator": "MA",
    "parameters": {{"period": 20}},
    "start_date": "2025-01-01",
    "end_date": "2025-09-13"
}}
```

**响应示例:**
```json
{{
    "status": "success",
    "data": {{
        "indicator": "MA",
        "parameters": {{"period": 20}},
        "values": [
            {{"date": "2025-09-13", "value": 10.65}}
        ]
    }}
}}
```

### 策略分析API

#### GET /api/v1/strategies
获取可用策略列表

**响应示例:**
```json
{{
    "status": "success",
    "data": [
        {{
            "name": "趋势跟踪策略",
            "description": "基于移动平均线的趋势跟踪",
            "type": "trend_following"
        }}
    ]
}}
```

#### POST /api/v1/strategies/analyze
执行策略分析

**请求体:**
```json
{{
    "strategy": "trend_following",
    "stock_code": "000001",
    "parameters": {{"ma_period": 20}},
    "start_date": "2025-01-01",
    "end_date": "2025-09-13"
}}
```

### 风险监控API

#### POST /api/v1/risk/assess
风险评估

**请求体:**
```json
{{
    "type": "comprehensive",
    "stock_codes": ["000001", "000002"],
    "portfolio_weights": [0.6, 0.4]
}}
```

### 实时监控API

#### GET /api/v1/monitoring/status
获取监控状态

**响应示例:**
```json
{{
    "status": "success",
    "data": {{
        "system_status": "running",
        "active_monitors": 5,
        "last_update": "2025-09-13T15:00:00"
    }}
}}
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
{{
    "type": "subscribe",
    "topic": "stock_prices"
}}
```

#### 取消订阅
```json
{{
    "type": "unsubscribe",
    "topic": "stock_prices"
}}
```

#### 心跳
```json
{{
    "type": "ping"
}}
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
{{
    "type": "stock_prices",
    "data": [
        {{
            "code": "000001",
            "name": "平安银行",
            "price": 10.70,
            "change": 0.20,
            "change_percent": 1.90,
            "volume": 1000000,
            "timestamp": "2025-09-13T15:00:00"
        }}
    ],
    "timestamp": "2025-09-13T15:00:00"
}}
```

#### 预警消息推送
```json
{{
    "type": "alerts",
    "data": [
        {{
            "id": "alert_20250913_150000_1234",
            "type": "technical_alert",
            "stock_code": "000001",
            "title": "技术指标信号",
            "message": "MACD金叉信号出现，建议关注",
            "level": "WARNING",
            "timestamp": "2025-09-13T15:00:00"
        }}
    ],
    "timestamp": "2025-09-13T15:00:00"
}}
```

## ❌ 错误处理

### HTTP状态码

- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

### 错误响应格式

```json
{{
    "status": "error",
    "error": {{
        "code": "INVALID_PARAMETER",
        "message": "股票代码格式不正确",
        "details": "股票代码必须是6位数字"
    }},
    "timestamp": "2025-09-13T15:00:00"
}}
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
    print(f"收到消息: {{data['type']}}")

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

ws.onmessage = function(event) {{
    const data = JSON.parse(event.data);
    console.log('收到消息:', data.type);
}};

// 订阅股票价格
ws.send(JSON.stringify({{
    type: 'subscribe',
    topic: 'stock_prices'
}}));
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

**更新时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**文档版本**: {self.api_info['version']}
"""
    
    @exception_handler(reraise=True)
    def _generate_openapi_specification(self):
        """生成OpenAPI规范文档"""
        logger.info("生成OpenAPI规范文档")
        
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.api_info['title'],
                "version": self.api_info['version'],
                "description": self.api_info['description']
            },
            "servers": [
                {
                    "url": self.api_info['base_url'],
                    "description": "开发服务器"
                }
            ],
            "paths": self._generate_openapi_paths(),
            "components": {
                "schemas": self._generate_openapi_schemas()
            }
        }
        
        output_file = self.output_dir / "openapi.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(openapi_spec, f, ensure_ascii=False, indent=2)
        
        logger.info(f"OpenAPI规范已生成: {output_file}")
    
    def _generate_openapi_paths(self) -> Dict[str, Any]:
        """生成OpenAPI路径定义"""
        return {
            "/health": {
                "get": {
                    "summary": "健康检查",
                    "responses": {
                        "200": {
                            "description": "系统健康",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/stocks": {
                "get": {
                    "summary": "获取股票列表",
                    "parameters": [
                        {
                            "name": "page",
                            "in": "query",
                            "schema": {"type": "integer", "default": 1}
                        },
                        {
                            "name": "size",
                            "in": "query",
                            "schema": {"type": "integer", "default": 20}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "股票列表",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/StockListResponse"}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def _generate_openapi_schemas(self) -> Dict[str, Any]:
        """生成OpenAPI数据模型"""
        return {
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "version": {"type": "string"}
                }
            },
            "StockListResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "data": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Stock"}
                    }
                }
            },
            "Stock": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "market": {"type": "string"}
                }
            }
        }
    
    @exception_handler(reraise=True)
    def _generate_postman_collection(self):
        """生成Postman集合"""
        logger.info("生成Postman集合")
        
        collection = {
            "info": {
                "name": self.api_info['title'],
                "description": self.api_info['description'],
                "version": self.api_info['version']
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": self.api_info['base_url']
                }
            ],
            "item": self._generate_postman_requests()
        }
        
        output_file = self.output_dir / "postman_collection.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(collection, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Postman集合已生成: {output_file}")
    
    def _generate_postman_requests(self) -> List[Dict[str, Any]]:
        """生成Postman请求"""
        return [
            {
                "name": "健康检查",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/health",
                        "host": ["{{base_url}}"],
                        "path": ["health"]
                    }
                }
            },
            {
                "name": "获取股票列表",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{base_url}}/api/v1/stocks?page=1&size=20",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "stocks"],
                        "query": [
                            {"key": "page", "value": "1"},
                            {"key": "size", "value": "20"}
                        ]
                    }
                }
            }
        ]
    
    @exception_handler(reraise=True)
    def _generate_websocket_documentation(self):
        """生成WebSocket文档"""
        logger.info("生成WebSocket文档")
        
        websocket_doc = f"""# WebSocket API 文档

## 连接信息

**WebSocket URL**: {self.api_info['websocket_url']}

## 连接示例

### JavaScript
```javascript
const ws = new WebSocket('{self.api_info['websocket_url']}');

ws.onopen = function(event) {{
    console.log('WebSocket连接已建立');
}};

ws.onmessage = function(event) {{
    const data = JSON.parse(event.data);
    console.log('收到消息:', data);
}};

ws.onclose = function(event) {{
    console.log('WebSocket连接已关闭');
}};
```

### Python
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"收到消息: {{data}}")

def on_open(ws):
    print("WebSocket连接已建立")
    # 订阅股票价格
    ws.send(json.dumps({{
        "type": "subscribe",
        "topic": "stock_prices"
    }}))

ws = websocket.WebSocketApp('{self.api_info['websocket_url']}',
                          on_message=on_message,
                          on_open=on_open)
ws.run_forever()
```

## 消息协议

### 客户端发送消息

#### 订阅主题
```json
{{
    "type": "subscribe",
    "topic": "stock_prices"
}}
```

#### 取消订阅
```json
{{
    "type": "unsubscribe",
    "topic": "stock_prices"
}}
```

#### 心跳
```json
{{
    "type": "ping"
}}
```

#### 获取统计信息
```json
{{
    "type": "get_stats"
}}
```

### 服务器推送消息

#### 欢迎消息
```json
{{
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
}}
```

#### 股票价格推送
```json
{{
    "type": "stock_prices",
    "data": [
        {{
            "code": "000001",
            "name": "股票000001",
            "price": 10.70,
            "change": 0.20,
            "change_percent": 1.90,
            "volume": 1000000,
            "timestamp": "2025-09-13T15:00:00"
        }}
    ],
    "timestamp": "2025-09-13T15:00:00"
}}
```

## 可用主题

- **stock_prices**: 股票价格实时推送
- **alerts**: 预警消息推送
- **monitoring**: 系统监控状态推送
- **indicators**: 技术指标推送
- **risk_updates**: 风险更新推送

## 错误处理

```json
{{
    "type": "error",
    "message": "错误描述",
    "timestamp": "2025-09-13T15:00:00"
}}
```

## 最佳实践

1. **心跳保活**: 定期发送ping消息保持连接
2. **重连机制**: 实现自动重连逻辑
3. **消息缓冲**: 处理网络延迟和消息堆积
4. **错误处理**: 妥善处理连接错误和消息错误
"""
        
        output_file = self.output_dir / "websocket_api.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(websocket_doc)
        
        logger.info(f"WebSocket文档已生成: {output_file}")
    
    @exception_handler(reraise=True)
    def _generate_integration_guide(self):
        """生成集成指南"""
        logger.info("生成集成指南")
        
        integration_guide = """# 系统集成指南

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
"""
        
        output_file = self.output_dir / "integration_guide.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(integration_guide)
        
        logger.info(f"集成指南已生成: {output_file}")

def main():
    """主函数"""
    print("🚀 开始生成API文档...")
    
    generator = APIDocumentationGenerator()
    generator.generate_complete_documentation()
    
    print("✅ API文档生成完成!")
    print(f"📁 文档位置: {generator.output_dir}")
    print("📋 生成的文档:")
    print("  - api_documentation.md (完整API文档)")
    print("  - openapi.json (OpenAPI规范)")
    print("  - postman_collection.json (Postman集合)")
    print("  - websocket_api.md (WebSocket文档)")
    print("  - integration_guide.md (集成指南)")

if __name__ == "__main__":
    main()
