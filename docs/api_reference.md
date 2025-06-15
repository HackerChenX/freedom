# API接口文档

## 📖 目录

- [概述](#概述)
- [认证](#认证)
- [买点分析API](#买点分析api)
- [技术指标API](#技术指标api)
- [性能优化API](#性能优化api)
- [批量处理API](#批量处理api)
- [系统监控API](#系统监控api)
- [错误处理](#错误处理)
- [SDK使用](#sdk使用)

---

## 概述

股票分析系统提供RESTful API接口，支持买点分析、技术指标计算、性能优化等功能。所有API都经过高性能优化，支持0.05秒/股的极致处理速度。

### 🚀 API特性

- **超高性能**: 0.05秒响应时间
- **高并发**: 支持72,000请求/小时
- **智能缓存**: 50%缓存命中率
- **实时监控**: 完整的性能指标

### 🔗 基础URL

```
生产环境: https://api.stockanalysis.com/v1
测试环境: https://test-api.stockanalysis.com/v1
本地开发: http://localhost:8000/v1
```

---

## 认证

### API密钥认证

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### 获取API密钥

```bash
curl -X POST https://api.stockanalysis.com/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**响应:**

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## 买点分析API

### 单个买点分析

分析单个股票的买点，返回详细的技术指标和买点评分。

```http
POST /api/v1/analyze/buypoint
```

**请求参数:**

```json
{
  "stock_code": "000001",
  "buypoint_date": "2025-01-01",
  "enable_cache": true,
  "enable_vectorization": true,
  "indicators": ["RSI", "MACD", "KDJ"],  // 可选，指定计算的指标
  "optimization_level": "high"           // low, medium, high
}
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "stock_code": "000001",
    "buypoint_date": "2025-01-01",
    "analysis_time": 0.0523,
    "indicator_count": 86,
    "indicators": {
      "RSI": {
        "value": 65.23,
        "signal": "中性",
        "trend": "上升",
        "strength": 0.75
      },
      "MACD": {
        "macd": 0.1234,
        "signal": 0.0987,
        "histogram": 0.0247,
        "trend": "金叉",
        "strength": 0.82
      },
      "KDJ": {
        "K": 78.45,
        "D": 72.31,
        "J": 90.73,
        "signal": "超买",
        "strength": 0.68
      }
    },
    "buypoint_analysis": {
      "score": 85.6,
      "grade": "A",
      "recommendation": "强烈买入",
      "confidence": 0.89,
      "risk_level": "中等"
    },
    "zxm_analysis": {
      "trend_strength": 78.5,
      "elasticity_score": 82.3,
      "market_breadth": 75.8,
      "final_score": 85.6
    }
  },
  "performance": {
    "cache_hit": true,
    "vectorization_used": true,
    "processing_time_ms": 52.3,
    "optimization_level": "high"
  },
  "metadata": {
    "api_version": "v1",
    "timestamp": "2025-06-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### 历史买点查询

查询股票的历史买点记录。

```http
GET /api/v1/analyze/buypoint/history/{stock_code}
```

**查询参数:**

```
?start_date=2024-01-01
&end_date=2024-12-31
&min_score=80
&limit=100
&offset=0
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "stock_code": "000001",
    "total_count": 25,
    "buypoints": [
      {
        "date": "2024-03-15",
        "score": 88.5,
        "grade": "A+",
        "recommendation": "强烈买入"
      },
      {
        "date": "2024-06-20",
        "score": 82.3,
        "grade": "A",
        "recommendation": "买入"
      }
    ]
  },
  "pagination": {
    "limit": 100,
    "offset": 0,
    "total": 25,
    "has_more": false
  }
}
```

---

## 技术指标API

### 单个指标计算

计算指定股票的单个技术指标。

```http
GET /api/v1/indicators/{stock_code}/{indicator_name}
```

**路径参数:**
- `stock_code`: 股票代码
- `indicator_name`: 指标名称 (RSI, MACD, KDJ等)

**查询参数:**

```
?date=2025-01-01
&period=14          // 指标周期
&enable_cache=true
&enable_vectorization=true
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "stock_code": "000001",
    "indicator_name": "RSI",
    "date": "2025-01-01",
    "result": {
      "value": 65.23,
      "signal": "中性",
      "trend": "上升",
      "strength": 0.75,
      "historical_data": [
        {"date": "2024-12-30", "value": 63.45},
        {"date": "2024-12-31", "value": 64.12},
        {"date": "2025-01-01", "value": 65.23}
      ]
    },
    "parameters": {
      "period": 14,
      "calculation_method": "vectorized"
    }
  },
  "performance": {
    "calculation_time_ms": 12.5,
    "cache_hit": false,
    "vectorization_used": true
  }
}
```

### 批量指标计算

一次性计算多个技术指标。

```http
POST /api/v1/indicators/batch
```

**请求参数:**

```json
{
  "stock_code": "000001",
  "date": "2025-01-01",
  "indicators": [
    {
      "name": "RSI",
      "parameters": {"period": 14}
    },
    {
      "name": "MACD",
      "parameters": {"fast": 12, "slow": 26, "signal": 9}
    },
    {
      "name": "KDJ",
      "parameters": {"period": 9}
    }
  ],
  "enable_optimizations": true
}
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "stock_code": "000001",
    "date": "2025-01-01",
    "indicators": {
      "RSI": {
        "value": 65.23,
        "calculation_time_ms": 8.5
      },
      "MACD": {
        "macd": 0.1234,
        "signal": 0.0987,
        "histogram": 0.0247,
        "calculation_time_ms": 12.3
      },
      "KDJ": {
        "K": 78.45,
        "D": 72.31,
        "J": 90.73,
        "calculation_time_ms": 10.8
      }
    }
  },
  "performance": {
    "total_calculation_time_ms": 31.6,
    "vectorization_rate": 100,
    "cache_hit_rate": 33.3
  }
}
```

### 支持的指标列表

```http
GET /api/v1/indicators/list
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "total_indicators": 86,
    "categories": {
      "trend": {
        "count": 23,
        "indicators": ["MA", "EMA", "MACD", "TRIX", "DMI", "ADX"]
      },
      "oscillator": {
        "count": 25,
        "indicators": ["RSI", "KDJ", "CCI", "WR", "STOCHRSI"]
      },
      "volume": {
        "count": 15,
        "indicators": ["OBV", "MFI", "VR", "VOLUME_RATIO"]
      },
      "volatility": {
        "count": 10,
        "indicators": ["BOLL", "ATR", "KC"]
      },
      "zxm": {
        "count": 13,
        "indicators": ["ZXM_TREND_DETECTOR", "ZXM_BUYPOINT_DETECTOR"]
      }
    },
    "vectorized_indicators": [
      "RSI", "MACD", "KDJ", "BOLL", "MA", "EMA"
    ]
  }
}
```

---

## 性能优化API

### 缓存管理

```http
POST /api/v1/performance/cache/clear
```

**请求参数:**

```json
{
  "cache_type": "all",  // memory, disk, all
  "stock_codes": ["000001", "000002"],  // 可选，清除特定股票缓存
  "force": false
}
```

### 性能统计

```http
GET /api/v1/performance/stats
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "system_performance": {
      "average_processing_time": 0.052,
      "requests_per_hour": 72000,
      "cpu_utilization": 65.5,
      "memory_usage": 78.2
    },
    "cache_performance": {
      "hit_rate": 50.0,
      "memory_cache_size": 1000,
      "disk_cache_size": "2.5GB"
    },
    "vectorization_stats": {
      "coverage_rate": 7.6,
      "performance_improvement": 45.5,
      "vectorized_calculations": 15420
    },
    "parallel_processing": {
      "active_workers": 8,
      "queue_size": 0,
      "completed_tasks": 98765
    }
  }
}
```

---

## 批量处理API

### 批量买点分析

```http
POST /api/v1/analyze/batch
```

**请求参数:**

```json
{
  "buypoints": [
    {"stock_code": "000001", "buypoint_date": "2025-01-01"},
    {"stock_code": "000002", "buypoint_date": "2025-01-01"},
    {"stock_code": "000003", "buypoint_date": "2025-01-02"}
  ],
  "parallel_workers": 8,
  "enable_optimizations": true,
  "callback_url": "https://your-domain.com/webhook"  // 可选
}
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "batch_id": "batch_123456789",
    "total_tasks": 3,
    "estimated_completion_time": "2025-06-15T10:30:15Z",
    "results": [
      {
        "stock_code": "000001",
        "buypoint_date": "2025-01-01",
        "status": "completed",
        "score": 85.6,
        "processing_time": 0.048
      },
      {
        "stock_code": "000002", 
        "buypoint_date": "2025-01-01",
        "status": "completed",
        "score": 78.3,
        "processing_time": 0.052
      },
      {
        "stock_code": "000003",
        "buypoint_date": "2025-01-02", 
        "status": "completed",
        "score": 92.1,
        "processing_time": 0.045
      }
    ]
  },
  "performance": {
    "total_processing_time": 0.145,
    "average_time_per_stock": 0.048,
    "parallel_efficiency": 95.2,
    "cache_hit_rate": 66.7
  }
}
```

### 批量任务状态查询

```http
GET /api/v1/analyze/batch/{batch_id}
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "batch_id": "batch_123456789",
    "status": "completed",
    "progress": {
      "total": 100,
      "completed": 100,
      "failed": 0,
      "percentage": 100.0
    },
    "performance": {
      "start_time": "2025-06-15T10:30:00Z",
      "end_time": "2025-06-15T10:30:05Z",
      "total_duration": 5.2,
      "average_time_per_task": 0.052
    }
  }
}
```

---

## 系统监控API

### 系统健康检查

```http
GET /api/v1/health
```

**响应示例:**

```json
{
  "status": "healthy",
  "timestamp": "2025-06-15T10:30:00Z",
  "version": "v2.0",
  "services": {
    "database": "healthy",
    "cache": "healthy", 
    "indicators": "healthy",
    "parallel_processing": "healthy"
  },
  "performance": {
    "response_time_ms": 2.5,
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.4
  }
}
```

### 实时监控指标

```http
GET /api/v1/monitoring/metrics
```

**响应示例:**

```json
{
  "status": "success",
  "data": {
    "real_time_metrics": {
      "requests_per_second": 20.5,
      "average_response_time": 0.052,
      "error_rate": 0.001,
      "active_connections": 45
    },
    "resource_usage": {
      "cpu_cores_used": 6.5,
      "memory_gb_used": 12.8,
      "disk_io_mb_per_sec": 15.2
    },
    "business_metrics": {
      "stocks_analyzed_today": 15420,
      "cache_hit_rate": 52.3,
      "vectorization_usage": 8.1
    }
  }
}
```

---

## 错误处理

### 错误响应格式

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_STOCK_CODE",
    "message": "股票代码格式不正确",
    "details": "股票代码必须是6位数字",
    "timestamp": "2025-06-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### 常见错误代码

| 错误代码 | HTTP状态码 | 描述 |
|---------|-----------|------|
| `INVALID_STOCK_CODE` | 400 | 股票代码格式错误 |
| `INVALID_DATE_FORMAT` | 400 | 日期格式错误 |
| `STOCK_DATA_NOT_FOUND` | 404 | 股票数据不存在 |
| `INDICATOR_NOT_SUPPORTED` | 400 | 不支持的技术指标 |
| `RATE_LIMIT_EXCEEDED` | 429 | 请求频率超限 |
| `INTERNAL_SERVER_ERROR` | 500 | 服务器内部错误 |
| `SERVICE_UNAVAILABLE` | 503 | 服务暂时不可用 |

---

## SDK使用

### Python SDK

```python
from stock_analysis_sdk import StockAnalysisClient

# 初始化客户端
client = StockAnalysisClient(
    api_key='your_api_key',
    base_url='https://api.stockanalysis.com/v1'
)

# 买点分析
result = client.analyze_buypoint(
    stock_code='000001',
    buypoint_date='2025-01-01',
    enable_optimizations=True
)

# 批量分析
batch_results = client.analyze_batch([
    {'stock_code': '000001', 'buypoint_date': '2025-01-01'},
    {'stock_code': '000002', 'buypoint_date': '2025-01-01'}
])

# 技术指标查询
indicators = client.get_indicators(
    stock_code='000001',
    date='2025-01-01',
    indicators=['RSI', 'MACD', 'KDJ']
)

# 性能统计
performance = client.get_performance_stats()
print(f"平均处理时间: {performance['average_processing_time']}秒")
```

### JavaScript SDK

```javascript
import { StockAnalysisClient } from 'stock-analysis-sdk';

const client = new StockAnalysisClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.stockanalysis.com/v1'
});

// 买点分析
const result = await client.analyzeBuypoint({
  stockCode: '000001',
  buypointDate: '2025-01-01',
  enableOptimizations: true
});

// 批量分析
const batchResults = await client.analyzeBatch([
  { stockCode: '000001', buypointDate: '2025-01-01' },
  { stockCode: '000002', buypointDate: '2025-01-01' }
]);

console.log(`分析完成: ${batchResults.length}个买点`);
```

---

## 🚀 性能优化建议

### 1. 启用缓存
```json
{
  "enable_cache": true,
  "cache_ttl": 3600
}
```

### 2. 使用向量化计算
```json
{
  "enable_vectorization": true,
  "optimization_level": "high"
}
```

### 3. 批量请求
```python
# 推荐：批量处理
client.analyze_batch(buypoints_list)

# 不推荐：逐个处理
for buypoint in buypoints_list:
    client.analyze_buypoint(buypoint)
```

### 4. 异步处理
```python
import asyncio

async def analyze_multiple_stocks():
    tasks = [
        client.analyze_buypoint_async('000001', '2025-01-01'),
        client.analyze_buypoint_async('000002', '2025-01-01')
    ]
    results = await asyncio.gather(*tasks)
    return results
```

---

*API文档版本: v2.0*  
*最后更新: 2025-06-15*  
*支持的API版本: v1*
