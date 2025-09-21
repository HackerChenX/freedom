
# WebSocket接口文档

## WebSocket行情数据(公有流)
### K线数据

 请求正文

参数 |类型  |  是否必填|  描述   |  说明 
--- | --- | --- | ---| ---
type| String |true  |类型|    sub
topic| String   |true|主题    |candle_stick
period  | String|true|K线周期| 1m, 5m, 15m, 30m, 60m
instrumentId| String|true|  币对| ETH/USDT, BTC/ETH ...

 响应正文

 参数  |类型 |  描述   |  说明   
 --- | --- | --- | ---
 type| String   | 类型|   sub-resp
 topic| String  | 主题|   candle_stick
 errorCode  | String| 错误码|  0000 ...
 errorMessage| String   | 错误信息|     success ...
 high| String|  高|  
 open| String   | 开|    
 low| String    | 低|    
 close| String  | 收|    
 instrumentId| String   | 币对|   ETH/USDT, BTC/ETH ...
 volume | String| 成交量 | 
timestamp   | String|时间戳|   1492420473027


``` java
请求样例：
{
    "type":"sub",
    "channel":{
        "topic":"candle_stick",
        "period":"1m",
        "instrumentId":"ETH/USDT"
    }
}

响应样例：
{
    "type":"sub-resp",
    "topic":"candle_stick",
    "errorCode":"0000",
    "errorMessage":"success",
    "data":{
        "close":"2",
        "high":"2",
        "low":"2",
        "open":"2",
        "instrumentId":"ETH/USDT",
        "timestamp":"1492420473027",
        "volume":"0"
    }
}
```


### 行情数据

 请求正文

|  参数 |类型  |是否必填|  描述   |  说明   |
| --- | --- | --- | ---| ---
| type| String  |true| 类型   | sub
| topic| String|true    | 主题    | market_data
| instrumentId| String|true|    币对|     ETH/USDT, BTC/ETH ... 

 响应正文

 参数  |类型 |  描述   |  说明   |
 --- | --- | --- | ---
type    | String| 类型    | sub-resp
topic| String   | 主题    | market_data
errorCode| String   | 错误码   | 0000 ...
errorMessage    | String| 错误信息|     success ...
high| String|   高   | 
low | String| 低 | 
close| String|  收   | 
instrumentId| String    | 币对| ETH/USDT, BTC/ETH ...
base| String    |  源币种    | 
target  | String |    目标币种   | 
lastPrice| String   |   最新价格    | 
lastPriceValuation| String  |  最新价格估值   | 
chgRate| String |  变化比例    | 
chgVol| String  |    变化量  | 
volume| String  | 成交量| 



``` json
请求样例：
{
    "type":"sub",
    "channel":{
        "topic":"market_data",
        "instrumentId":"ETH/USDT"
    }
} 

响应样例：
{
    "type":"sub-resp",
    "topic":"market_data",
    "errorCode":"0000",
    "errorMessage":"success",
    "data":[
        {
            "close":"0",
            "high":"0",
            "low":"0",
            "base":"USDT",
            "target":"ETH",
            "instrumentId":"ETH/USDT",
            "chgRate":"0",
            "chgVol":"0",
            "lastPrice":"3",
            "lastPriceValuation":0,
            "volume":"2"
        },
        {
            "close":"0",
            "high":"0",
            "low":"0",
            "base":"USDT",
            "target":"ETH",
            "instrumentId":"ETH/USDT",
            "chgRate":"0",
            "chgVol":"0",
            "lastPrice":"3",
            "lastPriceValuation":0,
            "volume":"2"
        }
    ]
}
```


### 深度行情数据

 请求正文

  参数  |类型 |是否必填|  描述   |  说明   |
 --- | --- | --- |--- | --- 
type    | String|true |类型 | sub
topic   | String |true|主题 | depth_market_data
instrumentId| String |true| 币对   |ETH/USDT, BTC/ETH ...


 响应正文

  参数 |类型  |  描述   |  说明   |
 --- | --- | --- | --- |
type    | String |类型 |  sub-resp
topic   | String |主题 |  depth_market_data
errorCode   | String |错误码 | 0000 ...
errorMessage| String     |错误信息 |    success ...
amount  | String |  成交量      |
price   | String |  单价   |
total   | String |  总金额 |


``` json
请求样例：
{
    "type":"sub",
    "channel":{
        "topic":"depth_market_data",
        "instrumentId":"ETH/USDT"
    }
}
响应样例：
{
    "type":"sub-resp",
    "topic":"depth_market_data",
    "errorCode":"0000",
    "errorMessage":"success",
    "data":{
        "ask":[
            {
                "amount":"3",
                "price":"2",
                "total":"6"
            },
            {
                "amount":"3",
                "price":"2",
                "total":"6"
            }
        ]
    }
}
```


### 全量成交回报数据


 请求正文

  参数  |类型 |是否必填|  描述   |  说明  
 --- | --- | --- |--- | --- 
type    | String|true |类型 | sub
topic   | String |true|主题 | trade_rtn_all
instrumentId| String |false|    币对   |ETH/USDT, BTC/ETH ...传空为全量


 响应正文

  参数  |类型 |  描述   |  说明   |
 --- | --- | --- | ---
type| String    |类型 |sub-resp
topic   | String|主题|    trade_rtn_all
errorCode| String|  错误码|    0000 ...
errorMessage| String    |错误信息|  success ...
fee | String|       费用  |
instrumentId    | String    |   币对  |
direction| String       |   交易方向    | 0-买入 ， 1-卖出
tradeId     | String|   交易ID        |
volume  | String    |       成交数量    |
total   | String    |   成交额     |
price   | String    |   价格      |
tradeTimestamp  | String    |   交易时间        |


``` json
请求样例：
{
    "type":"sub",
    "channel":{
        "topic":"trade_rtn_all",
        "instrumentId":"ETH/USDT"
    }
}
响应样例：
{
    "type":"sub-resp",
    "topic":"trade_rtn_all",
    "errorCode":"0000",
    "errorMessage":"success",
    "data":[
            {
                "fee":"2",
                "instrumentId":"ETH-USDT",
                "direction":"1",
                "tradeId":"1000001",
                "volume":"2",
                "total":"2",
                "price":"2",
                "tradeTimestamp":"2020/07/01 18:08:08"
            },
            {
                "fee":"2",
                "instrumentId":"ETH-USDT",
                "direction":"1",
                "tradeId":"1000001",
                "volume":"2",
                "total":"2",
                "price":"2",
                "tradeTimestamp":"2020/07/01 18:08:08"
            }
        ]
}
```

## WebSocket订单交易数据(私有流)

### 订单数据


 请求正文

  参数  |类型 |是否必填|  描述   |  说明   
 --- | --- | --- | --- | ---
type| String|true|  类型| sub
topic   | String|true|主题    |order_rtn
instrumentId| String|false| 币对  |ETH/USDT, BTC/ETH ...传空为全量

 响应正文

  参数   | 类型 |  描述   |  说明   |
 --- | --- | --- | --- |
type     | String|类型    |sub-resp
topic    | String|主题|   order_rtn
errorCode | String| 错误码|    0000 ...
errorMessage | String   |错误信息|  success ...
apiKey  | String  | apiKey  | 
orderId  | String  | 订单号  | 不能重复
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止盈止损 、4-冰山订单 、5-隐藏订单
limtPrice  | String  | 价格  | 
stopPrice  | String  | 触发价  | 
displaySize  | String   | 展示数量| 
volume  | String | 数量  | 
orderStatus  | String | 订单状态  | 0-全部成交、1-部分成交 、2-未成交、3-撤单、4-部分成交部分撤单、5-已触发止损
orderTimestamp  | String | 操作时间戳  | 13位


``` json
请求样例：
{
    "type":"sub",
    "channel":{
        "apiKey":"xxxxxxxxxx",
        "topic":"order_rtn",
        "instrumentId":"ETH/USDT"
    }
}
响应样例：
{
    "type":"sub-resp",
    "topic":"order_rtn",
    "errorCode":"0000",
    "errorMessage":"success",
    "data":[
        {
            "apiKey":"testApiKey",
            "orderId":"000000001",
            "orderType":"1",
            "instrumentId":"BTC-ETH",
            "direction":"0",
            "limtPrice":"1",
            "volume":"2",
            "stopPrice":"0",
            "displaySize":"0",
            "orderStatus":"0",
            "orderTimestamp":"1234567890123"
        },
        {
            "apiKey":"testApiKey",
            "orderId":"000000002",
            "orderType":"1",
            "instrumentId":"BTC-ETH",
            "direction":"0",
            "limtPrice":"1",
            "volume":"2",
            "stopPrice":"0",
            "displaySize":"0",
            "orderStatus":"0",
            "orderTimestamp":"1234567890123"
        }
    ]
}
```

### 交易数据


 请求正文

  参数   | 类型|是否必填 |  描述   |  说明   
 --- | --- | --- | --- | ---
type| String|true|  类型| sub
topic   | String|true|主题    |trade_rtn
instrumentId| String|false| 币对  |ETH/USDT, BTC/ETH ...传空为全量



 响应正文

  参数   | 类型 |  描述   |  说明   |
 --- | --- | --- | --- | 
type  | String  |类型 |sub-resp
topic     | String |主题| trade_rtn
errorCode  | String |   错误码|    0000 ...
errorMessage      | String |错误信息|   success ...
apiKey  | String  | apiKey  | 
tradeId  | String  | 成交号  | 
orderId  | String  | 订单号  | 
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
price  | String  | 价格  | 
volume  | String  | 数量  | 
fee  | String   | 手续费| 
tradeTimestamp  | String | 操作时间戳  | 13位


``` json
请求样例：
{
    "type":"sub",
    "channel":{
        "apiKey":"xxxxxxxxxx",
        "topic":"trade_rtn",
        "instrumentId":"ETH/USDT"
    }
}
响应样例：
{
    "type":"sub-resp",
    "topic":"trade_rtn",
    "errorCode":"0000",
    "errorMessage":"success",
    "data":[
        {
            "apiKey":"testApiKey",
            "orderId":"000000001",
            "tradeId":"1",
            "instrumentId":"BTC-ETH",
            "direction":"0",
            "price":"1",
            "volume":"2",
            "fee":"0.05",
            "tradeTimestamp":"1234567890123"
        },
        {
            "apiKey":"testApiKey",
            "orderId":"000000001",
            "tradeId":"2",
            "instrumentId":"BTC-ETH",
            "direction":"1",
            "price":"1",
            "volume":"2",
            "fee":"0.05",
            "tradeTimestamp":"1234567890123"
        }
    ]
}
```
