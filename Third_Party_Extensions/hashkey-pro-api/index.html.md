

#  简介：
欢迎使用HashKey API！此文档是HashKey API的唯一官方文档，HashKey API提供的能力会在此持续更新，请大家及时关注。


# 接入说明：
## 接入准备
如需使用API ，请先登录网页端，完成API key的申请和权限配置，再据此文档详情进行开发和交易。
每个账户最多可创建10组Api Key，每个Api Key可对应设置读取、交易、提现三种权限。
权限说明如下：
- 读取权限：读取权限用于对数据的查询接口，例如：订单查询、成交查询等。
- 交易权限：交易权限用于下单、撤单、划转类接口。
- 提现权限：提现权限用于创建提币订单、取消提币订单操作。

创建成功后请务必记住以下信息：
- Access Key API 访问密钥
- Secret Key 签名认证加密所使用的密钥（仅申请时可见）

## SDK示例
具体示例请参考具体请求样例。


## 签名认证
### 签名说明

> 签名样例：

```java
         /**
         * 签名说明：签名所需字段apiKey、apiSecret、timestamp、version、各请求接口字段。
         *                 字段名和字段值采用key-value形式，且对Key使用ASCII码排序.
         *                 以下为测试数据demo
         */
        String key="84dd8e670471a888e3a7547e120886cb";
        String secret = "3211b4306fcb1d9670cbee5abc69dead";
        String timeOfRequest ="1478692862000";

        TreeMap<String,String> treeMap = new TreeMap<>();

        // 各feild可能是基本类型，也可能是对象
        treeMap.put("x-access-key",key); //api-key
        treeMap.put("x-access-timestamp",timeOfRequest); //当前时间戳
        treeMap.put("x-access-version","1");//版本号默认1

        treeMap.put("feild1","1");//请求字段mock1
        treeMap.put("feild2","2");//请求字段mock2
        treeMap.put("feild3","3");//请求字段mock3

        String requestText = JSONObject.toJSONString(treeMap);
        //根据Map key的ASCII 排序 {"feild1":"1","feild2":"2","feild3":"3","x-access-key":"84dd8e670471a888e3a7547e120886cb","x-access-timestamp":"1478692862000","x-access-version":"1"}

        System.out.println(createSignature(requestText,secret));
        //PiB3bk0wWgY2mN9efTmCrKCtn6PZUsrlHhCsXNLUPtU=
		
		private static String createSignature(String requestText,String secret) 
		             throws Exception {
              return Base64Utils.encodeToString(hmacSha1(requestText, secret));
        }

        private static byte[] hmacSha1(String plainData, String secret)  
		            throws Exception{
            Mac mac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKey = 
			     new SecretKeySpec(Base64Utils.decodeFromString(secret), "HmacSHA256");
            mac.init(secretKey);
            return mac.doFinal(plainData.getBytes());
         }
```

API 请求在通过 internet 传输的过程中极有可能被篡改，为了确保请求未被更改，除公共接口（基础信息，行情数据）外的私有接口均必须使用您的 API Key 做签名认证，以校验参数或参数值在传输途中是否发生了更改。
每一个API Key需要有适当的权限才能访问相应的接口，每个新创建的API Key都需要分配权限。在使用接口前，请查看每个接口的权限类型，并确认你的API Key有相应的权限。

对API端点的所有HTTP请求都需要身份验证和授权。
用户可以通过创建API key流程获得key和sercet，key和sercet用于验证和授权所有请求。
以下标头应添加到所有HTTP请求中：

Key  | Value | Description
------------- | ------------- | -----------
x-access-key  | <API_KEY> |您申请的 API Key 中的 Access Key
x-access-sign  |  \<signatureOfRequest>| 签名计算得出的值，用于确保签名有效和未被篡改
x-access-timestamp  | \<timeOfRequest> |当前时间格式为当前的时间戳（以毫秒为单位）
x-access-version  |  \<versionOfRequest>   | 签名协议的版本，默认为1


## 接口概览：
### 接口类型
HashKey Pro为用户提供两种接口，您可根据自己的使用场景来选择适合的方式。
**Rest API**
REST，即Representational State Transfer的缩写，是目前较为流行的基于HTTP的一种通信机制，每一个URL代表一种资源。
交易或资产提币等一次性操作，建议开发者使用REST API进行操作。

**WebSocket API**
WebSocket是HTML5一种新的协议（Protocol）。它实现了客户端与服务器全双工通信，通过一次简单的握手就可以建立客户端和服务器连接，服务器可以根据业务规则主动推送信息给客户端。
市场行情和买卖深度等信息，建议开发者使用WebSocket API进行获取。

## 环境信息
### 测试环境
Restful URL: https://int.hashkey3.com

WebSocket: wss://int.hashkey3.com

### 生产环境
Restful URL: https://int.hashkey3.com

WebSocket: wss://int.hashkey3.com

# 常见问题：
暂无

# Rest API
## 接入说明
### 限频规则
除已单独标注限频值的接口外 - 每个API Key 在1秒之内限制10次

比如：
资产订单类接口调用根据API Key进行限频：1秒10次

### 请求格式
所有的API请求都是restful，目前只有两种方法：GET和POST
- GET请求：业务参数都在路径参数里
- POST请求，业务参数以JSON格式发送在请求主体（body）里

### 返回格式
所有的接口都是JSON格式。在JSON最上层有四个字段：status, ch, ts 和 data。前三个字段表示请求状态和属性，实际的业务数据在data字段里。

参数名称|	类型|	描述
--- | --- | ---
errorCode	|String|	API接口返回状态
errorMessage	|String|	API接口错误描述信息
data	|Object	|接口返回数据主体

# Websocket API
## 接入说明
### 接入限制
-  同一apiKey最多允许创建10条session长连接。比如：订单成交数据订阅，同一apiKey最多允许建立10条session连接。
-  同一ip最多允许建立5条session长连接。比如：行情数据推送无需apiKey，同一ip地址最多允许建立5条连接。

### 心跳消息
当用户的Websocket客户端连接到HashKey Websocket服务器后，服务器会定期（当前设为10秒）向其发送ping消息并包含一整数值如下：

``{"type":"ping","data"="1492420473027"}``

当用户的Websocket客户端接收到此心跳消息后，应返回pong消息并包含同一整数值：

``{"type":"pong","data"="1492420473027"}``

注：当Websocket服务器连续两次发送了`ping`消息却没有收到任何一次`pong`消息返回后，服务器将主动断开与此客户端的连接。

### 请求格式
所有的请求都是JSON格式，订阅不同topic传递不同参数，请参考具体接口请求样例。

### 返回格式
所有的响应都是JSON格式，不同topic返回data各有不同，请参考具体接口响应样例。

## 鉴权

> 鉴权
>
> 请求样例：

```json
{
    "type":"auth",
    "auth":{
        "x-access-key":"xxxxxxxxx",
        "x-access-sign":"xxxxxxxxx",
        "x-access-timestamp":"150782303000"，
		"x-access-version":"2",
    }
}
```

> 响应样例：

```json        
{
    "type":"auth",
    "errorCode":"000",
    "errorMessage":"success"    
}
```

**请求正文**

|  参数  |类型 |  是否必填|注释   |  说明   |
| --- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为auth
x-access-key | String |true |访问Key  |您申请的 API Key 中的 Access Key
x-access-sign | String |true | 签名 | 签名计算得出的值，用于确保签名有效和未被篡改
x-access-timestamp| String |true  | 时间戳 |当前时间格式为当前的时间戳（以毫秒为单位）
x-access-version  | String|true |  版本号   | 签名协议的版本，默认为1

**响应正文**

|  参数  |类型 |  是否必填|注释   |  说明   |
| --- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为auth
errorCode	| String |true|错误码 |	000
errorMessage| String	|true |错误信息 |	success


## 订阅主题

> 订阅主题
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "topic":"xxxxxxxxx",
		......
    }
}
```

> 响应样例：

```json
{
    "type":"sub",
    "errorCode":"000",
    "errorMessage":"success",
	"data":{
	......
	}
}
```

**请求正文**

|  参数  |类型 |  是否必填|注释   |  说明   |
| --- | --- |--- | --- |---
type	| String|true |操作类型 |	sub
topic | String |true |订阅的主题  |
apiKey | String |false | apiKey 私有流必传  |
...|||


**响应正文**

|  参数  |类型 |  是否必填|注释   |  说明   |
| --- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为sub
errorCode	| String |true|错误码 |	000
errorMessage| String	|true |错误信息 |	success
data | Object |true|快照数据|


## 取消订阅

> 取消订阅
>
> 请求样例：

```json
{
    "type":"unsub",
    "channel":{
        "topic":"xxxxxxxxx",
		......
    }
}
```

> 响应样例：

```json
{
    "type":"unsub",
    "errorCode":"000",
    "errorMessage":"success"
}
```

**请求正文**

|  参数  |类型 |  是否必填|注释   |  说明   |
| --- | --- |--- | --- |---
type	| String|true |操作类型 |	unsub
topic | String |true |订阅的主题  |
...|||


**响应正文**

|  参数  |类型 |  是否必填|注释   |  说明   |
| --- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为unsub
errorCode	| String |true|错误码 |	000
errorMessage| String	|true |错误信息 |	success


# 错误信息
**错误码划分规范**

| 错误码 | 描述|
| --- | ---|
000~099  |公共错误码
100~199  |WebSocket错误码
200~299 | 行情错误码

**错误码对照表**

|错误类型 |  错误码   |  错误信息   |  描述   |          
| --- | --- | --- | --- | 
|  公共错误码   | 000    | success   |  成功   |
|  公共错误码   | 092    | network error   |  网络连接异常   |
|  公共错误码   | 096    | system error   |  系统异常   |



# Rest API接口文档
## 1、通用
### 1.1、查询服务器时间

> 查询服务器时间
>
> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
		"timestamp":"timestamp"
	}
}
```

**Http request:**  GET info/time

**请求正文：** 无

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
timestamp  | String  | 时间戳  | UTC时间


### 1.2、查询API发行号

> 查询API发行号
>
> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
		"version":"version"
	}
}
```

**Http request:**  GET info/version

**请求正文：** 无

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
version  | String  | 版本号  |


## 2、交易
### 2.1、创建限价单

> 创建限价单
>
> 请求样例：

```json
{
	"orderId":"000000001",
	"instrumentId":"BTC-ETH",
	"direction":"0",
	"price":"1",
	"volume":"1"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
		"apiKey":"testApiKey",
		"orderId":"000000001",
     	"orderType":"1",
    	"instrumentId":"BTC-ETH",
    	"direction":"0",
    	"limtPrice":"1",
    	"volume":"1",
    	"stopPrice":"0",
    	"displaySize":"0",
    	"timestamp":"1234567890123"
	}
}
```

**Http request:**  POST limit/order/create

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
orderId  | String   | true  | 订单号  | 不能重复
instrumentId  | String   | true  | 合约号  | 样例 ETH-BTC
direction  |  String  | true  | 方向 | 0-买 1-卖
price  | String | true  | 价格  |
volume  | String | true | 数量  |

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止盈止损 、4-冰山订单 、5-隐藏订单
limtPrice  | String  | 价格  |
stopPrice  | String  | 触发价  |
displaySize  | String   | 展示数量|
volume  | String | 数量  |
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 下单成功、订单已经提交处理
207  | 下单请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效


### 2.2、创建市价单

> 创建市价单
>
> 请求样例：

```json
{
	"orderId":"000000001",
	"instrumentId":"BTC-ETH",
	"direction":"0",
	"volume":"1"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
		"apiKey":"testApiKey",
		"orderId":"000000001",
     	"orderType":"2",
    	"instrumentId":"BTC-ETH",
    	"direction":"0",
    	"limtPrice":"1",
    	"volume":"1",
    	"stopPrice":"0",
    	"displaySize":"0",
    	"timestamp":"1234567890123"
	}
}
```

**Http request:**  POST market/order/create

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
orderId  | String   | true  | 订单号  | 不能重复
instrumentId  | String   | true  | 合约号  | 样例 ETH-BTC
direction  |  String  | true  | 方向 | 0-买 1-卖
volume  | String | true | 数量  |

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止盈止损 、4-冰山订单 、5-隐藏订单
limtPrice  | String  | 价格  |
stopPrice  | String  | 触发价  |
displaySize  | String   | 展示数量|
volume  | String | 数量  |
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 下单成功、订单已经提交处理
207  | 下单请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效


### 2.3、创建限价止损单

> 创建限价止损单
>
> 请求样例：

```json
{
	"orderId":"000000001",
	"instrumentId":"BTC-ETH",
	"direction":"0",
	"price":"1",
	"volume":"1"，
	"stopPrice":"2"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
	    "apiKey":"testApiKey",
		"orderId":"000000001",
     	"orderType":"3",
    	"instrumentId":"BTC-ETH",
    	"direction":"0",
    	"limtPrice":"1",
    	"volume":"1",
    	"stopPrice":"2",
    	"displaySize":"0",
    	"timestamp":"1234567890123"
	}
}
```

**Http request:**  POST stopL/order/create

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
orderId  | String   | true  | 订单号  | 不能重复
instrumentId  | String   | true  | 合约号  | 样例 ETH-BTC
direction  |  String  | true  | 方向 | 0-买 1-卖
price  | String | true  | 价格  |
volume  | String | true | 数量  |
stopPrice  | String | true | 触发价  |

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止损限价单 、4-冰山订单 、5-隐藏订单 、6-止损市价单
limtPrice  | String  | 价格  |
stopPrice  | String  | 触发价  |
displaySize  | String   | 展示数量|
volume  | String | 数量  |
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 下单成功、订单已经提交处理
207  | 下单请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效


### 2.4、创建市价止损单

> 创建市价止损单
>
> 请求样例：

```json
{
	"orderId":"000000001",
	"instrumentId":"BTC-ETH",
	"direction":"0",
	"volume":"1",
	"stopPrice":"2"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
	    "apiKey":"testApiKey",
		"orderId":"000000001",
     	"orderType":"6",
    	"instrumentId":"BTC-ETH",
    	"direction":"0",
    	"limtPrice":"1",
    	"volume":"1",
    	"stopPrice":"2",
    	"displaySize":"0",
    	"timestamp":"1234567890123"
	}
}
```

**Http request:**  POST market/order/create

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
orderId  | String   | true  | 订单号  | 不能重复
instrumentId  | String   | true  | 合约号  | 样例 ETH-BTC
direction  |  String  | true  | 方向 | 0-买 1-卖
volume  | String | true | 数量  |
stopPrice  | String | true | 触发价  |

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止损限价单 、4-冰山订单 、5-隐藏订单 、6-止损市价单
limtPrice  | String  | 价格  |
stopPrice  | String  | 触发价  |
displaySize  | String   | 展示数量|
volume  | String | 数量  |
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 下单成功、订单已经提交处理
207  | 下单请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效


### 2.5、撤销订单

> 撤销订单
>
> 请求样例：

```json
{
	"orderId":"000000001"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
	"data":{
	    "apiKey":"testApiKey",
		"orderId":"000000001",
     	"orderType":"1",
    	"instrumentId":"BTC-ETH",
    	"direction":"0",
    	"limtPrice":"1",
    	"volume":"2",
    	"stopPrice":"0",
    	"displaySize":"0",
    	"timestamp":"1234567890123"
	}
}
```

**Http request:**  POST /order/cancel

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
orderId  | String   | true  | 订单号  |

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
apiKey  | String  | apiKey  |
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止盈止损 、4-冰山订单 、5-隐藏订单
limtPrice  | String  | 价格  |
stopPrice  | String  | 触发价  |
displaySize  | String   | 展示数量|
volume  | String | 数量  |
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 撤单成功、订单已经提交处理
207  | 撤单请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效


### 2.6、查询订单

> 查询订单
>
> 请求样例：

```json
{
	"instrumentId":"BTC-ETH"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
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
			"timestamp":"1234567890123"
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
			"timestamp":"1234567890123"
		}
	]
}
```

**Http request:**  GET /order/find

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
orderId  | String   | false  | 订单号  |
instrumentId  | String   | false  | 合约号  | 样例 ETH-BTC
direction  |  String  | false  | 方向 | 0-买 1-卖
orderType  | String  | false  | 订单类型  | 1-限价单、2-市价单 、3-止盈止损 、4-冰山订单 、5-隐藏订单
states | String  | false  | 订单状态 0-全部订单，1-已完成订单，2-pending订单
start  | String  | false  | 开始时间  | yyyyMMdd HH:mm:ss
end  | String  | false  | 结束时间  | yyyyMMdd HH:mm:ss
pageNo | String  | false  | 当前页码  | 默认第一页
pageSize | String  | false  | 每页数量  | 默认返回1000条

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
apiKey  | String  | apiKey  |
orderId  | String  | 订单号  | 不能重复
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
orderType  | String  | 订单类型  | 1-限价单、2-市价单 、3-止盈止损 、4-冰山订单 、5-隐藏订单
limtPrice  | String  | 价格  | 当订单类型为3 时必填
stopPrice  | String  | 触发价  | 当订单类型为4 时必填
displaySize  | String   | 展示数量|
volume  | String | 数量  |
orderStatus  | String | 订单状态  | 0-全部成交、1-部分成交 、2-未成交、3-撤单、4-部分成交部分撤单、5-已触发止损
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 查单成功
207  | 查单请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效


### 2.7、查询成交

> 查询成交
>
> 请求样例：

```json
{
	"instrumentId":"BTC-ETH"
}
```

> 响应样例：

```json
{
	"errNo":"200",
	"errMsg":"",
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
			"timestamp":"1234567890123"
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
			"timestamp":"1234567890123"
		}
	]
}
```

**Http request:**  GET /trade/find

**请求正文：**

参数名  | 类型  | 是否必填   | 注释   | 说明
------------- | -------------  | -------------  | -------------  | -------------
instrumentId  | String   | false  | 合约号  | 样例 ETH-BTC
direction  |  String  | false  | 方向 | 0-买 1-卖
start  | String  | false  | 开始时间  | yyyyMMdd HH:mm:ss
end  | String  | false  | 结束时间  | yyyyMMdd HH:mm:ss
pageNo | String  | false  | 当前页码  | 默认第一页
pageSize | String  | false  | 每页数量  | 默认返回1000条

**响应正文：**

参数名  | 类型 | 注释   | 说明
------------- | ------------  | -------------  | -------------
apiKey  | String  | apiKey  |
tradeId  | String  | 成交号  |
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
price  | String  | 价格  |
volume  | String  | 数量  |
fee  | String   | 手续费|
timestamp  | String | 操作时间戳  | 13位

**响应码：**

响应码  | 描述
------------- | ------------ 
200  | 成交查询成功
207  | 成交查询请求包含格式验证错误。查看结果以获取详细信息
400  | 请求格式无效



# WebSocket接口文档

## 1、WebSocket行情数据(公有流)
### 1.1、K线数据

> K线数据
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "topic":"candle_stick",
        "period":"1m",
        "instrumentId":"ETH/USDT"
    }
}
```

> 响应样例：

```json
{
    "type":"sub",
    "topic":"candle_stick",
    "errorCode":"000",
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

**请求正文**

|  参数 |类型  |  是否必填|  描述   |  说明   |
| --- | --- | --- | ---| ---
type| String |true	|类型|	sub
topic| String	|true|主题	|candle_stick
period	| String|true|K线周期|	1m, 5m, 15m, 30m, 60m
instrumentId| String|true|	币对|	ETH/USDT, BTC/ETH ...

**响应正文**

|  参数  |类型 |  描述   |  说明   |
| --- | --- | --- | ---
| type| String	| 类型| 	sub
| topic| String	| 主题| 	candle_stick
| errorCode	| String| 错误码| 	000 ...
| errorMessage| String	| 错误信息| 	success ...
| high| String| 	高|
| open| String	| 开|
| low| String	| 低|
| close| String	| 收|
| instrumentId| String	| 币对| 	ETH/USDT, BTC/ETH ...
| volume	| String| 成交量 |
|timestamp	| String|时间戳|	1492420473027


### 1.2、行情数据

> 行情数据
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "topic":"market_data",
        "instrumentId":"ETH/USDT"
    }
} 
```

> 响应样例：

```json
{
    "type":"sub",
    "topic":"market_data",
    "errorCode":"000",
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

**请求正文**

|  参数 |类型  |是否必填|  描述   |  说明   |
| --- | --- | --- | ---| ---
| type| String	|true| 类型	| sub
| topic| String|true	| 主题	| market_data
| instrumentId| String|true| 	币对| 	ETH/USDT, BTC/ETH ...

**响应正文**

|  参数  |类型 |  描述   |  说明   |
| --- | --- | --- | ---
|type	| String| 类型	| sub
|topic| String	| 主题	| candle_stick
|errorCode| String	| 错误码	| 000 ...
|errorMessage	| String| 错误信息| 	success ...
|high| String| 	高	|
|low	| String| 低	|
|close| String| 	收	|
|instrumentId| String	| 币对| ETH/USDT, BTC/ETH ...
|base| String	|  源币种    |
|target	| String |    目标币种   |
|lastPrice| String	|   最新价格    |
|lastPriceValuation| String	|  最新价格估值   |
|chgRate| String	|  变化比例    |
|chgVol| String	|    变化量  |
|volume| String	| 成交量|


### 1.3、深度行情数据

> 深度行情数据
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "topic":"depth_market_data",
        "instrumentId":"ETH/USDT"
    }
}
```

> 响应样例：

```json
{
    "type":"sub",
    "topic":"depth_market_data",
    "errorCode":"000",
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

**请求正文**

|  参数  |类型 |是否必填|  描述   |  说明   |
| --- | --- | --- |--- | --- 
type	| String|true |类型 |	sub
topic	| String |true|主题 |	depth_market_data
instrumentId| String |true|	币对	 |ETH/USDT, BTC/ETH ...

**响应正文**

|  参数 |类型  |  描述   |  说明   |
| --- | --- | --- | --- |
type	| String |类型 |	sub
topic	| String |主题 |	depth_market_data
errorCode	| String |错误码 |	000 ...
errorMessage| String	 |错误信息 |	success ...
amount	| String |  成交量  	 |
price	| String |	单价	 |
total	| String |	总金额 |


### 1.4、全量成交回报数据

> 全量成交回报数据
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "topic":"trade_rtn_all",
        "instrumentId":"ETH/USDT"
    }
}
```

> 响应样例：

```json
{
    "type":"sub",
    "topic":"trade_rtn_all",
    "errorCode":"000",
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

**请求正文**

|  参数  |类型 |是否必填|  描述   |  说明   |
| --- | --- | --- |--- | --- 
type	| String|true |类型 |	sub
topic	| String |true|主题 |	trade_rtn_all
instrumentId| String |false|	币对	 |ETH/USDT, BTC/ETH ...传空为全量


**响应正文**

|  参数  |类型 |  描述   |  说明   |
| --- | --- | --- | ---
type| String	|类型	|sub
topic	| String|主题|	trade_rtn_all
errorCode| String|	错误码|	000 ...
errorMessage| String	|错误信息|	success ...
fee	| String|		费用	|
instrumentId	| String	|	币对 	|
direction| String		|	交易方向	| 0-买入 ， 1-卖出
tradeId		| String|	交易ID		|
volume	| String	|		成交数量	|
total	| String	|	成交额		|
price	| String	|	价格		|
tradeTimestamp	| String	|	交易时间		|


## 2、WebSocket订单交易数据(私有流)

### 1.1、订单数据

> 订单数据
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "apiKey":"xxxxxxxxxx",
        "topic":"order_rtn",
        "instrumentId":"ETH/USDT"
    }
}
```

> 响应样例：

```json
{
    "type":"sub",
    "topic":"order_rtn",
    "errorCode":"000",
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

**请求正文**

|  参数  |类型 |是否必填|  描述   |  说明   |
| --- | --- | --- | --- | ---
type| String|true|	类型|	sub
topic	| String|true|主题	|order_rtn
instrumentId| String|false|	币对	|ETH/USDT, BTC/ETH ...传空为全量

**响应正文**

|  参数   | 类型 |  描述   |  说明   |
| --- | --- | --- | --- |
type	 | String|类型	|sub
topic	 | String|主题|	order_rtn
errorCode | String|	错误码|	000 ...
errorMessage | String	|错误信息|	success ...
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

### 1.2、交易数据

> 交易数据
>
> 请求样例：

```json
{
    "type":"sub",
    "channel":{
        "apiKey":"xxxxxxxxxx",
        "topic":"trade_rtn",
        "instrumentId":"ETH/USDT"
    }
}
```

> 响应样例：

```json
{
    "type":"sub",
    "topic":"trade_rtn",
    "errorCode":"000",
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

**请求正文**

|  参数   | 类型|是否必填 |  描述   |  说明   |
| --- | --- | --- | --- | ---
type| String|true|	类型|	sub
topic	| String|true|主题	|trade_rtn
instrumentId| String|false|	币对	|ETH/USDT, BTC/ETH ...传空为全量



**响应正文**

|  参数   | 类型 |  描述   |  说明   |
| --- | --- | --- | --- | 
type  | String 	|类型	|sub
topic	  | String |主题|	trade_rtn
errorCode  | String |	错误码|	000 ...
errorMessage	  | String |错误信息|	success ...
apiKey  | String  | apiKey  |
tradeId  | String  | 成交号  |
orderId  | String  | 订单号  |
instrumentId  | String  | 合约号  | 样例 ETH-BTC
direction  |  String  | 方向 | 0-买 1-卖
price  | String  | 价格  |
volume  | String  | 数量  |
fee  | String   | 手续费|
tradeTimestamp  | String | 操作时间戳  | 13位

