

#  简介
欢迎使用HashKey API！此文档是HashKey API的唯一官方文档，HashKey API提供的能力会在此持续更新，请大家及时关注。


## 联系我们
使用过程中如有问题或者建议，您可选择以下任一方式联系我们。


# 接入说明
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
API 请求在通过 internet 传输的过程中极有可能被篡改，为了确保请求未被更改，除公共接口（基础信息，行情数据）外的私有接口均必须使用您的 API Key 做签名认证，以校验参数或参数值在传输途中是否发生了更改。
每一个API Key需要有适当的权限才能访问相应的接口，每个新创建的API Key都需要分配权限。在使用接口前，请查看每个接口的权限类型，并确认你的API Key有相应的权限。

对API端点的所有HTTP请求都需要身份验证和授权。
用户可以通过创建API key流程获得key和sercet，key和sercet用于验证和授权所有请求。
以下标头应添加到所有HTTP请求中：

Key  | Value | Description
------------- | ------------- | -----------
x-access-key  | \<API_KEY> |您申请的 API Key 中的 Access Key
x-access-sign  |  \<signatureOfRequest>| 签名计算得出的值，用于确保签名有效和未被篡改
x-access-timestamp  | \<timeOfRequest> |当前时间格式为当前的时间戳（以毫秒为单位）
x-access-version  |  \<versionOfRequest>   | 签名协议的版本，默认为1

### 签名步骤

### 样例：
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

## 接口概览
### 接口类型
HashKey Pro为用户提供两种接口，您可根据自己的使用场景来选择适合的方式。
**Rest API**
REST，即Representational State Transfer的缩写，是目前较为流行的基于HTTP的一种通信机制，每一个URL代表一种资源。
交易或资产提币等一次性操作，建议开发者使用REST API进行操作。

**WebSocket API**
WebSocket是HTML5一种新的协议（Protocol）。它实现了客户端与服务器全双工通信，通过一次简单的握手就可以建立客户端和服务器连接，服务器可以根据业务规则主动推送信息给客户端。
市场行情和买卖深度等信息，建议开发者使用WebSocket API进行获取。
### 接口简述

## 环境信息
### 测试环境
Restful URL: https://int.hashkey3.com

WebSocket: wss://int.hashkey3.com

### 生产环境
Restful URL: https://int.hashkey3.com

WebSocket: wss://int.hashkey3.com

# 常见问题


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

## REST API
[rest-api](https://github.com/hashkey-pro/open-api/blob/master/rest-api.md)

# Websocket API
## 接入说明
### 接入地址
-  /api/websocket-api/v1/stream
### 接入限制
-  同一apiKey最多允许创建10条session长连接。比如：订单成交数据订阅，同一apiKey最多允许建立10条session连接。
-  同一ip最多允许建立5条session长连接。比如：行情数据推送无需apiKey，同一ip地址最多允许建立5条连接。

### 心跳消息
当用户的Websocket客户端连接到HashKey Websocket服务器后，服务器会定期（当前设为10秒）向其发送ping消息并包含一整数值如下：
``` java
{"type":"ping","data"="1492420473027"}
```

当用户的Websocket客户端接收到此心跳消息后，应返回pong消息并包含同一整数值：

``` java
{"type":"pong","data"="1492420473027"}
```

   注：当Websocket服务器连续两次发送了`ping`消息却没有收到任何一次`pong`消息返回后，服务器将主动断开与此客户端的连接。

### 请求格式
所有的请求都是JSON格式，订阅不同topic传递不同参数，请参考具体接口请求样例。

### 返回格式
所有的响应都是JSON格式，不同topic返回data各有不同，请参考具体接口响应样例。

## 鉴权

### 请求正文

参数 | 类型 | 是否必填 | 注释 | 说明 
---|---|---|---|---
type	| String|true |操作类型 |	默认为auth
x-access-key | String |true |访问Key  |您申请的 API Key 中的 Access Key
x-access-sign | String |true | 签名 | 签名计算得出的值，用于确保签名有效和未被篡改
x-access-timestamp| String |true  | 时间戳 |当前时间格式为当前的时间戳（以毫秒为单位）
x-access-version  | String|true |  版本号   | 签名协议的版本，默认为1

### 响应正文

参数  |类型 |  是否必填|注释   |  说明   
--- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为auth-resp
errorCode	| String |true|错误码 |	0000 
errorMessage| String	|true |错误信息 |	success

``` java
请求样例：
{
    "type":"auth",
    "auth":{
        "x-access-key":"xxxxxxxxx",
        "x-access-sign":"xxxxxxxxx",
        "x-access-timestamp":"150782303000",
		"x-access-version":"2"
    }
}
响应样例：
{
    "type":"auth-resp",
    "errorCode":"0000",
    "errorMessage":"success"    
}
```

## 订阅主题 

### 请求正文
参数  |类型 |  是否必填|注释   |  说明   
--- | --- |--- | --- |---
type	| String|true |操作类型 |	sub
topic | String |true |订阅的主题  | 
apiKey | String |false | apiKey 私有流必传  | 
...|||


### 响应正文
参数  |类型 |  是否必填|注释   |  说明  
--- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为sub-resp
errorCode	| String |true|错误码 |	0000 
errorMessage| String	|true |错误信息 |	success
data | Object |true|数据|

``` java
请求样例：
{
    "type":"sub",
    "channels":  [
		{
        "topic":"xxxxxxxxx",
		......
    	},
		{
        "topic":"xxxxxxxxx",
		......
    	}
	]
}
响应样例：
{
    "type":"sub-resp",
    "errorCode":"0000",
    "errorMessage":"success",
	"data": [
		{
		   ....
		}
	]
}
```

## 取消订阅
### 请求正文
 参数  |类型 |  是否必填|注释   |  说明   
 --- | --- |--- | --- |---
type	| String|true |操作类型 |	unsub
topic | String |true |订阅的主题  | 
...|||


### 响应正文
 参数  |类型 |  是否必填|注释   |  说明   
--- | --- |--- | --- |---
type	| String|true |操作类型 |	默认为unsub-resp
errorCode	| String |true|错误码 |	0000 
errorMessage| String	|true |错误信息 |	success


``` java
请求样例：
{
    "type":"unsub",
    "channel":  {
        "topic":"xxxxxxxxx",
		......
	}
}
响应样例：
{
    "type":"unsub-resp",
    "errorCode":"0000",
    "errorMessage":"success"
}
```


## websocket API
[websocket-api](https://github.com/hashkey-pro/open-api/blob/master/websocket-api.md)

# 错误信息
### 错误码划分规范
错误码 | 描述
--- | ---
0000~0099  |公共错误码 
0100~0199  |WebSocket错误码
0200~0299 | 行情错误码

### 错误码对照表
错误类型 |  错误码   |  错误信息   |  描述          
--- | --- | --- | --- 
公共错误码   |0000    | success   |  成功   |
公共错误码   | 0092    | network error   |  网络连接异常 
公共错误码   | 0096    | system error   |  系统异常 


