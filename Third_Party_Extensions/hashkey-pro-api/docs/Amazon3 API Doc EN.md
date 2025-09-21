

[TOC]



# Change log
|Doc Version|API Version|Date|Modification|Author|
|-------|----|--------------|--------|--------|
|v0.1|v1.0|2022/03/21|Initial version|Dingchun|


# 1. Getting Started

The exchange provides two interface modes for API users:

* **REST**: Use synchronous calls (request/response) to complete functions such as order creation, order cancellation, query transaction history,order history, portfolio, transfer history, etc.
* **WebSocket**: Use an asynchronous method (pubsub) to complete the subscribed push notification of orders, transactions, market data and other information.
In order to facilitate users understanding following are the common steps to access and implement API to complete trading programs

## 1.1 Preparations

### 1.1.1 API-KEY Management

* Users have to log in the exchange website and apply for an API-KEY, please make sure to remember the following information when creating an API key:
  * **Access key**: API access key
  * **secret key**：the key used for signature authentication encryption (visible to the application only)

* Users have to assign permissions to API-KEY. There are four kinds of permissions,
  * **READ** read permission is used for data query interfaces such as order query, transaction query, etc.
  * **TRADE** trade permission is used for order placing, order cancelling, etc.
  * **TRANSFER** transfer permission is used for transfer interfaces, user can transfer between sub-accounts under the same main trading account.

* Users can set IP whitelist for API-KEY. If user set IP for the API-KEY, only the IPs in the whitelist can call the API. Each API-KEY will be bound to a maximum of 5 IPs. If the user has multiple  API-KEYs, they have to set IP whitelist for each API-KEY respectively.

* Both REST and WebSocket modes require users to authenticate the transaction through the API-KEY. Refer to the following chapters for the signature algorithm of the API-KEY..

### 1.1.2 Access Preparation

Before making a transaction, users have to query the server information to ensure that the program is in the correct state:

* Users have to query the server time to ensure that the local time and the server time are consistent
* Users need to query the version number to ensure that the version number is included in each request header. The request may be rejected if the version number is incorrect.

## 1.2 Signature Authentication

### 1.2.1 Signature

API requests are likely to be tampered during transmission through the internet. To ensure that the request remains unchanged, all private interfaces other than public interfaces (basic information, market data) must be verified by signature authentication via API-KEY to make sure the parameters or configurations are unchanged during transmission. Each created API-KEY need to be assigned with appropriate permissions in order to access the corresponding interface. Before using the interface, users is required to check the permission type for each interface and confirm there is appropriate permissions.
All HTTP requests to API endpoints require authentication and authorization. Users can obtain x-access-key and x-access-secret  by creating an API key flow. x-access-key and x-access-secret are used to verify and authorize all requests. The following headers should be added to all HTTP requests:

| Key                | Value                  | Description                                                                                                |
| ------------------ |------------------------|------------------------------------------------------------------------------------------------------------|
| x-access-key       | < API-KEY >            | The API Access Key you applied for                                                                         |
| x-access-sign      | < signatureOfRequest > | The value calculated by the hash value from request to ensure that the signature is valid and not tampered |
| x-access-timestamp | < timeOfRequest >      | The timestamp represents the time of request (in milliseconds)                                             |
| x-access-version   | < versionOfRequest >   | Signature protocol version.                                                       |
| Content-Type       | application/json       | Content-Type                                                                                               |

### 1.2.2 Signature Procedures

**Example:**
* The request is in a Map format and is serialized into a Json string. The map requires at least 3 keys: x-access-key, x-access-timestamp, x-access-version, include all columns in body when HTTP method is POST, include all columns in params where HTTP method is GET or DELETE. The example below illustrates how to sign a map.

```java
/*
  *Signature description：required fields for Signature include apiKey, apiSecret, timestamp, version, interface fields of each request.
  *field name and field value are in the form of key-value, and the Key is sorted by ASCII code.
  *The Following is test data demo:
*/

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;
import java.util.TreeMap;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.codec.binary.Hex;

public class SignDemo {

    public static void main(String[] args) throws Exception {
        String key="84dd8e670471a888e3a7547e120886cb";
        String secret = "3211b4306fcb1d9670cbee5abc69dead";
        String timeOfRequest ="1478692862000";

        TreeMap<String,String> treeMap = new TreeMap<>();

        treeMap.put("x-access-key",key); //api-key
        treeMap.put("x-access-timestamp",timeOfRequest); //current timestamp
        treeMap.put("x-access-version","v1.0");

        treeMap.put("feild1","1");//request field mock1
        treeMap.put("feild2","2");//request field mock2
        treeMap.put("feild3","3");//request field mock3

        String requestText = JSONObject.toJSONString(treeMap);
        //Sort by ASCII of Map key {"feild1":"1","feild2":"2","feild3":"3","x-access-key":"84dd8e670471a888e3a7547e120886cb","x-access-timestamp":"1478692862000","x-access-version":"1"}

        System.out.println(createSignature(requestText,secret));
        //cVfsAceIN7w+3vDf4WEIpA+iWJnK2TjKcmwgVARi8DI=
    }

    private static String createSignature(String requestText,String secret)
            throws Exception {
        return Base64.getEncoder().encodeToString(hmacSha1(requestText, secret));
    }

    private static byte[] hmacSha1(String plainData, String secret)
            throws Exception{
        Mac mac = Mac.getInstance("HmacSHA256");
        SecretKeySpec secretKey =
                new SecretKeySpec(Hex.decodeHex(secret.toCharArray()), "HmacSHA256");
        mac.init(secretKey);
        return mac.doFinal(plainData.getBytes());
    }
}
```

## 1.2 Environment

### 1.2.1 Testing Environment

Restful URL: <https://api-pro-sim.hashkey.com>

WebSocket: wss://api-pro-sim.hashkey.com

### 1.2.2 Production Environment

**Note:** Not public access, application needed

Restful URL: <https://api.pro.hashkey.com>

WebSocket: wss://api.pro.hashkey.com

# 2. REST API

## 2.1 Access

### 2.1.1 REST API Rate Limit
* Unless specified, each API Key has a rate limit of 10 requests per second. For instance, order-related endpoints have a rate limit of 10 requests per second.
* Unless specified, each IP has a rate limit of 10 requests per second.

### 2.1.2 Request Format

All API requests are under RESTful framework. Parameters can be set and sent by the request body in JSON format.

### 2.1.3 Response Format

All endpoints are in JSON standard format.  There are three fields, namely  **error_code**, **error_message**, and **data**. These specific business data are contained in the field **data**.

| **PARAMETER** | **TYPE** | **DESCRIPTION**   |
| ------------- | -------- | ----------------- |
| error_code    | string   | API return status |
| error_message | string   | API error message |
| data          | Object   | API return data   |

## 2.2 Universal

### 2.2.1 Get Timestamp

**Http Request:** GET info/time

**Request Content：** null

**Response Content：**

| **PARAMETER** | **TYPE** | **DESCRIPTION**        |
| ------------- | -------- | ---------------------- |
| timestamp     | int64    | millisecond time-stamp |

**Response Example：**

```json
{
    "error_code":"0000",            // Error code
    "error_message":"",             // Error message
    "data":{
        "timestamp": 1478692862000 // Server timestamp
    }
}
```

### 2.2.2 Get API Version

**Http Request:** GET info/version

**Request Content：** null

**Response Content：**

| **PARAMETER** | **TYPE** | **DESCRIPTION** |
| ------------- | -------- | --------------- |
| version       | string   | Version No.     |

**Response Example：**

```json
{
    "error_code":"0000",        // Error code
    "error_message":"",         // Error message
    "data":{
        "version":"v1.0"        // Server version
    }
}
```

### 2.2.3 Query instruments

**Http Request:** GET info/instruments

**Request Content：** null

**Response Content：**

| **PARAMETER**           | **TYPE** | **DESCRIPTION**                                                                                                                                                    |
|-------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| instrument_id           | string   | Instrument ID.                                                                                                                                                     |
| base_asset              | string   | Base  Asset.                                                                                                                                                       |
| quote_asset             | string   | Quote Asset.                                                                                                                                                       |
| product_type            | string   | Product Type(Token/Token:digital assets exchange, Token/Fiat:exchange digital assets to fiat currency, ST/Token:exchange ST to digital assets, ST/Fiat:exchange ST to fiat currency) |
| price_tick              | string    | Price tick.                                                                                                                                                        |
| max_market_order_volume | string    | Max market order volume.                                                                                                                                           |
| min_market_order_volume | string    | Min market order volume.                                                                                                                                           |
| max_limit_order_volume  | string    | Max limit order volume.                                                                                                                                            |
| min_limit_order_volume  | string    | Min limit order volume.                                                                                                                                            |

**Response Example：**

```json
{
  "error_code":"0000",        
  "error_message":"",         
  "data":[{
    "instrument_id":"BTC-ETH",                 
    "base_asset":"BTC",                        
    "quote_asset":"ETH",                      
    "product_type": "Token/Token",
    "price_tick": "0.00000001",                
    "max_market_order_volume": "10000000.1",   
    "min_market_order_volume": "0.00000001",   
    "max_limit_order_volume": "10000000.1",    
    "min_limit_order_volume": "0.00000001"
  }]
}
```

### 2.2.4 Query instrument status

**Http Request:** GET info/instrument_status/{:instrument_id}
* instrument_id is required in the above url

**Request example：**

```context
 GET "https://domain/info/instrument_status/ETH-BTC"
```

**Response Content：**

| **PARAMETER** | **TYPE** | **DESCRIPTION**                                                                                                 |
|---------------|----------|-----------------------------------------------------------------------------------------------------------------|
| instrument_id | string   | Instrument ID.                                                                                                  |
| status        | string   | Status  "BeforeTrading"、"NoTrading"、"Continuous"、"AuctionOrdering"、 "AuctionBalance"、 "AuctionMatch"、"Closed"   |

| instrument status       | description        |
|-------------------------|--------------------|
| BeforeTrading           | Before trading     |
| NoTrading               | No trading         |
| Continuous              | Continuous trading |
| AuctionOrdering         | Auction ordering   |
| AuctionBalance          | Auction balance    |
| AuctionMatch            | Auction match      |
| Closed                  | Closed             |

**Response Example：**

```json
{
  "error_code":"0000",       
  "error_message":"",       
  "data":[{
    "instrument_id":"ETH-BTC",          
    "status":"Continuous"                          
  }]
}
```

## 2.3 Trading

### 2.3.1 Create An Order(TRADE permission is required)

**Http Request:** POST /order

**Request Content：**

| **PARAMETER**   | **TYPE** | **REQUIRED**                                                  | **DESCRIPTION**                                                                                              |
| --------------- | -------- |---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| type            | string   | true                                                          | "limit": limit order; "market": market order; "stopLimit": stop limit order; "stopMarket": stop market order |
| client_order_id | string   | true                                                          | Max length: 20. Must be unique                                                                               |
| instrument_id   | string   | true                                                          | e.g. "ETH-BTC"                                                                                               |
| direction       | string   | true                                                          | "buy" or "sell"                                                                                              |
| stop_price      | string   | false                                                         | Required when order type is stopLimit or stopMarket                                                          |
| price           | string   | false                                                         | Limit price. Required when order type is limit or stopLimit                                                  |
| volume          | string   | true                                                          | Total Volume                                                                                                 |
| post_only       | bool     | false                                                         | Only maker  default false                                                                                           |
| time_in_force (currently unused) | string   | default: limit and stopLimit: GTC, market and stopMarket: IOC |

 **Response Content：**

| **PARAMETER**   | **TYPE** | **DESCRIPTION**                                                                                              |
| --------------- | -------- |--------------------------------------------------------------------------------------------------------------|
| type            | string   | "limit": limit order; "market": market order; "stopLimit": stop limit order; "stopMarket": stop market order |
| sys_order_id    | string   | Server order id.                                                                                             |
| client_order_id | string   | Client order id.                                                                                             |
| instrument_id   | string   | e.g. "ETH-BTC"                                                                                               |
| direction       | string   | "buy" or "sell"                                                                                              |
| stop_price      | string   | Required when order type is stopLimit or stopMarket                                                          |
| price           | string   | Limit Price. Required when order type is limit or stopLimit                                                  |
| volume          | string   | Total Volume                                                                                                 |
| post_only       | bool     | Only maker                                                                                                   |
| timestamp       | int64    | millisecond time-stamp                                                                                       |
| time_in_force (currently unused) | string   | default: limit and stopLimit: GTC, market and stopMarket: IOC                                                |

**Request Example：**

```json
{
    "type": "limit",                    // Order type:
    "client_order_id":"000000001",      // Client order ID
    "instrument_id":"ETH-BTC",          // Instrument ID
    "direction":"buy",                  // Trade direction: buy、sell
    "price":"1",                        // Limit price
    "volume":"1",                       // Total Volume
    "post_only": true                   // Whether to only be a maker
}
```

**Response Example：**

```json
{
    "error_code":"0000",    // Errorcode
    "error_message":"",     // Errormessage
    "data":{
        "type":"limit",                // Order type
        "sys_order_id":"1550849345000001",     // System order ID
        "client_order_id":"000000001",  // Client order ID
        "instrument_id":"ETH-BTC",      // Instrument ID
        "direction":"buy",              // Trade direction
        "stop_price": "0",              // Stop price
        "price":"1",                    // Limit price
        "volume":"1",                   // Total Volume
        "post_only": true,              // Whether to only be a maker
        "timestamp": 1478692862000      // Order timestamp
    }
}
```

### 2.3.2 Cancel An Order(TRADE permission is required)

**Http Request:** DELETE /order/

**Query Parameters：**

| **PARAMETER**   | **TYPE**   | **REQUIRED**  | **DESCRIPTION**                                         |
|-----------------|------------|---------------|---------------------------------------------------------|
| sys_order_id    | string     | false         | This filed is required when client_order_id is null     |
| client_order_id | string     | false         | This filed is required when sys_order_id is null        |
| volume          | string     | false         | If it is not input, the whole order will be cancelled   |
Note: If neither sys_order_id nor client_order_id is empty, then client_order_id is ignored.

**Response Content：**
null

**Request example：**

```context
 DELETE "https://domain/order?&sys_order_id=1550849345000001"
```
**Response Example：**

```json
{
  "error_code":"0000",    // 错误码
  "error_message":""      // 错误描述
}
```

### 2.3.3 Cancel all orders（TRADE permission is required）

**Http Request:** DELETE /orders

**Query Parameters：**

| **PARAMETER** | **TYPE**   | **REQUIRED** | **DESCRIPTION**                         |
|------------|------------|--------------|-----------------------------------------|
| instrument_id       | string     | false        | Cancel orders on a specific instrument_id only |


**Response Content：**
null

**Request example：**

```context
 DELETE "https://domain/orders?instrument_id=BTC-ETH"
```

**Response Example：**

```json
{
    "error_code":"0000",    // Error code
    "error_message":""     // Error message
}
```

### 2.3.4 Get Order Data(READ permission is required)

**Http Request:**  GET /orders

**Query Parameters：**

| **PARAMETER**   | **TYPE**     | **REQUIRED** | **DESCRIPTION**                                              |
|-----------------| ------------ |--------------|--------------------------------------------------------------|
| sys_order_id    | string       | false        | Server Order ID                                              |
| instrument_id   | string       | false        | e.g. "ETH-BTC"                                               |
| sorting         | string       | false        | "desc" or "asc"  default "asc"                                 |
| direction       | string       | false        | "buy" or "sell"                                              |
| type            | string       | false        | Order type                                                   |
| status          | array string | false        | Array with order statuses to filter by.                      |
| start_timestamp | string        | true         | millisecond time-stamp                                       |
| end_timestamp   | string        | true         | millisecond time-stamp                                       |
| limit           | string        | true        | Limit on number of results to return. min 1 max 200. |
| page            | string        | true         | Used for pagination. Page number.                            |

**Response Content：**

| **PARAMETER**     | **TYPE** | **DESCRIPTION**                                                                                              |
|-------------------| -------- |--------------------------------------------------------------------------------------------------------------|
| type              | string   | "limit": limit order; "market": market order; "stopLimit": stop limit order; "stopMarket": stop market order |
| sys_order_id      | string   | Server Order ID                                                                                              |
| client_order_id   | string | Client order id.                                                                                             |
| instrument_id     | string   | e.g. "ETH-BTC"                                                                                               |
| direction         | string   | "buy" or "sell"                                                                                              |
| stop_price        | string   | Required when order type is stopLimit or stopMarket                                                          |
| price             | string   | Limit Price. Required when order type is limit or stopLimit                                                  |
| volume            | string   | Original Total Volume                                                                                        |
| status            | string   | Order status                                                                                                 |
| post_only         | bool     | Only maker                                                                                                   |
| timestamp         | int64    | millisecond time-stamp                                                                                       |
| filled_size       | string   | The size that has been filled                                                                                |
| unfilled_size     | string   | The size that has not been filled                                                                            |
| avg_filled_price  | string   | Average filled price                                                                                         |
| sum_trade_amount | string   | cumulative trading amount(turnover)                                                                          |

**Order status**

| order status               | description                                                                            |
|----------------------------|----------------------------------------------------------------------------------------|
| -------------------------- | ------------------------------------------------------------                           |
| NEW                        | The order has been accepted by the engine                                              |
| FILLED                     | The order has been completed                                                           |
| PARTIALLY_FILLED           | A part of the order has been filled, and the rest remains in the order book            |
| CANCELED                   | The order has been canceled by the user, whitout any trade                             |
| PARTIALLY_CANCELED         | A part of the order has been filled, and the  remaining part has been canceled by user |
| NOT_TRIGGERED              | The Stop order  is not triggered                                                       |
| REJECTED                   | The order was not accepted by the engine and not processed. (Currently not enabled)    |

**Request Example：**

```context
 GET "http://domain/orders?sys_order_id=1550849345000001&instrument_id=ETH-BTC&start_timestamp=1656928657000&end_timestamp=1656928717000&limit=50&page=1"
```

**Response Example：**

```json
{
    "error_code":"0000",    // Error code
    "error_message":"",     // Error message
    "data":[{
        "type": "limit",                // Order type
        "sys_order_id":"1550849345000001",     // System order ID
        "client_order_id":"000000001",  // Client order ID
        "instrument_id":"ETH-BTC",      // InstrumentID
        "direction":"buy",              // Trade direction
        "stop_price":"0",               // Stop price
        "price":"1",                    // Limit Price
        "volume":"1",                   // Original Total Volume
        "status": "FILLED",             // Order status
        "post_only": false,             // Only as "maker"
        "timestamp": 1478692862000,     // Order timestamp
        "filled_size": "1",             // The size that has been filled
        "avg_filled_price": "1",        // Average filled price
        "unfilled_size": "0",           // The size that has not been filled
        "sum_trade_amount": "1"         // turnover
    }]
}
```

### 2.3.5 Retrieve Trade Data(READ permission is required)

**Http Request:** GET /trades

**Query Parameters** **:**

| **PARAMETER**   | **TYPE** | **REQUIRED** | **DESCRIPTION**                                              |
|-----------------| -------- |--------------|--------------------------------------------------------------|
| instrument_id   | string   | false        | e.g. "ETH-BTC"                                               |
| sys_order_id    | string   | false        | Server Order ID                                              |
| direction       | string   | false        | "buy" or "sell"                                              |
| sorting         | string   | false        | "desc" or "asc"    default "asc"                               |
| limit           | string    | true        | Limit on number of results to return. min 1 max 200. |
| page            | string    | true         | Used for pagination. Page number.                            |
| start_timestamp | string    | true         | millisecond time-stamp                                       |
| end_timestamp   | string    | true         | millisecond time-stamp                                       |

 **Response Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**        |
| ------------- | -------- |------------------------|
| trade_id      | string   | Trade ID               |
| sys_order_id  | string   | Order ID               |
| instrument_id | string   | e.g. "ETH-BTC"         |
| direction     | string   | "buy" or "sell"        |
| price         | string   | Price                  |
| volume        | string   | Volume                 |
| fee           | string   | Fee                    |
| fee_ccy       | string   | Fee Currency           |
| timestamp     | int64    | millisecond time-stamp |
| trade_type    | string   | "Common", "Invalid"    |

**Trade Type**

| Trade type         | description |
|--------------------|-------------|
| Common             | Common Trade        |
| Invalid            |  Invalid Trade        |


**Request Example:**

```context
 GET "http://domain/trades?sys_order_id=1550849345000001&instrument_id=ETH-BTC&start_timestamp=1656928657000&end_timestamp=1656928717000&limit=50&page=1"
```

**Response Example：**

```json
{
    "error_code":"0000",    // Error code
    "error_message":"",     // Error message
    "data":[{
        "sys_order_id":"1550849345000001",     // System order ID
        "trade_id":"1",                 // Trade ID
        "instrument_id":"ETH-BTC",      // Instrument ID
        "direction":"buy",              // Trade direction
        "price":"1",                    // Price
        "volume":"1",                   // Volume
        "fee":"0.05",                   // Transaction Fee
        "fee_ccy": "BTC",              // Transaction Fee currency
        "timestamp": 1478692862000,      // Trade timestamp
        "trade_type": "Common"          // Trade type
    }]
}
```

## 2.4 Asset

### 2.4.1 Query assets(READ permission is required)

**Http Request:** GET /assets

**Request Content：** null

**Response Content：**

| **PARAMETER** | **TYPE** | **DESCRIPTION**        |
| ------------- | -------- | ---------------------- |
| asset         | string   | Asset type, e.g. "BTC" |
| free          | string   | Avalible balance       |
| freeze        | string   | Frozen balance         |

**Response Example：**

```json
{
    "error_code":"0000",
    "error_message":"",
    "data":[{
        "asset":"ETH",
        "free": "1",
        "freeze": "0",
    }, {
        // other asset
    }]
}
```

### 2.4.2 Transfer between trading accounts(TRANSFER permission is required)

**Http Request:** POST /assets/transfer

**Request Content：**

| **PARAMETER**   | **TYPE** | **REQUIRED** | **DESCRIPTION** |
|-----------------| -------- | ------------ |-----------------|
| asset           | string   | true         | Asset ID        |
| amount          | string   | true         | Amount          |
| to_account_id   | string   | true         | To Account ID   |
| from_account_id | string   | true         | From Account ID |

**Response Content：**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION** |
| ------------- | -------- | ------------ | --------------- |
| error_code    | string   | true         | Error Code      |
| error_message | string   | true         | Error Message   |
| data          | string   | true         | Data            |

**Request Example：**

```json
{
    "asset":"ETH",
    "amount":"1",
    "from_account_id": "B000000000001",
    "to_account_id":"B000000000002"
}
```

**Response Example：**

```json
{
    "error_code":"0000",
    "error_message":"",
    "data":{}
}

```

### 2.4.3 Query withdrawal history（READ permission is required）

**Http Request:** GET /withdraw/history

**Query Parameters** **:**

| **PARAMETER**     | **TYPE** | **REQUIRED** | **DESCRIPTION**                                            |
|-------------------| -------- |--------------|------------------------------------------------------------|
| currency          | string   | false        | Currency                                            |
| status            | string   | false        | Status   "failed"、"withdrawing"、"successful"、"cancelling"、"cancelled"              |
| limit             | string    | true         | Limit on number of results to return. min 1 max 200 |
| page              | string    | true         | Used for pagination. Page number.                          |
| start_timestamp   | string    | true         | millisecond time-stamp                                     |
| end_timestamp     | string    | true         | millisecond time-stamp                                     |


**Response Content：**

| **PARAMETER**     | **TYPE** | **DESCRIPTION**                |
|-------------------|----------|--------------------------------|
| withdraw_order_id | string   | withdraw order ID              |
| txn_id            | string   | Txn ID                         |
| network           | string   | Network (currently unused)     |
| currency          | string   | Currency                       |
| address           | string   | Withdrawal destination address |
| memo              | string   | Memo                           |
| volume            | string   | Volume                         |
| status            | string   | Status                         |
| gas_fee           | string   | Gas Fee                        |
| gas_fee_ccy       | string   | Gas Fee Currency               |
| fee               | string   | Fee                            |
| fee_ccy           | string   | Fee Currency                   |
| timestamp         | int64    | Timestamp                      |

**Request example：**

```context
 GET "https://domain/withdraw/history?currency=BTC&start_timestamp=1656928657000&end_timestamp=1656928717000&limit=50&page=1"
```


**Response Example：**

```json
{
  "error_code":"0000",
  "error_message":"",
  "data":[{
    "withdraw_order_id": "00000001",
    "txn_id": "60fd9007ebfddc753455f95fafa808c4302c836e4d1eebc5a132c36c1d8ac354",
    "currency": "BTC",
    "address": "1FZdVHtiBqMrWdjPyRPULCUceZPJ2WLCsB",
    "memo": "",
    "volume": "1",
    "status": "successful",
    "fee": "0.004",
    "fee_ccy": "BTC",
    "gas_fee": "0.0001",
    "gas_fee_ccy": "BTC",
    "timestamp": 1478692862000
  }]
}
```

### 2.4.4 Query deposit history（READ permission is required）

**Http Request:** GET /deposit/history

**Query Parameters** **:**

| **PARAMETER**     | **TYPE** | **REQUIRED** | **DESCRIPTION**                                            |
|-------------------| -------- |--------------|------------------------------------------------------------|
| currency          | string   | false        | Currency                                                                                          |
| status           | string    | false        | Status  "addressToBeVerified"、"underReview"、"successful"、"failed"、"refundInProgress"、"refundComplete"、"refundFailed" |
| page              | string    | true         | Used for pagination. Page number.                   |
| limit             | string    | true        | Limit on number of results to return. min 1 max 200 |
| start_timestamp   | string    | true         | millisecond time-stamp                                     |
| end_timestamp     | string    | true         | millisecond time-stamp                                     |


**Response Content：**

| **PARAMETER**    | **TYPE** | **DESCRIPTION**            |
|------------------|----------|----------------------------|
| deposit_order_id | string   | Deposit order ID           |
| txn_id           | string   | Txn ID                     |
| network           | string   | Network (currently unused) |
| currency         | string   | Currency                   |
| address          | string   | Deposit source address     |
| memo             | string   | Memo                       |
| volume           | string   | Volume                     |
| status           | string   | Status                     |
| fee              | string   | Fee                        |
| fee_ccy          | string   | Fee Currency               |
| timestamp        | int64    | Timestamp                  |

**Request example：**

```context
 GET "https://domain/deposit/history?currency=BTC&start_timestamp=1656928657000&end_timestamp=1656928717000&limit=50&page=1"
```

**Response Example：**

```json
{
  "error_code":"0000",
  "error_message":"",
  "data":[{
    "deposit_order_id": "00000001",
    "txn_id": "60fd9007ebfddc753455f95fafa808c4302c836e4d1eebc5a132c36c1d8ac354",
    "currency": "BTC",
    "address": "1FZdVHtiBqMrWdjPyRPULCUceZPJ2WLCsB",
    "memo": "",
    "volume": "1",
    "status": "successful",
    "fee": "0.004",
    "fee_ccy": "BTC",
    "timestamp": 1478692862000
  }]
}
```

### 2.4.5 Query assets transfer history（READ permission is required）

**Http Request:** GET /assets/transfer/history

**Query Parameters** **:**

| **PARAMETER**     | **TYPE** | **REQUIRED** | **DESCRIPTION**                                            |
|-------------------| -------- |--------------|------------------------------------------------------------|
| start_timestamp   | string    | true         | millisecond time-stamp                                     |
| end_timestamp     | string    | true         | millisecond time-stamp                                     |
| limit             | string    | true        | Limit on number of results to return. min 1 max 200 |
| page              | string    | true         | Used for pagination. Page number.                          |

**Response Content：**

| **PARAMETER**    | **TYPE** | **DESCRIPTION**    |
|------------------|----------|--------------------|
| asset           | string   | Asset ID           |
| amount          | string   | Amount             |
| from_account_id | string   | From Account ID    |
| to_account_id   | string   | To Account ID      |
| status          | string   | "successful", "failed" |
| timestamp       | int64    | Timestamp          |


**Request example：**

```context
 GET "https://domain/assets/transfer/history?start_timestamp=1656928657000&end_timestamp=1656928717000&limit=50&page=1"
```

**Response Example：**

```json
{
    "error_code":"0000",
    "error_message":"",
    "data":[{
      "asset":"ETH",
      "amount":"1",
      "from_account_id": "B000000000001",
      "to_account_id":"B000000000002",
      "status": "successful",
      "timestamp": 1478692862000
    }]
}
```

## 2.5 Market

### 2.5.1 Get Kline

**Http Request:** GET market/kline

**Query Parameters：**

| **PARAMETER**    | **TYPE** | **REQUIRED** | **DESCRIPTION**                                                                                                                                                     |
|------------------|----------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| instrument_id    | string   | true         | e.g. "ETH-BTC"                                                                                                                                                      |
| period           | string   | true         | m -> minutes; h -> hours; d -> days; w -> weeks; M -> months;<br/> "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",<br/> "6h", "8h", "12h", "1d", "3d", "1w", "1M" |
| start_timestamp  | string   | true         | millisecond time-stamp  start from 000 milliseconds of this period                                                                                                  |
| end_timestamp    | string   | true         | millisecond time-stamp  end at 000 milliseconds of the next period                                                                                                  |
| page             | string   | true         | Used for pagination. Page number.                   |
| limit            | string   | true         | min 1 max 200                                                                                                                                                       |

**Response Content：**

| **PARAMETER**   | **TYPE** | **DESCRIPTION**        |
|-----------------| -------- | ---------------------- |
| instrument_id   | string   | e.g. "ETH-BTC"               |
| open            | string   | Open Price                   |
| close           | string   | Close Price                  |
| high            | string   | High Price                   |
| low             | string   | Low Price                    |
| volume          | string   | Volume in base asset, e.g, ETH in ETH-BTC |
| start_timestamp | int64    | Start millisecond time-stamp |
| end_timestamp   | int64    | End millisecond time-stamp   |

**Response Example：**

```json
{
    "error_code":"0000",    // Error code
    "error_message":"",     // Error message
    "data":[
        {
            "instrument_id":"ETH-BTC",      // Instrument ID
            "open":"10",                    // Open price
            "close":"10",                   // Close price
            "high":"10",                    // High price
            "low":"10",                     // Low price
            "volume":"100",                 // Volume
            "start_timestamp":1646213700000,     // Start time
            "end_timestamp":1646213760000        // End time
        }
    ]
}
```

### 2.5.2 Get Trade List

**Http Request:** GET /trades/market

**Query Parameters：**

| **PARAMETER**   | **TYPE** | **REQUIRED** | **DESCRIPTION**                   |
|-----------------|----------|--------------|-----------------------------------|
| instrument_id   | string   | false        | e.g. "ETH-BTC"                    |
| start_timestamp | string    | true         | millisecond time-stamp            |
| end_timestamp   | string    | true         | millisecond time-stamp            |
| limit           | string    | true        | min 1 max 200.          |
| page            | string    | true         | Used for pagination. Page number. |

**Response Content：**

| **PARAMETER** | **TYPE** | **DESCRIPTION**  |
| ------------- | -------- | ---------------- |
| instrument_id | string   | e.g. "ETH-BTC"   |
| trade_id      | string   | Trade ID         |
| price         | string   | Price            |
| volume        | string   | Volume           |
| timestamp     | int64    | millisecond time-stamp |
| direction     | string   | Taker direction        |

**Response Example：**

```json
{
    "error_code":"0000",    // Error code
    "error_message":"",     // Error message
    "data":[
        {
            "instrument_id": "ETH-BTC",     // Instrument ID
            "trade_id": "123456789",        // Trade ID
            "price": "10",                  // Price
            "volume": "100",                // Volume
            "timestamp": 1478692862000      // Trade timestamp
            "direction": "buy"              // Taker direction
        }
    ]
}
```

# 3. Websocket API

## 3.1 Access

### 3.1.1 WebSocket API session Limit
* Use apikey to authorize websocket sessions, and 10 sessions can be authorized at the same time.
* Only authorized sessions can subscribe to private data and public data.

**URL for Access:**
 /stream

### 3.1.2 Heartbeat Message

When user's Websocket client application gets connected with the HashKey Websocket server, the server will periodically (every 10 seconds currently) send a ping message containing the sessionID and current timestamp.

```json
 {"type":"ping","sessionID":"74939a43-0523-4cb1-a870-0dbadfda6a62","data":"1492420473027"}
```

When the user receives the mentioned message from the Websocket client application, a pong message containing the same timestamp should be returned.

```json
{"type":"pong","data":"1492420473027"}
```

**Note**: When the Websocket server sends two ping messages successively but gets no pong message in return, the connection will be disconnected.

### 3.1.3 Request Format

All subscription request bodies are expected to be in valid JSON format. Each topic has its own parameter set. For details, please refer to the endpoint request example.

### 3.1.4 Response Format

All response bodies are expected to be in valid JSON format.  For details, please refer to the endpoint response example.

### 3.1.5 Authentication

**Request Content:**

| **PARAMETER**      | **TYPE** | **REQUIRED** | **DESCRIPTION**                                                                          |
| ------------------ |----------| ------------ |------------------------------------------------------------------------------------------|
| type               | string   | true         | Default: auth                                                                            |
| x-access-key       | string   | true         | The API Access Key you applied for.                                                      |
| x-access-sign      | string   | true         | A value calculated by the hash value from request to ensure it is valid and not tampered. |
| x-access-timestamp | string   | true         | The timestamp represents the time of request (in milliseconds).                          |
| x-access-version   | string   | true         | Signature protocol version. Default version is 1.                                        |

**Request Example：**

 ```json
{
    "type":"auth",
    "auth":{
        "x-access-key":"xxxxxxxxx",
        "x-access-sign":"xxxxxxxxx",
        "x-access-timestamp": "1478692862000",
        "x-access-version":"v1.0"
    },
    "id": 1
}
 ```

**Response Example：**

```json
{
    "type":"auth-resp",
    "error_code": "0000",
    "error_message": "success"
}
```

### 3.1.6 Subscribe

**Request Content:**

| **PARAMETER** | **TYPE**     | **REQUIRED** | **DESCRIPTION**      |
|---------------| ------------ | ------------ |----------------------|
| type          | string       | true         | "sub"                |
| id            | integer      | true         | Unique request id    |
| parameters    | object array | true         | Subscribe parameters |
| => topic      | string       | true         | Topic                |

**Response Content:**

| **PARAMETER** | **TYPE**     | **DESCRIPTION**                                |
| ------------- | ------------ |------------------------------------------------|
| id            | integer      | Unique request id                              |
| result        | object array | Topic list                                     |
| error_code    | string       | 0000: success 0100: partial success 0001: fail |

**Request Example:**

```json
{
    "type": "sub",            // Message type, sub: subscribe
    "parameters":[{
        "topic":"order_rtn",            // Subscribe topoic
        "instrument_id":"ETH-USDT"      // Instrument ID
    }],
    "id": 1                             // Message ID
}
```

**Response Example:**

```json
{
    "id": 1,                          // Message ID
    "result": null,                   // Result is null when all topics are successfully subscribed; otherwise, it returns the topics that are successfully subscribed
    "error_code": "0000"              // Error code is 0000 when all topics are successfully subscribed; otherwise, it returns the error code that unsuccessfully subscribed
}
```

### 3.1.7 Unsubscribe

**Request Content:**

| **PARAMETER** | **TYPE**     | **REQUIRED** | **DESCRIPTION**        |
|---------------|--------------| ------------ |------------------------|
| type          | string       | true         | "unsub"                |
| id            | int64        | true         | Unique request id      |
| parameters    | object array | true         | Unsubscribe parameters |
| => topic      | string       | true         | Topic                  |
| error_code    | string       | true         | Error code             |
| error_message | string       | true         | Explain error info     |

**Response Content:**

| **PARAMETER** | **TYPE**     | **DESCRIPTION**                                |
| ------------- |--------------|------------------------------------------------|
| id            | int64        | Unique request id                              |
| result        | object array | Topic list                                     |
| error_code    | string       | 0000: success 0100: partial success 0001: fail |

**Request Example:**

 ```json
{
    "type":"unsub",
    "parameters":  [{
        "topic":"xxxxxxxxx",
        ......
    }],
    "id": 2
}
 ```

**Response Example:**

```json
{
    "id": 1,
    "result": null,
    "error_code": "0000"
}
```



## 3.2 Market Data (Public stream)

### 3.2.1 Kline

**Push Frequency**
Push every 1000 milliseconds

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**                                              |
| ------------- | -------- | ------------ | ------------------------------------------------------------ |
| type          | string   | true         | "sub"                                                        |
| topic         | string   | true         | "kline"                                                      |
| period        | string   | true         | m -> minutes; h -> hours; d -> days; w -> weeks; M -> months;<br/> "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",<br/> "6h", "8h", "12h", "1d", "3d", "1w", "1M" |
| instrument_id | string   | true         | e.g. "ETH-USDT", "ETH-BTC"                                   |

**Response Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**                               |
| ------------- | -------- | --------------------------------------------- |
| type          | string   | "sub-resp"                                    |
| topic         | string   | "kline"                                       |

**Data Content:**

| **PARAMETER**   | **TYPE** | **DESCRIPTION**                               |
|-----------------| -------- | --------------------------------------------- |
| instrument_id   | string   | e.g. "ETH-USDT", "ETH-BTC"                    |
| high            | string   | High  price                                   |
| open            | string   | Open price                                    |
| low             | string   | Low  price                                    |
| close           | string   | Close  price                                  |
| volume          | string   | Volume in base asset, e.g, ETH in ETH-BTC     |
| start_timestamp | int64    | millisecond time-stamp     e.g. 1646213700000 |
| end_timestamp   | int64    | millisecond time-stamp     e.g. 1646213800000 |

**How to Subscribe：**

```json
{
    "type":"sub",                       // Message type
    "parameters":[{
        "topic":"kline",                // Subscribe topoic(kline)
        "period":"1m",                  // Time period
        "instrument_id":"ETH-USDT"      // Instrument ID
    }],
    "id": 1                             // Message ID
}
```

**Response Example：**

```json
{
    "type":"sub-resp",                      // Message type
    "topic":"kline",                        // Subscribe topoic
    "data":{
            "instrument_id":"ETH-BTC",      // Instrument ID
            "open":"10",                    // Open price
            "close":"10",                   // Close price
            "high":"10",                    // Highest price
            "low":"10",                     // Lowest price
            "volume":"100",                 // Volume in base asset
            "start_timestamp":1646213700000,     // Start time
            "end_timestamp":1646213760000        // End time
    }
}
```

### 3.2.2 Market Data

**Push frequency**
Push every 1000 milliseconds

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**            |
| ------------- | -------- | ------------ | -------------------------- |
| type          | string   | true         | "sub"                      |
| topic         | string   | true         | "market_data"              |
| instrument_id | string   | true         | e.g. "ETH-USDT", "ETH-BTC" |

**Response Content:**

| **PARAMETER**        | **TYPE** | **DESCRIPTION**        |
| -------------------- | -------- | ---------------------- |
| type                 | string   | "sub-resp"             |
| topic                | string   | "market_data"          |

**Data Content:**

| **PARAMETER**        | **TYPE** | **DESCRIPTION**                           |
|----------------------| -------- |-------------------------------------------|
| high                 | string   | High price                                |
| low                  | string   | Low price                                 |
| open                 | string   | Open price                                |
| instrument_id        | string   | e.g. "ETH-USDT", "ETH-BTC"                |
| base                 | string   | Base asset, e.g, ETH in ETH-BTC           |
| quote                | string   | Quote asset, e.g, BTC in ETH-BTC          |
| last_price           | string   | Price of the latest trade                 |
| price_change_rate    | string   | Price change rate of 24 hours             |
| price_change         | string   | Price change  of 24 hours                 |
| volume               | string   | Volume in base asset, e.g, ETH in ETH-BTC |

**How to Subscribe：**

```json
{
    "type":"sub",                           // Message type
    "parameters":[{
        "topic":"market_data",              // Subscribe topoic
        "instrument_id":"ETH-USDT"          // Instrument ID
    }],
    "id": 1                                 // Message ID
}
```

**Response Example：**

```json
{
    "type":"sub-resp",                  // Message type: sub-resp: Subscription results
    "topic":"market_data",              // Subscribe topic
    "data":[
        {
            "open":"0",                 // Open price
            "high":"0",                 // Highest price
            "low":"0",                  // Lowest price
            "base":"ETH",               // Base asset
            "quote":"USDT",             // Quote asset
            "instrument_id":"ETH-USDT", // Instrument ID
            "price_change_rate":"0",    // Price change rate of 24 hours
            "price_change":"0",         // Price change rate of 24 hours
            "last_price":"0",           // Price of the latest trade
            "volume":"0"                // Volume of 24 hours
        }
    ]
}
```

### 3.2.3 Order book Data

**Push frequency**
Push every 500 milliseconds(If there is any change)

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**            |
| ------------- | -------- | ------------ | -------------------------- |
| type          | string   | true         | "sub"                      |
| topic         | string   | true         | "depth_market_data"        |
| instrument_id | string   | true         | e.g. "ETH-USDT", "ETH-BTC" |

**Response Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**              |
| ------------- | -------- |------------------------------|
| type          | string   | "sub-resp"                   |
| topic         | string   | "depth_market_data"          |
| instrument_id | string   | e.g. "ETH-USDT", "ETH-BTC"   |

**Data Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**     |
| ------------- | -------- | ------------------- |
| volume        | string   | Volume              |
| price         | string   | Price               |
| timestamp     | int64    | Timestamp       |

**How to Subscribe：**

```json
{
    "type":"sub",                           // Message type
    "parameters":[{
        "topic":"depth_market_data",        // Subscribe topoic
        "instrument_id":"ETH-USDT"          // Instrument ID
    }],
    "id": 1                                 // Message ID
}
```

**Response Example：**

```json
{
    "type":"sub-resp",                      // Message type
    "topic":"depth_market_data",            // Subscribe topoic
    "instrument_id":"ETH-USDT",
    "data":{
        "asks":[                            // Sell 50 levels, sorted from small to large according to the price
            {
                "volume":"3",               // volume
                "price":"1.7"                // price
            },
            {
                "volume":"3",
                "price":"2"
            }
        ],
        "bids":[
            {                              // Buy 50 levels, sorted from large to small according to the price
                "volume":"3",
                "price":"1.5"
            }
        ]
    },
  "timestamp": 1646213700000
}
```

### 3.2.4 Total Trade Data

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**            |
| ------------- | -------- |--------------|----------------------------|
| type          | string   | true         | "sub"                      |
| topic         | string   | true         | "trade_rtn_all"            |
| instrument_id | string   | false        | e.g. "ETH-USDT", "ETH-BTC" |

**Response Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**  |
| ------------- | -------- | ---------------- |
| type          | string   | "sub-resp"       |
| topic         | string   | Topic            |


**Data Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**        |
|---------------| -------- |------------------------|
| instrument_id | string   | Instrument Id          |
| trade_id      | string   | Trade Id               |
| volume        | string   | Volume                 |
| price         | string   | Price                  |
| timestamp     | int64x   | millisecond time-stamp |
| direction     | string   | Taker direction        |

**How to Subscribe：**

```json
{
    "type":"sub",
    "parameters":[{
        "topic":"trade_rtn_all",
        "instrument_id":"ETH-USDT"
    }],
    "id": 1
}
```

**Request Response：**

```json
{
    "type":"sub-resp",
    "topic":"trade_rtn_all",                    // Subscribe topoic
    "data":[
            {
                "instrument_id":"ETH-USDT",     // Instrument ID
                "trade_id":"1000001",           // Trade ID
                "volume":"2",                   // Volume
                "price":"2",                    // Price
                "timestamp":1478692862000,      // Trade time
                "direction": "buy"              // Taker direction
            },
            {
                "instrument_id":"ETH-USDT",
                "trade_id":"1000002",
                "volume":"2",
                "price":"2",
                "timestamp":1478692862000,
                "direction": "buy"              
            }
        ]
}
```

### 3.2.5 Instruments status change

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**             |
| ------------- | -------- | ------------ |-----------------------------|
| type          | string   | true         | "sub"                       |
| topic         | string   | true         | "instruments_status_change" |
| instrument_id | string   | false        | e.g. "ETH-USDT", "ETH-BTC"  |

**Response Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**  |
| ------------- | -------- | ---------------- |
| type          | string   | "sub-resp"               |
| topic         | string   | Topic                    |

**Data Content:**

| **PARAMETER** | **TYPE** | **DESCRIPTION**        |
|---------------| -------- |------------------------|
| instrument_id | string   | Instrument Id          |
| status        | string   | Status                 |


**How to Subscribe：**

```json
{
    "type":"sub",
    "parameters":[{
        "topic":"instruments_status_change",
    }],
    "id": 1
}
```

**Request Response：**

```json
{
    "type":"sub-resp",
    "topic":"instruments_status_change",                    
    "data":[
            {
                "instrument_id":"ETH-BTC",      
                "status":""                     
            }
        ]
}
```


## 3.3  Transaction Data (Private Stream)

### 3.3.1 Order Data

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**            |
| ------------- | -------- | ------------ | -------------------------- |
| type          | string   | true         | "sub"                      |
| topic         | string   | true         | "order_rtn"                |
| instrument_id | string   | false        | e.g. "ETH-USDT", "ETH-BTC" |

**Response Content:**

| **PARAMETER**   | **TYPE** | **DESCRIPTION**                                              |
| --------------- | -------- | ------------------------------------------------------------ |
| type            | string   | "sub-resp"                                                   |
| topic           | string   | "order_rtn"                                                  |

**Data Content:**

| **PARAMETER**   | **TYPE** | **DESCRIPTION**                                                                                              |
| --------------- | -------- |--------------------------------------------------------------------------------------------------------------|
| type            | string   | "limit": limit order; "market": market order; "stopLimit": stop limit order; "stopMarket": stop market order |
| sys_order_id    | string   | Server Order ID                                                                                              |
| client_order_id | string   | Client order id.                                                                                             |
| instrument_id   | string   | e.g. "ETH-BTC"                                                                                               |
| direction       | string   | "buy" or "sell"                                                                                              |
| stop_price      | string   | Required when order type is stopLimit or stopMarket                                                          |
| price           | string   | Limit Price. Required when order type is limit or stopLimit                                                  |
| volume          | string   | Original Total Volume                                                                                        |
| status          | string   | Order status                                                                                                 |
| post_only       | bool     | Only maker                                                                                                   |
| timestamp       | int64    | millisecond time-stamp                                                                                       |
| filled_size     | string   | The size that has been filled                                                                                |
| unfilled_size    | string   | The size that has not been filled                                                                            |
| avg_filled_price | string   | Average filled price                                                                                         |
| sum_trade_amount | string   | cumulative trading amount(turnover)                                                                          |

**How to Subscribe：**

```json
{
    "type":"sub",
    "parameters":[{
        "topic":"order_rtn",
        "instrument_id":"ETH-USDT"
    }],
    "id": 1
}
```

**Response Example：**

```json
{
    "type":"sub-resp",
    "topic":"order_rtn",
    "data":[{
        "type": "limit",                // Order type
        "sys_order_id":"1550849345000001",     // System order ID
        "client_order_id":"000000001",  // Client order ID
        "instrument_id":"ETH-BTC",      // InstrumentID
        "direction":"buy",              // Trade direction
        "stop_price":"0",               // Stop price
        "price":"1",                    // Limit Price
        "volume":"1",                   // Original Total Volume
        "status": "FILLED",             // Order status
        "post_only": false,             // Only as "maker"
        "timestamp": 1478692862000,     // Order timestamp
        "filled_size": "1",             // The size that has been filled
        "avg_filled_price": "1",        // Average filled price
        "unfilled_size": "0",           // The size that has not been filled
        "sum_trade_amount": "1"         // turnover
    }]
}
```

### 3.3.2 Trade Data

**Request Content:**

| **PARAMETER** | **TYPE** | **REQUIRED** | **DESCRIPTION**            |
| ------------- | -------- | ------------ | -------------------------- |
| type          | string   | true         | "sub"                      |
| topic         | string   | true         | "trade_rtn"                |
| instrument_id | string   | false        | e.g. "ETH-USDT", "ETH-BTC" |

**Response Content:**

| **PARAMETER**   | **TYPE** | **DESCRIPTION**                      |
|-----------------| -------- |--------------------------------------|
| type            | string   | "sub-resp"                           |
| topic           | string   | "trade_rtn"                          |
| trade_id        | string   | Trade Id                             |
| client_order_id | string   | Client Order Id                      |
| sys_order_id    | string   | Server Order Id                      |
| instrument_id   | string   | e.g. "ETH-BTC"                       |
| direction       | string   | "buy" or "sell"                      |
| price           | string   | Price                                |
| volume          | string   | Volume                               |
| fee             | string   | Fee                                  |
| fee_ccy         | string   | Transaction fee currency, e.g. "ETH" |
| timestamp | int64    | Trade millisecond time-stamp         |
| trade_type    | string   | "Common", "Invalid"                  |


**How to Subscribe：**

```json
{
    "type":"sub",
    "parameters":[{
        "topic":"trade_rtn",
        "instrument_id":"ETH-USDT"
    }],
    "id": 1
}
```

**Response Example：**

```json
{
    "type":"sub-resp",
    "topic":"trade_rtn",
    "data":[
        {
            "sys_order_id":"1550849345000001",     // System order ID
            "client_order_id":"000000001",  // Client order ID
            "trade_id":"1",                 // Trade ID
            "instrument_id":"ETH-BTC",      // Instrument ID
            "direction":"buy",              // Trade direction
            "price":"1",                    // Price
            "volume":"1",                   // Volume
            "fee":"0.05",                   // Transaction fee
            "fee_ccy": "BTC",               // Transaction fee currency
            "timestamp": 1478692862000,     // Trade time
            "trade_type": "Common"          // Trade type
        },
        {
            "sys_order_id":"1550849345000002",
            "client_order_id":"000000002",
            "trade_id":"2",
            "instrument_id":"ETH-BTC",
            "direction":"sell",
            "price":"1",
            "volume":"2",
            "fee":"0.05",
            "fee_ccy": "BTC",
            "timestamp":1478692862000,
            "trade_type": "Common"
        }
    ]
}
```

# 4. Error code

## 4.1 Error Code Classification Standard

| **ERROR CODE RANGE** | **DESCRIPTION**    |
| -------------------- | ------------------ |
| 0000~0099            | Public error code      |
| 0100~0199            | WebSocket error code   |

## 4.2 Error Code Example

| **ERROR CODE**   | **TYPE**   | **ERROR MESSAGE**                                  |
|------------------| ---------- |----------------------------------------------------|
| 0000             | Public error code    | Success                                            |
| 0001             | Public error code    | common parameter error                                    |
| 0002             | Public error code    | System error                                       |
| 0003             | Public error code    | Request not allowed                                |
| 0004             | Public error code | Insufficient account funds                         | |
| 0005             | Public error code | Order not found                                    |
| 0006             | Public error code | Instrument not found                               |
| 0007             | Public error code | Invalid volume                               |
| 0008             | Public error code | Not enough volume to cancel                        |
| 0009             | Public error code | Unsatisfied with the price precision               |
| 0010             | Public error code | Invalid price                                      |
| 0011             | Public error code | Operation for the APIKey user not allowed               |
| 0012             | Public error code | Unsupported order type                             |
| 0013             | Public error code | duplicate orders                                   |
| 0100             | WebSocket error code | Subscription failed                                |
| 0101             | WebSocket error code | Partial subscriptions succeeded                    |

