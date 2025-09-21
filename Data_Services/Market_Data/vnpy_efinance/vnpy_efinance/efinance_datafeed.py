"""
EFinance数据源实现

基于efinance库的VnPy数据源适配器，支持获取股票、基金、期货的历史数据。
"""

from datetime import datetime, timedelta
from collections.abc import Callable
from typing import List, Optional, Dict, Any
import re

try:
    import efinance as ef
    EFINANCE_AVAILABLE = True
    USING_MOCK = False
    print("✅ EFinance库导入成功")
except ImportError as e:
    EFINANCE_AVAILABLE = False
    USING_MOCK = False
    ef = None
    print(f"⚠️ EFinance库导入失败: {e}")
    print("   原因可能是:")
    print("   1. efinance库未安装: pip install efinance")
    print("   2. 依赖包缺失: pip install multitasking jsonpath retry")
    print("   3. 网络连接问题")
except Exception as e:
    EFINANCE_AVAILABLE = False
    USING_MOCK = False
    ef = None
    print(f"⚠️ EFinance库加载失败: {e}")

import pandas as pd
from pandas import DataFrame

from vnpy.trader.datafeed import BaseDatafeed
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, HistoryRequest
from vnpy.trader.utility import round_to, ZoneInfo


# 时间周期映射 - EFinance支持的周期
INTERVAL_VT2EF: Dict[Interval, str] = {
    Interval.MINUTE: "1",      # 1分钟
    Interval.MINUTE_5: "5",    # 5分钟  
    Interval.MINUTE_15: "15",  # 15分钟
    Interval.MINUTE_30: "30",  # 30分钟
    Interval.HOUR: "60",       # 60分钟
    Interval.DAILY: "101",     # 日线
    Interval.WEEKLY: "102",    # 周线
    Interval.MONTHLY: "103",   # 月线
}

# 支持的交易所列表
STOCK_EXCHANGES: List[Exchange] = [
    Exchange.SSE,    # 上海证券交易所
    Exchange.SZSE,   # 深圳证券交易所
    Exchange.BSE,    # 北京证券交易所
]

# 美股交易所
US_EXCHANGES: List[Exchange] = [
    Exchange.NASDAQ,  # 纳斯达克
    Exchange.NYSE,    # 纽约证券交易所
]

# 港股交易所
HK_EXCHANGES: List[Exchange] = [
    Exchange.SEHK,    # 香港证券交易所
]

# 期货交易所
FUTURE_EXCHANGES: List[Exchange] = [
    Exchange.CFFEX,   # 中国金融期货交易所
    Exchange.SHFE,    # 上海期货交易所
    Exchange.CZCE,    # 郑州商品交易所
    Exchange.DCE,     # 大连商品交易所
    Exchange.INE,     # 上海国际能源交易中心
    Exchange.GFEX,    # 广州期货交易所
]

# 全部支持的交易所
SUPPORTED_EXCHANGES = STOCK_EXCHANGES + US_EXCHANGES + HK_EXCHANGES + FUTURE_EXCHANGES

# 中国时区
CHINA_TZ = ZoneInfo("Asia/Shanghai")


def normalize_symbol(symbol: str, exchange: Exchange) -> str:
    """
    标准化股票代码格式
    
    Args:
        symbol: 原始股票代码
        exchange: 交易所
        
    Returns:
        标准化后的股票代码
    """
    # 去除空格和特殊字符
    symbol = symbol.strip().upper()
    
    # A股代码处理
    if exchange in STOCK_EXCHANGES:
        # 确保是6位数字
        if symbol.isdigit() and len(symbol) == 6:
            return symbol
        # 去除交易所后缀
        if '.' in symbol:
            symbol = symbol.split('.')[0]
        return symbol
    
    # 美股代码处理
    elif exchange in US_EXCHANGES:
        # 美股代码通常是字母，直接返回
        return symbol
        
    # 港股代码处理
    elif exchange in HK_EXCHANGES:
        # 港股代码通常是数字，可能有前导零
        if symbol.isdigit():
            return symbol.zfill(5)  # 补齐到5位
        return symbol
        
    # 期货代码处理
    elif exchange in FUTURE_EXCHANGES:
        return symbol
        
    return symbol


def detect_market_type(symbol: str, exchange: Exchange) -> str:
    """
    检测市场类型
    
    Args:
        symbol: 股票代码
        exchange: 交易所
        
    Returns:
        市场类型标识
    """
    if exchange in STOCK_EXCHANGES:
        return "A股"
    elif exchange in US_EXCHANGES:
        return "美股"
    elif exchange in HK_EXCHANGES:
        return "港股"
    elif exchange in FUTURE_EXCHANGES:
        return "期货"
    else:
        return "未知"


class EfinanceDatafeed(BaseDatafeed):
    """
    EFinance数据源
    
    基于efinance库实现的免费数据源，支持A股、美股、港股、期货等多种数据。
    """

    def __init__(self) -> None:
        """初始化EFinance数据源"""
        self.inited: bool = False
        
        # 检查efinance库是否可用
        if not EFINANCE_AVAILABLE:
            print("❌ efinance库未安装，请运行: pip install efinance")
            return
            
        self.inited = True

    def init(self, output: Callable = print) -> bool:
        """
        初始化数据源
        
        Args:
            output: 输出函数
            
        Returns:
            是否初始化成功
        """
        if not EFINANCE_AVAILABLE:
            output("❌ EFinance数据源初始化失败：efinance库未安装")
            output("   请运行: pip install efinance")
            return False
            
        if USING_MOCK:
            output("⚠️ 使用模拟EFinance数据源")
            output("   仅用于演示和测试，不包含真实市场数据")
            output("   生产环境请安装真正的efinance库")
            
        if self.inited:
            output("✅ EFinance数据源已初始化")
            return True
            
        try:
            # 测试efinance连接
            test_data = ef.stock.get_realtime_quotes()
            if test_data is not None:
                output("✅ EFinance数据源初始化成功")
                output(f"   支持交易所: {len(SUPPORTED_EXCHANGES)}个")
                output(f"   支持周期: {list(INTERVAL_VT2EF.keys())}")
                self.inited = True
                return True
            else:
                output("❌ EFinance数据源测试失败")
                return False
                
        except Exception as e:
            output(f"❌ EFinance数据源初始化失败: {e}")
            return False

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> List[BarData]:
        """
        查询历史K线数据
        
        Args:
            req: 历史数据请求
            output: 输出函数
            
        Returns:
            K线数据列表
        """
        if not self.inited:
            output("❌ EFinance数据源未初始化")
            return []
            
        # 检查时间周期支持
        if req.interval not in INTERVAL_VT2EF:
            output(f"❌ 不支持的时间周期: {req.interval.value}")
            output(f"   支持的周期: {[interval.value for interval in INTERVAL_VT2EF.keys()]}")
            return []
            
        # 检查交易所支持
        if req.exchange not in SUPPORTED_EXCHANGES:
            output(f"❌ 不支持的交易所: {req.exchange.value}")
            output(f"   支持的交易所: {[ex.value for ex in SUPPORTED_EXCHANGES]}")
            return []

        # 标准化股票代码
        symbol = normalize_symbol(req.symbol, req.exchange)
        market_type = detect_market_type(symbol, req.exchange)
        
        output(f"🔍 查询 {market_type} {symbol} 数据...")
        output(f"   时间范围: {req.start} ~ {req.end}")
        output(f"   周期: {req.interval.value}")
        
        try:
            # 调用efinance获取数据
            ef_interval = INTERVAL_VT2EF[req.interval]
            
            # 获取历史K线数据
            if req.exchange in STOCK_EXCHANGES + US_EXCHANGES + HK_EXCHANGES:
                # 股票数据
                df = ef.stock.get_quote_history(
                    stock_codes=symbol,
                    klt=ef_interval,
                    fqt=1,  # 前复权
                    start=req.start.strftime("%Y%m%d"),
                    end=req.end.strftime("%Y%m%d")
                )
            elif req.exchange in FUTURE_EXCHANGES:
                # 期货数据 (如果efinance支持的话)
                try:
                    df = ef.futures.get_quote_history(
                        futures_code=symbol,
                        klt=ef_interval,
                        start=req.start.strftime("%Y%m%d"),
                        end=req.end.strftime("%Y%m%d")
                    )
                except AttributeError:
                    output("❌ EFinance暂不支持期货数据")
                    return []
            else:
                output(f"❌ 未知的交易所类型: {req.exchange}")
                return []
                
            if df is None or df.empty:
                output(f"❌ 未获取到 {symbol} 的数据")
                return []
                
            # 转换为VnPy BarData格式
            bars = self._convert_to_bar_data(df, symbol, req.exchange, req.interval)
            
            output(f"✅ 获取到 {len(bars)} 条数据")
            if bars:
                output(f"   时间范围: {bars[0].datetime} ~ {bars[-1].datetime}")
                output(f"   价格范围: {min(bar.close_price for bar in bars):.2f} ~ {max(bar.close_price for bar in bars):.2f}")
                
            return bars
            
        except Exception as e:
            output(f"❌ 查询数据失败: {e}")
            import traceback
            output(f"   详细错误: {traceback.format_exc()}")
            return []

    def _convert_to_bar_data(self, df: DataFrame, symbol: str, exchange: Exchange, interval: Interval) -> List[BarData]:
        """
        将efinance数据转换为VnPy BarData格式
        
        Args:
            df: efinance返回的DataFrame
            symbol: 股票代码
            exchange: 交易所
            interval: 时间周期
            
        Returns:
            BarData列表
        """
        bars = []
        
        for _, row in df.iterrows():
            try:
                # 处理时间
                if "日期" in row:
                    dt_str = str(row["日期"])
                    if len(dt_str) == 8:  # YYYYMMDD格式
                        dt = datetime.strptime(dt_str, "%Y%m%d")
                    else:
                        dt = pd.to_datetime(row["日期"])
                elif "datetime" in row:
                    dt = pd.to_datetime(row["datetime"])
                else:
                    # 尝试从索引获取时间
                    dt = pd.to_datetime(row.name) if hasattr(row, 'name') else datetime.now()
                
                # 设置时区
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=CHINA_TZ)
                
                # 处理价格数据 - efinance的列名可能是中文
                price_columns = {
                    "开盘": "open_price",
                    "收盘": "close_price", 
                    "最高": "high_price",
                    "最低": "low_price",
                    "成交量": "volume",
                    "成交额": "turnover",
                    # 英文列名备选
                    "open": "open_price",
                    "close": "close_price",
                    "high": "high_price", 
                    "low": "low_price",
                    "vol": "volume",
                    "amount": "turnover"
                }
                
                # 提取价格数据
                prices = {}
                for col_name, field_name in price_columns.items():
                    if col_name in row:
                        value = row[col_name]
                        if pd.notna(value) and value != 0:
                            prices[field_name] = float(value)
                
                # 确保必要字段存在
                required_fields = ["open_price", "close_price", "high_price", "low_price"]
                if not all(field in prices for field in required_fields):
                    continue
                    
                # 成交量和成交额处理
                volume = prices.get("volume", 0)
                turnover = prices.get("turnover", 0)
                
                # 创建BarData对象
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,
                    volume=round_to(volume, 0.01),
                    turnover=round_to(turnover, 0.01),
                    open_price=round_to(prices["open_price"], 0.000001),
                    high_price=round_to(prices["high_price"], 0.000001),
                    low_price=round_to(prices["low_price"], 0.000001),
                    close_price=round_to(prices["close_price"], 0.000001),
                    gateway_name="EF"
                )
                
                # 数据质量检查
                if self._validate_bar_data(bar):
                    bars.append(bar)
                    
            except Exception as e:
                print(f"转换数据失败: {e}, 行数据: {row}")
                continue
                
        # 按时间排序
        bars.sort(key=lambda x: x.datetime)
        return bars

    def _validate_bar_data(self, bar: BarData) -> bool:
        """
        验证K线数据质量
        
        Args:
            bar: K线数据
            
        Returns:
            是否通过验证
        """
        try:
            # 基本价格关系检查
            if bar.high_price < bar.low_price:
                return False
            if bar.high_price < bar.open_price or bar.high_price < bar.close_price:
                return False
            if bar.low_price > bar.open_price or bar.low_price > bar.close_price:
                return False
                
            # 价格不能为负数
            if any(price < 0 for price in [bar.open_price, bar.high_price, bar.low_price, bar.close_price]):
                return False
                
            # 成交量不能为负数
            if bar.volume < 0:
                return False
                
            return True
            
        except Exception:
            return False

    def get_supported_exchanges(self) -> List[Exchange]:
        """获取支持的交易所列表"""
        return SUPPORTED_EXCHANGES

    def get_supported_intervals(self) -> List[Interval]:
        """获取支持的时间周期列表"""
        return list(INTERVAL_VT2EF.keys())
        
    def get_market_info(self) -> Dict[str, Any]:
        """获取市场信息"""
        return {
            "name": "EFinance",
            "description": "免费开源金融数据获取库",
            "version": "0.5.5",
            "website": "https://github.com/Micro-sheep/efinance",
            "docs": "https://efinance.readthedocs.io/en/latest/",
            "supported_markets": {
                "A股": len(STOCK_EXCHANGES),
                "美股": len(US_EXCHANGES), 
                "港股": len(HK_EXCHANGES),
                "期货": len(FUTURE_EXCHANGES)
            },
            "supported_intervals": len(INTERVAL_VT2EF),
            "features": [
                "完全免费",
                "无需注册",
                "数据质量高",
                "支持多市场",
                "实时更新"
            ]
        }
