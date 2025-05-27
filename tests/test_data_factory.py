#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
测试数据工厂模块

提供标准化的测试数据生成方法，用于各类测试场景
"""

import os
import json
import datetime
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

# 设置随机种子，确保测试数据可重现
np.random.seed(42)
random.seed(42)


class TestDataFactory:
    """测试数据工厂类，提供标准化的测试数据创建方法"""

    # 测试数据目录
    TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'test_data')
    
    # 确保测试数据目录存在
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    @classmethod
    def create_stock_list(cls, count: int = 100, 
                          include_indices: bool = True, 
                          seed: int = 42) -> pd.DataFrame:
        """
        创建模拟股票列表
        
        Args:
            count: 股票数量
            include_indices: 是否包含指数
            seed: 随机种子，用于生成可重现的数据
            
        Returns:
            包含股票代码、名称等信息的DataFrame
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # 股票代码前缀
        sh_prefix = "60"  # 上证
        sz_prefix = "00"  # 深证
        cyb_prefix = "30"  # 创业板
        
        stock_data = []
        
        # 生成上证股票
        sh_count = count // 3
        for i in range(sh_count):
            code = f"{sh_prefix}{random.randint(1000, 9999)}"
            name = f"测试股票{i+1}"
            stock_data.append({
                "code": code,
                "name": name,
                "market": "SH",
                "is_index": False,
                "industry": random.choice(["金融", "科技", "医药", "消费", "制造"]),
                "list_date": cls._random_date(2000, 2022)
            })
        
        # 生成深证股票
        sz_count = count // 3
        for i in range(sz_count):
            code = f"{sz_prefix}{random.randint(1000, 9999)}"
            name = f"测试股票{sh_count+i+1}"
            stock_data.append({
                "code": code,
                "name": name,
                "market": "SZ",
                "is_index": False,
                "industry": random.choice(["金融", "科技", "医药", "消费", "制造"]),
                "list_date": cls._random_date(2000, 2022)
            })
        
        # 生成创业板股票
        cyb_count = count - sh_count - sz_count
        for i in range(cyb_count):
            code = f"{cyb_prefix}{random.randint(1000, 9999)}"
            name = f"测试股票{sh_count+sz_count+i+1}"
            stock_data.append({
                "code": code,
                "name": name,
                "market": "SZ",
                "is_index": False,
                "industry": random.choice(["金融", "科技", "医药", "消费", "制造"]),
                "list_date": cls._random_date(2000, 2022)
            })
        
        # 添加指数
        if include_indices:
            indices = [
                {"code": "000001", "name": "上证指数", "market": "SH", "is_index": True, "industry": "指数", "list_date": "1990-12-19"},
                {"code": "399001", "name": "深证成指", "market": "SZ", "is_index": True, "industry": "指数", "list_date": "1991-04-03"},
                {"code": "399006", "name": "创业板指", "market": "SZ", "is_index": True, "industry": "指数", "list_date": "2010-06-01"},
                {"code": "000016", "name": "上证50", "market": "SH", "is_index": True, "industry": "指数", "list_date": "2004-01-02"},
                {"code": "000300", "name": "沪深300", "market": "SH", "is_index": True, "industry": "指数", "list_date": "2005-04-08"},
                {"code": "000905", "name": "中证500", "market": "SH", "is_index": True, "industry": "指数", "list_date": "2007-01-15"}
            ]
            stock_data.extend(indices)
        
        return pd.DataFrame(stock_data)
    
    @classmethod
    def create_kline_data(cls, 
                         stock_code: str, 
                         start_date: str = "2020-01-01", 
                         end_date: str = "2022-12-31", 
                         seed: int = 42) -> pd.DataFrame:
        """
        创建模拟K线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            seed: 随机种子，用于生成可重现的数据
            
        Returns:
            包含OHLCV等信息的DataFrame
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # 生成日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B'表示工作日
        
        # 生成初始价格（50-500之间的随机数）
        init_price = random.uniform(50, 500)
        
        # 创建价格序列，使用随机游走模型
        # 价格变化率在 -2% 到 2% 之间
        returns = np.random.normal(0.0005, 0.015, size=len(date_range))
        prices = [init_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # 去掉初始价格
        
        # 生成K线数据
        kline_data = []
        
        for i, date in enumerate(date_range):
            close_price = prices[i]
            
            # 生成开盘价（前一日收盘价基础上波动）
            if i == 0:
                open_price = close_price * (1 + random.uniform(-0.01, 0.01))
            else:
                open_price = prices[i-1] * (1 + random.uniform(-0.01, 0.01))
            
            # 生成最高价和最低价
            price_range = close_price * random.uniform(0.02, 0.05)  # 当日价格波动范围
            high_price = max(close_price, open_price) + random.uniform(0, price_range)
            low_price = min(close_price, open_price) - random.uniform(0, price_range)
            
            # 生成成交量（均值在当日股价的10000倍左右）
            volume = int(close_price * random.uniform(8000, 12000))
            
            # 生成成交额
            amount = volume * close_price
            
            kline_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "code": stock_code,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
                "amount": round(amount, 2),
                "turnover": round(random.uniform(1, 8), 2)  # 换手率
            })
        
        return pd.DataFrame(kline_data)
    
    @classmethod
    def create_indicator_data(cls, kline_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        基于K线数据创建各类指标数据
        
        Args:
            kline_data: K线数据DataFrame
            
        Returns:
            包含各类指标的字典，键为指标名称，值为指标数据DataFrame
        """
        indicators = {}
        
        # 确保数据按日期排序
        kline_data = kline_data.sort_values("date")
        
        # 计算简单移动平均线
        for period in [5, 10, 20, 30, 60]:
            ma_name = f"MA{period}"
            indicators[ma_name] = cls._calculate_ma(kline_data, period)
        
        # 计算MACD
        indicators["MACD"] = cls._calculate_macd(kline_data)
        
        # 计算KDJ
        indicators["KDJ"] = cls._calculate_kdj(kline_data)
        
        # 计算RSI
        for period in [6, 12, 24]:
            rsi_name = f"RSI{period}"
            indicators[rsi_name] = cls._calculate_rsi(kline_data, period)
        
        return indicators
    
    @classmethod
    def create_strategy_config(cls, 
                              strategy_name: str, 
                              complexity: str = "medium") -> Dict[str, Any]:
        """
        创建策略配置
        
        Args:
            strategy_name: 策略名称
            complexity: 策略复杂度，可选值为："simple", "medium", "complex"
            
        Returns:
            策略配置字典
        """
        if complexity == "simple":
            # 简单策略，单一条件
            config = {
                "strategy": {
                    "name": strategy_name,
                    "description": f"简单测试策略：{strategy_name}",
                    "conditions": [
                        {
                            "indicator_id": "MA_CROSS",
                            "params": {
                                "fast_period": 5,
                                "slow_period": 20
                            }
                        }
                    ]
                }
            }
        elif complexity == "medium":
            # 中等复杂度策略，多个条件组合
            config = {
                "strategy": {
                    "name": strategy_name,
                    "description": f"中等复杂度测试策略：{strategy_name}",
                    "conditions": [
                        {
                            "indicator_id": "MA_CROSS",
                            "params": {
                                "fast_period": 5,
                                "slow_period": 20
                            }
                        },
                        {
                            "indicator_id": "RSI",
                            "params": {
                                "period": 14,
                                "lower_bound": 30,
                                "upper_bound": 70
                            }
                        },
                        {
                            "logic": "AND"
                        }
                    ]
                }
            }
        else:  # complex
            # 复杂策略，嵌套条件
            config = {
                "strategy": {
                    "name": strategy_name,
                    "description": f"复杂测试策略：{strategy_name}",
                    "conditions": [
                        {
                            "indicator_id": "MA_CROSS",
                            "params": {
                                "fast_period": 5,
                                "slow_period": 20
                            }
                        },
                        {
                            "conditions": [
                                {
                                    "indicator_id": "RSI",
                                    "params": {
                                        "period": 14,
                                        "lower_bound": 30,
                                        "upper_bound": 70
                                    }
                                },
                                {
                                    "indicator_id": "MACD",
                                    "params": {
                                        "fast_period": 12,
                                        "slow_period": 26,
                                        "signal_period": 9
                                    }
                                },
                                {
                                    "logic": "OR"
                                }
                            ]
                        },
                        {
                            "logic": "AND"
                        }
                    ]
                }
            }
        
        return config
    
    @classmethod
    def save_test_data(cls, data: Any, filename: str) -> str:
        """
        保存测试数据到文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        file_path = os.path.join(cls.TEST_DATA_DIR, filename)
        
        # 判断数据类型并使用合适的方式保存
        if isinstance(data, pd.DataFrame):
            if filename.endswith(".csv"):
                data.to_csv(file_path, index=False)
            elif filename.endswith(".parquet"):
                data.to_parquet(file_path, index=False)
            else:
                # 默认保存为CSV
                file_path = file_path + ".csv"
                data.to_csv(file_path, index=False)
        elif isinstance(data, dict):
            if filename.endswith(".json"):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                # 默认保存为JSON
                file_path = file_path + ".json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持保存类型为 {type(data)} 的数据")
        
        return file_path
    
    @classmethod
    def load_test_data(cls, filename: str) -> Any:
        """
        从文件加载测试数据
        
        Args:
            filename: 文件名
            
        Returns:
            加载的数据
        """
        file_path = os.path.join(cls.TEST_DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"测试数据文件不存在: {file_path}")
        
        if filename.endswith(".csv"):
            return pd.read_csv(file_path)
        elif filename.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
    
    @classmethod
    def generate_standard_test_dataset(cls) -> None:
        """
        生成标准测试数据集
        
        创建一组标准化的测试数据，用于各类测试场景
        """
        # 1. 创建股票列表
        stock_list = cls.create_stock_list(count=100, include_indices=True)
        cls.save_test_data(stock_list, "stock_list.csv")
        
        # 2. 为部分股票创建K线数据
        # 选择10只股票和3个指数
        selected_stocks = stock_list[~stock_list["is_index"]].sample(10)
        selected_indices = stock_list[stock_list["is_index"]].sample(3)
        selected = pd.concat([selected_stocks, selected_indices])
        
        for idx, row in selected.iterrows():
            code = row["code"]
            kline_data = cls.create_kline_data(code)
            cls.save_test_data(kline_data, f"kline_{code}.csv")
            
            # 为第一只股票创建指标数据
            if idx == selected.index[0]:
                indicators = cls.create_indicator_data(kline_data)
                for name, data in indicators.items():
                    cls.save_test_data(data, f"indicator_{name}_{code}.csv")
        
        # 3. 创建不同复杂度的策略配置
        for complexity in ["simple", "medium", "complex"]:
            strategy_config = cls.create_strategy_config(f"测试策略_{complexity}", complexity)
            cls.save_test_data(strategy_config, f"strategy_config_{complexity}.json")
    
    # 辅助方法
    
    @staticmethod
    def _random_date(start_year: int, end_year: int) -> str:
        """生成指定年份范围内的随机日期"""
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # 简化处理，避免月份天数问题
        return f"{year}-{month:02d}-{day:02d}"
    
    @staticmethod
    def _calculate_ma(data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算简单移动平均线"""
        result = data.copy()
        result[f'MA{period}'] = result['close'].rolling(window=period).mean().round(2)
        return result[[f'MA{period}', 'date', 'code']]
    
    @staticmethod
    def _calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        result = data.copy()
        # 计算12日EMA
        result['EMA12'] = result['close'].ewm(span=12, adjust=False).mean()
        # 计算26日EMA
        result['EMA26'] = result['close'].ewm(span=26, adjust=False).mean()
        # 计算DIF
        result['DIF'] = result['EMA12'] - result['EMA26']
        # 计算DEA (9日EMA平滑后的DIF)
        result['DEA'] = result['DIF'].ewm(span=9, adjust=False).mean()
        # 计算MACD柱状图 (DIF-DEA)*2
        result['MACD'] = (result['DIF'] - result['DEA']) * 2
        
        # 四舍五入到2位小数
        for col in ['DIF', 'DEA', 'MACD']:
            result[col] = result[col].round(2)
        
        return result[['DIF', 'DEA', 'MACD', 'date', 'code']]
    
    @staticmethod
    def _calculate_kdj(data: pd.DataFrame, n: int = 9) -> pd.DataFrame:
        """计算KDJ指标"""
        result = data.copy()
        
        # 计算N日内的最低价和最高价
        low_n = result['low'].rolling(window=n).min()
        high_n = result['high'].rolling(window=n).max()
        
        # 计算RSV
        rsv = (result['close'] - low_n) / (high_n - low_n) * 100
        
        # 初始化K、D值
        k = np.zeros(len(result))
        d = np.zeros(len(result))
        
        # 计算K值
        k[0] = 50  # 第一天K值默认为50
        for i in range(1, len(result)):
            k[i] = 2/3 * k[i-1] + 1/3 * rsv[i]
        
        # 计算D值
        d[0] = 50  # 第一天D值默认为50
        for i in range(1, len(result)):
            d[i] = 2/3 * d[i-1] + 1/3 * k[i]
        
        # 计算J值
        j = 3 * k - 2 * d
        
        # 添加到结果
        result['K'] = k.round(2)
        result['D'] = d.round(2)
        result['J'] = j.round(2)
        
        return result[['K', 'D', 'J', 'date', 'code']]
    
    @staticmethod
    def _calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        result = data.copy()
        
        # 计算价格变化
        delta = result['close'].diff()
        
        # 区分上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和平均下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 避免除以零
        avg_loss = avg_loss.replace(0, 0.001)
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 添加到结果
        result[f'RSI{period}'] = rsi.round(2)
        
        return result[[f'RSI{period}', 'date', 'code']]


if __name__ == "__main__":
    # 生成标准测试数据集
    TestDataFactory.generate_standard_test_dataset()
    print(f"标准测试数据集已生成到: {TestDataFactory.TEST_DATA_DIR}") 