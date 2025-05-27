"""
性能测试模块

测试策略执行的性能
"""

import unittest
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PerformanceTest(unittest.TestCase):
    """性能测试基类"""
    
    def setUp(self):
        """测试前准备"""
        # 设置性能测试参数
        self.iterations = 10  # 重复执行次数
        self.warmup_iterations = 2  # 预热次数
        
        # 创建性能测试数据
        self.prepare_test_data()
        
    def prepare_test_data(self):
        """准备测试数据"""
        # 创建测试用股票列表（100支股票）
        stock_codes = [f'{i:06d}' for i in range(100)]
        stock_names = [f'测试股票{i}' for i in range(100)]
        markets = ['主板', '创业板', '科创板'] * 34
        industries = ['金融', '科技', '医药', '能源', '消费'] * 20
        market_caps = np.random.uniform(50, 5000, 100)
        
        self.stock_list = pd.DataFrame({
            'stock_code': stock_codes,
            'stock_name': stock_names,
            'market': markets[:100],
            'industry': industries[:100],
            'market_cap': market_caps
        })
        
        # 创建测试K线数据（一年的日线数据）
        self.kline_data = {}
        dates = pd.date_range('2023-01-01', periods=252)
        
        for stock_code in stock_codes:
            # 创建随机价格数据
            closes = np.cumsum(np.random.normal(0, 1, 252)) + 100
            opens = closes + np.random.normal(0, 2, 252)
            highs = np.maximum(opens, closes) + np.random.uniform(0, 3, 252)
            lows = np.minimum(opens, closes) - np.random.uniform(0, 3, 252)
            volumes = np.random.uniform(10000, 50000, 252)
            
            # 确保价格为正数
            opens = np.maximum(1, opens)
            highs = np.maximum(opens, highs)
            lows = np.maximum(1, lows)
            closes = np.maximum(lows, closes)
            
            self.kline_data[stock_code] = pd.DataFrame({
                'date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
        
    def run_benchmark(self, test_func, *args, **kwargs):
        """运行性能基准测试"""
        # 预热
        for _ in range(self.warmup_iterations):
            test_func(*args, **kwargs)
        
        # 测量执行时间
        start_time = time.time()
        
        for _ in range(self.iterations):
            test_func(*args, **kwargs)
            
        end_time = time.time()
        
        # 计算平均执行时间
        avg_time = (end_time - start_time) / self.iterations
        
        return avg_time


class StrategyPerformanceTest(PerformanceTest):
    """策略性能测试类"""
    
    def test_strategy_execution_performance(self):
        """测试策略执行性能"""
        # 定义测试函数
        def execute_strategy():
            """模拟策略执行过程"""
            # 1. 过滤股票列表
            filtered_stocks = self.stock_list[
                (self.stock_list['market'].isin(['主板', '科创板'])) &
                (self.stock_list['market_cap'] >= 100) &
                (self.stock_list['market_cap'] <= 2000)
            ]
            
            # 2. 对每支股票执行指标计算
            results = []
            for idx, stock in filtered_stocks.iterrows():
                stock_code = stock['stock_code']
                kline = self.kline_data[stock_code]
                
                # 计算MA5和MA20
                kline['ma5'] = kline['close'].rolling(5).mean()
                kline['ma20'] = kline['close'].rolling(20).mean()
                
                # 计算RSI
                delta = kline['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, 0.001)
                kline['rsi'] = 100 - (100 / (1 + rs))
                
                # 判断是否满足条件
                last_idx = len(kline) - 1
                if last_idx >= 20:  # 确保有足够的数据
                    # 条件1: MA5上穿MA20
                    ma_cross = (kline['ma5'].iloc[last_idx] > kline['ma20'].iloc[last_idx]) and \
                               (kline['ma5'].iloc[last_idx-1] <= kline['ma20'].iloc[last_idx-1])
                    
                    # 条件2: RSI小于30（超卖）
                    rsi_oversold = kline['rsi'].iloc[last_idx] < 30
                    
                    # 组合条件
                    if ma_cross and rsi_oversold:
                        results.append({
                            'stock_code': stock_code,
                            'stock_name': stock['stock_name'],
                            'market': stock['market'],
                            'industry': stock['industry'],
                            'market_cap': stock['market_cap'],
                            'ma5': kline['ma5'].iloc[last_idx],
                            'ma20': kline['ma20'].iloc[last_idx],
                            'rsi': kline['rsi'].iloc[last_idx]
                        })
            
            # 3. 排序结果
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('market_cap')
            
            return results
        
        # 运行性能测试
        avg_time = self.run_benchmark(execute_strategy)
        
        print(f"\n策略执行平均耗时: {avg_time:.4f}秒")
        
        # 验证性能满足要求
        self.assertLess(avg_time, 1.0, "策略执行时间应小于1秒")
    
    def test_data_filtering_performance(self):
        """测试数据过滤性能"""
        # 定义测试函数
        def filter_stocks():
            """模拟股票列表过滤过程"""
            return self.stock_list[
                (self.stock_list['market'].isin(['主板', '科创板'])) &
                (self.stock_list['market_cap'] >= 100) &
                (self.stock_list['market_cap'] <= 2000)
            ]
        
        # 运行性能测试
        avg_time = self.run_benchmark(filter_stocks)
        
        print(f"\n数据过滤平均耗时: {avg_time:.4f}秒")
        
        # 验证性能满足要求
        self.assertLess(avg_time, 0.01, "数据过滤时间应小于0.01秒")
    
    def test_indicator_calculation_performance(self):
        """测试指标计算性能"""
        # 获取一支样本股票的K线数据
        sample_kline = self.kline_data['000000']
        
        # 定义测试函数
        def calculate_indicators():
            """模拟指标计算过程"""
            kline = sample_kline.copy()
            
            # 计算MA系列
            kline['ma5'] = kline['close'].rolling(5).mean()
            kline['ma10'] = kline['close'].rolling(10).mean()
            kline['ma20'] = kline['close'].rolling(20).mean()
            kline['ma30'] = kline['close'].rolling(30).mean()
            kline['ma60'] = kline['close'].rolling(60).mean()
            
            # 计算EMA系列
            kline['ema12'] = kline['close'].ewm(span=12).mean()
            kline['ema26'] = kline['close'].ewm(span=26).mean()
            
            # 计算MACD
            kline['macd_dif'] = kline['ema12'] - kline['ema26']
            kline['macd_dea'] = kline['macd_dif'].ewm(span=9).mean()
            kline['macd_hist'] = 2 * (kline['macd_dif'] - kline['macd_dea'])
            
            # 计算KDJ
            low_min = kline['low'].rolling(window=9).min()
            high_max = kline['high'].rolling(window=9).max()
            
            kline['rsv'] = 100 * ((kline['close'] - low_min) / (high_max - low_min)).replace([np.inf, -np.inf], np.nan).fillna(0)
            kline['k'] = kline['rsv'].ewm(alpha=1/3, adjust=False).mean()
            kline['d'] = kline['k'].ewm(alpha=1/3, adjust=False).mean()
            kline['j'] = 3 * kline['k'] - 2 * kline['d']
            
            # 计算RSI
            delta = kline['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            
            rs = avg_gain / avg_loss.replace(0, 0.001)
            kline['rsi'] = 100 - (100 / (1 + rs))
            
            return kline
        
        # 运行性能测试
        avg_time = self.run_benchmark(calculate_indicators)
        
        print(f"\n指标计算平均耗时: {avg_time:.4f}秒")
        
        # 验证性能满足要求
        self.assertLess(avg_time, 0.05, "指标计算时间应小于0.05秒")


if __name__ == '__main__':
    unittest.main() 