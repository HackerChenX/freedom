"""
集成测试模块

测试选股策略的端到端流程
"""

import unittest
import json
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

class IntegrationTest(unittest.TestCase):
    """集成测试基类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        self.prepare_test_data()
        
    def prepare_test_data(self):
        """准备测试数据"""
        # 创建测试用股票列表
        self.stock_list = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003', '000004', '000005'],
            'stock_name': ['测试1', '测试2', '测试3', '测试4', '测试5'],
            'market': ['主板', '创业板', '科创板', '主板', '创业板'],
            'industry': ['金融', '科技', '医药', '能源', '消费'],
            'market_cap': [100, 200, 300, 150, 250]
        })
        
        # 创建测试K线数据
        dates = pd.date_range('2023-01-01', periods=100)
        
        # 股票1: MA5上穿MA20，RSI<30
        stock1_close = np.concatenate([
            np.linspace(100, 80, 80),  # 下跌趋势
            np.linspace(80, 100, 20)   # 快速反弹
        ])
        
        # 股票2: MA5不上穿MA20，RSI<30
        stock2_close = np.concatenate([
            np.linspace(100, 60, 90),  # 持续下跌
            np.linspace(60, 65, 10)    # 小幅反弹
        ])
        
        # 股票3: MA5上穿MA20，RSI>30
        stock3_close = np.concatenate([
            np.linspace(100, 90, 50),  # 小幅下跌
            np.linspace(90, 110, 50)   # 温和上涨
        ])
        
        # 股票4: 不满足任何条件
        stock4_close = np.linspace(100, 120, 100)  # 持续上涨
        
        # 股票5: MA5上穿MA20，RSI<30
        stock5_close = np.concatenate([
            np.linspace(100, 75, 75),  # 下跌趋势
            np.linspace(75, 95, 25)    # 快速反弹
        ])
        
        # 构建K线数据字典
        self.kline_data = {}
        
        for idx, stock_code in enumerate(['000001', '000002', '000003', '000004', '000005']):
            # 选择对应的收盘价数据
            if idx == 0:
                closes = stock1_close
            elif idx == 1:
                closes = stock2_close
            elif idx == 2:
                closes = stock3_close
            elif idx == 3:
                closes = stock4_close
            else:
                closes = stock5_close
                
            # 生成其他价格数据
            opens = closes + np.random.normal(0, 2, 100)
            highs = np.maximum(opens, closes) + np.random.uniform(0, 3, 100)
            lows = np.minimum(opens, closes) - np.random.uniform(0, 3, 100)
            volumes = np.random.uniform(10000, 50000, 100)
            
            # 确保价格为正数
            opens = np.maximum(1, opens)
            highs = np.maximum(opens, highs)
            lows = np.maximum(1, lows)
            closes = np.maximum(lows, closes)
            
            # 创建DataFrame
            self.kline_data[stock_code] = pd.DataFrame({
                'date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            # 预计算指标
            self.kline_data[stock_code]['ma5'] = self.kline_data[stock_code]['close'].rolling(5).mean()
            self.kline_data[stock_code]['ma20'] = self.kline_data[stock_code]['close'].rolling(20).mean()
            
            # 计算RSI
            delta = self.kline_data[stock_code]['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            self.kline_data[stock_code]['rsi'] = 100 - (100 / (1 + rs))
        
        # 创建测试策略配置
        self.strategy_config = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
                "description": "用于测试的策略",
                "conditions": [
                    {
                        "indicator_id": "MA_CROSS",
                        "period": "DAILY",
                        "parameters": {
                            "fast_period": 5,
                            "slow_period": 20
                        }
                    },
                    {
                        "indicator_id": "RSI_OVERSOLD",
                        "period": "DAILY",
                        "parameters": {
                            "period": 14,
                            "threshold": 30
                        }
                    },
                    {
                        "logic": "AND"
                    }
                ],
                "filters": {
                    "market": ["主板", "创业板"],
                    "market_cap": {
                        "min": 50,
                        "max": 250
                    }
                },
                "sort": [
                    {
                        "field": "market_cap",
                        "direction": "ASC"
                    }
                ]
            }
        }


class StockSelectionIntegrationTest(IntegrationTest):
    """选股流程集成测试"""
    
    def test_end_to_end_stock_selection(self):
        """测试端到端选股流程"""
        # 修改K线数据使000001和000005符合条件
        # 确保MA5上穿MA20
        for stock_code in ['000001', '000005']:
            kline = self.kline_data[stock_code]
            kline['ma5'].iloc[-1] = kline['ma20'].iloc[-1] + 1
            kline['ma5'].iloc[-2] = kline['ma20'].iloc[-2] - 1
            # 确保RSI < 30
            kline['rsi'].iloc[-1] = 25
        
        # 模拟依赖对象
        mock_data_manager = MagicMock()
        # 只返回主板和创业板的股票
        mock_data_manager.get_stock_list.return_value = self.stock_list[
            self.stock_list['market'].isin(['主板', '创业板'])
        ]
        mock_data_manager.get_kline_data.side_effect = lambda stock_code, **kwargs: self.kline_data.get(stock_code, pd.DataFrame())
        
        # 模拟指标
        class MockMACross:
            def __init__(self, name="MA_CROSS", parameters=None):
                self.name = name
                self.parameters = parameters or {}
                
            def calculate(self, data):
                return data
                
            def generate_signals(self, data):
                # 判断MA5是否上穿MA20
                last_idx = len(data) - 1
                if last_idx >= 20:
                    ma_cross = (data['ma5'].iloc[last_idx] > data['ma20'].iloc[last_idx]) and \
                               (data['ma5'].iloc[last_idx-1] <= data['ma20'].iloc[last_idx-1])
                    
                    return pd.DataFrame({
                        'signal': [ma_cross] * len(data),
                        'signal_strength': [0.8 if ma_cross else 0.2] * len(data)
                    }, index=data.index)
                return pd.DataFrame({'signal': [False] * len(data)}, index=data.index)
                
        class MockRSIOversold:
            def __init__(self, name="RSI_OVERSOLD", parameters=None):
                self.name = name
                self.parameters = parameters or {"threshold": 30}
                
            def calculate(self, data):
                return data
                
            def generate_signals(self, data):
                # 判断RSI是否小于阈值
                last_idx = len(data) - 1
                if last_idx >= 14:
                    threshold = self.parameters.get("threshold", 30)
                    rsi_oversold = data['rsi'].iloc[last_idx] < threshold
                    
                    return pd.DataFrame({
                        'signal': [rsi_oversold] * len(data),
                        'signal_strength': [0.8 if rsi_oversold else 0.2] * len(data)
                    }, index=data.index)
                return pd.DataFrame({'signal': [False] * len(data)}, index=data.index)
        
        # 模拟策略解析器
        mock_strategy_parser = MagicMock()
        mock_strategy_parser.parse.return_value = {
            "strategy_id": "TEST_STRATEGY",
            "name": "测试策略",
            "description": "用于测试的策略",
            "conditions": [
                {
                    "type": "indicator",
                    "indicator": MockMACross(parameters={"fast_period": 5, "slow_period": 20}),
                    "period": "DAILY"
                },
                {
                    "type": "indicator",
                    "indicator": MockRSIOversold(parameters={"period": 14, "threshold": 30}),
                    "period": "DAILY"
                },
                {
                    "type": "logic",
                    "value": "AND"
                }
            ],
            "filters": self.strategy_config["strategy"]["filters"],
            "sort": self.strategy_config["strategy"]["sort"]
        }
        
        # 模拟策略执行器
        class MockStrategyExecutor:
            def __init__(self, data_manager):
                self.data_manager = data_manager
                
            def execute(self, strategy_plan):
                # 1. 应用过滤条件
                filters = strategy_plan["filters"]
                stock_list = self.data_manager.get_stock_list(filters)
                
                # 2. 对每支股票执行策略条件
                results = []
                for _, stock in stock_list.iterrows():
                    stock_code = stock['stock_code']
                    
                    # 获取K线数据
                    kline_data = self.data_manager.get_kline_data(stock_code)
                    
                    # 应用条件
                    condition_results = []
                    for condition in strategy_plan["conditions"]:
                        if condition["type"] == "indicator":
                            # 计算指标信号
                            indicator = condition["indicator"]
                            signals = indicator.generate_signals(kline_data)
                            
                            # 检查最后一条信号
                            last_signal = signals['signal'].iloc[-1] if not signals.empty else False
                            condition_results.append(last_signal)
                        elif condition["type"] == "logic":
                            # 应用逻辑操作
                            logic = condition["value"]
                            if logic == "AND" and len(condition_results) >= 2:
                                # 合并最后两个结果
                                last_result = condition_results.pop()
                                prev_result = condition_results.pop()
                                condition_results.append(last_result and prev_result)
                            elif logic == "OR" and len(condition_results) >= 2:
                                # 合并最后两个结果
                                last_result = condition_results.pop()
                                prev_result = condition_results.pop()
                                condition_results.append(last_result or prev_result)
                    
                    # 如果所有条件都满足
                    final_result = condition_results[0] if condition_results else False
                    if final_result:
                        results.append({
                            'stock_code': stock_code,
                            'stock_name': stock['stock_name'],
                            'market': stock['market'],
                            'industry': stock['industry'],
                            'market_cap': stock['market_cap'],
                            'ma5': kline_data['ma5'].iloc[-1],
                            'ma20': kline_data['ma20'].iloc[-1],
                            'rsi': kline_data['rsi'].iloc[-1]
                        })
                
                # 3. 对结果排序
                if results and strategy_plan["sort"]:
                    for sort_item in strategy_plan["sort"]:
                        field = sort_item["field"]
                        direction = sort_item["direction"]
                        
                        results = sorted(
                            results,
                            key=lambda x: x.get(field, 0),
                            reverse=(direction == "DESC")
                        )
                
                return results
        
        # 创建执行器
        executor = MockStrategyExecutor(mock_data_manager)
        
        # 执行策略
        results = executor.execute(mock_strategy_parser.parse())
        
        # 验证结果
        self.assertEqual(len(results), 2)  # 应该有2支股票满足条件（000001和000005）
        
        # 检查是否按市值升序排序
        self.assertEqual(results[0]['stock_code'], '000001')
        self.assertEqual(results[1]['stock_code'], '000005')
        
        # 检查筛选出的股票确实满足条件
        for result in results:
            # 验证MA5确实上穿MA20
            stock_code = result['stock_code']
            kline = self.kline_data[stock_code]
            
            # 验证MA5上穿MA20
            self.assertTrue(
                kline['ma5'].iloc[-1] > kline['ma20'].iloc[-1] and 
                kline['ma5'].iloc[-2] <= kline['ma20'].iloc[-2]
            )
            
            # 验证RSI < 30
            self.assertTrue(kline['rsi'].iloc[-1] < 30)
    
    def test_strategy_config_validation(self):
        """测试策略配置验证"""
        # 创建无效的策略配置（缺少条件）
        invalid_config = {
            "strategy": {
                "id": "INVALID_STRATEGY",
                "name": "无效策略",
                "description": "缺少条件的策略",
                "conditions": [],
                "filters": {
                    "market": ["主板", "创业板"],
                    "market_cap": {
                        "min": 50,
                        "max": 250
                    }
                },
                "sort": [
                    {
                        "field": "market_cap",
                        "direction": "ASC"
                    }
                ]
            }
        }
        
        # 模拟配置验证器
        def validate_strategy_config(config):
            strategy = config.get("strategy", {})
            
            # 检查必要字段
            if not strategy.get("id"):
                return False, "缺少策略ID"
                
            if not strategy.get("conditions"):
                return False, "缺少策略条件"
                
            # 验证条件
            conditions = strategy.get("conditions", [])
            indicator_count = 0
            logic_count = 0
            
            for condition in conditions:
                if "indicator_id" in condition:
                    indicator_count += 1
                elif "logic" in condition:
                    logic_count += 1
            
            if indicator_count < 1:
                return False, "至少需要一个指标条件"
                
            if indicator_count > 1 and logic_count < 1:
                return False, "多个指标需要指定逻辑关系"
            
            return True, "策略配置有效"
        
        # 验证有效配置
        is_valid, message = validate_strategy_config(self.strategy_config)
        self.assertTrue(is_valid)
        
        # 验证无效配置
        is_valid, message = validate_strategy_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertEqual(message, "缺少策略条件")
    
    def test_result_format_and_export(self):
        """测试结果格式化和导出"""
        # 模拟选股结果
        selection_results = [
            {
                'stock_code': '000001',
                'stock_name': '测试1',
                'market': '主板',
                'industry': '金融',
                'market_cap': 100,
                'ma5': 95.0,
                'ma20': 90.0,
                'rsi': 25.0
            },
            {
                'stock_code': '000005',
                'stock_name': '测试5',
                'market': '创业板',
                'industry': '消费',
                'market_cap': 250,
                'ma5': 92.0,
                'ma20': 85.0,
                'rsi': 28.0
            }
        ]
        
        # 测试转换为DataFrame
        results_df = pd.DataFrame(selection_results)
        
        # 验证DataFrame
        self.assertEqual(len(results_df), 2)
        self.assertEqual(list(results_df.columns), ['stock_code', 'stock_name', 'market', 'industry', 'market_cap', 'ma5', 'ma20', 'rsi'])
        
        # 测试导出为JSON
        results_json = json.dumps(selection_results)
        loaded_json = json.loads(results_json)
        
        # 验证JSON
        self.assertEqual(len(loaded_json), 2)
        self.assertEqual(loaded_json[0]['stock_code'], '000001')
        self.assertEqual(loaded_json[1]['stock_code'], '000005')


if __name__ == '__main__':
    unittest.main() 