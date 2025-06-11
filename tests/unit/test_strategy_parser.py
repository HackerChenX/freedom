"""
策略解析器单元测试模块

用于测试策略解析器的功能和异常处理
"""

import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from strategy.strategy_parser import StrategyParser
from indicators.factory import IndicatorFactory
from enums.period import Period
from utils.exceptions import (
    StrategyParseError, 
    StrategyValidationError, 
    ConfigFileError,
    IndicatorNotFoundError
)


class TestStrategyParser(unittest.TestCase):
    """策略解析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.parser = StrategyParser()
        
        # 创建测试用的有效策略配置
        self.valid_strategy = {
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
                    "market": ["主板", "科创板"],
                    "market_cap": {
                        "min": 50,
                        "max": 2000
                    }
                },
                "sort": [
                    {
                        "field": "signal_strength",
                        "direction": "DESC"
                    }
                ]
            }
        }
        
        # 模拟指标工厂
        self.mock_indicator1 = MagicMock()
        self.mock_indicator2 = MagicMock()
        
        # 创建临时文件和目录
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件和目录
        self.temp_dir.cleanup()
    
    def test_parse_strategy_valid(self):
        """测试解析有效的策略配置"""
        # 模拟指标创建
        with patch.object(IndicatorFactory, 'create') as mock_create:
            mock_create.side_effect = [self.mock_indicator1, self.mock_indicator2]
            
            # 解析策略
            result = self.parser.parse_strategy(self.valid_strategy)
            
            # 验证结果
            self.assertEqual(result["strategy_id"], "TEST_STRATEGY")
            self.assertEqual(result["name"], "测试策略")
            self.assertEqual(len(result["conditions"]), 3)
            self.assertEqual(result["conditions"][0]["indicator_id"], "MA_CROSS")
            self.assertEqual(result["conditions"][0]["period"], "日线")
            self.assertEqual(result["conditions"][1]["indicator_id"], "RSI_OVERSOLD")
            self.assertEqual(result["conditions"][2]["type"], "logic")
            self.assertEqual(result["conditions"][2]["value"], "AND")
            
            # 验证过滤器和排序
            self.assertEqual(result["filters"]["market"], ["主板", "科创板"])
            self.assertEqual(result["filters"]["market_cap"]["min"], 50)
            self.assertEqual(result["sort"][0]["field"], "signal_strength")
            self.assertEqual(result["sort"][0]["direction"], "DESC")
            
            # 验证指标工厂调用
            mock_create.assert_any_call("MA_CROSS", fast_period=5, slow_period=20)
            mock_create.assert_any_call("RSI_OVERSOLD", period=14, threshold=30)
    
    def test_parse_strategy_missing_strategy_node(self):
        """测试解析缺少strategy节点的配置"""
        invalid_config = {"not_strategy": {}}
        
        with self.assertRaises(StrategyValidationError) as context:
            self.parser.parse_strategy(invalid_config)
        
        self.assertIn("缺少'strategy'节点", str(context.exception))
    
    def test_parse_strategy_missing_required_fields(self):
        """测试解析缺少必要字段的配置"""
        # 缺少name字段
        invalid_config = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "conditions": []
            }
        }
        
        with self.assertRaises(StrategyValidationError) as context:
            self.parser.parse_strategy(invalid_config)
        
        self.assertIn("缺少必要字段", str(context.exception))
        
        # 缺少id字段
        invalid_config = {
            "strategy": {
                "name": "测试策略",
                "conditions": []
            }
        }
        
        with self.assertRaises(StrategyValidationError) as context:
            self.parser.parse_strategy(invalid_config)
        
        self.assertIn("缺少必要字段", str(context.exception))
    
    def test_parse_conditions_invalid_logic(self):
        """测试解析无效的逻辑运算符"""
        invalid_strategy = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
                "conditions": [
                    {
                        "logic": "INVALID_LOGIC"
                    }
                ]
            }
        }
        
        with self.assertRaises(StrategyValidationError) as context:
            self.parser.parse_strategy(invalid_strategy)
        
        self.assertIn("不支持的逻辑运算符", str(context.exception))
    
    def test_parse_conditions_missing_indicator_id(self):
        """测试解析缺少indicator_id的条件"""
        invalid_strategy = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
                "conditions": [
                    {
                        "period": "DAILY"
                    }
                ]
            }
        }
        
        with self.assertRaises(StrategyValidationError) as context:
            self.parser.parse_strategy(invalid_strategy)
        
        self.assertIn("缺少 indicator_id 字段", str(context.exception))
    
    def test_parse_conditions_missing_period(self):
        """测试解析缺少period的条件"""
        invalid_strategy = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
                "conditions": [
                    {
                        "indicator_id": "MA_CROSS"
                    }
                ]
            }
        }
        
        with self.assertRaises(StrategyValidationError) as context:
            self.parser.parse_strategy(invalid_strategy)
        
        self.assertIn("缺少 period 字段", str(context.exception))
    
    def test_parse_conditions_invalid_period(self):
        """测试解析无效的周期类型"""
        invalid_strategy = {
            "strategy": {
                "id": "TEST_STRATEGY",
                "name": "测试策略",
                "conditions": [
                    {
                        "indicator_id": "MA_CROSS",
                        "period": "INVALID_PERIOD"
                    }
                ]
            }
        }
        
        # 模拟指标创建，使其不抛出异常，以便测试周期验证
        with patch.object(IndicatorFactory, 'create', return_value=MagicMock()):
            with self.assertRaises(StrategyValidationError) as context:
                self.parser.parse_strategy(invalid_strategy)
        
        self.assertIn("不支持的周期类型", str(context.exception))
    
    def test_parse_conditions_indicator_not_found(self):
        """测试解析不存在的指标"""
        with patch.object(IndicatorFactory, 'create', side_effect=KeyError("指标不存在")):
            with self.assertRaises(IndicatorNotFoundError) as context:
                self.parser.parse_strategy(self.valid_strategy)
            
            self.assertIn("指标不存在", str(context.exception))
    
    def test_parse_from_file_json(self):
        """测试从JSON文件解析策略"""
        # 创建临时JSON文件
        temp_file = os.path.join(self.temp_dir.name, "test_strategy.json")
        with open(temp_file, 'w') as f:
            json.dump(self.valid_strategy, f)
        
        # 模拟指标创建
        with patch.object(IndicatorFactory, 'create') as mock_create:
            mock_create.side_effect = [self.mock_indicator1, self.mock_indicator2]
            
            # 从文件解析策略
            result = self.parser.parse_from_file(temp_file)
            
            # 验证结果
            self.assertEqual(result["strategy_id"], "TEST_STRATEGY")
            self.assertEqual(result["name"], "测试策略")
    
    def test_parse_from_file_not_exists(self):
        """测试解析不存在的文件"""
        non_existent_file = os.path.join(self.temp_dir.name, "non_existent.json")
        
        with self.assertRaises(ConfigFileError) as context:
            self.parser.parse_from_file(non_existent_file)
        
        self.assertIn("不存在", str(context.exception))
    
    def test_parse_from_file_unsupported_format(self):
        """测试解析不支持的文件格式"""
        temp_file = os.path.join(self.temp_dir.name, "test_strategy.txt")
        with open(temp_file, 'w') as f:
            f.write("This is not a valid format")
        
        with self.assertRaises(ConfigFileError) as context:
            self.parser.parse_from_file(temp_file)
        
        self.assertIn("不支持的配置文件格式", str(context.exception))
    
    def test_parse_from_string_json(self):
        """测试从JSON字符串解析策略"""
        json_str = json.dumps(self.valid_strategy)
        
        # 模拟指标创建
        with patch.object(IndicatorFactory, 'create') as mock_create:
            mock_create.side_effect = [self.mock_indicator1, self.mock_indicator2]
            
            # 从字符串解析策略
            result = self.parser.parse_from_string(json_str, 'json')
            
            # 验证结果
            self.assertEqual(result["strategy_id"], "TEST_STRATEGY")
            self.assertEqual(result["name"], "测试策略")
    
    def test_parse_from_string_invalid_format(self):
        """测试从无效格式的字符串解析策略"""
        with self.assertRaises(StrategyParseError) as context:
            self.parser.parse_from_string("This is not a valid JSON", 'json')
        
        self.assertIn("JSON解析错误", str(context.exception))
    
    def test_validate_strategy(self):
        """测试验证策略配置"""
        # 模拟指标创建
        with patch.object(IndicatorFactory, 'create') as mock_create:
            mock_create.side_effect = [self.mock_indicator1, self.mock_indicator2]
            
            # 验证有效策略
            result = self.parser.validate_strategy(self.valid_strategy)
            self.assertTrue(result)
            
            # 验证无效策略
            invalid_strategy = {"strategy": {"id": "INVALID"}}
            with self.assertRaises(StrategyValidationError):
                self.parser.validate_strategy(invalid_strategy)


if __name__ == '__main__':
    unittest.main() 