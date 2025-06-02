#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
股票分析系统测试脚本

测试指标系统、回测系统和选股系统的功能
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta
from typing import List, Dict, Optional
# 单独导入Any类型
from typing import Any

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.clickhouse_db import get_clickhouse_db
from indicators.factory import IndicatorFactory
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest
from strategy.strategy_executor import StrategyExecutor
from strategy.strategy_manager import StrategyManager
from enums.period import Period
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)


class TestIndicatorsAndBacktest(unittest.TestCase):
    """测试指标系统、回测系统和选股系统的功能"""

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前执行"""
        cls.db = get_clickhouse_db()
        cls.indicator_factory = IndicatorFactory()
        cls.backtest = ConsolidatedBacktest(cpu_cores=4)
        cls.strategy_manager = StrategyManager()
        cls.strategy_executor = StrategyExecutor(max_workers=4)
        
        # 获取测试股票列表
        try:
            # 获取最新交易日期
            cls.latest_trade_date = cls.db.get_stock_max_date().strftime('%Y-%m-%d')
            # 获取30天前的日期
            cls.date_30_days_ago = (datetime.strptime(cls.latest_trade_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            # 获取90天前的日期
            cls.date_90_days_ago = (datetime.strptime(cls.latest_trade_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # 获取上证50股票列表进行测试
            cls.test_stocks_df = cls.db.query("SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code FROM stock.index_weight WHERE index_code = '000016.SH' LIMIT 10)")
            cls.test_stocks = cls.test_stocks_df['stock_code'].tolist()
            
            logger.info(f"测试准备完成, 测试股票数量: {len(cls.test_stocks)}, 最新交易日期: {cls.latest_trade_date}")
            
        except Exception as e:
            logger.error(f"测试准备阶段出错: {e}")
            raise

    def test_1_indicators_calculation(self):
        """测试指标计算功能"""
        logger.info("开始测试指标计算功能")
        
        # 测试常用指标
        test_indicators = ['MACD', 'KDJ', 'RSI', 'BOLL', 'MA', 'VOL']
        success_count = 0
        
        # 选择一只测试股票
        stock_code = self.test_stocks[0]
        
        try:
            # 获取该股票的日线数据
            stock_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=self.date_90_days_ago,
                end_date=self.latest_trade_date,
                period='day'
            )
            
            logger.info(f"获取股票 {stock_code} 数据，数据长度: {len(stock_data)}")
            
            # 测试每个指标
            for indicator_name in test_indicators:
                indicator = self.indicator_factory.create(indicator_name)
                self.assertIsNotNone(indicator, f"创建指标 {indicator_name} 失败")
                
                result = indicator.calculate(stock_data)
                self.assertIsNotNone(result, f"计算指标 {indicator_name} 失败")
                
                # 验证结果是否包含必要的列
                if indicator_name == 'MACD':
                    self.assertTrue('DIFF' in result, "MACD结果中应包含DIFF列")
                    self.assertTrue('DEA' in result, "MACD结果中应包含DEA列")
                    self.assertTrue('MACD' in result, "MACD结果中应包含MACD列")
                elif indicator_name == 'KDJ':
                    self.assertTrue('K' in result, "KDJ结果中应包含K列")
                    self.assertTrue('D' in result, "KDJ结果中应包含D列")
                    self.assertTrue('J' in result, "KDJ结果中应包含J列")
                elif indicator_name == 'RSI':
                    self.assertTrue('RSI' in result, "RSI结果中应包含RSI列")
                
                # 测试形态识别功能
                patterns = indicator.get_patterns(data=stock_data, result=result)
                logger.info(f"指标 {indicator_name} 识别到 {len(patterns)} 个形态")
                
                success_count += 1
                logger.info(f"指标 {indicator_name} 测试成功")
            
            logger.info(f"指标计算测试完成，成功率: {success_count}/{len(test_indicators)}")
            self.assertEqual(success_count, len(test_indicators), "部分指标测试失败")
            
        except Exception as e:
            logger.error(f"指标计算测试出错: {e}")
            raise

    def test_2_backtest_system(self):
        """测试回测系统功能"""
        logger.info("开始测试回测系统功能")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 分析单个股票
            buy_date = self.date_30_days_ago
            
            logger.info(f"分析股票 {stock_code} 在 {buy_date} 的买点")
            result = self.backtest.analyze_stock(
                code=stock_code,
                buy_date=buy_date
            )
            
            self.assertIsNotNone(result, "回测系统分析结果不应为None")
            self.assertEqual(result.stock_code, stock_code, "回测结果股票代码不匹配")
            self.assertEqual(result.buy_date, buy_date, "回测结果买入日期不匹配")
            
            # 验证至少有一个周期的分析结果
            self.assertTrue(len(result.period_results) > 0, "回测结果应包含至少一个周期的分析")
            
            # 验证日线周期的分析结果
            daily_result = result.period_results.get(Period.DAILY)
            self.assertIsNotNone(daily_result, "回测结果应包含日线周期的分析")
            
            # 验证至少有一个指标的分析结果
            self.assertTrue(len(daily_result.indicator_results) > 0, "日线分析结果应包含至少一个指标的分析")
            
            # 批量回测测试
            logger.info("开始测试批量回测功能")
            test_stock_list = self.test_stocks[:2]  # 使用前两只股票进行测试
            test_dates = [self.date_30_days_ago]
            
            # 创建临时输入文件
            temp_input_file = os.path.join(root_dir, 'data', 'result', 'temp_backtest_input.csv')
            temp_output_file = os.path.join(root_dir, 'data', 'result', 'temp_backtest_output.xlsx')
            
            # 准备输入数据
            input_data = []
            for stock in test_stock_list:
                for date in test_dates:
                    input_data.append({
                        'stock_code': stock,
                        'buy_date': date
                    })
            
            input_df = pd.DataFrame(input_data)
            input_df.to_csv(temp_input_file, index=False)
            
            # 执行批量回测
            batch_results = self.backtest.batch_analyze(
                input_file=temp_input_file,
                output_file=temp_output_file
            )
            
            self.assertTrue(len(batch_results) > 0, "批量回测结果不应为空")
            self.assertTrue(os.path.exists(temp_output_file), "回测结果文件应存在")
            
            # 生成策略
            strategy = self.backtest.generate_strategy(
                results=batch_results,
                output_file=os.path.join(root_dir, 'data', 'result', 'temp_strategy.json'),
                threshold=1
            )
            
            self.assertIsNotNone(strategy, "生成的策略不应为None")
            self.assertTrue('strategy' in strategy, "生成的策略应包含strategy字段")
            self.assertTrue('conditions' in strategy['strategy'], "策略应包含conditions字段")
            
            logger.info("回测系统测试完成")
            
            # 清理临时文件
            try:
                if os.path.exists(temp_input_file):
                    os.remove(temp_input_file)
                if os.path.exists(temp_output_file):
                    os.remove(temp_output_file)
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")
            
        except Exception as e:
            logger.error(f"回测系统测试出错: {e}")
            raise

    def test_3_strategy_execution(self):
        """测试选股策略执行功能"""
        logger.info("开始测试选股策略执行功能")
        
        try:
            # 加载测试策略
            strategy_dir = os.path.join(root_dir, 'config', 'strategies')
            
            # 如果目录不存在，创建它
            if not os.path.exists(strategy_dir):
                os.makedirs(strategy_dir)
            
            # 测试策略ID
            test_strategy_id = 'TEST_STRATEGY_001'
            
            # 创建测试策略
            test_strategy = {
                "strategy": {
                    "id": test_strategy_id,
                    "name": "测试策略",
                    "description": "用于测试的MACD金叉策略",
                    "create_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "author": "System Test",
                    "version": "1.0",
                    "tags": ["MACD", "金叉", "测试"],
                    "conditions": [
                        {
                            "period": "daily",
                            "indicator_id": "MACD",
                            "pattern_id": "golden_cross",
                            "min_strength": 0.6,
                            "score_threshold": 60
                        }
                    ],
                    "trading_rules": {
                        "entry": {
                            "min_score": 60,
                            "max_spread": 0.03
                        },
                        "exit": {
                            "take_profit": 0.15,
                            "stop_loss": 0.07
                        }
                    }
                }
            }
            
            # 保存测试策略
            strategy_file = os.path.join(strategy_dir, f"{test_strategy_id}.json")
            with open(strategy_file, 'w', encoding='utf-8') as f:
                json.dump(test_strategy, f, ensure_ascii=False, indent=2)
            
            # 执行策略
            result_df = self.strategy_executor.execute_strategy_by_id(
                strategy_id=test_strategy_id,
                strategy_manager=self.strategy_manager,
                start_date=None,
                end_date=self.latest_trade_date
            )
            
            # 验证结果
            self.assertIsNotNone(result_df, "策略执行结果不应为None")
            
            # 验证结果包含必要的列
            expected_columns = ['stock_code', 'stock_name', 'score']
            for col in expected_columns:
                self.assertTrue(col in result_df.columns, f"结果应包含{col}列")
            
            logger.info(f"策略执行成功，选出股票数量: {len(result_df)}")
            
            # 清理测试策略文件
            try:
                if os.path.exists(strategy_file):
                    os.remove(strategy_file)
            except Exception as e:
                logger.warning(f"清理测试策略文件时出错: {e}")
            
        except Exception as e:
            logger.error(f"选股策略执行测试出错: {e}")
            raise

    def test_4_integration(self):
        """测试系统集成功能"""
        logger.info("开始测试系统集成功能")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 1. 使用回测系统分析买点
            buy_date = self.date_30_days_ago
            logger.info(f"分析股票 {stock_code} 在 {buy_date} 的买点")
            
            analysis_result = self.backtest.analyze_stock(
                code=stock_code,
                buy_date=buy_date
            )
            
            self.assertIsNotNone(analysis_result, "回测分析结果不应为None")
            
            # 2. 基于分析结果生成策略
            logger.info("基于分析结果生成策略")
            
            strategy = self.backtest.generate_strategy(
                results=[analysis_result],
                output_file=None,  # 不保存到文件
                threshold=1
            )
            
            self.assertIsNotNone(strategy, "生成的策略不应为None")
            
            # 3. 使用生成的策略进行选股
            logger.info("使用生成的策略进行选股")
            
            # 解析策略
            from strategy.strategy_parser import StrategyParser
            parser = StrategyParser()
            strategy_plan = parser.parse_strategy(strategy)
            
            # 执行策略
            result_df = self.strategy_executor.execute_strategy(
                strategy_plan=strategy_plan,
                start_date=None,
                end_date=self.latest_trade_date
            )
            
            # 验证结果
            self.assertIsNotNone(result_df, "策略执行结果不应为None")
            
            logger.info(f"集成测试成功，选出股票数量: {len(result_df)}")
            if not result_df.empty:
                logger.info(f"选出的第一只股票: {result_df.iloc[0]['stock_code']} {result_df.iloc[0]['stock_name']}")
            
        except Exception as e:
            logger.error(f"系统集成测试出错: {e}")
            raise


if __name__ == '__main__':
    unittest.main() 