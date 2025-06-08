#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
多周期分析测试脚本

测试系统对不同周期数据的处理能力
"""

import unittest
import pandas as pd
import numpy as np

from analysis.multi_period_analysis import MultiPeriodAnalysis
from indicators.ma import MA
from indicators.macd import MACD
from datetime import datetime, timedelta
from typing import List, Dict, Optional
# 单独导入Any类型
from typing import Any

from db.clickhouse_db import get_clickhouse_db
from indicators.factory import IndicatorFactory
from scripts.backtest.consolidated_backtest import ConsolidatedBacktest
from enums.period import Period
from utils.period_manager import PeriodManager
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)


class TestMultiPeriodAnalysis(unittest.TestCase):
    """测试多周期分析功能"""

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前执行"""
        cls.db = get_clickhouse_db()
        cls.indicator_factory = IndicatorFactory()
        cls.backtest = ConsolidatedBacktest(cpu_cores=4)
        cls.period_manager = PeriodManager.get_instance()
        
        # 获取测试股票列表
        try:
            # 获取最新交易日期
            cls.latest_trade_date = cls.db.get_stock_max_date().strftime('%Y-%m-%d')
            # 获取30天前的日期
            cls.date_30_days_ago = (datetime.strptime(cls.latest_trade_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
            # 获取180天前的日期
            cls.date_180_days_ago = (datetime.strptime(cls.latest_trade_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')
            
            # 获取上证50股票列表进行测试
            cls.test_stocks_df = cls.db.query("SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code FROM stock.index_weight WHERE index_code = '000016.SH' LIMIT 3)")
            cls.test_stocks = cls.test_stocks_df['stock_code'].tolist()
            
            logger.info(f"测试准备完成, 测试股票数量: {len(cls.test_stocks)}, 最新交易日期: {cls.latest_trade_date}")
            
        except Exception as e:
            logger.error(f"测试准备阶段出错: {e}")
            raise

    def test_1_different_period_data_fetch(self):
        """测试不同周期数据的获取"""
        logger.info("开始测试不同周期数据的获取")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 测试周期列表
            test_periods = [
                (Period.DAILY, 'day'),
                (Period.MIN_60, '60min'),
                (Period.MIN_30, '30min'),
                (Period.MIN_15, '15min'),
                (Period.WEEKLY, 'week'),
                (Period.MONTHLY, 'month')
            ]
            
            for period_enum, period_str in test_periods:
                logger.info(f"获取 {stock_code} 的 {period_str} 周期数据")
                
                # 获取该股票的特定周期数据
                try:
                    stock_data = self.db.get_kline_data(
                        stock_code=stock_code,
                        start_date=self.date_180_days_ago,
                        end_date=self.latest_trade_date,
                        period=period_str
                    )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        logger.info(f"{period_str} 周期数据获取成功，数据长度: {len(stock_data)}")
                        
                        # 验证数据包含必要的列
                        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                        for col in required_columns:
                            self.assertTrue(col in stock_data.columns, f"{period_str} 周期数据应包含 {col} 列")
                    else:
                        logger.warning(f"{period_str} 周期数据获取失败或为空")
                        
                except Exception as e:
                    logger.error(f"获取 {period_str} 周期数据时出错: {e}")
            
            logger.info("不同周期数据获取测试完成")
            
        except Exception as e:
            logger.error(f"不同周期数据获取测试出错: {e}")
            raise

    def test_2_period_manager(self):
        """测试周期管理器功能"""
        logger.info("开始测试周期管理器功能")
        
        try:
            # 测试周期转换
            self.assertEqual(self.period_manager.convert_to_period_enum('day'), Period.DAILY, "周期转换错误")
            self.assertEqual(self.period_manager.convert_to_period_enum('60min'), Period.MIN_60, "周期转换错误")
            self.assertEqual(self.period_manager.convert_to_period_enum('week'), Period.WEEKLY, "周期转换错误")
            
            # 测试周期显示名称
            self.assertEqual(self.period_manager.get_period_display_name(Period.DAILY), "日线", "周期显示名称错误")
            self.assertEqual(self.period_manager.get_period_display_name(Period.MIN_60), "60分钟线", "周期显示名称错误")
            self.assertEqual(self.period_manager.get_period_display_name(Period.WEEKLY), "周线", "周期显示名称错误")
            
            # 测试周期数据需求
            daily_days = self.period_manager.get_days_needed_for_period(Period.DAILY)
            weekly_days = self.period_manager.get_days_needed_for_period(Period.WEEKLY)
            monthly_days = self.period_manager.get_days_needed_for_period(Period.MONTHLY)
            
            logger.info(f"日线所需天数: {daily_days}")
            logger.info(f"周线所需天数: {weekly_days}")
            logger.info(f"月线所需天数: {monthly_days}")
            
            self.assertTrue(daily_days < weekly_days < monthly_days, "周期数据需求天数应随周期增长而增加")
            
            logger.info("周期管理器测试通过")
            
        except Exception as e:
            logger.error(f"周期管理器测试出错: {e}")
            raise

    def test_3_multi_period_indicator_calculation(self):
        """测试多周期指标计算"""
        logger.info("开始测试多周期指标计算")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 测试周期
            test_periods = [Period.DAILY, Period.WEEKLY]
            
            # 测试指标
            test_indicators = ['MACD', 'KDJ', 'RSI']
            
            # 对每个周期和指标进行测试
            for period in test_periods:
                period_str = self.period_manager.convert_to_period_string(period)
                logger.info(f"测试 {period_str} 周期的指标计算")
                
                # 获取该股票的周期数据
                stock_data = self.db.get_kline_data(
                    stock_code=stock_code,
                    start_date=self.date_180_days_ago,
                    end_date=self.latest_trade_date,
                    period=period_str
                )
                
                if stock_data is None or len(stock_data) == 0:
                    logger.warning(f"{period_str} 周期数据为空，跳过")
                    continue
                
                logger.info(f"{period_str} 周期数据长度: {len(stock_data)}")
                
                # 对每个指标进行计算
                for indicator_name in test_indicators:
                    indicator = self.indicator_factory.create(indicator_name)
                    self.assertIsNotNone(indicator, f"创建 {indicator_name} 指标失败")
                    
                    result = indicator.calculate(stock_data)
                    self.assertIsNotNone(result, f"计算 {period_str} 周期的 {indicator_name} 指标失败")
                    
                    # 验证结果是否包含必要的列
                    if indicator_name == 'MACD':
                        self.assertTrue('DIFF' in result, f"{period_str} 周期的 MACD 结果中应包含 DIFF 列")
                        self.assertTrue('DEA' in result, f"{period_str} 周期的 MACD 结果中应包含 DEA 列")
                        self.assertTrue('MACD' in result, f"{period_str} 周期的 MACD 结果中应包含 MACD 列")
                    elif indicator_name == 'KDJ':
                        self.assertTrue('K' in result, f"{period_str} 周期的 KDJ 结果中应包含 K 列")
                        self.assertTrue('D' in result, f"{period_str} 周期的 KDJ 结果中应包含 D 列")
                        self.assertTrue('J' in result, f"{period_str} 周期的 KDJ 结果中应包含 J 列")
                    elif indicator_name == 'RSI':
                        self.assertTrue('RSI' in result, f"{period_str} 周期的 RSI 结果中应包含 RSI 列")
                    
                    logger.info(f"{period_str} 周期的 {indicator_name} 指标计算成功")
            
            logger.info("多周期指标计算测试通过")
            
        except Exception as e:
            logger.error(f"多周期指标计算测试出错: {e}")
            raise

    def test_4_period_isolation(self):
        """测试周期隔离机制"""
        logger.info("开始测试周期隔离机制")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            buy_date = self.date_30_days_ago
            
            # 分析股票
            logger.info(f"分析股票 {stock_code} 在 {buy_date} 的买点")
            result = self.backtest.analyze_stock(
                code=stock_code,
                buy_date=buy_date
            )
            
            self.assertIsNotNone(result, "回测系统分析结果不应为None")
            
            # 验证结果中包含多个周期
            self.assertTrue(len(result.period_results) > 1, "回测结果应包含多个周期的分析")
            
            # 验证不同周期的结果是否独立存储
            for period, period_result in result.period_results.items():
                logger.info(f"验证 {period} 周期的分析结果")
                
                # 验证周期结果对象
                self.assertEqual(period_result.period, period, "周期结果对象的周期属性应与期望周期一致")
                
                # 验证指标结果不为空
                self.assertTrue(len(period_result.indicator_results) > 0, f"{period} 周期的指标结果不应为空")
                
                # 检查几个关键指标
                for indicator_id in ['MACD', 'KDJ', 'RSI']:
                    if indicator_id in period_result.indicator_results:
                        indicator_result = period_result.indicator_results[indicator_id]
                        
                        # 验证指标结果包含必要的信息
                        self.assertEqual(indicator_result.indicator_id, indicator_id, "指标结果的ID应与期望ID一致")
                        self.assertIsNotNone(indicator_result.score, f"{period} 周期的 {indicator_id} 指标评分不应为None")
                        self.assertIsNotNone(indicator_result.patterns, f"{period} 周期的 {indicator_id} 形态列表不应为None")
            
            # 验证不同周期同名指标的结果是否不同
            # 以MACD为例，日线和周线的MACD结果应该不同
            if Period.DAILY in result.period_results and Period.WEEKLY in result.period_results:
                daily_result = result.period_results[Period.DAILY]
                weekly_result = result.period_results[Period.WEEKLY]
                
                if 'MACD' in daily_result.indicator_results and 'MACD' in weekly_result.indicator_results:
                    daily_macd = daily_result.indicator_results['MACD']
                    weekly_macd = weekly_result.indicator_results['MACD']
                    
                    # 验证两个周期的MACD结果是否不同
                    self.assertNotEqual(daily_macd.score, weekly_macd.score, "日线和周线的MACD评分应该不同")
                    
                    logger.info(f"日线MACD评分: {daily_macd.score}, 周线MACD评分: {weekly_macd.score}")
                    logger.info(f"日线MACD形态数量: {len(daily_macd.patterns)}, 周线MACD形态数量: {len(weekly_macd.patterns)}")
            
            logger.info("周期隔离机制测试通过")
            
        except Exception as e:
            logger.error(f"周期隔离机制测试出错: {e}")
            raise

    def test_5_multi_period_backtest(self):
        """测试多周期回测功能"""
        logger.info("开始测试多周期回测功能")
        
        try:
            # 选择测试股票
            stock_code = self.test_stocks[0]
            buy_date = self.date_30_days_ago
            
            # 设置自定义配置，只使用日线和周线
            custom_config = {
                'periods': [Period.DAILY, Period.WEEKLY],
                'indicators': {
                    'MACD': {'enabled': True},
                    'KDJ': {'enabled': True},
                    'RSI': {'enabled': True}
                }
            }
            
            # 执行回测
            logger.info(f"使用自定义配置执行 {stock_code} 在 {buy_date} 的回测")
            result = self.backtest.analyze_stock(
                code=stock_code,
                buy_date=buy_date,
                custom_config=custom_config
            )
            
            self.assertIsNotNone(result, "回测结果不应为None")
            
            # 验证结果中只包含指定的周期
            self.assertEqual(len(result.period_results), 2, "回测结果应只包含日线和周线两个周期")
            self.assertTrue(Period.DAILY in result.period_results, "回测结果应包含日线周期")
            self.assertTrue(Period.WEEKLY in result.period_results, "回测结果应包含周线周期")
            
            # 验证结果中只包含指定的指标
            for period, period_result in result.period_results.items():
                logger.info(f"验证 {period} 周期的指标结果")
                
                # 应该只有3个指标
                self.assertEqual(len(period_result.indicator_results), 3, f"{period} 周期应只包含3个指标")
                
                # 验证具体指标
                self.assertTrue('MACD' in period_result.indicator_results, f"{period} 周期应包含MACD指标")
                self.assertTrue('KDJ' in period_result.indicator_results, f"{period} 周期应包含KDJ指标")
                self.assertTrue('RSI' in period_result.indicator_results, f"{period} 周期应包含RSI指标")
            
            logger.info("多周期回测功能测试通过")
            
        except Exception as e:
            logger.error(f"多周期回测功能测试出错: {e}")
            raise
    
    def test_6_batch_multi_period_analysis(self):
        """测试批量多周期分析功能"""
        logger.info("开始测试批量多周期分析功能")
        
        try:
            # 创建临时输入文件
            temp_input_file = os.path.join(root_dir, 'data', 'result', 'temp_multi_period_input.csv')
            temp_output_file = os.path.join(root_dir, 'data', 'result', 'temp_multi_period_output.xlsx')
            
            # 准备输入数据
            input_data = []
            for stock_code in self.test_stocks:
                input_data.append({
                    'stock_code': stock_code,
                    'buy_date': self.date_30_days_ago
                })
            
            input_df = pd.DataFrame(input_data)
            input_df.to_csv(temp_input_file, index=False)
            
            # 设置自定义配置，包含多个周期
            custom_config = {
                'periods': [Period.DAILY, Period.WEEKLY, Period.MIN_60],
                'indicators': {
                    'MACD': {'enabled': True},
                    'KDJ': {'enabled': True},
                    'RSI': {'enabled': True}
                }
            }
            
            # 执行批量回测
            logger.info("执行批量多周期回测")
            batch_results = self.backtest.batch_analyze(
                input_file=temp_input_file,
                output_file=temp_output_file,
                custom_config=custom_config
            )
            
            self.assertTrue(len(batch_results) > 0, "批量回测结果不应为空")
            self.assertTrue(os.path.exists(temp_output_file), "回测结果文件应存在")
            
            # 验证批量结果
            for result in batch_results:
                # 验证结果包含指定的周期
                self.assertEqual(len(result.period_results), 3, "回测结果应包含三个周期")
                self.assertTrue(Period.DAILY in result.period_results, "回测结果应包含日线周期")
                self.assertTrue(Period.WEEKLY in result.period_results, "回测结果应包含周线周期")
                self.assertTrue(Period.MIN_60 in result.period_results, "回测结果应包含60分钟线周期")
            
            logger.info("批量多周期分析功能测试通过")
            
            # 清理临时文件
            try:
                if os.path.exists(temp_input_file):
                    os.remove(temp_input_file)
                if os.path.exists(temp_output_file):
                    os.remove(temp_output_file)
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")
            
        except Exception as e:
            logger.error(f"批量多周期分析功能测试出错: {e}")
            raise


if __name__ == '__main__':
    unittest.main() 