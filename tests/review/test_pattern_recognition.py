#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
形态识别测试脚本

专门测试系统的形态识别功能
"""

import unittest
import pandas as pd
import numpy as np

from indicators.pattern_recognition import PatternRecognition
from indicators.ma import MA
from indicators.macd import MACD

class TestPatternRecognition(unittest.TestCase):
    """测试形态识别功能"""

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前执行"""
        cls.db = get_clickhouse_db()
        cls.indicator_factory = IndicatorFactory()
        
        # 获取测试股票列表
        try:
            # 获取最新交易日期
            cls.latest_trade_date = cls.db.get_stock_max_date().strftime('%Y-%m-%d')
            # 获取180天前的日期
            cls.date_180_days_ago = (datetime.strptime(cls.latest_trade_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')
            
            # 获取上证50股票列表进行测试
            cls.test_stocks_df = cls.db.query("SELECT stock_code, stock_name FROM stock.stock_info WHERE stock_code IN (SELECT stock_code FROM stock.index_weight WHERE index_code = '000016.SH' LIMIT 5)")
            cls.test_stocks = cls.test_stocks_df['stock_code'].tolist()
            
            logger.info(f"测试准备完成, 测试股票数量: {len(cls.test_stocks)}, 最新交易日期: {cls.latest_trade_date}")
            
        except Exception as e:
            logger.error(f"测试准备阶段出错: {e}")
            raise

    def test_1_macd_pattern_recognition(self):
        """测试MACD形态识别功能"""
        logger.info("开始测试MACD形态识别功能")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 获取该股票的日线数据
            stock_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=self.date_180_days_ago,
                end_date=self.latest_trade_date,
                period='day'
            )
            
            logger.info(f"获取股票 {stock_code} 数据，数据长度: {len(stock_data)}")
            
            # 创建MACD指标
            macd = self.indicator_factory.create('MACD')
            self.assertIsNotNone(macd, "创建MACD指标失败")
            
            # 计算MACD值
            macd_result = macd.calculate(stock_data)
            self.assertIsNotNone(macd_result, "计算MACD指标失败")
            
            # 识别MACD形态
            patterns = macd.get_patterns(data=stock_data, result=macd_result)
            
            # 记录形态数量
            logger.info(f"MACD形态识别结果: 共识别出 {len(patterns)} 个形态")
            
            # 输出每种形态的数量统计
            pattern_counts = {}
            for pattern in patterns:
                pattern_id = pattern.get('pattern_id', 'unknown')
                if pattern_id not in pattern_counts:
                    pattern_counts[pattern_id] = 0
                pattern_counts[pattern_id] += 1
            
            for pattern_id, count in pattern_counts.items():
                logger.info(f"  - {pattern_id}: {count}个")
            
            # 验证形态识别结果
            self.assertTrue(len(patterns) > 0, "应识别出至少一个MACD形态")
            
            # 验证形态中包含必要的字段
            if patterns:
                pattern = patterns[0]
                required_fields = ['pattern_id', 'start_index', 'end_index', 'strength', 'description']
                for field in required_fields:
                    self.assertTrue(field in pattern, f"形态结果应包含{field}字段")
            
            logger.info("MACD形态识别测试通过")
            
        except Exception as e:
            logger.error(f"MACD形态识别测试出错: {e}")
            raise

    def test_2_kdj_pattern_recognition(self):
        """测试KDJ形态识别功能"""
        logger.info("开始测试KDJ形态识别功能")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 获取该股票的日线数据
            stock_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=self.date_180_days_ago,
                end_date=self.latest_trade_date,
                period='day'
            )
            
            logger.info(f"获取股票 {stock_code} 数据，数据长度: {len(stock_data)}")
            
            # 创建KDJ指标
            kdj = self.indicator_factory.create('KDJ')
            self.assertIsNotNone(kdj, "创建KDJ指标失败")
            
            # 计算KDJ值
            kdj_result = kdj.calculate(stock_data)
            self.assertIsNotNone(kdj_result, "计算KDJ指标失败")
            
            # 识别KDJ形态
            patterns = kdj.get_patterns(data=stock_data, result=kdj_result)
            
            # 记录形态数量
            logger.info(f"KDJ形态识别结果: 共识别出 {len(patterns)} 个形态")
            
            # 输出每种形态的数量统计
            pattern_counts = {}
            for pattern in patterns:
                pattern_id = pattern.get('pattern_id', 'unknown')
                if pattern_id not in pattern_counts:
                    pattern_counts[pattern_id] = 0
                pattern_counts[pattern_id] += 1
            
            for pattern_id, count in pattern_counts.items():
                logger.info(f"  - {pattern_id}: {count}个")
            
            # 验证形态识别结果
            self.assertTrue(len(patterns) > 0, "应识别出至少一个KDJ形态")
            
            # 验证形态中包含必要的字段
            if patterns:
                pattern = patterns[0]
                required_fields = ['pattern_id', 'start_index', 'end_index', 'strength', 'description']
                for field in required_fields:
                    self.assertTrue(field in pattern, f"形态结果应包含{field}字段")
            
            logger.info("KDJ形态识别测试通过")
            
        except Exception as e:
            logger.error(f"KDJ形态识别测试出错: {e}")
            raise

    def test_3_boll_pattern_recognition(self):
        """测试BOLL形态识别功能"""
        logger.info("开始测试BOLL形态识别功能")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 获取该股票的日线数据
            stock_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=self.date_180_days_ago,
                end_date=self.latest_trade_date,
                period='day'
            )
            
            logger.info(f"获取股票 {stock_code} 数据，数据长度: {len(stock_data)}")
            
            # 创建BOLL指标
            boll = self.indicator_factory.create('BOLL')
            self.assertIsNotNone(boll, "创建BOLL指标失败")
            
            # 计算BOLL值
            boll_result = boll.calculate(stock_data)
            self.assertIsNotNone(boll_result, "计算BOLL指标失败")
            
            # 识别BOLL形态
            patterns = boll.get_patterns(data=stock_data, result=boll_result)
            
            # 记录形态数量
            logger.info(f"BOLL形态识别结果: 共识别出 {len(patterns)} 个形态")
            
            # 输出每种形态的数量统计
            pattern_counts = {}
            for pattern in patterns:
                pattern_id = pattern.get('pattern_id', 'unknown')
                if pattern_id not in pattern_counts:
                    pattern_counts[pattern_id] = 0
                pattern_counts[pattern_id] += 1
            
            for pattern_id, count in pattern_counts.items():
                logger.info(f"  - {pattern_id}: {count}个")
            
            # 验证形态识别结果
            self.assertTrue(len(patterns) > 0, "应识别出至少一个BOLL形态")
            
            # 验证形态中包含必要的字段
            if patterns:
                pattern = patterns[0]
                required_fields = ['pattern_id', 'start_index', 'end_index', 'strength', 'description']
                for field in required_fields:
                    self.assertTrue(field in pattern, f"形态结果应包含{field}字段")
            
            logger.info("BOLL形态识别测试通过")
            
        except Exception as e:
            logger.error(f"BOLL形态识别测试出错: {e}")
            raise

    def test_4_multi_stocks_pattern_analysis(self):
        """对多只股票进行形态识别分析"""
        logger.info("开始多只股票形态识别分析")
        
        try:
            # 测试指标
            test_indicators = ['MACD', 'KDJ', 'RSI', 'BOLL']
            
            # 结果统计
            all_results = {}
            
            # 对每只股票进行分析
            for stock_code in self.test_stocks:
                # 获取该股票的日线数据
                stock_data = self.db.get_kline_data(
                    stock_code=stock_code,
                    start_date=self.date_180_days_ago,
                    end_date=self.latest_trade_date,
                    period='day'
                )
                
                if stock_data is None or len(stock_data) == 0:
                    logger.warning(f"股票 {stock_code} 数据为空，跳过")
                    continue
                
                logger.info(f"分析股票 {stock_code} 的形态，数据长度: {len(stock_data)}")
                
                # 对每个指标进行分析
                for indicator_name in test_indicators:
                    # 创建指标
                    indicator = self.indicator_factory.create(indicator_name)
                    if indicator is None:
                        logger.warning(f"创建指标 {indicator_name} 失败，跳过")
                        continue
                    
                    # 计算指标值
                    result = indicator.calculate(stock_data)
                    if result is None:
                        logger.warning(f"计算指标 {indicator_name} 失败，跳过")
                        continue
                    
                    # 识别形态
                    patterns = indicator.get_patterns(data=stock_data, result=result)
                    
                    # 记录结果
                    key = f"{stock_code}_{indicator_name}"
                    all_results[key] = patterns
                    
                    # 输出形态数量
                    pattern_counts = {}
                    for pattern in patterns:
                        pattern_id = pattern.get('pattern_id', 'unknown')
                        if pattern_id not in pattern_counts:
                            pattern_counts[pattern_id] = 0
                        pattern_counts[pattern_id] += 1
                    
                    logger.info(f"股票 {stock_code} 的 {indicator_name} 形态: 共 {len(patterns)} 个")
                    for pattern_id, count in pattern_counts.items():
                        logger.info(f"  - {pattern_id}: {count}个")
            
            # 统计所有股票的形态分布
            indicator_pattern_stats = {}
            for key, patterns in all_results.items():
                _, indicator_name = key.split('_', 1)
                
                if indicator_name not in indicator_pattern_stats:
                    indicator_pattern_stats[indicator_name] = {}
                
                for pattern in patterns:
                    pattern_id = pattern.get('pattern_id', 'unknown')
                    if pattern_id not in indicator_pattern_stats[indicator_name]:
                        indicator_pattern_stats[indicator_name][pattern_id] = 0
                    indicator_pattern_stats[indicator_name][pattern_id] += 1
            
            # 输出总体统计
            logger.info("所有股票形态分布统计:")
            for indicator_name, pattern_stats in indicator_pattern_stats.items():
                logger.info(f"{indicator_name} 形态统计:")
                total_patterns = sum(pattern_stats.values())
                for pattern_id, count in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_patterns) * 100 if total_patterns > 0 else 0
                    logger.info(f"  - {pattern_id}: {count}个 ({percentage:.2f}%)")
            
            logger.info("多只股票形态识别分析完成")
            
        except Exception as e:
            logger.error(f"多只股票形态识别分析出错: {e}")
            raise

    def test_5_pattern_validation(self):
        """验证形态识别的准确性"""
        logger.info("开始验证形态识别的准确性")
        
        try:
            # 选择一只测试股票
            stock_code = self.test_stocks[0]
            
            # 获取该股票的日线数据
            stock_data = self.db.get_kline_data(
                stock_code=stock_code,
                start_date=self.date_180_days_ago,
                end_date=self.latest_trade_date,
                period='day'
            )
            
            logger.info(f"获取股票 {stock_code} 数据，数据长度: {len(stock_data)}")
            
            # 测试MACD金叉形态识别
            macd = self.indicator_factory.create('MACD')
            self.assertIsNotNone(macd, "创建MACD指标失败")
            
            # 计算MACD值
            macd_result = macd.calculate(stock_data)
            self.assertIsNotNone(macd_result, "计算MACD指标失败")
            
            # 识别MACD形态
            patterns = macd.get_patterns(data=stock_data, result=macd_result)
            
            # 查找金叉形态
            golden_cross_patterns = [p for p in patterns if p.get('pattern_id') == 'golden_cross']
            
            if golden_cross_patterns:
                logger.info(f"找到 {len(golden_cross_patterns)} 个MACD金叉形态")
                
                # 验证第一个金叉形态
                pattern = golden_cross_patterns[0]
                start_idx = pattern.get('start_index')
                end_idx = pattern.get('end_index')
                
                # 验证金叉前后的DIFF和DEA值变化
                if start_idx is not None and end_idx is not None and start_idx < end_idx:
                    diff_before = macd_result.iloc[start_idx]['DIFF']
                    dea_before = macd_result.iloc[start_idx]['DEA']
                    diff_after = macd_result.iloc[end_idx]['DIFF']
                    dea_after = macd_result.iloc[end_idx]['DEA']
                    
                    logger.info(f"MACD金叉前: DIFF={diff_before:.4f}, DEA={dea_before:.4f}")
                    logger.info(f"MACD金叉后: DIFF={diff_after:.4f}, DEA={dea_after:.4f}")
                    
                    # 金叉定义: DIFF从下方穿越DEA
                    cross_condition_1 = diff_before < dea_before
                    cross_condition_2 = diff_after > dea_after
                    
                    self.assertTrue(cross_condition_1, "金叉前DIFF应小于DEA")
                    self.assertTrue(cross_condition_2, "金叉后DIFF应大于DEA")
                    
                    logger.info("MACD金叉形态验证通过")
                else:
                    logger.warning("无法验证MACD金叉形态，索引无效")
            else:
                logger.warning("未找到MACD金叉形态，无法验证")
            
            # 测试KDJ超买形态识别
            kdj = self.indicator_factory.create('KDJ')
            self.assertIsNotNone(kdj, "创建KDJ指标失败")
            
            # 计算KDJ值
            kdj_result = kdj.calculate(stock_data)
            self.assertIsNotNone(kdj_result, "计算KDJ指标失败")
            
            # 识别KDJ形态
            patterns = kdj.get_patterns(data=stock_data, result=kdj_result)
            
            # 查找超买形态
            overbought_patterns = [p for p in patterns if p.get('pattern_id') == 'overbought']
            
            if overbought_patterns:
                logger.info(f"找到 {len(overbought_patterns)} 个KDJ超买形态")
                
                # 验证第一个超买形态
                pattern = overbought_patterns[0]
                idx = pattern.get('end_index')
                
                # 验证超买条件
                if idx is not None:
                    k_value = kdj_result.iloc[idx]['K']
                    d_value = kdj_result.iloc[idx]['D']
                    j_value = kdj_result.iloc[idx]['J']
                    
                    logger.info(f"KDJ超买点: K={k_value:.2f}, D={d_value:.2f}, J={j_value:.2f}")
                    
                    # 超买定义: K、D值都大于80
                    overbought_condition_1 = k_value > 80
                    overbought_condition_2 = d_value > 80
                    
                    self.assertTrue(overbought_condition_1, "超买形态K值应大于80")
                    self.assertTrue(overbought_condition_2, "超买形态D值应大于80")
                    
                    logger.info("KDJ超买形态验证通过")
                else:
                    logger.warning("无法验证KDJ超买形态，索引无效")
            else:
                logger.warning("未找到KDJ超买形态，无法验证")
            
            logger.info("形态识别准确性验证完成")
            
        except Exception as e:
            logger.error(f"形态识别准确性验证出错: {e}")
            raise


if __name__ == '__main__':
    unittest.main() 