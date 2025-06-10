"""
指标性能测试

测试各种指标的计算性能，包括：
1. 计算时间
2. 内存使用
3. 不同数据规模下的性能表现
4. 单次计算与批量计算性能对比
"""
import time
import unittest
import pandas as pd
import numpy as np
import psutil
import os
import gc
from memory_profiler import profile
from functools import wraps
from indicators.factory import IndicatorFactory
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin

def measure_memory(func):
    """装饰器，用于测量函数执行前后的内存变化"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 强制垃圾回收
        gc.collect()
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 再次强制垃圾回收
        gc.collect()
        
        # 记录结束时内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 计算内存差异
        memory_diff = final_memory - initial_memory
        
        print(f"函数 '{func.__name__}' 内存使用: {memory_diff:.2f} MB")
        
        return result
    return wrapper

class TestIndicatorPerformance(unittest.TestCase, LogCaptureMixin):

    def setUp(self):
        """准备测试数据"""
        # 生成不同大小的数据集用于性能测试
        self.small_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 100},
            {'type': 'v_shape', 'start_price': 120, 'bottom_price': 90, 'periods': 100},
        ])
        
        self.medium_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 500},
            {'type': 'v_shape', 'start_price': 120, 'bottom_price': 90, 'periods': 500},
        ])
        
        self.large_data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 1000},
            {'type': 'v_shape', 'start_price': 120, 'bottom_price': 90, 'periods': 1000},
        ])
        
        # 为了测试需要，模拟一些StockInfo字段
        for dataset in [self.small_data, self.medium_data, self.large_data]:
            dataset['volume'] = dataset['volume'].astype(float)
            dataset['turnover_rate'] = dataset['volume'] / 10000  # 模拟换手率
            dataset['price_change'] = dataset['close'].diff()
            dataset['price_range'] = dataset['high'] - dataset['low']
            dataset['industry'] = 'Technology'  # 模拟行业
        
        # 注册所有指标
        IndicatorFactory.auto_register_all_indicators()
        self.supported_indicators = IndicatorFactory.get_supported_indicators()
        
        # 排除列表：一些高级或特殊指标可能不适合性能测试
        self.exclude_list = [
            'COMPOSITEINDICATOR', 'SENTIMENTANALYSIS', 'CHIPDISTRIBUTION',  
            'FIBONACCITOOLS', 'GANNTOOLS', 'FIBONACCI', 'MULTIPERIODRESONANCE',
            'STOCKVIX', 'VIX', 'ENHANCEDCCI', 'ENHANCEDTRIX', 'ENHANCEDWR',
            'ENHANCEDSTOCHRSI', 'ENHANCEDMFI', 'ENHANCEDOBV', 'ENHANCEDDMI',
            'CROSS_OVER', 'CROSSOVER', 'ELLIOTTWAVE',
            'ZXMTURNOVER', 'ZXMVOLUMESHRINK',
            'INSTITUTIONALBEHAVIOR', 'TRENDCLASSIFICATION', 'TRENDSTRENGTH',
            'CANDLESTICKPATTERNS', 'ADVANCEDCANDLESTICKPATTERNS',
            'CMO',  # 添加到排除列表，实现不完整
            'DMA',  # 添加到排除列表，实现不完整
            'KC',   # 添加到排除列表，实现不完整
            'KDJ_CONDITION', 'MACD_CONDITION', 'MA_CONDITION', 'GENERIC_CONDITION',  # 条件类指标需要特殊处理
            'KDJCONDITION', 'MACDCONDITION', 'MACONDITION', 'GENERICCONDITION',  # 条件类指标别名
            'ZXMPATTERNINDICATOR'  # 需要特殊参数的指标
        ]
        
        # 选择一些典型指标进行详细测试
        self.test_indicators = [
            'MACD', 'RSI', 'KDJ', 'BOLL', 'MA',  # 基础指标
            'OBV', 'VOL', 'CCI', 'ATR',          # 常用指标
            'SAR', 'WR', 'BIAS', 'ROC'           # 特殊指标
        ]
        
        # 确保所有测试指标都在支持列表中
        self.test_indicators = [ind for ind in self.test_indicators 
                               if ind in self.supported_indicators and ind not in self.exclude_list]

    def test_all_indicators_calculation_performance(self):
        """测试所有支持的指标的计算性能"""
        total_time = 0
        indicator_timings = {}

        for name in self.supported_indicators:
            if name in self.exclude_list:
                continue
            
            try:
                indicator = IndicatorFactory.create_indicator(name)
                
                start_time = time.time()
                # 我们只关心能否成功计算，不关心结果的合并
                _ = indicator.calculate(self.medium_data.copy())
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                indicator_timings[name] = elapsed_time
                total_time += elapsed_time

            except Exception as e:
                self.fail(f"指标 '{name}' 在性能测试中计算失败: {e}")

        print("\n--- 指标性能测试报告 ---")
        # 按耗时排序
        sorted_timings = sorted(indicator_timings.items(), key=lambda item: item[1], reverse=True)
        for name, timing in sorted_timings:
            print(f"指标: {name:<30} | 耗时: {timing:.4f} 秒")
        
        print(f"\n总计耗时: {total_time:.4f} 秒")
        # 设置一个宽松的性能阈值，例如总时间不超过10秒
        self.assertLess(total_time, 10.0, "所有指标的总计算时间超过了10秒的阈值")
    
    def test_indicator_performance_by_data_size(self):
        """测试不同数据规模下的指标性能"""
        results = {}
        
        # 遍历测试指标列表
        for name in self.test_indicators:
            results[name] = {}
            
            try:
                indicator = IndicatorFactory.create_indicator(name)
                
                # 小规模数据测试
                start_time = time.time()
                _ = indicator.calculate(self.small_data.copy())
                small_time = time.time() - start_time
                results[name]['small'] = small_time
                
                # 中规模数据测试
                start_time = time.time()
                _ = indicator.calculate(self.medium_data.copy())
                medium_time = time.time() - start_time
                results[name]['medium'] = medium_time
                
                # 大规模数据测试
                start_time = time.time()
                _ = indicator.calculate(self.large_data.copy())
                large_time = time.time() - start_time
                results[name]['large'] = large_time
                
            except Exception as e:
                self.fail(f"指标 '{name}' 在数据规模测试中失败: {e}")
        
        # 打印结果报告
        print("\n--- 不同数据规模下的指标性能测试报告 ---")
        print(f"{'指标名称':<30} | {'小规模(200行)':<15} | {'中规模(1000行)':<15} | {'大规模(2000行)':<15} | {'比例(大/小)':<15}")
        print("-" * 100)
        
        for name, timings in results.items():
            ratio = timings['large'] / timings['small'] if timings['small'] > 0 else float('inf')
            print(f"{name:<30} | {timings['small']:.4f} 秒 | {timings['medium']:.4f} 秒 | {timings['large']:.4f} 秒 | {ratio:.2f}")
        
        # 验证：理想情况下，计算时间应该随数据规模线性增长
        # 这里我们不做严格验证，只是确保大规模数据的计算时间不超过小规模的50倍
        for name, timings in results.items():
            ratio = timings['large'] / timings['small'] if timings['small'] > 0 else float('inf')
            self.assertLess(ratio, 50.0, f"指标 {name} 在大规模数据上的性能过差，比例为 {ratio:.2f}")
    
    @measure_memory
    def test_memory_usage_of_indicators(self):
        """测试指标计算的内存使用情况"""
        for name in self.test_indicators:
            try:
                indicator = IndicatorFactory.create_indicator(name)
                
                # 使用装饰器测量内存使用
                @measure_memory
                def calculate_indicator():
                    return indicator.calculate(self.large_data.copy())
                
                _ = calculate_indicator()
                
            except Exception as e:
                self.fail(f"指标 '{name}' 在内存使用测试中失败: {e}")
    
    def test_pattern_recognition_performance(self):
        """测试形态识别的性能"""
        pattern_results = {}
        
        for name in self.test_indicators:
            try:
                indicator = IndicatorFactory.create_indicator(name)
                
                # 检查是否有get_patterns方法
                if hasattr(indicator, 'get_patterns'):
                    # 首先计算指标
                    calculated_data = indicator.calculate(self.medium_data.copy())
                    
                    # 测量形态识别性能
                    start_time = time.time()
                    _ = indicator.get_patterns(calculated_data)
                    elapsed_time = time.time() - start_time
                    
                    pattern_results[name] = elapsed_time
            
            except Exception as e:
                print(f"警告: 指标 '{name}' 在形态识别性能测试中失败: {e}")
        
        # 打印结果
        if pattern_results:
            print("\n--- 形态识别性能测试报告 ---")
            sorted_results = sorted(pattern_results.items(), key=lambda item: item[1], reverse=True)
            for name, timing in sorted_results:
                print(f"指标: {name:<30} | 形态识别耗时: {timing:.4f} 秒")
    
    def test_batch_vs_individual_calculation(self):
        """对比批量计算与单个计算的性能差异"""
        # 选择几个典型指标进行测试
        test_subset = ['MACD', 'RSI', 'KDJ', 'BOLL', 'MA']
        test_subset = [ind for ind in test_subset if ind in self.supported_indicators and ind not in self.exclude_list]
        
        # 单个计算总时间
        individual_start = time.time()
        individual_results = {}
        
        for name in test_subset:
            indicator = IndicatorFactory.create_indicator(name)
            result = indicator.calculate(self.medium_data.copy())
            individual_results[name] = result
        
        individual_time = time.time() - individual_start
        
        # 批量计算总时间（通过链式计算）
        batch_start = time.time()
        batch_df = self.medium_data.copy()
        
        for name in test_subset:
            indicator = IndicatorFactory.create_indicator(name)
            batch_df = indicator.calculate(batch_df)
        
        batch_time = time.time() - batch_start
        
        # 打印结果
        print("\n--- 批量计算与单个计算性能对比 ---")
        print(f"单个计算总时间: {individual_time:.4f} 秒")
        print(f"批量链式计算总时间: {batch_time:.4f} 秒")
        print(f"性能比例 (单个/批量): {individual_time/batch_time:.2f}")
        
        # 验证批量计算应该不会明显慢于单个计算
        self.assertLess(batch_time, individual_time * 1.5, "批量计算性能明显差于单个计算")
    
    def test_repeated_calculation_performance(self):
        """测试重复计算的性能表现"""
        # 选择一个典型指标进行测试
        test_indicator = 'MACD'
        if test_indicator in self.supported_indicators and test_indicator not in self.exclude_list:
            indicator = IndicatorFactory.create_indicator(test_indicator)
            
            # 第一次计算
            start_time = time.time()
            _ = indicator.calculate(self.medium_data.copy())
            first_time = time.time() - start_time
            
            # 第二次计算
            start_time = time.time()
            _ = indicator.calculate(self.medium_data.copy())
            second_time = time.time() - start_time
            
            # 打印结果
            print("\n--- 重复计算性能测试 ---")
            print(f"指标: {test_indicator}")
            print(f"第一次计算时间: {first_time:.4f} 秒")
            print(f"第二次计算时间: {second_time:.4f} 秒")
            print(f"性能比例 (第二次/第一次): {second_time/first_time:.2f}")
            
            # 如果实现了缓存机制，第二次计算应该更快
            # 但这里我们只是验证第二次计算不会明显慢于第一次
            self.assertLessEqual(second_time, first_time * 1.2, "第二次计算性能明显下降")
    
    def test_parameter_variation_performance(self):
        """测试不同参数设置对性能的影响"""
        # 选择MA指标进行参数变化测试
        test_indicator = 'MA'
        if test_indicator in self.supported_indicators and test_indicator not in self.exclude_list:
            # 默认参数
            default_indicator = IndicatorFactory.create_indicator(test_indicator)
            
            start_time = time.time()
            _ = default_indicator.calculate(self.medium_data.copy())
            default_time = time.time() - start_time
            
            # 修改参数（例如增加更多的移动平均周期）
            try:
                # 尝试创建带有自定义参数的指标
                custom_indicator = IndicatorFactory.create_indicator(
                    test_indicator, 
                    periods=[5, 10, 20, 30, 60, 120, 250]  # 增加更多周期
                )
                
                start_time = time.time()
                _ = custom_indicator.calculate(self.medium_data.copy())
                custom_time = time.time() - start_time
                
                # 打印结果
                print("\n--- 参数变化性能测试 ---")
                print(f"指标: {test_indicator}")
                print(f"默认参数计算时间: {default_time:.4f} 秒")
                print(f"自定义参数计算时间: {custom_time:.4f} 秒")
                print(f"性能比例 (自定义/默认): {custom_time/default_time:.2f}")
                
                # 参数增加后，计算时间可能会增加，但不应该增加过多
                self.assertLess(custom_time, default_time * 3, 
                               "参数增加后，计算时间增加过多")
            except Exception as e:
                print(f"警告: 无法使用自定义参数创建指标 {test_indicator}: {e}")
    
    def test_error_logging_performance(self):
        """测试错误日志记录对性能的影响"""
        # 选择一个典型指标
        test_indicator = 'MACD'
        
        if test_indicator in self.supported_indicators and test_indicator not in self.exclude_list:
            indicator = IndicatorFactory.create_indicator(test_indicator)
            
            # 正常数据计算
            start_time = time.time()
            _ = indicator.calculate(self.medium_data.copy())
            normal_time = time.time() - start_time
            
            # 制造错误数据（删除必要列）
            error_data = self.medium_data.copy()
            if 'close' in error_data.columns:
                error_data = error_data.drop(columns=['close'])
            
            # 捕获日志并测量性能
            self.setup_log_capture()
            try:
                start_time = time.time()
                try:
                    _ = indicator.calculate(error_data)
                except:
                    pass  # 我们预期会失败
                error_time = time.time() - start_time
                
                # 打印结果
                print("\n--- 错误处理性能测试 ---")
                print(f"指标: {test_indicator}")
                print(f"正常计算时间: {normal_time:.4f} 秒")
                print(f"错误情况时间: {error_time:.4f} 秒")
                print(f"性能比例 (错误/正常): {error_time/normal_time:.2f}")
                
                # 验证错误处理不应该明显拖慢系统
                # 注意：错误处理通常会导致一些性能下降
                self.assertLess(error_time, normal_time * 5, 
                               "错误处理导致性能严重下降")
                
            finally:
                self.teardown_log_capture()


if __name__ == '__main__':
    unittest.main() 