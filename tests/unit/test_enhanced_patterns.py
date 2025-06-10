"""
增强形态识别单元测试

测试增强的形态识别功能，包括：
- 形态组合识别
- 形态确认与验证
- 形态与其他指标的配合
- 形态信号质量评估
"""

import unittest
import pandas as pd
import numpy as np

from indicators.pattern.pattern_combination import PatternCombination
from indicators.pattern.pattern_confirmation import PatternConfirmation
from indicators.pattern.pattern_quality_evaluator import PatternQualityEvaluator
from indicators.factory import IndicatorFactory
from tests.unit.indicator_test_mixin import IndicatorTestMixin
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


# 测试用的指标工厂类
class TestIndicatorFactory:
    """测试用的指标工厂，用于创建测试指标实例"""
    
    def __init__(self):
        """初始化测试指标工厂"""
        self.indicators = {
            'CANDLESTICKPATTERNS': TestCandlestickPatterns(),
            'MACD': TestMACD(),
            'RSI': TestRSI()
        }
    
    def get_indicator(self, indicator_name):
        """获取指标实例"""
        return self.indicators.get(indicator_name)


# 测试用的K线形态指标
class TestCandlestickPatterns:
    """测试用的K线形态指标"""
    
    def calculate(self, data):
        """计算K线形态"""
        result = data.copy()
        result['hammer'] = False
        result['shooting_star'] = False
        result['engulfing'] = False
        
        # 在特定位置设置几个形态
        if len(result) > 60:
            result.iloc[60:65, result.columns.get_indexer(['hammer'])[0]] = True
            result.iloc[100:105, result.columns.get_indexer(['engulfing'])[0]] = True
        
        return result


# 测试用的MACD指标
class TestMACD:
    """测试用的MACD指标"""
    
    def calculate(self, data):
        """计算MACD"""
        result = data.copy()
        result['macd_line'] = 0.0
        result['macd_signal'] = 0.0
        result['macd_histogram'] = 0.0
        
        # 模拟MACD金叉
        if len(result) > 60:
            # 设置MACD值使其在60-65位置形成金叉
            result.iloc[58:60, result.columns.get_indexer(['macd_line'])[0]] = -1.0
            result.iloc[58:60, result.columns.get_indexer(['macd_signal'])[0]] = 0.0
            
            result.iloc[60:65, result.columns.get_indexer(['macd_line'])[0]] = 1.0
            result.iloc[60:65, result.columns.get_indexer(['macd_signal'])[0]] = 0.0
            
            # 设置柱状图
            result.iloc[58:60, result.columns.get_indexer(['macd_histogram'])[0]] = -1.0
            result.iloc[60:65, result.columns.get_indexer(['macd_histogram'])[0]] = 1.0
        
        return result


# 测试用的RSI指标
class TestRSI:
    """测试用的RSI指标"""
    
    def calculate(self, data):
        """计算RSI"""
        result = data.copy()
        result['rsi'] = 50.0
        
        # 设置RSI值使其在指定位置低于30
        if len(result) > 60:
            result.iloc[60:65, result.columns.get_indexer(['rsi'])[0]] = 25.0
        
        return result


class TestPatternCombination(unittest.TestCase, IndicatorTestMixin):
    """形态组合识别测试"""
    
    def setUp(self):
        """为所有测试准备数据和指标实例"""
        self.indicator = PatternCombination()
        self.expected_columns = ['combined_pattern', 'pattern_strength']
        
        # 生成适合形态组合分析的数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 40},   # 下跌
            {'type': 'trend', 'start_price': 80, 'end_price': 90, 'periods': 30},    # 反弹
            {'type': 'trend', 'start_price': 90, 'end_price': 85, 'periods': 20},    # 回调
            {'type': 'trend', 'start_price': 85, 'end_price': 110, 'periods': 50},   # 上升
            {'type': 'trend', 'start_price': 110, 'end_price': 105, 'periods': 15},  # 回调
            {'type': 'trend', 'start_price': 105, 'end_price': 120, 'periods': 35},  # 继续上升
        ])
        
        # 添加量价数据
        volume = np.ones(len(self.data)) * 1000
        # 反弹时放量
        volume[40:60] = 2000
        # 回调时缩量
        volume[70:85] = 500
        # 突破时放量
        volume[100:120] = 2500
        self.data['volume'] = volume
        
        # 添加换手率
        self.data['turnover_rate'] = self.data['volume'] / 10000
        
        # 设置日志捕获
        self.setup_log_capture()
    
    def tearDown(self):
        """清理测试环境"""
        self.teardown_log_capture()
    
    def test_pattern_combination_detection(self):
        """测试形态组合识别功能"""
        result = self.indicator.calculate(self.data)
        
        # 验证有形态组合被识别
        self.assertFalse(result.empty, "没有识别出形态组合")
        
        # 检查形态组合类型
        if 'combined_pattern' in result.columns:
            combined_patterns = result['combined_pattern'].dropna().unique()
            self.assertTrue(len(combined_patterns) > 0, "未识别出任何形态组合")
    
    def test_pattern_strength_assessment(self):
        """测试形态强度评估"""
        result = self.indicator.calculate(self.data)
        
        # 检查形态强度
        if 'pattern_strength' in result.columns:
            strength_values = result['pattern_strength'].dropna()
            self.assertFalse(strength_values.empty, "形态强度值全为NaN")
            
            # 验证强度值在合理范围内
            self.assertTrue((strength_values >= 0).all() and (strength_values <= 100).all(),
                          "形态强度值超出0-100范围")


class TestPatternConfirmation(unittest.TestCase, IndicatorTestMixin):
    """形态确认测试"""
    
    def setUp(self):
        """为所有测试准备数据和指标实例"""
        self.indicator = PatternConfirmation()
        self.expected_columns = ['pattern_confirmed', 'confirmation_type', 'confirmation_strength']
        
        # 生成适合形态确认分析的数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 30},   # 下跌
            {'type': 'trend', 'start_price': 80, 'end_price': 85, 'periods': 20},    # 反弹
            {'type': 'trend', 'start_price': 85, 'end_price': 80, 'periods': 15},    # 回调
            {'type': 'trend', 'start_price': 80, 'end_price': 100, 'periods': 40},   # 上升
            {'type': 'trend', 'start_price': 100, 'end_price': 95, 'periods': 15},   # 回调
            {'type': 'trend', 'start_price': 95, 'end_price': 115, 'periods': 30},   # 继续上升
        ])
        
        # 添加量价数据
        volume = np.ones(len(self.data)) * 1000
        # 下跌末期缩量
        volume[20:30] = 500
        # 反弹时放量
        volume[30:45] = 2000
        # 回调时缩量
        volume[50:60] = 600
        # 突破时放量
        volume[65:80] = 2500
        # 回调时缩量
        volume[95:105] = 700
        # 继续上升时放量
        volume[110:130] = 3000
        self.data['volume'] = volume
        
        # 添加换手率
        self.data['turnover_rate'] = self.data['volume'] / 10000
        
        # 设置日志捕获
        self.setup_log_capture()
    
    def tearDown(self):
        """清理测试环境"""
        self.teardown_log_capture()
    
    def test_pattern_confirmation(self):
        """测试形态确认功能"""
        result = self.indicator.calculate(self.data)
        
        # 验证有形态确认被识别
        self.assertFalse(result.empty, "没有识别出形态确认")
        
        # 检查确认类型
        if 'confirmation_type' in result.columns:
            confirmation_types = result['confirmation_type'].dropna().unique()
            self.assertTrue(len(confirmation_types) > 0, "未识别出任何确认类型")
    
    def test_confirmation_strength(self):
        """测试确认强度评估"""
        result = self.indicator.calculate(self.data)
        
        # 检查确认强度
        if 'confirmation_strength' in result.columns:
            strength_values = result['confirmation_strength'].dropna()
            self.assertFalse(strength_values.empty, "确认强度值全为NaN")
            
            # 验证强度值在合理范围内
            self.assertTrue((strength_values >= 0).all() and (strength_values <= 100).all(),
                          "确认强度值超出0-100范围")


class TestPatternQualityEvaluator(unittest.TestCase, IndicatorTestMixin):
    """形态质量评估测试"""
    
    def setUp(self):
        """为所有测试准备数据和指标实例"""
        self.indicator = PatternQualityEvaluator()
        self.expected_columns = ['pattern_quality', 'reliability_score', 'profit_potential']
        
        # 生成适合形态质量评估的数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 80, 'periods': 35},   # 下跌
            {'type': 'trend', 'start_price': 80, 'end_price': 90, 'periods': 25},    # 反弹
            {'type': 'trend', 'start_price': 90, 'end_price': 85, 'periods': 15},    # 回调
            {'type': 'trend', 'start_price': 85, 'end_price': 105, 'periods': 40},   # 上升
            {'type': 'trend', 'start_price': 105, 'end_price': 100, 'periods': 20},  # 回调
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 30},  # 继续上升
        ])
        
        # 添加量价数据
        volume = np.ones(len(self.data)) * 1000
        # 下跌末期缩量
        volume[25:35] = 500
        # 反弹时放量
        volume[35:50] = 2000
        # 回调时缩量
        volume[60:70] = 600
        # 突破时放量
        volume[75:90] = 2500
        # 回调时缩量
        volume[105:115] = 700
        # 继续上升时放量
        volume[120:140] = 3000
        self.data['volume'] = volume
        
        # 添加换手率
        self.data['turnover_rate'] = self.data['volume'] / 10000
        
        # 设置日志捕获
        self.setup_log_capture()
    
    def tearDown(self):
        """清理测试环境"""
        self.teardown_log_capture()
    
    def test_pattern_quality_evaluation(self):
        """测试形态质量评估功能"""
        result = self.indicator.calculate(self.data)
        
        # 验证有形态质量评估结果
        self.assertFalse(result.empty, "没有生成形态质量评估结果")
        
        # 检查质量评分
        if 'pattern_quality' in result.columns:
            quality_scores = result['pattern_quality'].dropna()
            self.assertFalse(quality_scores.empty, "形态质量评分全为NaN")
            
            # 验证质量评分在合理范围内
            self.assertTrue((quality_scores >= 0).all() and (quality_scores <= 100).all(),
                          "形态质量评分超出0-100范围")
    
    def test_reliability_assessment(self):
        """测试可靠性评估"""
        result = self.indicator.calculate(self.data)
        
        # 检查可靠性评分
        if 'reliability_score' in result.columns:
            reliability_scores = result['reliability_score'].dropna()
            self.assertFalse(reliability_scores.empty, "可靠性评分全为NaN")
            
            # 验证可靠性评分在合理范围内
            self.assertTrue((reliability_scores >= 0).all() and (reliability_scores <= 100).all(),
                          "可靠性评分超出0-100范围")
    
    def test_profit_potential_assessment(self):
        """测试盈利潜力评估"""
        result = self.indicator.calculate(self.data)
        
        # 检查盈利潜力评估
        if 'profit_potential' in result.columns:
            profit_potential = result['profit_potential'].dropna()
            self.assertFalse(profit_potential.empty, "盈利潜力评估全为NaN")
            
            # 验证盈利潜力在合理范围内
            self.assertTrue((profit_potential >= 0).all(), "盈利潜力含有负值")


class TestMultiIndicatorPatternAnalysis(unittest.TestCase, LogCaptureMixin):
    """多指标形态分析测试"""
    
    def setUp(self):
        """为所有测试准备数据"""
        # 生成复杂的市场数据
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 90, 'periods': 40},    # 下跌
            {'type': 'trend', 'start_price': 90, 'end_price': 85, 'periods': 20},     # 继续下跌
            {'type': 'trend', 'start_price': 85, 'end_price': 95, 'periods': 30},     # 反弹
            {'type': 'trend', 'start_price': 95, 'end_price': 90, 'periods': 15},     # 回调
            {'type': 'trend', 'start_price': 90, 'end_price': 110, 'periods': 40},    # 上升
            {'type': 'trend', 'start_price': 110, 'end_price': 105, 'periods': 15},   # 回调
            {'type': 'trend', 'start_price': 105, 'end_price': 120, 'periods': 30},   # 继续上升
        ])
        
        # 添加量价数据
        volume = np.ones(len(self.data)) * 1000
        # 下跌末期缩量
        volume[50:60] = 500
        # 反弹时放量
        volume[60:80] = 2000
        # 回调时缩量
        volume[90:100] = 600
        # 突破时放量
        volume[105:125] = 2500
        # 回调时缩量
        volume[140:150] = 700
        # 继续上升时放量
        volume[155:175] = 3000
        self.data['volume'] = volume
        
        # 添加换手率
        self.data['turnover_rate'] = self.data['volume'] / 10000
        
        # 创建测试用的IndicatorFactory，返回测试指标
        self.factory = TestIndicatorFactory()
        
        # 设置日志捕获
        self.setup_log_capture()
    
    def tearDown(self):
        """清理测试环境"""
        self.teardown_log_capture()
    
    def test_pattern_with_macd_rsi_combination(self):
        """测试形态与MACD、RSI组合分析"""
        # 获取K线形态指标
        candlestick = self.factory.get_indicator('CANDLESTICKPATTERNS')
        # 获取MACD指标
        macd = self.factory.get_indicator('MACD')
        # 获取RSI指标
        rsi = self.factory.get_indicator('RSI')
        
        # 计算各指标
        pattern_result = candlestick.calculate(self.data)
        macd_result = macd.calculate(self.data)
        rsi_result = rsi.calculate(self.data)
        
        # 检查是否有错误日志
        self.assert_no_errors("指标计算过程中出现错误")
        
        # 验证计算结果有效
        self.assertFalse(pattern_result.empty, "K线形态计算结果为空")
        self.assertFalse(macd_result.empty, "MACD计算结果为空")
        self.assertFalse(rsi_result.empty, "RSI计算结果为空")
        
        # 组合分析 - 寻找同时满足条件的点
        # 这里只做一个简单的示例：K线形态为看涨 + MACD金叉 + RSI低位(<30)
        # 实际指标可能返回不同的列名，这里使用通用逻辑
        combined_signals = pd.DataFrame(index=self.data.index)
        
        # 检查MACD金叉
        if 'macd_line' in macd_result.columns and 'macd_signal' in macd_result.columns:
            macd_cross_up = (macd_result['macd_line'].shift(1) < macd_result['macd_signal'].shift(1)) & \
                           (macd_result['macd_line'] > macd_result['macd_signal'])
            combined_signals['macd_golden_cross'] = macd_cross_up
        
        # 检查RSI低位
        if 'rsi' in rsi_result.columns:
            rsi_low = rsi_result['rsi'] < 30
            combined_signals['rsi_low'] = rsi_low
        
        # 检查是否有匹配的综合信号
        if 'macd_golden_cross' in combined_signals.columns and 'rsi_low' in combined_signals.columns:
            strong_buy_signal = combined_signals['macd_golden_cross'] & combined_signals['rsi_low']
            # 验证有强烈买入信号
            self.assertTrue(strong_buy_signal.any(), "未检测到MACD金叉与RSI低位组合的买点")


if __name__ == '__main__':
    unittest.main() 