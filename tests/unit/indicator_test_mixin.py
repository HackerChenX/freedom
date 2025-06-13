"""
指标测试混入模块

提供指标测试类的通用功能和方法
"""

import pandas as pd
import numpy as np
from tests.helper.log_capture import LogCaptureMixin
from typing import List, Dict, Any, Union
import pytest


class IndicatorTestMixin:
    """
    指标测试混入类，提供指标测试的通用功能
    
    使用此混入类的测试类需要定义以下属性：
    - indicator: 要测试的指标实例
    - data: 用于测试的数据
    - expected_columns: 预期输出的列名列表
    """
    
    def _ensure_stock_info_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        确保数据包含所有必需的StockInfo字段
        
        Args:
            data: 输入数据
            
        Returns:
            包含所有必需字段的数据
        """
        # 检查索引是否为日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                # 创建日期索引
                data.index = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
        
        # 添加必需字段
        required_fields = {
            'code': '000001',
            'name': '测试股票',
            'level': 'D',
            'industry': '软件服务',
            'datetime_value': data.index,
            'seq': range(len(data)),
            'turnover_rate': 5.0,
            'price_change': 0.0,
            'price_range': 2.0
        }
        
        for field, default_value in required_fields.items():
            if field not in data.columns:
                if field == 'price_change':
                    data[field] = data['close'].diff().fillna(0)
                elif field == 'price_range':
                    data[field] = (data['high'] - data['low']) / data['close'] * 100
                elif field == 'datetime_value':
                    data[field] = data.index
                elif field == 'seq':
                    data[field] = range(len(data))
                else:
                    data[field] = default_value
        
        return data
    
    def _verify_indicator_basics(self, indicator, expected_properties: List[str] = None):
        """
        验证指标的基本属性
        
        Args:
            indicator: 要验证的指标
            expected_properties: 预期的属性列表
        """
        # 验证基本属性
        assert indicator is not None, "指标不应为None"
        assert hasattr(indicator, 'calculate'), "指标应有calculate方法"
        
        # 验证预期属性
        if expected_properties:
            for prop in expected_properties:
                assert hasattr(indicator, prop), f"指标应有{prop}属性"
    
    def _verify_calculation_result(self, result: Union[pd.DataFrame, Dict], 
                                  expected_columns: List[str] = None):
        """
        验证计算结果
        
        Args:
            result: 计算结果
            expected_columns: 预期的列名列表
        """
        # 验证结果不为None
        assert result is not None, "计算结果不应为None"
        
        # 如果结果是DataFrame
        if isinstance(result, pd.DataFrame):
            # 验证预期列
            if expected_columns:
                for col in expected_columns:
                    assert col in result.columns, f"结果应包含{col}列"
        
        # 如果结果是字典
        elif isinstance(result, dict):
            # 验证预期键
            if expected_columns:
                for key in expected_columns:
                    assert key in result, f"结果应包含{key}键"
    
    def _verify_raw_score(self, score: pd.Series):
        """
        验证原始评分
        
        Args:
            score: 评分Series
        """
        # 验证评分为Series
        assert isinstance(score, pd.Series), "评分应为Series"
        
        # 验证评分在0-100范围内
        valid_scores = score.dropna()
        assert all(0 <= s <= 100 for s in valid_scores), "评分应在0-100范围内"
    
    def _verify_signals(self, signals: List[Dict[str, Any]], min_expected: int = 0):
        """
        验证信号
        
        Args:
            signals: 信号列表
            min_expected: 最小预期信号数
        """
        # 验证信号为列表
        assert isinstance(signals, list), "信号应为列表"
        
        # 验证信号数量
        if min_expected > 0:
            assert len(signals) >= min_expected, f"信号数量应不少于{min_expected}"
        
        # 验证信号格式
        if signals:
            for signal in signals:
                assert isinstance(signal, dict), "信号项应为字典"
                required_keys = ['indicator', 'buy_signal', 'sell_signal', 'score']
                for key in required_keys:
                    assert key in signal, f"信号应包含{key}键"
    
    def _verify_pattern_result(self, result: Dict[str, List[int]], min_patterns: int = 0):
        """
        验证形态识别结果
        
        Args:
            result: 形态识别结果
            min_patterns: 最小预期形态数
        """
        # 验证结果为字典
        assert isinstance(result, dict), "形态识别结果应为字典"
        
        # 验证形态数量
        pattern_count = sum(len(positions) for positions in result.values())
        if min_patterns > 0:
            assert pattern_count >= min_patterns, f"识别出的形态数量应不少于{min_patterns}"
        
        # 验证形态位置
        for pattern, positions in result.items():
            assert isinstance(positions, list), f"{pattern}的位置应为列表"
            for pos in positions:
                assert isinstance(pos, int), "位置应为整数"
    
    def _mock_stock_data(self, periods: int = 100, 
                        start_price: float = 100.0, 
                        trend: str = 'up') -> pd.DataFrame:
        """
        创建模拟股票数据
        
        Args:
            periods: 周期数
            start_price: 起始价格
            trend: 趋势类型，可选'up'(上涨),'down'(下跌),'flat'(横盘)
            
        Returns:
            模拟股票数据
        """
        # 确定价格变化
        if trend == 'up':
            price_change = 0.01
        elif trend == 'down':
            price_change = -0.01
        else:
            price_change = 0
        
        # 生成价格序列
        noise = np.random.normal(0, 0.02, periods)
        changes = price_change + noise
        
        # 累积变化
        cum_changes = np.cumprod(1 + changes)
        close_prices = start_price * cum_changes
        
        # 生成OHLC数据
        data = pd.DataFrame({
            'close': close_prices,
            'open': close_prices * (1 - 0.01 + 0.02 * np.random.random(periods)),
            'high': close_prices * (1 + 0.01 + 0.02 * np.random.random(periods)),
            'low': close_prices * (1 - 0.01 - 0.02 * np.random.random(periods)),
            'volume': np.random.randint(1000, 10000, periods)
        })
        
        # 创建日期索引
        data.index = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        
        # 添加StockInfo字段
        self._ensure_stock_info_fields(data)
        
        return data
    
    def test_calculation_runs_without_error(self):
        """测试指标的 calculate 方法是否能无错运行。"""
        try:
            self.indicator.calculate(self.data)
        except Exception as e:
            pytest.fail(f"指标 {self.indicator.__class__.__name__} 计算失败，错误: {e}")
    
    def test_returns_dataframe(self):
        """测试 calculate 方法是否返回一个 pandas DataFrame。"""
        result = self.indicator.calculate(self.data)
        assert isinstance(result, pd.DataFrame), "计算结果不是 DataFrame"
    
    def test_output_has_expected_columns(self):
        """测试计算结果是否包含所有预期的列。"""
        result = self.indicator.calculate(self.data)
        
        for col in self.expected_columns:
            assert col in result.columns, f"结果中缺少预期列: {col}"
    
    def test_output_has_no_unexpected_all_nan_columns(self):
        """测试输出中不应有完全由NaN组成的意外列。"""
        result = self.indicator.calculate(self.data)
        
        # 只检查预期的列
        for col in self.expected_columns:
            if col in result.columns:
                assert not result[col].isna().all(), f"预期列 '{col}' 全是 NaN"
    
    def test_calculate_with_missing_columns(self):
        """测试当缺少必需列时的处理情况。"""
        if not hasattr(self.indicator, 'REQUIRED_COLUMNS'):
            pytest.skip(f"指标 {self.indicator.__class__.__name__} 没有 REQUIRED_COLUMNS 属性，跳过此测试")

        required = self.indicator.REQUIRED_COLUMNS
        if not required:
            pytest.skip(f"指标 {self.indicator.__class__.__name__} 的 REQUIRED_COLUMNS 为空，跳过此测试")

        cols_to_keep = self.data.columns.tolist()
        col_to_remove = None
        for col in required:
            if col in cols_to_keep:
                col_to_remove = col
                break
        
        if not col_to_remove:
            pytest.skip(f"无法从数据中找到一个可移除的必需列，跳过测试")

        cols_to_keep.remove(col_to_remove)
        minimal_data = self.data[cols_to_keep].copy()
        
        # 有些指标实现会抛出异常，有些则会返回带有警告的结果
        # 我们只需确保两种情况都能正确处理
        
        # 方法1: 捕获可能的异常
        try:
            result = self.indicator.calculate(minimal_data)
            # 如果没有异常，结果应该是有效的DataFrame
            assert isinstance(result, pd.DataFrame), "计算结果应为DataFrame"
        except ValueError as e:
            # 如果有异常，应该提到缺少的列
            assert col_to_remove in str(e), f"错误信息应包含缺少的列名: {col_to_remove}"
    
    def test_patterns_run_without_error(self):
        """测试指标的 get_patterns 方法是否能无错运行。"""
        if not hasattr(self.indicator, 'get_patterns'):
            pytest.skip(f"指标 {self.indicator.__class__.__name__} 没有 get_patterns 方法")
        
        try:
            self.indicator.get_patterns(self.data)
        except Exception as e:
            pytest.fail(f"指标 {self.indicator.__class__.__name__} 的 get_patterns 失败，错误: {e}")
    
    def test_patterns_return_valid_type(self):
        """测试 get_patterns 方法是否返回一个有效的类型。"""
        if not hasattr(self.indicator, 'get_patterns'):
            pytest.skip(f"指标 {self.indicator.__class__.__name__} 没有 get_patterns 方法")
    
        all_patterns = self.indicator.get_patterns(self.data)
        assert isinstance(all_patterns, pd.DataFrame), "get_patterns 的返回类型不是 DataFrame"
    
    def test_no_errors_during_calculation(self):
        """测试在计算过程中是否记录了ERROR级别的日志。"""
        # 清除之前测试产生的日志
        if hasattr(self, 'clear_logs'):
            self.clear_logs()
        self.indicator.calculate(self.data)
        self.assert_no_logs('ERROR')
    
    def test_no_errors_during_pattern_detection(self):
        """测试在形态检测过程中是否记录了ERROR级别的日志。"""
        if not hasattr(self.indicator, 'get_patterns'):
            pytest.skip(f"指标 {self.indicator.__class__.__name__} 没有 get_patterns 方法")

        # 清除之前测试产生的日志
        if hasattr(self, 'clear_logs'):
            self.clear_logs()
        self.indicator.get_patterns(self.data)
        self.assert_no_logs('ERROR')