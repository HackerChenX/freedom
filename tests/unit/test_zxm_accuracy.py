"""
ZXM体系指标准确性测试
基于ZXM体系3.0版权威文档验证指标计算准确性
"""
import unittest
import pandas as pd
import numpy as np
from indicators.zxm_absorb import ZXMAbsorb
from indicators.zxm_washplate import ZXMWashPlate
from tests.helper.data_generator import TestDataGenerator
from tests.helper.log_capture import LogCaptureMixin


class TestZXMAccuracy(unittest.TestCase, LogCaptureMixin):
    """ZXM体系指标准确性测试类"""
    
    def setUp(self):
        """设置测试环境"""
        LogCaptureMixin.setUp(self)
        
        self.zxm_absorb = ZXMAbsorb()
        self.zxm_washplate = ZXMWashPlate()
        
        # 生成足够长的测试数据用于验证计算准确性
        self.data = TestDataGenerator.generate_price_sequence([
            {'type': 'trend', 'start_price': 100, 'end_price': 120, 'periods': 150}
        ])
    
    def tearDown(self):
        """清理日志捕获器"""
        LogCaptureMixin.tearDown(self)
    
    def test_zxm_absorb_v11_formula_accuracy(self):
        """测试ZXM吸筹V11公式计算准确性"""
        self.clear_logs()
        
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证V11计算结果
        self.assertIn('V11', result.columns)
        v11_values = result['V11'].dropna()
        
        if len(v11_values) > 0:
            # V11值应该在合理范围内（基于KDJ衍生）
            self.assertTrue(all(-50 <= v <= 150 for v in v11_values), 
                          f"V11值应在-50到150范围内，实际范围: {v11_values.min():.2f} - {v11_values.max():.2f}")
    
    def test_zxm_absorb_v12_momentum_accuracy(self):
        """测试ZXM吸筹V12动量指标计算准确性"""
        self.clear_logs()
        
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证V12计算结果
        self.assertIn('V12', result.columns)
        v12_values = result['V12'].dropna()
        
        if len(v12_values) > 0:
            # V12是动量指标，应该有正负值
            self.assertTrue(any(v > 0 for v in v12_values), "V12应该有正值")
            self.assertTrue(any(v < 0 for v in v12_values), "V12应该有负值")
    
    def test_zxm_absorb_aa_bb_conditions(self):
        """测试ZXM吸筹AA和BB条件准确性"""
        self.clear_logs()
        
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证AA和BB条件
        self.assertIn('AA', result.columns)
        self.assertIn('BB', result.columns)
        self.assertIn('EMA_V11_3', result.columns)
        
        # 验证AA条件：EMA_V11_3 <= 13
        for i in range(len(result)):
            if pd.notna(result['EMA_V11_3'].iloc[i]):
                ema_v11 = result['EMA_V11_3'].iloc[i]
                aa_condition = result['AA'].iloc[i]
                expected_aa = ema_v11 <= 13
                
                self.assertEqual(aa_condition, expected_aa, 
                               f"第{i}行AA条件不正确: EMA_V11_3={ema_v11:.2f}, AA={aa_condition}, 期望={expected_aa}")
        
        # 验证BB条件：EMA_V11_3 <= 13 AND V12 > 13
        for i in range(len(result)):
            if pd.notna(result['EMA_V11_3'].iloc[i]) and pd.notna(result['V12'].iloc[i]):
                ema_v11 = result['EMA_V11_3'].iloc[i]
                v12 = result['V12'].iloc[i]
                bb_condition = result['BB'].iloc[i]
                expected_bb = (ema_v11 <= 13) and (v12 > 13)
                
                self.assertEqual(bb_condition, expected_bb,
                               f"第{i}行BB条件不正确: EMA_V11_3={ema_v11:.2f}, V12={v12:.2f}, BB={bb_condition}, 期望={expected_bb}")
    
    def test_zxm_absorb_xg_calculation_accuracy(self):
        """测试ZXM吸筹XG计算准确性"""
        self.clear_logs()
        
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证XG计算
        self.assertIn('XG', result.columns)
        
        # 手动验证XG计算逻辑
        for i in range(6, len(result)):
            if all(pd.notna(result[col].iloc[j]) for col in ['AA', 'BB'] for j in range(i-5, i+1)):
                # 计算近6天内AA或BB条件满足的次数
                aa_bb_count = 0
                for j in range(i-5, i+1):
                    if result['AA'].iloc[j] or result['BB'].iloc[j]:
                        aa_bb_count += 1
                
                actual_xg = result['XG'].iloc[i]
                self.assertEqual(actual_xg, aa_bb_count,
                               f"第{i}行XG计算不正确: 实际={actual_xg}, 期望={aa_bb_count}")
    
    def test_zxm_absorb_buy_signal_accuracy(self):
        """测试ZXM吸筹BUY信号准确性"""
        self.clear_logs()
        
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证BUY信号
        self.assertIn('BUY', result.columns)
        
        # 验证BUY信号逻辑：XG >= 3
        for i in range(len(result)):
            if pd.notna(result['XG'].iloc[i]):
                xg_value = result['XG'].iloc[i]
                buy_signal = result['BUY'].iloc[i]
                expected_buy = xg_value >= 3
                
                self.assertEqual(buy_signal, expected_buy,
                               f"第{i}行BUY信号不正确: XG={xg_value}, BUY={buy_signal}, 期望={expected_buy}")
    
    def test_zxm_absorb_buy_point_four_elements(self):
        """测试ZXM吸筹买点四要素验证"""
        self.clear_logs()
        
        # 先计算指标
        result = self.zxm_absorb.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 测试买点四要素验证函数
        for i in range(120, len(self.data), 10):  # 每10个点测试一次
            four_elements = self.zxm_absorb.validate_buy_point_four_elements(self.data, i)
            
            # 验证返回结构
            expected_keys = ['trend_intact', 'volume_shrink', 'pullback_support', 'bs_signal']
            for key in expected_keys:
                self.assertIn(key, four_elements, f"缺少买点要素: {key}")
                self.assertIsInstance(four_elements[key], bool, f"{key}应该是布尔值")
    
    def test_zxm_washplate_shock_wash_accuracy(self):
        """测试ZXM洗盘横盘震荡洗盘计算准确性"""
        self.clear_logs()
        
        result = self.zxm_washplate.calculate(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证横盘震荡洗盘
        shock_wash_col = "横盘震荡洗盘"
        self.assertIn(shock_wash_col, result.columns)
        
        # 手动验证横盘震荡洗盘逻辑
        window = 10
        for i in range(window, len(result)):
            # 价格区间相对波动小于7%
            price_window = self.data['close'].iloc[i-window:i]
            price_range = price_window.max() - price_window.min()
            price_range_ratio = price_range / price_window.min()
            
            # 成交量波动大于2倍
            vol_window = self.data['volume'].iloc[i-window:i]
            vol_range_ratio = vol_window.max() / vol_window.min()
            
            expected_shock = (price_range_ratio < 0.07) and (vol_range_ratio > 2)
            actual_shock = result[shock_wash_col].iloc[i]
            
            self.assertEqual(actual_shock, expected_shock,
                           f"第{i}行横盘震荡洗盘不正确: 价格波动={price_range_ratio:.4f}, 成交量比例={vol_range_ratio:.2f}")
    
    def test_zxm_washplate_score_calculation_accuracy(self):
        """测试ZXM洗盘评分计算准确性"""
        self.clear_logs()
        
        score = self.zxm_washplate.calculate_raw_score(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证评分范围
        valid_scores = score.dropna()
        self.assertTrue(all(0 <= s <= 100 for s in valid_scores), 
                       f"评分应在0-100范围内，实际范围: {valid_scores.min():.2f} - {valid_scores.max():.2f}")
        
        # 验证基础分数为50
        result = self.zxm_washplate.calculate(self.data)
        no_wash_indices = []
        for i in range(len(result)):
            has_wash = False
            for wash_type in ['横盘震荡洗盘', '回调洗盘', '假突破洗盘', '时间洗盘', '连续阴线洗盘']:
                if wash_type in result.columns and result[wash_type].iloc[i]:
                    has_wash = True
                    break
            if not has_wash:
                no_wash_indices.append(i)
        
        # 没有洗盘形态的位置评分应该接近50
        if no_wash_indices:
            base_scores = score.iloc[no_wash_indices]
            self.assertTrue(all(45 <= s <= 55 for s in base_scores), 
                           "没有洗盘形态时评分应该接近50分")
    
    def test_no_errors_in_comprehensive_calculation(self):
        """测试综合计算过程中无ERROR日志"""
        self.clear_logs()
        
        # 执行ZXM吸筹指标完整计算流程
        absorb_result = self.zxm_absorb.calculate(self.data)
        absorb_score = self.zxm_absorb.calculate_raw_score(self.data)
        absorb_patterns = self.zxm_absorb.get_patterns(self.data)
        absorb_signals = self.zxm_absorb.generate_trading_signals(self.data)
        
        # 执行ZXM洗盘指标完整计算流程
        washplate_result = self.zxm_washplate.calculate(self.data)
        washplate_score = self.zxm_washplate.calculate_raw_score(self.data)
        washplate_patterns = self.zxm_washplate.get_patterns(self.data)
        washplate_signals = self.zxm_washplate.generate_trading_signals(self.data)
        
        # 验证无ERROR日志
        self.assert_no_logs('ERROR')
        
        # 验证所有结果都不为空
        self.assertIsInstance(absorb_result, pd.DataFrame)
        self.assertIsInstance(absorb_score, pd.Series)
        self.assertIsInstance(absorb_patterns, pd.DataFrame)
        self.assertIsInstance(absorb_signals, dict)
        
        self.assertIsInstance(washplate_result, pd.DataFrame)
        self.assertIsInstance(washplate_score, pd.Series)
        self.assertIsInstance(washplate_patterns, pd.DataFrame)
        self.assertIsInstance(washplate_signals, dict)


if __name__ == '__main__':
    unittest.main()
