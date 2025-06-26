"""
ZXM指标信号语义验证测试模块

专门测试ZXM指标的信号生成逻辑是否符合业务语义
解决测试覆盖盲区：验证指标输出值与信号生成的语义一致性
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入需要测试的ZXM指标
from indicators.zxm.buy_point_indicators import ZXMTurnover, ZXMVolumeShrink, ZXMBSAbsorb
from indicators.zxm.trend_indicators import ZXMDailyTrendUp, ZXMWeeklyTrendUp
from indicators.zxm.elasticity_indicators import AmplitudeElasticity, ZXMRiseElasticity
from indicators.zxm.score_indicators import ZXMElasticityScore, ZXMBuyPointScore, StockScoreCalculator
from indicators.zxm.selection_model import SelectionModel


class TestZXMSignalSemanticValidation(unittest.TestCase):
    """ZXM指标信号语义验证测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 生成特定语义的测试数据，而不是随机数据
        self.test_scenarios = self._generate_semantic_test_scenarios()
    
    def _generate_semantic_test_scenarios(self):
        """生成语义测试场景"""
        scenarios = {}
        
        # 场景1：换手率测试数据
        scenarios['turnover_high'] = self._create_high_turnover_scenario()
        scenarios['turnover_low'] = self._create_low_turnover_scenario()
        
        # 场景2：缩量测试数据
        scenarios['volume_shrink'] = self._create_volume_shrink_scenario()
        scenarios['volume_expand'] = self._create_volume_expand_scenario()
        
        # 场景3：吸筹测试数据
        scenarios['absorb_strong'] = self._create_strong_absorb_scenario()
        scenarios['absorb_weak'] = self._create_weak_absorb_scenario()
        
        # 场景4：趋势测试数据
        scenarios['trend_up'] = self._create_trend_up_scenario()
        scenarios['trend_down'] = self._create_trend_down_scenario()
        
        # 场景5：弹性测试数据
        scenarios['high_elasticity'] = self._create_high_elasticity_scenario()
        scenarios['low_elasticity'] = self._create_low_elasticity_scenario()
        
        return scenarios
    
    def _create_high_turnover_scenario(self):
        """创建高换手率场景：换手率持续>0.7%"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100,
            'turnover_rate': [1.5] * 100,  # 持续高换手率
        })
        return data
    
    def _create_low_turnover_scenario(self):
        """创建低换手率场景：换手率持续<0.7%"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': [1000000] * 100,
            'turnover_rate': [0.3] * 100,  # 持续低换手率
        })
        return data
    
    def _create_volume_shrink_scenario(self):
        """创建缩量场景：成交量持续缩减"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # 创建递减的成交量序列
        volumes = [1000000 * (0.8 ** i) for i in range(100)]
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': volumes,
            'turnover_rate': [0.5] * 100,
        })
        return data
    
    def _create_volume_expand_scenario(self):
        """创建放量场景：成交量持续放大"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        # 创建递增的成交量序列
        volumes = [1000000 * (1.2 ** i) for i in range(100)]
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [100] * 100,
            'volume': volumes,
            'turnover_rate': [0.5] * 100,
        })
        return data
    
    def _create_strong_absorb_scenario(self):
        """创建强吸筹场景：满足吸筹条件的数据"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='30min')  # 30分钟数据
        # 创建满足吸筹条件的价格和成交量数据
        prices = []
        volumes = []
        base_price = 100
        base_volume = 1000000
        
        for i in range(200):
            # 价格小幅波动，模拟吸筹过程
            price = base_price + np.sin(i * 0.1) * 2
            # 成交量适度放大
            volume = base_volume * (1 + 0.5 * np.random.random())
            prices.append(price)
            volumes.append(volume)
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': volumes,
        })
        return data
    
    def _create_weak_absorb_scenario(self):
        """创建弱吸筹场景：不满足吸筹条件的数据"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='30min')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 200,
            'high': [101] * 200,
            'low': [99] * 200,
            'close': [100] * 200,
            'volume': [500000] * 200,  # 低成交量
        })
        return data
    
    def _create_trend_up_scenario(self):
        """创建上升趋势场景：均线持续上移"""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        # 创建上升趋势的价格序列
        prices = [100 + i * 0.5 for i in range(150)]  # 持续上涨
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 150,
        })
        return data
    
    def _create_trend_down_scenario(self):
        """创建下降趋势场景：均线持续下移"""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        # 创建下降趋势的价格序列
        prices = [100 - i * 0.3 for i in range(150)]  # 持续下跌
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 150,
        })
        return data
    
    def _create_high_elasticity_scenario(self):
        """创建高弹性场景：有大振幅或大涨幅"""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        prices = [100] * 150
        highs = [100] * 150
        lows = [100] * 150
        
        # 在第50天和第100天设置大振幅
        prices[50] = 110  # 10%涨幅
        highs[50] = 115
        lows[50] = 105
        
        prices[100] = 90   # 大跌后
        highs[100] = 95
        lows[100] = 85     # 大振幅
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [1000000] * 150,
        })
        return data
    
    def _create_low_elasticity_scenario(self):
        """创建低弹性场景：无大振幅或大涨幅"""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 150,
            'high': [101] * 150,  # 小振幅
            'low': [99] * 150,
            'close': [100] * 150,
            'volume': [1000000] * 150,
        })
        return data
    
    def test_zxm_turnover_semantic_consistency(self):
        """测试ZXM换手率指标的语义一致性"""
        indicator = ZXMTurnover()
        
        # 测试高换手率场景
        high_turnover_data = self.test_scenarios['turnover_high']
        result = indicator.calculate(high_turnover_data)
        
        # 验证语义：换手率>0.7时，XG应该为True，buy_signal应该为True
        high_turnover_rows = result[result['Turnover'] > 0.7]
        self.assertTrue(len(high_turnover_rows) > 0, "应该有高换手率的数据")
        self.assertTrue(high_turnover_rows['XG'].all(), "高换手率时XG应该为True")
        self.assertTrue(high_turnover_rows['buy_signal'].all(), "高换手率时buy_signal应该为True")
        
        # 测试低换手率场景
        low_turnover_data = self.test_scenarios['turnover_low']
        result = indicator.calculate(low_turnover_data)
        
        # 验证语义：换手率<0.7时，XG应该为False，buy_signal应该为False
        low_turnover_rows = result[result['Turnover'] < 0.7]
        self.assertTrue(len(low_turnover_rows) > 0, "应该有低换手率的数据")
        self.assertFalse(low_turnover_rows['XG'].any(), "低换手率时XG应该为False")
        self.assertFalse(low_turnover_rows['buy_signal'].any(), "低换手率时buy_signal应该为False")
        
        print("✅ ZXM换手率指标语义验证通过")
    
    def test_zxm_volume_shrink_semantic_consistency(self):
        """测试ZXM缩量指标的语义一致性"""
        indicator = ZXMVolumeShrink()
        
        # 测试缩量场景
        shrink_data = self.test_scenarios['volume_shrink']
        result = indicator.calculate(shrink_data)
        
        # 验证语义：缩量时（VOL_RATIO < 0.9），XG应该为True，buy_signal应该为True
        shrink_rows = result[(result['VOL_RATIO'] < 0.9) & (~result['VOL_RATIO'].isna())]
        if len(shrink_rows) > 0:
            self.assertTrue(shrink_rows['XG'].all(), "缩量时XG应该为True")
            self.assertTrue(shrink_rows['buy_signal'].all(), "缩量时buy_signal应该为True")
        
        print("✅ ZXM缩量指标语义验证通过")

    def test_zxm_daily_trend_up_semantic_consistency(self):
        """测试ZXM日线上移趋势指标的语义一致性"""
        indicator = ZXMDailyTrendUp()

        # 测试上升趋势场景
        trend_up_data = self.test_scenarios['trend_up']
        result = indicator.calculate(trend_up_data)

        # 验证语义：当均线上移时，XG应该为True，buy_signal应该为True
        trend_up_rows = result[result['XG'] == True]
        if len(trend_up_rows) > 0:
            self.assertTrue(trend_up_rows['buy_signal'].all(), "均线上移时buy_signal应该为True")

        # 测试下降趋势场景
        trend_down_data = self.test_scenarios['trend_down']
        result = indicator.calculate(trend_down_data)

        # 验证语义：当均线下移时，XG应该为False，buy_signal应该为False
        trend_down_rows = result[result['XG'] == False]
        if len(trend_down_rows) > 0:
            self.assertFalse(trend_down_rows['buy_signal'].any(), "均线下移时buy_signal应该为False")

        print("✅ ZXM日线上移趋势指标语义验证通过")

    def test_zxm_amplitude_elasticity_semantic_consistency(self):
        """测试ZXM振幅弹性指标的语义一致性"""
        indicator = AmplitudeElasticity()

        # 测试高弹性场景
        high_elasticity_data = self.test_scenarios['high_elasticity']
        result = indicator.calculate(high_elasticity_data)

        # 验证语义：有弹性时，XG应该为True，buy_signal应该为True
        elasticity_rows = result[result['XG'] == True]
        if len(elasticity_rows) > 0:
            self.assertTrue(elasticity_rows['buy_signal'].all(), "有振幅弹性时buy_signal应该为True")

        # 测试低弹性场景
        low_elasticity_data = self.test_scenarios['low_elasticity']
        result = indicator.calculate(low_elasticity_data)

        # 验证语义：无弹性时，XG应该为False，buy_signal应该为False
        no_elasticity_rows = result[result['XG'] == False]
        if len(no_elasticity_rows) > 0:
            self.assertFalse(no_elasticity_rows['buy_signal'].any(), "无振幅弹性时buy_signal应该为False")

        print("✅ ZXM振幅弹性指标语义验证通过")

    def test_zxm_rise_elasticity_semantic_consistency(self):
        """测试ZXM涨幅弹性指标的语义一致性"""
        indicator = ZXMRiseElasticity()

        # 测试高弹性场景
        high_elasticity_data = self.test_scenarios['high_elasticity']
        result = indicator.calculate(high_elasticity_data)

        # 验证语义：有涨幅弹性时，XG应该为True，buy_signal应该为True
        elasticity_rows = result[result['XG'] == True]
        if len(elasticity_rows) > 0:
            self.assertTrue(elasticity_rows['buy_signal'].all(), "有涨幅弹性时buy_signal应该为True")

        print("✅ ZXM涨幅弹性指标语义验证通过")

    def test_zxm_elasticity_score_semantic_consistency(self):
        """测试ZXM弹性评分指标的语义一致性"""
        indicator = ZXMElasticityScore(threshold=75)

        # 测试高弹性场景
        high_elasticity_data = self.test_scenarios['high_elasticity']
        result = indicator.calculate(high_elasticity_data)

        # 验证语义：当弹性评分>=75时，Signal应该为True，buy_signal应该为True
        high_score_rows = result[result['ElasticityScore'] >= 75]
        if len(high_score_rows) > 0:
            self.assertTrue(high_score_rows['Signal'].all(), "弹性评分>=75时Signal应该为True")
            self.assertTrue(high_score_rows['buy_signal'].all(), "弹性评分>=75时buy_signal应该为True")

        # 测试低弹性场景
        low_elasticity_data = self.test_scenarios['low_elasticity']
        result = indicator.calculate(low_elasticity_data)

        # 验证语义：当弹性评分<75时，Signal应该为False，buy_signal应该为False
        low_score_rows = result[result['ElasticityScore'] < 75]
        if len(low_score_rows) > 0:
            self.assertFalse(low_score_rows['Signal'].any(), "弹性评分<75时Signal应该为False")
            self.assertFalse(low_score_rows['buy_signal'].any(), "弹性评分<75时buy_signal应该为False")

        print("✅ ZXM弹性评分指标语义验证通过")

    def test_zxm_buypoint_score_semantic_consistency(self):
        """测试ZXM买点评分指标的语义一致性"""
        indicator = ZXMBuyPointScore(threshold=75)

        # 测试高评分场景（使用上升趋势数据，更可能触发买点）
        trend_up_data = self.test_scenarios['trend_up']
        result = indicator.calculate(trend_up_data)

        # 验证语义：当买点评分>=75时，Signal应该为True，buy_signal应该为True
        high_score_rows = result[result['BuyPointScore'] >= 75]
        if len(high_score_rows) > 0:
            self.assertTrue(high_score_rows['Signal'].all(), "买点评分>=75时Signal应该为True")
            self.assertTrue(high_score_rows['buy_signal'].all(), "买点评分>=75时buy_signal应该为True")

        # 验证语义：当买点评分<75时，Signal应该为False，buy_signal应该为False
        low_score_rows = result[result['BuyPointScore'] < 75]
        if len(low_score_rows) > 0:
            self.assertFalse(low_score_rows['Signal'].any(), "买点评分<75时Signal应该为False")
            self.assertFalse(low_score_rows['buy_signal'].any(), "买点评分<75时buy_signal应该为False")

        print("✅ ZXM买点评分指标语义验证通过")

    def test_stock_score_calculator_semantic_consistency(self):
        """测试股票综合评分指标的语义一致性"""
        indicator = StockScoreCalculator()

        # 测试高评分场景
        trend_up_data = self.test_scenarios['trend_up']
        result = indicator.calculate(trend_up_data)

        # 验证语义：当总分>70时，BuySignal应该为True，buy_signal应该为True
        high_score_rows = result[result['TotalScore'] > 70]
        if len(high_score_rows) > 0:
            self.assertTrue(high_score_rows['BuySignal'].all(), "总分>70时BuySignal应该为True")
            self.assertTrue(high_score_rows['buy_signal'].all(), "总分>70时buy_signal应该为True")

        # 验证语义：当总分<30时，SellSignal应该为True，sell_signal应该为True
        low_score_rows = result[result['TotalScore'] < 30]
        if len(low_score_rows) > 0:
            self.assertTrue(low_score_rows['SellSignal'].all(), "总分<30时SellSignal应该为True")
            self.assertTrue(low_score_rows['sell_signal'].all(), "总分<30时sell_signal应该为True")

        print("✅ 股票综合评分指标语义验证通过")

    def test_selection_model_semantic_consistency(self):
        """测试ZXM选股模型的语义一致性"""
        indicator = SelectionModel()

        # 测试选股场景
        trend_up_data = self.test_scenarios['trend_up']
        result = indicator.calculate(trend_up_data)

        # 验证语义：当FinalSelect为True时，buy_signal应该为True
        selected_rows = result[result['FinalSelect'] == True]
        if len(selected_rows) > 0:
            self.assertTrue(selected_rows['buy_signal'].all(), "FinalSelect=True时buy_signal应该为True")

        # 验证语义：当FinalSelect为False时，buy_signal应该为False
        not_selected_rows = result[result['FinalSelect'] == False]
        if len(not_selected_rows) > 0:
            self.assertFalse(not_selected_rows['buy_signal'].any(), "FinalSelect=False时buy_signal应该为False")

        print("✅ ZXM选股模型语义验证通过")


if __name__ == '__main__':
    unittest.main()
