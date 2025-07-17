"""
603359真实数据验证测试

使用603359股票的真实数据验证ZXM指标修复的有效性
这是对P0级修复的关键验证测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from db.unified_data_manager import get_unified_data_manager
from indicators.complete_indicator_registry import complete_registry
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level=get_config('logging.level', 'INFO'))
logger = get_logger(__name__)


class Test603359_real_data_validation(unittest.Test_case):
    """603359真实数据验证测试类"""
    
    def set_up_Validation_Test_603359_Real_Data_Validation(self):
        """设置测试环境"""
        self.data_manager = get_unified_data_manager()
        self.target_stock = "603359"
        self.target_date = "2025-05-12"
        
        # 获取真实数据
        try:
            # 获取30分钟数据用于ZXM_BS_ABSORB
            self.min30_data = self.data_manager.get_stock_data(
                stock_code=self.target_stock,
                end_date=self.target_date,
                period='30min',
                lookback_days=90
            )
            
            # 获取日线数据用于其他指标
            self.daily_data = self.data_manager.get_stock_data(
                stock_code=self.target_stock,
                end_date=self.target_date,
                period='daily',
                lookback_days=150
            )
            
            logger.info(f"获取到30分钟数据: {len(self.min30_data)} 条")
            logger.info(f"获取到日线数据: {len(self.daily_data)} 条")
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            self.skipTest(f"无法获取{self.target_stock}的数据: {e}")
    
    def test_zxm_bs_absorb_real_data_validation(self):
        """验证ZXM_BS_ABSORB指标在真实数据上的表现"""
        logger.info("=== 测试ZXM_BS_ABSORB指标真实数据验证 ===")
        
        # 创建指标
        zxm_absorb = complete_registry.create_indicator('ZXM_BS_ABSORB')
        
        # 计算指标
        result = zxm_absorb.calculate(self.min30_data)
        
        # 分析目标日期的结果
        target_data = result[result['date'] == self.target_date]
        
        logger.info(f"目标日期 {self.target_date} 的ZXM_BS_ABSORB分析:")
        logger.info(f"数据条数: {len(target_data)}")
        
        if not target_data.empty:
            buy_signal_count = 0
            xg_positive_count = 0
            
            for idx, row in target_data.iterrows():
                time_str = row.get('time', '')
                xg = row.get('XG', 0)
                buy_signal = row.get('buy_signal', False)
                
                logger.info(f"  {time_str}: XG={xg}, buy_signal={buy_signal}")
                
                if xg > 0:
                    xg_positive_count += 1
                if buy_signal:
                    buy_signal_count += 1
            
            # 关键验证：XG > 0 的数量应该等于 buy_signal = True 的数量
            self.assert_equal(xg_positive_count, buy_signal_count, 
                           f"XG>0的数量({xg_positive_count})应该等于buy_signal=True的数量({buy_signal_count})")
            
            # 验证语义一致性：当XG > 0时，buy_signal应该为True
            xg_positive_rows = target_data[target_data['XG'] > 0]
            if len(xg_positive_rows) > 0:
                self.assertTrue(xg_positive_rows['buy_signal'].all(), 
                              "当XG > 0时，buy_signal应该全部为True")
            
            # 验证语义一致性：当XG = 0时，buy_signal应该为False
            xg_zero_rows = target_data[target_data['XG'] == 0]
            if len(xg_zero_rows) > 0:
                self.assertFalse(xg_zero_rows['buy_signal'].any(), 
                               "当XG = 0时，buy_signal应该全部为False")
            
            logger.info(f"✅ 语义验证通过: XG>0数量={xg_positive_count}, BUY信号数量={buy_signal_count}")
        else:
            logger.warning(f"目标日期 {self.target_date} 没有数据")
    
    def test_zxm_turnover_real_data_validation(self):
        """验证ZXM换手率指标在真实数据上的表现"""
        logger.info("=== 测试ZXM换手率指标真实数据验证 ===")
        
        # 创建指标
        zxm_turnover = complete_registry.create_indicator('ZXM_TURNOVER')
        
        # 计算指标
        result = zxm_turnover.calculate(self.daily_data)
        
        # 分析目标日期的结果
        target_data = result[result['date'] == self.target_date]
        
        if not target_data.empty:
            row = target_data.iloc[0]
            turnover = row.get('Turnover', 0)
            xg = row.get('XG', False)
            buy_signal = row.get('buy_signal', False)
            
            logger.info(f"目标日期 {self.target_date} 的ZXM换手率分析:")
            logger.info(f"  换手率: {turnover}")
            logger.info(f"  XG: {xg}")
            logger.info(f"  buy_signal: {buy_signal}")
            
            # 验证语义一致性
            if turnover > 0.7:
                self.assertTrue(xg, "换手率>0.7时，XG应该为True")
                self.assertTrue(buy_signal, "换手率>0.7时，buy_signal应该为True")
            else:
                self.assertFalse(xg, "换手率<=0.7时，XG应该为False")
                self.assertFalse(buy_signal, "换手率<=0.7时，buy_signal应该为False")
            
            logger.info("✅ ZXM换手率指标语义验证通过")
    
    def test_zxm_volume_shrink_real_data_validation(self):
        """验证ZXM缩量指标在真实数据上的表现"""
        logger.info("=== 测试ZXM缩量指标真实数据验证 ===")
        
        # 创建指标
        zxm_volume_shrink = complete_registry.create_indicator('ZXM_VOLUME_SHRINK')
        
        # 计算指标
        result = zxm_volume_shrink.calculate(self.daily_data)
        
        # 分析目标日期的结果
        target_data = result[result['date'] == self.target_date]
        
        if not target_data.empty:
            row = target_data.iloc[0]
            vol_ratio = row.get('VOL_RATIO', np.nan)
            xg = row.get('XG', False)
            buy_signal = row.get('buy_signal', False)
            
            logger.info(f"目标日期 {self.target_date} 的ZXM缩量分析:")
            logger.info(f"  量比: {vol_ratio}")
            logger.info(f"  XG: {xg}")
            logger.info(f"  buy_signal: {buy_signal}")
            
            # 验证语义一致性
            if not pd.isna(vol_ratio):
                if vol_ratio < 0.9:
                    self.assertTrue(xg, "量比<0.9时，XG应该为True")
                    self.assertTrue(buy_signal, "量比<0.9时，buy_signal应该为True")
                else:
                    self.assertFalse(xg, "量比>=0.9时，XG应该为False")
                    self.assertFalse(buy_signal, "量比>=0.9时，buy_signal应该为False")
            
            logger.info("✅ ZXM缩量指标语义验证通过")
    
    def test_zxm_daily_trend_up_real_data_validation(self):
        """验证ZXM日线上移趋势指标在真实数据上的表现"""
        logger.info("=== 测试ZXM日线上移趋势指标真实数据验证 ===")
        
        # 创建指标
        zxm_daily_trend = complete_registry.create_indicator('ZXM_DAILY_TREND_UP')
        
        # 计算指标
        result = zxm_daily_trend.calculate(self.daily_data)
        
        # 分析目标日期的结果
        target_data = result[result['date'] == self.target_date]
        
        if not target_data.empty:
            row = target_data.iloc[0]
            xg = row.get('XG', False)
            buy_signal = row.get('buy_signal', False)
            j1 = row.get('J1', False)
            j2 = row.get('J2', False)
            
            logger.info(f"目标日期 {self.target_date} 的ZXM日线上移趋势分析:")
            logger.info(f"  J1(60日均线上移): {j1}")
            logger.info(f"  J2(120日均线上移): {j2}")
            logger.info(f"  XG: {xg}")
            logger.info(f"  buy_signal: {buy_signal}")
            
            # 验证语义一致性：XG应该等于(J1 OR J2)
            expected_xg = j1 or j2
            self.assertEqual(xg, expected_xg, f"XG({xg})应该等于J1 OR J2({expected_xg})")
            
            # 验证信号一致性：buy_signal应该等于XG
            self.assertEqual(buy_signal, xg, f"buy_signal({buy_signal})应该等于XG({xg})")
            
            logger.info("✅ ZXM日线上移趋势指标语义验证通过")
    
    def test_comprehensive_signal_consistency(self):
        """综合测试所有修复指标的信号一致性"""
        logger.info("=== 综合信号一致性测试 ===")
        
        indicators_to_test = [
            ('ZXM_BS_ABSORB', self.min30_data),
            ('ZXM_TURNOVER', self.daily_data),
            ('ZXM_VOLUME_SHRINK', self.daily_data),
            ('ZXM_DAILY_TREND_UP', self.daily_data),
            ('ZXM_WEEKLY_TREND_UP', self.daily_data),
            ('ZXM_AMPLITUDE_ELASTICITY', self.daily_data),
            ('ZXM_RISE_ELASTICITY', self.daily_data),
        ]
        
        all_passed = True
        
        for indicator_name, data in indicators_to_test:
            try:
                indicator = complete_registry.create_indicator(indicator_name)
                result = indicator.calculate(data)
                
                # 验证所有信号字段都存在且类型正确
                self.assertIn('buy_signal', result.columns, f"{indicator_name}缺少buy_signal列")
                self.assertIn('sell_signal', result.columns, f"{indicator_name}缺少sell_signal列")
                self.assertIn('hold_signal', result.columns, f"{indicator_name}缺少hold_signal列")
                
                # 验证信号字段的数据类型
                self.assertEqual(result['buy_signal'].dtype, bool, f"{indicator_name}的buy_signal应该是布尔类型")
                self.assertEqual(result['sell_signal'].dtype, bool, f"{indicator_name}的sell_signal应该是布尔类型")
                self.assertEqual(result['hold_signal'].dtype, bool, f"{indicator_name}的hold_signal应该是布尔类型")
                
                logger.info(f"✅ {indicator_name} 信号一致性验证通过")
                
            except Exception as e:
                logger.error(f"❌ {indicator_name} 测试失败: {e}")
                all_passed = False
        
        self.assertTrue(all_passed, "所有指标的信号一致性测试应该通过")
        logger.info("✅ 综合信号一致性测试全部通过")


if __name__ == '__main__':
    unittest.main()
