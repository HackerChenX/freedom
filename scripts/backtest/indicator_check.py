#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Any, Optional

from formula import formula
from utils.logger import get_logger
from indicators.factory import IndicatorFactory
from scripts.backtest.unified_backtest import UnifiedBacktest

# 获取日志记录器
logger = get_logger(__name__)

class IndicatorChecker:
    """
    技术指标检查工具
    
    用于验证所有技术指标是否能正常工作，以及修复可能存在的问题
    """
    
    def __init__(self):
        """初始化指标检查器"""
        self.test_code = "000001"  # 使用上证指数进行测试
        
        # 获取最近60天的数据
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y%m%d")
        
        # 获取测试数据
        self.f = formula.Formula(self.test_code, start=start_date, end=end_date)
        
        # 准备测试数据
        self.test_df = pd.DataFrame({
            'date': self.f.dataDay.history['date'],
            'open': self.f.dataDay.open,
            'high': self.f.dataDay.high,
            'low': self.f.dataDay.low,
            'close': self.f.dataDay.close,
            'volume': self.f.dataDay.volume
        })
        
        # 获取所有支持的指标
        self.all_indicators = IndicatorFactory.get_supported_indicators()
        logger.info(f"系统支持的指标数量: {len(self.all_indicators)}")
        
    def check_all_indicators(self):
        """检查所有指标"""
        logger.info("开始检查所有指标...")
        # 检查各个指标
        self.check_ma_indicator()
        self.check_macd_indicator()
        self.check_kdj_indicator()
        self.check_rsi_indicator()
        self.check_boll_indicator()
        self.check_vol_indicator()
        self.check_v_shaped_reversal_indicator()
        self.check_trix_indicator()
        self.check_zxm_elasticity_score_indicator()
        self.check_zxm_buypoint_score_indicator()
        self.check_divergence_indicator()
        self.check_bias_indicator()
        self.check_sar_indicator()
        self.check_obv_indicator()
        self.check_dmi_indicator()
        self.check_wr_indicator()
        self.check_cci_indicator()
        self.check_roc_indicator()
        self.check_vosc_indicator()
        self.check_mfi_indicator()
        self.check_stochrsi_indicator()
        self.check_momentum_indicator()
        self.check_rsima_indicator()
        self.check_intraday_volatility_indicator()
        self.check_atr_indicator()
        self.check_emv_indicator()
        self.check_volume_ratio_indicator()
        logger.info("所有指标检查完成")
        
    def check_vol_indicator(self):
        """专门检查VOL指标的实现"""
        logger.info("检查VOL指标实现...")
        
        try:
            vol_indicator = IndicatorFactory.create_indicator("VOL")
            result = vol_indicator.compute(self.test_df)
            
            # 检查结果
            if 'vol' in result.columns and 'vol_ma5' in result.columns:
                logger.info("VOL指标实现正常")
                return True
            else:
                logger.error(f"VOL指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"VOL指标测试失败: {e}")
            return False
            
    def check_v_shaped_reversal_indicator(self):
        """专门检查V_SHAPED_REVERSAL指标的实现"""
        logger.info("检查V_SHAPED_REVERSAL指标实现...")
        
        try:
            v_shaped_indicator = IndicatorFactory.create_indicator("V_SHAPED_REVERSAL")
            result = v_shaped_indicator.compute(self.test_df)
            
            # 检查结果
            if 'v_reversal' in result.columns:
                logger.info("V_SHAPED_REVERSAL指标实现正常")
                return True
            else:
                logger.error(f"V_SHAPED_REVERSAL指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"V_SHAPED_REVERSAL指标测试失败: {e}")
            return False
            
    def check_trix_indicator(self):
        """专门检查TRIX指标的实现"""
        logger.info("检查TRIX指标实现...")
        
        try:
            trix_indicator = IndicatorFactory.create_indicator("TRIX")
            result = trix_indicator.compute(self.test_df)
            
            # 检查结果
            if 'TRIX' in result.columns and 'MATRIX' in result.columns:
                logger.info("TRIX指标实现正常")
                return True
            else:
                logger.error(f"TRIX指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"TRIX指标测试失败: {e}")
            return False
            
    def check_zxm_elasticity_score_indicator(self):
        """专门检查ZXM弹性评分指标的实现"""
        logger.info("检查ZXM_ELASTICITY_SCORE指标实现...")
        
        try:
            zxm_elasticity_score = IndicatorFactory.create_indicator("ZXM_ELASTICITY_SCORE")
            result = zxm_elasticity_score.compute(self.test_df)
            
            # 检查结果
            if 'ElasticityScore' in result.columns and 'Signal' in result.columns:
                logger.info("ZXM弹性评分指标实现正常")
                return True
            else:
                logger.error(f"ZXM弹性评分指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"ZXM弹性评分指标测试失败: {e}")
            return False
            
    def check_zxm_buypoint_score_indicator(self):
        """专门检查ZXM买点评分指标的实现"""
        logger.info("检查ZXM_BUYPOINT_SCORE指标实现...")
        
        try:
            zxm_buypoint_score = IndicatorFactory.create_indicator("ZXM_BUYPOINT_SCORE")
            result = zxm_buypoint_score.compute(self.test_df)
            
            # 检查结果
            if 'BuyPointScore' in result.columns and 'Signal' in result.columns:
                logger.info("ZXM买点评分指标实现正常")
                return True
            else:
                logger.error(f"ZXM买点评分指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"ZXM买点评分指标测试失败: {e}")
            return False
            
    def check_divergence_indicator(self):
        """专门检查DIVERGENCE指标的实现"""
        logger.info("检查DIVERGENCE指标实现...")
        
        try:
            divergence_indicator = IndicatorFactory.create_indicator("DIVERGENCE")
            result = divergence_indicator.compute(self.test_df)
            
            # 打印结果列名
            logger.info(f"DIVERGENCE指标输出列: {result.columns.tolist()}")
            
            # 检查是否有任何结果列
            if len(result.columns) > 0:
                logger.info("DIVERGENCE指标实现正常")
                return True
            else:
                logger.error("DIVERGENCE指标计算结果为空")
                return False
                
        except Exception as e:
            logger.error(f"DIVERGENCE指标测试失败: {e}")
            return False
    
    def check_bias_indicator(self):
        """专门检查BIAS指标的实现"""
        logger.info("检查BIAS指标实现...")
        
        try:
            bias_indicator = IndicatorFactory.create_indicator("BIAS", periods=[6, 12, 24])
            result = bias_indicator.compute(self.test_df)
            
            # 检查结果
            if 'BIAS6' in result.columns and 'BIAS12' in result.columns and 'BIAS24' in result.columns:
                logger.info("BIAS指标实现正常")
                return True
            else:
                logger.error(f"BIAS指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"BIAS指标测试失败: {e}")
            return False
            
    def check_sar_indicator(self):
        """专门检查SAR指标的实现"""
        logger.info("检查SAR指标实现...")
        
        try:
            sar_indicator = IndicatorFactory.create_indicator("SAR", acceleration=0.02, maximum=0.2)
            result = sar_indicator.compute(self.test_df)
            
            # 检查结果
            if 'SAR' in result.columns or 'sar' in result.columns:
                logger.info("SAR指标实现正常")
                return True
            else:
                logger.error(f"SAR指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"SAR指标测试失败: {e}")
            return False
            
    def check_obv_indicator(self):
        """专门检查OBV指标的实现"""
        logger.info("检查OBV指标实现...")
        
        try:
            obv_indicator = IndicatorFactory.create_indicator("OBV")
            result = obv_indicator.compute(self.test_df)
            
            # 检查结果
            if 'OBV' in result.columns or 'obv' in result.columns:
                logger.info("OBV指标实现正常")
                return True
            else:
                logger.error(f"OBV指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"OBV指标测试失败: {e}")
            return False
            
    def check_dmi_indicator(self):
        """专门检查DMI指标的实现"""
        logger.info("检查DMI指标实现...")
        
        try:
            dmi_indicator = IndicatorFactory.create_indicator("DMI")
            result = dmi_indicator.compute(self.test_df)
            
            # 检查结果
            if 'PDI' in result.columns and 'MDI' in result.columns and 'ADX' in result.columns:
                logger.info("DMI指标实现正常")
                return True
            else:
                logger.error(f"DMI指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"DMI指标测试失败: {e}")
            return False
            
    def check_wr_indicator(self):
        """专门检查WR指标的实现"""
        logger.info("检查WR指标实现...")
        
        try:
            wr_indicator = IndicatorFactory.create_indicator("WR", periods=[6, 14])
            result = wr_indicator.compute(self.test_df)
            
            # 检查结果
            if 'WR6' in result.columns and 'WR14' in result.columns:
                logger.info("WR指标实现正常")
                return True
            else:
                logger.error(f"WR指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"WR指标测试失败: {e}")
            return False
            
    def check_cci_indicator(self):
        """专门检查CCI指标的实现"""
        logger.info("检查CCI指标实现...")
        
        try:
            cci_indicator = IndicatorFactory.create_indicator("CCI", periods=[14, 20])
            result = cci_indicator.compute(self.test_df)
            
            # 检查结果
            if 'CCI14' in result.columns or 'CCI20' in result.columns:
                logger.info("CCI指标实现正常")
                return True
            else:
                logger.error(f"CCI指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"CCI指标测试失败: {e}")
            return False
            
    def check_roc_indicator(self):
        """检查ROC指标"""
        logger.info("检查ROC指标...")
        try:
            # 获取测试数据
            data = self._get_test_data()
            
            # 创建回测系统
            backtest = UnifiedBacktest()
            
            # 定义结果字典
            result = {'indicators': {}, 'patterns': []}
            
            # 计算ROC指标
            backtest._calculate_roc_indicators(data, 50, result)
            
            # 检查结果
            assert 'roc' in result['indicators'], "ROC指标计算失败，没有返回结果"
            
            # 检查各项指标值
            assert 'roc6' in result['indicators']['roc'], "ROC6值缺失"
            assert 'roc12' in result['indicators']['roc'], "ROC12值缺失"
            assert 'roc24' in result['indicators']['roc'], "ROC24值缺失"
            assert 'roc6_signal' in result['indicators']['roc'], "ROC6信号线值缺失"
            
            logger.info("ROC指标检查通过")
            return True
        except Exception as e:
            logger.error(f"ROC指标检查失败: {e}")
            return False
            
    def check_vosc_indicator(self):
        """检查VOSC指标"""
        logger.info("检查VOSC指标...")
        try:
            # 获取测试数据
            data = self._get_test_data()
            
            # 创建回测系统
            backtest = UnifiedBacktest()
            
            # 定义结果字典
            result = {'indicators': {}, 'patterns': []}
            
            # 计算VOSC指标
            backtest._calculate_vosc_indicators(data, 50, result)
            
            # 检查结果
            assert 'vosc' in result['indicators'], "VOSC指标计算失败，没有返回结果"
            
            # 检查各项指标值
            assert 'vosc' in result['indicators']['vosc'], "VOSC值缺失"
            assert 'vosc_ma' in result['indicators']['vosc'], "VOSC_MA值缺失"
            assert 'vosc_diff' in result['indicators']['vosc'], "VOSC_DIFF值缺失"
            
            logger.info("VOSC指标检查通过")
            return True
        except Exception as e:
            logger.error(f"VOSC指标检查失败: {e}")
            return False
            
    def check_mfi_indicator(self):
        """检查MFI指标"""
        logger.info("检查MFI指标...")
        try:
            # 获取测试数据
            data = self._get_test_data()
            
            # 创建回测系统
            backtest = UnifiedBacktest()
            
            # 定义结果字典
            result = {'indicators': {}, 'patterns': []}
            
            # 计算MFI指标
            backtest._calculate_mfi_indicators(data, 50, result)
            
            # 检查结果
            assert 'mfi' in result['indicators'], "MFI指标计算失败，没有返回结果"
            
            # 检查各项指标值
            assert 'mfi' in result['indicators']['mfi'], "MFI值缺失"
            assert 'mfi_prev' in result['indicators']['mfi'], "MFI_PREV值缺失"
            assert 'mfi_diff' in result['indicators']['mfi'], "MFI_DIFF值缺失"
            
            logger.info("MFI指标检查通过")
            return True
        except Exception as e:
            logger.error(f"MFI指标检查失败: {e}")
            return False
            
    def check_stochrsi_indicator(self):
        """检查STOCHRSI指标"""
        logger.info("检查STOCHRSI指标...")
        try:
            # 获取测试数据
            data = self._get_test_data()
            
            # 创建回测系统
            backtest = UnifiedBacktest()
            
            # 定义结果字典
            result = {'indicators': {}, 'patterns': []}
            
            # 计算STOCHRSI指标
            backtest._calculate_stochrsi_indicators(data, 50, result)
            
            # 检查结果
            assert 'stochrsi' in result['indicators'], "STOCHRSI指标计算失败，没有返回结果"
            
            # 检查各项指标值
            assert 'k' in result['indicators']['stochrsi'], "STOCHRSI K值缺失"
            assert 'd' in result['indicators']['stochrsi'], "STOCHRSI D值缺失"
            assert 'k_prev' in result['indicators']['stochrsi'], "STOCHRSI K_PREV值缺失"
            assert 'd_prev' in result['indicators']['stochrsi'], "STOCHRSI D_PREV值缺失"
            
            logger.info("STOCHRSI指标检查通过")
            return True
        except Exception as e:
            logger.error(f"STOCHRSI指标检查失败: {e}")
            return False
            
    def check_momentum_indicator(self):
        """专门检查MOMENTUM指标的实现"""
        logger.info("检查MOMENTUM指标实现...")
        
        try:
            momentum_indicator = IndicatorFactory.create_indicator("MTM", period=14, signal_period=6)
            result = momentum_indicator.compute(self.test_df)
            
            # 检查结果
            if 'mtm' in result.columns and 'signal' in result.columns:
                logger.info("MOMENTUM指标实现正常")
                
                # 如果使用回测系统，测试在回测中的计算
                try:
                    backtest = UnifiedBacktest()
                    test_result = {}
                    backtest._calculate_momentum_indicators(self.test_df, 10, test_result)
                    if 'indicators' in test_result and 'momentum' in test_result['indicators']:
                        logger.info("MOMENTUM指标在回测系统中运行正常")
                    else:
                        logger.warning("MOMENTUM指标在回测系统中运行异常")
                except Exception as e:
                    logger.error(f"在回测系统中测试MOMENTUM指标失败: {e}")
                
                return True
            else:
                logger.error(f"MOMENTUM指标计算结果缺少必要列，实际列: {result.columns.tolist()}")
                return False
                
        except Exception as e:
            logger.error(f"MOMENTUM指标测试失败: {e}")
            return False
            
    def check_rsima_indicator(self):
        """专门检查RSIMA指标的实现"""
        logger.info("检查RSIMA指标实现...")
        
        try:
            rsima_indicator = IndicatorFactory.create_indicator("RSIMA", rsi_period=14, ma_periods=[5, 10, 20])
            result = rsima_indicator.compute(self.test_df)
            
            # 检查结果
            if 'rsi' in result.columns and 'rsi_ma5' in result.columns and 'rsi_ma10' in result.columns:
                logger.info("RSIMA指标实现正常")
                
                # 如果使用回测系统，测试在回测中的计算
                try:
                    backtest = UnifiedBacktest()
                    result_dict = {'indicators': {}, 'patterns': []}
                    backtest._calculate_rsima_indicators(self.test_df, 20, result_dict)
                    logger.info(f"回测系统中RSIMA指标计算结果: {result_dict['indicators']['rsima']}")
                    logger.info(f"回测系统中识别出的RSIMA形态: {[p for p in result_dict['patterns'] if 'RSI' in p or 'RSIMA' in p]}")
                except Exception as e:
                    logger.error(f"在回测系统中测试RSIMA指标时出错: {e}")
            else:
                logger.warning("RSIMA指标计算结果缺少必要的列")
        except Exception as e:
            logger.error(f"检查RSIMA指标时出错: {e}")

    def check_intraday_volatility_indicator(self):
        """专门检查INTRADAY_VOLATILITY指标的实现"""
        logger.info("检查INTRADAY_VOLATILITY指标实现...")
        
        try:
            volatility_indicator = IndicatorFactory.create_indicator("INTRADAY_VOLATILITY", smooth_period=5)
            result = volatility_indicator.compute(self.test_df)
            
            # 检查结果
            if 'volatility' in result.columns and 'volatility_ma' in result.columns:
                logger.info("INTRADAY_VOLATILITY指标实现正常")
                
                # 如果使用回测系统，测试在回测中的计算
                try:
                    backtest = UnifiedBacktest()
                    result_dict = {'indicators': {}, 'patterns': []}
                    backtest._calculate_intraday_volatility_indicators(self.test_df, 20, result_dict)
                    logger.info(f"回测系统中INTRADAY_VOLATILITY指标计算结果: {result_dict['indicators']['intraday_volatility']}")
                    logger.info(f"回测系统中识别出的INTRADAY_VOLATILITY形态: {[p for p in result_dict['patterns'] if '波动率' in p]}")
                except Exception as e:
                    logger.error(f"在回测系统中测试INTRADAY_VOLATILITY指标时出错: {e}")
            else:
                logger.warning("INTRADAY_VOLATILITY指标计算结果缺少必要的列")
        except Exception as e:
            logger.error(f"检查INTRADAY_VOLATILITY指标时出错: {e}")
            
    def check_atr_indicator(self):
        """专门检查ATR指标的实现"""
        logger.info("检查ATR指标实现...")
        
        try:
            atr_indicator = IndicatorFactory.create_indicator("ATR", period=14)
            result = atr_indicator.compute(self.test_df)
            
            # 检查结果
            if 'ATR' in result.columns:
                logger.info("ATR指标实现正常")
                
                # 如果使用回测系统，测试在回测中的计算
                try:
                    backtest = UnifiedBacktest()
                    result_dict = {'indicators': {}, 'patterns': []}
                    backtest._calculate_atr_indicators(self.test_df, 20, result_dict)
                    
                    if 'atr' in result_dict['indicators']:
                        logger.info(f"回测系统中ATR指标计算结果: {result_dict['indicators']['atr']}")
                        logger.info(f"回测系统中识别出的ATR形态: {[p for p in result_dict['patterns'] if 'ATR' in p or '波动性' in p]}")
                        return True
                    else:
                        logger.warning("回测系统中ATR指标计算结果不存在")
                        return False
                except Exception as e:
                    logger.error(f"在回测系统中测试ATR指标时出错: {e}")
                    return False
            else:
                logger.warning("ATR指标计算结果缺少必要的列")
                return False
        except Exception as e:
            logger.error(f"检查ATR指标时出错: {e}")
            return False
            
    def check_emv_indicator(self):
        """专门检查EMV指标的实现"""
        logger.info("检查EMV指标实现...")
        
        try:
            emv_indicator = IndicatorFactory.create_indicator("EMV", period=14, volume_scale=10000)
            result = emv_indicator.compute(self.test_df)
            
            # 检查结果
            if 'EMV' in result.columns and 'daily_emv' in result.columns:
                logger.info("EMV指标实现正常")
                
                # 如果使用回测系统，测试在回测中的计算
                try:
                    backtest = UnifiedBacktest()
                    result_dict = {'indicators': {}, 'patterns': []}
                    backtest._calculate_emv_indicators(self.test_df, 20, result_dict)
                    
                    if 'emv' in result_dict['indicators']:
                        logger.info(f"回测系统中EMV指标计算结果: {result_dict['indicators']['emv']}")
                        logger.info(f"回测系统中识别出的EMV形态: {[p for p in result_dict['patterns'] if 'EMV' in p]}")
                        return True
                    else:
                        logger.warning("回测系统中EMV指标计算结果不存在")
                        return False
                except Exception as e:
                    logger.error(f"在回测系统中测试EMV指标时出错: {e}")
                    return False
            else:
                logger.warning(f"EMV指标计算结果缺少必要的列，实际列: {result.columns.tolist()}")
                return False
        except Exception as e:
            logger.error(f"检查EMV指标时出错: {e}")
            return False
            
    def check_volume_ratio_indicator(self):
        """专门检查量比指标的实现"""
        logger.info("检查量比指标实现...")
        
        try:
            vr_indicator = IndicatorFactory.create_indicator("VOLUME_RATIO", reference_period=5, ma_period=3)
            result = vr_indicator.compute(self.test_df)
            
            # 检查结果
            if 'volume_ratio' in result.columns and 'volume_ratio_ma' in result.columns:
                logger.info("量比指标实现正常")
                
                # 如果使用回测系统，测试在回测中的计算
                try:
                    backtest = UnifiedBacktest()
                    result_dict = {'indicators': {}, 'patterns': []}
                    backtest._calculate_volume_ratio_indicators(self.test_df, 20, result_dict)
                    
                    if 'volume_ratio' in result_dict['indicators']:
                        logger.info(f"回测系统中量比指标计算结果: {result_dict['indicators']['volume_ratio']}")
                        logger.info(f"回测系统中识别出的量比形态: {[p for p in result_dict['patterns'] if '量比' in p or '放量' in p or '缩量' in p]}")
                        return True
                    else:
                        logger.warning("回测系统中量比指标计算结果不存在")
                        return False
                except Exception as e:
                    logger.error(f"在回测系统中测试量比指标时出错: {e}")
                    return False
            else:
                logger.warning(f"量比指标计算结果缺少必要的列，实际列: {result.columns.tolist()}")
                return False
        except Exception as e:
            logger.error(f"检查量比指标时出错: {e}")
            return False
            
    def _get_test_data(self):
        """
        获取测试数据
        
        Returns:
            pd.DataFrame: 包含测试数据的DataFrame
        """
        # 使用已经初始化的测试数据
        return self.test_df.copy()
            
    def run_all_checks(self):
        """运行所有检查"""
        checks = [
            ("MA", self.check_ma_indicator),
            ("MACD", self.check_macd_indicator),
            ("KDJ", self.check_kdj_indicator),
            ("RSI", self.check_rsi_indicator),
            ("BOLL", self.check_boll_indicator),
            ("VOL", self.check_vol_indicator),
            ("V形反转", self.check_v_shaped_reversal_indicator),
            ("TRIX", self.check_trix_indicator),
            ("ZXM弹性评分", self.check_zxm_elasticity_score_indicator),
            ("ZXM买点评分", self.check_zxm_buypoint_score_indicator),
            ("量价背离", self.check_divergence_indicator),
            ("BIAS", self.check_bias_indicator),
            ("SAR", self.check_sar_indicator),
            ("OBV", self.check_obv_indicator),
            ("DMI", self.check_dmi_indicator),
            ("WR", self.check_wr_indicator),
            ("CCI", self.check_cci_indicator),
            ("ROC", self.check_roc_indicator),
            ("VOSC", self.check_vosc_indicator),
            ("MFI", self.check_mfi_indicator),
            ("STOCHRSI", self.check_stochrsi_indicator),
            ("MOMENTUM", self.check_momentum_indicator),
            ("RSIMA", self.check_rsima_indicator),
            ("INTRADAY_VOLATILITY", self.check_intraday_volatility_indicator),
            ("ATR", self.check_atr_indicator),
            ("EMV", self.check_emv_indicator),
            ("量比", self.check_volume_ratio_indicator)
        ]
        
        results = {}
        all_passed = True
        
        for name, check_func in checks:
            result = check_func()
            results[name] = "通过" if result else "失败"
            if not result:
                all_passed = False
                
        # 打印结果
        logger.info("=" * 30)
        logger.info("指标检查结果汇总")
        logger.info("=" * 30)
        
        for name, result in results.items():
            logger.info(f"{name}指标: {result}")
            
        logger.info("=" * 30)
        logger.info(f"总体结果: {'全部通过' if all_passed else '存在失败项'}")
        logger.info("=" * 30)
        
        return all_passed

def main():
    """主程序入口"""
    import sys
    
    # 获取命令行参数
    args = sys.argv[1:]
    
    checker = IndicatorChecker()
    
    if not args:
        # 如果没有参数，运行所有检查
        checker.run_all_checks()
    else:
        # 如果有参数，只检查指定的指标
        results = []
        for indicator_name in args:
            indicator_name = indicator_name.upper()  # 转换为大写
            check_method_name = f"check_{indicator_name.lower()}_indicator"
            
            if hasattr(checker, check_method_name):
                check_method = getattr(checker, check_method_name)
                result = check_method()
                results.append((indicator_name, result))
                logger.info(f"{indicator_name}指标: {'通过' if result else '失败'}")
            else:
                logger.error(f"未找到{indicator_name}指标的检查方法")
                results.append((indicator_name, False))
        
        # 总结结果
        logger.info("\n检查结果总结:")
        for name, result in results:
            logger.info(f"{name}指标: {'通过' if result else '失败'}")
        
        # 返回是否全部通过
        return all(result for _, result in results)
    
if __name__ == "__main__":
    main() 