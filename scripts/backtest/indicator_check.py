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
        """检查所有指标是否可用"""
        logger.info("开始检查所有技术指标...")
        
        valid_indicators = []
        issue_indicators = []
        
        for indicator_name in self.all_indicators:
            try:
                logger.info(f"正在测试指标: {indicator_name}")
                indicator = IndicatorFactory.create_indicator(indicator_name)
                
                # 测试计算
                result = indicator.compute(self.test_df)
                
                # 如果返回的不是DataFrame，或者结果为空，则记录问题
                if not isinstance(result, pd.DataFrame) or result.empty:
                    logger.error(f"指标 {indicator_name} 计算结果无效")
                    issue_indicators.append((indicator_name, "计算结果无效"))
                else:
                    logger.info(f"指标 {indicator_name} 测试通过, 输出列: {result.columns.tolist()}")
                    valid_indicators.append(indicator_name)
                    
            except Exception as e:
                logger.error(f"指标 {indicator_name} 测试失败: {e}")
                issue_indicators.append((indicator_name, str(e)))
        
        # 打印测试结果
        logger.info(f"\n测试总结:")
        logger.info(f"成功指标: {len(valid_indicators)}/{len(self.all_indicators)}")
        logger.info(f"失败指标: {len(issue_indicators)}/{len(self.all_indicators)}")
        
        if issue_indicators:
            logger.info("\n问题指标详情:")
            for name, error in issue_indicators:
                logger.info(f"- {name}: {error}")
                
        return valid_indicators, issue_indicators
    
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
    
    def run_all_checks(self):
        """运行所有检查"""
        logger.info("开始全面指标检查...")
        
        # 先检查所有指标
        valid_indicators, issue_indicators = self.check_all_indicators()
        
        # 专门检查几个关键指标
        vol_check = self.check_vol_indicator()
        v_shaped_check = self.check_v_shaped_reversal_indicator()
        trix_check = self.check_trix_indicator()
        divergence_check = self.check_divergence_indicator()
        
        # 总结
        logger.info("\n检查结果总结:")
        logger.info(f"全部指标: {len(valid_indicators)}/{len(self.all_indicators)} 通过")
        logger.info(f"VOL指标: {'通过' if vol_check else '失败'}")
        logger.info(f"V形反转指标: {'通过' if v_shaped_check else '失败'}")
        logger.info(f"TRIX指标: {'通过' if trix_check else '失败'}")
        logger.info(f"DIVERGENCE指标: {'通过' if divergence_check else '失败'}")
        
        return {
            "all_indicators": (valid_indicators, issue_indicators),
            "vol_check": vol_check,
            "v_shaped_check": v_shaped_check,
            "trix_check": trix_check,
            "divergence_check": divergence_check
        }

def main():
    """主程序入口"""
    checker = IndicatorChecker()
    checker.run_all_checks()
    
if __name__ == "__main__":
    main() 