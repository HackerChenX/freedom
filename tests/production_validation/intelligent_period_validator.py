#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能周期验证器 - 基于指标最小周期要求的动态验证

使用每个指标定义的minimum_periods属性，动态确定数据窗口大小：
1. 自动获取指标的最小周期要求
2. 使用稳定周期进行验证，确保计算准确性
3. 解决双向验证中的数据窗口不一致问题
4. 为不同指标提供个性化的验证策略
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from db.interfaces.data_access_interface import ClickHouseDataAccess
from db.services.stock_data_service import StockDataService
from utils.logger import get_logger

logger = get_logger(__name__)

class IntelligentPeriodValidator:
    """智能周期验证器 - 基于指标最小周期要求"""
    
    def __init__(self, indicator_name: str, indicator_instance: Any):
        """初始化智能周期验证器"""
        self.indicator_name = indicator_name.upper()
        self.indicator_instance = indicator_instance
        self.pattern_registry = get_unified_pattern_registry()
        
        # 使用真实数据访问
        clickhouse_data_access = ClickHouseDataAccess()
        self.stock_data_service = StockDataService(clickhouse_data_access)
        
        # 获取指标的周期要求
        self.period_requirements = self._get_indicator_period_requirements()
        
        # 验证标准（基于指标周期要求）
        self.validation_standards = {
            'min_stocks_to_test': 8,
            'min_patterns_found': 3,
            'min_data_days': self.period_requirements['minimum_periods'],
            'recommended_data_days': self.period_requirements['recommended_periods'],
            'stable_data_days': self.period_requirements['stable_periods'],
            'required_levels': ['日线'],
            'min_verification_rate': 0.8  # 80%验证率
        }
        
        # 动态获取指标支持的形态
        self.supported_patterns = self._get_indicator_patterns()
        
        logger.info(f"🔥 {self.indicator_name}智能周期验证器初始化完成")
        logger.info(f"📊 周期要求: 最小{self.period_requirements['minimum_periods']}, 推荐{self.period_requirements['recommended_periods']}, 稳定{self.period_requirements['stable_periods']}")
    
    def _get_indicator_period_requirements(self) -> Dict[str, int]:
        """获取指标的周期要求"""
        try:
            # 检查指标是否有minimum_periods属性
            if hasattr(self.indicator_instance, 'minimum_periods'):
                minimum = self.indicator_instance.minimum_periods
                recommended = max(minimum * 2, 60)
                stable = max(minimum * 3, 120)
                
                return {
                    'minimum_periods': minimum,
                    'recommended_periods': recommended,
                    'stable_periods': stable
                }
            else:
                # 降级处理：使用默认值
                logger.warning(f"⚠️ {self.indicator_name}指标没有minimum_periods属性，使用默认值")
                return {
                    'minimum_periods': 30,
                    'recommended_periods': 60,
                    'stable_periods': 120
                }
        except Exception as e:
            logger.error(f"❌ 获取{self.indicator_name}周期要求失败: {e}")
            return {
                'minimum_periods': 30,
                'recommended_periods': 60,
                'stable_periods': 120
            }
    
    def _get_indicator_patterns(self) -> List[str]:
        """动态获取指标支持的形态"""
        try:
            indicator_specific_patterns = {
                'MACD': ['GOLDEN_CROSS', 'DEATH_CROSS', 'MACD_ABOVE_ZERO_GOLDEN', 'BEARISH_DIVERGENCE'],
                'RSI': ['OVERBOUGHT', 'OVERSOLD', 'RSI_GOLDEN_CROSS', 'RSI_DEATH_CROSS'],
                'KDJ': ['KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS', 'KDJ_OVERBOUGHT', 'KDJ_OVERSOLD'],
                'BOLL': ['BOLL_UPPER_BREAKOUT', 'BOLL_LOWER_BREAKOUT', 'BOLL_SQUEEZE', 'BOLL_EXPANSION'],
                'MA': ['MA_GOLDEN_CROSS', 'MA_DEATH_CROSS', 'MA_SUPPORT', 'MA_RESISTANCE'],
                'EMA': ['EMA_GOLDEN_CROSS', 'EMA_DEATH_CROSS', 'EMA_TREND_UP', 'EMA_TREND_DOWN']
            }
            
            return indicator_specific_patterns.get(self.indicator_name, ['GOLDEN_CROSS', 'DEATH_CROSS'])
            
        except Exception as e:
            logger.warning(f"获取{self.indicator_name}支持形态失败: {e}")
            return ['GOLDEN_CROSS', 'DEATH_CROSS']
    
    def run_intelligent_period_validation(self) -> Dict[str, Any]:
        """运行智能周期验证"""
        
        print(f"🔥 开始{self.indicator_name}智能周期验证")
        print("=" * 80)
        print("📋 验证目标: 基于指标最小周期要求的智能验证")
        print(f"🎯 周期策略: 使用稳定周期{self.period_requirements['stable_periods']}天数据")
        print("🏗️ 架构: 严格遵循数据层接口规范")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Intelligent_Period_Validation',
            'indicator_name': self.indicator_name,
            'indicator_type': type(self.indicator_instance).__name__,
            'validation_timestamp': datetime.now().isoformat(),
            'period_requirements': self.period_requirements,
            'database_connection_validation': {},
            'intelligent_data_access_validation': {},
            'period_aware_calculation_validation': {},
            'period_aware_pattern_detection': {},
            'intelligent_bidirectional_verification': {},
            'overall_assessment': {},
            'validation_passed': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 验证数据库连接
            print("\n🔌 步骤1: 验证真实数据库连接")
            db_connection = self._validate_database_connection()
            validation_result['database_connection_validation'] = db_connection
            
            if not db_connection['connected']:
                validation_result['issues_found'].extend(db_connection['issues'])
                return validation_result
            
            print(f"✅ 数据库连接验证通过")
            
            # 步骤2: 智能数据访问验证
            print(f"\n📊 步骤2: 智能数据访问验证（稳定周期{self.period_requirements['stable_periods']}天）")
            data_access_validation = self._validate_intelligent_data_access()
            validation_result['intelligent_data_access_validation'] = data_access_validation
            
            if not data_access_validation['success']:
                validation_result['issues_found'].extend(data_access_validation['issues'])
                return validation_result
            
            print(f"✅ 智能数据访问验证通过: 获取{data_access_validation['stocks_count']}支股票")
            
            # 步骤3: 周期感知的指标计算验证
            print(f"\n🎯 步骤3: 周期感知的{self.indicator_name}指标计算验证")
            calculation_validation = self._validate_period_aware_calculation(data_access_validation['sample_data'])
            validation_result['period_aware_calculation_validation'] = calculation_validation
            
            if not calculation_validation['success']:
                validation_result['issues_found'].extend(calculation_validation['issues'])
                return validation_result
            
            print(f"✅ {self.indicator_name}周期感知计算验证通过")
            
            # 步骤4: 周期感知的形态检测验证
            print(f"\n🔍 步骤4: 周期感知的{self.indicator_name}形态检测验证")
            pattern_validation = self._validate_period_aware_pattern_detection(data_access_validation['sample_data'])
            validation_result['period_aware_pattern_detection'] = pattern_validation
            
            if not pattern_validation['success']:
                validation_result['issues_found'].extend(pattern_validation['issues'])
            else:
                print(f"✅ {self.indicator_name}周期感知形态检测验证通过: 检测到{pattern_validation['patterns_detected']}个形态")
            
            # 步骤5: 智能双向验证
            print(f"\n🛡️ 步骤5: {self.indicator_name}智能双向验证（统一稳定周期）")
            bidirectional_validation = self._intelligent_bidirectional_verification(
                pattern_validation.get('pattern_details', []),
                data_access_validation['sample_data']
            )
            validation_result['intelligent_bidirectional_verification'] = bidirectional_validation
            
            if bidirectional_validation['verified_patterns'] > 0:
                print(f"✅ 智能双向验证通过: {bidirectional_validation['verified_patterns']}个形态验证成功，验证率: {bidirectional_validation['verification_rate']:.1%}")
            else:
                print(f"⚠️ 智能双向验证需要进一步优化")
            
            # 步骤6: 综合评估
            print(f"\n🏆 步骤6: {self.indicator_name}综合评估")
            overall_assessment = self._comprehensive_assessment_intelligent(validation_result)
            validation_result['overall_assessment'] = overall_assessment
            validation_result['validation_passed'] = overall_assessment['passed']
            
            if overall_assessment['passed']:
                print(f"🎉 {self.indicator_name}指标智能周期验证通过！")
            else:
                print(f"❌ {self.indicator_name}指标智能周期验证失败")
        
        except Exception as e:
            logger.error(f"❌ {self.indicator_name}智能周期验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _validate_database_connection(self) -> Dict[str, Any]:
        """验证数据库连接"""
        result = {
            'connected': False,
            'database_info': {},
            'issues': []
        }
        
        try:
            stock_codes = self.stock_data_service.get_stock_list(limit=1)
            
            if stock_codes and len(stock_codes) > 0:
                result['connected'] = True
                result['database_info'] = {
                    'database_type': 'ClickHouse',
                    'connection_method': 'StockDataService + ClickHouseDataAccess',
                    'test_query_success': True
                }
            else:
                result['issues'].append("数据库连接成功但无法获取股票数据")
        
        except Exception as e:
            result['issues'].append(f"数据库连接失败: {str(e)}")
        
        return result
    
    def _validate_intelligent_data_access(self) -> Dict[str, Any]:
        """智能数据访问验证"""
        result = {
            'success': False,
            'stocks_count': 0,
            'sample_data': {},
            'data_window_used': self.period_requirements['stable_periods'],
            'issues': []
        }
        
        try:
            stock_codes = self.stock_data_service.get_stock_list(limit=12)
            
            if not stock_codes:
                result['issues'].append("无法获取股票列表")
                return result
            
            result['stocks_count'] = len(stock_codes)
            print(f"    📋 获取到{len(stock_codes)}支股票代码")
            
            # 使用指标的稳定周期获取数据
            sample_data = {}
            stable_periods = self.period_requirements['stable_periods']
            
            for stock_code in stock_codes[:8]:  # 测试前8支
                try:
                    # 使用稳定周期数据
                    df = self.stock_data_service.get_stock_data(stock_code, days=stable_periods)
                    
                    if df is not None and len(df) >= self.validation_standards['min_data_days']:
                        sample_data[stock_code] = df
                        print(f"    ✅ {stock_code}: {len(df)}天数据（稳定周期{stable_periods}）")
                    else:
                        print(f"    ⚠️ {stock_code}: 数据不足")
                        
                except Exception as e:
                    print(f"    ❌ {stock_code}: 数据获取失败 - {e}")
                    continue
            
            result['sample_data'] = sample_data
            
            if len(sample_data) >= 5:
                result['success'] = True
            else:
                result['issues'].append("获取到的有效股票数据不足")
        
        except Exception as e:
            result['issues'].append(f"智能数据访问验证异常: {str(e)}")
        
        return result

    def _validate_period_aware_calculation(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """周期感知的指标计算验证"""
        result = {
            'success': False,
            'calculations_successful': 0,
            'total_calculations': 0,
            'period_validation_details': {},
            'issues': []
        }

        try:
            for stock_code, df in sample_data.items():
                result['total_calculations'] += 1

                try:
                    print(f"    🔍 计算{stock_code}的{self.indicator_name}指标（稳定周期{len(df)}天）")

                    # 验证数据长度是否满足指标要求
                    if hasattr(self.indicator_instance, 'validate_data_length'):
                        validation = self.indicator_instance.validate_data_length(df, strict=False)

                        result['period_validation_details'][stock_code] = validation

                        if not validation['valid']:
                            print(f"      ⚠️ 数据长度验证失败: {validation['message']}")
                            result['issues'].append(f"{stock_code}: {validation['message']}")
                            continue
                        else:
                            print(f"      ✅ 数据长度验证通过: {validation['validation_level']}")

                    # 使用指标实例计算
                    indicator_result = self.indicator_instance.calculate(df)

                    if indicator_result is not None and not indicator_result.empty:
                        result['calculations_successful'] += 1
                        print(f"      ✅ {self.indicator_name}计算成功: {len(indicator_result)}条结果")
                    else:
                        print(f"      ❌ {self.indicator_name}计算返回空结果")

                except Exception as e:
                    print(f"      ❌ {self.indicator_name}计算失败: {e}")
                    result['issues'].append(f"{stock_code}: {str(e)}")
                    continue

            # 成功率检查
            if result['total_calculations'] > 0:
                success_rate = result['calculations_successful'] / result['total_calculations']
                if success_rate >= 0.8:  # 80%成功率
                    result['success'] = True
                else:
                    result['issues'].append(f"指标计算成功率过低: {success_rate:.1%}")
            else:
                result['issues'].append("没有进行任何指标计算")

        except Exception as e:
            result['issues'].append(f"周期感知计算验证异常: {str(e)}")

        return result
