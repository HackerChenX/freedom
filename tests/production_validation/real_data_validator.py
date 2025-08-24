#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据验证器 - 严格按照架构规范使用真实ClickHouse数据

基于已修复的架构合规数据层，使用真实股票数据进行技术指标验证：
1. 移除模拟数据依赖，使用ClickHouseDataAccess
2. 连接真实ClickHouse数据库进行验证
3. 严格遵循数据层架构规范，不直接写SQL
4. 通过StockDataService服务层进行所有数据操作
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

class RealDataValidator:
    """真实数据验证器 - 严格遵循架构规范使用真实数据"""
    
    def __init__(self, indicator_name: str, indicator_instance: Any):
        """
        初始化真实数据验证器
        
        Args:
            indicator_name: 指标名称（如"MACD", "RSI", "KDJ"等）
            indicator_instance: 指标实例（如MacdMacd(), RsiRsi()等）
        """
        self.indicator_name = indicator_name.upper()
        self.indicator_instance = indicator_instance
        self.pattern_registry = get_unified_pattern_registry()
        
        # 使用真实ClickHouse数据访问，严格遵循架构规范
        try:
            # 创建ClickHouse数据访问实例
            clickhouse_data_access = ClickHouseDataAccess()
            
            # 通过服务层封装数据访问
            self.stock_data_service = StockDataService(clickhouse_data_access)
            
            logger.info("✅ 使用真实ClickHouse数据访问的股票数据服务初始化成功")
        except Exception as e:
            logger.error(f"❌ 真实数据访问初始化失败: {e}")
            raise RuntimeError(f"无法初始化真实数据访问: {e}")
        
        # 验证标准（真实数据环境）
        self.validation_standards = {
            'min_stocks_to_test': 10,           # 至少测试10支股票
            'min_patterns_found': 3,            # 至少找到3个形态
            'min_data_days': 60,                # 至少60天数据
            'required_levels': ['日线'],         # 必须的时间级别
            'min_verification_rate': 0.7        # 70%验证率
        }
        
        # 动态获取指标支持的形态
        self.supported_patterns = self._get_indicator_patterns()
        
        logger.info(f"🔥 {self.indicator_name}真实数据验证器初始化完成")
    
    def _get_indicator_patterns(self) -> List[str]:
        """动态获取指标支持的形态"""
        try:
            # 根据指标名称返回特定形态
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
    
    def run_real_data_validation(self) -> Dict[str, Any]:
        """运行真实数据验证"""
        
        print(f"🔥 开始{self.indicator_name}真实数据验证")
        print("=" * 80)
        print("📋 验证目标: 使用真实ClickHouse数据进行技术指标验证")
        print("🎯 特点: 连接真实数据库，使用真实股票市场数据")
        print("🏗️ 架构: 严格遵循数据层接口规范，不直接写SQL")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Real_Data_Validation',
            'indicator_name': self.indicator_name,
            'indicator_type': type(self.indicator_instance).__name__,
            'validation_timestamp': datetime.now().isoformat(),
            'database_connection_validation': {},
            'real_data_access_validation': {},
            'indicator_calculation_validation': {},
            'pattern_detection_validation': {},
            'bidirectional_verification': {},
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
                print(f"❌ 数据库连接验证失败")
                return validation_result
            
            print(f"✅ 数据库连接验证通过: {db_connection['database_info']}")
            
            # 步骤2: 验证真实数据访问
            print("\n📊 步骤2: 验证真实数据访问")
            data_access_validation = self._validate_real_data_access()
            validation_result['real_data_access_validation'] = data_access_validation
            
            if not data_access_validation['success']:
                validation_result['issues_found'].extend(data_access_validation['issues'])
                print(f"❌ 真实数据访问验证失败")
                return validation_result
            
            print(f"✅ 真实数据访问验证通过: 获取{data_access_validation['stocks_count']}支股票，{data_access_validation['total_records']}条记录")
            
            # 步骤3: 验证指标计算
            print(f"\n🎯 步骤3: 验证{self.indicator_name}指标计算（真实数据）")
            indicator_validation = self._validate_indicator_calculation_real_data(data_access_validation['sample_data'])
            validation_result['indicator_calculation_validation'] = indicator_validation
            
            if not indicator_validation['success']:
                validation_result['issues_found'].extend(indicator_validation['issues'])
                print(f"❌ {self.indicator_name}指标计算验证失败")
                return validation_result
            
            print(f"✅ {self.indicator_name}指标计算验证通过: {indicator_validation['calculations_successful']}/{indicator_validation['total_calculations']}成功")
            
            # 步骤4: 验证形态检测
            print(f"\n🔍 步骤4: 验证{self.indicator_name}形态检测（真实数据）")
            pattern_validation = self._validate_pattern_detection_real_data(data_access_validation['sample_data'])
            validation_result['pattern_detection_validation'] = pattern_validation
            
            if not pattern_validation['success']:
                validation_result['issues_found'].extend(pattern_validation['issues'])
                print(f"❌ {self.indicator_name}形态检测验证失败")
            else:
                print(f"✅ {self.indicator_name}形态检测验证通过: 检测到{pattern_validation['patterns_detected']}个形态")
            
            # 步骤5: 双向验证（真实数据）
            print(f"\n🛡️ 步骤5: {self.indicator_name}双向验证（真实数据）")
            bidirectional_validation = self._validate_bidirectional_real_data(pattern_validation.get('pattern_details', []))
            validation_result['bidirectional_verification'] = bidirectional_validation
            
            if bidirectional_validation['verified_patterns'] > 0:
                print(f"✅ 双向验证通过: {bidirectional_validation['verified_patterns']}个形态验证成功，验证率: {bidirectional_validation['verification_rate']:.1%}")
            else:
                print(f"⚠️ 双向验证需要改进: 验证率过低")
            
            # 步骤6: 综合评估
            print(f"\n🏆 步骤6: {self.indicator_name}综合评估")
            overall_assessment = self._comprehensive_assessment_real_data(validation_result)
            validation_result['overall_assessment'] = overall_assessment
            validation_result['validation_passed'] = overall_assessment['passed']
            
            if overall_assessment['passed']:
                print(f"🎉 {self.indicator_name}指标真实数据验证通过！")
            else:
                print(f"❌ {self.indicator_name}指标真实数据验证失败")
        
        except Exception as e:
            logger.error(f"❌ {self.indicator_name}真实数据验证异常: {e}")
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
            # 通过服务层测试数据库连接
            # 尝试获取股票列表来验证连接
            stock_codes = self.stock_data_service.get_stock_list(limit=1)
            
            if stock_codes and len(stock_codes) > 0:
                result['connected'] = True
                result['database_info'] = {
                    'database_type': 'ClickHouse',
                    'connection_method': 'StockDataService + ClickHouseDataAccess',
                    'test_query_success': True,
                    'sample_stock_code': stock_codes[0]
                }
                logger.info("✅ 真实数据库连接验证成功")
            else:
                result['issues'].append("数据库连接成功但无法获取股票数据")
                logger.warning("⚠️ 数据库连接成功但无数据")
        
        except Exception as e:
            result['issues'].append(f"数据库连接失败: {str(e)}")
            logger.error(f"❌ 数据库连接验证失败: {e}")
        
        return result
    
    def _validate_real_data_access(self) -> Dict[str, Any]:
        """验证真实数据访问"""
        
        result = {
            'success': False,
            'stocks_count': 0,
            'total_records': 0,
            'sample_data': {},
            'issues': []
        }
        
        try:
            # 通过服务层获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=20)
            
            if not stock_codes:
                result['issues'].append("无法获取股票列表")
                return result
            
            result['stocks_count'] = len(stock_codes)
            print(f"    📋 获取到{len(stock_codes)}支股票代码")
            
            # 获取样本股票数据
            sample_data = {}
            total_records = 0
            
            for i, stock_code in enumerate(stock_codes[:10]):  # 测试前10支
                try:
                    # 通过服务层获取股票数据
                    df = self.stock_data_service.get_stock_data(stock_code, days=120)
                    
                    if df is not None and len(df) >= self.validation_standards['min_data_days']:
                        sample_data[stock_code] = df
                        total_records += len(df)
                        print(f"    ✅ {stock_code}: {len(df)}天真实数据")
                    else:
                        print(f"    ⚠️ {stock_code}: 数据不足")
                        
                except Exception as e:
                    print(f"    ❌ {stock_code}: 数据获取失败 - {e}")
                    continue
            
            result['total_records'] = total_records
            result['sample_data'] = sample_data
            
            if len(sample_data) >= 5:  # 至少5支股票有数据
                result['success'] = True
                logger.info(f"✅ 真实数据访问验证成功: {len(sample_data)}支股票，{total_records}条记录")
            else:
                result['issues'].append("获取到的有效真实股票数据不足")
                logger.warning(f"⚠️ 真实数据不足: 仅{len(sample_data)}支股票有数据")
        
        except Exception as e:
            result['issues'].append(f"真实数据访问验证异常: {str(e)}")
            logger.error(f"❌ 真实数据访问验证失败: {e}")
        
        return result
    
    def _validate_indicator_calculation_real_data(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证指标计算（真实数据）"""
        
        result = {
            'success': False,
            'calculations_successful': 0,
            'total_calculations': 0,
            'calculation_details': {},
            'issues': []
        }
        
        try:
            for stock_code, df in sample_data.items():
                result['total_calculations'] += 1
                
                try:
                    print(f"    🔍 计算{stock_code}的{self.indicator_name}指标（真实数据）")
                    
                    # 验证数据质量
                    if df.empty or len(df) < 20:
                        print(f"      ⚠️ {stock_code}: 数据量不足")
                        result['issues'].append(f"{stock_code}: 数据量不足")
                        continue
                    
                    # 使用指标实例计算真实数据
                    indicator_result = self.indicator_instance.calculate(df)
                    
                    if indicator_result is not None and not indicator_result.empty:
                        result['calculations_successful'] += 1
                        result['calculation_details'][stock_code] = {
                            'input_records': len(df),
                            'output_records': len(indicator_result),
                            'calculation_success': True,
                            'date_range': f"{df['date'].min()} 到 {df['date'].max()}"
                        }
                        print(f"      ✅ {self.indicator_name}计算成功: {len(indicator_result)}条结果")
                    else:
                        print(f"      ❌ {self.indicator_name}计算返回空结果")
                        result['calculation_details'][stock_code] = {
                            'calculation_success': False,
                            'error': '计算返回空结果'
                        }
                        
                except Exception as e:
                    print(f"      ❌ {self.indicator_name}计算失败: {e}")
                    result['issues'].append(f"{stock_code}: {str(e)}")
                    result['calculation_details'][stock_code] = {
                        'calculation_success': False,
                        'error': str(e)
                    }
                    continue
            
            # 成功率检查
            if result['total_calculations'] > 0:
                success_rate = result['calculations_successful'] / result['total_calculations']
                if success_rate >= 0.7:  # 至少70%成功率
                    result['success'] = True
                    logger.info(f"✅ {self.indicator_name}指标计算验证成功: {success_rate:.1%}成功率")
                else:
                    result['issues'].append(f"指标计算成功率过低: {success_rate:.1%}")
                    logger.warning(f"⚠️ {self.indicator_name}指标计算成功率过低: {success_rate:.1%}")
            else:
                result['issues'].append("没有进行任何指标计算")
        
        except Exception as e:
            result['issues'].append(f"指标计算验证异常: {str(e)}")
            logger.error(f"❌ 指标计算验证异常: {e}")
        
        return result

    def _validate_pattern_detection_real_data(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证形态检测（真实数据）"""

        result = {
            'success': False,
            'patterns_detected': 0,
            'stocks_with_patterns': 0,
            'pattern_details': [],
            'pattern_distribution': {},
            'issues': []
        }

        try:
            for stock_code, df in sample_data.items():
                try:
                    print(f"    🔍 检测{stock_code}的{self.indicator_name}形态（真实数据）")

                    # 获取指标形态
                    patterns = self.indicator_instance.get_patterns(df)

                    if patterns is not None and not patterns.empty:
                        stock_patterns = 0

                        # 检查每个支持的形态
                        for pattern_name in self.supported_patterns:
                            if pattern_name in patterns.columns:
                                pattern_signals = patterns[patterns[pattern_name] == True]

                                if not pattern_signals.empty:
                                    pattern_count = len(pattern_signals)
                                    stock_patterns += pattern_count
                                    result['patterns_detected'] += pattern_count

                                    # 统计形态分布
                                    if pattern_name not in result['pattern_distribution']:
                                        result['pattern_distribution'][pattern_name] = 0
                                    result['pattern_distribution'][pattern_name] += pattern_count

                                    # 记录形态详情（真实数据）
                                    for idx in pattern_signals.index:
                                        if idx < len(df):
                                            pattern_detail = {
                                                'stock_code': stock_code,
                                                'pattern_name': pattern_name,
                                                'date': df.iloc[idx]['date'].strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
                                                'close_price': float(df.iloc[idx]['close']) if 'close' in df.columns else 0.0,
                                                'data_source': 'real_clickhouse_data'
                                            }
                                            result['pattern_details'].append(pattern_detail)

                        if stock_patterns > 0:
                            result['stocks_with_patterns'] += 1
                            print(f"      ✅ 检测到{stock_patterns}个{self.indicator_name}形态（真实数据）")
                        else:
                            print(f"      ⚠️ 未检测到{self.indicator_name}形态")
                    else:
                        print(f"      ❌ {self.indicator_name}形态检测返回空结果")

                except Exception as e:
                    print(f"      ❌ {self.indicator_name}形态检测失败: {e}")
                    result['issues'].append(f"{stock_code}: {str(e)}")
                    continue

            # 成功标准：至少检测到一些形态
            if result['patterns_detected'] >= self.validation_standards['min_patterns_found']:
                result['success'] = True
                logger.info(f"✅ {self.indicator_name}形态检测验证成功: 检测到{result['patterns_detected']}个形态")
            else:
                result['issues'].append(f"检测到的形态数量不足: {result['patterns_detected']} < {self.validation_standards['min_patterns_found']}")
                logger.warning(f"⚠️ {self.indicator_name}形态检测数量不足")

        except Exception as e:
            result['issues'].append(f"形态检测验证异常: {str(e)}")
            logger.error(f"❌ 形态检测验证异常: {e}")

        return result

    def _validate_bidirectional_real_data(self, pattern_details: List[Dict]) -> Dict[str, Any]:
        """双向验证（真实数据）"""

        result = {
            'verified_patterns': 0,
            'total_patterns': len(pattern_details),
            'verification_rate': 0.0,
            'verification_details': []
        }

        try:
            if not pattern_details:
                logger.warning("⚠️ 没有形态需要进行双向验证")
                return result

            verified_count = 0

            # 限制验证数量以避免过长时间
            patterns_to_verify = pattern_details[:10]  # 只验证前10个形态

            for pattern_info in patterns_to_verify:
                stock_code = pattern_info['stock_code']
                pattern_date = pattern_info['date']
                pattern_name = pattern_info['pattern_name']

                try:
                    print(f"    🔍 双向验证: {stock_code} - {pattern_name} @ {pattern_date}（真实数据）")

                    # 使用服务层获取验证数据
                    target_date = datetime.strptime(pattern_date, '%Y-%m-%d')
                    start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
                    end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')

                    verify_df = self.stock_data_service.get_stock_data_by_date_range(
                        stock_code, start_date, end_date
                    )

                    if verify_df is not None and len(verify_df) >= 20:
                        verify_df = verify_df.sort_values('date').reset_index(drop=True)

                        # 重新计算指标
                        verify_indicator = self.indicator_instance.calculate(verify_df)
                        verify_patterns = self.indicator_instance.get_patterns(verify_df)

                        # 查找目标日期的索引
                        target_date_obj = pd.to_datetime(pattern_date).date()
                        target_idx = verify_df[verify_df['date'].dt.date == target_date_obj].index

                        if len(target_idx) > 0 and verify_patterns is not None:
                            idx = target_idx[0]

                            # 验证形态是否确实存在
                            if pattern_name in verify_patterns.columns and idx < len(verify_patterns):
                                pattern_exists = verify_patterns.iloc[idx][pattern_name] if not pd.isna(verify_patterns.iloc[idx][pattern_name]) else False

                                if pattern_exists:
                                    verified_count += 1
                                    verification_detail = {
                                        'stock_code': stock_code,
                                        'pattern_name': pattern_name,
                                        'date': pattern_date,
                                        'verified': True,
                                        'data_source': 'real_clickhouse_verification'
                                    }
                                    result['verification_details'].append(verification_detail)
                                    print(f"      ✅ 双向验证通过（真实数据）")
                                else:
                                    print(f"      ❌ 双向验证失败: 形态不存在")
                            else:
                                print(f"      ❌ 双向验证失败: 无法找到目标日期或形态")
                        else:
                            print(f"      ❌ 双向验证失败: 目标日期不存在")
                    else:
                        print(f"      ❌ 双向验证失败: 验证数据不足")

                except Exception as e:
                    print(f"      ❌ 双向验证异常: {e}")
                    continue

            result['verified_patterns'] = verified_count
            result['verification_rate'] = verified_count / len(patterns_to_verify) if patterns_to_verify else 0

            logger.info(f"✅ 双向验证完成: {verified_count}/{len(patterns_to_verify)}验证通过，验证率: {result['verification_rate']:.1%}")

        except Exception as e:
            logger.error(f"❌ 双向验证异常: {e}")

        return result

    def _comprehensive_assessment_real_data(self, validation_result: Dict) -> Dict[str, Any]:
        """综合评估（真实数据）"""

        assessment = {
            'passed': False,
            'overall_score': 0.0,
            'component_scores': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'real_data_quality': {}
        }

        try:
            # 计算各组件得分
            db_score = 20 if validation_result['database_connection_validation']['connected'] else 0
            data_score = 20 if validation_result['real_data_access_validation']['success'] else 0
            indicator_score = 30 if validation_result['indicator_calculation_validation']['success'] else 0
            pattern_score = 20 if validation_result['pattern_detection_validation']['success'] else 0
            verification_score = 10 if validation_result['bidirectional_verification']['verified_patterns'] > 0 else 0

            assessment['component_scores'] = {
                'database_connection': db_score,
                'real_data_access': data_score,
                'indicator_calculation': indicator_score,
                'pattern_detection': pattern_score,
                'bidirectional_verification': verification_score
            }

            assessment['overall_score'] = sum(assessment['component_scores'].values())

            # 真实数据质量评估
            if validation_result['real_data_access_validation']['success']:
                data_validation = validation_result['real_data_access_validation']
                assessment['real_data_quality'] = {
                    'stocks_count': data_validation['stocks_count'],
                    'total_records': data_validation['total_records'],
                    'data_source': 'ClickHouse真实数据库',
                    'data_completeness': 'HIGH' if data_validation['total_records'] > 1000 else 'MEDIUM'
                }

            # 评估优势和劣势
            if db_score > 0:
                assessment['strengths'].append("真实数据库连接正常")
            else:
                assessment['weaknesses'].append("真实数据库连接异常")
                assessment['recommendations'].append("检查ClickHouse数据库连接配置")

            if data_score > 0:
                assessment['strengths'].append("真实数据访问正常")
            else:
                assessment['weaknesses'].append("真实数据访问异常")
                assessment['recommendations'].append("检查数据访问接口实现")

            if indicator_score > 0:
                assessment['strengths'].append(f"{validation_result['indicator_name']}指标计算正常（真实数据）")
            else:
                assessment['weaknesses'].append(f"{validation_result['indicator_name']}指标计算异常")
                assessment['recommendations'].append("检查指标计算逻辑")

            if pattern_score > 0:
                assessment['strengths'].append(f"{validation_result['indicator_name']}形态检测正常（真实数据）")
            else:
                assessment['weaknesses'].append(f"{validation_result['indicator_name']}形态检测异常")
                assessment['recommendations'].append("检查形态检测逻辑")

            if verification_score > 0:
                assessment['strengths'].append("双向验证通过（真实数据）")
            else:
                assessment['weaknesses'].append("双向验证需要改进")
                assessment['recommendations'].append("提高双向验证逻辑的准确性")

            # 通过标准：总分>=80分
            assessment['passed'] = assessment['overall_score'] >= 80

            logger.info(f"✅ 综合评估完成: 总分{assessment['overall_score']}/100，通过: {assessment['passed']}")

        except Exception as e:
            assessment['recommendations'].append(f"综合评估异常: {str(e)}")
            logger.error(f"❌ 综合评估异常: {e}")

        return assessment

def main():
    """主函数 - 演示真实数据验证器"""

    # 导入指标类
    from indicators.macd import MacdMacd

    try:
        # 创建真实数据验证器
        validator = RealDataValidator("MACD", MacdMacd())

        # 运行真实数据验证
        results = validator.run_real_data_validation()

        print("\n" + "="*80)
        print(f"🏆 {validator.indicator_name}真实数据验证结果汇总")
        print("="*80)

        print(f"🎯 验证通过: {results['validation_passed']}")
        print(f"📊 指标类型: {results['indicator_type']}")

        if results['database_connection_validation']:
            db_res = results['database_connection_validation']
            print(f"🔌 数据库连接: {db_res['connected']}")
            if db_res['connected']:
                print(f"    数据库类型: {db_res['database_info'].get('database_type', 'Unknown')}")

        if results['real_data_access_validation']:
            data_res = results['real_data_access_validation']
            print(f"📊 真实数据访问: {data_res['success']} (股票: {data_res['stocks_count']}支, 记录: {data_res['total_records']}条)")

        if results['indicator_calculation_validation']:
            calc_res = results['indicator_calculation_validation']
            success_rate = calc_res['calculations_successful'] / calc_res['total_calculations'] * 100 if calc_res['total_calculations'] > 0 else 0
            print(f"🎯 指标计算: {calc_res['success']} (成功率: {success_rate:.1f}%)")

        if results['pattern_detection_validation']:
            pattern_res = results['pattern_detection_validation']
            print(f"🔍 形态检测: {pattern_res['success']} (检测: {pattern_res['patterns_detected']}个形态)")
            if pattern_res['pattern_distribution']:
                print(f"    形态分布: {pattern_res['pattern_distribution']}")

        if results['bidirectional_verification']:
            verify_res = results['bidirectional_verification']
            print(f"🛡️ 双向验证: {verify_res['verified_patterns']}个形态验证通过，验证率: {verify_res['verification_rate']:.1%}")

        if results['overall_assessment']:
            assessment = results['overall_assessment']
            print(f"📊 综合得分: {assessment['overall_score']:.1f}/100")

            if assessment['real_data_quality']:
                quality = assessment['real_data_quality']
                print(f"📈 数据质量: {quality['data_source']}, 完整性: {quality['data_completeness']}")

            if assessment['strengths']:
                print(f"💪 优势: {', '.join(assessment['strengths'])}")

            if assessment['weaknesses']:
                print(f"⚠️ 劣势: {', '.join(assessment['weaknesses'])}")

            if assessment['recommendations']:
                print(f"💡 建议: {', '.join(assessment['recommendations'])}")

        if results['validation_passed']:
            print(f"\n🎉 {validator.indicator_name}指标真实数据验证通过！")
            print(f"✅ 成功连接真实ClickHouse数据库")
            print(f"✅ 使用真实股票市场数据进行验证")
            print(f"✅ 严格遵循架构规范，不直接写SQL")
            print(f"✅ 通过数据层服务进行所有数据操作")
        else:
            print(f"\n🔧 {validator.indicator_name}指标真实数据验证需要改进")
            if results['issues_found']:
                print(f"❌ 问题: {', '.join(results['issues_found'])}")

    except Exception as e:
        logger.error(f"真实数据验证演示失败: {e}")
        print(f"❌ 真实数据验证演示失败: {e}")

if __name__ == "__main__":
    main()
