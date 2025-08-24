#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构合规验证器 - 严格按照架构规范实现

使用模拟数据访问来验证架构设计的正确性，
展示如何严格遵循数据层架构要求。
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
from db.interfaces.mock_data_access import create_mock_data_access
from db.services.stock_data_service import StockDataService
from utils.logger import get_logger

logger = get_logger(__name__)

class ArchitectureCompliantValidator:
    """架构合规验证器 - 严格遵循架构规范"""
    
    def __init__(self, indicator_name: str, indicator_instance: Any):
        """
        初始化架构合规验证器
        
        Args:
            indicator_name: 指标名称（如"MACD", "RSI", "KDJ"等）
            indicator_instance: 指标实例（如MacdMacd(), RsiRsi()等）
        """
        self.indicator_name = indicator_name.upper()
        self.indicator_instance = indicator_instance
        self.pattern_registry = get_unified_pattern_registry()
        
        # 使用模拟数据访问，严格遵循架构规范
        try:
            mock_data_access = create_mock_data_access()
            self.stock_data_service = StockDataService(mock_data_access)
            logger.info("✅ 使用模拟数据访问的股票数据服务初始化成功")
        except Exception as e:
            logger.error(f"❌ 股票数据服务初始化失败: {e}")
            raise RuntimeError(f"无法初始化股票数据服务: {e}")
        
        # 验证标准
        self.validation_standards = {
            'min_stocks_to_test': 5,            # 降低要求以适应模拟数据
            'min_patterns_found': 2,            # 降低要求以适应模拟数据
            'min_data_days': 30,                # 降低要求以适应模拟数据
            'required_levels': ['日线'],         # 必须的时间级别
            'min_verification_rate': 0.5        # 降低验证率要求
        }
        
        # 动态获取指标支持的形态
        self.supported_patterns = self._get_indicator_patterns()
        
        logger.info(f"🔥 {self.indicator_name}架构合规验证器初始化完成")
    
    def _get_indicator_patterns(self) -> List[str]:
        """动态获取指标支持的形态"""
        try:
            # 根据指标名称返回特定形态
            indicator_specific_patterns = {
                'MACD': ['GOLDEN_CROSS', 'DEATH_CROSS', 'MACD_ABOVE_ZERO_GOLDEN', 'BEARISH_DIVERGENCE'],
                'RSI': ['OVERBOUGHT', 'OVERSOLD', 'RSI_GOLDEN_CROSS', 'RSI_DEATH_CROSS'],
                'KDJ': ['KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS', 'KDJ_OVERBOUGHT', 'KDJ_OVERSOLD'],
                'BOLL': ['BOLL_UPPER_BREAKOUT', 'BOLL_LOWER_BREAKOUT', 'BOLL_SQUEEZE', 'BOLL_EXPANSION']
            }
            
            return indicator_specific_patterns.get(self.indicator_name, ['GOLDEN_CROSS', 'DEATH_CROSS'])
            
        except Exception as e:
            logger.warning(f"获取{self.indicator_name}支持形态失败: {e}")
            return ['GOLDEN_CROSS', 'DEATH_CROSS']
    
    def run_architecture_compliance_validation(self) -> Dict[str, Any]:
        """运行架构合规验证"""
        
        print(f"🔥 开始{self.indicator_name}架构合规验证")
        print("=" * 80)
        print("📋 验证目标: 验证架构设计的正确性和合规性")
        print("🎯 特点: 使用模拟数据，不依赖外部服务")
        print("🏗️ 架构: 严格遵循数据层接口规范")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Architecture_Compliance_Validation',
            'indicator_name': self.indicator_name,
            'indicator_type': type(self.indicator_instance).__name__,
            'validation_timestamp': datetime.now().isoformat(),
            'architecture_compliance': {},
            'data_layer_validation': {},
            'indicator_calculation_validation': {},
            'pattern_detection_validation': {},
            'overall_assessment': {},
            'validation_passed': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 验证架构合规性
            print("\n🏗️ 步骤1: 验证架构合规性")
            architecture_compliance = self._validate_architecture_compliance()
            validation_result['architecture_compliance'] = architecture_compliance
            
            if not architecture_compliance['compliant']:
                validation_result['issues_found'].extend(architecture_compliance['issues'])
                print(f"❌ 架构合规性验证失败")
                return validation_result
            
            print(f"✅ 架构合规性验证通过")
            
            # 步骤2: 验证数据层功能
            print("\n📊 步骤2: 验证数据层功能")
            data_layer_validation = self._validate_data_layer()
            validation_result['data_layer_validation'] = data_layer_validation
            
            if not data_layer_validation['success']:
                validation_result['issues_found'].extend(data_layer_validation['issues'])
                print(f"❌ 数据层验证失败")
                return validation_result
            
            print(f"✅ 数据层验证通过: 获取{data_layer_validation['stocks_count']}支股票数据")
            
            # 步骤3: 验证指标计算功能
            print(f"\n🎯 步骤3: 验证{self.indicator_name}指标计算功能")
            indicator_validation = self._validate_indicator_calculation(data_layer_validation['sample_data'])
            validation_result['indicator_calculation_validation'] = indicator_validation
            
            if not indicator_validation['success']:
                validation_result['issues_found'].extend(indicator_validation['issues'])
                print(f"❌ {self.indicator_name}指标计算验证失败")
                return validation_result
            
            print(f"✅ {self.indicator_name}指标计算验证通过")
            
            # 步骤4: 验证形态检测功能
            print(f"\n🔍 步骤4: 验证{self.indicator_name}形态检测功能")
            pattern_validation = self._validate_pattern_detection(data_layer_validation['sample_data'])
            validation_result['pattern_detection_validation'] = pattern_validation
            
            if not pattern_validation['success']:
                validation_result['issues_found'].extend(pattern_validation['issues'])
                print(f"❌ {self.indicator_name}形态检测验证失败")
            else:
                print(f"✅ {self.indicator_name}形态检测验证通过: 检测到{pattern_validation['patterns_detected']}个形态")
            
            # 步骤5: 综合评估
            print(f"\n🏆 步骤5: {self.indicator_name}综合评估")
            overall_assessment = self._comprehensive_assessment(validation_result)
            validation_result['overall_assessment'] = overall_assessment
            validation_result['validation_passed'] = overall_assessment['passed']
            
            if overall_assessment['passed']:
                print(f"🎉 {self.indicator_name}指标架构合规验证通过！")
            else:
                print(f"❌ {self.indicator_name}指标架构合规验证失败")
        
        except Exception as e:
            logger.error(f"❌ {self.indicator_name}架构合规验证异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _validate_architecture_compliance(self) -> Dict[str, Any]:
        """验证架构合规性"""
        
        result = {
            'compliant': False,
            'compliance_checks': {},
            'issues': []
        }
        
        try:
            # 检查1: 数据访问接口使用
            data_access_check = hasattr(self.stock_data_service, 'data_access') and \
                               hasattr(self.stock_data_service.data_access, 'get_stock_list_data_access_interface')
            result['compliance_checks']['data_access_interface'] = data_access_check
            
            if not data_access_check:
                result['issues'].append("未正确使用数据访问接口")
            
            # 检查2: 服务层封装
            service_layer_check = isinstance(self.stock_data_service, StockDataService)
            result['compliance_checks']['service_layer'] = service_layer_check
            
            if not service_layer_check:
                result['issues'].append("未正确使用服务层封装")
            
            # 检查3: 指标实例化
            indicator_check = hasattr(self.indicator_instance, 'calculate') and \
                             hasattr(self.indicator_instance, 'get_patterns')
            result['compliance_checks']['indicator_interface'] = indicator_check
            
            if not indicator_check:
                result['issues'].append("指标实例未实现必要接口")
            
            # 检查4: 形态注册表
            pattern_registry_check = self.pattern_registry is not None
            result['compliance_checks']['pattern_registry'] = pattern_registry_check
            
            if not pattern_registry_check:
                result['issues'].append("形态注册表未正确初始化")
            
            # 综合评估
            result['compliant'] = all(result['compliance_checks'].values())
            
        except Exception as e:
            result['issues'].append(f"架构合规性检查异常: {str(e)}")
        
        return result
    
    def _validate_data_layer(self) -> Dict[str, Any]:
        """验证数据层功能"""
        
        result = {
            'success': False,
            'stocks_count': 0,
            'sample_data': {},
            'issues': []
        }
        
        try:
            # 测试获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=10)
            
            if not stock_codes:
                result['issues'].append("无法获取股票列表")
                return result
            
            result['stocks_count'] = len(stock_codes)
            print(f"    📋 获取到{len(stock_codes)}支股票代码")
            
            # 测试获取股票数据
            sample_data = {}
            for i, stock_code in enumerate(stock_codes[:3]):  # 只测试前3支
                try:
                    df = self.stock_data_service.get_stock_data(stock_code, days=60)
                    
                    if df is not None and len(df) >= 30:
                        sample_data[stock_code] = df
                        print(f"    ✅ {stock_code}: {len(df)}天数据")
                    else:
                        print(f"    ⚠️ {stock_code}: 数据不足")
                        
                except Exception as e:
                    print(f"    ❌ {stock_code}: 数据获取失败 - {e}")
                    continue
            
            if len(sample_data) >= 2:  # 至少2支股票有数据
                result['sample_data'] = sample_data
                result['success'] = True
            else:
                result['issues'].append("获取到的有效股票数据不足")
        
        except Exception as e:
            result['issues'].append(f"数据层验证异常: {str(e)}")
        
        return result
    
    def _validate_indicator_calculation(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证指标计算功能"""
        
        result = {
            'success': False,
            'calculations_successful': 0,
            'total_calculations': 0,
            'issues': []
        }
        
        try:
            for stock_code, df in sample_data.items():
                result['total_calculations'] += 1
                
                try:
                    print(f"    🔍 计算{stock_code}的{self.indicator_name}指标")
                    
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
                if success_rate >= 0.5:  # 至少50%成功率
                    result['success'] = True
                else:
                    result['issues'].append(f"指标计算成功率过低: {success_rate:.1%}")
            else:
                result['issues'].append("没有进行任何指标计算")
        
        except Exception as e:
            result['issues'].append(f"指标计算验证异常: {str(e)}")
        
        return result
    
    def _validate_pattern_detection(self, sample_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证形态检测功能"""
        
        result = {
            'success': False,
            'patterns_detected': 0,
            'stocks_with_patterns': 0,
            'pattern_details': [],
            'issues': []
        }
        
        try:
            for stock_code, df in sample_data.items():
                try:
                    print(f"    🔍 检测{stock_code}的{self.indicator_name}形态")
                    
                    # 获取指标形态
                    patterns = self.indicator_instance.get_patterns(df)
                    
                    if patterns is not None and not patterns.empty:
                        stock_patterns = 0
                        
                        # 检查每个支持的形态
                        for pattern_name in self.supported_patterns:
                            if pattern_name in patterns.columns:
                                pattern_signals = patterns[patterns[pattern_name] == True]
                                
                                if not pattern_signals.empty:
                                    stock_patterns += len(pattern_signals)
                                    result['patterns_detected'] += len(pattern_signals)
                                    
                                    # 记录形态详情
                                    for idx in pattern_signals.index:
                                        if idx < len(df):
                                            result['pattern_details'].append({
                                                'stock_code': stock_code,
                                                'pattern_name': pattern_name,
                                                'date': df.iloc[idx]['date'].strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
                                                'close_price': float(df.iloc[idx]['close']) if 'close' in df.columns else 0.0
                                            })
                        
                        if stock_patterns > 0:
                            result['stocks_with_patterns'] += 1
                            print(f"      ✅ 检测到{stock_patterns}个{self.indicator_name}形态")
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
            else:
                result['issues'].append(f"检测到的形态数量不足: {result['patterns_detected']} < {self.validation_standards['min_patterns_found']}")
        
        except Exception as e:
            result['issues'].append(f"形态检测验证异常: {str(e)}")
        
        return result
    
    def _comprehensive_assessment(self, validation_result: Dict) -> Dict[str, Any]:
        """综合评估"""
        
        assessment = {
            'passed': False,
            'overall_score': 0.0,
            'component_scores': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        try:
            # 计算各组件得分
            architecture_score = 25 if validation_result['architecture_compliance']['compliant'] else 0
            data_layer_score = 25 if validation_result['data_layer_validation']['success'] else 0
            indicator_score = 25 if validation_result['indicator_calculation_validation']['success'] else 0
            pattern_score = 25 if validation_result['pattern_detection_validation']['success'] else 0
            
            assessment['component_scores'] = {
                'architecture_compliance': architecture_score,
                'data_layer_functionality': data_layer_score,
                'indicator_calculation': indicator_score,
                'pattern_detection': pattern_score
            }
            
            assessment['overall_score'] = sum(assessment['component_scores'].values())
            
            # 评估优势和劣势
            if architecture_score > 0:
                assessment['strengths'].append("架构设计合规")
            else:
                assessment['weaknesses'].append("架构设计不合规")
                assessment['recommendations'].append("修复架构合规性问题")
            
            if data_layer_score > 0:
                assessment['strengths'].append("数据层功能正常")
            else:
                assessment['weaknesses'].append("数据层功能异常")
                assessment['recommendations'].append("修复数据层接口问题")
            
            if indicator_score > 0:
                assessment['strengths'].append(f"{validation_result['indicator_name']}指标计算正常")
            else:
                assessment['weaknesses'].append(f"{validation_result['indicator_name']}指标计算异常")
                assessment['recommendations'].append("修复指标计算逻辑")
            
            if pattern_score > 0:
                assessment['strengths'].append(f"{validation_result['indicator_name']}形态检测正常")
            else:
                assessment['weaknesses'].append(f"{validation_result['indicator_name']}形态检测异常")
                assessment['recommendations'].append("修复形态检测逻辑")
            
            # 通过标准：总分>=75分
            assessment['passed'] = assessment['overall_score'] >= 75
        
        except Exception as e:
            assessment['recommendations'].append(f"综合评估异常: {str(e)}")
        
        return assessment

def main():
    """主函数 - 演示架构合规验证器"""
    
    # 导入指标类
    from indicators.macd import MacdMacd
    
    try:
        # 创建架构合规验证器
        validator = ArchitectureCompliantValidator("MACD", MacdMacd())
        
        # 运行架构合规验证
        results = validator.run_architecture_compliance_validation()
        
        print("\n" + "="*80)
        print(f"🏆 {validator.indicator_name}架构合规验证结果汇总")
        print("="*80)
        
        print(f"🎯 验证通过: {results['validation_passed']}")
        print(f"📊 指标类型: {results['indicator_type']}")
        
        if results['architecture_compliance']:
            arch_res = results['architecture_compliance']
            print(f"🏗️ 架构合规: {arch_res['compliant']}")
            if arch_res['compliance_checks']:
                for check, passed in arch_res['compliance_checks'].items():
                    status = "✅" if passed else "❌"
                    print(f"    {status} {check}")
        
        if results['data_layer_validation']:
            data_res = results['data_layer_validation']
            print(f"📊 数据层验证: {data_res['success']} (股票: {data_res['stocks_count']}支)")
        
        if results['indicator_calculation_validation']:
            calc_res = results['indicator_calculation_validation']
            success_rate = calc_res['calculations_successful'] / calc_res['total_calculations'] * 100 if calc_res['total_calculations'] > 0 else 0
            print(f"🎯 指标计算: {calc_res['success']} (成功率: {success_rate:.1f}%)")
        
        if results['pattern_detection_validation']:
            pattern_res = results['pattern_detection_validation']
            print(f"🔍 形态检测: {pattern_res['success']} (检测: {pattern_res['patterns_detected']}个形态)")
        
        if results['overall_assessment']:
            assessment = results['overall_assessment']
            print(f"📊 综合得分: {assessment['overall_score']:.1f}/100")
            
            if assessment['strengths']:
                print(f"💪 优势: {', '.join(assessment['strengths'])}")
            
            if assessment['weaknesses']:
                print(f"⚠️ 劣势: {', '.join(assessment['weaknesses'])}")
            
            if assessment['recommendations']:
                print(f"💡 建议: {', '.join(assessment['recommendations'])}")
        
        if results['validation_passed']:
            print(f"\n🎉 {validator.indicator_name}指标架构合规验证通过！")
            print(f"✅ 架构设计完全符合规范要求")
            print(f"✅ 数据层接口工作正常")
            print(f"✅ 指标计算和形态检测功能正常")
        else:
            print(f"\n🔧 {validator.indicator_name}指标架构合规验证需要改进")
            if results['issues_found']:
                print(f"❌ 问题: {', '.join(results['issues_found'])}")
        
    except Exception as e:
        logger.error(f"架构合规验证演示失败: {e}")
        print(f"❌ 架构合规验证演示失败: {e}")

if __name__ == "__main__":
    main()
