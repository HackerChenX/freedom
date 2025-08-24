#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用技术指标验证器使用示例

展示如何为不同的技术指标创建验证器实例，
严格遵循架构要求，使用数据层接口。
"""

import sys
import os
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from universal_indicator_validator import UniversalIndicatorValidator, create_indicator_validator
from utils.logger import get_logger

logger = get_logger(__name__)

def create_p0_validators() -> List[UniversalIndicatorValidator]:
    """创建P0核心指标验证器"""
    
    validators = []
    
    try:
        # P0核心指标
        from indicators.macd import MacdMacd
        from indicators.rsi import RsiRsi
        from indicators.kdj import KdjKdj
        from indicators.boll import BollBoll
        from indicators.ma import MaMa
        from indicators.ema import EmaEma
        
        p0_indicators = [
            ("MACD", MacdMacd),
            ("RSI", RsiRsi),
            ("KDJ", KdjKdj),
            ("BOLL", BollBoll),
            ("MA", MaMa),
            ("EMA", EmaEma)
        ]
        
        print("🔥 创建P0核心指标验证器")
        for indicator_name, indicator_class in p0_indicators:
            try:
                validator = UniversalIndicatorValidator(indicator_name, indicator_class())
                validators.append(validator)
                print(f"  ✅ {indicator_name}验证器创建成功")
            except Exception as e:
                print(f"  ❌ {indicator_name}验证器创建失败: {e}")
                
    except ImportError as e:
        logger.error(f"导入P0指标类失败: {e}")
    
    return validators

def create_p1_validators() -> List[UniversalIndicatorValidator]:
    """创建P1重要指标验证器"""
    
    validators = []
    
    try:
        # P1重要指标
        from indicators.atr import AtrAtr
        from indicators.cci import CciCci
        from indicators.mfi import MfiMfi
        from indicators.obv import ObvObv
        from indicators.stochrsi import StochRsiStochRsi
        
        p1_indicators = [
            ("ATR", AtrAtr),
            ("CCI", CciCci),
            ("MFI", MfiMfi),
            ("OBV", ObvObv),
            ("STOCHRSI", StochRsiStochRsi)
        ]
        
        print("🔥 创建P1重要指标验证器")
        for indicator_name, indicator_class in p1_indicators:
            try:
                validator = UniversalIndicatorValidator(indicator_name, indicator_class())
                validators.append(validator)
                print(f"  ✅ {indicator_name}验证器创建成功")
            except Exception as e:
                print(f"  ❌ {indicator_name}验证器创建失败: {e}")
                
    except ImportError as e:
        logger.error(f"导入P1指标类失败: {e}")
    
    return validators

def run_batch_validation(validators: List[UniversalIndicatorValidator]) -> Dict[str, Any]:
    """批量运行验证器"""
    
    batch_results = {
        'total_validators': len(validators),
        'passed_validators': 0,
        'failed_validators': 0,
        'validation_details': {},
        'summary': {}
    }
    
    print(f"\n🚀 开始批量验证 {len(validators)} 个指标")
    print("=" * 80)
    
    for i, validator in enumerate(validators, 1):
        indicator_name = validator.indicator_name
        
        try:
            print(f"\n📊 [{i}/{len(validators)}] 验证 {indicator_name} 指标")
            print("-" * 60)
            
            # 运行验证
            result = validator.run_universal_validation()
            
            # 记录结果
            batch_results['validation_details'][indicator_name] = result
            
            if result['validation_passed']:
                batch_results['passed_validators'] += 1
                print(f"✅ {indicator_name} 验证通过")
            else:
                batch_results['failed_validators'] += 1
                print(f"❌ {indicator_name} 验证失败")
                
        except Exception as e:
            batch_results['failed_validators'] += 1
            batch_results['validation_details'][indicator_name] = {
                'validation_passed': False,
                'error': str(e)
            }
            print(f"❌ {indicator_name} 验证异常: {e}")
    
    # 生成汇总
    batch_results['summary'] = {
        'success_rate': batch_results['passed_validators'] / batch_results['total_validators'] * 100,
        'passed_indicators': [name for name, result in batch_results['validation_details'].items() 
                             if result.get('validation_passed', False)],
        'failed_indicators': [name for name, result in batch_results['validation_details'].items() 
                             if not result.get('validation_passed', False)]
    }
    
    return batch_results

def print_batch_summary(batch_results: Dict[str, Any]):
    """打印批量验证汇总"""
    
    print("\n" + "=" * 80)
    print("🏆 批量验证结果汇总")
    print("=" * 80)
    
    summary = batch_results['summary']
    
    print(f"📊 总体统计:")
    print(f"  总验证器数: {batch_results['total_validators']}")
    print(f"  通过验证: {batch_results['passed_validators']}")
    print(f"  验证失败: {batch_results['failed_validators']}")
    print(f"  成功率: {summary['success_rate']:.1f}%")
    
    if summary['passed_indicators']:
        print(f"\n✅ 通过验证的指标 ({len(summary['passed_indicators'])}个):")
        for indicator in summary['passed_indicators']:
            result = batch_results['validation_details'][indicator]
            patterns_count = len(result.get('pattern_detection_results', {}).get('patterns_found', []))
            verified_count = len(result.get('bidirectional_verification', {}).get('verified_patterns', []))
            print(f"  {indicator}: 发现{patterns_count}个形态，验证{verified_count}个")
    
    if summary['failed_indicators']:
        print(f"\n❌ 验证失败的指标 ({len(summary['failed_indicators'])}个):")
        for indicator in summary['failed_indicators']:
            result = batch_results['validation_details'][indicator]
            if 'error' in result:
                print(f"  {indicator}: 异常 - {result['error']}")
            else:
                issues = result.get('issues_found', ['未知问题'])
                print(f"  {indicator}: {', '.join(issues)}")

def demonstrate_universal_validator():
    """演示通用验证器的使用"""
    
    print("🎯 通用技术指标验证器演示")
    print("=" * 80)
    print("📋 目标: 展示如何为不同技术指标创建验证器")
    print("🏗️ 架构: 严格遵循数据层接口，不直接写SQL")
    print("🔧 特性: 通用化设计，支持任意技术指标")
    print("=" * 80)
    
    # 创建P0核心指标验证器
    p0_validators = create_p0_validators()
    
    if p0_validators:
        print(f"\n✅ 成功创建 {len(p0_validators)} 个P0核心指标验证器")
        
        # 运行单个验证器示例
        if len(p0_validators) > 0:
            print(f"\n🔍 运行单个验证器示例: {p0_validators[0].indicator_name}")
            single_result = p0_validators[0].run_universal_validation()
            
            print(f"\n📋 {p0_validators[0].indicator_name} 验证结果:")
            print(f"  验证通过: {single_result['validation_passed']}")
            print(f"  支持形态: {', '.join(single_result['supported_patterns'])}")
            
            if single_result['pattern_detection_results']:
                patterns_found = len(single_result['pattern_detection_results']['patterns_found'])
                print(f"  发现形态: {patterns_found}个")
            
            if single_result['bidirectional_verification']:
                verified_patterns = len(single_result['bidirectional_verification']['verified_patterns'])
                verification_rate = single_result['bidirectional_verification']['verification_rate']
                print(f"  验证通过: {verified_patterns}个 ({verification_rate:.1%})")
        
        # 批量验证示例（仅前3个，避免时间过长）
        print(f"\n🚀 批量验证示例（前3个指标）")
        batch_validators = p0_validators[:3]
        batch_results = run_batch_validation(batch_validators)
        print_batch_summary(batch_results)
        
    else:
        print("❌ 未能创建任何验证器")
    
    # 展示工厂方法使用
    print(f"\n💡 工厂方法使用示例:")
    print(f"# 使用工厂方法创建验证器")
    print(f"# from indicators.macd import MacdMacd")
    print(f"# macd_validator = create_indicator_validator('MACD', MacdMacd)")
    print(f"# result = macd_validator.run_universal_validation()")
    
    # 展示架构合规性
    print(f"\n🏗️ 架构合规性特点:")
    print(f"✅ 数据层接口: 使用DataLayerInterface封装数据访问")
    print(f"✅ 不直接写SQL: 所有数据查询通过数据层方法")
    print(f"✅ 通用化设计: 支持任意技术指标的验证")
    print(f"✅ 标准化报告: 统一的验证结果格式")
    print(f"✅ 可复用性: 一次开发，所有指标都能使用")

def main():
    """主函数"""
    try:
        demonstrate_universal_validator()
    except Exception as e:
        logger.error(f"演示失败: {e}")
        print(f"❌ 演示失败: {e}")

if __name__ == "__main__":
    main()
