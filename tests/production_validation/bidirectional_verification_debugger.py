#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向验证调试器 - 分析双向验证失败的原因

深入分析为什么双向验证会出现"形态不存在"的问题：
1. 数据一致性问题
2. 时间窗口问题
3. 指标计算差异
4. 形态检测逻辑问题
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from db.interfaces.data_access_interface import ClickHouseDataAccess
from db.services.stock_data_service import StockDataService
from utils.logger import get_logger

logger = get_logger(__name__)

class BidirectionalVerificationDebugger:
    """双向验证调试器"""
    
    def __init__(self, indicator_name: str, indicator_instance: Any):
        """初始化调试器"""
        self.indicator_name = indicator_name.upper()
        self.indicator_instance = indicator_instance
        
        # 使用真实数据访问
        clickhouse_data_access = ClickHouseDataAccess()
        self.stock_data_service = StockDataService(clickhouse_data_access)
        
        logger.info(f"🔍 {self.indicator_name}双向验证调试器初始化完成")
    
    def debug_pattern_verification(self, stock_code: str, pattern_name: str, pattern_date: str) -> Dict[str, Any]:
        """调试单个形态的双向验证"""
        
        debug_result = {
            'stock_code': stock_code,
            'pattern_name': pattern_name,
            'pattern_date': pattern_date,
            'original_detection': {},
            'verification_detection': {},
            'data_comparison': {},
            'issues_found': [],
            'verification_passed': False
        }
        
        print(f"\n🔍 调试双向验证: {stock_code} - {pattern_name} @ {pattern_date}")
        print("=" * 60)
        
        try:
            # 步骤1: 获取原始检测数据
            print("📊 步骤1: 获取原始检测数据")
            original_data = self._get_original_detection_data(stock_code, pattern_date)
            debug_result['original_detection'] = original_data
            
            if not original_data['success']:
                debug_result['issues_found'].extend(original_data['issues'])
                return debug_result
            
            # 步骤2: 获取验证数据
            print("🔍 步骤2: 获取验证数据")
            verification_data = self._get_verification_data(stock_code, pattern_date)
            debug_result['verification_detection'] = verification_data
            
            if not verification_data['success']:
                debug_result['issues_found'].extend(verification_data['issues'])
                return debug_result
            
            # 步骤3: 比较数据差异
            print("📈 步骤3: 比较数据差异")
            data_comparison = self._compare_data_differences(
                original_data['data'], 
                verification_data['data'],
                pattern_date
            )
            debug_result['data_comparison'] = data_comparison
            
            # 步骤4: 分析形态检测差异
            print("🎯 步骤4: 分析形态检测差异")
            pattern_analysis = self._analyze_pattern_detection_differences(
                original_data, 
                verification_data, 
                pattern_name, 
                pattern_date
            )
            debug_result.update(pattern_analysis)
            
            # 步骤5: 提供修复建议
            print("💡 步骤5: 提供修复建议")
            recommendations = self._provide_fix_recommendations(debug_result)
            debug_result['recommendations'] = recommendations
            
        except Exception as e:
            debug_result['issues_found'].append(f"调试过程异常: {str(e)}")
            logger.error(f"❌ 调试过程异常: {e}")
        
        return debug_result
    
    def _get_original_detection_data(self, stock_code: str, pattern_date: str) -> Dict[str, Any]:
        """获取原始检测数据"""
        
        result = {
            'success': False,
            'data': None,
            'patterns': None,
            'issues': []
        }
        
        try:
            # 获取较长时间窗口的数据（用于原始检测）
            df = self.stock_data_service.get_stock_data(stock_code, days=120)
            
            if df is None or df.empty:
                result['issues'].append("无法获取原始股票数据")
                return result
            
            print(f"    📊 原始数据: {len(df)}天数据，日期范围: {df['date'].min()} 到 {df['date'].max()}")
            
            # 计算指标和形态
            indicator_result = self.indicator_instance.calculate(df)
            patterns = self.indicator_instance.get_patterns(df)
            
            if patterns is None or patterns.empty:
                result['issues'].append("原始形态检测返回空结果")
                return result
            
            result['success'] = True
            result['data'] = df
            result['patterns'] = patterns
            result['indicator_result'] = indicator_result
            
            print(f"    ✅ 原始检测成功: 指标{len(indicator_result)}条，形态{len(patterns)}条")
            
        except Exception as e:
            result['issues'].append(f"获取原始检测数据失败: {str(e)}")
            logger.error(f"❌ 获取原始检测数据失败: {e}")
        
        return result
    
    def _get_verification_data(self, stock_code: str, pattern_date: str) -> Dict[str, Any]:
        """获取验证数据"""
        
        result = {
            'success': False,
            'data': None,
            'patterns': None,
            'issues': []
        }
        
        try:
            # 获取验证时间窗口的数据
            target_date = datetime.strptime(pattern_date, '%Y-%m-%d')
            start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
            
            verify_df = self.stock_data_service.get_stock_data_by_date_range(
                stock_code, start_date, end_date
            )
            
            if verify_df is None or verify_df.empty:
                result['issues'].append("无法获取验证数据")
                return result
            
            verify_df = verify_df.sort_values('date').reset_index(drop=True)
            
            print(f"    📊 验证数据: {len(verify_df)}天数据，日期范围: {verify_df['date'].min()} 到 {verify_df['date'].max()}")
            
            # 重新计算指标和形态
            verify_indicator = self.indicator_instance.calculate(verify_df)
            verify_patterns = self.indicator_instance.get_patterns(verify_df)
            
            if verify_patterns is None or verify_patterns.empty:
                result['issues'].append("验证形态检测返回空结果")
                return result
            
            result['success'] = True
            result['data'] = verify_df
            result['patterns'] = verify_patterns
            result['indicator_result'] = verify_indicator
            
            print(f"    ✅ 验证检测成功: 指标{len(verify_indicator)}条，形态{len(verify_patterns)}条")
            
        except Exception as e:
            result['issues'].append(f"获取验证数据失败: {str(e)}")
            logger.error(f"❌ 获取验证数据失败: {e}")
        
        return result
    
    def _compare_data_differences(self, original_df: pd.DataFrame, verify_df: pd.DataFrame, pattern_date: str) -> Dict[str, Any]:
        """比较数据差异"""
        
        comparison = {
            'data_length_diff': len(original_df) - len(verify_df),
            'date_range_diff': {},
            'target_date_analysis': {},
            'price_data_consistency': {}
        }
        
        try:
            # 日期范围比较
            comparison['date_range_diff'] = {
                'original_start': original_df['date'].min(),
                'original_end': original_df['date'].max(),
                'verify_start': verify_df['date'].min(),
                'verify_end': verify_df['date'].max()
            }
            
            print(f"    📅 数据长度差异: {comparison['data_length_diff']}天")
            print(f"    📅 原始数据范围: {comparison['date_range_diff']['original_start']} 到 {comparison['date_range_diff']['original_end']}")
            print(f"    📅 验证数据范围: {comparison['date_range_diff']['verify_start']} 到 {comparison['date_range_diff']['verify_end']}")
            
            # 目标日期分析
            target_date_obj = pd.to_datetime(pattern_date).date()
            
            # 在原始数据中查找目标日期
            original_target_idx = original_df[original_df['date'].dt.date == target_date_obj].index
            verify_target_idx = verify_df[verify_df['date'].dt.date == target_date_obj].index
            
            comparison['target_date_analysis'] = {
                'target_date': pattern_date,
                'in_original_data': len(original_target_idx) > 0,
                'in_verify_data': len(verify_target_idx) > 0,
                'original_idx': original_target_idx.tolist() if len(original_target_idx) > 0 else [],
                'verify_idx': verify_target_idx.tolist() if len(verify_target_idx) > 0 else []
            }
            
            print(f"    🎯 目标日期{pattern_date}在原始数据中: {comparison['target_date_analysis']['in_original_data']}")
            print(f"    🎯 目标日期{pattern_date}在验证数据中: {comparison['target_date_analysis']['in_verify_data']}")
            
            # 价格数据一致性检查
            if len(original_target_idx) > 0 and len(verify_target_idx) > 0:
                original_row = original_df.iloc[original_target_idx[0]]
                verify_row = verify_df.iloc[verify_target_idx[0]]
                
                comparison['price_data_consistency'] = {
                    'close_price_match': abs(original_row['close'] - verify_row['close']) < 0.01,
                    'original_close': float(original_row['close']),
                    'verify_close': float(verify_row['close']),
                    'price_diff': abs(original_row['close'] - verify_row['close'])
                }
                
                print(f"    💰 价格一致性: {comparison['price_data_consistency']['close_price_match']}")
                print(f"    💰 原始收盘价: {comparison['price_data_consistency']['original_close']}")
                print(f"    💰 验证收盘价: {comparison['price_data_consistency']['verify_close']}")
            
        except Exception as e:
            comparison['error'] = str(e)
            print(f"    ❌ 数据比较异常: {e}")
        
        return comparison
    
    def _analyze_pattern_detection_differences(self, original_data: Dict, verification_data: Dict, 
                                             pattern_name: str, pattern_date: str) -> Dict[str, Any]:
        """分析形态检测差异"""
        
        analysis = {
            'pattern_column_exists': {},
            'pattern_values_comparison': {},
            'verification_passed': False
        }
        
        try:
            target_date_obj = pd.to_datetime(pattern_date).date()
            
            # 检查形态列是否存在
            original_patterns = original_data['patterns']
            verify_patterns = verification_data['patterns']
            
            analysis['pattern_column_exists'] = {
                'in_original': pattern_name in original_patterns.columns,
                'in_verify': pattern_name in verify_patterns.columns
            }
            
            print(f"    🔍 形态列'{pattern_name}'在原始数据中: {analysis['pattern_column_exists']['in_original']}")
            print(f"    🔍 形态列'{pattern_name}'在验证数据中: {analysis['pattern_column_exists']['in_verify']}")
            
            if not analysis['pattern_column_exists']['in_original']:
                analysis['issues'] = [f"原始数据中不存在形态列'{pattern_name}'"]
                return analysis
            
            if not analysis['pattern_column_exists']['in_verify']:
                analysis['issues'] = [f"验证数据中不存在形态列'{pattern_name}'"]
                return analysis
            
            # 查找目标日期的形态值
            original_df = original_data['data']
            verify_df = verification_data['data']
            
            original_target_idx = original_df[original_df['date'].dt.date == target_date_obj].index
            verify_target_idx = verify_df[verify_df['date'].dt.date == target_date_obj].index
            
            if len(original_target_idx) > 0 and len(verify_target_idx) > 0:
                original_idx = original_target_idx[0]
                verify_idx = verify_target_idx[0]
                
                # 获取形态值
                original_pattern_value = original_patterns.iloc[original_idx][pattern_name] if original_idx < len(original_patterns) else None
                verify_pattern_value = verify_patterns.iloc[verify_idx][pattern_name] if verify_idx < len(verify_patterns) else None
                
                analysis['pattern_values_comparison'] = {
                    'original_value': original_pattern_value,
                    'verify_value': verify_pattern_value,
                    'values_match': original_pattern_value == verify_pattern_value,
                    'original_is_true': bool(original_pattern_value) if original_pattern_value is not None else False,
                    'verify_is_true': bool(verify_pattern_value) if verify_pattern_value is not None else False
                }
                
                print(f"    📊 原始形态值: {original_pattern_value}")
                print(f"    📊 验证形态值: {verify_pattern_value}")
                print(f"    📊 值是否匹配: {analysis['pattern_values_comparison']['values_match']}")
                
                # 判断验证是否通过
                analysis['verification_passed'] = analysis['pattern_values_comparison']['verify_is_true']
                
                if analysis['verification_passed']:
                    print(f"    ✅ 双向验证应该通过")
                else:
                    print(f"    ❌ 双向验证失败: 验证数据中形态值为{verify_pattern_value}")
            else:
                analysis['issues'] = ["无法在验证数据中找到目标日期"]
                print(f"    ❌ 无法在验证数据中找到目标日期{pattern_date}")
        
        except Exception as e:
            analysis['error'] = str(e)
            print(f"    ❌ 形态检测分析异常: {e}")
        
        return analysis
    
    def _provide_fix_recommendations(self, debug_result: Dict) -> List[str]:
        """提供修复建议"""
        
        recommendations = []
        
        try:
            # 基于调试结果提供建议
            if debug_result['issues_found']:
                recommendations.append("修复数据获取问题")
            
            data_comparison = debug_result.get('data_comparison', {})
            if data_comparison.get('data_length_diff', 0) > 50:
                recommendations.append("考虑统一原始检测和验证的数据时间窗口")
            
            target_analysis = data_comparison.get('target_date_analysis', {})
            if not target_analysis.get('in_verify_data', True):
                recommendations.append("确保验证数据包含目标日期")
            
            price_consistency = data_comparison.get('price_data_consistency', {})
            if not price_consistency.get('close_price_match', True):
                recommendations.append("检查数据一致性问题")
            
            pattern_values = debug_result.get('pattern_values_comparison', {})
            if not pattern_values.get('values_match', True):
                recommendations.append("分析指标计算差异导致的形态检测不一致")
            
            if not recommendations:
                recommendations.append("双向验证逻辑正常，可能是正常的市场数据变化")
        
        except Exception as e:
            recommendations.append(f"生成建议时异常: {str(e)}")
        
        return recommendations

def main():
    """主函数 - 调试双向验证问题"""
    
    from indicators.macd import MacdMacd
    
    # 创建调试器
    debugger = BidirectionalVerificationDebugger("MACD", MacdMacd())
    
    # 调试失败的案例
    failed_cases = [
        ("000001", "DEATH_CROSS", "2025-03-18"),
        ("000002", "DEATH_CROSS", "2025-05-22"),
        ("000007", "GOLDEN_CROSS", "2025-03-17"),
        ("000007", "DEATH_CROSS", "2025-03-24")
    ]
    
    print("🔍 开始调试双向验证失败案例")
    print("=" * 80)
    
    for i, (stock_code, pattern_name, pattern_date) in enumerate(failed_cases, 1):
        print(f"\n📋 案例{i}: {stock_code} - {pattern_name} @ {pattern_date}")
        
        try:
            debug_result = debugger.debug_pattern_verification(stock_code, pattern_name, pattern_date)
            
            print(f"\n📊 调试结果汇总:")
            print(f"  验证通过: {debug_result.get('verification_passed', False)}")
            
            if debug_result.get('issues_found'):
                print(f"  发现问题: {', '.join(debug_result['issues_found'])}")
            
            if debug_result.get('recommendations'):
                print(f"  修复建议: {', '.join(debug_result['recommendations'])}")
            
        except Exception as e:
            print(f"  ❌ 调试异常: {e}")
        
        if i < len(failed_cases):
            print("\n" + "-" * 60)

if __name__ == "__main__":
    main()
