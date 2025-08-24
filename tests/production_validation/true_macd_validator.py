#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD真正的双向验证器 - 阶段3验证

真正的双向验证逻辑：
1. 正向验证：使用MACD指标计算真实股票数据，识别具体形态
2. 反向验证：验证识别的形态确实存在于指定时间/级别的数据中
3. 详细报告：明确说明哪支股票在什么时间什么级别符合哪个MACD技术形态
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.macd import MacdMacd
from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from db.db_manager import DBManager
from utils.logger import get_logger

logger = get_logger(__name__)

class TrueMACDValidator:
    """MACD真正的双向验证器"""

    def __init__(self):
        """初始化验证器"""
        self.macd = MacdMacd()
        self.pattern_registry = get_unified_pattern_registry()

        # 使用系统数据层接口
        try:
            self.db_manager = DBManager()
            logger.info("✅ 数据管理器初始化成功")
        except Exception as e:
            logger.error(f"❌ 数据管理器初始化失败: {e}")
            raise RuntimeError(f"无法初始化数据管理器: {e}")
        
        # 验证标准
        self.validation_standards = {
            'min_stocks_to_test': 10,           # 至少测试10支股票
            'min_patterns_found': 3,            # 至少找到3个形态
            'min_data_days': 60,                # 至少60天数据
            'required_levels': ['日线'],         # 必须的时间级别
            'macd_patterns': [                  # MACD支持的形态
                'GOLDEN_CROSS',                 # 金叉
                'DEATH_CROSS',                  # 死叉
                'MACD_ABOVE_ZERO_GOLDEN',       # 零轴上方金叉
                'BEARISH_DIVERGENCE'            # 顶背离
            ]
        }
        
        logger.info("🔥 MACD真正的双向验证器初始化完成")
    
    def run_true_validation(self) -> Dict[str, Any]:
        """运行真正的双向验证"""
        
        print("🔥 开始MACD真正的双向验证")
        print("=" * 80)
        print("📋 验证目标: 真正的技术形态识别和双向验证")
        print("🎯 要求: 明确说明哪支股票在什么时间什么级别符合哪个MACD形态")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Stage3_True_Bidirectional',
            'indicator_name': 'MACD',
            'validation_timestamp': datetime.now().isoformat(),
            'stock_analysis_results': {},
            'pattern_detection_results': {},
            'bidirectional_verification': {},
            'detailed_findings': [],
            'validation_passed': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 获取测试股票数据
            print("\n📊 步骤1: 获取测试股票的历史数据")
            stock_data = self._get_test_stock_data()
            validation_result['stock_analysis_results'] = {
                'total_stocks': len(stock_data),
                'stocks_with_sufficient_data': sum(1 for data in stock_data.values() if len(data) >= self.validation_standards['min_data_days'])
            }
            
            if len(stock_data) < self.validation_standards['min_stocks_to_test']:
                validation_result['issues_found'].append(f"测试股票不足: {len(stock_data)} < {self.validation_standards['min_stocks_to_test']}")
                return validation_result
            
            print(f"✅ 获取{len(stock_data)}支股票数据")
            
            # 步骤2: 正向验证 - MACD形态识别
            print("\n🎯 步骤2: 正向验证 - MACD形态识别")
            pattern_results = self._forward_validation(stock_data)
            validation_result['pattern_detection_results'] = pattern_results
            
            if not pattern_results['patterns_found']:
                validation_result['issues_found'].append("未识别到任何MACD形态")
                return validation_result
            
            print(f"✅ 识别到{len(pattern_results['patterns_found'])}个MACD形态")
            
            # 步骤3: 反向验证 - 验证形态的真实性
            print("\n🛡️ 步骤3: 反向验证 - 验证形态的真实性")
            verification_results = self._backward_validation(pattern_results['patterns_found'])
            validation_result['bidirectional_verification'] = verification_results
            
            # 步骤4: 生成详细报告
            print("\n📋 步骤4: 生成详细验证报告")
            detailed_findings = self._generate_detailed_report(pattern_results, verification_results)
            validation_result['detailed_findings'] = detailed_findings
            
            # 步骤5: 综合评估
            print("\n🏆 步骤5: 综合评估")
            validation_passed = self._comprehensive_assessment(validation_result)
            validation_result['validation_passed'] = validation_passed
            
            if validation_passed:
                print(f"🎉 MACD指标通过真正的双向验证！")
            else:
                print(f"❌ MACD指标双向验证失败")
        
        except Exception as e:
            logger.error(f"❌ 验证过程异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _get_test_stock_data(self) -> Dict[str, pd.DataFrame]:
        """获取测试股票的历史数据"""

        stock_data = {}

        try:
            # 使用DBManager获取股票列表
            # 先获取一些活跃股票的代码
            query = """
                SELECT DISTINCT code
                FROM stock_info
                WHERE level = '日线'
                AND close > 5.0
                LIMIT 20
            """

            stock_list_result = self.db_manager.query_manager(query)

            if stock_list_result is None:
                logger.warning("未获取到股票列表")
                return stock_data

            # 提取股票代码
            if isinstance(stock_list_result, list):
                stock_codes = [item[0] if isinstance(item, (list, tuple)) else item for item in stock_list_result]
            else:
                logger.error("股票列表格式不正确")
                return stock_data

            # 限制测试股票数量
            stock_codes = stock_codes[:20]

            for stock_code in stock_codes:
                try:
                    # 使用DBManager获取股票历史数据
                    data_query = f"""
                        SELECT date, open, high, low, close, volume
                        FROM stock_info
                        WHERE code = '{stock_code}'
                        AND level = '日线'
                        ORDER BY date ASC
                        LIMIT 120
                    """

                    raw_data = self.db_manager.query_manager(data_query)

                    if raw_data and len(raw_data) >= self.validation_standards['min_data_days']:
                        # 转换为DataFrame
                        df = pd.DataFrame(raw_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)

                        # 确保数据类型正确
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        stock_data[stock_code] = df
                        print(f"    ✅ {stock_code}: {len(df)}天数据")
                    else:
                        print(f"    ⚠️ {stock_code}: 数据不足({len(raw_data) if raw_data else 0}天)")

                except Exception as e:
                    print(f"    ❌ {stock_code}: 数据获取失败 - {e}")
                    continue

        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")

        return stock_data
    
    def _forward_validation(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """正向验证 - MACD形态识别"""
        
        result = {
            'patterns_found': [],
            'stocks_analyzed': 0,
            'patterns_by_type': {},
            'analysis_details': {}
        }
        
        for stock_code, df in stock_data.items():
            result['stocks_analyzed'] += 1
            
            try:
                print(f"    🔍 分析股票: {stock_code}")
                
                # 使用MACD指标计算
                macd_result = self.macd.calculate(df)
                
                if macd_result is None or macd_result.empty:
                    print(f"      ❌ MACD计算失败")
                    continue
                
                # 获取MACD形态
                patterns = self.macd.get_patterns(df)
                
                if patterns is None or patterns.empty:
                    print(f"      ⚠️ 未识别到形态")
                    continue
                
                # 分析每个形态
                for pattern_name in self.validation_standards['macd_patterns']:
                    if pattern_name in patterns.columns:
                        pattern_signals = patterns[patterns[pattern_name] == True]
                        
                        if not pattern_signals.empty:
                            for idx in pattern_signals.index:
                                pattern_info = {
                                    'stock_code': stock_code,
                                    'pattern_name': pattern_name,
                                    'date': df.iloc[idx]['date'].strftime('%Y-%m-%d'),
                                    'level': '日线',
                                    'close_price': float(df.iloc[idx]['close']),
                                    'macd_value': float(macd_result.iloc[idx]['MACD']) if 'MACD' in macd_result.columns else None,
                                    'signal_value': float(macd_result.iloc[idx]['Signal']) if 'Signal' in macd_result.columns else None,
                                    'histogram_value': float(macd_result.iloc[idx]['Histogram']) if 'Histogram' in macd_result.columns else None
                                }
                                
                                result['patterns_found'].append(pattern_info)
                                
                                # 按类型统计
                                if pattern_name not in result['patterns_by_type']:
                                    result['patterns_by_type'][pattern_name] = 0
                                result['patterns_by_type'][pattern_name] += 1
                                
                                print(f"      ✅ 发现{pattern_name}: {pattern_info['date']}, 价格: {pattern_info['close_price']:.2f}")
                
                result['analysis_details'][stock_code] = {
                    'data_points': len(df),
                    'macd_calculated': macd_result is not None,
                    'patterns_detected': len([p for p in result['patterns_found'] if p['stock_code'] == stock_code])
                }
                
            except Exception as e:
                print(f"      ❌ 分析失败: {e}")
                result['analysis_details'][stock_code] = {
                    'error': str(e)
                }
                continue
        
        return result
    
    def _backward_validation(self, patterns_found: List[Dict]) -> Dict[str, Any]:
        """反向验证 - 验证形态的真实性"""

        result = {
            'verified_patterns': [],
            'verification_rate': 0.0,
            'verification_details': {}
        }

        verified_count = 0
        total_count = len(patterns_found)

        for pattern_info in patterns_found:
            stock_code = pattern_info['stock_code']
            pattern_date = pattern_info['date']
            pattern_name = pattern_info['pattern_name']

            try:
                print(f"    🔍 验证: {stock_code} - {pattern_name} @ {pattern_date}")

                # 使用DBManager获取验证数据
                from datetime import datetime, timedelta
                target_date = datetime.strptime(pattern_date, '%Y-%m-%d')
                start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')

                verify_query = f"""
                    SELECT date, open, high, low, close, volume
                    FROM stock_info
                    WHERE code = '{stock_code}'
                    AND level = '日线'
                    AND date BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY date ASC
                """

                verify_data = self.db_manager.query_manager(verify_query)

                verify_df = None
                if verify_data:
                    verify_df = pd.DataFrame(verify_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    verify_df['date'] = pd.to_datetime(verify_df['date'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        verify_df[col] = pd.to_numeric(verify_df[col], errors='coerce')

                if verify_df is not None and len(verify_df) >= 20:  # 至少20天数据用于验证
                    verify_df = verify_df.sort_values('date').reset_index(drop=True)

                    # 重新计算MACD
                    verify_macd = self.macd.calculate(verify_df)
                    verify_patterns = self.macd.get_patterns(verify_df)

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
                                pattern_info['verified'] = True
                                pattern_info['verification_details'] = {
                                    'macd_value': float(verify_macd.iloc[idx]['MACD']) if 'MACD' in verify_macd.columns and idx < len(verify_macd) else None,
                                    'signal_value': float(verify_macd.iloc[idx]['Signal']) if 'Signal' in verify_macd.columns and idx < len(verify_macd) else None,
                                    'histogram_value': float(verify_macd.iloc[idx]['Histogram']) if 'Histogram' in verify_macd.columns and idx < len(verify_macd) else None
                                }
                                result['verified_patterns'].append(pattern_info)
                                print(f"      ✅ 验证通过")
                            else:
                                pattern_info['verified'] = False
                                print(f"      ❌ 验证失败: 形态不存在")
                        else:
                            pattern_info['verified'] = False
                            print(f"      ❌ 验证失败: 无法找到目标日期或形态")
                    else:
                        pattern_info['verified'] = False
                        print(f"      ❌ 验证失败: 目标日期不存在")
                else:
                    pattern_info['verified'] = False
                    print(f"      ❌ 验证失败: 验证数据不足")

                result['verification_details'][f"{stock_code}_{pattern_name}_{pattern_date}"] = pattern_info

            except Exception as e:
                print(f"      ❌ 验证异常: {e}")
                pattern_info['verified'] = False
                pattern_info['verification_error'] = str(e)

        result['verification_rate'] = verified_count / total_count if total_count > 0 else 0

        return result
    
    def _generate_detailed_report(self, pattern_results: Dict, verification_results: Dict) -> List[Dict]:
        """生成详细验证报告"""
        
        detailed_findings = []
        
        # 汇总验证通过的形态
        for pattern_info in verification_results['verified_patterns']:
            finding = {
                'stock_code': pattern_info['stock_code'],
                'stock_name': f"股票{pattern_info['stock_code']}",
                'pattern_name': pattern_info['pattern_name'],
                'pattern_description': self._get_pattern_description(pattern_info['pattern_name']),
                'detection_date': pattern_info['date'],
                'time_level': pattern_info['level'],
                'close_price': pattern_info['close_price'],
                'macd_indicators': pattern_info.get('verification_details', {}),
                'verification_status': 'VERIFIED',
                'confidence_level': 'HIGH'
            }
            detailed_findings.append(finding)
        
        return detailed_findings
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """获取形态描述"""
        descriptions = {
            'GOLDEN_CROSS': 'MACD金叉：MACD线上穿信号线，买入信号',
            'DEATH_CROSS': 'MACD死叉：MACD线下穿信号线，卖出信号',
            'MACD_ABOVE_ZERO_GOLDEN': 'MACD零轴上方金叉：强势买入信号',
            'BEARISH_DIVERGENCE': 'MACD顶背离：价格创新高但MACD不创新高，看跌信号'
        }
        return descriptions.get(pattern_name, f'{pattern_name}形态')
    
    def _comprehensive_assessment(self, validation_result: Dict) -> bool:
        """综合评估"""
        
        try:
            # 检查是否有足够的验证通过的形态
            verified_patterns = validation_result.get('bidirectional_verification', {}).get('verified_patterns', [])
            verification_rate = validation_result.get('bidirectional_verification', {}).get('verification_rate', 0)
            
            # 通过标准
            min_verified_patterns = self.validation_standards['min_patterns_found']
            min_verification_rate = 0.7  # 70%验证率
            
            if len(verified_patterns) >= min_verified_patterns and verification_rate >= min_verification_rate:
                return True
            else:
                if len(verified_patterns) < min_verified_patterns:
                    validation_result['issues_found'].append(f"验证通过的形态不足: {len(verified_patterns)} < {min_verified_patterns}")
                if verification_rate < min_verification_rate:
                    validation_result['issues_found'].append(f"验证率过低: {verification_rate:.1%} < {min_verification_rate:.1%}")
                return False
        
        except Exception as e:
            validation_result['issues_found'].append(f"综合评估异常: {str(e)}")
            return False

def main():
    """主函数"""
    validator = TrueMACDValidator()
    
    # 运行真正的双向验证
    results = validator.run_true_validation()
    
    print("\n" + "="*80)
    print("🏆 MACD真正的双向验证结果汇总")
    print("="*80)
    
    print(f"🎯 验证通过: {results['validation_passed']}")
    
    if results['stock_analysis_results']:
        stock_res = results['stock_analysis_results']
        print(f"📊 股票分析: {stock_res['total_stocks']}支股票，{stock_res['stocks_with_sufficient_data']}支有足够数据")
    
    if results['pattern_detection_results']:
        pattern_res = results['pattern_detection_results']
        print(f"🎯 形态识别: 分析{pattern_res['stocks_analyzed']}支股票，发现{len(pattern_res['patterns_found'])}个形态")
        
        if pattern_res['patterns_by_type']:
            print(f"📋 形态分布:")
            for pattern_type, count in pattern_res['patterns_by_type'].items():
                print(f"    {pattern_type}: {count}个")
    
    if results['bidirectional_verification']:
        verify_res = results['bidirectional_verification']
        print(f"🛡️ 双向验证: {len(verify_res['verified_patterns'])}个形态验证通过，验证率: {verify_res['verification_rate']:.1%}")
    
    if results['detailed_findings']:
        print(f"\n📋 详细验证结果:")
        for finding in results['detailed_findings'][:5]:  # 显示前5个
            print(f"  ✅ {finding['stock_code']} - {finding['pattern_description']}")
            print(f"     时间: {finding['detection_date']} ({finding['time_level']})")
            print(f"     价格: {finding['close_price']:.2f}元")
            if finding['macd_indicators']:
                macd_info = finding['macd_indicators']
                print(f"     MACD: {macd_info.get('macd_value', 'N/A'):.4f}, Signal: {macd_info.get('signal_value', 'N/A'):.4f}")
    
    if results['validation_passed']:
        print(f"\n🎉 MACD指标通过真正的双向验证！")
        print(f"✅ 成功识别并验证了具体的技术形态")
        print(f"✅ 明确说明了哪支股票在什么时间什么级别符合哪个MACD形态")
    else:
        print(f"\n🔧 MACD指标双向验证需要改进")
        if results['issues_found']:
            print(f"❌ 问题: {', '.join(results['issues_found'])}")

if __name__ == "__main__":
    main()
