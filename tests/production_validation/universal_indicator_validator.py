#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用技术指标双向验证器 - 阶段3验证

通用双向验证逻辑：
1. 正向验证：使用任意技术指标计算真实股票数据，识别具体形态
2. 反向验证：验证识别的形态确实存在于指定时间/级别的数据中
3. 详细报告：明确说明哪支股票在什么时间什么级别符合哪个技术形态
4. 架构合规：严格使用数据层接口，不直接写SQL

使用示例：
    macd_validator = UniversalIndicatorValidator("MACD", MacdMacd())
    rsi_validator = UniversalIndicatorValidator("RSI", RsiRsi())
    kdj_validator = UniversalIndicatorValidator("KDJ", KdjKdj())
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

from indicators.pattern.unified_pattern_registry import get_unified_pattern_registry
from db.services.stock_data_service import get_stock_data_service, StockDataService
from utils.logger import get_logger

logger = get_logger(__name__)

class UniversalIndicatorValidator:
    """通用技术指标双向验证器"""
    
    def __init__(self, indicator_name: str, indicator_instance: Any):
        """
        初始化通用验证器
        
        Args:
            indicator_name: 指标名称（如"MACD", "RSI", "KDJ"等）
            indicator_instance: 指标实例（如MacdMacd(), RsiRsi()等）
        """
        self.indicator_name = indicator_name.upper()
        self.indicator_instance = indicator_instance
        self.pattern_registry = get_unified_pattern_registry()
        
        # 使用符合架构规范的数据层服务
        try:
            self.stock_data_service = get_stock_data_service()
            logger.info("✅ 股票数据服务初始化成功")
        except Exception as e:
            logger.error(f"❌ 股票数据服务初始化失败: {e}")
            raise RuntimeError(f"无法初始化股票数据服务: {e}")
        
        # 通用验证标准
        self.validation_standards = {
            'min_stocks_to_test': 10,           # 至少测试10支股票
            'min_patterns_found': 3,            # 至少找到3个形态
            'min_data_days': 60,                # 至少60天数据
            'required_levels': ['日线'],         # 必须的时间级别
            'min_verification_rate': 0.7        # 最低验证率70%
        }
        
        # 动态获取指标支持的形态
        self.supported_patterns = self._get_indicator_patterns()
        
        logger.info(f"🔥 {self.indicator_name}通用双向验证器初始化完成")
    
    def _get_indicator_patterns(self) -> List[str]:
        """动态获取指标支持的形态"""
        try:
            # 尝试从指标实例获取支持的形态
            if hasattr(self.indicator_instance, 'get_supported_patterns'):
                return self.indicator_instance.get_supported_patterns()
            
            # 通用形态类型（适用于大多数指标）
            common_patterns = [
                'GOLDEN_CROSS',         # 金叉
                'DEATH_CROSS',          # 死叉
                'BULLISH_SIGNAL',       # 看涨信号
                'BEARISH_SIGNAL',       # 看跌信号
                'OVERBOUGHT',           # 超买
                'OVERSOLD',             # 超卖
                'BREAKOUT_UP',          # 向上突破
                'BREAKOUT_DOWN',        # 向下突破
                'DIVERGENCE_BULLISH',   # 底背离
                'DIVERGENCE_BEARISH'    # 顶背离
            ]
            
            # 根据指标名称返回特定形态
            indicator_specific_patterns = {
                'MACD': ['GOLDEN_CROSS', 'DEATH_CROSS', 'MACD_ABOVE_ZERO_GOLDEN', 'BEARISH_DIVERGENCE'],
                'RSI': ['OVERBOUGHT', 'OVERSOLD', 'RSI_GOLDEN_CROSS', 'RSI_DEATH_CROSS'],
                'KDJ': ['KDJ_GOLDEN_CROSS', 'KDJ_DEATH_CROSS', 'KDJ_OVERBOUGHT', 'KDJ_OVERSOLD'],
                'BOLL': ['BOLL_UPPER_BREAKOUT', 'BOLL_LOWER_BREAKOUT', 'BOLL_SQUEEZE', 'BOLL_EXPANSION'],
                'MA': ['MA_GOLDEN_CROSS', 'MA_DEATH_CROSS', 'MA_SUPPORT', 'MA_RESISTANCE'],
                'EMA': ['EMA_GOLDEN_CROSS', 'EMA_DEATH_CROSS', 'EMA_TREND_UP', 'EMA_TREND_DOWN']
            }
            
            return indicator_specific_patterns.get(self.indicator_name, common_patterns)
            
        except Exception as e:
            logger.warning(f"获取{self.indicator_name}支持形态失败: {e}")
            return ['GOLDEN_CROSS', 'DEATH_CROSS', 'BULLISH_SIGNAL', 'BEARISH_SIGNAL']
    
    def run_universal_validation(self) -> Dict[str, Any]:
        """运行通用双向验证"""
        
        print(f"🔥 开始{self.indicator_name}通用双向验证")
        print("=" * 80)
        print("📋 验证目标: 通用技术形态识别和双向验证")
        print("🎯 要求: 明确说明哪支股票在什么时间什么级别符合哪个技术形态")
        print("🏗️ 架构: 严格使用数据层接口，不直接写SQL")
        print("=" * 80)
        
        validation_result = {
            'stage': 'Stage3_Universal_Bidirectional',
            'indicator_name': self.indicator_name,
            'indicator_type': type(self.indicator_instance).__name__,
            'validation_timestamp': datetime.now().isoformat(),
            'supported_patterns': self.supported_patterns,
            'stock_analysis_results': {},
            'pattern_detection_results': {},
            'bidirectional_verification': {},
            'detailed_findings': [],
            'validation_passed': False,
            'issues_found': []
        }
        
        try:
            # 步骤1: 获取测试股票数据（使用数据层接口）
            print("\n📊 步骤1: 通过数据层接口获取测试股票数据")
            stock_data = self._get_test_stock_data_via_interface()
            validation_result['stock_analysis_results'] = {
                'total_stocks': len(stock_data),
                'stocks_with_sufficient_data': sum(1 for data in stock_data.values() if len(data) >= self.validation_standards['min_data_days'])
            }
            
            if len(stock_data) < self.validation_standards['min_stocks_to_test']:
                validation_result['issues_found'].append(f"测试股票不足: {len(stock_data)} < {self.validation_standards['min_stocks_to_test']}")
                return validation_result
            
            print(f"✅ 通过数据层接口获取{len(stock_data)}支股票数据")
            
            # 步骤2: 正向验证 - 通用形态识别
            print(f"\n🎯 步骤2: 正向验证 - {self.indicator_name}形态识别")
            pattern_results = self._forward_validation_universal(stock_data)
            validation_result['pattern_detection_results'] = pattern_results
            
            if not pattern_results['patterns_found']:
                validation_result['issues_found'].append(f"未识别到任何{self.indicator_name}形态")
                return validation_result
            
            print(f"✅ 识别到{len(pattern_results['patterns_found'])}个{self.indicator_name}形态")
            
            # 步骤3: 反向验证 - 验证形态的真实性（使用数据层接口）
            print(f"\n🛡️ 步骤3: 反向验证 - 验证{self.indicator_name}形态的真实性")
            verification_results = self._backward_validation_universal(pattern_results['patterns_found'])
            validation_result['bidirectional_verification'] = verification_results
            
            # 步骤4: 生成详细报告
            print(f"\n📋 步骤4: 生成{self.indicator_name}详细验证报告")
            detailed_findings = self._generate_universal_detailed_report(pattern_results, verification_results)
            validation_result['detailed_findings'] = detailed_findings
            
            # 步骤5: 综合评估
            print(f"\n🏆 步骤5: {self.indicator_name}综合评估")
            validation_passed = self._comprehensive_assessment_universal(validation_result)
            validation_result['validation_passed'] = validation_passed
            
            if validation_passed:
                print(f"🎉 {self.indicator_name}指标通过通用双向验证！")
            else:
                print(f"❌ {self.indicator_name}指标双向验证失败")
        
        except Exception as e:
            logger.error(f"❌ {self.indicator_name}验证过程异常: {e}")
            validation_result['issues_found'].append(f"验证过程异常: {str(e)}")
        
        return validation_result
    
    def _get_test_stock_data_via_interface(self) -> Dict[str, pd.DataFrame]:
        """通过架构合规的数据层服务获取测试股票数据"""

        stock_data = {}

        try:
            # 使用股票数据服务获取股票列表
            stock_codes = self.stock_data_service.get_stock_list(limit=20)

            if not stock_codes:
                logger.warning("股票数据服务未返回股票列表")
                return stock_data

            print(f"    📋 股票数据服务返回{len(stock_codes)}支股票代码")

            for stock_code in stock_codes:
                try:
                    # 使用股票数据服务获取股票历史数据
                    df = self.stock_data_service.get_stock_data(stock_code, days=120)

                    if df is not None and len(df) >= self.validation_standards['min_data_days']:
                        stock_data[stock_code] = df
                        print(f"    ✅ {stock_code}: {len(df)}天数据")
                    else:
                        print(f"    ⚠️ {stock_code}: 数据不足")

                except Exception as e:
                    print(f"    ❌ {stock_code}: 数据获取失败 - {e}")
                    continue

        except Exception as e:
            logger.error(f"通过股票数据服务获取股票数据失败: {e}")

        return stock_data

    def _forward_validation_universal(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """通用正向验证 - 技术形态识别"""

        result = {
            'patterns_found': [],
            'stocks_analyzed': 0,
            'patterns_by_type': {},
            'analysis_details': {}
        }

        for stock_code, df in stock_data.items():
            result['stocks_analyzed'] += 1

            try:
                print(f"    🔍 分析股票: {stock_code} ({self.indicator_name})")

                # 使用指标实例计算
                indicator_result = self.indicator_instance.calculate(df)

                if indicator_result is None or indicator_result.empty:
                    print(f"      ❌ {self.indicator_name}计算失败")
                    continue

                # 获取指标形态
                patterns = self.indicator_instance.get_patterns(df)

                if patterns is None or patterns.empty:
                    print(f"      ⚠️ 未识别到{self.indicator_name}形态")
                    continue

                # 分析每个支持的形态
                for pattern_name in self.supported_patterns:
                    if pattern_name in patterns.columns:
                        pattern_signals = patterns[patterns[pattern_name] == True]

                        if not pattern_signals.empty:
                            for idx in pattern_signals.index:
                                if idx < len(df):  # 确保索引有效
                                    pattern_info = {
                                        'stock_code': stock_code,
                                        'pattern_name': pattern_name,
                                        'date': df.iloc[idx]['date'].strftime('%Y-%m-%d'),
                                        'level': '日线',
                                        'close_price': float(df.iloc[idx]['close']),
                                        'indicator_values': self._extract_indicator_values(indicator_result, idx)
                                    }

                                    result['patterns_found'].append(pattern_info)

                                    # 按类型统计
                                    if pattern_name not in result['patterns_by_type']:
                                        result['patterns_by_type'][pattern_name] = 0
                                    result['patterns_by_type'][pattern_name] += 1

                                    print(f"      ✅ 发现{pattern_name}: {pattern_info['date']}, 价格: {pattern_info['close_price']:.2f}")

                result['analysis_details'][stock_code] = {
                    'data_points': len(df),
                    'indicator_calculated': indicator_result is not None,
                    'patterns_detected': len([p for p in result['patterns_found'] if p['stock_code'] == stock_code])
                }

            except Exception as e:
                print(f"      ❌ {self.indicator_name}分析失败: {e}")
                result['analysis_details'][stock_code] = {
                    'error': str(e)
                }
                continue

        return result

    def _extract_indicator_values(self, indicator_result: pd.DataFrame, idx: int) -> Dict[str, float]:
        """提取指标值（通用方法）"""
        values = {}

        try:
            if idx < len(indicator_result):
                # 提取所有数值列的值
                for col in indicator_result.columns:
                    if indicator_result[col].dtype in ['float64', 'int64']:
                        val = indicator_result.iloc[idx][col]
                        if not pd.isna(val):
                            values[col] = float(val)
        except Exception as e:
            logger.warning(f"提取指标值失败: {e}")

        return values

    def _backward_validation_universal(self, patterns_found: List[Dict]) -> Dict[str, Any]:
        """通用反向验证 - 验证形态的真实性（使用数据层接口）"""

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

                # 使用股票数据服务获取验证数据
                target_date = datetime.strptime(pattern_date, '%Y-%m-%d')
                start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')

                verify_df = self.stock_data_service.get_stock_data_by_date_range(
                    stock_code, start_date, end_date
                )

                if verify_df is not None and len(verify_df) >= 20:  # 至少20天数据用于验证
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
                                pattern_info['verified'] = True
                                pattern_info['verification_details'] = self._extract_indicator_values(verify_indicator, idx)
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

    def _generate_universal_detailed_report(self, pattern_results: Dict, verification_results: Dict) -> List[Dict]:
        """生成通用详细验证报告"""

        detailed_findings = []

        # 汇总验证通过的形态
        for pattern_info in verification_results['verified_patterns']:
            finding = {
                'stock_code': pattern_info['stock_code'],
                'stock_name': f"股票{pattern_info['stock_code']}",
                'indicator_name': self.indicator_name,
                'pattern_name': pattern_info['pattern_name'],
                'pattern_description': self._get_universal_pattern_description(pattern_info['pattern_name']),
                'detection_date': pattern_info['date'],
                'time_level': pattern_info['level'],
                'close_price': pattern_info['close_price'],
                'indicator_values': pattern_info.get('verification_details', {}),
                'verification_status': 'VERIFIED',
                'confidence_level': 'HIGH'
            }
            detailed_findings.append(finding)

        return detailed_findings

    def _get_universal_pattern_description(self, pattern_name: str) -> str:
        """获取通用形态描述"""

        # 通用形态描述映射
        universal_descriptions = {
            'GOLDEN_CROSS': f'{self.indicator_name}金叉：买入信号',
            'DEATH_CROSS': f'{self.indicator_name}死叉：卖出信号',
            'BULLISH_SIGNAL': f'{self.indicator_name}看涨信号：多头信号',
            'BEARISH_SIGNAL': f'{self.indicator_name}看跌信号：空头信号',
            'OVERBOUGHT': f'{self.indicator_name}超买：价格可能回调',
            'OVERSOLD': f'{self.indicator_name}超卖：价格可能反弹',
            'BREAKOUT_UP': f'{self.indicator_name}向上突破：强势信号',
            'BREAKOUT_DOWN': f'{self.indicator_name}向下突破：弱势信号',
            'DIVERGENCE_BULLISH': f'{self.indicator_name}底背离：看涨背离信号',
            'DIVERGENCE_BEARISH': f'{self.indicator_name}顶背离：看跌背离信号'
        }

        # 指标特定描述
        specific_descriptions = {
            'MACD': {
                'MACD_ABOVE_ZERO_GOLDEN': 'MACD零轴上方金叉：强势买入信号',
                'BEARISH_DIVERGENCE': 'MACD顶背离：价格创新高但MACD不创新高'
            },
            'RSI': {
                'RSI_GOLDEN_CROSS': 'RSI金叉：相对强弱指标买入信号',
                'RSI_DEATH_CROSS': 'RSI死叉：相对强弱指标卖出信号'
            },
            'KDJ': {
                'KDJ_GOLDEN_CROSS': 'KDJ金叉：随机指标买入信号',
                'KDJ_DEATH_CROSS': 'KDJ死叉：随机指标卖出信号',
                'KDJ_OVERBOUGHT': 'KDJ超买：K值和D值过高',
                'KDJ_OVERSOLD': 'KDJ超卖：K值和D值过低'
            },
            'BOLL': {
                'BOLL_UPPER_BREAKOUT': 'BOLL上轨突破：价格突破布林带上轨',
                'BOLL_LOWER_BREAKOUT': 'BOLL下轨突破：价格跌破布林带下轨',
                'BOLL_SQUEEZE': 'BOLL收缩：布林带收缩，波动率降低',
                'BOLL_EXPANSION': 'BOLL扩张：布林带扩张，波动率增加'
            }
        }

        # 优先使用指标特定描述
        if self.indicator_name in specific_descriptions:
            if pattern_name in specific_descriptions[self.indicator_name]:
                return specific_descriptions[self.indicator_name][pattern_name]

        # 使用通用描述
        return universal_descriptions.get(pattern_name, f'{self.indicator_name}_{pattern_name}形态')

    def _comprehensive_assessment_universal(self, validation_result: Dict) -> bool:
        """通用综合评估"""

        try:
            # 检查是否有足够的验证通过的形态
            verified_patterns = validation_result.get('bidirectional_verification', {}).get('verified_patterns', [])
            verification_rate = validation_result.get('bidirectional_verification', {}).get('verification_rate', 0)

            # 通用通过标准
            min_verified_patterns = self.validation_standards['min_patterns_found']
            min_verification_rate = self.validation_standards['min_verification_rate']

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

def create_indicator_validator(indicator_name: str, indicator_class: type) -> UniversalIndicatorValidator:
    """
    工厂方法：创建指标验证器

    Args:
        indicator_name: 指标名称
        indicator_class: 指标类

    Returns:
        通用指标验证器实例
    """
    try:
        indicator_instance = indicator_class()
        return UniversalIndicatorValidator(indicator_name, indicator_instance)
    except Exception as e:
        logger.error(f"创建{indicator_name}验证器失败: {e}")
        raise RuntimeError(f"无法创建{indicator_name}验证器: {e}")

def main():
    """主函数 - 演示通用验证器的使用"""

    # 导入指标类
    from indicators.macd import MacdMacd

    try:
        # 创建MACD验证器
        macd_validator = UniversalIndicatorValidator("MACD", MacdMacd())

        # 运行通用双向验证
        results = macd_validator.run_universal_validation()

        print("\n" + "="*80)
        print(f"🏆 {macd_validator.indicator_name}通用双向验证结果汇总")
        print("="*80)

        print(f"🎯 验证通过: {results['validation_passed']}")
        print(f"📊 指标类型: {results['indicator_type']}")
        print(f"🔧 支持形态: {', '.join(results['supported_patterns'])}")

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
                if finding['indicator_values']:
                    values_str = ', '.join([f"{k}: {v:.4f}" for k, v in finding['indicator_values'].items()])
                    print(f"     指标值: {values_str}")

        if results['validation_passed']:
            print(f"\n🎉 {macd_validator.indicator_name}指标通过通用双向验证！")
            print(f"✅ 成功识别并验证了具体的技术形态")
            print(f"✅ 明确说明了哪支股票在什么时间什么级别符合哪个形态")
            print(f"✅ 严格遵循架构要求，使用数据层接口")
        else:
            print(f"\n🔧 {macd_validator.indicator_name}指标双向验证需要改进")
            if results['issues_found']:
                print(f"❌ 问题: {', '.join(results['issues_found'])}")

        # 演示如何为其他指标创建验证器
        print(f"\n💡 使用示例:")
        print(f"# 为其他指标创建验证器")
        print(f"# rsi_validator = UniversalIndicatorValidator('RSI', RsiRsi())")
        print(f"# kdj_validator = UniversalIndicatorValidator('KDJ', KdjKdj())")
        print(f"# boll_validator = UniversalIndicatorValidator('BOLL', BollBoll())")

    except Exception as e:
        logger.error(f"验证器演示失败: {e}")
        print(f"❌ 验证器演示失败: {e}")

if __name__ == "__main__":
    main()
