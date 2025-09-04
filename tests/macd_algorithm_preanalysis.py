#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD算法差异预分析

基于RSI项目经验，提前分析MACD算法实现，识别潜在的算法差异问题
确保使用标准EMA计算方法，避免重复RSI项目中发现的算法不一致问题
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from clickhouse_driver import Client
    from indicators.macd import MacdMacd
    from utils.technical_utils import calculate_macd_Utils, calculate_ema_Utils
except ImportError as e:
    print(f"导入错误: {e}")

class MACDAlgorithmPreAnalyzer:
    """MACD算法差异预分析器"""
    
    def __init__(self):
        """初始化预分析器"""
        self.analyzer_name = "MACD算法差异预分析器"
        
        # 连接数据库
        self.client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        self.macd_indicator = MacdMacd()
        
        print(f"✅ {self.analyzer_name}初始化完成")
        print(f"🎯 基于RSI项目经验，预防算法差异问题")
    
    def get_test_data(self, stock_code: str = '000001', days: int = 100) -> Optional[pd.DataFrame]:
        """获取测试数据"""
        
        try:
            # 查询可用股票
            available_query = """
            SELECT code, COUNT(*) as data_count
            FROM stock_info
            WHERE level = '日线'
            AND date >= '2025-01-01'
            GROUP BY code
            HAVING data_count >= 100
            ORDER BY data_count DESC
            LIMIT 5
            """
            
            available_result = self.client.execute(available_query)
            
            if available_result:
                available_stocks = [row[0] for row in available_result]
                print(f"📊 可用股票: {available_stocks}")
                
                # 使用第一个可用股票
                if available_stocks:
                    stock_code = available_stocks[0]
                    print(f"📈 使用股票: {stock_code}")
            
            # 查询股票数据
            query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '2025-01-01'
            AND date <= '2025-12-31'
            ORDER BY date ASC
            LIMIT {days}
            """
            
            result = self.client.execute(query)
            
            if result:
                df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_sorted = df.sort_values('date').reset_index(drop=True)
                print(f"📊 获取到{len(df_sorted)}条数据")
                return df_sorted
            else:
                print(f"❌ 股票{stock_code}无数据")
                return None
                
        except Exception as e:
            print(f"❌ 获取测试数据失败: {e}")
            return None
    
    def analyze_ema_calculation_methods(self) -> Dict[str, Any]:
        """分析不同EMA计算方法"""
        
        print(f"\n🔍 分析EMA计算方法差异")
        print("=" * 80)
        
        analysis_result = {
            'analysis_type': 'EMA_CALCULATION_METHODS',
            'timestamp': datetime.now().isoformat(),
            'test_data_info': {},
            'ema_methods': {},
            'comparison_results': {},
            'potential_issues': []
        }
        
        try:
            # 获取测试数据
            test_data = self.get_test_data('000001', 50)
            
            if test_data is None or len(test_data) < 30:
                print("❌ 无法获取足够的测试数据")
                return analysis_result
            
            analysis_result['test_data_info'] = {
                'data_points': len(test_data),
                'date_range': {
                    'start': test_data['date'].min().isoformat(),
                    'end': test_data['date'].max().isoformat()
                },
                'price_range': {
                    'min': float(test_data['close'].min()),
                    'max': float(test_data['close'].max())
                }
            }
            
            print(f"📊 测试数据: {len(test_data)}个数据点")
            print(f"📅 日期范围: {test_data['date'].min().date()} 到 {test_data['date'].max().date()}")
            print(f"💰 价格范围: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
            
            # 测试不同EMA计算方法
            ema_methods = ['standard', 'sma_init', 'pandas']
            ema_periods = [12, 26]
            
            for period in ema_periods:
                print(f"\n🔧 测试EMA{period}的不同计算方法")
                
                method_results = {}
                
                for method in ema_methods:
                    try:
                        ema_result = calculate_ema_Utils(test_data['close'], period, method)
                        
                        if ema_result is not None and not ema_result.empty:
                            ema_clean = ema_result.dropna()
                            
                            if len(ema_clean) > 0:
                                method_results[method] = {
                                    'data_points': len(ema_clean),
                                    'values_sample': ema_clean.tail(5).tolist(),
                                    'latest_value': float(ema_clean.iloc[-1]),
                                    'range': {'min': float(ema_clean.min()), 'max': float(ema_clean.max())}
                                }
                                print(f"  ✅ {method}: {len(ema_clean)}个值, 最新值: {ema_clean.iloc[-1]:.3f}")
                            else:
                                method_results[method] = {'status': 'NO_VALID_VALUES'}
                                print(f"  ❌ {method}: 无有效值")
                        else:
                            method_results[method] = {'status': 'CALCULATION_FAILED'}
                            print(f"  ❌ {method}: 计算失败")
                    
                    except Exception as e:
                        method_results[method] = {'status': 'ERROR', 'error': str(e)}
                        print(f"  ❌ {method}: 异常 - {e}")
                
                analysis_result['ema_methods'][f'ema_{period}'] = method_results
                
                # 比较不同方法的差异
                if len(method_results) >= 2:
                    self._compare_ema_methods(method_results, period, analysis_result)
            
        except Exception as e:
            analysis_result['error'] = str(e)
            print(f"❌ EMA方法分析异常: {e}")
        
        return analysis_result
    
    def _compare_ema_methods(self, method_results: Dict, period: int, analysis_result: Dict):
        """比较不同EMA方法的结果"""
        
        print(f"  📊 EMA{period}方法对比:")
        
        # 收集有效的方法结果
        valid_methods = {}
        for method, result in method_results.items():
            if 'latest_value' in result:
                valid_methods[method] = result['latest_value']
        
        if len(valid_methods) >= 2:
            values = list(valid_methods.values())
            max_diff = max(values) - min(values)
            avg_value = sum(values) / len(values)
            relative_diff = max_diff / avg_value if avg_value != 0 else 0
            
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均值: {avg_value:.3f}")
            print(f"    相对差异: {relative_diff:.1%}")
            
            # 记录潜在问题
            if relative_diff > 0.01:  # 1%以上差异
                issue = {
                    'type': 'EMA_METHOD_DIFFERENCE',
                    'period': period,
                    'max_difference': max_diff,
                    'relative_difference': relative_diff,
                    'severity': 'HIGH' if relative_diff > 0.05 else 'MEDIUM'
                }
                analysis_result['potential_issues'].append(issue)
                print(f"    ⚠️ 发现潜在问题: EMA{period}方法差异{relative_diff:.1%}")
    
    def analyze_macd_calculation_methods(self) -> Dict[str, Any]:
        """分析不同MACD计算方法"""
        
        print(f"\n🔍 分析MACD计算方法差异")
        print("=" * 80)
        
        analysis_result = {
            'analysis_type': 'MACD_CALCULATION_METHODS',
            'timestamp': datetime.now().isoformat(),
            'test_data_info': {},
            'macd_methods': {},
            'comparison_results': {},
            'potential_issues': []
        }
        
        try:
            # 获取测试数据
            test_data = self.get_test_data('000001', 50)
            
            if test_data is None or len(test_data) < 40:
                print("❌ 无法获取足够的测试数据")
                return analysis_result
            
            print(f"📊 测试MACD计算方法")
            
            # 方法1: 系统MACD指标类
            print(f"\n🔧 方法1: 系统MACD指标类")
            try:
                system_macd_result = self.macd_indicator._calculate_macd(test_data)
                
                if system_macd_result is not None and not system_macd_result.empty:
                    macd_line = system_macd_result.get('macd_line', pd.Series())
                    macd_signal = system_macd_result.get('macd_signal', pd.Series())
                    macd_histogram = system_macd_result.get('macd_histogram', pd.Series())
                    
                    if not macd_line.empty:
                        macd_clean = macd_line.dropna()
                        signal_clean = macd_signal.dropna()
                        hist_clean = macd_histogram.dropna()
                        
                        analysis_result['macd_methods']['system_macd'] = {
                            'method': 'MacdMacd._calculate_macd()',
                            'macd_data_points': len(macd_clean),
                            'signal_data_points': len(signal_clean),
                            'histogram_data_points': len(hist_clean),
                            'latest_macd': float(macd_clean.iloc[-1]) if len(macd_clean) > 0 else None,
                            'latest_signal': float(signal_clean.iloc[-1]) if len(signal_clean) > 0 else None,
                            'latest_histogram': float(hist_clean.iloc[-1]) if len(hist_clean) > 0 else None
                        }
                        print(f"  ✅ 系统MACD计算成功")
                        print(f"  📊 MACD: {macd_clean.iloc[-1]:.6f}, 信号: {signal_clean.iloc[-1]:.6f}, 柱状图: {hist_clean.iloc[-1]:.6f}")
                    else:
                        analysis_result['macd_methods']['system_macd'] = {'status': 'NO_VALID_VALUES'}
                        print(f"  ❌ 系统MACD无有效值")
                else:
                    analysis_result['macd_methods']['system_macd'] = {'status': 'CALCULATION_FAILED'}
                    print(f"  ❌ 系统MACD计算失败")
            except Exception as e:
                analysis_result['macd_methods']['system_macd'] = {'status': 'ERROR', 'error': str(e)}
                print(f"  ❌ 系统MACD异常: {e}")
            
            # 方法2: 技术工具函数 - 不同EMA方法
            ema_methods = ['standard', 'sma_init', 'pandas']
            
            for ema_method in ema_methods:
                print(f"\n🔧 方法2-{ema_method}: 技术工具函数({ema_method})")
                
                try:
                    dif, dea, macd = calculate_macd_Utils(
                        test_data['close'], 
                        fast_period=12, 
                        slow_period=26, 
                        signal_period=9,
                        method=ema_method
                    )
                    
                    if dif is not None and not dif.empty:
                        dif_clean = dif.dropna()
                        dea_clean = dea.dropna()
                        macd_clean = macd.dropna()
                        
                        analysis_result['macd_methods'][f'utils_macd_{ema_method}'] = {
                            'method': f'calculate_macd_Utils({ema_method})',
                            'dif_data_points': len(dif_clean),
                            'dea_data_points': len(dea_clean),
                            'macd_data_points': len(macd_clean),
                            'latest_dif': float(dif_clean.iloc[-1]) if len(dif_clean) > 0 else None,
                            'latest_dea': float(dea_clean.iloc[-1]) if len(dea_clean) > 0 else None,
                            'latest_macd': float(macd_clean.iloc[-1]) if len(macd_clean) > 0 else None
                        }
                        print(f"  ✅ 工具函数({ema_method})计算成功")
                        print(f"  📊 DIF: {dif_clean.iloc[-1]:.6f}, DEA: {dea_clean.iloc[-1]:.6f}, MACD: {macd_clean.iloc[-1]:.6f}")
                    else:
                        analysis_result['macd_methods'][f'utils_macd_{ema_method}'] = {'status': 'NO_VALID_VALUES'}
                        print(f"  ❌ 工具函数({ema_method})无有效值")
                
                except Exception as e:
                    analysis_result['macd_methods'][f'utils_macd_{ema_method}'] = {'status': 'ERROR', 'error': str(e)}
                    print(f"  ❌ 工具函数({ema_method})异常: {e}")
            
            # 比较分析
            comparison_results = self._compare_macd_methods(analysis_result['macd_methods'])
            analysis_result['comparison_results'] = comparison_results
            
        except Exception as e:
            analysis_result['error'] = str(e)
            print(f"❌ MACD方法分析异常: {e}")
        
        return analysis_result
    
    def _compare_macd_methods(self, methods: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同MACD计算方法"""
        
        print(f"\n📊 MACD计算方法对比分析")
        print("=" * 60)
        
        comparison = {
            'method_count': 0,
            'successful_methods': [],
            'failed_methods': [],
            'value_comparisons': {},
            'accuracy_analysis': {}
        }
        
        # 收集成功的方法
        successful_methods = {}
        for method_name, method_info in methods.items():
            if isinstance(method_info, dict) and 'latest_macd' in method_info and method_info['latest_macd'] is not None:
                successful_methods[method_name] = method_info
                comparison['successful_methods'].append(method_name)
            else:
                comparison['failed_methods'].append(method_name)
        
        comparison['method_count'] = len(successful_methods)
        
        if len(successful_methods) >= 2:
            # 值对比 - 使用MACD线(DIF)进行对比
            print(f"📈 最新MACD值对比:")
            macd_values = {}
            signal_values = {}
            
            for method_name, method_info in successful_methods.items():
                # 优先使用DIF值，如果没有则使用MACD值
                if 'latest_dif' in method_info and method_info['latest_dif'] is not None:
                    macd_value = method_info['latest_dif']
                    signal_value = method_info.get('latest_dea', method_info.get('latest_signal'))
                else:
                    macd_value = method_info.get('latest_macd')
                    signal_value = method_info.get('latest_signal')
                
                if macd_value is not None:
                    macd_values[method_name] = macd_value
                    if signal_value is not None:
                        signal_values[method_name] = signal_value
                    
                    print(f"  {method_name}: MACD={macd_value:.6f}")
            
            comparison['value_comparisons'] = {
                'macd_values': macd_values,
                'signal_values': signal_values
            }
            
            # 计算差异
            if len(macd_values) >= 2:
                values = list(macd_values.values())
                max_diff = max(values) - min(values)
                avg_value = sum(values) / len(values)
                relative_diff = max_diff / abs(avg_value) if avg_value != 0 else 0
                
                comparison['accuracy_analysis'] = {
                    'max_difference': max_diff,
                    'average_value': avg_value,
                    'relative_difference': relative_diff,
                    'consistency_level': 'HIGH' if max_diff < 0.001 else 'MEDIUM' if max_diff < 0.01 else 'LOW'
                }
                
                print(f"\n📊 差异分析:")
                print(f"  最大差异: {max_diff:.6f}")
                print(f"  平均值: {avg_value:.6f}")
                print(f"  相对差异: {relative_diff:.1%}")
                print(f"  一致性水平: {comparison['accuracy_analysis']['consistency_level']}")
        
        return comparison
    
    def run_complete_preanalysis(self) -> Dict[str, Any]:
        """运行完整的预分析"""
        
        print(f"\n🎯 MACD算法差异预分析")
        print("基于RSI项目经验，预防算法差异问题")
        print("=" * 80)
        
        complete_analysis = {
            'analysis_type': 'COMPLETE_MACD_PREANALYSIS',
            'start_time': datetime.now().isoformat(),
            'ema_analysis': {},
            'macd_analysis': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        try:
            # 1. EMA计算方法分析
            ema_analysis = self.analyze_ema_calculation_methods()
            complete_analysis['ema_analysis'] = ema_analysis
            
            # 2. MACD计算方法分析
            macd_analysis = self.analyze_macd_calculation_methods()
            complete_analysis['macd_analysis'] = macd_analysis
            
            # 3. 风险评估
            risk_assessment = self._assess_algorithm_risks(ema_analysis, macd_analysis)
            complete_analysis['risk_assessment'] = risk_assessment
            
            # 4. 生成建议
            recommendations = self._generate_recommendations(risk_assessment)
            complete_analysis['recommendations'] = recommendations
            
            complete_analysis['end_time'] = datetime.now().isoformat()
            
            print(f"\n🏆 MACD算法预分析完成")
            print(f"风险等级: {risk_assessment.get('overall_risk_level', 'UNKNOWN')}")
            print(f"需要修复: {'是' if risk_assessment.get('needs_fix', False) else '否'}")
            
        except Exception as e:
            complete_analysis['error'] = str(e)
            print(f"❌ 完整预分析异常: {e}")
        
        return complete_analysis
    
    def _assess_algorithm_risks(self, ema_analysis: Dict, macd_analysis: Dict) -> Dict[str, Any]:
        """评估算法风险"""
        
        risk_assessment = {
            'overall_risk_level': 'LOW',
            'needs_fix': False,
            'identified_risks': [],
            'confidence_level': 'HIGH'
        }
        
        # 检查EMA方法差异风险
        ema_issues = ema_analysis.get('potential_issues', [])
        for issue in ema_issues:
            if issue.get('severity') == 'HIGH':
                risk_assessment['identified_risks'].append({
                    'type': 'EMA_ALGORITHM_DIFFERENCE',
                    'description': f"EMA{issue['period']}方法差异{issue['relative_difference']:.1%}",
                    'severity': 'HIGH'
                })
                risk_assessment['overall_risk_level'] = 'HIGH'
                risk_assessment['needs_fix'] = True
        
        # 检查MACD方法一致性
        macd_comparison = macd_analysis.get('comparison_results', {})
        accuracy_analysis = macd_comparison.get('accuracy_analysis', {})
        
        if accuracy_analysis:
            consistency_level = accuracy_analysis.get('consistency_level', 'UNKNOWN')
            relative_diff = accuracy_analysis.get('relative_difference', 0)
            
            if consistency_level == 'LOW' or relative_diff > 0.05:
                risk_assessment['identified_risks'].append({
                    'type': 'MACD_ALGORITHM_INCONSISTENCY',
                    'description': f"MACD方法一致性{consistency_level}，相对差异{relative_diff:.1%}",
                    'severity': 'HIGH'
                })
                risk_assessment['overall_risk_level'] = 'HIGH'
                risk_assessment['needs_fix'] = True
            elif consistency_level == 'MEDIUM' or relative_diff > 0.01:
                risk_assessment['identified_risks'].append({
                    'type': 'MACD_ALGORITHM_INCONSISTENCY',
                    'description': f"MACD方法一致性{consistency_level}，相对差异{relative_diff:.1%}",
                    'severity': 'MEDIUM'
                })
                if risk_assessment['overall_risk_level'] == 'LOW':
                    risk_assessment['overall_risk_level'] = 'MEDIUM'
        
        return risk_assessment
    
    def _generate_recommendations(self, risk_assessment: Dict) -> List[str]:
        """生成建议"""
        
        recommendations = []
        
        if risk_assessment.get('needs_fix', False):
            recommendations.extend([
                '统一MACD计算算法实现',
                '确保EMA计算方法一致性',
                '建立标准MACD基准数据集',
                '在验证前先修复算法差异'
            ])
        else:
            recommendations.extend([
                '当前算法实现基本一致',
                '可以继续进行5阶段验证',
                '建立监控机制确保一致性'
            ])
        
        # 基于RSI经验的通用建议
        recommendations.extend([
            '使用标准EMA计算方法',
            '建立完整的验证测试用例',
            '确保服务层数据获取正常'
        ])
        
        return recommendations

def main():
    """主函数"""
    print("🎯 MACD算法差异预分析")
    print("基于RSI项目经验，预防算法差异问题")
    
    # 创建预分析器
    analyzer = MACDAlgorithmPreAnalyzer()
    
    # 运行完整预分析
    results = analyzer.run_complete_preanalysis()
    
    # 显示结果摘要
    print(f"\n📊 MACD算法预分析结果摘要")
    print("=" * 80)
    
    if 'risk_assessment' in results:
        risk_assessment = results['risk_assessment']
        print(f"风险等级: {risk_assessment.get('overall_risk_level', 'UNKNOWN')}")
        print(f"需要修复: {'是' if risk_assessment.get('needs_fix', False) else '否'}")
        
        if risk_assessment.get('identified_risks'):
            print(f"\n⚠️ 识别的风险:")
            for risk in risk_assessment['identified_risks']:
                print(f"  • {risk['description']} ({risk['severity']})")
    
    if 'recommendations' in results:
        recommendations = results['recommendations']
        print(f"\n💡 建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

if __name__ == "__main__":
    main()
