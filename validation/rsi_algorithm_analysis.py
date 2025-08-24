#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI算法差异深度分析

深入分析系统RSI与基准RSI计算方法的差异，找出准确率仅61.6%的根本原因
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
    from indicators.rsi import RsiRsi
    from utils.technical_utils import calculate_rsi_Utils
except ImportError as e:
    print(f"导入错误: {e}")

class RSIAlgorithmAnalyzer:
    """RSI算法差异分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.analyzer_name = "RSI算法差异分析器"
        
        # 连接数据库
        self.client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        self.rsi_indicator = RsiRsi()
        
        print(f"✅ {self.analyzer_name}初始化完成")
    
    def get_test_data(self, stock_code: str = '000001', days: int = 100) -> pd.DataFrame:
        """获取测试数据"""

        try:
            # 先查询可用的股票
            available_query = """
            SELECT code, COUNT(*) as data_count
            FROM stock_info
            WHERE level = '日线'
            AND date >= '2024-01-01'
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

            # 查询股票数据，使用更大的日期范围
            end_date = '2025-12-31'
            start_date = '2024-01-01'

            query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_info
            WHERE code = '{stock_code}'
            AND level = '日线'
            AND date >= '{start_date}'
            AND date <= '{end_date}'
            ORDER BY date ASC
            """

            result = self.client.execute(query)

            if result:
                df = pd.DataFrame(result, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])

                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 取最后的days天数据
                df_sorted = df.sort_values('date').tail(days).reset_index(drop=True)
                print(f"📊 获取到{len(df_sorted)}条数据")
                return df_sorted
            else:
                print(f"❌ 股票{stock_code}无数据")
                return None

        except Exception as e:
            print(f"❌ 获取测试数据失败: {e}")
            return None
    
    def analyze_rsi_calculation_methods(self) -> Dict[str, Any]:
        """分析不同RSI计算方法"""
        
        print(f"\n🔍 分析RSI计算方法差异")
        print("=" * 80)
        
        analysis_result = {
            'analysis_type': 'RSI_CALCULATION_METHODS',
            'timestamp': datetime.now().isoformat(),
            'test_data_info': {},
            'calculation_methods': {},
            'comparison_results': {},
            'root_cause_analysis': {}
        }
        
        try:
            # 获取测试数据
            test_data = self.get_test_data('000001', 50)
            
            if test_data is None or len(test_data) < 30:
                print("❌ 无法获取足够的测试数据")
                return analysis_result
            
            analysis_result['test_data_info'] = {
                'stock_code': '000001',
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
            
            # 方法1: 系统RSI指标类
            print(f"\n🔧 方法1: 系统RSI指标类")
            system_rsi_result = self.rsi_indicator._calculate_rsi(test_data)
            
            if 'rsi_14' in system_rsi_result.columns:
                system_rsi = system_rsi_result['rsi_14'].dropna()
                analysis_result['calculation_methods']['system_rsi'] = {
                    'method': 'RsiRsi._calculate_rsi()',
                    'data_points': len(system_rsi),
                    'values_sample': system_rsi.tail(5).tolist(),
                    'latest_value': float(system_rsi.iloc[-1]) if len(system_rsi) > 0 else None,
                    'range': {'min': float(system_rsi.min()), 'max': float(system_rsi.max())} if len(system_rsi) > 0 else None
                }
                print(f"  ✅ 系统RSI计算成功: {len(system_rsi)}个值")
                print(f"  📊 最新值: {system_rsi.iloc[-1]:.3f}")
                print(f"  📈 范围: {system_rsi.min():.2f} - {system_rsi.max():.2f}")
            else:
                print(f"  ❌ 系统RSI计算失败")
                analysis_result['calculation_methods']['system_rsi'] = {'method': 'RsiRsi._calculate_rsi()', 'status': 'FAILED'}
            
            # 方法2: 技术工具函数
            print(f"\n🔧 方法2: 技术工具函数")
            utils_rsi = calculate_rsi_Utils(test_data['close'], 14)
            
            if utils_rsi is not None and not utils_rsi.empty:
                utils_rsi_clean = utils_rsi.dropna()
                analysis_result['calculation_methods']['utils_rsi'] = {
                    'method': 'calculate_rsi_Utils()',
                    'data_points': len(utils_rsi_clean),
                    'values_sample': utils_rsi_clean.tail(5).tolist(),
                    'latest_value': float(utils_rsi_clean.iloc[-1]) if len(utils_rsi_clean) > 0 else None,
                    'range': {'min': float(utils_rsi_clean.min()), 'max': float(utils_rsi_clean.max())} if len(utils_rsi_clean) > 0 else None
                }
                print(f"  ✅ 工具函数RSI计算成功: {len(utils_rsi_clean)}个值")
                print(f"  📊 最新值: {utils_rsi_clean.iloc[-1]:.3f}")
                print(f"  📈 范围: {utils_rsi_clean.min():.2f} - {utils_rsi_clean.max():.2f}")
            else:
                print(f"  ❌ 工具函数RSI计算失败")
                analysis_result['calculation_methods']['utils_rsi'] = {'method': 'calculate_rsi_Utils()', 'status': 'FAILED'}
            
            # 方法3: 标准Wilder RSI算法
            print(f"\n🔧 方法3: 标准Wilder RSI算法")
            wilder_rsi = self._calculate_wilder_rsi(test_data['close'], 14)
            
            if wilder_rsi is not None and len(wilder_rsi) > 0:
                analysis_result['calculation_methods']['wilder_rsi'] = {
                    'method': 'Standard Wilder RSI',
                    'data_points': len(wilder_rsi),
                    'values_sample': wilder_rsi[-5:].tolist(),
                    'latest_value': float(wilder_rsi[-1]),
                    'range': {'min': float(np.min(wilder_rsi)), 'max': float(np.max(wilder_rsi))}
                }
                print(f"  ✅ Wilder RSI计算成功: {len(wilder_rsi)}个值")
                print(f"  📊 最新值: {wilder_rsi[-1]:.3f}")
                print(f"  📈 范围: {np.min(wilder_rsi):.2f} - {np.max(wilder_rsi):.2f}")
            else:
                print(f"  ❌ Wilder RSI计算失败")
                analysis_result['calculation_methods']['wilder_rsi'] = {'method': 'Standard Wilder RSI', 'status': 'FAILED'}
            
            # 方法4: 简单移动平均RSI算法
            print(f"\n🔧 方法4: 简单移动平均RSI算法")
            sma_rsi = self._calculate_sma_rsi(test_data['close'], 14)
            
            if sma_rsi is not None and len(sma_rsi) > 0:
                analysis_result['calculation_methods']['sma_rsi'] = {
                    'method': 'Simple Moving Average RSI',
                    'data_points': len(sma_rsi),
                    'values_sample': sma_rsi[-5:].tolist(),
                    'latest_value': float(sma_rsi[-1]),
                    'range': {'min': float(np.min(sma_rsi)), 'max': float(np.max(sma_rsi))}
                }
                print(f"  ✅ SMA RSI计算成功: {len(sma_rsi)}个值")
                print(f"  📊 最新值: {sma_rsi[-1]:.3f}")
                print(f"  📈 范围: {np.min(sma_rsi):.2f} - {np.max(sma_rsi):.2f}")
            else:
                print(f"  ❌ SMA RSI计算失败")
                analysis_result['calculation_methods']['sma_rsi'] = {'method': 'Simple Moving Average RSI', 'status': 'FAILED'}
            
            # 比较分析
            comparison_results = self._compare_rsi_methods(analysis_result['calculation_methods'])
            analysis_result['comparison_results'] = comparison_results
            
            # 根本原因分析
            root_cause = self._analyze_root_cause(analysis_result['calculation_methods'], comparison_results)
            analysis_result['root_cause_analysis'] = root_cause
            
        except Exception as e:
            analysis_result['error'] = str(e)
            print(f"❌ RSI算法分析异常: {e}")
        
        return analysis_result
    
    def _calculate_wilder_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """计算标准Wilder RSI"""
        
        try:
            prices_array = prices.values
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Wilder平滑方法
            avg_gains = np.zeros(len(gains))
            avg_losses = np.zeros(len(losses))
            
            # 初始平均值
            if len(gains) >= period:
                avg_gains[period-1] = np.mean(gains[:period])
                avg_losses[period-1] = np.mean(losses[:period])
                
                # Wilder平滑
                for i in range(period, len(gains)):
                    avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
                    avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
            
            # 计算RSI
            rsi = np.zeros(len(avg_gains))
            for i in range(period-1, len(avg_gains)):
                if avg_losses[i] == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi[i] = 100 - (100 / (1 + rs))
            
            return rsi[period-1:]
            
        except Exception as e:
            print(f"❌ Wilder RSI计算异常: {e}")
            return None
    
    def _calculate_sma_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """计算简单移动平均RSI"""
        
        try:
            prices_array = prices.values
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # 简单移动平均
            rsi = []
            for i in range(period-1, len(gains)):
                avg_gain = np.mean(gains[i-period+1:i+1])
                avg_loss = np.mean(losses[i-period+1:i+1])
                
                if avg_loss == 0:
                    rsi.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
            
            return np.array(rsi)
            
        except Exception as e:
            print(f"❌ SMA RSI计算异常: {e}")
            return None
    
    def _compare_rsi_methods(self, methods: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同RSI计算方法"""
        
        print(f"\n📊 RSI计算方法对比分析")
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
            if 'latest_value' in method_info and method_info['latest_value'] is not None:
                successful_methods[method_name] = method_info
                comparison['successful_methods'].append(method_name)
            else:
                comparison['failed_methods'].append(method_name)
        
        comparison['method_count'] = len(successful_methods)
        
        if len(successful_methods) >= 2:
            # 值对比
            print(f"📈 最新RSI值对比:")
            for method_name, method_info in successful_methods.items():
                latest_value = method_info['latest_value']
                print(f"  {method_name}: {latest_value:.3f}")
                comparison['value_comparisons'][method_name] = latest_value
            
            # 计算差异
            values = list(comparison['value_comparisons'].values())
            max_diff = max(values) - min(values)
            avg_value = sum(values) / len(values)
            
            comparison['accuracy_analysis'] = {
                'max_difference': max_diff,
                'average_value': avg_value,
                'relative_difference': max_diff / avg_value if avg_value != 0 else 0,
                'consistency_level': 'HIGH' if max_diff < 1 else 'MEDIUM' if max_diff < 5 else 'LOW'
            }
            
            print(f"\n📊 差异分析:")
            print(f"  最大差异: {max_diff:.3f}")
            print(f"  平均值: {avg_value:.3f}")
            print(f"  相对差异: {comparison['accuracy_analysis']['relative_difference']:.1%}")
            print(f"  一致性水平: {comparison['accuracy_analysis']['consistency_level']}")
        
        return comparison
    
    def _analyze_root_cause(self, methods: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
        """分析根本原因"""
        
        print(f"\n🔍 根本原因分析")
        print("=" * 60)
        
        root_cause = {
            'primary_issues': [],
            'algorithm_differences': [],
            'recommendations': [],
            'severity': 'UNKNOWN'
        }
        
        # 分析一致性水平
        consistency = comparison.get('accuracy_analysis', {}).get('consistency_level', 'UNKNOWN')
        max_diff = comparison.get('accuracy_analysis', {}).get('max_difference', 0)
        
        if consistency == 'LOW':
            root_cause['primary_issues'].append({
                'issue': '算法实现差异显著',
                'description': f'不同RSI计算方法的最大差异达到{max_diff:.3f}，超出可接受范围',
                'impact': 'HIGH'
            })
            root_cause['severity'] = 'HIGH'
        elif consistency == 'MEDIUM':
            root_cause['primary_issues'].append({
                'issue': '算法实现存在差异',
                'description': f'不同RSI计算方法存在{max_diff:.3f}的差异，需要统一',
                'impact': 'MEDIUM'
            })
            root_cause['severity'] = 'MEDIUM'
        else:
            root_cause['primary_issues'].append({
                'issue': '算法实现基本一致',
                'description': f'不同RSI计算方法差异较小({max_diff:.3f})，在可接受范围内',
                'impact': 'LOW'
            })
            root_cause['severity'] = 'LOW'
        
        # 分析具体的算法差异
        if 'system_rsi' in methods and 'utils_rsi' in methods:
            system_val = methods['system_rsi'].get('latest_value')
            utils_val = methods['utils_rsi'].get('latest_value')
            
            if system_val is not None and utils_val is not None:
                diff = abs(system_val - utils_val)
                if diff > 1:
                    root_cause['algorithm_differences'].append({
                        'comparison': 'system_rsi vs utils_rsi',
                        'difference': diff,
                        'system_value': system_val,
                        'utils_value': utils_val,
                        'possible_cause': '平滑方法或初始化差异'
                    })
        
        # 生成建议
        if root_cause['severity'] in ['HIGH', 'MEDIUM']:
            root_cause['recommendations'].extend([
                '统一RSI计算算法实现',
                '采用标准Wilder平滑方法',
                '确保初始化参数一致',
                '建立标准RSI基准数据集'
            ])
        else:
            root_cause['recommendations'].extend([
                '当前算法实现可接受',
                '可进行微调优化',
                '建立监控机制'
            ])
        
        # 输出分析结果
        print(f"🎯 严重程度: {root_cause['severity']}")
        
        if root_cause['primary_issues']:
            print(f"\n❗ 主要问题:")
            for issue in root_cause['primary_issues']:
                print(f"  • {issue['issue']}: {issue['description']}")
        
        if root_cause['algorithm_differences']:
            print(f"\n🔧 算法差异:")
            for diff in root_cause['algorithm_differences']:
                print(f"  • {diff['comparison']}: 差异{diff['difference']:.3f}")
        
        if root_cause['recommendations']:
            print(f"\n💡 建议:")
            for i, rec in enumerate(root_cause['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        return root_cause
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整的RSI算法分析"""
        
        print(f"\n🎯 RSI算法差异深度分析")
        print("分析系统RSI与基准RSI计算方法差异，找出准确率问题根本原因")
        print("=" * 80)
        
        complete_analysis = {
            'analysis_type': 'COMPLETE_RSI_ALGORITHM_ANALYSIS',
            'start_time': datetime.now().isoformat(),
            'algorithm_analysis': {},
            'final_assessment': {},
            'action_plan': {}
        }
        
        try:
            # 1. RSI计算方法分析
            algorithm_analysis = self.analyze_rsi_calculation_methods()
            complete_analysis['algorithm_analysis'] = algorithm_analysis
            
            # 2. 最终评估
            final_assessment = self._generate_final_assessment(algorithm_analysis)
            complete_analysis['final_assessment'] = final_assessment
            
            # 3. 行动计划
            action_plan = self._generate_action_plan(algorithm_analysis, final_assessment)
            complete_analysis['action_plan'] = action_plan
            
            complete_analysis['end_time'] = datetime.now().isoformat()
            
            print(f"\n🏆 RSI算法分析完成")
            print(f"问题严重程度: {final_assessment.get('severity', 'UNKNOWN')}")
            print(f"需要修复: {'是' if final_assessment.get('needs_fix', False) else '否'}")
            
        except Exception as e:
            complete_analysis['error'] = str(e)
            print(f"❌ 完整分析异常: {e}")
        
        return complete_analysis
    
    def _generate_final_assessment(self, algorithm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终评估"""
        
        root_cause = algorithm_analysis.get('root_cause_analysis', {})
        comparison = algorithm_analysis.get('comparison_results', {})
        
        assessment = {
            'severity': root_cause.get('severity', 'UNKNOWN'),
            'needs_fix': False,
            'confidence_level': 'HIGH',
            'impact_on_accuracy': 'UNKNOWN'
        }
        
        # 判断是否需要修复
        max_diff = comparison.get('accuracy_analysis', {}).get('max_difference', 0)
        
        if max_diff > 5:
            assessment['needs_fix'] = True
            assessment['impact_on_accuracy'] = 'HIGH'
        elif max_diff > 2:
            assessment['needs_fix'] = True
            assessment['impact_on_accuracy'] = 'MEDIUM'
        else:
            assessment['needs_fix'] = False
            assessment['impact_on_accuracy'] = 'LOW'
        
        return assessment
    
    def _generate_action_plan(self, algorithm_analysis: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
        """生成行动计划"""
        
        action_plan = {
            'immediate_actions': [],
            'medium_term_actions': [],
            'success_criteria': []
        }
        
        if assessment.get('needs_fix', False):
            action_plan['immediate_actions'] = [
                '统一RSI计算算法实现',
                '修复系统RSI与基准RSI的差异',
                '重新验证基准准确率'
            ]
            
            action_plan['medium_term_actions'] = [
                '建立标准RSI基准数据库',
                '完善RSI算法测试用例',
                '建立持续监控机制'
            ]
            
            action_plan['success_criteria'] = [
                '基准准确率达到95%以上',
                '不同算法实现差异小于1%',
                '通过完整的阶段4验证'
            ]
        else:
            action_plan['immediate_actions'] = [
                '当前算法实现可接受',
                '进行微调优化',
                '继续阶段4验证'
            ]
            
            action_plan['success_criteria'] = [
                '维持当前算法精度',
                '通过阶段4验证'
            ]
        
        return action_plan

def main():
    """主函数"""
    print("🎯 RSI算法差异深度分析")
    print("找出系统RSI与基准RSI准确率仅61.6%的根本原因")
    
    # 创建分析器
    analyzer = RSIAlgorithmAnalyzer()
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    # 显示结果摘要
    print(f"\n📊 分析结果摘要")
    print("=" * 80)
    
    if 'final_assessment' in results:
        assessment = results['final_assessment']
        print(f"问题严重程度: {assessment.get('severity', 'UNKNOWN')}")
        print(f"需要修复: {'是' if assessment.get('needs_fix', False) else '否'}")
        print(f"对准确率影响: {assessment.get('impact_on_accuracy', 'UNKNOWN')}")
    
    if 'action_plan' in results:
        action_plan = results['action_plan']
        if action_plan.get('immediate_actions'):
            print(f"\n🎯 立即行动:")
            for action in action_plan['immediate_actions']:
                print(f"  • {action}")

if __name__ == "__main__":
    main()
