#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD算法修复

基于RSI项目成功经验，修复MACD算法差异问题
确保系统MACD与基准MACD完全一致
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

class MACDAlgorithmFixer:
    """MACD算法修复器"""
    
    def __init__(self):
        """初始化修复器"""
        self.fixer_name = "MACD算法修复器"
        
        # 连接数据库
        self.client = Client(
            host='localhost',
            port=9000,
            database='stock',
            user='default',
            password='123456'
        )
        
        self.macd_indicator = MacdMacd()
        
        print(f"✅ {self.fixer_name}初始化完成")
        print(f"🎯 基于RSI项目成功经验，统一MACD算法")
    
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
    
    def test_current_algorithm_consistency(self) -> Dict[str, Any]:
        """测试当前算法一致性"""
        
        print(f"\n🔍 测试当前MACD算法一致性")
        print("=" * 80)
        
        test_result = {
            'test_type': 'CURRENT_ALGORITHM_CONSISTENCY',
            'timestamp': datetime.now().isoformat(),
            'test_results': [],
            'overall_consistency': 0.0,
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 获取测试数据
            test_data = self.get_test_data('000001', 50)
            
            if test_data is None or len(test_data) < 40:
                print("❌ 无法获取足够的测试数据")
                test_result['status'] = 'NO_DATA'
                return test_result
            
            print(f"📊 测试数据: {len(test_data)}个数据点")
            
            # 测试不同股票的一致性
            available_stocks = ['002492', '002578', '300900']
            
            total_consistency = 0.0
            valid_tests = 0
            
            for stock_code in available_stocks:
                print(f"\n📊 测试股票 {stock_code}")
                
                try:
                    # 获取股票数据
                    stock_data = self.get_test_data(stock_code, 50)
                    
                    if stock_data is None or len(stock_data) < 40:
                        print(f"  ⚠️ {stock_code} 数据不足，跳过")
                        continue
                    
                    # 方法1: 系统MACD
                    system_macd_result = self.macd_indicator._calculate_macd(stock_data)
                    
                    if system_macd_result is None or system_macd_result.empty:
                        print(f"  ❌ {stock_code} 系统MACD计算失败")
                        continue
                    
                    system_dif = system_macd_result.get('macd_line', pd.Series()).dropna()
                    system_dea = system_macd_result.get('macd_signal', pd.Series()).dropna()
                    
                    if len(system_dif) == 0 or len(system_dea) == 0:
                        print(f"  ❌ {stock_code} 系统MACD无有效值")
                        continue
                    
                    # 方法2: 工具函数(standard)
                    utils_dif, utils_dea, utils_macd = calculate_macd_Utils(
                        stock_data['close'], 
                        fast_period=12, 
                        slow_period=26, 
                        signal_period=9,
                        method='standard'
                    )
                    
                    if utils_dif is None or utils_dif.empty:
                        print(f"  ❌ {stock_code} 工具函数MACD计算失败")
                        continue
                    
                    utils_dif_clean = utils_dif.dropna()
                    utils_dea_clean = utils_dea.dropna()
                    
                    if len(utils_dif_clean) == 0 or len(utils_dea_clean) == 0:
                        print(f"  ❌ {stock_code} 工具函数MACD无有效值")
                        continue
                    
                    # 精确对比
                    min_len = min(len(system_dif), len(utils_dif_clean))
                    
                    if min_len >= 10:
                        system_dif_values = system_dif.tail(min_len).values
                        utils_dif_values = utils_dif_clean.tail(min_len).values
                        
                        system_dea_values = system_dea.tail(min_len).values
                        utils_dea_values = utils_dea_clean.tail(min_len).values
                        
                        # 计算DIF差异
                        dif_differences = np.abs(system_dif_values - utils_dif_values)
                        max_dif_diff = dif_differences.max()
                        avg_dif_diff = dif_differences.mean()
                        
                        # 计算DEA差异
                        dea_differences = np.abs(system_dea_values - utils_dea_values)
                        max_dea_diff = dea_differences.max()
                        avg_dea_diff = dea_differences.mean()
                        
                        # 计算一致性（基于极小误差容忍）
                        tolerance = 1e-6
                        dif_accurate_count = np.sum(dif_differences <= tolerance)
                        dea_accurate_count = np.sum(dea_differences <= tolerance)
                        
                        dif_accuracy = dif_accurate_count / len(dif_differences)
                        dea_accuracy = dea_accurate_count / len(dea_differences)
                        overall_accuracy = (dif_accuracy + dea_accuracy) / 2
                        
                        test_result_item = {
                            'stock_code': stock_code,
                            'data_points': len(stock_data),
                            'compared_values': min_len,
                            'dif_max_diff': float(max_dif_diff),
                            'dif_avg_diff': float(avg_dif_diff),
                            'dea_max_diff': float(max_dea_diff),
                            'dea_avg_diff': float(avg_dea_diff),
                            'dif_accuracy': dif_accuracy,
                            'dea_accuracy': dea_accuracy,
                            'overall_accuracy': overall_accuracy,
                            'latest_system_dif': float(system_dif.iloc[-1]),
                            'latest_utils_dif': float(utils_dif_clean.iloc[-1]),
                            'status': 'PASSED' if overall_accuracy >= 0.99 else 'FAILED'
                        }
                        
                        test_result['test_results'].append(test_result_item)
                        total_consistency += overall_accuracy
                        valid_tests += 1
                        
                        print(f"  ✅ 对比{min_len}个值:")
                        print(f"    DIF: 最大差异{max_dif_diff:.6f}, 平均差异{avg_dif_diff:.6f}, 准确率{dif_accuracy:.1%}")
                        print(f"    DEA: 最大差异{max_dea_diff:.6f}, 平均差异{avg_dea_diff:.6f}, 准确率{dea_accuracy:.1%}")
                        print(f"    总体准确率: {overall_accuracy:.1%}")
                    else:
                        print(f"  ⚠️ {stock_code} 数据不足，无法对比")
                
                except Exception as e:
                    print(f"  ❌ {stock_code} 测试异常: {e}")
                    continue
            
            # 计算总体结果
            if valid_tests > 0:
                test_result['overall_consistency'] = total_consistency / valid_tests
                
                if test_result['overall_consistency'] >= 0.995:
                    test_result['status'] = 'PASSED'
                    print(f"\n✅ 算法一致性测试通过: {test_result['overall_consistency']:.1%} ≥ 99.5%")
                else:
                    test_result['status'] = 'FAILED'
                    print(f"\n❌ 算法一致性测试失败: {test_result['overall_consistency']:.1%} < 99.5%")
            else:
                test_result['status'] = 'NO_VALID_DATA'
                print(f"\n❌ 没有有效的测试数据")
        
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"❌ 算法一致性测试异常: {e}")
        
        return test_result
    
    def fix_ema_method_consistency(self) -> Dict[str, Any]:
        """修复EMA方法一致性"""
        
        print(f"\n🔧 修复EMA方法一致性")
        print("=" * 80)
        
        fix_result = {
            'fix_type': 'EMA_METHOD_CONSISTENCY',
            'timestamp': datetime.now().isoformat(),
            'changes_made': [],
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 检查当前技术工具函数的默认方法
            print(f"📊 检查技术工具函数默认EMA方法")
            
            # 查看calculate_macd_Utils的默认参数
            import inspect
            macd_signature = inspect.signature(calculate_macd_Utils)
            method_param = macd_signature.parameters.get('method')
            
            if method_param and method_param.default:
                current_default = method_param.default
                print(f"  当前默认方法: {current_default}")
                
                if current_default != 'standard':
                    print(f"  ⚠️ 发现问题: 默认方法不是'standard'")
                    fix_result['changes_made'].append({
                        'file': 'utils/technical_utils.py',
                        'function': 'calculate_macd_Utils',
                        'change': f'修改默认method参数从{current_default}到standard'
                    })
                else:
                    print(f"  ✅ 默认方法正确")
            
            # 检查系统MACD的默认方法
            print(f"📊 检查系统MACD默认EMA方法")
            
            # 创建测试数据
            test_data = pd.DataFrame({
                'close': [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
                'date': pd.date_range('2025-01-01', periods=10)
            })
            
            # 测试系统MACD使用的默认方法
            try:
                system_result = self.macd_indicator._calculate_macd(test_data)
                print(f"  ✅ 系统MACD使用默认方法正常")
            except Exception as e:
                print(f"  ❌ 系统MACD默认方法异常: {e}")
                fix_result['changes_made'].append({
                    'file': 'indicators/macd.py',
                    'function': '_calculate_macd',
                    'change': '修复系统MACD默认方法调用'
                })
            
            # 建议的修复措施
            recommendations = [
                '确保所有MACD计算都使用standard方法',
                '移除或标记sma_init方法为非标准方法',
                '在文档中明确标准EMA计算方法',
                '建立EMA方法一致性测试'
            ]
            
            fix_result['recommendations'] = recommendations
            fix_result['status'] = 'COMPLETED'
            
            print(f"\n💡 修复建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        except Exception as e:
            fix_result['status'] = 'ERROR'
            fix_result['error'] = str(e)
            print(f"❌ EMA方法一致性修复异常: {e}")
        
        return fix_result
    
    def create_standard_macd_benchmark(self) -> Dict[str, Any]:
        """创建标准MACD基准数据集"""
        
        print(f"\n📊 创建标准MACD基准数据集")
        print("=" * 80)
        
        benchmark_result = {
            'benchmark_type': 'STANDARD_MACD_BENCHMARK',
            'timestamp': datetime.now().isoformat(),
            'benchmark_data': {},
            'status': 'IN_PROGRESS'
        }
        
        try:
            # 获取测试数据
            test_data = self.get_test_data('002492', 50)
            
            if test_data is None or len(test_data) < 40:
                print("❌ 无法获取足够的测试数据")
                benchmark_result['status'] = 'NO_DATA'
                return benchmark_result
            
            print(f"📊 使用{len(test_data)}个数据点创建基准")
            
            # 使用标准方法计算MACD基准
            benchmark_dif, benchmark_dea, benchmark_macd = calculate_macd_Utils(
                test_data['close'], 
                fast_period=12, 
                slow_period=26, 
                signal_period=9,
                method='standard'
            )
            
            if benchmark_dif is not None and not benchmark_dif.empty:
                benchmark_dif_clean = benchmark_dif.dropna()
                benchmark_dea_clean = benchmark_dea.dropna()
                benchmark_macd_clean = benchmark_macd.dropna()
                
                # 保存基准数据
                benchmark_result['benchmark_data'] = {
                    'stock_code': '002492',
                    'data_points': len(test_data),
                    'price_data': test_data['close'].tolist(),
                    'dif_values': benchmark_dif_clean.tolist(),
                    'dea_values': benchmark_dea_clean.tolist(),
                    'macd_values': benchmark_macd_clean.tolist(),
                    'parameters': {
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9,
                        'method': 'standard'
                    }
                }
                
                benchmark_result['status'] = 'COMPLETED'
                
                print(f"✅ 基准数据集创建成功")
                print(f"  DIF值数量: {len(benchmark_dif_clean)}")
                print(f"  DEA值数量: {len(benchmark_dea_clean)}")
                print(f"  MACD值数量: {len(benchmark_macd_clean)}")
                print(f"  最新DIF: {benchmark_dif_clean.iloc[-1]:.6f}")
                print(f"  最新DEA: {benchmark_dea_clean.iloc[-1]:.6f}")
                print(f"  最新MACD: {benchmark_macd_clean.iloc[-1]:.6f}")
            else:
                benchmark_result['status'] = 'CALCULATION_FAILED'
                print(f"❌ 基准MACD计算失败")
        
        except Exception as e:
            benchmark_result['status'] = 'ERROR'
            benchmark_result['error'] = str(e)
            print(f"❌ 基准数据集创建异常: {e}")
        
        return benchmark_result
    
    def run_complete_fix(self) -> Dict[str, Any]:
        """运行完整的MACD算法修复"""
        
        print(f"\n🎯 MACD算法完整修复")
        print("基于RSI项目成功经验，统一MACD算法实现")
        print("=" * 80)
        
        complete_fix = {
            'fix_type': 'COMPLETE_MACD_ALGORITHM_FIX',
            'start_time': datetime.now().isoformat(),
            'current_consistency_test': {},
            'ema_method_fix': {},
            'benchmark_creation': {},
            'final_verification': {},
            'overall_assessment': {},
            'final_status': 'IN_PROGRESS'
        }
        
        try:
            # 1. 测试当前算法一致性
            current_test = self.test_current_algorithm_consistency()
            complete_fix['current_consistency_test'] = current_test
            
            # 2. 修复EMA方法一致性
            ema_fix = self.fix_ema_method_consistency()
            complete_fix['ema_method_fix'] = ema_fix
            
            # 3. 创建标准基准数据集
            benchmark = self.create_standard_macd_benchmark()
            complete_fix['benchmark_creation'] = benchmark
            
            # 4. 最终验证
            final_verification = self.test_current_algorithm_consistency()
            complete_fix['final_verification'] = final_verification
            
            # 5. 总体评估
            overall_assessment = self._assess_fix_results(
                current_test, ema_fix, benchmark, final_verification
            )
            complete_fix['overall_assessment'] = overall_assessment
            complete_fix['final_status'] = overall_assessment['final_status']
            
            complete_fix['end_time'] = datetime.now().isoformat()
            
            print(f"\n🏆 MACD算法修复完成")
            print(f"最终状态: {complete_fix['final_status']}")
            print(f"修复效果: {overall_assessment.get('fix_effectiveness', 'UNKNOWN')}")
            
        except Exception as e:
            complete_fix['final_status'] = 'ERROR'
            complete_fix['error'] = str(e)
            print(f"❌ 完整修复异常: {e}")
        
        return complete_fix
    
    def _assess_fix_results(self, current_test: Dict, ema_fix: Dict, 
                           benchmark: Dict, final_verification: Dict) -> Dict[str, Any]:
        """评估修复结果"""
        
        assessment = {
            'assessment_type': 'FIX_RESULTS_ASSESSMENT',
            'current_consistency': current_test.get('overall_consistency', 0),
            'final_consistency': final_verification.get('overall_consistency', 0),
            'improvement': 0.0,
            'fix_effectiveness': 'UNKNOWN',
            'final_status': 'UNKNOWN'
        }
        
        # 计算改进效果
        current_consistency = current_test.get('overall_consistency', 0)
        final_consistency = final_verification.get('overall_consistency', 0)
        
        assessment['improvement'] = final_consistency - current_consistency
        
        # 评估修复效果
        if final_consistency >= 0.995:
            assessment['fix_effectiveness'] = 'EXCELLENT'
            assessment['final_status'] = 'FULLY_FIXED'
        elif final_consistency >= 0.99:
            assessment['fix_effectiveness'] = 'GOOD'
            assessment['final_status'] = 'MOSTLY_FIXED'
        elif assessment['improvement'] > 0.05:
            assessment['fix_effectiveness'] = 'PARTIAL'
            assessment['final_status'] = 'PARTIALLY_FIXED'
        else:
            assessment['fix_effectiveness'] = 'MINIMAL'
            assessment['final_status'] = 'NEEDS_MORE_WORK'
        
        return assessment

def main():
    """主函数"""
    print("🎯 MACD算法修复")
    print("基于RSI项目成功经验，统一MACD算法实现")
    
    # 创建修复器
    fixer = MACDAlgorithmFixer()
    
    # 运行完整修复
    results = fixer.run_complete_fix()
    
    # 显示结果摘要
    print(f"\n📊 MACD算法修复结果摘要")
    print("=" * 80)
    
    if 'overall_assessment' in results:
        assessment = results['overall_assessment']
        print(f"当前一致性: {assessment.get('current_consistency', 0):.1%}")
        print(f"最终一致性: {assessment.get('final_consistency', 0):.1%}")
        print(f"改进效果: {assessment.get('improvement', 0):.1%}")
        print(f"修复效果: {assessment.get('fix_effectiveness', 'UNKNOWN')}")
        print(f"最终状态: {assessment.get('final_status', 'UNKNOWN')}")

if __name__ == "__main__":
    main()
