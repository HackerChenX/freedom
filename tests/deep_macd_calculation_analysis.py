#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析MACD计算错误的根本原因

发现的问题：
1. 000017在2025-05-14: DIFF差异15.3%, MACD差异122.3%
2. 用户选中了2025-05-09日期，可能暗示日期处理问题
3. DEA计算准确但DIFF计算有误，说明问题在EMA12/EMA26计算

深入分析方向：
1. 日期索引对齐问题
2. EMA计算的初始化方法
3. 数据预处理差异
4. 时间窗口边界问题
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from indicators.macd import MacdMacd
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class DeepMacdAnalyzer:
    """深入MACD计算分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.macd_indicator = MacdMacd()
        self.stock_data_service = get_stock_data_service()
        
        # 测试用例：已知有问题的股票
        self.test_cases = {
            '000017': {
                'problem_date': '2025-05-14',
                'real_data': {'MACD': 0.037, 'DIFF': 0.149, 'DEA': 0.13},
                'system_data': {'MACD': -0.008249, 'DIFF': 0.126218, 'DEA': 0.130343}
            },
            '000001': {
                'benchmark_date': '2025-05-12', 
                'real_data': {'MACD': 0.073, 'DIFF': -0.039, 'DEA': -0.076},
                'expected_accurate': True
            }
        }
        
        print("🔍 深入MACD计算错误分析器初始化完成")
    
    def analyze_ema_calculation_methods(self, stock_code: str, target_date: str) -> dict:
        """分析EMA计算方法的差异"""
        
        print(f"\n📊 分析{stock_code}的EMA计算方法 (目标日期: {target_date})")
        print("=" * 80)
        
        try:
            # 获取股票数据
            df = self.stock_data_service.get_stock_data(stock_code, days=200)
            
            if df is None or len(df) == 0:
                return {'error': f'无法获取{stock_code}数据'}
            
            # 查找目标日期
            target_date_obj = pd.to_datetime(target_date).date()
            target_rows = df[df['date'].dt.date == target_date_obj]
            
            if target_rows.empty:
                return {'error': f'未找到{target_date}的数据'}
            
            target_idx = target_rows.index[0]
            close_prices = df['close'].values
            
            print(f"📈 数据基本信息:")
            print(f"  总数据量: {len(df)}天")
            print(f"  目标日期索引: {target_idx}")
            print(f"  目标日期收盘价: {close_prices[target_idx]:.3f}")
            
            # 方法1: 标准EMA计算（从第一天开始）
            ema12_standard = self._calculate_ema_standard(close_prices, 12)
            ema26_standard = self._calculate_ema_standard(close_prices, 26)
            diff_standard = ema12_standard[target_idx] - ema26_standard[target_idx]
            
            # 方法2: SMA初始化的EMA计算
            ema12_sma_init = self._calculate_ema_sma_init(close_prices, 12)
            ema26_sma_init = self._calculate_ema_sma_init(close_prices, 26)
            diff_sma_init = ema12_sma_init[target_idx] - ema26_sma_init[target_idx]
            
            # 方法3: 系统计算结果
            macd_result = self.macd_indicator.calculate(df)
            system_diff = macd_result.iloc[target_idx]['macd_line']
            system_dea = macd_result.iloc[target_idx]['macd_signal']
            system_macd = macd_result.iloc[target_idx]['macd_histogram']
            
            # 方法4: 不同起始点的EMA计算
            ema12_from_26 = self._calculate_ema_from_index(close_prices, 12, 26)
            ema26_from_26 = self._calculate_ema_from_index(close_prices, 26, 26)
            diff_from_26 = ema12_from_26[target_idx] - ema26_from_26[target_idx] if target_idx < len(ema12_from_26) else None
            
            analysis_result = {
                'stock_code': stock_code,
                'target_date': target_date,
                'target_index': target_idx,
                'close_price': close_prices[target_idx],
                'ema_methods': {
                    'standard': {
                        'ema12': ema12_standard[target_idx],
                        'ema26': ema26_standard[target_idx],
                        'diff': diff_standard,
                        'description': '从第一天开始的标准EMA'
                    },
                    'sma_init': {
                        'ema12': ema12_sma_init[target_idx],
                        'ema26': ema26_sma_init[target_idx], 
                        'diff': diff_sma_init,
                        'description': 'SMA初始化的EMA'
                    },
                    'system': {
                        'diff': system_diff,
                        'dea': system_dea,
                        'macd': system_macd,
                        'description': '系统计算结果'
                    },
                    'from_26th': {
                        'ema12': ema12_from_26[target_idx] if diff_from_26 is not None else None,
                        'ema26': ema26_from_26[target_idx] if diff_from_26 is not None else None,
                        'diff': diff_from_26,
                        'description': '从第26天开始的EMA'
                    }
                }
            }
            
            # 显示对比结果
            print(f"\n📊 EMA计算方法对比:")
            print(f"{'方法':<15} {'EMA12':<12} {'EMA26':<12} {'DIFF':<12} {'与系统差异':<12}")
            print("-" * 75)
            
            for method_name, method_data in analysis_result['ema_methods'].items():
                if method_name == 'system':
                    print(f"{'系统计算':<15} {'-':<12} {'-':<12} {method_data['diff']:<12.6f} {'基准':<12}")
                else:
                    if method_data['diff'] is not None:
                        diff_vs_system = abs(method_data['diff'] - system_diff)
                        print(f"{method_data['description'][:14]:<15} {method_data['ema12']:<12.6f} {method_data['ema26']:<12.6f} {method_data['diff']:<12.6f} {diff_vs_system:<12.6f}")
                    else:
                        print(f"{method_data['description'][:14]:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            
            return analysis_result
            
        except Exception as e:
            return {'error': f'分析过程异常: {str(e)}'}
    
    def _calculate_ema_standard(self, prices: np.ndarray, period: int) -> np.ndarray:
        """标准EMA计算（从第一天开始）"""
        ema = np.zeros(len(prices))
        multiplier = 2 / (period + 1)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _calculate_ema_sma_init(self, prices: np.ndarray, period: int) -> np.ndarray:
        """SMA初始化的EMA计算"""
        if len(prices) < period:
            return self._calculate_ema_standard(prices, period)
        
        ema = np.zeros(len(prices))
        multiplier = 2 / (period + 1)
        
        # 前period天用SMA初始化
        ema[period-1] = np.mean(prices[:period])
        
        # 从第period+1天开始用EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        # 前period-1天设为NaN或使用SMA
        for i in range(period-1):
            ema[i] = np.mean(prices[:i+1]) if i > 0 else prices[0]
        
        return ema
    
    def _calculate_ema_from_index(self, prices: np.ndarray, period: int, start_idx: int) -> np.ndarray:
        """从指定索引开始计算EMA"""
        if start_idx >= len(prices):
            return np.full(len(prices), np.nan)
        
        ema = np.full(len(prices), np.nan)
        multiplier = 2 / (period + 1)
        
        # 从start_idx开始计算
        ema[start_idx] = prices[start_idx]
        
        for i in range(start_idx + 1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def analyze_date_alignment_issues(self, stock_code: str) -> dict:
        """分析日期对齐问题"""
        
        print(f"\n📅 分析{stock_code}的日期对齐问题")
        print("=" * 80)
        
        try:
            # 获取不同天数的数据
            df_30 = self.stock_data_service.get_stock_data(stock_code, days=30)
            df_60 = self.stock_data_service.get_stock_data(stock_code, days=60)
            df_120 = self.stock_data_service.get_stock_data(stock_code, days=120)
            df_200 = self.stock_data_service.get_stock_data(stock_code, days=200)
            
            datasets = {
                '30天': df_30,
                '60天': df_60, 
                '120天': df_120,
                '200天': df_200
            }
            
            target_date = '2025-05-14'
            target_date_obj = pd.to_datetime(target_date).date()
            
            alignment_results = {}
            
            for name, df in datasets.items():
                if df is None or len(df) == 0:
                    alignment_results[name] = {'error': '无数据'}
                    continue
                
                # 查找目标日期
                target_rows = df[df['date'].dt.date == target_date_obj]
                
                if target_rows.empty:
                    alignment_results[name] = {'error': '未找到目标日期'}
                    continue
                
                target_idx = target_rows.index[0]
                
                # 计算MACD
                macd_result = self.macd_indicator.calculate(df)
                
                if macd_result is None or target_idx >= len(macd_result):
                    alignment_results[name] = {'error': 'MACD计算失败'}
                    continue
                
                macd_data = macd_result.iloc[target_idx]
                
                alignment_results[name] = {
                    'data_length': len(df),
                    'target_index': target_idx,
                    'date_range': f"{df['date'].min().date()} 到 {df['date'].max().date()}",
                    'macd_values': {
                        'diff': macd_data['macd_line'],
                        'dea': macd_data['macd_signal'],
                        'macd': macd_data['macd_histogram']
                    }
                }
            
            # 显示对比结果
            print(f"📊 不同数据长度的MACD计算结果对比 (目标日期: {target_date}):")
            print(f"{'数据集':<8} {'长度':<6} {'索引':<6} {'DIFF':<12} {'DEA':<12} {'MACD':<12}")
            print("-" * 70)
            
            for name, result in alignment_results.items():
                if 'error' in result:
                    print(f"{name:<8} {'N/A':<6} {'N/A':<6} {result['error']:<36}")
                else:
                    values = result['macd_values']
                    print(f"{name:<8} {result['data_length']:<6} {result['target_index']:<6} {values['diff']:<12.6f} {values['dea']:<12.6f} {values['macd']:<12.6f}")
            
            return {
                'stock_code': stock_code,
                'target_date': target_date,
                'alignment_results': alignment_results
            }
            
        except Exception as e:
            return {'error': f'日期对齐分析异常: {str(e)}'}
    
    def analyze_data_preprocessing_differences(self, stock_code: str, target_date: str) -> dict:
        """分析数据预处理差异"""
        
        print(f"\n🔄 分析{stock_code}的数据预处理差异")
        print("=" * 80)
        
        try:
            df = self.stock_data_service.get_stock_data(stock_code, days=200)
            
            if df is None or len(df) == 0:
                return {'error': f'无法获取{stock_code}数据'}
            
            target_date_obj = pd.to_datetime(target_date).date()
            target_rows = df[df['date'].dt.date == target_date_obj]
            
            if target_rows.empty:
                return {'error': f'未找到{target_date}的数据'}
            
            target_idx = target_rows.index[0]
            
            # 检查数据质量
            print(f"📊 数据质量检查:")
            print(f"  总数据量: {len(df)}")
            print(f"  空值检查: {df['close'].isnull().sum()}个空值")
            print(f"  重复日期: {df['date'].duplicated().sum()}个重复")
            print(f"  价格异常: {(df['close'] <= 0).sum()}个异常价格")
            
            # 检查目标日期前后的数据连续性
            print(f"\n📅 目标日期前后数据连续性:")
            start_check = max(0, target_idx - 5)
            end_check = min(len(df), target_idx + 6)
            
            for i in range(start_check, end_check):
                date_str = df.iloc[i]['date'].strftime('%Y-%m-%d')
                close_price = df.iloc[i]['close']
                marker = " ← 目标" if i == target_idx else ""
                print(f"  {date_str}: {close_price:.3f}{marker}")
            
            # 检查是否存在数据跳跃
            close_prices = df['close'].values
            price_changes = np.diff(close_prices)
            large_changes = np.where(np.abs(price_changes) > np.std(price_changes) * 3)[0]
            
            print(f"\n📈 价格异常波动检查:")
            if len(large_changes) > 0:
                print(f"  发现{len(large_changes)}个异常波动点:")
                for idx in large_changes[-5:]:  # 显示最近5个
                    if idx < len(df) - 1:
                        date1 = df.iloc[idx]['date'].strftime('%Y-%m-%d')
                        date2 = df.iloc[idx+1]['date'].strftime('%Y-%m-%d')
                        price1 = df.iloc[idx]['close']
                        price2 = df.iloc[idx+1]['close']
                        change = (price2 - price1) / price1 * 100
                        print(f"    {date1} → {date2}: {price1:.3f} → {price2:.3f} ({change:+.1f}%)")
            else:
                print(f"  ✅ 未发现异常价格波动")
            
            # 检查MACD计算的稳定性
            print(f"\n🔄 MACD计算稳定性检查:")
            
            # 多次计算同一数据
            macd_results = []
            for i in range(3):
                macd_result = self.macd_indicator.calculate(df.copy())
                if macd_result is not None and target_idx < len(macd_result):
                    macd_data = macd_result.iloc[target_idx]
                    macd_results.append({
                        'diff': macd_data['macd_line'],
                        'dea': macd_data['macd_signal'],
                        'macd': macd_data['macd_histogram']
                    })
            
            if len(macd_results) > 1:
                print(f"  计算一致性检查:")
                for i, result in enumerate(macd_results):
                    print(f"    第{i+1}次: DIFF={result['diff']:.6f}, DEA={result['dea']:.6f}, MACD={result['macd']:.6f}")
                
                # 检查计算差异
                diff_std = np.std([r['diff'] for r in macd_results])
                dea_std = np.std([r['dea'] for r in macd_results])
                macd_std = np.std([r['macd'] for r in macd_results])
                
                print(f"  计算稳定性: DIFF标准差={diff_std:.8f}, DEA标准差={dea_std:.8f}, MACD标准差={macd_std:.8f}")
                
                if diff_std < 1e-10 and dea_std < 1e-10 and macd_std < 1e-10:
                    print(f"  ✅ 计算结果高度一致")
                else:
                    print(f"  ⚠️ 计算结果存在差异")
            
            return {
                'stock_code': stock_code,
                'target_date': target_date,
                'data_quality': {
                    'total_records': len(df),
                    'null_values': df['close'].isnull().sum(),
                    'duplicate_dates': df['date'].duplicated().sum(),
                    'abnormal_prices': (df['close'] <= 0).sum(),
                    'large_changes': len(large_changes)
                },
                'calculation_stability': macd_results
            }
            
        except Exception as e:
            return {'error': f'数据预处理分析异常: {str(e)}'}
    
    def comprehensive_root_cause_analysis(self) -> dict:
        """综合根因分析"""
        
        print(f"\n🎯 MACD计算错误综合根因分析")
        print("=" * 80)
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'test_cases': {},
            'root_causes': [],
            'recommendations': []
        }
        
        # 分析每个测试用例
        for stock_code, test_case in self.test_cases.items():
            print(f"\n📊 分析{stock_code}股票:")
            
            if 'problem_date' in test_case:
                target_date = test_case['problem_date']
            else:
                target_date = test_case['benchmark_date']
            
            # EMA计算方法分析
            ema_analysis = self.analyze_ema_calculation_methods(stock_code, target_date)
            
            # 日期对齐分析
            alignment_analysis = self.analyze_date_alignment_issues(stock_code)
            
            # 数据预处理分析
            preprocessing_analysis = self.analyze_data_preprocessing_differences(stock_code, target_date)
            
            analysis_results['test_cases'][stock_code] = {
                'ema_analysis': ema_analysis,
                'alignment_analysis': alignment_analysis,
                'preprocessing_analysis': preprocessing_analysis
            }
        
        # 生成根因分析
        self._generate_root_cause_conclusions(analysis_results)
        
        return analysis_results
    
    def _generate_root_cause_conclusions(self, analysis_results: dict):
        """生成根因分析结论"""
        
        print(f"\n🏆 根因分析结论:")
        print("=" * 80)
        
        root_causes = []
        recommendations = []
        
        # 分析EMA计算方法差异
        print(f"1. EMA计算方法分析:")
        for stock_code, results in analysis_results['test_cases'].items():
            ema_analysis = results.get('ema_analysis', {})
            if 'ema_methods' in ema_analysis:
                methods = ema_analysis['ema_methods']
                system_diff = methods['system']['diff']
                
                # 找到最接近的方法
                closest_method = None
                min_diff = float('inf')
                
                for method_name, method_data in methods.items():
                    if method_name != 'system' and method_data['diff'] is not None:
                        diff = abs(method_data['diff'] - system_diff)
                        if diff < min_diff:
                            min_diff = diff
                            closest_method = method_name
                
                print(f"  {stock_code}: 系统计算与{closest_method}最接近 (差异: {min_diff:.6f})")
        
        # 分析日期对齐问题
        print(f"\n2. 日期对齐问题分析:")
        for stock_code, results in analysis_results['test_cases'].items():
            alignment_analysis = results.get('alignment_analysis', {})
            if 'alignment_results' in alignment_analysis:
                alignment_results_data = alignment_analysis['alignment_results']
                
                # 检查不同数据长度的一致性
                valid_results = {k: v for k, v in alignment_results_data.items() if 'error' not in v}
                
                if len(valid_results) > 1:
                    diff_values = [v['macd_values']['diff'] for v in valid_results.values()]
                    diff_std = np.std(diff_values)
                    
                    if diff_std > 0.001:
                        print(f"  {stock_code}: 不同数据长度导致DIFF差异 (标准差: {diff_std:.6f})")
                        root_causes.append(f"{stock_code}存在数据长度相关的计算差异")
                    else:
                        print(f"  {stock_code}: 不同数据长度计算一致 ✅")
        
        # 分析数据质量问题
        print(f"\n3. 数据质量问题分析:")
        for stock_code, results in analysis_results['test_cases'].items():
            preprocessing_analysis = results.get('preprocessing_analysis', {})
            if 'data_quality' in preprocessing_analysis:
                quality = preprocessing_analysis['data_quality']
                
                issues = []
                if quality['null_values'] > 0:
                    issues.append(f"{quality['null_values']}个空值")
                if quality['duplicate_dates'] > 0:
                    issues.append(f"{quality['duplicate_dates']}个重复日期")
                if quality['abnormal_prices'] > 0:
                    issues.append(f"{quality['abnormal_prices']}个异常价格")
                if quality['large_changes'] > 5:
                    issues.append(f"{quality['large_changes']}个异常波动")
                
                if issues:
                    print(f"  {stock_code}: 发现数据质量问题 - {', '.join(issues)}")
                    root_causes.append(f"{stock_code}存在数据质量问题: {', '.join(issues)}")
                else:
                    print(f"  {stock_code}: 数据质量良好 ✅")
        
        # 生成建议
        print(f"\n💡 改进建议:")
        
        if any("数据长度" in cause for cause in root_causes):
            recommendations.append("统一使用固定长度的历史数据（如200天）进行MACD计算")
            print(f"  1. 统一数据长度标准")
        
        if any("数据质量" in cause for cause in root_causes):
            recommendations.append("加强数据质量检查和清洗机制")
            print(f"  2. 加强数据质量控制")
        
        recommendations.extend([
            "验证EMA初始化方法的一致性",
            "建立MACD计算结果的基准测试",
            "实施多重验证机制确保计算准确性"
        ])
        
        print(f"  3. 验证EMA计算方法")
        print(f"  4. 建立基准测试体系")
        print(f"  5. 实施多重验证机制")
        
        analysis_results['root_causes'] = root_causes
        analysis_results['recommendations'] = recommendations

def main():
    """主函数"""
    print("🔍 深入分析MACD计算错误的根本原因")
    print("基于000017股票的计算差异，深入挖掘系统性问题")
    
    # 创建深度分析器
    analyzer = DeepMacdAnalyzer()
    
    # 运行综合根因分析
    results = analyzer.comprehensive_root_cause_analysis()
    
    # 保存分析结果
    results_file = "validation/deep_macd_root_cause_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 深度分析结果已保存到: {results_file}")
    
    print(f"\n🎯 关键发现总结:")
    if results['root_causes']:
        for i, cause in enumerate(results['root_causes'], 1):
            print(f"  {i}. {cause}")
    else:
        print(f"  未发现明显的系统性问题，可能是外部数据源差异")
    
    print(f"\n🚀 下一步行动:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")

if __name__ == "__main__":
    main()
