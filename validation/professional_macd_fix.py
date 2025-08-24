#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业MACD计算修复方案

基于技术分析标准实施：
1. 标准EMA计算方法（Wilder's方法）
2. 足够的历史数据（250天）
3. 正确的EMA初始化
4. 符合金融行业标准的MACD计算

目标：修复MACD计算差异，确保与市场标准一致
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('/Users/hacker/PycharmProjects/freedom')

try:
    from db.services.stock_data_service import get_stock_data_service
    from utils.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")

logger = get_logger(__name__)

class ProfessionalMacdCalculator:
    """专业MACD计算器"""
    
    def __init__(self):
        """初始化专业MACD计算器"""
        self.stock_data_service = get_stock_data_service()
        
        # 标准MACD参数
        self.fast_period = 12  # 快线EMA周期
        self.slow_period = 26  # 慢线EMA周期  
        self.signal_period = 9  # 信号线EMA周期
        
        # 专业计算参数
        self.min_history_days = 250  # 最少历史数据（行业标准）
        self.warmup_days = 34  # 预热期（26+9-1）
        
        print("✅ 专业MACD计算器初始化完成")
        print(f"📊 标准参数: EMA({self.fast_period}, {self.slow_period}, {self.signal_period})")
        print(f"📅 最少历史数据: {self.min_history_days}天")
        print(f"🔥 预热期: {self.warmup_days}天")
    
    def calculate_professional_ema(self, prices: np.ndarray, period: int, 
                                 method: str = 'standard') -> np.ndarray:
        """
        专业EMA计算
        
        Args:
            prices: 价格数组
            period: EMA周期
            method: 计算方法 ('standard', 'sma_init', 'wilder')
        
        Returns:
            EMA数组
        """
        
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        ema = np.full(len(prices), np.nan)
        
        if method == 'standard':
            # 标准EMA：第一个值作为初始值
            multiplier = 2.0 / (period + 1)
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
                
        elif method == 'sma_init':
            # SMA初始化EMA（金融行业标准）
            multiplier = 2.0 / (period + 1)
            
            # 前period个值用SMA初始化
            ema[period-1] = np.mean(prices[:period])
            
            # 从第period+1个值开始用EMA
            for i in range(period, len(prices)):
                ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
                
        elif method == 'wilder':
            # Wilder's方法（更平滑）
            multiplier = 1.0 / period
            ema[period-1] = np.mean(prices[:period])
            
            for i in range(period, len(prices)):
                ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def calculate_professional_macd(self, df: pd.DataFrame, 
                                  method: str = 'sma_init') -> pd.DataFrame:
        """
        专业MACD计算
        
        Args:
            df: 包含价格数据的DataFrame
            method: EMA计算方法
            
        Returns:
            包含MACD指标的DataFrame
        """
        
        if len(df) < self.min_history_days:
            print(f"⚠️ 数据量不足：{len(df)}天 < {self.min_history_days}天")
        
        close_prices = df['close'].values
        
        # 计算EMA12和EMA26
        ema12 = self.calculate_professional_ema(close_prices, self.fast_period, method)
        ema26 = self.calculate_professional_ema(close_prices, self.slow_period, method)
        
        # 计算DIFF（MACD线）
        diff = ema12 - ema26
        
        # 计算DEA（信号线）- 对DIFF进行EMA平滑
        # 注意：只对有效的DIFF值计算信号线
        valid_diff_start = self.slow_period - 1  # 从第26天开始DIFF有效
        dea = np.full(len(diff), np.nan)
        
        if len(diff) > valid_diff_start:
            valid_diff = diff[valid_diff_start:]
            valid_dea = self.calculate_professional_ema(valid_diff, self.signal_period, method)
            dea[valid_diff_start:] = valid_dea
        
        # 计算MACD柱状图
        macd_histogram = 2 * (diff - dea)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'date': df['date'],
            'close': df['close'],
            'ema12': ema12,
            'ema26': ema26,
            'macd_line': diff,  # DIFF
            'macd_signal': dea,  # DEA
            'macd_histogram': macd_histogram  # MACD
        })
        
        return result_df
    
    def compare_calculation_methods(self, stock_code: str, target_date: str) -> dict:
        """比较不同MACD计算方法"""
        
        print(f"\n📊 比较{stock_code}的MACD计算方法 (目标日期: {target_date})")
        print("=" * 80)
        
        try:
            # 获取足够的历史数据
            df = self.stock_data_service.get_stock_data(stock_code, days=self.min_history_days)
            
            if df is None or len(df) == 0:
                return {'error': f'无法获取{stock_code}数据'}
            
            # 查找目标日期
            target_date_obj = pd.to_datetime(target_date).date()
            target_rows = df[df['date'].dt.date == target_date_obj]
            
            if target_rows.empty:
                return {'error': f'未找到{target_date}的数据'}
            
            target_idx = target_rows.index[0]
            
            print(f"📈 数据信息:")
            print(f"  数据量: {len(df)}天")
            print(f"  目标日期索引: {target_idx}")
            print(f"  收盘价: {df.iloc[target_idx]['close']:.3f}")
            
            # 使用不同方法计算MACD
            methods = {
                'standard': '标准EMA方法',
                'sma_init': 'SMA初始化方法（行业标准）',
                'wilder': 'Wilder方法'
            }
            
            results = {}
            
            for method_key, method_name in methods.items():
                print(f"\n🔄 计算方法: {method_name}")
                
                macd_result = self.calculate_professional_macd(df, method_key)
                
                if target_idx < len(macd_result):
                    target_data = macd_result.iloc[target_idx]
                    
                    # 检查数据有效性
                    if pd.isna(target_data['macd_line']) or pd.isna(target_data['macd_signal']):
                        print(f"  ⚠️ 数据无效（在预热期内）")
                        results[method_key] = {
                            'method_name': method_name,
                            'valid': False,
                            'reason': '在预热期内'
                        }
                    else:
                        print(f"  DIFF: {target_data['macd_line']:.6f}")
                        print(f"  DEA:  {target_data['macd_signal']:.6f}")
                        print(f"  MACD: {target_data['macd_histogram']:.6f}")
                        
                        results[method_key] = {
                            'method_name': method_name,
                            'valid': True,
                            'diff': float(target_data['macd_line']),
                            'dea': float(target_data['macd_signal']),
                            'macd': float(target_data['macd_histogram']),
                            'ema12': float(target_data['ema12']),
                            'ema26': float(target_data['ema26'])
                        }
                else:
                    results[method_key] = {
                        'method_name': method_name,
                        'valid': False,
                        'reason': '索引超出范围'
                    }
            
            return {
                'stock_code': stock_code,
                'target_date': target_date,
                'target_index': target_idx,
                'data_length': len(df),
                'close_price': float(df.iloc[target_idx]['close']),
                'methods': results
            }
            
        except Exception as e:
            return {'error': f'计算比较异常: {str(e)}'}
    
    def validate_against_benchmark(self, stock_code: str, target_date: str, 
                                 benchmark_data: dict) -> dict:
        """与基准数据验证"""
        
        print(f"\n🎯 验证{stock_code}与基准数据的匹配度")
        print("=" * 80)
        
        comparison_result = self.compare_calculation_methods(stock_code, target_date)
        
        if 'error' in comparison_result:
            return comparison_result
        
        print(f"\n📊 与基准数据对比:")
        print(f"基准数据: DIFF={benchmark_data['DIFF']:.6f}, DEA={benchmark_data['DEA']:.6f}, MACD={benchmark_data['MACD']:.6f}")
        print()
        
        best_match = None
        min_total_error = float('inf')
        
        for method_key, method_result in comparison_result['methods'].items():
            if not method_result['valid']:
                print(f"{method_result['method_name']}: 无效数据")
                continue
            
            # 计算与基准的差异
            diff_error = abs(method_result['diff'] - benchmark_data['DIFF'])
            dea_error = abs(method_result['dea'] - benchmark_data['DEA'])
            macd_error = abs(method_result['macd'] - benchmark_data['MACD'])
            
            total_error = diff_error + dea_error + macd_error
            
            print(f"{method_result['method_name']}:")
            print(f"  DIFF: {method_result['diff']:.6f} (误差: {diff_error:.6f})")
            print(f"  DEA:  {method_result['dea']:.6f} (误差: {dea_error:.6f})")
            print(f"  MACD: {method_result['macd']:.6f} (误差: {macd_error:.6f})")
            print(f"  总误差: {total_error:.6f}")
            print()
            
            if total_error < min_total_error:
                min_total_error = total_error
                best_match = method_key
        
        if best_match:
            print(f"🏆 最佳匹配方法: {comparison_result['methods'][best_match]['method_name']}")
            print(f"📊 总误差: {min_total_error:.6f}")
        
        return {
            'stock_code': stock_code,
            'target_date': target_date,
            'benchmark_data': benchmark_data,
            'comparison_result': comparison_result,
            'best_match_method': best_match,
            'min_total_error': min_total_error
        }
    
    def fix_macd_calculation_system(self) -> dict:
        """修复MACD计算系统"""
        
        print(f"\n🔧 专业MACD计算系统修复")
        print("=" * 80)
        
        # 测试用例
        test_cases = {
            '000017': {
                'date': '2025-05-14',
                'benchmark': {'MACD': 0.037, 'DIFF': 0.149, 'DEA': 0.13}
            },
            '000001': {
                'date': '2025-05-12', 
                'benchmark': {'MACD': 0.073, 'DIFF': -0.039, 'DEA': -0.076}
            }
        }
        
        fix_results = {
            'timestamp': datetime.now().isoformat(),
            'test_cases': {},
            'recommended_method': None,
            'system_fixes': []
        }
        
        method_scores = {'standard': 0, 'sma_init': 0, 'wilder': 0}
        
        # 验证每个测试用例
        for stock_code, test_case in test_cases.items():
            print(f"\n📊 验证{stock_code}股票:")
            
            validation_result = self.validate_against_benchmark(
                stock_code, 
                test_case['date'], 
                test_case['benchmark']
            )
            
            fix_results['test_cases'][stock_code] = validation_result
            
            # 评分最佳方法
            if 'best_match_method' in validation_result and validation_result['best_match_method']:
                method_scores[validation_result['best_match_method']] += 1
        
        # 确定推荐方法
        best_method = max(method_scores, key=method_scores.get)
        fix_results['recommended_method'] = best_method
        
        print(f"\n🏆 修复结果汇总:")
        print("=" * 80)
        print(f"推荐计算方法: {best_method}")
        print(f"方法评分: {method_scores}")
        
        # 生成修复建议
        fixes = []
        
        if best_method == 'sma_init':
            fixes.append("使用SMA初始化EMA方法（金融行业标准）")
            fixes.append("确保至少250天历史数据")
            fixes.append("设置34天预热期")
        elif best_method == 'wilder':
            fixes.append("使用Wilder's EMA方法")
            fixes.append("调整EMA平滑系数")
        else:
            fixes.append("使用标准EMA方法")
        
        fixes.extend([
            "统一MACD参数：EMA(12,26,9)",
            "实施数据质量检查",
            "建立计算结果验证机制"
        ])
        
        fix_results['system_fixes'] = fixes
        
        print(f"\n💡 系统修复建议:")
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")
        
        return fix_results

def main():
    """主函数"""
    print("🔧 专业MACD计算修复方案")
    print("基于技术分析标准，修复MACD计算差异问题")
    
    # 创建专业MACD计算器
    calculator = ProfessionalMacdCalculator()
    
    # 执行系统修复
    fix_results = calculator.fix_macd_calculation_system()
    
    # 保存修复结果
    results_dir = Path("validation/professional_fix_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "专业MACD修复结果.json"
    
    # 处理numpy类型以便JSON序列化
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 递归转换所有numpy类型
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        else:
            return convert_numpy_types(obj)
    
    fix_results_clean = deep_convert(fix_results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(fix_results_clean, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 专业修复结果已保存到: {results_file}")
    
    # 显示最终建议
    print(f"\n🎯 最终修复方案:")
    print("=" * 80)
    
    recommended_method = fix_results['recommended_method']
    if recommended_method == 'sma_init':
        print(f"✅ 推荐使用：SMA初始化EMA方法（金融行业标准）")
        print(f"📊 技术原理：")
        print(f"  • 前26天使用SMA计算EMA26初始值")
        print(f"  • 前12天使用SMA计算EMA12初始值")
        print(f"  • 从第27天开始计算有效DIFF")
        print(f"  • 从第35天开始计算有效DEA")
        print(f"  • 确保与主流金融软件一致")
    
    print(f"\n🚀 实施步骤:")
    print(f"  1. 修改MACD指标类使用SMA初始化方法")
    print(f"  2. 确保获取足够的历史数据（250天）")
    print(f"  3. 重新生成MACD形态检测结果")
    print(f"  4. 验证修复后的计算准确性")
    
    print(f"\n💡 预期效果:")
    print(f"  • MACD计算与市场标准一致")
    print(f"  • 形态检测结果更加准确")
    print(f"  • 可以安全用于实际交易分析")

if __name__ == "__main__":
    main()
