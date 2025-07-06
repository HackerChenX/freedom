#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
批量演示测试脚本

对所有核心指标进行反向验证测试，生成综合报告
"""

import sys
import os
import subprocess
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def run_single_indicator_test(indicator: str) -> dict:
    """运行单个指标测试"""
    print(f"正在测试指标: {indicator}")

    # 构建命令
    cmd = [
        sys.executable,
        os.path.join(current_dir, 'demo_single_indicator.py'),
        indicator,
        '--save-results'
    ]

    try:
        # 运行测试
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        # 查找生成的结果文件
        result_files = [f for f in os.listdir('.') if f.startswith(f'demo_validation_results_{indicator}_')]

        if result_files:
            # 读取最新的结果文件
            latest_file = max(result_files, key=lambda x: os.path.getctime(x))
            with open(latest_file, 'r', encoding='utf-8') as f:
                test_result = json.load(f)

            # 添加执行信息
            test_result['execution_info'] = {
                'return_code': result.returncode,
                'stdout_lines': len(result.stdout.split('\n')),
                'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0,
                'result_file': latest_file
            }

            return test_result
        else:
            return {
                'indicator': indicator,
                'error': 'No result file generated',
                'execution_info': {
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            }

    except Exception as e:
        return {
            'indicator': indicator,
            'error': str(e),
            'execution_info': {'exception': True}
        }


def generate_comprehensive_report_Batch_Demo_Test(all_results: dict) -> str:
    """生成综合报告"""
    report_lines = []

    # 报告标题
    report_lines.append("# 选股系统反向验证测试综合报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**测试模式**: 演示模式（简化验证）")
    report_lines.append("")

    # 总体统计
    total_indicators = len(all_results)
    successful_indicators = sum(1 for r in all_results.values() if r.get('success_rate', 0) >= 0.6)
    total_patterns = sum(r.get('total_patterns', 0) for r in all_results.values())
    total_successful_patterns = sum(r.get('successful_patterns', 0) for r in all_results.values())

    overall_success_rate = total_successful_patterns / total_patterns if total_patterns > 0 else 0

    report_lines.append("## 总体统计")
    report_lines.append("")
    report_lines.append(f"- **测试指标数**: {total_indicators}")
    report_lines.append(f"- **表现良好指标数**: {successful_indicators}")
    report_lines.append(f"- **总形态数**: {total_patterns}")
    report_lines.append(f"- **成功识别形态数**: {total_successful_patterns}")
    report_lines.append(f"- **整体成功率**: {overall_success_rate:.2%}")
    report_lines.append("")

    # 指标排名
    valid_results = [(name, result) for name, result in all_results.items()
                    if 'success_rate' in result and 'error' not in result]
    valid_results.sort(key=lambda x: x[1]['success_rate'], reverse=True)

    if valid_results:
        report_lines.append("## 指标表现排名")
        report_lines.append("")
        report_lines.append("| 排名 | 指标 | 成功率 | 平均匹配分 | 成功形态数 | 总形态数 |")
        report_lines.append("|------|------|--------|------------|------------|----------|")

        for i, (indicator, result) in enumerate(valid_results, 1):
            success_rate = f"{result['success_rate']:.2%}"
            avg_score = f"{result.get('average_score', 0):.3f}"
            successful = result.get('successful_patterns', 0)
            total = result.get('total_patterns', 0)

            report_lines.append(f"| {i} | {indicator} | {success_rate} | {avg_score} | {successful} | {total} |")

        report_lines.append("")

    return "\n".join(report_lines)


def main_batchdemotest():
    """主函数"""
    print("=" * 60)
    print("批量反向验证测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 要测试的指标列表
    indicators = ['RSI', 'MACD', 'KDJ', 'BOLL', 'MA', 'EMA']

    all_results = {}

    # 逐个测试指标
    for indicator in indicators:
        try:
            result = run_single_indicator_test(indicator)
            all_results[indicator] = result

            # 显示简要结果
            if 'error' in result:
                print(f"  ❌ {indicator}: 测试失败 - {result['error']}")
            else:
                success_rate = result.get('success_rate', 0)
                status = "✅" if success_rate >= 0.6 else "⚠️" if success_rate >= 0.4 else "❌"
                print(f"  {status} {indicator}: 成功率 {success_rate:.2%}")

        except Exception as e:
            print(f"  ❌ {indicator}: 异常 - {e}")
            all_results[indicator] = {'error': str(e)}

    print()
    print("=" * 60)
    print("生成综合报告...")

    # 生成报告
    report_content = generate_comprehensive_report_Batch_Demo_Test(all_results)

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comprehensive_validation_report_{timestamp}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # 保存原始数据
    json_file = f"comprehensive_validation_data_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"报告已保存到: {report_file}")
    print(f"原始数据已保存到: {json_file}")
    print()

    # 显示简要总结
    total_indicators = len(all_results)
    successful_indicators = sum(1 for r in all_results.values() if r.get('success_rate', 0) >= 0.6)

    print("测试总结:")
    print(f"- 测试指标数: {total_indicators}")
    print(f"- 表现良好: {successful_indicators}")
    print(f"- 表现良好率: {successful_indicators/total_indicators:.2%}")

    return 0 if successful_indicators >= total_indicators * 0.5 else 1


if __name__ == '__main__':
    exit_code = main_batchdemotest()
    sys.exit(exit_code)