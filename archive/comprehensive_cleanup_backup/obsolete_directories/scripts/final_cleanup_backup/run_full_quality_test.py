#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整指标质量测试脚本
一次性测试所有103个已验证指标的质量状态
"""

import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.unified_indicator_quality_monitor import UnifiedIndicatorQualityMonitor
from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def run_full_quality_test():
    """运行完整的质量测试"""
    logger.info("🚀 开始完整指标质量测试...")
    logger.info("=" * 100)
    
    start_time = time.time()
    
    try:
        # 创建监控器
        monitor = UnifiedIndicatorQualityMonitor()
        
        # 获取所有可用指标
        available_indicators = monitor.get_available_indicators()
        total_indicators = len(available_indicators)
        
        logger.info(f"📊 准备测试 {total_indicators} 个指标...")
        logger.info("=" * 100)
        
        # 分批测试以避免内存问题
        batch_size = 25
        all_results = {}
        all_summary = {
            'total_indicators': total_indicators,
            'passed_indicators': 0,
            'failed_indicators': 0,
            'error_indicators': 0,
            'execution_time': 0,
            'test_timestamp': datetime.now().isoformat()
        }
        
        for i in range(0, total_indicators, batch_size):
            batch_indicators = available_indicators[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_indicators + batch_size - 1) // batch_size
            
            logger.info(f"📦 批次 {batch_num}/{total_batches}: 测试 {len(batch_indicators)} 个指标")
            logger.info(f"   指标: {', '.join(batch_indicators[:5])}{'...' if len(batch_indicators) > 5 else ''}")
            
            # 运行批次测试
            batch_results = monitor.run_unified_quality_test(
                target_indicators=batch_indicators
            )
            
            # 合并结果
            all_results.update(batch_results['results'])
            all_summary['passed_indicators'] += batch_results['summary']['passed_indicators']
            all_summary['failed_indicators'] += batch_results['summary']['failed_indicators']
            all_summary['error_indicators'] += batch_results['summary']['error_indicators']
            
            logger.info(f"   ✅ 批次 {batch_num} 完成: {batch_results['summary']['passed_indicators']}/{len(batch_indicators)} 通过")
            logger.info("-" * 80)
        
        # 计算总执行时间
        all_summary['execution_time'] = time.time() - start_time
        
        # 生成最终报告
        generate_final_report(all_results, all_summary)
        
        # 显示最终结果
        logger.info("=" * 100)
        logger.info("🎉 完整指标质量测试完成！")
        logger.info(f"📊 最终结果:")
        logger.info(f"   - 总指标数: {all_summary['total_indicators']}")
        logger.info(f"   - 通过指标: {all_summary['passed_indicators']}")
        logger.info(f"   - 失败指标: {all_summary['failed_indicators']}")
        logger.info(f"   - 错误指标: {all_summary['error_indicators']}")
        logger.info(f"   - 通过率: {(all_summary['passed_indicators']/all_summary['total_indicators']*100):.1f}%")
        logger.info(f"   - 总耗时: {all_summary['execution_time']:.1f} 秒")
        
        # 返回结果
        return {
            'summary': all_summary,
            'results': all_results
        }
        
    except Exception as e:
        logger.error(f"❌ 完整测试过程中发生异常: {e}")
        raise


def generate_final_report(results: dict, summary: dict):
    """生成最终测试报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"results/full_quality_test_{timestamp}.md"
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# 完整指标质量测试报告

## 📊 测试概要

- **测试时间**: {summary['test_timestamp']}
- **测试指标数**: {summary['total_indicators']}
- **通过指标数**: {summary['passed_indicators']}
- **失败指标数**: {summary['failed_indicators']}
- **错误指标数**: {summary['error_indicators']}
- **执行时间**: {summary['execution_time']:.1f} 秒
- **通过率**: {(summary['passed_indicators']/summary['total_indicators']*100):.1f}%

## 🎯 质量分布

""")
        
        # 按状态分组统计
        status_counts = {}
        score_distribution = {'100分': 0, '95分': 0, '其他': 0}
        
        for result in results.values():
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            score = result.get('score', 0)
            if score == 100:
                score_distribution['100分'] += 1
            elif score == 95:
                score_distribution['95分'] += 1
            else:
                score_distribution['其他'] += 1
        
        for status, count in status_counts.items():
            percentage = count / summary['total_indicators'] * 100
            f.write(f"- **{status}**: {count}个 ({percentage:.1f}%)\n")
        
        f.write(f"""
## 📈 分数分布

""")
        
        for score_range, count in score_distribution.items():
            percentage = count / summary['total_indicators'] * 100
            f.write(f"- **{score_range}**: {count}个 ({percentage:.1f}%)\n")
        
        f.write(f"""
## 📋 详细结果

| 指标名称 | 状态 | 分数 | 验证脚本 | 执行时间 | 备注 |
|---------|------|------|----------|----------|------|
""")
        
        # 按状态和分数排序
        sorted_results = sorted(results.items(), 
                              key=lambda x: (x[1].get('status', 'ZZZ'), -x[1].get('score', 0)))
        
        for indicator_name, result in sorted_results:
            status = result['status']
            score = result.get('score', 0)
            exec_time = result.get('execution_time', 0)
            
            # 获取验证脚本名称
            from scripts.unified_indicator_quality_monitor import UnifiedIndicatorQualityMonitor
            monitor = UnifiedIndicatorQualityMonitor()
            script_path = monitor.validation_scripts.get(indicator_name, 'N/A')
            script_name = os.path.basename(script_path) if script_path != 'N/A' else 'N/A'
            
            status_emoji = {
                'PASSED': '🟢',
                'SUCCESS': '🟢',
                'WARNING': '🟡',
                'FAILED': '🔴',
                'ERROR': '❌',
                'TIMEOUT': '⏰'
            }
            
            emoji = status_emoji.get(status, '❓')
            
            # 备注信息
            notes = []
            if result.get('error'):
                notes.append(f"错误: {result['error'][:30]}...")
            if result.get('message'):
                notes.append(result['message'][:20])
            note_text = "; ".join(notes) if notes else "-"
            
            f.write(f"| {indicator_name} | {emoji} {status} | {score} | {script_name} | {exec_time:.2f}s | {note_text} |\n")
        
        f.write(f"""
## 🏆 优秀指标 (100分)

""")
        
        excellent_indicators = [name for name, result in results.items() if result.get('score', 0) == 100]
        if excellent_indicators:
            for indicator in sorted(excellent_indicators):
                f.write(f"- **{indicator}** 🌟\n")
        else:
            f.write("暂无100分指标\n")
        
        f.write(f"""
## ⚠️ 需要关注的指标

""")
        
        # 列出失败和错误的指标
        failed_indicators = []
        error_indicators = []
        
        for indicator_name, result in results.items():
            if result['status'] in ['FAILED', 'WARNING']:
                failed_indicators.append((indicator_name, result))
            elif result['status'] in ['ERROR', 'TIMEOUT']:
                error_indicators.append((indicator_name, result))
        
        if failed_indicators:
            f.write("### 🔴 失败指标\n\n")
            for indicator_name, result in failed_indicators:
                f.write(f"- **{indicator_name}**: 分数 {result.get('score', 0)}\n")
                if result.get('error'):
                    f.write(f"  - 错误: {result['error']}\n")
        
        if error_indicators:
            f.write("### ❌ 错误指标\n\n")
            for indicator_name, result in error_indicators:
                f.write(f"- **{indicator_name}**: {result.get('error', '未知错误')}\n")
        
        f.write(f"""
## 📈 系统健康度评估

### 整体质量评级
""")
        
        pass_rate = summary['passed_indicators'] / summary['total_indicators'] * 100
        
        if pass_rate >= 95:
            f.write("🟢 **优秀** - 系统质量优秀，可直接投入生产使用\n")
        elif pass_rate >= 90:
            f.write("🟡 **良好** - 系统质量良好，少量优化后可投入使用\n")
        elif pass_rate >= 80:
            f.write("🟠 **一般** - 系统质量一般，需要重点优化失败指标\n")
        else:
            f.write("🔴 **较差** - 系统质量较差，需要全面检查和修复\n")
        
        f.write(f"""
### 建议措施
1. 对于失败指标，优先检查验证脚本的兼容性
2. 对于错误指标，确认验证脚本的依赖和路径
3. 定期运行此完整测试，监控系统质量变化
4. 建立质量基线，确保新增指标达到相同标准

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**测试覆盖率**: 100% (103个已验证指标)  
**测试方式**: 统一调用现有验证脚本
""")
    
    logger.info(f"📄 完整测试报告已生成: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='完整指标质量测试')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式（仅测试前30个指标）')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            logger.info("🏃 快速测试模式")
            monitor = UnifiedIndicatorQualityMonitor()
            results = monitor.run_unified_quality_test(max_indicators=30)
        else:
            results = run_full_quality_test()
        
        # 返回状态码
        summary = results['summary']
        if summary['failed_indicators'] + summary['error_indicators'] == 0:
            return 0  # 全部通过
        elif summary['passed_indicators'] / summary['total_indicators'] >= 0.9:
            return 1  # 90%以上通过，可接受
        else:
            return 2  # 通过率过低，需要关注
            
    except Exception as e:
        logger.error(f"❌ 完整测试过程中发生异常: {e}")
        return 3  # 异常


if __name__ == "__main__":
    exit(main())
