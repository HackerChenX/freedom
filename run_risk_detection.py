#!/usr/bin/env python3
"""
运行自动化风险检测并生成报告
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from tools.automated_risk_detection import AutomatedRiskDetector, RiskLevel
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """主函数"""
    print("🔍 开始自动化风险检测...")
    
    # 创建风险检测器
    detector = AutomatedRiskDetector()
    
    # 扫描所有指标
    results = detector.scan_all_indicators()
    
    if not results:
        print("❌ 未能获取检测结果")
        return
    
    # 统计信息
    total_count = len(results)
    high_risk_count = sum(1 for r in results.values() if r.risk_level == RiskLevel.HIGH)
    medium_risk_count = sum(1 for r in results.values() if r.risk_level == RiskLevel.MEDIUM)
    low_risk_count = sum(1 for r in results.values() if r.risk_level == RiskLevel.LOW)
    
    print(f"\n📊 检测结果统计:")
    print(f"   总指标数: {total_count}")
    print(f"   高风险指标: {high_risk_count}")
    print(f"   中风险指标: {medium_risk_count}")
    print(f"   低风险指标: {low_risk_count}")
    
    # 显示高风险指标详情
    if high_risk_count > 0:
        print(f"\n🚨 高风险指标详情 ({high_risk_count}个):")
        print("=" * 60)
        
        high_risk_indicators = []
        for name, result in results.items():
            if result.risk_level == RiskLevel.HIGH:
                high_risk_indicators.append(name)
                print(f"\n【{name}】")
                print(f"  指标类型: {result.indicator_type.value}")
                print(f"  信号一致性评分: {result.signal_consistency_score:.1f}")
                print(f"  风险因素: {', '.join(result.risk_factors) if result.risk_factors else '无'}")
                print(f"  修复建议: {'; '.join(result.recommendations) if result.recommendations else '无'}")
        
        print(f"\n📋 高风险指标清单:")
        for i, indicator in enumerate(high_risk_indicators, 1):
            print(f"  {i:2d}. {indicator}")
            
        # 按优先级分类
        p0_indicators = []
        p1_indicators = []
        p2_indicators = []
        p3_indicators = []
        
        for indicator in high_risk_indicators:
            if any(keyword in indicator.upper() for keyword in ['ICHIMOKU', 'STOCHRSI']):
                p0_indicators.append(indicator)
            elif any(keyword in indicator.upper() for keyword in ['ENHANCED_RSI', 'ENHANCED_STOCHRSI', 'ENHANCED_WR', 'ENHANCED_MACD_ROOT']):
                p1_indicators.append(indicator)
            elif any(keyword in indicator.upper() for keyword in ['COMPOSITE', 'UNIFIED_MA', 'CHIP_DISTRIBUTION', 'INSTITUTIONAL_BEHAVIOR', 'STOCK_VIX']):
                p2_indicators.append(indicator)
            elif any(keyword in indicator.upper() for keyword in ['FIBONACCI_TOOLS', 'GANN_TOOLS', 'ELLIOTT_WAVE']):
                p3_indicators.append(indicator)
            else:
                p3_indicators.append(indicator)  # 默认归为P3
        
        print(f"\n🎯 按修复优先级分类:")
        if p0_indicators:
            print(f"  P0级核心技术指标 ({len(p0_indicators)}个): {', '.join(p0_indicators)}")
        if p1_indicators:
            print(f"  P1级增强指标 ({len(p1_indicators)}个): {', '.join(p1_indicators)}")
        if p2_indicators:
            print(f"  P2级复合指标 ({len(p2_indicators)}个): {', '.join(p2_indicators)}")
        if p3_indicators:
            print(f"  P3级工具指标 ({len(p3_indicators)}个): {', '.join(p3_indicators)}")
    
    else:
        print("\n✅ 太好了！没有发现高风险指标")
    
    # 生成完整报告
    report = detector.generate_risk_report()
    report_file = "risk_detection_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 详细报告已保存到: {report_file}")
    print("\n🔍 风险检测完成！")

if __name__ == "__main__":
    main() 