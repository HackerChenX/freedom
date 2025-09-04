#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查所有已注册指标脚本
分析当前系统中实际注册的所有指标，与验证进度表进行对比
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Set, Any

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


class RegisteredIndicatorsChecker:
    """已注册指标检查器"""
    
    def __init__(self):
        self.registered_indicators = set()
        self.failed_indicators = set()
        self.indicator_details = {}
        
        # 验证进度表中声称已完成的63个指标
        self.claimed_completed_indicators = {
            # BaseIndicator指标 (9个)
            'ADX', 'ROC', 'MFI', 'OBV', 'KC', 'VIX', 'MTM', 'SYNERGY', 'UNIFIED_MA',
            
            # ZXM体系指标 (35个)
            'ZXM_DAILY_MACD', 'ZXM_WEEKLY_MACD', 'ZXM_MONTHLY_MACD', 'ZXM_DAILY_KDJ', 'ZXM_WEEKLY_KDJ',
            'ZXM_DAILY_RSI', 'ZXM_WEEKLY_RSI', 'ZXM_DAILY_BOLL', 'ZXM_WEEKLY_BOLL', 'ZXM_DAILY_MA',
            'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_VOLUME_BREAKOUT', 'ZXM_VOLUME_PRICE_TREND',
            'ZXM_BUYPOINT_SCORE', 'ZXM_TREND_SCORE', 'ZXM_COMPREHENSIVE_SCORE',
            'ZXM_PRICE_POSITION', 'ZXM_TREND_STRENGTH', 'ZXM_SUPPORT_RESISTANCE', 'ZXM_BREAKOUT_SIGNAL',
            'ZXM_HOT_SPOT', 'ZXM_SECTOR_ROTATION',
            'ZXM_RISK_CONTROL', 'ZXM_POSITION_SIZING', 'ZXM_TIMING_SIGNAL', 'ZXM_STOP_LOSS',
            'ZXM_PORTFOLIO_OPTIMIZATION', 'ZXM_STRATEGY_COMBINATION', 'ZXM_PERFORMANCE_ATTRIBUTION',
            'ZXM_ALPHA_GENERATION', 'ZXM_BETA_HEDGING',
            'ZXM_MARKET_SENTIMENT', 'ZXM_LIQUIDITY_ANALYSIS', 'ZXM_VOLATILITY_FORECAST', 'ZXM_CORRELATION_MATRIX',
            
            # 形态识别指标 (19个)
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
            'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
            'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
            'WEDGE', 'FLAG', 'PENNANT'
        }
        
    def get_all_registered_indicators(self) -> Dict[str, Any]:
        """获取所有已注册的指标"""
        logger.info("🔍 检查所有已注册指标...")
        
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            
            # 获取所有已注册的指标
            all_indicators = registry.get_all_indicators()
            
            logger.info(f"📊 注册表中共有 {len(all_indicators)} 个指标")
            
            return all_indicators
            
        except Exception as e:
            logger.error(f"❌ 获取注册表失败: {e}")
            return {}
    
    def test_indicator_creation(self, indicator_name: str) -> Dict[str, Any]:
        """测试指标创建"""
        result = {
            'name': indicator_name,
            'can_create': False,
            'has_calculate': False,
            'has_get_patterns': False,
            'error': None
        }
        
        try:
            from indicators.complete_indicator_registry import get_indicator_registry
            registry = get_indicator_registry()
            
            # 尝试创建指标实例
            indicator = registry.create_indicator(indicator_name)
            
            if indicator is not None:
                result['can_create'] = True
                
                # 检查是否有calculate方法
                if hasattr(indicator, 'calculate'):
                    result['has_calculate'] = True
                
                # 检查是否有get_patterns方法
                if hasattr(indicator, 'get_patterns'):
                    result['has_get_patterns'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def analyze_all_indicators(self) -> Dict[str, Any]:
        """分析所有指标的状态"""
        logger.info("🚀 开始分析所有指标状态...")
        
        start_time = time.time()
        
        # 获取所有已注册的指标
        all_registered = self.get_all_registered_indicators()
        
        # 分析每个指标
        analysis_results = {}
        
        for indicator_name in all_registered.keys():
            logger.info(f"📦 分析指标: {indicator_name}")
            
            test_result = self.test_indicator_creation(indicator_name)
            analysis_results[indicator_name] = test_result
            
            if test_result['can_create']:
                self.registered_indicators.add(indicator_name)
            else:
                self.failed_indicators.add(indicator_name)
        
        analysis_time = time.time() - start_time
        
        # 分类统计
        baseindicator_indicators = set()
        zxm_indicators = set()
        pattern_indicators = set()
        other_indicators = set()
        
        for name in self.registered_indicators:
            if name.startswith('ZXM_'):
                zxm_indicators.add(name)
            elif name in ['DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'HARAMI', 'PIERCING_LINE',
                         'DARK_CLOUD_COVER', 'MORNING_STAR', 'EVENING_STAR', 'THREE_BLACK_CROWS', 'THREE_WHITE_SOLDIERS',
                         'V_SHAPED_REVERSAL', 'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'TRIANGLE',
                         'WEDGE', 'FLAG', 'PENNANT']:
                pattern_indicators.add(name)
            elif name in ['ADX', 'ROC', 'MFI', 'OBV', 'KC', 'VIX', 'MTM', 'SYNERGY', 'UNIFIED_MA']:
                baseindicator_indicators.add(name)
            else:
                other_indicators.add(name)
        
        # 与声称完成的指标进行对比
        actually_registered = self.registered_indicators
        claimed_completed = self.claimed_completed_indicators
        
        verified_indicators = actually_registered & claimed_completed
        missing_indicators = claimed_completed - actually_registered
        extra_indicators = actually_registered - claimed_completed
        
        summary = {
            'analysis_type': 'ALL_REGISTERED_INDICATORS_CHECK',
            'total_registered': len(self.registered_indicators),
            'total_failed': len(self.failed_indicators),
            'analysis_time': analysis_time,
            'categories': {
                'baseindicator': list(baseindicator_indicators),
                'zxm': list(zxm_indicators),
                'pattern': list(pattern_indicators),
                'other': list(other_indicators)
            },
            'verification_status': {
                'claimed_completed_count': len(claimed_completed),
                'actually_registered_count': len(actually_registered),
                'verified_count': len(verified_indicators),
                'missing_count': len(missing_indicators),
                'extra_count': len(extra_indicators)
            },
            'verified_indicators': list(verified_indicators),
            'missing_indicators': list(missing_indicators),
            'extra_indicators': list(extra_indicators),
            'failed_indicators': list(self.failed_indicators),
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"🎯 指标分析完成!")
        logger.info(f"📊 总注册指标: {len(self.registered_indicators)}个")
        logger.info(f"📊 声称完成: {len(claimed_completed)}个")
        logger.info(f"📊 实际验证: {len(verified_indicators)}个")
        logger.info(f"📊 缺失指标: {len(missing_indicators)}个")
        logger.info(f"📊 额外指标: {len(extra_indicators)}个")
        logger.info(f"⏱️ 分析时间: {analysis_time:.2f}秒")
        
        return summary
    
    def generate_detailed_report(self, analysis_result: Dict[str, Any]) -> str:
        """生成详细报告"""
        
        report = f"""# 所有已注册指标检查报告

## 检查概览
- **检查类型**: 所有已注册指标状态检查
- **检查时间**: {analysis_result['timestamp']}
- **总注册指标**: {analysis_result['total_registered']}个
- **注册失败**: {analysis_result['total_failed']}个

## 验证状态对比
- **声称完成指标**: {analysis_result['verification_status']['claimed_completed_count']}个
- **实际注册指标**: {analysis_result['verification_status']['actually_registered_count']}个
- **验证通过指标**: {analysis_result['verification_status']['verified_count']}个
- **缺失指标**: {analysis_result['verification_status']['missing_count']}个
- **额外指标**: {analysis_result['verification_status']['extra_count']}个

## 指标分类统计

### BaseIndicator指标 ({len(analysis_result['categories']['baseindicator'])}个)
{chr(10).join([f"- {name}" for name in sorted(analysis_result['categories']['baseindicator'])])}

### ZXM体系指标 ({len(analysis_result['categories']['zxm'])}个)
{chr(10).join([f"- {name}" for name in sorted(analysis_result['categories']['zxm'])])}

### 形态识别指标 ({len(analysis_result['categories']['pattern'])}个)
{chr(10).join([f"- {name}" for name in sorted(analysis_result['categories']['pattern'])])}

### 其他指标 ({len(analysis_result['categories']['other'])}个)
{chr(10).join([f"- {name}" for name in sorted(analysis_result['categories']['other'])])}

## 验证结果详情

### ✅ 验证通过的指标 ({len(analysis_result['verified_indicators'])}个)
{chr(10).join([f"- **{name}**: 已注册且可创建" for name in sorted(analysis_result['verified_indicators'])])}

### ❌ 缺失的指标 ({len(analysis_result['missing_indicators'])}个)
{chr(10).join([f"- **{name}**: 声称完成但未注册" for name in sorted(analysis_result['missing_indicators'])])}

### 🔍 额外的指标 ({len(analysis_result['extra_indicators'])}个)
{chr(10).join([f"- **{name}**: 已注册但未在完成列表中" for name in sorted(analysis_result['extra_indicators'])])}

### ⚠️ 注册失败的指标 ({len(analysis_result['failed_indicators'])}个)
{chr(10).join([f"- **{name}**: 注册失败或无法创建" for name in sorted(analysis_result['failed_indicators'])])}

## 问题分析

### 主要问题
"""
        
        if analysis_result['verification_status']['missing_count'] > 0:
            report += f"""
1. **缺失指标问题**: 有{analysis_result['verification_status']['missing_count']}个指标声称已完成但实际未注册
   - 这表明验证进度表与实际系统状态不符
   - 需要重新检查这些指标的实际状态
"""
        
        if analysis_result['verification_status']['extra_count'] > 0:
            report += f"""
2. **额外指标发现**: 发现{analysis_result['verification_status']['extra_count']}个未在完成列表中的已注册指标
   - 这些指标可能是系统中的其他指标
   - 需要评估是否应该包含在验证范围内
"""
        
        if analysis_result['total_failed'] > 0:
            report += f"""
3. **注册失败问题**: 有{analysis_result['total_failed']}个指标注册失败
   - 这些指标可能存在实现问题
   - 需要检查指标的实现和注册路径
"""
        
        verification_rate = (analysis_result['verification_status']['verified_count'] / 
                           analysis_result['verification_status']['claimed_completed_count'] * 100 
                           if analysis_result['verification_status']['claimed_completed_count'] > 0 else 0)
        
        report += f"""

## 验证结论

### 验证通过率: {verification_rate:.1f}%

{'### 🎉 验证基本通过' if verification_rate >= 80 else '### ⚠️ 验证存在问题'}

{'大部分声称完成的指标都能正常注册和创建。' if verification_rate >= 80 else '存在较多指标注册或创建问题，需要进一步检查。'}

### 建议行动
1. **优先处理缺失指标**: 检查缺失指标的实际实现状态
2. **验证额外指标**: 评估额外指标是否应该纳入验证范围
3. **修复失败指标**: 解决注册失败的指标问题
4. **更新进度表**: 根据实际情况更新验证进度表

---
*检查时间: {analysis_result['analysis_time']:.2f}秒*
*检查工具: 已注册指标检查系统*
"""
        
        return report


def main():
    """主函数"""
    logger.info("🔍 开始检查所有已注册指标...")
    
    checker = RegisteredIndicatorsChecker()
    result = checker.analyze_all_indicators()
    
    # 生成详细报告
    report_content = checker.generate_detailed_report(result)
    
    # 保存报告
    report_file = f"docs/finaltesting/indicators/all_registered_indicators_check_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"📄 检查报告已保存: {report_file}")
    
    # 输出关键信息
    print(f"\n=== 关键发现 ===")
    print(f"声称完成指标: {result['verification_status']['claimed_completed_count']}个")
    print(f"实际注册指标: {result['verification_status']['actually_registered_count']}个")
    print(f"验证通过指标: {result['verification_status']['verified_count']}个")
    print(f"缺失指标: {result['verification_status']['missing_count']}个")
    print(f"额外指标: {result['verification_status']['extra_count']}个")
    
    if result['missing_indicators']:
        print(f"\n=== 缺失的指标 ===")
        for indicator in sorted(result['missing_indicators']):
            print(f"- {indicator}")
    
    if result['extra_indicators']:
        print(f"\n=== 额外的指标 ===")
        for indicator in sorted(result['extra_indicators']):
            print(f"- {indicator}")
    
    return result


if __name__ == "__main__":
    main()
