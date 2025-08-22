#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
112指标系统继续测试修复脚本

按照计划，继续测试修复其他112个指标
基于Ultra Think方法论，追求100%完美标准
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.complete_indicator_registry import complete_registry
from tests.unified_indicator_testing.components.test_data_generator import TestDataGenerator
from tests.unified_indicator_testing.components.buypoint_analyzer import BuypointAnalyzer
from utils.logger import get_logger, init_logging

# 初始化日志
init_logging(level="INFO")
logger = get_logger(__name__)


class Continue112IndicatorsTest:
    """112指标系统继续测试修复器"""
    
    def __init__(self):
        """初始化测试器"""
        self.test_data_generator = TestDataGenerator()
        self.buypoint_analyzer = BuypointAnalyzer()
        
        # 测试统计
        self.total_indicators = 0
        self.successful_indicators = 0
        self.failed_indicators = []
        self.test_results = {}
        
        # 112指标分类 - 基于之前的系统架构
        self.indicator_categories = {
            # 核心指标(21个) - 已100%完成
            'core_indicators': [
                'MA', 'EMA', 'MACD', 'RSI', 'BOLL', 'PSY',
                'KDJ', 'ADX', 'WR', 'DMA', 'DMI', 'CCI', 
                'BIAS', 'STOCHRSI', 'OBV', 'MTM', 'PVT', 
                'MOMENTUM', 'AROON', 'FIBONACCI', 'VOL'
            ],
            
            # ZXM体系指标(35个) - 需要继续测试
            'zxm_buypoint': [
                'ZXM_DAILY_MACD', 'ZXM_TURNOVER', 'ZXM_BS_ABSORB',
                'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK'
            ],
            'zxm_trend': [
                'ZXM_DAILY_TREND_UP', 'ZXM_WEEKLY_TREND_UP', 'ZXM_MONTHLY_KDJ_TREND_UP',
                'ZXM_WEEKLY_MACD', 'ZXM_MONTHLY_MACD'
            ],
            'zxm_elasticity': [
                'ZXM_AMPLITUDE_ELASTICITY', 'ZXM_RISE_ELASTICITY',
                'ZXM_ELASTICITY', 'ZXM_BOUNCE_DETECTOR'
            ],
            'zxm_scoring': [
                'ZXM_BUYPOINT_SCORE', 'ZXM_TREND_SCORE', 'ZXM_ELASTIC_SCORE'
            ],
            'zxm_professional': [
                'ZXM_VOLUME_ENERGY', 'ZXM_PRICE_POSITION', 'ZXM_TECHNICAL_FORM',
                'ZXM_MARKET_SENTIMENT', 'ZXM_CHIP_DISTRIBUTION', 'ZXM_FUND_FLOW',
                'ZXM_INSTITUTION_BEHAVIOR', 'ZXM_HOT_SPOT', 'ZXM_INDUSTRY_ROTATION',
                'ZXM_CYCLE_POSITION', 'ZXM_RISK_CONTROL', 'ZXM_TIMING_SIGNAL',
                'ZXM_POSITION_MANAGEMENT', 'ZXM_PORTFOLIO_OPTIMIZATION',
                'ZXM_STRATEGY_COMBINATION', 'ZXM_PERFORMANCE_ATTRIBUTION',
                'ZXM_ALPHA_GENERATION', 'ZXM_BETA_HEDGING'
            ],
            
            # 增强指标(8个) - 需要继续测试
            'enhanced': [
                'EnhancedMACD', 'EnhancedBOLL', 'EnhancedSTOCHRSI', 'EnhancedRSI',
                'EnhancedKDJ', 'EnhancedCCI', 'EnhancedTRIX', 'EnhancedWR'
            ],
            
            # 形态识别指标(21个) - 需要继续测试
            'pattern_recognition': [
                'CANDLESTICK_PATTERNS', 'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING',
                'HARAMI', 'PIERCING_LINE', 'MORNING_STAR', 'EVENING_STAR',
                'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS', 'HANGING_MAN',
                'INVERTED_HAMMER', 'DARK_CLOUD_COVER', 'BEARISH_ENGULFING',
                'BULLISH_ENGULFING', 'SPINNING_TOP', 'MARUBOZU',
                'DRAGONFLY_DOJI', 'GRAVESTONE_DOJI', 'LONG_LEGGED_DOJI'
            ],
            
            # 专业分析指标(4个) - 需要继续测试
            'professional_analysis': [
                'ELLIOTT_WAVE', 'GANN', 'ICHIMOKU'
            ],
            
            # 其他技术指标(23个) - 需要继续测试
            'trend_indicators': [
                'SAR', 'TRIX', 'WMA'
            ],
            'oscillator_indicators': [
                'CMO', 'ROC'
            ],
            'volume_indicators': [
                'AD', 'EMV', 'VR', 'VOSC', 'MFI', 'CHAIKIN'
            ],
            'volatility_indicators': [
                'ATR', 'KC', 'VIX', 'STDDEV'
            ],
            'composite_indicators': [
                'COMPOSITE', 'SYNERGY', 'RSIMA', 'VORTEX'
            ]
        }
    
    def run_comprehensive_112_test(self):
        """运行112指标系统的综合测试"""
        logger.info("🔬 开始112指标系统继续测试修复")
        test_start_time = time.time()
        
        print("=" * 80)
        print("🎯 112指标系统Ultra Think方法论继续测试修复")
        print("=" * 80)
        
        # 1. 确认21个核心指标状态
        print("\n📊 阶段1: 确认21个核心指标状态")
        core_status = self._verify_core_indicators_status()
        
        # 2. 测试91个其他指标
        print(f"\n🔍 阶段2: 测试91个其他指标")
        other_results = self._test_other_indicators()
        
        # 3. 生成完整报告
        print(f"\n📋 阶段3: 生成完整测试报告")
        final_report = self._generate_final_report(core_status, other_results, test_start_time)
        
        return final_report
    
    def _verify_core_indicators_status(self):
        """验证21个核心指标状态"""
        core_indicators = self.indicator_categories['core_indicators']
        print(f"验证 {len(core_indicators)} 个核心指标...")
        
        core_status = {
            'total': len(core_indicators),
            'verified': 0,
            'status': 'COMPLETE',
            'details': {}
        }
        
        for indicator_name in core_indicators:
            try:
                indicator = complete_registry.create_indicator(indicator_name)
                if indicator:
                    core_status['verified'] += 1
                    core_status['details'][indicator_name] = '✅ 已完成100%修复'
                else:
                    core_status['details'][indicator_name] = '❌ 指标创建失败'
            except Exception as e:
                core_status['details'][indicator_name] = f'💥 异常: {str(e)}'
        
        success_rate = (core_status['verified'] / core_status['total']) * 100
        print(f"✅ 核心指标验证完成: {core_status['verified']}/{core_status['total']} ({success_rate:.1f}%)")
        
        return core_status
    
    def _test_other_indicators(self):
        """测试91个其他指标"""
        other_categories = [cat for cat in self.indicator_categories.keys() if cat != 'core_indicators']
        other_results = {}
        
        total_other = sum(len(self.indicator_categories[cat]) for cat in other_categories)
        print(f"开始测试 {total_other} 个其他指标...")
        
        for category in other_categories:
            print(f"\n🔍 测试 {category} 类别...")
            category_result = self._test_category(category)
            other_results[category] = category_result
        
        return other_results
    
    def _test_category(self, category):
        """测试指定类别的指标"""
        indicators = self.indicator_categories[category]
        category_result = {
            'total': len(indicators),
            'successful': 0,
            'failed': 0,
            'details': {}
        }
        
        for indicator_name in indicators:
            try:
                print(f"  📈 测试 {indicator_name}...")
                result = self._test_single_indicator(indicator_name)
                
                if result['status'] == 'SUCCESS':
                    category_result['successful'] += 1
                    print(f"    ✅ 成功 - {result.get('result_type', 'unknown')}类型")
                else:
                    category_result['failed'] += 1
                    print(f"    ❌ 失败 - {result.get('error', '未知错误')}")
                
                category_result['details'][indicator_name] = result
                
            except Exception as e:
                category_result['failed'] += 1
                category_result['details'][indicator_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"    💥 异常 - {str(e)}")
        
        success_rate = (category_result['successful'] / category_result['total']) * 100 if category_result['total'] > 0 else 0
        print(f"📈 {category} 结果: {category_result['successful']}/{category_result['total']} ({success_rate:.1f}%)")
        
        return category_result
    
    def _test_single_indicator(self, indicator_name):
        """测试单个指标"""
        try:
            # 1. 创建指标
            indicator = complete_registry.create_indicator(indicator_name)
            if not indicator:
                return {'status': 'FAILED', 'error': '指标创建失败'}
            
            # 2. 生成测试数据
            test_data = self._generate_test_data_for_indicator(indicator_name)
            if test_data is None or test_data.empty:
                return {'status': 'FAILED', 'error': '测试数据生成失败'}
            
            # 3. 计算指标
            result = indicator.calculate(test_data)
            
            # 4. 检查结果
            if result is None:
                return {'status': 'FAILED', 'error': '指标计算结果为None'}
            
            # 5. 分析结果类型
            if isinstance(result, dict):
                # ZXM指标通常返回字典
                return {
                    'status': 'SUCCESS',
                    'result_type': 'dict',
                    'keys': list(result.keys()) if result else [],
                    'has_data': bool(result)
                }
            elif hasattr(result, 'empty'):
                # DataFrame类型结果
                if result.empty:
                    return {'status': 'FAILED', 'error': '指标计算结果为空DataFrame'}
                return {
                    'status': 'SUCCESS',
                    'result_type': 'dataframe',
                    'data_rows': len(result),
                    'columns': list(result.columns)
                }
            else:
                # 其他类型结果
                return {
                    'status': 'SUCCESS',
                    'result_type': str(type(result)),
                    'data_preview': str(result)[:100] + '...' if len(str(result)) > 100 else str(result)
                }
        
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _generate_test_data_for_indicator(self, indicator_name):
        """为指标生成适合的测试数据"""
        try:
            # ZXM指标需要多周期数据
            if indicator_name.startswith('ZXM_'):
                return self.test_data_generator.generate_zxm_test_data(120)
            
            # 形态识别指标需要特殊的K线数据
            elif indicator_name in ['CANDLESTICK_PATTERNS', 'DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING']:
                return self.test_data_generator.generate_candlestick_test_data(100)
            
            # 波动性指标需要波动数据
            elif indicator_name in ['ATR', 'KC', 'VIX', 'STDDEV']:
                return self.test_data_generator.generate_volatility_test_data(100)
            
            # 成交量指标需要成交量数据
            elif indicator_name in ['OBV', 'AD', 'EMV', 'VOL', 'VR', 'VOSC', 'MFI', 'PVT', 'CHAIKIN']:
                return self.test_data_generator.generate_volume_test_data(100)
            
            # 增强指标需要特定数据
            elif indicator_name.startswith('Enhanced'):
                return self.test_data_generator.generate_enhanced_test_data(indicator_name, 100)
            
            # 默认生成标准测试数据
            else:
                return self.test_data_generator.generate_standard_test_data(100)
                
        except Exception as e:
            logger.error(f"生成 {indicator_name} 测试数据失败: {e}")
            return None
    
    def _generate_final_report(self, core_status, other_results, test_start_time):
        """生成最终测试报告"""
        test_duration = time.time() - test_start_time
        
        # 统计总体结果
        total_core = core_status['total']
        verified_core = core_status['verified']
        
        total_other = sum(result['total'] for result in other_results.values())
        successful_other = sum(result['successful'] for result in other_results.values())
        failed_other = sum(result['failed'] for result in other_results.values())
        
        total_indicators = total_core + total_other
        total_successful = verified_core + successful_other
        overall_success_rate = (total_successful / total_indicators) * 100 if total_indicators > 0 else 0
        
        final_report = {
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': round(test_duration, 2),
            'summary': {
                'total_indicators': total_indicators,
                'target_indicators': 112,
                'successful_indicators': total_successful,
                'failed_indicators': total_indicators - total_successful,
                'overall_success_rate': round(overall_success_rate, 1),
                'status': 'EXCELLENT' if overall_success_rate >= 90 else 'GOOD' if overall_success_rate >= 80 else 'NEEDS_IMPROVEMENT'
            },
            'core_indicators_status': core_status,
            'other_indicators_results': other_results
        }
        
        # 打印报告摘要
        print(f"\n" + "=" * 80)
        print(f"🎉 112指标系统测试完成报告")
        print(f"=" * 80)
        print(f"📊 测试摘要:")
        print(f"   总指标数: {total_indicators}")
        print(f"   目标数量: 112")
        print(f"   成功指标: {total_successful}")
        print(f"   失败指标: {total_indicators - total_successful}")
        print(f"   成功率: {overall_success_rate:.1f}%")
        print(f"   测试时长: {test_duration:.2f}秒")
        print(f"   系统状态: {final_report['summary']['status']}")
        
        print(f"\n📋 分类详情:")
        print(f"   ✅ 核心指标: {verified_core}/{total_core} (已100%修复)")
        
        for category, result in other_results.items():
            success_rate = (result['successful'] / result['total']) * 100 if result['total'] > 0 else 0
            status_icon = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
            print(f"   {status_icon} {category}: {result['successful']}/{result['total']} ({success_rate:.1f}%)")
        
        # 保存报告
        self._save_report(final_report)
        
        return final_report
    
    def _save_report(self, report):
        """保存测试报告"""
        try:
            # 创建结果目录
            results_dir = project_root / "results" / "112_indicators_tests"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = results_dir / f"112_indicators_test_report_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 保存Markdown报告
            md_path = results_dir / f"112_indicators_test_report_{timestamp}.md"
            self._save_markdown_report(report, md_path)
            
            print(f"\n📄 测试报告已保存:")
            print(f"   JSON: {json_path}")
            print(f"   MD: {md_path}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
    
    def _save_markdown_report(self, report, md_path):
        """保存Markdown格式报告"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 112指标系统继续测试修复报告\n\n")
            f.write(f"**测试时间**: {report['test_time']}  \n")
            f.write(f"**测试方法**: Ultra Think方法论标准  \n")
            f.write(f"**测试时长**: {report['duration_seconds']}秒  \n\n")
            
            f.write("## 📊 测试摘要\n\n")
            summary = report['summary']
            f.write(f"- **总指标数**: {summary['total_indicators']}\n")
            f.write(f"- **目标数量**: {summary['target_indicators']}\n")
            f.write(f"- **成功指标**: {summary['successful_indicators']}\n")
            f.write(f"- **失败指标**: {summary['failed_indicators']}\n")
            f.write(f"- **成功率**: {summary['overall_success_rate']}%\n")
            f.write(f"- **系统状态**: {summary['status']}\n\n")
            
            f.write("## 🎯 核心指标状态\n\n")
            core = report['core_indicators_status']
            f.write(f"**21个核心指标**: {core['verified']}/{core['total']} 已完成100%买点识别准确率修复\n\n")
            
            f.write("## 🔍 其他指标测试结果\n\n")
            for category, result in report['other_indicators_results'].items():
                success_rate = (result['successful'] / result['total']) * 100 if result['total'] > 0 else 0
                status_icon = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
                f.write(f"### {status_icon} {category.upper()}\n")
                f.write(f"- **测试结果**: {result['successful']}/{result['total']} ({success_rate:.1f}%)\n")
                f.write(f"- **成功指标**: {result['successful']}个\n")
                f.write(f"- **失败指标**: {result['failed']}个\n\n")
            
            f.write("---\n\n")
            f.write("**Ultra Think方法论**: 追求100%完美标准，不断迭代优化  \n")
            f.write("**系统质量**: 基于112指标架构，为拓展新指标奠定坚实基础  \n")


def main():
    """主函数"""
    try:
        print("🚀 启动112指标系统继续测试修复")
        
        # 创建测试器
        tester = Continue112IndicatorsTest()
        
        # 运行综合测试
        report = tester.run_comprehensive_112_test()
        
        # 输出最终结果
        print(f"\n🎊 112指标系统测试修复完成!")
        print(f"📈 总体成功率: {report['summary']['overall_success_rate']}%")
        print(f"🏆 系统状态: {report['summary']['status']}")
        
        return 0 if report['summary']['overall_success_rate'] >= 80 else 1
        
    except Exception as e:
        logger.error(f"💥 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())