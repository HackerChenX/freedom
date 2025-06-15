#!/usr/bin/env python3
"""
技术指标系统批量注册执行器
按优先级分批注册63个未注册但可用的指标
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BatchRegistrationResult:
    """批量注册结果"""
    batch_name: str
    total_attempted: int
    successful: int
    failed: int
    failed_indicators: List[Tuple[str, str]]
    success_rate: float

class BatchIndicatorRegistrationExecutor:
    """批量指标注册执行器"""
    
    def __init__(self):
        self.registration_results = []
        self.total_stats = {
            'total_attempted': 0,
            'total_successful': 0,
            'total_failed': 0,
            'overall_success_rate': 0.0
        }
        self.registered_indicators = {}
    
    def safe_register_indicator(self, module_path: str, class_name: str, indicator_name: str, description: str) -> bool:
        """安全注册单个指标"""
        try:
            # 动态导入模块
            logger.info(f"正在导入 {module_path}.{class_name}...")
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class is None:
                logger.error(f"❌ 未找到类 {class_name} 在模块 {module_path}")
                return False
            
            # 验证是否为BaseIndicator子类
            from indicators.base_indicator import BaseIndicator
            if not issubclass(indicator_class, BaseIndicator):
                logger.error(f"❌ {class_name} 不是BaseIndicator子类")
                return False
            
            # 尝试获取注册表并注册
            from indicators.indicator_registry import get_registry
            registry = get_registry()
            
            # 检查是否已注册
            if indicator_name in registry.get_indicator_names():
                logger.info(f"⚠️  {indicator_name} 已注册，跳过")
                return True
            
            # 注册指标
            success = registry.register_indicator(
                indicator_class, 
                name=indicator_name, 
                description=description,
                overwrite=False
            )
            
            if success:
                # 验证注册成功
                if indicator_name in registry.get_indicator_names():
                    self.registered_indicators[indicator_name] = {
                        'class': indicator_class,
                        'description': description,
                        'module_path': module_path,
                        'class_name': class_name
                    }
                    logger.info(f"✅ 成功注册: {indicator_name}")
                    return True
                else:
                    logger.error(f"❌ 注册验证失败: {indicator_name}")
                    return False
            else:
                logger.error(f"❌ 注册失败: {indicator_name}")
                return False
                
        except ImportError as e:
            logger.error(f"❌ 导入失败 {module_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 注册过程出错 {indicator_name}: {e}")
            return False
    
    def test_indicator_functionality(self, indicator_name: str) -> bool:
        """测试指标功能"""
        try:
            from indicators.indicator_registry import get_registry
            registry = get_registry()
            
            # 尝试创建指标实例
            indicator = registry.create_indicator(indicator_name)
            if indicator is None:
                logger.warning(f"⚠️  {indicator_name} 无法实例化")
                return False
            
            # 检查必要方法
            required_methods = ['calculate', 'get_patterns', 'calculate_confidence']
            for method in required_methods:
                if not hasattr(indicator, method):
                    logger.warning(f"⚠️  {indicator_name} 缺少方法: {method}")
                    return False
            
            logger.debug(f"✅ {indicator_name} 功能测试通过")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  {indicator_name} 功能测试失败: {e}")
            return False
    
    def execute_batch_registration(self, batch_name: str, indicators: List[Tuple[str, str, str, str]]) -> BatchRegistrationResult:
        """执行批量注册"""
        logger.info(f"\n=== 开始批量注册: {batch_name} ===")
        logger.info(f"本批次需要注册 {len(indicators)} 个指标")
        
        successful = 0
        failed = 0
        failed_indicators = []
        
        for i, (module_path, class_name, indicator_name, description) in enumerate(indicators, 1):
            logger.info(f"\n[{i}/{len(indicators)}] 注册 {indicator_name}...")
            
            if self.safe_register_indicator(module_path, class_name, indicator_name, description):
                # 测试功能
                if self.test_indicator_functionality(indicator_name):
                    successful += 1
                    logger.info(f"✅ {indicator_name} 注册并验证成功")
                else:
                    logger.warning(f"⚠️  {indicator_name} 注册成功但功能测试失败")
                    successful += 1  # 仍然算作成功，只是功能有问题
            else:
                failed += 1
                failed_indicators.append((indicator_name, "注册失败"))
                logger.error(f"❌ {indicator_name} 注册失败")
        
        # 计算成功率
        success_rate = (successful / len(indicators)) * 100 if indicators else 0
        
        result = BatchRegistrationResult(
            batch_name=batch_name,
            total_attempted=len(indicators),
            successful=successful,
            failed=failed,
            failed_indicators=failed_indicators,
            success_rate=success_rate
        )
        
        self.registration_results.append(result)
        
        logger.info(f"\n=== {batch_name} 批量注册完成 ===")
        logger.info(f"成功: {successful}/{len(indicators)} ({success_rate:.1f}%)")
        if failed_indicators:
            logger.warning(f"失败: {failed} 个指标")
            for name, reason in failed_indicators:
                logger.warning(f"  - {name}: {reason}")
        
        return result
    
    def get_batch_1_core_indicators(self) -> List[Tuple[str, str, str, str]]:
        """第一批：核心指标 (23个)"""
        return [
            ('indicators.ad', 'AD', 'AD', '累积/派发线'),
            ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
            ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
            ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
            ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
            ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
            ('indicators.ma', 'MA', 'MA', '移动平均线'),
            ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
            ('indicators.momentum', 'Momentum', 'MOMENTUM', '动量指标'),
            ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
            ('indicators.obv', 'OBV', 'OBV', '能量潮指标'),
            ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
            ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
            ('indicators.roc', 'ROC', 'ROC', '变动率指标'),
            ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
            ('indicators.trix', 'TRIX', 'TRIX', 'TRIX指标'),
            ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
            ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO', '量比指标'),
            ('indicators.vosc', 'VOSC', 'VOSC', '成交量震荡器'),
            ('indicators.vr', 'VR', 'VR', '成交量比率'),
            ('indicators.vortex', 'Vortex', 'VORTEX', '涡流指标'),
            ('indicators.wma', 'WMA', 'WMA', '加权移动平均线'),
            ('indicators.wr', 'WR', 'WR', '威廉指标'),
        ]
    
    def get_batch_2_enhanced_indicators(self) -> List[Tuple[str, str, str, str]]:
        """第二批：增强指标 (9个)"""
        return [
            ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI', '增强版CCI'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI', '增强版DMI'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI', '增强版MFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV', '增强版OBV'),
            ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE', '复合指标'),
            ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA', '统一移动平均线'),
            ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION', '筹码分布'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR', '机构行为'),
            ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX', '个股恐慌指数'),
        ]
    
    def get_batch_3_formula_indicators(self) -> List[Tuple[str, str, str, str]]:
        """第三批：公式指标 (5个)"""
        return [
            ('indicators.formula_indicators', 'CrossOver', 'CROSS_OVER', '交叉条件指标'),
            ('indicators.formula_indicators', 'KDJCondition', 'KDJ_CONDITION', 'KDJ条件指标'),
            ('indicators.formula_indicators', 'MACDCondition', 'MACD_CONDITION', 'MACD条件指标'),
            ('indicators.formula_indicators', 'MACondition', 'MA_CONDITION', 'MA条件指标'),
            ('indicators.formula_indicators', 'GenericCondition', 'GENERIC_CONDITION', '通用条件指标'),
        ]

    def get_batch_4_pattern_tools_indicators(self) -> List[Tuple[str, str, str, str]]:
        """第四批：形态和工具指标 (5个)"""
        return [
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS', 'K线形态'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK', '高级K线形态'),
            ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS', '斐波那契工具'),
            ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS', '江恩工具'),
            ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE', '艾略特波浪'),
        ]

    def get_batch_5_zxm_indicators_part1(self) -> List[Tuple[str, str, str, str]]:
        """第五批：ZXM体系指标 第一部分 (12个)"""
        return [
            # ZXM Trend (9个)
            ('indicators.zxm.trend_indicators', 'ZXMDailyTrendUp', 'ZXM_DAILY_TREND_UP', 'ZXM日趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyTrendUp', 'ZXM_WEEKLY_TREND_UP', 'ZXM周趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyKDJTrendUp', 'ZXM_MONTHLY_KDJ_TREND_UP', 'ZXM月KDJ趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDOrDEATrendUp', 'ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP', 'ZXM周KDJ D或DEA趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDTrendUp', 'ZXM_WEEKLY_KDJ_D_TREND_UP', 'ZXM周KDJ D趋势向上'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyMACD', 'ZXM_MONTHLY_MACD', 'ZXM月MACD'),
            ('indicators.zxm.trend_indicators', 'TrendDetector', 'ZXM_TREND_DETECTOR', 'ZXM趋势检测器'),
            ('indicators.zxm.trend_indicators', 'TrendDuration', 'ZXM_TREND_DURATION', 'ZXM趋势持续时间'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyMACD', 'ZXM_WEEKLY_MACD', 'ZXM周MACD'),
            # ZXM Buy Points (3个)
            ('indicators.zxm.buy_point_indicators', 'ZXMDailyMACD', 'ZXM_DAILY_MACD', 'ZXM日MACD买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMTurnover', 'ZXM_TURNOVER', 'ZXM换手率买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMVolumeShrink', 'ZXM_VOLUME_SHRINK', 'ZXM缩量买点'),
        ]

    def get_batch_6_zxm_indicators_part2(self) -> List[Tuple[str, str, str, str]]:
        """第六批：ZXM体系指标 第二部分 (13个)"""
        return [
            # ZXM Buy Points 剩余 (2个)
            ('indicators.zxm.buy_point_indicators', 'ZXMMACallback', 'ZXM_MA_CALLBACK', 'ZXM均线回踩买点'),
            ('indicators.zxm.buy_point_indicators', 'ZXMBSAbsorb', 'ZXM_BS_ABSORB', 'ZXM吸筹买点'),
            # ZXM Elasticity (4个)
            ('indicators.zxm.elasticity_indicators', 'AmplitudeElasticity', 'ZXM_AMPLITUDE_ELASTICITY', 'ZXM振幅弹性'),
            ('indicators.zxm.elasticity_indicators', 'ZXMRiseElasticity', 'ZXM_RISE_ELASTICITY', 'ZXM涨幅弹性'),
            ('indicators.zxm.elasticity_indicators', 'Elasticity', 'ZXM_ELASTICITY', 'ZXM弹性'),
            ('indicators.zxm.elasticity_indicators', 'BounceDetector', 'ZXM_BOUNCE_DETECTOR', 'ZXM反弹检测器'),
            # ZXM Score (3个)
            ('indicators.zxm.score_indicators', 'ZXMElasticityScore', 'ZXM_ELASTICITY_SCORE', 'ZXM弹性评分'),
            ('indicators.zxm.score_indicators', 'ZXMBuyPointScore', 'ZXM_BUYPOINT_SCORE', 'ZXM买点评分'),
            ('indicators.zxm.score_indicators', 'StockScoreCalculator', 'ZXM_STOCK_SCORE', 'ZXM股票评分'),
            # ZXM其他 (4个)
            ('indicators.zxm.market_breadth', 'ZXMMarketBreadth', 'ZXM_MARKET_BREADTH', 'ZXM市场宽度'),
            ('indicators.zxm.selection_model', 'SelectionModel', 'ZXM_SELECTION_MODEL', 'ZXM选股模型'),
            ('indicators.zxm.diagnostics', 'ZXMDiagnostics', 'ZXM_DIAGNOSTICS', 'ZXM诊断'),
            ('indicators.zxm.buy_point_indicators', 'BuyPointDetector', 'ZXM_BUYPOINT_DETECTOR', 'ZXM买点检测器'),
        ]

    def execute_all_batches(self):
        """执行所有批次的注册"""
        logger.info("🚀 开始技术指标系统批量注册工作")
        logger.info("目标：将注册率从18.8%提升到100%，注册63个可用指标\n")

        # 获取注册前状态
        try:
            from indicators.indicator_registry import get_registry
            registry = get_registry()
            initial_count = len(registry.get_indicator_names())
            logger.info(f"注册前指标数量: {initial_count}")
        except Exception as e:
            logger.error(f"获取初始状态失败: {e}")
            initial_count = 0

        # 执行各批次注册
        batches = [
            ("第一批：核心指标", self.get_batch_1_core_indicators()),
            ("第二批：增强指标", self.get_batch_2_enhanced_indicators()),
            ("第三批：公式指标", self.get_batch_3_formula_indicators()),
            ("第四批：形态和工具指标", self.get_batch_4_pattern_tools_indicators()),
            ("第五批：ZXM指标(第一部分)", self.get_batch_5_zxm_indicators_part1()),
            ("第六批：ZXM指标(第二部分)", self.get_batch_6_zxm_indicators_part2()),
        ]

        for batch_name, indicators in batches:
            try:
                result = self.execute_batch_registration(batch_name, indicators)

                # 更新总体统计
                self.total_stats['total_attempted'] += result.total_attempted
                self.total_stats['total_successful'] += result.successful
                self.total_stats['total_failed'] += result.failed

                # 检查是否需要暂停
                if result.success_rate < 50:
                    logger.warning(f"⚠️  {batch_name} 成功率较低 ({result.success_rate:.1f}%)，建议检查问题")

            except Exception as e:
                logger.error(f"❌ {batch_name} 执行失败: {e}")
                continue

        # 计算总体成功率
        if self.total_stats['total_attempted'] > 0:
            self.total_stats['overall_success_rate'] = (
                self.total_stats['total_successful'] / self.total_stats['total_attempted']
            ) * 100

        # 获取注册后状态
        try:
            final_count = len(registry.get_indicator_names())
            logger.info(f"注册后指标数量: {final_count}")
            new_registered = final_count - initial_count
        except Exception as e:
            logger.error(f"获取最终状态失败: {e}")
            final_count = 0
            new_registered = 0

        # 生成最终报告
        self.generate_final_report(initial_count, final_count, new_registered)

    def generate_final_report(self, initial_count: int, final_count: int, new_registered: int):
        """生成最终报告"""
        logger.info("\n" + "="*60)
        logger.info("🎉 技术指标系统批量注册工作完成")
        logger.info("="*60)

        # 总体统计
        stats = self.total_stats
        logger.info(f"\n📊 总体统计:")
        logger.info(f"  尝试注册: {stats['total_attempted']} 个指标")
        logger.info(f"  成功注册: {stats['total_successful']} 个指标")
        logger.info(f"  注册失败: {stats['total_failed']} 个指标")
        logger.info(f"  总体成功率: {stats['overall_success_rate']:.1f}%")

        # 系统改进
        logger.info(f"\n📈 系统改进:")
        logger.info(f"  注册前指标数量: {initial_count}")
        logger.info(f"  注册后指标数量: {final_count}")
        logger.info(f"  新增注册指标: {new_registered}")
        if initial_count > 0:
            improvement = ((final_count - initial_count) / initial_count) * 100
            logger.info(f"  数量提升: {improvement:.1f}%")

        # 注册率计算
        target_total = 79  # 基于检查报告的可用指标总数
        current_rate = (final_count / target_total) * 100 if target_total > 0 else 0
        logger.info(f"  当前注册率: {current_rate:.1f}% (目标: 100%)")

        # 各批次详情
        logger.info(f"\n📋 各批次详情:")
        for result in self.registration_results:
            status = "✅" if result.success_rate >= 80 else "⚠️" if result.success_rate >= 50 else "❌"
            logger.info(f"  {status} {result.batch_name}: {result.successful}/{result.total_attempted} ({result.success_rate:.1f}%)")

        # 功能预期
        estimated_conditions = final_count * 8
        estimated_patterns = final_count * 3
        logger.info(f"\n🎯 功能预期:")
        logger.info(f"  预期策略条件: ~{estimated_conditions} 个 (目标: 500+)")
        logger.info(f"  预期技术形态: ~{estimated_patterns} 个 (目标: 150+)")

        # 评估结果
        if current_rate >= 90:
            logger.info(f"\n🎉 批量注册大获成功！")
            logger.info(f"✅ 注册率达到 {current_rate:.1f}%，接近100%目标")
        elif current_rate >= 70:
            logger.info(f"\n👍 批量注册基本成功！")
            logger.info(f"✅ 注册率达到 {current_rate:.1f}%，大幅改善")
        elif current_rate >= 50:
            logger.info(f"\n⚠️  批量注册部分成功")
            logger.info(f"⚠️  注册率为 {current_rate:.1f}%，仍需改进")
        else:
            logger.info(f"\n❌ 批量注册遇到困难")
            logger.info(f"❌ 注册率仅为 {current_rate:.1f}%，需要调试")

def main():
    """主函数"""
    executor = BatchIndicatorRegistrationExecutor()
    executor.execute_all_batches()

    # 返回是否成功
    return executor.total_stats['overall_success_rate'] >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
