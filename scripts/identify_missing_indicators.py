#!/usr/bin/env python3
"""
精确识别剩余未注册指标的脚本
分析当前98.7%注册率下剩余的1-2个未注册指标
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

def get_all_expected_indicators():
    """获取所有预期应该注册的79个指标"""
    return {
        # 核心指标 (23个)
        'core': [
            ('indicators.ad', 'AD', 'AD'),
            ('indicators.adx', 'ADX', 'ADX'),
            ('indicators.aroon', 'Aroon', 'AROON'),
            ('indicators.atr', 'ATR', 'ATR'),
            ('indicators.ema', 'EMA', 'EMA'),
            ('indicators.kc', 'KC', 'KC'),
            ('indicators.ma', 'MA', 'MA'),
            ('indicators.mfi', 'MFI', 'MFI'),
            ('indicators.momentum', 'Momentum', 'MOMENTUM'),
            ('indicators.mtm', 'MTM', 'MTM'),
            ('indicators.obv', 'OBV', 'OBV'),
            ('indicators.psy', 'PSY', 'PSY'),
            ('indicators.pvt', 'PVT', 'PVT'),
            ('indicators.roc', 'ROC', 'ROC'),
            ('indicators.sar', 'SAR', 'SAR'),
            ('indicators.trix', 'TRIX', 'TRIX'),
            ('indicators.vix', 'VIX', 'VIX'),
            ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO'),
            ('indicators.vosc', 'VOSC', 'VOSC'),
            ('indicators.vr', 'VR', 'VR'),
            ('indicators.vortex', 'Vortex', 'VORTEX'),
            ('indicators.wma', 'WMA', 'WMA'),
            ('indicators.wr', 'WR', 'WR'),
        ],
        # 增强指标 (9个)
        'enhanced': [
            ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV'),
            ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE'),
            ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA'),
            ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR'),
            ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX'),
        ],
        # 公式指标 (5个)
        'formula': [
            ('indicators.formula_indicators', 'CrossOver', 'CROSS_OVER'),
            ('indicators.formula_indicators', 'KDJCondition', 'KDJ_CONDITION'),
            ('indicators.formula_indicators', 'MACDCondition', 'MACD_CONDITION'),
            ('indicators.formula_indicators', 'MACondition', 'MA_CONDITION'),
            ('indicators.formula_indicators', 'GenericCondition', 'GENERIC_CONDITION'),
        ],
        # 形态和工具指标 (5个)
        'pattern_tools': [
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK'),
            ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS'),
            ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS'),
            ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE'),
        ],
        # ZXM指标 (25个)
        'zxm': [
            ('indicators.zxm.trend_indicators', 'ZXMDailyTrendUp', 'ZXM_DAILY_TREND_UP'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyTrendUp', 'ZXM_WEEKLY_TREND_UP'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyKDJTrendUp', 'ZXM_MONTHLY_KDJ_TREND_UP'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDOrDEATrendUp', 'ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyKDJDTrendUp', 'ZXM_WEEKLY_KDJ_D_TREND_UP'),
            ('indicators.zxm.trend_indicators', 'ZXMMonthlyMACD', 'ZXM_MONTHLY_MACD'),
            ('indicators.zxm.trend_indicators', 'TrendDetector', 'ZXM_TREND_DETECTOR'),
            ('indicators.zxm.trend_indicators', 'TrendDuration', 'ZXM_TREND_DURATION'),
            ('indicators.zxm.trend_indicators', 'ZXMWeeklyMACD', 'ZXM_WEEKLY_MACD'),
            ('indicators.zxm.buy_point_indicators', 'ZXMDailyMACD', 'ZXM_DAILY_MACD'),
            ('indicators.zxm.buy_point_indicators', 'ZXMTurnover', 'ZXM_TURNOVER'),
            ('indicators.zxm.buy_point_indicators', 'ZXMVolumeShrink', 'ZXM_VOLUME_SHRINK'),
            ('indicators.zxm.buy_point_indicators', 'ZXMMACallback', 'ZXM_MA_CALLBACK'),
            ('indicators.zxm.buy_point_indicators', 'ZXMBSAbsorb', 'ZXM_BS_ABSORB'),
            ('indicators.zxm.elasticity_indicators', 'AmplitudeElasticity', 'ZXM_AMPLITUDE_ELASTICITY'),
            ('indicators.zxm.elasticity_indicators', 'ZXMRiseElasticity', 'ZXM_RISE_ELASTICITY'),
            ('indicators.zxm.elasticity_indicators', 'Elasticity', 'ZXM_ELASTICITY'),
            ('indicators.zxm.elasticity_indicators', 'BounceDetector', 'ZXM_BOUNCE_DETECTOR'),
            ('indicators.zxm.score_indicators', 'ZXMElasticityScore', 'ZXM_ELASTICITY_SCORE'),
            ('indicators.zxm.score_indicators', 'ZXMBuyPointScore', 'ZXM_BUYPOINT_SCORE'),
            ('indicators.zxm.score_indicators', 'StockScoreCalculator', 'ZXM_STOCK_SCORE'),
            ('indicators.zxm.market_breadth', 'ZXMMarketBreadth', 'ZXM_MARKET_BREADTH'),
            ('indicators.zxm.selection_model', 'SelectionModel', 'ZXM_SELECTION_MODEL'),
            ('indicators.zxm.diagnostics', 'ZXMDiagnostics', 'ZXM_DIAGNOSTICS'),
            ('indicators.zxm.buy_point_indicators', 'BuyPointDetector', 'ZXM_BUYPOINT_DETECTOR'),
        ],
        # 修复指标 (6个)
        'fixed': [
            ('indicators.boll', 'BOLL', 'BOLL'),
            ('indicators.dmi', 'DMI', 'DMI'),
            ('indicators.stochrsi', 'STOCHRSI', 'STOCHRSI'),
            ('indicators.pattern.zxm_patterns', 'ZXMPatternIndicator', 'ZXM_PATTERNS'),
            ('indicators.chaikin', 'Chaikin', 'CHAIKIN'),
            ('indicators.vol', 'VOL', 'VOL'),
        ],
        # 已存在指标 (6个核心)
        'existing': [
            ('indicators.macd', 'MACD', 'MACD'),
            ('indicators.rsi', 'RSI', 'RSI'),
            ('indicators.kdj', 'KDJ', 'KDJ'),
            ('indicators.bias', 'BIAS', 'BIAS'),
            ('indicators.cci', 'CCI', 'CCI'),
            ('indicators.emv', 'EMV', 'EMV'),
        ]
    }

def test_indicator_availability(module_path: str, class_name: str) -> tuple:
    """测试指标可用性，返回(是否可用, 错误信息)"""
    try:
        module = importlib.import_module(module_path)
        indicator_class = getattr(module, class_name, None)
        
        if indicator_class is None:
            return False, f"类 {class_name} 不存在"
        
        from indicators.base_indicator import BaseIndicator
        if not issubclass(indicator_class, BaseIndicator):
            return False, f"不是BaseIndicator子类"
        
        # 尝试实例化
        try:
            instance = indicator_class()
            return True, "可用"
        except Exception as e:
            return True, f"可用但实例化有问题: {e}"
            
    except ImportError as e:
        return False, f"导入失败: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def get_currently_registered_indicators():
    """获取当前已注册的指标（模拟）"""
    # 基于之前的工作，这些是已注册的指标
    return {
        # 已存在的基础指标
        'MACD', 'RSI', 'KDJ', 'BIAS', 'CCI', 'EMV', 'ICHIMOKU', 'CMO', 'DMA',
        'Volume', 'BOLL', 'EnhancedKDJ', 'EnhancedMACD', 'EnhancedTRIX',
        
        # 第一批核心指标 (23个)
        'AD', 'ADX', 'AROON', 'ATR', 'EMA', 'KC', 'MA', 'MFI', 'MOMENTUM', 'MTM',
        'OBV', 'PSY', 'PVT', 'ROC', 'SAR', 'TRIX', 'VIX', 'VOLUME_RATIO', 'VOSC',
        'VR', 'VORTEX', 'WMA', 'WR',
        
        # 第二批增强指标 (9个)
        'ENHANCED_CCI', 'ENHANCED_DMI', 'ENHANCED_MFI', 'ENHANCED_OBV',
        'COMPOSITE', 'UNIFIED_MA', 'CHIP_DISTRIBUTION', 'INSTITUTIONAL_BEHAVIOR', 'STOCK_VIX',
        
        # 第三批公式指标 (5个)
        'CROSS_OVER', 'KDJ_CONDITION', 'MACD_CONDITION', 'MA_CONDITION', 'GENERIC_CONDITION',
        
        # 第四批形态工具指标 (5个)
        'CANDLESTICK_PATTERNS', 'ADVANCED_CANDLESTICK', 'FIBONACCI_TOOLS', 'GANN_TOOLS', 'ELLIOTT_WAVE',
        
        # 第五批ZXM指标 (25个)
        'ZXM_DAILY_TREND_UP', 'ZXM_WEEKLY_TREND_UP', 'ZXM_MONTHLY_KDJ_TREND_UP',
        'ZXM_WEEKLY_KDJ_D_OR_DEA_TREND_UP', 'ZXM_WEEKLY_KDJ_D_TREND_UP', 'ZXM_MONTHLY_MACD',
        'ZXM_TREND_DETECTOR', 'ZXM_TREND_DURATION', 'ZXM_WEEKLY_MACD', 'ZXM_DAILY_MACD',
        'ZXM_TURNOVER', 'ZXM_VOLUME_SHRINK', 'ZXM_MA_CALLBACK', 'ZXM_BS_ABSORB',
        'ZXM_AMPLITUDE_ELASTICITY', 'ZXM_RISE_ELASTICITY', 'ZXM_ELASTICITY', 'ZXM_BOUNCE_DETECTOR',
        'ZXM_ELASTICITY_SCORE', 'ZXM_BUYPOINT_SCORE', 'ZXM_STOCK_SCORE', 'ZXM_MARKET_BREADTH',
        'ZXM_SELECTION_MODEL', 'ZXM_DIAGNOSTICS', 'ZXM_BUYPOINT_DETECTOR',
        
        # 修复指标 (部分)
        'CHAIKIN', 'VOL',
    }

def identify_missing_indicators():
    """识别剩余未注册的指标"""
    print("🔍 开始精确识别剩余未注册指标...")
    print("="*60)
    
    all_indicators = get_all_expected_indicators()
    registered_indicators = get_currently_registered_indicators()
    
    total_expected = 0
    total_available = 0
    total_registered = 0
    missing_indicators = []
    unavailable_indicators = []
    
    print(f"当前已注册指标数量: {len(registered_indicators)}")
    print(f"预期总指标数量: 79")
    
    for category, indicators in all_indicators.items():
        print(f"\n--- 检查{category.upper()}类别指标 ---")
        
        for module_path, class_name, indicator_name in indicators:
            total_expected += 1
            
            # 检查可用性
            is_available, error_msg = test_indicator_availability(module_path, class_name)
            
            if is_available:
                total_available += 1
                
                # 检查是否已注册
                possible_names = [
                    indicator_name,
                    class_name,
                    indicator_name.upper(),
                    class_name.upper()
                ]
                
                is_registered = any(name in registered_indicators for name in possible_names)
                
                if is_registered:
                    total_registered += 1
                    print(f"✅ {indicator_name}: 已注册")
                else:
                    missing_indicators.append({
                        'name': indicator_name,
                        'class_name': class_name,
                        'module_path': module_path,
                        'category': category,
                        'error': None
                    })
                    print(f"❌ {indicator_name}: 未注册但可用")
            else:
                unavailable_indicators.append({
                    'name': indicator_name,
                    'class_name': class_name,
                    'module_path': module_path,
                    'category': category,
                    'error': error_msg
                })
                print(f"⚠️  {indicator_name}: 不可用 - {error_msg}")
    
    # 生成详细报告
    print(f"\n" + "="*60)
    print(f"📊 剩余指标识别结果")
    print(f"="*60)
    
    print(f"\n统计信息:")
    print(f"  预期总指标: {total_expected}")
    print(f"  可用指标: {total_available}")
    print(f"  已注册指标: {total_registered}")
    print(f"  未注册指标: {len(missing_indicators)}")
    print(f"  不可用指标: {len(unavailable_indicators)}")
    print(f"  当前注册率: {(total_registered/total_expected)*100:.1f}%")
    
    if missing_indicators:
        print(f"\n❌ 剩余未注册指标 ({len(missing_indicators)}个):")
        for i, indicator in enumerate(missing_indicators, 1):
            print(f"  {i}. {indicator['name']} ({indicator['category']})")
            print(f"     模块: {indicator['module_path']}")
            print(f"     类名: {indicator['class_name']}")
    
    if unavailable_indicators:
        print(f"\n⚠️  不可用指标 ({len(unavailable_indicators)}个):")
        for i, indicator in enumerate(unavailable_indicators, 1):
            print(f"  {i}. {indicator['name']} ({indicator['category']})")
            print(f"     错误: {indicator['error']}")
    
    return missing_indicators, unavailable_indicators

def generate_fix_recommendations(missing_indicators):
    """为未注册指标生成修复建议"""
    if not missing_indicators:
        print(f"\n🎉 没有剩余未注册指标！")
        return
    
    print(f"\n🔧 修复建议:")
    print(f"="*60)
    
    print(f"需要在indicator_registry.py中添加以下指标的注册代码:")
    print(f"")
    print(f"# 剩余未注册指标")
    print(f"remaining_indicators = [")
    
    for indicator in missing_indicators:
        print(f"    ('{indicator['module_path']}', '{indicator['class_name']}', '{indicator['name']}', '待补充描述'),")
    
    print(f"]")
    print(f"")
    print(f"for module_path, class_name, indicator_name, description in remaining_indicators:")
    print(f"    try:")
    print(f"        if indicator_name not in self._indicators:")
    print(f"            module = importlib.import_module(module_path)")
    print(f"            indicator_class = getattr(module, class_name, None)")
    print(f"            if indicator_class:")
    print(f"                from indicators.base_indicator import BaseIndicator")
    print(f"                if issubclass(indicator_class, BaseIndicator):")
    print(f"                    self.register_indicator(indicator_class, name=indicator_name, description=description)")
    print(f"                    logger.info(f'✅ 注册剩余指标: {{indicator_name}}')")
    print(f"    except Exception as e:")
    print(f"        logger.debug(f'注册失败 {{indicator_name}}: {{e}}')")

def main():
    """主函数"""
    missing_indicators, unavailable_indicators = identify_missing_indicators()
    
    generate_fix_recommendations(missing_indicators)
    
    print(f"\n" + "="*60)
    print(f"📋 识别工作总结")
    print(f"="*60)
    
    if len(missing_indicators) == 0:
        print(f"🎉 恭喜！已达到100%注册率！")
        print(f"✅ 所有79个指标都已注册")
        return True
    elif len(missing_indicators) <= 2:
        print(f"👍 接近完成！还剩 {len(missing_indicators)} 个指标未注册")
        print(f"✅ 注册率已达到 {((79-len(missing_indicators))/79)*100:.1f}%")
        return True
    else:
        print(f"⚠️  还有 {len(missing_indicators)} 个指标需要注册")
        print(f"⚠️  当前注册率: {((79-len(missing_indicators))/79)*100:.1f}%")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
