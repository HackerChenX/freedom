#!/usr/bin/env python3
"""
检查最后剩余未注册指标的简化脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_specific_indicators():
    """检查特定的可能未注册指标"""
    print("🔍 检查可能剩余的未注册指标...")
    
    # 基于目录结构，这些可能是剩余的指标
    potential_missing = [
        # 可能遗漏的核心指标
        ('indicators.unified_indicators', 'UnifiedIndicators', 'UNIFIED_INDICATORS'),
        ('indicators.technical_indicators', 'TechnicalIndicators', 'TECHNICAL_INDICATORS'),
        
        # 可能遗漏的增强指标
        ('indicators.enhanced_stochrsi', 'EnhancedStochasticRSI', 'ENHANCED_STOCHRSI'),
        ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI_ROOT'),
        ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR_ROOT'),
        
        # 可能遗漏的形态指标
        ('indicators.pattern.pattern_combination', 'PatternCombination', 'PATTERN_COMBINATION'),
        ('indicators.pattern.pattern_confirmation', 'PatternConfirmation', 'PATTERN_CONFIRMATION'),
        ('indicators.pattern.pattern_quality_evaluator', 'PatternQualityEvaluator', 'PATTERN_QUALITY'),
        
        # 可能遗漏的ZXM指标
        ('indicators.zxm_absorb', 'ZXMAbsorb', 'ZXM_ABSORB_ROOT'),
        ('indicators.zxm_washplate', 'ZXMWashplate', 'ZXM_WASHPLATE'),
        
        # 可能遗漏的分析工具
        ('indicators.trend_classification', 'TrendClassification', 'TREND_CLASSIFICATION'),
        ('indicators.trend_strength', 'TrendStrength', 'TREND_STRENGTH'),
        ('indicators.market_env', 'MarketEnvironment', 'MARKET_ENV'),
        ('indicators.sentiment_analysis', 'SentimentAnalysis', 'SENTIMENT_ANALYSIS'),
        
        # 可能遗漏的评分指标
        ('indicators.boll_score', 'BOLLScore', 'BOLL_SCORE'),
        ('indicators.kdj_score', 'KDJScore', 'KDJ_SCORE'),
        ('indicators.macd_score', 'MACDScore', 'MACD_SCORE'),
        ('indicators.rsi_score', 'RSIScore', 'RSI_SCORE'),
        ('indicators.volume_score', 'VolumeScore', 'VOLUME_SCORE'),
        
        # 可能遗漏的特殊指标
        ('indicators.divergence', 'Divergence', 'DIVERGENCE'),
        ('indicators.multi_period_resonance', 'MultiPeriodResonance', 'MULTI_PERIOD_RESONANCE'),
        ('indicators.time_cycle_analysis', 'TimeCycleAnalysis', 'TIME_CYCLE'),
        ('indicators.intraday_volatility', 'IntradayVolatility', 'INTRADAY_VOLATILITY'),
        ('indicators.platform_breakout', 'PlatformBreakout', 'PLATFORM_BREAKOUT'),
        ('indicators.island_reversal', 'IslandReversal', 'ISLAND_REVERSAL'),
        ('indicators.v_shaped_reversal', 'VShapedReversal', 'V_SHAPED_REVERSAL'),
    ]
    
    available_indicators = []
    unavailable_indicators = []
    
    for module_path, class_name, indicator_name in potential_missing:
        try:
            print(f"检查 {indicator_name}...")
            
            # 尝试导入模块
            import importlib
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class is_Check_Final_Missing None:
                print(f"  ❌ 类 {class_name} 不存在")
                unavailable_indicators.append((indicator_name, f"类 {class_name} 不存在"))
                continue
            
            # 检查是否为BaseIndicator子类
            from indicators.base_indicator import BaseIndicator
            if not issubclass(indicator_class, BaseIndicator):
                print(f"  ❌ {class_name} 不是BaseIndicator子类")
                unavailable_indicators.append((indicator_name, "不是BaseIndicator子类"))
                continue
            
            # 尝试实例化
            try:
                instance = indicator_class()
                print(f"  ✅ {indicator_name}: 可用")
                available_indicators.append((module_path, class_name, indicator_name))
            except Exception as e:
                print(f"  ⚠️  {indicator_name}: 可用但实例化有问题 - {e}")
                available_indicators.append((module_path, class_name, indicator_name))
                
        except Import_error as e:
            print(f"  ❌ {indicator_name}: 导入失败 - {e}")
            unavailable_indicators.append((indicator_name, f"导入失败: {e}"))
        except Exception as e:
            print(f"  ❌ {indicator_name}: 其他错误 - {e}")
            unavailable_indicators.append((indicator_name, f"其他错误: {e}"))
    
    print(f"\n" + "="*60)
    print(f"📊 检查结果")
    print(f"="*60)
    
    print(f"可用指标: {len(available_indicators)}")
    print(f"不可用指标: {len(unavailable_indicators)}")
    
    if available_indicators:
        print(f"\n✅ 可用但可能未注册的指标:")
        for i, (module_path, class_name, indicator_name) in enumerate(available_indicators, 1):
            print(f"  {i}. {indicator_name}")
            print(f"     模块: {module_path}")
            print(f"     类名: {class_name}")
    
    if unavailable_indicators:
        print(f"\n❌ 不可用的指标:")
        for i, (indicator_name, error) in enumerate(unavailable_indicators, 1):
            print(f"  {i}. {indicator_name}: {error}")
    
    return available_indicators

def generate_registration_code(available_indicators):
    """为可用指标生成注册代码"""
    if not available_indicators:
        print(f"\n🎉 没有发现额外的可用指标")
        return
    
    print(f"\n🔧 为发现的可用指标生成注册代码:")
    print(f"="*60)
    
    print(f"# 新发现的可用指标注册代码")
    print(f"additional_indicators = [")
    
    for module_path, class_name, indicator_name in available_indicators:
        print(f"    ('{module_path}', '{class_name}', '{indicator_name}', '{indicator_name}指标'),")
    
    print(f"]")
    print(f"")
    print(f"# 在indicator_registry.py的register_standard_indicators方法中添加:")
    print(f"for module_path, class_name, indicator_name, description in additional_indicators:")
    print(f"    try:")
    print(f"        if indicator_name not in self._indicators:")
    print(f"            module = importlib.import_module(module_path)")
    print(f"            indicator_class = getattr(module, class_name, None)")
    print(f"            if indicator_class:")
    print(f"                from indicators.base_indicator import BaseIndicator")
    print(f"                if issubclass(indicator_class, BaseIndicator):")
    print(f"                    self.register_indicator(indicator_class, name=indicator_name, description=description)")
    print(f"                    logger.info(f'✅ 注册新发现指标: {{indicator_name}}')")
    print(f"    except Exception as e:")
    print(f"        logger.debug(f'注册失败 {{indicator_name}}: {{e}}')")

def estimate_final_count(available_indicators):
    """估算最终指标数量"""
    current_registered = 78  # 当前已注册
    new_available = len(available_indicators)
    
    print(f"\n📈 最终指标数量估算:")
    print(f"="*60)
    
    print(f"当前已注册: {current_registered}")
    print(f"新发现可用: {new_available}")
    print(f"预期最终: {current_registered + new_available}")
    
    if new_available > 0:
        final_rate = ((current_registered + new_available) / 79) * 100
        print(f"预期注册率: {final_rate:.1f}%")
        
        if final_rate >= 100:
            print(f"🎉 将超过100%注册率目标！")
        elif final_rate >= 99:
            print(f"👍 将接近100%注册率目标！")
        else:
            print(f"⚠️  仍未达到100%注册率目标")
    else:
        print(f"当前注册率: 98.7%")
        print(f"⚠️  没有发现额外可注册指标")

def main_check_final_missing():
    """主函数"""
    available_indicators = check_specific_indicators()
    
    generate_registration_code(available_indicators)
    
    estimate_final_count(available_indicators)
    
    print(f"\n" + "="*60)
    print(f"📋 检查总结")
    print(f"="*60)
    
    if len(available_indicators) > 0:
        print(f"🎉 发现 {len(available_indicators)} 个额外可用指标！")
        print(f"✅ 可以进一步提升注册率")
        return True
    else:
        print(f"⚠️  没有发现额外的可用指标")
        print(f"⚠️  当前98.7%可能是实际最高注册率")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
