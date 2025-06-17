#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pattern Polarity Analysis
Systematically analyzes each pattern to verify if polarity annotations are technically correct
"""

import importlib
from indicators.pattern_registry import PatternRegistry
from utils.logger import get_logger

logger = get_logger(__name__)

def analyze_pattern_technical_meaning(pattern_id: str, display_name: str, description: str, 
                                    pattern_type: str, score_impact: float, polarity: str) -> dict:
    """Analyze if a pattern's polarity annotation is technically correct"""
    
    analysis = {
        'pattern_id': pattern_id,
        'display_name': display_name,
        'description': description,
        'pattern_type': pattern_type,
        'score_impact': score_impact,
        'current_polarity': polarity,
        'issues': [],
        'recommended_polarity': None,
        'reasoning': []
    }
    
    pattern_lower = pattern_id.lower()
    desc_lower = description.lower() if description else ""
    name_lower = display_name.lower() if display_name else ""
    
    # 分析技术含义
    technical_signals = []
    
    # 1. 明确的看涨信号
    bullish_patterns = [
        ('golden_cross', '金叉', '上穿信号线通常是看涨信号'),
        ('cross_up', '上穿', '向上穿越通常是看涨信号'),
        ('cross_above', '上穿', '向上穿越通常是看涨信号'),
        ('breakout', '突破', '向上突破通常是看涨信号'),
        ('break_up', '向上突破', '向上突破是看涨信号'),
        ('bullish', '看涨', '明确标注为看涨'),
        ('rising', '上升', '上升趋势是看涨信号'),
        ('uptrend', '上升趋势', '上升趋势是看涨信号'),
        ('oversold', '超卖', '超卖区域通常预示反弹机会'),
        ('support', '支撑', '支撑位通常是买入机会'),
        ('bounce', '反弹', '反弹是看涨信号'),
        ('bottom', '底部', '底部形态通常是看涨信号'),
        ('buy', '买入', '买入信号明确看涨'),
        ('accumulation', '吸筹', '吸筹阶段通常看涨'),
        ('absorption', '吸收', '机构吸收筹码通常看涨'),
        ('rally', '拉升', '拉升阶段明确看涨'),
        ('completion', '完成', '吸筹完成通常看涨'),
        ('rebound', '反弹', '反弹是看涨信号'),
        ('low_volatility', '低波动', 'VIX低波动通常看涨'),
        ('extreme_panic', '极度恐慌', 'VIX极度恐慌通常是买入机会'),
        ('high_panic', '高度恐慌', 'VIX高度恐慌通常看涨'),
        ('top_reversal', '见顶回落', 'VIX见顶回落通常看涨'),
        ('rapid_rise', '快速上升', 'VIX快速上升通常看涨'),
        ('historical_high', '历史高位', 'VIX历史高位通常看涨'),
        ('arrangement', '多头排列', '多头排列明确看涨'),
        ('strong_up', '强势上涨', '强势上涨明确看涨'),
        ('consecutive_rising', '连续上升', '连续上升明确看涨'),
    ]
    
    # 2. 明确的看跌信号
    bearish_patterns = [
        ('death_cross', '死叉', '下穿信号线通常是看跌信号'),
        ('cross_down', '下穿', '向下穿越通常是看跌信号'),
        ('cross_below', '下穿', '向下穿越通常是看跌信号'),
        ('breakdown', '跌破', '向下跌破通常是看跌信号'),
        ('break_down', '向下跌破', '向下跌破是看跌信号'),
        ('bearish', '看跌', '明确标注为看跌'),
        ('falling', '下降', '下降趋势是看跌信号'),
        ('downtrend', '下降趋势', '下降趋势是看跌信号'),
        ('overbought', '超买', '超买区域通常预示回调'),
        ('resistance', '阻力', '阻力位通常是卖出机会'),
        ('top', '顶部', '顶部形态通常是看跌信号'),
        ('sell', '卖出', '卖出信号明确看跌'),
        ('distribution', '出货', '出货阶段通常看跌'),
        ('washout', '洗盘', '洗盘过程通常看跌'),
        ('strong_down', '强势下跌', '强势下跌明确看跌'),
        ('consecutive_falling', '连续下降', '连续下降明确看跌'),
        ('high_volatility', '高波动', 'VIX高波动通常看跌'),
        ('extreme_optimism', '极度乐观', 'VIX极度乐观通常看跌'),
        ('bottom_reversal', '见底回升', 'VIX见底回升通常看跌'),
        ('historical_low', '历史低位', 'VIX历史低位通常看跌'),
        ('bearish_arrangement', '空头排列', '空头排列明确看跌'),
        ('hanging_man', '上吊线', '上吊线是看跌信号'),
        ('shooting_star', '流星线', '流星线是看跌信号'),
        ('evening_star', '黄昏之星', '黄昏之星是看跌信号'),
        ('dark_cloud', '乌云盖顶', '乌云盖顶是看跌信号'),
        ('engulfing_bearish', '看跌吞没', '看跌吞没明确看跌'),
    ]
    
    # 3. 中性信号
    neutral_patterns = [
        ('neutral', '中性', '明确标注为中性'),
        ('consolidation', '整理', '整理阶段通常中性'),
        ('time', '时间', '时间周期通常中性'),
        ('cycle', '周期', '周期信号通常中性'),
        ('confirmation', '确认', '确认信号通常中性'),
        ('anomaly', '异常', '异常波动通常中性'),
        ('extreme', '极值', '极值状态可能中性'),
        ('wide', '宽幅', '宽幅通道通常中性'),
        ('narrow', '窄幅', '窄幅通道通常中性'),
        ('doji', '十字星', '十字星通常中性'),
        ('island', '岛形', '岛形反转可能中性'),
        ('triangle', '三角形', '三角形整理通常中性'),
        ('convergence', '收敛', '收敛通常中性'),
        ('shrink', '缩量', '缩量可能中性'),
        ('range', '区间', '区间震荡通常中性'),
        ('volume_confirmation', '成交量确认', '成交量确认通常中性'),
        ('trend_alignment', '趋势对齐', '趋势对齐可能中性'),
    ]
    
    # 检查模式匹配
    for keyword, chinese, reason in bullish_patterns:
        if keyword in pattern_lower or chinese in name_lower or chinese in desc_lower:
            technical_signals.append(('POSITIVE', reason))
            break
    
    for keyword, chinese, reason in bearish_patterns:
        if keyword in pattern_lower or chinese in name_lower or chinese in desc_lower:
            technical_signals.append(('NEGATIVE', reason))
            break
    
    for keyword, chinese, reason in neutral_patterns:
        if keyword in pattern_lower or chinese in name_lower or chinese in desc_lower:
            technical_signals.append(('NEUTRAL', reason))
            break
    
    # 基于pattern_type和score_impact的分析
    type_signal = None
    score_signal = None
    
    if pattern_type:
        if pattern_type.upper() == 'BULLISH':
            type_signal = ('POSITIVE', f'模式类型为BULLISH，应该是POSITIVE极性')
        elif pattern_type.upper() == 'BEARISH':
            type_signal = ('NEGATIVE', f'模式类型为BEARISH，应该是NEGATIVE极性')
        elif pattern_type.upper() == 'NEUTRAL':
            type_signal = ('NEUTRAL', f'模式类型为NEUTRAL，应该是NEUTRAL极性')
    
    if score_impact is not None:
        if score_impact > 0:
            score_signal = ('POSITIVE', f'评分影响为正值({score_impact})，应该是POSITIVE极性')
        elif score_impact < 0:
            score_signal = ('NEGATIVE', f'评分影响为负值({score_impact})，应该是NEGATIVE极性')
        else:
            score_signal = ('NEUTRAL', f'评分影响为零({score_impact})，应该是NEUTRAL极性')
    
    # 综合分析
    all_signals = []
    if technical_signals:
        all_signals.extend(technical_signals)
    if type_signal:
        all_signals.append(type_signal)
    if score_signal:
        all_signals.append(score_signal)
    
    # 检查一致性
    polarities = [signal[0] for signal in all_signals]
    reasons = [signal[1] for signal in all_signals]
    
    if len(set(polarities)) == 1:
        # 所有信号一致
        recommended_polarity = polarities[0]
        analysis['reasoning'] = reasons
    else:
        # 信号不一致，需要人工判断
        analysis['issues'].append(f"信号不一致: {set(polarities)}")
        analysis['reasoning'] = reasons
        # 优先考虑技术含义
        if technical_signals:
            recommended_polarity = technical_signals[0][0]
        elif type_signal:
            recommended_polarity = type_signal[0]
        else:
            recommended_polarity = score_signal[0] if score_signal else 'NEUTRAL'
    
    analysis['recommended_polarity'] = recommended_polarity
    
    # 检查当前极性是否正确
    if polarity != recommended_polarity:
        analysis['issues'].append(f"极性不匹配: 当前{polarity}, 建议{recommended_polarity}")
    
    return analysis

def analyze_all_patterns():
    """Analyze all registered patterns for polarity correctness"""
    
    print("🔍 PATTERN POLARITY TECHNICAL ANALYSIS")
    print("=" * 80)
    print("系统性分析每个模式的极性标注是否符合技术含义")
    print()
    
    # Initialize all indicators
    indicators_to_init = [
        ('indicators.vix', 'VIX'),
        ('indicators.bias', 'BIAS'),
        ('indicators.pvt', 'PVT'),
        ('indicators.zxm_absorb', 'ZXMAbsorb'),
        ('indicators.sar', 'SAR'),
        ('indicators.wma', 'WMA'),
        ('indicators.vortex', 'Vortex'),
        ('indicators.ma', 'MA'),
        ('indicators.fibonacci_tools', 'FibonacciTools'),
        ('indicators.kdj', 'KDJ'),
        ('indicators.emv', 'EMV'),
        ('indicators.zxm_washplate', 'ZXMWashPlate'),
        ('indicators.composite_indicator', 'CompositeIndicator'),
        ('indicators.vosc', 'VOSC'),
        ('indicators.macd', 'MACD'),
        ('indicators.stochrsi', 'STOCHRSI'),
        ('indicators.ema', 'EMA'),
        ('indicators.cmo', 'CMO'),
        ('indicators.atr', 'ATR'),
        ('indicators.boll', 'BOLL'),
        ('indicators.roc', 'ROC'),
        ('indicators.dma', 'DMA'),
        ('indicators.gann_tools', 'GannTools'),
        ('indicators.cci', 'CCI'),
        ('indicators.institutional_behavior', 'InstitutionalBehavior'),
        ('indicators.psy', 'PSY'),
        ('indicators.chip_distribution', 'ChipDistribution'),
        ('indicators.mfi', 'MFI'),
        ('indicators.ichimoku', 'Ichimoku'),
        ('indicators.obv', 'OBV'),
        ('indicators.vol', 'VOL'),
        ('indicators.chaikin', 'Chaikin'),
        ('indicators.unified_ma', 'UnifiedMA'),
        ('indicators.volume_ratio', 'VolumeRatio'),
        ('indicators.wr', 'WR'),
        ('indicators.mtm', 'MTM'),
        ('indicators.vr', 'VR'),
        ('indicators.stock_vix', 'StockVIX'),
        ('indicators.kc', 'KC'),
        ('indicators.elliott_wave', 'ElliottWave'),
        ('indicators.rsi', 'RSI'),
        ('indicators.trix', 'TRIX'),
        ('indicators.aroon', 'Aroon'),
        ('indicators.momentum', 'Momentum'),
        ('indicators.dmi', 'DMI'),
        ('indicators.trend.enhanced_dmi', 'EnhancedDMI'),
        ('indicators.trend.enhanced_macd', 'EnhancedMACD'),
        ('indicators.trend.enhanced_cci', 'EnhancedCCI'),
        ('indicators.trend.enhanced_trix', 'EnhancedTRIX'),
        ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ'),
        ('indicators.volume.enhanced_mfi', 'EnhancedMFI'),
        ('indicators.volume.enhanced_obv', 'EnhancedOBV'),
        ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns'),
        ('indicators.pattern.zxm_patterns', 'ZXMPatternIndicator'),
        ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns'),
    ]
    
    for module_name, class_name in indicators_to_init:
        try:
            module = importlib.import_module(module_name)
            indicator_class = getattr(module, class_name)
            
            indicator = None
            try:
                indicator = indicator_class()
            except:
                try:
                    indicator = indicator_class(periods=[5, 10, 20])
                except:
                    try:
                        indicator = indicator_class(period=14)
                    except:
                        pass
            
            if indicator and hasattr(indicator, 'register_patterns'):
                indicator.register_patterns()
                
        except Exception as e:
            print(f"Failed to initialize {class_name}: {e}")
    
    # Get all patterns
    registry = PatternRegistry()
    all_patterns = registry.get_all_patterns()
    
    print(f"📊 分析 {len(all_patterns)} 个模式的极性标注...")
    print()
    
    issues_found = []
    correct_patterns = 0
    
    for pattern_id, pattern_info in all_patterns.items():
        display_name = pattern_info.get('display_name', '')
        description = pattern_info.get('description', '')
        pattern_type = pattern_info.get('pattern_type', '').name if hasattr(pattern_info.get('pattern_type', ''), 'name') else str(pattern_info.get('pattern_type', ''))
        score_impact = pattern_info.get('score_impact', 0)
        polarity = pattern_info.get('polarity', '').name if hasattr(pattern_info.get('polarity', ''), 'name') else str(pattern_info.get('polarity', ''))
        
        analysis = analyze_pattern_technical_meaning(
            pattern_id, display_name, description, pattern_type, score_impact, polarity
        )
        
        if analysis['issues']:
            issues_found.append(analysis)
        else:
            correct_patterns += 1
    
    # Report results
    print(f"📋 分析结果:")
    print(f"- 总模式数: {len(all_patterns)}")
    print(f"- 正确标注: {correct_patterns}")
    print(f"- 需要检查: {len(issues_found)}")
    print(f"- 准确率: {correct_patterns/len(all_patterns)*100:.1f}%")
    
    if issues_found:
        print(f"\n❌ 需要检查的模式 ({len(issues_found)}):")
        print("=" * 80)
        
        for analysis in issues_found:
            print(f"\n🔍 {analysis['pattern_id']}")
            print(f"   名称: {analysis['display_name']}")
            print(f"   描述: {analysis['description']}")
            print(f"   类型: {analysis['pattern_type']} | 评分: {analysis['score_impact']} | 当前极性: {analysis['current_polarity']}")
            print(f"   建议极性: {analysis['recommended_polarity']}")
            print(f"   问题: {', '.join(analysis['issues'])}")
            print(f"   分析理由:")
            for reason in analysis['reasoning']:
                print(f"     - {reason}")
    else:
        print(f"\n🎉 所有模式的极性标注都符合技术含义！")
    
    return issues_found

if __name__ == "__main__":
    issues = analyze_all_patterns()
    
    if issues:
        print(f"\n💡 建议:")
        print(f"请仔细检查上述 {len(issues)} 个模式，确认极性标注是否符合技术分析原理")
    else:
        print(f"\n✅ 极性标注质量检查通过！")
