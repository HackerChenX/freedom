#!/usr/bin/env python3
"""
手动指标注册脚本
直接创建一个新的注册表，避免循环导入问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from typing import Dict, Any

class ManualIndicatorRegistry:
    """手动指标注册表"""
    
    def __init__(self):
        self._indicators = {}
        self._stats = {'successful': 0, 'failed': 0, 'failed_list': []}
    
    def register_indicator_manual(self, indicator_class, name: str, description: str = ""):
        """手动注册指标"""
        try:
            from indicators.base_indicator import BaseIndicator
            if issubclass(indicator_class, BaseIndicator):
                self._indicators[name] = {
                    'class': indicator_class,
                    'description': description,
                    'is_available': True
                }
                self._stats['successful'] += 1
                print(f"✅ 成功注册: {name}")
                return True
            else:
                print(f"❌ {name} 不是BaseIndicator子类")
                self._stats['failed'] += 1
                self._stats['failed_list'].append(name)
                return False
        except Exception as e:
            print(f"❌ 注册 {name} 失败: {e}")
            self._stats['failed'] += 1
            self._stats['failed_list'].append(name)
            return False
    
    def register_from_import(self, module_path: str, class_name: str, indicator_name: str, description: str = ""):
        """从导入注册指标"""
        try:
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class:
                return self.register_indicator_manual(indicator_class, indicator_name, description)
            else:
                print(f"❌ 未找到类 {class_name} 在 {module_path}")
                self._stats['failed'] += 1
                self._stats['failed_list'].append(indicator_name)
                return False
        except Import_error as e:
            print(f"❌ 导入失败 {module_path}: {e}")
            self._stats['failed'] += 1
            self._stats['failed_list'].append(indicator_name)
            return False
        except Exception as e:
            print(f"❌ 注册过程出错 {indicator_name}: {e}")
            self._stats['failed'] += 1
            self._stats['failed_list'].append(indicator_name)
            return False
    
    def register_all_available_indicators(self):
        """注册所有可用指标"""
        print("=== 开始手动注册所有可用指标 ===\n")
        
        # 核心指标
        print("注册核心指标...")
        core_indicators = [
            ('indicators.ma', 'MA', 'MA', '移动平均线'),
            ('indicators.ema', 'EMA', 'EMA', '指数移动平均线'),
            ('indicators.wma', 'WMA', 'WMA', '加权移动平均线'),
            ('indicators.sar', 'SAR', 'SAR', '抛物线转向指标'),
            ('indicators.adx', 'ADX', 'ADX', '平均趋向指标'),
            ('indicators.aroon', 'Aroon', 'AROON', 'Aroon指标'),
            ('indicators.atr', 'ATR', 'ATR', '平均真实波幅'),
            ('indicators.kc', 'KC', 'KC', '肯特纳通道'),
            ('indicators.mfi', 'MFI', 'MFI', '资金流量指标'),
            ('indicators.momentum', 'Momentum', 'MOMENTUM', '动量指标'),
            ('indicators.mtm', 'MTM', 'MTM', '动量指标'),
            ('indicators.obv', 'OBV', 'OBV', '能量潮指标'),
            ('indicators.psy', 'PSY', 'PSY', '心理线指标'),
            ('indicators.pvt', 'PVT', 'PVT', '价量趋势指标'),
            ('indicators.roc', 'ROC', 'ROC', '变动率指标'),
            ('indicators.trix', 'TRIX', 'TRIX', 'TRIX指标'),
            ('indicators.vix', 'VIX', 'VIX', '恐慌指数'),
            ('indicators.volume_ratio', 'VolumeRatio', 'VOLUME_RATIO', '量比指标'),
            ('indicators.vosc', 'VOSC', 'VOSC', '成交量震荡器'),
            ('indicators.vr', 'VR', 'VR', '成交量比率'),
            ('indicators.vortex', 'Vortex', 'VORTEX', '涡流指标'),
            ('indicators.wr', 'WR', 'WR', '威廉指标'),
            ('indicators.ad', 'AD', 'AD', '累积/派发线'),
            # 已注册的基础指标
            ('indicators.macd', 'MACD', 'MACD', '移动平均线收敛散度指标'),
            ('indicators.rsi', 'RSI', 'RSI', '相对强弱指数'),
            ('indicators.kdj', 'KDJ', 'KDJ', 'KDJ随机指标'),
            ('indicators.bias', 'BIAS', 'BIAS', '乖离率'),
            ('indicators.cci', 'CCI', 'CCI', '顺势指标'),
            ('indicators.emv', 'EMV', 'EMV', '简易波动指标'),
            ('indicators.ichimoku', 'Ichimoku', 'ICHIMOKU', '一目均衡表'),
            ('indicators.cmo', 'CMO', 'CMO', '钱德动量摆动指标'),
            ('indicators.dma', 'DMA', 'DMA', '动态移动平均线'),
        ]
        
        for module_path, class_name, indicator_name, description in core_indicators:
            self.register_from_import(module_path, class_name, indicator_name, description)
        
        # 增强指标
        print("\n注册增强指标...")
        enhanced_indicators = [
            ('indicators.trend.enhanced_cci', 'EnhancedCCI', 'ENHANCED_CCI', '增强版CCI'),
            ('indicators.trend.enhanced_dmi', 'EnhancedDMI', 'ENHANCED_DMI', '增强版DMI'),
            ('indicators.trend.enhanced_macd', 'EnhancedMACD', 'ENHANCED_MACD_TREND', '增强版MACD(趋势)'),
            ('indicators.trend.enhanced_trix', 'EnhancedTRIX', 'ENHANCED_TRIX', '增强版TRIX'),
            ('indicators.oscillator.enhanced_kdj', 'EnhancedKDJ', 'ENHANCED_KDJ_OSC', '增强版KDJ(震荡)'),
            ('indicators.volume.enhanced_mfi', 'EnhancedMFI', 'ENHANCED_MFI', '增强版MFI'),
            ('indicators.volume.enhanced_obv', 'EnhancedOBV', 'ENHANCED_OBV', '增强版OBV'),
            ('indicators.enhanced_rsi', 'EnhancedRSI', 'ENHANCED_RSI', '增强版RSI'),
            ('indicators.enhanced_wr', 'EnhancedWR', 'ENHANCED_WR', '增强版威廉指标'),
            ('indicators.enhanced_macd', 'EnhancedMACD', 'ENHANCED_MACD_ROOT', '增强版MACD(根目录)'),
        ]
        
        for module_path, class_name, indicator_name, description in enhanced_indicators:
            self.register_from_import(module_path, class_name, indicator_name, description)
        
        # 复合指标
        print("\n注册复合指标...")
        composite_indicators = [
            ('indicators.composite_indicator', 'CompositeIndicator', 'COMPOSITE', '复合指标'),
            ('indicators.unified_ma', 'UnifiedMA', 'UNIFIED_MA', '统一移动平均线'),
            ('indicators.chip_distribution', 'ChipDistribution', 'CHIP_DISTRIBUTION', '筹码分布'),
            ('indicators.institutional_behavior', 'InstitutionalBehavior', 'INSTITUTIONAL_BEHAVIOR', '机构行为'),
            ('indicators.stock_vix', 'StockVIX', 'STOCK_VIX', '个股恐慌指数'),
        ]
        
        for module_path, class_name, indicator_name, description in composite_indicators:
            self.register_from_import(module_path, class_name, indicator_name, description)
        
        # 形态指标
        print("\n注册形态指标...")
        pattern_indicators = [
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS', 'K线形态'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK', '高级K线形态'),
        ]
        
        for module_path, class_name, indicator_name, description in pattern_indicators:
            self.register_from_import(module_path, class_name, indicator_name, description)
        
        # 工具指标
        print("\n注册工具指标...")
        tool_indicators = [
            ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS', '斐波那契工具'),
            ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS', '江恩工具'),
            ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE', '艾略特波浪'),
        ]
        
        for module_path, class_name, indicator_name, description in tool_indicators:
            self.register_from_import(module_path, class_name, indicator_name, description)
        
        # 公式指标
        print("\n注册公式指标...")
        formula_indicators = [
            ('indicators.formula_indicators', 'CrossOver', 'CROSS_OVER', '交叉条件指标'),
            ('indicators.formula_indicators', 'KDJCondition', 'KDJ_CONDITION', 'KDJ条件指标'),
            ('indicators.formula_indicators', 'MACDCondition', 'MACD_CONDITION', 'MACD条件指标'),
            ('indicators.formula_indicators', 'MACondition', 'MA_CONDITION', 'MA条件指标'),
            ('indicators.formula_indicators', 'GenericCondition', 'GENERIC_CONDITION', '通用条件指标'),
        ]
        
        for module_path, class_name, indicator_name, description in formula_indicators:
            self.register_from_import(module_path, class_name, indicator_name, description)
    
    def get_indicator_names_Registration(self):
        """获取所有指标名称"""
        return list(self._indicators.keys())
    
    def create_indicator_Registration(self, name: str, **kwargs):
        """创建指标实例"""
        if name not in self._indicators:
            return None
        
        try:
            indicator_class = self._indicators[name]['class']
            return indicator_class(**kwargs)
        except Exception as e:
            print(f"创建指标 {name} 实例失败: {e}")
            return None
    
    def print_summary_Registration_Manual_Indicator_Registration(self):
        """打印注册摘要"""
        total_registered = len(self._indicators)
        stats = self._stats
        
        print(f"\n=== 手动注册摘要 ===")
        print(f"成功注册: {stats['successful']} 个")
        print(f"注册失败: {stats['failed']} 个")
        print(f"最终注册: {total_registered} 个")
        
        if stats['failed_list']:
            print(f"失败指标: {stats['failed_list']}")
        
        print(f"\n所有已注册指标 ({total_registered}个):")
        for i, name in enumerate(sorted(self._indicators.keys()), 1):
            print(f"{i:2d}. {name}")

def main_manualindicatorregistration():
    """主函数"""
    registry = Manual_indicator_registry()
    registry.register_all_available_indicators()
    registry.print_summary_Registration_Manual_Indicator_Registration()
    
    # 测试指标实例化
    print(f"\n=== 测试指标实例化 ===")
    test_indicators = list(registry.get_indicator_names_Registration())[:10]
    successful_instances = 0
    
    for indicator_name in test_indicators:
        try:
            indicator = registry.create_indicator_Registration(indicator_name)
            if indicator:
                successful_instances += 1
                print(f"✅ {indicator_name}: 实例化成功")
            else:
                print(f"❌ {indicator_name}: 实例化失败")
        except Exception as e:
            print(f"❌ {indicator_name}: 实例化异常 - {e}")
    
    print(f"\n实例化测试: {successful_instances}/{len(test_indicators)} 成功")
    
    total_registered = len(registry.get_indicator_names_Registration())
    if total_registered >= 50:
        print(f"\n🎉 手动注册成功！注册了 {total_registered} 个指标")
        return True
    else:
        print(f"\n⚠️  手动注册部分成功，注册了 {total_registered} 个指标")
        return False

if __name__ == "__main__":
    success = main_manualindicatorregistration()
    sys.exit(0 if success else 1)
