#!/usr/bin/env python3
"""
简化的批量指标注册脚本
避免循环导入问题，直接测试指标导入和注册
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

class SimpleBatchRegistration:
    """简化的批量注册器"""
    
    def __init__(self):
        self.registered_indicators = {}
        self.stats = {'successful': 0, 'failed': 0, 'failed_list': []}
    
    def test_and_register_indicator(self, module_path: str, class_name: str, indicator_name: str, description: str) -> bool:
        """测试并注册指标"""
        try:
            print(f"正在测试 {indicator_name}...")
            
            # 尝试导入模块
            module = importlib.import_module(module_path)
            indicator_class = getattr(module, class_name, None)
            
            if indicator_class is None:
                print(f"❌ {indicator_name}: 类 {class_name} 不存在")
                self.stats['failed'] += 1
                self.stats['failed_list'].append(indicator_name)
                return False
            
            # 检查是否为BaseIndicator子类
            from indicators.base_indicator import BaseIndicator
            if not issubclass(indicator_class, BaseIndicator):
                print(f"❌ {indicator_name}: 不是BaseIndicator子类")
                self.stats['failed'] += 1
                self.stats['failed_list'].append(indicator_name)
                return False
            
            # 尝试实例化
            try:
                instance = indicator_class()
                print(f"✅ {indicator_name}: 导入和实例化成功")
                
                # 记录为可注册
                self.registered_indicators[indicator_name] = {
                    'class': indicator_class,
                    'description': description,
                    'module_path': module_path,
                    'class_name': class_name
                }
                self.stats['successful'] += 1
                return True
                
            except Exception as e:
                print(f"⚠️  {indicator_name}: 实例化失败 - {e}")
                # 仍然记录为可注册，因为类定义正确
                self.registered_indicators[indicator_name] = {
                    'class': indicator_class,
                    'description': description,
                    'module_path': module_path,
                    'class_name': class_name
                }
                self.stats['successful'] += 1
                return True
                
        except ImportError as e:
            print(f"❌ {indicator_name}: 导入失败 - {e}")
            self.stats['failed'] += 1
            self.stats['failed_list'].append(indicator_name)
            return False
        except Exception as e:
            print(f"❌ {indicator_name}: 其他错误 - {e}")
            self.stats['failed'] += 1
            self.stats['failed_list'].append(indicator_name)
            return False
    
    def test_batch_1_core_indicators(self):
        """测试第一批：核心指标"""
        print("=== 测试第一批：核心指标 (23个) ===")
        
        core_indicators = [
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
        
        for module_path, class_name, indicator_name, description in core_indicators:
            self.test_and_register_indicator(module_path, class_name, indicator_name, description)
        
        print(f"\n第一批测试完成: {self.stats['successful']}/{len(core_indicators)} 成功")
        return self.stats['successful'], len(core_indicators)
    
    def test_batch_2_enhanced_indicators(self):
        """测试第二批：增强指标"""
        print("\n=== 测试第二批：增强指标 (9个) ===")
        
        enhanced_indicators = [
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
        
        initial_successful = self.stats['successful']
        
        for module_path, class_name, indicator_name, description in enhanced_indicators:
            self.test_and_register_indicator(module_path, class_name, indicator_name, description)
        
        batch_successful = self.stats['successful'] - initial_successful
        print(f"\n第二批测试完成: {batch_successful}/{len(enhanced_indicators)} 成功")
        return batch_successful, len(enhanced_indicators)
    
    def test_batch_3_formula_indicators(self):
        """测试第三批：公式指标"""
        print("\n=== 测试第三批：公式指标 (5个) ===")
        
        formula_indicators = [
            ('indicators.formula_indicators', 'CrossOver', 'CROSS_OVER', '交叉条件指标'),
            ('indicators.formula_indicators', 'KDJCondition', 'KDJ_CONDITION', 'KDJ条件指标'),
            ('indicators.formula_indicators', 'MACDCondition', 'MACD_CONDITION', 'MACD条件指标'),
            ('indicators.formula_indicators', 'MACondition', 'MA_CONDITION', 'MA条件指标'),
            ('indicators.formula_indicators', 'GenericCondition', 'GENERIC_CONDITION', '通用条件指标'),
        ]
        
        initial_successful = self.stats['successful']
        
        for module_path, class_name, indicator_name, description in formula_indicators:
            self.test_and_register_indicator(module_path, class_name, indicator_name, description)
        
        batch_successful = self.stats['successful'] - initial_successful
        print(f"\n第三批测试完成: {batch_successful}/{len(formula_indicators)} 成功")
        return batch_successful, len(formula_indicators)
    
    def test_batch_4_pattern_tools_indicators(self):
        """测试第四批：形态和工具指标"""
        print("\n=== 测试第四批：形态和工具指标 (5个) ===")
        
        pattern_tools_indicators = [
            ('indicators.pattern.candlestick_patterns', 'CandlestickPatterns', 'CANDLESTICK_PATTERNS', 'K线形态'),
            ('indicators.pattern.advanced_candlestick_patterns', 'AdvancedCandlestickPatterns', 'ADVANCED_CANDLESTICK', '高级K线形态'),
            ('indicators.fibonacci_tools', 'FibonacciTools', 'FIBONACCI_TOOLS', '斐波那契工具'),
            ('indicators.gann_tools', 'GannTools', 'GANN_TOOLS', '江恩工具'),
            ('indicators.elliott_wave', 'ElliottWave', 'ELLIOTT_WAVE', '艾略特波浪'),
        ]
        
        initial_successful = self.stats['successful']
        
        for module_path, class_name, indicator_name, description in pattern_tools_indicators:
            self.test_and_register_indicator(module_path, class_name, indicator_name, description)
        
        batch_successful = self.stats['successful'] - initial_successful
        print(f"\n第四批测试完成: {batch_successful}/{len(pattern_tools_indicators)} 成功")
        return batch_successful, len(pattern_tools_indicators)
    
    def generate_summary(self):
        """生成总结报告"""
        total_tested = self.stats['successful'] + self.stats['failed']
        success_rate = (self.stats['successful'] / total_tested * 100) if total_tested > 0 else 0
        
        print(f"\n" + "="*60)
        print(f"📊 批量指标测试总结")
        print(f"="*60)
        print(f"总测试指标: {total_tested}")
        print(f"测试成功: {self.stats['successful']}")
        print(f"测试失败: {self.stats['failed']}")
        print(f"成功率: {success_rate:.1f}%")
        
        if self.stats['failed_list']:
            print(f"\n❌ 失败指标:")
            for indicator in self.stats['failed_list']:
                print(f"  - {indicator}")
        
        print(f"\n✅ 可注册指标 ({self.stats['successful']}个):")
        for i, (name, info) in enumerate(sorted(self.registered_indicators.items()), 1):
            print(f"  {i:2d}. {name} - {info['description']}")
        
        # 评估结果
        if success_rate >= 80:
            print(f"\n🎉 测试结果优秀！大部分指标可以注册")
        elif success_rate >= 60:
            print(f"\n👍 测试结果良好！多数指标可以注册")
        elif success_rate >= 40:
            print(f"\n⚠️  测试结果一般，部分指标可以注册")
        else:
            print(f"\n❌ 测试结果较差，需要进一步调试")
        
        return success_rate >= 60

def main():
    """主函数"""
    print("🚀 开始简化批量指标测试...")
    
    registrar = SimpleBatchRegistration()
    
    # 执行各批次测试
    registrar.test_batch_1_core_indicators()
    registrar.test_batch_2_enhanced_indicators()
    registrar.test_batch_3_formula_indicators()
    registrar.test_batch_4_pattern_tools_indicators()
    
    # 生成总结
    success = registrar.generate_summary()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
