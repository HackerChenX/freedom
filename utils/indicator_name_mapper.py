#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标名称映射器

用于解决买点分析生成的策略中的指标名称与Complete_indicator_registry中注册的指标名称不匹配的问题
"""

from typing import Dict, Optional
from utils.logger import getLogger

logger = getLogger(__name__)


class IndicatorNameMapper:
    """指标名称映射器"""
    
    def __init___48(self):
        """初始化映射器"""
        # 策略中的指标名称 -> 注册表中的指标名称
        self.name_mapping = {
            # 核心指标（大部分已匹配）
            'VOL': 'VOL',
            'SAR': 'SAR', 
            'KC': 'KC',
            'MTM': 'MTM',
            'PSY': 'PSY',
            'PVT': 'PVT',
            'TRIX': 'TRIX',
            'VIX': 'VIX',
            'VOSC': 'VOSC',
            'VR': 'VR',
            'WR': 'WR',
            'MACD': 'MACD',
            'BOLL': 'BOLL',
            'KDJ': 'KDJ',
            'BIAS': 'BIAS',
            'DMI': 'DMI',
            'EMV': 'EMV',
            'CMO': 'CMO',
            'DMA': 'DMA',
            'RSI': 'RSI',
            
            # 增强指标映射
            'EnhancedMACD': 'ENHANCED_MACD_TREND',
            'EnhancedTRIX': 'ENHANCED_TRIX',
            'EnhancedKDJ': 'ENHANCED_KDJ_OSC',
            'EnhancedOBV': 'ENHANCED_OBV',
            'EnhancedCCI': 'ENHANCED_CCI',
            'EnhancedRSI': 'ENHANCED_RSI',
            'EnhancedWR': 'ENHANCED_WR',
            'EnhancedMFI': 'ENHANCED_MFI',
            'EnhancedStochRSI': 'ENHANCED_STOCHRSI',
            'EnhancedDMI': 'ENHANCED_DMI',
            
            # 形态指标映射
            'CandlestickPatterns': 'CANDLESTICK_PATTERNS',
            'AdvancedCandlestickPatterns': 'ADVANCED_CANDLESTICK',
            'ZXMPattern': 'ZXM_PATTERNS',
            'ZXMPatterns': 'ZXM_PATTERNS',
            
            # ZXM指标映射
            'TrendDetector': 'ZXM_TREND_DETECTOR',
            'TrendDuration': 'ZXM_TREND_DURATION',
            'ZXMTurnover': 'ZXM_TURNOVER',
            'ZXMVolumeShrink': 'ZXM_VOLUME_SHRINK',
            'ZXMBSAbsorb': 'ZXM_BS_ABSORB',
            'AmplitudeElasticity': 'ZXM_AMPLITUDE_ELASTICITY',
            'ZXMRiseElasticity': 'ZXM_RISE_ELASTICITY',
            'Elasticity': 'ZXM_ELASTICITY',
            'BounceDetector': 'ZXM_BOUNCE_DETECTOR',
            'ZXMElasticityScore': 'ZXM_ELASTICITY_SCORE',
            'ZXMBuyPointScore': 'ZXM_BUYPOINT_SCORE',
            'StockScoreCalculator': 'ZXM_STOCK_SCORE',
            'SelectionModel': 'ZXM_SELECTION_MODEL',
            'ZXMDiagnostics': 'ZXM_DIAGNOSTICS',
            
            # 其他可能的映射
            'MA': 'MA',
            'EMA': 'EMA',
            'WMA': 'WMA',
            'ADX': 'ADX',
            'AROON': 'AROON',
            'ATR': 'ATR',
            'MFI': 'MFI',
            'MOMENTUM': 'MOMENTUM',
            'OBV': 'OBV',
            'ROC': 'ROC',
            'CCI': 'CCI',
            'CHAIKIN': 'CHAIKIN',
            'ICHIMOKU': 'ICHIMOKU',
            'STOCHRSI': 'STOCHRSI',
            'AD': 'AD',
            'VORTEX': 'VORTEX',
            'VOLUME_RATIO': 'VOLUME_RATIO',
        }
        
        # 反向映射（注册表名称 -> 策略名称）
        self.reverse_mapping = {v: k for k, v in self.name_mapping.items()}
        
        logger.info(f"指标名称映射器初始化完成，包含 {len(self.name_mapping)} 个映射")
    
    def map_to_registry_name(self, strategy_name: str) -> Optional[str]:
        """
        将策略中的指标名称映射到注册表中的名称
        
        Args:
            strategy_name: 策略中使用的指标名称
            
        Returns:
            注册表中的指标名称，如果没有映射则返回原名称
        """
        mapped_name = self.name_mapping.get(strategy_name, strategy_name)
        
        if mapped_name != strategy_name:
            logger.debug(f"指标名称映射: {strategy_name} -> {mapped_name}")
        
        return mapped_name
    
    def map_to_strategy_name(self, registry_name: str) -> Optional[str]:
        """
        将注册表中的指标名称映射到策略中的名称
        
        Args:
            registry_name: 注册表中的指标名称
            
        Returns:
            策略中的指标名称，如果没有映射则返回原名称
        """
        mapped_name = self.reverse_mapping.get(registry_name, registry_name)
        
        if mapped_name != registry_name:
            logger.debug(f"反向指标名称映射: {registry_name} -> {mapped_name}")
        
        return mapped_name
    
    def get_all_mappings(self) -> Dict[str, str]:
        """获取所有映射关系"""
        return self.name_mapping.copy()
    
    def add_mapping(self, strategy_name: str, registry_name: str):
        """
        添加新的映射关系
        
        Args:
            strategy_name: 策略中的指标名称
            registry_name: 注册表中的指标名称
        """
        self.name_mapping[strategy_name] = registry_name
        self.reverse_mapping[registry_name] = strategy_name
        logger.info(f"添加新的指标映射: {strategy_name} -> {registry_name}")
    
    def remove_mapping(self, strategy_name: str):
        """
        移除映射关系
        
        Args:
            strategy_name: 策略中的指标名称
        """
        if strategy_name in self.name_mapping:
            registry_name = self.name_mapping[strategy_name]
            del self.name_mapping[strategy_name]
            if registry_name in self.reverse_mapping:
                del self.reverse_mapping[registry_name]
            logger.info(f"移除指标映射: {strategy_name}")
    
    def validate_mapping(self, indicator_registry) -> Dict[str, bool]:
        """
        验证映射关系是否正确
        
        Args:
            indicator_registry: 指标注册表实例
            
        Returns:
            验证结果字典，键为策略名称，值为是否有效
        """
        validation_results = {}
        registered_names = set(indicator_registry.get_indicator_names())
        
        for strategy_name, registry_name in self.name_mapping.items():
            is_valid = registry_name in registered_names
            validation_results[strategy_name] = is_valid
            
            if not is_valid:
                logger.warning(f"映射无效: {strategy_name} -> {registry_name} (未注册)")
        
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        logger.info(f"映射验证完成: {valid_count}/{total_count} 个映射有效")
        
        return validation_results
    
    def get_missing_indicators(self, strategy_indicators: list, indicator_registry) -> list:
        """
        获取策略中使用但未在注册表中找到的指标
        
        Args:
            strategy_indicators: 策略中使用的指标名称列表
            indicator_registry: 指标注册表实例
            
        Returns:
            缺失的指标名称列表
        """
        registered_names = set(indicator_registry.get_indicator_names())
        missing_indicators = []
        
        for indicator_name in strategy_indicators:
            mapped_name = self.map_to_registry_name(indicator_name)
            if mapped_name not in registered_names:
                missing_indicators.append(indicator_name)
        
        if missing_indicators:
            logger.warning(f"发现 {len(missing_indicators)} 个缺失的指标: {missing_indicators}")
        
        return missing_indicators
    
    def print_mapping_summary(self):
        """打印映射摘要"""
        print("\n=== 指标名称映射摘要 ===")
        print(f"总映射数: {len(self.name_mapping)}")
        
        # 按类别分组
        categories = {
            '核心指标': ['VOL', 'SAR', 'KC', 'MTM', 'PSY', 'PVT', 'TRIX', 'VIX', 'VOSC', 'VR', 'WR', 'MACD', 'BOLL', 'KDJ', 'BIAS', 'DMI', 'EMV', 'CMO', 'DMA', 'RSI'],
            '增强指标': ['EnhancedMACD', 'EnhancedTRIX', 'EnhancedKDJ', 'EnhancedOBV', 'EnhancedCCI', 'EnhancedRSI', 'EnhancedWR', 'EnhancedMFI'],
            '形态指标': ['CandlestickPatterns', 'AdvancedCandlestickPatterns', 'ZXMPattern'],
            'ZXM指标': ['TrendDetector', 'TrendDuration', 'ZXMTurnover', 'ZXMVolumeShrink', 'ZXMBSAbsorb', 'AmplitudeElasticity', 'ZXMRiseElasticity', 'Elasticity', 'BounceDetector', 'ZXMElasticityScore', 'ZXMBuyPointScore', 'StockScoreCalculator', 'SelectionModel']
        }
        
        for category, indicators in categories.items():
            print(f"\n{category}:")
            for indicator in indicators:
                if indicator in self.name_mapping:
                    mapped = self.name_mapping[indicator]
                    status = "✅" if indicator != mapped else "➡️"
                    print(f"  {status} {indicator} -> {mapped}")


# 创建全局实例
indicator_name_mapper = Indicator_name_mapper()
