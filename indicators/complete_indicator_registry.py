"""
简化的指标注册管理器
避免复杂的依赖和常量问题
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SimpleIndicatorRegistry:
    """简化的指标注册管理器"""
    
    def __init__(self):
        self._indicators = {}
        
    def register_core_indicators(self):
        """注册核心指标"""
        logger.info("=== 开始注册核心指标 ===")
        
        # 只注册确实存在的指标
        core_indicators = {
            'MA': 'indicators.ma.MA',
            'EMA': 'indicators.ema.EMA', 
            'MACD': 'indicators.macd.MACD',
            'RSI': 'indicators.rsi.RSI',
            'BOLL': 'indicators.boll.BOLL',
            'PSY': 'indicators.psy.PSY',
        }
        
        for name, class_path in core_indicators.items():
            try:
                self._indicators[name] = class_path
                logger.info(f"✅ 成功注册指标: {name}")
            except Exception as e:
                logger.error(f"❌ 注册指标失败 {name}: {e}")
        
        logger.info(f"核心指标注册完成: {len(self._indicators)}/{len(core_indicators)}")
        
    def get_indicator(self, name: str):
        """获取指标"""
        return self._indicators.get(name)
    
    def get_all_indicators(self) -> Dict[str, Any]:
        """获取所有指标"""
        return self._indicators.copy()

# 创建全局实例
complete_registry = SimpleIndicatorRegistry()

# 执行注册
def initialize_indicators():
    """初始化指标"""
    try:
        complete_registry.register_core_indicators()
        logger.info("指标注册完成")
    except Exception as e:
        logger.error(f"指标注册失败: {e}")

# 自动初始化
initialize_indicators()
