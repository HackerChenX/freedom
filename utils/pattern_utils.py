from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class PatternType(Enum):
    """形态类型枚举"""
    BULLISH = "bullish"  # 看涨
    BEARISH = "bearish"  # 看跌
    REVERSAL = "reversal"  # 反转
    CONTINUATION = "continuation"  # 持续
    NEUTRAL = "neutral"  # 中性

@dataclass
class PatternResult:
    """形态识别结果类"""
    pattern_name: str  # 形态名称
    pattern_type: PatternType  # 形态类型
    start_idx: int  # 形态开始位置
    end_idx: int  # 形态结束位置
    strength: float  # 形态强度 (0-1)
    reliability: float  # 形态可靠性 (0-1)
    description: str  # 形态描述
    parameters: Dict[str, Any]  # 形态参数
    signals: List[Dict[str, Any]]  # 形态信号列表
    
    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.pattern_type, PatternType):
            self.pattern_type = PatternType(self.pattern_type)
            
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type.value,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "strength": self.strength,
            "reliability": self.reliability,
            "description": self.description,
            "parameters": self.parameters,
            "signals": self.signals
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternResult':
        """从字典创建实例"""
        return cls(
            pattern_name=data["pattern_name"],
            pattern_type=data["pattern_type"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            strength=data["strength"],
            reliability=data["reliability"],
            description=data["description"],
            parameters=data["parameters"],
            signals=data["signals"]
        ) 