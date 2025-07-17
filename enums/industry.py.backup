"""
行业列表枚举模块
"""

from enum import Enum
from typing import List


class Industry:
    """
    行业分类
    
    提供集中管理的行业列表，避免重复定义
    """
    
    # 所有支持的行业列表
    ALL_INDUSTRIES = [
        '电子元件', '半导体', '互联网', '电子信息', '汽车零部件', '通讯设备', '仪器仪表', 
        '化工行业', '工业机械', '电力行业', '煤炭行业', '化学制药', '贵金属', '家用电器', 
        '游戏', '电池', '塑胶制品', '造纸印刷', '新材料', '有色金属', '保险', '玻璃陶瓷', 
        '水泥建材', '农药兽药', '旅游酒店', '软件服务', '机场航运', '石油行业', '食品饮料', 
        '装修装饰', '园林工程', '安防设备', '公用事业', '电子商务', '船舶制造', '环保工程'
    ]
    
    # 按类别分组的行业
    TECH_INDUSTRIES = ['电子元件', '半导体', '互联网', '电子信息', '通讯设备', '软件服务', '电子商务']
    MANUFACTURING_INDUSTRIES = ['汽车零部件', '工业机械', '家用电器', '电池', '塑胶制品', '仪器仪表']
    RESOURCE_INDUSTRIES = ['化工行业', '电力行业', '煤炭行业', '贵金属', '有色金属', '石油行业', '水泥建材', '新材料']
    CONSUMER_INDUSTRIES = ['食品饮料', '旅游酒店', '游戏']
    
    @classmethod
    def get_all_industries(cls) -> List[str]:
        """
        获取所有行业列表
        
        Returns:
            List[str]: 行业列表
        """
        return cls.ALL_INDUSTRIES.copy()
    
    @classmethod
    def get_tech_industries(cls) -> List[str]:
        """
        获取科技行业列表
        
        Returns:
            List[str]: 科技行业列表
        """
        return cls.TECH_INDUSTRIES.copy()
    
    @classmethod
    def get_manufacturing_industries(cls) -> List[str]:
        """
        获取制造业行业列表
        
        Returns:
            List[str]: 制造业行业列表
        """
        return cls.MANUFACTURING_INDUSTRIES.copy()
    
    @classmethod
    def get_resource_industries(cls) -> List[str]:
        """
        获取资源行业列表
        
        Returns:
            List[str]: 资源行业列表
        """
        return cls.RESOURCE_INDUSTRIES.copy()
    
    @classmethod
    def get_consumer_industries(cls) -> List[str]:
        """
        获取消费行业列表
        
        Returns:
            List[str]: 消费行业列表
        """
        return cls.CONSUMER_INDUSTRIES.copy() 