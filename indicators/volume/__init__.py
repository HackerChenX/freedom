"""
量价类指标包

包含所有与成交量相关的技术指标
"""

# 导入各模块，使其可以通过包名直接引用
# from indicators.volume.enhanced_obv import EnhancedOBV
# from indicators.volume.enhanced_mfi import EnhancedMFI
# from indicators.volume.enhanced_vr import EnhancedVR

# 导入VOL类作为Volume
from .vol import VOL as Volume

# 版本信息
__version__ = '0.1.0' 

__all__ = ['Volume'] 

from __future__ import absolute_import, division, print_function, unicode_literals

# -*- coding: utf-8 -*-
from .ad import AD
from .obv import OBV
from .pvt import PVT
from .vosc import VOSC 