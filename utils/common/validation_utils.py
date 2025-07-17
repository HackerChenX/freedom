#!/usr/bin/env python3
"""
通用验证工具类
统一项目中的各种验证逻辑
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re

class ValidationUtils:
    """通用验证工具类"""
    
    @staticmethod
    def validate_stock_code(code: str) -> bool:
        """验证股票代码格式"""
        if not code or not isinstance(code, str):
            return False
        return bool(re.match(r'^[0-9]{6}$', code))
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> bool:
        """验证日期范围"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            return start <= end
        except ValueError:
            return False
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: float = None, max_val: float = None) -> bool:
        """验证数值范围"""
        if not isinstance(value, (int, float)):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> bool:
        """验证必填字段"""
        return all(field in data and data[field] is not None for field in required_fields)
