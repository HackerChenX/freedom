"""
兼容性包装器 - 解决方法签名不匹配问题

提供向后兼容的方法包装，确保系统稳定运行
"""

from functools import wraps
from typing import Any, Callable

def flexible_parameters(func: Callable) -> Callable:
    """
    创建灵活参数装饰器，允许不同的参数名
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 处理code/stock_code参数的兼容性
        if 'stock_code' in kwargs and 'code' not in kwargs:
            kwargs['code'] = kwargs.pop('stock_code')
        elif 'code' in kwargs and 'stock_code' in kwargs:
            # 如果两个都有，优先使用code
            kwargs.pop('stock_code', None)
        
        return func(*args, **kwargs)
    
    return wrapper

def container_compatibility_wrapper(container):
    """
    为ServiceContainer添加兼容性方法
    """
    if not hasattr(container, 'get'):
        def get(key: str, default=None):
            try:
                if isinstance(key, str):
                    if key == 'data_access':
                        from db.interfaces.data_access_interface import DataAccessInterface
                        return container.resolve(DataAccessInterface)
                    else:
                        return default
                else:
                    return container.resolve(key)
            except Exception:
                return default
        
        container.get = get
    
    return container
