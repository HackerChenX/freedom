"""
完整的服务初始化器
"""

import logging
from typing import Optional
from utils.dependency_injection import ServiceContainer, get_container
from utils.logger import get_logger

logger = get_logger(__name__)

def initialize_all_services() -> ServiceContainer:
    """
    初始化所有必需的服务
    """
    container = get_container()
    
    try:
        logger.info("开始初始化所有服务...")
        
        # 清空现有配置
        container.clear()
        
        # 1. 注册数据访问接口
        logger.info("注册数据访问接口...")
        try:
            from db.interfaces.data_access_interface import DataAccessInterface
            from db.data_access_manager import DataAccessManager
            
            def create_data_access_manager():
                return DataAccessManager()
            
            container.register_singleton(DataAccessInterface, factory=create_data_access_manager)
            logger.info("  ✅ 数据访问接口注册成功")
        except Exception as e:
            logger.warning(f"  ⚠️ 数据访问接口注册失败: {e}")
        
        # 2. 注册指标计算器接口
        logger.info("注册指标计算器接口...")
        try:
            from db.interfaces.indicator_calculator_interface import IindicatorCalculator
            from indicators.indicator_calculator import IndicatorCalculator
            
            def create_indicator_calculator():
                return IndicatorCalculator()
            
            container.register_singleton(IindicatorCalculator, factory=create_indicator_calculator)
            logger.info("  ✅ 指标计算器接口注册成功")
        except Exception as e:
            logger.warning(f"  ⚠️ 指标计算器接口注册失败: {e}")
        
        # 3. 注册缓存服务
        logger.info("注册缓存服务...")
        try:
            from utils.cache import MemoryCache
            container.register_singleton(MemoryCache, factory=lambda: MemoryCache())
            logger.info("  ✅ 缓存服务注册成功")
        except Exception as e:
            logger.warning(f"  ⚠️ 缓存服务注册失败: {e}")
        
        # 4. 注册数据库管理器
        logger.info("注册数据库管理器...")
        try:
            from db.db_manager import DBManager
            
            def create_db_manager():
                try:
                    data_access = container.resolve(DataAccessInterface)
                    return DBManager(data_access=data_access)
                except:
                    return DBManager()
            
            container.register_singleton(DBManager, factory=create_db_manager)
            logger.info("  ✅ 数据库管理器注册成功")
        except Exception as e:
            logger.warning(f"  ⚠️ 数据库管理器注册失败: {e}")
        
        logger.info(f"服务初始化完成，共注册 {len(container._services)} 个服务")
        return container
        
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        # 返回部分初始化的容器而不是抛出异常
        return container

def ensure_services_registered() -> bool:
    """确保关键服务已注册"""
    container = get_container()
    
    registered_count = 0
    total_services = 4
    
    try:
        from db.interfaces.data_access_interface import DataAccessInterface
        if container.is_registered(DataAccessInterface):
            registered_count += 1
    except:
        pass
    
    try:
        from db.interfaces.indicator_calculator_interface import IindicatorCalculator
        if container.is_registered(IindicatorCalculator):
            registered_count += 1
    except:
        pass
    
    try:
        from utils.cache import MemoryCache
        if container.is_registered(MemoryCache):
            registered_count += 1
    except:
        pass
    
    try:
        from db.db_manager import DBManager
        if container.is_registered(DBManager):
            registered_count += 1
    except:
        pass
    
    logger.info(f"已注册服务: {registered_count}/{total_services}")
    return registered_count >= 2  # 至少注册2个核心服务

if __name__ == "__main__":
    try:
        container = initialize_all_services()
        success = ensure_services_registered()
        print(f"服务初始化: {'成功' if success else '部分成功'}")
    except Exception as e:
        print(f"服务初始化失败: {e}")
