"""
数据服务接口层单元测试

测试接口定义和依赖注入容器的功能
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.interfaces.data_access_interface import IDataAccess
from db.interfaces.cache_interface import ICacheManager
from db.interfaces.connection_interface import IConnectionManager
from db.container import ServiceContainer, LifecycleType, get_container, reset_container


class TestServiceContainer(unittest.TestCase):
    """测试服务容器"""
    
    def setUp(self):
        """测试前准备"""
        self.container = ServiceContainer()
    
    def tearDown(self):
        """测试后清理"""
        reset_container()
    
    def test_register_and_resolve_transient(self):
        """测试注册和解析瞬态服务"""
        
        # 创建模拟接口和实现
        class ITestService:
            pass
        
        class TestService(ITestService):
            def __init__(self):
                self.created_time = id(self)
        
        # 注册服务
        self.container.register_transient(ITestService, TestService)
        
        # 解析服务
        service1 = self.container.resolve(ITestService)
        service2 = self.container.resolve(ITestService)
        
        # 验证
        self.assertIsInstance(service1, TestService)
        self.assertIsInstance(service2, TestService)
        self.assertNotEqual(service1.created_time, service2.created_time)  # 瞬态服务每次创建新实例
    
    def test_register_and_resolve_singleton(self):
        """测试注册和解析单例服务"""
        
        # 创建模拟接口和实现
        class ITestService:
            pass
        
        class TestService(ITestService):
            def __init__(self):
                self.created_time = id(self)
        
        # 注册单例服务
        self.container.register_singleton(ITestService, TestService)
        
        # 解析服务
        service1 = self.container.resolve(ITestService)
        service2 = self.container.resolve(ITestService)
        
        # 验证
        self.assertIsInstance(service1, TestService)
        self.assertIsInstance(service2, TestService)
        self.assertEqual(service1.created_time, service2.created_time)  # 单例服务返回同一实例
    
    def test_register_with_factory(self):
        """测试使用工厂方法注册服务"""
        
        class ITestService:
            pass
        
        class TestService(ITestService):
            def __init__(self, value):
                self.value = value
        
        # 使用工厂方法注册
        def factory():
            return TestService("factory_created")
        
        self.container.register(ITestService, factory=factory)
        
        # 解析服务
        service = self.container.resolve(ITestService)
        
        # 验证
        self.assertIsInstance(service, TestService)
        self.assertEqual(service.value, "factory_created")
    
    def test_register_with_instance(self):
        """测试使用实例注册服务"""
        
        class ITestService:
            pass
        
        class TestService(ITestService):
            def __init__(self, value):
                self.value = value
        
        # 创建实例
        instance = TestService("instance_value")
        
        # 使用实例注册
        self.container.register(ITestService, instance=instance, lifecycle=LifecycleType.SINGLETON)
        
        # 解析服务
        service = self.container.resolve(ITestService)
        
        # 验证
        self.assertIs(service, instance)
        self.assertEqual(service.value, "instance_value")
    
    def test_resolve_unregistered_service(self):
        """测试解析未注册的服务"""
        
        class IUnregisteredService:
            pass
        
        # 验证抛出异常
        with self.assertRaises(ValueError) as context:
            self.container.resolve(IUnregisteredService)
        
        self.assertIn("服务未注册", str(context.exception))
    
    def test_is_registered(self):
        """测试检查服务是否已注册"""
        
        class ITestService:
            pass
        
        class TestService(ITestService):
            pass
        
        # 初始状态未注册
        self.assertFalse(self.container.is_registered(ITestService))
        
        # 注册后已注册
        self.container.register_transient(ITestService, TestService)
        self.assertTrue(self.container.is_registered(ITestService))


class TestDataAccessInterface(unittest.TestCase):
    """测试数据访问接口"""
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        
        # 验证接口有必要的方法
        required_methods = [
            'get_stock_info',
            'get_stock_list', 
            'get_industry_list',
            'query',
            'execute',
            'test_connection',
            'get_stock_max_date',
            'get_industry_max_date',
            'get_avg_price'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(IDataAccess, method_name), 
                          f"IDataAccess接口缺少方法: {method_name}")


class TestCacheInterface(unittest.TestCase):
    """测试缓存接口"""
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        
        # 验证接口有必要的方法
        required_methods = [
            'get',
            'set',
            'delete',
            'exists',
            'clear',
            'get_stats'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(ICacheManager, method_name), 
                          f"ICacheManager接口缺少方法: {method_name}")


class TestConnectionInterface(unittest.TestCase):
    """测试连接接口"""
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        
        # 验证接口有必要的方法
        required_methods = [
            'get_connection',
            'release_connection',
            'test_connection',
            'get_connection_stats',
            'close_all_connections'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(IConnectionManager, method_name), 
                          f"IConnectionManager接口缺少方法: {method_name}")


class TestGlobalContainer(unittest.TestCase):
    """测试全局容器"""
    
    def setUp(self):
        """测试前准备"""
        reset_container()
    
    def tearDown(self):
        """测试后清理"""
        reset_container()
    
    @patch('db.container._setup_default_services')
    def test_get_container_singleton(self, mock_setup):
        """测试全局容器单例模式"""
        
        # 获取容器实例
        container1 = get_container()
        container2 = get_container()
        
        # 验证是同一个实例
        self.assertIs(container1, container2)
        
        # 验证设置默认服务被调用
        mock_setup.assert_called_once()


if __name__ == '__main__':
    unittest.main() 