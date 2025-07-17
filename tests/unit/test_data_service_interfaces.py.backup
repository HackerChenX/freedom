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

from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.cache_interface import ICache_manager
from db.interfaces.connection_interface import IconnectionManager
from db.container import Service_container, Lifecycle_type, get_container, reset_container


class Test_service_container(unittest.Test_case):
    """测试服务容器"""
    
    def set_up_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
        """测试前准备"""
        self.container = Service_container()
    
    def tear_down_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
        """测试后清理"""
        reset_container()
    
    def test_register_and_resolve_transient(self):
        """测试注册和解析瞬态服务"""
        
        # 创建模拟接口和实现
        class Itestservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces:
            pass
        
        class Testservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces(ITest_service):
            def __init__(self):
                self.created_time = id(self)
        
        # 注册服务
        self.container.register_transient(ITest_service, Test_service)
        
        # 解析服务
        service1 = self.get_service(Data_access_interface)
        service2 = self.get_service(Data_access_interface)
        
        # 验证
        self.assert_is_instance(service1, Test_service)
        self.assert_is_instance(service2, Test_service)
        self.assert_not_equal(service1.created_time, service2.created_time)  # 瞬态服务每次创建新实例
    
    def test_register_and_resolve_singleton(self):
        """测试注册和解析单例服务"""
        
        # 创建模拟接口和实现
        class Itestservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces:
            pass
        
        class Testservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces(ITest_service):
            def __init__(self):
                self.created_time = id(self)
        
        # 注册单例服务
        self.container.register_singleton(ITest_service, Test_service)
        
        # 解析服务
        service1 = self.get_service(Data_access_interface)
        service2 = self.get_service(Data_access_interface)
        
        # 验证
        self.assert_is_instance(service1, Test_service)
        self.assert_is_instance(service2, Test_service)
        self.assert_equal(service1.created_time, service2.created_time)  # 单例服务返回同一实例
    
    def test_register_with_factory(self):
        """测试使用工厂方法注册服务"""
        
        class Itestservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces:
            pass
        
        class Testservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces(ITest_service):
            def __init__(self, value):
                self.value = value
        
        # 使用工厂方法注册
        def factory():
            return TestService_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_InterfacesTestdataserviceinterfaces("factory_created")
        
        self.container.register(ITest_service, factory=factory)
        
        # 解析服务
        service = self.get_service(Data_access_interface)
        
        # 验证
        self.assert_is_instance(service, Test_service)
        self.assertEqual(service.value, "factory_created")
    
    def test_register_with_instance(self):
        """测试使用实例注册服务"""
        
        class Itestservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces:
            pass
        
        class Testservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces(ITest_service):
            def __init__(self, value):
                self.value = value
        
        # 创建实例
        instance = TestService_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_InterfacesTestdataserviceinterfaces("instance_value")
        
        # 使用实例注册
        self.container.register(ITest_service, instance=instance, lifecycle=Lifecycle_type.SINGLETON)
        
        # 解析服务
        service = self.get_service(Data_access_interface)
        
        # 验证
        self.assert_is(service, instance)
        self.assertEqual(service.value, "instance_value")
    
    def test_resolve_unregistered_service(self):
        """测试解析未注册的服务"""
        
        class IUnregistered_service:
            pass
        
        # 验证抛出异常
        with self.assert_raises(Value_error) as context:
            self.get_service(Data_access_interface)
        
        self.assertIn("服务未注册", str(context.exception))
    
    def test_is_registered(self):
        """测试检查服务是否已注册"""
        
        class Itestservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces:
            pass
        
        class Testservice_interfaces_test_data_service_interfaces_test_data_service_interfacestestdataserviceinterfaces(ITest_service):
            pass
        
        # 初始状态未注册
        self.assert_false(self.container.is_registered(ITest_service))
        
        # 注册后已注册
        self.container.register_transient(ITest_service, Test_service)
        self.assert_true(self.container.is_registered(ITest_service))


class Test_data_access_interface(unittest.Test_case):
    """测试数据访问接口"""
    
    def test_interface_methods_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
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
            self.assert_true(hasattr(DataAccessInterface, method_name), 
                          f"IDataAccess接口缺少方法: {method_name}")


class Test_cache_interface(unittest.Test_case):
    """测试缓存接口"""
    
    def test_interface_methods_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
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
            self.assert_true(hasattr(ICache_manager, method_name), 
                          f"ICacheManager接口缺少方法: {method_name}")


class Test_connection_interface(unittest.Test_case):
    """测试连接接口"""
    
    def test_interface_methods_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
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
            self.assert_true(hasattr(IconnectionManager, method_name), 
                          f"IConnectionManager接口缺少方法: {method_name}")


class Test_global_container(unittest.Test_case):
    """测试全局容器"""
    
    def set_up_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
        """测试前准备"""
        reset_container()
    
    def tear_down_Interfaces_Test_Data_Service_Interfaces_Test_Data_Service_Interfaces_testdataserviceinterfaces(self):
        """测试后清理"""
        reset_container()
    
    @patch('db.container._setup_default_services')
    def test_get_container_singleton(self, mock_setup):
        """测试全局容器单例模式"""
        
        # 获取容器实例
        container1 = get_container()
        container2 = get_container()
        
        # 验证是同一个实例
        self.assert_is(container1, container2)
        
        # 验证设置默认服务被调用
        mock_setup.assert_called_once()


if __name__ == '__main__':
    unittest.main() 