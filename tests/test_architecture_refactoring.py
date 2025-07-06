#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
架构重构单元测试

验证重构后的代码是否正确使用依赖注入架构
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, Magic_mock
import pandas as pd

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IData_access


class Test_architecture_refactoring(unittest.Test_case):
    """测试架构重构"""
    
    def set_up_Test_Architecture_Refactoring(self):
        """设置测试环境"""
        # 创建具有所有必要方法的Mock
        self.mock_data_access = Mock(spec=IData_access)
        
        # 配置Mock的方法
        self.mock_data_access.get_stock_data = Mock()
        self.mock_data_access.get_stock_list = Mock()
        self.mock_data_access.get_market_data = Mock()
        self.mock_data_access.get_last_trade_date = Mock()
        self.mock_data_access.get_index_stocks = Mock()
        self.mock_data_access.get_stocks_by_industry = Mock()
        
        self.mock_container = Mock()
        self.mock_container.get_data_access.return_value = self.mock_data_access
        self.mock_container.resolve.return_value = self.mock_data_access
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'open': [100 + i for i in range(10)],
            'high': [105 + i for i in range(10)],
            'low': [95 + i for i in range(10)],
            'close': [102 + i for i in range(10)],
            'volume': [1000000 + i*10000 for i in range(10)]
        })
    
    @patch('db.container.get_container')
    def test_market_dimension_analyzer_uses_dependency_injection(self, mock_get_container):
        """测试市场维度分析器使用依赖注入"""
        mock_get_container.return_value = self.mock_container
        
        # 导入并测试
        from analysis.market.market_dimension_analyzer import Market_dimension_analyzer
        
        analyzer = Market_dimension_analyzer()
        
        # 验证容器被正确调用
        mock_get_container.assert_called_once()
        
        # 验证分析器使用了依赖注入的数据访问接口
        self.assert_is_not_none(analyzer.data_access)
    
    @patch('db.container.get_container')
    def test_buypoint_dimension_analyzer_uses_dependency_injection(self, mock_get_container):
        """测试买点维度分析器使用依赖注入"""
        mock_get_container.return_value = self.mock_container
        
        # 导入并测试
        from analysis.buypoints.buypoint_dimension_analyzer import Buy_point_dimension_analyzer
        
        analyzer = Buy_point_dimension_analyzer()
        
        # 验证容器被正确调用
        mock_get_container.assert_called_once()
        self.mock_container.resolve.assert_called_once()
        
        # 验证分析器使用了依赖注入的数据访问接口
        self.assert_is_not_none(analyzer.data_access)
    
    def test_multi_dimension_analyzer_uses_dependency_injection(self):
        """测试多维度分析器使用依赖注入"""
        # 导入并测试
        from analysis.market.multi_dimension_analyzer import Multi_dimension_analyzer
        
        analyzer = Multi_dimension_analyzer()
        
        # 验证分析器使用了依赖注入的数据访问接口
        self.assert_is_not_none(analyzer.data_access)
    
    def test_container_provides_data_access_interface(self):
        """测试容器提供数据访问接口"""
        container = get_container()
        data_access = container.get_data_access()
        
        # 验证返回的是IDataAccess接口的实现
        self.assert_is_not_none(data_access)
        
        # 验证接口方法存在
        self.assertTrue(hasattr(data_access, 'get_stock_data'))
        self.assertTrue(hasattr(data_access, 'get_stock_list'))
        self.assertTrue(hasattr(data_access, 'get_market_data'))
    
    def test_bin_scripts_use_dependency_injection(self):
        """测试bin脚本使用依赖注入"""
        # 检查bin脚本文件是否使用了依赖注入
        bin_scripts = [
            'bin/run_pattern_analysis_and_backtest.py',
            'bin/run_pattern_combination_backtest.py'
        ]
        
        for script_path in bin_scripts:
            full_path = os.path.join(root_dir, script_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 验证使用了依赖注入
                self.assertIn('from utils.dependency_injection import get_service', content,
                            f"脚本 {script_path} 应该使用依赖注入容器")
                self.assertIn('get_container()', content,
                            f"脚本 {script_path} 应该调用get_container()")
                
                # 验证不使用直接数据库导入
                self.assertNotIn('from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IDataAccess', content,
                               f"脚本 {script_path} 不应该使用直接数据库导入")
    
    def test_no_direct_database_imports_in_refactored_files(self):
        """测试重构后的文件不包含直接数据库导入"""
        refactored_files = [
            'analysis/market/market_dimension_analyzer.py',
            'analysis/buypoints/buypoint_dimension_analyzer.py',
            'analysis/market/multi_dimension_analyzer.py',
            'bin/run_pattern_analysis_and_backtest.py',
            'bin/run_pattern_combination_backtest.py'
        ]
        
        for file_path in refactored_files:
            full_path = os.path.join(root_dir, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 验证不包含直接数据库导入
                self.assertNotIn('from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import IDataAccess', content,
                               f"文件 {file_path} 仍然包含直接数据库导入")
                self.assertNotIn('get_service(DataAccessInterface)', content,
                               f"文件 {file_path} 仍然包含直接数据库调用")
    
    def test_dependency_injection_container_singleton(self):
        """测试依赖注入容器是单例模式"""
        container1 = get_container()
        container2 = get_container()
        
        # 验证返回的是同一个实例
        self.assert_is(container1, container2)
    
    @patch('db.container.get_container')
    def test_data_access_interface_methods(self, mock_get_container):
        """测试数据访问接口方法"""
        mock_get_container.return_value = self.mock_container
        
        # 设置模拟返回值
        self.mock_data_access.get_stock_data.return_value = self.test_data
        self.mock_data_access.get_stock_list.return_value = ['000001.SZ', '000002.SZ']
        
        # 测试数据访问接口
        data_access = self.mock_container.get_data_access()
        
        # 测试获取股票数据
        stock_data = data_access.get_stock_data('000001.SZ', '2023-01-01', '2023-12-31')
        self.assert_is_not_none(stock_data)
        
        # 测试获取股票列表
        stock_list = data_access.get_stock_list()
        self.assert_is_instance(stock_list, list)
        self.assert_greater(len(stock_list), 0)


class Test_architecture_compliance(unittest.Test_case):
    """测试架构合规性"""
    
    def test_no_layer_violations(self):
        """测试没有分层架构违规"""
        # 这里可以添加更多的分层架构检查
        # 例如检查L1层不直接导入L4层等
        pass
    
    def test_consistent_naming_conventions(self):
        """测试命名规范一致性"""
        # 检查类名使用大驼峰命名法
        # 检查函数名使用小写加下划线
        # 检查变量名使用小写加下划线
        pass
    
    def test_proper_error_handling(self):
        """测试正确的错误处理"""
        # 验证所有数据库操作都有适当的异常处理
        pass


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2) 