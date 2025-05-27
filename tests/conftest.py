"""
测试配置文件

用于设置测试环境和公共测试夹具
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import patch

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import setup_logger
from utils.path_utils import get_config_dir


# 设置测试日志级别
setup_logger(level="INFO")


@pytest.fixture(scope="session")
def temp_test_dir():
    """创建临时测试目录"""
    temp_dir = tempfile.mkdtemp(prefix="stock_test_")
    yield temp_dir
    # 测试完成后清理
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config_dir(temp_test_dir):
    """创建测试配置目录"""
    config_dir = os.path.join(temp_test_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # 修补get_config_dir函数，返回测试配置目录
    with patch('utils.path_utils.get_config_dir', return_value=config_dir):
        yield config_dir


@pytest.fixture(scope="session")
def test_data_dir(temp_test_dir):
    """创建测试数据目录"""
    data_dir = os.path.join(temp_test_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    yield data_dir


@pytest.fixture(scope="session")
def test_result_dir(temp_test_dir):
    """创建测试结果目录"""
    result_dir = os.path.join(temp_test_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    yield result_dir 