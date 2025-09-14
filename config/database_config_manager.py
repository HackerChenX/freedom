from db.query_executor import get_query_executor
from db.sql_manager import QueryType
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from config import get_config
"""
统一的数据库配置管理器

提供Click_house数据库配置的统一管理，支持：
1. 环境变量优先级配置
2. 配置文件加载
3. 密码加密存储
4. 配置验证和默认值
5. 多环境配置支持
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import getpass
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

class DatabaseConfigManager:
    """统一的数据库配置管理器"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        'host': get_config('db.host', 'localhost'),
        'port': get_config('db.port', 9000),
        'database': get_config('db.database', 'stock'),
        'user': get_config('db.user', 'default'),
        'password': get_config('db.password', '123456'),
        'timeout': 30,
        'compression': True,
        'pool': {
            'min_size': 5,
            'max_size': 20,
            'timeout': 30,
            'max_overflow': 10
        },
        'query': {
            'timeout': 30,
            'max_rows': 1000000,
            'max_memory_usage': 20000000000
        },
        'cache': {
            'enabled': True,
            'ttl': 3600,
            'max_size': 1000
        }
    }
    
    # 环境变量映射
    ENV_MAPPING = {
        'host': 'CLICKHOUSE_HOST',
        'port': 'CLICKHOUSE_PORT',
        'database': 'CLICKHOUSE_DATABASE',
        'user': 'CLICKHOUSE_USER',
        'password': 'CLICKHOUSE_PASSWORD',
        'timeout': 'CLICKHOUSE_TIMEOUT'
    }
    
    def __init__(self, config_dir: str = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config
        """
        if config_dir is None:
            # 获取项目根目录
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_dir = project_root / 'config'
        
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / 'database.yaml'
        self.user_config_file = self.config_dir / 'user_config.json'
        self.env_file = self.config_dir / '.env'
        
        # 加密密钥文件
        self.key_file = self.config_dir / '.db_key'
        
        # 当前配置
        self._config = None
        self._encryption_key = None
        
        # 加载配置
        self._load_config_Database_Config_Manager()
    
    def _get_encryption_key(self) -> bytes:
        """获取或生成加密密钥"""
        if self._encryption_key is not None:
            return self._encryption_key
        
        # 尝试从环境变量获取
        env_key = os.environ.get('CLICKHOUSE_ENCRYPTION_KEY')
        if env_key:
            try:
                self._encryption_key = base64.urlsafe_b64decode(env_key)
                return self._encryption_key
            except Exception:
                logger.warning("环境变量中的加密密钥格式无效")
        
        # 尝试从文件加载
        if self.key_file.exists():
            try:
                with open(self.key_file, 'rb') as f:
                    self._encryption_key = base64.urlsafe_b64decode(f.read())
                return self._encryption_key
            except Exception as e:
                logger.warning(f"无法从文件加载加密密钥: {e}")
        
        # 生成新密钥
        self._encryption_key = Fernet.generate_key()
        
        # 保存密钥到文件
        try:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.key_file, 'wb') as f:
                f.write(base64.urlsafe_b64encode(self._encryption_key))
            # 设置文件权限为仅所有者可读写
            os.chmod(self.key_file, 0o600)
            logger.info(f"生成新的加密密钥并保存到: {self.key_file}")
        except Exception as e:
            logger.error(f"保存加密密钥失败: {e}")
        
        return self._encryption_key
    
    def _encrypt_password(self, password: str) -> str:
        """加密密码"""
        if not password:
            return password
        
        key = self._get_encryption_key()
        fernet = Fernet(key)
        encrypted = fernet.encrypt(password.encode())
        return f"ENC:{base64.urlsafe_b64encode(encrypted).decode()}"
    
    def _decrypt_password(self, encrypted_password: str) -> str:
        """解密密码"""
        if not encrypted_password or not encrypted_password.startswith("ENC:"):
            return encrypted_password
        
        try:
            key = self._get_encryption_key()
            fernet = Fernet(key)
            encrypted_data = base64.urlsafe_b64decode(encrypted_password[4:])
            return fernet.decrypt(encrypted_data).decode()
        except Exception as e:
            logger.error(f"解密密码失败: {e}")
            return ""
    
    def _load_env_file(self):
        """加载.env文件"""
        if not self.env_file.exists():
            return
        
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            logger.debug(f"已加载环境变量文件: {self.env_file}")
        except Exception as e:
            logger.warning(f"加载环境变量文件失败: {e}")
    
    def _load_config_Database_Config_Manager(self):
        """加载配置"""
        # 1. 从默认配置开始
        config = self.DEFAULT_CONFIG.copy()
        
        # 2. 加载.env文件
        self._load_env_file()
        
        # 3. 加载YAML配置文件
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config and 'clickhouse' in yaml_config:
                        config.update(yaml_config['clickhouse'])
                logger.debug(f"已加载YAML配置文件: {self.config_file}")
            except Exception as e:
                logger.warning(f"加载YAML配置文件失败: {e}")
        
        # 4. 加载JSON用户配置文件
        if self.user_config_file.exists():
            try:
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    if user_config and 'db' in user_config:
                        # 将JSON配置映射到ClickHouse配置
                        db_config = user_config['db']
                        for key in ['host', 'port', 'database', 'user', 'password']:
                            if key in db_config:
                                config[key] = db_config[key]
                logger.debug(f"已加载JSON用户配置文件: {self.user_config_file}")
            except Exception as e:
                logger.warning(f"加载JSON用户配置文件失败: {e}")
        
        # 5. 环境变量覆盖（优先级最高）
        for config_key, env_key in self.ENV_MAPPING.items():
            env_value = os.environ.get(env_key)
            if env_value is not None:
                # 类型转换
                if config_key == 'port':
                    try:
                        config[config_key] = int(env_value)
                    except ValueError:
                        logger.warning(f"环境变量 {env_key} 的值 '{env_value}' 不是有效的端口号")
                else:
                    config[config_key] = env_value
                logger.debug(f"从环境变量 {env_key} 加载配置: {config_key}")
        
        # 6. 处理密码解密
        if config.get('password') and config['password'].startswith("ENC:"):
            config['password'] = self._decrypt_password(config['password'])
        
        self._config = config
    
    def get_config_Manager(self) -> Dict[str, Any]:
        """获取完整的数据库配置"""
        if self._config is None:
            self._load_config_Database_Config_Manager()
        return self._config.copy()
    
    def get_connection_config(self) -> Dict[str, Any]:
        """获取连接配置（仅包含连接相关参数）"""
        config = self.get_config_Manager()
        # 过滤掉ClickHouse客户端不支持的参数
        clickhouse_config = {
            'host': config['host'],
            'port': config['port'],
            'database': config['database'],
            'user': config['user'],
            'password': config['password'],
            'compression': False  # 禁用压缩以避免lz4依赖问题
        }

        # 添加ClickHouse客户端支持的其他参数
        if 'settings' in config:
            clickhouse_config['settings'] = config['settings']

        return clickhouse_config
    
    def set_password(self, password: str, encrypt: bool = True, save_to_file: bool = True):
        """
        设置数据库密码
        
        Args:
            password: 密码
            encrypt: 是否加密存储
            save_to_file: 是否保存到配置文件
        """
        if encrypt:
            encrypted_password = self._encrypt_password(password)
        else:
            encrypted_password = password
        
        # 更新内存中的配置
        if self._config is None:
            self._load_config_Database_Config_Manager()
        self._config['password'] = password  # 内存中保存明文
        
        # 保存到文件
        if save_to_file:
            self._save_password_to_file(encrypted_password)
    
    def _save_password_to_file(self, password: str):
        """保存密码到配置文件"""
        try:
            # 更新YAML文件
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                
                if 'clickhouse' not in yaml_config:
                    yaml_config['clickhouse'] = {}
                
                yaml_config['clickhouse']['password'] = password
                
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
                
                logger.info("密码已保存到YAML配置文件")
            
            # 更新JSON文件
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                if 'db' not in user_config:
                    user_config['db'] = {}
                
                user_config['db']['password'] = password
                
                with open(self.user_config_file, 'w', encoding='utf-8') as f:
                    json.dump(user_config, f, ensure_ascii=False, indent=4)
                
                logger.info("密码已保存到JSON配置文件")
                
        except Exception as e:
            logger.error(f"保存密码到配置文件失败: {e}")
    
    def request_password_interactive(self) -> str:
        """交互式请求密码"""
        if self._config and self._config.get('password'):
            return self._config['password']
        
        print("请输入ClickHouse数据库密码:")
        password = getpass.getpass()
        
        if password:
            # 询问是否保存密码
            save_choice = input("是否保存密码到配置文件? (y/N): ").lower()
            if save_choice in ['y', 'yes']:
                encrypt_choice = input("是否加密存储密码? (Y/n): ").lower()
                encrypt = encrypt_choice not in ['n', 'no']
                self.set_password(password, encrypt=encrypt, save_to_file=True)
            else:
                # 仅在内存中保存
                if self._config is None:
                    self._load_config_Database_Config_Manager()
                self._config['password'] = password
        
        return password
    
    def validate_config_Manager(self) -> bool:
        """验证配置的有效性"""
        config = self.get_config_Manager()
        
        # 检查必需的配置项
        required_fields = ['host', 'port', 'database', 'user']
        for field in required_fields:
            if not config.get(field):
                logger.error(f"缺少必需的配置项: {field}")
                return False
        
        # 检查端口号
        try:
            port = int(config['port'])
            if not (1 <= port <= 65535):
                logger.error(f"端口号无效: {port}")
                return False
        except (ValueError, TypeError):
            logger.error(f"端口号格式错误: {config['port']}")
            return False
        
        return True
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            from clickhouse_driver import Client
            
            config = self.get_connection_config()
            client = Client(**config)
            
            # 执行简单查询测试连接
            result = client.execute("SELECT 1")
            return result == [(1,)]
            
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def reload_config(self):
        """重新加载配置"""
        self._config = None
        self._load_config_Database_Config_Manager()
        logger.info("配置已重新加载")


# 全局配置管理器实例
_db_config_manager = None

def get_database_config_manager() -> DatabaseConfigManager:
    """获取全局数据库配置管理器实例"""
    global _db_config_manager
    if _db_config_manager is None:
        _db_config_manager = DatabaseConfigManager()
    return _db_config_manager

def get_clickhouse_config() -> Dict[str, Any]:
    """获取ClickHouse配置的便捷函数"""
    return get_database_config_manager().get_config_Manager()

def get_clickhouse_connection_config() -> Dict[str, Any]:
    """获取ClickHouse连接配置的便捷函数"""
    return get_database_config_manager().get_connection_config()
