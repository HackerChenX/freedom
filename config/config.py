"""
配置文件，集中管理项目的所有配置项
提供配置加密和环境变量支持
"""

import os
import json
import logging
import base64
from typing import Dict, Any, Optional, List, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# 默认配置
DEFAULT_CONFIG = {
    'db': {
        'host': 'localhost',
        'port': 8123, 
        'user': 'default',
        'password': '123456',  # 密码已隐藏，将从环境变量或用户输入获取
        'database': 'stock'
    },
    'paths': {
        'output': os.path.expanduser('~/Documents/StockResults/'),
        'doc': 'doc',
        'logs': 'logs'
    },
    'dates': {
        'default_start': '20000101',
        'default_end': '20241231'
    },
    'stock': {
        'default_date': '20230316'
    },
    'api': {
        'retry_times': 3,
        'timeout': 30
    },
    'log': {
        'level': 'info',
        'max_size_mb': 10,
        'backup_count': 5
    },
    'security': {
        'encrypt_sensitive': True,
        'key_file': '.config_key'
    }
}

# 必需的配置项定义，用于验证配置是否完整
REQUIRED_CONFIG = {
    'db': ['host', 'port', 'user', 'database'],
    'paths': ['output'],
    'dates': ['default_start', 'default_end'],
    'log': ['level']
}

# 敏感配置项列表，这些项将被加密存储
SENSITIVE_CONFIG = [
    'db.password'
]

# 用户配置文件路径
USER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'user_config.json')
# 加密密钥文件路径
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), '.config_key')

# 合并配置
CONFIG = DEFAULT_CONFIG.copy()

class ConfigError(Exception):
    """配置错误异常类"""
    pass

def get_encryption_key() -> bytes:
    """
    获取或生成加密密钥
    
    Returns:
        bytes: 加密密钥
    """
    # 尝试从环境变量获取密钥
    env_key = os.environ.get('STOCK_CONFIG_KEY')
    if env_key:
        try:
            return base64.urlsafe_b64decode(env_key)
        except Exception:
            logging.warning("环境变量中的加密密钥格式无效，将生成新密钥")
    
    # 尝试从文件加载密钥
    if os.path.exists(KEY_FILE_PATH):
        try:
            with open(KEY_FILE_PATH, 'rb') as f:
                return base64.urlsafe_b64decode(f.read())
        except Exception as e:
            logging.warning(f"无法从文件加载加密密钥: {e}，将生成新密钥")
    
    # 生成新密钥
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
    
    # 保存密钥到文件
    try:
        with open(KEY_FILE_PATH, 'wb') as f:
            f.write(key)
        os.chmod(KEY_FILE_PATH, 0o600)  # 设置文件权限为仅所有者可读写
    except Exception as e:
        logging.warning(f"无法保存加密密钥: {e}")
    
    return base64.urlsafe_b64decode(key)

def encrypt_value(value: str) -> str:
    """
    加密配置值
    
    Args:
        value: 要加密的值
        
    Returns:
        str: 加密后的值（Base64编码）
    """
    key = get_encryption_key()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(value.encode())
    return f"ENC:{base64.urlsafe_b64encode(encrypted).decode()}"

def decrypt_value(value: str) -> str:
    """
    解密配置值
    
    Args:
        value: 加密的值（Base64编码，以ENC:开头）
        
    Returns:
        str: 解密后的值
    """
    if not value.startswith("ENC:"):
        return value
    
    key = get_encryption_key()
    fernet = Fernet(key)
    encrypted = base64.urlsafe_b64decode(value[4:])
    return fernet.decrypt(encrypted).decode()

def get_env_value(key: str, default: Any = None) -> Any:
    """
    从环境变量获取配置值
    
    Args:
        key: 配置键（点号分隔的多级键）
        default: 默认值
        
    Returns:
        Any: 环境变量值或默认值
    """
    # 将点号分隔的键转换为环境变量名（大写，下划线分隔）
    env_key = f"STOCK_{'_'.join(key.split('.')).upper()}"
    return os.environ.get(env_key, default)

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    验证配置是否有效，检查必需的配置项是否存在
    
    Args:
        config: 要验证的配置字典
        
    Returns:
        List[str]: 错误信息列表，如果为空则表示配置有效
    """
    errors = []
    
    # 检查必需的配置项
    for section, keys in REQUIRED_CONFIG.items():
        if section not in config:
            errors.append(f"缺少必需的配置节 '{section}'")
            continue
            
        for key in keys:
            if key not in config[section]:
                errors.append(f"配置节 '{section}' 中缺少必需的配置项 '{key}'")
    
    # 检查数据库配置
    if 'db' in config:
        db_config = config['db']
        if 'port' in db_config and not isinstance(db_config['port'], int):
            errors.append(f"数据库端口必须是整数，当前值: {db_config['port']}")
    
    # 检查日期格式
    if 'dates' in config:
        date_config = config['dates']
        for key in ['default_start', 'default_end']:
            if key in date_config:
                date_str = date_config[key]
                if not (isinstance(date_str, str) and len(date_str) == 8 and date_str.isdigit()):
                    errors.append(f"日期格式错误 '{key}': {date_str}，应为8位数字的字符串，如'20230101'")
    
    return errors

def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    递归合并字典
    
    Args:
        target: 目标字典
        source: 源字典
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = value

def load_user_config() -> bool:
    """
    加载用户配置文件，如果存在则覆盖默认配置
    
    Returns:
        bool: 加载是否成功
    """
    global CONFIG
    if os.path.exists(USER_CONFIG_PATH):
        try:
            with open(USER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
                # 递归合并配置
                deep_update(CONFIG, user_config)
                
            # 验证合并后的配置
            errors = validate_config(CONFIG)
            if errors:
                error_msg = "\n".join(errors)
                logging.error(f"配置验证失败:\n{error_msg}")
                return False
                
            return True
        except json.JSONDecodeError as e:
            logging.error(f"解析用户配置文件失败，JSON格式错误: {e}")
            return False
        except Exception as e:
            logging.error(f"加载用户配置文件失败: {e}")
            return False
    
    # 如果用户配置文件不存在，则创建一个
    try:
        save_user_config(CONFIG)
        return True
    except Exception as e:
        logging.error(f"创建用户配置文件失败: {e}")
        return False

def process_sensitive_configs() -> None:
    """
    处理敏感配置项：加密存储，并从环境变量加载
    """
    # 检查是否启用敏感信息加密
    encrypt_sensitive = CONFIG.get('security', {}).get('encrypt_sensitive', True)
    
    for key_path in SENSITIVE_CONFIG:
        keys = key_path.split('.')
        
        # 从环境变量加载
        env_value = get_env_value(key_path)
        if env_value is not None:
            # 设置配置值（不加密环境变量值）
            current = CONFIG
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = env_value
            continue
        
        # 如果环境变量中没有值，检查配置中的值
        current = CONFIG
        try:
            for k in keys[:-1]:
                current = current[k]
            
            value = current.get(keys[-1])
            if value and isinstance(value, str):
                if value.startswith("ENC:"):
                    # 解密已加密的值
                    current[keys[-1]] = decrypt_value(value)
                elif encrypt_sensitive and value != "******":
                    # 加密未加密的值（非占位符）
                    current[keys[-1]] = encrypt_value(value)
        except (KeyError, TypeError):
            pass

def save_user_config(config: Dict[str, Any]) -> bool:
    """
    保存用户配置到文件
    
    Args:
        config: 要保存的配置字典
        
    Returns:
        bool: 是否保存成功
    """
    # 验证配置
    errors = validate_config(config)
    if errors:
        error_msg = "\n".join(errors)
        logging.error(f"配置验证失败，无法保存:\n{error_msg}")
        return False
    
    # 创建配置副本，用于保存
    config_copy = json.loads(json.dumps(config))
    
    # 处理敏感配置项
    encrypt_sensitive = config.get('security', {}).get('encrypt_sensitive', True)
    if encrypt_sensitive:
        for key_path in SENSITIVE_CONFIG:
            keys = key_path.split('.')
            current = config_copy
            try:
                for k in keys[:-1]:
                    current = current[k]
                
                value = current.get(keys[-1])
                if value and isinstance(value, str) and not value.startswith("ENC:") and value != "******":
                    current[keys[-1]] = encrypt_value(value)
            except (KeyError, TypeError):
                pass
    
    try:
        with open(USER_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logging.error(f"保存用户配置文件失败: {e}")
        return False

def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    获取配置项
    
    Args:
        key: 配置键，支持点号分隔的多级键，如 'db.password'
        default: 默认值，当配置项不存在时返回
        
    Returns:
        配置值或默认值
    """
    # 先尝试从环境变量获取
    if key is not None:
        env_value = get_env_value(key)
        if env_value is not None:
            return env_value
    
    if key is None:
        return CONFIG
        
    keys = key.split('.')
    value = CONFIG
    
    try:
        for k in keys:
            value = value[k]
            
            # 如果是加密的值，解密后返回
            if isinstance(value, str) and value.startswith("ENC:"):
                value = decrypt_value(value)
                
        return value
    except (KeyError, TypeError):
        return default

def set_config(key: str, value: Any) -> bool:
    """
    设置配置项，并保存到用户配置文件
    
    Args:
        key: 配置键，支持点号分隔的多级键，如 'db.password'
        value: 配置值
        
    Returns:
        bool: 是否设置成功
    """
    keys = key.split('.')
    config_copy = CONFIG.copy()
    
    # 检查是否为敏感配置项
    is_sensitive = key in SENSITIVE_CONFIG
    encrypt_sensitive = CONFIG.get('security', {}).get('encrypt_sensitive', True)
    
    # 定位到最后一级键的父级
    current = config_copy
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    
    # 设置值
    if is_sensitive and encrypt_sensitive and isinstance(value, str):
        current[keys[-1]] = encrypt_value(value)
    else:
        current[keys[-1]] = value
    
    # 验证修改后的配置
    errors = validate_config(config_copy)
    if errors:
        error_msg = "\n".join(errors)
        logging.error(f"配置验证失败，无法设置:\n{error_msg}")
        return False
    
    # 应用更改
    current = CONFIG
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    if is_sensitive and encrypt_sensitive and isinstance(value, str):
        current[keys[-1]] = encrypt_value(value)
    else:
        current[keys[-1]] = value
    
    # 保存到用户配置文件
    return save_user_config(CONFIG)

def request_db_password() -> None:
    """
    交互式请求数据库密码
    """
    # 如果已从环境变量获取密码，则跳过
    if get_env_value('db.password') is not None:
        return
    
    # 如果配置中已有非占位符密码，则跳过
    db_password = CONFIG.get('db', {}).get('password')
    if db_password and db_password != "******" and not (isinstance(db_password, str) and db_password.startswith("ENC:")):
        return
    
    import getpass
    print("数据库密码未配置，请输入:")
    password = getpass.getpass()
    if password:
        set_config('db.password', password)

# 初始化时加载用户配置
load_user_config()

# 处理敏感配置项
process_sensitive_configs()

# 暴露数据库配置便于直接使用
def get_db_config() -> Dict[str, Union[str, int]]:
    """获取数据库配置"""
    db_config = CONFIG['db'].copy()
    
    # 如果密码是加密的，解密
    if isinstance(db_config.get('password'), str) and db_config['password'].startswith("ENC:"):
        db_config['password'] = decrypt_value(db_config['password'])
    
    # 如果密码是占位符，从环境变量获取
    if db_config.get('password') == "******":
        env_password = get_env_value('db.password')
        if env_password:
            db_config['password'] = env_password
        else:
            # 如果环境变量中也没有，则交互式请求
            request_db_password()
            
            # 重新获取密码
            db_password = CONFIG.get('db', {}).get('password')
            if isinstance(db_password, str) and db_password.startswith("ENC:"):
                db_config['password'] = decrypt_value(db_password)
            else:
                db_config['password'] = db_password
    
    return db_config 