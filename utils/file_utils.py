"""
文件工具模块

提供文件处理的辅助函数
"""

import os
import shutil
import json
import csv
import pickle
from typing import Dict, List, Any, Union, Optional, Tuple, BinaryIO, TextIO
import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(directory: str) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        bool: 操作是否成功
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"创建目录: {directory}")
            return True
        except Exception as e:
            logger.error(f"创建目录失败 {directory}: {e}")
            return False
    return True


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict[str, Any]: 文件信息字典
    """
    if not os.path.exists(file_path):
        return {
            'exists': False,
            'error': '文件不存在'
        }
    
    try:
        stats = os.stat(file_path)
        
        return {
            'exists': True,
            'path': os.path.abspath(file_path),
            'size': stats.st_size,
            'size_human': format_file_size(stats.st_size),
            'created': datetime.datetime.fromtimestamp(stats.st_ctime),
            'modified': datetime.datetime.fromtimestamp(stats.st_mtime),
            'accessed': datetime.datetime.fromtimestamp(stats.st_atime),
            'is_file': os.path.isfile(file_path),
            'is_dir': os.path.isdir(file_path),
            'extension': os.path.splitext(file_path)[1].lower() if os.path.isfile(file_path) else None
        }
    except Exception as e:
        logger.error(f"获取文件信息失败 {file_path}: {e}")
        return {
            'exists': True,
            'error': str(e)
        }


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        str: 格式化后的文件大小
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def safe_delete(path: str) -> bool:
    """
    安全删除文件或目录
    
    Args:
        path: 文件或目录路径
        
    Returns:
        bool: 操作是否成功
    """
    if not os.path.exists(path):
        logger.warning(f"删除失败，路径不存在: {path}")
        return False
    
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug(f"删除文件: {path}")
        else:
            shutil.rmtree(path)
            logger.debug(f"删除目录: {path}")
        return True
    except Exception as e:
        logger.error(f"删除失败 {path}: {e}")
        return False


def copy_file(src: str, dst: str, overwrite: bool = True) -> bool:
    """
    复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        overwrite: 是否覆盖已存在的文件
        
    Returns:
        bool: 操作是否成功
    """
    if not os.path.exists(src):
        logger.error(f"源文件不存在: {src}")
        return False
    
    if os.path.exists(dst) and not overwrite:
        logger.warning(f"目标文件已存在且不允许覆盖: {dst}")
        return False
    
    try:
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst)
        ensure_dir(dst_dir)
        
        shutil.copy2(src, dst)
        logger.debug(f"复制文件: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"复制文件失败 {src} -> {dst}: {e}")
        return False


def move_file(src: str, dst: str, overwrite: bool = True) -> bool:
    """
    移动文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        overwrite: 是否覆盖已存在的文件
        
    Returns:
        bool: 操作是否成功
    """
    if not os.path.exists(src):
        logger.error(f"源文件不存在: {src}")
        return False
    
    if os.path.exists(dst) and not overwrite:
        logger.warning(f"目标文件已存在且不允许覆盖: {dst}")
        return False
    
    try:
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst)
        ensure_dir(dst_dir)
        
        shutil.move(src, dst)
        logger.debug(f"移动文件: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"移动文件失败 {src} -> {dst}: {e}")
        return False


def save_json(data: Any, file_path: str, ensure_ascii: bool = False, indent: int = 4) -> bool:
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        ensure_ascii: 是否确保ASCII编码
        indent: 缩进空格数
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_dir(directory)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
        
        logger.debug(f"保存JSON文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存JSON文件失败 {file_path}: {e}")
        return False


def load_json(file_path: str, default: Any = None) -> Any:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        default: 文件不存在或加载失败时返回的默认值
        
    Returns:
        Any: 加载的数据或默认值
    """
    if not os.path.exists(file_path):
        logger.warning(f"JSON文件不存在: {file_path}")
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"加载JSON文件: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载JSON文件失败 {file_path}: {e}")
        return default


def save_csv(data: List[Dict[str, Any]], file_path: str, fieldnames: Optional[List[str]] = None) -> bool:
    """
    保存数据为CSV文件
    
    Args:
        data: 要保存的数据，列表的每个元素是一个字典
        file_path: 文件路径
        fieldnames: 列名列表，如果为None则使用第一个字典的键
        
    Returns:
        bool: 操作是否成功
    """
    if not data:
        logger.warning("没有数据可保存")
        return False
    
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_dir(directory)
        
        # 确定列名
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.debug(f"保存CSV文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存CSV文件失败 {file_path}: {e}")
        return False


def load_csv(file_path: str, default: Any = None) -> List[Dict[str, str]]:
    """
    从CSV文件加载数据
    
    Args:
        file_path: 文件路径
        default: 文件不存在或加载失败时返回的默认值
        
    Returns:
        List[Dict[str, str]]: 加载的数据或默认值
    """
    if not os.path.exists(file_path):
        logger.warning(f"CSV文件不存在: {file_path}")
        return default if default is not None else []
    
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        
        logger.debug(f"加载CSV文件: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载CSV文件失败 {file_path}: {e}")
        return default if default is not None else []


def save_pickle(data: Any, file_path: str) -> bool:
    """
    保存数据为Pickle文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        ensure_dir(directory)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"保存Pickle文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存Pickle文件失败 {file_path}: {e}")
        return False


def load_pickle(file_path: str, default: Any = None) -> Any:
    """
    从Pickle文件加载数据
    
    Args:
        file_path: 文件路径
        default: 文件不存在或加载失败时返回的默认值
        
    Returns:
        Any: 加载的数据或默认值
    """
    if not os.path.exists(file_path):
        logger.warning(f"Pickle文件不存在: {file_path}")
        return default
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"加载Pickle文件: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载Pickle文件失败 {file_path}: {e}")
        return default


def list_files(directory: str, pattern: str = '*', recursive: bool = False) -> List[str]:
    """
    列出目录中的文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式（支持通配符）
        recursive: 是否递归搜索子目录
        
    Returns:
        List[str]: 文件路径列表
    """
    import glob
    
    if not os.path.exists(directory):
        logger.warning(f"目录不存在: {directory}")
        return []
    
    try:
        search_path = os.path.join(directory, pattern)
        
        if recursive:
            return glob.glob(search_path, recursive=True)
        else:
            return glob.glob(search_path)
    except Exception as e:
        logger.error(f"列出文件失败 {directory}: {e}")
        return [] 