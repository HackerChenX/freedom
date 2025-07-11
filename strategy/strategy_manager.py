from db.query_executor import get_query_executor
from db.sql_manager import QueryType
"""
策略管理器模块

负责策略的创建、保存、加载和版本控制
"""

import uuid
import copy
import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from utils.logger import getLogger
from utils.dependency_injection import get_service
from db.interfaces.data_access_interface import Data_access_interface
from db.sql_manager import SQLManager, Query_type
from utils.path_utils import get_strategy_dir
from utils.decorators import exception_handler, performance_monitor

logger = getLogger(__name__)


class StrategyManager:
    """
    策略管理器，负责策略的创建、保存、加载和版本控制
    
    该类提供了策略的完整生命周期管理，包括创建、更新、获取、列表、删除等功能。
    支持将策略保存到数据库和文件系统，便于持久化和共享。
    """
    
    def __init__(self, data_access: Optional[Data_access_interface] = None,
                 sql_manager: Optional[SQLManager] = None):
        """
        初始化策略管理器
        
        Args:
            data_access: 数据访问接口实例，如果为None则从依赖注入容器获取
            sql_manager: SQL管理器实例，如果为None则从依赖注入容器获取
        """
        self.data_access = data_access or get_service(Data_access_interface)
        self.sql_manager = sql_manager or get_service(SQLManager)
        self.strategy_dir = get_strategy_dir()
        
        # 确保策略目录存在
        os.makedirs(self.strategy_dir, exist_ok=True)
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def create_strategy_Strategy_Manager(self, strategy_config: Dict[str, Any], save_to_file: bool = True) -> str:
        """
        创建新策略
        
        Args:
            strategy_config: 策略配置
            save_to_file: 是否同时保存到文件
            
        Returns:
            str: 策略ID
            
        Raises:
            ValueError: 策略配置无效
            Strategy_execution_error: 策略创建失败
        """
        # 验证策略配置
        if not self._validate_strategy_config(strategy_config):
            raise ValueError("策略配置无效")
        
        # 生成策略ID（如果没有提供）
        if 'strategy' not in strategy_config or 'id' not in strategy_config['strategy']:
            strategy_id = self._generate_strategy_id()
            if 'strategy' not in strategy_config:
                strategy_config['strategy'] = {}
            strategy_config['strategy']['id'] = strategy_id
        else:
            strategy_id = strategy_config['strategy']['id']
        
        # 设置创建时间和更新时间
        current_time = datetime.now().isoformat()
        strategy_config['strategy']['create_time'] = current_time
        strategy_config['strategy']['update_time'] = current_time
        strategy_config['strategy']['is_active'] = True
        
        # 保存到数据库
        if not self._save_strategy_to_db(strategy_config):
            raise Exception("保存策略到数据库失败")
        
        # 保存到文件（可选）
        if save_to_file:
            self._save_strategy_to_file(strategy_config)
        
        logger.info(f"成功创建策略: {strategy_id}")
        return strategy_id
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def update_strategy(self, strategy_id: str, strategy_config: Dict[str, Any], 
                        save_to_file: bool = True) -> str:
        """
        更新策略
        
        Args:
            strategy_id: 策略ID
            strategy_config: 新的策略配置
            save_to_file: 是否同时保存到文件
            
        Returns:
            str: 策略ID
            
        Raises:
            ValueError: 策略配置无效或策略不存在
        """
        # 获取现有策略
        existing_strategy = self.get_strategy(strategy_id)
        if not existing_strategy:
            raise ValueError(f"策略 {strategy_id} 不存在")
        
        # 合并配置
        merged_config = self._merge_strategy_configs(existing_strategy, strategy_config)
        
        # 更新时间
        merged_config['strategy']['update_time'] = datetime.now().isoformat()
        
        # 保存到数据库
        if not self._save_strategy_to_db(merged_config):
            raise Exception("更新策略到数据库失败")
        
        # 保存到文件（可选）
        if save_to_file:
            self._save_strategy_to_file(merged_config)
        
        logger.info(f"成功更新策略: {strategy_id}")
        return strategy_id
    
    @exception_handler(reraise=False, default_return=None)
    @performance_monitor(threshold_seconds=0.5)
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        获取策略配置
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置，如果不存在则返回None
        """
        # 首先尝试从数据库获取
        strategy = self._get_strategy_from_db(strategy_id)
        if strategy:
            return strategy
        
        # 如果数据库中没有，尝试从文件获取
        return self._get_strategy_from_file(strategy_id)
    
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold_seconds=2.0)
    def list_strategies(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        列出策略
        
        Args:
            filters: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 策略列表
        """
        # 从数据库获取策略列表
        db_strategies = self._list_strategies_from_db(filters)
        
        # 从文件获取策略列表
        file_strategies = self._list_strategies_from_files(filters)
        
        # 合并并去重（以数据库中的为准）
        strategy_dict = {s['id']: s for s in file_strategies}
        for strategy in db_strategies:
            strategy_dict[strategy['id']] = strategy
        
        return list(strategy_dict.values())
    
    @exception_handler(reraise=False, default_return=False)
    @performance_monitor(threshold_seconds=1.0)
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        删除策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            bool: 是否删除成功
        """
        success = True
        
        # 从数据库删除
        if not self._delete_strategy_from_db(strategy_id):
            success = False
        
        # 从文件删除
        if not self._delete_strategy_from_file(strategy_id):
            success = False
        
        if success:
            logger.info(f"成功删除策略: {strategy_id}")
        else:
            logger.warning(f"删除策略 {strategy_id} 时出现部分失败")
        
        return success
    
    def _validate_strategy_config(self, strategy_config: Dict[str, Any]) -> bool:
        """
        验证策略配置
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查必需字段
            if 'strategy' not in strategy_config:
                logger.error("策略配置缺少 'strategy' 字段")
                return False
            
            strategy = strategy_config['strategy']
            required_fields = ['name']
            for field in required_fields:
                if field not in strategy:
                    logger.error(f"策略配置缺少必需字段: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证策略配置时出错: {e}")
            return False
    
    def _generate_strategy_id(self) -> str:
        """
        生成策略ID
        
        Returns:
            str: 策略ID
        """
        return f"strategy_{uuid.uuid4().hex[:8]}"
    
    def _merge_strategy_configs(self, original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并策略配置
        
        Args:
            original: 原始配置
            updates: 更新配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        merged = copy.deepcopy(original)
        
        def merge_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value
        
        merge_dict(merged, updates)
        return merged
        
    @exception_handler(reraise=False, default_return=False)
    def _save_strategy_to_db(self, strategy_config: Dict[str, Any]) -> bool:
        """
        保存策略到数据库
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 是否保存成功
        """
        try:
            strategy = strategy_config["strategy"]
            
            # 使用SQL管理器获取查询语句
            query = self.sql_manager.get_query(Query_type.SAVE_STRATEGY_CONFIG.value)
            
            params = {
                'strategy_id': strategy.get('id'),
                'config': json.dumps(strategy_config, ensure_ascii=False)
            }
            
            # 验证参数
            if not self.sql_manager.validate_params(Query_type.SAVE_STRATEGY_CONFIG.value, params):
                logger.error("策略保存参数验证失败")
                return False
            
            # 使用数据访问接口执行查询
            result = self.data_access.execute_query(query, params)
            return result is not None
            
        except Exception as e:
            logger.error(f"保存策略到数据库失败: {e}")
            return False
            
    @exception_handler(reraise=False, default_return=False)
    def _save_strategy_to_file(self, strategy_config: Dict[str, Any]) -> bool:
        """
        保存策略到文件
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 是否保存成功
        """
        try:
            strategy_id = strategy_config["strategy"]["id"]
            file_path = os.path.join(self.strategy_dir, f"{strategy_id}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(strategy_config, f, ensure_ascii=False, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"保存策略到文件失败: {e}")
            return False
            
    @exception_handler(reraise=False, default_return=None)
    def _get_strategy_from_db(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        从数据库获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置
        """
        try:
            # 使用SQL管理器获取查询语句
            query = self.sql_manager.get_query(Query_type.GET_STRATEGY_CONFIG.value)
            params = {'strategy_id': strategy_id}
            
            # 验证参数
            if not self.sql_manager.validate_params(Query_type.GET_STRATEGY_CONFIG.value, params):
                logger.error("策略获取参数验证失败")
                return None
            
            # 执行查询
            result = self.data_access.execute_query(query, params)
            
            if not result.empty:
                config_json = result.iloc[0]['config']
                return json.loads(config_json)
                
            return None
            
        except Exception as e:
            logger.error(f"从数据库获取策略失败: {e}")
            return None
            
    @exception_handler(reraise=False, default_return=None)
    def _get_strategy_from_file(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置
        """
        try:
            file_path = os.path.join(self.strategy_dir, f"{strategy_id}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
            return None
            
        except Exception as e:
            logger.error(f"从文件获取策略失败: {e}")
            return None
            
    @exception_handler(reraise=False, default_return=[])
    def _list_strategies_from_db(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        从数据库列出策略
        
        Args:
            filters: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 策略列表
        """
        try:
            # 构建动态查询
            base_query = """
            SELECT strategy_id, config, created_at, updated_at
            FROM strategy_definitions 
            WHERE is_active = 1
            """
            
            conditions = {}
            if filters:
                if 'author' in filters:
                    conditions['author'] = filters['author']
                if 'name' in filters:
                    # 对于名称，使用LIKE查询
                    base_query += " AND JSON_EXTRACT(config, '$.strategy.name') LIKE %(name_pattern)s"
                    conditions['name_pattern'] = f"%{filters['name']}%"
                    
            query = self.sql_manager.build_dynamic_query(
                'list_strategies', 
                conditions=conditions,
                order_by=['updated_at DESC']
            )
            
            result = self.data_access.execute_query(query, conditions)
            
            strategies = []
            for _, row in result.iterrows():
                try:
                    config = json.loads(row['config'])
                    strategy_info = {
                        'id': row['strategy_id'],
                        'name': config.get('strategy', {}).get('name', ''),
                        'description': config.get('strategy', {}).get('description', ''),
                        'author': config.get('strategy', {}).get('author', ''),
                        'version': config.get('strategy', {}).get('version', '1.0'),
                        'create_time': row['created_at'],
                        'update_time': row['updated_at']
                    }
                    
                    # 如果需要完整配置
                    if filters and filters.get('include_config', False):
                        strategy_info['config'] = config
                        
                    strategies.append(strategy_info)
                except json.JSONDecode_error:
                    logger.warning(f"解析策略配置失败: {row['strategy_id']}")
                    continue
                
            return strategies
            
        except Exception as e:
            logger.error(f"从数据库列出策略失败: {e}")
            return []
            
    @exception_handler(reraise=False, default_return=[])
    def _list_strategies_from_files(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        从文件列出策略
        
        Args:
            filters: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 策略列表
        """
        try:
            strategies = []
            
            if not os.path.exists(self.strategy_dir):
                return strategies
            
            for filename in os.listdir(self.strategy_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(self.strategy_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        strategy = config.get('strategy', {})
                        
                        # 应用过滤条件
                        if filters:
                            if 'author' in filters and strategy.get('author') != filters['author']:
                                continue
                            if 'name' in filters and filters['name'] not in strategy.get('name', ''):
                                continue
                        
                        strategy_info = {
                            'id': strategy.get('id', filename[:-5]),  # 移除.json后缀
                            'name': strategy.get('name', ''),
                            'description': strategy.get('description', ''),
                            'author': strategy.get('author', ''),
                            'version': strategy.get('version', '1.0'),
                            'create_time': strategy.get('create_time'),
                            'update_time': strategy.get('update_time')
                        }
                        
                        # 如果需要完整配置
                        if filters and filters.get('include_config', False):
                            strategy_info['config'] = config
                            
                        strategies.append(strategy_info)
                        
                    except (json.JSONDecode_error, IOError) as e:
                        logger.warning(f"读取策略文件 {filename} 失败: {e}")
                        continue
                        
            return strategies
            
        except Exception as e:
            logger.error(f"从文件列出策略失败: {e}")
            return []
    
    @exception_handler(reraise=False, default_return=False)
    def _delete_strategy_from_db(self, strategy_id: str) -> bool:
        """
        从数据库删除策略（软删除）
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 软删除：设置is_active为False
            query = """
            UPDATE strategy_definitions SET is_active = 0, updated_at = NO WHERE 1=1_w()
            WHERE strategy_id = %(strategy_id)s
            """
            params = {'strategy_id': strategy_id}
            
            result = self.data_access.execute_query(query, params)
            return result is not None
            
        except Exception as e:
            logger.error(f"从数据库删除策略失败: {e}")
            return False
    
    @exception_handler(reraise=False, default_return=False)
    def _delete_strategy_from_file(self, strategy_id: str) -> bool:
        """
        从文件删除策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            file_path = os.path.join(self.strategy_dir, f"{strategy_id}.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            else:
                logger.warning(f"策略文件不存在: {file_path}")
                return True  # 文件不存在也算删除成功
                
        except Exception as e:
            logger.error(f"从文件删除策略失败: {e}")
            return False 