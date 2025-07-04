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

from utils.logger import get_logger
from db.container import get_container
from db.interfaces.data_access_interface import IDataAccess
from utils.path_utils import get_strategy_dir
from utils.decorators import exception_handler, performance_monitor

logger = get_logger(__name__)


class StrategyManager:
    """
    策略管理器，负责策略的创建、保存、加载和版本控制
    
    该类提供了策略的完整生命周期管理，包括创建、更新、获取、列表、删除等功能。
    支持将策略保存到数据库和文件系统，便于持久化和共享。
    """
    
    def __init__(self, data_access: Optional[IDataAccess] = None):
        """
        初始化策略管理器
        
        Args:
            data_access: 数据访问接口实例，如果为None则从容器获取
        """
        container = get_container()
        self.data_access = data_access or container.resolve(IDataAccess)
        self.strategy_dir = get_strategy_dir()
        
        # 确保策略目录存在
        os.makedirs(self.strategy_dir, exist_ok=True)
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def create_strategy(self, strategy_config: Dict[str, Any], save_to_file: bool = True) -> str:
        """
        创建新策略
        
        Args:
            strategy_config: 策略配置字典，包含策略的完整定义
            save_to_file: 是否保存到文件，默认为True
            
        Returns:
            str: 策略ID
            
        Raises:
            ValueError: 策略配置无效时抛出
        """
        # 验证策略配置
        self._validate_strategy_config(strategy_config)
        
        # 生成策略ID
        if "id" not in strategy_config["strategy"]:
            strategy_config["strategy"]["id"] = self._generate_strategy_id()
            
        # 设置创建和更新时间
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        strategy_config["strategy"]["create_time"] = now
        strategy_config["strategy"]["update_time"] = now
        
        # 保存策略
        strategy_id = strategy_config["strategy"]["id"]
        
        # 保存到数据库
        self._save_strategy_to_db(strategy_config)
        
        # 保存到文件
        if save_to_file:
            self._save_strategy_to_file(strategy_config)
            
        logger.info(f"创建策略成功: {strategy_id}")
        return strategy_id
        
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def update_strategy(self, strategy_id: str, strategy_config: Dict[str, Any], 
                        save_to_file: bool = True) -> str:
        """
        更新现有策略
        
        Args:
            strategy_id: 策略ID
            strategy_config: 策略配置，包含要更新的字段
            save_to_file: 是否保存到文件，默认为True
            
        Returns:
            str: 策略ID
            
        Raises:
            ValueError: 策略不存在或配置无效时抛出
        """
        # 获取原策略
        original_strategy = self.get_strategy(strategy_id)
        if not original_strategy:
            raise ValueError(f"策略 {strategy_id} 不存在")
            
        # 合并配置
        merged_config = self._merge_strategy_configs(original_strategy, strategy_config)
        
        # 更新时间
        merged_config["strategy"]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存策略
        # 保存到数据库
        self._save_strategy_to_db(merged_config)
        
        # 保存到文件
        if save_to_file:
            self._save_strategy_to_file(merged_config)
            
        logger.info(f"更新策略成功: {strategy_id}")
        return strategy_id
        
    @exception_handler(reraise=False, default_return=None)
    @performance_monitor(threshold_seconds=0.5)
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        获取策略定义
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置，如果不存在则返回None
        """
        # 先从数据库获取
        strategy = self._get_strategy_from_db(strategy_id)
        
        # 如果数据库中不存在，则尝试从文件获取
        if not strategy:
            strategy = self._get_strategy_from_file(strategy_id)
            
        return strategy
        
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold_seconds=2.0)
    def list_strategies(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        列出所有策略
        
        Args:
            filters: 过滤条件，支持按作者、名称、版本等筛选
            
        Returns:
            List[Dict[str, Any]]: 策略列表
        """
        # 从数据库获取
        strategies = self._list_strategies_from_db(filters)
        
        # 如果数据库中没有，则从文件获取
        if not strategies:
            strategies = self._list_strategies_from_files(filters)
            
        return strategies
        
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
        # 从数据库删除
        db_result = self._delete_strategy_from_db(strategy_id)
        
        # 从文件删除
        file_result = self._delete_strategy_from_file(strategy_id)
        
        return db_result or file_result
        
    def _validate_strategy_config(self, strategy_config: Dict[str, Any]) -> bool:
        """
        验证策略配置
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 是否验证通过
            
        Raises:
            ValueError: 验证失败时抛出，包含详细错误信息
        """
        # 验证必要字段
        if "strategy" not in strategy_config:
            raise ValueError("无效的策略配置: 缺少'strategy'节点")
            
        strategy = strategy_config["strategy"]
        
        required_fields = ["name", "conditions"]
        for field in required_fields:
            if field not in strategy:
                raise ValueError(f"策略配置缺少必要字段: {field}")
                
        # 验证条件配置
        if not isinstance(strategy["conditions"], list) or len(strategy["conditions"]) == 0:
            raise ValueError("策略条件不能为空")
            
        return True
        
    def _generate_strategy_id(self) -> str:
        """
        生成策略ID
        
        Returns:
            str: 策略ID
        """
        return str(uuid.uuid4())
        
    def _merge_strategy_configs(self, original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并策略配置
        
        Args:
            original: 原始策略配置
            updates: 更新的配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        merged = copy.deepcopy(original)
        
        # 递归合并字典
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
            
            # 构建SQL语句
            sql = """
            INSERT INTO strategy_config 
            (id, name, description, author, version, conditions, risk_control, 
             create_time, update_time, is_active, config_json)
            VALUES (%(id)s, %(name)s, %(description)s, %(author)s, %(version)s, 
                    %(conditions)s, %(risk_control)s, %(create_time)s, %(update_time)s, 
                    %(is_active)s, %(config_json)s)
            ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            description = VALUES(description),
            author = VALUES(author),
            version = VALUES(version),
            conditions = VALUES(conditions),
            risk_control = VALUES(risk_control),
            update_time = VALUES(update_time),
            is_active = VALUES(is_active),
            config_json = VALUES(config_json)
            """
            
            params = {
                'id': strategy.get('id'),
                'name': strategy.get('name', ''),
                'description': strategy.get('description', ''),
                'author': strategy.get('author', ''),
                'version': strategy.get('version', '1.0'),
                'conditions': json.dumps(strategy.get('conditions', []), ensure_ascii=False),
                'risk_control': json.dumps(strategy.get('risk_control', {}), ensure_ascii=False),
                'create_time': strategy.get('create_time'),
                'update_time': strategy.get('update_time'),
                'is_active': strategy.get('is_active', True),
                'config_json': json.dumps(strategy_config, ensure_ascii=False)
            }
            
            # 使用数据访问接口执行SQL
            result = self.data_access.execute_sql(sql, params)
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
            sql = "SELECT config_json FROM strategy_config WHERE id = %s AND is_active = 1"
            result = self.data_access.query(sql, (strategy_id,))
            
            if result and len(result) > 0:
                config_json = result[0][0]
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
            sql = """
            SELECT id, name, description, author, version, create_time, update_time, config_json
            FROM strategy_config 
            WHERE is_active = 1
            """
            params = []
            
            # 应用过滤条件
            if filters:
                if 'author' in filters:
                    sql += " AND author = %s"
                    params.append(filters['author'])
                if 'name' in filters:
                    sql += " AND name LIKE %s"
                    params.append(f"%{filters['name']}%")
                    
            sql += " ORDER BY update_time DESC"
            
            result = self.data_access.query(sql, params)
            
            strategies = []
            for row in result:
                strategy_info = {
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'author': row[3],
                    'version': row[4],
                    'create_time': row[5],
                    'update_time': row[6]
                }
                
                # 如果需要完整配置，解析config_json
                if filters and filters.get('include_config', False):
                    strategy_info['config'] = json.loads(row[7])
                    
                strategies.append(strategy_info)
                
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
            
            for filename in os.listdir(self.strategy_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.strategy_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            strategy_config = json.load(f)
                            
                        strategy = strategy_config.get('strategy', {})
                        
                        # 应用过滤条件
                        if filters:
                            if 'author' in filters and strategy.get('author') != filters['author']:
                                continue
                            if 'name' in filters and filters['name'] not in strategy.get('name', ''):
                                continue
                                
                        strategy_info = {
                            'id': strategy.get('id'),
                            'name': strategy.get('name'),
                            'description': strategy.get('description'),
                            'author': strategy.get('author'),
                            'version': strategy.get('version'),
                            'create_time': strategy.get('create_time'),
                            'update_time': strategy.get('update_time')
                        }
                        
                        # 如果需要完整配置
                        if filters and filters.get('include_config', False):
                            strategy_info['config'] = strategy_config
                            
                        strategies.append(strategy_info)
                        
                    except Exception as e:
                        logger.warning(f"读取策略文件 {filename} 失败: {e}")
                        continue
                        
            # 按更新时间倒序排列
            strategies.sort(key=lambda x: x.get('update_time', ''), reverse=True)
            return strategies
            
        except Exception as e:
            logger.error(f"从文件列出策略失败: {e}")
            return []
            
    @exception_handler(reraise=False, default_return=False)
    def _delete_strategy_from_db(self, strategy_id: str) -> bool:
        """
        从数据库删除策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 软删除，设置is_active为0
            sql = "UPDATE strategy_config SET is_active = 0 WHERE id = %s"
            result = self.data_access.execute_sql(sql, (strategy_id,))
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
                
            return False
            
        except Exception as e:
            logger.error(f"从文件删除策略失败: {e}")
            return False
            
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=2.0)
    def import_strategy(self, file_path: str) -> str:
        """
        从文件导入策略
        
        Args:
            file_path: 策略文件路径
            
        Returns:
            str: 策略ID
            
        Raises:
            ValueError: 文件不存在或格式无效时抛出
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"策略文件不存在: {file_path}")
            
        try:
            # 根据文件扩展名选择解析方式
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    strategy_config = json.load(f)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    strategy_config = yaml.safe_load(f)
            else:
                raise ValueError("不支持的文件格式，仅支持JSON和YAML格式")
                
            # 生成新的策略ID避免冲突
            if "strategy" in strategy_config:
                strategy_config["strategy"]["id"] = self._generate_strategy_id()
                
            # 创建策略
            return self.create_strategy(strategy_config)
            
        except Exception as e:
            logger.error(f"导入策略失败: {e}")
            raise ValueError(f"导入策略失败: {e}")
            
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def export_strategy(self, strategy_id: str, file_path: str, format: str = 'json') -> bool:
        """
        导出策略到文件
        
        Args:
            strategy_id: 策略ID
            file_path: 导出文件路径
            format: 导出格式，支持'json'和'yaml'
            
        Returns:
            bool: 是否导出成功
            
        Raises:
            ValueError: 策略不存在或格式不支持时抛出
        """
        strategy_config = self.get_strategy(strategy_id)
        if not strategy_config:
            raise ValueError(f"策略 {strategy_id} 不存在")
            
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(strategy_config, f, ensure_ascii=False, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(strategy_config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
                
            logger.info(f"策略 {strategy_id} 导出成功: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出策略失败: {e}")
            raise ValueError(f"导出策略失败: {e}")
            
    @exception_handler(reraise=True)
    @performance_monitor(threshold_seconds=1.0)
    def clone_strategy(self, strategy_id: str, new_name: Optional[str] = None) -> str:
        """
        克隆策略
        
        Args:
            strategy_id: 原策略ID
            new_name: 新策略名称，如果为None则在原名称后加上"_copy"
            
        Returns:
            str: 新策略ID
            
        Raises:
            ValueError: 原策略不存在时抛出
        """
        original_strategy = self.get_strategy(strategy_id)
        if not original_strategy:
            raise ValueError(f"策略 {strategy_id} 不存在")
            
        # 创建副本
        cloned_strategy = copy.deepcopy(original_strategy)
        
        # 生成新ID和名称
        cloned_strategy["strategy"]["id"] = self._generate_strategy_id()
        
        if new_name:
            cloned_strategy["strategy"]["name"] = new_name
        else:
            original_name = cloned_strategy["strategy"].get("name", "未命名策略")
            cloned_strategy["strategy"]["name"] = f"{original_name}_copy"
            
        # 更新时间
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cloned_strategy["strategy"]["create_time"] = now
        cloned_strategy["strategy"]["update_time"] = now
        
        # 创建新策略
        new_strategy_id = self.create_strategy(cloned_strategy)
        
        logger.info(f"策略克隆成功: {strategy_id} -> {new_strategy_id}")
        return new_strategy_id
        
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold_seconds=2.0)
    def get_strategy_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        获取策略历史版本
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            List[Dict[str, Any]]: 历史版本列表
        """
        try:
            sql = """
            SELECT version, update_time, author, description, config_json
            FROM strategy_config_history 
            WHERE strategy_id = %s 
            ORDER BY update_time DESC
            """
            
            result = self.data_access.query(sql, (strategy_id,))
            
            history = []
            for row in result:
                version_info = {
                    'version': row[0],
                    'update_time': row[1],
                    'author': row[2],
                    'description': row[3],
                    'config': json.loads(row[4])
                }
                history.append(version_info)
                
            return history
            
        except Exception as e:
            logger.error(f"获取策略历史失败: {e}")
            return [] 