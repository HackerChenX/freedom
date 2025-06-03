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
from db.clickhouse_db import get_clickhouse_db
from utils.path_utils import get_strategy_dir

logger = get_logger(__name__)


class StrategyManager:
    """
    策略管理器，负责策略的创建、保存、加载和版本控制
    
    该类提供了策略的完整生命周期管理，包括创建、更新、获取、列表、删除等功能。
    支持将策略保存到数据库和文件系统，便于持久化和共享。
    """
    
    def __init__(self, db_manager=None):
        """
        初始化策略管理器
        
        Args:
            db_manager: 数据库管理器实例，如果为None则使用默认的ClickHouse连接
        """
        self.db_manager = db_manager or get_clickhouse_db()
        self.strategy_dir = get_strategy_dir()
        
        # 确保策略目录存在
        os.makedirs(self.strategy_dir, exist_ok=True)
        
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
        for condition in strategy["conditions"]:
            if "logic" in condition:
                if condition["logic"].upper() not in ["AND", "OR", "NOT"]:
                    raise ValueError(f"不支持的逻辑运算符: {condition['logic']}")
            elif "group" in condition or "end_group" in condition:
                # 分组条件，不需要额外验证
                pass
            else:
                if "indicator_id" not in condition:
                    raise ValueError("条件缺少 indicator_id 字段")
                if "period" not in condition:
                    raise ValueError("条件缺少 period 字段")
                    
        return True
        
    def _generate_strategy_id(self) -> str:
        """
        生成唯一策略ID
        
        Returns:
            str: 格式为"STRATEGY_{8位随机字符}"的唯一ID
        """
        return f"STRATEGY_{uuid.uuid4().hex[:8].upper()}"
        
    def _merge_strategy_configs(self, original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并策略配置
        
        Args:
            original: 原始策略配置
            updates: 更新的策略配置
            
        Returns:
            Dict[str, Any]: 合并后的策略配置
        """
        merged = copy.deepcopy(original)
        for key, value in updates["strategy"].items()``:
            if key != "id" and key != "create_time":
                merged["strategy"][key] = value
        return merged
            
    def _save_strategy_to_db(self, strategy_config: Dict[str, Any]) -> bool:
        """
        将策略保存到数据库
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            bool: 是否保存成功
        """
        try:
            strategy_id = strategy_config["strategy"]["id"]
            strategy_name = strategy_config["strategy"]["name"]
            
            # 转换为JSON字符串
            config_json = json.dumps(strategy_config, ensure_ascii=False)
            
            # 构建SQL语句
            # 首先尝试删除可能存在的同ID策略
            delete_sql = f"ALTER TABLE strategy_definitions DELETE WHERE strategy_id = '{strategy_id}'"
            self.db_manager.execute(delete_sql)
            
            # 然后插入新策略
            insert_sql = f"""
            INSERT INTO strategy_definitions 
            (strategy_id, name, config, create_time, update_time)
            VALUES 
            (
                '{strategy_id}', 
                '{strategy_name}', 
                '{config_json}', 
                '{strategy_config["strategy"]["create_time"]}', 
                '{strategy_config["strategy"]["update_time"]}'
            )
            """
            self.db_manager.execute(insert_sql)
            
            logger.info(f"策略已保存到数据库: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"保存策略到数据库失败: {e}")
            return False
            
    def _save_strategy_to_file(self, strategy_config: Dict[str, Any]) -> bool:
        """
        将策略保存到文件
        
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
                
            logger.info(f"策略已保存到文件: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存策略到文件失败: {e}")
            return False
            
    def _get_strategy_from_db(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        从数据库获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置，如果不存在则返回None
        """
        try:
            sql = f"SELECT config FROM strategy_definitions WHERE strategy_id = '{strategy_id}' LIMIT 1"
            result = self.db_manager.query(sql)
            
            if result.empty:
                return None
                
            # 解析JSON
            config_json = result.iloc[0]['config']
            return json.loads(config_json)
        except Exception as e:
            logger.error(f"从数据库获取策略失败: {e}")
            return None
            
    def _get_strategy_from_file(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Optional[Dict[str, Any]]: 策略配置，如果不存在则返回None
        """
        try:
            file_path = os.path.join(self.strategy_dir, f"{strategy_id}.json")
            
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"从文件获取策略失败: {e}")
            return None
            
    def _list_strategies_from_db(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        从数据库获取策略列表
        
        Args:
            filters: 过滤条件
            
        Returns:
            List[Dict[str, Any]]: 策略列表
        """
        try:
            # 构建查询条件
            where_clause = "WHERE 1=1"
            if filters:
                if 'author' in filters:
                    where_clause += f" AND config LIKE '%\"author\":\"{filters['author']}\"%'"
                if 'name' in filters:
                    where_clause += f" AND name LIKE '%{filters['name']}%'"
                if 'version' in filters:
                    where_clause += f" AND config LIKE '%\"version\":\"{filters['version']}\"%'"
            
            # 构建SQL查询
            sql = f"""
            SELECT config
            FROM strategy_definitions
            {where_clause}
            ORDER BY update_time DESC
            """
            
            result = self.db_manager.query(sql)
            
            # 解析结果
            strategies = []
            for _, row in result.iterrows():
                try:
                    strategy_config = json.loads(row['config'])
                    strategies.append(strategy_config)
                except Exception as e:
                    logger.warning(f"解析策略配置失败: {e}")
            
            return strategies
        except Exception as e:
            logger.error(f"从数据库获取策略列表失败: {e}")
            return []
            
    def _list_strategies_from_files(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        从文件获取策略列表
        
        Args:
            filters: 过滤条件，支持按作者、名称、版本等筛选
            
        Returns:
            List[Dict[str, Any]]: 策略列表
        """
        try:
            strategies = []
            
            # 获取所有JSON文件
            for file_name in os.listdir(self.strategy_dir):
                if not file_name.endswith('.json'):
                    continue
                    
                file_path = os.path.join(self.strategy_dir, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        strategy_config = json.load(f)
                        
                    # 应用过滤条件
                    if filters:
                        strategy = strategy_config["strategy"]
                        if 'author' in filters and strategy.get('author') != filters['author']:
                            continue
                        if 'name' in filters and filters['name'] not in strategy.get('name', ''):
                            continue
                        if 'version' in filters and strategy.get('version') != filters['version']:
                            continue
                            
                    strategies.append(strategy_config)
                except Exception as e:
                    logger.warning(f"解析策略文件失败: {file_name}, 错误: {e}")
                    
            # 按更新时间排序
            strategies.sort(key=lambda s: s["strategy"].get("update_time", ""), reverse=True)
            
            return strategies
        except Exception as e:
            logger.error(f"从文件获取策略列表失败: {e}")
            return []
            
    def _delete_strategy_from_db(self, strategy_id: str) -> bool:
        """
        从数据库删除策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            sql = f"ALTER TABLE strategy_definitions DELETE WHERE strategy_id = '{strategy_id}'"
            self.db_manager.execute(sql)
            logger.info(f"已从数据库删除策略: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"从数据库删除策略失败: {e}")
            return False
            
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
            
            if not os.path.exists(file_path):
                return False
                
            os.remove(file_path)
            logger.info(f"已从文件删除策略: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"从文件删除策略失败: {e}")
            return False
            
    def import_strategy(self, file_path: str) -> str:
        """
        导入策略
        
        Args:
            file_path: 策略文件路径
            
        Returns:
            str: 策略ID
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持或解析失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"策略文件不存在: {file_path}")
            
        # 根据文件扩展名确定解析方式
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    strategy_config = json.load(f)
            elif file_ext in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    strategy_config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的策略文件格式: {file_ext}")
                
            # 验证策略配置
            self._validate_strategy_config(strategy_config)
            
            # 生成新的策略ID
            strategy_config["strategy"]["id"] = self._generate_strategy_id()
            
            # 设置创建和更新时间
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            strategy_config["strategy"]["create_time"] = now
            strategy_config["strategy"]["update_time"] = now
            
            # 保存策略
            return self.create_strategy(strategy_config)
        except Exception as e:
            logger.error(f"导入策略失败: {e}")
            raise ValueError(f"导入策略失败: {e}")
            
    def export_strategy(self, strategy_id: str, file_path: str, format: str = 'json') -> bool:
        """
        导出策略到文件
        
        Args:
            strategy_id: 策略ID
            file_path: 导出的文件路径
            format: 导出格式，支持'json'和'yaml'，默认为'json'
            
        Returns:
            bool: 是否导出成功
            
        Raises:
            ValueError: 策略不存在或导出格式不支持
        """
        # 获取策略
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"策略 {strategy_id} 不存在")
            
        try:
            # 根据格式导出
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(strategy, f, ensure_ascii=False, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(strategy, f, allow_unicode=True, default_flow_style=False)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
                
            logger.info(f"策略已导出到文件: {file_path}")
            return True
        except Exception as e:
            logger.error(f"导出策略失败: {e}")
            return False
            
    def clone_strategy(self, strategy_id: str, new_name: Optional[str] = None) -> str:
        """
        克隆策略
        
        Args:
            strategy_id: 源策略ID
            new_name: 新策略名称，如果为None则自动生成
            
        Returns:
            str: 新策略ID
            
        Raises:
            ValueError: 源策略不存在
        """
        # 获取源策略
        source_strategy = self.get_strategy(strategy_id)
        if not source_strategy:
            raise ValueError(f"策略 {strategy_id} 不存在")
            
        # 创建新策略配置
        new_strategy = copy.deepcopy(source_strategy)
        
        # 生成新的ID
        new_strategy["strategy"]["id"] = self._generate_strategy_id()
        
        # 设置新名称
        if new_name:
            new_strategy["strategy"]["name"] = new_name
        else:
            new_strategy["strategy"]["name"] = f"{source_strategy['strategy']['name']} (复制)"
            
        # 设置创建和更新时间
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_strategy["strategy"]["create_time"] = now
        new_strategy["strategy"]["update_time"] = now
        
        # 保存新策略
        return self.create_strategy(new_strategy)
            
    def get_strategy_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        获取策略历史版本
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            List[Dict[str, Any]]: 策略历史版本列表
        """
        try:
            # 查询策略历史记录
            sql = f"""
            SELECT version, config, update_time
            FROM strategy_history
            WHERE strategy_id = '{strategy_id}'
            ORDER BY version DESC
            """
            
            result = self.db_manager.query(sql)
            
            # 解析结果
            history = []
            for _, row in result.iterrows():
                try:
                    version = row['version']
                    config = json.loads(row['config'])
                    update_time = row['update_time']
                    
                    history.append({
                        'version': version,
                        'config': config,
                        'update_time': update_time
                    })
                except Exception as e:
                    logger.warning(f"解析策略历史记录失败: {e}")
            
            return history
        except Exception as e:
            logger.error(f"获取策略历史版本失败: {e}")
            return [] 