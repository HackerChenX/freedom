#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略管理器 - 策略生命周期管理核心组件
"""

import os
import json
import copy
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from db.interfaces.data_access_interface import DataAccessInterface
from db.interfaces.cache_interface import ICacheService
from db.sql_manager import QueryType, get_sql_manager
from utils.dependency_injection import get_service, get_container
from utils.logger import get_logger
from utils.decorators import performance_monitor, safe_run, exception_handler
from utils.path_utils import get_strategy_dir
from utils.exceptions import (
    StrategyExecutionError,
    StrategyValidationError,
    DataAccessError
)

logger = get_logger(__name__)


class SimpleStrategy:
    """简化的策略实现"""

    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.config = config
        self.name = config.get('name', strategy_id)
        self.description = config.get('description', '')
        self.type = config.get('type', 'unknown')

    def generate_signal(self, data) -> str:
        """生成策略信号"""
        try:
            # 简化的信号生成逻辑
            if len(data) < 20:
                return "HOLD"

            # 基于策略类型生成不同的信号
            if self.type == 'momentum':
                return self._momentum_signal(data)
            elif self.type == 'trend':
                return self._trend_signal(data)
            elif self.type == 'volume_price':
                return self._volume_price_signal(data)
            elif self.type == 'volatility':
                return self._volatility_signal(data)
            else:
                return "HOLD"

        except Exception as e:
            logger.error(f"策略 {self.strategy_id} 信号生成失败: {e}")
            return "HOLD"

    def calculate_score(self, data) -> float:
        """计算策略评分"""
        try:
            # 简化的评分逻辑
            if len(data) < 20:
                return 0.0

            # 基于最近价格变化计算评分
            recent_data = data.tail(10)
            = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]

            # 根据策略类型调整评分
            if self.type == 'momentum':
                return max(0.0, min(100.0, 50.0 + * 1000))
            elif self.type == 'trend':
                return max(0.0, min(100.0, 45.0 + * 800))
            elif self.type == 'volume_price':
                volume_trend = recent_data['volume'].pct_change().mean()
                return max(0.0, min(100.0, 40.0 + * 600 + volume_trend * 200))
            elif self.type == 'volatility':
                volatility = recent_data['close'].pct_change().std()
                return max(0.0, min(100.0, 35.0 + volatility * 1000))
            else:
                return 0.0

        except Exception as e:
            logger.error(f"策略 {self.strategy_id} 评分计算失败: {e}")
            return 0.0

    def _momentum_signal(self, data) -> str:
        """动量策略信号"""
        recent_data = data.tail(5)
        = recent_data['close'].pct_change().mean()
        if > 0.02:
            return "BUY"
        elif < -0.02:
            return "SELL"
        return "HOLD"

    def _trend_signal(self, data) -> str:
        """趋势策略信号"""
        if len(data) < 20:
            return "HOLD"
        ma_short = data['close'].tail(5).mean()
        ma_long = data['close'].tail(20).mean()
        if ma_short > ma_long * 1.01:
            return "BUY"
        elif ma_short < ma_long * 0.99:
            return "SELL"
        return "HOLD"

    def _volume_price_signal(self, data) -> str:
        """量价策略信号"""
        recent_data = data.tail(3)
        price_up = recent_data['close'].iloc[-1] > recent_data['close'].iloc[0]
        volume_up = recent_data['volume'].iloc[-1] > recent_data['volume'].mean()
        if price_up and volume_up:
            return "BUY"
        elif not price_up and volume_up:
            return "SELL"
        return "HOLD"

    def _volatility_signal(self, data) -> str:
        """波动率策略信号"""
        recent_data = data.tail(10)
        volatility = recent_data['close'].pct_change().std()
        = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        if volatility > 0.03 and > 0:
            return "BUY"
        elif volatility > 0.03 and < 0:
            return "SELL"
        return "HOLD"


class StrategyManager:
    """
    策略管理器
    """
    
    def __init__(self, data_access: Optional[DataAccessInterface] = None,
                 cache_manager: Optional[ICacheService] = None):
        """
        初始化策略管理器

        Args:
            data_access: 数据访问接口
            cache_manager: 缓存管理器
        """
        # 使用依赖注入容器获取服务
        try:
            self.data_access = data_access or get_service(DataAccessInterface)
        except Exception as e:
            logger.warning(f"数据访问服务不可用: {e}")
            self.data_access = None

        # 缓存服务为可选
        try:
            self.cache_manager = cache_manager or get_service(ICacheService)
        except Exception as e:
            logger.warning(f"缓存服务不可用: {e}")
            self.cache_manager = None

        # 初始化SQL管理器
        try:
            # 使用正确的SQL管理器
            self.sql_manager = get_sql_manager()
        except Exception as e:
            logger.warning(f"SQL管理器初始化失败: {e}")
            self.sql_manager = None

        self.strategy_dir = get_strategy_dir()

        # 确保策略目录存在
        os.makedirs(self.strategy_dir, exist_ok=True)

        # 初始化内置策略
        self.strategies = self._initialize_builtin_strategies()
    
    @exception_handler(reraise=True)
    @performance_monitor(threshold=1.0)
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
    @performance_monitor(threshold=1.0)
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
    @performance_monitor(threshold=0.5)
    def get_strategy(self, strategy_id: str):
        """
        获取策略实例

        Args:
            strategy_id: 策略ID

        Returns:
            策略实例，如果不存在则返回None
        """
        # 首先检查内置策略
        if strategy_id in self.strategies:
            return self.strategies[strategy_id]

        # 然后尝试从数据库获取
        strategy_config = self._get_strategy_from_db(strategy_id)
        if strategy_config:
            # 创建策略实例
            return SimpleStrategy(strategy_id, strategy_config)

        # 最后尝试从文件获取
        strategy_config = self._get_strategy_from_file(strategy_id)
        if strategy_config:
            return SimpleStrategy(strategy_id, strategy_config)

        return None
    
    @exception_handler(reraise=False, default_return=[])
    @performance_monitor(threshold=2.0)
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

    @exception_handler(reraise=False, default_return={})
    @performance_monitor(threshold=1.0)
    def get_available_strategies(self) -> Dict[str, Any]:
        """
        获取可用策略列表

        Returns:
            Dict[str, Any]: 可用策略字典，键为策略ID，值为策略信息
        """
        try:
            # 获取所有策略
            strategies = self.list_strategies()

            # 转换为字典格式
            available_strategies = {}
            for strategy in strategies:
                strategy_id = strategy.get('id', strategy.get('strategy', {}).get('id'))
                if strategy_id:
                    available_strategies[strategy_id] = {
                        'name': strategy.get('name', strategy.get('strategy', {}).get('name', strategy_id)),
                        'description': strategy.get('description', strategy.get('strategy', {}).get('description', '')),
                        'type': strategy.get('type', strategy.get('strategy', {}).get('type', 'unknown')),
                        'is_active': strategy.get('is_active', strategy.get('strategy', {}).get('is_active', True))
                    }

            # 添加内置策略
            builtin_strategies = self._get_builtin_strategies()
            available_strategies.update(builtin_strategies)

            logger.info(f"获取到 {len(available_strategies)} 个可用策略")
            return available_strategies

        except Exception as e:
            logger.error(f"获取可用策略失败: {e}")
            return {}

    def _initialize_builtin_strategies(self) -> Dict[str, Any]:
        """初始化内置策略实例"""
        from strategy.unified_base_strategy import UnifiedBaseStrategy
from db.sql_manager import SQLManager, QueryType

        strategies = {}

        # 创建内置策略实例
        builtin_configs = {
            'MULTI_PERIOD_MOMENTUM': {
                'name': '多周期动量策略',
                'description': '基于多周期动量指标的选股策略',
                'type': 'momentum',
                'is_active': True
            },
            'TREND_FOLLOWING': {
                'name': '趋势跟踪策略',
                'description': '基于趋势指标的跟踪策略',
                'type': 'trend',
                'is_active': True
            },
            'VOLUME_PRICE_ANALYSIS': {
                'name': '量价分析策略',
                'description': '基于成交量和价格关系的分析策略',
                'type': 'volume_price',
                'is_active': True
            },
            'VOLATILITY_BREAKOUT': {
                'name': '波动率突破策略',
                'description': '基于波动率突破的选股策略',
                'type': 'volatility',
                'is_active': True
            }
        }

        for strategy_id, config in builtin_configs.items():
            try:
                # 创建策略实例
                strategy = SimpleStrategy(strategy_id, config)
                strategies[strategy_id] = strategy
                logger.info(f"成功初始化内置策略: {strategy_id}")
            except Exception as e:
                logger.error(f"初始化策略 {strategy_id} 失败: {e}")

        return strategies

    def _get_builtin_strategies(self) -> Dict[str, Any]:
        """获取内置策略信息"""
        return {
            'MULTI_PERIOD_MOMENTUM': {
                'name': '多周期动量策略',
                'description': '基于多周期动量指标的选股策略',
                'type': 'momentum',
                'is_active': True
            },
            'TREND_FOLLOWING': {
                'name': '趋势跟踪策略',
                'description': '基于趋势指标的跟踪策略',
                'type': 'trend',
                'is_active': True
            },
            'VOLUME_PRICE_ANALYSIS': {
                'name': '量价分析策略',
                'description': '基于成交量和价格关系的分析策略',
                'type': 'volume_price',
                'is_active': True
            },
            'VOLATILITY_BREAKOUT': {
                'name': '波动率突破策略',
                'description': '基于波动率突破的选股策略',
                'type': 'volatility',
                'is_active': True
            }
        }

    @exception_handler(reraise=False, default_return=False)
    @performance_monitor(threshold=1.0)
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
            query = self.sql_manager.get_query(QueryType.SAVE_STRATEGY_CONFIG.value)
            
            params = {
                'strategy_id': strategy.get('id'),
                'config': json.dumps(strategy_config, ensure_ascii=False)
            }
            
            # 验证参数
            if not self.sql_manager.validate_params(QueryType.SAVE_STRATEGY_CONFIG.value, params):
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
            query = self.sql_manager.get_query(QueryType.STRATEGY_CONFIG.value)
            params = {'strategy_id': strategy_id}

            # 验证参数
            if not self.sql_manager.validate_params(QueryType.STRATEGY_CONFIG.value, params):
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
                except json.JSONDecodeError:
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
                        
                    except (json.JSONDecodeError, IOError) as e:
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
            UPDATE strategy_definitions SET is_active = 0, updated_at = NOW()
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