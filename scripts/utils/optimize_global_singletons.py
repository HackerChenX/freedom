#!/usr/bin/env python3
"""
全局单例优化脚本

将全局单例模式迁移到依赖注入容器，减少全局状态依赖。
"""

import os
import sys
import re
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from utils.logger import get_logger
from utils.path_utils import get_project_root

logger = get_logger(__name__)


class SingletonOptimizer:
    """全局单例优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.project_root = get_project_root()
        
        # 需要优化的单例模块
        self.singleton_modules = [
            {
                'file': 'strategy/strategy_format_converter.py',
                'class': 'StrategyFormatConverter',
                'instance_var': '_converter_instance',
                'getter_func': 'get_strategy_format_converter'
            },
            {
                'file': 'db/db_manager.py',
                'class': 'DBManager',
                'instance_var': '_instance',
                'getter_func': 'get_db_manager'
            },
            {
                'file': 'db/data_access.py',
                'class': 'ClickHouseDB',
                'instance_var': '_instance',
                'getter_func': 'get_clickhouse_db'
            },
            {
                'file': 'indicators/pattern_registry.py',
                'class': 'PatternRegistry',
                'instance_var': '_instance',
                'getter_func': 'get_pattern_registry'
            },
            {
                'file': 'utils/period_manager.py',
                'class': 'PeriodManager',
                'instance_var': '_instance',
                'getter_func': 'get_period_manager'
            },
            {
                'file': 'utils/cache.py',
                'class': 'CacheManager',
                'instance_var': '_instance',
                'getter_func': 'get_cache_manager'
            },
            {
                'file': 'analysis/integration/unified_data_adapter.py',
                'class': 'UnifiedDataAdapter',
                'instance_var': '_unified_adapter_instance',
                'getter_func': 'get_unified_data_adapter'
            },
            {
                'file': 'analysis/integration/unified_analysis_engine.py',
                'class': 'UnifiedAnalysisEngine',
                'instance_var': '_unified_engine_instance',
                'getter_func': 'get_unified_analysis_engine'
            }
        ]
        
        self.optimization_stats = {
            'total_modules': len(self.singleton_modules),
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'container_registrations': 0
        }
    
    def optimize_all_singletons(self) -> Dict[str, int]:
        """优化所有全局单例"""
        logger.info("开始优化全局单例模块...")
        
        for module_info in self.singleton_modules:
            self.optimize_singleton_module(module_info)
        
        # 更新容器配置
        self.update_container_configuration()
        
        # 输出统计信息
        logger.info("全局单例优化完成！统计信息:")
        logger.info(f"  总模块数: {self.optimization_stats['total_modules']}")
        logger.info(f"  成功优化: {self.optimization_stats['successful_optimizations']}")
        logger.info(f"  失败优化: {self.optimization_stats['failed_optimizations']}")
        logger.info(f"  容器注册: {self.optimization_stats['container_registrations']}")
        
        return self.optimization_stats
    
    def optimize_singleton_module(self, module_info: Dict[str, str]) -> bool:
        """优化单个单例模块"""
        try:
            file_path = os.path.join(self.project_root, module_info['file'])
            logger.info(f"优化单例模块: {module_info['file']}")
            
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {module_info['file']}")
                self.optimization_stats['failed_optimizations'] += 1
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已经优化过
            if 'from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons' in content:
                logger.info(f"模块已优化: {module_info['file']}")
                self.optimization_stats['successful_optimizations'] += 1
                return True
            
            # 添加容器导入
            content = self._add_container_import(content)
            
            # 修改getter函数
            content = self._modify_getter_function(content, module_info)
            
            # 添加容器注册说明
            content = self._add_container_registration_comment(content, module_info)
            
            # 写入修改后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.optimization_stats['successful_optimizations'] += 1
            logger.info(f"单例模块优化完成: {module_info['file']}")
            return True
            
        except Exception as e:
            logger.error(f"优化单例模块失败: {module_info['file']}, 错误: {e}")
            self.optimization_stats['failed_optimizations'] += 1
            return False
    
    def _add_container_import(self, content: str) -> str:
        """添加容器导入"""
        if 'from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons' not in content:
            # 在现有导入后添加容器导入
            import_pattern = r'(from *? import .*?\n)'
            matches = list(re.finditer(import_pattern, content))
            if matches:
                # 在最后一个导入后添加
                last_import = matches[-1]
                insert_pos = last_import.end()
                content = (content[:insert_pos] + 
                          'from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons\n' + 
                          content[insert_pos:])
            else:
                # 在文件开头添加
                content = 'from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons\n' + content
        
        return content
    
    def _modify_getter_function(self, content: str, module_info: Dict[str, str]) -> str:
        """修改getter函数使用容器"""
        func_name = module_info['getter_func']
        class_name = module_info['class']
        
        # 查找getter函数
        func_pattern = rf'def {func_name}\([^)]*\):[^:]*?return.*?{module_info["instance_var"]}'
        
        # 新的函数实现
        new_func = f'''def {func_name}():
    """获取{class_name}实例（通过依赖注入容器）"""
    container = get_container()
    if not container.is_registered({class_name}):
        # 如果未注册，使用单例模式注册
        container.register_singleton({class_name}, {class_name})
    return get_service_Optimize_Global_Singletons_Optimize_Global_Singletons(DataAccessInterface)'''
        
        # 替换函数实现
        content = re.sub(func_pattern, new_func, content, flags=re.DOTALL)
        
        return content
    
    def _add_container_registration_comment(self, content: str, module_info: Dict[str, str]) -> str:
        """添加容器注册说明注释"""
        class_name = module_info['class']
        
        comment = f'''
# 注意：{class_name}现在通过依赖注入容器管理
# 可以在应用启动时预注册：
# container.register_singleton({class_name}, {class_name})
'''
        
        # 在文件末尾添加注释
        content += comment
        
        return content
    
    def update_container_configuration(self):
        """更新容器配置"""
        logger.info("更新容器配置...")
        
        container_config_path = os.path.join(self.project_root, 'config', 'container_config.py')
        
        # 创建容器配置文件
        config_content = '''"""
容器配置文件

定义依赖注入容器的服务注册配置
"""

from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons
from db.interfaces.data_access_interface import IData_access
from db.managers.data_access_manager import Data_access_manager

# 导入需要注册的服务类
from strategy.strategy_format_converter import Strategy_format_converter
from db.db_manager import DBManager
from db.interfaces.data_access_interface import Data_access_interface
from indicators.pattern_registry import Pattern_registry
from utils.period_manager import Period_manager
from utils.cache import Cache_manager
from analysis.integration.unified_data_adapter import Unified_data_adapter
from analysis.integration.unified_analysis_engine import Unified_analysis_engine


def configure_container_Optimize_Global_Singletons():
    """配置依赖注入容器"""
    container = get_container()
    
    # 注册核心服务
    container.register_singleton(IData_access, Data_access_manager)
    
    # 注册原全局单例服务
    container.register_singleton(Strategy_format_converter, Strategy_format_converter)
    container.register_singleton(DBManager, DBManager)
    container.register_singleton(Click_house_dB, Click_house_dB)
    container.register_singleton(Pattern_registry, Pattern_registry)
    container.register_singleton(Period_manager, Period_manager)
    container.register_singleton(Cache_manager, Cache_manager)
    container.register_singleton(Unified_data_adapter, Unified_data_adapter)
    container.register_singleton(Unified_analysis_engine, Unified_analysis_engine)
    
    return container


def get_configured_container_Optimize_Global_Singletons():
    """获取配置好的容器实例"""
    return configure_container_Optimize_Global_Singletons()
'''
        
        os.makedirs(os.path.dirname(container_config_path), exist_ok=True)
        with open(container_config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        self.optimization_stats['container_registrations'] = 8
        logger.info(f"容器配置文件已创建: {container_config_path}")
    
    def generate_optimization_report_Singletons(self) -> str:
        """生成优化报告"""
        report = f"""
# 全局单例优化报告

## 优化统计

- **总模块数**: {self.optimization_stats['total_modules']}
- **成功优化**: {self.optimization_stats['successful_optimizations']}
- **失败优化**: {self.optimization_stats['failed_optimizations']}
- **容器注册**: {self.optimization_stats['container_registrations']}

## 优化内容

### 优化的单例模块

1. **strategy/strategy_format_converter.py** - Strategy_format_converter
2. **db/db_manager.py** - DBManager
3. **db/data_access.py** - Click_house_dB
4. **indicators/pattern_registry.py** - Pattern_registry
5. **utils/period_manager.py** - Period_manager
6. **utils/cache.py** - Cache_manager
7. **analysis/integration/unified_data_adapter.py** - Unified_data_adapter
8. **analysis/integration/unified_analysis_engine.py** - Unified_analysis_engine

### 优化效果

- ✅ 消除了全局状态依赖
- ✅ 建立了统一的依赖注入机制
- ✅ 提高了代码的可测试性
- ✅ 增强了模块间的解耦
- ✅ 支持生命周期管理

## 架构改进

### 优化前
使用全局单例模式，通过全局变量管理实例状态。

### 优化后
使用依赖注入容器模式，通过容器管理服务生命周期。

## 使用指南

### 应用启动时配置
```python
from config.container_config import configure_container_Optimize_Global_Singletons

# 在应用启动时配置容器
container = configure_container_Optimize_Global_Singletons()
```

### 获取服务实例
```python
# 通过容器获取服务
from utils.dependency_injection import get_service_Optimize_Global_Singletons_Optimize_Global_Singletons

container = get_container()
service = get_service_Optimize_Global_Singletons_Optimize_Global_Singletons(Data_access_interface)
```

"""
        return report


def main_optimizeglobalsingletons():
    """主函数"""
    print("开始全局单例优化...")
    
    # 创建优化器实例
    optimizer = SingletonOptimizer()
    
    # 执行优化
    stats = optimizer.optimize_all_singletons()
    
    # 输出结果
    print(f"\n优化完成！")
    print(f"总模块数: {stats['total_modules']}")
    print(f"成功优化: {stats['successful_optimizations']}")
    print(f"失败优化: {stats['failed_optimizations']}")
    print(f"容器注册: {stats['container_registrations']}")
    
    # 生成报告
    report = optimizer.generate_optimization_report_Singletons()
    print(f"\n优化报告:\n{report}")


if __name__ == "__main__":
    main_optimizeglobalsingletons() 