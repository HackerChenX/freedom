#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略配置迁移脚本

执行第一阶段重构：基础设施整合
将现有策略配置迁移到统一格式，遵循六层架构规范
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.logger import getLogger
from utils.strategy_config_migrator import migrate_strategies
from utils.strategy_validator import UnifiedStrategyConfigValidator
from utils.cache import cleanup_all_caches, get_cache_stats

logger = getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='策略配置迁移工具')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='是否备份原始文件（默认：True）')
    parser.add_argument('--validate-only', action='store_true', default=False,
                       help='仅验证配置格式，不执行迁移')
    parser.add_argument('--clean-cache', action='store_true', default=False,
                       help='迁移前清理缓存')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("策略配置迁移工具 - 第一阶段：基础设施整合")
    logger.info("=" * 60)
    
    try:
        # 清理缓存（如果需要）
        if args.clean_cache:
            logger.info("清理缓存...")
            cleanup_all_caches()
            logger.info("缓存清理完成")
        
        # 显示缓存统计
        if args.verbose:
            cache_stats = get_cache_stats()
            logger.info(f"缓存统计: {cache_stats}")
        
        # 验证配置格式
        if args.validate_only:
            logger.info("开始验证现有配置格式...")
            validate_existing_configs()
            logger.info("配置格式验证完成")
            return
        
        # 执行迁移
        logger.info("开始策略配置迁移...")
        migration_result = migrate_strategies(backup_original=args.backup)
        
        # 显示迁移结果
        display_migration_results(migration_result)
        
        # 验证迁移后的配置
        logger.info("验证迁移后的配置...")
        validate_migrated_configs()
        
        logger.info("策略配置迁移完成！")
        
    except Exception as e:
        logger.error(f"策略配置迁移失败: {e}")
        sys.exit(1)


def validate_existing_configs():
    """验证现有配置格式"""
    validator = UnifiedStrategyConfigValidator()
    config_dir = os.path.join(root_dir, 'config', 'strategies')
    
    if not os.path.exists(config_dir):
        logger.warning(f"策略配置目录不存在: {config_dir}")
        return
    
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    for root, dirs, files in os.walk(config_dir):
        # 跳过已统一格式的目录
        if 'standardized' in root or 'legacy' in root:
            continue
        
        for file in files:
            if file.endswith(('.yaml', '.yml', '.json')):
                file_path = os.path.join(root, file)
                total_files += 1
                
                try:
                    # 加载配置文件
                    import json
                    import yaml
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.endswith('.json'):
                            config = json.load(f)
                        else:
                            config = yaml.safe_load(f)
                    
                    # 验证配置
                    validation_result = validator.validate_strategy_config(config)
                    
                    if validation_result['is_valid']:
                        valid_files += 1
                        logger.debug(f"配置有效: {file}")
                    else:
                        invalid_files.append({
                            'file': file,
                            'errors': validation_result['errors']
                        })
                        logger.warning(f"配置无效: {file}, 错误: {validation_result['errors']}")
                
                except Exception as e:
                    invalid_files.append({
                        'file': file,
                        'errors': [f"加载失败: {str(e)}"]
                    })
                    logger.error(f"加载配置文件失败: {file}, 错误: {e}")
    
    logger.info(f"配置验证完成 - 总计: {total_files}, 有效: {valid_files}, 无效: {len(invalid_files)}")
    
    if invalid_files:
        logger.info("无效配置文件详情:")
        for item in invalid_files:
            logger.info(f"  - {item['file']}: {item['errors']}")


def validate_migrated_configs():
    """验证迁移后的配置"""
    validator = UnifiedStrategyConfigValidator()
    unified_config_dir = os.path.join(root_dir, 'config', 'strategies', 'standardized')
    
    if not os.path.exists(unified_config_dir):
        logger.warning(f"统一配置目录不存在: {unified_config_dir}")
        return
    
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    for file in os.listdir(unified_config_dir):
        if file.endswith('.json') and not file.startswith('migration_report'):
            file_path = os.path.join(unified_config_dir, file)
            total_files += 1
            
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                validation_result = validator.validate_strategy_config(config)
                
                if validation_result['is_valid']:
                    valid_files += 1
                    logger.debug(f"统一配置有效: {file}")
                else:
                    invalid_files.append({
                        'file': file,
                        'errors': validation_result['errors']
                    })
                    logger.error(f"统一配置无效: {file}, 错误: {validation_result['errors']}")
            
            except Exception as e:
                invalid_files.append({
                    'file': file,
                    'errors': [f"加载失败: {str(e)}"]
                })
                logger.error(f"加载统一配置失败: {file}, 错误: {e}")
    
    logger.info(f"统一配置验证完成 - 总计: {total_files}, 有效: {valid_files}, 无效: {len(invalid_files)}")
    
    if invalid_files:
        logger.error("发现无效的统一配置文件:")
        for item in invalid_files:
            logger.error(f"  - {item['file']}: {item['errors']}")
        raise Exception("统一配置验证失败")


def display_migration_results(migration_result):
    """显示迁移结果"""
    logger.info("=" * 50)
    logger.info("迁移结果统计")
    logger.info("=" * 50)
    
    logger.info(f"总文件数: {migration_result['total_files']}")
    logger.info(f"成功迁移: {migration_result['migrated_successfully']}")
    logger.info(f"迁移失败: {migration_result['migration_failed']}")
    logger.info(f"跳过文件: {migration_result['skipped_files']}")
    
    if migration_result['migrated_files']:
        logger.info("\n成功迁移的文件:")
        for item in migration_result['migrated_files']:
            logger.info(f"  - {os.path.basename(item['original_file'])} -> {os.path.basename(item['unified_file'])}")
    
    if migration_result['failed_files']:
        logger.warning("\n迁移失败的文件:")
        for item in migration_result['failed_files']:
            logger.warning(f"  - {os.path.basename(item['file'])}: {item['error']}")
    
    # 计算成功率
    if migration_result['total_files'] > 0:
        success_rate = (migration_result['migrated_successfully'] / migration_result['total_files']) * 100
        logger.info(f"\n迁移成功率: {success_rate:.1f}%")
    
    # 显示时间信息
    start_time = datetime.fromisoformat(migration_result['start_time'])
    end_time = datetime.fromisoformat(migration_result['end_time'])
    duration = end_time - start_time
    logger.info(f"迁移耗时: {duration.total_seconds():.2f} 秒")


if __name__ == "__main__":
    main()
