#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全面系统检查：识别所有潜在问题
"""

import sys
import os
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def check_indicator_functionality():
    """检查指标功能完整性"""
    logger.info("🔧 检查指标功能完整性...")
    
    issues = []
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        all_indicators = registry.get_all_indicators()
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000000 + i * 1000 for i in range(100)]
        })
        
        # 测试每个指标的基本功能
        functional_count = 0
        broken_indicators = []
        
        for indicator_name in list(all_indicators.keys())[:10]:  # 测试前10个指标
            try:
                indicator = registry.create_indicator(indicator_name)
                result = indicator.calculate(test_data)
                
                if result is not None and not result.empty:
                    functional_count += 1
                else:
                    broken_indicators.append(f"{indicator_name}: 返回空结果")
                    
            except Exception as e:
                broken_indicators.append(f"{indicator_name}: {str(e)[:100]}")
        
        if broken_indicators:
            issues.append(f"功能性问题: {len(broken_indicators)}个指标存在计算问题")
            for issue in broken_indicators[:5]:  # 只显示前5个
                logger.warning(f"  ⚠️ {issue}")
        else:
            logger.info(f"  ✅ 测试的{functional_count}个指标功能正常")
            
    except Exception as e:
        issues.append(f"指标功能检查失败: {e}")
        logger.error(f"  ❌ 指标功能检查失败: {e}")
    
    return issues


def check_documentation_completeness():
    """检查文档完整性"""
    logger.info("📚 检查文档完整性...")
    
    issues = []
    
    # 检查关键文档文件
    required_docs = [
        'docs/finaltesting/技术指标验证进度表.md',
        'README.md',
        'docs/architecture.md'
    ]
    
    missing_docs = []
    for doc_path in required_docs:
        full_path = Path(root_dir) / doc_path
        if not full_path.exists():
            missing_docs.append(doc_path)
    
    if missing_docs:
        issues.append(f"文档缺失: {len(missing_docs)}个关键文档文件不存在")
        for doc in missing_docs:
            logger.warning(f"  ⚠️ 缺失文档: {doc}")
    else:
        logger.info("  ✅ 关键文档文件完整")
    
    # 检查进度表数据一致性
    try:
        progress_file = Path(root_dir) / "docs/finaltesting/技术指标验证进度表.md"
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查统计数据一致性
        total_pattern = r'验证通过指标数.*?(\d+)个'
        total_match = re.search(total_pattern, content)
        
        table_pattern = r'\|\s*\*\*([A-Z_]+)\*\*\s*\|\s*✅ PASSED'
        table_matches = re.findall(table_pattern, content)
        
        if total_match:
            declared_total = int(total_match.group(1))
            actual_total = len(table_matches)
            
            if declared_total != actual_total:
                issues.append(f"进度表数据不一致: 声明{declared_total}个，实际{actual_total}个")
                logger.warning(f"  ⚠️ 进度表数据不一致: 声明{declared_total}个，实际{actual_total}个")
            else:
                logger.info(f"  ✅ 进度表数据一致: {actual_total}个指标")
        
    except Exception as e:
        issues.append(f"进度表检查失败: {e}")
        logger.error(f"  ❌ 进度表检查失败: {e}")
    
    return issues


def check_code_quality():
    """检查代码质量问题"""
    logger.info("🔍 检查代码质量...")
    
    issues = []
    
    # 检查关键文件的代码质量
    key_files = [
        'indicators/complete_indicator_registry.py',
        'indicators/base_indicator.py'
    ]
    
    for file_path in key_files:
        full_path = Path(root_dir) / file_path
        if not full_path.exists():
            issues.append(f"关键文件缺失: {file_path}")
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查常见代码质量问题
            lines = content.split('\n')
            
            # 检查过长的行
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
            if len(long_lines) > 10:
                issues.append(f"{file_path}: {len(long_lines)}行代码过长")
            
            # 检查TODO/FIXME注释
            todo_count = content.count('TODO') + content.count('FIXME')
            if todo_count > 5:
                issues.append(f"{file_path}: {todo_count}个待办事项")
            
            # 检查重复代码模式
            if content.count('def calculate(') > 3:
                issues.append(f"{file_path}: 可能存在重复的calculate方法")
                
        except Exception as e:
            issues.append(f"代码质量检查失败 {file_path}: {e}")
    
    if not issues:
        logger.info("  ✅ 代码质量检查通过")
    else:
        for issue in issues:
            logger.warning(f"  ⚠️ {issue}")
    
    return issues


def check_performance_issues():
    """检查性能问题"""
    logger.info("⚡ 检查性能问题...")
    
    issues = []
    
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        
        # 检查注册性能
        import time
        start_time = time.time()
        registry.get_all_indicators()
        registration_time = time.time() - start_time
        
        if registration_time > 2.0:
            issues.append(f"注册性能问题: 指标注册耗时{registration_time:.2f}秒")
            logger.warning(f"  ⚠️ 注册性能问题: 耗时{registration_time:.2f}秒")
        else:
            logger.info(f"  ✅ 注册性能良好: {registration_time:.3f}秒")
        
        # 检查内存使用
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 500:
            issues.append(f"内存使用过高: {memory_mb:.1f}MB")
            logger.warning(f"  ⚠️ 内存使用: {memory_mb:.1f}MB")
        else:
            logger.info(f"  ✅ 内存使用正常: {memory_mb:.1f}MB")
            
    except Exception as e:
        issues.append(f"性能检查失败: {e}")
        logger.error(f"  ❌ 性能检查失败: {e}")
    
    return issues


def check_dependency_issues():
    """检查依赖问题"""
    logger.info("📦 检查依赖问题...")
    
    issues = []
    
    # 检查关键依赖导入
    critical_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('utils.dependency_injection', 'get_logger')
    ]
    
    for module, alias in critical_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
        except ImportError as e:
            issues.append(f"依赖缺失: {module} - {e}")
            logger.error(f"  ❌ 依赖缺失: {module}")
    
    if not issues:
        logger.info("  ✅ 关键依赖完整")
    
    # 检查循环导入
    try:
        from indicators.complete_indicator_registry import get_indicator_registry
        registry = get_indicator_registry()
        # 如果能成功创建，说明没有严重的循环导入
        logger.info("  ✅ 无严重循环导入问题")
    except Exception as e:
        issues.append(f"可能存在循环导入: {e}")
        logger.error(f"  ❌ 可能存在循环导入: {e}")
    
    return issues


def check_configuration_issues():
    """检查配置问题"""
    logger.info("⚙️ 检查配置问题...")
    
    issues = []
    
    # 检查配置文件
    config_files = [
        'config/database_config_manager.py',
        'config/config.py'
    ]
    
    missing_configs = []
    for config_file in config_files:
        full_path = Path(root_dir) / config_file
        if not full_path.exists():
            missing_configs.append(config_file)
    
    if missing_configs:
        issues.append(f"配置文件缺失: {missing_configs}")
        for config in missing_configs:
            logger.warning(f"  ⚠️ 配置文件缺失: {config}")
    else:
        logger.info("  ✅ 配置文件完整")
    
    return issues


def comprehensive_system_check():
    """执行全面系统检查"""
    logger.info("🚀 开始全面系统检查...")
    logger.info("=" * 80)
    
    all_issues = []
    
    # 执行各项检查
    checks = [
        ("指标功能完整性", check_indicator_functionality),
        ("文档完整性", check_documentation_completeness),
        ("代码质量", check_code_quality),
        ("性能问题", check_performance_issues),
        ("依赖问题", check_dependency_issues),
        ("配置问题", check_configuration_issues)
    ]
    
    for check_name, check_func in checks:
        try:
            issues = check_func()
            if issues:
                all_issues.extend(issues)
        except Exception as e:
            all_issues.append(f"{check_name}检查失败: {e}")
            logger.error(f"❌ {check_name}检查失败: {e}")
    
    # 生成最终报告
    logger.info("=" * 80)
    logger.info("📊 全面系统检查报告")
    logger.info("=" * 80)
    
    if not all_issues:
        logger.info("🎉 系统检查完美通过！")
        logger.info("✅ 未发现任何问题")
        logger.info("✅ 系统处于最佳状态")
        return True
    else:
        logger.warning(f"⚠️ 发现 {len(all_issues)} 个问题需要关注:")
        
        # 按严重程度分类
        critical_issues = [issue for issue in all_issues if any(keyword in issue.lower() for keyword in ['缺失', '失败', '错误', '异常'])]
        warning_issues = [issue for issue in all_issues if issue not in critical_issues]
        
        if critical_issues:
            logger.error(f"🔴 严重问题 ({len(critical_issues)}个):")
            for issue in critical_issues:
                logger.error(f"  ❌ {issue}")
        
        if warning_issues:
            logger.warning(f"🟡 警告问题 ({len(warning_issues)}个):")
            for issue in warning_issues:
                logger.warning(f"  ⚠️ {issue}")
        
        # 提供修复建议
        logger.info("\n💡 修复建议:")
        if critical_issues:
            logger.info("  1. 优先修复严重问题，确保系统基本功能")
        if warning_issues:
            logger.info("  2. 逐步改进警告问题，提升系统质量")
        logger.info("  3. 定期运行系统检查，保持系统健康")
        
        return len(critical_issues) == 0


def main():
    """主函数"""
    try:
        success = comprehensive_system_check()
        
        if success:
            logger.info("🎉 全面系统检查完成：系统状态优秀！")
        else:
            logger.warning("⚠️ 全面系统检查完成：发现需要修复的问题")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 系统检查过程中发生异常: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
