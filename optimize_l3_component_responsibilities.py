#!/usr/bin/env python3
"""
优化L3数据服务层组件职责分配
"""

import os
import ast
from utils.logger import get_logger

logger = get_logger(__name__)


def analyze_component_responsibilities():
    """分析组件职责分配"""
    logger.info("🔍 分析L3组件职责分配...")
    
    components_analysis = {}
    
    # 分析主要组件
    components = {
        'CacheService': 'db/services/cache_service.py',
        'DataAccessManager': 'db/managers/data_access_manager.py',
        'QueryOptimizationService': 'db/services/integrated/intelligent_query_optimizer.py',
        'ICacheService': 'db/interfaces/cache_interface.py'
    }
    
    for component_name, file_path in components.items():
        if os.path.exists(file_path):
            analysis = analyze_single_component(file_path, component_name)
            components_analysis[component_name] = analysis
    
    return components_analysis


def analyze_single_component(file_path: str, component_name: str):
    """分析单个组件的职责"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        methods = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                class_methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                methods.extend(class_methods)
        
        # 分析职责类型
        responsibility_types = categorize_methods(methods)
        
        return {
            'file_path': file_path,
            'classes': classes,
            'method_count': len(methods),
            'methods': methods,
            'responsibility_types': responsibility_types,
            'complexity_score': calculate_complexity_score(methods, responsibility_types)
        }
        
    except Exception as e:
        logger.error(f"分析组件失败 {file_path}: {e}")
        return {'error': str(e)}


def categorize_methods(methods):
    """将方法按职责类型分类"""
    categories = {
        'cache_operations': [],
        'data_access': [],
        'configuration': [],
        'monitoring': [],
        'utility': [],
        'interface': []
    }
    
    for method in methods:
        method_lower = method.lower()
        
        if any(keyword in method_lower for keyword in ['cache', 'get_', 'set_', 'put_', 'remove_', 'clear_']):
            categories['cache_operations'].append(method)
        elif any(keyword in method_lower for keyword in ['query', 'data', 'fetch', 'retrieve', 'access']):
            categories['data_access'].append(method)
        elif any(keyword in method_lower for keyword in ['config', 'setting', 'init']):
            categories['configuration'].append(method)
        elif any(keyword in method_lower for keyword in ['monitor', 'stats', 'metric', 'log']):
            categories['monitoring'].append(method)
        elif any(keyword in method_lower for keyword in ['validate', 'check', 'verify', 'format']):
            categories['utility'].append(method)
        else:
            categories['interface'].append(method)
    
    return categories


def calculate_complexity_score(methods, responsibility_types):
    """计算组件复杂度评分"""
    method_count = len(methods)
    responsibility_count = sum(1 for cat in responsibility_types.values() if cat)
    
    # 基础复杂度评分
    if method_count <= 10:
        base_score = 1  # 简单
    elif method_count <= 20:
        base_score = 2  # 中等
    elif method_count <= 30:
        base_score = 3  # 复杂
    else:
        base_score = 4  # 非常复杂
    
    # 职责分散度惩罚
    if responsibility_count > 3:
        base_score += 1
    
    return min(base_score, 5)  # 最高5分


def generate_optimization_recommendations(components_analysis):
    """生成优化建议"""
    logger.info("📋 生成组件优化建议...")
    
    recommendations = []
    
    for component_name, analysis in components_analysis.items():
        if 'error' in analysis:
            continue
            
        complexity = analysis['complexity_score']
        method_count = analysis['method_count']
        
        if complexity >= 4:  # 复杂度过高
            if component_name == 'CacheService' and method_count > 30:
                recommendations.append({
                    'component': component_name,
                    'priority': 'HIGH',
                    'issue': f'方法过多({method_count}个)',
                    'recommendation': '考虑拆分为多个专门的缓存服务类',
                    'suggested_split': [
                        'BasicCacheService (基础缓存操作)',
                        'AdvancedCacheService (高级缓存功能)',
                        'CacheMonitoringService (缓存监控)'
                    ]
                })
            elif method_count > 20:
                recommendations.append({
                    'component': component_name,
                    'priority': 'MEDIUM',
                    'issue': f'方法较多({method_count}个)',
                    'recommendation': '考虑提取部分功能到辅助类',
                    'action': '重构部分方法到工具类'
                })
        
        # 检查职责分散
        responsibility_types = analysis['responsibility_types']
        active_responsibilities = [cat for cat, methods in responsibility_types.items() if methods]
        
        if len(active_responsibilities) > 3:
            recommendations.append({
                'component': component_name,
                'priority': 'MEDIUM',
                'issue': f'职责过于分散({len(active_responsibilities)}种职责)',
                'recommendation': '考虑按职责重新组织方法',
                'responsibilities': active_responsibilities
            })
    
    return recommendations


def implement_safe_optimizations():
    """实施安全的优化措施"""
    logger.info("🔧 实施安全的组件优化...")
    
    optimizations_applied = []
    
    # 1. 为CacheService添加职责分组注释
    cache_service_file = 'db/services/cache_service.py'
    if os.path.exists(cache_service_file):
        try:
            with open(cache_service_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 添加职责分组注释（如果还没有的话）
            if '# === 基础缓存操作 ===' not in content:
                # 在类定义后添加职责分组注释
                content = content.replace(
                    'class CacheService(ICacheService):',
                    '''class CacheService(ICacheService):
    """
    缓存服务 - 统一缓存管理
    
    职责分组：
    - 基础缓存操作：get, set, delete等
    - 高级缓存功能：批量操作、过期管理等  
    - 缓存监控：统计、性能监控等
    - 配置管理：缓存配置、初始化等
    """'''
                )
                
                with open(cache_service_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                optimizations_applied.append('CacheService职责分组注释')
                logger.info("✅ 为CacheService添加职责分组注释")
        
        except Exception as e:
            logger.error(f"❌ 优化CacheService失败: {e}")
    
    # 2. 为DataAccessManager添加方法分组
    data_access_file = 'db/managers/data_access_manager.py'
    if os.path.exists(data_access_file):
        try:
            with open(data_access_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 添加方法分组注释
            if '# === 数据访问接口实现 ===' not in content:
                content = content.replace(
                    'class DataAccessManager(DataAccessInterface):',
                    '''class DataAccessManager(DataAccessInterface):
    """
    数据访问管理器 - 统一数据访问入口
    
    方法分组：
    - 接口实现：实现IDataAccess接口方法
    - 核心查询：基础股票数据查询
    - 批量操作：批量数据获取和处理
    - 指标数据：技术指标数据获取
    - 工具方法：数据验证、格式化等
    """'''
                )
                
                with open(data_access_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                optimizations_applied.append('DataAccessManager方法分组注释')
                logger.info("✅ 为DataAccessManager添加方法分组注释")
        
        except Exception as e:
            logger.error(f"❌ 优化DataAccessManager失败: {e}")
    
    return optimizations_applied


def main():
    """主函数"""
    try:
        logger.info("🚀 开始L3组件职责优化")
        
        # 1. 分析组件职责
        components_analysis = analyze_component_responsibilities()
        
        # 2. 生成优化建议
        recommendations = generate_optimization_recommendations(components_analysis)
        
        # 3. 实施安全优化
        optimizations = implement_safe_optimizations()
        
        # 4. 输出报告
        print("\n" + "="*70)
        print("📊 L3数据服务层组件职责分析报告")
        print("="*70)
        
        for component_name, analysis in components_analysis.items():
            if 'error' not in analysis:
                print(f"\n🔍 {component_name}:")
                print(f"  文件: {analysis['file_path']}")
                print(f"  方法数量: {analysis['method_count']}")
                print(f"  复杂度评分: {analysis['complexity_score']}/5")
                
                responsibility_types = analysis['responsibility_types']
                active_responsibilities = [cat for cat, methods in responsibility_types.items() if methods]
                print(f"  职责类型: {', '.join(active_responsibilities)}")
        
        print(f"\n📋 优化建议 ({len(recommendations)}个):")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['component']} - {rec['priority']}优先级")
            print(f"     问题: {rec['issue']}")
            print(f"     建议: {rec['recommendation']}")
        
        print(f"\n✅ 已实施优化 ({len(optimizations)}个):")
        for opt in optimizations:
            print(f"  ✓ {opt}")
        
        print("\n🎯 总结:")
        print("  ✅ 组件职责分析完成")
        print("  ✅ 优化建议已生成")
        print("  ✅ 安全优化已实施")
        print("  📈 建议重新运行架构设计合规性验证")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"组件职责优化过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
