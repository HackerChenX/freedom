#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
更新技术指标验证进度表
整理当前验证状态，梳理未验证指标，更新进度表
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, List, Tuple
import re

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.dependency_injection import get_logger

logger = get_logger(__name__)


def get_validated_indicators_from_reports() -> Dict[str, Dict]:
    """从验证报告获取已验证的指标及其状态"""
    reports_dir = Path(root_dir) / "docs/finaltesting/indicators"
    validated_indicators = {}
    
    if not reports_dir.exists():
        return validated_indicators
    
    for report_file in reports_dir.glob("*_validation_report.md"):
        try:
            indicator_name = report_file.stem.replace("_validation_report", "").upper()
            
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取验证状态和评分
            status = "UNKNOWN"
            score = 0.0
            validation_time = "未知"
            
            # 查找验证状态
            status_patterns = [
                r'验证状态.*?✅\s*PASSED_PRODUCTION_READY',
                r'验证状态.*?✅\s*PASSED_ARCHITECTURE_COMPLIANT', 
                r'最终状态.*?PASSED_PRODUCTION_READY',
                r'最终状态.*?PASSED_ARCHITECTURE_COMPLIANT',
                r'最终状态.*?CONDITIONAL_PASS',
                r'最终状态.*?FAILED'
            ]
            
            for pattern in status_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    if 'PASSED_PRODUCTION_READY' in pattern:
                        status = "PASSED_PRODUCTION_READY"
                    elif 'PASSED_ARCHITECTURE_COMPLIANT' in pattern:
                        status = "PASSED_ARCHITECTURE_COMPLIANT"
                    elif 'CONDITIONAL_PASS' in pattern:
                        status = "CONDITIONAL_PASS"
                    elif 'FAILED' in pattern:
                        status = "FAILED"
                    break
            
            # 查找评分
            score_match = re.search(r'总体评分.*?(\d+\.?\d*)/100分', content)
            if score_match:
                score = float(score_match.group(1))
            else:
                # 查找综合评分
                score_match = re.search(r'综合评分.*?(\d+\.?\d*)/100', content)
                if score_match:
                    score = float(score_match.group(1))
            
            # 查找验证时间
            time_match = re.search(r'验证时间.*?(\d{4}-\d{2}-\d{2})', content)
            if time_match:
                validation_time = time_match.group(1)
            else:
                time_match = re.search(r'验证日期.*?(\d{4}-\d{2}-\d{2})', content)
                if time_match:
                    validation_time = time_match.group(1)
            
            validated_indicators[indicator_name] = {
                'status': status,
                'score': score,
                'validation_time': validation_time,
                'report_file': str(report_file)
            }
            
        except Exception as e:
            logger.warning(f"解析验证报告 {report_file} 失败: {e}")
    
    return validated_indicators


def get_all_indicators_from_registry() -> Set[str]:
    """从指标注册表获取所有指标"""
    try:
        from indicators.complete_indicator_registry import complete_registry
        return set(complete_registry.get_all_indicators().keys())
    except Exception as e:
        logger.error(f"获取注册表指标失败: {e}")
        return set()


def get_indicators_from_files() -> Set[str]:
    """从indicators目录获取所有指标文件"""
    indicators_dir = Path(root_dir) / "indicators"
    indicator_files = set()
    
    if indicators_dir.exists():
        for file_path in indicators_dir.glob("*.py"):
            if file_path.name not in ["__init__.py", "base_indicator.py"]:
                indicator_name = file_path.stem.upper()
                indicator_files.add(indicator_name)
    
    return indicator_files


def check_base_indicator_inheritance(indicator_name: str) -> bool:
    """检查指标是否继承BaseIndicator"""
    try:
        indicator_file = Path(root_dir) / "indicators" / f"{indicator_name.lower()}.py"
        if indicator_file.exists():
            with open(indicator_file, 'r', encoding='utf-8') as f:
                content = f.read()
                return "BaseIndicator" in content and "class" in content
        return False
    except Exception:
        return False


def categorize_indicators() -> Dict[str, List[str]]:
    """分类指标"""
    registry_indicators = get_all_indicators_from_registry()
    file_indicators = get_indicators_from_files()
    validated_indicators = get_validated_indicators_from_reports()
    
    all_indicators = registry_indicators.union(file_indicators)
    validated_names = set(validated_indicators.keys())
    
    # 分类
    categories = {
        'validated_passed': [],
        'validated_failed': [],
        'unvalidated_base_indicator': [],
        'unvalidated_factory_pattern': [],
        'system_files': []
    }
    
    for indicator in sorted(all_indicators):
        if indicator in validated_names:
            status = validated_indicators[indicator]['status']
            if status in ['PASSED_PRODUCTION_READY', 'PASSED_ARCHITECTURE_COMPLIANT']:
                categories['validated_passed'].append(indicator)
            else:
                categories['validated_failed'].append(indicator)
        else:
            # 未验证的指标
            if check_base_indicator_inheritance(indicator):
                # 检查是否是系统文件
                system_keywords = [
                    'FACTORY', 'REGISTRY', 'MANAGER', 'CALCULATOR', 'ADAPTER',
                    'COMMON', 'COMPLETE_INDICATOR_REGISTRY', 'PATTERN_REGISTRY',
                    'SERVICE_REGISTRY', 'VECTORIZATION', 'OPTIMIZER'
                ]
                
                if any(keyword in indicator for keyword in system_keywords):
                    categories['system_files'].append(indicator)
                else:
                    categories['unvalidated_base_indicator'].append(indicator)
            else:
                categories['unvalidated_factory_pattern'].append(indicator)
    
    return categories, validated_indicators


def generate_updated_progress_table() -> str:
    """生成更新后的进度表"""
    categories, validated_indicators = categorize_indicators()
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 统计数据
    total_indicators = sum(len(cat) for cat in categories.values())
    validated_count = len(categories['validated_passed']) + len(categories['validated_failed'])
    passed_count = len(categories['validated_passed'])
    failed_count = len(categories['validated_failed'])
    unvalidated_count = len(categories['unvalidated_base_indicator']) + len(categories['unvalidated_factory_pattern'])
    
    validation_rate = (validated_count / total_indicators * 100) if total_indicators > 0 else 0
    pass_rate = (passed_count / validated_count * 100) if validated_count > 0 else 0
    
    report = f"""# 技术指标验证进度表

## 📊 验证概览

**更新时间**: {current_time}

### 🎯 总体统计
- **总指标数**: {total_indicators}个
- **已验证指标数**: {validated_count}个 ({validation_rate:.1f}%)
- **验证通过数**: {passed_count}个
- **验证失败数**: {failed_count}个
- **待验证指标数**: {unvalidated_count}个
- **系统文件数**: {len(categories['system_files'])}个

### 📈 验证成功率
- **总体验证率**: {validation_rate:.1f}%
- **验证通过率**: {pass_rate:.1f}%

## ✅ 已验证通过指标 ({passed_count}个)

| 指标名称 | 验证状态 | 评分 | 验证时间 | 备注 |
|---------|---------|------|----------|------|
"""
    
    # 添加已验证通过的指标
    for indicator in sorted(categories['validated_passed']):
        if indicator in validated_indicators:
            info = validated_indicators[indicator]
            status_emoji = "🎉" if info['status'] == "PASSED_PRODUCTION_READY" else "✅"
            report += f"| **{indicator}** | {status_emoji} {info['status']} | {info['score']:.1f}/100 | {info['validation_time']} | 验证通过 |\n"
    
    report += f"""
## ❌ 已验证失败指标 ({failed_count}个)

| 指标名称 | 验证状态 | 评分 | 验证时间 | 失败原因 |
|---------|---------|------|----------|----------|
"""
    
    # 添加已验证失败的指标
    for indicator in sorted(categories['validated_failed']):
        if indicator in validated_indicators:
            info = validated_indicators[indicator]
            status_emoji = "⚠️" if info['status'] == "CONDITIONAL_PASS" else "❌"
            failure_reason = "架构不合规" if info['score'] < 50 else "部分功能缺失"
            report += f"| **{indicator}** | {status_emoji} {info['status']} | {info['score']:.1f}/100 | {info['validation_time']} | {failure_reason} |\n"
    
    report += f"""
## ⏸️ 待验证指标 ({unvalidated_count}个)

### 🎯 真正的BaseIndicator实现 ({len(categories['unvalidated_base_indicator'])}个)
这些指标继承了BaseIndicator，可以进行标准5阶段验证：

| 序号 | 指标名称 | 优先级 | 预期难度 | 建议验证顺序 |
|------|---------|--------|----------|-------------|
"""
    
    # 添加待验证的BaseIndicator实现
    priority_mapping = {
        'ADX': 'P1-高', 'MFI': 'P1-高', 'OBV': 'P1-高', 'ROC': 'P1-高',
        'KC': 'P2-中', 'VIX': 'P2-中', 'MTM': 'P2-中',
        'SYNERGY': 'P3-低', 'UNIFIED_MA': 'P3-低'
    }
    
    for i, indicator in enumerate(sorted(categories['unvalidated_base_indicator']), 1):
        priority = priority_mapping.get(indicator, 'P3-低')
        difficulty = "中等" if indicator in ['ADX', 'MFI', 'VIX'] else "简单"
        order = "建议优先" if priority.startswith('P1') else "次要"
        report += f"| {i:2d} | **{indicator}** | {priority} | {difficulty} | {order} |\n"
    
    report += f"""
### 🏭 工厂模式实现 ({len(categories['unvalidated_factory_pattern'])}个)
这些指标通过工厂模式创建，需要特殊验证方法：

| 序号 | 指标名称 | 实现方式 | 验证状态 | 备注 |
|------|---------|----------|----------|------|
"""
    
    # 添加工厂模式指标
    for i, indicator in enumerate(sorted(categories['unvalidated_factory_pattern']), 1):
        impl_type = "工厂模式" if indicator not in categories['system_files'] else "系统文件"
        status = "需要特殊验证" if impl_type == "工厂模式" else "无需验证"
        note = "通过RealIndicatorFactory创建" if impl_type == "工厂模式" else "系统支持文件"
        report += f"| {i:2d} | **{indicator}** | {impl_type} | {status} | {note} |\n"
    
    report += f"""
## 🔧 系统文件 ({len(categories['system_files'])}个)
这些是系统支持文件，无需验证：

"""
    
    # 添加系统文件列表
    for i, system_file in enumerate(sorted(categories['system_files']), 1):
        report += f"{i:2d}. **{system_file}**\n"
    
    report += f"""
## 🚀 下一步验证计划

### 📋 优先验证列表
基于重要性和实现复杂度，建议按以下顺序验证：

1. **P1级别指标** (高优先级):
   - ADX (平均趋向指数) - 重要趋势强度指标
   - MFI (资金流量指数) - 重要成交量指标
   - OBV (能量潮指标) - 经典成交量指标
   - ROC (变动率指标) - 重要动量指标

2. **P2级别指标** (中优先级):
   - KC (肯特纳通道) - 波动性指标
   - VIX (波动率指数) - 市场恐慌指标
   - MTM (动量指标) - 价格动量分析

3. **P3级别指标** (低优先级):
   - SYNERGY (协同指标) - 复合分析指标
   - UNIFIED_MA (统一移动平均) - 移动平均系统

### 🔧 工厂模式指标处理
对于工厂模式创建的指标，需要：
1. 开发专门的验证框架
2. 适配dict返回格式
3. 验证核心算法逻辑
4. 确保API兼容性

## 📈 验证质量标准

### ✅ 通过标准
- **算法真实性**: ≥99.0分 (绝对不可妥协)
- **基础功能**: ≥95.0分
- **形态识别**: ≥90.0分 (根据指标类型调整)
- **架构合规**: ≥95.0分
- **生产就绪**: ≥95.0分
- **总体平均**: ≥95.0分，最低≥90.0分

### 🎯 验证目标
- **短期目标**: 完成所有P1级别指标验证
- **中期目标**: 完成所有BaseIndicator实现验证
- **长期目标**: 建立工厂模式指标验证体系

---

**最后更新**: {current_time}
**验证工具**: 严格标准化5阶段验证系统
**质量保证**: 100%算法真实性 + 生产级架构合规性
"""
    
    return report


def main():
    """主函数"""
    logger.info("🔄 开始更新技术指标验证进度表...")
    
    try:
        # 生成更新后的进度表
        updated_table = generate_updated_progress_table()
        
        # 保存更新后的进度表
        progress_file = Path(root_dir) / "docs/finaltesting/技术指标验证进度表.md"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write(updated_table)
        
        logger.info(f"✅ 验证进度表已更新: {progress_file}")
        
        # 输出统计摘要
        categories, validated_indicators = categorize_indicators()
        total_indicators = sum(len(cat) for cat in categories.values())
        validated_count = len(categories['validated_passed']) + len(categories['validated_failed'])
        
        logger.info("📊 验证进度统计摘要:")
        logger.info(f"  总指标数: {total_indicators}")
        logger.info(f"  已验证: {validated_count} ({validated_count/total_indicators*100:.1f}%)")
        logger.info(f"  验证通过: {len(categories['validated_passed'])}")
        logger.info(f"  验证失败: {len(categories['validated_failed'])}")
        logger.info(f"  待验证BaseIndicator: {len(categories['unvalidated_base_indicator'])}")
        logger.info(f"  工厂模式指标: {len(categories['unvalidated_factory_pattern'])}")
        logger.info(f"  系统文件: {len(categories['system_files'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 更新验证进度表失败: {e}")
        return False


if __name__ == "__main__":
    main()
