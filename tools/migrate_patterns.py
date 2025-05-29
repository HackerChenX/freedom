#!/usr/bin/env python3
"""
形态识别迁移工具

将回测脚本中的形态识别逻辑迁移到各个指标类中
"""

import sys
import os
import re
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.logger import get_logger

logger = get_logger(__name__)


class PatternMigrator:
    """形态识别迁移器"""
    
    def __init__(self):
        """初始化迁移器"""
        self.pattern_mappings = {
            'MACD': [
                'MACD零轴上方金叉',
                'MACD零轴下方金叉', 
                'MACD零轴上方死叉',
                'MACD零轴下方死叉',
                'MACD柱状图由负转正',
                'MACD柱状图由正转负',
                'MACD背离'
            ],
            'KDJ': [
                'KDJ超卖区金叉',
                'KDJ超买区死叉',
                'KDJ低位钝化',
                'KDJ高位钝化',
                'KDJ背离'
            ],
            'RSI': [
                'RSI超卖反弹',
                'RSI超买回落',
                'RSI背离',
                'RSI中线突破'
            ],
            'BOLL': [
                'BOLL下轨支撑',
                'BOLL上轨阻力',
                'BOLL带宽收缩',
                'BOLL带宽扩张',
                'BOLL突破'
            ],
            'Volume': [
                '放量突破',
                '缩量整理',
                '量价背离',
                '天量天价',
                '地量地价'
            ]
        }
    
    def scan_backtest_patterns(self, backtest_file: str) -> Dict[str, List[str]]:
        """
        扫描回测文件中的形态识别代码
        
        Args:
            backtest_file: 回测文件路径
            
        Returns:
            Dict[str, List[str]]: 按指标分类的形态识别代码
        """
        logger.info(f"扫描回测文件: {backtest_file}")
        
        if not os.path.exists(backtest_file):
            logger.error(f"回测文件不存在: {backtest_file}")
            return {}
        
        with open(backtest_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        patterns_found = {}
        
        # 查找形态识别相关的代码段
        pattern_blocks = self._extract_pattern_blocks(content)
        
        for block in pattern_blocks:
            indicator_type = self._identify_indicator_type(block)
            if indicator_type:
                if indicator_type not in patterns_found:
                    patterns_found[indicator_type] = []
                patterns_found[indicator_type].append(block)
        
        logger.info(f"发现 {len(patterns_found)} 个指标的形态识别代码")
        return patterns_found
    
    def _extract_pattern_blocks(self, content: str) -> List[str]:
        """
        提取形态识别代码块
        
        Args:
            content: 文件内容
            
        Returns:
            List[str]: 代码块列表
        """
        blocks = []
        
        # 查找包含形态识别关键词的代码段
        pattern_keywords = [
            '金叉', '死叉', '突破', '支撑', '阻力', '背离',
            '超买', '超卖', '放量', '缩量', '天量', '地量'
        ]
        
        lines = content.split('\n')
        current_block = []
        in_pattern_block = False
        
        for i, line in enumerate(lines):
            # 检查是否包含形态识别关键词
            if any(keyword in line for keyword in pattern_keywords):
                if not in_pattern_block:
                    # 开始新的代码块，包含前面几行上下文
                    start_idx = max(0, i - 5)
                    current_block = lines[start_idx:i+1]
                    in_pattern_block = True
                else:
                    current_block.append(line)
            elif in_pattern_block:
                current_block.append(line)
                
                # 检查是否到达代码块结束
                if (line.strip() == '' or 
                    line.strip().startswith('except') or
                    line.strip().startswith('finally') or
                    i == len(lines) - 1):
                    
                    # 添加后面几行上下文
                    end_idx = min(len(lines), i + 3)
                    current_block.extend(lines[i+1:end_idx])
                    
                    blocks.append('\n'.join(current_block))
                    current_block = []
                    in_pattern_block = False
        
        return blocks
    
    def _identify_indicator_type(self, code_block: str) -> str:
        """
        识别代码块对应的指标类型
        
        Args:
            code_block: 代码块
            
        Returns:
            str: 指标类型
        """
        code_lower = code_block.lower()
        
        if 'macd' in code_lower:
            return 'MACD'
        elif 'kdj' in code_lower or ('k' in code_lower and 'd' in code_lower and 'j' in code_lower):
            return 'KDJ'
        elif 'rsi' in code_lower:
            return 'RSI'
        elif 'boll' in code_lower or 'bollinger' in code_lower:
            return 'BOLL'
        elif 'volume' in code_lower or 'obv' in code_lower or '成交量' in code_block:
            return 'Volume'
        elif 'ma' in code_lower and 'macd' not in code_lower:
            return 'MA'
        
        return None
    
    def generate_migration_report(self, patterns_found: Dict[str, List[str]]) -> str:
        """
        生成迁移报告
        
        Args:
            patterns_found: 发现的形态识别代码
            
        Returns:
            str: 迁移报告
        """
        report = []
        report.append("# 形态识别迁移报告")
        report.append("=" * 50)
        report.append("")
        
        for indicator_type, code_blocks in patterns_found.items():
            report.append(f"## {indicator_type} 指标")
            report.append(f"发现 {len(code_blocks)} 个形态识别代码块")
            report.append("")
            
            for i, block in enumerate(code_blocks, 1):
                report.append(f"### 代码块 {i}")
                report.append("```python")
                report.append(block)
                report.append("```")
                report.append("")
                
                # 分析代码块并提供迁移建议
                suggestions = self._analyze_code_block(block, indicator_type)
                if suggestions:
                    report.append("**迁移建议:**")
                    for suggestion in suggestions:
                        report.append(f"- {suggestion}")
                    report.append("")
        
        # 添加总结和下一步行动
        report.append("## 迁移总结")
        report.append("")
        report.append("### 需要迁移的指标:")
        for indicator_type in patterns_found.keys():
            report.append(f"- {indicator_type}")
        report.append("")
        
        report.append("### 建议的迁移步骤:")
        report.append("1. 为每个指标创建专门的形态识别方法")
        report.append("2. 将形态识别逻辑从回测脚本中移除")
        report.append("3. 在指标类中实现 `identify_patterns()` 方法")
        report.append("4. 更新回测脚本，调用指标的形态识别方法")
        report.append("5. 进行测试验证，确保迁移后功能正常")
        
        return '\n'.join(report)
    
    def _analyze_code_block(self, code_block: str, indicator_type: str) -> List[str]:
        """
        分析代码块并提供迁移建议
        
        Args:
            code_block: 代码块
            indicator_type: 指标类型
            
        Returns:
            List[str]: 迁移建议列表
        """
        suggestions = []
        
        # 检查是否包含硬编码的阈值
        if re.search(r'\d+\.?\d*', code_block):
            suggestions.append("代码中包含硬编码数值，建议将其作为参数配置")
        
        # 检查是否包含复杂的逻辑
        if 'if' in code_block and 'elif' in code_block:
            suggestions.append("包含复杂的条件逻辑，建议拆分为多个子方法")
        
        # 检查是否包含数据访问
        if 'data[' in code_block or '.iloc[' in code_block:
            suggestions.append("直接访问数据，需要确保在指标类中有相应的数据接口")
        
        # 根据指标类型提供特定建议
        if indicator_type == 'MACD':
            if '金叉' in code_block or '死叉' in code_block:
                suggestions.append("MACD金叉死叉逻辑可以封装为 `detect_crossover()` 方法")
            if '背离' in code_block:
                suggestions.append("背离检测逻辑可以使用基类的 `detect_divergence()` 方法")
        
        elif indicator_type == 'KDJ':
            if '超买' in code_block or '超卖' in code_block:
                suggestions.append("超买超卖判断可以作为KDJ类的基础方法")
        
        elif indicator_type == 'Volume':
            if '放量' in code_block or '缩量' in code_block:
                suggestions.append("成交量变化检测可以封装为独立的方法")
        
        return suggestions
    
    def create_migration_template(self, indicator_type: str, patterns: List[str]) -> str:
        """
        创建迁移模板代码
        
        Args:
            indicator_type: 指标类型
            patterns: 形态列表
            
        Returns:
            str: 模板代码
        """
        template = []
        template.append(f"# {indicator_type} 指标形态识别迁移模板")
        template.append("")
        template.append(f"def identify_patterns(self, data: pd.DataFrame, **kwargs) -> List[str]:")
        template.append(f'    """')
        template.append(f'    识别{indicator_type}技术形态')
        template.append(f'    """')
        template.append(f'    patterns = []')
        template.append("")
        
        for pattern in patterns:
            method_name = self._pattern_to_method_name(pattern)
            template.append(f'    # 检测{pattern}')
            template.append(f'    if self.{method_name}(data):')
            template.append(f'        patterns.append("{pattern}")')
            template.append("")
        
        template.append('    return patterns')
        template.append("")
        
        # 为每个形态生成方法模板
        for pattern in patterns:
            method_name = self._pattern_to_method_name(pattern)
            template.append(f"def {method_name}(self, data: pd.DataFrame) -> bool:")
            template.append(f'    """')
            template.append(f'    检测{pattern}形态')
            template.append(f'    """')
            template.append(f'    # TODO: 实现{pattern}检测逻辑')
            template.append(f'    return False')
            template.append("")
        
        return '\n'.join(template)
    
    def _pattern_to_method_name(self, pattern: str) -> str:
        """
        将形态名称转换为方法名
        
        Args:
            pattern: 形态名称
            
        Returns:
            str: 方法名
        """
        # 移除特殊字符，转换为下划线命名
        method_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', pattern)
        method_name = f"detect_{method_name.lower()}"
        return method_name


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='形态识别迁移工具')
    parser.add_argument('--backtest-file', 
                       default='scripts/backtest/unified_backtest.py',
                       help='回测文件路径')
    parser.add_argument('--output-dir', 
                       default='migration_output',
                       help='输出目录')
    parser.add_argument('--generate-templates', 
                       action='store_true',
                       help='生成迁移模板代码')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    migrator = PatternMigrator()
    
    # 扫描回测文件
    patterns_found = migrator.scan_backtest_patterns(args.backtest_file)
    
    # 生成迁移报告
    report = migrator.generate_migration_report(patterns_found)
    
    # 保存报告
    report_file = os.path.join(args.output_dir, 'migration_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"迁移报告已保存到: {report_file}")
    
    # 生成模板代码
    if args.generate_templates:
        for indicator_type, code_blocks in patterns_found.items():
            if indicator_type in migrator.pattern_mappings:
                patterns = migrator.pattern_mappings[indicator_type]
                template = migrator.create_migration_template(indicator_type, patterns)
                
                template_file = os.path.join(args.output_dir, f'{indicator_type.lower()}_migration_template.py')
                with open(template_file, 'w', encoding='utf-8') as f:
                    f.write(template)
                
                logger.info(f"{indicator_type} 迁移模板已保存到: {template_file}")
    
    print(f"迁移分析完成，结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 