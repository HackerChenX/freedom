#!/usr/bin/env python3
"""
技术标准验证和修正脚本
用于确保所有模块遵循统一的技术标准
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Tuple, Any
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TechnicalStandardsValidator:
    """技术标准验证器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.corrections = []
        
        # 标准形态名称
        self.standard_patterns = {
            "GOLDEN_CROSS", "DEATH_CROSS", "CROSS_UP", "CROSS_DOWN",
            "UPPER_BREAK", "LOWER_BREAK", "RESISTANCE_BREAK", "SUPPORT_BREAK",
            "OVERBOUGHT", "OVERSOLD", "NEUTRAL_ZONE", "OVERSOLD_RECOVERY", "OVERBOUGHT_CORRECTION",
            "BULLISH_DIVERGENCE", "BEARISH_DIVERGENCE", "HIDDEN_BULLISH_DIV", "HIDDEN_BEARISH_DIV",
            "DOJI", "HAMMER", "SHOOTING_STAR", "ENGULFING_BULLISH", "ENGULFING_BEARISH",
            "MORNING_STAR", "EVENING_STAR", "VOLUME_SURGE", "VOLUME_SHRINK",
            "VOLUME_BREAKTHROUGH", "PRICE_VOLUME_CONFIRM", "VOLATILITY_EXPANSION",
            "VOLATILITY_CONTRACTION", "TREND_ACCELERATION", "TREND_DECELERATION"
        }
        
        # 标准周期名称
        self.standard_periods = {
            "15min", "30min", "60min", "daily", "weekly", "monthly"
        }
        
        # 标准指标名称
        self.standard_indicators = {
            "MA", "EMA", "MACD", "RSI", "KDJ", "BOLL", "CCI", "WR", "BIAS", "PSY",
            "VR", "ARBR", "DMA", "MTM", "ROC", "OSC", "UOS", "CANDLESTICK", "VOL", "PRICE"
        }
        
        # 需要修正的形态名称映射
        self.pattern_corrections = {
            "RSI_OVERBOUGHT": "OVERBOUGHT",
            "RSI_OVERSOLD": "OVERSOLD", 
            "RSI_BULLISH_DIVERGENCE": "BULLISH_DIVERGENCE",
            "RSI_BEARISH_DIVERGENCE": "BEARISH_DIVERGENCE",
            "KDJ_GOLDEN_CROSS": "GOLDEN_CROSS",
            "KDJ_DEATH_CROSS": "DEATH_CROSS",
            "KDJ_OVERBOUGHT": "OVERBOUGHT",
            "KDJ_OVERSOLD": "OVERSOLD",
            "MACD_GOLDEN_CROSS": "GOLDEN_CROSS",
            "MACD_DEATH_CROSS": "DEATH_CROSS",
            "MACD_BULLISH_DIVERGENCE": "BULLISH_DIVERGENCE",
            "MACD_BEARISH_DIVERGENCE": "BEARISH_DIVERGENCE"
        }
        
        # 周期名称映射
        self.period_corrections = {
            "15分钟": "15min",
            "30分钟": "30min", 
            "60分钟": "60min",
            "日线": "daily",
            "周线": "weekly",
            "月线": "monthly",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "1d": "daily",
            "1w": "weekly",
            "1M": "monthly"
        }
    
    def validate_python_file(self, file_path: Path) -> bool:
        """验证Python文件的技术标准合规性"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_errors = []
            file_warnings = []
            
            # 检查形态名称
            self._check_pattern_names(content, file_path, file_errors, file_warnings)
            
            # 检查周期名称
            self._check_period_names(content, file_path, file_errors, file_warnings)
            
            # 检查指标名称
            self._check_indicator_names(content, file_path, file_errors, file_warnings)
            
            self.errors.extend(file_errors)
            self.warnings.extend(file_warnings)
            
            return len(file_errors) == 0
            
        except Exception as e:
            self.errors.append(f"文件读取错误 {file_path}: {e}")
            return False
    
    def _check_pattern_names(self, content: str, file_path: Path, errors: List, warnings: List):
        """检查形态名称"""
        # 查找可能的非标准形态名称
        pattern_regex = r'["\']([A-Z_]+_(?:GOLDEN_CROSS|DEATH_CROSS|OVERBOUGHT|OVERSOLD|DIVERGENCE))["\']'
        matches = re.findall(pattern_regex, content)
        
        for match in matches:
            if match in self.pattern_corrections:
                warnings.append(f"{file_path}: 建议修正形态名称 '{match}' → '{self.pattern_corrections[match]}'")
                self.corrections.append({
                    "file": str(file_path),
                    "type": "pattern_name",
                    "old": match,
                    "new": self.pattern_corrections[match]
                })
            elif match not in self.standard_patterns:
                errors.append(f"{file_path}: 非标准形态名称 '{match}'")
    
    def _check_period_names(self, content: str, file_path: Path, errors: List, warnings: List):
        """检查周期名称"""
        # 查找可能的周期名称
        period_regex = r'["\'](\d+分钟|日线|周线|月线|\d+[mhd]|1[wM])["\']'
        matches = re.findall(period_regex, content)
        
        for match in matches:
            if match in self.period_corrections:
                warnings.append(f"{file_path}: 建议标准化周期名称 '{match}' → '{self.period_corrections[match]}'")
                self.corrections.append({
                    "file": str(file_path),
                    "type": "period_name", 
                    "old": match,
                    "new": self.period_corrections[match]
                })
            elif match not in self.standard_periods:
                errors.append(f"{file_path}: 非标准周期名称 '{match}'")
    
    def _check_indicator_names(self, content: str, file_path: Path, errors: List, warnings: List):
        """检查指标名称"""
        # 查找类名中的指标名称
        class_regex = r'class\s+(\w+)(?:Indicator|指标)'
        matches = re.findall(class_regex, content)
        
        for match in matches:
            indicator_name = match.upper()
            if indicator_name not in self.standard_indicators:
                warnings.append(f"{file_path}: 可能的非标准指标名称 '{indicator_name}'")
    
    def validate_config_file(self, file_path: Path) -> bool:
        """验证配置文件的技术标准合规性"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    config = json.load(f)
                else:
                    # 简单的文本配置文件检查
                    content = f.read()
                    return self._check_config_content(content, file_path)
            
            return self._validate_config_structure(config, file_path)
            
        except Exception as e:
            self.errors.append(f"配置文件读取错误 {file_path}: {e}")
            return False
    
    def _validate_config_structure(self, config: Dict, file_path: Path) -> bool:
        """验证配置文件结构"""
        file_errors = []
        
        # 检查策略配置
        if 'conditions' in config:
            conditions = config['conditions']
            if 'pattern_conditions' in conditions:
                for condition in conditions['pattern_conditions']:
                    pattern = condition.get('pattern')
                    if pattern and pattern not in self.standard_patterns:
                        if pattern in self.pattern_corrections:
                            self.corrections.append({
                                "file": str(file_path),
                                "type": "config_pattern",
                                "old": pattern,
                                "new": self.pattern_corrections[pattern]
                            })
                        else:
                            file_errors.append(f"{file_path}: 配置中的非标准形态名称 '{pattern}'")
                    
                    indicator = condition.get('indicator')
                    if indicator and indicator not in self.standard_indicators:
                        file_errors.append(f"{file_path}: 配置中的非标准指标名称 '{indicator}'")
                    
                    period = condition.get('period')
                    if period and period not in self.standard_periods:
                        if period in self.period_corrections:
                            self.corrections.append({
                                "file": str(file_path),
                                "type": "config_period",
                                "old": period,
                                "new": self.period_corrections[period]
                            })
                        else:
                            file_errors.append(f"{file_path}: 配置中的非标准周期名称 '{period}'")
        
        self.errors.extend(file_errors)
        return len(file_errors) == 0
    
    def _check_config_content(self, content: str, file_path: Path) -> bool:
        """检查配置文件内容"""
        file_errors = []
        
        # 简单的文本匹配检查
        for old_pattern, new_pattern in self.pattern_corrections.items():
            if old_pattern in content:
                self.corrections.append({
                    "file": str(file_path),
                    "type": "config_text",
                    "old": old_pattern,
                    "new": new_pattern
                })
        
        return len(file_errors) == 0
    
    def scan_directory(self, directory: Path, file_patterns: List[str] = None) -> bool:
        """扫描目录中的文件"""
        if file_patterns is None:
            file_patterns = ['*.py', '*.json', '*.yaml', '*.yml']
        
        all_valid = True
        
        for pattern in file_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    if pattern == '*.py':
                        valid = self.validate_python_file(file_path)
                    else:
                        valid = self.validate_config_file(file_path)
                    
                    all_valid = all_valid and valid
        
        return all_valid
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("技术标准验证报告")
        report.append("=" * 60)
        
        if self.errors:
            report.append(f"\n❌ 发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                report.append(f"  • {error}")
        
        if self.warnings:
            report.append(f"\n⚠️  发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                report.append(f"  • {warning}")
        
        if self.corrections:
            report.append(f"\n🔧 建议 {len(self.corrections)} 个修正:")
            for correction in self.corrections:
                report.append(f"  • {correction['file']}: {correction['old']} → {correction['new']}")
        
        if not self.errors and not self.warnings:
            report.append("\n✅ 所有检查通过，符合技术标准！")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def apply_corrections(self, dry_run: bool = True) -> bool:
        """应用修正建议"""
        if not self.corrections:
            print("没有需要修正的内容")
            return True
        
        print(f"准备应用 {len(self.corrections)} 个修正...")
        
        if dry_run:
            print("(这是预览模式，不会实际修改文件)")
        
        file_corrections = {}
        for correction in self.corrections:
            file_path = correction['file']
            if file_path not in file_corrections:
                file_corrections[file_path] = []
            file_corrections[file_path].append(correction)
        
        for file_path, corrections in file_corrections.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                for correction in corrections:
                    content = content.replace(correction['old'], correction['new'])
                
                if not dry_run and content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✅ 已修正文件: {file_path}")
                elif dry_run:
                    print(f"🔍 预览修正: {file_path}")
                    for correction in corrections:
                        print(f"    {correction['old']} → {correction['new']}")
                
            except Exception as e:
                print(f"❌ 修正文件失败 {file_path}: {e}")
                return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description='技术标准验证和修正工具')
    parser.add_argument('--directory', '-d', type=str, default='.', 
                       help='要检查的目录路径 (默认: 当前目录)')
    parser.add_argument('--fix', action='store_true', 
                       help='自动应用修正建议')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='预览模式，不实际修改文件 (默认)')
    parser.add_argument('--strict', action='store_true',
                       help='严格模式，有错误时返回非零退出码')
    
    args = parser.parse_args()
    
    validator = TechnicalStandardsValidator()
    directory = Path(args.directory)
    
    print(f"开始扫描目录: {directory.absolute()}")
    
    # 扫描文件
    all_valid = validator.scan_directory(directory)
    
    # 生成报告
    report = validator.generate_report()
    print(report)
    
    # 应用修正
    if args.fix:
        validator.apply_corrections(dry_run=args.dry_run)
    
    # 退出码
    if args.strict and (validator.errors or validator.warnings):
        sys.exit(1)
    elif validator.errors:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
