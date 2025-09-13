#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
预警配置命令行工具

提供命令行界面来管理预警规则配置，支持：
1. 查看、创建、修改、删除预警规则
2. 管理预警规则模板
3. 导入导出配置
4. 配置验证和测试
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.alert_config_manager import get_alert_config_manager, AlertRuleConfig
from monitoring.intelligent_alert_system import IntelligentAlertSystem
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertConfigCLI:
    """预警配置命令行工具"""
    
    def __init__(self):
        """初始化CLI工具"""
        self.config_manager = get_alert_config_manager()
        self.alert_system = None
    
    def list_rules(self, enabled_only: bool = False):
        """列出预警规则"""
        if enabled_only:
            rules = self.config_manager.get_enabled_rules()
            print("📋 启用的预警规则:")
        else:
            rules = self.config_manager.get_all_rules()
            print("📋 所有预警规则:")
        
        if not rules:
            print("  (无规则)")
            return
        
        print(f"{'ID':<25} {'名称':<20} {'类型':<15} {'优先级':<8} {'状态':<8}")
        print("-" * 85)
        
        for rule_id, rule_config in rules.items():
            status = "启用" if rule_config.enabled else "禁用"
            print(f"{rule_id:<25} {rule_config.name:<20} {rule_config.signal_type:<15} "
                  f"{rule_config.priority:<8} {status:<8}")
    
    def show_rule(self, rule_id: str):
        """显示规则详情"""
        rule_config = self.config_manager.get_rule(rule_id)
        if not rule_config:
            print(f"❌ 规则 {rule_id} 不存在")
            return
        
        print(f"📋 规则详情: {rule_id}")
        print("-" * 50)
        print(f"名称: {rule_config.name}")
        print(f"描述: {rule_config.description}")
        print(f"指标: {', '.join(rule_config.indicators)}")
        print(f"信号类型: {rule_config.signal_type}")
        print(f"优先级: {rule_config.priority}")
        print(f"状态: {'启用' if rule_config.enabled else '禁用'}")
        print(f"创建时间: {rule_config.created_at}")
        print(f"更新时间: {rule_config.updated_at}")
        print("\n条件:")
        for key, value in rule_config.conditions.items():
            print(f"  {key}: {value}")
    
    def create_rule(self, rule_data: Dict[str, Any]):
        """创建预警规则"""
        try:
            rule_config = AlertRuleConfig(**rule_data)
            success = self.config_manager.create_rule(rule_config)
            
            if success:
                print(f"✅ 成功创建规则: {rule_config.id}")
            else:
                print(f"❌ 创建规则失败: {rule_config.id}")
            
            return success
            
        except Exception as e:
            print(f"❌ 创建规则失败: {e}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]):
        """更新预警规则"""
        success = self.config_manager.update_rule(rule_id, updates)
        
        if success:
            print(f"✅ 成功更新规则: {rule_id}")
        else:
            print(f"❌ 更新规则失败: {rule_id}")
        
        return success
    
    def delete_rule(self, rule_id: str):
        """删除预警规则"""
        success = self.config_manager.delete_rule(rule_id)
        
        if success:
            print(f"✅ 成功删除规则: {rule_id}")
        else:
            print(f"❌ 删除规则失败: {rule_id}")
        
        return success
    
    def list_templates(self):
        """列出模板"""
        templates = self.config_manager.get_templates()
        
        print("📋 可用模板:")
        if not templates:
            print("  (无模板)")
            return
        
        for template_id, template_data in templates.items():
            print(f"\n🔧 {template_id}:")
            print(f"  名称: {template_data.get('name', 'N/A')}")
            print(f"  描述: {template_data.get('description', 'N/A')}")
            print(f"  指标: {', '.join(template_data.get('indicators', []))}")
            print(f"  信号类型: {template_data.get('signal_type', 'N/A')}")
    
    def create_from_template(self, template_id: str, rule_id: str, 
                           rule_name: str, custom_conditions: Dict[str, Any] = None):
        """从模板创建规则"""
        success = self.config_manager.create_rule_from_template(
            template_id, rule_id, rule_name, custom_conditions
        )
        
        if success:
            print(f"✅ 成功从模板 {template_id} 创建规则: {rule_id}")
        else:
            print(f"❌ 从模板创建规则失败: {template_id} -> {rule_id}")
        
        return success
    
    def test_alert_system(self):
        """测试预警系统"""
        print("🧪 测试预警系统...")
        
        try:
            self.alert_system = IntelligentAlertSystem()
            
            # 重新加载配置
            success = self.alert_system.reload_rules_from_config()
            if success:
                print("✅ 配置加载成功")
            else:
                print("❌ 配置加载失败")
                return False
            
            # 获取规则统计
            rules = self.alert_system.get_alert_rules()
            print(f"✅ 加载规则数量: {len(rules)}")
            
            # 显示规则摘要
            for rule in rules[:3]:  # 只显示前3个
                print(f"  - {rule['name']} ({rule['signal_type']})")
            
            if len(rules) > 3:
                print(f"  ... 还有 {len(rules) - 3} 个规则")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试预警系统失败: {e}")
            return False
    
    def export_config(self, output_file: str):
        """导出配置"""
        try:
            rules = self.config_manager.get_all_rules()
            templates = self.config_manager.get_templates()
            
            export_data = {
                "rules": {
                    rule_id: {
                        'id': rule_config.id,
                        'name': rule_config.name,
                        'description': rule_config.description,
                        'indicators': rule_config.indicators,
                        'conditions': rule_config.conditions,
                        'signal_type': rule_config.signal_type,
                        'priority': rule_config.priority,
                        'enabled': rule_config.enabled
                    }
                    for rule_id, rule_config in rules.items()
                },
                "templates": templates
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 配置已导出到: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ 导出配置失败: {e}")
            return False
    
    def import_config(self, input_file: str):
        """导入配置"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入规则
            rules_data = import_data.get('rules', {})
            imported_count = 0
            
            for rule_id, rule_data in rules_data.items():
                try:
                    rule_config = AlertRuleConfig(**rule_data)
                    if self.config_manager.create_rule(rule_config):
                        imported_count += 1
                except Exception as e:
                    print(f"⚠️ 导入规则 {rule_id} 失败: {e}")
            
            print(f"✅ 成功导入 {imported_count}/{len(rules_data)} 个规则")
            return True
            
        except Exception as e:
            print(f"❌ 导入配置失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预警配置管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出规则
    list_parser = subparsers.add_parser('list', help='列出预警规则')
    list_parser.add_argument('--enabled-only', action='store_true', help='只显示启用的规则')
    
    # 显示规则详情
    show_parser = subparsers.add_parser('show', help='显示规则详情')
    show_parser.add_argument('rule_id', help='规则ID')
    
    # 创建规则
    create_parser = subparsers.add_parser('create', help='创建预警规则')
    create_parser.add_argument('--config', required=True, help='规则配置JSON文件')
    
    # 更新规则
    update_parser = subparsers.add_parser('update', help='更新预警规则')
    update_parser.add_argument('rule_id', help='规则ID')
    update_parser.add_argument('--config', required=True, help='更新配置JSON文件')
    
    # 删除规则
    delete_parser = subparsers.add_parser('delete', help='删除预警规则')
    delete_parser.add_argument('rule_id', help='规则ID')
    
    # 列出模板
    subparsers.add_parser('templates', help='列出可用模板')
    
    # 从模板创建
    template_parser = subparsers.add_parser('create-from-template', help='从模板创建规则')
    template_parser.add_argument('template_id', help='模板ID')
    template_parser.add_argument('rule_id', help='新规则ID')
    template_parser.add_argument('rule_name', help='新规则名称')
    template_parser.add_argument('--conditions', help='自定义条件JSON文件')
    
    # 测试系统
    subparsers.add_parser('test', help='测试预警系统')
    
    # 导出配置
    export_parser = subparsers.add_parser('export', help='导出配置')
    export_parser.add_argument('output_file', help='输出文件路径')
    
    # 导入配置
    import_parser = subparsers.add_parser('import', help='导入配置')
    import_parser.add_argument('input_file', help='输入文件路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = AlertConfigCLI()
    
    try:
        if args.command == 'list':
            cli.list_rules(args.enabled_only)
        
        elif args.command == 'show':
            cli.show_rule(args.rule_id)
        
        elif args.command == 'create':
            with open(args.config, 'r', encoding='utf-8') as f:
                rule_data = json.load(f)
            cli.create_rule(rule_data)
        
        elif args.command == 'update':
            with open(args.config, 'r', encoding='utf-8') as f:
                updates = json.load(f)
            cli.update_rule(args.rule_id, updates)
        
        elif args.command == 'delete':
            cli.delete_rule(args.rule_id)
        
        elif args.command == 'templates':
            cli.list_templates()
        
        elif args.command == 'create-from-template':
            custom_conditions = None
            if args.conditions:
                with open(args.conditions, 'r', encoding='utf-8') as f:
                    custom_conditions = json.load(f)
            cli.create_from_template(args.template_id, args.rule_id, args.rule_name, custom_conditions)
        
        elif args.command == 'test':
            cli.test_alert_system()
        
        elif args.command == 'export':
            cli.export_config(args.output_file)
        
        elif args.command == 'import':
            cli.import_config(args.input_file)
        
    except Exception as e:
        print(f"❌ 执行命令失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
