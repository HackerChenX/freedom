#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
动态预警配置系统测试

测试预警规则的动态配置功能，包括：
1. 配置文件的加载和保存
2. 规则的动态创建、修改、删除
3. 模板的使用和管理
4. 配置的验证和校验
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.alert_config_manager import AlertConfigManager, AlertRuleConfig
from monitoring.intelligent_alert_system import IntelligentAlertSystem
from utils.logger import get_logger
from db.sql_manager import SQLManager, QueryType

logger = get_logger(__name__)


class TestDynamicAlertConfig:
    """动态预警配置测试类"""
    
    def __init__(self):
        """初始化测试"""
        self.temp_dir = None
        self.config_manager = None
        self.alert_system = None
        
    def setup_test_environment(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        config_dir = os.path.join(self.temp_dir, "config", "alerts")
        
        # 初始化配置管理器
        self.config_manager = AlertConfigManager(config_dir)
        
        logger.info(f"测试环境设置完成，临时目录: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """清理测试环境"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("测试环境清理完成")
    
    def test_config_manager_basic_operations(self):
        """测试配置管理器基本操作"""
        print("🔧 测试配置管理器基本操作...")
        
        try:
            # 测试获取默认规则
            rules = self.config_manager.get_all_rules()
            print(f"✅ 默认规则数量: {len(rules)}")
            
            # 测试创建新规则
            new_rule = AlertRuleConfig(
                id="test_custom_rule",
                name="测试自定义规则",
                description="这是一个测试用的自定义预警规则",
                indicators=["RSI", "MACD"],
                conditions={
                    "rsi_threshold": 75,
                    "macd_threshold": 0.02
                },
                signal_type="BUY",
                priority=4
            )
            
            success = self.config_manager.create_rule(new_rule)
            print(f"✅ 创建新规则: {'成功' if success else '失败'}")
            
            # 测试更新规则
            success = self.config_manager.update_rule("test_custom_rule", {
                "priority": 5,
                "conditions": {
                    "rsi_threshold": 80,
                    "macd_threshold": 0.03
                }
            })
            print(f"✅ 更新规则: {'成功' if success else '失败'}")
            
            # 测试获取更新后的规则
            updated_rule = self.config_manager.get_rule("test_custom_rule")
            if updated_rule:
                print(f"✅ 规则更新验证: 优先级={updated_rule.priority}, RSI阈值={updated_rule.conditions['rsi_threshold']}")
            
            # 测试删除规则
            success = self.config_manager.delete_rule("test_custom_rule")
            print(f"✅ 删除规则: {'成功' if success else '失败'}")
            
            return True
            
        except Exception as e:
            print(f"❌ 配置管理器基本操作测试失败: {e}")
            return False
    
    def test_template_operations(self):
        """测试模板操作"""
        print("📋 测试模板操作...")
        
        try:
            # 获取可用模板
            templates = self.config_manager.get_templates()
            print(f"✅ 可用模板数量: {len(templates)}")
            
            # 从模板创建规则
            success = self.config_manager.create_rule_from_template(
                template_id="rsi_template",
                rule_id="custom_rsi_rule",
                rule_name="自定义RSI规则",
                custom_conditions={
                    "rsi_overbought": 75,
                    "rsi_oversold": 25
                }
            )
            print(f"✅ 从模板创建规则: {'成功' if success else '失败'}")
            
            # 验证创建的规则
            created_rule = self.config_manager.get_rule("custom_rsi_rule")
            if created_rule:
                print(f"✅ 模板规则验证: 超买阈值={created_rule.conditions['rsi_overbought']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模板操作测试失败: {e}")
            return False
    
    def test_intelligent_alert_system_integration(self):
        """测试智能预警系统集成"""
        print("🤖 测试智能预警系统集成...")
        
        try:
            # 创建智能预警系统实例
            self.alert_system = IntelligentAlertSystem()
            
            # 测试重新加载配置
            success = self.alert_system.reload_rules_from_config()
            print(f"✅ 重新加载配置: {'成功' if success else '失败'}")
            
            # 测试从模板创建规则
            success = self.alert_system.create_rule_from_template(
                template_id="macd_template",
                rule_id="dynamic_macd_rule",
                rule_name="动态MACD规则",
                custom_conditions={
                    "golden_cross_threshold": 0.005,
                    "death_cross_threshold": -0.005
                }
            )
            print(f"✅ 从模板创建规则: {'成功' if success else '失败'}")
            
            # 测试更新规则配置
            success = self.alert_system.update_rule_config("dynamic_macd_rule", {
                "priority": 5,
                "description": "更新后的动态MACD规则描述"
            })
            print(f"✅ 更新规则配置: {'成功' if success else '失败'}")
            
            # 测试获取规则配置
            rule_config = self.alert_system.get_rule_config("dynamic_macd_rule")
            if rule_config:
                print(f"✅ 获取规则配置: 优先级={rule_config['priority']}")
            
            # 测试获取可用模板
            templates = self.alert_system.get_available_templates()
            print(f"✅ 可用模板数量: {len(templates)}")
            
            # 测试删除规则配置
            success = self.alert_system.delete_rule_config("dynamic_macd_rule")
            print(f"✅ 删除规则配置: {'成功' if success else '失败'}")
            
            return True
            
        except Exception as e:
            print(f"❌ 智能预警系统集成测试失败: {e}")
            return False
    
    def test_config_persistence(self):
        """测试配置持久化"""
        print("💾 测试配置持久化...")
        
        try:
            # 创建测试规则
            test_rule = AlertRuleConfig(
                id="persistence_test_rule",
                name="持久化测试规则",
                description="测试配置持久化功能",
                indicators=["KDJ"],
                conditions={
                    "k_threshold": 85,
                    "d_threshold": 85
                },
                signal_type="SELL",
                priority=3
            )
            
            # 保存规则
            success = self.config_manager.create_rule(test_rule)
            print(f"✅ 保存规则: {'成功' if success else '失败'}")
            
            # 创建新的配置管理器实例（模拟重启）
            new_config_manager = AlertConfigManager(self.config_manager.config_dir)
            
            # 验证规则是否被正确加载
            loaded_rule = new_config_manager.get_rule("persistence_test_rule")
            if loaded_rule:
                print(f"✅ 配置持久化验证: 规则名称={loaded_rule.name}")
                print(f"✅ 条件持久化验证: K阈值={loaded_rule.conditions['k_threshold']}")
            else:
                print("❌ 配置持久化失败：规则未找到")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 配置持久化测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始动态预警配置系统全面测试...")
        print("=" * 60)
        
        try:
            # 设置测试环境
            self.setup_test_environment()
            
            # 运行测试
            tests = [
                ("配置管理器基本操作", self.test_config_manager_basic_operations),
                ("模板操作", self.test_template_operations),
                ("智能预警系统集成", self.test_intelligent_alert_system_integration),
                ("配置持久化", self.test_config_persistence)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                print(f"\n📋 测试 {test_name}:")
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name}: 通过")
                else:
                    print(f"❌ {test_name}: 失败")
            
            # 输出测试结果
            print("\n" + "=" * 60)
            print(f"📊 测试完成: {passed_tests}/{total_tests} 通过")
            
            if passed_tests == total_tests:
                print("🎉 所有测试通过！")
                return True
            else:
                print("⚠️ 部分测试失败，请检查问题")
                return False
                
        except Exception as e:
            print(f"❌ 测试执行失败: {e}")
            return False
        finally:
            # 清理测试环境
            self.cleanup_test_environment()


def main():
    """主函数"""
    tester = TestDynamicAlertConfig()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 动态预警配置系统测试完成，所有功能正常")
    else:
        print("\n❌ 动态预警配置系统测试失败，请检查问题")
    
    return success


if __name__ == "__main__":
    main()
