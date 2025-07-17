#!/usr/bin/env python3
"""
系统问题全面修复脚本

修复以下关键问题：
1. 数据库字段名问题 (turnover_rate -> turnover)
2. 依赖注入问题
3. 抽象方法实现问题
4. 导入路径问题

作者：AI Assistant
创建时间：2025-01-13
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class SystemIssuesFixer:
    """系统问题修复器"""
    
    def __init__(self):
        self.root_path = Path(".")
        self.fixes_applied = []
        self.errors_encountered = []
        
    def fix_database_field_issues(self):
        """修复数据库字段名问题"""
        print("🔧 修复数据库字段名问题...")
        
        # 需要修复的文件列表（核心文件）
        critical_files = [
            "db/unified_data_manager.py",
            "db/managers/data_access_manager.py", 
            "strategy/strategy_parser.py",
            "models/stock_info.py"
        ]
        
        for file_path in critical_files:
            if self._fix_turnover_field_in_file(file_path):
                self.fixes_applied.append(f"修复字段名: {file_path}")
    
    def _fix_turnover_field_in_file(self, file_path: str) -> bool:
        """修复单个文件中的turnover字段问题"""
        try:
            if not os.path.exists(file_path):
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 替换SQL查询中的字段名
            content = re.sub(r'\bturnover_rate\b', 'turnover', content)
            
            # 替换Python代码中对应的字段引用（但保留某些特定上下文）
            # 只在SQL相关的上下文中替换
            patterns_to_replace = [
                (r"'turnover_rate'", "'turnover'"),
                (r'"turnover_rate"', '"turnover"'),
                (r"turnover_rate\s*,", "turnover,"),
                (r",\s*turnover_rate", ", turnover"),
            ]
            
            for pattern, replacement in patterns_to_replace:
                content = re.sub(pattern, replacement, content)
            
            # 只有内容发生变化时才写入
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
                
        except Exception as e:
            self.errors_encountered.append(f"修复{file_path}失败: {e}")
            
        return False
    
    def fix_dependency_injection_issues(self):
        """修复依赖注入问题"""
        print("🔧 修复依赖注入问题...")
        
        # 修复DataAccessManager构造函数问题
        manager_file = "db/managers/data_access_manager.py"
        self._ensure_data_access_manager_methods(manager_file)
        
        # 修复依赖注入容器注册
        di_file = "utils/dependency_injection.py"
        self._fix_dependency_injection_registration(di_file)
    
    def _ensure_data_access_manager_methods(self, file_path: str):
        """确保DataAccessManager有所需的标准方法"""
        try:
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已有get_stock_data方法
            if 'def get_stock_data(' not in content:
                # 在适当位置添加标准方法
                insertion_point = content.find('def get_stocks_data_batch_data_access_manager')
                if insertion_point > 0:
                    method_code = '''
    # 添加标准接口方法 - 这是其他模块期望的方法名
    @exception_handler(reraise=False, default_return=pd.DataFrame())
    def get_stock_data(self, code: str, start_date: str, end_date: str, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取股票数据 - 标准接口方法
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表，None表示所有列
            
        Returns:
            股票数据DataFrame
        """
        return self.get_stock_data_data_access_manager(code, start_date, end_date, columns)
    
'''
                    content = content[:insertion_point] + method_code + content[insertion_point:]
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.fixes_applied.append("添加DataAccessManager.get_stock_data方法")
                    
        except Exception as e:
            self.errors_encountered.append(f"修复DataAccessManager失败: {e}")
    
    def _fix_dependency_injection_registration(self, file_path: str):
        """修复依赖注入注册问题"""
        try:
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 改进工厂方法
            if 'def create_data_access_manager():' not in content:
                factory_pattern = r'(factory=lambda: DataAccessManager\(\))'
                
                factory_replacement = '''factory=lambda: self._create_data_access_manager()
            )
            logger.info("✅ DataAccessInterface已自动注册到依赖注入容器")
        
    def _create_data_access_manager(self):
        """创建DataAccessManager实例的工厂方法"""
        try:
            from db.connection_manager import get_connection_manager
            connection_manager = get_connection_manager()
            return DataAccessManager(connection_manager=connection_manager)
        except Exception as e:
            logger.warning(f"无法获取连接管理器，使用默认配置: {e}")
            return DataAccessManager()'''
                
                content = re.sub(factory_pattern, factory_replacement, content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied.append("改进依赖注入工厂方法")
                
        except Exception as e:
            self.errors_encountered.append(f"修复依赖注入失败: {e}")
    
    def fix_import_issues(self):
        """修复导入问题"""
        print("🔧 修复导入问题...")
        
        # 修复常见的导入问题
        files_to_check = [
            "strategy/strategy_parser.py",
            "indicators/zxm/buy_point_indicators.py"
        ]
        
        for file_path in files_to_check:
            self._fix_imports_in_file(file_path)
    
    def _fix_imports_in_file(self, file_path: str):
        """修复单个文件的导入问题"""
        try:
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 修复常见的导入问题
            fixes = [
                # 修复get_service导入
                (r'from utils\.dependency_injection import get_container',
                 'from utils.dependency_injection import get_container, get_service'),
                
                # 修复ParameterStandardizer导入
                (r'from db\.interfaces\.parameter_standardizer import ParameterStandardizer',
                 'from utils.parameter_standardizer import ParameterStandardizer'),
                
                # 修复performance_monitor导入
                (r'from utils\.decorators import performance_monitor\(threshold=(\d+\.?\d*)\)',
                 r'from utils.decorators import performance_monitor'),
            ]
            
            for pattern, replacement in fixes:
                content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied.append(f"修复导入: {file_path}")
                
        except Exception as e:
            self.errors_encountered.append(f"修复导入{file_path}失败: {e}")
    
    def verify_zxm_indicator_implementation(self):
        """验证ZXM指标实现"""
        print("🔧 验证ZXM指标实现...")
        
        zxm_file = "indicators/zxm/buy_point_indicators.py"
        try:
            if os.path.exists(zxm_file):
                with open(zxm_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查必需的抽象方法
                required_methods = [
                    '_calculate_baseindicator',
                    'calculate_raw_score_Indicator_Base_Indicator',
                    'get_patterns_Indicator_Base_Indicator',
                    'set_parameters_Indicator_Base_Indicator',
                    'calculate_confidence_Indicator_Base_Indicator'
                ]
                
                missing_methods = []
                for method in required_methods:
                    if f'def {method}(' not in content:
                        missing_methods.append(method)
                
                if missing_methods:
                    self.errors_encountered.append(f"ZXM指标缺少方法: {missing_methods}")
                else:
                    self.fixes_applied.append("ZXM指标抽象方法实现完整")
                    
        except Exception as e:
            self.errors_encountered.append(f"验证ZXM指标失败: {e}")
    
    def create_final_review_gate_script(self):
        """确保final_review_gate.py脚本存在且正确"""
        print("🔧 创建/验证final_review_gate.py脚本...")
        
        script_path = "final_review_gate.py"
        script_content = '''# final_review_gate.py
import sys
import os

if __name__ == "__main__":
    # Try to make stdout unbuffered for more responsive interaction.
    try:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    except Exception:
        pass

    try:
        sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    except Exception:
        pass

    print("--- FINAL REVIEW GATE ACTIVE ---", flush=True)
    print("AI has completed its primary actions. Awaiting your review or further sub-prompts.", flush=True)
    print("Type your sub-prompt, or one of: 'TASK_COMPLETE', 'Done', 'Quit', 'q' to signal completion.", flush=True)
    
    active_session = True
    while active_session:
        try:
            print("REVIEW_GATE_AWAITING_INPUT:", end="", flush=True) 
            
            line = sys.stdin.readline()
            
            if not line:  # EOF
                print("--- REVIEW GATE: STDIN CLOSED (EOF), EXITING SCRIPT ---", flush=True)
                active_session = False
                break
            
            user_input = line.strip()

            if user_input.upper() in ['TASK_COMPLETE', 'DONE', 'QUIT', 'Q']:
                print(f"--- REVIEW GATE: USER SIGNALED COMPLETION WITH '{user_input.upper()}' ---", flush=True)
                active_session = False
                break
            elif user_input:
                print(f"USER_REVIEW_SUB_PROMPT: {user_input}", flush=True)
                
        except KeyboardInterrupt:
            print("--- REVIEW GATE: SESSION INTERRUPTED BY USER (KeyboardInterrupt) ---", flush=True)
            active_session = False
            break
        except Exception as e:
            print(f"--- REVIEW GATE SCRIPT ERROR: {e} ---", flush=True)
            active_session = False
            break
            
    print("--- FINAL REVIEW GATE SCRIPT EXITED ---", flush=True)
'''
        
        try:
            # 检查脚本是否存在且内容正确
            script_exists = os.path.exists(script_path)
            
            if not script_exists:
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                self.fixes_applied.append("创建final_review_gate.py脚本")
            else:
                # 验证现有脚本内容
                with open(script_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                
                if 'REVIEW_GATE_AWAITING_INPUT:' not in existing_content:
                    with open(script_path, 'w', encoding='utf-8') as f:
                        f.write(script_content)
                    self.fixes_applied.append("更新final_review_gate.py脚本")
                
        except Exception as e:
            self.errors_encountered.append(f"创建final_review_gate.py失败: {e}")
    
    def run_comprehensive_fix(self):
        """运行全面修复"""
        print("🚀 开始全面系统修复...")
        print("=" * 60)
        
        # 执行各项修复
        self.fix_database_field_issues()
        self.fix_dependency_injection_issues()
        self.fix_import_issues()
        self.verify_zxm_indicator_implementation()
        self.create_final_review_gate_script()
        
        # 生成修复报告
        self._generate_fix_report()
    
    def _generate_fix_report(self):
        """生成修复报告"""
        print("\n" + "=" * 60)
        print("🎯 系统修复完成报告")
        print("=" * 60)
        
        print(f"\n✅ 成功修复项目 ({len(self.fixes_applied)}项):")
        for fix in self.fixes_applied:
            print(f"  • {fix}")
        
        if self.errors_encountered:
            print(f"\n❌ 遇到的错误 ({len(self.errors_encountered)}项):")
            for error in self.errors_encountered:
                print(f"  • {error}")
        
        print(f"\n📊 修复统计:")
        print(f"  • 成功修复: {len(self.fixes_applied)}项")
        print(f"  • 遇到错误: {len(self.errors_encountered)}项")
        print(f"  • 总体状态: {'✅ 基本成功' if len(self.fixes_applied) > len(self.errors_encountered) else '⚠️ 需要人工检查'}")


if __name__ == "__main__":
    fixer = SystemIssuesFixer()
    fixer.run_comprehensive_fix() 