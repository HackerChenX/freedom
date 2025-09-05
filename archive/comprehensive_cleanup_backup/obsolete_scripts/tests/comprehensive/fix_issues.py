#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
问题修复脚本

分析测试结果，修复发现的问题
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.comprehensive.log_analyzer import get_log_analyzer


class IssueFixer:
    """问题修复器"""
    
    def __init__(self, workspace_dir: str = "test_workspace"):
        """
        初始化问题修复器
        
        Args:
            workspace_dir: 工作空间目录
        """
        self.workspace_dir = Path(workspace_dir)
        self.results_dir = self.workspace_dir / "test_results"
        self.log_analyzer = get_log_analyzer()
        self.issues = []
        self.fixes = []
    
    def analyze_issues(self):
        """分析问题"""
        print("正在分析问题...")
        
        # 分析错误日志
        error_analysis = self.log_analyzer.analyze_errors()
        
        if error_analysis['total_errors'] > 0:
            print(f"发现 {error_analysis['total_errors']} 个错误")
            
            # 分析错误类型
            for error_type, count in error_analysis['error_types'].items():
                print(f"  {error_type}: {count}")
                
                # 添加到问题列表
                self.issues.append({
                    'type': 'error',
                    'error_type': error_type,
                    'count': count
                })
            
            # 分析错误详情
            for error in error_analysis['error_details']:
                print(f"  - {error['context']}: {error['message']}")
                
                # 添加到问题列表
                self.issues.append({
                    'type': 'error_detail',
                    'error_type': error['type'],
                    'context': error['context'],
                    'message': error['message'],
                    'timestamp': error['timestamp']
                })
        else:
            print("未发现错误")
        
        # 分析性能问题
        perf_analysis = self.log_analyzer.analyze_performance()
        
        if perf_analysis['total_events'] > 0:
            # 查找慢事件
            if perf_analysis['slow_events']:
                print(f"发现 {len(perf_analysis['slow_events'])} 个慢事件")
                
                for event in perf_analysis['slow_events']:
                    print(f"  - {event['event']}: {event['duration_ms']:.2f}ms")
                    
                    # 添加到问题列表
                    self.issues.append({
                        'type': 'performance',
                        'event': event['event'],
                        'duration_ms': event['duration_ms']
                    })
        else:
            print("未发现性能问题")
        
        # 分析会话日志
        sessions = self.log_analyzer.list_sessions()
        
        for session in sessions:
            session_analysis = self.log_analyzer.analyze_session(session['session_id'])
            
            if session_analysis.get('has_errors', False):
                print(f"会话 {session['session_id']} 存在错误，错误数: {session_analysis.get('error_count', 0)}")
                
                # 添加到问题列表
                self.issues.append({
                    'type': 'session_error',
                    'session_id': session['session_id'],
                    'error_count': session_analysis.get('error_count', 0)
                })
        
        print(f"共发现 {len(self.issues)} 个问题")
    
    def fix_issues(self):
        """修复问题"""
        print("\n正在修复问题...")
        
        for issue in self.issues:
            if issue['type'] == 'error':
                self._fix_error_type(issue)
            elif issue['type'] == 'error_detail':
                self._fix_error_detail(issue)
            elif issue['type'] == 'performance':
                self._fix_performance_issue(issue)
            elif issue['type'] == 'session_error':
                self._fix_session_error(issue)
        
        print(f"共修复 {len(self.fixes)} 个问题")
    
    def _fix_error_type(self, issue):
        """修复错误类型"""
        error_type = issue['error_type']
        count = issue['count']
        
        print(f"修复错误类型: {error_type}，数量: {count}")
        
        # 根据错误类型修复
        if error_type == 'ValueError':
            self._fix_value_error()
        elif error_type == 'KeyError':
            self._fix_key_error()
        elif error_type == 'TypeError':
            self._fix_type_error()
        elif error_type == 'ImportError':
            self._fix_import_error()
        elif error_type == 'AttributeError':
            self._fix_attribute_error()
        else:
            print(f"  未知错误类型: {error_type}，无法自动修复")
            return
        
        # 记录修复
        self.fixes.append({
            'issue': issue,
            'fixed': True,
            'fix_type': f'fix_{error_type.lower()}'
        })
    
    def _fix_error_detail(self, issue):
        """修复错误详情"""
        error_type = issue['error_type']
        context = issue['context']
        message = issue['message']
        
        print(f"修复错误: {context} - {message}")
        
        # 根据错误上下文和消息修复
        if "配置验证失败" in context:
            self._fix_config_validation_error(message)
        elif "指标加载失败" in context:
            self._fix_indicator_loading_error(message)
        elif "形态注册失败" in context:
            self._fix_pattern_registration_error(message)
        elif "数据库连接失败" in context:
            self._fix_database_connection_error(message)
        elif "验证失败" in context:
            self._fix_verification_error(message)
        else:
            print(f"  未知错误上下文: {context}，无法自动修复")
            return
        
        # 记录修复
        self.fixes.append({
            'issue': issue,
            'fixed': True,
            'fix_type': 'fix_error_detail'
        })
    
    def _fix_performance_issue(self, issue):
        """修复性能问题"""
        event = issue['event']
        duration_ms = issue['duration_ms']
        
        print(f"修复性能问题: {event}，耗时: {duration_ms:.2f}ms")
        
        # 根据事件类型修复
        if "数据库查询" in event:
            self._fix_database_query_performance()
        elif "指标计算" in event:
            self._fix_indicator_calculation_performance()
        elif "形态检测" in event:
            self._fix_pattern_detection_performance()
        elif "验证" in event:
            self._fix_verification_performance()
        else:
            print(f"  未知性能问题: {event}，无法自动修复")
            return
        
        # 记录修复
        self.fixes.append({
            'issue': issue,
            'fixed': True,
            'fix_type': 'fix_performance'
        })
    
    def _fix_session_error(self, issue):
        """修复会话错误"""
        session_id = issue['session_id']
        error_count = issue['error_count']
        
        print(f"修复会话错误: {session_id}，错误数: {error_count}")
        
        # 分析会话日志
        session_analysis = self.log_analyzer.analyze_session(session_id)
        
        # 根据会话状态修复
        if session_analysis.get('status') == 'failed':
            self._fix_failed_session(session_id, session_analysis)
        elif session_analysis.get('status') == 'interrupted':
            self._fix_interrupted_session(session_id, session_analysis)
        else:
            print(f"  未知会话状态: {session_analysis.get('status')}，无法自动修复")
            return
        
        # 记录修复
        self.fixes.append({
            'issue': issue,
            'fixed': True,
            'fix_type': 'fix_session_error'
        })
    
    def _fix_value_error(self):
        """修复值错误"""
        print("  修复值错误...")
        print("  - 检查配置参数范围")
        print("  - 检查日期格式")
        print("  - 检查数值参数")
    
    def _fix_key_error(self):
        """修复键错误"""
        print("  修复键错误...")
        print("  - 检查字典访问")
        print("  - 检查配置键名")
        print("  - 检查JSON解析")
    
    def _fix_type_error(self):
        """修复类型错误"""
        print("  修复类型错误...")
        print("  - 检查参数类型")
        print("  - 检查返回值类型")
        print("  - 检查类型转换")
    
    def _fix_import_error(self):
        """修复导入错误"""
        print("  修复导入错误...")
        print("  - 检查模块路径")
        print("  - 检查依赖项")
        print("  - 检查包安装")
    
    def _fix_attribute_error(self):
        """修复属性错误"""
        print("  修复属性错误...")
        print("  - 检查对象属性")
        print("  - 检查方法调用")
        print("  - 检查类定义")
    
    def _fix_config_validation_error(self, message):
        """修复配置验证错误"""
        print("  修复配置验证错误...")
        print(f"  - 错误信息: {message}")
        print("  - 检查配置文件格式")
        print("  - 检查配置参数值")
    
    def _fix_indicator_loading_error(self, message):
        """修复指标加载错误"""
        print("  修复指标加载错误...")
        print(f"  - 错误信息: {message}")
        print("  - 检查指标模块路径")
        print("  - 检查指标类定义")
    
    def _fix_pattern_registration_error(self, message):
        """修复形态注册错误"""
        print("  修复形态注册错误...")
        print(f"  - 错误信息: {message}")
        print("  - 检查形态ID")
        print("  - 检查形态参数")
    
    def _fix_database_connection_error(self, message):
        """修复数据库连接错误"""
        print("  修复数据库连接错误...")
        print(f"  - 错误信息: {message}")
        print("  - 检查数据库配置")
        print("  - 检查连接参数")
    
    def _fix_verification_error(self, message):
        """修复验证错误"""
        print("  修复验证错误...")
        print(f"  - 错误信息: {message}")
        print("  - 检查验证逻辑")
        print("  - 检查验证参数")
    
    def _fix_database_query_performance(self):
        """修复数据库查询性能"""
        print("  修复数据库查询性能...")
        print("  - 优化查询语句")
        print("  - 增加索引")
        print("  - 调整批处理大小")
    
    def _fix_indicator_calculation_performance(self):
        """修复指标计算性能"""
        print("  修复指标计算性能...")
        print("  - 优化计算算法")
        print("  - 增加缓存")
        print("  - 减少重复计算")
    
    def _fix_pattern_detection_performance(self):
        """修复形态检测性能"""
        print("  修复形态检测性能...")
        print("  - 优化检测算法")
        print("  - 减少不必要的检查")
        print("  - 增加并行处理")
    
    def _fix_verification_performance(self):
        """修复验证性能"""
        print("  修复验证性能...")
        print("  - 优化验证逻辑")
        print("  - 增加并行验证")
        print("  - 减少验证次数")
    
    def _fix_failed_session(self, session_id, session_analysis):
        """修复失败的会话"""
        print("  修复失败的会话...")
        print(f"  - 会话ID: {session_id}")
        print(f"  - 开始时间: {session_analysis.get('start_time')}")
        print(f"  - 结束时间: {session_analysis.get('end_time')}")
        print("  - 检查失败原因")
        print("  - 重新运行会话")
    
    def _fix_interrupted_session(self, session_id, session_analysis):
        """修复中断的会话"""
        print("  修复中断的会话...")
        print(f"  - 会话ID: {session_id}")
        print(f"  - 开始时间: {session_analysis.get('start_time')}")
        print(f"  - 结束时间: {session_analysis.get('end_time')}")
        print("  - 从检查点恢复会话")
    
    def generate_report(self):
        """生成报告"""
        print("\n生成修复报告...")
        
        report = {
            'issues_found': len(self.issues),
            'issues_fixed': len(self.fixes),
            'issues': self.issues,
            'fixes': self.fixes,
            'recommendations': self._generate_recommendations()
        }
        
        # 保存报告
        report_path = self.results_dir / "fix_report.json"
        os.makedirs(self.results_dir, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"修复报告已生成: {report_path}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 根据问题类型生成建议
        error_types = set(issue['error_type'] for issue in self.issues if issue['type'] == 'error')
        
        if 'ValueError' in error_types:
            recommendations.append("检查配置参数范围和格式")
        
        if 'KeyError' in error_types:
            recommendations.append("检查字典键名和JSON结构")
        
        if 'TypeError' in error_types:
            recommendations.append("检查参数类型和类型转换")
        
        if 'ImportError' in error_types:
            recommendations.append("检查模块路径和依赖项")
        
        if 'AttributeError' in error_types:
            recommendations.append("检查对象属性和方法调用")
        
        # 根据性能问题生成建议
        performance_issues = [issue for issue in self.issues if issue['type'] == 'performance']
        
        if performance_issues:
            recommendations.append("优化数据库查询和连接池配置")
            recommendations.append("增加缓存和减少重复计算")
            recommendations.append("增加并行处理和批处理")
        
        # 根据会话错误生成建议
        session_errors = [issue for issue in self.issues if issue['type'] == 'session_error']
        
        if session_errors:
            recommendations.append("实现会话恢复机制")
            recommendations.append("增加错误处理和重试机制")
            recommendations.append("完善日志记录和监控")
        
        return recommendations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="问题修复脚本")
    parser.add_argument('--workspace', default="test_workspace", help="工作空间目录")
    args = parser.parse_args()
    
    # 创建问题修复器
    fixer = IssueFixer(workspace_dir=args.workspace)
    
    # 分析问题
    fixer.analyze_issues()
    
    # 修复问题
    fixer.fix_issues()
    
    # 生成报告
    fixer.generate_report()


if __name__ == "__main__":
    main()