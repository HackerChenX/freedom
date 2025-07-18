#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志分析工具

提供日志分析、统计和可视化功能，支持错误分析、性能分析和测试结果分析。
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from .logging_config import get_test_log_manager


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        初始化日志分析器
        
        Args:
            log_dir: 日志目录，如果为None则使用默认日志目录
        """
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            log_manager = get_test_log_manager()
            self.log_dir = log_manager.log_dir
        
        self.error_log_dir = self.log_dir / "errors"
        self.performance_log_dir = self.log_dir / "performance"
        self.audit_log_dir = self.log_dir / "audit"
        self.session_log_dir = self.log_dir / "sessions"
    
    def analyze_errors(self, days: int = 7) -> Dict[str, Any]:
        """
        分析错误日志
        
        Args:
            days: 分析的天数
            
        Returns:
            Dict[str, Any]: 错误分析结果
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        error_files = []
        
        # 收集错误日志文件
        for file in self.error_log_dir.glob("error_*.log"):
            try:
                # 从文件名中提取时间
                timestamp_str = file.name.replace("error_", "").replace(".log", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if timestamp >= cutoff_time:
                    error_files.append(file)
            except Exception:
                # 如果无法解析时间戳，使用文件修改时间
                if file.stat().st_mtime >= cutoff_time.timestamp():
                    error_files.append(file)
        
        # 分析错误
        error_types = Counter()
        error_contexts = Counter()
        error_details = []
        
        for file in error_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 提取错误类型
                    error_type_match = re.search(r"错误类型: (.+)", content)
                    error_type = error_type_match.group(1) if error_type_match else "Unknown"
                    
                    # 提取上下文
                    context_match = re.search(r"上下文: (.+)", content)
                    context = context_match.group(1) if context_match else "Unknown"
                    
                    # 提取时间
                    time_match = re.search(r"时间: (.+)", content)
                    timestamp = time_match.group(1) if time_match else file.name
                    
                    # 提取错误信息
                    message_match = re.search(r"错误信息: (.+)", content)
                    message = message_match.group(1) if message_match else "Unknown"
                    
                    error_types[error_type] += 1
                    error_contexts[context] += 1
                    
                    error_details.append({
                        'timestamp': timestamp,
                        'type': error_type,
                        'context': context,
                        'message': message,
                        'file': str(file)
                    })
            except Exception as e:
                print(f"分析错误日志文件失败 {file}: {e}")
        
        return {
            'total_errors': len(error_files),
            'error_types': dict(error_types.most_common()),
            'error_contexts': dict(error_contexts.most_common()),
            'error_details': sorted(error_details, key=lambda x: x['timestamp'], reverse=True)
        }
    
    def analyze_performance(self, days: int = 1) -> Dict[str, Any]:
        """
        分析性能日志
        
        Args:
            days: 分析的天数
            
        Returns:
            Dict[str, Any]: 性能分析结果
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        perf_data = []
        
        # 收集性能日志
        for file in self.performance_log_dir.glob("performance_*.jsonl"):
            try:
                # 从文件名中提取日期
                date_str = file.name.replace("performance_", "").replace(".jsonl", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date >= cutoff_time:
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                entry_time = datetime.fromisoformat(entry['timestamp'])
                                
                                if entry_time >= cutoff_time:
                                    perf_data.append(entry)
                            except Exception:
                                continue
            except Exception as e:
                print(f"分析性能日志文件失败 {file}: {e}")
        
        if not perf_data:
            return {
                'total_events': 0,
                'events_by_type': {},
                'avg_duration': 0,
                'max_duration': 0,
                'slow_events': []
            }
        
        # 转换为DataFrame进行分析
        df = pd.DataFrame(perf_data)
        
        # 按事件类型分组
        events_by_type = df.groupby('event').agg({
            'duration_ms': ['count', 'mean', 'min', 'max']
        }).reset_index()
        
        events_by_type.columns = ['event', 'count', 'avg_duration', 'min_duration', 'max_duration']
        
        # 找出慢事件
        slow_threshold = df['duration_ms'].mean() * 2  # 平均时间的2倍
        slow_events = df[df['duration_ms'] > slow_threshold].sort_values('duration_ms', ascending=False)
        
        return {
            'total_events': len(df),
            'events_by_type': events_by_type.to_dict('records'),
            'avg_duration': df['duration_ms'].mean(),
            'max_duration': df['duration_ms'].max(),
            'slow_events': slow_events.to_dict('records')[:10]  # 最慢的10个事件
        }
    
    def analyze_audit_logs(self, days: int = 7) -> Dict[str, Any]:
        """
        分析审计日志
        
        Args:
            days: 分析的天数
            
        Returns:
            Dict[str, Any]: 审计日志分析结果
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        audit_entries = []
        
        # 收集审计日志
        for file in self.audit_log_dir.glob("audit_*.log"):
            try:
                # 从文件名中提取日期
                date_str = file.name.replace("audit_", "").replace(".log", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date >= cutoff_time:
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                entry_time = datetime.fromisoformat(entry['timestamp'])
                                
                                if entry_time >= cutoff_time:
                                    audit_entries.append(entry)
                            except Exception:
                                continue
            except Exception as e:
                print(f"分析审计日志文件失败 {file}: {e}")
        
        if not audit_entries:
            return {
                'total_entries': 0,
                'actions': {},
                'resources': {},
                'results': {},
                'users': {}
            }
        
        # 分析审计日志
        actions = Counter()
        resources = Counter()
        results = Counter()
        users = Counter()
        
        for entry in audit_entries:
            actions[entry.get('action', 'unknown')] += 1
            resources[entry.get('resource', 'unknown')] += 1
            results[entry.get('result', 'unknown')] += 1
            users[entry.get('user', 'system')] += 1
        
        return {
            'total_entries': len(audit_entries),
            'actions': dict(actions.most_common()),
            'resources': dict(resources.most_common()),
            'results': dict(results.most_common()),
            'users': dict(users.most_common())
        }
    
    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """
        分析特定会话的日志
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict[str, Any]: 会话分析结果
        """
        session_dir = self.session_log_dir / session_id
        if not session_dir.exists():
            return {'error': f"会话 {session_id} 不存在"}
        
        session_logs = list(session_dir.glob("session_*.log"))
        if not session_logs:
            return {'error': f"会话 {session_id} 没有日志文件"}
        
        # 按修改时间排序，获取最新的日志
        latest_log = max(session_logs, key=lambda f: f.stat().st_mtime)
        
        # 分析日志
        log_entries = []
        phases = []
        current_phase = None
        start_time = None
        end_time = None
        status = "unknown"
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取时间戳和消息
                match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)', line)
                if match:
                    timestamp_str, level, message = match.groups()
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    
                    log_entries.append({
                        'timestamp': timestamp,
                        'level': level,
                        'message': message
                    })
                    
                    # 提取会话开始时间
                    if "会话开始" in message:
                        start_time = timestamp
                    
                    # 提取会话结束时间和状态
                    if "会话结束" in message:
                        end_time = timestamp
                    
                    if "执行状态" in message:
                        status_match = re.search(r"执行状态: (.+)", message)
                        if status_match:
                            status = status_match.group(1)
                    
                    # 提取阶段信息
                    if "开始执行阶段" in message:
                        phase_match = re.search(r"开始执行阶段: (\w+) - (.+)", message)
                        if phase_match:
                            phase_name, phase_desc = phase_match.groups()
                            current_phase = {
                                'name': phase_name,
                                'description': phase_desc,
                                'start_time': timestamp,
                                'end_time': None,
                                'status': 'running',
                                'duration': None
                            }
                    
                    # 提取阶段完成信息
                    if current_phase and "阶段" in message and "完成" in message:
                        phase_end_match = re.search(r"阶段 (\w+) 完成，耗时: ([\d\.]+)秒", message)
                        if phase_end_match and phase_end_match.group(1) == current_phase['name']:
                            current_phase['end_time'] = timestamp
                            current_phase['duration'] = float(phase_end_match.group(2))
                            
                            # 提取状态
                            status_match = re.search(r"状态: (\w+)", message)
                            if status_match:
                                current_phase['status'] = status_match.group(1)
                            
                            phases.append(current_phase)
                            current_phase = None
        
        # 计算总执行时间
        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        
        # 分析日志级别分布
        level_counts = Counter(entry['level'] for entry in log_entries)
        
        return {
            'session_id': session_id,
            'log_file': str(latest_log),
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'duration': duration,
            'status': status,
            'total_log_entries': len(log_entries),
            'level_distribution': dict(level_counts),
            'phases': phases,
            'has_errors': any(entry['level'] == 'ERROR' for entry in log_entries),
            'error_count': sum(1 for entry in log_entries if entry['level'] == 'ERROR')
        }
    
    def generate_performance_chart(self, output_file: str, days: int = 1) -> str:
        """
        生成性能图表
        
        Args:
            output_file: 输出文件路径
            days: 分析的天数
            
        Returns:
            str: 图表文件路径
        """
        perf_analysis = self.analyze_performance(days)
        
        if perf_analysis['total_events'] == 0:
            return None
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 事件类型分布
        events_df = pd.DataFrame(perf_analysis['events_by_type'])
        if not events_df.empty:
            plt.subplot(2, 2, 1)
            events_df.sort_values('count', ascending=False).head(10).plot(
                kind='bar', x='event', y='count', ax=plt.gca()
            )
            plt.title('Top 10 Event Types')
            plt.xlabel('Event Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            events_df.sort_values('avg_duration', ascending=False).head(10).plot(
                kind='bar', x='event', y='avg_duration', ax=plt.gca()
            )
            plt.title('Top 10 Slowest Events (Avg Duration)')
            plt.xlabel('Event Type')
            plt.ylabel('Avg Duration (ms)')
            plt.xticks(rotation=45)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def generate_error_report(self, output_file: str, days: int = 7) -> str:
        """
        生成错误报告
        
        Args:
            output_file: 输出文件路径
            days: 分析的天数
            
        Returns:
            str: 报告文件路径
        """
        error_analysis = self.analyze_errors(days)
        
        # 创建HTML报告
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>错误分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .error-message {{ color: #d9534f; }}
            </style>
        </head>
        <body>
            <h1>错误分析报告</h1>
            <p>分析时间范围: 过去{days}天</p>
            <p>总错误数: {error_analysis['total_errors']}</p>
            
            <h2>错误类型分布</h2>
            <table>
                <tr>
                    <th>错误类型</th>
                    <th>数量</th>
                </tr>
        """
        
        for error_type, count in error_analysis['error_types'].items():
            html += f"""
                <tr>
                    <td>{error_type}</td>
                    <td>{count}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>错误上下文分布</h2>
            <table>
                <tr>
                    <th>错误上下文</th>
                    <th>数量</th>
                </tr>
        """
        
        for context, count in error_analysis['error_contexts'].items():
            html += f"""
                <tr>
                    <td>{context}</td>
                    <td>{count}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>最近错误详情</h2>
            <table>
                <tr>
                    <th>时间</th>
                    <th>类型</th>
                    <th>上下文</th>
                    <th>错误信息</th>
                </tr>
        """
        
        for error in error_analysis['error_details'][:20]:  # 最近20个错误
            html += f"""
                <tr>
                    <td>{error['timestamp']}</td>
                    <td>{error['type']}</td>
                    <td>{error['context']}</td>
                    <td class="error-message">{error['message']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_file
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话
        
        Returns:
            List[Dict[str, Any]]: 会话列表
        """
        sessions = []
        
        for session_dir in self.session_log_dir.glob("*"):
            if session_dir.is_dir():
                session_id = session_dir.name
                
                # 获取会话日志
                log_files = list(session_dir.glob("session_*.log"))
                if not log_files:
                    continue
                
                # 按修改时间排序，获取最新的日志
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                
                # 提取基本信息
                start_time = None
                end_time = None
                status = "unknown"
                
                with open(latest_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "会话开始" in line:
                            time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if time_match:
                                start_time = time_match.group(1)
                        
                        if "会话结束" in line:
                            time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if time_match:
                                end_time = time_match.group(1)
                        
                        if "执行状态" in line:
                            status_match = re.search(r"执行状态: (.+)", line)
                            if status_match:
                                status = status_match.group(1)
                
                sessions.append({
                    'session_id': session_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'status': status,
                    'log_file': str(latest_log)
                })
        
        # 按开始时间排序
        return sorted(sessions, key=lambda s: s['start_time'] if s['start_time'] else "", reverse=True)


def get_log_analyzer() -> LogAnalyzer:
    """
    获取日志分析器实例
    
    Returns:
        LogAnalyzer: 日志分析器实例
    """
    return LogAnalyzer()


if __name__ == "__main__":
    analyzer = get_log_analyzer()
    
    # 测试错误分析
    error_analysis = analyzer.analyze_errors()
    print(f"总错误数: {error_analysis['total_errors']}")
    
    # 测试性能分析
    perf_analysis = analyzer.analyze_performance()
    print(f"总性能事件: {perf_analysis['total_events']}")
    
    # 测试会话列表
    sessions = analyzer.list_sessions()
    print(f"总会话数: {len(sessions)}")