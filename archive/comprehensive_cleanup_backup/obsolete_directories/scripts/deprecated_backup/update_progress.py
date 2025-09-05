#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标验证进度表更新脚本

用于在完成指标验证后自动更新进度表
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ProgressUpdater:
    """进度表更新器"""
    
    def __init__(self):
        """初始化更新器"""
        self.progress_file = Path("docs/技术指标验证进度表.md")
        self.backup_dir = Path("docs/backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # 状态映射
        self.status_map = {
            'PASSED': '✅ PASSED',
            'CONDITIONAL_PASS': '⚠️ CONDITIONAL_PASS', 
            'FAILED': '❌ FAILED',
            'IN_PROGRESS': '🔄 IN_PROGRESS',
            'PENDING': '⏸️ PENDING'
        }
    
    def backup_progress_file(self):
        """备份当前进度表"""
        if self.progress_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"进度表备份_{timestamp}.md"
            
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 进度表已备份: {backup_file}")
    
    def update_indicator_status(self, indicator_name: str, status: str, score: float, 
                              completion_date: str, issues: str = "无", notes: str = ""):
        """更新指标状态"""
        
        if not self.progress_file.exists():
            print(f"❌ 进度表文件不存在: {self.progress_file}")
            return False
        
        # 备份文件
        self.backup_progress_file()
        
        # 读取当前内容
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并更新指标行
        status_display = self.status_map.get(status, status)
        score_display = f"{score:.1f}/100" if score > 0 else "-"
        
        # 构建新的行内容
        new_row = f"| **{indicator_name}** | {status_display} | {score_display} | {completion_date} | {issues} | {notes} |"
        
        # 查找指标行的模式
        pattern = rf"\| \*\*{re.escape(indicator_name)}\*\* \|[^\n]*\|"
        
        if re.search(pattern, content):
            # 更新现有行
            content = re.sub(pattern, new_row, content)
            print(f"✅ 更新指标: {indicator_name}")
        else:
            print(f"⚠️ 未找到指标: {indicator_name}")
            return False
        
        # 更新总体统计
        content = self._update_statistics(content)
        
        # 更新时间戳
        content = self._update_timestamp(content)
        
        # 添加更新日志
        content = self._add_update_log(content, indicator_name, status, score)
        
        # 写回文件
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 进度表已更新: {indicator_name} -> {status_display}")
        return True
    
    def _update_statistics(self, content: str) -> str:
        """更新统计数据"""
        
        # 统计各状态的指标数量
        passed_count = len(re.findall(r'✅ PASSED', content))
        conditional_count = len(re.findall(r'⚠️ CONDITIONAL_PASS', content))
        failed_count = len(re.findall(r'❌ FAILED', content))
        in_progress_count = len(re.findall(r'🔄 IN_PROGRESS', content))
        pending_count = len(re.findall(r'⏸️ PENDING', content))
        
        completed_count = passed_count + conditional_count
        total_count = 82  # 总指标数
        completion_rate = (completed_count / total_count) * 100
        
        # 更新总体进度概览
        overview_pattern = r'(\*\*完成指标\*\*: )\d+/\d+ \([^)]+\)'
        new_overview = f"**完成指标**: {completed_count}/{total_count} ({completion_rate:.1f}%)"
        content = re.sub(overview_pattern, f"\\1{completed_count}/{total_count} ({completion_rate:.1f}%)", content)
        
        # 更新完成情况统计
        stats_section = f"""### 完成情况统计
- **已完成**: {completed_count}个指标 ({completion_rate:.1f}%)
- **进行中**: {in_progress_count}个指标 ({(in_progress_count/total_count)*100:.1f}%)
- **待验证**: {pending_count}个指标 ({(pending_count/total_count)*100:.1f}%)

### 质量统计
- **完全通过**: {passed_count}个指标
- **条件通过**: {conditional_count}个指标
- **验证失败**: {failed_count}个指标"""
        
        # 替换统计部分
        stats_pattern = r'### 完成情况统计.*?### 问题统计'
        if re.search(stats_pattern, content, re.DOTALL):
            content = re.sub(stats_pattern, stats_section + "\n\n### 问题统计", content, flags=re.DOTALL)
        
        return content
    
    def _update_timestamp(self, content: str) -> str:
        """更新时间戳"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 更新文档顶部的更新时间
        timestamp_pattern = r'(\*\*更新时间\*\*: )\d{4}-\d{2}-\d{2}'
        content = re.sub(timestamp_pattern, f"\\g<1>{current_date}", content)
        
        return content
    
    def _add_update_log(self, content: str, indicator_name: str, status: str, score: float) -> str:
        """添加更新日志"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        status_display = self.status_map.get(status, status)
        
        new_log_entry = f"""### {current_date}
- ✅ 完成{indicator_name}指标5阶段验证
- 📊 {indicator_name}获得{status.replace('_', ' ')}状态，评分{score:.1f}/100
- 📈 验证进度持续推进"""
        
        # 查找更新日志部分
        log_pattern = r'(## 📝 更新日志\n\n)'
        if re.search(log_pattern, content):
            content = re.sub(log_pattern, f"\\1{new_log_entry}\n\n", content)
        
        return content
    
    def add_indicator_details(self, indicator_name: str, details: Dict[str, Any]):
        """添加指标详细验证结果"""
        
        if not self.progress_file.exists():
            return False
        
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 构建详细结果部分
        detail_section = f"""
### {details.get('status_icon', '⚠️')} {indicator_name}指标验证详情
- **验证时间**: {details.get('completion_date', 'N/A')}
- **总体评分**: {details.get('total_score', 0):.1f}/100
- **各阶段评分**:
  - 算法差异预分析: {details.get('algorithm_analysis', 'N/A')}
  - 阶段1基础功能: {details.get('stage1_score', 'N/A')}
  - 阶段2形态识别: {details.get('stage2_score', 'N/A')}
  - 阶段3服务层集成: {details.get('stage3_score', 'N/A')}
  - 阶段4&5综合验证: {details.get('stage45_score', 'N/A')}
- **关键成就**: {details.get('achievements', '无')}
- **需要改进**: {details.get('improvements', '无')}
- **生产状态**: {details.get('production_status', '待确定')}"""
        
        # 查找详细验证结果部分
        details_pattern = r'(## 📊 详细验证结果\n)'
        if re.search(details_pattern, content):
            content = re.sub(details_pattern, f"\\1{detail_section}\n", content)
        
        # 写回文件
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已添加{indicator_name}详细验证结果")
        return True

def main():
    """主函数 - 命令行接口"""
    
    if len(sys.argv) < 6:
        print("用法: python update_progress.py <指标名称> <状态> <评分> <完成日期> <问题描述> [备注]")
        print("状态: PASSED, CONDITIONAL_PASS, FAILED, IN_PROGRESS, PENDING")
        print("示例: python update_progress.py MACD CONDITIONAL_PASS 80.0 2025-08-24 '参数管理方法缺失' '算法一致性100%'")
        return
    
    indicator_name = sys.argv[1]
    status = sys.argv[2]
    score = float(sys.argv[3])
    completion_date = sys.argv[4]
    issues = sys.argv[5]
    notes = sys.argv[6] if len(sys.argv) > 6 else ""
    
    updater = ProgressUpdater()
    success = updater.update_indicator_status(
        indicator_name=indicator_name,
        status=status,
        score=score,
        completion_date=completion_date,
        issues=issues,
        notes=notes
    )
    
    if success:
        print(f"🎉 {indicator_name}指标进度更新成功！")
    else:
        print(f"❌ {indicator_name}指标进度更新失败！")

if __name__ == "__main__":
    main()
