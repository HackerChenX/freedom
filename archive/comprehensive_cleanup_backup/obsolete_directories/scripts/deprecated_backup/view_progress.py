#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标验证进度查看脚本

快速查看当前验证进度和统计信息
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

class ProgressViewer:
    """进度查看器"""
    
    def __init__(self):
        """初始化查看器"""
        self.progress_file = Path("docs/技术指标验证进度表.md")
    
    def get_progress_summary(self) -> Dict[str, any]:
        """获取进度摘要"""
        
        if not self.progress_file.exists():
            return {"error": "进度表文件不存在"}
        
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取基本信息
        update_time = self._extract_update_time(content)
        completion_info = self._extract_completion_info(content)
        
        # 统计各状态指标
        status_stats = self._count_status(content)
        
        # 获取已完成指标列表
        completed_indicators = self._get_completed_indicators(content)
        
        # 获取下一个待验证指标
        next_indicators = self._get_next_indicators(content)
        
        return {
            "update_time": update_time,
            "completion_info": completion_info,
            "status_stats": status_stats,
            "completed_indicators": completed_indicators,
            "next_indicators": next_indicators
        }
    
    def _extract_update_time(self, content: str) -> str:
        """提取更新时间"""
        match = re.search(r'\*\*更新时间\*\*: (\d{4}-\d{2}-\d{2})', content)
        return match.group(1) if match else "未知"
    
    def _extract_completion_info(self, content: str) -> Dict[str, str]:
        """提取完成信息"""
        match = re.search(r'\*\*完成指标\*\*: (\d+)/(\d+) \(([^)]+)\)', content)
        if match:
            return {
                "completed": match.group(1),
                "total": match.group(2),
                "percentage": match.group(3)
            }
        return {"completed": "0", "total": "82", "percentage": "0.0%"}
    
    def _count_status(self, content: str) -> Dict[str, int]:
        """统计各状态数量"""
        return {
            "passed": len(re.findall(r'✅ PASSED', content)),
            "conditional_pass": len(re.findall(r'⚠️ CONDITIONAL_PASS', content)),
            "failed": len(re.findall(r'❌ FAILED', content)),
            "in_progress": len(re.findall(r'🔄 IN_PROGRESS', content)),
            "pending": len(re.findall(r'⏸️ PENDING', content))
        }
    
    def _get_completed_indicators(self, content: str) -> List[Dict[str, str]]:
        """获取已完成指标列表"""
        completed = []
        
        # 查找所有已完成的指标行
        pattern = r'\| \*\*([^*]+)\*\* \| (✅ PASSED|⚠️ CONDITIONAL_PASS) \| ([^|]+) \| ([^|]+) \|'
        matches = re.findall(pattern, content)
        
        for match in matches:
            completed.append({
                "name": match[0],
                "status": match[1],
                "score": match[2].strip(),
                "date": match[3].strip()
            })
        
        return completed
    
    def _get_next_indicators(self, content: str, limit: int = 5) -> List[str]:
        """获取下一批待验证指标"""
        next_indicators = []
        
        # 按优先级顺序查找待验证指标
        sections = [
            "核心指标验证进度 (P0级别)",
            "重要指标验证进度 (P1级别)",
            "常用指标验证进度 (P2级别)"
        ]
        
        for section in sections:
            if len(next_indicators) >= limit:
                break
                
            # 查找该部分的待验证指标
            section_pattern = rf'## 📈 {re.escape(section)}.*?(?=## |$)'
            section_match = re.search(section_pattern, content, re.DOTALL)
            
            if section_match:
                section_content = section_match.group(0)
                pending_pattern = r'\| \*\*([^*]+)\*\* \| ⏸️ PENDING'
                pending_matches = re.findall(pending_pattern, section_content)
                
                for indicator in pending_matches:
                    if len(next_indicators) < limit:
                        next_indicators.append(indicator)
        
        return next_indicators
    
    def display_progress(self):
        """显示进度信息"""
        
        summary = self.get_progress_summary()
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("🎯 技术指标验证进度概览")
        print("=" * 60)
        
        # 基本信息
        print(f"📅 更新时间: {summary['update_time']}")
        print(f"📊 完成进度: {summary['completion_info']['completed']}/{summary['completion_info']['total']} ({summary['completion_info']['percentage']})")
        
        # 状态统计
        stats = summary['status_stats']
        print(f"\n📈 状态统计:")
        print(f"  ✅ 完全通过: {stats['passed']}个")
        print(f"  ⚠️ 条件通过: {stats['conditional_pass']}个")
        print(f"  ❌ 验证失败: {stats['failed']}个")
        print(f"  🔄 进行中: {stats['in_progress']}个")
        print(f"  ⏸️ 待验证: {stats['pending']}个")
        
        # 已完成指标
        completed = summary['completed_indicators']
        if completed:
            print(f"\n🏆 已完成指标 ({len(completed)}个):")
            for indicator in completed:
                print(f"  {indicator['status']} {indicator['name']} - {indicator['score']} ({indicator['date']})")
        
        # 下一批待验证指标
        next_indicators = summary['next_indicators']
        if next_indicators:
            print(f"\n🎯 下一批待验证指标:")
            for i, indicator in enumerate(next_indicators, 1):
                print(f"  {i}. {indicator}")
        
        # 进度条
        completed_count = int(summary['completion_info']['completed'])
        total_count = int(summary['completion_info']['total'])
        progress_bar = self._create_progress_bar(completed_count, total_count)
        print(f"\n📊 总体进度:")
        print(f"  {progress_bar}")
        print(f"  {completed_count}/{total_count} 指标已完成")
    
    def _create_progress_bar(self, completed: int, total: int, width: int = 40) -> str:
        """创建进度条"""
        percentage = completed / total
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:.1%}"
    
    def get_indicator_details(self, indicator_name: str) -> Dict[str, str]:
        """获取指标详细信息"""
        
        if not self.progress_file.exists():
            return {"error": "进度表文件不存在"}
        
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找指标行
        pattern = rf'\| \*\*{re.escape(indicator_name)}\*\* \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|'
        match = re.search(pattern, content)
        
        if match:
            return {
                "name": indicator_name,
                "status": match.group(1).strip(),
                "score": match.group(2).strip(),
                "date": match.group(3).strip(),
                "issues": match.group(4).strip(),
                "notes": match.group(5).strip()
            }
        
        return {"error": f"未找到指标: {indicator_name}"}

def main():
    """主函数"""
    import sys
    
    viewer = ProgressViewer()
    
    if len(sys.argv) > 1:
        # 查看特定指标详情
        indicator_name = sys.argv[1]
        details = viewer.get_indicator_details(indicator_name)
        
        if "error" in details:
            print(f"❌ {details['error']}")
        else:
            print(f"📊 {details['name']} 指标详情")
            print("=" * 40)
            print(f"状态: {details['status']}")
            print(f"评分: {details['score']}")
            print(f"完成时间: {details['date']}")
            print(f"问题: {details['issues']}")
            print(f"备注: {details['notes']}")
    else:
        # 显示总体进度
        viewer.display_progress()

if __name__ == "__main__":
    main()
