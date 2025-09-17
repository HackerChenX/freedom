#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复指标计算问题
解决ZXM指标实例化、Score指标calculate方法缺失、形态识别指标问题
"""

import os
import re
import shutil
from typing import List, Dict, Any
from datetime import datetime

class IndicatorCalculationFixer:
    """指标计算问题修复器"""
    
    def __init__(self):
        self.backup_dir = f"backup/indicator_calculation_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixed_files = []
        self.fix_stats = {
            'files_processed': 0,
            'files_modified': 0,
            'zxm_fixes': 0,
            'score_fixes': 0,
            'pattern_fixes': 0,
            'enum_fixes': 0
        }
    
    def fix_all_calculation_issues(self) -> Dict[str, Any]:
        """修复所有指标计算问题"""
        print("🔧 开始修复指标计算问题")
        print("=" * 50)
        
        # 创建备份目录
        os.makedirs(self.backup_dir, exist_ok=True)
        
        results = {
            "fix_type": "INDICATOR_CALCULATION_FIX",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fixes_applied": {},
            "summary": {}
        }
        
        # 1. 修复ZXM指标构造函数问题
        results["fixes_applied"]["zxm_constructor_fix"] = self.fix_zxm_constructor_issues()
        
        # 2. 修复Score指标calculate方法缺失
        results["fixes_applied"]["score_calculate_fix"] = self.fix_score_calculate_methods()
        
        # 3. 修复形态识别指标问题
        results["fixes_applied"]["pattern_recognition_fix"] = self.fix_pattern_recognition_issues()
        
        # 4. 修复Advanced_pattern_type导入问题
        results["fixes_applied"]["enum_import_fix"] = self.fix_enum_import_issues()
        
        # 5. 修复Period枚举JSON序列化问题
        results["fixes_applied"]["period_serialization_fix"] = self.fix_period_serialization()
        
        # 计算总结
        self._calculate_summary(results)
        
        return results
    
    def fix_zxm_constructor_issues(self) -> Dict[str, Any]:
        """修复ZXM指标构造函数问题"""
        print("🔧 修复ZXM指标构造函数问题")
        
        fix_result = {
            "action": "fix_zxm_constructor_issues",
            "files_modified": [],
            "fixes_applied": []
        }
        
        # 修复指标注册表中的ZXM指标实例化
        registry_file = 'indicators/complete_indicator_registry.py'
        
        if os.path.exists(registry_file):
            try:
                # 备份文件
                backup_path = os.path.join(self.backup_dir, 'complete_indicator_registry.py')
                shutil.copy2(registry_file, backup_path)
                
                # 读取文件
                with open(registry_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 修复ZXM指标的实例化，移除不必要的参数
                zxm_patterns = [
                    (r'(\s+)(["\']ZXM_[^"\']+["\']:\s*lambda[^:]*:\s*)([A-Za-z_]+)\([^)]*\)', r'\1\2\3()'),
                    (r'(\s+)(["\']ZXM_[^"\']+["\']:\s*)([A-Za-z_]+)\([^)]*\)', r'\1\2\3()'),
                ]
                
                modified = False
                for pattern, replacement in zxm_patterns:
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        modified = True
                        fix_result["fixes_applied"].append(f"修复ZXM指标实例化参数: {pattern}")
                
                # 写入修改后的内容
                if modified:
                    with open(registry_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fix_result["files_modified"].append(registry_file)
                    self.fix_stats['zxm_fixes'] += 1
                    print(f"  ✅ 已修复: {registry_file}")
                
            except Exception as e:
                print(f"  ❌ 修复失败: {e}")
        
        return fix_result
    
    def fix_score_calculate_methods(self) -> Dict[str, Any]:
        """修复Score指标calculate方法缺失"""
        print("🔧 修复Score指标calculate方法缺失")
        
        fix_result = {
            "action": "fix_score_calculate_methods",
            "files_created": [],
            "fixes_applied": []
        }
        
        # 创建通用Score指标基类
        score_base_content = '''"""
通用Score指标基类
为所有Score类指标提供统一的calculate方法实现
"""

import pandas as pd
from abc import ABC, abstractmethod
from indicators.base_indicator import BaseIndicator

class BaseScoreIndicator(BaseIndicator, ABC):
    """Score指标基类"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算Score指标
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含评分的结果
        """
        if data.empty:
            return pd.DataFrame()
        
        # 计算原始评分
        scores = self.calculate_scores(data)
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=data.index)
        result['score'] = scores
        result['signal'] = self.generate_score_signal(scores)
        
        return result
    
    @abstractmethod
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        计算具体的评分
        
        Args:
            data: 输入数据
            
        Returns:
            pd.Series: 评分序列
        """
        pass
    
    def generate_score_signal(self, scores: pd.Series) -> pd.Series:
        """
        根据评分生成信号
        
        Args:
            scores: 评分序列
            
        Returns:
            pd.Series: 信号序列
        """
        signals = pd.Series('HOLD', index=scores.index)
        signals[scores >= 70] = 'BUY'
        signals[scores <= 30] = 'SELL'
        return signals

class MACDScoreIndicator(BaseScoreIndicator):
    """MACD评分指标"""
    
    def __init__(self):
        super().__init__("MACD_SCORE")
    
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD评分"""
        # 简化的MACD评分逻辑
        close = data['close']
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        # 基于MACD和信号线的关系计算评分
        scores = pd.Series(50, index=data.index)  # 基础分50
        scores[macd > signal] += 20  # 金叉加分
        scores[macd < signal] -= 20  # 死叉减分
        
        return scores.clip(0, 100)

class RSIScoreIndicator(BaseScoreIndicator):
    """RSI评分指标"""
    
    def __init__(self):
        super().__init__("RSI_SCORE")
    
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI评分"""
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 基于RSI值计算评分
        scores = pd.Series(50, index=data.index)
        scores[rsi < 30] += 30  # 超卖区域
        scores[rsi > 70] -= 30  # 超买区域
        
        return scores.clip(0, 100)

class BOLLScoreIndicator(BaseScoreIndicator):
    """BOLL评分指标"""
    
    def __init__(self):
        super().__init__("BOLL_SCORE")
    
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """计算BOLL评分"""
        close = data['close']
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        # 基于价格在布林带中的位置计算评分
        scores = pd.Series(50, index=data.index)
        scores[close <= lower] += 30  # 接近下轨
        scores[close >= upper] -= 30  # 接近上轨
        
        return scores.clip(0, 100)

class KDJScoreIndicator(BaseScoreIndicator):
    """KDJ评分指标"""
    
    def __init__(self):
        super().__init__("KDJ_SCORE")
    
    def calculate_scores(self, data: pd.DataFrame) -> pd.Series:
        """计算KDJ评分"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算KDJ
        llv = low.rolling(window=9).min()
        hhv = high.rolling(window=9).max()
        rsv = (close - llv) / (hhv - llv) * 100
        k = rsv.ewm(alpha=1/3).mean()
        d = k.ewm(alpha=1/3).mean()
        j = 3 * k - 2 * d
        
        # 基于KDJ值计算评分
        scores = pd.Series(50, index=data.index)
        scores[(k < 20) & (d < 20)] += 30  # 超卖
        scores[(k > 80) & (d > 80)] -= 30  # 超买
        scores[k > d] += 10  # K线在D线上方
        
        return scores.clip(0, 100)
'''
        
        # 保存Score指标基类文件
        score_base_file = 'indicators/score_indicators.py'
        with open(score_base_file, 'w', encoding='utf-8') as f:
            f.write(score_base_content)
        
        fix_result["files_created"].append(score_base_file)
        fix_result["fixes_applied"].append("创建通用Score指标基类和实现")
        self.fix_stats['score_fixes'] += 1
        
        print(f"  ✅ 已创建: {score_base_file}")
        
        return fix_result
    
    def fix_pattern_recognition_issues(self) -> Dict[str, Any]:
        """修复形态识别指标问题"""
        print("🔧 修复形态识别指标问题")
        
        fix_result = {
            "action": "fix_pattern_recognition_issues",
            "files_created": [],
            "fixes_applied": []
        }
        
        # 创建简化的形态识别指标
        pattern_content = '''"""
简化的形态识别指标实现
为THREE_BLACK_CROWS、THREE_WHITE_SOLDIERS、V_SHAPED_REVERSAL提供基础实现
"""

import pandas as pd
import numpy as np
from indicators.base_indicator import BaseIndicator

class ThreeBlackCrowsIndicator(BaseIndicator):
    """三黑鸦形态识别指标"""
    
    def __init__(self):
        super().__init__("THREE_BLACK_CROWS")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算三黑鸦形态"""
        if len(data) < 3:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        result['three_black_crows'] = False
        
        # 简化的三黑鸦识别逻辑
        close = data['close']
        open_price = data['open']
        
        # 连续三根阴线
        bear1 = close < open_price
        bear2 = bear1.shift(1)
        bear3 = bear1.shift(2)
        
        # 收盘价逐步下降
        declining = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
        
        pattern = bear1 & bear2 & bear3 & declining
        result.loc[pattern.index[2:], 'three_black_crows'] = pattern[2:]
        
        return result

class ThreeWhiteSoldiersIndicator(BaseIndicator):
    """三白兵形态识别指标"""
    
    def __init__(self):
        super().__init__("THREE_WHITE_SOLDIERS")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算三白兵形态"""
        if len(data) < 3:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        result['three_white_soldiers'] = False
        
        # 简化的三白兵识别逻辑
        close = data['close']
        open_price = data['open']
        
        # 连续三根阳线
        bull1 = close > open_price
        bull2 = bull1.shift(1)
        bull3 = bull1.shift(2)
        
        # 收盘价逐步上升
        rising = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
        
        pattern = bull1 & bull2 & bull3 & rising
        result.loc[pattern.index[2:], 'three_white_soldiers'] = pattern[2:]
        
        return result

class VShapedReversalIndicator(BaseIndicator):
    """V型反转形态识别指标"""
    
    def __init__(self):
        super().__init__("V_SHAPED_REVERSAL")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算V型反转形态"""
        if len(data) < 5:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        result['v_shaped_reversal'] = False
        
        # 简化的V型反转识别逻辑
        close = data['close']
        low = data['low']
        
        # 寻找局部最低点
        local_min = (low < low.shift(1)) & (low < low.shift(-1))
        
        # 在最低点前后价格快速下跌和上涨
        for i in range(2, len(data) - 2):
            if local_min.iloc[i]:
                # 检查前两天是否下跌，后两天是否上涨
                before_decline = (close.iloc[i-1] < close.iloc[i-2]) and (close.iloc[i] < close.iloc[i-1])
                after_rise = (close.iloc[i+1] > close.iloc[i]) and (close.iloc[i+2] > close.iloc[i+1])
                
                if before_decline and after_rise:
                    result.iloc[i, result.columns.get_loc('v_shaped_reversal')] = True
        
        return result
'''
        
        # 保存形态识别指标文件
        pattern_file = 'indicators/pattern_indicators.py'
        with open(pattern_file, 'w', encoding='utf-8') as f:
            f.write(pattern_content)
        
        fix_result["files_created"].append(pattern_file)
        fix_result["fixes_applied"].append("创建简化的形态识别指标实现")
        self.fix_stats['pattern_fixes'] += 1
        
        print(f"  ✅ 已创建: {pattern_file}")
        
        return fix_result
    
    def fix_enum_import_issues(self) -> Dict[str, Any]:
        """修复枚举导入问题"""
        print("🔧 修复枚举导入问题")
        
        fix_result = {
            "action": "fix_enum_import_issues",
            "files_modified": [],
            "fixes_applied": []
        }
        
        # 确保Advanced_pattern_type可以正确导入
        pattern_file = 'indicators/pattern/advanced_candlestick_patterns.py'
        
        if os.path.exists(pattern_file):
            try:
                # 备份文件
                backup_path = os.path.join(self.backup_dir, 'advanced_candlestick_patterns.py')
                shutil.copy2(pattern_file, backup_path)
                
                # 读取文件
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 确保AdvancedPatternType被正确导出
                if 'Advanced_pattern_type' in content and 'AdvancedPatternType' not in content:
                    content = content.replace('Advanced_pattern_type', 'AdvancedPatternType')
                    
                    with open(pattern_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fix_result["files_modified"].append(pattern_file)
                    fix_result["fixes_applied"].append("修复AdvancedPatternType枚举名称")
                    self.fix_stats['enum_fixes'] += 1
                    print(f"  ✅ 已修复: {pattern_file}")
                
            except Exception as e:
                print(f"  ❌ 修复失败: {e}")
        
        return fix_result
    
    def fix_period_serialization(self) -> Dict[str, Any]:
        """修复Period枚举JSON序列化问题"""
        print("🔧 修复Period枚举JSON序列化问题")
        
        fix_result = {
            "action": "fix_period_serialization",
            "files_created": [],
            "fixes_applied": []
        }
        
        # 创建JSON序列化工具
        serialization_content = '''"""
JSON序列化工具
解决Period枚举等对象的JSON序列化问题
"""

import json
from enum import Enum
from datetime import datetime, date
from db.sql_manager import SQLManager, QueryType
import pandas as pd
import numpy as np

class EnhancedJSONEncoder(json.JSONEncoder):
    """增强的JSON编码器，支持更多数据类型"""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """安全的JSON序列化"""
    try:
        return json.dumps(obj, cls=EnhancedJSONEncoder, ensure_ascii=False, **kwargs)
    except Exception as e:
        # 如果序列化失败，转换为字符串
        return json.dumps(str(obj), ensure_ascii=False, **kwargs)

def safe_json_loads(json_str):
    """安全的JSON反序列化"""
    try:
        return json.loads(json_str)
    except Exception as e:
        return {"error": f"JSON解析失败: {e}", "raw_data": json_str}
'''
        
        # 保存序列化工具文件
        serialization_file = 'utils/json_serializer.py'
        with open(serialization_file, 'w', encoding='utf-8') as f:
            f.write(serialization_content)
        
        fix_result["files_created"].append(serialization_file)
        fix_result["fixes_applied"].append("创建JSON序列化工具")
        
        print(f"  ✅ 已创建: {serialization_file}")
        
        return fix_result
    
    def _calculate_summary(self, results: Dict[str, Any]):
        """计算修复总结"""
        summary = results["summary"]
        summary.update(self.fix_stats)
        summary["backup_directory"] = self.backup_dir
        summary["fixed_files"] = self.fixed_files
        
        # 计算总修复数
        total_fixes = (summary["zxm_fixes"] + summary["score_fixes"] + 
                      summary["pattern_fixes"] + summary["enum_fixes"])
        summary["total_fixes"] = total_fixes

def main():
    """主函数"""
    print("🔧 指标计算问题修复")
    print("=" * 50)
    
    fixer = IndicatorCalculationFixer()
    results = fixer.fix_all_calculation_issues()
    
    # 显示修复结果
    print(f"\n📊 修复摘要:")
    print(f"ZXM指标修复: {results['summary']['zxm_fixes']}")
    print(f"Score指标修复: {results['summary']['score_fixes']}")
    print(f"形态识别修复: {results['summary']['pattern_fixes']}")
    print(f"枚举问题修复: {results['summary']['enum_fixes']}")
    print(f"总修复数: {results['summary']['total_fixes']}")
    
    # 显示修复详情
    print(f"\n📋 修复详情:")
    for fix_type, fix_result in results["fixes_applied"].items():
        if isinstance(fix_result, dict) and "fixes_applied" in fix_result:
            if fix_result["fixes_applied"]:
                print(f"  ✅ {fix_type}:")
                for fix in fix_result["fixes_applied"]:
                    print(f"    - {fix}")
    
    print(f"\n💾 备份目录: {results['summary']['backup_directory']}")
    print(f"\n✅ 指标计算问题修复完成")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
