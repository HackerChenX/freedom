#!/usr/bin/env python3
"""
批量指标修复工具
用于快速修复高风险指标的信号生成语义问题
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class Batch_indicator_fixer:
    """批量指标修复器"""
    
    def __init__(self):
        self.indicators_dir = Path("indicators")
        self.fixed_count = 0
        self.failed_count = 0
        
    def get_high_risk_indicators_Fix(self) -> List[str]:
        """获取高风险指标列表"""
        from tools.automated_risk_detection import Automated_risk_detector
        
        detector = Automated_risk_detector()
        results = detector.scan_all_indicators()
        
        high_risk_indicators = []
        for name, result in results.items():
            if result.risk_level.value == 'high':
                high_risk_indicators.append(name)
        
        return high_risk_indicators
    
    def find_indicator_file(self, indicator_name: str) -> str:
        """查找指标文件路径"""
        # 常见的指标文件名模式
        possible_names = [
            indicator_name.lower() + ".py",
            indicator_name + ".py"
        ]
        
        # 搜索指标文件
        for root, dirs, files in os.walk(self.indicators_dir):
            for file in files:
                if file in possible_names:
                    return os.path.join(root, file)
        
        return None
    
    def detect_signal_generation_pattern(self, file_path: str) -> Tuple[bool, int, int]:
        """检测信号生成模式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找add_signal_generation调用
            pattern = r'(\s+)# 添加形态识别和信号生成\s*\n(\s+).*add_pattern_detection.*\n(\s+).*add_signal_generation.*\n\s*\n(\s+)return'
            match = re.search(pattern, content, re.MULTILINE)
            
            if match:
                # 找到插入点
                start_pos = match.start(4)  # return语句的位置
                lines = content[:start_pos].split('\n')
                insert_line = len(lines) - 1
                return True, insert_line, start_pos
            
            return False, -1, -1
            
        except Exception as e:
            logger.error(f"检测信号生成模式失败 {file_path}: {e}")
            return False, -1, -1
    
    def generate_signal_logic_code(self, indicator_name: str, indicator_type: str) -> str:
        """生成信号逻辑代码"""
        method_name = f"_apply_{indicator_name.lower()}_signal_logic"
        
        if indicator_type == 'state_type':
            signal_logic = '''
            # 状态型指标：基于布尔输出
            if 'XG' in df.columns:
                df.loc[:, 'buy_signal'] = df["XG"] == True
                df.loc[:, 'sell_signal'] = df["XG"] == False
                df.loc[:, 'hold_signal'] = df["XG"] == False
            else:
                # 使用第一个布尔列作为信号源
                bool_cols = [col for col in df.columns if df[col].dtype == bool]
                if bool_cols:
                    signal_col = bool_cols[0]
                    df.loc[:, 'buy_signal'] = df[signal_col] == True
                    df.loc[:, 'sell_signal'] = df[signal_col] == False
                    df.loc[:, 'hold_signal'] = df[signal_col] == False
                else:
                    # 默认信号
                    df.loc[:, 'buy_signal'] = False
                    df.loc[:, 'sell_signal'] = False
                    df.loc[:, 'hold_signal'] = True'''
                    
        elif indicator_type == 'count_type':
            signal_logic = '''
            # 计数型指标：基于数值阈值
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            signal_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if signal_cols:
                signal_col = signal_cols[0]  # 使用第一个数值列
                df.loc[:, 'buy_signal'] = df[signal_col] > 0
                df.loc[:, 'sell_signal'] = df[signal_col] <= 0
                df.loc[:, 'hold_signal'] = df[signal_col] <= 0
            else:
                # 默认信号
                df.loc[:, 'buy_signal'] = False
                df.loc[:, 'sell_signal'] = False
                df.loc[:, 'hold_signal'] = True'''
                
        elif indicator_type == 'level_type':
            signal_logic = '''
            # 等级型指标：基于评分阈值
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            signal_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            if signal_cols:
                signal_col = signal_cols[0]  # 使用第一个数值列
                threshold = df[signal_col].median()  # 使用中位数作为阈值
                df.loc[:, 'buy_signal'] = df[signal_col] > threshold
                df.loc[:, 'sell_signal'] = df[signal_col] < threshold
                df.loc[:, 'hold_signal'] = df[signal_col] == threshold
            else:
                # 默认信号
                df.loc[:, 'buy_signal'] = False
                df.loc[:, 'sell_signal'] = False
                df.loc[:, 'hold_signal'] = True'''
                
        else:  # composite_type 或其他
            signal_logic = '''
            # 复合型指标：检查专用信号字段
            if 'BuySignal' in df.columns and 'SellSignal' in df.columns:
                df.loc[:, 'buy_signal'] = df["BuySignal"] == True
                df.loc[:, 'sell_signal'] = df["SellSignal"] == True
                df.loc[:, 'hold_signal'] = ~(df['buy_signal'] | df['sell_signal'])
            elif 'Signal' in df.columns:
                df.loc[:, 'buy_signal'] = df["Signal"] == True
                df.loc[:, 'sell_signal'] = df["Signal"] == False
                df.loc[:, 'hold_signal'] = df["Signal"] == False
            else:
                # 默认信号
                df.loc[:, 'buy_signal'] = False
                df.loc[:, 'sell_signal'] = False
                df.loc[:, 'hold_signal'] = True'''
        
        return f'''
        # 重写信号生成逻辑（{indicator_name}指标特定逻辑）
        df = self.{method_name}(df)

        return df
    
    def {method_name}(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用{indicator_name}指标特定的信号生成逻辑
        """
        try:{signal_logic}
            
            # 确保信号类型为布尔值
            df['buy_signal'] = df['buy_signal'].astype(bool)
            df['sell_signal'] = df['sell_signal'].astype(bool)
            df['hold_signal'] = df['hold_signal'].astype(bool)
            
        except Exception as e:
            logger.warning(f"{indicator_name}信号生成失败: {{e}}")
            # 如果出错，使用默认信号
            df.loc[:, 'buy_signal'] = False
            df.loc[:, 'sell_signal'] = False
            df.loc[:, 'hold_signal'] = True
        
        return df'''
    
    def apply_fix_to_file(self, file_path: str, indicator_name: str, indicator_type: str) -> bool:
        """应用修复到文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检测插入点
            has_pattern, insert_line, insert_pos = self.detect_signal_generation_pattern(file_path)
            
            if not has_pattern:
                logger.warning(f"未找到信号生成模式: {file_path}")
                return False
            
            # 生成修复代码
            fix_code = self.generate_signal_logic_code(indicator_name, indicator_type)
            
            # 插入修复代码
            lines = content.split('\n')
            
            # 找到return语句的位置
            return_line_idx = -1
            for i in range(len(lines)):
                if 'return df' in lines[i] and 'add_signal_generation' in '\n'.join(lines[max(0, i-5):i]):
                    return_line_idx = i
                    break
            
            if return_line_idx == -1:
                logger.warning(f"未找到return语句: {file_path}")
                return False
            
            # 插入修复代码
            lines.insert(return_line_idx, fix_code)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"✅ 成功修复指标: {indicator_name}")
            self.fixed_count += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ 修复指标失败 {indicator_name}: {e}")
            self.failed_count += 1
            return False
    
    def fix_indicator(self, indicator_name: str, indicator_type: str) -> bool:
        """修复单个指标"""
        # 查找指标文件
        file_path = self.find_indicator_file(indicator_name)
        
        if not file_path:
            logger.warning(f"未找到指标文件: {indicator_name}")
            return False
        
        # 应用修复
        return self.apply_fix_to_file(file_path, indicator_name, indicator_type)
    
    def batch_fix_indicators(self, indicators_info: Dict[str, str]) -> Dict[str, bool]:
        """批量修复指标"""
        results = {}
        
        logger.info(f"开始批量修复 {len(indicators_info)} 个指标...")
        
        for indicator_name, indicator_type in indicators_info.items():
            logger.info(f"修复指标: {indicator_name} ({indicator_type})")
            success = self.fix_indicator(indicator_name, indicator_type)
            results[indicator_name] = success
        
        logger.info(f"批量修复完成: 成功 {self.fixed_count}, 失败 {self.failed_count}")
        return results

if __name__ == "__main__":
    fixer = Batch_indicator_fixer()
    
    # 获取高风险指标
    high_risk_indicators = fixer.get_high_risk_indicators_Fix()
    print(f"发现 {len(high_risk_indicators)} 个高风险指标")
    
    # 这里需要手动指定指标类型，因为自动检测可能不准确
    # 可以根据风险检测结果来设置
