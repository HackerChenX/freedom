#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为缺少calculate_raw_score方法的指标类添加实现
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# 为ZXMAmplitudeElasticity类添加calculate_raw_score方法
def add_amplitude_elasticity_raw_score():
    """为ZXMAmplitudeElasticity添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/zxm/elasticity_indicators.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经实现了该方法
    if 'def calculate_raw_score(' in content and 'class ZXMAmplitudeElasticity(' in content:
        print("ZXMAmplitudeElasticity类已经实现了calculate_raw_score方法")
        return
    
    # 定义要添加的方法
    method_to_add = '''
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM振幅弹性指标的原始评分
        
        Args:
            data: 输入数据，包含OHLC数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 有振幅弹性信号时加分
        score[result["XG"]] += 40
        
        # 根据振幅大小给予额外加分
        if "Amplitude" in result.columns:
            # 振幅越大，加分越多（最多额外加10分）
            amplitude_bonus = result["Amplitude"].apply(lambda x: min(10, max(0, (x - 8.1) / 2)))
            score += amplitude_bonus
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
'''
    
    # 找到类定义结束的位置
    class_end_marker = "class ZXMRiseElasticity(BaseIndicator):"
    if class_end_marker not in content:
        print("无法找到ZXMAmplitudeElasticity类的结束位置")
        return
    
    # 在类定义结束前插入方法
    insert_position = content.find(class_end_marker)
    modified_content = content[:insert_position] + method_to_add + content[insert_position:]
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为ZXMAmplitudeElasticity类添加calculate_raw_score方法")

# 为ZXMRiseElasticity类添加calculate_raw_score方法
def add_rise_elasticity_raw_score():
    """为ZXMRiseElasticity添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/zxm/elasticity_indicators.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经实现了该方法
    if 'def calculate_raw_score(' in content and 'class ZXMRiseElasticity(' in content:
        if 'def calculate_raw_score(self' in content[content.find('class ZXMRiseElasticity('):content.find('class ElasticityIndicator(')]:
            print("ZXMRiseElasticity类已经实现了calculate_raw_score方法")
            return
    
    # 定义要添加的方法
    method_to_add = '''
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM涨幅弹性指标的原始评分
        
        Args:
            data: 输入数据，包含收盘价数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50, index=data.index)
        
        # 有涨幅弹性信号时加分
        score[result["XG"]] += 40
        
        # 根据涨幅大小给予额外加分
        if "RiseRatio" in result.columns:
            # 涨幅越大，加分越多（最多额外加10分）
            rise_bonus = result["RiseRatio"].apply(lambda x: min(10, max(0, (x - 1.07) * 100)))
            score += rise_bonus
        
        # 确保评分在0-100范围内
        score = score.clip(0, 100)
        
        return score
'''
    
    # 找到类定义结束的位置
    class_end_marker = "class ElasticityIndicator(BaseIndicator):"
    if class_end_marker not in content:
        print("无法找到ZXMRiseElasticity类的结束位置")
        return
    
    # 在类定义结束前插入方法
    insert_position = content.find(class_end_marker)
    modified_content = content[:insert_position] + method_to_add + content[insert_position:]
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为ZXMRiseElasticity类添加calculate_raw_score方法")

# 为ZXMElasticityScore类添加calculate_raw_score方法
def add_elasticity_score_raw_score():
    """为ZXMElasticityScore添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/zxm/score_indicators.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经实现了该方法
    if 'def calculate_raw_score(' in content and 'class ZXMElasticityScore(' in content:
        if 'def calculate_raw_score(self' in content[content.find('class ZXMElasticityScore('):content.find('class ZXMBuyPointScore(')]:
            print("ZXMElasticityScore类已经实现了calculate_raw_score方法")
            return
    
    # 定义要添加的方法
    method_to_add = '''
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM弹性评分指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 直接使用计算的弹性评分作为原始评分
        return result["ElasticityScore"]
'''
    
    # 找到类定义结束的位置
    class_end_marker = "class ZXMBuyPointScore(BaseIndicator):"
    if class_end_marker not in content:
        print("无法找到ZXMElasticityScore类的结束位置")
        return
    
    # 在类定义结束前插入方法
    insert_position = content.find(class_end_marker)
    modified_content = content[:insert_position] + method_to_add + content[insert_position:]
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为ZXMElasticityScore类添加calculate_raw_score方法")

# 为ZXMBuyPointScore类添加calculate_raw_score方法
def add_buy_point_score_raw_score():
    """为ZXMBuyPointScore添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/zxm/score_indicators.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经实现了该方法
    if 'def calculate_raw_score(' in content and 'class ZXMBuyPointScore(' in content:
        if 'def calculate_raw_score(self' in content[content.find('class ZXMBuyPointScore('):content.find('class StockScoreCalculator(')]:
            print("ZXMBuyPointScore类已经实现了calculate_raw_score方法")
            return
    
    # 定义要添加的方法
    method_to_add = '''
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM买点评分指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 计算指标
        result = self.calculate(data)
        
        # 直接使用计算的买点评分作为原始评分
        return result["BuyPointScore"]
'''
    
    # 找到类定义结束的位置
    class_end_marker = "class StockScoreCalculator(BaseIndicator):"
    if class_end_marker not in content:
        print("无法找到ZXMBuyPointScore类的结束位置")
        return
    
    # 在类定义结束前插入方法
    insert_position = content.find(class_end_marker)
    modified_content = content[:insert_position] + method_to_add + content[insert_position:]
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为ZXMBuyPointScore类添加calculate_raw_score方法")

# 为ZXMPatternIndicator类添加calculate_raw_score方法
def add_pattern_indicator_raw_score():
    """为ZXMPatternIndicator添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/pattern/zxm_patterns.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经实现了该方法
    if 'def calculate_raw_score(' in content and 'class ZXMPatternIndicator(' in content:
        print("ZXMPatternIndicator类已经实现了calculate_raw_score方法")
        return
    
    # 获取类的末尾位置
    class_start = content.find('class ZXMPatternIndicator(BaseIndicator):')
    if class_start == -1:
        print("无法找到ZXMPatternIndicator类")
        return
    
    # 找到_identify_absorption_patterns方法的位置
    method_pos = content.find('def _identify_absorption_patterns', class_start)
    if method_pos == -1:
        # 如果找不到这个方法，则查找类中最后一个方法的结束位置
        pattern = r'def [^(]+\([^)]*\):'
        import re
        matches = list(re.finditer(pattern, content[class_start:]))
        if not matches:
            print("无法确定在哪里插入新方法")
            return
        last_method_start = matches[-1].start() + class_start
        # 找到该方法的结束位置（大致估计为下一个方法的开始或者文件结束）
        next_method = content.find('\ndef ', last_method_start + 1)
        if next_method == -1:
            insert_position = len(content)
        else:
            # 向上查找最后一个}或者缩进减少的地方
            pos = next_method
            while pos > last_method_start:
                if content[pos] == '}' or (content[pos:pos+8].count('\n') >= 2 and content[pos:pos+8].count('    ') == 0):
                    break
                pos -= 1
            insert_position = pos
    else:
        insert_position = method_pos
    
    # 定义要添加的方法
    method_to_add = '''
    def calculate_raw_score(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        计算ZXM体系买点和吸筹形态指标的原始评分
        
        Args:
            data: 输入数据，包含OHLCV数据
            **kwargs: 其他参数
            
        Returns:
            pd.Series: 评分结果，0-100分
        """
        # 确保数据包含必需的列
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                return pd.Series(50.0, index=data.index)  # 返回默认中性评分
        
        # 计算指标
        result = self.calculate(
            data["open"].values, 
            data["high"].values, 
            data["low"].values, 
            data["close"].values, 
            data["volume"].values
        )
        
        # 初始化评分为基础分50分（中性）
        score = pd.Series(50.0, index=data.index)
        
        # 买点形态评分
        for i, idx in enumerate(data.index):
            current_score = 50.0
            
            # 一类买点（最强买点）：+40分
            if i < len(result.get('class_one_buy', [])) and result['class_one_buy'][i]:
                current_score += 40
            
            # 二类买点（强势回调买点）：+30分
            elif i < len(result.get('class_two_buy', [])) and result['class_two_buy'][i]:
                current_score += 30
            
            # 三类买点（超跌反弹买点）：+20分
            elif i < len(result.get('class_three_buy', [])) and result['class_three_buy'][i]:
                current_score += 20
            
            # 其他特殊买点：+15-25分
            special_buy_signals = [
                'breakout_pullback_buy',      # 强势突破回踩型：+25分
                'volume_shrink_platform_buy', # 连续缩量平台型：+20分
                'long_shadow_support_buy',    # 长下影线支撑型：+15分
                'ma_converge_diverge_buy'     # 均线粘合发散型：+20分
            ]
            
            for signal_name in special_buy_signals:
                if signal_name in result and i < len(result[signal_name]) and result[signal_name][i]:
                    if signal_name == 'breakout_pullback_buy':
                        current_score += 25
                    elif signal_name in ['volume_shrink_platform_buy', 'ma_converge_diverge_buy']:
                        current_score += 20
                    elif signal_name == 'long_shadow_support_buy':
                        current_score += 15
                    break  # 只取最高的一个特殊买点信号
            
            # 吸筹形态评分（如果有这些指标）
            absorption_signals = [
                'large_scale_absorption',     # 大级别吸筹：+15分
                'stealth_absorption',         # 隐蔽性吸筹：+10分
                'repeated_absorption',        # 反复性吸筹：+12分
                'breakthrough_absorption'     # 突破性吸筹：+18分
            ]
            
            for signal_name in absorption_signals:
                if signal_name in result and i < len(result[signal_name]) and result[signal_name][i]:
                    if signal_name == 'large_scale_absorption':
                        current_score += 15
                    elif signal_name == 'stealth_absorption':
                        current_score += 10
                    elif signal_name == 'repeated_absorption':
                        current_score += 12
                    elif signal_name == 'breakthrough_absorption':
                        current_score += 18
            
            score[idx] = min(100, max(0, current_score))  # 确保评分在0-100范围内
        
        return score
    
'''
    
    # 插入方法
    modified_content = content[:insert_position] + method_to_add + content[insert_position:]
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为ZXMPatternIndicator类添加calculate_raw_score方法")

def main():
    """主函数"""
    print("开始为缺少calculate_raw_score方法的指标类添加实现...")
    
    # 添加实现
    add_amplitude_elasticity_raw_score()
    add_rise_elasticity_raw_score()
    add_elasticity_score_raw_score()
    add_buy_point_score_raw_score()
    add_pattern_indicator_raw_score()
    
    print("完成!")

if __name__ == "__main__":
    main() 