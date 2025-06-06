#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门为ZXMAmplitudeElasticity类添加calculate_raw_score方法
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def add_amplitude_elasticity_raw_score():
    """为ZXMAmplitudeElasticity添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/zxm/elasticity_indicators.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经实现了该方法
    class_start = content.find('class ZXMAmplitudeElasticity(BaseIndicator):')
    class_end = content.find('class ZXMRiseElasticity(BaseIndicator):')
    
    if 'def calculate_raw_score(' in content[class_start:class_end]:
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
    
    # 插入方法到类定义的适当位置
    # 找到calculate方法的结束位置
    calculate_method_start = content.find('def calculate(self', class_start, class_end)
    if calculate_method_start != -1:
        # 找到calculate方法的结束位置（下一个def或class的开始位置）
        next_method = content.find('\n    def ', calculate_method_start + 1, class_end)
        next_class = content.find('\nclass ', calculate_method_start + 1, class_end)
        
        if next_method != -1:
            insert_position = next_method
        elif next_class != -1:
            insert_position = next_class
        else:
            # 如果找不到下一个方法或类，直接在类结束前插入
            insert_position = class_end
    else:
        # 如果找不到calculate方法，直接在类结束前插入
        insert_position = class_end
    
    modified_content = content[:insert_position] + method_to_add + content[insert_position:]
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为ZXMAmplitudeElasticity类添加calculate_raw_score方法")

def add_elasticity_score_raw_score():
    """为score_indicators.py中的ZXMAmplitudeElasticity引用添加calculate_raw_score方法"""
    file_path = os.path.join(root_dir, 'indicators/zxm/score_indicators.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查文件中是否引用了ZXMAmplitudeElasticity
    import_pos = content.find('from indicators.zxm.elasticity_indicators import ZXMAmplitudeElasticity')
    if import_pos == -1:
        print("score_indicators.py中没有引用ZXMAmplitudeElasticity")
        return
    
    # 判断是否需要修复
    if 'class ZXMAmplitudeElasticity(' in content:
        print("score_indicators.py中重新定义了ZXMAmplitudeElasticity类，需要添加calculate_raw_score方法")
        
        # 找到ZXMAmplitudeElasticity类的位置
        class_start = content.find('class ZXMAmplitudeElasticity(')
        if class_start == -1:
            print("无法找到ZXMAmplitudeElasticity类的定义")
            return
        
        # 找到类的结束位置（下一个class的开始）
        next_class = content.find('\nclass ', class_start + 1)
        if next_class == -1:
            print("无法确定ZXMAmplitudeElasticity类的结束位置")
            return
        
        # 检查是否已经实现了该方法
        if 'def calculate_raw_score(' in content[class_start:next_class]:
            print("score_indicators.py中的ZXMAmplitudeElasticity类已经实现了calculate_raw_score方法")
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
        
        # 插入方法到类定义的适当位置
        # 找到calculate方法的结束位置
        calculate_method_start = content.find('def calculate(self', class_start, next_class)
        if calculate_method_start != -1:
            # 找到calculate方法的结束位置（下一个def或class的开始位置）
            next_method = content.find('\n    def ', calculate_method_start + 1, next_class)
            
            if next_method != -1:
                insert_position = next_method
            else:
                # 如果找不到下一个方法，直接在类结束前插入
                insert_position = next_class
        else:
            # 如果找不到calculate方法，直接在类结束前插入
            insert_position = next_class
        
        modified_content = content[:insert_position] + method_to_add + content[insert_position:]
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("已为score_indicators.py中的ZXMAmplitudeElasticity类添加calculate_raw_score方法")
    else:
        print("score_indicators.py中只引用了ZXMAmplitudeElasticity，不需要添加方法")

def main():
    """主函数"""
    print("开始为ZXMAmplitudeElasticity类添加calculate_raw_score方法...")
    
    # 添加实现
    add_amplitude_elasticity_raw_score()
    add_elasticity_score_raw_score()
    
    print("完成!")

if __name__ == "__main__":
    main() 