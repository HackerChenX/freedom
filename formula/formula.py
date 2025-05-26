"""
为了向后兼容，从formula模块导入StockFormula并重命名为Formula
"""

from formula.stock_formula import StockFormula as Formula, StockData, IndustryData, 主线, 吸筹板块

# 重新导出所有类和函数，以便保持向后兼容性
__all__ = ['Formula', 'StockData', 'IndustryData', '主线', '吸筹板块'] 