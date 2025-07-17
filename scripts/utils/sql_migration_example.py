#!/usr/bin/env python3
"""
SQL迁移示例

演示如何将直接的SQL查询迁移到统一的SQL管理系统。
这个示例展示了迁移的具体步骤和最佳实践。
"""

import os
import sys

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

def example_before_migration():
    """迁移前的代码示例（不要运行，仅作参考）"""
    print("=== 迁移前的代码示例 ===")
    
    # 这是迁移前的典型代码模式
    code_before = '''
    # 直接SQL查询 - 迁移前
    with data_manager.connection_pool.get_connection() as conn:
        daily_query = f"""
        SELECT date, open, high, low, close, volume, turnover_rate
        FROM stock_info 
        WHERE code = '{stock_code}'
        AND date BETWEEN '{start_date}' AND '{end_date}'
        AND level = '日线'
        ORDER BY date
        """
        daily_df = conn.query_dataframe(daily_query)
    '''
    
    print("迁移前的问题：")
    print("1. SQL语句分散在业务代码中")
    print("2. 硬编码的表名和字段名")
    print("3. 没有参数验证")
    print("4. 没有统一的错误处理")
    print("5. 难以维护和优化")
    print()

def example_after_migration():
    """迁移后的代码示例"""
    print("=== 迁移后的代码示例 ===")
    
    try:
        from db.query_executor import get_query_executor
        from db.sql_manager import QueryType
        
        # 获取查询执行器
        query_executor = get_query_executor()
        
        # 示例1: 获取股票数据
        print("1. 股票数据查询迁移")
        stock_code = "603359"
        start_date = "2025-05-01"
        end_date = "2025-05-15"
        
        # 迁移后的代码 - 使用统一接口
        stock_data = query_executor.get_stock_data(
            code=stock_code,
            start_date=start_date,
            end_date=end_date,
            level='日线'
        )
        print(f"✅ 获取股票数据成功: {len(stock_data)} 条记录")
        
        # 示例2: 批量股票数据查询
        print("\n2. 批量股票数据查询迁移")
        stock_codes = ["603359", "000001", "000002"]
        
        batch_data = query_executor.get_batch_stock_data(
            codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            level='日线'
        )
        print(f"✅ 获取批量股票数据成功: {len(batch_data)} 条记录")
        
        # 示例3: 股票列表查询
        print("\n3. 股票列表查询迁移")
        stock_list = query_executor.get_stock_list(level='日线')
        print(f"✅ 获取股票列表成功: {len(stock_list)} 只股票")
        
        # 示例4: 股票计数查询
        print("\n4. 股票计数查询迁移")
        stock_count = query_executor.get_stock_count(level='日线')
        print(f"✅ 获取股票总数成功: {stock_count} 只")
        
        # 示例5: 自定义查询（对于复杂查询）
        print("\n5. 自定义查询迁移")
        custom_params = {
            'code': stock_code,
            'start_date': start_date,
            'end_date': end_date,
            'level': '日线',
            'min_volume': 1000000
        }
        
        custom_data = query_executor.execute_query(
            QueryType.STOCK_DATA, 
            custom_params
        )
        print(f"✅ 执行自定义查询成功: {len(custom_data)} 条记录")
        
    except Exception as e:
        print(f"❌ 查询执行失败: {e}")
        print("可能的原因：")
        print("1. 数据库连接问题")
        print("2. SQL管理器未正确初始化")
        print("3. 参数验证失败")
    
    print("\n迁移后的优势：")
    print("1. 统一的查询接口")
    print("2. 自动参数验证")
    print("3. 统一的错误处理")
    print("4. 查询模板化管理")
    print("5. 更好的可维护性")

def show_migration_steps():
    """展示具体的迁移步骤"""
    print("=== 具体迁移步骤 ===")
    
    steps = [
        "1. 识别原始SQL查询",
        "2. 分析查询类型和参数",
        "3. 选择合适的QueryType",
        "4. 替换为query_executor调用",
        "5. 更新错误处理逻辑",
        "6. 测试验证迁移效果"
    ]
    
    for step in steps:
        print(step)
    
    print("\n迁移模板：")
    template = '''
# 迁移模板
# 原始代码：
# query = "SELECT * FROM stock_info WHERE code = '%s'" % code
# result = conn.execute(query)

# 迁移后：
from db.query_executor import get_query_executor
from db.sql_manager import QueryType

query_executor = get_query_executor()
result = query_executor.execute_query(
    QueryType.STOCK_DATA,
    {'code': code, 'start_date': start_date, 'end_date': end_date, 'level': '日线'}
)
'''
    print(template)

def create_migration_checklist():
    """创建迁移检查清单"""
    print("=== 迁移检查清单 ===")
    
    checklist = [
        "□ 确认SQL查询类型（SELECT/INSERT/UPDATE/DELETE）",
        "□ 识别查询参数和表名",
        "□ 选择合适的QueryType枚举值",
        "□ 验证参数格式和类型",
        "□ 替换直接SQL为query_executor调用",
        "□ 更新异常处理逻辑",
        "□ 添加必要的日志记录",
        "□ 运行单元测试验证功能",
        "□ 性能测试确保无回归",
        "□ 更新相关文档"
    ]
    
    for item in checklist:
        print(item)

def demonstrate_error_handling():
    """演示错误处理的改进"""
    print("=== 错误处理改进 ===")
    
    print("迁移前的错误处理：")
    before_error = '''
try:
    query = f"SELECT * FROM stock_info WHERE code = '{code}'"
    result = conn.execute(query)
except Exception as e:
    print(f"查询失败: {e}")  # 错误信息不够详细
'''
    print(before_error)
    
    print("迁移后的错误处理：")
    after_error = '''
try:
    result = query_executor.get_stock_data(code=code, ...)
except ParameterValidationError as e:
    logger.error(f"参数验证失败: {e}")
except DatabaseConnectionError as e:
    logger.error(f"数据库连接失败: {e}")
except QueryExecutionError as e:
    logger.error(f"查询执行失败: {e}")
except Exception as e:
    logger.error(f"未知错误: {e}")
'''
    print(after_error)

def main_sql_migration_example():
    """主函数"""
    print("SQL迁移示例和指南")
    print("=" * 50)
    
    # 显示迁移前后的对比
    example_before_migration()
    print()
    example_after_migration()
    print()
    
    # 展示迁移步骤
    show_migration_steps()
    print()
    
    # 创建检查清单
    create_migration_checklist()
    print()
    
    # 演示错误处理改进
    demonstrate_error_handling()
    print()
    
    print("迁移总结：")
    print("1. 统一SQL管理提高了代码质量")
    print("2. 标准化接口降低了维护成本")
    print("3. 参数验证减少了运行时错误")
    print("4. 模板化查询提高了开发效率")
    print("5. 集中管理便于性能优化")

if __name__ == "__main__":
    main_sql_migration_example() 