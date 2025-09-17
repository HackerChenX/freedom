#!/usr/bin/python
# -*- coding: UTF-8 -*-

from config.unified_config_manager import get_config
"""
数据库配置管理工具

提供命令行界面来管理Click_house数据库配置，包括：
1. 查看当前配置
2. 设置密码
3. 测试连接
4. 配置验证
5. 配置迁移
"""

import sys
import os
import argparse
import getpass
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database_config_manager import Database_config_manager, get_database_config_manager

def show_config_Config(args):
    """显示当前配置"""
    manager = get_database_config_manager()
    config = manager.get_config()
    
    print("📋 当前ClickHouse数据库配置:")
    print("=" * 50)
    
    # 隐藏密码显示
    display_config = config.copy()
    if display_config.get('password'):
        display_config['password'] = '*' * len(display_config['password'])
    
    for key, value in display_config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 50)
    
    # 显示配置来源
    print("\n🔍 配置来源优先级:")
    print("1. 环境变量 (最高优先级)")
    print("2. database.yaml 文件")
    print("3. user_config.json 文件")
    print("4. 默认配置 (最低优先级)")

def set_password_Config(args):
    """设置数据库密码"""
    manager = get_database_config_manager()
    
    if args.password:
        password = args.password
    else:
        password = getpass.getpass("请输入ClickHouse数据库密码: ")
    
    if not password:
        print("❌ 密码不能为空")
        return
    
    # 设置密码
    encrypt = not args.no_encrypt
    save_to_file = not args.no_save
    
    manager.set_password_Config(password, encrypt=encrypt, save_to_file=save_to_file)
    
    if save_to_file:
        encrypt_status = "加密" if encrypt else "明文"
        print(f"✅ 密码已设置并以{encrypt_status}形式保存到配置文件")
    else:
        print("✅ 密码已设置（仅在内存中，未保存到文件）")

def test_connection_Config(args):
    """测试数据库连接"""
    manager = get_database_config_manager()
    
    print("🔍 正在测试数据库连接...")
    
    # 验证配置
    if not manager.validate_config_Config():
        print("❌ 配置验证失败")
        return
    
    # 检查密码
    config = manager.get_config()
    if not config.get('password'):
        print("⚠️  未设置密码，尝试交互式输入...")
        password = manager.request_password_interactive()
        if not password:
            print("❌ 密码为空，无法连接")
            return
    
    # 测试连接
    if manager.test_connection_Config():
        print("✅ 数据库连接测试成功！")
        
        # 显示连接信息
        conn_config = manager.get_connection_config()
        print(f"📡 连接信息: {conn_config['user']}@{conn_config['host']}:{conn_config['port']}/{conn_config['database']}")
    else:
        print("❌ 数据库连接测试失败")
        print("请检查:")
        print("  1. ClickHouse服务是否运行")
        print("  2. 连接参数是否正确")
        print("  3. 用户名和密码是否正确")
        print("  4. 网络连接是否正常")

def validate_config_Config(args):
    """验证配置"""
    manager = get_database_config_manager()
    
    print("🔍 正在验证配置...")
    
    if manager.validate_config_Config():
        print("✅ 配置验证通过")
        
        # 显示配置摘要
        config = manager.get_config()
        print(f"📡 配置摘要: {config['user']}@{config['host']}:{config['port']}/{config['database']}")
        
        # 检查密码
        if config.get('password'):
            print("🔑 密码: 已设置")
        else:
            print("⚠️  密码: 未设置")
    else:
        print("❌ 配置验证失败")

def migrate_config(args):
    """迁移旧配置"""
    print("🔄 正在迁移配置...")
    
    manager = get_database_config_manager()
    
    # 检查是否有旧的硬编码配置需要迁移
    old_configs = []
    
    # 检查 data_access.py 中的硬编码配置
    clickhouse_db_file = project_root / 'db' / 'data_access.py'
    if clickhouse_db_file.exists():
        with open(clickhouse_db_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "get_config('database.password')" in content:
                old_configs.append("db/data_access.py 中发现硬编码密码 '123456'")
    
    # 检查测试脚本中的硬编码配置
    test_script_file = project_root / 'scripts' / 'simple_clickhouse_test.py'
    if test_script_file.exists():
        with open(test_script_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "get_config('database.password')" in content:
                old_configs.append("scripts/simple_clickhouse_test.py 中发现硬编码密码 '123456'")
    
    if old_configs:
        print("⚠️  发现以下硬编码配置:")
        for config in old_configs:
            print(f"  - {config}")
        
        print("\n建议:")
        print("1. 使用环境变量 CLICKHOUSE_PASSWORD 设置密码")
        print("2. 或使用此工具设置密码: python scripts/manage_db_config.py set-password")
        print("3. 移除代码中的硬编码密码")
    else:
        print("✅ 未发现硬编码配置")
    
    # 重新加载配置
    manager.reload_config()
    print("✅ 配置迁移完成")

def show_env_vars(args):
    """显示环境变量配置"""
    print("🌍 ClickHouse相关环境变量:")
    print("=" * 50)
    
    env_vars = [
        'CLICKHOUSE_HOST',
        'CLICKHOUSE_PORT', 
        'CLICKHOUSE_DATABASE',
        'CLICKHOUSE_USER',
        'CLICKHOUSE_PASSWORD',
        'CLICKHOUSE_TIMEOUT',
        'CLICKHOUSE_ENCRYPTION_KEY'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # 隐藏密码和密钥
            if 'PASSWORD' in var or 'KEY' in var:
                display_value = '*' * len(value)
            else:
                display_value = value
            print(f"{var}: {display_value}")
        else:
            print(f"{var}: (未设置)")
    
    print("=" * 50)
    print("\n💡 提示:")
    print("可以在 config/.env 文件中设置这些环境变量")
    print("或者在系统环境中直接设置")

def main_managedbconfig():
    """主函数"""
    parser = argparse.Argument_parser(
        description="ClickHouse数据库配置管理工具",
        formatter_class=argparse.Raw_description_help_formatter,
        epilog="""
示例用法:
  python scripts/manage_db_config.py show              # 显示当前配置
  python scripts/manage_db_config.py set-password     # 交互式设置密码
  python scripts/manage_db_config.py test             # 测试数据库连接
  python scripts/manage_db_config.py validate         # 验证配置
  python scripts/manage_db_config.py env              # 显示环境变量
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # show 命令
    show_parser = subparsers.add_parser('show', help='显示当前配置')
    show_parser.set_defaults(func=show_config)
    
    # set-password 命令
    password_parser = subparsers.add_parser('set-password', help='设置数据库密码')
    password_parser.add_argument('--password', help='密码（不建议在命令行中使用）')
    password_parser.add_argument('--no-encrypt', action='store_true', help='不加密存储密码')
    password_parser.add_argument('--no-save', action='store_true', help='不保存到配置文件')
    password_parser.set_defaults(func=set_password)
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='测试数据库连接')
    test_parser.set_defaults(func=test_connection)
    
    # validate 命令
    validate_parser = subparsers.add_parser('validate', help='验证配置')
    validate_parser.set_defaults(func=validate_config)
    
    # migrate 命令
    migrate_parser = subparsers.add_parser('migrate', help='迁移旧配置')
    migrate_parser.set_defaults(func=migrate_config)
    
    # env 命令
    env_parser = subparsers.add_parser('env', help='显示环境变量配置')
    env_parser.set_defaults(func=show_env_vars)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Keyboard_interrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        if args.command == 'test':
            print("请检查ClickHouse服务是否运行，以及配置是否正确")

if __name__ == '__main__':
    main_managedbconfig()
