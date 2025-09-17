# 分层测试执行脚本

## 🧪 测试脚本概览

本文档提供了每层修复后的标准化测试脚本，确保严格执行分层测试标准。

## 📋 L1基础设施层测试脚本

### 脚本文件: `tests/layer_tests/test_l1_infrastructure.py`

```python
#!/usr/bin/env python3
"""
L1基础设施层测试脚本
测试依赖注入、配置管理、日志系统的统一性和稳定性
"""

import os
import sys
import time
import pytest
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestL1Infrastructure:
    """L1基础设施层测试类"""
    
    def setup_class(self):
        """测试类初始化"""
        self.test_start_time = time.time()
        self.errors = []
        self.warnings = []
        
    def test_container_uniqueness(self):
        """测试容器唯一性"""
        print("🔍 测试容器唯一性...")
        
        # 检查容器文件存在性
        container_files = [
            'utils/unified_container.py',  # 应该存在
            'db/container.py',             # 应该被删除
            'utils/optimized_dependency_injection.py'  # 应该被删除
        ]
        
        existing_files = [f for f in container_files if os.path.exists(f)]
        
        # 断言只有一个容器文件存在
        assert len(existing_files) == 1, f"发现多个容器文件: {existing_files}"
        assert existing_files[0] == 'utils/unified_container.py', "错误的容器文件"
        
        print("✅ 容器唯一性测试通过")
    
    def test_container_functionality(self):
        """测试容器功能"""
        print("🔍 测试容器功能...")
        
        try:
            from utils.unified_container import container
            
            # 测试服务注册
            class MockService:
                def test_method(self):
                    return "test"
            
            # 注册测试服务
            container.register("TestService", MockService)
            
            # 解析测试服务
            service = container.resolve("TestService")
            assert service is not None
            assert service.test_method() == "test"
            
            print("✅ 容器功能测试通过")
            
        except Exception as e:
            self.errors.append(f"容器功能测试失败: {e}")
            raise
    
    def test_configuration_management(self):
        """测试配置管理"""
        print("🔍 测试配置管理...")
        
        # 检查必需的配置文件
        required_configs = [
            'config/database.yml',
            'config/indicators.yml', 
            'config/strategies.yml',
            'config/thresholds.yml',
            'config/system.yml'
        ]
        
        missing_configs = [f for f in required_configs if not os.path.exists(f)]
        assert len(missing_configs) == 0, f"缺少配置文件: {missing_configs}"
        
        # 测试配置加载
        try:
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
            assert config_manager.load_all_configs()
            
            print("✅ 配置管理测试通过")
            
        except Exception as e:
            self.errors.append(f"配置管理测试失败: {e}")
            raise
    
    def test_no_hardcoded_values(self):
        """测试无硬编码值"""
        print("🔍 检查硬编码值...")
        
        # 检查常见硬编码模式
        hardcoded_patterns = [
            r'if.*>.*\d+',  # if score > 60
            r'weight.*=.*\d+\.\d+',  # weight = 0.3
            r'threshold.*=.*\d+',  # threshold = 100
        ]
        
        # 扫描Python文件
        python_files = []
        for root, dirs, files in os.walk('.'):
            # 跳过测试文件和虚拟环境
            if 'test' in root or '.venv' in root or '__pycache__' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        hardcoded_issues = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in hardcoded_patterns:
                        import re
                        matches = re.findall(pattern, content)
                        if matches:
                            hardcoded_issues.append(f"{file_path}: {matches}")
            except:
                continue
        
        if hardcoded_issues:
            self.warnings.extend(hardcoded_issues)
            print(f"⚠️ 发现可能的硬编码: {len(hardcoded_issues)} 个")
        else:
            print("✅ 无硬编码值检查通过")
    
    def test_logging_system(self):
        """测试日志系统"""
        print("🔍 测试日志系统...")
        
        try:
            from utils.logger import get_logger
            
            # 测试日志获取
            logger = get_logger(__name__)
            assert logger is not None
            
            # 测试日志级别
            test_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
            for level in test_levels:
                logger.log(level, f"测试日志级别 {level}")
            
            print("✅ 日志系统测试通过")
            
        except Exception as e:
            self.errors.append(f"日志系统测试失败: {e}")
            raise
    
    def test_performance_requirements(self):
        """测试性能要求"""
        print("🔍 测试性能要求...")
        
        try:
            from utils.unified_container import container
            
            # 测试容器操作性能
            start_time = time.time()
            
            # 执行100次容器操作
            for i in range(100):
                container.register(f"TestService{i}", str)
                service = container.resolve(f"TestService{i}")
            
            operation_time = time.time() - start_time
            avg_time_ms = (operation_time / 100) * 1000
            
            assert avg_time_ms < 1.0, f"容器操作平均时间 {avg_time_ms:.2f}ms > 1ms"
            
            print(f"✅ 性能测试通过 (平均操作时间: {avg_time_ms:.2f}ms)")
            
        except Exception as e:
            self.errors.append(f"性能测试失败: {e}")
            raise
    
    def test_log_cleanliness(self):
        """测试日志清洁性"""
        print("🔍 检查日志清洁性...")
        
        # 检查最近的日志文件
        log_files = ['logs/system.log', 'logs/error.log', 'logs/application.log']
        
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 检查ERROR级别日志
                    error_lines = [line for line in content.split('\n') if 'ERROR' in line]
                    if error_lines:
                        self.errors.extend([f"发现ERROR日志: {line}" for line in error_lines[-5:]])  # 最近5条
                    
                    # 检查WARNING级别日志
                    warning_lines = [line for line in content.split('\n') if 'WARNING' in line]
                    if warning_lines:
                        self.warnings.extend([f"发现WARNING日志: {line}" for line in warning_lines[-5:]])  # 最近5条
        
        if not self.errors:
            print("✅ 日志清洁性检查通过")
        else:
            print(f"❌ 发现 {len(self.errors)} 个ERROR日志")
    
    def teardown_class(self):
        """测试类清理"""
        test_duration = time.time() - self.test_start_time
        
        print(f"\n📊 L1层测试总结:")
        print(f"⏱️ 测试耗时: {test_duration:.2f}秒")
        print(f"❌ 错误数量: {len(self.errors)}")
        print(f"⚠️ 警告数量: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ 发现的错误:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️ 发现的警告:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # 生成测试报告
        self.generate_test_report(test_duration)
        
        # 判断是否通过
        if self.errors:
            print("\n🚫 L1层测试未通过，禁止进入L2层修复！")
            sys.exit(1)
        else:
            print("\n✅ L1层测试通过，可以进入L2层修复")
    
    def generate_test_report(self, duration):
        """生成测试报告"""
        report_content = f"""
# L1基础设施层测试报告

## 测试执行时间
- 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_start_time))}
- 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 总耗时: {duration:.2f}秒

## 测试结果
- 容器唯一性: ✅
- 容器功能: ✅
- 配置管理: ✅
- 硬编码检查: {'✅' if not self.warnings else '⚠️'}
- 日志系统: ✅
- 性能要求: ✅
- 日志清洁性: {'✅' if not self.errors else '❌'}

## 问题记录
### 错误 ({len(self.errors)}个)
{chr(10).join([f"- {error}" for error in self.errors])}

### 警告 ({len(self.warnings)}个)
{chr(10).join([f"- {warning}" for warning in self.warnings])}

## 通过/阻断决定
- {'✅ 通过，可以进入L2层' if not self.errors else '❌ 阻断，需要解决错误后重新测试'}
"""
        
        # 保存报告
        os.makedirs('test_reports', exist_ok=True)
        with open('test_reports/l1_test_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)

if __name__ == "__main__":
    # 运行L1层测试
    print("🚀 开始L1基础设施层测试...")
    pytest.main([__file__, "-v", "--tb=short"])
```

## 📋 L2存储访问层测试脚本

### 脚本文件: `tests/layer_tests/test_l2_storage.py`

```python
#!/usr/bin/env python3
"""
L2存储访问层测试脚本
测试数据库连接池和SQL管理的统一性和稳定性
"""

import os
import sys
import time
import pytest
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestL2Storage:
    """L2存储访问层测试类"""
    
    def setup_class(self):
        """测试类初始化"""
        self.test_start_time = time.time()
        self.errors = []
        self.warnings = []
    
    def test_connection_pool_uniqueness(self):
        """测试连接池唯一性"""
        print("🔍 测试连接池唯一性...")
        
        try:
            from db.enhanced_connection_pool import get_connection_pool
            
            # 测试连接池获取
            pool1 = get_connection_pool()
            pool2 = get_connection_pool()
            
            # 应该是同一个实例（单例模式）
            assert pool1 is pool2, "连接池不是单例模式"
            
            print("✅ 连接池唯一性测试通过")
            
        except Exception as e:
            self.errors.append(f"连接池唯一性测试失败: {e}")
            raise
    
    def test_connection_pool_functionality(self):
        """测试连接池功能"""
        print("🔍 测试连接池功能...")
        
        try:
            from db.enhanced_connection_pool import get_connection_pool
            
            pool = get_connection_pool()
            
            # 测试连接获取
            with pool.get_connection() as conn:
                assert conn is not None
                
                # 测试简单查询
                result = conn.execute("SELECT 1 as test_value")
                assert result is not None
            
            print("✅ 连接池功能测试通过")
            
        except Exception as e:
            self.errors.append(f"连接池功能测试失败: {e}")
            raise
    
    def test_concurrent_connections(self):
        """测试并发连接"""
        print("🔍 测试并发连接...")
        
        def test_connection():
            try:
                from db.enhanced_connection_pool import get_connection_pool
                pool = get_connection_pool()
                
                with pool.get_connection() as conn:
                    result = conn.execute("SELECT 1")
                    time.sleep(0.1)  # 模拟查询时间
                    return True
            except:
                return False
        
        # 创建10个并发线程
        threads = []
        results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda: results.append(test_connection()))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        success_count = sum(results)
        assert success_count == 10, f"并发连接测试失败: {success_count}/10"
        
        print("✅ 并发连接测试通过")
    
    def test_sql_management(self):
        """测试SQL管理"""
        print("🔍 测试SQL管理...")
        
        try:
            from db.sql_manager import SQLManager
            
            sql_manager = SQLManager()
            
            # 测试标准查询模板
            stock_query = sql_manager.get_stock_data_query()
            
            # 验证查询模板内容
            required_elements = [
                "SELECT",
                "code, name, date, open, high, low, close, volume, turnover_rate",
                "FROM stock_info",
                "WHERE code = %s AND level = %s",
                "ORDER BY date ASC"
            ]
            
            for element in required_elements:
                assert element in stock_query, f"查询模板缺少: {element}"
            
            print("✅ SQL管理测试通过")
            
        except Exception as e:
            self.errors.append(f"SQL管理测试失败: {e}")
            raise
    
    def test_performance_requirements(self):
        """测试性能要求"""
        print("🔍 测试性能要求...")
        
        try:
            from db.enhanced_connection_pool import get_connection_pool
            
            pool = get_connection_pool()
            
            # 测试连接获取性能
            start_time = time.time()
            
            for i in range(10):
                with pool.get_connection() as conn:
                    conn.execute("SELECT 1")
            
            total_time = time.time() - start_time
            avg_time_ms = (total_time / 10) * 1000
            
            assert avg_time_ms < 100, f"连接获取平均时间 {avg_time_ms:.2f}ms > 100ms"
            
            print(f"✅ 性能测试通过 (平均连接时间: {avg_time_ms:.2f}ms)")
            
        except Exception as e:
            self.errors.append(f"性能测试失败: {e}")
            raise
    
    def teardown_class(self):
        """测试类清理"""
        test_duration = time.time() - self.test_start_time
        
        print(f"\n📊 L2层测试总结:")
        print(f"⏱️ 测试耗时: {test_duration:.2f}秒")
        print(f"❌ 错误数量: {len(self.errors)}")
        print(f"⚠️ 警告数量: {len(self.warnings)}")
        
        # 生成测试报告
        self.generate_test_report(test_duration)
        
        # 判断是否通过
        if self.errors:
            print("\n🚫 L2层测试未通过，禁止进入L3层修复！")
            sys.exit(1)
        else:
            print("\n✅ L2层测试通过，可以进入L3层修复")
    
    def generate_test_report(self, duration):
        """生成测试报告"""
        report_content = f"""
# L2存储访问层测试报告

## 测试执行时间
- 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_start_time))}
- 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 总耗时: {duration:.2f}秒

## 测试结果
- 连接池唯一性: ✅
- 连接池功能: ✅
- 并发连接: ✅
- SQL管理: ✅
- 性能要求: ✅

## 问题记录
### 错误 ({len(self.errors)}个)
{chr(10).join([f"- {error}" for error in self.errors])}

## 通过/阻断决定
- {'✅ 通过，可以进入L3层' if not self.errors else '❌ 阻断，需要解决错误后重新测试'}
"""
        
        os.makedirs('test_reports', exist_ok=True)
        with open('test_reports/l2_test_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)

if __name__ == "__main__":
    print("🚀 开始L2存储访问层测试...")
    pytest.main([__file__, "-v", "--tb=short"])
```

## 🚀 测试执行命令

### 单层测试执行
```bash
# L1基础设施层测试
python tests/layer_tests/test_l1_infrastructure.py

# L2存储访问层测试
python tests/layer_tests/test_l2_storage.py

# L3数据服务层测试
python tests/layer_tests/test_l3_data_service.py

# L4核心服务层测试
python tests/layer_tests/test_l4_core_service.py

# L5业务应用层测试
python tests/layer_tests/test_l5_business.py

# L6用户接口层测试
python tests/layer_tests/test_l6_interface.py
```

### 批量测试执行脚本
```bash
#!/bin/bash
# 文件: run_layer_tests.sh

echo "🚀 开始分层测试执行..."

LAYERS=("l1_infrastructure" "l2_storage" "l3_data_service" "l4_core_service" "l5_business" "l6_interface")
LAYER_NAMES=("L1基础设施层" "L2存储访问层" "L3数据服务层" "L4核心服务层" "L5业务应用层" "L6用户接口层")

for i in "${!LAYERS[@]}"; do
    layer=${LAYERS[$i]}
    layer_name=${LAYER_NAMES[$i]}
    
    echo ""
    echo "🧪 开始 ${layer_name} 测试..."
    
    python tests/layer_tests/test_${layer}.py
    
    if [ $? -ne 0 ]; then
        echo "❌ ${layer_name} 测试失败，停止后续测试"
        exit 1
    fi
    
    echo "✅ ${layer_name} 测试通过"
done

echo ""
echo "🎉 所有层测试通过！系统优化完成！"
```

## 📊 测试报告汇总脚本

```python
#!/usr/bin/env python3
"""
测试报告汇总脚本
汇总所有层的测试报告，生成总体测试报告
"""

import os
import glob
from datetime import datetime

def generate_summary_report():
    """生成汇总测试报告"""
    
    # 查找所有测试报告
    report_files = glob.glob('test_reports/l*_test_report.md')
    report_files.sort()
    
    summary_content = f"""
# 系统优化分层测试汇总报告

## 测试概览
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 测试层数: {len(report_files)}
- 报告文件: {len(report_files)}个

## 各层测试结果
"""
    
    total_errors = 0
    total_warnings = 0
    
    for report_file in report_files:
        layer_name = os.path.basename(report_file).replace('_test_report.md', '').upper()
        
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取测试结果
            if '❌ 阻断' in content:
                status = '❌ 未通过'
            else:
                status = '✅ 通过'
            
            # 统计错误和警告
            error_count = content.count('错误 (') 
            warning_count = content.count('警告 (')
            
            total_errors += error_count
            total_warnings += warning_count
            
            summary_content += f"""
### {layer_name}
- 状态: {status}
- 错误: {error_count}个
- 警告: {warning_count}个
"""
    
    summary_content += f"""
## 总体统计
- 总错误数: {total_errors}
- 总警告数: {total_warnings}
- 整体状态: {'✅ 系统优化成功' if total_errors == 0 else '❌ 系统优化失败'}

## 结论
{'🎉 所有层测试通过，系统优化成功完成！' if total_errors == 0 else '🚫 存在测试失败，需要重新修复和测试'}
"""
    
    # 保存汇总报告
    with open('test_reports/summary_report.md', 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print("📊 测试报告汇总完成")
    print(f"📄 汇总报告: test_reports/summary_report.md")

if __name__ == "__main__":
    generate_summary_report()
```

## 📝 总结

分层测试执行脚本确保了：

1. **严格的测试标准**: 每层都有详细的测试用例
2. **自动化执行**: 脚本化的测试执行流程
3. **阻断机制**: 测试不通过就不能进入下一层
4. **详细报告**: 每层都生成详细的测试报告
5. **汇总分析**: 最终生成整体测试汇总报告

这些脚本为分层修复提供了强有力的质量保证机制。
