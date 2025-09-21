#!/bin/bash
# Cursor终端快速重置脚本

echo "🔧 Cursor终端快速重置"
echo "===================="

# 1. 杀死可能卡住的Python进程
echo "🔍 停止Python进程..."
pkill -f "python.*freedom.*monitoring" 2>/dev/null || true
pkill -f "python.*freedom.*monitor" 2>/dev/null || true
pkill -f "python.*freedom.*self_healing" 2>/dev/null || true
pkill -f "python.*freedom.*risk" 2>/dev/null || true

# 2. 清理环境变量
echo "🔄 清理环境变量..."
unset MONITORING_ACTIVE
unset SELF_HEALING_ACTIVE  
unset RISK_MONITOR_ACTIVE
unset PYTHONPATH

# 3. 清理Python缓存
echo "🧹 清理Python缓存..."
find /Users/hacker/PycharmProjects/freedom -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /Users/hacker/PycharmProjects/freedom -name "*.pyc" -delete 2>/dev/null || true

# 4. 重置终端
echo "🔄 重置终端..."
reset 2>/dev/null || true

# 5. 创建简单测试
echo "📝 创建测试文件..."
cat > /Users/hacker/PycharmProjects/freedom/simple_test.py << 'EOF'
#!/usr/bin/env python3
print("✅ Cursor终端工作正常!")
print("当前时间:", __import__('datetime').datetime.now())
print("Python版本:", __import__('sys').version.split()[0])
EOF

chmod +x /Users/hacker/PycharmProjects/freedom/simple_test.py

echo ""
echo "🎉 快速重置完成!"
echo ""
echo "📋 测试步骤:"
echo "1. 重启Cursor应用"
echo "2. 运行: python3 simple_test.py"
echo "3. 如果看到成功消息，终端已恢复正常"
echo ""
echo "💡 如果问题仍然存在，请运行: python3 fix_cursor_terminal.py"

