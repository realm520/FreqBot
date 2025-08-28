#!/usr/bin/env python3
"""
FreqBot做市商策略启动脚本
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path


def check_freqtrade_installation():
    """检查FreqTrade安装"""
    try:
        result = subprocess.run(['uv', 'run', 'freqtrade', '--version'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ FreqTrade已安装: {result.stdout.strip().split('\\n')[-1]}")
            return True
        else:
            print("❌ FreqTrade未正确安装")
            return False
    except FileNotFoundError:
        print("❌ 未找到uv命令")
        return False


def install_frequi():
    """安装FreqUI"""
    print("🔧 安装FreqUI...")
    try:
        result = subprocess.run(['uv', 'run', 'freqtrade', 'install-ui'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FreqUI安装成功")
            return True
        else:
            print(f"❌ FreqUI安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ FreqUI安装错误: {e}")
        return False


def start_freqtrade(config_file: str, strategy: str, mode: str = "trade"):
    """启动FreqTrade"""
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"🚀 启动FreqTrade做市商策略...")
    print(f"   配置文件: {config_file}")
    print(f"   策略: {strategy}")
    print(f"   模式: {mode}")
    
    cmd = [
        'uv', 'run', 'freqtrade', mode,
        '--config', config_file,
        '--strategy', strategy
    ]
    
    if mode == "webserver":
        cmd.append('--enable-webserver')
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 启动FreqTrade
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FreqBot做市商策略启动器')
    parser.add_argument('--config', default='user_data/market_maker_config.json',
                       help='配置文件路径')
    parser.add_argument('--strategy', default='FreqTradeMarketMaker',
                       help='策略名称')
    parser.add_argument('--mode', choices=['trade', 'webserver'], default='webserver',
                       help='运行模式')
    parser.add_argument('--install-ui', action='store_true',
                       help='安装FreqUI')
    
    args = parser.parse_args()
    
    print("🤖 FreqBot做市商策略启动器")
    print("="*50)
    
    # 检查FreqTrade安装
    if not check_freqtrade_installation():
        print("请先安装FreqTrade: pip install freqtrade")
        sys.exit(1)
    
    # 安装FreqUI（如果需要）
    if args.install_ui:
        if not install_frequi():
            print("FreqUI安装失败，但可以继续运行")
    
    # 启动FreqTrade
    process = start_freqtrade(args.config, args.strategy, args.mode)
    
    if process:
        print("✅ FreqTrade已启动")
        print(f"🌐 WebUI地址: http://127.0.0.1:8080")
        print(f"👤 用户名: market_maker")
        print(f"🔑 密码: mm123456")
        print("")
        print("💡 提示:")
        print("   - 使用 Ctrl+C 停止")
        print("   - 使用监控工具: python market_maker_monitor.py")
        print("   - 查看日志了解详细运行状态")
        
        try:
            # 等待进程结束
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号，正在关闭...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            print("✅ 已停止")
    else:
        print("❌ 启动失败")
        sys.exit(1)


if __name__ == "__main__":
    main()