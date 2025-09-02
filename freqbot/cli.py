"""FreqBot CLI 命令行接口"""

import argparse
import sys
import json
import threading
import time
from pathlib import Path
from typing import Optional
import logging

from .config.manager import ConfigManager
from .strategies.registry import StrategyRegistry
from .strategies.loader import StrategyLoader
from .core.engine import TradingEngine
from .core.monitor import TradeMonitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreqBotCLI:
    """FreqBot 命令行接口"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.strategy_registry = StrategyRegistry()
        self.strategy_loader = StrategyLoader()
        self.trading_engine = TradingEngine()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            prog='freqbot',
            description='FreqBot - 统一量化交易平台',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # list-strategies 命令
        list_parser = subparsers.add_parser('list-strategies', help='列出所有可用策略')
        list_parser.add_argument('--category', help='按分类筛选')
        
        # list-envs 命令
        list_envs_parser = subparsers.add_parser('list-envs', help='列出所有环境')
        
        # run 命令
        run_parser = subparsers.add_parser('run', help='运行策略')
        run_parser.add_argument('--strategy', required=True, help='策略名称')
        run_parser.add_argument('--env', default='demo', help='环境名称 (默认: demo)')
        run_parser.add_argument('--dry-run', action='store_true', help='强制干跑模式')
        run_parser.add_argument('--live', action='store_true', help='强制实盘模式')
        run_parser.add_argument('--monitor', action='store_true', help='启用监控')
        
        # backtest 命令
        backtest_parser = subparsers.add_parser('backtest', help='回测策略')
        backtest_parser.add_argument('--strategy', required=True, help='策略名称')
        backtest_parser.add_argument('--env', default='demo', help='环境名称')
        backtest_parser.add_argument('--timerange', help='时间范围 (例: 20240101-20240201)')
        
        # monitor 命令
        monitor_parser = subparsers.add_parser('monitor', help='监控运行中的交易')
        monitor_parser.add_argument('--db', default='demo_trades.sqlite', help='数据库文件')
        monitor_parser.add_argument('--interval', type=int, default=15, help='更新间隔(秒)')
        
        # download-data 命令
        download_parser = subparsers.add_parser('download-data', help='下载历史数据')
        download_parser.add_argument('--env', default='demo', help='环境名称')
        download_parser.add_argument('--pairs', nargs='+', help='交易对列表')
        download_parser.add_argument('--timeframe', help='时间框架')
        download_parser.add_argument('--timerange', help='时间范围')
        
        # init-config 命令
        init_parser = subparsers.add_parser('init-config', help='初始化配置')
        init_parser.add_argument('--strategy', required=True, help='策略名称')
        init_parser.add_argument('--env', required=True, help='环境名称')
        init_parser.add_argument('--template', help='使用的模板名称')
        
        # migrate-config 命令
        migrate_parser = subparsers.add_parser('migrate-config', help='迁移旧配置文件')
        migrate_parser.add_argument('--file', required=True, help='旧配置文件路径')
        migrate_parser.add_argument('--env', required=True, help='目标环境名称')
        
        return parser
    
    def list_strategies(self, args):
        """列出策略"""
        print("🔍 发现策略中...")
        self.strategy_registry.discover_strategies()
        
        strategies = self.strategy_registry.list_strategies(args.category)
        
        if not strategies:
            print("❌ 未找到策略")
            return
        
        print(f"\\n📋 可用策略 ({len(strategies)}个):")
        print("=" * 80)
        
        current_category = None
        for strategy in strategies:
            if strategy.category != current_category:
                current_category = strategy.category
                print(f"\\n📁 {current_category.upper()}")
                print("-" * 40)
            
            print(f"  📊 {strategy.name}")
            if strategy.description:
                print(f"      {strategy.description}")
            if strategy.author:
                print(f"      作者: {strategy.author} | 版本: {strategy.version}")
            print(f"      文件: {strategy.file_path}")
    
    def list_environments(self, args):
        """列出环境"""
        envs = self.config_manager.list_environments()
        
        if not envs:
            print("❌ 未找到环境配置")
            return
        
        print(f"\\n🏗️ 可用环境 ({len(envs)}个):")
        print("=" * 40)
        
        for env in envs:
            print(f"  🌍 {env}")
            try:
                config = self.config_manager.get_environment_config(env)
                mode = "💰 实盘" if not config.get("dry_run", True) else "🎯 模拟"
                exchange = config.get("exchange", {}).get("name", "未知")
                print(f"      {mode} | 交易所: {exchange}")
            except Exception as e:
                print(f"      ❌ 配置错误: {e}")
    
    def run_strategy(self, args):
        """运行策略"""
        strategy_name = args.strategy
        env_name = args.env
        
        print(f"🚀 启动策略: {strategy_name} (环境: {env_name})")
        
        # 验证策略
        self.strategy_registry.discover_strategies()
        if not self.strategy_registry.validate_strategy(strategy_name):
            print(f"❌ 策略验证失败: {strategy_name}")
            return
        
        # 生成配置文件
        try:
            config = self.config_manager.create_freqtrade_config(strategy_name, env_name)
            
            # 处理运行模式覆盖
            if args.dry_run:
                config["dry_run"] = True
                print("🎯 强制启用干跑模式")
            elif args.live:
                config["dry_run"] = False
                print("💰 强制启用实盘模式")
            
            # 保存临时配置文件
            temp_config = Path(f"temp_{env_name}_{strategy_name}.json")
            with open(temp_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            print(f"📄 配置文件: {temp_config}")
            
            # 启动交易引擎
            dry_run_mode = config.get("dry_run", True)
            mode_text = "模拟交易" if dry_run_mode else "实盘交易"
            print(f"⚡ 启动{mode_text}...")
            
            def log_callback(line: str):
                # 过滤和格式化日志
                if any(keyword in line for keyword in ['BUY', 'SELL', 'Entry', 'Exit']):
                    print(f"🔔 {line}")
                elif 'ERROR' in line:
                    print(f"❌ {line}")
                elif 'WARNING' in line:
                    print(f"⚠️ {line}")
            
            success = self.trading_engine.start_trading(
                str(temp_config),
                dry_run=dry_run_mode,
                log_callback=log_callback
            )
            
            if not success:
                print("❌ 启动失败")
                return
            
            # 启动监控
            if args.monitor:
                monitor = TradeMonitor(config.get("db_url", "").replace("sqlite:///", ""))
                
                def print_stats_callback(stats):
                    monitor.print_stats(stats)
                
                monitor_thread = threading.Thread(
                    target=monitor.start_monitoring,
                    args=(print_stats_callback,),
                    daemon=True
                )
                monitor_thread.start()
                print("📊 已启动交易监控")
            
            print("\\n💡 运行说明:")
            print("- 按 Ctrl+C 停止运行")
            if args.monitor:
                print("- 监控信息会自动显示")
            print("")
            
            try:
                # 等待用户中断
                while self.trading_engine.is_running():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\\n🛑 用户中断")
            
            print("🔄 正在停止...")
            self.trading_engine.stop_trading()
            
            # 清理临时文件
            if temp_config.exists():
                temp_config.unlink()
            
            print("✅ 策略已停止")
            
        except Exception as e:
            logger.error(f"运行策略失败: {e}")
            print(f"❌ 运行失败: {e}")
    
    def backtest_strategy(self, args):
        """回测策略"""
        strategy_name = args.strategy
        env_name = args.env
        
        print(f"🔍 回测策略: {strategy_name} (环境: {env_name})")
        
        try:
            config = self.config_manager.create_freqtrade_config(strategy_name, env_name)
            
            # 保存临时配置文件
            temp_config = Path(f"backtest_{env_name}_{strategy_name}.json")
            with open(temp_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 运行回测
            result = self.trading_engine.run_backtest(
                str(temp_config),
                timerange=args.timerange,
                strategy_list=[strategy_name] if strategy_name else None
            )
            
            if result["success"]:
                print("✅ 回测完成")
                print(result["stdout"])
            else:
                print("❌ 回测失败")
                print(result["stderr"])
            
            # 清理临时文件
            if temp_config.exists():
                temp_config.unlink()
                
        except Exception as e:
            logger.error(f"回测失败: {e}")
            print(f"❌ 回测失败: {e}")
    
    def monitor_trades(self, args):
        """监控交易"""
        print(f"📊 开始监控交易 (数据库: {args.db})")
        print("=" * 60)
        print("按 Ctrl+C 停止监控\\n")
        
        monitor = TradeMonitor(args.db)
        monitor.update_interval = args.interval
        
        def print_callback(stats):
            monitor.print_stats(stats)
        
        try:
            monitor.start_monitoring(print_callback)
        except KeyboardInterrupt:
            print("\\n🛑 停止监控")
            monitor.stop_monitoring()
    
    def download_data(self, args):
        """下载数据"""
        env_name = args.env
        
        print(f"📥 下载数据 (环境: {env_name})")
        
        try:
            config = self.config_manager.get_environment_config(env_name)
            
            # 保存临时配置文件
            temp_config = Path(f"temp_download_{env_name}.json")
            with open(temp_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            success = self.trading_engine.download_data(
                str(temp_config),
                pairs=args.pairs,
                timeframe=args.timeframe,
                timerange=args.timerange
            )
            
            if success:
                print("✅ 数据下载成功")
            else:
                print("❌ 数据下载失败")
            
            # 清理临时文件
            if temp_config.exists():
                temp_config.unlink()
                
        except Exception as e:
            logger.error(f"下载数据失败: {e}")
            print(f"❌ 下载数据失败: {e}")
    
    def init_config(self, args):
        """初始化配置"""
        strategy_name = args.strategy
        env_name = args.env
        template = args.template or "demo_environment"
        
        print(f"⚙️ 初始化配置: {strategy_name} -> {env_name} (模板: {template})")
        
        # 这里可以实现配置模板的复制和自定义
        print("🚧 功能开发中...")
    
    def migrate_config(self, args):
        """迁移配置"""
        file_path = args.file
        env_name = args.env
        
        print(f"🔄 迁移配置: {file_path} -> {env_name}")
        
        try:
            strategy_name, migrated_env = self.config_manager.migrate_legacy_config(file_path, env_name)
            print(f"✅ 迁移完成: 策略 {strategy_name}, 环境 {migrated_env}")
        except Exception as e:
            logger.error(f"迁移配置失败: {e}")
            print(f"❌ 迁移失败: {e}")
    
    def run(self):
        """运行CLI"""
        parser = self.create_parser()
        
        if len(sys.argv) == 1:
            parser.print_help()
            return
        
        args = parser.parse_args()
        
        try:
            if args.command == 'list-strategies':
                self.list_strategies(args)
            elif args.command == 'list-envs':
                self.list_environments(args)
            elif args.command == 'run':
                self.run_strategy(args)
            elif args.command == 'backtest':
                self.backtest_strategy(args)
            elif args.command == 'monitor':
                self.monitor_trades(args)
            elif args.command == 'download-data':
                self.download_data(args)
            elif args.command == 'init-config':
                self.init_config(args)
            elif args.command == 'migrate-config':
                self.migrate_config(args)
            else:
                parser.print_help()
        
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            print(f"❌ 错误: {e}")

def main():
    """CLI 入口点"""
    cli = FreqBotCLI()
    cli.run()

if __name__ == "__main__":
    main()