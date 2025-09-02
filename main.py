#!/usr/bin/env python3
"""
FreqBot - 统一量化交易平台主入口

使用方法:
  python main.py list-strategies    # 列出所有策略
  python main.py run --strategy AdvancedMarketMakerV2 --env demo
  python main.py monitor
"""

from freqbot.cli import main

if __name__ == "__main__":
    main()
