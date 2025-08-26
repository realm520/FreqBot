#!/bin/bash
# FreqBot 模拟交易启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    FreqBot 网格交易机器人启动${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查配置文件
if [ ! -f "dry_run_config.json" ]; then
    echo -e "${RED}错误: 找不到配置文件 dry_run_config.json${NC}"
    exit 1
fi

# 检查策略文件
if [ ! -f "strategies/GridTradingStrategy.py" ]; then
    echo -e "${RED}错误: 找不到策略文件 GridTradingStrategy.py${NC}"
    exit 1
fi

echo -e "${YELLOW}配置检查:${NC}"
echo -e "  ✓ 配置文件: dry_run_config.json"
echo -e "  ✓ 策略文件: GridTradingStrategy.py"
echo -e "  ✓ 交易模式: 模拟交易 (dry_run)"
echo

# 显示重要信息
echo -e "${YELLOW}重要信息:${NC}"
echo -e "  • 这是模拟交易，不会使用真实资金"
echo -e "  • Web界面: http://localhost:8080"
echo -e "  • 用户名: freqbot, 密码: freqbot123"
echo -e "  • 按 Ctrl+C 停止交易机器人"
echo

# 询问是否继续
read -p "$(echo -e ${YELLOW}是否启动FreqBot模拟交易? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}操作已取消${NC}"
    exit 0
fi

echo -e "${GREEN}启动FreqBot...${NC}"
echo

# 启动交易机器人
uv run freqtrade trade \
    --config dry_run_config.json \
    --userdir . \
    --strategy GridTradingStrategy \
    --logfile logs/freqbot.log