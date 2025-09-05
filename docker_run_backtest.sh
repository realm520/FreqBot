#!/bin/bash

# FreqTrade Docker回测脚本
# 用于运行VATSMStrategy策略的回测

echo "========================================="
echo "FreqTrade Docker 回测工具"
echo "========================================="

# 默认参数
STRATEGY="VATSMStrategy"
TIMEFRAME="15m"
TIMERANGE="20240101-20240826"
BREAKDOWN="day"
EXPORT_TRADES="trades"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --timeframe)
            TIMEFRAME="$2"
            shift 2
            ;;
        --timerange)
            TIMERANGE="$2"
            shift 2
            ;;
        --breakdown)
            BREAKDOWN="$2"
            shift 2
            ;;
        --no-export)
            EXPORT_TRADES=""
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --strategy   策略名称 (默认: VATSMStrategy)"
            echo "  --timeframe  时间框架 (默认: 15m)"
            echo "  --timerange  时间范围 (默认: 20240101-20240826)"
            echo "  --breakdown  统计周期 (默认: day, 可选: week, month)"
            echo "  --no-export  不导出交易记录"
            echo "  --help       显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --strategy VATSMStrategy --timeframe 1h --timerange 20230101-20231231"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "回测参数:"
echo "  策略: $STRATEGY"
echo "  时间框架: $TIMEFRAME"
echo "  时间范围: $TIMERANGE"
echo "  统计周期: $BREAKDOWN"
if [ -n "$EXPORT_TRADES" ]; then
    echo "  导出交易: 是"
else
    echo "  导出交易: 否"
fi
echo ""

# 创建回测结果目录
mkdir -p ./user_data/backtest_results

# 构建回测命令
BACKTEST_CMD="backtesting \
    --config /freqtrade/user_data/config_docker.json \
    --strategy $STRATEGY \
    --timeframe $TIMEFRAME \
    --timerange $TIMERANGE \
    --breakdown $BREAKDOWN"

# 添加导出选项
if [ -n "$EXPORT_TRADES" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPORT_FILENAME="backtest_results/${STRATEGY}_${TIMEFRAME}_${TIMESTAMP}.json"
    BACKTEST_CMD="$BACKTEST_CMD \
        --export $EXPORT_TRADES \
        --export-filename $EXPORT_FILENAME"
fi

# 运行回测
echo "开始运行回测..."
docker run --rm \
    -v "./user_data:/freqtrade/user_data" \
    freqtradeorg/freqtrade:stable \
    $BACKTEST_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 回测完成!"
    if [ -n "$EXPORT_TRADES" ]; then
        echo "回测结果保存在: ./user_data/$EXPORT_FILENAME"
    fi
    echo ""
    echo "提示: 运行以下命令启动WebUI查看结果:"
    echo "  docker-compose up -d"
    echo "  然后访问: http://localhost:8080"
    echo "  用户名: freqtrade"
    echo "  密码: freqtrade123"
else
    echo ""
    echo "❌ 回测失败!"
    exit 1
fi