#!/bin/bash

# FreqTrade Docker数据下载脚本
# 用于下载历史数据供回测使用

echo "========================================="
echo "FreqTrade Docker 数据下载工具"
echo "========================================="

# 默认参数
PAIRS="ETH/USDT BTC/USDT"
TIMEFRAME="15m"
TIMERANGE="20240101-20240826"
EXCHANGE="binance"
DOWNLOAD_FUNDING=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --pairs)
            PAIRS="$2"
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
        --exchange)
            EXCHANGE="$2"
            shift 2
            ;;
        --download-funding)
            DOWNLOAD_FUNDING=true
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --pairs             交易对列表 (默认: 'ETH/USDT BTC/USDT')"
            echo "  --timeframe         时间框架 (默认: 15m，仅K线数据)"
            echo "  --timerange         时间范围 (默认: 20240101-20240826)"
            echo "  --exchange          交易所 (默认: binance)"
            echo "  --download-funding  下载资金费率历史数据（而非K线）"
            echo "  --help              显示帮助信息"
            echo ""
            echo "示例:"
            echo "  下载K线数据:"
            echo "    $0 --pairs 'ETH/USDT' --timeframe 1h --timerange 20230101-20231231"
            echo ""
            echo "  下载资金费率历史:"
            echo "    $0 --pairs 'BTC/USDT:USDT ETH/USDT:USDT' --timerange 20240101-20241003 --download-funding"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "下载参数:"
echo "  交易对: $PAIRS"
if [ "$DOWNLOAD_FUNDING" = false ]; then
    echo "  时间框架: $TIMEFRAME"
fi
echo "  时间范围: $TIMERANGE"
echo "  交易所: $EXCHANGE"
echo "  数据类型: $([ "$DOWNLOAD_FUNDING" = true ] && echo "资金费率" || echo "K线数据")"
echo ""

# 使用docker运行下载任务
if [ "$DOWNLOAD_FUNDING" = true ]; then
    echo "开始下载资金费率历史数据..."
    docker run --rm \
        -v "./user_data:/freqtrade/user_data" \
        freqtradeorg/freqtrade:stable \
        download-data \
        --config /freqtrade/user_data/config_docker.json \
        --pairs $PAIRS \
        --timerange $TIMERANGE \
        --exchange $EXCHANGE \
        --trading-mode futures \
        --candle-types funding_rate \
        --data-format json

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 资金费率数据下载完成!"
        echo "数据保存在: ./user_data/data/$EXCHANGE/*-funding_rate.json"
    else
        echo ""
        echo "❌ 资金费率数据下载失败!"
        exit 1
    fi
else
    echo "开始下载K线数据..."
    docker run --rm \
        -v "./user_data:/freqtrade/user_data" \
        freqtradeorg/freqtrade:stable \
        download-data \
        --config /freqtrade/user_data/config_docker.json \
        --pairs $PAIRS \
        --timeframe $TIMEFRAME \
        --timerange $TIMERANGE \
        --exchange $EXCHANGE \
        --data-format json

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ K线数据下载完成!"
        echo "数据保存在: ./user_data/data/$EXCHANGE/"
    else
        echo ""
        echo "❌ K线数据下载失败!"
        exit 1
    fi
fi