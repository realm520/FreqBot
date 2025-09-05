# FreqTrade Docker WebUI 使用指南

本指南介绍如何使用Docker运行FreqTrade WebUI来查看VATSMStrategy策略的回测结果。

## 快速开始

### 1. 下载历史数据

```bash
# 下载默认数据 (ETH/USDT, BTC/USDT, 15分钟, 2024年数据)
./docker_download_data.sh

# 自定义下载
./docker_download_data.sh --pairs "ETH/USDT BTC/USDT SOL/USDT" --timeframe 5m --timerange 20230101-20231231
```

### 2. 运行回测

```bash
# 使用默认参数运行VATSMStrategy回测
./docker_run_backtest.sh

# 自定义回测参数
./docker_run_backtest.sh --strategy VATSMStrategy --timeframe 1h --timerange 20230601-20240601
```

### 3. 启动WebUI

```bash
# 启动WebUI服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 4. 访问WebUI

1. 打开浏览器访问: http://localhost:8080
2. 登录信息:
   - 用户名: `freqtrade`
   - 密码: `freqtrade123`

## WebUI功能介绍

### Dashboard (仪表板)
- 实时显示机器人状态
- 账户余额和收益统计
- 活跃交易概览

### Backtesting (回测)
- **运行新回测**: 
  1. 选择策略: VATSMStrategy
  2. 设置时间范围
  3. 配置参数
  4. 点击"Run Backtest"
- **查看历史回测结果**:
  - 在结果列表中选择要查看的回测
  - 查看详细统计和图表

### Graph (图表)
- 交易可视化
- K线图和技术指标
- 买卖信号标记
- 支持的VATSMStrategy指标:
  - Momentum (动量)
  - Volume Ratio (成交量比率)
  - ATR (平均真实波动范围)
  - Target Exposure (目标仓位)

### Trade History (交易历史)
- 所有历史交易记录
- 交易详情和盈亏统计
- 可按日期、交易对筛选
- 导出CSV功能

### Performance (性能分析)
- 策略总体表现
- 关键指标:
  - 总收益率
  - 夏普比率
  - 最大回撤
  - 胜率
- 月度/年度收益表
- 交易对表现对比

## Docker命令参考

### 服务管理

```bash
# 启动所有服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f freqtrade

# 进入容器
docker exec -it freqtrade_webui /bin/bash
```

### 单独运行任务

```bash
# 仅下载数据
docker-compose run --rm freqtrade-download

# 仅运行回测
docker-compose run --rm freqtrade-backtesting

# 使用自定义配置运行
docker run --rm -v "./user_data:/freqtrade/user_data" \
  freqtradeorg/freqtrade:stable \
  backtesting \
  --config /freqtrade/user_data/config_docker.json \
  --strategy VATSMStrategy
```

## 配置说明

### config_docker.json
主要配置文件，包含:
- API服务器设置 (已启用，监听0.0.0.0:8080)
- 策略路径配置
- 交易对设置
- 数据格式配置 (JSON格式，兼容性更好)

### docker-compose.yml
定义了三个服务:
1. **freqtrade**: WebUI主服务 (注意: webserver命令不需要--strategy参数，策略在config文件中指定)
2. **freqtrade-backtesting**: 回测服务 (profile: backtesting)
3. **freqtrade-download**: 数据下载服务 (profile: download)

## 常见问题

### Q: 无法访问WebUI
A: 检查:
1. Docker服务是否正常运行: `docker-compose ps`
2. 端口是否被占用: `netstat -tlnp | grep 8080`
3. 防火墙设置

### Q: 回测没有数据
A: 确保:
1. 已下载对应时间范围的数据
2. 数据路径正确映射
3. 配置文件中的交易对与下载的数据匹配

### Q: 策略未找到
A: 检查:
1. 策略文件存在: `ls user_data/strategies/VATSMStrategy.py`
2. 策略类名正确
3. 策略路径配置正确

### Q: 修改配置后不生效
A: 需要重启服务:
```bash
docker-compose restart
```

## 高级配置

### 修改WebUI端口
编辑 `docker-compose.yml`:
```yaml
ports:
  - "9090:8080"  # 改为9090端口
```

### 添加更多策略
1. 将策略文件放入 `user_data/strategies/`
2. 在WebUI中选择新策略进行回测

### 配置实盘交易
1. 编辑 `user_data/config_docker.json`
2. 设置 `dry_run: false`
3. 添加交易所API密钥
4. **警告**: 请先在模拟环境充分测试!

## 性能优化建议

1. **数据格式**: 使用feather格式可提升加载速度
2. **时间范围**: 合理设置回测时间范围，避免过长
3. **Docker资源**: 可调整Docker内存和CPU限制
4. **并行回测**: 可同时运行多个回测容器

## 安全建议

1. **修改默认密码**: 编辑config_docker.json中的username和password
2. **JWT密钥**: 使用强随机字符串替换jwt_secret_key
3. **网络访问**: 生产环境建议使用反向代理和SSL
4. **API密钥**: 永远不要在配置文件中硬编码真实API密钥

## 更多信息

- FreqTrade官方文档: https://www.freqtrade.io/
- Docker Hub: https://hub.docker.com/r/freqtradeorg/freqtrade
- 策略开发指南: 参见 `docs/strategies/VATSMStrategy_Guide.md`