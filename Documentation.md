# Investment
Agentic AI agent for investment assistance

## Current Modules

- **Data Provider (`data_provider.py`)**  
  - 支持获取股票价格（目前通过 `stooq` 和合成数据`synthetic`，可选 `yfinance`）。  
  - 元数据与基本面默认调用 Financial Modeling Prep（`fmp_client.py`），同时保留 yfinance 作为后备。  
  - 缓存写入采用原子方式，避免进程中断导致的半成品文件；可配置 FMP 的重试冷却与逐条日志输出。  

- **FMP Client (`fmp_client.py`)**  
  - 封装 FMP 的 `profile`、`ratios-ttm` 与 `key-metrics-ttm` 接口，一次请求一个股票。  
  - 自适应节流：遭遇 429 自动放慢节奏，连续成功后逐步回落；Premium/授权错误会对单票进入冷却，避免无谓重试。  

- **Portfolio Construction & Optimization (`optimizer.py`)**  
  - 等权重组合。  
  - 基于 **PyPortfolioOpt** 的均值方差优化（最大化 Sharpe 等目标）。  
  - 新增多目标优化引擎（`engine=multi_objective`），可同时最大化 Sharpe / 收益并最小化波动与回撤。  
  - 输出统一权重与核心绩效指标，方便与报告模块衔接。  

- **Performance Metrics (`perf_metrics.py`)**  
  - 抽取常用绩效指标（年化收益、波动、Sharpe、Sortino、最大回撤等）。  
  - 供优化与回测模块共享，避免重复实现。  

- **Backtesting (`backtest.py`)**  
  - 支持 in-sample 与 walk-forward 模式，复用策略优化逻辑生成滚动权重。  
  - 可配置再平衡频率、训练/测试窗口、交易成本。  
  - 结果保存为 JSON，便于后续分析或报告。  

- **Policy / Universe (`policy.py`, `policy.yaml`, `universe.txt`)**  
  - YAML 文件定义投资策略：股票池、约束条件、集中度规则。  
  - Universe 列表管理可选股票集合。  

- **Risk Tools (`risk_tools.py`)**  
  - 核心风险指标：VaR（在险价值）、ES（期望损失）。  
  - 最大回撤计算。  

- **Selector / Concentration (`selector.py`, `concentration.py`)**  
  - 支持用户自定义约束（例如前 5 大股票占比 > 50%）。  
  - 股票筛选和集中度控制。  

- **Stock Analysis & Candidate Pool (`stock_analysis.py`)**  
  - `CandidatePoolBuilder` 读取 policy.yaml 的 `candidate_pool` 配置，串联元数据过滤、横截面因子与行业/市值中性化。  
  - 自动将打分靠前的 ~200 只美股写入 `data/universe/universe_YYYYMMDD_*.txt` 并更新 `universe.txt`。  
  - `selector` 运行前可自动读取该候选池，叠加策略级筛选与约束。  
  - 默认 DataProvider 读取 `candidate_pool.base_universe_file`（或 `data.universe_file`）的种子列表，价格来自 Stooq，元数据/基本面由 FMP 获取（需配置 `FMP_API_KEY`）。  
  - 使用 `python -m portfolio_agent.tools.build_base_universe --out data/universe/all_us_common.txt` 可从 Nasdaq Trader 目录快速生成美股全量清单。  
  - 因子计算新增 FCF 收益率、ROIC、净债务/EBITDA 等回退指标，数据缺位时可保持更多股票。  

- **Reporting (`qs_wrapper.py`, `reporting_extras.py`)**  
  - 基于 QuantStats 生成单策略报告（等权 / 优化）。  
  - 已扩展支持 **组合报告**（两条策略曲线 + 指标对比）。  
  - 输出图表（净值曲线、价格走势、相关性矩阵）。  

- **Main Script (`run.py`)**  
  - 程序统一入口，支持命令行参数。  
  - 输出 CSV 和 HTML 报告。  

## To be developed

* **Transaction 模块（`transaction.py`）**  
  - 统一管理交易摩擦项与税务。包含手续费计算、滑点模型、FIFO 税务引擎。  
  - Backtester 通过接口调用，便于后续替换不同市场费用和税率设定。  

* **Strategy 模块（`strategy.py`）**  
  - 策略即权重生成器。提供静态权重策略与滚动均值方差策略接口。  
  - 回测在再平衡日调用策略生成目标权重，便于扩展风格与约束。  

* **Investor Profile 模块（`profile.py`）**  
  - 把投资者风险画像与目标转成可执行参数。  
  - 例如风险等级映射到目标函数、权重上下限、再平衡频率与定投金额。  

* **Stress Test 模块（`stress_test.py`）**  
  - 预置历史情景窗口并复用回测引擎评估组合在危机期的表现。  
  - 输出净值、回撤、恢复时间与最差月份等关键指标。  

* **Monte Carlo 模块（`mc_sim.py`）**  
  - 基于历史或参数化分布进行路径模拟，评估财富目标达成概率。  
  - 支持历史重采样与高斯或学生 t 分布，输出区间带与成功概率。  

* **Narrative Report 模块（`nl_report.py`）**  
  - 生成面向普通投资者的自然语言总结，解释收益与风险来源。  
  - 可与现有报告合并，突出费用与税务拖累以及目标达成情况。  

* **Asset Provider 扩展模块（`asset_provider.py`）**  
  - 扩展多资产数据加载，支持 ETF 债券黄金房地产与加密资产等。  
  - 统一符号规范与数据源映射，提升股票池维护体验。  
