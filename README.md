# 作文评分 Prompt 进化系统

基于遗传算法的作文自动评分 Prompt 优化框架。

## 功能特性

- **结构化 Prompt 进化**：使用 PromptGenome + 遗传算法优化评分指令
- **双模式 Few-shot ICL**：
  - 策略模式：进化采样策略（random, balanced, extreme 等）
  - 索引模式：直接进化示例索引列表（更精细控制）
- **多指标评估**：QWK / Pearson / RMSE / Accuracy
- **LLM 驱动变异**：基于偏差统计的智能 Prompt 优化
- **模板池管理**：自动积累和复用高质量模板
- **断点续传**：自动保存进度，网络中断后可继续运行

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

配置 `.env` 文件：

设置OPENROUTER_API_KEY环境变量

```bash
# API 配置
export OPENROUTER_API_KEY=your_key_here
```

### 准备数据

```bash
python prepare_data.py
```

### 运行进化

```bash
python run_evolution.py
```

## 项目结构

```
├── evolver/                      # 进化算法核心
│   ├── alphaevolve_multi.py      # 主进化流程
│   ├── prompt_evolver.py         # 遗传算子
│   ├── data_aware_prompt.py      # Prompt 构建（支持多数据集）
│   ├── prompt_templates.json     # 数据集特定的 Prompt 模板
│   ├── icl_sampler.py            # ICL 示例采样
│   └── checkpoint.py             # 断点续传
├── llm_api/                      # LLM API 封装
│   └── openrouter_api.py         # OpenRouter API 调用
├── scorer/                       # 评分器和特征提取
│   ├── infer_scorer.py           # 推理评分器
│   ├── corrector.py              # 分数校正器
│   └── features.py               # 特征提取
├── utils/                        # 工具函数
│   ├── metrics.py                # 评估指标
│   └── preprocess.py             # 数据预处理
├── data/                         # 数据集（被 .gitignore 忽略）
│   ├── ielts_chillies/           # IELTS Chillies 数据集
│   │   ├── raw/                  # 原始数据
│   │   └── processed/            # 清洗后的数据
│   ├── ielts_kaggle/             # IELTS Kaggle 数据集
│   ├── asap_1/ ... asap_8/       # ASAP 8 个 essay sets
│   └── ...
├── logs/                         # 运行日志和结果
│   ├── checkpoints/              # 断点续传检查点
│   ├── best_scoring_prompt_hf.json
│   └── metrics_curve_hf.png
├── prepare_data.py               # 数据准备脚本
├── run_evolution.py              # 启动脚本
├── .env                          # 环境配置（数据集选择等）
└── requirements.txt              # Python 依赖
```

## 输出结果

- `logs/best_scoring_prompt_hf.json` - 最佳 Prompt 配置
- `logs/best_scoring_prompt_hf.txt` - 最佳 Prompt 文本
- `logs/best_prompt_predictions_hf.csv` - 预测结果
- `logs/metrics_curve_hf.png` - 指标曲线图
- `logs/template_pool.json` - 模板池
