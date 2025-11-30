#!/usr/bin/env python
"""
数据准备脚本
1. 根据 .env 中的 DATASET_NAME 下载原始数据到 data/raw/
2. 清洗数据并保存到 data/processed/
3. 生成 train_clean.csv 和 eval_clean.csv 供 run_evolution.py 使用
"""
import os
import re
import random
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

BASE_DIR = Path(__file__).parent
DATASET_NAME = os.getenv("DATASET_NAME", "ielts_chillies")


# ==================== 数据下载函数 ==================== #

def download_ielts_chillies():
    """从 Hugging Face 下载 IELTS Chillies 数据集"""
    from datasets import load_dataset
    
    print("📥 正在从 Hugging Face 下载 IELTS Chillies 数据集...")
    
    output_dir = BASE_DIR / "data" / "ielts_chillies" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset("chillies/IELTS-writing-task-2-evaluation")
    df = dataset["train"].to_pandas()
    
    output_path = output_dir / "train.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"✅ 下载完成: {output_path} ({len(df)} 行)")
    return output_path


def download_ielts_kaggle():
    """从 Kaggle 下载 IELTS 数据集"""
    print("📥 正在下载 IELTS Kaggle 数据集...")
    
    output_dir = BASE_DIR / "data" / "ielts_kaggle" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有旧数据可以复制
    legacy_file = BASE_DIR / "data_legacy" / "raw" / "kaggle" / "ielts_writing_dataset.csv"
    output_file = output_dir / "ielts_writing_dataset.csv"
    
    if legacy_file.exists():
        import shutil
        print(f"   从旧数据目录复制: {legacy_file}")
        shutil.copy(legacy_file, output_file)
        print(f"✅ 复制完成: {output_file}")
        return output_file
    
    # 如果没有旧数据，尝试从 Kaggle 下载
    try:
        import kaggle
        print("⚠️  Kaggle IELTS 数据集需要手动下载")
        print("   请访问 Kaggle 搜索 'IELTS writing dataset' 并下载")
        print(f"   然后将文件放到: {output_file}")
        return None
    except ImportError:
        print("❌ 未安装 kaggle 包，请运行: pip install kaggle")
        return None


def download_asap():
    """从 Kaggle 下载 ASAP 数据集"""
    print("📥 正在下载 ASAP 数据集...")
    
    output_dir = BASE_DIR / "data" / "asap" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有数据文件
    output_path = output_dir / "training_set_rel3.tsv"
    if output_path.exists():
        print(f"✅ 数据文件已存在: {output_path}")
        return output_path
    
    try:
        import kaggle
        import zipfile
        
        print("   尝试从 Kaggle API 下载...")
        kaggle.api.competition_download_files('asap-aes', path=output_dir, quiet=False)
        
        zip_path = output_dir / "asap-aes.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()
            
            if output_path.exists():
                print(f"✅ 下载完成: {output_path}")
                return output_path
    except ImportError:
        print("❌ 未安装 kaggle 包，请运行: pip install kaggle")
    except Exception as e:
        print(f"❌ API 下载失败: {e}")
    
    # 提供手动下载指引
    print("\n" + "="*60)
    print("⚠️  ASAP 数据集需要手动下载")
    print("="*60)
    print("\n请按以下步骤操作：")
    print("\n1. 访问 ASAP-AES 竞赛页面:")
    print("   https://www.kaggle.com/c/asap-aes")
    print("\n2. 点击 'Join Competition' 并接受规则")
    print("\n3. 进入 Data 标签页:")
    print("   https://www.kaggle.com/c/asap-aes/data")
    print("\n4. 下载 'training_set_rel3.tsv' 文件")
    print(f"\n5. 将文件放到以下目录:")
    print(f"   {output_dir}/")
    print("\n6. 重新运行: uv run prepare_data.py")
    print("="*60)
    
    return None


# ==================== 数据清洗函数 ==================== #

def safe_parse_band(val) -> Optional[float]:
    """
    解析 band 分数，支持多种格式:
    - "7.5", "5.0\\n\\n"
    - "<4", "<4\\n\\n"
    - "Band: 6.5"
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    
    band = float(nums[0])
    if s.lstrip().startswith("<"):
        band -= 0.5
    
    # 限制在 0-9 范围，四舍五入到 0.5
    band = max(0.0, min(9.0, band))
    band = round(band * 2) / 2.0
    return band


def clean_ielts_chillies_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """清洗 IELTS Chillies 数据集（HuggingFace）"""
    stats = {
        "raw_rows": len(df),
        "drop_na": 0,
        "bad_band": 0,
        "bad_len": 0,
        "dedup": 0,
    }
    
    # 删除缺失值
    df = df.dropna(subset=["prompt", "essay", "band"]).reset_index(drop=True)
    stats["drop_na"] = stats["raw_rows"] - len(df)
    
    # 解析 band 分数
    df["band_clean"] = df["band"].apply(safe_parse_band)
    bad_band_mask = df["band_clean"].isna()
    stats["bad_band"] = int(bad_band_mask.sum())
    df = df[~bad_band_mask].reset_index(drop=True)
    
    # 过滤字数不合理的文章
    df["word_count"] = df["essay"].apply(lambda x: len(str(x).split()))
    bad_len_mask = (df["word_count"] < 50) | (df["word_count"] > 1200)
    stats["bad_len"] = int(bad_len_mask.sum())
    df = df[~bad_len_mask].reset_index(drop=True)
    
    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "essay"]).reset_index(drop=True)
    stats["dedup"] = before - len(df)
    
    # 重命名列
    df = df.drop(columns=["band"], errors="ignore")
    df = df.rename(columns={"band_clean": "band"})
    df = df.drop(columns=["word_count"], errors="ignore")
    
    stats["clean_rows"] = len(df)
    return df, stats


def clean_ielts_kaggle_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """清洗 IELTS Kaggle 数据集"""
    stats = {
        "raw_rows": len(df),
        "drop_na": 0,
        "bad_band": 0,
        "bad_len": 0,
        "dedup": 0,
    }
    
    # Kaggle 数据集列名: Question, Essay, Overall
    # 重命名为统一格式
    df = df.rename(columns={
        "Question": "prompt",
        "Essay": "essay",
        "Overall": "band"
    })
    
    # 只保留需要的列
    df = df[["prompt", "essay", "band"]].copy()
    
    # 删除缺失值
    df = df.dropna(subset=["prompt", "essay", "band"]).reset_index(drop=True)
    stats["drop_na"] = stats["raw_rows"] - len(df)
    
    # 解析 band 分数
    df["band_clean"] = df["band"].apply(safe_parse_band)
    bad_band_mask = df["band_clean"].isna()
    stats["bad_band"] = int(bad_band_mask.sum())
    df = df[~bad_band_mask].reset_index(drop=True)
    
    # 过滤字数不合理的文章
    df["word_count"] = df["essay"].apply(lambda x: len(str(x).split()))
    bad_len_mask = (df["word_count"] < 50) | (df["word_count"] > 1200)
    stats["bad_len"] = int(bad_len_mask.sum())
    df = df[~bad_len_mask].reset_index(drop=True)
    
    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "essay"]).reset_index(drop=True)
    stats["dedup"] = before - len(df)
    
    # 重命名列
    df = df.drop(columns=["band"], errors="ignore")
    df = df.rename(columns={"band_clean": "band"})
    df = df.drop(columns=["word_count"], errors="ignore")
    
    stats["clean_rows"] = len(df)
    return df, stats


def clean_asap_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """清洗 ASAP 数据集"""
    stats = {
        "raw_rows": len(df),
        "drop_na": 0,
        "bad_len": 0,
        "dedup": 0,
    }
    
    # ASAP 数据集列名: essay_id, essay_set, essay, rater1_domain1, rater2_domain1, domain1_score
    # 我们需要转换为统一格式: prompt, essay, band
    
    # 为每个 essay_set 创建对应的 prompt
    essay_set_prompts = {
        1: "Write an essay about the effects of computers on people.",
        2: "Write an essay about censorship in libraries.",
        3: "Write an essay about the advantages and disadvantages of RFID technology.",
        4: "Write an essay about the role of patience in life.",
        5: "Write an essay describing a person who has influenced you.",
        6: "Write an essay about the importance of laughter.",
        7: "Write an essay about the value of persistence.",
        8: "Write an essay about the benefits of laughter in difficult times.",
    }
    
    # 添加 prompt 列
    df["prompt"] = df["essay_set"].map(essay_set_prompts)
    
    # 使用 domain1_score 作为分数
    df["band"] = df["domain1_score"]
    
    # 只保留需要的列
    df = df[["prompt", "essay", "band"]].copy()
    
    # 删除缺失值
    df = df.dropna(subset=["prompt", "essay", "band"]).reset_index(drop=True)
    stats["drop_na"] = stats["raw_rows"] - len(df)
    
    # 标准化分数到 0-9 范围（ASAP 分数范围因 essay_set 而异）
    # 简单处理：将分数归一化到 0-9
    min_score = df["band"].min()
    max_score = df["band"].max()
    if max_score > min_score:
        df["band"] = ((df["band"] - min_score) / (max_score - min_score)) * 9.0
        df["band"] = (df["band"].round() * 0.5).round(1)  # 四舍五入到 0.5
    
    # 过滤字数
    df["word_count"] = df["essay"].apply(lambda x: len(str(x).split()))
    bad_len_mask = (df["word_count"] < 50) | (df["word_count"] > 1200)
    stats["bad_len"] = int(bad_len_mask.sum())
    df = df[~bad_len_mask].reset_index(drop=True)
    
    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["prompt", "essay"]).reset_index(drop=True)
    stats["dedup"] = before - len(df)
    df = df.drop(columns=["word_count"], errors="ignore")
    
    stats["clean_rows"] = len(df)
    return df, stats


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_per_band: int = 100,
    eval_per_band: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 band 分层采样，生成训练集和评估集"""
    train_rows = []
    eval_rows = []
    
    for band, sub in df.groupby("band"):
        sub = sub.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # 先取评估集
        n_eval = min(eval_per_band, max(1, len(sub) // 5))
        eval_part = sub.iloc[:n_eval]
        rest = sub.iloc[n_eval:]
        
        # 再取训练集
        n_train = min(train_per_band, len(rest))
        train_part = rest.iloc[:n_train]
        
        eval_rows.append(eval_part)
        train_rows.append(train_part)
    
    train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    eval_df = pd.concat(eval_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return train_df, eval_df


# ==================== 主函数 ==================== #

def main():
    print(f"🚀 数据准备脚本")
    print(f"   数据集: {DATASET_NAME}\n")
    
    # 1. 下载原始数据
    raw_path = None
    if DATASET_NAME == "ielts_chillies":
        raw_path = download_ielts_chillies()
        clean_func = clean_ielts_chillies_data
        output_dir = BASE_DIR / "data" / "ielts_chillies" / "processed"
    elif DATASET_NAME == "ielts_kaggle":
        raw_path = download_ielts_kaggle()
        clean_func = clean_ielts_kaggle_data
        output_dir = BASE_DIR / "data" / "ielts_kaggle" / "processed"
    elif DATASET_NAME == "asap":
        raw_path = download_asap()
        clean_func = clean_asap_data
        output_dir = BASE_DIR / "data" / "asap" / "processed"
    else:
        print(f"❌ 未知的数据集: {DATASET_NAME}")
        print("   支持的数据集: ielts_chillies, ielts_kaggle, asap")
        return
    
    if raw_path is None or not Path(raw_path).exists():
        print("❌ 数据下载失败")
        return
    
    # 2. 加载原始数据
    print(f"\n📂 加载原始数据: {raw_path}")
    if DATASET_NAME == "asap":
        df = pd.read_csv(raw_path, sep='\t', encoding='latin-1')
    else:
        df = pd.read_csv(raw_path)
    
    # 3. 清洗数据
    print(f"\n🧹 清洗数据...")
    clean_df, stats = clean_func(df)
    
    print("\n=== 清洗统计 ===")
    for k, v in stats.items():
        print(f"  {k:>12}: {v}")
    
    print("\n=== Band 分布 ===")
    print(clean_df["band"].value_counts().sort_index())
    
    # 4. 保存清洗后的完整数据
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 保存完整数据: {clean_path}")
    
    # 5. 分层采样生成训练集和评估集
    print(f"\n✂️  分层采样...")
    train_df, eval_df = stratified_split(clean_df)
    
    train_path = output_dir / "train_clean.csv"
    eval_path = output_dir / "eval_clean.csv"
    
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    
    print(f"   训练集: {train_path} ({len(train_df)} 行)")
    print(f"   评估集: {eval_path} ({len(eval_df)} 行)")
    
    print(f"\n✨ 数据准备完成！")
    print(f"   现在可以运行: python run_evolution.py")


if __name__ == "__main__":
    main()
