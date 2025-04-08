#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速启动评估脚本
用法:
    python run_evaluate.py --model-path runs/mlp_20230330_120000/weights/best.pt
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    parser = argparse.ArgumentParser(description="启动模型评估")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型权重文件路径")
    parser.add_argument("--config", type=str, default="configs/mpbdnet.yaml",
                        help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()
    
    # 如果未指定输出目录，创建默认输出目录
    if args.output_dir is None:
        model_dir = Path(args.model_path).parent.parent
        args.output_dir = str(model_dir / "evaluation")
    
    # 构建评估命令
    cmd = [
        "python", "evaluate.py",
        "--model-path", args.model_path,
        "--config", args.config,
        "--output-dir", args.output_dir,
        "--seed", str(args.seed)
    ]
    
    # 打印命令
    print("执行评估命令:")
    print(" ".join(cmd))
    
    # 运行评估
    subprocess.call(cmd)


if __name__ == "__main__":
    main() 