#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速启动训练脚本
用法:
    python run_train.py --config configs/config.yaml
"""

import os
import argparse
import subprocess
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    parser = argparse.ArgumentParser(description="启动模型训练")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument("--exp-name", type=str, default=None,
                        help="实验名称")
    args = parser.parse_args()
    
    # 构建训练命令
    cmd = [
        "python", "train.py",
        "--config", args.config,
        "--seed", str(args.seed)
    ]
    if args.resume_from:
        cmd.extend(["--resume_from", args.resume_from])
        
    if args.exp_name:
        cmd.extend(["--exp-name", args.exp_name])
    
    # 打印命令
    print("执行训练命令:")
    print(" ".join(cmd))
    
    # 运行训练
    subprocess.call(cmd)


if __name__ == "__main__":
    main() 