#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查模型checkpoint文件信息
用法:
    python check_model.py --model-path runs/mlp_20230330_120000/weights/best.pt
"""

import argparse
import json
from utils.general import check_checkpoint_model_type

def parse_args():
    parser = argparse.ArgumentParser(description='检查模型checkpoint信息')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重文件路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查模型类型
    model_info = check_checkpoint_model_type(args.model_path)
    
    # 打印模型信息
    print("==== 模型信息 ====")
    print(f"模型路径: {args.model_path}")
    
    if 'error' in model_info:
        print(f"错误: {model_info['error']}")
        return
    
    print(f"模型类型: {model_info['model_type']}")
    
    if 'note' in model_info:
        print(f"注意: {model_info['note']}")
    
    print("\n模型配置:")
    print(json.dumps(model_info['model_config'], indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main() 