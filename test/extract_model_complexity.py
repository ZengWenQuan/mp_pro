#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import re
import glob
from pathlib import Path

def extract_model_complexity_from_log(log_file):
    """
    从训练日志文件中提取模型复杂度信息
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        包含模型参数量信息的字典，如果未找到则返回None
    """
    try:
        complexity_info = {}
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 提取模型名称
            model_name_match = re.search(r'模型结构总结', content)
            if model_name_match:
                # 查找参数总数
                total_params_match = re.search(r'参数总数: (\d+[,\d]*)', content)
                if total_params_match:
                    # 处理可能包含逗号的数字
                    total_params = total_params_match.group(1).replace(',', '')
                    complexity_info['total_params'] = int(total_params)
                
                # 查找可训练参数
                trainable_params_match = re.search(r'可训练参数: (\d+[,\d]*)', content)
                if trainable_params_match:
                    # 处理可能包含逗号的数字
                    trainable_params = trainable_params_match.group(1).replace(',', '')
                    complexity_info['trainable_params'] = int(trainable_params)
                
                return complexity_info
        
        return None
    except Exception as e:
        print(f"处理文件 {log_file} 时出错: {e}")
        return None

def collect_model_complexity(base_dir, output_file):
    """
    收集所有模型的复杂度信息并保存到CSV文件
    
    Args:
        base_dir: 结果的基础目录
        output_file: 输出文件路径
    """
    # 存储所有模型的复杂度信息
    model_complexities = []
    
    # 遍历全部特征和500特征目录
    dataset_types = ["全部特征", "500特征"]
    
    for dataset_type in dataset_types:
        # 结果目录
        results_dir = os.path.join(base_dir, dataset_type)
        
        # 遍历目录下的所有模型文件夹
        model_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
        
        for model_dir in model_dirs:
            # 从目录名称中提取模型名称
            model_name = os.path.basename(model_dir).split('_')[0]
            
            # 查找日志文件
            log_files = []
            for ext in ['log', 'txt']:
                log_files.extend(glob.glob(os.path.join(model_dir, f"*.{ext}")))
            
            # 如果没有找到日志文件，查找子目录中的日志文件
            if not log_files:
                log_files = list(Path(model_dir).rglob("*.log")) + list(Path(model_dir).rglob("*.txt"))
            
            complexity_info = None
            
            # 尝试从每个日志文件中提取信息
            for log_file in log_files:
                complexity_info = extract_model_complexity_from_log(log_file)
                if complexity_info:
                    break
            
            # 如果找到了复杂度信息，添加到结果列表
            if complexity_info:
                model_complexities.append({
                    'model': model_name,
                    'dataset': dataset_type,
                    'total_params': complexity_info.get('total_params', 'N/A'),
                    'trainable_params': complexity_info.get('trainable_params', 'N/A')
                })
            else:
                # 如果未找到复杂度信息，添加一个占位条目
                model_complexities.append({
                    'model': model_name,
                    'dataset': dataset_type,
                    'total_params': 'N/A',
                    'trainable_params': 'N/A'
                })
    
    # 保存到CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'dataset', 'total_params', 'trainable_params']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for complexity in model_complexities:
            writer.writerow(complexity)
    
    print(f"已保存模型复杂度信息到 {output_file}")

def extract_training_time(base_dir, output_file):
    """
    提取所有模型的训练时间信息并保存到CSV文件
    
    Args:
        base_dir: 结果的基础目录
        output_file: 输出文件路径
    """
    # 存储所有模型的训练时间信息
    training_times = []
    
    # 遍历全部特征和500特征目录
    dataset_types = ["全部特征", "500特征"]
    
    for dataset_type in dataset_types:
        # 结果目录
        results_dir = os.path.join(base_dir, dataset_type)
        
        # 遍历目录下的所有模型文件夹
        model_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
        
        for model_dir in model_dirs:
            # 从目录名称中提取模型名称
            model_name = os.path.basename(model_dir).split('_')[0]
            
            # 查找日志文件
            log_files = []
            for ext in ['log', 'txt']:
                log_files.extend(glob.glob(os.path.join(model_dir, f"*.{ext}")))
            
            # 如果没有找到日志文件，查找子目录中的日志文件
            if not log_files:
                log_files = list(Path(model_dir).rglob("*.log")) + list(Path(model_dir).rglob("*.txt"))
            
            training_time = "N/A"
            
            # 尝试从每个日志文件中提取训练时间信息
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # 查找训练时间
                        time_match = re.search(r'总训练时间: ([\d.]+)小时 ([\d.]+)分钟 ([\d.]+)秒', content)
                        if time_match:
                            hours = float(time_match.group(1))
                            minutes = float(time_match.group(2))
                            seconds = float(time_match.group(3))
                            
                            # 转换为小时
                            training_time = round(hours + minutes/60 + seconds/3600, 2)
                            break
                        
                        # 尝试其他可能的格式
                        time_match2 = re.search(r'总训练时间: ([\d.]+)分钟 ([\d.]+)秒', content)
                        if time_match2:
                            minutes = float(time_match2.group(1))
                            seconds = float(time_match2.group(2))
                            
                            # 转换为小时
                            training_time = round(minutes/60 + seconds/3600, 2)
                            break
                except Exception as e:
                    print(f"处理文件 {log_file} 训练时间时出错: {e}")
            
            # 添加到结果列表
            training_times.append({
                'model': model_name,
                'dataset': dataset_type,
                'training_time_hours': training_time
            })
    
    # 保存到CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'dataset', 'training_time_hours']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for time_info in training_times:
            writer.writerow(time_info)
    
    print(f"已保存模型训练时间信息到 {output_file}")

def create_combined_summary(complexity_file, time_file, output_file):
    """
    将模型复杂度和训练时间信息合并到一个CSV文件中
    
    Args:
        complexity_file: 复杂度信息CSV文件路径
        time_file: 训练时间信息CSV文件路径
        output_file: 输出文件路径
    """
    # 读取复杂度信息
    complexity_data = {}
    with open(complexity_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['model']}_{row['dataset']}"
            complexity_data[key] = {
                'model': row['model'],
                'dataset': row['dataset'],
                'total_params': row['total_params'],
                'trainable_params': row['trainable_params']
            }
    
    # 读取训练时间信息
    time_data = {}
    with open(time_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['model']}_{row['dataset']}"
            time_data[key] = {
                'training_time_hours': row['training_time_hours']
            }
    
    # 合并数据
    combined_data = []
    for key, complexity in complexity_data.items():
        row = complexity.copy()
        if key in time_data:
            row.update(time_data[key])
        else:
            row['training_time_hours'] = 'N/A'
        combined_data.append(row)
    
    # 保存到CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'dataset', 'total_params', 'trainable_params', 'training_time_hours']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in combined_data:
            writer.writerow(row)
    
    print(f"已保存综合模型信息到 {output_file}")

def main():
    # 结果目录
    base_results_dir = "results"
    
    # 输出目录
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 复杂度信息CSV文件路径
    complexity_file = os.path.join(output_dir, "model_complexity.csv")
    
    # 训练时间信息CSV文件路径
    time_file = os.path.join(output_dir, "training_time.csv")
    
    # 综合信息CSV文件路径
    combined_file = os.path.join(output_dir, "model_summary.csv")
    
    # 收集模型复杂度信息
    collect_model_complexity(base_results_dir, complexity_file)
    
    # 提取训练时间信息
    extract_training_time(base_results_dir, time_file)
    
    # 创建综合摘要
    create_combined_summary(complexity_file, time_file, combined_file)
    
    print("所有模型复杂度和训练时间信息已提取并保存")

if __name__ == "__main__":
    main() 