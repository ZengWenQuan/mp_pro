#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def collect_model_results(base_dir, dataset_type):
    """
    收集特定数据集类型的所有模型评估结果
    
    Args:
        base_dir: 结果的基础目录
        dataset_type: 数据集类型 ('全部特征' 或 '500特征')
    
    Returns:
        包含所有模型评估指标的DataFrame
    """
    # 结果目录
    results_dir = os.path.join(base_dir, dataset_type)
    
    # 存储结果的列表
    all_results = []
    
    # 遍历目录下的所有模型文件夹
    model_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
    
    for model_dir in model_dirs:
        # 从目录名称中提取模型名称
        model_name = os.path.basename(model_dir).split('_')[0]
        
        # 评估结果文件路径
        eval_file = os.path.join(model_dir, "evaluation", "train_test_comparison.csv")
        
        if not os.path.exists(eval_file):
            print(f"警告: 未找到评估文件 {eval_file}")
            continue
        
        # 读取评估结果
        try:
            results_df = pd.read_csv(eval_file)
            
            # 只保留测试集结果和特定指标
            if 'test_mse' in results_df.columns and 'test_rmse' in results_df.columns and 'test_mae' in results_df.columns:
                # 获取各个特征的结果
                for _, row in results_df.iterrows():
                    # 过滤掉overall行
                    feature_name = row.iloc[0] if isinstance(row.iloc[0], str) else 'overall'
                    if feature_name not in ['overall', 'nan', '', np.nan]:
                        result = {
                            'model': model_name,
                            'dataset': dataset_type,
                            'feature': feature_name,
                            'mse': row['test_mse'] if 'test_mse' in results_df.columns else None,
                            'rmse': row['test_rmse'] if 'test_rmse' in results_df.columns else None, 
                            'mae': row['test_mae'] if 'test_mae' in results_df.columns else None,
                            'r2': row['test_r2'] if 'test_r2' in results_df.columns else None
                        }
                        all_results.append(result)
                
                # 添加overall结果
                overall_row = results_df[results_df.iloc[:, 0] == 'overall'] if 'overall' in results_df.iloc[:, 0].values else results_df.iloc[-1]
                if len(overall_row) > 0:
                    result = {
                        'model': model_name,
                        'dataset': dataset_type,
                        'feature': 'overall',
                        'mse': overall_row['test_mse'].values[0] if 'test_mse' in results_df.columns else None,
                        'rmse': overall_row['test_rmse'].values[0] if 'test_rmse' in results_df.columns else None,
                        'mae': overall_row['test_mae'].values[0] if 'test_mae' in results_df.columns else None,
                        'r2': None  # overall通常没有R2值
                    }
                    all_results.append(result)
            else:
                print(f"警告: 文件 {eval_file} 缺少所需列")
        except Exception as e:
            print(f"处理文件 {eval_file} 时出错: {e}")
    
    # 转换为DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        print(f"在 {dataset_type} 中未找到任何结果")
        return pd.DataFrame()

def create_comparison_tables(results_df, output_dir):
    """
    创建各种比较表并保存为CSV
    
    Args:
        results_df: 包含所有结果的DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 按指标分类的表格
    metrics = ['mse', 'rmse', 'mae', 'r2']
    for metric in metrics:
        if metric in results_df.columns:
            # 透视表: 行=模型，列=特征，值=指标
            pivot_df = results_df.pivot_table(
                index=['model', 'dataset'],
                columns='feature',
                values=metric,
                aggfunc='first'
            )
            
            # 保存为CSV
            output_file = os.path.join(output_dir, f"{metric}_comparison.csv")
            pivot_df.to_csv(output_file)
            print(f"已保存比较表到 {output_file}")
    
    # 2. 按数据集分类的表格
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        
        # 按特征创建表格
        features = [f for f in dataset_df['feature'].unique() if f != 'overall']
        for feature in features:
            feature_df = dataset_df[dataset_df['feature'] == feature]
            
            # 创建一个宽格式表，每列是一个指标
            wide_df = feature_df.pivot(
                index='model',
                columns='feature',
                values=['mse', 'rmse', 'mae', 'r2']
            )
            
            # 标准化列名
            wide_df.columns = [f"{col[0]}_{col[1]}" for col in wide_df.columns]
            
            # 保存为CSV
            safe_dataset = dataset.replace('/', '_')
            output_file = os.path.join(output_dir, f"{safe_dataset}_{feature}_comparison.csv")
            wide_df.to_csv(output_file)
            print(f"已保存比较表到 {output_file}")
    
    # 3. 综合比较表 - 所有模型、所有特征、所有指标
    # 按模型和数据集分组，每组只保留overall指标
    overall_df = results_df[results_df['feature'] == 'overall']
    
    # 创建一个包含所有指标的宽格式表
    wide_overall_df = pd.pivot_table(
        overall_df,
        index='model',
        columns='dataset',
        values=['mse', 'rmse', 'mae']
    )
    
    # 标准化列名
    wide_overall_df.columns = [f"{col[0]}_{col[1]}" for col in wide_overall_df.columns]
    
    # 保存综合比较表
    output_file = os.path.join(output_dir, "overall_model_comparison.csv")
    wide_overall_df.to_csv(output_file)
    print(f"已保存综合比较表到 {output_file}")
    
    # 4. 模型排名表
    # 为每个指标和数据集创建排名
    ranking_results = []
    
    for dataset in results_df['dataset'].unique():
        for feature in ['logg', 'feh', 'teff', 'overall']:
            subset = results_df[(results_df['dataset'] == dataset) & (results_df['feature'] == feature)]
            
            if len(subset) > 0:
                # 对每个指标进行排名
                for metric in ['mse', 'rmse', 'mae']:
                    if metric in subset.columns:
                        # 按指标值从小到大排序
                        ranked = subset.sort_values(by=metric)
                        for i, (_, row) in enumerate(ranked.iterrows()):
                            ranking_results.append({
                                'model': row['model'],
                                'dataset': dataset,
                                'feature': feature,
                                'metric': metric,
                                'value': row[metric],
                                'rank': i + 1
                            })
                
                # 对R2指标特殊处理（越大越好）
                if 'r2' in subset.columns:
                    ranked = subset.sort_values(by='r2', ascending=False)
                    for i, (_, row) in enumerate(ranked.iterrows()):
                        if not pd.isna(row['r2']):
                            ranking_results.append({
                                'model': row['model'],
                                'dataset': dataset,
                                'feature': feature,
                                'metric': 'r2',
                                'value': row['r2'],
                                'rank': i + 1
                            })
    
    ranking_df = pd.DataFrame(ranking_results)
    
    # 保存排名表
    output_file = os.path.join(output_dir, "model_rankings.csv")
    ranking_df.to_csv(output_file, index=False)
    print(f"已保存模型排名表到 {output_file}")
    
    # 5. 最佳模型摘要表
    best_models = []
    
    for dataset in results_df['dataset'].unique():
        for feature in ['logg', 'feh', 'teff', 'overall']:
            for metric in ['mse', 'rmse', 'mae']:
                subset = ranking_df[(ranking_df['dataset'] == dataset) & 
                                    (ranking_df['feature'] == feature) &
                                    (ranking_df['metric'] == metric) &
                                    (ranking_df['rank'] == 1)]
                if len(subset) > 0:
                    row = subset.iloc[0]
                    best_models.append({
                        'dataset': dataset,
                        'feature': feature,
                        'metric': metric,
                        'best_model': row['model'],
                        'value': row['value']
                    })
            
            # R2指标（越大越好）
            subset = ranking_df[(ranking_df['dataset'] == dataset) & 
                               (ranking_df['feature'] == feature) &
                               (ranking_df['metric'] == 'r2') &
                               (ranking_df['rank'] == 1)]
            if len(subset) > 0:
                row = subset.iloc[0]
                best_models.append({
                    'dataset': dataset,
                    'feature': feature,
                    'metric': 'r2',
                    'best_model': row['model'],
                    'value': row['value']
                })
    
    best_df = pd.DataFrame(best_models)
    
    # 保存最佳模型摘要
    output_file = os.path.join(output_dir, "best_models_summary.csv")
    best_df.to_csv(output_file, index=False)
    print(f"已保存最佳模型摘要到 {output_file}")

def main():
    # 结果目录
    base_results_dir = "results"
    
    # 输出目录
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集全部特征和500特征的结果
    full_feature_results = collect_model_results(base_results_dir, "全部特征")
    limited_feature_results = collect_model_results(base_results_dir, "500特征")
    
    # 合并结果
    all_results = pd.concat([full_feature_results, limited_feature_results], ignore_index=True)
    
    # 保存合并后的原始数据
    all_results.to_csv(os.path.join(output_dir, "all_model_results_raw.csv"), index=False)
    print(f"已保存原始结果数据到 {os.path.join(output_dir, 'all_model_results_raw.csv')}")
    
    # 创建比较表
    create_comparison_tables(all_results, output_dir)
    
    print("所有比较表已生成")

if __name__ == "__main__":
    main() 