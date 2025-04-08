#!/usr/bin/env python
"""
智能安装脚本 (Smart Installation Script):
1. 检查已安装的库，避免重复安装
2. 只安装缺失或版本不符的库
3. 自动检测GPU/CUDA可用性，选择合适的PyTorch版本
4. 使用清华源加速下载
"""
import subprocess
import sys
import importlib
import pkg_resources

# 配置pip源
TSINGHUA_SOURCE = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_SOURCE = "https://download.pytorch.org/whl/cu118"  # GPU版本
PYTORCH_CPU_SOURCE = "https://download.pytorch.org/whl/cpu"  # CPU版本

def is_gpu_available():
    """检查CUDA是否可用，通过运行nvidia-smi命令"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def is_installed(package_name):
    """检查库是否已安装"""
    try:
        # 移除版本说明符
        name = package_name.split('>=')[0].split('==')[0].strip()
        pkg_resources.get_distribution(name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def get_installed_version(package_name):
    """获取已安装库的版本"""
    try:
        name = package_name.split('>=')[0].split('==')[0].strip()
        return pkg_resources.get_distribution(name).version
    except pkg_resources.DistributionNotFound:
        return None

def parse_requirements_file():
    """解析requirements.txt文件"""
    with open('requirements.txt', 'r') as file:
        lines = file.readlines()
    
    requirements = []
    extra_index_url = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('--extra-index-url'):
            extra_index_url = line.split(' ')[1]
        else:
            requirements.append(line)
    
    return requirements, extra_index_url

def main():
    """根据环境智能安装所需的库"""
    print("=" * 60)
    print("MP_PRO 智能环境配置工具")
    print("=" * 60)
    
    # 检查GPU
    has_gpu = is_gpu_available()
    if has_gpu:
        print("✓ 检测到GPU! 将安装支持CUDA的PyTorch版本")
    else:
        print("✗ 未检测到GPU. 将安装CPU版本的PyTorch")
    
    # 解析requirements文件
    all_requirements, _ = parse_requirements_file()
    
    # 筛选出未安装的库
    to_install = []
    already_installed = []
    
    for req in all_requirements:
        if not req or req.startswith('#') or req.startswith('--'):
            continue
            
        if is_installed(req):
            current_version = get_installed_version(req.split('>=')[0].split('==')[0])
            if '>=' in req:
                required_version = req.split('>=')[1]
                if pkg_resources.parse_version(current_version) >= pkg_resources.parse_version(required_version):
                    already_installed.append(f"{req.split('>=')[0]} (当前版本: {current_version})")
                    continue
            else:
                already_installed.append(f"{req} (当前版本: {current_version})")
                continue
                
        to_install.append(req)
    
    # 特殊处理PyTorch
    torch_to_install = []
    final_to_install = []
    
    for req in to_install:
        if req.startswith('torch') or req.startswith('torchvision'):
            torch_to_install.append(req)
        else:
            final_to_install.append(req)
    
    # 报告已安装的库
    if already_installed:
        print("\n已安装的库:")
        for pkg in already_installed:
            print(f"  ✓ {pkg}")
    
    # 安装其他依赖
    if final_to_install:
        print("\n正在安装缺失的标准库:")
        for pkg in final_to_install:
            print(f"  → {pkg}")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-i", TSINGHUA_SOURCE], check=True)
    else:
        print("\n所有标准库已安装完成!")
    
    # 安装PyTorch
    if torch_to_install:
        print("\n正在安装PyTorch (GPU版本):" if has_gpu else "\n正在安装PyTorch (CPU版本):")
        
        # 准备PyTorch安装命令
        torch_cmd = [sys.executable, "-m", "pip", "install"]
        torch_cmd.extend(torch_to_install)
        
        if has_gpu:
            torch_cmd.extend(["--extra-index-url", PYTORCH_SOURCE])
        else:
            # 为不带cpu后缀的torch包添加+cpu
            for i, req in enumerate(torch_cmd):
                if req.startswith('torch') and '+cpu' not in req:
                    version_spec = None
                    if '>=' in req:
                        name, version_spec = req.split('>=')
                        torch_cmd[i] = f"{name}+cpu>={version_spec}"
                    elif '==' in req:
                        name, version_spec = req.split('==')
                        torch_cmd[i] = f"{name}+cpu=={version_spec}"
                    else:
                        torch_cmd[i] = f"{req}+cpu"
            
            torch_cmd.extend(["--extra-index-url", PYTORCH_CPU_SOURCE])
        
        # 执行PyTorch安装
        for pkg in torch_cmd[2:]:
            if not pkg.startswith('--'):
                print(f"  → {pkg}")
        subprocess.run(torch_cmd, check=True)
    else:
        print("\nPyTorch已安装!")
    
    # 验证安装
    print("\n验证PyTorch安装...")
    verify_cmd = [
        sys.executable, "-c", 
        "import torch; print(f'PyTorch版本: {torch.__version__}'); "
        "print(f'CUDA可用: {torch.cuda.is_available()}'); "
        "print(f'GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
    ]
    subprocess.run(verify_cmd)
    
    print("\n所有依赖安装完成!")
    print("=" * 60)

if __name__ == "__main__":
    main() 