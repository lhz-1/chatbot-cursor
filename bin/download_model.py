import os
import sys
import time
import glob

# 获取当前目录和父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def download_model(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    下载模型
    """
    print(f"准备下载模型: {model_id}")
    print("这可能需要一些时间，取决于您的网络速度...")
    print("下载开始，请耐心等待...")
    
    # 设置缓存目录
    cache_dir = os.path.join(parent_dir, "models_cache")
    os.environ["MODELSCOPE_CACHE"] = cache_dir
    
    try:
        # 使用 modelscope 下载
        from modelscope import snapshot_download
        
        # 显示下载信息
        start_time = time.time()
        print(f"开始下载 {model_id}，时间: {time.strftime('%H:%M:%S', time.localtime())}")
        
        # 基本的下载，无进度条
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        
        end_time = time.time()
        download_time = end_time - start_time
        
        # 显示下载完成信息
        print(f"下载完成！耗时: {download_time:.1f} 秒 ({download_time/60:.1f} 分钟)")
        print(f"模型保存在: {model_dir}")
        return True, model_dir
    
    except ImportError:
        print("未找到 modelscope 库，请安装: pip install modelscope")
        return False, None
    except Exception as e:
        print(f"模型下载失败: {e}")
        print("请检查网络连接或手动下载模型")
        return False, None

def check_model_exists(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    检查模型是否已经下载
    """
    # 设置缓存目录
    cache_dir = os.path.join(parent_dir, "models_cache")
    os.environ["MODELSCOPE_CACHE"] = cache_dir
    model_name = model_id.split("/")[-1]
    org_name = model_id.split("/")[0] if "/" in model_id else ""
    
    # 更全面地检查模型目录的各种可能位置
    potential_paths = [
        os.path.join(cache_dir, model_id),
        os.path.join(cache_dir, "hub", model_id),
        os.path.join(cache_dir, model_name),
        os.path.join(cache_dir, "models--" + model_id.replace("/", "--")),
        # HF 缓存风格
        os.path.join(cache_dir, org_name, model_name),
        # 模型直接在根目录
        os.path.join(parent_dir, model_name),
        # 本地目录中的任何包含 DeepSeek 或 1.5B 的目录
        *glob.glob(os.path.join(cache_dir, "**", "*DeepSeek*"), recursive=True),
        *glob.glob(os.path.join(parent_dir, "**", "*DeepSeek*"), recursive=True),
        *glob.glob(os.path.join(cache_dir, "**", "*1.5B*"), recursive=True),
        *glob.glob(os.path.join(parent_dir, "**", "*1.5B*"), recursive=True)
    ]
    
    # 打印所有检查的路径，帮助调试
    print("检查以下路径是否存在模型:")
    for i, path in enumerate(potential_paths):
        print(f"{i+1}. {path}")
        if os.path.exists(path):
            print(f"  - 路径存在!")
            # 检查特定文件是否存在
            if os.path.exists(os.path.join(path, "config.json")):
                print(f"  - 找到 config.json")
                
                # 检查模型文件
                if os.path.exists(os.path.join(path, "pytorch_model.bin")):
                    print(f"  - 找到 pytorch_model.bin")
                    return True, path
                elif os.path.exists(os.path.join(path, "model.safetensors")):
                    print(f"  - 找到 model.safetensors")
                    return True, path
                # 检查分片模型文件 (多文件模型)
                elif glob.glob(os.path.join(path, "pytorch_model-*.bin")):
                    print(f"  - 找到分片模型文件")
                    return True, path
                elif glob.glob(os.path.join(path, "model-*.safetensors")):
                    print(f"  - 找到分片safetensors文件")
                    return True, path
                elif glob.glob(os.path.join(path, "**", "*.bin"), recursive=True):
                    print(f"  - 在子目录中找到模型文件")
                    return True, path
    
    # 在当前环境中尝试直接从 transformers 加载模型
    try:
        from transformers import AutoConfig
        print("尝试直接使用transformers验证模型可访问性...")
        AutoConfig.from_pretrained(model_id)
        print(f"  - transformers可以直接访问模型 {model_id}")
        return True, "online" 
    except Exception as e:
        print(f"  - transformers无法直接访问模型: {e}")
    
    print("未找到模型文件")
    return False, None

if __name__ == "__main__":
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # 检查模型是否已下载
    exists, model_path = check_model_exists(model_id)
    
    if exists:
        print(f"模型已存在于: {model_path}")
        choice = input("是否重新下载模型? (y/n): ").strip().lower()
        if choice != 'y':
            print("使用已下载的模型。")
            sys.exit(0)
    
    # 下载模型
    success, _ = download_model(model_id)
    
    if success:
        print("模型下载完成，现在可以运行 bin/car_sales_bot.py 了")
    else:
        print("模型下载失败，请检查网络连接并重试") 