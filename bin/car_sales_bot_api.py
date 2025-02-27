import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
from contextlib import contextmanager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

# 导入下载模块
from download_model import check_model_exists, download_model

# 设置环境变量和路径
os.environ["MODELSCOPE_CACHE"] = os.path.join(parent_dir, "models_cache")

# 定义请求和响应模型
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

# 创建 FastAPI 应用
app = FastAPI(title="车车 - 二手车销售顾问")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarSalesBot:
    def __init__(self):
        self.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        # 添加联系方式信息
        self.contact_info = {
            "地点": "北京市昌平区西二旗地铁口车车总店",
            "电话": "400-888-8888",
            "销售顾问": "张先生",
            "工作时间": "周一至周日 9:00-21:00"
        }
        
        # 汽车销售相关的知识库
        self.car_database = {
            "奔驰C级": {"价格": "28-35万", "年份": "2020", "里程": "3万公里", "状况": "极好", "颜色": "银色"},
            "宝马3系": {"价格": "25-32万", "年份": "2019", "里程": "4.5万公里", "状况": "良好", "颜色": "黑色"},
            "奥迪A4L": {"价格": "22-30万", "年份": "2019", "里程": "5万公里", "状况": "良好", "颜色": "白色"},
            "大众帕萨特": {"价格": "15-20万", "年份": "2018", "里程": "6万公里", "状况": "良好", "颜色": "灰色"},
            "丰田凯美瑞": {"价格": "16-22万", "年份": "2019", "里程": "4万公里", "状况": "极好", "颜色": "红色"},
            "本田雅阁": {"价格": "18-24万", "年份": "2020", "里程": "3.8万公里", "状况": "极好", "颜色": "蓝色"}
        }
        
        # 更新系统提示
        self.system_prompt = """# Role: 二手车销售顾问张先生

## Profile
- language: 中文
- description: 你是北京市昌平区西二旗地铁口车车总店的二手车销售顾问张先生。你的任务是根据客户的需求提供车辆信息，推荐适合的车型，并确保回答自然流畅、简洁专业，同时提供试驾等后续服务。

## Skills
- 提供具体车型的详细信息，如价格、车况、里程等
- 专业分析车辆的优缺点，帮助客户做出决策
- 有效地与客户沟通，保持热情专业的销售态度
- 推荐符合客户需求的车型，并邀约客户到店看车试驾
- 提供购车流程、付款方式和联系方式

## Background
你在车车总店担任销售顾问，熟悉各种车型，能够为客户提供专业的购车建议。

## Goals
1. 根据客户的需求和预算推荐合适的车型
2. 提供清晰的车辆信息，帮助客户做出选择
3. 通过电话或面对面沟通，确保客户对购车流程有清晰了解
4. 邀请客户到店看车，并提供试驾机会

## OutputFormat
- 用中文简洁、自然地与客户对话
- 对于车型具体信息，可以提供价格、车况、特点
- 对于非数据库中的车型，提供市场分析及建议
- 回答需突出车辆的优势，确保专业性

## Rules
1. 必须使用中文回答，不得使用“数据库”或“信息”这些词。
2. 强调车辆的具体特点、价格、车况，避免提及“几种车型”之类的笼统说法。
3. 不要提及“数据库”，要专注于展示每辆车的实际情况。
4. 保持热情、专业的销售语气，尽量为客户提供合适的购车方案。
5. 对于不在数据库中的车型，需提供大致的价格范围，并提醒客户实地看车。

## Workflows
1. 当客户询问车辆时，提供详细的车况、价格、里程等信息。
2. 如客户没有明确车型，主动询问客户的预算和需求，推荐合适的车型。
3. 提供购车建议，阐述每款车的优缺点，帮助客户做出决策。
4. 如果是非数据库中的车型，根据市场情况和经验，给出价格区间和推荐意见。
5. 每次对话都要明确提供销售顾问的联系方式，鼓励客户到店看车、试驾。

## Init
你好，我是张先生，欢迎来到车车总店！如果你有任何购车问题，随时告诉我，我会根据你的需求为你推荐最适合的车型。我们店里有很多车辆，价格合理，车况一流，也可以带你试驾。你有什么特别关注的车型吗？如果不确定，我们可以根据你的预算和需求来推荐。

"""

    @contextmanager
    def error_handler(self):
        """错误处理上下文管理器"""
        try:
            yield
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU内存不足")
            if hasattr(self, 'model'):
                self.model.cpu()  # 将模型移到CPU
            torch.cuda.empty_cache()  # 清理GPU内存
            raise HTTPException(status_code=503, detail="GPU内存不足，请稍后重试")
        except Exception as e:
            logger.error(f"发生错误: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

    async def initialize(self):
        """异步初始化模型"""
        if self.initialized:
            return

        with self.error_handler():
            # 检查模型是否已下载
            exists, model_path = check_model_exists(self.model_id)
            
            if not exists:
                logger.info(f"模型 {self.model_id} 未找到，开始下载...")
                success, model_path = download_model(self.model_id)
                if not success:
                    raise Exception("模型下载失败")
            
            logger.info(f"从 {model_path} 加载模型...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 加载模型
            try:
                import accelerate
                device_map = {"": 0} if self.device == "cuda" else "auto"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True
                )
            except ImportError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                if self.device == "cuda":
                    self.model = self.model.to("cuda")
            
            self.initialized = True
            logger.info("模型初始化完成")

    async def generate_response(self, messages: List[Message]) -> str:
        """生成回复"""
        with self.error_handler():
            # 确保模型已初始化
            if not self.initialized:
                await self.initialize()
            
            # 检查是否是直接询问联系方式
            current_message = messages[-1].content.strip()
            if current_message == "联系方式":
                return f"销售顾问：{self.contact_info['销售顾问']}\n电话：{self.contact_info['电话']}"
            
            # 构建输入文本
            prompt = ""
            formatted_system_prompt = self.system_prompt.format(
                car_database=self.car_database,
                contact_info=self.contact_info
            )
            prompt += f"<|system|>\n{formatted_system_prompt}\n"
            
            # 添加对话历史
            for msg in messages:
                if msg.role == "user":
                    prompt += f"<|user|>\n{msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"<|assistant|>\n{msg.content}\n"
            
            prompt += "<|assistant|>\n"
            
            # 生成回复
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                start_time = time.time()
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_length=2048,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=0.7,
                        repetition_penalty=1.1,
                        do_sample=True
                    )
                
                end_time = time.time()
                logger.info(f"生成耗时: {end_time - start_time:.2f}秒")
                
                # 解码输出
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # 提取回复，确保只返回最后一个回复
                assistant_parts = generated_text.split("<|assistant|>")
                if len(assistant_parts) > 1:
                    response = assistant_parts[-1].strip()
                    # 移除可能的思考过程（以"让我"、"我需要"、"首先"等开头的内容）
                    response_lines = response.split('\n')
                    cleaned_lines = [line for line in response_lines 
                                   if not any(line.strip().startswith(prefix) for prefix in 
                                            ["让我", "我需要", "首先", "接下来", "然后", "最后", "总结"])]
                    response = '\n'.join(cleaned_lines)
                else:
                    response = generated_text[len(prompt):].strip()
                
                return response.strip()
                
            except Exception as e:
                logger.error(f"生成回复时出错: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"生成回复失败: {str(e)}")

# 创建机器人实例
bot = CarSalesBot()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    try:
        response = await bot.generate_response(request.messages)
        return ChatResponse(response=response)
    except HTTPException as e:
        return ChatResponse(response="", error=str(e.detail))
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        return ChatResponse(response="", error=f"服务器错误: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": bot.initialized}

def main():
    """启动服务器"""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main() 