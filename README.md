二手车销售顾问机器人
一个基于大型语言模型的智能二手车销售顾问聊天机器人，为客户提供专业的二手车咨询服务。

### 功能特点
- 🚗 专业的二手车销售咨询
- 💬 自然流畅的对话体验
- 📊 内置二手车数据库
- 🧠 基于 DeepSeek 先进 AI 模型
- 💻 同时支持 CPU 和 GPU 运行
- 🌐 简洁易用的网页界面

### 安装要求
- Python 3.10
- PyTorch 2.0+
- 至少 4GB RAM (推荐 8GB+)
- 支持 CUDA 的 NVIDIA GPU (可选，但推荐用于更快的响应速度)
- 约 4GB 的磁盘空间用于模型存储

### 安装步骤
1. 克隆仓库
```bash
git clone https://github.com/lhz-1/chatbot-cursor.git
cd chatbot-cursor
```
2. 安装依赖
```bash
pip install -r requirements.txt
```
3. 下载模型
```bash
python bin/download_model.py
```
4. 运行机器人
```bash
python bin/car_sales_bot_api.py
```
在浏览器打开 `index.html`。

### 项目结构
```text
car_sales_bot/
├── bin/
│   ├── car_sales_bot_api.py # 主 API 文件
│   ├── download_model.py # 模型下载脚本
├── frontend/
│   # vue3 (还在调试暂不支持)
├── index.html # 前端界面
├── requirements.txt # 依赖列表
├── README.md # 项目说明
```

### 技术细节
- **AI 模型**: DeepSeek-R1-Distill-Qwen-1.5B，这是一个轻量级但性能优秀的大语言模型
- **框架**: 使用 Transformers 库加载和运行 AI 模型
- **前端**: 使用 vue（还在调试暂不支持）
- **知识库**: 内置了常见二手车型的基本信息
- **设备适配**: 自动检测并利用 GPU 加速，也支持纯 CPU 运行

### 自定义与扩展
- 添加更多汽车品牌和型号: 编辑 `car_database` 字典
- 修改系统提示: 编辑 `system_prompt` 变量
- 使用不同模型: 修改 `model_id` 变量并重新下载模型

### 许可证
本项目采用 MIT 许可证。DeepSeek 模型使用需遵循其原始许可条款。

### 致谢
- DeepSeek AI 提供的优秀开源模型
- Hugging Face Transformers 提供的模型加载与推理框架
- Gradio 提供的易用界面开发工具

---

希望这个二手车销售顾问机器人能为您提供有用的服务！如有问题或建议，欢迎提交 Issue 或 Pull Request。