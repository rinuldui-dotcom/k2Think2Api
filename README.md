# k2Think2Api Deno版本

环境变量配置：

在 Deno Deploy 中设置以下环境变量：

CLIENT_API_KEYS: 

逗号分隔的有效 API 密钥列表（如：sk-key1,sk-key2）

MODELS_JSON: 

模型映射的 JSON 字符串（可选）


部署步骤：

将代码 main.ts 
在 Deno Deploy 控制台创建新项目
复制粘贴

设置环境变量
部署即可

主要功能保持不变：

/v1/models - 获取模型列表
/v1/chat/completions - 聊天完成接口
支持流式响应
K2Think API 集成

推理内容和答案内容的分离处理

Docker部署指南
快速开始
1. 克隆或准备项目文件
确保你有以下文件：

main.py (你的FastAPI应用代码)
Dockerfile
docker-compose.yml
requirements.txt
.dockerignore
models.json

2. 使用Docker Compose启动（推荐）
3. # 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

配置说明
API密钥配置
首次启动前，系统会自动生成 client_api_keys.json 文件。你也可以手动创建：

[
  "sk-talkai-your-custom-key-here",
  "sk-talkai-another-key-here"
]

使用方式
健康检查
curl http://localhost:8001/v1/models

聊天接口

curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-talkai-your-api-key" \
  -d '{
    "model": "k2-think",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": true
  }'




