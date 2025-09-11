# k2Think2Api Deno版本

环境变量配置：
在 Deno Deploy 中设置以下环境变量：

CLIENT_API_KEYS: 逗号分隔的有效 API 密钥列表（如：sk-key1,sk-key2）
MODELS_JSON: 模型映射的 JSON 字符串（可选）

部署步骤：

将代码 main.ts 复制
在 Deno Deploy 控制台创建新项目
上传文件或连接 GitHub 仓库
设置环境变量
部署即可

主要功能保持不变：

/v1/models - 获取模型列表
/v1/chat/completions - 聊天完成接口
支持流式响应
K2Think API 集成
推理内容和答案内容的分离处理

