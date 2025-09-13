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

# Docker部署指南

## 快速开始

### 1. 克隆或准备项目文件

确保你有以下文件：
- `main.py` (你的FastAPI应用代码)
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.dockerignore`
- `models.json`

### 2. 使用Docker Compose启动

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 配置说明

### API密钥配置

首次启动前，系统会自动生成 `client_api_keys.json` 文件。你也可以手动创建：

```json
[
  "sk-talkai-your-custom-key-here",
  "sk-talkai-another-key-here"
]
```

### 模型配置

`models.json` 文件定义了可用的模型映射：

```json
{
  "k2-think": "MBZUAI-IFM/K2-Think",
  "deepseek-r1": "MBZUAI-IFM/K2-Think"
}
```

## 使用方式

### 健康检查

```bash
curl http://localhost:8001/v1/models
```

### 聊天接口

```bash
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
```

## 环境变量

可以通过环境变量进行配置：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| PORT | 8001 | 服务端口 |
| HOST | 0.0.0.0 | 绑定地址 |

在 `docker-compose.yml` 中添加：

```yaml
environment:
  - PORT=8001
  - HOST=0.0.0.0
```

## 数据持久化

Docker Compose配置了以下卷挂载：

- `./client_api_keys.json` - API密钥配置
- `./models.json` - 模型配置
- `./data` - 应用数据目录

## 日志管理

```bash
# 查看实时日志
docker-compose logs -f talkai-adapter

# 查看最近的日志
docker-compose logs --tail=100 talkai-adapter
```

## 更新部署

```bash
# 停止当前服务
docker-compose down

# 重新构建镜像
docker-compose build

# 启动新版本
docker-compose up -d
```

## 故障排除

### 1. 端口被占用

修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8002:8001"  # 将宿主机端口改为8002
```

### 2. 权限问题

确保配置文件有正确的权限：

```bash
chmod 644 client_api_keys.json models.json
chmod 755 data/
```

### 3. 容器无法启动

检查日志：

```bash
docker-compose logs talkai-adapter
```

### 4. 依赖安装失败

如果需要添加系统依赖，修改 `Dockerfile`：

```dockerfile
RUN apt-get update && apt-get install -y \
    gcc \
    curl \  # 添加新的依赖
    && rm -rf /var/lib/apt/lists/*
```

## 生产环境建议

1. **使用反向代理**：通过Nginx或Traefik进行反向代理
2. **SSL/TLS**：配置HTTPS证书
3. **监控**：添加Prometheus监控
4. **日志**：配置日志收集和轮转
5. **资源限制**：在docker-compose.yml中添加资源限制

```yaml
deploy:
  resources:
    limits:
      memory: 512M
      cpus: '0.5'
```

## 多环境部署

创建不同的docker-compose文件：

- `docker-compose.dev.yml` - 开发环境
- `docker-compose.prod.yml` - 生产环境

```bash
# 使用特定环境配置
docker-compose -f docker-compose.prod.yml up -d
```




