# QQBot

基于 NapCat、OneBot 11 反向 WebSocket 和 OpenAI 兼容接口的 QQ 聊天机器人。

## 当前功能

- 通过 WebSocket 接收 NapCat 推送的私聊和群聊消息
- 自动忽略机器人自己发送的消息
- 私聊消息直接处理，群聊消息仅在 @ 机器人时处理
- 调用 OpenAI 兼容的聊天接口生成回复
- 按会话保留最近 20 条消息上下文
- 使用 QQ 引用消息进行回复
- 将 OneBot 消息转换为内部统一模型，并保留暂未支持的消息段

当前监听流程只会处理消息事件；撤回和群文件事件已经定义了内部模型，但还没有接入业务处理流程。

## 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- 已运行并配置 OneBot 11 反向 WebSocket 的 NapCat
- 一个 OpenAI 兼容的 LLM API

## 安装

```bash
uv sync
```

复制环境变量模板：

```bash
cp .env.example .env
```

然后编辑 `.env`：

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `NAPCAT_WS_URL` | NapCat 反向 WebSocket 地址 | `ws://127.0.0.1:3001` |
| `NAPCAT_TOKEN` | NapCat WebSocket 鉴权 Token | 必填 |
| `LLM_API_KEY` | LLM API 密钥 | 必填 |
| `LLM_BASE_URL` | OpenAI 兼容 API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 使用的模型名称 | `deepseek-flash` |

不要将 `.env` 或 API 密钥提交到 Git；`.env.example` 可以提交并用于说明配置格式。

## 运行

启动消息监听器：

```bash
uv run python -m qqbot.listen
```

正常连接后会输出：

```text
已连接 NapCat
```

程序会打印收到的消息，并将符合路由条件的消息交给 LLM。群聊消息必须 @ 当前机器人，私聊消息则会直接处理。

## 项目结构

```text
src/qqbot/
├── agent.py       # LLM 客户端和会话上下文
├── context.py     # 内存中的会话历史
├── listen.py      # WebSocket 监听入口
├── normalizer.py  # OneBot 事件归一化
├── router.py      # 消息路由规则
├── schema.py      # 内部事件和消息模型
├── system.md      # Agent 系统提示词
└── transport.py   # NapCat 消息发送
```

## 开发

依赖和锁文件由 uv 管理。修改依赖后同步环境：

```bash
uv sync
```

当前仓库尚未配置自动化测试；启动监听器前请确认 NapCat 已连接到与 `NAPCAT_WS_URL` 相同的反向 WebSocket 地址。
