# 移动端小程序（微信小程序优先）—— 前端骨架

对接 `app/api/http/mobile_gateway.py`（移动端 BFF，端口 8002）。**本目录只含前端**，
业务逻辑全在服务端（服务端重、客户端轻）。

## 目录结构

```
mobile_miniprogram/
├── app.js / app.json / app.wxss      # 小程序入口（可选登录态）
├── config.js                          # 后端地址 / 鉴权开关 / 流式开关
├── project.config.json                # 开发者工具工程配置（appid 待替换）
├── sitemap.json
├── utils/ws.js                        # ChatSocket：连接/心跳/重连/帧分发
└── pages/chat/                        # 对话页（流式增量渲染）
    ├── chat.js / chat.wxml / chat.wxss / chat.json
```

## 通信协议（与 BFF 对齐）

发送（JSON 文本帧）：

```json
{ "type": "query", "query": "成都三日游怎么安排？", "session_id": "mp_xxx",
  "user_id": "可选", "token": "可选(JWT)", "stream": true }
{ "type": "ping" }                       // 心跳保活，服务端回 {"type":"pong"}
{ "type": "stop", "session_id": "mp_xxx" } // 停止生成
```

接收：

```json
{ "type": "ready",    "session_id": "..." }
{ "type": "progress", "session_id": "...", "data": {...} }
{ "type": "delta",    "session_id": "...", "data": { "content": "增量文本" } }
{ "type": "final",    "session_id": "...", "data": { "answer": "完整答案" } }
{ "type": "stop" | "pong" }
{ "type": "error", "code": "unauthorized|rate_limited|INJECTION_BLOCKED|UPSTREAM_ERROR",
  "message": "...", "retry_after": 1 }
```

## 本地联调

1. 启动查询服务：`query_server`（8001，主服务）。
2. 启动网关：`python app/api/http/mobile_gateway.py`（8002）。
3. 微信开发者工具导入本目录 → 详情 → 本地设置 → 勾选
   **「不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书」**（因本地为 `ws://`）。
4. 保持 `config.js` 的 `AUTH_ENABLE=false` 与后端 `AUTH_ENABLE=false` 一致，即可直接对话。
5. 若要联调登录：后端置 `AUTH_ENABLE=true` + `JWT_SECRET=<≥32字节随机串>` +
   `AUTH_MOCK_OPENID=dev-openid`（免真实微信凭据），前端 `config.js` 置 `AUTH_ENABLE=true`。

## ⚠️ 上线待办（依赖主人侧资源，本次「下不表」）

| # | 事项 | 说明 | 归属 |
|---|---|---|---|
| 1 | 小程序 AppID | 替换 `project.config.json` 的 `appid`；`WX_APPID/WX_SECRET` 写入服务端 `.env` | 主人 |
| 2 | 域名备案 + HTTPS | 域名需完成 ICP 备案，配置 HTTPS 证书 | 主人 / 运维 |
| 3 | WSS 接入 | `mobile_gateway` 前置 Nginx/网关终结 TLS，前端 `config.js` 改 `wss://<域名>/ws/chat` | 运维 |
| 4 | 服务器域名白名单 | 微信公众平台 → 开发设置 → 服务器域名 添加 `wss://<域名>` 与 `https://<域名>` | 主人 |
| 5 | 真机测试 | 开发者工具 + 真机预览，验证流式/鉴权/限流 | 主人 |
| 6 | 图片入口 | 对话页尚未接 `wx.chooseMedia`（图片）；后端导入侧已有 OCR 能力，查询侧接口待定 | 待设计 |

> 结论：前端骨架已就位（可读、可导入开发者工具），**备案域名与真机上线无法在本地完成**，
> 需主人提供 AppID / 备案域名后继续。
