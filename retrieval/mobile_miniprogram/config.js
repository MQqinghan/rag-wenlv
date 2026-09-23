/**
 * 移动端配置（微信小程序）。
 *
 * ⚠️ 微信小程序要求所有网络请求走 https / wss，且域名需在
 *    「微信公众平台 → 开发管理 → 开发设置 → 服务器域名」中完成备案与配置。
 *    开发阶段可在微信开发者工具里勾选「不校验合法域名…」直连本机。
 */
module.exports = {
  // 生产：wss://<备案域名>/ws/chat
  // 开发：ws://127.0.0.1:8002/ws/chat（开发者工具勾选「不校验合法域名」）
  WS_URL: 'ws://127.0.0.1:8002/ws/chat',

  // BFF HTTP 基址（/auth/login 用）
  API_BASE: 'http://127.0.0.1:8002',

  // 是否启用登录：需与后端 AUTH_ENABLE 一致（后端 false 时本项也置 false）
  AUTH_ENABLE: false,

  // 是否走流式（后端 T11 已支持；false 则一次性返回）
  STREAM: true,

  // 心跳间隔（毫秒）——保活，防中间层 idle 断连
  HEARTBEAT_MS: 30000,
};
