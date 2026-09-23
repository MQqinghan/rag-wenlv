/**
 * 对话页：连接 BFF WebSocket，发送提问并流式渲染 delta。
 */
const config = require('../../config');
const { ChatSocket } = require('../../utils/ws');

const app = getApp();

Page({
  data: {
    messages: [],   // { role: 'user'|'assistant', content, streaming?, error? }
    input: '',
    sending: false,
    sessionId: '',
    scrollTo: '',
  },

  onLoad() {
    this.socket = new ChatSocket({
      url: config.WS_URL,
      heartbeatMs: config.HEARTBEAT_MS,
      onFrame: (f) => this._onFrame(f),
      onOpen: () => this._ensureSession(),
      onClose: (code) => {
        if (code === 4401) {
          this._pushAssistant({ content: '登录态失效，请重新进入小程序或重新登录。', error: true });
        }
      },
    });
    this.socket.connect();
  },

  onUnload() {
    if (this.socket) this.socket.close();
  },

  _ensureSession() {
    if (!this.data.sessionId) {
      this.setData({
        sessionId: `mp_${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
      });
    }
  },

  onInput(e) {
    this.setData({ input: e.detail.value });
  },

  /** 发送提问 */
  onSend() {
    const text = (this.data.input || '').trim();
    if (!text || this.data.sending) return;
    this._ensureSession();

    const messages = this.data.messages.concat([
      { role: 'user', content: text },
      { role: 'assistant', content: '', streaming: true },
    ]);
    this.setData({ messages, input: '', sending: true }, () => this._scrollToEnd());

    this.socket.sendQuery({
      query: text,
      session_id: this.data.sessionId,
      user_id: app.globalData.userId || undefined,
      token: app.globalData.token || undefined,
      stream: config.STREAM,
    });
  },

  /** 停止生成 */
  onStop() {
    this.socket.sendStop(this.data.sessionId);
    this._finishStreaming();
  },

  // ---------------- 帧处理 ----------------

  _onFrame(frame) {
    const type = frame && frame.type;
    if (type === 'ready') {
      if (frame.session_id) this.setData({ sessionId: frame.session_id });
      return;
    }
    if (type === 'delta') {
      const d = frame.data || {};
      this._appendAssistantText(d.content || d.delta || '');
      return;
    }
    if (type === 'final') {
      const d = frame.data || {};
      if (d.answer) this._setAssistantText(d.answer);
      this._finishStreaming();
      return;
    }
    if (type === 'error') {
      const code = frame.code || 'error';
      const msg = frame.message || '服务异常';
      const hint = code === 'rate_limited' && frame.retry_after
        ? `（请 ${frame.retry_after} 秒后重试）` : '';
      this._setAssistantText(`[${code}] ${msg}${hint}`, true);
      this._finishStreaming();
      return;
    }
    // progress / stop 等其他帧：暂不渲染
  },

  _appendAssistantText(chunk) {
    if (!chunk) return;
    const idx = this.data.messages.length - 1;
    if (idx < 0) return;
    const key = `messages[${idx}].content`;
    this.setData({ [key]: (this.data.messages[idx].content || '') + chunk }, () => this._scrollToEnd());
  },

  _setAssistantText(text, isError) {
    const idx = this.data.messages.length - 1;
    if (idx < 0) return;
    this.setData({
      [`messages[${idx}].content`]: text,
      [`messages[${idx}].error`]: !!isError,
    }, () => this._scrollToEnd());
  },

  _pushAssistant(msg) {
    this.setData({ messages: this.data.messages.concat([msg]) }, () => this._scrollToEnd());
  },

  _finishStreaming() {
    const idx = this.data.messages.length - 1;
    const patch = { sending: false };
    if (idx >= 0) patch[`messages[${idx}].streaming`] = false;
    this.setData(patch);
  },

  _scrollToEnd() {
    const idx = this.data.messages.length - 1;
    this.setData({ scrollTo: idx >= 0 ? `msg-${idx}` : '' });
  },
});
