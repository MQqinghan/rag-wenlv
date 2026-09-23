/**
 * ChatSocket —— 与 mobile_gateway `/ws/chat` 通信的 WebSocket 封装。
 *
 * 协议（与 app/api/http/mobile_gateway.py 对齐）：
 *   发送：{ type:'query', query, session_id?, user_id?, token?, stream? }
 *   接收：{ type:'ready'|'progress'|'delta'|'final'|'stop', session_id, data }
 *         { type:'error', code, message, retry_after? }
 *
 * 说明：小程序 wx.connectSocket 同一时刻只能维持少量连接，故本类按「单连接复用」设计，
 *      断线自动重连（指数退避，上限 15s），重连后由业务侧重发未完成的问题。
 */
class ChatSocket {
  /**
   * @param {object} opts
   * @param {string} opts.url         ws/wss 地址
   * @param {number} [opts.heartbeatMs] 心跳间隔
   * @param {function} [opts.onFrame]  收到业务帧回调 (frame) => void
   * @param {function} [opts.onOpen]   连接建立回调
   * @param {function} [opts.onClose]  连接关闭回调 (code, reason)
   */
  constructor(opts) {
    this.url = opts.url;
    this.heartbeatMs = opts.heartbeatMs || 30000;
    this.onFrame = opts.onFrame || function () {};
    this.onOpen = opts.onOpen || function () {};
    this.onClose = opts.onClose || function () {};

    this.task = null;         // SocketTask
    this.connected = false;
    this.manualClose = false;
    this._retry = 0;
    this._heartbeatTimer = null;
    this._queue = [];         // 未连接时的发送队列
  }

  connect() {
    if (this.task || this.connected) return;
    this.manualClose = false;

    this.task = wx.connectSocket({ url: this.url });

    this.task.onOpen(() => {
      this.connected = true;
      this._retry = 0;
      this._startHeartbeat();
      // 补发排队消息
      while (this._queue.length) {
        this._send(this._queue.shift());
      }
      this.onOpen();
    });

    this.task.onMessage((res) => {
      let frame = null;
      try {
        frame = JSON.parse(res.data);
      } catch (e) {
        frame = { type: 'error', code: 'BAD_JSON', message: '服务端返回非 JSON' };
      }
      if (frame && frame.type === 'pong') return; // 心跳响应，不派发
      this.onFrame(frame);
    });

    this.task.onClose((res) => {
      this.connected = false;
      this.task = null;
      this._stopHeartbeat();
      this.onClose(res && res.code, res && res.reason);
      if (!this.manualClose) this._scheduleReconnect();
    });

    this.task.onError(() => {
      // onError 后通常伴随 onClose，重连交给 onClose 统一处理
      this.connected = false;
    });
  }

  close() {
    this.manualClose = true;
    this._stopHeartbeat();
    if (this.task) {
      this.task.close({ code: 1000, reason: 'client close' });
      this.task = null;
    }
    this.connected = false;
  }

  /**
   * 发送一次提问。
   * @param {object} payload { query, session_id?, user_id?, token?, stream? }
   */
  sendQuery(payload) {
    const frame = Object.assign({ type: 'query' }, payload);
    this._send(frame);
  }

  /** 请求停止当前生成（后端 /stop 由 BFF 透传；此处仅发信号帧） */
  sendStop(sessionId) {
    this._send({ type: 'stop', session_id: sessionId });
  }

  _send(frame) {
    const text = JSON.stringify(frame);
    if (!this.connected || !this.task) {
      this._queue.push(frame);
      this.connect();
      return;
    }
    this.task.send({ data: text });
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this._heartbeatTimer = setInterval(() => {
      if (this.connected && this.task) {
        this.task.send({ data: JSON.stringify({ type: 'ping' }) });
      }
    }, this.heartbeatMs);
  }

  _stopHeartbeat() {
    if (this._heartbeatTimer) {
      clearInterval(this._heartbeatTimer);
      this._heartbeatTimer = null;
    }
  }

  _scheduleReconnect() {
    this._retry += 1;
    const delay = Math.min(15000, 1000 * Math.pow(2, Math.min(this._retry, 4)));
    setTimeout(() => {
      if (!this.manualClose) this.connect();
    }, delay);
  }
}

module.exports = { ChatSocket };
