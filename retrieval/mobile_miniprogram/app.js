/**
 * 小程序入口：登录态（可选）+ 全局配置。
 */
const config = require('./config');

App({
  globalData: {
    token: '',
    userId: '',
    openid: '',
  },

  onLaunch() {
    if (config.AUTH_ENABLE) {
      this.login().catch((err) => {
        console.error('[app] 登录失败', err);
      });
    }
  },

  /**
   * 微信登录：wx.login() → code → BFF /auth/login → 自签 JWT。
   * 后端未接微信凭据时，可用 AUTH_MOCK_OPENID 联调（见 mobile_gateway 文档串）。
   * @returns {Promise<{access_token:string, refresh_token:string, user_id:string}>}
   */
  login() {
    return new Promise((resolve, reject) => {
      wx.login({
        success: (res) => {
          if (!res.code) {
            reject(new Error('wx.login 未返回 code'));
            return;
          }
          wx.request({
            url: `${config.API_BASE}/auth/login`,
            method: 'POST',
            header: { 'content-type': 'application/json' },
            data: { code: res.code },
            success: (r) => {
              if (r.statusCode === 200 && r.data && r.data.access_token) {
                this.globalData.token = r.data.access_token;
                this.globalData.userId = r.data.user_id || '';
                resolve(r.data);
              } else {
                reject(new Error((r.data && r.data.message) || `登录失败(${r.statusCode})`));
              }
            },
            fail: (e) => reject(new Error(e.errMsg || '网络异常')),
          });
        },
        fail: (e) => reject(new Error(e.errMsg || 'wx.login 失败')),
      });
    });
  },
});
