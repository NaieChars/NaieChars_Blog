import type { FooterConfig } from "../types/config";

// 页脚配置
export const footerConfig: FooterConfig = {
	enable: true, // 是否启用Footer HTML注入功能
	customHtml: `
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:center;font-size:14px;">
    <img src="/备案图标.png" alt="公安备案图标" style="width:16px;height:16px;display:inline-block;vertical-align:middle;">
    <a href="https://beian.mps.gov.cn/#/query/webSearch?code=51010802033357" rel="noreferrer" target="_blank">川公网安备51010802033357号</a>
    <span style="color:#999;">|</span>
    <a href="https://beian.miit.gov.cn/" target="_blank" rel="noopener noreferrer">蜀ICP备2026040788号-2</a>
  </div>
`,
};
