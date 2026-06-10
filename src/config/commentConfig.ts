import type { CommentConfig } from "../types/config";
import { SITE_LANG } from "./siteConfig";

// 评论系统配置
export const commentConfig: CommentConfig = {
	enable: true, // 启用评论功能。当设置为 false 时，评论组件将不会显示在文章区域。
	system: "giscus", // 评论系统选择: "twikoo" | "giscus"
	twikoo: {
		envId: "https://twikoo.vercel.app",
		lang: "zh-CN",
	},
	giscus: {
		repo: "NaieChars/NaieChars_Blog",
		repoId: "R_kgDOS08bkg",
		category: "Announcements",
		categoryId: "DIC_kwDOS08bks4C-3Tn",
		mapping: "pathname",
		strict: "0",
		reactionsEnabled: "1",
		emitMetadata: "0",
		inputPosition: "bottom",
		theme: "preferred_color_scheme",
		lang: "zh-CN",
		loading: "lazy",
	},
};
