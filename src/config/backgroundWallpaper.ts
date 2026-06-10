import type { FullscreenWallpaperConfig } from "../types/config";

export const fullscreenWallpaperConfig: FullscreenWallpaperConfig = {
	enable: true,
	src: {
		desktop: [
			"https://honaisu.cd.bcebos.com/naie-char/1.webp",
		],
		mobile: [
			"/assets/mobile-banner/1.webp",
		],
	},
	position: "center",
	carousel: {
		enable: true,
		interval: 5,
	},
	zIndex: -1,
	opacity: 1,
	blur: 0,
	switchable: true,
	overlay: {
		opacity: 1, // 壁纸不透明度，0-1
		blur: 0, // 背景模糊半径（px）
		cardOpacity: 0.45, // 卡片不透明度，0-1
		switchable: {
			opacity: true,
			blur: true,
			cardOpacity: true,
		},
	},
	fullscreen: {
		switchable: {
			opacity: true,
			blur: true,
		},
	},
};
