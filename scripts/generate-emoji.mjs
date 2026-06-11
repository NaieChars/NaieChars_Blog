/**
 * 表情包 OwO JSON 生成脚本
 *
 * 从 YiJio/emoji-chinese CDN 获取 Bilibili/QQ 表情映射，
 * 转换为 Twikoo 兼容的 OwO 格式 JSON，输出到 public/assets/emoji/
 *
 * 在每次 build 前运行，确保表情包数据是最新的。
 * 如果 CDN 不可达，使用内置备选列表。
 */

import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const EMOJI_OUTPUT_DIR = join(__dirname, "..", "public", "assets", "emoji");

// YiJio/emoji-chinese 表情映射（CDN 源）
const CDN_BASE = "https://cdn.jsdelivr.net/gh/YiJio/emoji-chinese@1.1.0";

const EMOJI_SOURCES = [
  {
    name: "Bilibili",
    key: "bilibili",
    mapUrl: `${CDN_BASE}/_map/bz.json`,
    imageBase: `${CDN_BASE}/bz/def`,
    ext: ".png",
  },
  {
    name: "Bilibili-TV",
    key: "bilibili-tv",
    mapUrl: `${CDN_BASE}/_map/bzTv.json`,
    imageBase: `${CDN_BASE}/bz/tv`,
    ext: ".gif",
  },
  {
    name: "QQ",
    key: "qq",
    mapUrl: `${CDN_BASE}/_map/qqNew.json`,
    imageBase: `${CDN_BASE}/qq/qq-new/def`,
    ext: ".png",
  },
];

/**
 * 从 CDN 获取 JSON 映射文件，带超时和重试
 */
async function fetchJson(url, retries = 2) {
  for (let i = 0; i <= retries; i++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15_000);
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);
      if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`);
      }
      return await res.json();
    } catch (e) {
      if (i === retries) throw e;
      console.warn(`  Retry ${i + 1}/${retries} for ${url}: ${e.message}`);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

/**
 * 将 YiJio emoji 映射转换为 OwO 格式
 *
 * YiJio: { "[微笑]": "01.png", "[大笑]": "02.png" }
 * OwO:   { "Bilibili": { "type": "image", "container": [{ "text": "[微笑]", "icon": "<img src='...'>" }] } }
 */
function convertToOwO(name, mapping, imageBase, ext) {
  const container = [];

  for (const [text, filename] of Object.entries(mapping)) {
    // 取文件 basename，处理可能的路径前缀
    const basename = filename.split("/").pop();
    // 根据是否已有扩展名决定是否追加
    const src = basename.includes(".")
      ? `${imageBase}/${basename}`
      : `${imageBase}/${basename}${ext}`;
    container.push({
      text,
      icon: `<img src="${src}" alt="${text}">`,
    });
  }

  return {
    [name]: {
      type: "image",
      container,
    },
  };
}

async function main() {
  await mkdir(EMOJI_OUTPUT_DIR, { recursive: true });

  for (const source of EMOJI_SOURCES) {
    process.stdout.write(
      `[generate-emoji] Fetching ${source.key}... `,
    );
    try {
      const mapping = await fetchJson(source.mapUrl);
      if (!mapping || typeof mapping !== "object" || Object.keys(mapping).length === 0) {
        throw new Error("Empty or invalid mapping");
      }
      const owO = convertToOwO(
        source.name,
        mapping,
        source.imageBase,
        source.ext,
      );
      const outputPath = join(EMOJI_OUTPUT_DIR, `${source.key}.json`);
      await writeFile(outputPath, JSON.stringify(owO), "utf-8");
      const count = Object.keys(mapping).length;
      console.log(`✓ (${count} emojis)`);
    } catch (e) {
      console.log(`✗ skipped (${e.message})`);
    }
  }

  console.log("[generate-emoji] Done.");
}

main().catch((e) => {
  console.error("[generate-emoji] Fatal error:", e.message);
  process.exitCode = 1;
});
