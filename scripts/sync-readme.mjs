#!/usr/bin/env node
/**
 * 把 book.json 的进度渲染进 README 的 <!-- progress:start/end --> 块。
 *
 *   node scripts/sync-readme.mjs          # 写入
 *   node scripts/sync-readme.mjs --check  # 只比对，过期即退出 1
 *
 * README 里只保留这一份进度；侧边栏与部分导言页的章节列表也来自同一份 book.json。
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const START = '<!-- progress:start -->'
const END = '<!-- progress:end -->'

export function renderProgress() {
  const book = JSON.parse(readFileSync(join(ROOT, 'docs/.vitepress/book.json'), 'utf8'))
  const rows = book.parts.map((part) => {
    const sections = part.sections.filter((s) => s.num)
    const status = part.status === 'done' ? '已完成' : part.status === 'draft' ? '草稿' : '占位'
    return `| ${part.title} | ${status} | ${sections.length} |`
  })
  const notes = book.parts
    .filter((part) => part.note)
    .map((part) => `- **${part.title}**：${part.note}`)

  return [
    '（本节由 `npm run docs:sync` 从 `docs/.vitepress/book.json` 生成，勿手改。）',
    '',
    '正文按 `docs/00-preface/` → `docs/08-delivery/` 的顺序推进。当前状态：',
    '',
    '| 部分 | 状态 | 节数 |',
    '|---|---|---|',
    ...rows,
    ...(notes.length ? ['', '已完成部分的实验代码与数据：', '', ...notes] : []),
  ].join('\n')
}

export function replaceBlock(text) {
  const start = text.indexOf(START)
  const end = text.indexOf(END)
  if (start === -1 || end === -1 || end < start) {
    throw new Error(`README.md 缺少 ${START} / ${END} 标记`)
  }
  return text.slice(0, start + START.length) + '\n' + renderProgress() + '\n' + text.slice(end)
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href
if (isMain) {
  const file = join(ROOT, 'README.md')
  const text = readFileSync(file, 'utf8')
  const next = replaceBlock(text)
  if (process.argv.includes('--check')) {
    if (next !== text) {
      console.error('README 的进度块已过期：跑 `npm run docs:sync` 更新。')
      process.exit(1)
    }
    console.log('README 进度块：最新')
  } else {
    if (next !== text) writeFileSync(file, next)
    console.log(next === text ? 'README 进度块：无变化' : 'README 进度块：已更新')
  }
}
