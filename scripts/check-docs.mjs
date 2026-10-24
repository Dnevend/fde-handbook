#!/usr/bin/env node
/**
 * 文档一致性检查。零依赖，只读。
 *
 *   node scripts/check-docs.mjs
 *
 * 覆盖：book.json ↔ 文件/frontmatter、部分导言页的 ChapterList、README 进度块、
 * 图号与文件名、引用编号与锚点、占位页形状、阅读路径、禁用词。
 * 退出码 0 = 无 error（warning 不影响退出码）。
 */
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, dirname, basename, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { replaceBlock } from './sync-readme.mjs'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const DOCS = join(ROOT, 'docs')
const BANNED = ['待实测', '液压站', '3 号线', '0-0-book-map']
const errors = []
const warnings = []

const err = (file, msg) => errors.push(`${rel(file)}: ${msg}`)
const warn = (file, msg) => warnings.push(`${rel(file)}: ${msg}`)
const rel = (file) => relative(ROOT, file) || '.'

const book = JSON.parse(readFileSync(join(DOCS, '.vitepress/book.json'), 'utf8'))
const read = (file) => readFileSync(file, 'utf8')

function docFile(route) {
  if (!route.startsWith('/')) return null
  if (route.endsWith('/')) return join(DOCS, route.slice(1), 'index.md')
  return join(DOCS, `${route.slice(1)}.md`)
}

function frontmatterTitle(text) {
  const block = text.match(/^---\n([\s\S]*?)\n---/)
  return block?.[1].match(/^title:\s*(.+)$/m)?.[1].trim()
}

function walkMarkdown(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (name === '.vitepress' || name === 'node_modules') continue
    const st = statSync(full)
    if (st.isDirectory()) walkMarkdown(full, out)
    else if (name.endsWith('.md')) out.push(full)
  }
  return out
}

// 1. book.json ↔ 文件、frontmatter
const sections = new Map()
for (const part of book.parts) {
  const index = docFile(`/${part.dir}/`)
  if (!existsSync(index)) {
    err(index, `book.json 的部分 ${part.id} 找不到导言页`)
    continue
  }
  if (!read(index).includes(`<ChapterList part="${part.id}" />`)) {
    err(index, `缺少 <ChapterList part="${part.id}" />`)
  }
  for (const section of part.sections) {
    sections.set(section.num, section)
    const file = docFile(section.path)
    if (!file || !existsSync(file)) {
      err(ROOT, `book.json 的 ${section.num ?? '导言'} 指向不存在的文件：${section.path}`)
      continue
    }
    const title = frontmatterTitle(read(file))
    if (title !== section.title) {
      err(file, `frontmatter title「${title ?? '缺失'}」与 book.json「${section.title}」不一致`)
    }
    if (section.status === 'placeholder') {
      const text = read(file)
      const lines = text.trim().split('\n').length
      if (lines > 25 || !text.includes('要回答的问题')) {
        err(file, '占位页应为「标题 + 要回答的问题」的短页（≤25 行）')
      }
    }
  }
}

// 2. README 进度块
const readme = join(ROOT, 'README.md')
if (!existsSync(readme)) err(readme, 'README.md 不存在')
else if (replaceBlock(read(readme)) !== read(readme)) {
  err(readme, '进度块已过期，跑 `npm run docs:sync`')
}

// 3. 反向检查：正文目录里不能有未登记的 Markdown（首页除外）
const registered = new Set([join(DOCS, 'index.md')])
for (const part of book.parts) registered.add(docFile(`/${part.dir}/`))
for (const section of sections.values()) registered.add(docFile(section.path))
for (const file of walkMarkdown(DOCS)) {
  if (!registered.has(file)) err(file, '未登记的 Markdown：请写进 book.json，或从正文目录移走')
}

// 3. 图：编号、文件、图源
const figureIds = new Map()
const usedSvgs = new Set()
const mdFiles = walkMarkdown(DOCS)
for (const file of mdFiles) {
  const text = read(file)
  for (const m of text.matchAll(/<Figure\s+([\s\S]*?)\/>/g)) {
    const attrs = Object.fromEntries([...m[1].matchAll(/(\w+)="([^"]*)"/g)].map((a) => [a[1], a[2]]))
    const { id, src } = attrs
    if (!id || !src) {
      err(file, 'Figure 缺少 id 或 src')
      continue
    }
    if (figureIds.has(id)) err(file, `图号 ${id} 重复（另见 ${rel(figureIds.get(id))}）`)
    else figureIds.set(id, file)

    const svg = join(DOCS, 'public', src.replace(/^\//, ''))
    if (!existsSync(svg)) {
      err(file, `Figure src 不存在：${src}`)
      continue
    }
    const name = basename(svg, '.svg')
    usedSvgs.add(basename(svg))
    if (/^\d+\.\d+-\d+$/.test(id)) {
      const prefix = `${id.split('-')[0].replace('.', '-')}-`
      if (!name.startsWith(prefix)) err(file, `图 ${id} 的文件名应以 ${prefix} 开头，实际是 ${name}.svg`)
    } else if (/^\d+-\d+$/.test(id)) {
      const p = id.split('-')[0]
      if (!name.startsWith(`${p}-part${p}-`)) {
        err(file, `图 ${id} 是部分导言图，文件名应以 ${p}-part${p}- 开头，实际是 ${name}.svg`)
      }
    } else {
      err(file, `图号 ${id} 不符合「节号-序号」或「部分号-序号」`)
    }
  }
}

const quantitative = new Set(
  readFileSync(join(ROOT, 'figures/quantitative.txt'), 'utf8')
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#')),
)
const srcDir = join(ROOT, 'figures/src')
for (const name of readdirSync(srcDir).filter((n) => n.endsWith('.json'))) {
  const svg = name.replace(/\.json$/, '.svg')
  if (!usedSvgs.has(svg)) err(join(srcDir, name), `图源 ${name} 没有被任何页面引用（产物 ${svg}）`)
}
for (const svg of usedSvgs) {
  if (!existsSync(join(srcDir, svg.replace(/\.svg$/, '.json'))) && !quantitative.has(svg)) {
    err(ROOT, `${svg} 既没有 figures/src 图源，也不在 figures/quantitative.txt`)
  }
}
for (const svg of quantitative) {
  if (!usedSvgs.has(svg)) err(join(ROOT, 'figures/quantitative.txt'), `白名单里的 ${svg} 没有被引用`)
}

// 4. 引用编号与锚点
for (const file of mdFiles) {
  const text = read(file)
  const refBlock = text.split('## 参考资料')[1]
  if (!refBlock) continue
  const list = [...refBlock.matchAll(/^\s*(\d+)\.\s/gm)].map((m) => Number(m[1]))
  const anchors = [...text.matchAll(/id="ref-(\d+)"/g)].map((m) => Number(m[1]))
  const targets = [...text.matchAll(/\(#ref-(\d+)\)/g)].map((m) => Number(m[1]))
  const expected = list.map((_, i) => i + 1)
  if (list.join(',') !== expected.join(',')) err(file, `参考资料编号不连续：${list.join(',')}`)
  for (const n of targets) {
    if (!anchors.includes(n)) err(file, `引用 [${n}] 没有对应锚点 #ref-${n}`)
    if (!list.includes(n)) err(file, `引用 [${n}] 超出参考资料条目数 ${list.length}`)
  }
  for (const n of anchors) {
    if (!list.includes(n)) err(file, `锚点 ref-${n} 没有对应的参考资料条目`)
  }
  // Figure 的 source 是 HTML 属性，里面没法放 Markdown 链接，裸引用只在正文里算错。
  const prose = text.replace(/<Figure[\s\S]*?\/>/g, '')
  if (/引用\s*\[(?![\\])/.test(prose)) {
    err(file, '出现裸引用（如「引用 [1]」），必须写成 [\\[1\\]](#ref-1)')
  }
  for (const m of prose.matchAll(/引用\s*\[\\\[(\d+)\\\]\](?!\(#ref-\d+\))/g)) {
    err(file, `引用 [${m[1]}] 没有链到 #ref-${m[1]}`)
  }
  for (const m of prose.matchAll(/引用\s*\[\\\[(\d+)\\\]\]\(#ref-(\d+)\)/g)) {
    if (m[1] !== m[2]) err(file, `引用编号 [${m[1]}] 链到了 #ref-${m[2]}`)
  }
}

// 5. 正文不得出现作者 TODO 式用语
const scanFiles = [...mdFiles, join(ROOT, 'AGENTS.md'), readme].filter(existsSync)
for (const file of scanFiles) {
  const text = read(file)
  for (const word of BANNED) {
    if (text.includes(word)) err(file, `出现禁用字符串「${word}」`)
  }
}

// 6. 阅读路径
for (const [id, nums] of Object.entries(book.readingPaths)) {
  for (const num of nums) {
    const section = sections.get(num)
    if (!section) {
      err(join(DOCS, '.vitepress/book.json'), `阅读路径 ${id} 引用了不存在的章节 ${num}`)
    } else if (section.status === 'placeholder') {
      warn(join(DOCS, '.vitepress/book.json'), `阅读路径 ${id} 含占位章节 ${num}（写作完成后自动消失）`)
    }
  }
}

// 输出
for (const line of warnings) console.warn(`WARN  ${line}`)
for (const line of errors) console.error(`ERROR ${line}`)
console.log(`\n检查完成：${errors.length} error / ${warnings.length} warning · ${mdFiles.length} 个 Markdown 文件`)
process.exit(errors.length ? 1 : 0)
