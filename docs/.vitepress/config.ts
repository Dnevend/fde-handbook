import { defineConfig } from 'vitepress'
import book from './book.json'

// 站点通过自定义域名 https://fde.dnevend.site/ 发布（GitHub Pages 项目页 +
// project custom domain），所以站点位于域名根，base 用默认的 '/'。
// 若改回 username.github.io/repo/ 形式，需要把 base 改成 '/repo/'。

// 侧边栏与部分导言页的章节列表都由 book.json 生成。
// 增删章节只改 book.json，再跑 `npm run docs:sync` 更新 README 的进度块。
const sidebar = book.parts.map((part) => ({
  text: part.nav,
  collapsed: true,
  items: part.sections.map((section) => ({
    text: section.num ? `${section.num} ${section.nav}` : section.nav,
    link: section.path,
  })),
}))

export default defineConfig({
  lang: 'zh-CN',
  title: 'FDE Handbook',
  description: '从本地大模型到 RAG 落地',
  base: '/',
  cleanUrls: true,
  srcExclude: ['**/README.md'],
  sitemap: { hostname: 'https://fde.dnevend.site/' },

  // head 里的链接不会自动加 base，这里显式写出根路径。
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }]],

  themeConfig: {
    // 深浅两版除颜色外必须一致；含义见 docs/public/logo.svg 的注释
    logo: { light: '/logo.svg', dark: '/logo-dark.svg' },

    nav: [{ text: '序', link: '/00-preface/' }],

    sidebar,

    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一节', next: '下一节' },
    lastUpdated: { text: '最后更新' },
    notFound: {
      title: '页面不存在',
      quote: '这个地址没有对应的章节，可能链接已过期，或路径拼错了。',
      linkLabel: '回到首页',
      linkText: '回到首页',
    },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    langMenuLabel: '语言',
    externalLinkIcon: true,

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清空',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
  },
})
