import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import HeroAurora from './components/HeroAurora.vue'
import Figure from './components/Figure.vue'
import ChapterList from './components/ChapterList.vue'
import ReadingPath from './components/ReadingPath.vue'
import './custom.css'

// 封面背景层通过默认主题的 home-hero-before 槽位注入，不 fork 主题。
// Figure / ChapterList / ReadingPath 全局注册，正文里直接写标签。
// ChapterList 与 ReadingPath 的数据都来自 docs/.vitepress/book.json。
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Figure', Figure)
    app.component('ChapterList', ChapterList)
    app.component('ReadingPath', ReadingPath)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-before': () => h(HeroAurora),
    })
  },
}
