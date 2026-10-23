import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import HeroAurora from './components/HeroAurora.vue'
import Figure from './components/Figure.vue'
import './custom.css'

// 封面背景层通过默认主题的 home-hero-before 槽位注入，不 fork 主题。
// Figure 全局注册，正文里直接写 <Figure ... />。
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Figure', Figure)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-before': () => h(HeroAurora),
    })
  },
}
