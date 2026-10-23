import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import HeroAurora from './components/HeroAurora.vue'
import './custom.css'

// 封面背景层通过默认主题的 home-hero-before 槽位注入，不 fork 主题。
export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-before': () => h(HeroAurora),
    })
  },
}
