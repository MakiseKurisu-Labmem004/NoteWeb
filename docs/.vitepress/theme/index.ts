import DefaultTheme from 'vitepress/theme'
import PaperLibrary from './PaperLibrary.vue'
import PaperMeta from './PaperMeta.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('PaperLibrary', PaperLibrary)
    app.component('PaperMeta', PaperMeta)
  }
}
