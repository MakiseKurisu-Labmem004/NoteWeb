import { defineConfig } from 'vitepress'

// Actions supplies the actual Pages base path, supporting project and user sites.
const base = process.env.BASE_PATH || '/'

export default defineConfig({
  lang: 'zh-CN',
  title: '纸间 · Paper Notes',
  description: '阅读、思考、连接。用 Markdown 持续积累论文阅读笔记。',
  base,
  lastUpdated: true,
  markdown: { math: true },
  themeConfig: {
    siteTitle: '纸间 / Paper Notes',
    nav: [
      { text: '论文书架', link: '/' },
      { text: '写作指南', link: '/guide' },
      { text: '关于', link: '/about' }
    ],
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          options: {
            tokenize: (text) => Array.from(new Intl.Segmenter('zh-CN', { granularity: 'word' }).segment(text))
              .filter(part => part.isWordLike).map(part => part.segment)
          }
        },
        locales: {
          root: {
            translations: {
              button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索笔记' },
              modal: {
                displayDetails: '显示详细内容', resetButtonTitle: '清除搜索',
                backButtonTitle: '关闭搜索', noResultsText: '没有找到相关笔记',
                footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
              }
            }
          }
        }
      }
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最近更新', formatOptions: { dateStyle: 'medium' } },
    returnToTopLabel: '回到顶部',
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '菜单',
    footer: { message: '阅读是起点，理解在字里行间。', copyright: 'Paper Notes · 持续生长的研究笔记' }
  }
})
