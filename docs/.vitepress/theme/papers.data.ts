import { createContentLoader } from 'vitepress'

export default createContentLoader('papers/**/*.md', {
  transform(raw) {
    return raw.map(({ url, frontmatter: f }) => ({
      url,
      title: String(f.title || '未命名笔记'),
      description: String(f.description || ''),
      date: String(f.date || '').slice(0, 10),
      tags: Array.isArray(f.tags) ? f.tags.map(String) : [],
      status: String(f.status || '待读'),
      venue: String(f.venue || ''),
      example: Boolean(f.example)
    })).sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title))
  }
})
