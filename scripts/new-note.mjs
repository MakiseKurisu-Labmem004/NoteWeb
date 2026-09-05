import { readFile, mkdir, writeFile } from 'node:fs/promises'

const [slug, title = '新论文阅读笔记'] = process.argv.slice(2)
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error('用法：npm run note -- paper-slug "论文标题"\n文件名只使用小写字母、数字和连字符。')
  process.exit(1)
}
const template = await readFile(new URL('../templates/paper.md', import.meta.url), 'utf8')
const now = new Date()
const date = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
const content = template.replace('title: "论文标题"', `title: ${JSON.stringify(title)}`)
  .replace('date: "2026-09-05"', `date: "${date}"`).replace('# 论文标题', `# ${title.replace(/[\r\n]/g, ' ')}`)
const target = new URL(`../docs/papers/${slug}.md`, import.meta.url)
await mkdir(new URL('../docs/papers/', import.meta.url), { recursive: true })
try {
  await writeFile(target, content, { encoding: 'utf8', flag: 'wx' })
  console.log(`已创建 docs/papers/${slug}.md`)
} catch (error) {
  if (error.code === 'EEXIST') { console.error('同名笔记已存在，请选择其他文件名。'); process.exit(1) }
  throw error
}
