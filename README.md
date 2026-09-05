# 纸间 · Paper Notes

一个用 Markdown 持续维护的论文笔记网站，基于 VitePress，自动部署到 GitHub Pages。

## 已包含

- 论文书架：日期排序、关键词筛选、主题标签、阅读状态
- Markdown 阅读页：全文搜索、自动目录、数学公式、代码高亮、深色模式
- 新笔记自动收录，支持多级目录，无需手动登记
- 可复用笔记模板和创建命令
- GitHub Actions 构建、PR 检查与 Pages 自动部署，自动适配仓库子路径

示例文章只演示笔记结构，不是真实论文总结；可以直接删除。

## 本地使用

安装 Node.js 22 或更新版本，然后执行：

```bash
npm ci
npm run dev
```

打开终端给出的本地地址。正式构建与预览：

```bash
npm run build
npm run preview
```

## 新增笔记

```bash
npm run note -- my-first-paper "我的第一篇论文笔记"
```

编辑新建的 `docs/papers/my-first-paper.md` 即可。也可以手动复制 [模板](templates/paper.md) 到 `docs/papers/`。文件顶部填写标题、摘要、日期、标签和阅读状态，正文直接使用 Markdown。

## 首次发布

1. 创建 GitHub 仓库，将项目推送到 `main`。确保 `.github/workflows/deploy.yml` 和 `package-lock.json` 一并提交。
2. 在 **Settings → Pages → Source** 选择 **GitHub Actions**。
3. 在 **Actions → Build and deploy notes → Run workflow** 启动部署。
4. 等待成功，在 **Settings → Pages** 查看网址。

后续修改 Markdown 并推送到 `main` 即自动更新。也可以直接在 GitHub 网页编辑 Markdown。

如使用 Git 命令，先将下面 URL 替换为你新建的空仓库地址：

```bash
git init -b main
git add .
git commit -m "Initialize paper notes website"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

若默认分支不是 `main`，同步修改工作流中的分支配置。站点路径自动从 Pages 配置读取，支持项目站点、用户站点；自定义域名需先在 GitHub Pages 设置中配置。PR 只构建，不发布。

## 项目结构

```text
docs/
  papers/                  # 你的 Markdown 论文笔记与配图
  .vitepress/config.mts    # 网站名称、导航、搜索等
  .vitepress/theme/        # 首页、笔记元信息和样式
  index.md                 # 首页入口
  guide.md                 # 完整写作维护指南
  about.md                 # 个人介绍
templates/paper.md         # 笔记模板
scripts/new-note.mjs       # 笔记创建工具
.github/workflows/         # 自动构建与部署
```

完整用法参见 [写作与维护指南](docs/guide.md)。部署配置参考 [VitePress 官方文档](https://vitepress.dev/guide/deploy) 和 [GitHub Pages 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
