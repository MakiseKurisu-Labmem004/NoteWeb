# 写作与维护指南

## 添加一篇论文笔记

将项目中的 `templates/paper.md` 复制到 `docs/papers/`，改名为 `your-paper.md`。也可以在项目根目录执行：

```bash
npm run note -- your-paper "你的论文标题"
```

修改文件顶部的元数据和正文。保存后，笔记会自动出现在首页、主题筛选和全文搜索中，不需要手动维护导航列表。支持 `docs/papers/` 下的多级目录。

```yaml
---
title: "你的论文标题"
description: "一句话概括研究问题与核心方法。"
date: "2026-09-05"
tags: ["具身智能", "模仿学习"]
status: "精读中"
venue: "会议 / 期刊 / 年份"
authors: "作者姓名"
paper: "https://example.com/paper"
code: ""
---
```

| 字段 | 用途 |
| --- | --- |
| `title` | 首页标题、搜索标题；正文一级标题需同步修改 |
| `description` | 首页卡片摘要 |
| `date` | 笔记日期，使用带引号的 `YYYY-MM-DD`，首页按此倒序排列 |
| `tags` | 主题标签，可以有多个 |
| `status` | 建议使用 `待读`、`精读中`、`已读`，也支持自定义状态 |
| `venue` | 会议、期刊和年份 |
| `authors` | 作者信息，可留空 |
| `paper` / `code` | 论文与代码的完整网址，可留空 |
| `example` | 仅示例笔记使用 `true`，自己的笔记请省略 |

在一级标题之后保留 `<PaperMeta />`，阅读页会自动展示日期、标签、论文链接与返回入口。

## Markdown 常用语法

### 标题、引用和任务

````md
## 研究问题
**重点**，以及 *需要留意的假设*。

> 这里写简短引文，并注明来源和页码。

- [x] 阅读摘要
- [ ] 复核实验

```python
print("Hello, research!")
```
````

### 数学公式

行内公式使用 `$E = mc^2$`，独立公式使用：

```text
$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$
```

### 图片与关联笔记

推荐把图片放在笔记旁边，例如 `docs/papers/images/method.png`：

```md
![方法框架图](./images/method.png)
[关联笔记](./another-paper.md)
```

相对路径可以兼容 GitHub Pages 的仓库子路径。引用图片时注明来源。

### 提示块

```md
::: tip 我的理解
这里写自己的推理与判断。
:::

::: warning 待核实
这个结论仍需要实验支持。
:::
```

## 本地预览

安装 Node.js 22 或更新版本，在项目根目录运行：

```bash
npm ci
npm run dev
```

打开终端显示的本地地址。修改 Markdown 后浏览器会自动刷新。

发布前可以检查正式构建：

```bash
npm run build
npm run preview
```

## 发布到 GitHub Pages

1. 在 GitHub 创建仓库，将本项目文件上传或推送到 `main` 分支，包含隐藏目录 `.github`，不要上传 `node_modules` 或 `.tools`。
2. 打开仓库 **Settings → Pages → Build and deployment → Source**，选择 **GitHub Actions**。
3. 打开 **Actions → Build and deploy notes → Run workflow** 手动运行首次部署；之后推送到 `main` 会自动部署。
4. 部署成功后，在仓库 **Settings → Pages** 或工作流的部署结果中打开网站。

普通项目通常访问 `https://用户名.github.io/仓库名/`；名为 `用户名.github.io` 的仓库使用根路径。工作流会从 Pages 配置获取路径，无需在代码中填写仓库名。

如果使用其他默认分支，请同步修改 `.github/workflows/deploy.yml` 中的两个 `main`。

详细说明见 [GitHub Pages 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 日常更新

新增或编辑 Markdown → 提交并推送 → 等待 Actions 完成 → 刷新网站。

也可以直接在 GitHub 网页里编辑 `docs/papers/` 下的 Markdown 并提交，不必每次启动本地环境。删除示例笔记后，首页会自动移除相应卡片。最近更新时间从 Git 提交记录读取；还未提交的本地文件可能没有更新时间。

## 个性化

- 网站名称、导航、页脚：`docs/.vitepress/config.mts`
- 首页介绍文案：`docs/.vitepress/theme/PaperLibrary.vue`
- 色彩和排版：`docs/.vitepress/theme/style.css`
- 个人介绍：`docs/about.md`

首页输入框筛选标题、摘要、标签和会议信息；顶部搜索按钮搜索笔记全文。站点同时提供深色模式、手机布局和阅读页目录。
