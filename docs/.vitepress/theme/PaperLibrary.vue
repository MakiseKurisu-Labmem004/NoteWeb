<script setup>
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data as papers } from './papers.data'

const query = ref('')
const tag = ref('全部')
const status = ref('全部状态')
const tags = computed(() => ['全部', ...new Set(papers.flatMap(p => p.tags))])
const filtered = computed(() => papers.filter(p =>
  (tag.value === '全部' || p.tags.includes(tag.value)) &&
  (status.value === '全部状态' || p.status === status.value) &&
  `${p.title} ${p.description} ${p.tags.join(' ')} ${p.venue}`.toLowerCase().includes(query.value.trim().toLowerCase())
))
const statuses = computed(() => ['全部状态', ...new Set(papers.map(p => p.status))])
function reset() { query.value = ''; tag.value = '全部'; status.value = '全部状态' }
</script>

<template>
  <main class="library">
    <section class="library-hero">
      <div>
        <p class="eyebrow">A GROWING RESEARCH JOURNAL</p>
        <h1>读过的论文，<br>留下自己的<span>思考。</span></h1>
        <p class="hero-description">把问题、方法与灵感写下来。<br>在一次次阅读中，连接知识的碎片。</p>
        <a class="guide-link" :href="withBase('/guide.html')">开始写一篇笔记 <span aria-hidden="true">↗</span></a>
      </div>
      <aside class="journal-card" aria-label="笔记库概览">
        <span class="journal-label">FIELD NOTES / 研究手记</span>
        <div class="journal-number">{{ String(papers.length).padStart(2, '0') }}<small>篇笔记</small></div>
        <div class="journal-line"></div>
        <p>保持好奇，<br>让每一次阅读都有迹可循。</p>
        <span class="journal-bottom">{{ tags.length - 1 }} 个研究主题 · 持续更新中</span>
      </aside>
    </section>

    <section class="shelf" aria-labelledby="shelf-title">
      <div class="shelf-heading"><div><p class="eyebrow">THE COLLECTION</p><h2 id="shelf-title">论文书架</h2></div><span class="result-count" aria-live="polite">{{ filtered.length }} 篇笔记</span></div>
      <div class="shelf-tools">
        <label class="search-field"><span aria-hidden="true">⌕</span><input v-model="query" type="search" placeholder="筛选标题、关键词或研究方向…" aria-label="筛选笔记"></label>
        <select v-model="status" aria-label="按阅读状态筛选"><option v-for="s in statuses" :key="s">{{ s }}</option></select>
      </div>
      <div class="tag-list" aria-label="研究主题"><button v-for="t in tags" :key="t" :class="{ active: tag === t }" :aria-pressed="tag === t" @click="tag = t">{{ t }}</button></div>
      <div v-if="filtered.length" class="paper-grid">
        <a v-for="paper in filtered" :key="paper.url" :href="withBase(paper.url)" class="paper-card">
          <div class="paper-top"><span>{{ paper.venue || 'READING NOTE' }}</span><span class="status">{{ paper.status }}</span></div>
          <h3>{{ paper.title }}</h3><p class="paper-description">{{ paper.description }}</p>
          <div class="paper-tags"><span v-for="t in paper.tags" :key="t">{{ t }}</span><span v-if="paper.example">示例笔记</span></div>
          <div class="paper-bottom"><time :datetime="paper.date">{{ paper.date }}</time><span aria-hidden="true">阅读笔记 ↗</span></div>
        </a>
      </div>
      <div v-else class="empty-state"><h3>暂时没有匹配的笔记</h3><p>试试其他关键词，或清除筛选条件。</p><button @click="reset">显示全部笔记</button></div>
    </section>
    <div class="library-footnote"><span>01 — 阅读 · 02 — 记录 · 03 — 连接</span><span>以 Markdown 书写，以思考积累。</span></div>
  </main>
</template>
