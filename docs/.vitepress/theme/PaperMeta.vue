<script setup>
import { useData, withBase } from 'vitepress'
const { frontmatter: f } = useData()
</script>
<template>
  <div class="reading-meta">
    <a :href="withBase('/')">← 返回论文书架</a>
    <p>{{ f.date }} · {{ f.status }}<template v-if="f.venue"> · {{ f.venue }}</template></p>
    <p v-if="f.authors"><strong>作者：</strong>{{ f.authors }}</p>
    <div class="paper-tags"><span v-for="tag in f.tags" :key="tag">{{ tag }}</span></div>
    <p v-if="f.paper || f.code"><a v-if="f.paper" :href="f.paper" target="_blank" rel="noopener noreferrer">论文原文 ↗</a> <a v-if="f.code" :href="f.code" target="_blank" rel="noopener noreferrer">代码 ↗</a></p>
  </div>
</template>
