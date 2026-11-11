<script setup lang="ts">
/**
 * 部分导言页末尾的章节列表。
 * 数据来自 docs/.vitepress/book.json，与侧边栏、README 进度同源，避免手工同步。
 */
import { withBase } from 'vitepress'
import book from '../../book.json'

const props = defineProps<{ part: string }>()
const sections = book.parts.find((p) => p.id === props.part)?.sections.filter((s) => s.num) ?? []
</script>

<template>
  <ul>
    <li v-for="s in sections" :key="s.path">
      <a :href="withBase(s.path)">{{ s.num }} {{ s.nav }}</a>
      <span v-if="s.status === 'placeholder'" class="chapter-list__tag">写作中</span>
      <span v-else-if="s.status === 'draft'" class="chapter-list__tag">草稿</span>
    </li>
  </ul>
</template>

<style scoped>
.chapter-list__tag {
  margin-left: 8px;
  padding: 0 6px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
}
</style>
