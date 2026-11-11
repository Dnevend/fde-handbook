<script setup lang="ts">
/**
 * 正文插图。图源在 figures/src/*.json，由 figures/render.sh 产出
 * docs/public/figures/*.svg；这里只负责版式和图注。
 *
 * 图注三段：图号、一句话说明、数据来源。只要图里有数字，source 必填，
 * 并沿用正文的"估算 / 引用 / 实测"标注口径。
 */
import { withBase } from 'vitepress'

withDefaults(
  defineProps<{
    id: string
    src: string
    alt: string
    caption: string
    source?: string
  }>(),
  { source: '' },
)
</script>

<template>
  <figure class="book-figure">
    <!-- 点开原始 SVG：小屏上正文栏只有 320px 左右，图缩得太小，开新标签能按原尺寸看。 -->
    <a class="book-figure__link" :href="withBase(src)" target="_blank" rel="noopener">
      <img class="book-figure__img" :src="withBase(src)" :alt="alt" loading="lazy" />
    </a>
    <figcaption class="book-figure__caption">
      <span class="book-figure__id">图 {{ id }}</span>
      <span class="book-figure__text">{{ caption }}</span>
      <span v-if="source" class="book-figure__source">{{ source }}</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.book-figure {
  margin: 32px 0;
}

/* 图和正文同宽：不向两侧借留白，保持左右边界对齐。
   960px 画布缩到正文宽度后图内文字会偏小，点开原图（figure 里的链接）可看全尺寸。 */

/* 所有插图是白底 SVG。浅色主题下白底与页面同色，不需要额外装饰；
   深色主题下给它一块白卡，避免图上的深色文字失去对比度。 */
.book-figure__img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 6px;
}

.dark .book-figure__img {
  background: #ffffff;
  box-shadow: 0 0 0 1px var(--vp-c-divider);
}

.book-figure__caption {
  margin-top: 10px;
  font-size: 13px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}

.book-figure__id {
  margin-right: 8px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.book-figure__source {
  display: block;
  margin-top: 2px;
  font-size: 12px;
  color: var(--vp-c-text-3);
}

@media (min-width: 640px) {
  .book-figure {
    margin-top: 36px;
    margin-bottom: 36px;
  }
}
</style>
