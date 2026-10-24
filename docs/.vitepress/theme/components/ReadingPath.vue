<script setup lang="ts">
/**
 * 阅读路径。章节编号与完稿状态来自 docs/.vitepress/book.json：
 * 路径里混进占位页时，页面会自动标灰并说明，不必手工维护提示语。
 */
import book from '../../book.json'

const props = defineProps<{ id: string }>()

const byNum = new Map(
  book.parts.flatMap((p) => p.sections).filter((s) => s.num).map((s) => [s.num as string, s]),
)
const items = (book.readingPaths as Record<string, string[]>)[props.id]
  ?.map((num) => byNum.get(num))
  .filter((s): s is NonNullable<typeof s> => Boolean(s)) ?? []
const pending = items.filter((s) => s.status === 'placeholder')
</script>

<template>
  <div class="reading-path">
    <p class="reading-path__seq">
      <template v-for="(s, i) in items" :key="s.path">
        <a :href="s.path" :class="{ 'is-pending': s.status === 'placeholder' }">{{ s.num }}</a>
        <span v-if="i < items.length - 1" class="reading-path__sep">→</span>
      </template>
    </p>
    <p v-if="pending.length" class="reading-path__note">
      标灰的 {{ pending.map((s) => s.num).join('、') }} 尚未完稿，这条路径会随写作更新。
    </p>
  </div>
</template>

<style scoped>
.reading-path__seq {
  margin: 12px 0 0;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
}

.reading-path__sep {
  margin: 0 8px;
  color: var(--vp-c-text-3);
}

.reading-path__seq a.is-pending {
  color: var(--vp-c-text-3);
  text-decoration-style: dotted;
}

.reading-path__note {
  margin: 8px 0 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
</style>
