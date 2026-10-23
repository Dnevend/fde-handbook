<script setup lang="ts">
// 首页 Hero 背景层：纯 CSS 的“噪声极光 + 颗粒”。
// 约束：零依赖、零客户端 JS、SSR 安全；动效在 prefers-reduced-motion 下全部关闭。
</script>

<template>
  <div class="hero-aurora" aria-hidden="true">
    <div class="hero-aurora__field">
      <span class="hero-aurora__blob hero-aurora__blob--1" />
      <span class="hero-aurora__blob hero-aurora__blob--2" />
      <span class="hero-aurora__blob hero-aurora__blob--3" />
    </div>
    <div class="hero-aurora__grain" />
  </div>
</template>

<style scoped>
.hero-aurora {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: min(92vh, 900px);
  height: min(92svh, 900px);
  overflow: hidden;
  pointer-events: none;
  z-index: 0;
  opacity: 0.9;
  /* 底部淡出，避免与下方 features 区形成硬边 */
  -webkit-mask-image: linear-gradient(
    to bottom,
    #000 0%,
    #000 56%,
    rgba(0, 0, 0, 0.55) 78%,
    transparent 96%
  );
  mask-image: linear-gradient(
    to bottom,
    #000 0%,
    #000 56%,
    rgba(0, 0, 0, 0.55) 78%,
    transparent 96%
  );
}

.hero-aurora__field {
  position: absolute;
  inset: -25%;
  filter: blur(60px) saturate(1.2);
}

.hero-aurora__blob {
  position: absolute;
  border-radius: 50%;
  will-change: transform;
  mix-blend-mode: screen;
}

/* ---- 暗色：屏幕混合，让光晕“发亮” ---- */
.hero-aurora__blob--1 {
  width: 62vmax;
  height: 62vmax;
  left: -6vmax;
  top: -20vmax;
  background: radial-gradient(circle at 50% 50%, rgba(79, 70, 229, 0.62) 0%, rgba(79, 70, 229, 0) 64%);
  animation: hero-float-1 42s ease-in-out infinite alternate;
}

.hero-aurora__blob--2 {
  width: 58vmax;
  height: 58vmax;
  right: -10vmax;
  top: 2vmax;
  background: radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.5) 0%, rgba(14, 165, 233, 0) 64%);
  animation: hero-float-2 56s ease-in-out infinite alternate;
}

.hero-aurora__blob--3 {
  width: 54vmax;
  height: 54vmax;
  left: 18vmax;
  bottom: -24vmax;
  background: radial-gradient(circle at 50% 50%, rgba(168, 85, 247, 0.46) 0%, rgba(168, 85, 247, 0) 66%);
  animation: hero-float-3 72s ease-in-out infinite alternate;
}

@keyframes hero-float-1 {
  to {
    transform: translate3d(6%, 8%, 0) scale(1.12);
  }
}

@keyframes hero-float-2 {
  to {
    transform: translate3d(-8%, 6%, 0) scale(1.08);
  }
}

@keyframes hero-float-3 {
  to {
    transform: translate3d(5%, -9%, 0) scale(1.16);
  }
}

/* ---- 亮色：正片叠底，避免白底被“漂白” ---- */
html:not(.dark) .hero-aurora {
  opacity: 0.62;
}

html:not(.dark) .hero-aurora__blob {
  mix-blend-mode: multiply;
}

html:not(.dark) .hero-aurora__blob--1 {
  background: radial-gradient(circle at 50% 50%, rgba(129, 140, 248, 0.55) 0%, rgba(129, 140, 248, 0) 64%);
}

html:not(.dark) .hero-aurora__blob--2 {
  background: radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.48) 0%, rgba(56, 189, 248, 0) 64%);
}

html:not(.dark) .hero-aurora__blob--3 {
  background: radial-gradient(circle at 50% 50%, rgba(192, 132, 252, 0.42) 0%, rgba(192, 132, 252, 0) 66%);
}

/* ---- 颗粒：内联 feTurbulence，不引入二进制资产，顺带消掉渐变色带 ---- */
.hero-aurora__grain {
  position: absolute;
  inset: 0;
  opacity: 0.05;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/><feColorMatrix type='saturate' values='0'/></filter><rect width='180' height='180' filter='url(%23n)'/></svg>");
  background-size: 180px 180px;
}

html:not(.dark) .hero-aurora__grain {
  opacity: 0.035;
}

@media (max-width: 768px) {
  .hero-aurora {
    height: min(80vh, 640px);
    height: min(80svh, 640px);
    opacity: 0.72;
  }

  .hero-aurora__field {
    filter: blur(44px) saturate(1.15);
  }

  /* 手机上 vmax 太小会把光晕挤成一条顶部色带，改用 vw 撑开 */
  .hero-aurora__blob--1 {
    width: 105vw;
    height: 105vw;
    left: -22vw;
    top: -18vw;
  }

  .hero-aurora__blob--2 {
    width: 100vw;
    height: 100vw;
    right: -25vw;
    top: 8vw;
  }

  .hero-aurora__blob--3 {
    width: 95vw;
    height: 95vw;
    left: 22vw;
    bottom: -28vw;
  }
}

@media (prefers-reduced-motion: reduce) {
  .hero-aurora__blob {
    animation: none;
  }
}
</style>
