// スマートフォン幅（CSS側の@media (max-width: 800px)と合わせる）では、
// .page-calendar-tab（右側のカレンダー＋階層タブ）を常時表示の本文埋め込みではなく、
// ボタンで開閉するポップアップ（ドロワー）として表示する。表示/非表示の切り替え自体は
// CSS側の.openクラスで行い、ここではボタン・背景（バックドロップ）のクリックに応じて
// そのクラスを付け外しするだけにする。

function initSidebarToggle() {
  const toggle = document.querySelector(".page-calendar-tab-toggle");
  const tab = document.querySelector(".page-calendar-tab");
  const backdrop = document.querySelector(".page-calendar-tab-backdrop");
  const close = document.querySelector(".page-calendar-tab-close");
  if (!toggle || !tab || !backdrop) return;

  function openTab() {
    tab.classList.add("open");
    backdrop.classList.add("open");
  }

  function closeTab() {
    tab.classList.remove("open");
    backdrop.classList.remove("open");
  }

  toggle.addEventListener("click", openTab);
  backdrop.addEventListener("click", closeTab);
  if (close) close.addEventListener("click", closeTab);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initSidebarToggle);
} else {
  initSidebarToggle();
}
