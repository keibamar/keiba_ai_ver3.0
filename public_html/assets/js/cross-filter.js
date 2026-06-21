// .cross-filter内の馬場状態/クラスの2つの<select>から選択値を読み、
// 事前にサーバー側で生成済みの.cross-filter-panel（data-ground-state/data-class属性
// 付き、組み合わせごとに1つ）のうち一致するものだけを表示する。section-tabs.jsと同様、
// JSは表示/非表示の切り替えのみ行い、表の内容自体は生成しない。

function updateCrossFilter(container) {
  const groundSelect = container.querySelector(".cross-filter-ground-state");
  const classSelect = container.querySelector(".cross-filter-class");
  const panels = container.querySelectorAll(".cross-filter-panel");
  const emptyMessage = container.querySelector(".cross-filter-empty");
  if (!groundSelect || !classSelect) return;

  const groundState = groundSelect.value;
  const classValue = classSelect.value;
  let matched = false;

  panels.forEach((panel) => {
    const isMatch = panel.dataset.groundState === groundState && panel.dataset.class === classValue;
    panel.hidden = !isMatch;
    if (isMatch) matched = true;
  });

  if (emptyMessage) {
    emptyMessage.hidden = matched;
  }
}

function initCrossFilters() {
  document.querySelectorAll(".cross-filter").forEach((container) => {
    const groundSelect = container.querySelector(".cross-filter-ground-state");
    const classSelect = container.querySelector(".cross-filter-class");
    if (groundSelect) groundSelect.addEventListener("change", () => updateCrossFilter(container));
    if (classSelect) classSelect.addEventListener("change", () => updateCrossFilter(container));
    updateCrossFilter(container);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCrossFilters);
} else {
  initCrossFilters();
}
