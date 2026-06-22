// .cross-filter内の馬場状態/クラス/（あれば）年度の<select>から選択値を読み、
// 事前にサーバー側で生成済みの.cross-filter-panel（data-ground-state/data-class/
// （あれば）data-year属性付き、組み合わせごとに1つ）のうち一致するものだけを表示する。
// section-tabs.jsと同様、JSは表示/非表示の切り替えのみ行い、表の内容自体は生成しない。
// 年度セレクトは2軸のみのページ（ai_performance_report_generator側）には存在しないため、
// 無ければ年度条件を無視する（2軸ページの動作はそのまま変わらない）。

function updateCrossFilter(container) {
  const groundSelect = container.querySelector(".cross-filter-ground-state");
  const classSelect = container.querySelector(".cross-filter-class");
  const yearSelect = container.querySelector(".cross-filter-year");
  const panels = container.querySelectorAll(".cross-filter-panel");
  const emptyMessage = container.querySelector(".cross-filter-empty");
  if (!groundSelect || !classSelect) return;

  const groundState = groundSelect.value;
  const classValue = classSelect.value;
  const yearValue = yearSelect ? yearSelect.value : null;
  let matched = false;

  panels.forEach((panel) => {
    const isMatch =
      panel.dataset.groundState === groundState &&
      panel.dataset.class === classValue &&
      (yearValue === null || panel.dataset.year === yearValue);
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
    const yearSelect = container.querySelector(".cross-filter-year");
    if (groundSelect) groundSelect.addEventListener("change", () => updateCrossFilter(container));
    if (classSelect) classSelect.addEventListener("change", () => updateCrossFilter(container));
    if (yearSelect) yearSelect.addEventListener("change", () => updateCrossFilter(container));
    updateCrossFilter(container);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCrossFilters);
} else {
  initCrossFilters();
}
