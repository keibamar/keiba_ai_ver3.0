// .tabbed-section内のタブ（.section-tabs button[data-target]）をクリックすると、
// 同じ.tabbed-section内の.section-panel[data-section]のうち一致するものだけを表示し、
// 他はhidden属性を付けて隠す。1ページに複数の.tabbed-sectionがあっても、
// それぞれ独立して動作する（.tabbed-section単位でクエリするため）。

function activateTab(section, target) {
  const tabs = section.querySelectorAll(".section-tabs button");
  const panels = section.querySelectorAll(".section-panel");

  tabs.forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.target === target));
  });

  panels.forEach((panel) => {
    if (panel.dataset.section === target) {
      panel.removeAttribute("hidden");
    } else {
      panel.setAttribute("hidden", "");
    }
  });
}

function initSectionTabs() {
  document.querySelectorAll(".tabbed-section").forEach((section) => {
    const tabs = section.querySelectorAll(".section-tabs button");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => activateTab(section, tab.dataset.target));
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initSectionTabs);
} else {
  initSectionTabs();
}
