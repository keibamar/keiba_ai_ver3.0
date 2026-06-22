// .page-calendar-tab（右側のカレンダー＋階層タブ）はfloat: rightのため、自身の中身
// （カレンダー＋階層リスト）の高さしか持たない。本文がそれより長いページでは、
// floatが終わった位置から本文が全幅に広がってしまい、グラフ等の終端位置が
// 途中でズレて見える。本文の実際の高さに合わせてmin-heightを動的に設定し、
// タブをページの最後まで伸ばす。
//
// タブ切り替え（section-tabs.js）・絞り込み（cross-filter.js）・折りたたみ
// （<details>）はいずれもhidden/open属性の変更で本文の高さを変えるため、
// MutationObserverで検知してその都度再計算する。

function resizeCalendarTab() {
  const tab = document.querySelector(".page-calendar-tab");
  if (!tab) return;

  tab.style.minHeight = "0px";
  const height = document.body.scrollHeight;
  tab.style.minHeight = `${height}px`;
}

function initCalendarTabHeight() {
  if (!document.querySelector(".page-calendar-tab")) return;

  resizeCalendarTab();
  window.addEventListener("resize", resizeCalendarTab);

  let pending = false;
  const observer = new MutationObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      resizeCalendarTab();
    });
  });
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["hidden", "open"],
    subtree: true,
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCalendarTabHeight);
} else {
  initCalendarTabHeight();
}
