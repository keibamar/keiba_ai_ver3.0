// .page-search内の入力欄（#page-search-input）で、page-search-index.jsが定義する
// PAGE_SEARCH_INDEX（{label, path}のサイト直下相対パス一覧）を部分一致検索し、
// 結果一覧（#page-search-results）をその場に表示する。pathはサイト直下からの
// 相対パスのため、遷移時は埋め込み先ページのbase_path（.page-searchのdata属性）を
// 前置する。

const PAGE_SEARCH_MAX_RESULTS = 15;

function matchesQuery(label, query) {
  const lowerLabel = label.toLowerCase();
  return query.split(/\s+/).filter(Boolean).every((token) => lowerLabel.includes(token.toLowerCase()));
}

function renderPageSearchResults(container, basePath, entries) {
  container.innerHTML = "";
  entries.forEach((entry) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = basePath + entry.path;
    a.textContent = entry.label;
    li.appendChild(a);
    container.appendChild(li);
  });
  container.hidden = entries.length === 0;
}

function initPageSearch() {
  const wrap = document.querySelector(".page-search");
  const input = document.getElementById("page-search-input");
  const results = document.getElementById("page-search-results");
  if (!wrap || !input || !results || typeof PAGE_SEARCH_INDEX === "undefined") return;

  const basePath = wrap.dataset.basePath || "";

  input.addEventListener("input", () => {
    const query = input.value.trim();
    if (!query) {
      results.hidden = true;
      results.innerHTML = "";
      return;
    }
    const matched = PAGE_SEARCH_INDEX.filter((entry) => matchesQuery(entry.label, query)).slice(
      0,
      PAGE_SEARCH_MAX_RESULTS
    );
    renderPageSearchResults(results, basePath, matched);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      results.hidden = true;
      results.innerHTML = "";
    } else if (event.key === "Enter") {
      const firstLink = results.querySelector("a");
      if (firstLink) {
        event.preventDefault();
        window.location.href = firstLink.href;
      }
    }
  });

  document.addEventListener("click", (event) => {
    if (!wrap.contains(event.target)) {
      results.hidden = true;
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initPageSearch);
} else {
  initPageSearch();
}
