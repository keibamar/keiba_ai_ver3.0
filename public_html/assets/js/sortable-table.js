// table.sortable のヘッダークリックで行を並び替える汎用スクリプト。
// 1ページに複数のsortableテーブルが並ぶ（コース詳細ページ等）ため、
// ソート状態（列・方向）は各<table>要素にdata属性として保持する
// （グローバル変数で持つとテーブル間で状態が混ざってしまうため）。

function extractSortValue(cell) {
  const text = cell.textContent.trim();

  // 「1:21.7」のような時計表記は分:秒として数値化する
  const timeMatch = text.match(/^(\d+):(\d{2}(?:\.\d+)?)$/);
  if (timeMatch) {
    return parseInt(timeMatch[1], 10) * 60 + parseFloat(timeMatch[2]);
  }

  const numMatch = text.match(/-?\d+(?:\.\d+)?/);
  if (numMatch) {
    return parseFloat(numMatch[0]);
  }

  return text.toLowerCase();
}

function sortTable(table, colIndex) {
  const tbody = table.querySelector("tbody");
  if (!tbody) return;

  const rows = Array.from(tbody.querySelectorAll("tr"));
  if (rows.length === 0) return;

  const currentDir = table.dataset.sortCol === String(colIndex) ? table.dataset.sortDir : null;
  const nextDir = currentDir === "asc" ? "desc" : "asc";

  const values = rows.map((row) => {
    const cell = row.cells[colIndex];
    return cell ? extractSortValue(cell) : "";
  });
  const isNumeric = values.every((v) => typeof v === "number");

  rows.sort((a, b) => {
    const va = extractSortValue(a.cells[colIndex]);
    const vb = extractSortValue(b.cells[colIndex]);
    let cmp;
    if (isNumeric) {
      cmp = va - vb;
    } else {
      cmp = String(va) > String(vb) ? 1 : String(va) < String(vb) ? -1 : 0;
    }
    return nextDir === "asc" ? cmp : -cmp;
  });

  rows.forEach((row) => tbody.appendChild(row));

  table.dataset.sortCol = String(colIndex);
  table.dataset.sortDir = nextDir;

  table.querySelectorAll("thead th").forEach((th, i) => {
    th.classList.remove("sort-asc", "sort-desc");
    if (i === colIndex) {
      th.classList.add(nextDir === "asc" ? "sort-asc" : "sort-desc");
    }
  });
}

function initSortableTables() {
  document.querySelectorAll("table.sortable").forEach((table) => {
    const headerCells = table.querySelectorAll("thead th");
    headerCells.forEach((th, colIndex) => {
      th.addEventListener("click", () => sortTable(table, colIndex));
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initSortableTables);
} else {
  initSortableTables();
}
