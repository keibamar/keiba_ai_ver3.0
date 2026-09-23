// race-marks.js — ユーザー予想マーク機能
(function () {
  'use strict';

  const MARKS  = ['', '◎', '〇', '▲', '△', '☆', '消'];
  const COLORS  = { '◎': '#e53935', '〇': '#1565c0', '▲': '#6a1b9a',
                    '△': '#2e7d32', '☆': '#f57c00', '消': '#999' };
  // 元の raceTable ソート対象列インデックス（印列挿入前）
  const ORIG_SORT_IDXS = [1, 7, 9, 10, 11];

  // ── localStorage ─────────────────────────────────────────
  const key = () => 'race_marks' + location.pathname;
  const load = () => { try { return JSON.parse(localStorage.getItem(key()) || '{}'); } catch { return {}; } };
  const save = m  => { try { localStorage.setItem(key(), JSON.stringify(m)); } catch {} };

  // ── スタイル注入 ───────────────────────────────────────────
  function injectStyles() {
    const s = document.createElement('style');
    s.textContent = `
      .mark-col, .mark-cell { text-align: center; width: 30px; padding: 1px 2px; }
      .mark-btn {
        background: none; border: 1px solid #ccc; border-radius: 3px;
        font-size: 1em; cursor: pointer; min-width: 26px; padding: 1px 3px;
        line-height: 1.3; font-weight: bold; transition: border-color .1s;
      }
      .mark-btn:hover { border-color: #888; background: #f0f0f0; }

      /* ドロップダウン */
      .mark-dropdown {
        position: absolute; z-index: 9999;
        background: #fff; border: 1px solid #ccc; border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,.2);
        display: flex; flex-direction: column; min-width: 52px;
        padding: 3px 0;
      }
      .mark-dropdown button {
        background: none; border: none; padding: 4px 10px;
        font-size: 1em; font-weight: bold; cursor: pointer;
        text-align: center; line-height: 1.5;
      }
      .mark-dropdown button:hover { background: #f0f0f0; }
      .mark-dropdown .mark-clear { color: #bbb; font-weight: normal; }

      /* 消: グレーアウト（背景・色はそのまま） */
      tr[data-mark="消"] td { opacity: 0.4; }

      /* ◎: 上下に赤線 + 印セルに赤い左線（背景色は上書きしない） */
      tr[data-mark="◎"] td {
        box-shadow: inset 0 2px 0 #e53935, inset 0 -1px 0 #e53935;
      }
      tr[data-mark="◎"] .mark-cell {
        box-shadow: inset 0 2px 0 #e53935, inset 0 -1px 0 #e53935, inset 4px 0 0 #e53935;
      }

      /* 〇▲△☆: 上下に青線（共通） */
      tr[data-mark="〇"] td, tr[data-mark="▲"] td,
      tr[data-mark="△"] td, tr[data-mark="☆"] td {
        box-shadow: inset 0 2px 0 #1565c0, inset 0 -1px 0 #1565c0;
      }
      tr[data-mark="〇"] .mark-cell, tr[data-mark="▲"] .mark-cell,
      tr[data-mark="△"] .mark-cell, tr[data-mark="☆"] .mark-cell {
        box-shadow: inset 0 2px 0 #1565c0, inset 0 -1px 0 #1565c0, inset 4px 0 0 #1565c0;
      }
    `;
    document.head.appendChild(s);
  }

  // ── ソート補正（印列を先頭に挿入したためインデックスが +1 ずれる） ──
  function getCellVal(tr, idx) {
    const td = tr.children[idx];
    if (td?.dataset.sort !== undefined && td.dataset.sort !== '') return Number(td.dataset.sort);
    const v = td ? td.innerText.trim() : '';
    return (v !== '' && !isNaN(v)) ? Number(v) : v;
  }

  function doSort(table, colIdx, th) {
    const tbody = table.tBodies[0];
    const arr   = [...tbody.querySelectorAll('tr')];
    const dir   = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
    th.dataset.sortDir = dir;

    table.querySelectorAll('th').forEach(h => {
      if (h !== th) h.dataset.sortDir = '';
      const ind = h.querySelector('.sort-ind');
      if (ind) ind.textContent = (h === th) ? (dir === 'asc' ? ' ▲' : ' ▼') : '';
    });

    arr.sort((a, b) => {
      const A = getCellVal(a, colIdx), B = getCellVal(b, colIdx);
      if (typeof A === 'number' && typeof B === 'number') return dir === 'asc' ? A - B : B - A;
      return dir === 'asc'
        ? String(A).localeCompare(String(B))
        : String(B).localeCompare(String(A));
    });
    arr.forEach(r => tbody.appendChild(r));
  }

  function patchSort(table) {
    // 印列挿入後の th リスト（先頭に印 th が追加済み）
    const allTh = table.querySelectorAll('th');
    ORIG_SORT_IDXS.forEach(origIdx => {
      const th = allTh[origIdx + 1]; // 印列分 +1
      if (!th) return;
      // クローンで旧リスナーを除去し、補正済みインデックスで再登録
      const clone = th.cloneNode(true);
      th.parentNode.replaceChild(clone, th);
      clone.style.cursor = 'pointer';
      clone.addEventListener('click', () => doSort(table, origIdx + 1, clone));
    });
  }

  // ── マーク適用 ────────────────────────────────────────────
  function applyMark(row, btn, mark) {
    row.dataset.mark    = mark || '';
    btn.textContent     = mark || '−';
    btn.style.color     = mark ? (COLORS[mark] || '#333') : '#bbb';
    btn.style.fontWeight = (mark && mark !== '消') ? 'bold' : 'normal';
  }

  // ── ドロップダウン ────────────────────────────────────────
  let activeDropdown = null;

  function closeDropdown() {
    if (activeDropdown) { activeDropdown.remove(); activeDropdown = null; }
  }

  function showDropdown(btn, row, k) {
    closeDropdown();

    const dd = document.createElement('div');
    dd.className = 'mark-dropdown';

    // 「なし」クリア項目
    const clearBtn = document.createElement('button');
    clearBtn.textContent = '−';
    clearBtn.className = 'mark-clear';
    clearBtn.addEventListener('click', e => { e.stopPropagation(); selectMark('', btn, row, k); });
    dd.appendChild(clearBtn);

    // 各印
    MARKS.filter(m => m).forEach(m => {
      const item = document.createElement('button');
      item.textContent = m;
      item.style.color = COLORS[m] || '#333';
      item.addEventListener('click', e => { e.stopPropagation(); selectMark(m, btn, row, k); });
      dd.appendChild(item);
    });

    // ボタン直下に配置
    document.body.appendChild(dd);
    const rect = btn.getBoundingClientRect();
    dd.style.left = (rect.left + window.scrollX) + 'px';
    dd.style.top  = (rect.bottom + window.scrollY + 2) + 'px';

    activeDropdown = dd;
  }

  function selectMark(mark, btn, row, k) {
    const m = load();
    m[k] = mark;
    save(m);
    applyMark(row, btn, mark);
    closeDropdown();
  }

  // ── メイン ───────────────────────────────────────────────
  function init() {
    const table = document.getElementById('raceTable');
    if (!table) return;

    injectStyles();

    // 印列ヘッダーを先頭に挿入
    const headerRow = table.querySelector('thead tr');
    const th = document.createElement('th');
    th.textContent = '印';
    th.className   = 'mark-col';
    headerRow.insertBefore(th, headerRow.firstChild);

    // ソートハンドラを補正（先頭挿入で列インデックスが +1 ずれる）
    patchSort(table);

    // 各行に印セルを先頭に挿入
    const marks = load();
    table.querySelectorAll('tbody tr').forEach(row => {
      // 馬番は挿入前の children[1]（= 元の 2列目）
      const umaban = row.children[1]?.textContent.trim() || String(Math.random());
      const k      = 'u' + umaban;
      const cur    = marks[k] || '';

      const td  = document.createElement('td');
      td.className = 'mark-cell';
      const btn = document.createElement('button');
      btn.className = 'mark-btn';
      btn.type      = 'button';
      applyMark(row, btn, cur);

      btn.addEventListener('click', e => {
        e.stopPropagation();
        if (activeDropdown) { closeDropdown(); return; }
        showDropdown(btn, row, k);
      });

      td.appendChild(btn);
      row.insertBefore(td, row.firstChild); // 先頭に挿入
    });

    // ページ外クリック・ESCでドロップダウンを閉じる
    document.addEventListener('click', closeDropdown);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDropdown(); });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
