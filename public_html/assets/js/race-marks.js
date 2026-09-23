// race-marks.js — ユーザー予想マーク機能
(function () {
  'use strict';

  var MARKS  = ['', '◎', '〇', '▲', '△', '☆', '消'];
  var COLORS = { '◎': '#e53935', '〇': '#1565c0', '▲': '#6a1b9a',
                 '△': '#2e7d32', '☆': '#f57c00', '消': '#aaa' };

  function storageKey() {
    return 'race_marks' + location.pathname;
  }
  function loadMarks() {
    try { return JSON.parse(localStorage.getItem(storageKey()) || '{}'); }
    catch (e) { return {}; }
  }
  function saveMarks(m) {
    try { localStorage.setItem(storageKey(), JSON.stringify(m)); } catch (e) {}
  }

  function applyMark(row, btn, mark) {
    row.dataset.mark = mark || '';
    btn.textContent  = mark || '−';
    btn.style.color      = mark ? (COLORS[mark] || '#333') : '#ccc';
    btn.style.fontWeight = (mark && mark !== '消') ? 'bold' : 'normal';
  }

  function injectStyles() {
    var s = document.createElement('style');
    s.textContent = [
      '.mark-col { width:34px; text-align:center; user-select:none; }',
      '.mark-cell { text-align:center; padding:1px 0; }',
      '.mark-btn {',
      '  background:none; border:1px solid #ddd; border-radius:3px;',
      '  font-size:1em; cursor:pointer; min-width:26px; padding:1px 2px;',
      '  line-height:1.3; transition:border-color .1s;',
      '}',
      '.mark-btn:hover { border-color:#999; background:#f5f5f5; }',
      /* 消: グレーアウト */
      'tr[data-mark="消"] td { opacity:0.4; }',
      /* ◎: 赤い左ボーダー＋薄い背景 */
      'tr[data-mark="◎"] td { background-color:#fff5f5 !important; }',
      'tr[data-mark="◎"] td:first-child { border-left:3px solid #e53935; }',
      /* 〇: 青い左ボーダー */
      'tr[data-mark="〇"] td:first-child { border-left:3px solid #1565c0; }',
      /* ▲: 紫の左ボーダー */
      'tr[data-mark="▲"] td:first-child { border-left:3px solid #6a1b9a; }',
    ].join('\n');
    document.head.appendChild(s);
  }

  function init() {
    var table = document.getElementById('raceTable');
    if (!table) return;

    injectStyles();

    var marks = loadMarks();

    // ヘッダー列を追加（末尾）
    var headerRow = table.querySelector('thead tr');
    var th = document.createElement('th');
    th.textContent = '印';
    th.className = 'mark-col';
    headerRow.appendChild(th);

    // 各行に印セルを追加
    var rows = table.querySelectorAll('tbody tr');
    rows.forEach(function (row) {
      // 馬番（2列目 = index 1）をキーに使用
      var umabanCell = row.children[1];
      var umaban = umabanCell ? umabanCell.textContent.trim() : String(Math.random());
      var key = 'u' + umaban;
      var currentMark = marks[key] || '';

      var td = document.createElement('td');
      td.className = 'mark-cell';

      var btn = document.createElement('button');
      btn.className = 'mark-btn';
      btn.type = 'button';
      applyMark(row, btn, currentMark);

      btn.addEventListener('click', function () {
        var m = loadMarks();
        var cur = m[key] || '';
        var nextIdx = (MARKS.indexOf(cur) + 1) % MARKS.length;
        var next = MARKS[nextIdx];
        m[key] = next;
        saveMarks(m);
        applyMark(row, btn, next);
      });

      td.appendChild(btn);
      row.appendChild(td);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
