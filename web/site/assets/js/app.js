const DATA_PATH = "data/predictions.json";

// JSON読み込み＆テーブル描画
async function loadAndRender(){
    const res = await fetch(DATA_PATH);
    const data = await res.json();
    document.getElementById("race-date").textContent = `${data.date}`;
    document.getElementById("cource").textContent = `${data.cource}`;
    document.getElementById("race-num").textContent = `${data.race_num}`;
    document.getElementById("race-title").textContent = `${data.race_title}`;

    const table = document.getElementById("predictions-table").querySelector("tbody");
    table.innerHTML = ""; // 前回行削除

    data.predictions.forEach((horse, idx)=>{
        const row = table.insertRow();
        row.classList.add("data-row");

        row.insertCell().textContent = idx+1;
        row.insertCell().textContent = horse.horse;
        row.insertCell().textContent = horse.weight_now;
        row.insertCell().textContent = horse.weight_diff;
        row.insertCell().textContent = horse.last3f_time.toFixed(2);
        row.insertCell().textContent = horse.last3f_diff.toFixed(2);
        row.insertCell().textContent = horse.odds.toFixed(2);
        row.insertCell().textContent = horse.score;
        row.insertCell().textContent = horse.expected_value.toFixed(2);
  });

  enableSorting();
}

// 列クリックで昇降順ソート
let sortDirection = {};
function enableSorting(){
  const table = document.getElementById("predictions-table");
  const headers = table.querySelectorAll("th");
  headers.forEach((th,i)=>{
    th.onclick = () => sortTableByColumn(table, i);
  });
}

function sortTableByColumn(table,colIndex){
  const tbody = table.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr.data-row"));
  if(rows.length===0) return;

  const sample = rows[0].cells[colIndex].textContent.trim();
  const isNum = !isNaN(Number(sample));

  const dir = sortDirection[colIndex]==='asc' ? 'desc' : 'asc';
  sortDirection[colIndex] = dir;

  rows.sort((a,b)=>{
    let va = a.cells[colIndex].textContent.trim();
    let vb = b.cells[colIndex].textContent.trim();
    if(isNum){va=Number(va); vb=Number(vb);}
    return (va>vb?1:va<vb?-1:0)*(dir==='asc'?1:-1);
  });

  rows.forEach(r=>tbody.appendChild(r));
}

loadAndRender().catch(e=>console.error("読み込み失敗:",e));
