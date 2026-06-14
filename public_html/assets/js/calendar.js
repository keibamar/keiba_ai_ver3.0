// calendar.js
let currentDate = new Date();

function renderCalendar(year, month) {
  const calendar = document.getElementById("calendar");
  const monthYear = document.getElementById("monthYear");

  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);

  const weekDays = ["月", "火", "水", "木", "金", "土", "日"];
  html = "<tr>";
  weekDays.forEach((d, i) => {
    let className = "";
    if (i === 5) className = "sat"; // 土曜
    if (i === 6) className = "sun"; // 日曜
    html += `<th class="${className}">${d}</th>`;
  });
  html += "</tr><tr>";

  let startDay = firstDay.getDay();
  startDay = (startDay === 0) ? 6 : startDay - 1;
  for (let i = 0; i < startDay; i++) {
    html += "<td></td>";
  }

  for (let d = 1; d <= lastDay.getDate(); d++) {
    const dateStr = `${year}${String(month+1).padStart(2,"0")}${String(d).padStart(2,"0")}`;
    let link = "";
    if (window.racedays && window.racedays.includes(dateStr)) {
      link = `<a href="../races/${dateStr}/index.html">${d}</a>`;
    } else {
      link = d;
    }

    html += `<td>${link}</td>`;

    if ((startDay + d) % 7 === 0) {
      html += "</tr><tr>";
    }
  }

  html += "</tr>";
  calendar.innerHTML = html;

  monthYear.textContent = `${year}年 ${month + 1}月`;
}

// 本日のレースリンクを表示
function showTodayRaceLink() {
  const todayRaceDiv = document.getElementById("todayRace");
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");
  const todayStr = `${yyyy}${mm}${dd}`;

  if (window.racedays && window.racedays.includes(todayStr)) {
    todayRaceDiv.innerHTML = `
      <a href="../races/${todayStr}/index.html">
        本日のレースを見る (${yyyy}/${mm}/${dd})
      </a>
    `;
  } else {
    todayRaceDiv.innerHTML = `<p>本日開催のレースはありません</p>`;
  }
}

// ボタン制御
document.getElementById("prevMonth").onclick = () => {
  currentDate.setMonth(currentDate.getMonth() - 1);
  renderCalendar(currentDate.getFullYear(), currentDate.getMonth());
};
document.getElementById("nextMonth").onclick = () => {
  currentDate.setMonth(currentDate.getMonth() + 1);
  renderCalendar(currentDate.getFullYear(), currentDate.getMonth());
};

// 初期描画
renderCalendar(currentDate.getFullYear(), currentDate.getMonth());
showTodayRaceLink();
