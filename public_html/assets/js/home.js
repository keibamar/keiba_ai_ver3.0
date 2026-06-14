// js/home.js
console.log("home.js loaded");

const INDEX_PATH = "data/race_index.json";

async function loadRaceIndex() {
  try {
    const res = await fetch(INDEX_PATH);
    const data = await res.json();
    console.log("race index loaded:", data);

    const tabsContainer = document.querySelector(".tabs");
    const raceList = document.getElementById("race-list");

    // 日付ごとにタブを作る
    Object.keys(data).forEach(date => {
      const btn = document.createElement("button");
      btn.textContent = date;
      btn.onclick = () => renderRaceList(date, data[date]);
      tabsContainer.appendChild(btn);
    });

    // 最初の日付を表示
    const firstDate = Object.keys(data)[0];
    if (firstDate) {
      renderRaceList(firstDate, data[firstDate]);
    }

  } catch (e) {
    console.error("レース一覧の読み込み失敗:", e);
  }
}

function renderRaceList(date, venues) {
  const raceList = document.getElementById("race-list");
  raceList.innerHTML = ""; // リセット

  const dateTitle = document.createElement("h2");
  dateTitle.textContent = `${date} の開催場一覧`;
  raceList.appendChild(dateTitle);

  // 開催場ごとに表を作成
  Object.keys(venues).forEach(venue => {
    const section = document.createElement("div");
    section.classList.add("venue-block");

    const h3 = document.createElement("h3");
    h3.textContent = venue;
    section.appendChild(h3);

    const ul = document.createElement("ul");
    venues[venue].forEach(raceNo => {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.textContent = `${venue} ${raceNo}`;
      // リンク先は races/yyyy-mm-dd/venue_rX.html の形式にする
      link.href = `races/${date}/${venue}_${raceNo.toLowerCase()}.html`;
      li.appendChild(link);
      ul.appendChild(li);
    });
    section.appendChild(ul);

    raceList.appendChild(section);
  });
}

loadRaceIndex();
