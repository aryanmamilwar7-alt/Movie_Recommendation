"""
Builds index.html - a self-contained movie recommendation website.

The Python model (movie_recommender.py) is trained on the ratings in ./data and
its movie-to-movie similarity matrix (content-based TF-IDF + item-based
collaborative filtering) is embedded in the page. The page then asks the visitor
for favourite genres and movies they liked, and recommends movies from that.

Run:   py build_website.py       ->  creates index.html  (double-click to open)
"""
import json
import os
import numpy as np
import movie_recommender as mr

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_MOVIES = 400          # keep page small when using a big MovieLens file


def build():
    movies, ratings = mr.load_data()
    # keep the most-rated movies only
    top_ids = ratings.movieId.value_counts().head(MAX_MOVIES).index
    movies = movies[movies.movieId.isin(top_ids)].reset_index(drop=True)
    ratings = ratings[ratings.movieId.isin(movies.movieId)]

    model = mr.Recommender(movies, ratings)
    sim = 0.5 * model.content_sim + 0.5 * np.clip(model.item_sim, 0, None)
    np.fill_diagonal(sim, 0)

    stats = ratings.groupby("movieId").rating.agg(["mean", "count"])
    items = []
    for _, row in movies.iterrows():
        s = stats.loc[row.movieId]
        items.append({
            "title": row.title,
            "genres": [g for g in str(row.genres).split("|") if g],
            "avg": round(float(s["mean"]), 2),
            "n": int(s["count"]),
        })
    genres = sorted({g for it in items for g in it["genres"]})
    data = {
        "movies": items,
        "genres": genres,
        "sim": np.round(sim, 3).tolist(),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = os.path.join(HERE, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built {out}  ({len(items)} movies, {len(genres)} genres, {os.path.getsize(out)//1024} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Movie Recommender</title>
<style>
  :root { --bg:#0f1220; --card:#181c30; --line:#2a2f4a; --text:#eef0fa; --mute:#9aa0c0;
          --accent:#ff5d8f; --accent2:#7c5cff; --ok:#3ddc97; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); }
  header { padding: 28px 20px 8px; text-align:center; }
  header h1 { margin:0; font-size: 2rem; }
  header p { margin:6px 0 0; color: var(--mute); }
  main { max-width: 1100px; margin: 0 auto; padding: 16px 20px 60px;
         display:grid; grid-template-columns: 380px 1fr; gap: 22px; }
  @media (max-width: 860px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--card); border:1px solid var(--line); border-radius: 14px; padding: 18px; }
  h2 { margin: 0 0 4px; font-size: 1.05rem; }
  .hint { color: var(--mute); font-size: .85rem; margin: 0 0 12px; }
  .step { margin-bottom: 20px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { border:1px solid var(--line); background:transparent; color:var(--text); padding:6px 12px;
          border-radius: 999px; cursor:pointer; font-size:.85rem; }
  .chip.on { background: var(--accent2); border-color: var(--accent2); }
  input[type=text] { width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--line);
         background:#0f1220; color:var(--text); font-size:.95rem; }
  .row { display:flex; align-items:center; gap:8px; padding:8px 0; border-bottom:1px solid var(--line); }
  .row .t { flex:1; font-size:.9rem; }
  .stars button { background:none; border:none; cursor:pointer; font-size:1.2rem; color:#555b7d; padding:0 1px; }
  .stars button.on { color:#ffc94d; }
  .x { background:none; border:none; color:var(--mute); cursor:pointer; font-size:1.1rem; }
  input[type=range] { width:100%; accent-color: var(--accent); }
  .slabels { display:flex; justify-content:space-between; color:var(--mute); font-size:.78rem; }
  .btns { display:flex; gap:10px; }
  button.primary { flex:1; padding:12px; border:none; border-radius:10px; color:white; font-weight:600;
         cursor:pointer; font-size:1rem; background: linear-gradient(90deg,var(--accent),var(--accent2)); }
  button.ghost { padding:12px 14px; border-radius:10px; border:1px solid var(--line);
         background:transparent; color:var(--text); cursor:pointer; }
  .results { display:grid; gap:12px; }
  .rec { display:grid; grid-template-columns: 40px 1fr; gap: 12px; padding:14px; border-radius:12px;
         border:1px solid var(--line); background:#13172a; }
  .rank { font-size:1.4rem; font-weight:700; color:var(--accent); }
  .rec h3 { margin:0 0 4px; font-size:1.05rem; }
  .tags { display:flex; flex-wrap:wrap; gap:6px; margin: 6px 0; }
  .tag { font-size:.72rem; padding:2px 8px; border-radius:999px; background:#232846; color:var(--mute); }
  .tag.hit { background:#2c2562; color:#cfc6ff; }
  .bar { height:6px; border-radius:99px; background:#232846; overflow:hidden; margin-top:8px; }
  .bar > i { display:block; height:100%; background: linear-gradient(90deg,var(--ok),var(--accent2)); }
  .why { color:var(--mute); font-size:.85rem; }
  .meta { color:var(--mute); font-size:.8rem; }
  .empty { color:var(--mute); text-align:center; padding:50px 10px; }
</style>
</head>
<body>
<header>
  <h1>&#127916; Movie Recommender</h1>
  <p>Tell us what you like &mdash; get movies picked for you.</p>
</header>

<main>
  <section class="card" id="inputs">
    <div class="step">
      <h2>1. Pick your favourite genres</h2>
      <p class="hint">Choose as many as you like.</p>
      <div class="chips" id="genreChips"></div>
    </div>

    <div class="step">
      <h2>2. Add movies you've watched</h2>
      <p class="hint">Start typing a title, choose it, then give it stars.</p>
      <input type="text" id="movieInput" list="movieList" placeholder="e.g. Inception" autocomplete="off">
      <datalist id="movieList"></datalist>
      <div id="ratedList"></div>
    </div>

    <div class="step">
      <h2>3. How should we choose?</h2>
      <input type="range" id="gw" min="0" max="100" value="35">
      <div class="slabels"><span>Similar to my movies</span><span>Match my genres</span></div>
    </div>

    <div class="step">
      <h2>4. How many suggestions?</h2>
      <input type="range" id="topn" min="3" max="20" value="8">
      <div class="slabels"><span>3</span><span id="topnVal">8</span><span>20</span></div>
    </div>

    <div class="btns">
      <button class="primary" id="go">Recommend movies</button>
      <button class="ghost" id="example">Example</button>
      <button class="ghost" id="reset">Reset</button>
    </div>
  </section>

  <section class="card">
    <h2>Your recommendations</h2>
    <p class="hint" id="summary">Fill in the left side and press <b>Recommend movies</b>.</p>
    <div class="results" id="results"><div class="empty">Nothing yet.</div></div>
  </section>
</main>

<script>
const DATA = __DATA__;

// CORE-START
// Recommendation logic (pure function; uses the trained similarity matrix in DATA).
function recommend(state, DATA, topN) {
  const { movies, sim } = DATA;
  const rated = state.rated;                               // { movieIndex: stars 1..5 }
  const ratedIdx = Object.keys(rated).map(Number);
  const sel = new Set(state.genres);
  const gw = ratedIdx.length ? state.genreWeight : 1;      // no movies rated -> rely on genres
  const cands = [];
  for (let m = 0; m < movies.length; m++) {
    if (rated[m] !== undefined) continue;
    let aff = 0, bestJ = -1, bestV = 0;
    for (const j of ratedIdx) {
      const w = (rated[j] - 2.5) / 2.5;                    // 5 stars -> +1, 1 star -> -0.6
      const v = sim[m][j] * w;
      aff += v;
      if (v > bestV) { bestV = v; bestJ = j; }
    }
    if (ratedIdx.length) aff /= ratedIdx.length;
    const hits = movies[m].genres.filter(g => sel.has(g));
    const gm = sel.size ? hits.length / sel.size : 0;
    cands.push({ m, aff, gm, hits, bestJ, quality: movies[m].avg / 5 });
  }
  const affs = cands.map(c => c.aff);
  const lo = Math.min(...affs), hi = Math.max(...affs);
  for (const c of cands) {
    const affN = hi > lo ? (c.aff - lo) / (hi - lo) : 0;
    c.score = (1 - gw) * affN + gw * c.gm + 0.15 * c.quality;
  }
  cands.sort((a, b) => b.score - a.score);
  const top = cands.slice(0, topN);
  const maxS = top.length ? top[0].score : 1;
  return top.map(c => ({
    index: c.m,
    title: movies[c.m].title,
    genres: movies[c.m].genres,
    hits: c.hits,
    avg: movies[c.m].avg,
    match: Math.max(1, Math.round(100 * c.score / maxS)),
    because: c.bestJ >= 0 ? movies[c.bestJ].title : null,
  }));
}
// CORE-END

// ---------- UI ----------
const state = { genres: [], rated: {} };
const $ = id => document.getElementById(id);
const titleToIdx = {};
DATA.movies.forEach((m, i) => { titleToIdx[m.title.toLowerCase()] = i; });

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// genre chips
DATA.genres.forEach(g => {
  const b = el('button', 'chip', g);
  b.onclick = () => {
    const i = state.genres.indexOf(g);
    if (i >= 0) state.genres.splice(i, 1); else state.genres.push(g);
    b.classList.toggle('on', i < 0);
  };
  $('genreChips').appendChild(b);
});

// movie autocomplete
const dl = $('movieList');
DATA.movies.slice().sort((a, b) => a.title.localeCompare(b.title)).forEach(m => {
  const o = document.createElement('option'); o.value = m.title; dl.appendChild(o);
});
$('movieInput').addEventListener('input', e => {
  const idx = titleToIdx[e.target.value.trim().toLowerCase()];
  if (idx !== undefined) {
    if (state.rated[idx] === undefined) state.rated[idx] = 4;
    e.target.value = '';
    renderRated();
  }
});

function renderRated() {
  const box = $('ratedList'); box.innerHTML = '';
  Object.keys(state.rated).forEach(k => {
    const idx = Number(k);
    const row = el('div', 'row');
    row.appendChild(el('div', 't', DATA.movies[idx].title));
    const stars = el('div', 'stars');
    for (let s = 1; s <= 5; s++) {
      const b = el('button', state.rated[idx] >= s ? 'on' : '', '★');
      b.title = s + ' star' + (s > 1 ? 's' : '');
      b.onclick = () => { state.rated[idx] = s; renderRated(); };
      stars.appendChild(b);
    }
    row.appendChild(stars);
    const x = el('button', 'x', '×');
    x.title = 'Remove';
    x.onclick = () => { delete state.rated[idx]; renderRated(); };
    row.appendChild(x);
    box.appendChild(row);
  });
}

$('topn').oninput = e => { $('topnVal').textContent = e.target.value; };

function show() {
  const res = $('results'); res.innerHTML = '';
  const nRated = Object.keys(state.rated).length;
  if (!nRated && !state.genres.length) {
    $('summary').textContent = 'Please pick at least one genre or add a movie you liked.';
    res.appendChild(el('div', 'empty', 'Nothing yet.'));
    return;
  }
  const recs = recommend({ genres: state.genres, rated: state.rated,
                           genreWeight: Number($('gw').value) / 100 }, DATA, Number($('topn').value));
  $('summary').textContent = 'Based on ' + (state.genres.length ? state.genres.join(', ') : 'no genre filter') +
      (nRated ? ' and ' + nRated + ' movie' + (nRated > 1 ? 's' : '') + ' you rated.' : '.');
  recs.forEach((r, i) => {
    const card = el('div', 'rec');
    card.appendChild(el('div', 'rank', String(i + 1)));
    const body = el('div');
    body.appendChild(el('h3', '', r.title));
    const tags = el('div', 'tags');
    r.genres.forEach(g => tags.appendChild(el('span', 'tag' + (r.hits.includes(g) ? ' hit' : ''), g)));
    body.appendChild(tags);
    let why = 'Popular with other viewers';
    if (r.because) why = 'Because you liked “' + r.because + '”';
    else if (r.hits.length) why = 'Matches your favourite genres';
    body.appendChild(el('div', 'why', why));
    body.appendChild(el('div', 'meta', 'Average rating ' + r.avg.toFixed(1) + ' / 5  •  match ' + r.match + '%'));
    const bar = el('div', 'bar'); const fill = el('i'); fill.style.width = r.match + '%';
    bar.appendChild(fill); body.appendChild(bar);
    card.appendChild(body);
    res.appendChild(card);
  });
}
$('go').onclick = show;

$('reset').onclick = () => {
  state.genres = []; state.rated = {};
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('on'));
  renderRated();
  $('results').innerHTML = '<div class="empty">Nothing yet.</div>';
  $('summary').innerHTML = 'Fill in the left side and press <b>Recommend movies</b>.';
};

$('example').onclick = () => {
  $('reset').onclick();
  ['Sci-Fi', 'Action'].forEach(g => {
    document.querySelectorAll('.chip').forEach(c => { if (c.textContent === g) c.click(); });
  });
  [['inception', 5], ['the matrix', 5], ['titanic', 2]].forEach(([t, s]) => {
    const i = titleToIdx[t]; if (i !== undefined) state.rated[i] = s;
  });
  renderRated(); show();
};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build()
