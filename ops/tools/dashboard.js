/* ops/tools/dashboard.js —— /metrics 看板的解析与渲染。
 * 纯手写、零依赖、零构建：浏览器直接 <script src="/dashboard.js">。
 * 解析的是 Prometheus 文本暴露格式（vLLM 的 /metrics 输出）。
 *
 * 注意：counter 是从「引擎启动」开始累计的，引擎重启会归零 ⇒ 所有速率都用差分算，
 *       差分出现负值就当作计数重置处理（显示为 —）。
 *
 * 渲染层的两条硬规矩（2026-10-01 重做看板显示时定下的）：
 *   1) 数字和单位永远包在 <span class="n"> 里 ⇒ white-space:nowrap，绝不把 "19.9 ms" 断成两行；
 *   2) 长注释一律单独一行（.note），不塞进数值那一列，否则窄卡片会把数字挤断。
 */
'use strict';

/* ---------------- Prometheus 文本解析 ---------------- */

function parseNumber(s) {
  if (s === undefined || s === null) return NaN;
  const t = String(s).trim();
  if (/^\+?inf$/i.test(t)) return Infinity;
  if (/^-?inf$/i.test(t)) return -Infinity;
  if (/^nan$/i.test(t)) return NaN;
  const v = Number(t);
  return Number.isFinite(v) ? v : NaN;
}

/* 按逗号切分标签，但引号内部的逗号不切（标签值里可能出现逗号） */
function splitLabels(inner) {
  const out = [];
  let cur = '', inQ = false;
  for (const ch of inner) {
    if (ch === '"') inQ = !inQ;
    if (ch === ',' && !inQ) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

function parseText(text) {
  const samples = [];
  for (const raw of String(text).split('\n')) {
    const line = raw.trim();
    if (!line || line[0] === '#') continue;           // 注释行（# HELP / # TYPE）
    const m = line.match(/^(.*?)[ \t]+([^\s]+)(?:[ \t]+([^\s]+))?$/);
    if (!m) continue;
    const spec = m[1], valRaw = m[2], tsRaw = m[3];
    let name = spec, labels = {};
    const brace = spec.indexOf('{');
    if (brace >= 0) {
      name = spec.slice(0, brace);
      let inner = spec.slice(brace + 1);
      if (inner.endsWith('}')) inner = inner.slice(0, -1);
      for (const pair of splitLabels(inner)) {
        const eq = pair.indexOf('=');
        if (eq <= 0) continue;
        labels[pair.slice(0, eq).trim()] = pair.slice(eq + 1).trim().replace(/^"|"$/g, '');
      }
    }
    const value = parseNumber(valRaw);
    if (Number.isNaN(value)) continue;                    // 解析不出数值就丢掉（+Inf 保留）
    if (name.endsWith('_created')) continue;          // vLLM 附带的时间戳序列，看板不用
    samples.push({ name, labels, value, ts: tsRaw ? Number(tsRaw) : null });
  }
  return samples;
}

function indexByName(samples) {
  const by = Object.create(null);
  for (const s of samples) (by[s.name] || (by[s.name] = [])).push(s);
  return by;
}

function pick(by, name, want) {
  const list = by[name] || [];
  if (!want) return list;
  return list.filter(s => Object.keys(want).every(k => s.labels[k] === want[k]));
}

function sumOf(by, name, want) {
  let t = 0, any = false;
  for (const s of pick(by, name, want)) { if (Number.isFinite(s.value)) { t += s.value; any = true; } }
  return any ? t : null;
}

/* 直方图分位数：累积桶 + 线性插值（Prometheus histogram_quantile 的简化版） */
function quantile(by, base, q, want) {
  const buckets = pick(by, base + '_bucket', want)
    .map(s => ({ le: parseNumber(s.labels.le), c: s.value }))
    .filter(b => Number.isFinite(b.c))
    .sort((a, b) => a.le - b.le);
  const count = sumOf(by, base + '_count', want);
  const sum = sumOf(by, base + '_sum', want);
  if (!count || count <= 0) return null;
  const target = q * count;
  let prevC = 0, prevLe = 0;
  for (const b of buckets) {
    if (b.c >= target) {
      const lo = prevLe, hi = Number.isFinite(b.le) ? b.le : (sum / count);
      if (hi <= lo) return hi;
      return lo + (hi - lo) * (target - prevC) / Math.max(1, b.c - prevC);
    }
    prevC = b.c; if (Number.isFinite(b.le)) prevLe = b.le;
  }
  return sum !== null ? sum / count : null;          // 落在 +Inf 之外时用均值兜底
}

function meanOf(by, base, want) {
  const c = sumOf(by, base + '_count', want), s = sumOf(by, base + '_sum', want);
  return (c && s !== null && c > 0) ? s / c : null;
}

/* ---------------- 差分算速率 ---------------- */
/* prev: { name -> { value, tMs } }，累计值（对所有标签求和） */
function rate(prev, name, totalNow, nowMs) {
  if (totalNow === null) return { state: 'nodata' };
  const p = prev[name];
  if (!p) return { state: 'first' };
  const dt = (nowMs - p.tMs) / 1000;
  if (dt <= 0.2) return { state: 'fast' };
  const d = totalNow - p.value;
  if (d < -1e-6) return { state: 'reset', dt };       // 引擎重启，计数归零
  return { state: 'ok', perSec: d / dt, d, dt };
}

/* ---------------- 格式化 ---------------- */
const nf0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

/* 缺数据时的统一写法：一个灰色的破折号（.n.na），不要到处手写 <span class="dim"> */
const NA = '<span class="n na">—</span>';
function n(v, f) {                      // 数字（nowrap + 等宽 + 表格数字），缺数据就是 NA
  if (v === null || v === undefined || !Number.isFinite(v)) return NA;
  return '<span class="n">' + (f || nf0).format(v) + '</span>';
}
function unit(u) { return u ? '<span class="u">' + esc(u) + '</span>' : ''; }
function sep() { return '<span class="sep">/</span>'; }
function pair(a, b, f) { return n(a, f) + sep() + n(b, f); }           // 两个同单位指标并排；f 是 Intl 格式化器
function pairH(a, b, fn) { return fn(a) + sep() + fn(b); }             // fn 是“返回整段 HTML”的格式化器（secs/ms/mib/pct）

function pct(v) { return v === null || !Number.isFinite(v) ? NA : n(v * 100, nf1) + '<span class="u">%</span>'; }
/* <1 s 用毫秒更好读（本项目 TTFT 常在几百毫秒） */
function secs(v) {
  if (v === null || !Number.isFinite(v)) return NA;
  if (v < 1) return n(v * 1000, nf0) + '<span class="u">ms</span>';
  if (v < 10) return n(v, nf2) + '<span class="u">s</span>';
  return n(v, nf1) + '<span class="u">s</span>';
}
function ms(v) {
  if (v === null || !Number.isFinite(v)) return NA;
  const t = v * 1000;
  return n(t, t < 100 ? nf1 : nf0) + '<span class="u">ms</span>';
}
function mib(bytes) {
  if (bytes === null || !Number.isFinite(bytes)) return NA;
  return n(bytes / 1048576, bytes > 10485760 ? nf0 : nf1) + '<span class="u">MiB</span>';
}
function gibps(bytesPerSec) {
  if (bytesPerSec === null || !Number.isFinite(bytesPerSec)) return NA;
  return n(bytesPerSec / 1073741824, nf2) + '<span class="u">GiB/s</span>';
}
function dur(v) {
  if (v === null || !Number.isFinite(v)) return NA;
  const d = Math.floor(v / 86400), h = Math.floor(v % 86400 / 3600), m = Math.floor(v % 3600 / 60), s = Math.floor(v % 60);
  return '<span class="n">' + esc((d ? d + 'd ' : '') + (h ? h + 'h ' : '') + m + 'm ' + s + 's') + '</span>';
}
function note(text) { return text ? '<span class="note">' + text + '</span>' : ''; }   // text 必须是已经拼好的 HTML

/* ---------------- 图形小工具 ---------------- */

/* 迷你曲线：viewBox 固定 100×30，靠 CSS 拉满卡片宽度；
 * vector-effect:non-scaling-stroke 保证线宽在非等比缩下放不变形。 */
function chart(arr, opts) {
  opts = opts || {};
  const W = 100, H = 30, P = 1.5;
  const vals = (arr || []).filter(v => Number.isFinite(v));
  if (vals.length < 2) {
    return '<svg class="chart idle" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
      '<line class="base" x1="0" y1="' + H + '" x2="' + W + '" y2="' + H + '"/>' +
      '<line class="idle" x1="0" y1="' + (H / 2) + '" x2="' + W + '" y2="' + (H / 2) + '"/>' +
      '</svg>';
  }
  let lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  const flat = hi === lo;                                        // 常数序列：画平线，别伪造 min/max 范围
  if (flat) { lo -= 0.5; hi += 0.5; }
  const span = hi - lo, k = vals.length;
  const xy = vals.map((v, i) => {
    const x = P + i * (W - 2 * P) / (k - 1);
    const y = H - P - (v - lo) / span * (H - 2 * P);
    return x.toFixed(2) + ',' + y.toFixed(2);
  });
  const last = xy[xy.length - 1].split(',');
  const area = xy.join(' ') + ' ' + last[0] + ',' + H + ' ' + xy[0].split(',')[0] + ',' + H;
  const fmt = opts.fmt || (v => nf0.format(v));
  const cap = flat
    ? '<div class="cap">恒定 <b>' + fmt(vals[vals.length - 1]) + '</b> · ' + vals.length + ' 个采样（这一段没动过）</div>'
    : '<div class="cap">min <b>' + fmt(lo) + '</b> · max <b>' + fmt(hi) +
      '</b> · 当前 <b>' + fmt(vals[vals.length - 1]) + '</b> · ' + vals.length + ' 个采样</div>';
  return '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
    '<line class="base" x1="0" y1="' + H + '" x2="' + W + '" y2="' + H + '"/>' +
    '<polygon class="area" points="' + area + '"/>' +
    '<polyline class="line" points="' + xy.join(' ') + '"/>' +
    '<line class="cursor" x1="' + last[0] + '" y1="0" x2="' + last[0] + '" y2="' + H + '"/>' +
    '</svg>' + cap;
}

function bar(p, tone) {
  if (p === null || !Number.isFinite(p)) return '<div class="bar"><i style="width:0"></i></div>';
  const w = Math.max(0, Math.min(1, p)) * 100;
  return '<div class="bar ' + (tone || '') + '"><i style="width:' + w.toFixed(1) + '%"></i>' +
    '<span class="tick" style="left:50%"></span></div>';
}
function barrow(p, tone, label) {                // 条 + 右边的百分比数字
  return '<div class="barrow"><span class="k" style="flex:none;font-size:11px;color:var(--fg-dim)">' +
    esc(label) + '</span>' + bar(p, tone) + '<span class="pctv">' + (p === null || !Number.isFinite(p) ? '—' : nf1.format(p * 100) + '%') + '</span></div>';
}
/* 阈值上色（判读标准写死，方便一眼看出异常） */
function usageTone(p)  { if (p === null || !Number.isFinite(p)) return ''; if (p < 0.7) return 'ok'; if (p < 0.9) return 'warn'; return 'bad'; }
function acceptTone(p) { if (p === null || !Number.isFinite(p)) return ''; if (p >= 0.7) return 'ok'; if (p >= 0.4) return 'warn'; return 'bad'; }
function hitTone(p)    { if (p === null || !Number.isFinite(p)) return ''; if (p >= 0.5) return 'ok'; if (p >= 0.2) return 'warn'; return 'bad'; }

/* ---------------- 结构小工具 ---------------- */

function kv(label, valueHtml, noteHtml, vcls) {
  return '<div class="kv"><span class="k">' + esc(label) + '</span>' +
    '<span class="v' + (vcls ? ' ' + vcls : '') + '">' + valueHtml + '</span>' + note(noteHtml) + '</div>';
}
function big(label, valueHtml, unitStr, noteHtml) {
  return '<div class="big"><div class="bl">' + esc(label) + '</div>' +
    '<div class="mv">' + valueHtml + (unitStr ? '<span class="mu">' + esc(unitStr) + '</span>' : '') + '</div>' +
    note(noteHtml) + '</div>';
}
function kpi(label, valueHtml, unitStr, footHtml, graphic) {
  return '<div class="kpi"><div class="kl">' + esc(label) + '</div>' +
    '<div class="mv">' + valueHtml + (unitStr ? '<span class="mu">' + esc(unitStr) + '</span>' : '') + '</div>' +
    (graphic || '') + '<div class="foot">' + (footHtml || '') + '</div></div>';
}
function sec(title) { return '<div class="sec">' + esc(title) + '</div>'; }
function hd(title, tag, badge) {
  return '<div class="hd"><h3>' + esc(title) + '</h3>' +
    (tag ? '<span class="tag">' + esc(tag) + '</span>' : '') +
    (badge ? '<span class="badge">' + badge + '</span>' : '') + '</div>';
}
/* 返回 {outerHTML} 包装：调用方一律用 card(...).outerHTML 塞进 innerHTML（和旧版一致） */
function card(title, body, opts) {
  opts = opts || {};
  if (Array.isArray(body)) body = body.join('');   // 传数组时拼成一段，别让它被隐式 join(",") 塞进逗号
  return { outerHTML: '<section class="card"><div class="hd"><h3>' + esc(title) + '</h3>' +
    (opts.tag ? '<span class="tag">' + esc(opts.tag) + '</span>' : '') +
    (opts.badge ? '<span class="badge">' + opts.badge + '</span>' : '') + '</div>' +
    '<div class="body">' + body + '</div></section>' };
}
function table(headers, rows) {
  if (!rows) return '';
  const th = headers.map((h, i) => '<th' + (h.right ? ' class="num"' : '') + '>' + esc(h.t || h) + '</th>').join('');
  return '<div class="tblwrap"><table class="tbl"><thead><tr>' + th + '</tr></thead><tbody>' + rows + '</tbody></table></div>';
}
function td(t, cls) { return '<td' + (cls ? ' class="' + cls + '"' : '') + '>' + t + '</td>'; }

/* counter 差分的可读呈现：窗口增量，或「为什么算不出来」 */
function dstr(r) {
  if (r.state === 'ok') return n(r.d);
  if (r.state === 'reset') return '<span class="bad">已归零（引擎重启过）</span>';
  return NA;
}
function why(r) {                        // 差分算不出来时，说清楚原因而不是只给一个破折号
  if (r.state === 'ok') return '';
  if (r.state === 'reset') return '';
  if (r.state === 'first') return '看板刚启动，还没有上一次抓取可以做差分';
  if (r.state === 'fast') return '两次抓取间隔太短（&lt;0.2 s），差分窗口不可信';
  if (r.state === 'nodata') return '引擎没有暴露这个指标';
  return '';
}

/* ---------------- 状态 ---------------- */

/* 需要算速率的 counter：短名 → 真实指标名（名字必须一字不差，不能拼接） */
const COUNTERS = {
  prompt:  'vllm:prompt_tokens_total',
  gen:     'vllm:generation_tokens_total',
  draftT:  'vllm:spec_decode_num_draft_tokens_total',
  accT:    'vllm:spec_decode_num_accepted_tokens_total',
  drafts:  'vllm:spec_decode_num_drafts_total',
  hits:    'vllm:prefix_cache_hits_total',
  queries: 'vllm:prefix_cache_queries_total',
  pre:     'vllm:num_preemptions_total',
  cached:  'vllm:prompt_tokens_cached_total',
  readB:   'vllm:estimated_read_bytes_per_gpu_total',
  writeB:  'vllm:estimated_write_bytes_per_gpu_total',
};

const HIST_MAX = 90;                     // 每条曲线保留 ~90 个采样点
const STALE_S = 15;                      // 超过这么久没成功抓取 ⇒ 明确标成“数据陈旧”
const state = {
  prev: Object.create(null),             // counter 累计值快照（差分用）
  hist: { kv: [], tps: [], ptps: [], run: [], wait: [] },
  lastText: '',
  lastModels: null,
  health: null,
  failStreak: 0,
  prevT: 0,
  samples: [],
  index: null,
  upstream: null,
};

let timer = null;

/* ---------------- 局部渲染片段 ---------------- */

function perConcurrencyTable(poolTokens, maxLen) {
  if (poolTokens === null) return '';
  const rows = [1, 2, 3, 4, 6, 8].map(nn => {
    const perPool = poolTokens / nn;
    const per = Math.min(perPool, maxLen === null ? Infinity : maxLen);
    const capped = maxLen !== null && perPool > maxLen;
    return '<tr>' + td(String(nn), 'num') + td(n(Math.floor(per), nf0), 'num') +
      td('<span class="dim">' + (capped ? '受 max_model_len 限制' : '受 KV 池限制') + '</span>') + '</tr>';
  }).join('');
  return table(['同时在跑', { t: '每条可用上下文', right: 1 }, '由谁决定'], rows) +
    '<div class="note">KV 池 ÷ 并发数就是每条请求能占用的上下文；取不到 max_model_len 时按池算。</div>';
}

function rateTable(R, T) {
  const rows = [
    ['生成 token', R.gen, T.gen],
    ['预填 token', R.prompt, T.prompt],
    ['draft token', R.draftT, T.draftT],
    ['接受 token', R.accT, T.accT],
    ['前缀命中 token', R.hits, T.hits],
    ['前缀查询 token', R.queries, T.queries],
    ['抢占次数', R.pre, T.pre],
  ];
  const body = rows.map(row => {
    const r = row[1], tot = row[2];
    const perSec = r.state === 'ok' ? n(r.perSec, nf1) : NA;
    return '<tr>' + td(esc(row[0])) + td(dstr(r), 'num') + td(perSec, 'num') + td(n(tot), 'num') + '</tr>';
  }).join('');
  return table([{ t: 'counter', right: 0 }, { t: '本窗口 Δ', right: 1 }, { t: '每秒', right: 1 }, { t: '自启动累计', right: 1 }], body);
}

/* ---------------- 渲染 ---------------- */

function render(nowMs) {
  const by = state.index;
  const el = id => document.getElementById(id);

  /* ---- 计数器总量（跨标签求和）与差分 ---- */
  const T = {}, R = {};
  for (const k of Object.keys(COUNTERS)) {
    T[k] = sumOf(by, COUNTERS[k]);
    R[k] = rate(state.prev, COUNTERS[k], T[k], nowMs);
  }

  /* ---- 仪表值 ---- */
  const kvPerc = sumOf(by, 'vllm:kv_cache_usage_perc');
  const running = sumOf(by, 'vllm:num_requests_running');
  const waiting = sumOf(by, 'vllm:num_requests_waiting');
  const cfgList = pick(by, 'vllm:cache_config_info');
  const cfg = cfgList.length ? cfgList[0].labels : {};
  const poolTokens = Number(cfg.kv_cache_size_tokens);
  const maxConc = Number(cfg.kv_cache_max_concurrency);
  const maxLen = state.lastModels && state.lastModels.max_model_len ? state.lastModels.max_model_len : null;

  /* ---- 历史记录 ---- */
  const tps = (R.gen.state === 'ok') ? R.gen.perSec : null;
  const ptps = (R.prompt.state === 'ok') ? R.prompt.perSec : null;
  state.hist.kv.push(kvPerc); state.hist.tps.push(tps); state.hist.ptps.push(ptps);
  state.hist.run.push(running); state.hist.wait.push(waiting);
  for (const k of Object.keys(state.hist)) { if (state.hist[k].length > HIST_MAX) state.hist[k].shift(); }

  /* ---- 顶栏 ---- */
  const healthOk = state.health === 200;
  const paused = !!(document.getElementById('pause') || {}).checked;
  const ageS = state.prevT ? (nowMs - state.prevT) / 1000 : null;
  const stale = !paused && ageS !== null && ageS > STALE_S;          // 手动暂停时不算“陈旧”
  el('dot').className = 'dot ' + (healthOk ? 'ok' : (state.health === null ? 'unk' : 'bad'));
  const pill = el('pill');
  pill.className = 'pill ' + (healthOk ? (paused ? 'unk' : (stale ? 'unk' : 'ok')) : (state.health === null ? 'unk' : 'bad'));
  pill.textContent = healthOk ? (paused ? '已暂停' : (stale ? '数据陈旧' : '引擎在线'))
    : (state.health === null ? '连不上看板' : '引擎不可达 ' + state.health);
  el('status-line').innerHTML = healthOk
    ? '指标来源 <code>' + esc(location.origin) + '</code> → 上游 <code>' + esc(state.upstream || '未知') + '</code>' +
      (paused ? ' · <span class="warn">抓取已暂停（“立即抓取”仍可用）</span>' : '')
    : '<span class="bad">上游不可达（' + esc(String(state.health === null ? '看板进程连不上引擎' : state.health)) + '）—— 看板保留最后一次成功的数据</span>';
  el('meta').innerHTML = '抓取 <b>' + new Date(nowMs).toLocaleTimeString() + '</b> · ' +
    '<b>' + (ageS === null ? '—' : nf1.format(ageS) + ' s') + '</b> 前 · ' +
    '连续失败 <b class="' + (state.failStreak ? 'bad' : '') + '">' + state.failStreak + '</b>' +
    (stale ? ' · <b class="warn">数据已停在这一刻</b>' : '');

  /* ---- KPI 条（最该一眼看到的六个数） ---- */
  const perStream = (running && tps !== null) ? tps / running : null;
  const accWin = (R.accT.state === 'ok' && R.draftT.state === 'ok' && R.draftT.d > 0) ? R.accT.d / R.draftT.d : null;
  const hitWin = (R.queries.state === 'ok' && R.queries.d > 0) ? R.hits.d / R.queries.d : null;
  const winS = R.gen.state === 'ok' ? nf1.format(R.gen.dt) : null;
  const idle = !running && (tps === null || tps === 0);
  const sparkPct = v => nf0.format(v * 100) + '%';
  const sparkTok = v => nf0.format(v);
  el('c-kpi').innerHTML = hd('一眼看板', '每次抓取刷新',
      (idle ? '<span class="tag" style="border-color:var(--ok);color:var(--ok)">空闲</span>' : '')) +
    '<div class="kpis">' +
    kpi('生成聚合', n(tps, nf1), 'tok/s',
        winS ? '窗口 Δ ' + dstr(R.gen) + ' tok / ' + winS + ' s' : (why(R.gen) || '窗口内没有生成'),
        chart(state.hist.tps, { fmt: sparkTok })) +
    kpi('预填', n(ptps, nf1), 'tok/s',
        R.prompt.state === 'ok' ? '窗口 Δ ' + dstr(R.prompt) + ' tok' : (why(R.prompt) || '窗口内没有预填'),
        chart(state.hist.ptps, { fmt: sparkTok })) +
    kpi('每流', n(perStream, nf1), 'tok/s',
        running ? '聚合 ÷ 在跑 ' + n(running) : '当前没有在跑的请求',
        '') +
    kpi('KV 占用', n(kvPerc === null ? null : kvPerc * 100, nf1), '%',
        Number.isFinite(poolTokens) ? '池 ' + nf0.format(poolTokens) + ' tok' : '池大小未知',
        bar(kvPerc, usageTone(kvPerc))) +
    kpi('MTP 接受率', n(accWin === null ? null : accWin * 100, nf1), '%',
        R.accT.state === 'ok' ? 'Δ接受 ÷ Δdraft' : (why(R.accT) || '窗口内没有 draft'),
        bar(accWin, acceptTone(accWin))) +
    kpi('在跑 / 排队', n(running) + '<span class="mu">run</span><span class="mu" style="margin-left:10px">' +
        n(waiting) + ' wait</span>', '',
        cfg.max_num_seqs ? 'max-num-seqs = ' + esc(cfg.max_num_seqs) : '并发硬顶不在 /metrics 里（默认 4）',
        chart(state.hist.run, { fmt: sparkTok })) +
    '</div>';

  /* ---- 吞吐 ---- */
  const readR = (R.readB.state === 'ok') ? R.readB.perSec : null;
  const writeR = (R.writeB.state === 'ok') ? R.writeB.perSec : null;
  el('c-throughput').innerHTML = card('吞吐与速率', [
    sec('按 counter 差分'),
    rateTable(R, T),
    note('Δ 就是两次抓取之差；每秒 = Δ ÷ 窗口（本轮窗口 ' + (R.gen.state === 'ok' ? nf1.format(R.gen.dt) + ' s' : '—') + '）。counter 从引擎启动累计，重启即归零。'),
    sec('SSD / PLE（vLLM 自己估的，不是实测带宽）'),
    kv('估算读入', gibps(readR), '来自 vllm:estimated_read_bytes_per_gpu_total；PLE 放 SSD 时看这个量级'),
    kv('估算写出', gibps(writeR), ''),
    kv('本窗口 Δ 读 / 写', dstr(R.readB) + sep() + dstr(R.writeB) + '<span class="u">bytes</span>', ''),
    sec('迭代粒度'),
    kv('每次迭代 token 数 p50 / 均值', pair(quantile(by, 'vllm:iteration_tokens_total', 0.5), meanOf(by, 'vllm:iteration_tokens_total'), nf1) + '<span class="u">tok</span>',
        '解码步 ≈2、预填 chunk =2048 ⇒ 均值被预填拉高'),
    kv('迭代次数（估算）', n(tps !== null && tps > 0 ? tps / 2 : null, nf1) + '<span class="u">it/s</span>',
        '按「每步 2 token（MTP=2 全中）」粗算，只用来判读趋势'),
  ], { tag: '窗口差分' }).outerHTML;

  /* ---- 延迟 ---- */
  const itl50 = quantile(by, 'vllm:inter_token_latency_seconds', 0.5);
  el('c-latency').innerHTML = card('延迟', [
    big('TTFT p50', secs(quantile(by, 'vllm:time_to_first_token_seconds', 0.5)), '',
        'p95 ' + secs(quantile(by, 'vllm:time_to_first_token_seconds', 0.95)) +
        ' · p99 ' + secs(quantile(by, 'vllm:time_to_first_token_seconds', 0.99)) +
        ' · 样本 ' + n(sumOf(by, 'vllm:time_to_first_token_seconds_count'), nf0)),
    kv('步时间 p50 / p95', pairH(itl50, quantile(by, 'vllm:inter_token_latency_seconds', 0.95), ms),
        '基线：MTP=1 ≈15.2–15.7 ms，MTP=2 ≈17.2–17.6 ms'),
    kv('单流隐含步时间', itl50 === null ? NA : n(itl50 * 1000, nf1) + '<span class="u">ms</span>', ''),
    kv('端到端 p50 / p95', pairH(quantile(by, 'vllm:e2e_request_latency_seconds', 0.5), quantile(by, 'vllm:e2e_request_latency_seconds', 0.95), secs), ''),
    kv('预填耗时 p50 / p95', pairH(quantile(by, 'vllm:request_prefill_time_seconds', 0.5), quantile(by, 'vllm:request_prefill_time_seconds', 0.95), secs),
        '基线：2048 ≈0.64–0.73 s · 8192 ≈2.0–2.4 s · 131072 ≈54–56 s'),
    kv('解码耗时 p50 / p95', pairH(quantile(by, 'vllm:request_decode_time_seconds', 0.5), quantile(by, 'vllm:request_decode_time_seconds', 0.95), secs), ''),
    kv('排队时间 p50 / p95', pairH(quantile(by, 'vllm:request_queue_time_seconds', 0.5), quantile(by, 'vllm:request_queue_time_seconds', 0.95), secs),
        '非零就是并发被顶满了'),
    kv('每 token 时间 p50 / p95', pairH(quantile(by, 'vllm:request_time_per_output_token_seconds', 0.5), quantile(by, 'vllm:request_time_per_output_token_seconds', 0.95), ms), ''),
    kv('prompt 长度 p50 / p95', pair(quantile(by, 'vllm:request_prompt_tokens', 0.5), quantile(by, 'vllm:request_prompt_tokens', 0.95), nf0) + '<span class="u">tok</span>', ''),
    kv('生成长度 p50 / p99', pair(quantile(by, 'vllm:request_generation_tokens', 0.5), quantile(by, 'vllm:request_max_num_generation_tokens', 0.99), nf0) + '<span class="u">tok</span>', ''),
    kv('已完成请求（累计）', n(sumOf(by, 'vllm:request_success_total'), nf0), ''),
  ], { tag: '直方图插值' }).outerHTML;

  /* ---- 投机解码 ---- */
  const accAll = (T.draftT && T.accT !== null) ? T.accT / T.draftT : null;
  const draftsNow = sumOf(by, 'vllm:spec_decode_num_drafts_total');
  const perPos = pick(by, 'vllm:spec_decode_num_accepted_tokens_per_pos_total')
    .slice().sort((a, b) => Number(a.labels.position) - Number(b.labels.position))
    .map(s => '<tr>' + td('draft #' + esc(s.labels.position || '?')) +
      td(n(s.value), 'num') +
      td(n(T.draftT ? s.value / T.draftT * 100 : null, nf1) + '<span class="u">%</span>', 'num') +
      td(n(draftsNow && s.value ? s.value / draftsNow : null, nf2), 'num') + '</tr>').join('');
  el('c-spec').innerHTML = card('MTP 投机解码', [
    barrow(accWin, acceptTone(accWin), '本窗口'),
    kv('接受率（累计）', pct(accAll), '自启动至今；本窗口那个才是“刚才这一段”的真实水平'),
    '<div class="trio">' +
      '<div class="cell"><div class="cl">draft 次数</div><div class="cv">' + n(draftsNow) + '</div><div class="cs">窗口 Δ ' + dstr(R.drafts) + '</div></div>' +
      '<div class="cell"><div class="cl">draft token</div><div class="cv">' + n(T.draftT) + '</div><div class="cs">窗口 Δ ' + dstr(R.draftT) + '</div></div>' +
      '<div class="cell"><div class="cl">接受 token</div><div class="cv">' + n(T.accT) + '</div><div class="cs">窗口 Δ ' + dstr(R.accT) + '</div></div>' +
    '</div>',
    kv('平均每次 draft 接受', (draftsNow && T.accT !== null) ? n(T.accT / draftsNow, nf2) + '<span class="u">tok</span>' : NA,
        '上限 = 1 + num_speculative_tokens（MTP=2 ⇒ ≈3.0 表示全中）'),
    table([{ t: 'draft 位置' }, { t: '累计接受', right: 1 }, { t: '占 draft token', right: 1 }, { t: '每次 draft', right: 1 }], perPos),
  ], { tag: '接受率 = Δacc ÷ Δdraft' }).outerHTML;

  /* ---- KV / 上下文 ---- */
  el('c-kv').innerHTML = card('KV 缓存与上下文上限', [
    barrow(kvPerc, usageTone(kvPerc), '占用'),
    chart(state.hist.kv, { fmt: sparkPct }),
    kv('KV 池容量', n(Number.isFinite(poolTokens) ? poolTokens : null) + '<span class="u">tokens</span>', ''),
    kv('满上下文并发', (Number.isFinite(maxConc) ? n(maxConc, nf2) + '<span class="mu">×</span>' : NA),
        '池 ÷ ' + (maxLen ? nf0.format(maxLen) : '?') + ' tokens'),
    kv('块大小 / 块数', n(cfg.block_size ? Number(cfg.block_size) : null) + '<span class="u">tok/块</span>' + sep() +
        n(cfg.num_gpu_blocks ? Number(cfg.num_gpu_blocks) : null) + '<span class="u">块</span>', ''),
    kv('单请求上限', maxLen ? n(maxLen) + '<span class="u">tokens</span>' : NA, ''),
    perConcurrencyTable(Number.isFinite(poolTokens) ? poolTokens : null, maxLen),
  ], { tag: '启动时定死' }).outerHTML;

  /* ---- 请求 / 排队 / 抢占 ---- */
  const byReason = pick(by, 'vllm:num_requests_waiting_by_reason')
    .slice().sort((a, b) => b.value - a.value)
    .map(s => '<tr>' + td(esc(s.labels.reason || '?')) + td(n(s.value), 'num') + '</tr>').join('');
  const successList = pick(by, 'vllm:request_success_total');
  const succTotal = successList.reduce((a, s) => a + (Number.isFinite(s.value) ? s.value : 0), 0);
  const success = successList.slice().sort((a, b) => b.value - a.value)
    .map(s => '<tr>' + td(esc(s.labels.finished_reason || '?')) + td(n(s.value), 'num') +
      td(n(succTotal ? s.value / succTotal * 100 : null, nf1) + '<span class="u">%</span>', 'num') + '</tr>').join('');
  el('c-requests').innerHTML = card('请求 / 排队 / 抢占', [
    '<div class="trio">' +
      '<div class="cell"><div class="cl">在跑</div><div class="cv' + (running > 0 ? ' ok' : '') + '">' + n(running) + '</div><div class="cs">窗口峰值 ' + n(state.hist.run.length ? Math.max.apply(null, state.hist.run.filter(Number.isFinite)) : null) + '</div></div>' +
      '<div class="cell"><div class="cl">排队</div><div class="cv' + (waiting > 0 ? ' warn' : '') + '">' + n(waiting) + '</div><div class="cs">窗口峰值 ' + n(state.hist.wait.length ? Math.max.apply(null, state.hist.wait.filter(Number.isFinite)) : null) + '</div></div>' +
      '<div class="cell"><div class="cl">抢占</div><div class="cv' + (T.pre > 0 ? ' bad' : '') + '">' + n(T.pre) + '</div><div class="cs">窗口 Δ ' + dstr(R.pre) + '</div></div>' +
    '</div>',
    chart(state.hist.wait, { fmt: sparkTok }),
    kv('max-num-seqs（并发硬顶）', cfg.max_num_seqs ? n(Number(cfg.max_num_seqs)) : NA,
        '不在 cache_config_info 里；本部署默认 4，看 config/engine.env 的 QWEN_SEQS'),
    kv('KV 池 token 总预算', Number.isFinite(poolTokens) ? n(poolTokens) : NA, ''),
    kv('每请求抢占次数 p95', n(quantile(by, 'vllm:request_num_preemptions', 0.95), nf2) + '<span class="u">次</span>',
        '非 0 就是 KV 不够、被抢占重算过'),
    table([{ t: '排队原因' }, { t: '数量', right: 1 }], byReason),
    table([{ t: '结束原因' }, { t: '请求数', right: 1 }, { t: '占已完成请求', right: 1 }], success),
  ], { tag: '瞬时值 + 累计' }).outerHTML;

  /* ---- 缓存 ---- */
  const hitAll = (T.queries && T.hits !== null) ? T.hits / T.queries : null;
  const mmHit = (function () {
    const h = sumOf(by, 'vllm:mm_cache_hits_total'), q = sumOf(by, 'vllm:mm_cache_queries_total');
    return q ? h / q : null;
  })();
  const extHit = (function () {
    const h = sumOf(by, 'vllm:external_prefix_cache_hits_total'), q = sumOf(by, 'vllm:external_prefix_cache_queries_total');
    return q ? h / q : null;
  })();
  const bySource = pick(by, 'vllm:prompt_tokens_by_source_total')
    .slice().sort((a, b) => b.value - a.value)
    .map(s => '<tr>' + td(esc(s.labels.source || '?'), 'lbl') + td(n(s.value), 'num') +
      td(n(T.prompt ? s.value / T.prompt * 100 : null, nf1) + '<span class="u">%</span>', 'num') + '</tr>').join('');
  el('c-cache').innerHTML = card('缓存命中', [
    barrow(hitWin, hitTone(hitWin), '本窗口'),
    kv('前缀缓存（累计）', pct(hitAll), '本窗口那个数才对应“刚才这一段”'),
    '<div class="trio">' +
      '<div class="cell"><div class="cl">命中 token</div><div class="cv">' + n(T.hits) + '</div><div class="cs">窗口 Δ ' + dstr(R.hits) + '</div></div>' +
      '<div class="cell"><div class="cl">查询 token</div><div class="cv">' + n(T.queries) + '</div><div class="cs">窗口 Δ ' + dstr(R.queries) + '</div></div>' +
      '<div class="cell"><div class="cl">复用 prompt</div><div class="cv">' + n(T.cached) + '</div><div class="cs">窗口 Δ ' + dstr(R.cached) + '</div></div>' +
    '</div>',
    note('这个版本的前缀缓存按 <b>token</b> 计，不是 block。'),
    kv('多模态缓存命中', pct(mmHit), ''),
    kv('外部 KV offload', (cfg.kv_offloading_backend || '—') + '<span class="u">后端</span>' + sep() +
        (cfg.kv_offloading_size || 'None') + '<span class="u">GiB</span>',
        'offload 后端是 CPU/磁盘那一路；PLE 表走的是自己的 O_DIRECT + AIO，不经过它'),
    kv('前缀共享', cfg.enable_prefix_caching === 'True' ? '<span class="ok">开</span>' : '<span class="bad">关</span>',
        '共享前缀的请求共用 KV 块'),
    table([{ t: 'prompt token 来源' }, { t: '累计', right: 1 }, { t: '占比', right: 1 }], bySource),
  ], { tag: '按 token 计' }).outerHTML;

  /* ---- 进程 ---- */
  const pStart = sumOf(by, 'process_start_time_seconds');
  const uptime = (pStart !== null) ? (nowMs / 1000 - pStart) : null;
  const py = pick(by, 'python_info')[0] || null;
  const maxFds = sumOf(by, 'process_max_fds'), openFds = sumOf(by, 'process_open_fds');
  el('c-process').innerHTML = card('API server 进程', [
    big('已运行', dur(uptime), '', '启动时刻 ' + (pStart !== null ? new Date(pStart * 1000).toLocaleTimeString() : '—')),
    kv('常驻内存 RSS', mib(sumOf(by, 'process_resident_memory_bytes')), '只算 API server；引擎显存/宿主内存在 bin/status.sh'),
    kv('虚拟内存', mib(sumOf(by, 'process_virtual_memory_bytes')), ''),
    kv('打开的 FD', n(openFds) + sep() + n(maxFds), maxFds && openFds / maxFds > 0.8 ? '<span class="warn">FD 快用满了</span>' : ''),
    kv('累计 CPU 时间', secs(sumOf(by, 'process_cpu_seconds_total')), 'API server 线程自己烧的 CPU，不含引擎'),
    kv('Python', py ? esc(py.labels.version || '—') : '—', ''),
    kv('GC 回收对象', n(sumOf(by, 'python_gc_objects_collected_total')), '不可回收 ' + n(sumOf(by, 'python_gc_objects_uncollectable_total'))),
  ], { tag: '进程级' }).outerHTML;

  /* ---- 配置 ---- */
  const cfgRows = Object.keys(cfg).sort().map(k =>
    '<tr>' + td(esc(k), 'lbl') + td(esc(cfg[k] === '' ? '(empty)' : cfg[k]), 'lbl') + '</tr>').join('');
  el('c-config').innerHTML = card('引擎配置', [
    kv('/v1/models', state.lastModels ? esc(state.lastModels.id) : NA,
        maxLen ? 'max_model_len = ' + n(maxLen) : 'max_model_len 未知'),
    kv('cache_config_info', cfgList.length ? n(cfgList.length) + '<span class="u">组</span>' : NA,
        '这些是启动时定死的常量，不随请求变化'),
    '<details><summary>展开全部 ' + Object.keys(cfg).length + ' 项配置</summary>' +
      '<div class="dbody">' + table([{ t: '键' }, { t: '值' }], cfgRows) + '</div></details>',
  ], { tag: '只读' }).outerHTML;

  renderRaw();
}

/* ---------------- 原始指标区（懒渲染 + 可过滤） ---------------- */

function rawRows(filter) {
  const q = (filter || '').trim().toLowerCase();
  const out = [];
  for (const s of state.samples) {
    if (!s.name.startsWith('vllm:')) continue;
    const lbl = Object.keys(s.labels).map(k => k + '=' + s.labels[k]).join(',');
    if (q && (s.name + ' ' + lbl).toLowerCase().indexOf(q) < 0) continue;
    out.push(s);
  }
  out.sort((a, b) => a.name < b.name ? -1 : (a.name > b.name ? 1 : 0));
  return out;
}

function renderRaw() {
  const det = document.getElementById('raw-det');
  const orig = document.getElementById('raw-orig');
  const box = document.getElementById('raw-body');
  const input = document.getElementById('raw-filter');
  const all = state.samples.filter(s => s.name.startsWith('vllm:'));
  const shown = rawRows(input.value);
  document.getElementById('raw-count').textContent =
    (input.value.trim() ? shown.length + ' / ' + all.length : all.length) + ' 条';
  /* 折叠着就别去重建这几百行（以前每 5 秒重建一次，纯属白烧 CPU） */
  if (!det.open) { box.innerHTML = ''; return; }
  const keep = box.scrollTop;
  const rows = shown.map(s => '<tr>' + td(esc(s.name), 'lbl') +
    td('<span class="dim">' + esc(Object.keys(s.labels).map(k => k + '=' + s.labels[k]).join(',')) + '</span>') +
    td(n(s.value, nf2), 'num') + '</tr>').join('');
  box.innerHTML = '<table class="tbl"><thead><tr><th>指标</th><th>标签</th><th class="num">值</th></tr></thead><tbody>' +
    (rows || '<tr><td colspan="3"><span class="dim">没有匹配 “' + esc(input.value.trim()) + '” 的指标</span></td></tr>') + '</tbody></table>';
  box.scrollTop = keep;
  const pre = document.getElementById('raw-orig-body');
  if (orig.open) { pre.textContent = state.lastText || '(还没有成功抓取的原文)'; }
  else { pre.textContent = ''; }
}

/* ---------------- 抓取与循环 ---------------- */

async function scrape() {
  let text = null, models = null, health = null;
  try {
    const r = await fetch('/api/health', { cache: 'no-store' });
    health = parseInt((await r.text()).trim(), 10);
  } catch (e) { health = null; }
  if (!state.upstream) {
    try { state.upstream = (await (await fetch('/api/info', { cache: 'no-store' })).json()).upstream; }
    catch (e) { state.upstream = '(上游未知)'; }
  }
  try {
    const r = await fetch('/api/metrics', { cache: 'no-store' });
    text = await r.text();
    if (!r.ok && !/vllm:/.test(text)) text = null;
  } catch (e) { text = null; }
  try {
    const r = await fetch('/api/models', { cache: 'no-store' });
    const j = await r.json();
    models = (j.data || [])[0] || null;
  } catch (e) { models = null; }

  const nowMs = Date.now();
  if (text) {
    state.lastText = text;
    state.samples = parseText(text);
    state.index = indexByName(state.samples);
    state.health = health;
    state.lastModels = models;
    state.failStreak = 0;
    render(nowMs);
    /* 差分基准：只在成功抓取后更新；名字一律用 COUNTERS 里的全名，不能拼 */
    for (const k of Object.keys(COUNTERS)) {
      const nm = COUNTERS[k];
      const tot = sumOf(state.index, nm);
      if (tot !== null) state.prev[nm] = { value: tot, tMs: nowMs };
    }
    state.prevT = nowMs;
    document.getElementById('err').style.display = 'none';
  } else {
    state.health = health === null ? null : health;
    state.failStreak++;
    render(nowMs);
    document.getElementById('err').style.display = 'block';
  }
}

function setTimer() {
  const sel = document.getElementById('interval');
  const v = Number(sel.value);
  if (timer) { clearInterval(timer); timer = null; }
  if (v > 0 && !document.getElementById('pause').checked) {
    timer = setInterval(scrape, v * 1000);
  }
  syncPauseLabel();
}

function syncPauseLabel() {
  const cb = document.getElementById('pause');
  const lbl = document.getElementById('pause-lbl');
  if (lbl) lbl.classList.toggle('on', cb.checked);
  /* 顶栏立刻跟上：不然要等到下一次抓取才能看出“已经暂停了” */
  const pill = document.getElementById('pill');
  if (state.health === 200 && pill) {
    pill.className = 'pill ' + (cb.checked ? 'unk' : 'ok');
    pill.textContent = cb.checked ? '已暂停' : '引擎在线';
  }
}

/* ---------------- 主题 ---------------- */

const THEME_KEY = 'qw-dashboard-privat…heme';
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* 隐私模式/禁用存储就算了 */ }
}
function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
}

/* ---------------- 初始化 ---------------- */

function init() {
  document.getElementById('refresh').addEventListener('click', scrape);
  document.getElementById('interval').addEventListener('change', setTimer);
  document.getElementById('pause').addEventListener('change', setTimer);
  document.getElementById('theme').addEventListener('click', toggleTheme);
  document.getElementById('raw-filter').addEventListener('input', renderRaw);
  document.getElementById('raw-det').addEventListener('toggle', renderRaw);
  document.getElementById('raw-orig').addEventListener('toggle', renderRaw);
  syncPauseLabel();
  /* Chrome 会把后台标签页的定时器节流到每分钟一次 ⇒ 切回这个标签时立刻补抓一次 */
  document.addEventListener('visibilitychange', () => { if (!document.hidden) scrape(); });
  /* 快捷键（输入框/按钮里打字时不拦截） */
  document.addEventListener('keydown', ev => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const t = ev.target;
    if (t && /INPUT|SELECT|TEXTAREA|BUTTON/.test(t.tagName)) return;
    const k = (ev.key || '').toLowerCase();
    const map = { '1': '2', '2': '5', '3': '10', '4': '30' };
    if (k === 'r') { ev.preventDefault(); scrape(); }
    else if (k === 'p') { ev.preventDefault(); const cb = document.getElementById('pause'); cb.checked = !cb.checked; setTimer(); }
    else if (k === 't') { ev.preventDefault(); toggleTheme(); }
    else if (map[k]) { ev.preventDefault(); document.getElementById('interval').value = map[k]; setTimer(); }
  });
  scrape().then(() => setTimer());
}

/* 浏览器里才启动；在 node 里加载时只暴露纯函数，方便单测 */
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}
