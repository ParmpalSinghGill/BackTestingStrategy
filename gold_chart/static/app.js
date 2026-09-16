const IST_OFFSET = 19800;
const DISPLAY_TZ = "Asia/Kolkata";
const UP_VOLUME = "#26a69a80";
const DOWN_VOLUME = "#ef535080";
const DEFAULT_DRAWING_COLOR = "#f4c430";
const DRAWING_STORAGE_KEY = "gc-chart-drawings-shared";
const REPLAY_STORAGE_KEY = "gc-chart-replay";
// TradingView-style drawing palette: greyscale, then hues from light to dark.
const TV_COLOR_GRID = [
  ["#ffffff", "#e0e3eb", "#b2b5be", "#787b86", "#5d606b", "#434651", "#2a2e39", "#000000"],
  ["#f8bbd0", "#ffcdd2", "#ffe0b2", "#fff9c4", "#c8e6c9", "#b2ebf2", "#bbdefb", "#e1bee7"],
  ["#f48fb1", "#ef9a9a", "#ffcc80", "#fff59d", "#a5d6a7", "#80deea", "#90caf9", "#ce93d8"],
  ["#e91e63", "#f23645", "#ff9800", "#f4c430", "#089981", "#00bcd4", "#2962ff", "#ab47bc"],
  ["#ad1457", "#c62828", "#ef6c00", "#f9a825", "#2e7d32", "#00838f", "#1565c0", "#6a1b9a"],
];
const TF_SECONDS = {
  "1m": 60,
  "5m": 300,
  "15m": 900,
  "1h": 3600,
  "4h": 14400,
  "1d": 86400,
  "1w": 604800,
};

const shell = document.getElementById("chart-shell");
const chartElement = document.getElementById("chart");
const canvas = document.getElementById("drawing-layer");
const ctx = canvas.getContext("2d");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const playButton = document.getElementById("play");
const selectionToolbar = document.getElementById("selection-toolbar");
const deleteOneButton = document.getElementById("delete-one");
const editTextButton = document.getElementById("edit-text");
const textEditor = document.getElementById("text-editor");
const colorToggle = document.getElementById("color-toggle");
const colorChip = document.getElementById("color-chip");
const colorPopover = document.getElementById("color-popover");
const colorGrid = document.getElementById("color-grid");
const colorInput = document.getElementById("drawing-color");
const colorOpacity = document.getElementById("color-opacity");
const colorOpacityValue = document.getElementById("color-opacity-value");
const liveDot = document.getElementById("live-dot");
const liveLabel = document.getElementById("live-label");

const chart = LightweightCharts.createChart(chartElement, {
  autoSize: true,
  layout: {
    background: { type: "solid", color: "#0b0e11" },
    textColor: "#b2b5be",
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  grid: {
    vertLines: { color: "#1e222d" },
    horzLines: { color: "#1e222d" },
  },
  crosshair: {
    mode: LightweightCharts.CrosshairMode.Normal,
    vertLine: { color: "#758696", width: 1, style: 3, labelBackgroundColor: "#2962ff" },
    horzLine: { color: "#758696", width: 1, style: 3, labelBackgroundColor: "#2962ff" },
  },
  rightPriceScale: {
    borderColor: "#2a2e39",
    scaleMargins: { top: 0.08, bottom: 0.25 },
  },
  timeScale: {
    borderColor: "#2a2e39",
    timeVisible: true,
    secondsVisible: false,
    rightOffset: 8,
    barSpacing: 7,
    minBarSpacing: 0.25,
    tickMarkFormatter: (time) => formatTime(time, currentTimeframe),
  },
  localization: {
    locale: "en-IN",
    timeFormatter: (time) => formatTimestamp(time),
    priceFormatter: (price) => Number(price).toFixed(2),
  },
  handleScroll: true,
  handleScale: true,
});

const candleSeries = chart.addCandlestickSeries({
  upColor: "#26a69a",
  downColor: "#ef5350",
  borderUpColor: "#26a69a",
  borderDownColor: "#ef5350",
  wickUpColor: "#26a69a",
  wickDownColor: "#ef5350",
  priceLineVisible: true,
  lastValueVisible: true,
});

const volumeSeries = chart.addHistogramSeries({
  priceFormat: { type: "volume" },
  priceScaleId: "volume",
  lastValueVisible: false,
  priceLineVisible: false,
});
chart.priceScale("volume").applyOptions({
  scaleMargins: { top: 0.82, bottom: 0 },
  borderVisible: false,
});

let sourceCandles = [];
let sourceVolumes = [];
let dailyCandles = [];
let sourceIndex = 0;
let replayTime = 0;
let currentTimeframe = "1m";
let currentCandles = [];
let currentVolumes = [];
let playing = false;
let playTimer = null;
let secondsPerCandle = 5;
let followHead = true;
let activeTool = "cursor";
let pendingPoint = null;
let pointerPreview = null;
let hoverPoint = null;
let hoverLocal = null;
let drawPress = null;
let measurement = null;
let shiftHeld = false;
let selectedDrawing = null;
let dragState = null;
let drawings = loadDrawings();
let lastUsedColor = DEFAULT_DRAWING_COLOR;
let lastUsedOpacity = 1;
let overlayWidth = 0;
let overlayHeight = 0;

const TWO_POINT_TOOLS = ["trend", "measure"];
const CLICK_TOOLS = ["horizontal", "ray", "long", "short", "text"];
const SNAP_TOOLS = ["trend"];
const TEXT_TOOLS = ["trend", "text", "horizontal"];
const TEXT_FONT = "12px Inter, -apple-system, 'Segoe UI', sans-serif";

// Intl formatters are rebuilt on every axis label and replay tick otherwise,
// which is the single biggest cost while panning.
const FMT_STAMP = new Intl.DateTimeFormat("en-IN", {
  timeZone: DISPLAY_TZ,
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const FMT_DAY = new Intl.DateTimeFormat("en-IN", {
  timeZone: DISPLAY_TZ,
  day: "2-digit",
  month: "short",
});
const FMT_CLOCK = new Intl.DateTimeFormat("en-IN", {
  timeZone: DISPLAY_TZ,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const FMT_PARTS = new Intl.DateTimeFormat("en-CA", {
  timeZone: DISPLAY_TZ,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function formatTimestamp(time) {
  return FMT_STAMP.format(new Date(Number(time) * 1000));
}

function formatTime(time, tf) {
  const formatter = ["1d", "1w"].includes(tf) ? FMT_DAY : FMT_CLOCK;
  return formatter.format(new Date(Number(time) * 1000));
}

const istPartsCache = new Map();

function istParts(time) {
  const key = Number(time);
  const cached = istPartsCache.get(key);
  if (cached) return cached;

  const parts = FMT_PARTS.formatToParts(new Date(key * 1000));
  const get = (type) => parts.find((part) => part.type === type)?.value || "";
  const value = {
    date: `${get("year")}-${get("month")}-${get("day")}`,
    time: `${get("hour")}:${get("minute")}`,
  };
  if (istPartsCache.size > 50000) istPartsCache.clear();
  istPartsCache.set(key, value);
  return value;
}

function parseIstInput(dateStr, timeStr) {
  if (!dateStr || !timeStr) return null;
  const clock = timeStr.length <= 5 ? `${timeStr}:00` : timeStr;
  const value = Date.parse(`${dateStr}T${clock}+05:30`);
  if (Number.isNaN(value)) return null;
  return Math.floor(value / 1000);
}

function bucketStart(utcSec, tf) {
  const local = utcSec + IST_OFFSET;
  if (tf === "1w") {
    const day = Math.floor(local / 86400);
    const weekday = (day + 4) % 7;
    return (day - weekday) * 86400 - IST_OFFSET;
  }
  const size = TF_SECONDS[tf];
  return Math.floor(local / size) * size - IST_OFFSET;
}

function lastIndexAtOrBefore(time, bars = sourceCandles) {
  let lo = 0;
  let hi = bars.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].time <= time) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

function firstIndexAtOrAfter(time) {
  let lo = 0;
  let hi = sourceCandles.length - 1;
  let ans = sourceCandles.length;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sourceCandles[mid].time >= time) {
      ans = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return ans;
}

function usesDailyArchive(tf = currentTimeframe) {
  return dailyCandles.length > 0 && (tf === "1d" || tf === "1w");
}

function applyTimeframe(timeframe) {
  currentTimeframe = timeframe;
  document.querySelectorAll(".tf").forEach((button) => {
    button.classList.toggle("active", button.dataset.tf === timeframe);
  });
}

function snapToReplayWindow() {
  if (!currentCandles.length) return;
  followHead = false;
  const bars = usesDailyArchive() || sourceIndex < 0
    ? 220
    : currentTimeframe === "1m"
      ? 160
      : 200;
  focusReplayWindow(bars);
}

function dataStartTime() {
  const times = [];
  if (dailyCandles.length) times.push(dailyCandles[0].time);
  if (sourceCandles.length) times.push(sourceCandles[0].time);
  return times.length ? Math.min(...times) : 0;
}

function dataEndTime() {
  const times = [];
  if (dailyCandles.length) times.push(dailyCandles[dailyCandles.length - 1].time);
  if (sourceCandles.length) times.push(sourceCandles[sourceCandles.length - 1].time);
  return times.length ? Math.max(...times) : 0;
}

function visibleSource() {
  if (sourceIndex < 0) return [];
  return sourceCandles.slice(0, sourceIndex + 1);
}

function visibleDaily() {
  const index = lastIndexAtOrBefore(replayTime, dailyCandles);
  if (index < 0) return [];
  return dailyCandles.slice(0, index + 1);
}

function seriesFromBars(bars) {
  const candles = bars.map((bar) => ({
    time: bar.time,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    volume: bar.volume ?? 0,
  }));
  const volumes = candles.map((bar) => ({
    time: bar.time,
    value: bar.volume,
    color: bar.close >= bar.open ? UP_VOLUME : DOWN_VOLUME,
  }));
  return { candles, volumes };
}

function resampleBars(bars, tf) {
  if (!bars.length) return { candles: [], volumes: [] };
  const buckets = new Map();
  for (const bar of bars) {
    const key = bucketStart(bar.time, tf);
    const volume = bar.volume ?? 0;
    const existing = buckets.get(key);
    if (!existing) {
      buckets.set(key, {
        time: key,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume,
      });
    } else {
      existing.high = Math.max(existing.high, bar.high);
      existing.low = Math.min(existing.low, bar.low);
      existing.close = bar.close;
      existing.volume += volume;
    }
  }
  return seriesFromBars([...buckets.values()]);
}

function buildVisibleSeries() {
  if (usesDailyArchive()) {
    const bars = visibleDaily();
    if (currentTimeframe === "1d") return seriesFromBars(bars);
    return resampleBars(bars, "1w");
  }
  const source = visibleSource();
  if (source.length) return resample(source);
  // Keep the clicked timeframe selected; draw daily so the pane is not empty
  // when replay sits before the 1-minute archive.
  const daily = visibleDaily();
  if (!daily.length) return { candles: [], volumes: [] };
  return seriesFromBars(daily);
}

function isReplayEnded() {
  if (usesDailyArchive()) {
    if (!dailyCandles.length) return true;
    return replayTime >= dailyCandles[dailyCandles.length - 1].time;
  }
  return !sourceCandles.length || sourceIndex >= sourceCandles.length - 1;
}

function resample(bars) {
  if (!bars.length) return { candles: [], volumes: [] };
  if (currentTimeframe === "1m") {
    // Copied rather than shared, so the chart library never mutates the source
    // bars and both render paths produce identically shaped candles.
    const candles = bars.map((bar, i) => ({
      time: bar.time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume: sourceVolumes[i]?.value ?? 0,
    }));
    const volumes = candles.map((bar) => ({
      time: bar.time,
      value: bar.volume,
      color: bar.close >= bar.open ? UP_VOLUME : DOWN_VOLUME,
    }));
    return { candles, volumes };
  }

  const buckets = new Map();
  for (let i = 0; i < bars.length; i += 1) {
    const bar = bars[i];
    const key = bucketStart(bar.time, currentTimeframe);
    const volume = sourceVolumes[i]?.value ?? 0;
    const existing = buckets.get(key);
    if (!existing) {
      buckets.set(key, {
        time: key,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume,
      });
    } else {
      existing.high = Math.max(existing.high, bar.high);
      existing.low = Math.min(existing.low, bar.low);
      existing.close = bar.close;
      existing.volume += volume;
    }
  }

  const candles = [...buckets.values()];
  const volumes = candles.map((bar) => ({
    time: bar.time,
    value: Math.max(0, bar.volume || 0),
    color: bar.close >= bar.open ? UP_VOLUME : DOWN_VOLUME,
  }));
  return { candles, volumes };
}

// Folds one 1-minute bar into the tail of the rendered series so a replay tick
// costs one series.update() instead of a full setData() of every bar.
function appendSourceBar(index) {
  const bar = sourceCandles[index];
  const volume = sourceVolumes[index]?.value ?? 0;
  const key = currentTimeframe === "1m"
    ? bar.time
    : bucketStart(bar.time, currentTimeframe);
  const lastCandle = currentCandles[currentCandles.length - 1];

  if (lastCandle && lastCandle.time === key) {
    const lastVolume = currentVolumes[currentVolumes.length - 1];
    lastCandle.high = Math.max(lastCandle.high, bar.high);
    lastCandle.low = Math.min(lastCandle.low, bar.low);
    lastCandle.close = bar.close;
    lastCandle.volume = (lastCandle.volume ?? 0) + volume;
    lastVolume.value += volume;
    lastVolume.color = lastCandle.close >= lastCandle.open ? UP_VOLUME : DOWN_VOLUME;
  } else {
    currentCandles.push({
      time: key,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume,
    });
    currentVolumes.push({
      time: key,
      value: volume,
      color: bar.close >= bar.open ? UP_VOLUME : DOWN_VOLUME,
    });
  }

  candleSeries.update(currentCandles[currentCandles.length - 1]);
  volumeSeries.update(currentVolumes[currentVolumes.length - 1]);
}

function loadDrawings() {
  try {
    const shared = JSON.parse(localStorage.getItem(DRAWING_STORAGE_KEY) || "null");
    if (!Array.isArray(shared)) return [];
    // Measurements used to be stored; they are transient now.
    return shared.filter((drawing) => drawing && drawing.type !== "measure");
  } catch {
    return [];
  }
}

function saveDrawings() {
  localStorage.setItem(DRAWING_STORAGE_KEY, JSON.stringify(drawings));
}

function loadReplayCursor() {
  try {
    const saved = JSON.parse(localStorage.getItem(REPLAY_STORAGE_KEY) || "null");
    if (!saved || typeof saved.time !== "number" || !Number.isFinite(saved.time)) return null;
    return saved;
  } catch {
    return null;
  }
}

function saveReplayCursor() {
  if (!replayTime) return;
  localStorage.setItem(
    REPLAY_STORAGE_KEY,
    JSON.stringify({ time: replayTime, timeframe: currentTimeframe })
  );
}

function drawingColor(drawing) {
  return drawing?.color || DEFAULT_DRAWING_COLOR;
}

function normalizeHex(color) {
  if (!color) return DEFAULT_DRAWING_COLOR;
  const value = String(color).trim().toLowerCase();
  if (/^#[0-9a-f]{6}$/.test(value)) return value;
  if (/^#[0-9a-f]{3}$/.test(value)) {
    return `#${value[1]}${value[1]}${value[2]}${value[2]}${value[3]}${value[3]}`;
  }
  return DEFAULT_DRAWING_COLOR;
}

function drawingOpacity(drawing) {
  const value = Number(drawing?.opacity);
  return Number.isFinite(value) ? Math.min(1, Math.max(0.2, value)) : 1;
}

function closeColorPopover() {
  colorPopover.hidden = true;
}

function openColorPopover() {
  colorPopover.hidden = false;
  const toolbar = selectionToolbar.getBoundingClientRect();
  const shellRect = shell.getBoundingClientRect();
  const popoverHeight = colorPopover.offsetHeight || 210;
  if (toolbar.bottom + 8 + popoverHeight > shellRect.bottom) {
    colorPopover.style.top = "auto";
    colorPopover.style.bottom = "calc(100% + 6px)";
  } else {
    colorPopover.style.top = "calc(100% + 6px)";
    colorPopover.style.bottom = "auto";
  }
}

function setSelectedColor(color, opacity) {
  const drawing = selectedDrawing != null ? drawings[selectedDrawing] : null;
  if (!drawing) return;
  const hex = normalizeHex(color);
  drawing.color = hex;
  lastUsedColor = hex;
  if (opacity != null) drawing.opacity = opacity;
  lastUsedOpacity = drawingOpacity(drawing);
  syncColorUi(drawing);
  saveDrawings();
  drawOverlay();
}

function syncColorUi(drawing) {
  const hex = normalizeHex(drawingColor(drawing));
  const opacity = Math.round(drawingOpacity(drawing) * 100);
  colorChip.style.background = hex;
  colorChip.style.opacity = String(opacity / 100);
  colorInput.value = hex;
  colorOpacity.value = String(opacity);
  colorOpacityValue.textContent = `${opacity}%`;
  colorGrid.querySelectorAll(".color-swatch").forEach((button) => {
    button.classList.toggle("active", button.dataset.color === hex);
  });
}

TV_COLOR_GRID.flat().forEach((hex) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "color-swatch";
  button.dataset.color = hex;
  button.style.background = hex;
  button.title = hex;
  button.addEventListener("pointerdown", (event) => event.stopPropagation());
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    setSelectedColor(hex);
  });
  colorGrid.appendChild(button);
});

function syncDateInputs() {
  if (!replayTime) return;
  const parts = istParts(replayTime);
  const dateInput = document.getElementById("jump-date");
  const timeInput = document.getElementById("jump-time");
  const editing =
    document.activeElement === dateInput || document.activeElement === timeInput;
  // Leave both fields alone while either is being edited so changing the time
  // cannot reset the date, and the other way around.
  if (!editing) {
    dateInput.value = parts.date;
    timeInput.value = parts.time;
  }
  document.getElementById("replay-clock").textContent =
    `REPLAY  ${parts.date}  ${parts.time} IST`;
}

function setLiveState(state, label) {
  liveDot.className = `dot ${state}`;
  liveLabel.textContent = label;
}

function updateOhlc(bar) {
  if (!bar) return;
  const color = bar.close >= bar.open ? "#26a69a" : "#ef5350";
  for (const [id, key] of [["o", "open"], ["h", "high"], ["l", "low"], ["c", "close"]]) {
    const element = document.getElementById(id);
    element.textContent = Number(bar[key]).toFixed(2);
    element.style.color = color;
  }
}

function updateStatus() {
  if (usesDailyArchive()) {
    const archiveLabel = currentTimeframe === "1w" ? "weekly from daily" : "daily archive";
    document.getElementById("bar-count").textContent =
      `${currentCandles.length.toLocaleString("en-IN")} ${currentTimeframe} · ` +
      `${dailyCandles.length.toLocaleString("en-IN")} ${archiveLabel}`;
    const start = visibleDaily()[0] || dailyCandles[0];
    const archiveStart = dailyCandles[0];
    document.getElementById("range").textContent = start
      ? `${formatTimestamp(archiveStart.time)} → ${formatTimestamp(replayTime)}`
      : "—";
    return;
  }

  const total = sourceCandles.length;
  if (sourceIndex < 0) {
    document.getElementById("bar-count").textContent =
      `No ${currentTimeframe} here · daily shown · 1m starts ${istParts(sourceCandles[0]?.time || replayTime).date}`;
    document.getElementById("range").textContent = formatTimestamp(replayTime);
    return;
  }
  document.getElementById("bar-count").textContent =
    `${currentCandles.length.toLocaleString("en-IN")} ${currentTimeframe} · ` +
    `${(sourceIndex + 1).toLocaleString("en-IN")}/${total.toLocaleString("en-IN")} 1m`;
  if (sourceCandles.length) {
    document.getElementById("range").textContent =
      `${formatTimestamp(sourceCandles[0].time)} → ${formatTimestamp(sourceCandles[sourceIndex].time)}`;
  }
}

function keepHeadInView() {
  if (!followHead || !currentCandles.length) return;
  const range = chart.timeScale().getVisibleLogicalRange();
  const last = currentCandles.length - 1;
  if (!range) {
    chart.timeScale().setVisibleLogicalRange({
      from: Math.max(0, last - 180),
      to: last + 5,
    });
    return;
  }
  if (last > range.to - 4 || last < range.from) {
    const width = Math.max(40, range.to - range.from);
    chart.timeScale().setVisibleLogicalRange({
      from: last - width + 5,
      to: last + 5,
    });
  }
}

function finishRender() {
  updateOhlc(currentCandles[currentCandles.length - 1]);
  updateStatus();
  syncDateInputs();
  keepHeadInView();
  drawOverlay();
}

function renderChart({ preserveRange = false } = {}) {
  const previousRange = preserveRange ? chart.timeScale().getVisibleRange() : null;
  const { candles, volumes } = buildVisibleSeries();
  currentCandles = candles;
  currentVolumes = volumes;
  candleSeries.setData(candles);
  volumeSeries.setData(volumes);
  updateOhlc(candles[candles.length - 1]);
  updateStatus();
  syncDateInputs();
  if (preserveRange && previousRange) {
    try {
      chart.timeScale().setVisibleRange(previousRange);
    } catch {
      keepHeadInView();
    }
  } else {
    keepHeadInView();
  }
  drawOverlay();
}

// Stepping further than this is cheaper as one setData() than as N updates.
const MAX_INCREMENTAL_STEP = 400;

function setReplayTime(time, { preserveRange = false } = {}) {
  const start = dataStartTime();
  const end = dataEndTime();
  if (!end) return;

  const previousIndex = sourceIndex;
  replayTime = Math.max(start, Math.min(time, end));
  sourceIndex = lastIndexAtOrBefore(replayTime, sourceCandles);

  const step = sourceIndex - previousIndex;
  const canAppend =
    !preserveRange &&
    !usesDailyArchive() &&
    previousIndex >= 0 &&
    sourceIndex >= 0 &&
    step > 0 &&
    step <= MAX_INCREMENTAL_STEP &&
    currentCandles.length > 0;

  if (canAppend) {
    for (let i = previousIndex + 1; i <= sourceIndex; i += 1) {
      appendSourceBar(i);
    }
    finishRender();
  } else {
    renderChart({ preserveRange });
  }
  saveReplayCursor();

  if (isReplayEnded()) {
    pauseReplay("End of data");
    setLiveState("ended", "End of data");
  } else if (!playing) {
    setLiveState("paused", "Paused");
  }
}

function setReplayIndex(index, { preserveRange = false } = {}) {
  if (!sourceCandles.length) return;
  const clamped = Math.max(0, Math.min(index, sourceCandles.length - 1));
  setReplayTime(sourceCandles[clamped].time, { preserveRange });
}

function jumpToTime(time) {
  setReplayTime(time);
}

function nextArchiveBar() {
  if (!dailyCandles.length) return;
  const idx = lastIndexAtOrBefore(replayTime, dailyCandles);
  if (idx < 0) {
    setReplayTime(dailyCandles[0].time);
    return;
  }
  if (currentTimeframe === "1d") {
    if (idx + 1 >= dailyCandles.length) {
      setReplayTime(dailyCandles[dailyCandles.length - 1].time);
      return;
    }
    setReplayTime(dailyCandles[idx + 1].time);
    return;
  }

  const currentWeek = bucketStart(replayTime, "1w");
  let index = idx + 1;
  while (
    index < dailyCandles.length &&
    bucketStart(dailyCandles[index].time, "1w") === currentWeek
  ) {
    index += 1;
  }
  if (index >= dailyCandles.length) {
    setReplayTime(dailyCandles[dailyCandles.length - 1].time);
    return;
  }
  const nextWeek = bucketStart(dailyCandles[index].time, "1w");
  let end = index;
  while (
    end + 1 < dailyCandles.length &&
    bucketStart(dailyCandles[end + 1].time, "1w") === nextWeek
  ) {
    end += 1;
  }
  setReplayTime(dailyCandles[end].time);
}

function nextCandle() {
  if (usesDailyArchive()) {
    if (isReplayEnded()) {
      pauseReplay("End of data");
      setLiveState("ended", "End of data");
      return;
    }
    nextArchiveBar();
    return;
  }
  if (!sourceCandles.length || isReplayEnded()) {
    pauseReplay("End of data");
    setLiveState("ended", "End of data");
    return;
  }
  if (sourceIndex < 0) {
    setReplayTime(sourceCandles[0].time);
    return;
  }
  if (currentTimeframe === "1m") {
    setReplayIndex(sourceIndex + 1);
    return;
  }
  const currentBucket = bucketStart(sourceCandles[sourceIndex].time, currentTimeframe);
  let index = sourceIndex + 1;
  while (
    index < sourceCandles.length &&
    bucketStart(sourceCandles[index].time, currentTimeframe) === currentBucket
  ) {
    index += 1;
  }
  if (index >= sourceCandles.length) {
    setReplayIndex(sourceCandles.length - 1);
    return;
  }
  const nextBucket = bucketStart(sourceCandles[index].time, currentTimeframe);
  let end = index;
  while (
    end + 1 < sourceCandles.length &&
    bucketStart(sourceCandles[end + 1].time, currentTimeframe) === nextBucket
  ) {
    end += 1;
  }
  setReplayIndex(end);
}

function jumpBySeconds(seconds) {
  if (usesDailyArchive()) {
    if (seconds >= 86400) {
      const days = Math.max(1, Math.round(seconds / 86400));
      const idx = lastIndexAtOrBefore(replayTime, dailyCandles);
      const next = Math.min(dailyCandles.length - 1, Math.max(0, idx) + days);
      setReplayTime(dailyCandles[next].time);
      return;
    }
    nextArchiveBar();
    return;
  }
  if (!sourceCandles.length) return;
  if (sourceIndex < 0) {
    setReplayTime(sourceCandles[0].time);
    return;
  }
  const target = sourceCandles[sourceIndex].time + seconds;
  let index = lastIndexAtOrBefore(target);
  if (index <= sourceIndex) {
    index = firstIndexAtOrAfter(target);
  }
  if (index >= sourceCandles.length) {
    setReplayIndex(sourceCandles.length - 1);
    return;
  }
  setReplayIndex(index);
}

function jumpNextDay() {
  if (usesDailyArchive()) {
    nextArchiveBar();
    return;
  }
  if (!sourceCandles.length) return;
  if (sourceIndex < 0) {
    setReplayTime(sourceCandles[0].time);
    return;
  }
  const current = istParts(sourceCandles[sourceIndex].time).date;
  for (let i = sourceIndex + 1; i < sourceCandles.length; i += 1) {
    if (istParts(sourceCandles[i].time).date !== current) {
      setReplayIndex(i);
      return;
    }
  }
  setReplayIndex(sourceCandles.length - 1);
}

function pauseReplay(label = "Paused") {
  playing = false;
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  playButton.textContent = "Play";
  playButton.classList.remove("playing");
  if (!isReplayEnded()) setLiveState("paused", label);
}

function startReplay() {
  if (isReplayEnded()) {
    setLiveState("ended", "End of data");
    return;
  }
  playing = true;
  playButton.textContent = "Pause";
  playButton.classList.add("playing");
  setLiveState("", "Playing");
  if (playTimer) clearInterval(playTimer);
  playTimer = setInterval(nextCandle, Math.max(100, secondsPerCandle * 1000));
}

function togglePlay() {
  if (playing) pauseReplay();
  else startReplay();
}

function restartPlayTimer() {
  if (!playing) return;
  startReplay();
}

async function loadSource() {
  loading.style.display = "block";
  errorBox.style.display = "none";
  try {
    const response = await fetch("/api/source");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load gold data");
    sourceCandles = payload.candles;
    sourceVolumes = payload.volumes;
    const dailyRaw = payload.dailyCandles || [];
    const dailyVols = payload.dailyVolumes || [];
    dailyCandles = dailyRaw.map((bar, i) => ({
      ...bar,
      volume: dailyVols[i]?.value ?? 0,
    }));
    const saved = loadReplayCursor();
    if (saved?.timeframe && TF_SECONDS[saved.timeframe]) {
      applyTimeframe(saved.timeframe);
    }
    const start = istParts(dataStartTime());
    const end = istParts(dataEndTime());
    document.getElementById("jump-date").min = start.date;
    document.getElementById("jump-date").max = end.date;
    document.getElementById("status-right").textContent =
      `${payload.timezone} · ${payload.files.join(", ")}`;
    if (saved?.time) {
      setReplayTime(saved.time);
      snapToReplayWindow();
    } else {
      const firstDate = istParts(sourceCandles[0].time).date;
      sourceIndex = 0;
      for (let i = 0; i < sourceCandles.length; i += 1) {
        if (istParts(sourceCandles[i].time).date !== firstDate) break;
        sourceIndex = i;
      }
      replayTime = sourceCandles[sourceIndex].time;
      saveReplayCursor();
      renderChart();
      chart.timeScale().setVisibleLogicalRange({
        from: -5,
        to: Math.min(180, sourceCandles.length) + 5,
      });
    }
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
}

function focusReplayWindow(bars = 220) {
  if (!currentCandles.length) return;
  const last = currentCandles.length - 1;
  chart.timeScale().setVisibleLogicalRange({
    from: Math.max(-8, last - bars),
    to: last + 10,
  });
}

function switchTimeframe(timeframe) {
  if (timeframe === currentTimeframe) return;
  applyTimeframe(timeframe);
  saveReplayCursor();
  renderChart();
  snapToReplayWindow();
}

// The overlay must stop where the axes begin, otherwise it swallows clicks meant
// for the right price scale and the bottom time scale.
function plotAreaSize() {
  const rect = shell.getBoundingClientRect();
  let axisWidth = 0;
  let axisHeight = 0;
  try {
    axisWidth = chart.priceScale("right").width() || 0;
    axisHeight = chart.timeScale().height() || 0;
  } catch {
    // Scales are not measurable before the first paint; full size is fine then.
  }
  return {
    width: Math.max(1, rect.width - axisWidth),
    height: Math.max(1, rect.height - axisHeight),
  };
}

function applyOverlayGeometry() {
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = plotAreaSize();
  const bitmapWidth = Math.max(1, Math.floor(width * ratio));
  const bitmapHeight = Math.max(1, Math.floor(height * ratio));

  // Reallocating the bitmap clears it, so only touch it on a real size change.
  if (canvas.width !== bitmapWidth || canvas.height !== bitmapHeight) {
    canvas.width = bitmapWidth;
    canvas.height = bitmapHeight;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  if (overlayWidth !== width || overlayHeight !== height) {
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    overlayWidth = width;
    overlayHeight = height;
  }
}

function resizeDrawingCanvas() {
  applyOverlayGeometry();
  drawOverlay();
}

function isInsidePlotArea(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  return (
    clientX >= rect.left &&
    clientX <= rect.left + overlayWidth &&
    clientY >= rect.top &&
    clientY <= rect.top + overlayHeight
  );
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return pointFromScreen(x, y);
}

function pointFromScreen(x, y) {
  const time = interpolatedTimeAtCoordinate(x);
  const price = candleSeries.coordinateToPrice(y);
  if (time == null || price == null) return null;
  return { time: Number(time), price: Number(price) };
}

// Bar-snapped time (coordinateToTime) is too coarse for drawing geometry, so
// interpolate between neighbouring bars to keep angles and Shift snaps exact.
function interpolatedTimeAtCoordinate(x) {
  return timeAtLogical(chart.timeScale().coordinateToLogical(x));
}

function timeAtLogical(logical) {
  if (logical == null || !currentCandles.length) return null;
  const step = TF_SECONDS[currentTimeframe];
  const last = currentCandles.length - 1;
  if (logical <= 0) {
    return currentCandles[0].time + Math.round(logical * step);
  }
  if (logical >= last) {
    return currentCandles[last].time + Math.round((logical - last) * step);
  }
  const index = Math.floor(logical);
  const fraction = logical - index;
  const from = currentCandles[index].time;
  const to = currentCandles[index + 1].time;
  return Math.round(from + (to - from) * fraction);
}

function logicalIndexForTime(time) {
  if (!currentCandles.length) return null;
  let low = 0;
  let high = currentCandles.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (currentCandles[middle].time < time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) {
    const span = Math.max(1, TF_SECONDS[currentTimeframe]);
    return (time - currentCandles[0].time) / span;
  }
  if (low >= currentCandles.length) {
    const last = currentCandles.length - 1;
    const span = Math.max(1, TF_SECONDS[currentTimeframe]);
    return last + (time - currentCandles[last].time) / span;
  }
  const before = currentCandles[low - 1];
  const after = currentCandles[low];
  return (low - 1) + (time - before.time) / Math.max(1, after.time - before.time);
}

function constrainStraightPoint(anchor, moving, lockStraight) {
  if (!lockStraight || !anchor || !moving) return moving;
  const a = screenPoint(anchor);
  const b = screenPoint(moving);
  if (!a || !b) return moving;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return moving;
  const snapped = Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) * (Math.PI / 4);
  const dist = Math.hypot(dx, dy);
  const next = pointFromScreen(
    a.x + Math.cos(snapped) * dist,
    a.y + Math.sin(snapped) * dist
  );
  if (next) return next;
  const axis = Math.abs(dx) >= Math.abs(dy)
    ? { time: moving.time, price: anchor.price }
    : { time: anchor.time, price: moving.price };
  return axis;
}

function timeAtCoordinate(x) {
  const exact = chart.timeScale().coordinateToTime(x);
  if (exact != null) return Number(exact);
  const logical = chart.timeScale().coordinateToLogical(x);
  if (logical == null || !currentCandles.length) return null;
  const secondsPerBar = TF_SECONDS[currentTimeframe];
  if (logical < 0) {
    return currentCandles[0].time + Math.round(logical * secondsPerBar);
  }
  const lastIndex = currentCandles.length - 1;
  return currentCandles[lastIndex].time +
    Math.round((logical - lastIndex) * secondsPerBar);
}

function screenPoint(point) {
  const x = coordinateForTime(point.time);
  const y = candleSeries.priceToCoordinate(point.price);
  return x == null || y == null ? null : { x, y };
}

function coordinateForTime(time) {
  if (!currentCandles.length) {
    const exact = chart.timeScale().timeToCoordinate(time);
    return exact == null ? null : exact;
  }
  // Always map through logical indices so points left of the first bar stay
  // left of the plot, instead of LWC clamping them to the left edge.
  const logical = logicalIndexForTime(time);
  if (logical == null) return null;
  return chart.timeScale().logicalToCoordinate(logical);
}

function drawOverlay() {
  // The price scale widens when labels get longer, so keep the overlay in step.
  const plot = plotAreaSize();
  if (
    Math.abs(plot.width - overlayWidth) > 0.5 ||
    Math.abs(plot.height - overlayHeight) > 0.5
  ) {
    applyOverlayGeometry();
  }
  ctx.clearRect(0, 0, overlayWidth, overlayHeight);
  drawings.forEach((drawing, index) => drawOne(drawing, false, index));
  if (measurement) drawOne(measurement, false, null);
  drawDraft();
  drawToolCrosshair();
  updateSelectionUi();
}

// Screen point the floating delete button should sit next to.
function selectionAnchorScreen() {
  const drawing = drawings[selectedDrawing];
  if (!drawing) return null;

  if (drawing.type === "horizontal") {
    const y = candleSeries.priceToCoordinate(drawing.price);
    return y == null ? null : { x: overlayWidth / 2, y };
  }
  if (drawing.type === "ray") {
    const start = screenPoint({ time: drawing.time, price: drawing.price });
    const y = candleSeries.priceToCoordinate(drawing.price);
    if (y == null) return null;
    return { x: start ? Math.max(0, start.x) : 0, y };
  }
  if (["long", "short"].includes(drawing.type)) {
    const points = positionScreenPoints(drawing);
    if (!points) return null;
    return {
      x: (points.left + points.right) / 2,
      y: Math.min(points.entry.y, points.target.y, points.stop.y),
    };
  }
  if (drawing.type === "text") {
    const box = textNoteBox(drawing);
    return box ? { x: box.left + box.width / 2, y: box.top } : null;
  }
  const a = screenPoint(drawing.a);
  const b = screenPoint(drawing.b);
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: Math.min(a.y, b.y) };
}

function updateSelectionUi() {
  const selected = selectedDrawing != null ? drawings[selectedDrawing] : null;
  const hasSelection = selected != null;
  deleteOneButton.disabled = !hasSelection;
  editTextButton.hidden = !hasSelection || !TEXT_TOOLS.includes(selected.type);

  if (!hasSelection || editingIndex != null) {
    closeColorPopover();
    selectionToolbar.hidden = true;
    return;
  }

  const anchor = selectionAnchorScreen();
  if (!anchor) {
    selectionToolbar.hidden = true;
    return;
  }

  selectionToolbar.hidden = false;
  syncColorUi(selected);
  const width = selectionToolbar.offsetWidth || 34;
  const height = selectionToolbar.offsetHeight || 30;
  const left = Math.max(4, Math.min(anchor.x - width / 2, overlayWidth - width - 4));
  const top = Math.max(4, Math.min(anchor.y - height - 10, overlayHeight - height - 4));
  selectionToolbar.style.left = `${left}px`;
  selectionToolbar.style.top = `${top}px`;
}

let editingIndex = null;
let editOriginalText = "";

function textAnchorScreen(drawing) {
  if (drawing.type === "text") {
    const box = textNoteBox(drawing);
    return box ? { x: box.left, y: box.top } : null;
  }
  if (drawing.type === "horizontal") {
    const y = candleSeries.priceToCoordinate(drawing.price);
    return y == null ? null : { x: 76, y: y - 28 };
  }
  const a = screenPoint(drawing.a);
  const b = screenPoint(drawing.b);
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2 - 40, y: Math.min(a.y, b.y) - 30 };
}

function startTextEdit(index) {
  const drawing = drawings[index];
  if (!drawing || !TEXT_TOOLS.includes(drawing.type)) return;

  editingIndex = index;
  editOriginalText = drawing.text || "";
  selectedDrawing = index;
  const anchor = textAnchorScreen(drawing) || { x: 20, y: 20 };
  const left = Math.max(4, Math.min(anchor.x, overlayWidth - 234));
  const top = Math.max(4, Math.min(anchor.y, overlayHeight - 30));
  textEditor.style.left = `${left}px`;
  textEditor.style.top = `${top}px`;
  textEditor.value = drawing.text || "";
  textEditor.hidden = false;
  selectionToolbar.hidden = true;
  textEditor.focus();
  textEditor.select();
}

function applyEditedText() {
  if (editingIndex == null) return;
  const drawing = drawings[editingIndex];
  if (drawing) drawing.text = textEditor.value;
  drawOverlay();
}

function finishTextEdit({ cancel = false } = {}) {
  if (editingIndex == null) return;
  const index = editingIndex;
  editingIndex = null;
  textEditor.hidden = true;

  const drawing = drawings[index];
  if (drawing) {
    // Typing edits the drawing live, so cancelling has to put the old text back.
    drawing.text = cancel ? editOriginalText : textEditor.value.trim();
    // An empty text note would be invisible and unselectable, so drop it.
    if (drawing.type === "text" && !drawing.text) {
      drawings.splice(index, 1);
      selectedDrawing = null;
    }
  }
  saveDrawings();
  drawOverlay();
}

function deleteSelectedDrawing() {
  if (selectedDrawing == null || !drawings[selectedDrawing]) return;
  if (editingIndex != null) {
    editingIndex = null;
    textEditor.hidden = true;
  }
  drawings.splice(selectedDrawing, 1);
  selectedDrawing = null;
  saveDrawings();
  drawOverlay();
}

function deleteAllDrawings() {
  drawings = [];
  pendingPoint = null;
  pointerPreview = null;
  selectedDrawing = null;
  saveDrawings();
  drawOverlay();
}

function draftAnchor() {
  if (pendingPoint) return pendingPoint;
  if (drawPress && drawPress.moved) return drawPress.point;
  return null;
}

function drawDraft() {
  if (activeTool === "cursor") return;

  if (TWO_POINT_TOOLS.includes(activeTool)) {
    const anchor = draftAnchor();
    if (!anchor || !pointerPreview) return;
    drawOne({ type: activeTool, a: anchor, b: pointerPreview, color: lastUsedColor }, true);
    return;
  }

  if (!hoverPoint) return;
  if (activeTool === "horizontal") {
    drawOne({ type: "horizontal", price: hoverPoint.price, color: lastUsedColor }, true);
  } else if (activeTool === "ray") {
    drawOne({ type: "ray", time: hoverPoint.time, price: hoverPoint.price, color: lastUsedColor }, true);
  }
}

function drawToolCrosshair() {
  if (activeTool === "cursor" || !hoverLocal) return;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#758696";
  ctx.beginPath();
  ctx.moveTo(0, hoverLocal.y);
  ctx.lineTo(overlayWidth, hoverLocal.y);
  ctx.moveTo(hoverLocal.x, 0);
  ctx.lineTo(hoverLocal.x, overlayHeight);
  ctx.stroke();
  ctx.restore();
}

function drawOne(drawing, preview = false, index = null) {
  const selected = !preview && index === selectedDrawing;
  const color = drawingColor(drawing);
  ctx.save();
  ctx.globalAlpha = (preview ? 0.65 : 1) * drawingOpacity(drawing);
  ctx.setLineDash(preview ? [5, 5] : []);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = selected ? 2 : 1.5;
  ctx.beginPath();

  if (drawing.type === "horizontal") {
    const y = candleSeries.priceToCoordinate(drawing.price);
    if (y == null) return ctx.restore();
    ctx.moveTo(0, y);
    ctx.lineTo(overlayWidth, y);
    ctx.stroke();
    drawPriceTag(drawing.price, y, selected, color);
    if (drawing.text) drawHorizontalLabel(drawing, y, color);
  } else if (drawing.type === "ray") {
    drawRay(drawing, selected, color);
  } else if (drawing.type === "measure") {
    drawMeasure(drawing, selected);
  } else if (["long", "short"].includes(drawing.type)) {
    drawPosition(drawing, selected, color);
  } else if (drawing.type === "text") {
    drawTextNote(drawing, selected, color);
  } else {
    const a = screenPoint(drawing.a);
    const b = screenPoint(drawing.b);
    if (!a || !b) return ctx.restore();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    if (drawing.text) drawTrendLabel(drawing, a, b, selected, color);
    if (selected) {
      drawHandle(a);
      drawHandle(b);
    }
  }
  ctx.restore();
}

// Label sits at the middle of the trendline, nudged clear of the line itself.
function drawTrendLabel(drawing, a, b, selected, color) {
  ctx.save();
  ctx.font = TEXT_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  const midX = (a.x + b.x) / 2;
  const midY = (a.y + b.y) / 2;
  const width = ctx.measureText(drawing.text).width;
  ctx.fillStyle = "#0b0e11cc";
  ctx.fillRect(midX - width / 2 - 4, midY - 22, width + 8, 17);
  ctx.fillStyle = color;
  ctx.fillText(drawing.text, midX, midY - 7);
  ctx.restore();
}

function textNoteBox(drawing) {
  const anchor = screenPoint({ time: drawing.time, price: drawing.price });
  if (!anchor) return null;
  ctx.save();
  ctx.font = TEXT_FONT;
  const width = Math.max(12, ctx.measureText(drawing.text || "").width);
  ctx.restore();
  return {
    anchor,
    left: anchor.x,
    top: anchor.y - 14,
    width,
    height: 18,
  };
}

function drawTextNote(drawing, selected, color) {
  const box = textNoteBox(drawing);
  if (!box) return;
  ctx.save();
  ctx.font = TEXT_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = color;
  ctx.fillText(drawing.text || "", box.left, box.anchor.y);
  if (selected) {
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(box.left - 4, box.top - 3, box.width + 8, box.height + 4);
  }
  ctx.restore();
}

function drawPriceTag(price, y, selected, color) {
  ctx.fillStyle = "#0b0e11";
  ctx.fillRect(7, y - 15, 64, 15);
  ctx.fillStyle = color;
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText(Number(price).toFixed(2), 11, y - 4);
}

function drawHorizontalLabel(drawing, y, color) {
  ctx.save();
  ctx.font = TEXT_FONT;
  const x = 76;
  const width = ctx.measureText(drawing.text).width;
  ctx.fillStyle = "#0b0e11cc";
  ctx.fillRect(x - 4, y - 16, width + 8, 16);
  ctx.fillStyle = color;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(drawing.text, x, y - 4);
  ctx.restore();
}

function drawRay(drawing, selected, color) {
  const start = screenPoint({ time: drawing.time, price: drawing.price });
  const y = candleSeries.priceToCoordinate(drawing.price);
  if (y == null) return;
  const width = overlayWidth;
  const x = start ? start.x : 0;
  if (start && start.x > width) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(Math.max(0, x), y);
  ctx.lineTo(width, y);
  ctx.stroke();
  if (start && start.x >= 0) {
    ctx.beginPath();
    ctx.arc(start.x, y, selected ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  drawPriceTag(drawing.price, y, selected, color);
}

function drawHandle(point, color = "#64b5f6") {
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.strokeStyle = "#0b0e11";
  ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawSquareHandle(point, color = "#64b5f6") {
  ctx.fillStyle = color;
  ctx.strokeStyle = "#0b0e11";
  ctx.fillRect(point.x - 5, point.y - 5, 10, 10);
  ctx.strokeRect(point.x - 5, point.y - 5, 10, 10);
}

function drawTextBox(lines, x, y, background) {
  ctx.font = "11px Inter, sans-serif";
  const padding = 6;
  const lineHeight = 15;
  const width = Math.max(...lines.map((line) => ctx.measureText(line).width)) + padding * 2;
  const height = lines.length * lineHeight + padding * 2 - 2;
  const left = Math.max(4, Math.min(x, overlayWidth - width - 4));
  const top = Math.max(4, Math.min(y, overlayHeight - height - 4));
  ctx.fillStyle = background;
  ctx.fillRect(left, top, width, height);
  ctx.strokeStyle = "#d1d4dc66";
  ctx.strokeRect(left, top, width, height);
  ctx.fillStyle = "#ffffff";
  lines.forEach((line, index) => {
    ctx.fillText(line, left + padding, top + padding + 10 + index * lineHeight);
  });
}

function formatDuration(seconds) {
  const totalMinutes = Math.max(0, Math.round(seconds / 60));
  if (totalMinutes < 60) return `${totalMinutes}m`;
  const hours = Math.floor(totalMinutes / 60);
  if (hours < 24) return `${hours}h ${totalMinutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

function drawMeasure(drawing, selected) {
  const a = screenPoint(drawing.a);
  const b = screenPoint(drawing.b);
  if (!a || !b) return;
  const rising = drawing.b.price >= drawing.a.price;
  const color = rising ? "#26a69a" : "#ef5350";
  const left = Math.min(a.x, b.x);
  const top = Math.min(a.y, b.y);
  const width = Math.abs(b.x - a.x);
  const height = Math.abs(b.y - a.y);
  const change = drawing.b.price - drawing.a.price;
  const percent = drawing.a.price ? (change / drawing.a.price) * 100 : 0;
  const seconds = Math.abs(drawing.b.time - drawing.a.time);

  ctx.fillStyle = `${color}22`;
  ctx.fillRect(left, top, width, height);
  ctx.strokeStyle = selected ? "#64b5f6" : color;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(left, top, width, height);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.setLineDash([]);
  drawTextBox(
    [
      `${change >= 0 ? "+" : ""}${change.toFixed(2)} (${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%)`,
      `${formatDuration(seconds)}`,
    ],
    left + width / 2 - 55,
    top + height / 2 - 22,
    rising ? "#10675d" : "#8f2d2b"
  );
  if (selected) {
    drawHandle(a);
    drawHandle(b);
  }
}

function drawPosition(drawing, selected, color) {
  const points = positionScreenPoints(drawing);
  if (!points) return;
  const { entry, target, stop, widthHandle, left, right } = points;
  const width = Math.max(2, right - left);
  const profitPts = Math.abs(drawing.target.price - drawing.entry.price);
  const lossPts = Math.abs(drawing.entry.price - drawing.stop.price);
  const profitPct = drawing.entry.price ? (profitPts / drawing.entry.price) * 100 : 0;
  const lossPct = drawing.entry.price ? (lossPts / drawing.entry.price) * 100 : 0;
  const ratio = lossPts ? profitPts / lossPts : 0;
  const targetTop = Math.min(entry.y, target.y);
  const stopTop = Math.min(entry.y, stop.y);

  ctx.fillStyle = "#26a69a33";
  ctx.fillRect(left, targetTop, width, Math.abs(target.y - entry.y));
  ctx.fillStyle = "#ef53503d";
  ctx.fillRect(left, stopTop, width, Math.abs(stop.y - entry.y));

  ctx.lineWidth = selected ? 2 : 1.25;
  ctx.strokeStyle = "#26a69a";
  ctx.strokeRect(left, targetTop, width, Math.abs(target.y - entry.y));
  ctx.strokeStyle = "#ef5350";
  ctx.strokeRect(left, stopTop, width, Math.abs(stop.y - entry.y));
  ctx.strokeStyle = color;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(left, entry.y);
  ctx.lineTo(right, entry.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // TradingView: points and percent inside each zone, both sides of entry.
  const labelX = left + 6;
  drawTextBox(
    [`Close  +${profitPts.toFixed(2)}  (+${profitPct.toFixed(2)}%)`],
    labelX,
    (entry.y + target.y) / 2 - 12,
    "#10675d"
  );
  drawTextBox(
    [`Stop  −${lossPts.toFixed(2)}  (−${lossPct.toFixed(2)}%)`],
    labelX,
    (entry.y + stop.y) / 2 - 12,
    "#8f2d2b"
  );
  drawTextBox(
    [`${drawing.type === "long" ? "LONG" : "SHORT"}  R:R ${ratio.toFixed(2)}`],
    right - 118,
    entry.y - 20,
    "#1e222d"
  );

  if (selected) {
    drawHandle(entry);
    drawSquareHandle(target, "#26a69a");
    drawSquareHandle(stop, "#ef5350");
    drawSquareHandle(widthHandle);
  }
}

function positionScreenPoints(drawing) {
  const entry = screenPoint(drawing.entry);
  const endTime = drawing.endTime ?? drawing.target.time;
  const rightX = coordinateForTime(endTime);
  const targetY = candleSeries.priceToCoordinate(drawing.target.price);
  const stopY = candleSeries.priceToCoordinate(drawing.stop.price);
  if (!entry || rightX == null || targetY == null || stopY == null) return null;

  const left = Math.min(entry.x, rightX);
  const right = Math.max(entry.x, rightX);
  const middle = left + (right - left) / 2;
  return {
    entry,
    target: { x: middle, y: targetY },
    stop: { x: middle, y: stopY },
    widthHandle: { x: rightX, y: entry.y },
    left,
    right,
  };
}

let chartCursor = "crosshair";

function setChartCursor(cursor) {
  if (chartCursor === cursor) return;
  chartCursor = cursor;
  chartElement.style.cursor = cursor;
}

function distanceToSegment(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return Math.hypot(point.x - a.x, point.y - a.y);
  const t = Math.max(0, Math.min(1,
    ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy)
  ));
  return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy));
}

function hitDrawing(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const point = { x: clientX - rect.left, y: clientY - rect.top };
  let nearest = null;

  drawings.forEach((drawing, index) => {
    if (drawing.type === "horizontal") {
      const y = candleSeries.priceToCoordinate(drawing.price);
      const distance = y == null ? Infinity : Math.abs(point.y - y);
      if (distance <= 8 && (!nearest || distance < nearest.distance)) {
        nearest = { index, part: "line", distance };
      }
      return;
    }

    if (drawing.type === "ray") {
      const start = screenPoint({ time: drawing.time, price: drawing.price });
      const y = candleSeries.priceToCoordinate(drawing.price);
      if (y == null) return;
      const startX = start ? start.x : -Infinity;
      if (point.x + 6 < startX) return;
      const distance = Math.abs(point.y - y);
      if (start && Math.hypot(point.x - start.x, point.y - y) <= 11) {
        nearest = { index, part: "start", distance: 0 };
        return;
      }
      if (distance <= 8 && (!nearest || distance < nearest.distance)) {
        nearest = { index, part: "line", distance };
      }
      return;
    }

    if (drawing.type === "text") {
      const box = textNoteBox(drawing);
      if (!box) return;
      const inside =
        point.x >= box.left - 6 &&
        point.x <= box.left + box.width + 6 &&
        point.y >= box.top - 5 &&
        point.y <= box.top + box.height + 5;
      if (inside && (!nearest || nearest.distance > 4)) {
        nearest = { index, part: "body", distance: 4 };
      }
      return;
    }

    if (["long", "short"].includes(drawing.type)) {
      const points = positionScreenPoints(drawing);
      if (!points) return;
      const { entry, target, stop, widthHandle, left, right } = points;
      const handles = [
        { part: "entry", point: entry },
        { part: "target", point: target },
        { part: "stop", point: stop },
        { part: "width", point: widthHandle },
      ];
      for (const handle of handles) {
        const distance = Math.hypot(point.x - handle.point.x, point.y - handle.point.y);
        if (distance <= 11 && (!nearest || distance < nearest.distance)) {
          nearest = { index, part: handle.part, distance };
        }
      }
      const top = Math.min(entry.y, target.y, stop.y);
      const bottom = Math.max(entry.y, target.y, stop.y);
      if (
        !nearest &&
        point.x >= left &&
        point.x <= right &&
        point.y >= top &&
        point.y <= bottom
      ) {
        nearest = { index, part: "body", distance: 6 };
      }
      return;
    }

    const a = screenPoint(drawing.a);
    const b = screenPoint(drawing.b);
    if (!a || !b) return;
    const distanceA = Math.hypot(point.x - a.x, point.y - a.y);
    const distanceB = Math.hypot(point.x - b.x, point.y - b.y);
    const lineDistance = distanceToSegment(point, a, b);
    let part = null;
    let distance = Infinity;
    const onHandleA = distanceA <= 5;
    const onHandleB = distanceB <= 5;
    const onLine = lineDistance <= 10;
    const alreadySelected = index === selectedDrawing;
    // Unselected lines always move as a whole. Stretch only from a selected corner.
    if (alreadySelected && onHandleA && (!onHandleB || distanceA <= distanceB)) {
      part = "a";
      distance = distanceA;
    } else if (alreadySelected && onHandleB) {
      part = "b";
      distance = distanceB;
    } else if (onLine || onHandleA || onHandleB) {
      part = "line";
      distance = onLine ? lineDistance : Math.min(distanceA, distanceB);
    }
    if (part && (!nearest || distance < nearest.distance)) {
      nearest = { index, part, distance };
    }
  });
  return nearest;
}

function eraseAt(clientX, clientY) {
  const hit = hitDrawing(clientX, clientY);
  if (hit) {
    drawings.splice(hit.index, 1);
    selectedDrawing = null;
    saveDrawings();
    drawOverlay();
  }
}

function commitDrawing(drawing) {
  // The ruler is a throwaway readout in TradingView: never stored, never
  // selectable, and it vanishes on the next click.
  if (drawing.type === "measure") {
    pendingPoint = null;
    pointerPreview = null;
    setTool("cursor");
    measurement = drawing;
    drawOverlay();
    return;
  }

  if (!drawing.color) drawing.color = lastUsedColor;
  if (drawing.opacity == null) drawing.opacity = lastUsedOpacity;
  drawings.push(drawing);
  selectedDrawing = drawings.length - 1;
  pendingPoint = null;
  pointerPreview = null;
  saveDrawings();
  setTool("cursor");
}

function clearMeasurement() {
  if (!measurement) return false;
  measurement = null;
  return true;
}

function snapEnd(anchor, point, shiftKey) {
  return SNAP_TOOLS.includes(activeTool)
    ? constrainStraightPoint(anchor, point, shiftKey)
    : point;
}

function buildPosition(entry, localX, localY, endPoint) {
  const riskProbe = candleSeries.coordinateToPrice(localY + 55);
  const risk = Math.max(
    Math.abs((riskProbe ?? entry.price * 0.995) - entry.price),
    entry.price * 0.002
  );
  const endTime = endPoint ? endPoint.time : timeAtCoordinate(localX + 125);
  const isLong = activeTool === "long";
  return {
    type: activeTool,
    entry: { ...entry },
    target: {
      time: endTime,
      price: entry.price + (isLong ? risk * 2 : -risk * 2),
    },
    stop: {
      time: endTime,
      price: entry.price + (isLong ? -risk : risk),
    },
    endTime,
  };
}

function updateHover(event) {
  const rect = canvas.getBoundingClientRect();
  hoverLocal = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  hoverPoint = pointFromScreen(hoverLocal.x, hoverLocal.y);
}

function refreshDraft(shiftKey) {
  shiftHeld = shiftKey;
  if (!TWO_POINT_TOOLS.includes(activeTool)) return;
  const anchor = draftAnchor();
  if (!anchor || !hoverPoint) return;
  pointerPreview = snapEnd(anchor, hoverPoint, shiftKey);
}

function toolHelpText(stage) {
  const help = {
    cursor: "Pan/zoom · drag drawings or handles to move",
    trend: stage === "second"
      ? "Trendline: release or click the second point · hold Shift for straight"
      : "Trendline: drag from one point to the other · hold Shift for straight",
    horizontal: "Horizontal line: click a price · double-click or T to add text",
    ray: "Horizontal ray: click a start point; it extends right",
    text: "Text: click the chart, then type · Enter saves, Esc cancels",
    measure: stage === "second"
      ? "Measure: release or click the second point"
      : "Measure: drag across the range you want to measure",
    long: "Long: click an entry, or drag to set the box width",
    short: "Short: click an entry, or drag to set the box width",
    erase: "Erase: click near a drawing",
  };
  document.getElementById("drawing-help").textContent = help[activeTool];
}

canvas.addEventListener("pointerdown", (event) => {
  if (activeTool === "cursor" || event.button !== 0) return;
  clearMeasurement();
  updateHover(event);
  if (!hoverPoint) return;
  drawPress = {
    pointerId: event.pointerId,
    screen: { x: event.clientX, y: event.clientY },
    local: { ...hoverLocal },
    point: hoverPoint,
    moved: false,
  };
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch {
    // Pointer capture is best-effort; drawing still works without it.
  }
});

canvas.addEventListener("pointermove", (event) => {
  updateHover(event);
  if (activeTool === "cursor") return;
  if (drawPress && event.pointerId === drawPress.pointerId && !drawPress.moved) {
    const distance = Math.hypot(
      event.clientX - drawPress.screen.x,
      event.clientY - drawPress.screen.y
    );
    if (distance > 3) {
      drawPress.moved = true;
      toolHelpText("second");
    }
  }
  refreshDraft(event.shiftKey);
  drawOverlay();
});

canvas.addEventListener("pointerup", (event) => {
  const press = drawPress;
  drawPress = null;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  if (activeTool === "cursor" || event.button !== 0) return;

  updateHover(event);
  const point = hoverPoint || press?.point;
  if (!point) return;

  if (activeTool === "erase") {
    eraseAt(event.clientX, event.clientY);
    return;
  }

  if (activeTool === "horizontal") {
    commitDrawing({ type: "horizontal", price: point.price, text: "" });
    return;
  }

  if (activeTool === "ray") {
    commitDrawing({ type: "ray", time: point.time, price: point.price });
    return;
  }

  if (activeTool === "text") {
    commitDrawing({ type: "text", time: point.time, price: point.price, text: "" });
    startTextEdit(selectedDrawing);
    return;
  }

  if (["long", "short"].includes(activeTool)) {
    const entry = press ? press.point : point;
    const local = press ? press.local : hoverLocal;
    const dragged = press && press.moved ? point : null;
    commitDrawing(buildPosition(entry, local.x, local.y, dragged));
    return;
  }

  if (TWO_POINT_TOOLS.includes(activeTool)) {
    if (press && press.moved) {
      commitDrawing({
        type: activeTool,
        a: press.point,
        b: snapEnd(press.point, point, event.shiftKey),
        ...(activeTool === "trend" ? { text: "" } : {}),
      });
      return;
    }
    if (!pendingPoint) {
      pendingPoint = point;
      toolHelpText("second");
      drawOverlay();
      return;
    }
    commitDrawing({
      type: activeTool,
      a: pendingPoint,
      b: snapEnd(pendingPoint, point, event.shiftKey),
      ...(activeTool === "trend" ? { text: "" } : {}),
    });
  }
});

canvas.addEventListener("pointerleave", () => {
  hoverPoint = null;
  hoverLocal = null;
  drawOverlay();
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
  cancelDraft();
});

function cancelDraft() {
  pendingPoint = null;
  pointerPreview = null;
  drawPress = null;
  clearMeasurement();
  toolHelpText();
  drawOverlay();
}

shell.addEventListener("pointerdown", (event) => {
  if (selectionToolbar.contains(event.target) || event.target === textEditor) return;
  if (editingIndex != null) finishTextEdit();
  if (activeTool !== "cursor" || event.button !== 0) return;
  // Let the price/time scales handle their own drags.
  if (!isInsidePlotArea(event.clientX, event.clientY)) return;
  const hadMeasurement = clearMeasurement();
  const hit = hitDrawing(event.clientX, event.clientY);
  if (!hit) {
    selectedDrawing = null;
    drawOverlay();
    return;
  }
  if (hadMeasurement) drawOverlay();

  const start = pointFromEvent(event);
  if (!start) return;
  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation();
  const alreadySelected = selectedDrawing === hit.index;
  selectedDrawing = hit.index;
  const rect = canvas.getBoundingClientRect();
  const drawing = drawings[hit.index];
  setChartInteraction(false);
  const movingWholeLine = drawing.a && drawing.b && (!alreadySelected || hit.part === "line" || hit.part === "body");
  dragState = {
    pointerId: event.pointerId,
    index: hit.index,
    part: movingWholeLine ? "line" : hit.part,
    start,
    startLocal: { x: event.clientX - rect.left, y: event.clientY - rect.top },
    startLogical: chart.timeScale().coordinateToLogical(event.clientX - rect.left),
    original: JSON.parse(JSON.stringify(drawing)),
  };
  shell.setPointerCapture(event.pointerId);
  setChartCursor("grabbing");
  drawOverlay();
}, true);

shell.addEventListener("pointermove", (event) => {
  if (activeTool !== "cursor") return;
  if (!dragState) {
    if (!isInsidePlotArea(event.clientX, event.clientY)) {
      setChartCursor("default");
      return;
    }
    const hit = hitDrawing(event.clientX, event.clientY);
    setChartCursor(hit
      ? (hit.part === "line" || hit.part === "body" ? "move" : "grab")
      : "crosshair");
    return;
  }

  if (event.pointerId !== dragState.pointerId) return;
  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation();
  const rect = canvas.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;
  const current = pointFromScreen(localX, localY);
  const drawing = drawings[dragState.index];
  const original = dragState.original;

  if (drawing.type === "horizontal") {
    if (!current) return;
    drawing.price = current.price;
  } else if (drawing.type === "text") {
    if (!current) return;
    drawing.time = original.time + (current.time - dragState.start.time);
    drawing.price = original.price + (current.price - dragState.start.price);
  } else if (drawing.type === "ray") {
    if (!current) return;
    if (dragState.part === "start") {
      drawing.time = current.time;
      drawing.price = current.price;
    } else {
      drawing.time = original.time + (current.time - dragState.start.time);
      drawing.price = current.price;
    }
  } else if (["long", "short"].includes(drawing.type)) {
    if (!current) return;
    if (dragState.part === "entry") {
      drawing.entry = current;
    } else if (dragState.part === "target") {
      drawing.target = {
        time: drawing.endTime ?? drawing.target.time,
        price: drawing.type === "long"
          ? Math.max(current.price, drawing.entry.price + 0.0001)
          : Math.min(current.price, drawing.entry.price - 0.0001),
      };
    } else if (dragState.part === "stop") {
      drawing.stop = {
        time: drawing.endTime ?? drawing.stop.time,
        price: drawing.type === "long"
          ? Math.min(current.price, drawing.entry.price - 0.0001)
          : Math.max(current.price, drawing.entry.price + 0.0001),
      };
    } else if (dragState.part === "width") {
      drawing.endTime = current.time;
      drawing.target.time = current.time;
      drawing.stop.time = current.time;
    } else {
      const timeDelta = current.time - dragState.start.time;
      const priceDelta = current.price - dragState.start.price;
      for (const key of ["entry", "target", "stop"]) {
        drawing[key] = {
          time: original[key].time + timeDelta,
          price: original[key].price + priceDelta,
        };
      }
      drawing.endTime = (original.endTime ?? original.target.time) + timeDelta;
    }
  } else if (dragState.part === "line" || (drawing.a && drawing.b && dragState.part !== "a" && dragState.part !== "b")) {
    const nowLogical = chart.timeScale().coordinateToLogical(localX);
    const dLogical = (nowLogical ?? 0) - (dragState.startLogical ?? 0);
    const startPrice = candleSeries.coordinateToPrice(dragState.startLocal.y);
    const nowPrice = candleSeries.coordinateToPrice(localY);
    const dPrice = startPrice == null || nowPrice == null ? 0 : nowPrice - startPrice;
    shiftTrendline(drawing, original, dLogical, dPrice);
  } else if (dragState.part === "a" || dragState.part === "b") {
    if (!current) return;
    const anchor = dragState.part === "a" ? original.b : original.a;
    drawing[dragState.part] = drawing.type === "trend"
      ? constrainStraightPoint(anchor, current, event.shiftKey)
      : current;
  }
  drawOverlay();
}, true);

function setChartInteraction(enabled) {
  chart.applyOptions({
    handleScroll: enabled,
    handleScale: enabled,
  });
}

function shiftTrendline(drawing, original, dLogical, dPrice) {
  const shiftEnd = (point) => {
    const logical = logicalIndexForTime(point.time);
    if (logical == null) {
      return {
        time: point.time,
        price: point.price + dPrice,
      };
    }
    const time = timeAtLogical(logical + dLogical);
    return {
      time: time == null ? point.time : time,
      price: point.price + dPrice,
    };
  };
  drawing.a = shiftEnd(original.a);
  drawing.b = shiftEnd(original.b);
}

function finishDrag(event) {
  if (!dragState || event.pointerId !== dragState.pointerId) return;
  event.preventDefault();
  event.stopPropagation();
  saveDrawings();
  if (shell.hasPointerCapture(event.pointerId)) {
    shell.releasePointerCapture(event.pointerId);
  }
  dragState = null;
  setChartInteraction(true);
  setChartCursor("crosshair");
  drawOverlay();
}

shell.addEventListener("pointerup", finishDrag, true);
shell.addEventListener("pointercancel", finishDrag, true);

function setTool(tool) {
  activeTool = tool;
  pendingPoint = null;
  pointerPreview = null;
  drawPress = null;
  if (tool === "cursor") {
    hoverPoint = null;
    hoverLocal = null;
  }
  canvas.style.pointerEvents = tool === "cursor" ? "none" : "auto";
  document.querySelectorAll(".tool").forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === tool);
  });
  toolHelpText();
  drawOverlay();
}

function goToEnteredTime() {
  const dateStr = document.getElementById("jump-date").value;
  let timeStr = document.getElementById("jump-time").value;
  if (!dateStr) return;
  if (usesDailyArchive() && !timeStr) timeStr = "23:59";

  const time = parseIstInput(dateStr, timeStr || "23:59");
  if (time == null) return;
  pauseReplay();
  followHead = false;
  jumpToTime(time);
  snapToReplayWindow();
}

document.querySelectorAll(".tf").forEach((button) => {
  button.addEventListener("click", () => switchTimeframe(button.dataset.tf));
});
document.querySelectorAll(".tool").forEach((button) => {
  button.addEventListener("click", () => setTool(button.dataset.tool));
});
document.getElementById("clear").addEventListener("click", deleteAllDrawings);
deleteOneButton.addEventListener("click", deleteSelectedDrawing);
document.getElementById("delete-selected").addEventListener("click", (event) => {
  event.stopPropagation();
  deleteSelectedDrawing();
});
editTextButton.addEventListener("click", (event) => {
  event.stopPropagation();
  startTextEdit(selectedDrawing);
});

colorToggle.addEventListener("pointerdown", (event) => event.stopPropagation());
colorToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  if (colorPopover.hidden) openColorPopover();
  else closeColorPopover();
});
colorPopover.addEventListener("pointerdown", (event) => event.stopPropagation());
colorInput.addEventListener("input", (event) => setSelectedColor(event.target.value));
colorInput.addEventListener("change", (event) => setSelectedColor(event.target.value));
colorOpacity.addEventListener("input", (event) => {
  setSelectedColor(colorInput.value, Number(event.target.value) / 100);
});
document.addEventListener("pointerdown", (event) => {
  if (!colorPopover.hidden && !selectionToolbar.contains(event.target)) {
    closeColorPopover();
  }
});

textEditor.addEventListener("input", applyEditedText);
textEditor.addEventListener("keydown", (event) => {
  event.stopPropagation();
  if (event.key === "Enter") finishTextEdit();
  if (event.key === "Escape") finishTextEdit({ cancel: true });
});
textEditor.addEventListener("blur", () => finishTextEdit());
textEditor.addEventListener("pointerdown", (event) => event.stopPropagation());

// Double-click a trendline or note to edit its text, like TradingView's dialog.
shell.addEventListener("dblclick", (event) => {
  if (activeTool !== "cursor") return;
  if (!isInsidePlotArea(event.clientX, event.clientY)) return;
  const hit = hitDrawing(event.clientX, event.clientY);
  if (!hit || !TEXT_TOOLS.includes(drawings[hit.index].type)) return;
  event.preventDefault();
  startTextEdit(hit.index);
});
document.getElementById("fit").addEventListener("click", () => {
  followHead = false;
  chart.timeScale().fitContent();
  drawOverlay();
});
document.getElementById("follow").addEventListener("click", () => {
  followHead = true;
  keepHeadInView();
  drawOverlay();
});
document.getElementById("jump").addEventListener("click", goToEnteredTime);
document.getElementById("jump-date").addEventListener("change", goToEnteredTime);
document.getElementById("jump-time").addEventListener("change", goToEnteredTime);
document.getElementById("jump-date").addEventListener("keydown", (event) => {
  if (event.key === "Enter") goToEnteredTime();
});
document.getElementById("jump-time").addEventListener("keydown", (event) => {
  if (event.key === "Enter") goToEnteredTime();
});
playButton.addEventListener("click", togglePlay);
document.getElementById("next").addEventListener("click", () => {
  pauseReplay();
  nextCandle();
});
document.getElementById("next-hour").addEventListener("click", () => {
  pauseReplay();
  jumpBySeconds(3600);
});
document.getElementById("next-day").addEventListener("click", () => {
  pauseReplay();
  jumpNextDay();
});

function setSpeed(value) {
  const seconds = Math.max(0.5, Math.min(60, Number(value) || 5));
  secondsPerCandle = seconds;
  document.getElementById("speed").value = String(Math.min(15, seconds));
  document.getElementById("speed-num").value = String(seconds);
  restartPlayTimer();
}
document.getElementById("speed").addEventListener("input", (event) => {
  setSpeed(event.target.value);
});
document.getElementById("speed-num").addEventListener("change", (event) => {
  setSpeed(event.target.value);
});

chart.subscribeCrosshairMove((param) => {
  const bar = param.seriesData.get(candleSeries);
  updateOhlc(bar || currentCandles[currentCandles.length - 1]);
});
// Painted synchronously so drawings move in the same frame as the candles
// instead of trailing them by one.
chart.timeScale().subscribeVisibleLogicalRangeChange(drawOverlay);
new ResizeObserver(resizeDrawingCanvas).observe(shell);

function refreshShiftConstraint(event) {
  refreshDraft(event.shiftKey);
  drawOverlay();
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Shift") refreshShiftConstraint(event);
  if (event.key === "Escape" && !colorPopover.hidden) {
    closeColorPopover();
    return;
  }
  const typing = ["INPUT", "TEXTAREA"].includes(event.target.tagName);
  if (event.code === "Space" && !typing) {
    event.preventDefault();
    togglePlay();
    return;
  }
  if (typing) return;
  if (event.key.toLowerCase() === "t") setTool("trend");
  if (event.key.toLowerCase() === "h") setTool("horizontal");
  if (event.key.toLowerCase() === "r") setTool("ray");
  if (event.key.toLowerCase() === "m") setTool("measure");
  if (event.key.toLowerCase() === "l") setTool("long");
  if (event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey) setTool("short");
  if (event.key === "ArrowRight") {
    pauseReplay();
    nextCandle();
  }
  if (event.key === "Escape") {
    if (draftAnchor()) cancelDraft();
    else setTool("cursor");
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    deleteSelectedDrawing();
  }
});
window.addEventListener("keyup", (event) => {
  if (event.key === "Shift") refreshShiftConstraint(event);
});

loadSource();
