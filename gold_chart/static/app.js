const IST_OFFSET = 19800;
const DISPLAY_TZ = "Asia/Kolkata";
const UP_VOLUME = "#26a69a80";
const DOWN_VOLUME = "#ef535080";
const DEFAULT_DRAWING_COLOR = "#f4c430";
const DRAWING_STORAGE_KEY = "gc-chart-drawings-shared";
const REPLAY_STORAGE_KEY = "gc-chart-replay";
const EVENTS_STORAGE_KEY = "gc-chart-events-on";
const EVENT_LABELS_KEY = "gc-chart-event-labels";
const PAPER_ARROWS_KEY = "gc-chart-paper-arrows";
const FOLLOW_STORAGE_KEY = "gc-chart-follow-head";
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
  "30m": 1800,
  "1h": 3600,
  "4h": 14400,
  "1d": 86400,
  "1w": 604800,
};

const STEP_CHOICES = [
  { id: "current", label: "Current candles" },
  { id: "1m", label: "1m" },
  { id: "5m", label: "5m" },
  { id: "15m", label: "15m" },
  { id: "30m", label: "30m" },
  { id: "1h", label: "1h" },
  { id: "4h", label: "4h" },
  { id: "1d", label: "1D" },
];

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

let followHead = true;
try {
  const savedFollow = localStorage.getItem(FOLLOW_STORAGE_KEY);
  if (savedFollow === "0") followHead = false;
  if (savedFollow === "1") followHead = true;
} catch {
  followHead = true;
}
let lockedPriceRange = null;
let pricePan = null;

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
  autoscaleInfoProvider: (original) => {
    if (!followHead && lockedPriceRange) {
      return {
        priceRange: {
          minValue: lockedPriceRange.minValue,
          maxValue: lockedPriceRange.maxValue,
        },
      };
    }
    const res = original();
    if (!res?.priceRange) return res;
    let min = res.priceRange.minValue;
    let max = res.priceRange.maxValue;
    paperPositions().forEach((pos) => {
      [pos.entryFill, pos.tp, pos.sl].forEach((value) => {
        const price = Number(value);
        if (Number.isFinite(price) && price > 0) {
          min = Math.min(min, price);
          max = Math.max(max, price);
        }
      });
    });
    const pad = Math.max(0.5, (max - min) * 0.04);
    return { ...res, priceRange: { minValue: min - pad, maxValue: max + pad } };
  },
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
let replayStep = "current";
let feedUnit = "XAU";
let feedDailyName = "Gold_Daily.csv";
let currentCandles = [];
let currentVolumes = [];
let playing = false;
let playTimer = null;
let secondsPerCandle = 5;
let followProgrammatic = 0;
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
let drawingUndo = [];
let drawingRedo = [];
const DRAWING_UNDO_LIMIT = 80;
let lastUsedColor = DEFAULT_DRAWING_COLOR;
let lastUsedOpacity = 1;
let overlayWidth = 0;
let overlayHeight = 0;
let chartEvents = [];
let chartEventLabels = [];
let eventsEnabled = false;
let eventLabelsOn = false;
let eventLabelCount = 1;
try {
  eventsEnabled = localStorage.getItem(EVENTS_STORAGE_KEY) === "1";
} catch {
  eventsEnabled = false;
}
try {
  const savedLabels = JSON.parse(localStorage.getItem(EVENT_LABELS_KEY) || "null");
  if (savedLabels && typeof savedLabels === "object") {
    eventLabelsOn = savedLabels.on === true;
    eventLabelCount = Math.max(1, Math.min(5, Number(savedLabels.count) || 1));
  }
} catch {
  eventLabelsOn = false;
  eventLabelCount = 1;
}
let eventsLoading = false;
let paperArrowsOn = true;
let paperArrowSize = 16;
try {
  const savedArrows = JSON.parse(localStorage.getItem(PAPER_ARROWS_KEY) || "null");
  if (savedArrows && typeof savedArrows === "object") {
    paperArrowsOn = savedArrows.on !== false;
    const size = Number(savedArrows.size);
    if (Number.isFinite(size)) paperArrowSize = Math.max(8, Math.min(32, size));
  }
} catch {
  paperArrowsOn = true;
}
let eventGuideLine = null;
const EVENT_TF_COLOR = {
  "2-Year": "#ab47bc",
  "1-Year": "#ff9800",
  Monthly: "#d1d4dc",
  Weekly: "#ef5350",
  Daily: "#2962ff",
  Hourly: "#26a69a",
  PrevDay: "#f4c430",
  Today: "#00bcd4",
};

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
  // XAUUSDT / CoinDCX day = UTC calendar day = 05:30 IST → next 05:29 IST.
  if (tf === "1d") {
    return Math.floor(utcSec / 86400) * 86400;
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

function chartTfSeconds(tf = currentTimeframe) {
  return TF_SECONDS[tf] || 60;
}

function effectiveStepTf() {
  if (replayStep !== "current" && TF_SECONDS[replayStep] && TF_SECONDS[replayStep] <= chartTfSeconds()) {
    return replayStep;
  }
  return currentTimeframe;
}

function inMinuteArchive() {
  return sourceCandles.length > 0 && replayTime >= sourceCandles[0].time;
}

function formingFromMinutes() {
  if (!inMinuteArchive()) return false;
  if (currentTimeframe !== "1d" && currentTimeframe !== "1w") return true;
  // 1D/1W "Current candles" keeps full Gold_Daily bars. A smaller step
  // builds that same chart candle from 1-minute data as Play/Next add time.
  return replayStep !== "current";
}

function usesDailyArchive() {
  if (!dailyCandles.length || (currentTimeframe !== "1d" && currentTimeframe !== "1w")) return false;
  return !formingFromMinutes();
}

function allowedStepIds(tf = currentTimeframe) {
  const maxSec = chartTfSeconds(tf);
  return STEP_CHOICES
    .filter((choice) => choice.id === "current" || TF_SECONDS[choice.id] <= maxSec)
    .map((choice) => choice.id);
}

function syncReplayStepSelect() {
  const select = document.getElementById("replay-step");
  if (!select) return;
  const allowed = allowedStepIds();
  if (replayStep !== "current" && !allowed.includes(replayStep)) {
    replayStep = "current";
  }
  select.innerHTML = "";
  for (const choice of STEP_CHOICES) {
    if (!allowed.includes(choice.id)) continue;
    const option = document.createElement("option");
    option.value = choice.id;
    option.textContent = choice.label;
    select.appendChild(option);
  }
  select.value = replayStep;
  updateNextButtonTitle();
}

function updateNextButtonTitle() {
  const next = document.getElementById("next");
  const play = document.getElementById("play");
  const step = effectiveStepTf();
  const stepLabel = replayStep === "current" ? `current ${currentTimeframe}` : step;
  if (next) {
    next.title = `Add the next ${stepLabel} of data. Current candles = one full ${currentTimeframe} bar. A smaller step builds that bar.`;
  }
  if (play) {
    play.title = `Play — each tick adds ${stepLabel} (Space)`;
  }
}

function setReplayStep(step, { render = true } = {}) {
  const allowed = allowedStepIds();
  replayStep = allowed.includes(step) ? step : "current";
  const select = document.getElementById("replay-step");
  if (select && select.value !== replayStep) select.value = replayStep;
  updateNextButtonTitle();
  saveReplayCursor();
  if (render && (currentCandles.length || dailyCandles.length)) {
    renderChart({ preserveRange: true });
  }
}

function applyTimeframe(timeframe) {
  currentTimeframe = timeframe;
  document.querySelectorAll(".tf").forEach((button) => {
    button.classList.toggle("active", button.dataset.tf === timeframe);
  });
  syncReplayStepSelect();
}

function snapToReplayWindow() {
  if (!currentCandles.length) return;
  const bars = usesDailyArchive() || sourceIndex < 0
    ? 220
    : currentTimeframe === "1m"
      ? 160
      : 200;
  followProgrammatic += 1;
  try {
    focusReplayWindow(bars);
  } finally {
    window.setTimeout(() => {
      followProgrammatic = Math.max(0, followProgrammatic - 1);
    }, 0);
  }
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

function buildHybridHigherTf() {
  const cut = bucketStart(sourceCandles[0].time, currentTimeframe);
  const older = dailyCandles.filter((bar) => bar.time < cut && bar.time <= replayTime);
  const fromDaily = currentTimeframe === "1w"
    ? resampleBars(older, "1w")
    : seriesFromBars(older);
  const fromMinute = resample(visibleSource());
  return {
    candles: fromDaily.candles.concat(fromMinute.candles),
    volumes: fromDaily.volumes.concat(fromMinute.volumes),
  };
}

function buildVisibleSeries() {
  if ((currentTimeframe === "1d" || currentTimeframe === "1w") && formingFromMinutes()) {
    return buildHybridHigherTf();
  }
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
  if (formingFromMinutes()) {
    return !sourceCandles.length || sourceIndex >= sourceCandles.length - 1;
  }
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

function cloneDrawings(list = drawings) {
  return JSON.parse(JSON.stringify(list));
}

function pushUndoSnapshot(previous) {
  const serialized = JSON.stringify(previous);
  const last = drawingUndo[drawingUndo.length - 1];
  if (last && JSON.stringify(last) === serialized) return;
  drawingRedo = [];
  drawingUndo.push(previous);
  if (drawingUndo.length > DRAWING_UNDO_LIMIT) drawingUndo.shift();
  syncUndoButtons();
}

function snapshotDrawings() {
  pushUndoSnapshot(cloneDrawings());
}

function applyDrawingSnapshot(list) {
  drawings = cloneDrawings(list);
  if (selectedDrawing == null || selectedDrawing >= drawings.length) {
    selectedDrawing = drawings.length ? drawings.length - 1 : null;
  }
  if (editingIndex != null) {
    editingIndex = null;
    if (textEditor) textEditor.hidden = true;
  }
  saveDrawings();
  drawOverlay();
  syncUndoButtons();
}

function undoDrawing() {
  if (!drawingUndo.length) return;
  drawingRedo.push(cloneDrawings());
  applyDrawingSnapshot(drawingUndo.pop());
}

function redoDrawing() {
  if (!drawingRedo.length) return;
  drawingUndo.push(cloneDrawings());
  applyDrawingSnapshot(drawingRedo.pop());
}

function syncUndoButtons() {
  const undoBtn = document.getElementById("undo-drawing");
  const redoBtn = document.getElementById("redo-drawing");
  if (undoBtn) undoBtn.disabled = !drawingUndo.length;
  if (redoBtn) redoBtn.disabled = !drawingRedo.length;
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
    JSON.stringify({ time: replayTime, timeframe: currentTimeframe, step: replayStep })
  );
}

function eventColor(evt) {
  return EVENT_TF_COLOR[evt.timeframe] || "#f4c430";
}

function navigationEvents() {
  const byTime = new Map();
  chartEvents.forEach((evt) => {
    const prev = byTime.get(evt.time);
    if (!prev || evt.status === "TOUCH") byTime.set(evt.time, evt);
  });
  return [...byTime.values()].sort((a, b) => a.time - b.time);
}

function eventAtOrBeforeReplay() {
  const list = navigationEvents();
  let found = null;
  for (const evt of list) {
    if (evt.time <= replayTime) found = evt;
    else break;
  }
  return found;
}

function saveEventLabelPrefs() {
  try {
    localStorage.setItem(EVENT_LABELS_KEY, JSON.stringify({
      on: eventLabelsOn,
      count: eventLabelCount,
    }));
  } catch {
    /* ignore */
  }
}

function syncEventLabelControls() {
  const toggle = document.getElementById("event-labels-toggle");
  const wrap = document.getElementById("event-label-count-wrap");
  const range = document.getElementById("event-label-count");
  const num = document.getElementById("event-label-count-num");
  if (toggle) {
    toggle.disabled = !eventsEnabled || eventsLoading;
    toggle.classList.toggle("active", eventsEnabled && eventLabelsOn && !eventsLoading);
  }
  if (wrap) wrap.hidden = !eventsEnabled || !eventLabelsOn;
  if (range) range.value = String(eventLabelCount);
  if (num && document.activeElement !== num) num.value = String(eventLabelCount);
}

function setEventLabelsEnabled(on) {
  eventLabelsOn = Boolean(on);
  saveEventLabelPrefs();
  syncEventLabelControls();
  drawOverlay();
}

function setEventLabelCount(value) {
  eventLabelCount = Math.max(1, Math.min(5, Math.round(Number(value) || 1)));
  saveEventLabelPrefs();
  syncEventLabelControls();
  drawOverlay();
}

function utcSessionDate(unix) {
  const date = new Date(Number(unix) * 1000);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function replaySessionRange() {
  const day = utcSessionDate(replayTime);
  let high = -Infinity;
  let low = Infinity;
  currentCandles.forEach((bar) => {
    if (utcSessionDate(bar.time) !== day) return;
    high = Math.max(high, bar.high);
    low = Math.min(low, bar.low);
  });
  return { high, low };
}

function hourlyLabelFloor(unix) {
  const sessionStart = Math.floor(Number(unix) / 86400) * 86400;
  return sessionStart - 86400;
}

function isEventLabelLive(lab, price, session, hourlyFrom) {
  const source = Number(lab.sourceTime);
  const ready = Number(lab.readyTime ?? lab.sourceTime);
  const cancel = lab.cancelTime == null ? null : Number(lab.cancelTime);
  if (!Number.isFinite(source) || source > replayTime) return false;
  if (Number.isFinite(ready) && ready > replayTime) return false;
  if (cancel != null && Number.isFinite(cancel) && cancel <= replayTime) return false;
  if (lab.hourly && source < hourlyFrom) return false;
  const level = Number(lab.price);
  if (!Number.isFinite(level) || level <= 0) return false;
  if (lab.type === "resistance") {
    if (!(level > price)) return false;
    if (Number.isFinite(session.high) && session.high >= level) return false;
  } else if (lab.type === "support") {
    if (!(level < price)) return false;
    if (Number.isFinite(session.low) && session.low <= level) return false;
  } else {
    return false;
  }
  return true;
}

function pickNearestEventLabels(labels, type, count) {
  const side = labels.filter((lab) => lab.type === type);
  side.sort((a, b) => (type === "resistance" ? a.price - b.price : b.price - a.price));
  const seen = new Set();
  const out = [];
  side.forEach((lab) => {
    const key = Number(lab.price).toFixed(2);
    if (seen.has(key) || out.length >= count) return;
    seen.add(key);
    out.push(lab);
  });
  return out;
}

function activeEventLabels() {
  if (!eventsEnabled || !eventLabelsOn || !chartEventLabels.length || !currentCandles.length) {
    return [];
  }
  const last = currentCandles[currentCandles.length - 1];
  const price = Number(last?.close);
  if (!Number.isFinite(price) || price <= 0) return [];
  const session = replaySessionRange();
  const hourlyFrom = hourlyLabelFloor(replayTime);
  const live = chartEventLabels.filter((lab) => isEventLabelLive(lab, price, session, hourlyFrom));
  return [
    ...pickNearestEventLabels(live, "support", eventLabelCount),
    ...pickNearestEventLabels(live, "resistance", eventLabelCount),
  ];
}

function drawEventLabelRays() {
  const labels = activeEventLabels();
  if (!labels.length) return;
  ctx.save();
  ctx.font = "11px Inter, sans-serif";
  ctx.lineWidth = 1;
  labels.forEach((lab) => {
    const y = candleSeries.priceToCoordinate(lab.price);
    if (y == null) return;
    const start = screenPoint({ time: lab.sourceTime, price: lab.price });
    const x = start ? start.x : 0;
    if (start && start.x > overlayWidth) return;
    const color = eventColor(lab);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.setLineDash(lab.type === "support" ? [6, 4] : []);
    ctx.beginPath();
    ctx.moveTo(Math.max(0, x), y);
    ctx.lineTo(overlayWidth, y);
    ctx.stroke();
    ctx.setLineDash([]);
    if (start && start.x >= 0 && start.x <= overlayWidth) {
      ctx.beginPath();
      ctx.arc(start.x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    const tag = lab.name || `${lab.timeframe} ${lab.type === "support" ? "S" : "R"}`;
    const textWidth = ctx.measureText(tag).width;
    const textX = Math.max(6, overlayWidth - textWidth - 8);
    const textY = lab.type === "support" ? y + 12 : y - 5;
    ctx.fillStyle = "#0b0e11cc";
    ctx.fillRect(textX - 3, textY - 10, textWidth + 6, 14);
    ctx.fillStyle = color;
    ctx.fillText(tag, textX, textY);
  });
  ctx.restore();
}

function syncEventScanUi() {
  const toggle = document.getElementById("events-toggle");
  const spinner = document.getElementById("events-spinner");
  const scanning = eventsEnabled && eventsLoading;
  if (toggle) {
    toggle.classList.toggle("active", eventsEnabled);
    toggle.classList.toggle("scanning", scanning);
    toggle.title = scanning
      ? "Scanning this feed in the background. Leave Events on until the spinner stops."
      : "Show EventFinder levels on the chart. Switching data turns Events off.";
  }
  if (spinner) spinner.hidden = !scanning;
}

function updateEventHud(evt = null) {
  const hud = document.getElementById("event-hud");
  const prevBtn = document.getElementById("event-prev");
  const nextBtn = document.getElementById("event-next");
  syncEventScanUi();
  syncEventLabelControls();
  if (!eventsEnabled) {
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    hud.textContent = "Events off";
    return;
  }
  if (eventsLoading) {
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    hud.textContent = "Scanning this feed… spinner means wait, do not toggle";
    return;
  }
  const list = navigationEvents();
  let previous = null;
  let next = null;
  for (const item of list) {
    if (item.time < replayTime) previous = item;
    if (item.time > replayTime && !next) next = item;
  }
  prevBtn.disabled = !previous;
  nextBtn.disabled = !next;
  const shown = evt || eventAtOrBeforeReplay();
  if (!list.length) {
    hud.textContent = "No events in local market data";
    return;
  }
  if (!shown) {
    hud.textContent = `${list.length} events · next ${next ? next.level : "—"}`;
    return;
  }
  hud.textContent =
    `${shown.status} ${shown.level} @ ${Number(shown.price).toFixed(2)}` +
    ` · ${list.length} events`;
}

function clearEventGuide() {
  if (!eventGuideLine) return;
  try {
    candleSeries.removePriceLine(eventGuideLine);
  } catch {
    // already gone
  }
  eventGuideLine = null;
}

function showEventGuide(evt) {
  if (!eventsEnabled || !evt) {
    clearEventGuide();
    return;
  }
  const options = {
    price: evt.price,
    color: eventColor(evt),
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: `${evt.status} ${evt.timeframe}`,
  };
  if (eventGuideLine) {
    eventGuideLine.applyOptions(options);
    return;
  }
  eventGuideLine = candleSeries.createPriceLine(options);
}

function snapEventMarkerTime(unix) {
  if (!currentCandles.length) return null;
  const idx = lastIndexAtOrBefore(unix, currentCandles);
  if (idx < 0) return null;
  return currentCandles[idx].time;
}

function paperFillMarkers() {
  return [];
}

function savePaperArrowPrefs() {
  try {
    localStorage.setItem(PAPER_ARROWS_KEY, JSON.stringify({
      on: paperArrowsOn,
      size: paperArrowSize,
    }));
  } catch {
    /* ignore */
  }
}

function setPaperArrowsEnabled(on) {
  paperArrowsOn = Boolean(on);
  const toggle = document.getElementById("paper-arrows-toggle");
  const wrap = document.getElementById("paper-arrow-size-wrap");
  if (toggle) toggle.classList.toggle("active", paperArrowsOn);
  if (wrap) wrap.hidden = !paperArrowsOn;
  savePaperArrowPrefs();
  drawOverlay();
  syncEventMarkers();
}

function setPaperArrowSize(value) {
  paperArrowSize = Math.max(8, Math.min(32, Number(value) || 16));
  const range = document.getElementById("paper-arrow-size");
  const num = document.getElementById("paper-arrow-size-num");
  if (range) range.value = String(paperArrowSize);
  if (num && document.activeElement !== num) num.value = String(paperArrowSize);
  savePaperArrowPrefs();
  drawOverlay();
}

function drawTradeArrow(x, y, up, color, size, label) {
  if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) return;
  ctx.beginPath();
  if (up) {
    ctx.moveTo(x, y);
    ctx.lineTo(x - size * 0.42, y + size);
    ctx.lineTo(x + size * 0.42, y + size);
  } else {
    ctx.moveTo(x, y);
    ctx.lineTo(x - size * 0.42, y - size);
    ctx.lineTo(x + size * 0.42, y - size);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  if (label) {
    ctx.font = `700 ${Math.max(9, Math.round(size * 0.55))}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = up ? "top" : "bottom";
    ctx.fillStyle = color;
    ctx.fillText(label, x, up ? y + size + 1 : y - size - 1);
    ctx.textAlign = "left";
  }
}

function drawPaperArrows() {
  if (!paperArrowsOn || !currentCandles.length) return;
  const size = paperArrowSize;
  const marks = paperState?.markers || [];
  ctx.save();
  marks.forEach((mark) => {
    const snapped = snapEventMarkerTime(mark.time);
    if (snapped == null || snapped > replayTime) return;
    const idx = lastIndexAtOrBefore(snapped, currentCandles);
    if (idx < 0 || currentCandles[idx].time !== snapped) return;
    const bar = currentCandles[idx];
    const x = coordinateForTime(bar.time);
    if (x == null) return;
    const text = mark.text || (mark.buy ? "B" : "S");
    const up = !!mark.buy || text === "B";
    const color = text === "X" ? "#f0b90b" : (up ? "#00c076" : "#f6465d");
    const price = Number(mark.price);
    const yPrice = Number.isFinite(price) && price > 0
      ? price
      : (up ? bar.low : bar.high);
    let y = candleSeries.priceToCoordinate(yPrice);
    if (y == null) return;
    y = up ? y + 2 : y - 2;
    drawTradeArrow(x, y, up, color, size, text);
  });
  ctx.restore();
}

function syncEventMarkers() {
  const markers = [];
  if (eventsEnabled && chartEvents.length && currentCandles.length) {
    const byTime = new Map();
    chartEvents.forEach((evt) => {
      if (evt.time > replayTime) return;
      const time = snapEventMarkerTime(evt.time);
      if (time == null) return;
      const prev = byTime.get(time);
      if (!prev || evt.status === "TOUCH") byTime.set(time, evt);
    });
    byTime.forEach((evt, time) => {
      const support = evt.type === "support";
      markers.push({
        time,
        position: support ? "belowBar" : "aboveBar",
        color: evt.status === "TOUCH" ? eventColor(evt) : "#2962ff",
        shape: support ? "arrowUp" : "arrowDown",
        text: evt.status === "TOUCH" ? "T" : "N",
      });
    });
    showEventGuide(eventAtOrBeforeReplay());
  } else if (!eventsEnabled) {
    clearEventGuide();
  }
  paperFillMarkers().forEach((marker) => markers.push(marker));
  markers.sort((a, b) => a.time - b.time || String(a.text).localeCompare(String(b.text)));
  try {
    candleSeries.setMarkers(markers);
  } catch {
    // series not ready
  }
  updateEventHud();
}

async function loadChartEvents() {
  if (eventsLoading) return;
  eventsLoading = true;
  updateEventHud();
  try {
    const response = await fetch("/api/events");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load events");
    if (!eventsEnabled) {
      chartEvents = [];
      chartEventLabels = [];
      return;
    }
    chartEvents = Array.isArray(payload.events) ? payload.events : [];
    chartEventLabels = Array.isArray(payload.labels) ? payload.labels : [];
  } catch (error) {
    chartEvents = [];
    chartEventLabels = [];
    if (eventsEnabled) {
      document.getElementById("event-hud").textContent = error.message;
    }
  } finally {
    eventsLoading = false;
    syncEventMarkers();
    drawOverlay();
  }
}

function setEventsEnabled(on) {
  eventsEnabled = Boolean(on);
  try {
    localStorage.setItem(EVENTS_STORAGE_KEY, eventsEnabled ? "1" : "0");
  } catch {
    // ignore
  }
  if (!eventsEnabled) {
    clearEventGuide();
    syncEventMarkers();
    updateEventHud();
    drawOverlay();
    return;
  }
  if (!chartEvents.length && !chartEventLabels.length) loadChartEvents();
  else {
    syncEventMarkers();
    drawOverlay();
  }
}

function jumpToEvent(evt) {
  if (!evt) return;
  pauseReplay();
  setReplayTime(evt.time, { preserveRange: !followHead });
  if (followHead) {
    snapToReplayWindow();
  } else {
    revealReplayHead();
  }
  showEventGuide(evt);
  updateEventHud(evt);
}

function jumpPrevEvent() {
  if (!eventsEnabled) return;
  const list = navigationEvents();
  let previous = null;
  for (const item of list) {
    if (item.time < replayTime) previous = item;
  }
  jumpToEvent(previous);
}

function jumpNextEvent() {
  if (!eventsEnabled) return;
  const next = navigationEvents().find((item) => item.time > replayTime);
  jumpToEvent(next);
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

function colorWithAlpha(color, opacity) {
  const hex = normalizeHex(color);
  const red = parseInt(hex.slice(1, 3), 16);
  const green = parseInt(hex.slice(3, 5), 16);
  const blue = parseInt(hex.slice(5, 7), 16);
  const alpha = Number.isFinite(opacity) ? Math.min(1, Math.max(0, opacity)) : 1;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
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

function setSelectedColor(color, opacity, { history = true } = {}) {
  const drawing = selectedDrawing != null ? drawings[selectedDrawing] : null;
  if (!drawing) return;
  const hex = normalizeHex(color);
  const nextOpacity = opacity != null ? opacity : drawingOpacity(drawing);
  if (normalizeHex(drawingColor(drawing)) === hex && drawingOpacity(drawing) === nextOpacity) return;
  if (history) snapshotDrawings();
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

function previousBarClose(bar) {
  if (!bar || !currentCandles.length) return null;
  const idx = lastIndexAtOrBefore(bar.time, currentCandles);
  if (idx <= 0) return null;
  if (currentCandles[idx].time === bar.time) {
    return Number(currentCandles[idx - 1].close);
  }
  return Number(currentCandles[idx].close);
}

function formatSigned(value, digits) {
  const n = Number(value);
  const text = Math.abs(n).toFixed(digits);
  if (n > 0) return `+${text}`;
  if (n < 0) return `−${text}`;
  return text;
}

function updateOhlc(bar) {
  if (!bar) return;
  const candleColor = bar.close >= bar.open ? "#26a69a" : "#ef5350";
  for (const [id, key] of [["o", "open"], ["h", "high"], ["l", "low"], ["c", "close"]]) {
    const text = Number(bar[key]).toFixed(2);
    const header = document.getElementById(id);
    const legend = document.getElementById(`l${id}`);
    if (header) {
      header.textContent = text;
      header.style.color = candleColor;
    }
    if (legend) {
      legend.textContent = text;
      legend.style.color = candleColor;
    }
  }

  const tfEl = document.getElementById("legend-tf");
  if (tfEl) {
    const step = effectiveStepTf();
    tfEl.textContent = replayStep === "current" || step === currentTimeframe
      ? currentTimeframe.toUpperCase()
      : `${currentTimeframe.toUpperCase()} · step ${step}`;
  }

  const prevClose = previousBarClose(bar);
  const changeEl = document.getElementById("chg");
  const legendChg = document.getElementById("legend-chg");
  if (prevClose == null || !Number.isFinite(prevClose) || prevClose === 0) {
    if (changeEl) {
      changeEl.textContent = "—";
      changeEl.style.color = "";
    }
    if (legendChg) {
      legendChg.textContent = "—";
      legendChg.style.color = "";
    }
    return;
  }

  const delta = Number(bar.close) - prevClose;
  const pct = (delta / prevClose) * 100;
  const chgColor = delta > 0 ? "#26a69a" : delta < 0 ? "#ef5350" : "#787b86";
  const chgText = `${formatSigned(delta, 2)} (${formatSigned(pct, 2)}%)`;
  if (changeEl) {
    changeEl.textContent = chgText;
    changeEl.style.color = chgColor;
  }
  if (legendChg) {
    legendChg.textContent = chgText;
    legendChg.style.color = chgColor;
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

function visiblePriceRange() {
  let height = overlayHeight;
  if (!height) {
    try {
      height = chart.paneSize?.()?.height || 0;
    } catch {
      height = 0;
    }
  }
  if (!height) return null;
  const top = candleSeries.coordinateToPrice(0);
  const bottom = candleSeries.coordinateToPrice(height);
  if (top == null || bottom == null) return null;
  const minValue = Math.min(top, bottom);
  const maxValue = Math.max(top, bottom);
  if (!(maxValue > minValue)) return null;
  return { minValue, maxValue };
}

function lockCurrentPriceRange() {
  const range = visiblePriceRange();
  if (range) lockedPriceRange = range;
  return lockedPriceRange;
}

function refreshLockedPriceScale() {
  if (followHead) {
    lockedPriceRange = null;
    try {
      chart.priceScale("right").applyOptions({ autoScale: true });
    } catch {
      /* scale not ready */
    }
    return;
  }
  if (!lockedPriceRange) lockCurrentPriceRange();
  if (!lockedPriceRange) return;
  try {
    chart.priceScale("right").applyOptions({ autoScale: true });
    chart.priceScale("right").applyOptions({ autoScale: false });
  } catch {
    /* scale not ready */
  }
}

function applyPricePan(event) {
  if (!pricePan || event.pointerId !== pricePan.pointerId) return;
  const height = pricePan.height || 1;
  const span = pricePan.max - pricePan.min;
  const shift = ((event.clientY - pricePan.startY) / height) * span;
  lockedPriceRange = {
    minValue: pricePan.min + shift,
    maxValue: pricePan.max + shift,
  };
  refreshLockedPriceScale();
  drawOverlay();
}

function zoomLockedPrice(event) {
  const range = lockedPriceRange || visiblePriceRange();
  if (!range) return;
  const height = overlayHeight || 1;
  const rect = canvas.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (event.clientY - rect.top) / height));
  const span = range.maxValue - range.minValue;
  const factor = event.deltaY > 0 ? 1.08 : 1 / 1.08;
  const nextSpan = Math.max(0.2, span * factor);
  const priceAtCursor = range.maxValue - frac * span;
  lockedPriceRange = {
    minValue: priceAtCursor - nextSpan * (1 - frac),
    maxValue: priceAtCursor + nextSpan * frac,
  };
  refreshLockedPriceScale();
  drawOverlay();
}

function isOnPriceAxis(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  return (
    clientX > rect.left + overlayWidth &&
    clientY >= rect.top &&
    clientY <= rect.top + overlayHeight
  );
}

function setFollowHead(on, { snap = true } = {}) {
  followHead = Boolean(on);
  document.querySelectorAll("[data-follow]").forEach((button) => {
    button.classList.toggle("active", followHead);
    button.textContent = followHead ? "Follow on" : "Follow off";
  });
  try {
    localStorage.setItem(FOLLOW_STORAGE_KEY, followHead ? "1" : "0");
  } catch {
    /* ignore */
  }
  if (followHead) {
    pricePan = null;
    lockedPriceRange = null;
    refreshLockedPriceScale();
    if (snap) revealReplayHead();
  } else {
    lockCurrentPriceRange();
    refreshLockedPriceScale();
  }
  drawOverlay();
}

function revealReplayHead() {
  if (!currentCandles.length) return;
  followProgrammatic += 1;
  try {
    const last = currentCandles.length - 1;
    const range = chart.timeScale().getVisibleLogicalRange();
    const width = range ? Math.max(40, range.to - range.from) : 180;
    chart.timeScale().setVisibleLogicalRange({
      from: last - width + 5,
      to: last + 5,
    });
  } finally {
    window.setTimeout(() => {
      followProgrammatic = Math.max(0, followProgrammatic - 1);
    }, 0);
  }
}

function keepHeadInView() {
  if (!followHead || !currentCandles.length) return;
  const range = chart.timeScale().getVisibleLogicalRange();
  const last = currentCandles.length - 1;
  followProgrammatic += 1;
  try {
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
  } finally {
    window.setTimeout(() => {
      followProgrammatic = Math.max(0, followProgrammatic - 1);
    }, 0);
  }
}

function finishRender() {
  updateOhlc(currentCandles[currentCandles.length - 1]);
  updateStatus();
  syncDateInputs();
  keepHeadInView();
  refreshLockedPriceScale();
  drawOverlay();
  syncEventMarkers();
  updatePaperHud();
  checkPaperStops();
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
  refreshLockedPriceScale();
  drawOverlay();
  scheduleOverlayDraw(4);
  syncEventMarkers();
  updatePaperHud();
  checkPaperStops();
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

function nextArchiveBar(stepTf = effectiveStepTf()) {
  const opts = { preserveRange: !followHead };
  if (!dailyCandles.length) return;
  const idx = lastIndexAtOrBefore(replayTime, dailyCandles);
  if (idx < 0) {
    setReplayTime(dailyCandles[0].time, opts);
    return;
  }
  const stepSec = chartTfSeconds(stepTf);
  // Daily files have no 1m/5m/1h prints. One Gold_Daily bar is the finest step.
  if (stepTf === "1d" || currentTimeframe === "1d" || stepSec < 86400) {
    if (idx + 1 >= dailyCandles.length) {
      setReplayTime(dailyCandles[dailyCandles.length - 1].time, opts);
      return;
    }
    setReplayTime(dailyCandles[idx + 1].time, opts);
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
    setReplayTime(dailyCandles[dailyCandles.length - 1].time, opts);
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
  setReplayTime(dailyCandles[end].time, opts);
}

function advanceSourceByTf(tf, opts) {
  if (!sourceCandles.length || isReplayEnded()) {
    pauseReplay("End of data");
    setLiveState("ended", "End of data");
    return;
  }
  if (sourceIndex < 0) {
    setReplayTime(sourceCandles[0].time, opts);
    return;
  }
  if (tf === "1m") {
    setReplayIndex(sourceIndex + 1, opts);
    return;
  }
  const currentBucket = bucketStart(sourceCandles[sourceIndex].time, tf);
  let index = sourceIndex + 1;
  while (
    index < sourceCandles.length &&
    bucketStart(sourceCandles[index].time, tf) === currentBucket
  ) {
    index += 1;
  }
  if (index >= sourceCandles.length) {
    setReplayIndex(sourceCandles.length - 1, opts);
    return;
  }
  const nextBucket = bucketStart(sourceCandles[index].time, tf);
  let end = index;
  while (
    end + 1 < sourceCandles.length &&
    bucketStart(sourceCandles[end + 1].time, tf) === nextBucket
  ) {
    end += 1;
  }
  setReplayIndex(end, opts);
}

function advanceBySelectedStep(opts) {
  const stepTf = effectiveStepTf();
  if (replayStep === "current" || stepTf === currentTimeframe) {
    advanceSourceByTf(currentTimeframe, opts);
    return;
  }
  jumpBySeconds(chartTfSeconds(stepTf), opts);
}

function nextCandle() {
  const opts = { preserveRange: !followHead };
  const stepTf = effectiveStepTf();

  if (formingFromMinutes()) {
    advanceBySelectedStep(opts);
    return;
  }

  if (usesDailyArchive()) {
    if (isReplayEnded()) {
      pauseReplay("End of data");
      setLiveState("ended", "End of data");
      return;
    }
    if (replayStep !== "current" && sourceCandles.length && replayTime < sourceCandles[0].time) {
      const idx = lastIndexAtOrBefore(replayTime, dailyCandles);
      const nextDaily = idx + 1 < dailyCandles.length ? dailyCandles[idx + 1] : null;
      if (!nextDaily || nextDaily.time >= sourceCandles[0].time) {
        setReplayTime(sourceCandles[0].time, opts);
        return;
      }
    }
    nextArchiveBar(stepTf);
    return;
  }

  if (!sourceCandles.length || isReplayEnded()) {
    pauseReplay("End of data");
    setLiveState("ended", "End of data");
    return;
  }
  if (sourceIndex < 0 || replayTime < sourceCandles[0].time) {
    setReplayTime(sourceCandles[0].time, opts);
    return;
  }
  advanceBySelectedStep(opts);
}

function jumpBySeconds(seconds) {
  const opts = { preserveRange: !followHead };
  if (formingFromMinutes()) {
    if (!sourceCandles.length) return;
    if (sourceIndex < 0) {
      setReplayTime(sourceCandles[0].time, opts);
      return;
    }
    const target = sourceCandles[sourceIndex].time + seconds;
    let index = lastIndexAtOrBefore(target);
    if (index <= sourceIndex) {
      index = firstIndexAtOrAfter(target);
    }
    if (index >= sourceCandles.length) {
      setReplayIndex(sourceCandles.length - 1, opts);
      return;
    }
    setReplayIndex(index, opts);
    return;
  }
  if (usesDailyArchive()) {
    if (seconds >= 86400) {
      const days = Math.max(1, Math.round(seconds / 86400));
      const idx = lastIndexAtOrBefore(replayTime, dailyCandles);
      const next = Math.min(dailyCandles.length - 1, Math.max(0, idx) + days);
      setReplayTime(dailyCandles[next].time, opts);
      return;
    }
    nextArchiveBar();
    return;
  }
  if (!sourceCandles.length) return;
  if (sourceIndex < 0) {
    setReplayTime(sourceCandles[0].time, opts);
    return;
  }
  const target = sourceCandles[sourceIndex].time + seconds;
  let index = lastIndexAtOrBefore(target);
  if (index <= sourceIndex) {
    index = firstIndexAtOrAfter(target);
  }
  if (index >= sourceCandles.length) {
    setReplayIndex(sourceCandles.length - 1, opts);
    return;
  }
  setReplayIndex(index, opts);
}

function jumpNextDay() {
  const opts = { preserveRange: !followHead };
  if (usesDailyArchive()) {
    nextArchiveBar();
    return;
  }
  if (!sourceCandles.length) return;
  if (sourceIndex < 0) {
    setReplayTime(sourceCandles[0].time, opts);
    return;
  }
  const sessionDay = currentTimeframe === "1d";
  const current = sessionDay
    ? utcSessionDate(sourceCandles[sourceIndex].time)
    : istParts(sourceCandles[sourceIndex].time).date;
  for (let i = sourceIndex + 1; i < sourceCandles.length; i += 1) {
    const day = sessionDay
      ? utcSessionDate(sourceCandles[i].time)
      : istParts(sourceCandles[i].time).date;
    if (day !== current) {
      setReplayIndex(i, opts);
      return;
    }
  }
  setReplayIndex(sourceCandles.length - 1, opts);
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

let paperState = null;
let paperLines = [];
let paperSettingsTimer = 0;
let paperStopsTimer = 0;
let paperChecking = false;
let paperSide = "long";
let paperPlaceKind = null;
let paperStopDrag = null;
let paperTagHits = [];
let paperPlacePress = null;

function paperPositions() {
  const raw = paperState?.open;
  if (!raw) return [];
  return Array.isArray(raw) ? raw.filter(Boolean) : [raw];
}

function paperById(id) {
  return paperPositions().find((pos) => Number(pos.id) === Number(id)) || null;
}

function setPaperPlaceHint(text, active) {
  const hint = paperField("paper-place-hint");
  if (!hint) return;
  hint.classList.toggle("active", !!active);
  if (text) hint.textContent = text;
}

function idlePaperPlaceHint() {
  paperPlaceKind = null;
  setPaperPlaceHint("Each order has its own line. Click TP or SL, then drag. × closes that order.", false);
}

function replayFillPrice() {
  const bar = currentCandles[currentCandles.length - 1];
  return bar ? Number(bar.close) : null;
}

function paperField(id) {
  return document.getElementById(id);
}

function isPaperShort(open = null) {
  if (open?.side) return String(open.side).toUpperCase() === "SHORT";
  return paperSide === "short";
}

function fmtPnl(value) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function paperExitFill(open, signal, kind) {
  const slip = Number(open.slippagePct) / 100;
  if (kind === "limit") return Number(signal);
  if (isPaperShort(open)) return Number(signal) * (1 + slip);
  return Number(signal) * (1 - slip);
}

function paperNetPnl(open, signal, kind) {
  if (!open || signal == null || !Number.isFinite(Number(signal))) return null;
  const qty = Number(open.quantity);
  const fill = paperExitFill(open, signal, kind);
  const entry = Number(open.entryFill);
  const gross = isPaperShort(open) ? (entry - fill) * qty : (fill - entry) * qty;
  const taker = Number(open.takerFeePct) / 100;
  const gst = Number(open.gstPct) / 100;
  const exitCharge = Math.abs(qty * fill) * taker * (1 + gst);
  return gross - exitCharge - Number(open.entryCharges || 0);
}

function paperInputPrice(id) {
  const el = paperField(id);
  const value = Number(el?.value);
  return el && el.value !== "" && Number.isFinite(value) && value > 0 ? value : null;
}

function readPaperStops() {
  return {
    tp: paperField("paper-tp")?.value ?? "",
    sl: paperField("paper-sl")?.value ?? "",
    tpPts: paperField("paper-tp-pts")?.value ?? "",
    slPts: paperField("paper-sl-pts")?.value ?? "",
    side: isPaperShort() ? "SHORT" : "LONG",
  };
}

function paperRefPrice() {
  return replayFillPrice();
}

function syncPtsFromPrices() {
  const ref = paperRefPrice();
  const tpPts = paperField("paper-tp-pts");
  const slPts = paperField("paper-sl-pts");
  if (ref == null || !tpPts || !slPts) return;
  const tp = paperInputPrice("paper-tp");
  const sl = paperInputPrice("paper-sl");
  const short = isPaperShort();
  if (document.activeElement !== tpPts) {
    tpPts.value = tp != null ? Math.max(0, short ? ref - tp : tp - ref).toFixed(2) : "";
  }
  if (document.activeElement !== slPts) {
    slPts.value = sl != null ? Math.max(0, short ? sl - ref : ref - sl).toFixed(2) : "";
  }
}

function applyPtsToPrice(kind) {
  const ref = paperRefPrice();
  if (ref == null) return;
  const short = isPaperShort();
  if (kind === "tp") {
    const pts = Number(paperField("paper-tp-pts").value);
    if (Number.isFinite(pts) && pts > 0) {
      paperField("paper-tp").value = (short ? ref - pts : ref + pts).toFixed(2);
    }
  } else {
    const pts = Number(paperField("paper-sl-pts").value);
    if (Number.isFinite(pts) && pts > 0) {
      paperField("paper-sl").value = (short ? ref + pts : ref - pts).toFixed(2);
    }
  }
}

function setPaperSide(side, { force = false } = {}) {
  paperSide = side === "short" ? "short" : "long";
  paperField("ticket-long")?.classList.toggle("active", paperSide === "long");
  paperField("ticket-short")?.classList.toggle("active", paperSide === "short");
  const ref = paperRefPrice();
  const tp = paperInputPrice("paper-tp");
  const sl = paperInputPrice("paper-sl");
  if (ref != null) {
    if (paperSide === "short") {
      if (tp != null && tp >= ref && paperField("paper-tp")) paperField("paper-tp").value = "";
      if (sl != null && sl <= ref && paperField("paper-sl")) paperField("paper-sl").value = "";
    } else {
      if (tp != null && tp <= ref && paperField("paper-tp")) paperField("paper-tp").value = "";
      if (sl != null && sl >= ref && paperField("paper-sl")) paperField("paper-sl").value = "";
    }
  }
  updateTicketButtons();
  updateTicketPreview();
  syncPtsFromPrices();
}

function togglePaperTpsl(forceOpen) {
  const box = paperField("paper-tpsl");
  if (!box) return;
  const open = forceOpen === true || (forceOpen !== false && box.hidden);
  box.hidden = !open;
  const toggle = paperField("paper-tpsl-toggle");
  if (toggle) toggle.textContent = open ? "TP / SL" : "+ Add TP / SL";
  if (open && !paperPlaceKind) idlePaperPlaceHint();
}

function paperEquity() {
  const cash = Number(paperState?.cash ?? 0);
  const now = replayFillPrice();
  const locked = paperPositions().reduce((sum, pos) => sum + Number(pos.moneyUsed || 0), 0);
  let pnl = 0;
  let mtm = 0;
  if (now != null) {
    paperPositions().forEach((pos) => {
      const net = paperNetPnl(pos, now, "market") || 0;
      pnl += net;
      mtm += net + Number(pos.entryCharges || 0);
    });
  }
  return { cash, locked, pnl, actual: cash + locked + mtm };
}

function fillPaperInputs(state, { forceCapital = false, syncSettings = false } = {}) {
  const capital = paperField("paper-capital");
  if (capital) {
    capital.disabled = paperPositions().length > 0;
    if (forceCapital || document.activeElement !== capital) {
      capital.value = String(state.setCapital ?? 10000);
    }
  }
  if (syncSettings) {
    const map = [
      ["paper-size", "sizePct"],
      ["paper-leverage", "leverage"],
      ["paper-slip", "slippagePct"],
      ["paper-taker", "takerFeePct"],
    ];
    map.forEach(([id, key]) => {
      const el = paperField(id);
      if (el && document.activeElement !== el && state[key] != null) {
        el.value = String(state[key]);
      }
    });
  }
  const sizeLabel = paperField("paper-size-label");
  if (sizeLabel) sizeLabel.textContent = `${Number(paperField("paper-size")?.value || 0)}%`;
  const sizeUsdt = paperField("paper-size-usdt");
  if (sizeUsdt && document.activeElement !== sizeUsdt) {
    const cash = Number(state.cash);
    const pct = Number(paperField("paper-size")?.value || state.sizePct || 0);
    sizeUsdt.value = Number.isFinite(cash * pct / 100) ? (cash * pct / 100).toFixed(2) : "";
  }
}

function upsertPaperLine(line, options) {
  if (!options) {
    if (line) {
      try { candleSeries.removePriceLine(line); } catch { /* gone */ }
    }
    return null;
  }
  if (line) {
    line.applyOptions(options);
    return line;
  }
  return candleSeries.createPriceLine(options);
}

function setPnlText(id, value) {
  const el = paperField(id);
  if (!el) return;
  el.textContent = fmtPnl(value);
  el.className = value > 0 ? "up" : value < 0 ? "down" : "";
}

function currentPaperLevels(pos) {
  if (!pos) return { tp: null, sl: null };
  const tp = pos.tp != null && pos.tp !== "" ? Number(pos.tp) : null;
  const sl = pos.sl != null && pos.sl !== "" ? Number(pos.sl) : null;
  return {
    tp: Number.isFinite(tp) && tp > 0 ? tp : null,
    sl: Number.isFinite(sl) && sl > 0 ? sl : null,
  };
}

function syncPaperLine() {
  const positions = paperPositions();
  const dotted = LightweightCharts.LineStyle.Dotted;
  while (paperLines.length > positions.length) {
    const extra = paperLines.pop();
    upsertPaperLine(extra.entry, null);
    upsertPaperLine(extra.tp, null);
    upsertPaperLine(extra.sl, null);
  }
  positions.forEach((pos, index) => {
    if (!paperLines[index]) paperLines[index] = { id: pos.id, entry: null, tp: null, sl: null };
    const short = isPaperShort(pos);
    const { tp, sl } = currentPaperLevels(pos);
    paperLines[index].id = pos.id;
    paperLines[index].entry = upsertPaperLine(paperLines[index].entry, {
      price: Number(pos.entryFill),
      color: short ? "#f6465d" : "#00c076",
      lineWidth: 1,
      lineStyle: dotted,
      axisLabelVisible: true,
      title: "",
    });
    paperLines[index].tp = upsertPaperLine(paperLines[index].tp, tp != null ? {
      price: tp,
      color: "#00c076",
      lineWidth: 1,
      lineStyle: dotted,
      axisLabelVisible: true,
      title: "",
    } : null);
    paperLines[index].sl = upsertPaperLine(paperLines[index].sl, sl != null ? {
      price: sl,
      color: "#f0b90b",
      lineWidth: 1,
      lineStyle: dotted,
      axisLabelVisible: true,
      title: "",
    } : null);
  });
  const box = paperField("paper-pl-box");
  if (box) box.hidden = true;
  refreshLockedPriceScale();
  scheduleOverlayDraw(2);
  syncEventMarkers();
}

function paperDisplayCash() {
  const cash = Number(paperState?.cash ?? 0);
  const typed = Number(paperField("paper-capital")?.value);
  const setCap = Number(paperState?.setCapital ?? cash);
  if (
    !paperPositions().length &&
    Number.isFinite(typed) &&
    typed > 0 &&
    Number.isFinite(setCap) &&
    Math.abs(typed - setCap) > 0.009
  ) {
    return typed;
  }
  return Number.isFinite(cash) ? cash : typed;
}

function updateTicketPreview() {
  const cash = paperDisplayCash();
  const pct = Number(paperField("paper-size")?.value || 0);
  const lev = Math.max(1, Number(paperField("paper-leverage")?.value || 1));
  const slip = Number(paperField("paper-slip")?.value || 0) / 100;
  const taker = Number(paperField("paper-taker")?.value || 0) / 100;
  const gst = Number(paperState?.gstPct ?? 18) / 100;
  const px = replayFillPrice();
  const short = isPaperShort();
  const mark = paperField("paper-mark-price");
  if (mark && document.activeElement !== mark) {
    mark.value = px != null ? `${px.toFixed(2)} · Market` : "Market Price";
  }
  const available = paperField("paper-available");
  if (available) available.textContent = Number.isFinite(cash) ? cash.toFixed(2) : "—";
  const setCapEl = paperField("paper-setcap");
  if (setCapEl) {
    const setCap = Number(paperState?.setCapital);
    setCapEl.textContent = Number.isFinite(setCap) ? setCap.toFixed(2) : "—";
  }
  const capitalNote = paperField("paper-capital-note");
  if (capitalNote) {
    capitalNote.textContent = paperPositions().length
      ? "Close open trades first. Available is leftover cash, not Set capital."
      : "Starting money. Save applies it to Available when you have no open trades. Actual capital above moves with P&L.";
  }
  const actualEl = paperField("paper-actual");
  if (actualEl) {
    const { actual, pnl } = paperEquity();
    actualEl.textContent = Number.isFinite(actual) ? actual.toFixed(2) : "—";
    actualEl.className = pnl > 0 ? "up" : pnl < 0 ? "down" : "";
  }
  const sizeLabel = paperField("paper-size-label");
  if (sizeLabel) sizeLabel.textContent = `${pct}%`;
  const maxEl = paperField("paper-max");
  if (maxEl) {
    const maxNotional = Math.max(0, cash * lev);
    maxEl.textContent = `${short ? "Max Sell" : "Max Buy"} ${maxNotional.toFixed(2)}`;
  }
  const sizeUsdt = paperField("paper-size-usdt");
  const margin = cash * (pct / 100);
  if (sizeUsdt && document.activeElement !== sizeUsdt) {
    sizeUsdt.value = pct > 0 ? margin.toFixed(2) : "";
  }
  const marginEl = paperField("paper-margin");
  const qtyEl = paperField("paper-qty");
  const liqEl = paperField("paper-liq");
  if (marginEl) marginEl.textContent = pct > 0 ? margin.toFixed(2) : "—";
  if (px == null || pct <= 0) {
    if (qtyEl) qtyEl.textContent = `~0.000 ${paperAssetUnit()}`;
    if (liqEl) liqEl.textContent = "—";
    return;
  }
  const fill = short ? px * (1 - slip) : px * (1 + slip);
  const chargeRate = taker * (1 + gst);
  const money = margin / (1 + lev * chargeRate);
  const qty = Math.floor((money * lev) / fill / 0.001) * 0.001;
  if (qtyEl) qtyEl.textContent = `~${Math.max(0, qty).toFixed(3)} ${paperAssetUnit()}`;
  if (liqEl) {
    const liq = short ? fill * (1 + 1 / lev) : fill * (1 - 1 / lev);
    liqEl.textContent = lev > 1 ? liq.toFixed(2) : "—";
  }
}

function updateTicketButtons() {
  const positions = paperPositions();
  const submit = paperField("paper-submit");
  const closeBtn = paperField("paper-close");
  const wantShort = paperSide === "short";
  paperField("ticket-long")?.classList.toggle("active", !wantShort);
  paperField("ticket-short")?.classList.toggle("active", wantShort);
  paperField("ticket-long")?.removeAttribute("disabled");
  paperField("ticket-short")?.removeAttribute("disabled");
  if (submit) {
    submit.className = `ticket-submit ${wantShort ? "short" : "long"}`;
    submit.textContent = wantShort ? "Sell / Short" : "Buy / Long";
    submit.hidden = false;
    submit.disabled = Number(paperField("paper-size")?.value || 0) <= 0;
  }
  if (closeBtn) {
    closeBtn.hidden = !positions.length;
    closeBtn.disabled = !positions.length;
    closeBtn.textContent = positions.length > 1 ? `Close all (${positions.length})` : "Close all";
  }
}

function paperQtyStep(qty) {
  const stepped = Math.floor((Number(qty) + 1e-9) / 0.001) * 0.001;
  return Math.max(0, Number(stepped.toFixed(3)));
}

function bindPaperPartial(box) {
  box.querySelectorAll(".paper-book-row").forEach((row) => {
    const id = Number(row.dataset.id);
    const pos = paperById(id);
    if (!pos) return;
    const remain = paperRemainQty(pos);
    const input = row.querySelector(".paper-partial-qty");
    row.querySelectorAll("[data-pct]").forEach((btn) => {
      btn.addEventListener("click", (event) => {
        event.preventDefault();
        const pct = Number(btn.dataset.pct);
        const qty = paperQtyStep(remain * (pct / 100));
        if (input) input.value = qty.toFixed(3);
      });
    });
    row.querySelector("[data-exit]")?.addEventListener("click", (event) => {
      event.preventDefault();
      const typed = Number(input?.value);
      const qty = Number.isFinite(typed) && typed > 0 ? Math.min(remain, typed) : remain;
      paperClose(id, qty);
    });
    input?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const typed = Number(input.value);
      const qty = Number.isFinite(typed) && typed > 0 ? Math.min(remain, typed) : remain;
      paperClose(id, qty);
    });
  });
}

function renderPaperHeld() {
  const el = paperField("paper-held");
  if (!el) return;
  const rows = paperPositions();
  const now = replayFillPrice();
  if (!rows.length) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = rows.map((pos) => {
    const remain = paperRemainQty(pos);
    const bought = paperBoughtQty(pos);
    const markVal = now != null ? remain * now : null;
    const leftover = bought > remain + 1e-9
      ? ` · after partial ${remain.toFixed(3)} left`
      : "";
    return `<div>Bought ${bought.toFixed(3)} ${paperAssetUnit()} · Remain ${remain.toFixed(3)} ${paperAssetUnit()}`
      + (markVal != null ? ` · Now ${markVal.toFixed(2)}` : "")
      + leftover
      + ` @ ${now != null ? now.toFixed(2) : Number(pos.entryFill).toFixed(2)}</div>`;
  }).join("");
}

function renderPaperBooks() {
  const box = paperField("paper-books");
  if (!box) return;
  const rows = paperPositions();
  const now = replayFillPrice();
  renderPaperHeld();
  if (!rows.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  const active = document.activeElement;
  const keep = active && box.contains(active) && active.classList.contains("paper-partial-qty")
    ? { id: active.closest("[data-id]")?.dataset.id, value: active.value, start: active.selectionStart, end: active.selectionEnd }
    : null;
  box.hidden = false;
  box.innerHTML = rows.map((pos) => {
    const short = isPaperShort(pos);
    const remain = paperRemainQty(pos);
    const bought = paperBoughtQty(pos);
    const pnl = now != null ? paperNetPnl(pos, now, "market") : 0;
    const cls = pnl > 0 ? "up" : pnl < 0 ? "down" : "";
    const markVal = now != null ? remain * now : null;
    return `<div class="paper-book-row ${short ? "short" : "long"}" data-id="${pos.id}">
      <span>${short ? "SHORT" : "LONG"}</span>
      <span>Bought ${bought.toFixed(3)} · Remain ${remain.toFixed(3)} @ ${Number(pos.entryFill).toFixed(2)}</span>
      <b class="${cls}">${fmtPnl(pnl)}</b>
      <div class="paper-book-value">${markVal != null ? `Remain now ${markVal.toFixed(2)} @ ${now.toFixed(2)}` : ""}</div>
      <div class="paper-partial">
        <input class="paper-partial-qty" type="number" min="0.001" step="0.001" max="${remain}" value="${remain.toFixed(3)}" title="Quantity to exit">
        <button type="button" data-pct="25">25%</button>
        <button type="button" data-pct="50">50%</button>
        <button type="button" data-pct="75">75%</button>
        <button type="button" data-pct="100">100%</button>
        <button type="button" data-exit="1">Exit qty</button>
      </div>
    </div>`;
  }).join("");
  bindPaperPartial(box);
  if (keep?.id) {
    const input = box.querySelector(`.paper-book-row[data-id="${keep.id}"] .paper-partial-qty`);
    if (input) {
      input.value = keep.value;
      input.focus();
      try { input.setSelectionRange(keep.start, keep.end); } catch { /* ignore */ }
    }
  }
}

function updatePositionStatus() {
  const wrap = paperField("paper-pos");
  const mark = paperField("paper-pos-mark");
  const title = paperField("paper-pos-title");
  const sub = paperField("paper-pos-sub");
  if (!wrap || !mark || !title || !sub) return;
  const rows = paperPositions();
  const last = paperState?.lastClosed;
  wrap.classList.remove("is-long", "is-short", "is-flat");
  renderPaperBooks();
  if (rows.length) {
    const longs = rows.filter((pos) => !isPaperShort(pos)).length;
    const shorts = rows.length - longs;
    const now = replayFillPrice();
    const pnl = now == null
      ? 0
      : rows.reduce((sum, pos) => sum + (paperNetPnl(pos, now, "market") || 0), 0);
    wrap.classList.add(shorts && !longs ? "is-short" : "is-long");
    mark.textContent = rows.length > 1 ? String(rows.length) : (shorts ? "▼" : "▲");
    title.textContent = rows.length > 1 ? `${rows.length} OPEN` : (shorts ? "IN SHORT" : "IN LONG");
    if (rows.length === 1) {
      const pos = rows[0];
      const remain = paperRemainQty(pos);
      const bought = paperBoughtQty(pos);
      const markVal = now != null ? remain * now : null;
      sub.textContent = `Bought ${bought.toFixed(3)} · Remain ${remain.toFixed(3)}`
        + (markVal != null ? ` · Now ${markVal.toFixed(2)}` : "")
        + ` · ${fmtPnl(pnl)}`;
    } else {
      sub.textContent = `${longs} long · ${shorts} short · ${fmtPnl(pnl)}`;
    }
    return;
  }
  wrap.classList.add("is-flat");
  if (last) {
    mark.textContent = "X";
    title.textContent = `EXITED ${last.side}`;
    sub.textContent =
      `${Number(last.quantity).toFixed(3)} ${paperAssetUnit()} @ ${Number(last.exitFill).toFixed(2)} · ${fmtPnl(Number(last.pnl))} · ${last.reason || "CLOSE"}`;
    return;
  }
  mark.textContent = "○";
  title.textContent = "No position";
  sub.textContent = "Click Buy / Long or Sell / Short to enter";
}

function updatePaperHud() {
  const hud = paperField("paper-hud");
  if (!hud) return;
  const state = paperState;
  if (!state) {
    hud.textContent = "Paper book loading…";
    hud.className = "paper-hud";
    return;
  }
  updateTicketButtons();
  updatePositionStatus();
  const { cash, actual, pnl } = paperEquity();
  updateTicketPreview();
  const rows = paperPositions();
  const cls = pnl > 0 ? "up" : pnl < 0 ? "down" : "";
  hud.className = `paper-hud ${cls}`.trim();
  const txns = state.transactionCount || 0;
  const closed = state.closedTrades || 0;
  const setCap = Number(state.setCapital || 0);
  if (!rows.length) {
    hud.textContent =
      `Actual ${actual.toFixed(2)} · Set ${setCap.toFixed(2)} · ${closed} trades · ${txns} txns`;
  } else {
    hud.textContent =
      `Actual ${actual.toFixed(2)} · ${fmtPnl(pnl)} · ${rows.length} open · avail ${cash.toFixed(2)}`;
  }
  hud.title = `${state.tradesFile || ""} | ${state.transactionsFile || ""} | ${state.dailyFile || ""}`;
  const txnPath = paperField("paper-file-txn");
  const tradePath = paperField("paper-file-trades");
  const dailyPath = paperField("paper-file-daily");
  if (txnPath) txnPath.textContent = state.transactionsFile || "";
  if (tradePath) tradePath.textContent = state.tradesFile || "";
  if (dailyPath) dailyPath.textContent = state.dailyFile || "";
  idlePaperPlaceHint();
  syncPaperLine();
}

function applyPaperState(state, { forceCapital = false, syncSettings = false } = {}) {
  paperState = state;
  fillPaperInputs(state, { forceCapital, syncSettings });
  updatePaperHud();
}

async function loadPaperState() {
  try {
    const response = await fetch("/api/paper");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load paper book");
    applyPaperState(payload, { forceCapital: true, syncSettings: true });
  } catch (error) {
    const hud = paperField("paper-hud");
    if (hud) hud.textContent = error.message;
  }
}

async function paperPost(body) {
  const response = await fetch("/api/paper", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Paper request failed");
  applyPaperState(payload, {
    forceCapital: body.action === "set_capital" || body.capital != null,
    syncSettings: body.action === "settings" || body.action === "set_capital",
  });
  return payload;
}

function paperMark() {
  const price = replayFillPrice();
  if (price == null || !replayTime) {
    throw new Error("No candle to trade. Jump to a date first.");
  }
  return { price, time: replayTime, timeframe: currentTimeframe };
}

async function paperSubmit() {
  try {
    await paperPost({
      action: paperSide === "short" ? "short" : "long",
      ...paperMark(),
      ...readPaperStops(),
      sizePct: Number(paperField("paper-size")?.value || paperState?.sizePct || 100),
      leverage: Number(paperField("paper-leverage")?.value || paperState?.leverage || 1),
    });
  } catch (error) {
    window.alert(error.message);
  }
}

async function paperClose(id, qty) {
  try {
    const body = { action: id == null ? "close_all" : "close", ...paperMark() };
    if (id != null) body.id = id;
    const closeQty = Number(qty);
    if (id != null && Number.isFinite(closeQty) && closeQty > 0) body.qty = closeQty;
    await paperPost(body);
  } catch (error) {
    window.alert(error.message);
  }
}

async function paperReverse(id) {
  try {
    await paperPost({ action: "reverse", id, ...paperMark() });
  } catch (error) {
    window.alert(error.message);
  }
}

async function paperSaveBook() {
  const btn = paperField("paper-book-save");
  const capital = Number(paperField("paper-capital")?.value);
  try {
    const body = {
      action: "settings",
      slippagePct: Number(paperField("paper-slip").value),
      takerFeePct: Number(paperField("paper-taker").value),
      leverage: Number(paperField("paper-leverage").value),
    };
    const sizePct = Number(paperField("paper-size").value);
    if (Number.isFinite(sizePct) && sizePct > 0) body.sizePct = sizePct;
    if (Number.isFinite(capital) && capital > 0) body.capital = capital;
    if (replayTime) body.time = replayTime;
    body.timeframe = currentTimeframe;
    const payload = await paperPost(body);
    if (payload?.capitalIgnored) {
      window.alert("Close all open trades before changing capital. Available is leftover cash while a trade is open.");
    }
    if (btn) {
      btn.textContent = "Saved";
      setTimeout(() => { if (btn) btn.textContent = "Save book settings"; }, 1200);
    }
  } catch (error) {
    window.alert(error.message);
    if (btn) btn.textContent = "Save book settings";
  }
}

function schedulePaperStops(id) {
  if (id == null) return;
  if (paperStopsTimer) clearTimeout(paperStopsTimer);
  paperStopsTimer = setTimeout(() => savePositionStops(id), 250);
}

function paperAssetUnit() {
  return feedUnit || "XAU";
}

function paperRemainQty(pos) {
  return Number(pos?.quantity || 0);
}

function paperBoughtQty(pos) {
  const remain = paperRemainQty(pos);
  const bought = Number(pos?.originalQuantity);
  return Number.isFinite(bought) && bought > 0 ? bought : remain;
}

function clampPaperStopPrice(kind, price, pos) {
  const mark = replayFillPrice();
  if (mark == null || !Number.isFinite(price)) return price;
  const tick = mark > 500 ? 0.1 : 0.01;
  const short = isPaperShort(pos);
  if (kind === "sl") {
    return short ? Math.max(price, mark + tick) : Math.min(price, mark - tick);
  }
  if (kind === "tp") {
    return short ? Math.min(price, mark - tick) : Math.max(price, mark + tick);
  }
  return price;
}

function stopArmedAt(pos, kind) {
  const raw = Number(kind === "sl" ? pos?.slArmedAt : pos?.tpArmedAt);
  if (Number.isFinite(raw) && raw > 0) return raw;
  return Number(pos?.entryTime) || 0;
}

function paperStopPad(price) {
  const px = Number(price);
  if (!Number.isFinite(px) || px <= 0) return 8;
  return Math.max(px * 0.004, px > 500 ? 8 : 0.05);
}

function suggestedPaperLevels(pos) {
  const ref = Number(pos?.entryFill ?? paperRefPrice());
  if (!Number.isFinite(ref) || ref <= 0) return { tp: null, sl: null };
  const now = replayFillPrice();
  const short = isPaperShort(pos);
  const px = now != null && Number.isFinite(now) ? now : ref;
  const pad = paperStopPad(px);
  if (short) {
    let sl = px + pad;
    if (now != null && now < ref) sl = Math.max(sl, (ref + now) / 2);
    return { tp: px - pad, sl };
  }
  let sl = px - pad;
  if (now != null && now > ref) sl = Math.min(sl, (ref + now) / 2);
  return { tp: px + pad, sl };
}

function clearPaperLevel(kind, id) {
  if (id != null) {
    const pos = paperById(id);
    if (!pos) return;
    pos[kind] = null;
    syncPaperLine();
    savePositionStops(id);
    return;
  }
  const el = paperField(kind === "tp" ? "paper-tp" : "paper-sl");
  const pts = paperField(kind === "tp" ? "paper-tp-pts" : "paper-sl-pts");
  if (el) el.value = "";
  if (pts) pts.value = "";
  syncPtsFromPrices();
}

function setPaperLevel(kind, price, id, persist = true) {
  if (id == null || !Number.isFinite(Number(price)) || Number(price) <= 0) return;
  const pos = paperById(id);
  if (!pos) return;
  pos[kind] = Number(clampPaperStopPrice(kind, Number(price), pos).toFixed(2));
  syncPaperLine();
  if (persist) savePositionStops(id);
}

async function savePositionStops(id) {
  const pos = paperById(id);
  if (!pos) return;
  try {
    await paperPost({
      action: "stops",
      id,
      tp: pos.tp ?? "",
      sl: pos.sl ?? "",
      price: replayFillPrice(),
      time: replayTime,
    });
  } catch (error) {
    window.alert(error.message);
    loadPaperState();
  }
}

function roundChip(x, y, w, h, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

function drawChip(x, y, w, h, border, fill, text, fg, action, extra) {
  roundChip(x, y, w, h, 4);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = border;
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.fillStyle = fg;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.fillText(text, x + w / 2, y + h / 2 + 0.5);
  ctx.textAlign = "left";
  paperTagHits.push({ x, y, w, h, action, ...extra });
}

function paperPriceY(price) {
  const y = candleSeries.priceToCoordinate(price);
  if (y != null && Number.isFinite(y)) {
    return Math.min(overlayHeight - 16, Math.max(16, y));
  }
  const top = candleSeries.coordinateToPrice(8);
  const bot = candleSeries.coordinateToPrice(overlayHeight - 8);
  if (top == null || bot == null) return overlayHeight / 2;
  const hi = Math.max(top, bot);
  const lo = Math.min(top, bot);
  if (price >= hi) return 16;
  if (price <= lo) return overlayHeight - 16;
  return overlayHeight / 2;
}

function paperClusterX(clusterW) {
  const range = chart.timeScale().getVisibleLogicalRange();
  let x = null;
  if (range) {
    const last = currentCandles.length ? currentCandles.length - 1 : range.to;
    x = chart.timeScale().logicalToCoordinate(Math.min(range.to, last));
  }
  if (x == null || !Number.isFinite(x)) {
    const lastBar = currentCandles[currentCandles.length - 1];
    x = lastBar ? coordinateForTime(lastBar.time) : overlayWidth * 0.55;
  }
  if (x == null || !Number.isFinite(x)) x = overlayWidth * 0.55;
  return Math.min(Math.max(8, x + 10), Math.max(8, overlayWidth - clusterW - 16));
}

function drawPaperTags() {
  paperTagHits = [];
  const rows = paperPositions();
  if (!rows.length) return;
  const now = replayFillPrice();
  ctx.save();
  ctx.font = "700 11px Inter, sans-serif";
  const usedY = [];
  const placeY = (raw) => {
    let y = raw;
    for (let i = 0; i < 10; i += 1) {
      if (usedY.every((other) => Math.abs(other - y) > 26)) break;
      y += 26;
    }
    usedY.push(y);
    return y;
  };

  const levelTag = (pos, price, label, pnl, border, kind) => {
    const y = candleSeries.priceToCoordinate(price);
    if (y == null) return;
    const text = `${label}  ${fmtPnl(pnl)}`;
    const h = 22;
    const w = Math.ceil(ctx.measureText(`${text}  ×`).width) + 18;
    const x = Math.max(8, overlayWidth - w - 14);
    const top = y - h / 2;
    roundChip(x, top, w, h, 4);
    ctx.fillStyle = "#0b0e11ee";
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.fillStyle = border;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 8, y + 0.5);
    ctx.fillText("×", x + w - 14, y + 0.5);
    paperTagHits.push({ x, y: top, w, h, action: "drag", kind, id: pos.id, price });
    paperTagHits.push({ x: x + w - 22, y: top, w: 22, h, action: "clear", kind, id: pos.id });
  };

  rows.forEach((pos) => {
    const { tp, sl } = currentPaperLevels(pos);
    const short = isPaperShort(pos);
    const suggest = suggestedPaperLevels(pos);
    const nowPnl = now != null ? paperNetPnl(pos, now, "market") : 0;
    const tpPnl = tp != null ? paperNetPnl(pos, tp, "limit") : null;
    const slPnl = sl != null ? paperNetPnl(pos, sl, "stop") : null;
    if (tp != null) levelTag(pos, tp, "TP", tpPnl, "#00c076", "tp");
    if (sl != null) levelTag(pos, sl, "SL", slPnl, "#f0b90b", "sl");
    const y = placeY(paperPriceY(Number(pos.entryFill)));
    const h = 24;
    const reverseW = 28;
    const tpW = 36;
    const slW = 36;
    const remain = paperRemainQty(pos);
    const bought = paperBoughtQty(pos);
    const qtyText = bought - remain > 1e-9
      ? `${remain.toFixed(3)}/${bought.toFixed(3)}`
      : remain.toFixed(3);
    const sideText = short ? "Short" : "Long";
    const pnlText = fmtPnl(nowPnl);
    const qtyW = Math.ceil(ctx.measureText(qtyText).width) + 16;
    const posW = Math.ceil(ctx.measureText(`${sideText} ${pnlText}`).width) + 16;
    const closeW = 26;
    const barW = qtyW + posW + closeW;
    const clusterW = reverseW + 4 + tpW + slW + 8 + barW;
    const x = paperClusterX(clusterW);
    const top = y - h / 2;
    const teal = "#00c076";
    const orange = "#f0b90b";
    const red = "#f6465d";
    const dark = "#0b0e11ee";
    drawChip(x, top, reverseW, h, red, dark, short ? "↑" : "↓", red, "reverse", { id: pos.id });
    drawChip(x + reverseW + 4, top, tpW, h, teal, tp != null ? "#00c07633" : dark, "TP", teal, "toggle-tp", {
      id: pos.id, price: suggest.tp,
    });
    drawChip(x + reverseW + 4 + tpW, top, slW, h, orange, sl != null ? "#f0b90b33" : dark, "SL", orange, "toggle-sl", {
      id: pos.id, price: suggest.sl,
    });
    const barX = x + reverseW + 4 + tpW + slW + 8;
    roundChip(barX, top, barW, h, 4);
    ctx.fillStyle = dark;
    ctx.fill();
    ctx.strokeStyle = short ? red : teal;
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.fillStyle = short ? red : teal;
    ctx.textBaseline = "middle";
    ctx.fillText(qtyText, barX + 8, y + 0.5);
    ctx.fillStyle = nowPnl < 0 ? red : teal;
    ctx.fillText(`${sideText} ${pnlText}`, barX + qtyW, y + 0.5);
    ctx.fillStyle = short ? red : teal;
    ctx.fillText("×", barX + qtyW + posW + 6, y + 0.5);
    paperTagHits.push({ x: barX, y: top, w: qtyW + posW, h, action: "none", id: pos.id });
    paperTagHits.push({ x: barX + qtyW + posW, y: top, w: closeW, h, action: "exit", id: pos.id });
  });
  ctx.restore();
}

function hitPaperTag(clientX, clientY) {
  if (!isInsidePlotArea(clientX, clientY) || !paperTagHits.length) return null;
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  for (let i = paperTagHits.length - 1; i >= 0; i -= 1) {
    const tag = paperTagHits[i];
    if (x >= tag.x && x <= tag.x + tag.w && y >= tag.y && y <= tag.y + tag.h) return tag;
  }
  return null;
}

function hitPaperStop(clientX, clientY) {
  if (!isInsidePlotArea(clientX, clientY)) return null;
  const rect = canvas.getBoundingClientRect();
  const y = clientY - rect.top;
  const hit = (price) => {
    const py = candleSeries.priceToCoordinate(price);
    return py != null && Math.abs(py - y) <= 8;
  };
  const rows = paperPositions();
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    const pos = rows[i];
    const { tp, sl } = currentPaperLevels(pos);
    if (tp != null && hit(tp)) return { kind: "tp", id: pos.id };
    if (sl != null && hit(sl)) return { kind: "sl", id: pos.id };
  }
  return null;
}

function placePaperStopAtPrice() {
  return false;
}

function canPlacePaperStops() {
  return false;
}

function paperStopSourceBars() {
  if (sourceIndex >= 0 && sourceCandles.length) {
    return sourceCandles.slice(0, sourceIndex + 1);
  }
  return currentCandles;
}

function paperStopTouched(pos, bar) {
  const { tp, sl } = currentPaperLevels(pos);
  const short = isPaperShort(pos);
  const entry = Number(pos.entryTime) || 0;
  if (bar.time <= entry) return false;
  if (short) {
    if (sl != null && bar.time > stopArmedAt(pos, "sl") && bar.high >= sl) return true;
    if (tp != null && bar.time > stopArmedAt(pos, "tp") && bar.low <= tp) return true;
    return false;
  }
  if (sl != null && bar.time > stopArmedAt(pos, "sl") && bar.low <= sl) return true;
  if (tp != null && bar.time > stopArmedAt(pos, "tp") && bar.high >= tp) return true;
  return false;
}

async function checkPaperStops() {
  if (paperStopDrag || paperChecking) return;
  const rows = paperPositions();
  if (!rows.length) return;
  const needs = rows.some((pos) => {
    const { tp, sl } = currentPaperLevels(pos);
    return tp != null || sl != null;
  });
  if (!needs) return;
  const source = paperStopSourceBars();
  const bars = source.filter((bar) => rows.some((pos) => paperStopTouched(pos, bar)));
  if (!bars.length) return;
  paperChecking = true;
  try {
    await paperPost({
      action: "check",
      time: replayTime,
      bars: bars.map((bar) => ({
        time: bar.time,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      })),
    });
  } catch {
    /* older server without check */
  } finally {
    paperChecking = false;
  }
}

function fillFeedSelect(payload) {
  const select = document.getElementById("data-feed");
  if (!select) return;
  const feeds = Array.isArray(payload.feeds) ? payload.feeds : [];
  const current = payload.feed || "";
  select.innerHTML = "";
  feeds.forEach((feed) => {
    const option = document.createElement("option");
    option.value = feed.id;
    option.textContent = feed.name || feed.id;
    select.appendChild(option);
  });
  if (current) select.value = current;
  select.disabled = feeds.length < 2;
}

function applyFeedMeta(payload) {
  const name = payload.name || payload.feed || "Gold";
  const symbol = payload.symbol || "";
  const note = payload.note || "";
  feedUnit = payload.unit || (String(symbol).includes("XAG") ? "XAG" : "XAU");
  feedDailyName = payload.dailyName || (feedUnit === "XAG" ? "Silver_Daily.csv" : "Gold_Daily.csv");
  const nameEl = document.getElementById("symbol-name");
  const codeEl = document.getElementById("symbol-code");
  const noteEl = document.getElementById("source-note");
  const iconEl = document.getElementById("symbol-icon");
  const loadingEl = document.getElementById("loading");
  if (nameEl) nameEl.textContent = name;
  if (iconEl) iconEl.textContent = payload.icon || (feedUnit === "XAG" ? "AG" : "AU");
  if (codeEl) {
    codeEl.textContent = symbol
      ? `${symbol} · MARKET_DATA/${payload.feed || ""} · replay`
      : "Local 1-minute + daily archive · replay";
  }
  if (noteEl) {
    noteEl.textContent = note || `1D/1W use ${feedDailyName} · 1m/intraday use month files · no download`;
  }
  if (loadingEl) loadingEl.textContent = `Loading local ${name} candles…`;
  document.title = `${name} Replay Chart`;
  fillFeedSelect(payload);
}

async function applySourcePayload(payload) {
  sourceCandles = payload.candles;
  sourceVolumes = payload.volumes;
  const dailyRaw = payload.dailyCandles || [];
  const dailyVols = payload.dailyVolumes || [];
  dailyCandles = dailyRaw.map((bar, i) => ({
    ...bar,
    volume: dailyVols[i]?.value ?? 0,
  }));
  applyFeedMeta(payload);
  const saved = loadReplayCursor();
  if (saved?.timeframe && TF_SECONDS[saved.timeframe]) {
    applyTimeframe(saved.timeframe);
  } else {
    syncReplayStepSelect();
  }
  if (typeof saved?.step === "string") {
    setReplayStep(saved.step, { render: false });
  }
  const start = istParts(dataStartTime());
  const end = istParts(dataEndTime());
  document.getElementById("jump-date").min = start.date;
  document.getElementById("jump-date").max = end.date;
  document.getElementById("status-right").textContent =
    `${payload.timezone} · MARKET_DATA/${payload.feed || ""} · ${payload.files.join(", ")}`;
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
  chartEvents = [];
  chartEventLabels = [];
  eventsLoading = false;
  updateEventHud();
  if (eventsEnabled) loadChartEvents();
  loadPaperState();
}

async function loadSource() {
  loading.style.display = "block";
  errorBox.style.display = "none";
  try {
    const response = await fetch("/api/source");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load market data");
    await applySourcePayload(payload);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
}

async function switchDataFeed(feedId) {
  if (!feedId) return;
  loading.style.display = "block";
  errorBox.style.display = "none";
  try {
    const response = await fetch("/api/feed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: feedId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not switch data feed");
    pauseReplay();
    setEventsEnabled(false);
    await applySourcePayload(payload);
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

let overlayLeft = 0;
let overlayTop = 0;
let overlaySyncRaf = 0;
let overlaySyncFrames = 0;
const horizontalPriceLines = [];

// Match the LWC series pane, not the whole shell: axis size changes when
// candles are zoomed, and a mismatch shifts every horizontal off its price.
function plotAreaMetrics() {
  const shellRect = shell.getBoundingClientRect();
  let width = 0;
  let height = 0;
  try {
    if (typeof chart.paneSize === "function") {
      const pane = chart.paneSize();
      width = pane?.width || 0;
      height = pane?.height || 0;
    }
  } catch {
    width = 0;
    height = 0;
  }
  if (!width || !height) {
    let axisWidth = 0;
    let axisHeight = 0;
    try {
      axisWidth = chart.priceScale("right").width() || 0;
      axisHeight = chart.timeScale().height() || 0;
    } catch {
      // Scales are not measurable before the first paint.
    }
    width = Math.max(1, shellRect.width - axisWidth);
    height = Math.max(1, shellRect.height - axisHeight);
  }

  let left = 0;
  let top = 0;
  const paneCanvas = chartElement.querySelector("table tr:first-child td:first-child canvas");
  if (paneCanvas) {
    const paneRect = paneCanvas.getBoundingClientRect();
    left = paneRect.left - shellRect.left;
    top = paneRect.top - shellRect.top;
  }
  return {
    left,
    top,
    width: Math.max(1, width),
    height: Math.max(1, height),
  };
}

function plotAreaSize() {
  const { width, height } = plotAreaMetrics();
  return { width, height };
}

function applyOverlayGeometry() {
  const ratio = window.devicePixelRatio || 1;
  const { left, top, width, height } = plotAreaMetrics();
  const bitmapWidth = Math.max(1, Math.floor(width * ratio));
  const bitmapHeight = Math.max(1, Math.floor(height * ratio));

  // Reallocating the bitmap clears it, so only touch it on a real size change.
  if (canvas.width !== bitmapWidth || canvas.height !== bitmapHeight) {
    canvas.width = bitmapWidth;
    canvas.height = bitmapHeight;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  if (
    overlayWidth !== width ||
    overlayHeight !== height ||
    overlayLeft !== left ||
    overlayTop !== top
  ) {
    canvas.style.left = `${left}px`;
    canvas.style.top = `${top}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    overlayLeft = left;
    overlayTop = top;
    overlayWidth = width;
    overlayHeight = height;
  }
}

function resizeDrawingCanvas() {
  applyOverlayGeometry();
  drawOverlay();
}

// Time-scale events fire before the price scale finishes layout. Draw now so
// pans stay glued, then again on the next frames once candle height is final.
function scheduleOverlayDraw(frames = 3) {
  overlaySyncFrames = Math.max(overlaySyncFrames, frames);
  if (overlaySyncRaf) return;
  const step = () => {
    overlaySyncRaf = 0;
    applyOverlayGeometry();
    drawOverlay();
    overlaySyncFrames -= 1;
    if (overlaySyncFrames > 0) {
      overlaySyncRaf = requestAnimationFrame(step);
    }
  };
  overlaySyncRaf = requestAnimationFrame(step);
}

function onChartViewChange() {
  drawOverlay();
  scheduleOverlayDraw(4);
}

let overlayPainting = false;
const horizontalPriceLineKeys = [];

function syncHorizontalPriceLines() {
  const horizontals = drawings.filter((drawing) => drawing.type === "horizontal");
  while (horizontalPriceLines.length > horizontals.length) {
    const line = horizontalPriceLines.pop();
    horizontalPriceLineKeys.pop();
    try {
      candleSeries.removePriceLine(line);
    } catch {
      // Series may already have dropped the line.
    }
  }
  horizontals.forEach((drawing, index) => {
    const selected = drawings.indexOf(drawing) === selectedDrawing;
    const options = {
      price: drawing.price,
      color: colorWithAlpha(drawingColor(drawing), drawingOpacity(drawing)),
      lineWidth: selected ? 2 : 1,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true,
      title: "",
    };
    const key = `${options.price}|${options.color}|${options.lineWidth}`;
    if (horizontalPriceLines[index]) {
      if (horizontalPriceLineKeys[index] === key) return;
      horizontalPriceLines[index].applyOptions(options);
    } else {
      horizontalPriceLines[index] = candleSeries.createPriceLine(options);
    }
    horizontalPriceLineKeys[index] = key;
  });
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
  if (overlayPainting) return;
  overlayPainting = true;
  try {
    // Axis width/height change when candle zoom changes; keep overlay on the pane.
    const plot = plotAreaMetrics();
    if (
      Math.abs(plot.width - overlayWidth) > 0.5 ||
      Math.abs(plot.height - overlayHeight) > 0.5 ||
      Math.abs(plot.left - overlayLeft) > 0.5 ||
      Math.abs(plot.top - overlayTop) > 0.5
    ) {
      applyOverlayGeometry();
    }
    syncHorizontalPriceLines();
    ctx.clearRect(0, 0, overlayWidth, overlayHeight);
    drawings.forEach((drawing, index) => drawOne(drawing, false, index));
    if (measurement) drawOne(measurement, false, null);
    drawDraft();
    drawToolCrosshair();
    drawPaperArrows();
    drawPaperTags();
    drawEventLabelRays();
    updateSelectionUi();
  } finally {
    overlayPainting = false;
  }
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
  if (!drawing) {
    saveDrawings();
    drawOverlay();
    return;
  }

  const nextText = cancel ? editOriginalText : textEditor.value.trim();
  const willDelete = drawing.type === "text" && !nextText;
  if (cancel && willDelete && !editOriginalText) {
    drawings.splice(index, 1);
    selectedDrawing = null;
    if (drawingUndo.length) drawingUndo.pop();
    syncUndoButtons();
    saveDrawings();
    drawOverlay();
    return;
  }

  const changed = (drawing.text || "") !== (editOriginalText || "") || willDelete;
  if (!cancel && changed) {
    const prev = cloneDrawings();
    if (prev[index]) prev[index].text = editOriginalText;
    pushUndoSnapshot(prev);
  }

  drawing.text = nextText;
  if (willDelete) {
    drawings.splice(index, 1);
    selectedDrawing = null;
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
  snapshotDrawings();
  drawings.splice(selectedDrawing, 1);
  selectedDrawing = null;
  saveDrawings();
  drawOverlay();
}

function deleteAllDrawings() {
  if (!drawings.length) return;
  snapshotDrawings();
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
    // Committed horizontals are native price lines so zoom cannot lift them
    // off the price. Preview and text still draw here.
    if (preview) {
      ctx.moveTo(0, y);
      ctx.lineTo(overlayWidth, y);
      ctx.stroke();
      drawPriceTag(drawing.price, y, selected, color);
    }
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

  if (selected) {
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
    snapshotDrawings();
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
  snapshotDrawings();
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
    cursor: followHead
      ? "Pan/zoom · drag drawings · Ctrl+Z undo · Ctrl+Y redo"
      : "Follow off · drag to pan · Ctrl+Z undo drawing · Shift+wheel zooms price",
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
  const tagHit = hitPaperTag(event.clientX, event.clientY);
  if (tagHit) {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    paperPlacePress = null;
    if (tagHit.action === "clear") {
      clearPaperLevel(tagHit.kind, tagHit.id);
      return;
    }
    if (tagHit.action === "toggle-tp") {
      const pos = paperById(tagHit.id);
      if (pos && currentPaperLevels(pos).tp != null) clearPaperLevel("tp", tagHit.id);
      else setPaperLevel("tp", tagHit.price, tagHit.id);
      return;
    }
    if (tagHit.action === "toggle-sl") {
      const pos = paperById(tagHit.id);
      if (pos && currentPaperLevels(pos).sl != null) clearPaperLevel("sl", tagHit.id);
      else setPaperLevel("sl", tagHit.price, tagHit.id);
      return;
    }
    if (tagHit.action === "exit") {
      paperClose(tagHit.id);
      return;
    }
    if (tagHit.action === "reverse") {
      paperReverse(tagHit.id);
      return;
    }
    if (tagHit.action === "drag") {
      paperStopDrag = { kind: tagHit.kind, id: tagHit.id, pointerId: event.pointerId, moved: false };
      setChartInteraction(false);
      try { shell.setPointerCapture(event.pointerId); } catch { /* optional */ }
      setChartCursor("ns-resize");
    }
    return;
  }
  const paperHit = hitPaperStop(event.clientX, event.clientY);
  if (paperHit) {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    paperStopDrag = { kind: paperHit.kind, id: paperHit.id, pointerId: event.pointerId, moved: false };
    paperPlacePress = null;
    setChartInteraction(false);
    try { shell.setPointerCapture(event.pointerId); } catch { /* optional */ }
    setChartCursor("ns-resize");
    return;
  }
  const hadMeasurement = clearMeasurement();
  const hit = hitDrawing(event.clientX, event.clientY);
  if (!hit) {
    selectedDrawing = null;
    if (!followHead && isInsidePlotArea(event.clientX, event.clientY)) {
      const range = lockedPriceRange || visiblePriceRange();
      const height = overlayHeight || plotAreaMetrics().height || 1;
      if (range) {
        pricePan = {
          pointerId: event.pointerId,
          startY: event.clientY,
          min: range.minValue,
          max: range.maxValue,
          height,
        };
      }
    }
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
  if (pricePan && event.pointerId === pricePan.pointerId) {
    applyPricePan(event);
    return;
  }
  if (paperStopDrag && event.pointerId === paperStopDrag.pointerId) {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    paperStopDrag.moved = true;
    const rect = canvas.getBoundingClientRect();
    const price = candleSeries.coordinateToPrice(event.clientY - rect.top);
    if (price == null || !Number.isFinite(price)) return;
    setPaperLevel(paperStopDrag.kind, price, paperStopDrag.id, false);
    return;
  }
  if (paperPlacePress && event.pointerId === paperPlacePress.pointerId) {
    const dist = Math.hypot(event.clientX - paperPlacePress.x, event.clientY - paperPlacePress.y);
    if (dist > 6) paperPlacePress = null;
  }
  if (!dragState) {
    if (!isInsidePlotArea(event.clientX, event.clientY)) {
      setChartCursor("default");
      return;
    }
    if (hitPaperTag(event.clientX, event.clientY)?.action === "drag" || hitPaperStop(event.clientX, event.clientY)) {
      setChartCursor("ns-resize");
      return;
    }
    if (hitPaperTag(event.clientX, event.clientY)) {
      setChartCursor("pointer");
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
  if (pricePan && event.pointerId === pricePan.pointerId) {
    applyPricePan(event);
    pricePan = null;
    lockCurrentPriceRange();
    return;
  }
  if (!followHead && isOnPriceAxis(event.clientX, event.clientY)) {
    window.requestAnimationFrame(() => {
      if (!followHead && !pricePan) lockCurrentPriceRange();
    });
  }
  if (paperStopDrag && event.pointerId === paperStopDrag.pointerId) {
    event.preventDefault();
    event.stopPropagation();
    if (shell.hasPointerCapture(event.pointerId)) {
      shell.releasePointerCapture(event.pointerId);
    }
    const drag = paperStopDrag;
    paperStopDrag = null;
    setChartInteraction(true);
    if (drag.id != null) savePositionStops(drag.id);
    setChartCursor("crosshair");
    return;
  }
  if (paperPlacePress && event.pointerId === paperPlacePress.pointerId) {
    paperPlacePress = null;
    return;
  }
  if (!dragState || event.pointerId !== dragState.pointerId) return;
  event.preventDefault();
  event.stopPropagation();
  const moved = drawings[dragState.index];
  const original = dragState.original;
  if (moved && original && JSON.stringify(moved) !== JSON.stringify(original)) {
    const prev = cloneDrawings();
    prev[dragState.index] = original;
    pushUndoSnapshot(prev);
  }
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
colorOpacity.addEventListener("pointerdown", () => {
  if (selectedDrawing != null) snapshotDrawings();
});
colorOpacity.addEventListener("input", (event) => {
  setSelectedColor(colorInput.value, Number(event.target.value) / 100, { history: false });
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
  setFollowHead(false, { snap: false });
  lockedPriceRange = null;
  followProgrammatic += 1;
  try {
    chart.priceScale("right").applyOptions({ autoScale: true });
    chart.timeScale().fitContent();
  } finally {
    window.setTimeout(() => {
      followProgrammatic = Math.max(0, followProgrammatic - 1);
      if (!followHead) {
        lockCurrentPriceRange();
        refreshLockedPriceScale();
      }
    }, 0);
  }
  drawOverlay();
});
document.querySelectorAll("[data-follow]").forEach((button) => {
  button.addEventListener("click", () => setFollowHead(!followHead));
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
document.getElementById("replay-step").addEventListener("change", (event) => {
  setReplayStep(event.target.value);
});
syncReplayStepSelect();
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
document.getElementById("events-toggle").addEventListener("click", () => {
  if (eventsLoading && eventsEnabled) return;
  setEventsEnabled(!eventsEnabled);
});
document.getElementById("event-prev").addEventListener("click", () => {
  pauseReplay();
  jumpPrevEvent();
});
document.getElementById("event-next").addEventListener("click", () => {
  pauseReplay();
  jumpNextEvent();
});
document.getElementById("event-labels-toggle").addEventListener("click", () => {
  if (!eventsEnabled) return;
  setEventLabelsEnabled(!eventLabelsOn);
});
document.getElementById("event-label-count").addEventListener("input", (event) => {
  setEventLabelCount(event.target.value);
});
document.getElementById("event-label-count-num").addEventListener("change", (event) => {
  setEventLabelCount(event.target.value);
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
document.getElementById("paper-submit").addEventListener("click", paperSubmit);
document.getElementById("paper-close").addEventListener("click", () => paperClose());
document.getElementById("paper-book-save").addEventListener("click", paperSaveBook);
document.getElementById("ticket-long").addEventListener("click", () => setPaperSide("long"));
document.getElementById("ticket-short").addEventListener("click", () => setPaperSide("short"));
document.getElementById("paper-tpsl-toggle").addEventListener("click", () => togglePaperTpsl());
document.getElementById("paper-size").addEventListener("input", () => {
  updateTicketPreview();
  const submit = paperField("paper-submit");
  if (submit) submit.disabled = Number(paperField("paper-size").value) <= 0;
});
document.getElementById("paper-size-usdt").addEventListener("input", () => {
  const cash = Number(paperState?.cash ?? 0);
  const usdt = Number(paperField("paper-size-usdt").value);
  if (cash > 0 && Number.isFinite(usdt)) {
    paperField("paper-size").value = String(Math.max(0, Math.min(100, (usdt / cash) * 100)));
  }
  updateTicketPreview();
});
["paper-leverage", "paper-slip", "paper-taker", "paper-capital"].forEach((id) => {
  document.getElementById(id).addEventListener("input", updateTicketPreview);
});
document.getElementById("paper-arrows-toggle").addEventListener("click", () => {
  setPaperArrowsEnabled(!paperArrowsOn);
});
document.getElementById("paper-arrow-size").addEventListener("input", (event) => {
  setPaperArrowSize(event.target.value);
});
document.getElementById("paper-arrow-size-num").addEventListener("change", (event) => {
  setPaperArrowSize(event.target.value);
});
setPaperArrowsEnabled(paperArrowsOn);
setPaperArrowSize(paperArrowSize);
["paper-tp", "paper-sl"].forEach((id) => {
  const el = document.getElementById(id);
  el.addEventListener("input", () => {
    syncPtsFromPrices();
  });
});
document.getElementById("paper-tp-pts").addEventListener("input", () => {
  applyPtsToPrice("tp");
});
document.getElementById("paper-sl-pts").addEventListener("input", () => {
  applyPtsToPrice("sl");
});
document.getElementById("paper-tp-clear").addEventListener("click", () => clearPaperLevel("tp"));
document.getElementById("paper-sl-clear").addEventListener("click", () => clearPaperLevel("sl"));

chart.subscribeCrosshairMove((param) => {
  let bar = param.seriesData.get(candleSeries);
  if (bar && param.time != null) {
    const time = typeof param.time === "number" ? param.time : Number(param.time);
    bar = { open: bar.open, high: bar.high, low: bar.low, close: bar.close, time };
  }
  updateOhlc(bar || currentCandles[currentCandles.length - 1]);
});
chart.timeScale().subscribeVisibleLogicalRangeChange(onChartViewChange);
chart.timeScale().subscribeVisibleTimeRangeChange(onChartViewChange);
new ResizeObserver(resizeDrawingCanvas).observe(shell);
shell.addEventListener("wheel", (event) => {
  scheduleOverlayDraw(6);
  if (followHead) return;
  if (isOnPriceAxis(event.clientX, event.clientY)) {
    window.setTimeout(() => {
      if (!followHead) lockCurrentPriceRange();
    }, 0);
    return;
  }
  if (!isInsidePlotArea(event.clientX, event.clientY)) return;
  if (event.shiftKey || event.altKey) {
    event.preventDefault();
    zoomLockedPrice(event);
  }
}, { passive: false });
chartElement.addEventListener("pointermove", (event) => {
  if (event.buttons) scheduleOverlayDraw(2);
  if (!followHead && event.buttons && isOnPriceAxis(event.clientX, event.clientY)) {
    lockCurrentPriceRange();
  }
}, { passive: true });

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
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redoDrawing();
    else undoDrawing();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault();
    redoDrawing();
    return;
  }
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
    if (paperPlaceKind) {
      idlePaperPlaceHint();
      drawOverlay();
      return;
    }
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

document.getElementById("undo-drawing")?.addEventListener("click", undoDrawing);
document.getElementById("redo-drawing")?.addEventListener("click", redoDrawing);
syncUndoButtons();
setFollowHead(followHead, { snap: false });
syncEventLabelControls();
document.getElementById("data-feed").addEventListener("change", (event) => {
  switchDataFeed(event.target.value);
});
loadSource();
