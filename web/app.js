/**
 * Mesa Self-Construction Interactive Dashboard
 * Connects to FastAPI backend running Mesa 3.5.1 model.
 * Handles high-performance Canvas rendering, pan/zoom, glider vector overlays,
 * proof-of-construction telemetry, and population charts.
 */

// Von Neumann 29-state names, mirroring von_neumann.STATE_NAMES.
const VN_STATE_NAMES = {
  0: "U blank",
  1: "T_O E", 2: "T_O N", 3: "T_O W", 4: "T_O S",
  5: "T_O* E", 6: "T_O* N", 7: "T_O* W", 8: "T_O* S",
  9: "T_S E", 10: "T_S N", 11: "T_S W", 12: "T_S S",
  13: "T_S* E", 14: "T_S* N", 15: "T_S* W", 16: "T_S* S",
  17: "C confluent", 18: "C* firing", 19: "C1 latched", 20: "C1* latched+firing",
  21: "V_O pend E", 22: "V_O pend N", 23: "V_O pend W", 24: "V_O pend S",
  25: "V_S pend E", 26: "V_S pend N", 27: "V_S pend W", 28: "V_S pend S",
};

class SelfConstructionApp {
  constructor() {
    // State
    this.modelState = null;
    this.isPlaying = false;
    this.fps = 15;
    this.playInterval = null;
    this.hoverCell = null;

    // Viewport transform (pan & zoom)
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    // View toggles
    this.showGrid = true;
    this.showTrajectories = true;
    this.showTargetZone = true;
    this.showGlow = true;

    // Canvas elements
    this.canvas = document.getElementById("life-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.sparklineCanvas = document.getElementById("sparkline-canvas");
    this.sparklineCtx = this.sparklineCanvas.getContext("2d");
    this.wrapper = document.getElementById("canvas-wrapper");

    // UI Elements
    this.presetSelect = document.getElementById("preset-select");
    this.btnPlayPause = document.getElementById("btn-play-pause");
    this.btnStepForward = document.getElementById("btn-step-forward");
    this.btnStepBack = document.getElementById("btn-step-back");
    this.btnReset = document.getElementById("btn-reset");
    this.btnRunProof = document.getElementById("btn-run-proof");
    this.speedSlider = document.getElementById("speed-slider");
    this.speedLabel = document.getElementById("speed-label");

    this.initEvents();
    this.resizeCanvas();
    this.fetchState();
  }

  initEvents() {
    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.render();
    });

    // Preset selector
    this.presetSelect.addEventListener("change", (e) => {
      this.loadPreset(e.target.value);
    });

    // Playback buttons
    this.btnPlayPause.addEventListener("click", () => this.togglePlay());
    this.btnStepForward.addEventListener("click", () => this.stepForward());
    this.btnStepBack.addEventListener("click", () => this.stepBack());
    this.btnReset.addEventListener("click", () => this.resetSimulation());
    this.btnRunProof.addEventListener("click", () => this.runToProof());

    // Speed slider
    this.speedSlider.addEventListener("input", (e) => {
      this.fps = parseInt(e.target.value, 10);
      this.speedLabel.textContent = `${this.fps} FPS`;
      if (this.isPlaying) {
        this.restartPlayLoop();
      }
    });

    // View toggles
    document.getElementById("toggle-grid").addEventListener("change", (e) => {
      this.showGrid = e.target.checked;
      this.render();
    });
    document.getElementById("toggle-trajectories").addEventListener("change", (e) => {
      this.showTrajectories = e.target.checked;
      this.render();
    });
    document.getElementById("toggle-target-zone").addEventListener("change", (e) => {
      this.showTargetZone = e.target.checked;
      this.render();
    });
    document.getElementById("toggle-glow").addEventListener("change", (e) => {
      this.showGlow = e.target.checked;
      this.render();
    });

    // Zoom controls
    document.getElementById("btn-zoom-in").addEventListener("click", () => this.zoom(1.25));
    document.getElementById("btn-zoom-out").addEventListener("click", () => this.zoom(0.8));
    document.getElementById("btn-zoom-reset").addEventListener("click", () => this.resetZoom());
    document.getElementById("btn-fit-target").addEventListener("click", () => this.fitToTarget());

    // Mouse drag pan
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.panX;
      this.dragStartY = e.clientY - this.panY;
    });

    window.addEventListener("mousemove", (e) => {
      if (this.isDragging) {
        this.panX = e.clientX - this.dragStartX;
        this.panY = e.clientY - this.dragStartY;
        this.render();
      }
      this.updateHoverCell(e);
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    // Mouse wheel zoom
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      this.panX = mouseX - (mouseX - this.panX) * zoomFactor;
      this.panY = mouseY - (mouseY - this.panY) * zoomFactor;
      this.scale *= zoomFactor;
      this.scale = Math.max(0.2, Math.min(this.scale, 8.0));
      this.render();
    });

    // Keyboard shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        this.togglePlay();
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        this.stepForward();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        this.stepBack();
      } else if (e.code === "KeyR") {
        e.preventDefault();
        this.resetSimulation();
      }
    });
  }

  resizeCanvas() {
    const rect = this.wrapper.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    // Reset transform before scaling: repeated resizes must not compound.
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.canvasWidth = rect.width;
    this.canvasHeight = rect.height;
  }

  async fetchState() {
    try {
      const res = await fetch("/api/state");
      if (res.ok) {
        this.modelState = await res.json();
        this.updateHUD();
        this.updateRightPanel();
        this.render();
        this.renderSparkline();
      }
    } catch (err) {
      console.warn("Could not fetch state from backend:", err);
    }
  }

  async stepForward() {
    try {
      const res = await fetch("/api/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ steps: 1 }),
      });
      if (res.ok) {
        this.modelState = await res.json();
        this.updateHUD();
        this.updateRightPanel();
        this.render();
        this.renderSparkline();
      }
    } catch (err) {
      console.warn("Step forward error:", err);
    }
  }

  async stepBack() {
    try {
      const res = await fetch("/api/step-back", { method: "POST" });
      if (res.ok) {
        this.modelState = await res.json();
        this.updateHUD();
        this.updateRightPanel();
        this.render();
        this.renderSparkline();
      }
    } catch (err) {
      console.warn("Step back error:", err);
    }
  }

  async resetSimulation() {
    this.pause();
    try {
      const res = await fetch("/api/reset", { method: "POST" });
      if (res.ok) {
        this.modelState = await res.json();
        this.updateHUD();
        this.updateRightPanel();
        this.resetZoom();
        this.render();
        this.renderSparkline();
      }
    } catch (err) {
      console.warn("Reset error:", err);
    }
  }

  async loadPreset(presetId) {
    this.pause();
    try {
      const res = await fetch(`/api/load-preset/${presetId}`, { method: "POST" });
      if (res.ok) {
        this.modelState = await res.json();
        this.updateHUD();
        this.updateRightPanel();
        this.resetZoom();
        this.render();
        this.renderSparkline();
      }
    } catch (err) {
      console.warn("Load preset error:", err);
    }
  }

  async runToProof() {
    this.pause();
    try {
      const res = await fetch("/api/run-to-validation", { method: "POST" });
      if (res.ok) {
        this.modelState = await res.json();
        this.updateHUD();
        this.updateRightPanel();
        this.render();
        this.renderSparkline();
      }
    } catch (err) {
      console.warn("Run to proof error:", err);
    }
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    this.isPlaying = true;
    document.getElementById("play-pause-label").textContent = "Pause";
    document.getElementById("icon-play").innerHTML =
      '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    this.btnPlayPause.classList.add("btn-accent");
    this.btnPlayPause.classList.remove("btn-primary");
    this.restartPlayLoop();
  }

  pause() {
    this.isPlaying = false;
    document.getElementById("play-pause-label").textContent = "Play";
    document.getElementById("icon-play").innerHTML =
      '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    this.btnPlayPause.classList.add("btn-primary");
    this.btnPlayPause.classList.remove("btn-accent");
    if (this.playInterval) {
      clearInterval(this.playInterval);
      this.playInterval = null;
    }
  }

  restartPlayLoop() {
    if (this.playInterval) clearInterval(this.playInterval);
    const intervalMs = Math.round(1000 / this.fps);
    this.playInterval = setInterval(() => {
      this.stepForward();
    }, intervalMs);
  }

  zoom(factor) {
    const centerX = this.canvasWidth / 2;
    const centerY = this.canvasHeight / 2;
    this.panX = centerX - (centerX - this.panX) * factor;
    this.panY = centerY - (centerY - this.panY) * factor;
    this.scale *= factor;
    this.scale = Math.max(0.2, Math.min(this.scale, 8.0));
    this.render();
  }

  resetZoom() {
    if (!this.modelState) return;
    const padding = 60;
    const gridCols = this.modelState.width;
    const gridRows = this.modelState.height;
    const cellWidth = (this.canvasWidth - padding * 2) / gridCols;
    const cellHeight = (this.canvasHeight - padding * 2) / gridRows;
    const cellSize = Math.floor(Math.min(cellWidth, cellHeight));

    this.baseCellSize = Math.max(8, cellSize);
    this.scale = 1.0;
    this.panX = (this.canvasWidth - gridCols * this.baseCellSize) / 2;
    this.panY = (this.canvasHeight - gridRows * this.baseCellSize) / 2;
    this.render();
  }

  fitToTarget() {
    if (!this.modelState || !this.modelState.target_box) return;
    const [rmin, rmax, cmin, cmax] = this.modelState.target_box;
    const targetW = (cmax - cmin + 3) * this.baseCellSize;
    const targetH = (rmax - rmin + 3) * this.baseCellSize;
    const scaleX = (this.canvasWidth * 0.6) / targetW;
    const scaleY = (this.canvasHeight * 0.6) / targetH;
    this.scale = Math.min(scaleX, scaleY, 5.0);

    const centerCol = (cmin + cmax) / 2;
    const centerRow = (rmin + rmax) / 2;
    this.panX = this.canvasWidth / 2 - centerCol * this.baseCellSize * this.scale;
    this.panY = this.canvasHeight / 2 - centerRow * this.baseCellSize * this.scale;
    this.render();
  }

  updateHoverCell(e) {
    if (!this.modelState) return;
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const cellSize = (this.baseCellSize || 12) * this.scale;
    const col = Math.floor((mouseX - this.panX) / cellSize);
    const row = Math.floor((mouseY - this.panY) / cellSize);
    const hudHover = document.getElementById("hud-hover");

    if (row >= 0 && row < this.modelState.height && col >= 0 && col < this.modelState.width) {
      this.hoverCell = { row, col };
      const state = this.cellStateAt(row, col);
      if (this.modelState.is_von_neumann) {
        hudHover.textContent = state > 0 ? `R:${row}, C:${col} · ${VN_STATE_NAMES[state] || state}` : `R:${row}, C:${col} · blank`;
      } else {
        hudHover.textContent = state > 0 ? `R:${row}, C:${col} · alive` : `R:${row}, C:${col} · dead`;
      }
    } else {
      this.hoverCell = null;
      hudHover.textContent = "-- , --";
    }
  }

  cellStateAt(row, col) {
    if (!this.modelState || !this.modelState.active_cells) return 0;
    // active_cells is sparse; linear scan is fine for hover (few cells).
    for (const [r, c, s] of this.modelState.active_cells) {
      if (r === row && c === col) return s;
    }
    return 0;
  }

  updateHUD() {
    if (!this.modelState) return;
    document.getElementById("hud-generation").textContent = this.modelState.step;
    document.getElementById("hud-population").textContent = this.modelState.population;
    document.getElementById("hud-stage").textContent = this.modelState.stage;

    // Experiment details
    document.getElementById("experiment-category").textContent = this.modelState.category;
    document.getElementById("experiment-title").textContent = this.modelState.preset_name;
    document.getElementById("experiment-description").textContent = this.modelState.description;

    // Pill badge
    const pill = document.getElementById("validation-pill");
    const statusText = document.getElementById("validation-status-text");
    const isPassed = this.modelState.validation && this.modelState.validation.passed;

    if (isPassed) {
      pill.className = "validation-pill status-passed";
      statusText.textContent = "Verified!";
    } else {
      pill.className = "validation-pill status-in-progress";
      statusText.textContent = `In Progress (${Math.round((this.modelState.validation?.match_ratio || 0) * 100)}%)`;
    }
  }

  updateRightPanel() {
    if (!this.modelState) return;
    const val = this.modelState.validation || {};
    const isPassed = val.passed || false;

    // Proof badge
    const badge = document.getElementById("proof-badge");
    badge.textContent = isPassed ? "VERIFIED" : "VALIDATING";
    badge.className = isPassed ? "badge-result verified" : "badge-result";

    document.getElementById("metric-match").textContent = `${Math.round((val.match_ratio || 0) * 100)}%`;
    document.getElementById("metric-settled").textContent = val.is_settled ? "Yes" : "No";
    document.getElementById("metric-gliders").textContent = val.gliders_used || this.modelState.gliders_injected || 0;
    document.getElementById("metric-target-pop").textContent = val.expected_population || "--";
    document.getElementById("proof-text").textContent = val.proof_message || "Analyzing construction...";

    // Glider Timeline
    const timelineEl = document.getElementById("glider-timeline");
    timelineEl.innerHTML = "";
    if (this.modelState.gliders && this.modelState.gliders.length > 0) {
      this.modelState.gliders.forEach((g, idx) => {
        const item = document.createElement("div");
        const isCurrent = this.modelState.step >= g.release_step && this.modelState.step < g.release_step + 40;
        item.className = `timeline-item ${isCurrent ? "active" : ""} ${g.injected ? "injected" : ""}`;
        item.innerHTML = `
          <span>${g.label}</span>
          <span class="mono-value">t=${g.release_step} [${g.direction}]</span>
        `;
        timelineEl.appendChild(item);
      });
    } else {
      timelineEl.innerHTML = `<div class="timeline-item"><span>No external gliders scheduled</span></div>`;
    }

    // Theory insight
    const theoryBadge = document.getElementById("theory-mode-badge");
    const theoryBody = document.getElementById("theory-body");
    if (this.modelState.is_von_neumann) {
      const built = this.modelState.vn_constructed ?? 0;
      if (this.modelState.preset_id === "vn_description_reader") {
        theoryBadge.textContent = "Von Neumann Description Reader";
        theoryBody.textContent =
          `Stored 3-bit tape [O, S, O] copied into built cells [1, 9, 1]: first pulse sensitizes each blank slot, second solidifies the described kind. ${built}/3 construction events so far. This read-and-build loop is the core mechanism universal construction scales up.`;
      } else {
        theoryBadge.textContent = "Von Neumann 29-State Constructor";
        theoryBody.textContent =
          `Full 29-state von Neumann system: directional ordinary/special transmission, confluent latch with one-tick delay, sensitized construct/kill. ${built} sensitized→wired construction events counted so far; the arm extends eastward as pulses arrive. Decoder programs and full self-description tapes remain out of scope.`;
      }
    } else {
      theoryBadge.textContent = "Conway Self-Construction";
      theoryBody.textContent =
        "In Conway's Life, construction is realized via slow salvos—timed glider packets colliding with an elbow block. Each collision relocates the elbow (0move) or synthesizes target seeds with zero lingering debris.";
    }
  }

  render() {
    if (!this.modelState) return;

    const ctx = this.ctx;
    const w = this.canvasWidth;
    const h = this.canvasHeight;
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.translate(this.panX, this.panY);

    const cellSize = (this.baseCellSize || 12) * this.scale;
    const cols = this.modelState.width;
    const rows = this.modelState.height;

    // 1. Grid Background & Mesh (Deep organic forest canvas)
    ctx.fillStyle = "#0A1611";
    ctx.fillRect(0, 0, cols * cellSize, rows * cellSize);

    if (this.showGrid && cellSize >= 6) {
      ctx.strokeStyle = "rgba(167, 193, 159, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let c = 0; c <= cols; c++) {
        ctx.moveTo(c * cellSize, 0);
        ctx.lineTo(c * cellSize, rows * cellSize);
      }
      for (let r = 0; r <= rows; r++) {
        ctx.moveTo(0, r * cellSize);
        ctx.lineTo(cols * cellSize, r * cellSize);
      }
      ctx.stroke();
    }

    // 2. Target Construction Zone
    if (this.showTargetZone && this.modelState.target_box) {
      const [rmin, rmax, cmin, cmax] = this.modelState.target_box;
      const tx = cmin * cellSize;
      const ty = rmin * cellSize;
      const tw = (cmax - cmin + 1) * cellSize;
      const th = (rmax - rmin + 1) * cellSize;

      ctx.fillStyle = "rgba(127, 168, 119, 0.09)";
      ctx.fillRect(tx, ty, tw, th);

      ctx.strokeStyle = "rgba(163, 201, 154, 0.65)";
      ctx.lineWidth = 1.25;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(tx, ty, tw, th);
      ctx.setLineDash([]);

      ctx.font = `500 ${Math.max(9, Math.round(10 * this.scale))}px Inter, sans-serif`;
      ctx.fillStyle = "#A3C99A";
      ctx.fillText("TARGET ZONE", tx + 4, ty - 6);
    }

    // 3. Glider Trajectories
    if (this.showTrajectories && this.modelState.gliders) {
      this.modelState.gliders.forEach((g) => {
        if (!g.consumed) {
          const gx = (g.col + 1.5) * cellSize;
          const gy = (g.row + 1.5) * cellSize;
          ctx.strokeStyle = "rgba(163, 201, 154, 0.22)";
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(gx, gy);
          // Draw diagonal trajectory
          let dx = g.direction.includes("E") ? 1 : -1;
          let dy = g.direction.includes("S") ? 1 : -1;
          ctx.lineTo(gx + dx * 220 * this.scale, gy + dy * 220 * this.scale);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    }

    // 4. Render Active Cells
    const activeCells = this.modelState.active_cells || [];
    const isVN = this.modelState.is_von_neumann;

    activeCells.forEach(([r, c, state]) => {
      const x = c * cellSize;
      const y = r * cellSize;
      const cellPad = Math.max(1, Math.round(1 * this.scale));
      const innerW = cellSize - cellPad * 2;
      const innerH = cellSize - cellPad * 2;

      let fillColor = "#A3C99A"; // Natural sage
      let glowColor = "rgba(163, 201, 154, 0.35)";

      if (isVN) {
        // Von Neumann 29-state colors (see von_neumann.py):
        // 1-4 ordinary ground, 5-8 ordinary excited,
        // 9-12 special ground, 13-16 special excited,
        // 17-20 confluent family, 21-24 sensitized-ordinary, 25-28 sensitized-special.
        if (state >= 1 && state <= 4) {
          fillColor = "#8EAF85"; // Ordinary transmission ground (quiet leaf green)
          glowColor = "rgba(142, 175, 133, 0.3)";
        } else if (state >= 5 && state <= 8) {
          fillColor = "#FFF9E6"; // Ordinary excitation pulse (warm light ivory)
          glowColor = "rgba(255, 249, 230, 0.6)";
        } else if (state >= 9 && state <= 12) {
          fillColor = "#7FB3C8"; // Special transmission ground (cool steel blue)
          glowColor = "rgba(127, 179, 200, 0.3)";
        } else if (state >= 13 && state <= 16) {
          fillColor = "#D6EDF5"; // Special excitation pulse (pale ice blue)
          glowColor = "rgba(214, 237, 245, 0.6)";
        } else if (state >= 17 && state <= 20) {
          fillColor = "#D9C38F"; // Confluent family (sand / ochre)
          glowColor = "rgba(217, 195, 143, 0.4)";
        } else if (state >= 21 && state <= 24) {
          fillColor = "#E2A654"; // Sensitized-ordinary (warm amber)
          glowColor = "rgba(226, 166, 84, 0.5)";
        } else if (state >= 25 && state <= 28) {
          fillColor = "#C77FB3"; // Sensitized-special (muted magenta)
          glowColor = "rgba(199, 127, 179, 0.5)";
        }
      } else {
        // In Conway's Game of Life
        if (this.modelState.target_box) {
          const [rmin, rmax, cmin, cmax] = this.modelState.target_box;
          if (r >= rmin && r <= rmax && c >= cmin && c <= cmax) {
            fillColor = "#F4F8F2"; // Constructed target cell (clean ivory)
            glowColor = "rgba(244, 248, 242, 0.5)";
          }
        }
      }

      if (this.showGlow && cellSize >= 8) {
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 4 * this.scale;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.fillStyle = fillColor;
      ctx.beginPath();
      const radius = Math.min(innerW / 3, 2.5);
      ctx.roundRect(x + cellPad, y + cellPad, innerW, innerH, radius);
      ctx.fill();
    });

    ctx.shadowBlur = 0;

    // 5. Construction-head marker (VN live front) + confluent junction.
    if (this.modelState.vn_head) {
      const [hr, hc] = this.modelState.vn_head;
      const hx = (hc + 0.5) * cellSize;
      const hy = (hr + 0.5) * cellSize;
      const rad = Math.max(5, cellSize * 0.9);
      ctx.strokeStyle = "#E2A654";
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      ctx.moveTo(hx, hy - rad);
      ctx.lineTo(hx + rad, hy);
      ctx.lineTo(hx, hy + rad);
      ctx.lineTo(hx - rad, hy);
      ctx.closePath();
      ctx.stroke();
      ctx.font = `600 ${Math.max(9, Math.round(10 * this.scale))}px Inter, sans-serif`;
      ctx.fillStyle = "#E2A654";
      ctx.fillText("BUILD FRONT", hx + rad + 4, hy + 3);
    }
    const junction = this.modelState.metadata && this.modelState.metadata.confluent_junction;
    if (junction && this.modelState.is_von_neumann) {
      const [jr, jc] = junction;
      const jx = (jc + 0.5) * cellSize;
      const jy = (jr + 0.5) * cellSize;
      ctx.strokeStyle = "rgba(217, 195, 143, 0.9)";
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(jx, jy, Math.max(4, cellSize * 0.7), 0, Math.PI * 2);
      ctx.stroke();
      ctx.font = `500 ${Math.max(9, Math.round(10 * this.scale))}px Inter, sans-serif`;
      ctx.fillStyle = "#D9C38F";
      ctx.fillText("CONFLUENT", jx + 8, jy - 8);
    }

    // 6. Hover indicator
    if (this.hoverCell) {
      const hx = this.hoverCell.col * cellSize;
      const hy = this.hoverCell.row * cellSize;
      ctx.strokeStyle = "rgba(244, 248, 242, 0.75)";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(hx, hy, cellSize, cellSize);
    }

    ctx.restore();
  }

  renderSparkline() {
    if (!this.modelState || !this.modelState.population_history) return;
    const history = this.modelState.population_history;
    if (history.length < 2) return;

    const ctx = this.sparklineCtx;
    const w = this.sparklineCanvas.width;
    const h = this.sparklineCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const min = Math.min(...history);
    const max = Math.max(...history, min + 1);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = "#A3C99A";
    ctx.lineWidth = 1.75;

    history.forEach((val, i) => {
      const x = (i / (history.length - 1)) * (w - 12) + 6;
      const y = h - ((val - min) / range) * (h - 16) - 8;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill gradient
    ctx.lineTo(w - 6, h);
    ctx.lineTo(6, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(163, 201, 154, 0.2)");
    grad.addColorStop(1, "rgba(163, 201, 154, 0.0)");
    ctx.fillStyle = grad;
    ctx.fill();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new SelfConstructionApp();
});
