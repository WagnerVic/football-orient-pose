// Renders the project page from static/data/paper.json. No build step, no dependencies.

const ICONS = {
  paper: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
  code: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>',
  bibtex: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>',
};

const STATUS_LINE = { accepted: "Accepted at", published: "Published at" };

// Small DOM helper: el("p", { class: "x" }, child, "text", ...)
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "html") setHTML(node, value);
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Trusted inline HTML from paper.json; anything that starts with "TODO" is highlighted.
function setHTML(node, value) {
  node.innerHTML = value;
  if (/^\s*TODO/.test(value)) node.classList.add("todo");
}

function fmt(value, decimals) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  return decimals === undefined ? String(value) : value.toFixed(decimals);
}

function paragraphs(texts) {
  return (texts || []).map((t) => el("p", { html: t }));
}

function renderIntro(data) {
  document.title = data.title;
  setHTML(document.getElementById("title"), data.title);

  const authors = document.getElementById("authors");
  data.authors.forEach((author, i) => {
    if (i > 0) authors.append(el("span", { class: "sep", "aria-hidden": "true" }, "·"));
    authors.append(
      author.url
        ? el("a", { class: "author", href: author.url, target: "_blank", rel: "noopener" }, author.name)
        : el("span", { class: "author" }, author.name),
    );
  });
  setHTML(document.getElementById("affiliation"), data.affiliation);

  const status = document.getElementById("status");
  if (STATUS_LINE[data.status] && data.venue) {
    status.textContent = `${STATUS_LINE[data.status]} ${data.venue.short}`;
    status.title = [data.venue.full, data.venue.where, data.venue.when].filter(Boolean).join(" · ");
    status.hidden = false;
  }

  const buttons = document.getElementById("buttons");
  const paper = data.links.paper
    ? el("a", { class: "btn", href: data.links.paper, target: "_blank", rel: "noopener" })
    : el("span", { class: "btn is-disabled", role: "link", "aria-disabled": "true", title: "Link to the publication coming soon" });
  paper.innerHTML = ICONS.paper + (data.links.paper ? "Paper" : "Paper (coming soon)");

  const code = el("a", { class: "btn", href: data.links.code, target: "_blank", rel: "noopener" });
  code.innerHTML = ICONS.code + "Code";

  const bib = el("button", { class: "btn", type: "button", "aria-expanded": "false", "aria-controls": "bibtex" });
  bib.innerHTML = ICONS.bibtex + "BibTeX";
  buttons.append(paper, code, bib);

  const block = document.getElementById("bibtex");
  document.getElementById("bibtex-code").textContent = data.bibtex;
  setHTML(document.getElementById("bibtex-note"), data.bibtexNote || "");
  bib.addEventListener("click", () => {
    block.hidden = !block.hidden;
    bib.setAttribute("aria-expanded", String(!block.hidden));
  });

  const copy = document.getElementById("bibtex-copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(data.bibtex);
    } catch (e) {
      // Fallback for browsers without the async clipboard API.
      const range = document.createRange();
      range.selectNodeContents(document.getElementById("bibtex-code"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand("copy");
      selection.removeAllRanges();
    }
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 1600);
  });
}

function renderHero(data) {
  const h = data.hero;
  const hero = document.getElementById("hero");
  hero.append(
    el("img", { src: h.src, width: h.width, height: h.height, alt: h.alt, fetchpriority: "high", decoding: "async" }),
    el("figcaption", { html: h.caption }),
  );
  setHTML(document.getElementById("tldr"), data.tldr);

  const list = document.getElementById("highlights");
  for (const item of data.highlights) {
    list.append(el("li", {}, el("span", { class: "value", html: item.value }), el("span", { class: "label", html: item.label })));
  }
  setHTML(document.getElementById("abstract"), data.abstract);
}

function table(spec) {
  const cls = (col) => [col.optional ? "optional" : "", col.best ? "best-col" : ""].join(" ").trim() || null;
  const head = el("tr", {}, ...spec.columns.map((col) => el("th", { scope: "col", class: cls(col), html: col.label })));
  const body = spec.rows.map((row) =>
    el(
      "tr",
      { class: row.best ? "best" : null },
      ...row.cells.map((value, i) => {
        const col = spec.columns[i];
        const cell = el(i === 0 ? "th" : "td", { scope: i === 0 ? "row" : null, class: cls(col) }, fmt(value, col.decimals));
        if (typeof value === "string" && /^\s*TODO/.test(value)) cell.classList.add("todo");
        return cell;
      }),
    ),
  );
  return el(
    "figure",
    { class: "table-figure" },
    el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body))),
    spec.caption ? el("figcaption", { html: spec.caption }) : null,
  );
}

function sectionShell(id, spec) {
  const section = document.getElementById(id);
  section.setAttribute("aria-labelledby", `${id}-title`);
  section.append(el("h2", { id: `${id}-title`, html: spec.title }), ...paragraphs(spec.text));
  return section;
}

function renderPipeline(spec) {
  const section = sectionShell("pipeline", spec);
  const f = spec.figure;
  section.append(
    el("figure", { class: "figure" }, el("img", { src: f.src, width: f.width, height: f.height, alt: f.alt, loading: "lazy", decoding: "async" })),
  );
}

function renderDetection(spec) {
  const section = sectionShell("detection", spec);
  section.append(...spec.tables.map(table));
}

function matrixCell(c) {
  const meta = el(
    "div",
    { class: "meta" },
    el("span", {}, `PDJ ${fmt(c.pdj, 1)}`),
    el("span", {}, `OKS ${fmt(c.oks, 1)}`),
    el("span", {}, `MPJPE ${fmt(c.mpjpe, 2)} px`),
    c.gap === null || c.gap === undefined ? null : el("span", {}, `gap ${fmt(c.gap, 1)} pp`),
  );
  return el("div", { class: c.best ? "cell best" : "cell" }, el("div", { class: "id" }, c.id), el("div", { class: "pck" }, fmt(c.pck, 1)), meta);
}

function renderAdaptation(spec) {
  const section = sectionShell("adaptation", spec);
  const m = spec.matrix;
  const grid = el("div", { class: "matrix", role: "group", "aria-label": `${m.rowHeader} × ${m.colHeader}, PCK@0.2` });
  // Each axis name appears once: the column axis spans both columns, the row axis sits in the corner.
  grid.append(
    el("div"),
    el("div", { class: "axis span" }, el("small", {}, m.colHeader)),
    el("div", { class: "axis corner" }, el("small", {}, m.rowHeader)),
    ...m.cols.map((c) => el("div", { class: "axis col" }, c)),
  );
  m.rows.forEach((r, i) => grid.append(el("div", { class: "axis row" }, r), ...m.cells[i].map(matrixCell)));

  const ref = m.reference;
  section.append(
    el(
      "figure",
      { class: "table-figure" },
      grid,
      el("p", { class: "reference" }, `${ref.label}: ${fmt(ref.pck, 1)} · PDJ ${fmt(ref.pdj, 1)} · OKS ${fmt(ref.oks, 1)} · MPJPE ${fmt(ref.mpjpe, 2)} px`),
      el("figcaption", { html: m.caption }),
    ),
  );

  const d = spec.details;
  if (d) section.append(el("details", { class: "more" }, el("summary", { html: d.summary }), table(d)));
}

function media(item) {
  if (item.type === "video") {
    // Played only while visible (see watchVideos); the poster shows meanwhile.
    const video = el("video", {
      muted: true,
      loop: true,
      playsinline: true,
      preload: "none",
      poster: item.poster,
      width: item.width,
      height: item.height,
      "aria-label": item.alt,
    });
    video.muted = true;
    for (const src of item.sources) video.append(el("source", { src, type: src.endsWith(".webm") ? "video/webm" : "video/mp4" }));
    return video;
  }
  return el("img", { src: item.src, width: item.width, height: item.height, alt: item.alt, loading: "lazy", decoding: "async" });
}

function renderGallery(spec) {
  const section = sectionShell("gallery", spec);
  for (const group of spec.groups) {
    section.append(
      el(
        "figure",
        { class: `gallery-group ${group.layout}` },
        el("div", { class: "grid" }, ...group.items.map(media)),
        group.caption ? el("figcaption", { html: group.caption }) : null,
      ),
    );
  }
  watchVideos(section.querySelectorAll("video"));
}

function watchVideos(videos) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) {
    videos.forEach((v) => {
      v.controls = reduceMotion;
      if (!reduceMotion) v.autoplay = true;
    });
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      for (const { target, isIntersecting } of entries) {
        if (isIntersecting) target.play().catch(() => {});
        else target.pause();
      }
    },
    { rootMargin: "200px 0px" },
  );
  videos.forEach((v) => observer.observe(v));
}

function renderFooter(f) {
  document.getElementById("footer").append(...[f.acknowledgements, f.license, f.footage].filter(Boolean).map((t) => el("p", { html: t })));
}

function setupThemeToggle() {
  const button = document.getElementById("theme-toggle");
  const root = document.documentElement;
  const current = () => root.getAttribute("data-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const paint = () => {
    button.innerHTML = current() === "dark" ? ICONS.sun : ICONS.moon;
  };
  button.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {}
    paint();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
  paint();
}

async function main() {
  setupThemeToggle();
  let data;
  try {
    const response = await fetch("static/data/paper.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    console.error("Could not load paper.json:", error);
    document.getElementById("load-error").hidden = false;
    return;
  }
  renderIntro(data);
  renderHero(data);
  renderPipeline(data.pipeline);
  renderDetection(data.detection);
  renderAdaptation(data.adaptation);
  renderGallery(data.gallery);
  renderFooter(data.footer);
}

main();
