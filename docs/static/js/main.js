// Renders the project page from static/data/paper.json. No build step, no dependencies.

const ICONS = {
  paper: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
  code: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>',
  bibtex: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>',
  moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
  sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>',
};

// Simplified flags (inline SVG: emoji flags do not render on Windows).
const FLAGS = {
  en: '<svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" fill="#B22234"/><rect y="1.538" width="30" height="1.538" fill="#fff"/><rect y="4.615" width="30" height="1.538" fill="#fff"/><rect y="7.692" width="30" height="1.538" fill="#fff"/><rect y="10.769" width="30" height="1.538" fill="#fff"/><rect y="13.846" width="30" height="1.538" fill="#fff"/><rect y="16.923" width="30" height="1.538" fill="#fff"/><rect width="12" height="10.769" fill="#3C3B6E"/><g fill="#fff"><circle cx="1.4" cy="1.4" r=".45"/><circle cx="3.7" cy="1.4" r=".45"/><circle cx="6" cy="1.4" r=".45"/><circle cx="8.3" cy="1.4" r=".45"/><circle cx="10.6" cy="1.4" r=".45"/><circle cx="1.4" cy="4" r=".45"/><circle cx="3.7" cy="4" r=".45"/><circle cx="6" cy="4" r=".45"/><circle cx="8.3" cy="4" r=".45"/><circle cx="10.6" cy="4" r=".45"/><circle cx="1.4" cy="6.6" r=".45"/><circle cx="3.7" cy="6.6" r=".45"/><circle cx="6" cy="6.6" r=".45"/><circle cx="8.3" cy="6.6" r=".45"/><circle cx="10.6" cy="6.6" r=".45"/><circle cx="1.4" cy="9.2" r=".45"/><circle cx="3.7" cy="9.2" r=".45"/><circle cx="6" cy="9.2" r=".45"/><circle cx="8.3" cy="9.2" r=".45"/><circle cx="10.6" cy="9.2" r=".45"/></g></svg>',
  pt: '<svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" fill="#009C3B"/><path d="M15 2.4 27.6 10 15 17.6 2.4 10z" fill="#FFDF00"/><circle cx="15" cy="10" r="4.9" fill="#002776"/><path d="M10.3 8.9c3.1-.9 6.6-.3 9.4 1.9" stroke="#fff" stroke-width=".9" fill="none"/></svg>',
  es: '<svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" fill="#AA151B"/><rect y="5" width="30" height="10" fill="#F1BF00"/></svg>',
  zh: '<svg viewBox="0 0 30 20" aria-hidden="true"><rect width="30" height="20" fill="#EE1C25"/><g fill="#FFDE00"><polygon points="5,2 5.67,4.07 7.85,4.07 6.09,5.35 6.76,7.43 5,6.15 3.24,7.43 3.91,5.35 2.15,4.07 4.33,4.07"/><polygon points="10,1 10.22,1.69 10.95,1.69 10.36,2.12 10.59,2.81 10,2.38 9.41,2.81 9.64,2.12 9.05,1.69 9.78,1.69"/><polygon points="12,3 12.22,3.69 12.95,3.69 12.36,4.12 12.59,4.81 12,4.38 11.41,4.81 11.64,4.12 11.05,3.69 11.78,3.69"/><polygon points="12,6 12.22,6.69 12.95,6.69 12.36,7.12 12.59,7.81 12,7.38 11.41,7.81 11.64,7.12 11.05,6.69 11.78,6.69"/><polygon points="10,8 10.22,8.69 10.95,8.69 10.36,9.12 10.59,9.81 10,9.38 9.41,9.81 9.64,9.12 9.05,8.69 9.78,8.69"/></g></svg>',
};

// Page chrome in each language; content lives in paper.json.
const LANGS = {
  en: {
    name: "English", htmlLang: "en", decimal: ".",
    abstract: "Abstract", paper: "Paper", paperSoon: "Paper (coming soon)", paperSoonTitle: "Link to the publication coming soon",
    code: "Code", copy: "Copy", copied: "Copied", nav: "Paper, code and citation", languages: "Language",
    theme: "Switch between light and dark mode", pck: "PCK@0.2", gap: "gap", pp: "pp",
    accepted: (v) => `Accepted at ${v}`, published: (v) => `Published at ${v}`,
  },
  pt: {
    name: "Português", htmlLang: "pt-BR", decimal: ",",
    abstract: "Resumo", paper: "Artigo", paperSoon: "Artigo (em breve)", paperSoonTitle: "Link para a publicação em breve",
    code: "Código", copy: "Copiar", copied: "Copiado", nav: "Artigo, código e citação", languages: "Idioma",
    theme: "Alternar entre modo claro e escuro", pck: "PCK@0,2", gap: "gap", pp: "pp",
    accepted: (v) => `Aceito no ${v}`, published: (v) => `Publicado no ${v}`,
  },
  es: {
    name: "Español", htmlLang: "es", decimal: ",",
    abstract: "Resumen", paper: "Artículo", paperSoon: "Artículo (próximamente)", paperSoonTitle: "Enlace a la publicación próximamente",
    code: "Código", copy: "Copiar", copied: "Copiado", nav: "Artículo, código y cita", languages: "Idioma",
    theme: "Cambiar entre modo claro y oscuro", pck: "PCK@0,2", gap: "brecha", pp: "p. p.",
    accepted: (v) => `Aceptado en ${v}`, published: (v) => `Publicado en ${v}`,
  },
  zh: {
    name: "中文", htmlLang: "zh-CN", decimal: ".",
    abstract: "摘要", paper: "论文", paperSoon: "论文（即将发布）", paperSoonTitle: "出版物链接即将发布",
    code: "代码", copy: "复制", copied: "已复制", nav: "论文、代码与引用", languages: "语言",
    theme: "切换浅色/深色模式", pck: "PCK@0.2", gap: "差距", pp: "个百分点",
    accepted: (v) => `已被 ${v} 接收`, published: (v) => `发表于 ${v}`,
  },
};

const state = { data: null, lang: "en" };
let videoObserver = null;

const ui = () => LANGS[state.lang];

// Picks the current language from a translated field ({en, pt, es, zh}); anything else passes through.
function t(value) {
  if (value && typeof value === "object" && !Array.isArray(value) && "en" in value) return value[state.lang] ?? value.en;
  return value;
}

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
  value = t(value) ?? "";
  node.innerHTML = value;
  node.classList.toggle("todo", /^\s*TODO/.test(value));
}

function fmt(value, decimals) {
  value = t(value);
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  const text = decimals === undefined ? String(value) : value.toFixed(decimals);
  return text.replace(".", ui().decimal);
}

function paragraphs(texts) {
  return (t(texts) || []).map((text) => el("p", { html: text }));
}

// ---------- language and theme ----------

function initialLang() {
  const fromUrl = new URLSearchParams(location.search).get("lang");
  if (fromUrl in LANGS) return fromUrl;
  try {
    const saved = localStorage.getItem("lang");
    if (saved in LANGS) return saved;
  } catch (e) {}
  return "en";
}

function setLang(code) {
  if (!(code in LANGS) || code === state.lang) return;
  state.lang = code;
  try {
    localStorage.setItem("lang", code);
  } catch (e) {}
  // Keep a shareable URL: ?lang=xx for every language but the default.
  const url = new URL(location.href);
  if (code === "en") url.searchParams.delete("lang");
  else url.searchParams.set("lang", code);
  history.replaceState(null, "", url);
  render();
}

function renderLangSwitch() {
  const group = document.getElementById("langs");
  group.setAttribute("aria-label", ui().languages);
  group.replaceChildren(
    ...Object.entries(LANGS).map(([code, lang]) => {
      const button = el("button", {
        class: "flag",
        type: "button",
        lang: lang.htmlLang,
        title: lang.name,
        "aria-label": lang.name,
        "aria-pressed": String(code === state.lang),
      });
      button.innerHTML = FLAGS[code];
      button.addEventListener("click", () => setLang(code));
      return button;
    }),
  );
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

// ---------- sections ----------

function renderIntro(data) {
  document.title = data.title;
  setHTML(document.getElementById("title"), data.title);

  const authors = document.getElementById("authors");
  authors.replaceChildren();
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
  const line = { accepted: ui().accepted, published: ui().published }[data.status];
  status.hidden = !(line && data.venue);
  if (!status.hidden) {
    status.textContent = line(data.venue.short);
    status.title = [data.venue.full, data.venue.where, t(data.venue.when)].filter(Boolean).join(" · ");
  }

  const block = document.getElementById("bibtex");
  const buttons = document.getElementById("buttons");
  buttons.setAttribute("aria-label", ui().nav);
  const paper = data.links.paper
    ? el("a", { class: "btn", href: data.links.paper, target: "_blank", rel: "noopener" })
    : el("span", { class: "btn is-disabled", role: "link", "aria-disabled": "true", title: ui().paperSoonTitle });
  paper.innerHTML = ICONS.paper + (data.links.paper ? ui().paper : ui().paperSoon);

  const code = el("a", { class: "btn", href: data.links.code, target: "_blank", rel: "noopener" });
  code.innerHTML = ICONS.code + ui().code;

  const bib = el("button", { class: "btn", type: "button", "aria-expanded": String(!block.hidden), "aria-controls": "bibtex" });
  bib.innerHTML = ICONS.bibtex + "BibTeX";
  bib.addEventListener("click", () => {
    block.hidden = !block.hidden;
    bib.setAttribute("aria-expanded", String(!block.hidden));
  });
  buttons.replaceChildren(paper, code, bib);

  document.getElementById("bibtex-code").textContent = data.bibtex;
  setHTML(document.getElementById("bibtex-note"), data.bibtexNote);
  document.getElementById("bibtex-copy").textContent = ui().copy;
}

function setupCopy() {
  const copy = document.getElementById("bibtex-copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(state.data.bibtex);
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
    copy.textContent = ui().copied;
    setTimeout(() => (copy.textContent = ui().copy), 1600);
  });
}

function renderHero(data) {
  const h = data.hero;
  document.getElementById("hero").replaceChildren(
    el("img", { src: h.src, width: h.width, height: h.height, alt: t(h.alt), fetchpriority: "high", decoding: "async" }),
    el("figcaption", { html: h.caption }),
  );
  setHTML(document.getElementById("tldr"), data.tldr);
  document.getElementById("highlights").replaceChildren(
    ...data.highlights.map((item) => el("li", {}, el("span", { class: "value", html: item.value }), el("span", { class: "label", html: item.label }))),
  );
  document.getElementById("abstract-title").textContent = ui().abstract;
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
        const text = fmt(value, col.decimals);
        const cell = el(i === 0 ? "th" : "td", { scope: i === 0 ? "row" : null, class: cls(col) }, text);
        if (/^\s*TODO/.test(text)) cell.classList.add("todo");
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
  section.replaceChildren(el("h2", { id: `${id}-title`, html: spec.title }), ...paragraphs(spec.text));
  return section;
}

function renderPipeline(spec) {
  const section = sectionShell("pipeline", spec);
  const f = spec.figure;
  section.append(
    el("figure", { class: "figure" }, el("img", { src: f.src, width: f.width, height: f.height, alt: t(f.alt), loading: "lazy", decoding: "async" })),
  );
}

function renderDetection(spec) {
  sectionShell("detection", spec).append(...spec.tables.map(table));
}

function matrixCell(c) {
  const meta = el(
    "div",
    { class: "meta" },
    el("span", {}, `PDJ ${fmt(c.pdj, 1)}`),
    el("span", {}, `OKS ${fmt(c.oks, 1)}`),
    el("span", {}, `MPJPE ${fmt(c.mpjpe, 2)} px`),
    c.gap === null || c.gap === undefined ? null : el("span", {}, `${ui().gap} ${fmt(c.gap, 1)} ${ui().pp}`),
  );
  return el("div", { class: c.best ? "cell best" : "cell" }, el("div", { class: "id" }, c.id), el("div", { class: "pck" }, fmt(c.pck, 1)), meta);
}

function renderAdaptation(spec) {
  const section = sectionShell("adaptation", spec);
  const m = spec.matrix;
  const grid = el("div", { class: "matrix", role: "group", "aria-label": `${t(m.rowHeader)} × ${t(m.colHeader)}, ${ui().pck}` });
  // Each axis name appears once: the column axis spans both columns, the row axis sits in the corner.
  grid.append(
    el("div"),
    el("div", { class: "axis span" }, el("small", {}, t(m.colHeader))),
    el("div", { class: "axis corner" }, el("small", {}, t(m.rowHeader))),
    ...m.cols.map((c) => el("div", { class: "axis col" }, t(c))),
  );
  m.rows.forEach((r, i) => grid.append(el("div", { class: "axis row" }, t(r)), ...m.cells[i].map(matrixCell)));

  const ref = m.reference;
  section.append(
    el(
      "figure",
      { class: "table-figure" },
      grid,
      el("p", { class: "reference" }, `${t(ref.label)}: ${fmt(ref.pck, 1)} · PDJ ${fmt(ref.pdj, 1)} · OKS ${fmt(ref.oks, 1)} · MPJPE ${fmt(ref.mpjpe, 2)} px`),
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
      "aria-label": t(item.alt),
    });
    video.muted = true;
    for (const src of item.sources) video.append(el("source", { src, type: src.endsWith(".webm") ? "video/webm" : "video/mp4" }));
    return video;
  }
  return el("img", { src: item.src, width: item.width, height: item.height, alt: t(item.alt), loading: "lazy", decoding: "async" });
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
  videoObserver = new IntersectionObserver(
    (entries) => {
      for (const { target, isIntersecting } of entries) {
        if (isIntersecting) target.play().catch(() => {});
        else target.pause();
      }
    },
    { rootMargin: "200px 0px" },
  );
  videos.forEach((v) => videoObserver.observe(v));
}

function renderFooter(f) {
  document.getElementById("footer").replaceChildren(...[f.acknowledgements, f.license, f.footage].filter(Boolean).map((text) => el("p", { html: text })));
}

function render() {
  const data = state.data;
  document.documentElement.lang = ui().htmlLang;
  document.getElementById("theme-toggle").setAttribute("aria-label", ui().theme);
  if (videoObserver) videoObserver.disconnect();
  renderLangSwitch();
  renderIntro(data);
  renderHero(data);
  renderPipeline(data.pipeline);
  renderDetection(data.detection);
  renderAdaptation(data.adaptation);
  renderGallery(data.gallery);
  renderFooter(data.footer);
}

async function main() {
  state.lang = initialLang();
  setupThemeToggle();
  setupCopy();
  try {
    const response = await fetch("static/data/paper.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
  } catch (error) {
    console.error("Could not load paper.json:", error);
    document.getElementById("load-error").hidden = false;
    return;
  }
  render();
}

main();
