# Project page

Static project page of the paper *A Quantitative Pipeline for Soccer Player Pose Estimation in
Broadcast Video* (WVC 2026), served by GitHub Pages at
**https://wagnervic.github.io/football-orient-pose/**.

Plain HTML + CSS + a small script, no build step:

```
docs/
  index.html              page skeleton + Open Graph / Twitter tags
  .nojekyll               serve files as-is (no Jekyll)
  assets/                 hero.gif and pipeline_stages.png (shared with the main README)
  static/data/paper.json  ALL text, links, numbers, captions and the BibTeX
  static/css/style.css    layout, light and dark themes
  static/js/main.js       renders paper.json into the page
  static/images/          showcase GIFs, video posters, og-preview.jpg
  static/videos/          showcase clips over ~3 MB, re-encoded as WebM + MP4
```

The paper PDF is **not** part of the page: `docs/**/*.pdf` is git-ignored, and Pages only serves
committed files. Do not `git add -f` it. The Paper button links to the publication page instead.

## Enable GitHub Pages (once)

Repository **Settings → Pages → Build and deployment**: Source *Deploy from a branch*, branch
`main`, folder `/docs`, **Save**. The page goes live a minute or two after the branch with
`docs/index.html` is merged into `main`.

## Test locally

The page loads `static/data/paper.json` with `fetch`, which browsers block on `file://`. Serve
the folder over HTTP:

```bash
cd docs
python -m http.server 8000
# open http://localhost:8000
```

## Update for the camera-ready

1. Edit `static/data/paper.json`: `abstract`, and `title`/`authors` if they changed.
2. If a number changed, recompute it from `results/tables/*.json` (×100, one decimal) and update it
   everywhere it appears in `paper.json` (`tldr`, `highlights`, section texts, tables, matrix).
   Settings A, B, C and the train–validation gaps have no JSON in `results/tables/`; they come from
   the paper's Tables III and IV.
3. If the title or the one-line description changed, update the `og:*` / `twitter:*` tags and
   `<title>` in `index.html` by hand (link-preview crawlers do not run JavaScript).
4. Test locally, commit, and tag: `git tag page-v-camera-ready`.

## When the paper is published

In `static/data/paper.json`:

1. `links.paper`: URL of the article page in the proceedings or journal (DOI or publisher page,
   never a PDF hosted here). The Paper button turns on by itself.
2. `status`: `"published"` (shows "Published at …"); adjust `venue.short` if needed.
3. `bibtex`: replace the `@software` entry with the publication reference, and update or empty
   `bibtexNote`.

## Adding showcase media

Copy the file into `static/images/` and add an item to `gallery.groups` in `paper.json`, with
`width`, `height` and a descriptive `alt`. Keep the originals in `results/showcase/` untouched. A
GIF over ~3 MB should become a looping video:

```bash
ffmpeg -i in.gif -vf "fps=5" -c:v libvpx-vp9 -crf 32 -b:v 0 -an out.webm
ffmpeg -i in.gif -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 20 \
       -pix_fmt yuv420p -movflags +faststart -an out.mp4
ffmpeg -i in.gif -vframes 1 -q:v 4 out_poster.jpg
```
