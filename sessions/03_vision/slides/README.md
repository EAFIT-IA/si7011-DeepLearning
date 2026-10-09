# Session 03 slides

[S03_slides.marp.md](S03_slides.marp.md) is the editable Marp source.
[../figures/figures.md](../figures/figures.md) maps every figure and animation to its slide and lists how each one is produced.

Figures and animations that are not ready yet are hidden comments in the source, so the deck renders without gaps (63 slides now, 74 with every figure integrated).

## Render

Run from the repository root:

```bash
npx @marp-team/marp-cli@4.5.1 sessions/03_vision/slides/S03_slides.marp.md --html -o sessions/03_vision/slides/S03_slides.html
```

Keep the `figures/` directory beside `slides/`. The HTML uses relative paths for figures, video posters and MP4 files. The `--html` flag enables the video elements.

For a static PDF:

```bash
npx @marp-team/marp-cli@4.5.1 sessions/03_vision/slides/S03_slides.marp.md --html --allow-local-files --pdf -o sessions/03_vision/slides/S03_slides.pdf
```

PDF does not play MP4 files. Animation slides provide static posters and links.
