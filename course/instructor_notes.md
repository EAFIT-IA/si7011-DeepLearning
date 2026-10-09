# Instructor notes: building the materials

Not for students. Production notes that used to live in the session READMEs.

## Rendering the slides

Run from the repository root. Keep `figures/` beside `slides/`: the HTML and PDF use relative paths for figures, video posters and MP4 files, and `--html` enables the video elements.

```bash
# HTML, plays the animations
npx @marp-team/marp-cli@4.5.1 sessions/01_learning/slides/S01_slides.marp.md --html -o sessions/01_learning/slides/S01_slides.html
# PDF, static posters instead of videos
npx @marp-team/marp-cli@4.5.1 sessions/01_learning/slides/S01_slides.marp.md --html --allow-local-files --pdf -o sessions/01_learning/slides/S01_slides.pdf
```

For S02, replace `01_learning/slides/S01` with `02_deep_training/slides/S02`. After editing a `.marp.md`, re-render the PDF: the student READMEs link to the PDF.

## Figures and animations

| Session | Inventory (slide ↔ figure, sources, Manim scenes) | Other |
|---|---|---|
| S01 | [`figures.md`](../sessions/01_learning/figures/figures.md) | Manim scenes in `escena*.py`, `escenas_s01.py` |
| S02 | [`figures.md`](../sessions/02_deep_training/figures/figures.md) | [`figure_prompts.md`](../sessions/02_deep_training/figures/figure_prompts.md) (image-generator prompts), `escenas_s02.py` |

If an animation is added or renamed, update the table in that session's `slides/README.md`.

## Notebooks

Open-in badges (Colab, Kaggle, Lightning) are inserted with `python tools/add_open_in_badges.py` from the repository root. The script also touches the S01 notebooks' cell ids; revert those if they were not meant to change.
