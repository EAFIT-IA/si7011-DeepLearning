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

For S02 and S03, replace `01_learning/slides/S01` with `02_deep_training/slides/S02` or `03_vision/slides/S03`. If Marp reports "No suitable browser", point `CHROME_PATH` to a Chrome or Chromium binary. After editing a `.marp.md`, re-render the PDF: the student READMEs link to the PDF.

## Figures and animations

| Session | Inventory (slide ↔ figure, sources, Manim scenes) | Other |
|---|---|---|
| S01 | [`figures.md`](../sessions/01_learning/figures/figures.md) | Manim scenes in `escena*.py`, `escenas_s01.py` |
| S02 | [`figures.md`](../sessions/02_deep_training/figures/figures.md) | [`figure_prompts.md`](../sessions/02_deep_training/figures/figure_prompts.md) (image-generator prompts), `escenas_s02.py` |
| S03 | [`figures.md`](../sessions/03_vision/figures/figures.md) | [`figure_prompts.md`](../sessions/03_vision/figures/figure_prompts.md), `escenas_s03.py`, `make_s03_*.py` and `fix_s03_f14_cnn_panel.py` (see below) |

If an animation is added or renamed, update the table in that session's `slides/README.md`.

S03 figures F02, F06, F10 and F11 are drawn from the saved outputs of notebooks 3.1, 3.4 and 3.5, so the slides show the run the students see. After re-running any of those notebooks, rebuild them and re-render the PDF:

```bash
cd sessions/03_vision/figures && python make_s03_figs_from_notebooks.py ../notebooks .
```

## Notebooks

Open-in badges (Colab, Kaggle, Lightning) are inserted with `python tools/add_open_in_badges.py` from the repository root. The script also touches the S01 notebooks' cell ids; revert those if they were not meant to change.

S03 notebooks load their data from Kaggle datasets (CIFAR-10, Oxford-IIIT Pets, EuroSAT); the handles are listed in [`sessions/03_vision/notebooks/README.md`](../sessions/03_vision/notebooks/README.md). Notebooks 3.1–3.5 are committed with the outputs of a Kaggle GPU run; the integrating exercise 3.6 is committed without outputs.
