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

For S02, replace `01_learning/slides/S01` with `02_deep_training/slides/S02`; for S04, with `04_representation/slides/S04`. If marp waits on stdin, add `--no-stdin`. After editing a `.marp.md`, re-render the PDF: the student READMEs link to the PDF.

## Figures and animations

| Session | Inventory (slide ↔ figure, sources, Manim scenes) | Other |
|---|---|---|
| S01 | [`figures.md`](../sessions/01_learning/figures/figures.md) | Manim scenes in `escena*.py`, `escenas_s01.py` |
| S02 | [`figures.md`](../sessions/02_deep_training/figures/figures.md) | [`figure_prompts.md`](../sessions/02_deep_training/figures/figure_prompts.md) (image-generator prompts), `escenas_s02.py` |
| S04 | [`figures.md`](../sessions/04_representation/figures/figures.md) | [`figure_prompts.md`](../sessions/04_representation/figures/figure_prompts.md), `escenas_s04.py`, `make_s04_a01_latent_snapshots.py`, `fix_s04_f04_f06.py`; slide plan in [`S04_plan.md`](../sessions/04_representation/S04_plan.md) |

If an animation is added or renamed, update the table in that session's `slides/README.md`.

## Notebooks

Open-in badges (Colab, Kaggle, Lightning) are inserted with `python tools/add_open_in_badges.py` from the repository root. The script also touches the S01 notebooks' cell ids; revert those if they were not meant to change.

## S04: the SimCLR checkpoint dataset

Notebooks 4.4 and 4.5 expect the Kaggle dataset **`juanmartinezv4399/si7011-s04-simclr`** (the handle is hard-coded as `CKPT_HANDLE` in both notebooks and in the notebooks README) with two files. Produce them once, before the session:

1. Open `sessions/04_representation/notebooks/sesion_04_4_simclr.ipynb` on Kaggle with a GPU (T4 or P100).
2. In section 4 set `ENTRENAR_LARGO, ABLACION = True, True` and run the notebook. It writes `simclr_resnet18_cifar10.pt` (100 epochs, with color jitter) and `simclr_sin_color.pt` (100 epochs, without color jitter and grayscale): about two hours in total.
3. Create the dataset `si7011-s04-simclr` under the account `juanmartinezv4399` from those two files (public, so students can add it with *Add Input* and Colab can fetch it with `kagglehub`) and set the flags back to `False`. To train on Kaggle, add `pankrzysiu/cifar10-python` as input first.

The S04 integrating-exercise solutions are not in the repository.
