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

For S02–S05, replace `01_learning/slides/S01` with `02_deep_training/slides/S02`, `03_vision/slides/S03` or `04_representation/slides/S04` or `05_sequences/slides/S05`. If Marp reports "No suitable browser", point `CHROME_PATH` to a Chrome or Chromium binary; if it waits on stdin, add `--no-stdin`. After editing a `.marp.md`, re-render the PDF: the student READMEs link to the PDF.

## Figures and animations

| Session | Inventory (slide ↔ figure, sources, Manim scenes) | Other |
|---|---|---|
| S01 | [`figures.md`](../sessions/01_learning/figures/figures.md) | Manim scenes in `escena*.py`, `escenas_s01.py` |
| S02 | [`figures.md`](../sessions/02_deep_training/figures/figures.md) | [`figure_prompts.md`](../sessions/02_deep_training/figures/figure_prompts.md) (image-generator prompts), `escenas_s02.py` |
| S03 | [`figures.md`](../sessions/03_vision/figures/figures.md) | [`figure_prompts.md`](../sessions/03_vision/figures/figure_prompts.md), `escenas_s03.py`, `make_s03_*.py` and `fix_s03_f14_cnn_panel.py` (see below) |
| S04 | [`figures.md`](../sessions/04_representation/figures/figures.md) | [`figure_prompts.md`](../sessions/04_representation/figures/figure_prompts.md), `escenas_s04.py`, `make_s04_a01_latent_snapshots.py`, `fix_s04_f04_f06.py`; slide plan in [`S04_plan.md`](plans/S04_plan.md) |
| S05 | [`figures.md`](../sessions/05_sequences/figures/figures.md) | [`figure_prompts.md`](../sessions/05_sequences/figures/figure_prompts.md); slide plan in [`S05_plan.md`](plans/S05_plan.md) |

If an animation is added or renamed, update the table in that session's `slides/README.md`.

S03 figures F02, F06, F10 and F11 are drawn from the saved outputs of notebooks 3.1, 3.4 and 3.5, so the slides show the run the students see. After re-running any of those notebooks, rebuild them and re-render the PDF:

```bash
cd sessions/03_vision/figures && python make_s03_figs_from_notebooks.py ../notebooks .
```

## Notebooks

Open-in badges (Colab, Kaggle, Lightning) are inserted with `python tools/add_open_in_badges.py` from the repository root. The script also touches the S01 notebooks' cell ids; revert those if they were not meant to change.

S03 notebooks load their data from Kaggle datasets (CIFAR-10, Oxford-IIIT Pets, EuroSAT); the handles are listed in [`sessions/03_vision/notebooks/README.md`](../sessions/03_vision/notebooks/README.md). Notebooks 3.1–3.5 are committed with the outputs of a Kaggle GPU run; the integrating exercise 3.6 is committed without outputs.

## S04: the SimCLR checkpoint model

Notebooks 4.4 and 4.5 load the SimCLR encoders from the **Kaggle model** [`juanmartinezv4399/si7011-s04-simclr`](https://www.kaggle.com/models/juanmartinezv4399/si7011-s04-simclr), framework/variation `pyTorch/default` (hard-coded as `CKPT_HANDLE` and `CKPT_VARIANTE` in both notebooks). On Kaggle students add it with *Add Input → Models*; the notebooks find the files anywhere under `/kaggle/input`. On Colab they download it with `kagglehub.model_download`.

Files: `simclr_resnet18_cifar10.pt` (required by 4.4 and 4.5) and `simclr_sin_color.pt` (only section 6 of 4.4; without it, that section prints a notice and is skipped).

To produce or redo them, use [`s04_entrenar_simclr.ipynb`](s04_entrenar_simclr.ipynb), which copies the data, views, model and loss cells of notebook 4.4:

1. Open it on Kaggle; *Add Input* `pankrzysiu/cifar10-python`; *Settings → Accelerator* GPU.
2. **Save Version → Save & Run All (Commit)**, so it runs in the background. After the first epoch it prints the estimated total time. Default: 30 epochs per file; lower `EPOCAS` if the estimate is too long.
3. It saves `*_parcial.pt` every epoch and stops by itself before `MAX_HORAS` (11 h). If it stopped: new version, *Add Input → Your Work →* the previous version's output, *Save & Run All* again. It resumes where it stopped and copies finished files to the new output.
4. Download the finished `.pt` files from the *Output* tab and upload them as a **new version** of the model (same framework and variation), so the handle does not change.

The S04 integrating-exercise solutions are not in the repository.

## S06: running notebook 6.3 (SFT)

Notebook 6.3 fine-tunes **Qwen2.5-0.5B** (the base model, not Instruct) on 2 000 no-context pairs of Dolly-15k in Spanish (`argilla/databricks-dolly-15k-curated-multilingual`, split `es`), with a hand-written masked loss and LoRA. There is no checkpoint to publish: the whole notebook runs in class. Both the model (about 1 GB) and the data come from Hugging Face, so on Kaggle the students need **Internet on** in *Settings*, plus a GPU (T4).

It is committed **without outputs** (it was not run on a GPU when written). Before class:

1. Run it once on Kaggle with a T4 and note the time and peak memory it prints (estimate: 10–15 minutes; batch 8, `T_MAX = 384`). If memory is short, lower `B` to 4.
2. Commit the executed copy and `resultados_sft.json` next to it, then draw F13 and F18:

```bash
cd sessions/06_foundation_models/figures && python make_s06_figuras_notebooks.py ../notebooks . f13,f18
```

3. Replace the F13 and F18 placeholders (slides 48 and 72) and write the measured answers in the notes of the prediction slides 47 and 73.

## S06: running notebook 6.5 (DistilBERT on IMDB)

Back in class (Practice B, after 6.3). It needs a T4 and **Internet on** (DistilBERT and `stanfordnlp/imdb` from Hugging Face; without Internet for the data, *Add Input* `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`, but the model still needs Internet). It trains its own LSTM baseline, so it does not depend on notebook 5.4. Committed without outputs; before class run it once (estimate: about 10 minutes), commit the executed copy with `resultados_adaptacion.json`, write the measured answer in the note of slide 76 and draw F19 (slide 82) with `python make_s06_figuras_notebooks.py ../notebooks . f19`.

## S06: the styled dataset of the integrating exercise

The integrator 6.6 reads `dolly_quijote_es.csv` from the Kaggle dataset `juanmartinezv4399/si7011-s06-dolly-quijote`, published on 2026-10-21 ([link](https://www.kaggle.com/datasets/juanmartinezv4399/si7011-s06-dolly-quijote)). To regenerate it, use [`s06_generar_estilo.ipynb`](s06_generar_estilo.ipynb):

1. Open it on Kaggle; *Settings*: GPU T4 and **Internet on** (it downloads Dolly-es and Qwen2.5-7B-Instruct, loaded in 4 bits with `bitsandbytes`).
2. **Save Version → Save & Run All (Commit)**. Estimate: about 40 minutes for 1 200 answers.
3. Read the samples it prints at the end; then download `dolly_quijote_es.csv` from *Output* and publish it as the dataset above, public, CC BY-SA 3.0 (Dolly's license).

The questions are positions 2 200–3 399 of the same filtered and shuffled Dolly-es that notebook 6.3 uses (positions 0–2 199), so the two never share a question. Then run 6.6 once with the solutions (kept outside the repository, like the S04 ones) to confirm the time and the ordering of the three conditions.

Notebook 6.4 is optional homework; to draw F15 (slide 56), run it once and then `python make_s06_figuras_notebooks.py ../notebooks . f15`.

