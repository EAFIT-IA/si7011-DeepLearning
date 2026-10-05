# SI7011 — Evaluation strategy

The evaluation strategy follows the conceptual organization of the course rather than individual architectures.

## Structure

| Assessment | Sessions | Weight | Main focus |
|---|---:|---:|---|
| **Evaluation Event 1 — Learning & Training** | S01–S02 | **20%** | learning loop, optimization, backpropagation, initialization, regularization, training diagnostics |
| **Evaluation Event 2 — Architecture & Representation** | S03–S04 | **20%** | inductive bias, CNN/ViT, transfer learning, embeddings, reconstruction and self-supervised learning |
| **Evaluation Event 3 — Foundation Models & Adaptation** | S05–S06 | **20%** | sequence modeling, attention, Transformers, pretrained/foundation models, adaptation, evaluation and inference |
| **Integrative Project** | transversal | **40%** | end-to-end formulation, modeling, experimentation, evaluation and deployment |

## Design principle

Each evaluation event is designed **after the corresponding pair of sessions has been fully designed**.

This allows the assessment to reflect:

- the actual concepts emphasized in class;
- the experiments students performed;
- the level of mathematical depth reached;
- the kinds of evidence students learned to interpret.

The sequence is therefore:

```text
Design S01 + S02
        ↓
Design Evaluation Event 1

Design S03 + S04
        ↓
Design Evaluation Event 2

Design S05 + S06
        ↓
Design Evaluation Event 3
```

## Evaluation events

The events should not be architecture-specific implementation exercises. Their purpose is to evaluate whether students can **reason about, diagnose and justify deep-learning decisions**.

Each event should combine two dimensions:

### Conceptual reasoning

Students should interpret evidence, explain mechanisms and distinguish related concepts.

Examples:

- identify an optimization failure from training curves;
- explain the effect of initialization or normalization;
- justify an architectural inductive bias;
- interpret an embedding space;
- distinguish a Transformer architecture from a foundation model;
- explain when fine-tuning is preferable to frozen features.

### Experimental reasoning

Students should work with a controlled experiment, notebook or model and use evidence to support a conclusion.

Typical tasks:

- formulate a prediction before execution;
- modify one factor while controlling the rest;
- compare experimental conditions;
- select appropriate metrics;
- inspect errors or representations;
- justify conclusions from observed evidence.

A useful design target is approximately:

[
40\%\ \text{conceptual reasoning}
+
60\%\ \text{experimental reasoning}
]

This proportion is a design guideline, not a rigid grading requirement.

## Transversal question

Across all three events, the central assessment question is:

> **What assumption does this modeling decision introduce, and what evidence shows whether that assumption helps in this problem?**

This question connects optimization, architectures, representation learning, foundation models and deployment under a common experimental perspective.

---

# Integrative project — 40%

The project is the transversal component of the course.

Its purpose is to demonstrate that students can move from a problem and data to a justified deep-learning system:

[
\text{problem}
\rightarrow
\text{data}
\rightarrow
\text{baseline}
\rightarrow
\text{model}
\rightarrow
\text{experiments}
\rightarrow
\text{evaluation}
\rightarrow
\text{application}
]

## Development across the course

### S01–S02
Define:

- problem;
- input and output variables;
- data source;
- baseline;
- initial evaluation metric.

### S03–S04
Justify:

- representation;
- architecture;
- transfer-learning strategy when relevant;
- experimental comparisons.

### S05
Evaluate whether a pretrained or foundation model is relevant to the problem.

### S06
Complete:

- final evaluation;
- error analysis;
- inference pipeline;
- deployment or executable demonstration;
- limitations.

## Checkpoints

Project checkpoints are primarily used for **feedback and course correction**, rather than becoming many independent graded deliverables.

The final project evaluation will be defined after the six sessions are fully designed.

A provisional internal structure is:

| Dimension | Reference weight |
|---|---:|
| Problem and data formulation | 5% |
| Experimental design | 10% |
| Results and analysis | 15% |
| Reproducibility / implementation | 5% |
| Final defense | 5% |
| **Total** | **40%** |

These internal weights remain provisional until the project rubric is designed.

## Individual accountability

The final evaluation should include a short individual defense or questioning component so that every team member can explain:

- the problem formulation;
- the model;
- the experimental decisions;
- the evidence supporting the conclusions;
- the main limitations.

---

# Assessment workflow

The evaluation design is intentionally iterative:

1. complete the design of two sessions;
2. identify the essential learning outcomes;
3. identify the experiments and evidence students encountered;
4. design the corresponding evaluation event;
5. define its rubric and expected artifacts;
6. verify alignment between teaching, practice and assessment.

This keeps assessment aligned with the course as it is actually taught.
