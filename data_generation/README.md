# Data Generation

This folder contains everything related to the datasets used in ID10M-JAM.

## id10m (fixed)

The `id10m_fixed/` folder contains our cleaned version of the id10m dataset, used
as the baseline in all experiments. English (178 sentences) and German (137
sentences), in both JSON and CSV format.

Each file contains the following columns:

| Column | Description |
|---|---|
| `sentence` | Full sentence text |
| `PIE` | Potentially idiomatic expression(s) present in the sentence |
| `true_idioms` | Idioms used figuratively — the ground-truth labels |
| `is_figurative` | Whether the sentence contains a figurative idiom use |
| `tokens` | Tokenized sentence |
| `tags` | BIO tags (`O`, `B-IDIOM`, `I-IDIOM`) |
| `tag_ids` | Numeric BIO tag IDs |
| `was_fixed` | `True` if the sentence was corrected or added during cleaning |

### Cleaning process

The original id10m dataset was manually reviewed. Sentences were removed or
corrected for the following reasons:

- **Dual-meaning sentences** — sentences where the idiom could plausibly be
  interpreted as either literal or figurative depending on context, making the
  ground-truth label ambiguous
- **Multiple idioms in one sentence** — sentences containing more than one idiom,
  which complicates identification and evaluation
- **Nonsensical sentences** — sentences that did not form coherent, natural
  utterances (e.g. malformed constructions from the original data generation)
- **Typos** — sentences with spelling or word errors in the original data
  (e.g. "hand" instead of "head", missing spaces) were corrected and marked
  `was_fixed = True`

The result is a clean, unambiguous benchmark where each sentence has a
well-defined ground truth. English started from 200 sentences (22 removed, 8
corrected or added). German started from a similar pool with 9 sentences corrected.

## id10m-jam

The full ID10M-JAM benchmark is hosted on HuggingFace:
[https://huggingface.co/datasets/Intellexus/ID10M-JAM](https://huggingface.co/datasets/Intellexus/ID10M-JAM)

Download it locally with:

```bash
python data_generation/download_id10m_jam.py
```

This saves `english.json` and `german.json` to `data_generation/id10m_jam/`
(gitignored — re-download as needed).

## Annotation Guidelines

The file [`ID10M-JAM Annotation Guildlines.pdf`](ID10M-JAM%20Annotation%20Guildlines.pdf) contains the annotation guidelines provided to the LLM during the confusing-context variant generation process.

## Data Generation Pipeline

The `generation/` folder contains the Gemini-based pipeline used to generate the
confusing-context variants that form the id10m-jam benchmark. Requires a Gemini
API key in `.env`:

```bash
cd data_generation/generation
python variants_generator.py
```
