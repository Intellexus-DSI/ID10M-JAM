# Data Generation

This folder contains everything related to the datasets used in ID10M-JAM.

## id10m (fixed)

The `id10m_fixed/` folder contains our cleaned version of the id10m dataset, used
as the baseline in all experiments. English (178 sentences) and German (137
sentences), in both JSON and CSV format.

The original id10m dataset was manually reviewed and the following sentence types
were removed:

- **Dual-meaning sentences** — sentences where the idiom could plausibly be
  interpreted as either literal or figurative depending on context, making the
  ground-truth label ambiguous
- **Multiple idioms in one sentence** — sentences containing more than one idiom,
  which complicates the identification task and evaluation
- **Nonsensical sentences** — sentences that did not form coherent, natural
  utterances (e.g. malformed constructions from the original data generation)

The result is a clean, unambiguous benchmark where each sentence has a
well-defined ground truth.

## id10m-jam

The full ID10M-JAM benchmark is hosted on HuggingFace:
[https://huggingface.co/datasets/Intellexus/ID10M-JAM](https://huggingface.co/datasets/Intellexus/ID10M-JAM)

Download it locally with:

```bash
python data_generation/download_id10m_jam.py
```

This saves `english.json` and `german.json` to `data_generation/id10m_jam/`
(gitignored — re-download as needed).

## Data Generation Pipeline

The `generation/` folder contains the Gemini-based pipeline used to generate the
confusing-context variants that form the id10m-jam benchmark. Requires a Gemini
API key in `.env`:

```bash
cd data_generation/generation
python variants_generator.py
```
