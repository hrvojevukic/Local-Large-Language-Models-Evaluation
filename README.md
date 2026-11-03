# Evaluation and Comparison of Local Large Language Models in News Summarization

**Authors:** Hrvoje Vukić

This project compares three large language models running locally through [Ollama](https://ollama.com) on one task: summarizing news articles in exactly three sentences. Each model summarized the same 100 articles, and the summaries were scored on speed, text overlap, semantic similarity and writing style.

The study is written up in a short report: [`evaluation of local large models.pdf`](evaluation%20of%20local%20large%20models.pdf).

## Key results

- **Llama 3.2** was the fastest (5.68 s per summary). It scored highest on every overlap metric, BERTScore, keyword coverage and readability.
- **Qwen 3 (4b)** was the slowest (42.14 s per summary). It scored highest on semantic similarity and lexical diversity, and had the lowest redundancy.
- **Gemma 3 (4b)** was in the middle on speed (14.20 s per summary) and had the highest novelty, meaning the largest share of words not found in the reference text.

No model leads on every measure. Llama 3.2 suits cases where speed and closeness to the source matter most. Qwen 3 and Gemma 3 suit cases where more varied wording matters and slower generation is acceptable.

## Models

| Model | Ollama tag | Name in the results |
| --- | --- | --- |
| Llama 3.2 | `llama3.2` | `llama` |
| Gemma 3 (4b) | `gemma3:4b` | `gemma` |
| Qwen 3 (4b) | `qwen3:4b` | `qwen` |

## Dataset

[`data/news_summary.csv`](data/news_summary.csv) contains 4,514 news items with the columns `author`, `date`, `headlines`, `read_more`, `text` and `ctext`. Two columns matter for this project:

- `ctext` is the full article, about 350 words on average in the evaluated subset.
- `text` is a short summary of the article, about 58 words on average in the evaluated subset.

The experiment uses the first 100 rows. Articles are cut to 5,000 characters before being sent to a model, which affected 3 of the 100.

## Method

### 1. Generating summaries

[`src/summarize.py`](src/summarize.py) sends each article to each model through the Ollama Python library, using the same prompt for all three:

```text
You are an expert news editor.
Summarize the following news article.

Requirements:
- Use exactly 3 sentences.
- Keep only the most important information.
- Use journalistic style.
- Avoid repetition.

Article:
{article}
```

For every summary, the script records the execution time and the peak Python memory allocation. The output is saved to [`data/news_summary_multi_model_metrics.csv`](data/news_summary_multi_model_metrics.csv).

### 2. Scoring summaries

[`src/project.ipynb`](src/project.ipynb) computes the metrics below for all 300 summaries, then aggregates and plots them. The saved outputs show it was run on Google Colab.

| Group | Metric | How it is computed |
| --- | --- | --- |
| Overlap | ROUGE-1, ROUGE-2, ROUGE-L | F-measure from `rouge-score`, with stemming |
| Overlap | BLEU | NLTK sentence BLEU with smoothing method 1 |
| Overlap | METEOR | NLTK METEOR on whitespace-separated tokens |
| Semantic | BERTScore | F1 from `bert-score`, using `distilroberta-base` |
| Semantic | Semantic Similarity | Cosine similarity of `all-MiniLM-L6-v2` embeddings |
| Coverage | Keyword Coverage | Share of the reference's unique words that appear in the summary, with no stopword filtering |
| Style | Lexical Diversity | Unique words divided by total words in the summary |
| Style | Redundancy | 1 minus Lexical Diversity |
| Style | Novelty | Share of the summary's unique words that are not in the reference |
| Style | Readability | Flesch Reading Ease from `textstat`; higher means easier to read |
| Structure | Summary length | Word count of the summary |
| Structure | Compression ratio | Summary word count divided by `text` word count |
| Performance | Execution time | Wall-clock time of the Ollama call |
| Performance | Memory usage | Peak Python allocation measured with `tracemalloc` |

### 3. Predicting summary quality

As an extra experiment, the notebook trains an FT-Transformer with `pytorch-tabular` to predict a summary's ROUGE-L score from 12 tabular features. See [Predicting ROUGE-L](#predicting-rouge-l) for the setup and results.

## Results

Mean values per model across the 100 articles. The best value in each scored row is in bold.

| Metric | Llama 3.2 | Gemma 3 (4b) | Qwen 3 (4b) |
| --- | --- | --- | --- |
| ROUGE-1 | **0.3476** | 0.2770 | 0.2992 |
| ROUGE-2 | **0.1781** | 0.1049 | 0.1359 |
| ROUGE-L | **0.2237** | 0.1747 | 0.2025 |
| BLEU | **0.0332** | 0.0119 | 0.0157 |
| METEOR | **0.1585** | 0.1112 | 0.1228 |
| BERTScore | **0.8781** | 0.8687 | 0.8768 |
| Semantic Similarity | 0.7774 | 0.7916 | **0.8052** |
| Keyword Coverage | **0.2478** | 0.1772 | 0.2016 |
| Lexical Diversity | 0.8052 | 0.8393 | **0.8523** |
| Redundancy (lower is better) | 0.1948 | 0.1607 | **0.1477** |
| Novelty | 0.4069 | **0.5338** | 0.4227 |
| Readability | **33.0092** | 17.2336 | 24.4339 |
| Execution time in seconds (lower is better) | **5.68** | 14.20 | 42.14 |
| Summary length in words | 91.88 | 78.89 | 71.22 |
| Compression ratio | 1.5736 | 1.3517 | 1.2192 |
| Memory usage in MB | 0.0752 | 0.0741 | 0.0743 |

Two summary views from the notebook point the same way:

| | Llama 3.2 | Qwen 3 (4b) | Gemma 3 (4b) |
| --- | --- | --- | --- |
| Overall score | **0.4142** | 0.4046 | 0.3916 |
| Articles won on ROUGE-L | **71** | 27 | 2 |

The overall score is the unweighted mean of ROUGE-L, BLEU, METEOR, BERTScore and Semantic Similarity.

The notebook's saved outputs also include a correlation heatmap, bar charts for each metric group, box plots of the per-article distributions, and a scatter plot of execution time against ROUGE-L.

### Predicting ROUGE-L

The features are the model, article length, reference length, summary length, compression ratio, execution time, memory usage, lexical diversity, readability, keyword coverage, redundancy and novelty. The 300 rows are split 80/20 with `random_state=42`, and training uses a batch size of 64 for up to 50 epochs. The test split also serves as the validation split during training.

| MAE | RMSE | R² |
| --- | --- | --- |
| 0.0845 | 0.1040 | -1.0701 |

The negative R² means the model predicts ROUGE-L worse than always guessing the mean, so this setup does not give a usable predictor.

## Repository contents

| File | Description |
| --- | --- |
| [`src/summarize.py`](src/summarize.py) | Generates the summaries with Ollama and records time and memory |
| [`src/project.ipynb`](src/project.ipynb) | Computes metrics, creates charts and trains the FT-Transformer |
| [`data/news_summary.csv`](data/news_summary.csv) | Source dataset |
| [`data/news_summary_multi_model_metrics.csv`](data/news_summary_multi_model_metrics.csv) | First 100 articles with each model's summary, time and memory |
| [`results/metrics_per_article.csv`](results/metrics_per_article.csv) | All metrics for each of the 300 summaries |
| [`results/evaluation_results.csv`](results/evaluation_results.csv) | Mean metrics per model |
| [`results/best_models.csv`](results/best_models.csv) | Best model for each of 10 metrics |
| [`results/descriptive_statistics.csv`](results/descriptive_statistics.csv) | Count, mean, standard deviation and quartiles for each metric |
| [`results/winner_per_article.csv`](results/winner_per_article.csv) | Model with the highest ROUGE-L for each article |
| [`results/summary_examples.csv`](results/summary_examples.csv) | Side-by-side summaries for the first 5 articles |
| [`results/tabular_model_results.csv`](results/tabular_model_results.csv) | FT-Transformer error metrics |
| [`evaluation of local large models.pdf`](evaluation%20of%20local%20large%20models.pdf) | Project report |

## Running the project

### Generate summaries

Install [Ollama](https://ollama.com), make sure it is running, and download the models:

```bash
ollama pull llama3.2
```

```bash
ollama pull gemma3:4b
```

```bash
ollama pull qwen3:4b
```

Install the script's dependencies:

```bash
pip install pandas ollama tqdm
```

Run the script from the `src` folder. This overwrites `data/news_summary_multi_model_metrics.csv`:

```bash
cd src
```

```bash
python summarize.py
```

The script's console messages are in Croatian. Its first message mentions 200 rows, but the code reads 100 (`nrows=100`).

### Evaluate summaries

Open `src/project.ipynb` in Jupyter and run all cells. The notebook reads from `data/` and overwrites the CSV files in `results/`. It installs its own dependencies with `pip`, and downloads the NLTK data and Hugging Face models it needs.

On Google Colab, clone the repository and change into `src/` first so the relative paths resolve.

## Notes and limitations

- **The reference is the full article.** The notebook uses `ctext` as `reference` and `text` as `article`. Overlap, semantic and coverage scores therefore measure how closely a summary matches the article it was generated from, not a short human summary. This also explains two results: compression ratios are above 1 because summaries are longer than `text`, and BLEU is low because a reference of about 350 words causes a heavy brevity penalty.
- **Differences from the report.** The report describes the reference as human-written and names Gemma 3 as having the lowest redundancy. This README follows the code and CSV files, which use `ctext` as the reference and show Qwen 3 with the lowest redundancy (0.1477).
- **Model output is scored as is.** 94 Llama summaries and 55 Gemma summaries open with a lead-in starting with "Here", such as "Here is a 3-sentence summary of the article:". At least one Llama output is a refusal rather than a summary. None of this text is removed before scoring.
- **Memory figures do not show model memory use.** `tracemalloc` only tracks the Python process, while inference runs in the Ollama service. The values of about 0.07 MB reflect the API call and cannot be used to compare the models.
- **Timings are approximate.** Each article was summarized once per model, and the hardware is not recorded, so the times are best read as a comparison between these models.
- **The sample is small.** The results cover 100 of the 4,514 articles in the dataset.
