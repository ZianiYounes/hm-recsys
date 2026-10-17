# H&M Fashion Recommender

A two-stage recommendation system trained on 31.8 million real transactions from H&M. For each customer it predicts the 12 articles they are most likely to buy next week, and serves those predictions through a real-time API.

![Demo](docs/images/demo.png)

## Results

Offline evaluation on a held-out week (72,019 customers who made a purchase). Every model only sees data from before that week.

| Model | MAP@12 | Recall@12 | Recall@100 | Hit rate@12 |
|---|---|---|---|---|
| Popularity (last week) | 0.0066 | 0.021 | 0.099 | 5.7% |
| Repurchase | 0.0215 | 0.034 | 0.035 | 6.1% |
| Rule-based blend | 0.0268 | 0.056 | 0.137 | 10.7% |
| Two-tower retrieval alone | 0.0151 | 0.041 | 0.143 | 9.4% |
| Rules + two-tower | 0.0272 | 0.058 | 0.156 | 11.6% |
| **Two-tower + LightGBM ranker** | **0.0358** | **0.076** | **0.176** | **15.2%** |

The ranker improves MAP@12 by 34% over the best rule-based approach. In plain terms, 15% of customers buy at least one of the 12 articles they are shown.

## How it works

```mermaid
flowchart LR
    T[31.8M transactions<br/>105k articles<br/>1.37M customers] --> S[Temporal split]
    S --> R1[Repurchase]
    S --> R2[Co-purchase]
    S --> R3[Popularity]
    S --> R4[Two-tower<br/>PyTorch]
    S --> R5[SASRec<br/>optional]
    R1 --> C[~145 candidates<br/>per customer]
    R2 --> C
    R3 --> C
    R4 --> C
    R5 --> C
    C --> F[50 features<br/>per pair]
    F --> L[LightGBM<br/>LambdaRank]
    L --> O[Top 12]
```

**Candidate generation.** A catalogue of 105k articles is too large to score with a heavy model for every customer, so several cheap sources each propose candidates. Three are simple rules. The fourth is a two-tower neural network: one tower encodes a customer from their last 32 purchases (attention pooling over item embeddings, plus recency and profile features), the other encodes an article from its ID and 12 attributes. Both live in the same 128-dimensional space, so retrieval is a nearest-neighbour search. It is trained with an in-batch softmax loss and logQ correction to avoid over-penalising popular items.

**Ranking.** A LightGBM LambdaRank model orders the candidates using 50 features: the score and rank given by each source, whether the customer already bought the article or the same model in another colour, recent sales and trend of the article, the price relative to the customer's usual spend, and the two-tower similarity. The two strongest signals are the department of the article and how recently the customer bought the same model.

**Evaluation protocol.** Weeks are counted backwards from the last day of data. To evaluate week W, retrievers are trained on everything before W, and the ranker is trained on weeks W+1 and W+2 with retrievers that never saw those weeks. Nothing from the evaluated week leaks into training.

**Serving.** A FastAPI service loads the models once, encodes the customer on the fly, retrieves candidates with a FAISS HNSW index, computes features with the same code used in training, and returns the top 12 with a short explanation for each item. A Streamlit app shows a customer's recent purchases next to their recommendations.

## Tech stack

PyTorch (nightly, CUDA 12.8), Polars, DuckDB, LightGBM, FAISS, FastAPI, Streamlit, MLflow, Docker, GitHub Actions. Trained on a single RTX 5080 laptop GPU: about 2 minutes per two-tower model and 30 seconds for the ranker.

## Getting started

Requirements: Python 3.11+, an NVIDIA GPU for training (serving runs on CPU), and a Kaggle account. The dataset is not included in this repository: accept the rules of the [H&M competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) first.

```bash
# Environment (Windows PowerShell script, adapt the venv activation on Linux/macOS)
./setup_venv.ps1
pip install -e .

# Data
kaggle auth login
python scripts/run_download.py
python scripts/run_preprocess.py
python scripts/run_split.py

# Models (validation week)
python scripts/run_baselines.py
python scripts/run_train.py --model two_tower
python scripts/run_train.py --model ranker

# Serving
python scripts/build_serving.py
uvicorn recsys.serving.api.main:app --port 8000
streamlit run app/streamlit_app.py
```

The API documentation is available at `http://localhost:8000/docs`. With Docker, after `build_serving.py`:

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Endpoint | Description |
|---|---|
| `GET /recommend/{customer_id}?k=12` | Top-k articles with reasons and latency breakdown |
| `GET /customers/{customer_id}/history` | Recent purchases |
| `GET /articles/{article_id}/similar` | Nearest articles in the two-tower embedding space |
| `GET /customers/sample` | Random customers for the demo |

## Project structure

```
configs/            one YAML file per step (baselines, two_tower, sasrec, ranker, serving)
scripts/            command-line entry points
src/recsys/
  data/             preprocessing, temporal split, GPU sequence store
  features/         item, customer and interaction features
  models/           baselines, two-tower, SASRec encoder, ranking
  training/         training loops for retrievers and ranker
  retrieval/        exact top-k on GPU, FAISS index for serving
  embeddings/       FashionCLIP image and text embeddings
  serving/          real-time recommender and FastAPI app
app/                Streamlit demo
tests/              unit tests and an end-to-end API test on synthetic data
```

## Going further

- `python scripts/run_train.py --model sasrec` trains a transformer sequence encoder (SASRec style) as a second retriever. Add it to `retrievers` in `configs/ranker.yaml` to use it in the ranker and the API.
- `scripts/run_embeddings.py` computes FashionCLIP embeddings from product photos and descriptions. They can feed the item tower (`multimodal: true`) and add a visual similarity feature to the ranker, which helps with new articles that have no sales yet.
- Every run is tracked in MLflow (`mlflow server --backend-store-uri sqlite:///mlflow.db`).

## Résumé en français

Système de recommandation en deux étages sur les données réelles de H&M. Des sources rapides (règles métier et réseau two-tower en PyTorch) proposent environ 145 candidats par client, puis un modèle LightGBM LambdaRank les classe à partir de 50 variables. Le MAP@12 passe de 0.0268 avec les règles seules à 0.0358, soit +34 %. Les recommandations sont servies par une API FastAPI avec un index FAISS, et une démo Streamlit permet de les explorer.

## License

Code under the MIT License. The H&M data is subject to the Kaggle competition rules and is not redistributed here.

Younes Ziani
