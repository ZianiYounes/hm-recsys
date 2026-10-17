"""Demo : streamlit run app/streamlit_app.py (l'API doit tourner sur API_URL)."""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
IMAGES_DIR = Path(os.getenv("DATA_DIR", "data")) / "raw" / "images"
N_COLS = 6

st.set_page_config(page_title="H&M Recommender", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .noimg {height: 160px; border-radius: 8px; background: rgba(128,128,128,.12);
              display: flex; align-items: center; justify-content: center;
              font-size: .8rem; opacity: .7; text-align: center; padding: .5rem;}
      .reason {font-size: .75rem; opacity: .8; margin: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def get(path: str, **params):
    r = httpx.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def cached_get(path: str, **params):
    return get(path, **params)


def image_or_placeholder(article: dict) -> None:
    img = IMAGES_DIR / article["image_path"]
    if img.exists():
        st.image(str(img), width="stretch")
    else:
        st.markdown(f"<div class='noimg'>{article['product_type'] or ''}</div>",
                    unsafe_allow_html=True)


def card(article: dict, subtitle: str | None = None, reasons: list[str] | None = None) -> None:
    image_or_placeholder(article)
    st.markdown(f"**{article['name']}**")
    st.caption(" · ".join(x for x in [article["colour"], article["product_type"]] if x))
    if subtitle:
        st.caption(subtitle)
    for r in reasons or []:
        st.markdown(f"<p class='reason'>{r}</p>", unsafe_allow_html=True)


def grid(items: list[dict], render) -> None:
    for i in range(0, len(items), N_COLS):
        for col, item in zip(st.columns(N_COLS), items[i : i + N_COLS], strict=False):
            with col:
                render(item)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("H&M Recommender")
try:
    health = cached_get("/health")
except httpx.HTTPError:
    st.error(
        f"API unreachable at {API_URL}. Start it with "
        "`uvicorn recsys.serving.api.main:app --port 8000`."
    )
    st.stop()

st.sidebar.caption(
    f"Predicting the week after {health['reference_date']}  \n"
    f"{health['n_customers']:,} customers · {health['n_articles']:,} articles  \n"
    f"Retrievers: {', '.join(health['retrievers'])} + LightGBM ranker"
)
clicked = st.sidebar.button("Random customer", type="primary")
if clicked or "customer" not in st.session_state:
    st.session_state.customer = get("/customers/sample", n=1)[0]
customer = st.sidebar.text_input("Customer ID", value=st.session_state.customer).strip()
k = st.sidebar.slider("Recommendations", min_value=6, max_value=24, value=12, step=6)

# ----------------------------------------------------------------------------- page
try:
    recs = get(f"/recommend/{customer}", k=k)
    hist = cached_get(f"/customers/{customer}/history", limit=N_COLS * 2)
except httpx.HTTPStatusError as e:
    st.warning(f"Customer not found ({e.response.status_code}).")
    st.stop()

st.title("Next-week recommendations")
st.caption(f"Customer `{customer}`")
m1, m2, m3 = st.columns(3)
m1.metric("Past purchases", f"{hist['n_purchases']:,}")
m2.metric("Candidates scored", f"{recs['n_candidates']:,}")
m3.metric("Latency", f"{recs['latency_ms']['total_ms']:.0f} ms")

st.subheader("Recent purchases")
if hist["purchases"]:
    grid(hist["purchases"], lambda a: card(a, subtitle=a["date"]))
else:
    st.info("New customer with no purchase history: recommendations rely on trends.")

st.subheader(f"Top {k} for next week")
grid(recs["recommendations"], lambda a: card(a, reasons=a["reasons"]))

with st.expander("Latency breakdown"):
    lat = {key.replace("_ms", ""): v for key, v in recs["latency_ms"].items() if key != "total_ms"}
    st.bar_chart(pd.DataFrame({"ms": lat}))

st.subheader("Explore similar items")
options = {f"{r['rank']}. {r['name']} ({r['colour']})": r["article_id"]
           for r in recs["recommendations"]}
choice = st.selectbox("Pick a recommended item", list(options))
if choice:
    sim = cached_get(f"/articles/{options[choice]}/similar", k=N_COLS)
    grid(sim["similar"], card)
