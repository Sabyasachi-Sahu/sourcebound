"""Phase 5 page: ask the documents. Shows the routed packs, the answer, exact
citations, the supporting passage for every claim, verification and latency.
Step 5.12 adds the risk tier, cache outcome, index version and an as-of date."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.rag.answer import ask
from core.rag.common import packs

ICON = {"supported": "✅", "unclear": "❔", "wrong": "❌"}

st.title("Ask the documents")
st.caption(
    "Hybrid search, reranking, exact citations and a check of every claim. "
    "Leave the pack list empty to let the router choose. Synthetic documents "
    "and public drug labels only."
)

chosen = st.multiselect("Packs (empty = automatic)", list(packs()))
question = st.text_input(
    "Question", placeholder="How quickly are approved claims paid?"
)
left, middle, right = st.columns(3)
use_cache = left.checkbox("Use cache", value=True)
judge = middle.checkbox("Model judge for unclear claims", value=False)
use_date = right.checkbox("Answer as of a date", value=False)
as_of = st.date_input("As of", disabled=not use_date)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving, answering and checking..."):
        try:
            r = ask(
                question.strip(),
                chosen or None,
                use_cache=use_cache,
                judge=judge,
                as_of=as_of.isoformat() if use_date else None,
            )
        except Exception as exc:  # noqa: BLE001 - show any failure, do not crash
            st.error(f"Could not answer: {type(exc).__name__}: {exc}")
            st.stop()

    as_of_text = f" | as of {r['as_of']}" if r.get("as_of") else ""
    st.info(
        f"Packs: **{', '.join(r['packs'])}** | risk **{r['risk']}** | "
        f"cache: **{r['cache']}** | index `{r['index_version']}`{as_of_text}"
    )
    if r.get("refreshed"):
        st.warning(
            f"A source file changed since the last ingest and was re-ingested "
            f"first: {r['refreshed']}"
        )
    if r["refused"]:
        st.warning(r["answer"])
    else:
        st.markdown(r["answer"])

    v = r["verification"]["counts"]
    a, b, c, d = st.columns(4)
    a.metric("Supported", v["supported"])
    b.metric("Unclear", v["unclear"])
    c.metric("Wrong", v["wrong"])
    d.metric("Evidence strength", r["strength"])

    if r["citations"]:
        st.subheader("Citations")
        st.markdown("\n".join(f"- `[{x['id']}]` {x['label']}" for x in r["citations"]))
    if r["evidence"]:
        st.subheader("Supporting passages")
        for e in r["evidence"]:
            st.markdown(f"{ICON[e['status']]} **{e['claim']}**")
            if e["passage"]:
                st.markdown(f"> {e['passage']}")
    if r.get("corrected"):
        x = r["corrected"]
        st.caption(
            f"Self-correction: evidence was weak ({x['strength_before']}); "
            f'retried as "{x["query"]}" ({x["strength_after"]}).'
        )
    st.caption(
        "Latency: "
        + ", ".join(f"{k[:-3]} {val:.0f} ms" for k, val in r["timings"].items())
    )
