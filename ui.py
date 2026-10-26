"""Gradio UI.  Run:  python ui.py  ->  http://localhost:7860"""
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from app import db, rag
from app.config import settings

EXAMPLES = [
    ["DA kitna badha hai 2024 mein?", "rewrite", "hybrid", False],
    ["शिक्षकों के तबादले के लिए आवेदन कब तक कर सकते हैं?", "rewrite", "hybrid", False],
    ["Ration card e-KYC ki last date kya hai?", "rewrite", "hybrid", False],
    ["What is the travel allowance per day for class 1 officers?", "rewrite", "hybrid", False],
    ["garmi mein school ka time kya rahega?", "rewrite", "hybrid", False],
]


def ask(question: str, strategy: str, mode: str, rerank: bool):
    if not question.strip():
        return "Please type a question.", ""
    try:
        r = rag.answer(question, strategy, mode, rerank=rerank)
    except Exception as e:
        return f"**Error:** {e}", ""
    meta = (f"_language: **{r.language}** · strategy: **{r.strategy}** · mode: **{r.mode}** · "
            f"rerank: **{'on' if rerank else 'off'}** · search query: `{r.search_query}` · "
            f"LLM: **{r.llm_provider or '?'}**_")
    if r.timings:
        meta += "  \n_time: " + " · ".join(f"{name} {sec:.1f}s" for name, sec in r.timings.items()) + \
                f" · **total {sum(r.timings.values()):.1f}s**_"
    lines = []
    for s in r.sources:
        head = " · ".join(x for x in [s["subject"], s["order_no"] and f"क्रमांक {s['order_no']}",
                                       s["order_date"] and f"दिनांक {s['order_date']}"] if x)
        lines.append(f"**[{s['n']}] {head}**  \n`{s['source']}` p.{s['page']}\n> {s['snippet']}")
    return f"{r.answer}\n\n{meta}", "\n\n".join(lines)


def list_docs():
    try:
        rows = db.list_documents(settings.collection)
    except Exception as e:
        return [[f"DB error: {e}", "", "", "", ""]]
    return [[d["source"], d["order_no"] or "", str(d["order_date"] or ""), d["subject"] or "",
             ", ".join(sorted(set(d["methods"] or [])))] for d in rows]


def upload(files, progress=gr.Progress()):
    """Index uploaded PDFs one by one, with a progress bar (scanned pages are OCR'd: slow on CPU)."""
    if not files:
        return "No file selected.", list_docs()
    from app.ingest import ingest_folder

    lines = []
    for i, f in enumerate(files):
        name = Path(f).name
        progress(i / len(files), desc=f"{i + 1}/{len(files)} {name}: reading / OCR + indexing...")
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(f, Path(tmp) / name)
            s = ingest_folder(Path(tmp), settings.collection, ocr=settings.ocr_enabled, quiet=False)
        methods = ", ".join(f"{m} {n}" for m, n in s["methods"].items()) or "no text found"
        lines.append(f"- `{name}`: {s['chunks']} chunks (pages: {methods})")
    progress(1.0, desc="done")
    return (f"**Indexed {len(files)} file(s)** into collection `{settings.collection}`. "
            f"You can ask about them in the Ask tab now.\n" + "\n".join(lines)), list_docs()


with gr.Blocks(title="Aadesh Khoj") as demo:
    gr.Markdown(
        "# Aadesh Khoj · आदेश खोज\n"
        "Ask about government orders in **Hindi, Hinglish or English**. Works even when the PDFs are in "
        "old **Kruti Dev** fonts or scanned. Answers cite the order number and date.\n\n"
        "_Always verify with the latest official order._"
    )
    with gr.Tab("Ask"):
        with gr.Row():
            q = gr.Textbox(label="Question", placeholder="e.g. chhutti ki list 2024", scale=4)
            strat = gr.Radio(list(rag.STRATEGIES), value="rewrite", label="Query strategy")
            mode = gr.Radio(list(rag.MODES), value="hybrid", label="Search mode")
            rerank = gr.Checkbox(value=settings.rerank_enabled, label="Rerank (slower, more accurate)")
        btn = gr.Button("Ask", variant="primary")
        ans = gr.Markdown()
        with gr.Accordion("Sources", open=False):
            srcs = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[q, strat, mode, rerank])
        btn.click(ask, [q, strat, mode, rerank], [ans, srcs])
        q.submit(ask, [q, strat, mode, rerank], [ans, srcs])
    with gr.Tab("Documents"):
        table = gr.Dataframe(headers=["file", "order no", "date", "subject", "extraction"], value=list_docs,
                             interactive=False, wrap=True)
        up = gr.File(label="Add PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
        status = gr.Markdown()
        up.upload(upload, up, [status, table])

if __name__ == "__main__":
    demo.launch()
