"""EcoSort AI: professional web demo for six-class RGB polymer screening.

This app provides *visual estimates*; it does not chemically verify polymers.
Requires user-generated model artifacts in ./artifacts/.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
MODEL_FILE = ARTIFACTS / "ecosort_resin_classifier.keras"
CLASSES_FILE = ARTIFACTS / "class_names.json"
THRESHOLD_FILE = ARTIFACTS / "decision_threshold.json"

POLYMERS = {
    "PET": ("Polyethylene terephthalate", "1", "Clean, separately collected PET is often suitable for mechanical recycling when a local recycler accepts it."),
    "HDPE": ("High-density polyethylene", "2", "Consider mechanical recycling when the item is clean and the local collection stream accepts it."),
    "PVC": ("Polyvinyl chloride", "3", "Keep separate for specialist review. Do not infer suitability for pyrolysis from an image."),
    "LDPE": ("Low-density polyethylene", "4", "Check whether a clean film/plastic-bag recycling stream exists locally; multilayer materials require review."),
    "PP": ("Polypropylene", "5", "Consider mechanical recycling if clean and accepted by local facilities."),
    "PS": ("Polystyrene", "6", "Check for a suitable specialist recycling route; contamination and form affect handling."),
}
OPTIONS = ["Not visible / uncertain"] + [f"{value[1]} — {label}" for label, value in POLYMERS.items()] + ["7 — Other / multilayer"]

st.set_page_config(page_title="EcoSort AI | Plastic Waste Screening", page_icon="♻️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
  :root {--ink:#16333b;--forest:#087f70;--muted:#66767b;--border:#dfe9e5;--paper:#f7faf9;}
  html, body, [class*=css], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;color:var(--ink)}
  [data-testid="stAppViewContainer"] {background:var(--paper)}
  [data-testid="stMainBlockContainer"] {max-width:1150px; padding-top:2.3rem; padding-bottom:3rem}
  h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.045em;color:var(--ink)}
  div[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:var(--border)}
  .ec-top {display:flex;align-items:center;justify-content:space-between;padding:0 0 22px;border-bottom:1px solid var(--border)}
  .ec-brand {font:800 22px 'Manrope',sans-serif;letter-spacing:-.055em;color:#14383b}
  .ec-logo {display:inline-flex;align-items:center;justify-content:center;background:#d4f1e9;color:#0d7c6c;border-radius:12px;width:37px;height:37px;margin-right:10px;font-size:22px}
  .ec-chip {font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#287f72;background:#e3f2ec;padding:8px 12px;border-radius:20px;border:1px solid #cce8db}
  .ec-hero {padding:38px 0 22px;max-width:870px}
  .ec-eyebrow {font-weight:800;font-size:12px;letter-spacing:.13em;color:#087f70;text-transform:uppercase;margin-bottom:10px}
  .ec-title {font:800 clamp(31px,4vw,52px)/1.15 'Manrope',sans-serif;letter-spacing:-.055em;color:#14343b;margin-bottom:14px}
  .ec-sub {font-size:16px;line-height:1.65;color:#657a7c;max-width:790px}
  .ec-panel {background:white;border:1px solid var(--border);border-radius:18px;padding:20px 22px;margin-bottom:14px}
  .ec-kicker {color:var(--forest);font-size:11px;letter-spacing:.1em;font-weight:800;text-transform:uppercase}
  .ec-panel-title {font:800 18px 'Manrope',sans-serif;margin:5px 0 12px}
  .ec-note {font-size:13px;color:var(--muted);line-height:1.6}
  .ec-label {font-size:12px;text-transform:uppercase;font-weight:800;letter-spacing:.06em;color:var(--muted)}
  .ec-status {display:inline-block;padding:7px 11px;font-size:12px;font-weight:700;border-radius:999px;border:1px solid #baddcd;background:#effaf4;color:#157251}
  .ec-warn {display:inline-block;padding:7px 11px;font-size:12px;font-weight:700;border-radius:999px;border:1px solid #efd9a8;background:#fff9e9;color:#935e14}
  .ec-result-name {font:800 38px 'Manrope',sans-serif;letter-spacing:-.05em;margin:10px 0 0}
  .ec-result-full {color:#647b7f;font-size:15px;margin-bottom:9px}
  .ec-metric {padding:21px;background:white;border:1px solid var(--border);border-radius:16px}
  .ec-metric-value {font:800 30px 'Manrope',sans-serif;letter-spacing:-.055em;margin:6px 0}
  .ec-divider {border-top:1px solid var(--border);margin:20px 0}
  .ec-footer {border-top:1px solid var(--border);margin-top:35px;padding-top:18px;font-size:12px;color:#718287;line-height:1.5}
  div[data-testid="stButton"] button[kind="primary"] {background:#087f70;border-color:#087f70;color:white;font-weight:700;border-radius:11px}
  div[data-testid="stButton"] button[kind="primary"]:hover {background:#066c60;border-color:#066c60}
  div[data-testid="stTabs"] button {font-weight:700}
  @media(max-width:768px){.ec-top{align-items:flex-start}.ec-chip{font-size:9px;padding:7px}.ec-hero{padding:24px 0 8px}.ec-panel{padding:14px}}
</style>
""", unsafe_allow_html=True)


def panel(number: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="ec-panel"><div class="ec-kicker">{escape(number)}</div>'
        f'<div class="ec-panel-title">{escape(title)}</div>'
        f'<div class="ec-note">{escape(description)}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading EcoSort model…")
def load_artifacts():
    import tensorflow as tf
    if not (MODEL_FILE.is_file() and CLASSES_FILE.is_file()):
        raise FileNotFoundError("Required model artifacts not found")
    names = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) < 2 or any(n not in POLYMERS for n in names):
        raise ValueError("Invalid classes in class_names.json")
    config = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8")) if THRESHOLD_FILE.exists() else {}
    threshold = float(config.get("threshold", .63))
    margin = float(config.get("margin", .10))
    # compile=False avoids needing training-time optimizer/loss objects on deployment.
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    return model, names, threshold, margin


def open_image(raw: bytes) -> Image.Image:
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("Image exceeds the 15 MB limit")
    with Image.open(io.BytesIO(raw)) as handle:
        img = ImageOps.exif_transpose(handle).convert("RGB")
    if img.width * img.height > 40_000_000:
        raise ValueError("Image resolution is too large")
    return img


def image_batch(images: list[Image.Image]) -> np.ndarray:
    # MATCH THE COLAB TRAINING NOTEBOOK: tf.image.resize(..., (224,224)).
    # Do not use ImageOps.pad: the earlier Streamlit version introduced black
    # borders at inference, while training images were stretched to 224x224.
    import tensorflow as tf
    batch = tf.stack([tf.image.resize(np.asarray(im, dtype=np.float32), (224, 224)) for im in images])
    return batch.numpy()


def fingerprint(payloads: list[bytes]) -> str:
    digest = hashlib.sha256()
    for p in payloads:
        digest.update(len(p).to_bytes(8, "little"))
        digest.update(p)
    return digest.hexdigest()


st.markdown('<div class="ec-top"><div class="ec-brand"><span class="ec-logo">♻</span>EcoSort <span style="color:#0a8b77">AI</span></div><span class="ec-chip">Research demonstration · v2.0</span></div>', unsafe_allow_html=True)
st.markdown('<div class="ec-hero"><div class="ec-eyebrow">Plastic waste · Computer vision</div><div class="ec-title">Smarter identification.<br>More responsible sorting.</div><div class="ec-sub">Upload or photograph a plastic item to obtain a six-class visual material estimate, uncertainty screening and conservative recovery guidance.</div></div>', unsafe_allow_html=True)

analyze_tab, results_tab, research_tab = st.tabs(["Identify material", "Evaluation & limitations", "About the research"])

with analyze_tab:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        panel("01 · Capture", "Provide an image", "Use your camera or upload up to three views of the same physical item. Good lighting and a simple background help.")
        source = st.radio("Input method", ["Upload photographs", "Take a camera photo"], horizontal=True, label_visibility="collapsed")
        input_items = []
        if source == "Upload photographs":
            uploaded = st.file_uploader("Select 1–3 images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
            input_items = list(uploaded or [])[:3]
            if uploaded and len(uploaded) > 3:
                st.warning("Only the first three images will be analyzed.")
        else:
            captured = st.camera_input("Take a photo of one plastic item")
            if captured is not None:
                input_items = [captured]
        image_payloads = [f.getvalue() for f in input_items]
        image_objects = []
        for i, raw in enumerate(image_payloads):
            try:
                image_objects.append(open_image(raw))
            except Exception as exc:
                st.error(f"Image {i + 1} could not be opened: {exc}")
                image_objects = []
                break
        if image_objects:
            preview_columns = st.columns(min(len(image_objects), 3), gap="small")
            for i, picture in enumerate(image_objects):
                preview_columns[i % len(preview_columns)].image(picture, caption=f"View {i + 1}", use_container_width=True)
    with right:
        panel("02 · Evidence", "Visible material code", "A resin identification code printed on the item provides useful additional evidence. Leave the field unselected if it is not clearly legible.")
        code_text = st.selectbox("Resin identification code (optional)", OPTIONS, index=0)
        st.caption("Code selection is **user-reported**; this application does not verify the printed marking automatically.")
        panel("03 · Interpret", "Evidence-aware assessment", "The app flags low-scoring or contradictory predictions for manual review. It does not establish chemical composition or industrial pyrolysis suitability.")
    run = st.button("Analyze plastic item", type="primary", use_container_width=True, disabled=(len(image_objects)==0))

    current_fingerprint = fingerprint(image_payloads) if image_objects else None
    if run:
        if current_fingerprint:
            try:
                model, labels, cutoff, min_margin = load_artifacts()
                with st.spinner("Running visual material screening…"):
                    batch = image_batch(image_objects)
                    scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
                    if scores.shape != (len(image_objects), len(labels)):
                        raise RuntimeError("Model output shape does not match class_names.json")
                    means = scores.mean(axis=0)
                    order = means.argsort()[::-1]
                    top, second = int(order[0]), int(order[1])
                    result = {"visual_class": labels[top], "score": float(means[top]),
                              "gap": float(means[top] - means[second]), "scores": {labels[i]: float(means[i]) for i in range(len(labels))},
                              "accepted": bool(means[top] >= cutoff and means[top] - means[second] >= min_margin),
                              "cutoff": cutoff, "margin_cutoff": min_margin,
                              "views": len(image_objects)}
                    st.session_state["ecosort_last_result"] = {"fingerprint": current_fingerprint, "data": result}
            except FileNotFoundError:
                st.error("Trained model missing. Copy the exported .keras model and class_names.json into the artifacts/ directory. See DEPLOY_GUIDE.md.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    stored = st.session_state.get("ecosort_last_result")
    if stored is not None and stored.get("fingerprint") == current_fingerprint and current_fingerprint:
        res = stored["data"]
        predicted = res["visual_class"]
        score = res["score"]
        code = code_text.split(" — ")[0] if code_text != OPTIONS[0] else None
        code_map = {details[1]: label for label, details in POLYMERS.items()}
        user_reported_class = code_map.get(code)
        disagreement = bool(user_reported_class and predicted != user_reported_class)
        is_other = code == "7"
        needs_review = not res["accepted"] or disagreement or is_other

        st.markdown('<div class="ec-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ec-kicker">Screening report</div>', unsafe_allow_html=True)
        a, b = st.columns([1.05, .95], gap="large")
        with a:
            tag = '<span class="ec-warn">Manual verification required</span>' if needs_review else '<span class="ec-status">Model passed screening threshold</span>'
            formal_name = POLYMERS.get(predicted, ("Unknown material", "—", ""))[0]
            st.markdown('<div class="ec-panel"><div class="ec-label">Visual model estimate</div>'
                        f'<div class="ec-result-name">{escape(predicted)}</div><div class="ec-result-full">{escape(formal_name)}</div>'
                        f'{tag}<div class="ec-divider"></div><div class="ec-label">Top softmax score</div>'
                        f'<div class="ec-result-name">{score:.1%}</div>'
                        '<div class="ec-note">This score is not a calibrated probability of chemical polymer identity.</div></div>', unsafe_allow_html=True)
            if disagreement:
                st.warning(f"Your reported resin code suggests **{user_reported_class}**, but the image model predicts **{predicted}**. Please verify the printed marking and the material before making a routing decision.")
            elif is_other:
                st.warning("You selected resin code 7 / Other. The six-class model does not represent this material and must not overrule that information.")
            elif not res["accepted"]:
                st.warning("The model did not meet the screening score or top-two separation threshold. Manual review is recommended.")
            elif user_reported_class:
                st.info(f"User-reported printed code: **{code} ({user_reported_class})**. This was not independently verified by the application.")
        with b:
            st.markdown('<div class="ec-panel"><div class="ec-label">Top candidate classes</div><div class="ec-panel-title">Model output distribution</div></div>', unsafe_allow_html=True)
            for name, value in sorted(res["scores"].items(), key=lambda item: item[1], reverse=True):
                st.write(f"**{name}** · {value:.1%}")
                st.progress(min(max(value, 0.0), 1.0))
        st.markdown("### Preliminary recovery guidance")
        if needs_review:
            st.info("**Manual review recommended.** Avoid assigning this item to a recycling or thermal-treatment stream solely on the basis of this prediction.")
        else:
            st.info(POLYMERS[predicted][2])
        st.caption("Routing guidance assumes accurate material identification, appropriate cleanliness and local processing availability. No pyrolysis suitability claim is made.")

        report = {"system": "EcoSort AI", "generated_at": datetime.now(timezone.utc).isoformat(),
                  "research_status": "Prototype; external field validation pending", "views": res["views"],
                  "visual_prediction": predicted, "model_softmax_score": score,
                  "top_two_margin": res["gap"], "individual_class_scores": res["scores"],
                  "score_threshold": res["cutoff"], "margin_threshold": res["margin_cutoff"],
                  "model_passed_screening": res["accepted"], "user_reported_visible_resin_code": code,
                  "reported_code_class": user_reported_class, "visual_code_disagreement": disagreement,
                  "manual_review": needs_review,
                  "disclaimer": "A standard RGB camera does not chemically verify polymer identity."}
        st.download_button("Download analysis report (.json)", data=json.dumps(report, indent=2),
                           file_name="ecosort_screening_report.json", mime="application/json")
    elif not MODEL_FILE.is_file():
        st.info("The professional interface is ready. To run predictions, add your trained model files to the `artifacts/` folder.")

with results_tab:
    st.markdown("### Preliminary model evaluation")
    st.caption("Metrics reproduced from the user's Colab results. They are not a guarantee of performance on new material streams.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out image accuracy", "86.7%")
    c2.metric("Macro F1", "0.879")
    c3.metric("Test images", "30")
    st.markdown("""The reported test set contained 30 images from **Roboflow Recy Clear Plastic Waste Type v1**. Some classes had only 3–7 test examples. Therefore, the evaluation is exploratory rather than a robust real-world accuracy estimate.

**Important limitations:**
- Public-source images have no reliable shared physical-item identifier, so exact deduplication cannot rule out near-duplicate or related samples across sets.
- Polymer labels in public RGB datasets have not been independently verified using laboratory testing in this project.
- Changes in lighting, backgrounds, labels and local waste streams can reduce accuracy.
- Model softmax scores should not be described as chemically calibrated certainty.
- An independent, physically verified test set from local plastic samples is still needed.
""")

with research_tab:
    st.markdown("### Research objective")
    st.write("EcoSort AI investigates whether low-cost RGB image screening combined with explicit uncertainty handling can support responsible plastic waste triage. The broader research program includes real-world sample collection, label verification and preliminary recovery decision support.")
    st.markdown("### Present development")
    st.markdown("""- Six-class EfficientNetB0 visual classifier (PET, HDPE, PVC, LDPE, PP, PS).
- Phone and laptop photo upload with optional camera snapshot.
- Multi-view prediction averaging and an uncertainty/manual-review rule.
- User-reported visible resin code treated as a separate evidence source.
- Local field validation and chemical/polymer label verification remain future work.
""")
    st.info("EcoSort does not claim that an image establishes polymer chemistry, recycling certification, contamination levels or safe thermal-processing suitability.")

st.markdown('<div class="ec-footer">EcoSort AI · Student research prototype · Preliminary plastic-waste screening only. Not a laboratory identification or waste-processing authorization system.</div>', unsafe_allow_html=True)
