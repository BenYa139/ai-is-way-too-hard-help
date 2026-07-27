import streamlit as st
import speech_recognition as sr
import tempfile
import os
import datetime
import numpy as np

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

st.set_page_config(page_title="MindCheck", page_icon="🧠", layout="centered")

# ─────────────────────────────────────────────
# CONTENT
# Cognitive test adapted from MoCA (Nasreddine et al., 2005)
# Speech pause analysis based on Cohen et al. (2026) & Lin et al. (2025)
# ─────────────────────────────────────────────
WORDS = ["face", "velvet", "church", "daisy", "red"]
SENTENCES = [
    "I only know that John is the one to help today",
    "The cat always hid under the couch when dogs were in the room",
]
KEYWORDS = {
    "vehicle":   ["vehicle", "transport", "transportation", "wheels", "ride", "travel", "move"],
    "furniture": ["furniture", "wood", "sit", "house", "home"],
    "watch":     ["watch"],
    "pen":       ["pen", "pencil"],
    "dog":       ["dog"],
}
MAX_SCORE = 30

# Keys that capture speech for pause analysis
SPEECH_KEYS = [
    "fwd", "bwd", "lang1", "lang2",
    "abs1_widget", "abs2_widget",
    "ori_day_name", "ori_date", "ori_month", "ori_year", "ori_season",
    "ori_city",
    "fluency_animals",
    "calc_serial7",
    "naming_watch", "naming_pen", "naming_dog",
    "recall_widget",
]

def current_context():
    now = datetime.datetime.now()
    month = now.month
    season = ("winter" if month in (12,1,2) else
              "spring" if month in (3,4,5) else
              "summer" if month in (6,7,8) else "fall")
    return {
        "day": now.day,
        "weekday": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": now.year,
        "season": season,
    }

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name
    try:
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            data = r.record(source)
        return r.recognize_google(data, language="en-US")
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        st.error(f"Speech service error: {e}")
        return None
    finally:
        os.unlink(path)

def similarity_score(spoken, reference):
    spoken_clean = spoken.lower().replace(" ", "")
    matches = sum(1 for w in reference.split() if w.lower() in spoken_clean)
    return matches / max(len(reference.split()), 1)

def digits_in_order(text, sequence):
    return [ch for ch in text if ch.isdigit()] == sequence

def contains_any(text, keywords):
    import re
    text_l = text.lower()
    for k in keywords:
        if re.search(r"\b" + re.escape(k.lower()) + r"\b", text_l):
            return True
    return False

def analyze_pause_ratio(audio_bytes):
    """
    Returns the ratio of silent frames within mid-speech (0.0 to 1.0).
    Based on methodology from Cohen et al. (2026) and Lin et al. (2025).
    """
    import wave
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if not raw or framerate == 0:
            return 0.0
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if not dtype:
            return 0.0
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if sampwidth == 1:
            samples -= 128.0
        if n_channels > 1:
            usable = (samples.size // n_channels) * n_channels
            samples = samples[:usable].reshape(-1, n_channels).mean(axis=1)
        frame_len = max(int(framerate * 30 / 1000), 1)
        n_full = samples.size // frame_len
        if n_full < 2:
            return 0.0
        frames = samples[:n_full * frame_len].reshape(n_full, frame_len)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        peak = float(np.max(rms))
        if peak == 0:
            return 0.0
        threshold = peak * 0.08
        voiced = np.nonzero(rms >= threshold)[0]
        if voiced.size == 0:
            return 0.0
        inner = rms[voiced[0]:voiced[-1] + 1]
        return int(np.sum(inner < threshold)) / len(inner) if len(inner) else 0.0
    except Exception:
        return 0.0
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

def get_average_pause_ratio():
    """Average pause ratio across all speech responses."""
    ratios = []
    for key in SPEECH_KEYS:
        ratio = st.session_state.get(f"{key}_pause_ratio")
        if ratio is not None:
            ratios.append(ratio)
    return sum(ratios) / len(ratios) if ratios else None

def pause_level(ratio):
    """Classify pause ratio into Normal / Elevated / High."""
    if ratio is None:
        return None
    if ratio < 0.20:
        return "normal"
    elif ratio < 0.40:
        return "elevated"
    else:
        return "high"

def get_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")

def ai_grade(question_text, spoken_answer):
    if not ANTHROPIC_SDK_AVAILABLE:
        return None
    api_key = get_api_key()
    if not api_key or not spoken_answer.strip():
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{"role": "user", "content":
                f'Grade this spoken answer. Reply YES or NO only.\nQuestion: "{question_text}"\nAnswer: "{spoken_answer}"'}],
        )
        return response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return None

def voice_input(key):
    """Record audio, transcribe, and store raw pause ratio separately."""
    audio = st.audio_input("🎙️ Record your answer", key=key)
    if audio is not None:
        audio_bytes = audio.read()
        if audio_bytes != st.session_state.get(f"{key}_bytes"):
            st.session_state[f"{key}_bytes"] = audio_bytes
            with st.spinner("Transcribing…"):
                result = transcribe_audio(audio_bytes)
            st.session_state[f"{key}_text"] = result or ""
            if result is None:
                st.error("Could not recognise — please try again.")
            # Store raw pause ratio (used for speech analysis, NOT for adjusting MoCA score)
            ratio = analyze_pause_ratio(audio_bytes)
            st.session_state[f"{key}_pause_ratio"] = ratio
    return st.session_state.get(f"{key}_text", "")

def show_answer(text):
    st.success(f"You said: **{text}**")

# ─────────────────────────────────────────────
# STEP FUNCTIONS (MoCA-aligned)
# ─────────────────────────────────────────────

def step_word_memory():
    st.subheader("📋 Word Memory")
    st.caption("MoCA Domain: Memory — Nasreddine et al., 2005")
    st.write("Memorize these words — you'll be asked again later:")
    cols = st.columns(len(WORDS))
    for i, w in enumerate(WORDS):
        cols[i].markdown(
            f"<div style='text-align:center;font-size:1.4rem;font-weight:700;"
            f"padding:1rem;background:#f0f4ff;border-radius:12px;'>{w}</div>",
            unsafe_allow_html=True
        )
    st.info("Read these carefully, then press **Next**.")

def step_forward_digits():
    st.subheader("🔢 Forward Digit Span")
    st.caption("MoCA Domain: Attention — Nasreddine et al., 2005")
    st.write("Say these numbers in the same order:")
    st.markdown(
        "<div style='font-size:2rem;font-weight:700;letter-spacing:0.5rem;"
        "text-align:center;padding:1.5rem;background:#f0f4ff;border-radius:12px;'>"
        "2 – 1 – 8 – 5 – 4</div>", unsafe_allow_html=True
    )
    ans = voice_input("fwd")
    if ans:
        show_answer(ans)
        ok = digits_in_order(ans, ["2","1","8","5","4"])
        st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
        st.session_state["fwd_ok"] = ok

def step_backward_digits():
    st.subheader("🔢 Backward Digit Span")
    st.caption("MoCA Domain: Attention — Nasreddine et al., 2005")
    st.write("Say 7 – 4 – 2 in **reverse** order:")
    st.markdown(
        "<div style='font-size:2rem;font-weight:700;letter-spacing:0.5rem;"
        "text-align:center;padding:1.5rem;background:#f0f4ff;border-radius:12px;'>"
        "7 – 4 – 2</div>", unsafe_allow_html=True
    )
    ans = voice_input("bwd")
    if ans:
        show_answer(ans)
        ok = digits_in_order(ans, ["2","4","7"])
        st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
        st.session_state["bwd_ok"] = ok

def make_sentence_step(i, sentence):
    def step():
        st.subheader(f"🗣️ Sentence Repetition ({i}/2)")
        st.caption("MoCA Domain: Language — Nasreddine et al., 2005")
        st.write("Repeat this sentence exactly:")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;font-style:italic;'>\"{sentence}\"</div>",
            unsafe_allow_html=True
        )
        ans = voice_input(f"lang{i}")
        if ans:
            show_answer(ans)
            sc = similarity_score(ans, sentence)
            st.progress(sc, text=f"Match: {sc:.0%}")
            st.session_state[f"lang{i}_score"] = sc
    return step

def make_abstraction_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🧩 Abstraction ({n}/2)")
        st.caption("MoCA Domain: Abstraction — Nasreddine et al., 2005")
        st.write("How are these two things similar?")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(f"abs{n}_widget")
        if ans:
            show_answer(ans)
            ok = ai_grade(question, ans)
            if ok is None:
                ok = contains_any(ans, KEYWORDS[kw_key])
            st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
            st.session_state[f"abs{n}_correct"] = ok
    return step

def make_ori_time_step(question, key, check_fn, truth):
    def step():
        st.subheader("🕐 Orientation to Time")
        st.caption("MoCA Domain: Orientation — Nasreddine et al., 2005")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            ok = check_fn(ans)
            st.write("✅ Correct! (1/1 pt)" if ok else f"❌ It's actually: **{truth}** (0/1 pt)")
            st.session_state[f"{key}_ok"] = ok
    return step

def step_ori_place():
    def step():
        st.subheader("📍 Orientation to Place")
        st.caption("MoCA Domain: Orientation — Nasreddine et al., 2005")
        st.caption("Your answer is recorded for a reviewer to verify.")
        st.markdown(
            "<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            "border-radius:12px;'>What city or town are you in right now?</div>",
            unsafe_allow_html=True
        )
        ans = voice_input("ori_city")
        if ans:
            show_answer(ans)
            st.session_state["ori_city_answered"] = True
    return step

def step_fluency_animals():
    st.subheader("🦁 Verbal Fluency")
    st.caption("MoCA Domain: Language — Nasreddine et al., 2005")
    st.write("⏱️ Name as many **animals** as you can in 1 minute:")
    ans = voice_input("fluency_animals")
    if ans:
        show_answer(ans)
        count = len(set(ans.lower().split()))
        st.write(f"📊 Approximate word count: **{count}** (need ≥11 for full point)")
        st.session_state["fluency_animals_count"] = count

def step_calculation():
    st.subheader("🧮 Calculation")
    st.caption("MoCA Domain: Attention — Nasreddine et al., 2005")
    st.write("Starting at 100, keep subtracting 7 and say **5 results** in a row:")
    st.markdown(
        "<div style='font-size:1.4rem;text-align:center;padding:1rem;"
        "background:#f0f4ff;border-radius:12px;'>100 → 93 → 86 → 79 → 72 → 65</div>",
        unsafe_allow_html=True
    )
    ans = voice_input("calc_serial7")
    if ans:
        show_answer(ans)
        expected = ["93","86","79","72","65"]
        spoken = [t for t in ans.replace(",", " ").split() if t.isdigit()]
        correct = sum(1 for e in expected if e in spoken)
        pts = 3 if correct >= 4 else 2 if correct == 3 else 1 if correct in (1,2) else 0
        st.write(f"📊 {correct}/5 correct → **{pts}/3 pts**")
        st.session_state["calc_correct_count"] = correct
        st.session_state["calc_pts"] = pts

def make_naming_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🏷️ Naming ({n}/3)")
        st.caption("MoCA Domain: Language — Nasreddine et al., 2005")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            ok = ai_grade(question, ans)
            if ok is None:
                ok = contains_any(ans, KEYWORDS[kw_key])
            st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
            st.session_state[f"{key}_ok"] = ok
    return step

def step_delayed_recall():
    st.subheader("🧠 Delayed Recall")
    st.caption("MoCA Domain: Memory — Nasreddine et al., 2005")
    st.write("Say as many words as you can remember from the very beginning:")
    ans = voice_input("recall_widget")
    if ans:
        show_answer(ans)
        found = [w for w in WORDS if w.lower() in ans.lower()]
        pct = len(found) / len(WORDS)
        st.progress(pct, text=f"Accuracy: {pct:.0%}")
        st.write(f"📊 {len(found)} / {len(WORDS)} words ({len(found)}/5 pts)")
        st.session_state["recall_score_count"] = len(found)

def step_results():
    st.subheader("📊 Results")

    # ── COMPUTE MOCA SCORE ──────────────────────
    score = 0
    details = []

    fwd_pt = 1 if st.session_state.get("fwd_ok") else 0
    score += fwd_pt
    details.append(f"{'✅' if fwd_pt else '❌'} Forward Digits: {fwd_pt}/1 pt")

    bwd_pt = 1 if st.session_state.get("bwd_ok") else 0
    score += bwd_pt
    details.append(f"{'✅' if bwd_pt else '❌'} Backward Digits: {bwd_pt}/1 pt")

    for i in range(1, 3):
        s = st.session_state.get(f"lang{i}_score", 0.0)
        pt = 1 if s >= 0.6 else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Sentence {i}: {pt}/1 pt ({s:.0%} match)")

    for i in range(1, 3):
        pt = 1 if st.session_state.get(f"abs{i}_correct") else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Abstraction {i}: {pt}/1 pt")

    for key in ["ori_day_name","ori_date","ori_month","ori_year","ori_season"]:
        pt = 1 if st.session_state.get(f"{key}_ok") else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Time ({key}): {pt}/1 pt")

    pt = 1 if st.session_state.get("ori_city_answered") else 0
    score += pt
    details.append(f"{'✅' if pt else '❌'} Place (city): {pt}/1 pt")

    count = st.session_state.get("fluency_animals_count", 0)
    pt = 1 if count >= 11 else 0
    score += pt
    details.append(f"{'✅' if pt else '❌'} Verbal Fluency: {count} animals → {pt}/1 pt")

    calc_pts = st.session_state.get("calc_pts", 0)
    score += calc_pts
    details.append(f"🧮 Calculation: {calc_pts}/3 pts")

    for key in ["naming_watch","naming_pen","naming_dog"]:
        pt = 1 if st.session_state.get(f"{key}_ok") else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Naming ({key}): {pt}/1 pt")

    found = st.session_state.get("recall_score_count", 0)
    score += found
    details.append(f"🧠 Delayed Recall: {found}/5 pts")

    # ── COMPUTE SPEECH PAUSE SCORE ───────────────
    avg_ratio = get_average_pause_ratio()
    p_level = pause_level(avg_ratio)

    # ── DETERMINE COGNITIVE RISK (MoCA) ──────────
    if score >= 26:
        cog_level = "normal"
        cog_label = "🟢 Normal"
        cog_desc = "No significant cognitive signs detected"
    elif score >= 18:
        cog_level = "mild"
        cog_label = "🟡 Mild"
        cog_desc = "Some early cognitive signs may be present"
    else:
        cog_level = "high"
        cog_label = "🔴 High"
        cog_desc = "Significant cognitive signs detected"

    # ── DETERMINE SPEECH RISK ─────────────────────
    if p_level == "normal" or p_level is None:
        speech_level = "normal"
        speech_label = "🟢 Normal"
        speech_desc = "No elevated pause patterns detected"
        speech_pct = f"{avg_ratio:.0%}" if avg_ratio is not None else "N/A"
    elif p_level == "elevated":
        speech_level = "elevated"
        speech_label = "🟡 Elevated"
        speech_desc = "Some elevated pause patterns detected"
        speech_pct = f"{avg_ratio:.0%}"
    else:
        speech_level = "high"
        speech_label = "🔴 High"
        speech_desc = "Significantly elevated pause patterns detected"
        speech_pct = f"{avg_ratio:.0%}"

    # ── COMBINED RISK (2x2 matrix) ────────────────
    if cog_level == "normal" and speech_level == "normal":
        risk_color = "#d4edda"
        risk_icon = "🟢"
        risk_title = "Low Risk"
        risk_desc = "No early signs of Alzheimer's detected in cognitive performance or speech patterns."
    elif cog_level == "normal" and speech_level == "elevated":
        risk_color = "#fff3cd"
        risk_icon = "🟡"
        risk_title = "Moderate Risk"
        risk_desc = "Cognitive performance is normal, but early speech pause patterns may be present. Consider monitoring."
    elif cog_level == "mild" and speech_level == "normal":
        risk_color = "#fff3cd"
        risk_icon = "🟡"
        risk_title = "Moderate Risk"
        risk_desc = "Some early cognitive signs are present. Speech patterns appear normal. Consult a doctor."
    elif cog_level == "high" and speech_level == "normal":
        risk_color = "#fff3cd"
        risk_icon = "🟡"
        risk_title = "Moderate-High Risk"
        risk_desc = "Significant cognitive signs detected. Speech patterns appear normal. Please seek clinical evaluation."
    elif cog_level == "normal" and speech_level == "high":
        risk_color = "#fff3cd"
        risk_icon = "🟡"
        risk_title = "Moderate Risk"
        risk_desc = "Cognitive performance is normal, but elevated speech pause patterns detected. Monitor closely."
    else:
        # Both elevated/high
        risk_color = "#f8d7da"
        risk_icon = "🔴"
        risk_title = "High Risk"
        risk_desc = "Both cognitive performance and speech pause patterns indicate early signs of Alzheimer's. Please seek clinical evaluation immediately."

    # ── DISPLAY RESULTS ───────────────────────────
    st.markdown("---")

    # Two indicators side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style='text-align:center;padding:1.5rem;background:#f0f4ff;
            border-radius:16px;margin-bottom:1rem;'>
                <div style='font-size:0.85rem;color:#666;margin-bottom:0.5rem;'>
                    🧠 Cognitive Score (MoCA)
                </div>
                <div style='font-size:2.5rem;font-weight:700;'>{score}/{MAX_SCORE}</div>
                <div style='font-size:1rem;margin-top:0.5rem;'>{cog_label}</div>
                <div style='font-size:0.8rem;color:#555;margin-top:0.25rem;'>{cog_desc}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='text-align:center;padding:1.5rem;background:#f0f4ff;
            border-radius:16px;margin-bottom:1rem;'>
                <div style='font-size:0.85rem;color:#666;margin-bottom:0.5rem;'>
                    🎙️ Speech Pause Analysis
                </div>
                <div style='font-size:2.5rem;font-weight:700;'>{speech_pct}</div>
                <div style='font-size:1rem;margin-top:0.5rem;'>{speech_label}</div>
                <div style='font-size:0.8rem;color:#555;margin-top:0.25rem;'>{speech_desc}</div>
            </div>
        """, unsafe_allow_html=True)

    # Combined risk result
    st.markdown(f"""
        <div style='text-align:center;padding:2rem;background:{risk_color};
        border-radius:16px;margin:1rem 0;'>
            <div style='font-size:1rem;color:#444;margin-bottom:0.5rem;'>Overall Early Alzheimer's Risk</div>
            <div style='font-size:2rem;font-weight:700;'>{risk_icon} {risk_title}</div>
            <div style='font-size:0.95rem;color:#555;margin-top:0.75rem;'>{risk_desc}</div>
        </div>
    """, unsafe_allow_html=True)

    # Score breakdown
    with st.expander("📋 Cognitive Score Breakdown"):
        for d in details:
            st.write(d)

    if avg_ratio is not None:
        with st.expander("📋 Speech Pause Breakdown"):
            st.write(f"Average pause ratio across all responses: **{avg_ratio:.1%}**")
            st.write("Thresholds: Normal < 20% | Elevated 20–40% | High > 40%")
            st.caption("Based on: Cohen et al. (2026), *Journal of the International Neuropsychological Society*, 32(1), 24-31; Lin et al. (2025), *Alzheimer's & Dementia*, doi:10.1002/alz.086309")

    st.markdown("---")
    st.caption("⚠️ This tool is a prototype and NOT a medical diagnosis. Please consult a qualified clinician for formal assessment.")
    st.caption("📚 Cognitive scoring adapted from: Nasreddine ZS et al. (2005). *Journal of the American Geriatrics Society*, 53(4), 695-699.")
    st.caption("📚 Speech pause analysis based on: Cohen et al. (2026). *Journal of the International Neuropsychological Society*, 32(1), 24-31 | Lin et al. (2025). *Alzheimer's & Dementia*. doi:10.1002/alz.086309")

    if st.button("🔁 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["step"] = 0
        st.rerun()

# ─────────────────────────────────────────────
# BUILD STEPS LIST
# ─────────────────────────────────────────────
ctx = current_context()

STEPS = [
    step_word_memory,
    step_forward_digits,
    step_backward_digits,
    make_sentence_step(1, SENTENCES[0]),
    make_sentence_step(2, SENTENCES[1]),
    make_abstraction_step(1, "How are a Train and a Bicycle similar?", "abs1", "vehicle"),
    make_abstraction_step(2, "How are a Table and a Chair similar?", "abs2", "furniture"),
    make_ori_time_step("What day of the week is it today?", "ori_day_name",
                       lambda a: ctx["weekday"].lower() in a.lower(), ctx["weekday"]),
    make_ori_time_step("What is today's date?", "ori_date",
                       lambda a: str(ctx["day"]) in a, ctx["day"]),
    make_ori_time_step("What month is it?", "ori_month",
                       lambda a: ctx["month"].lower() in a.lower(), ctx["month"]),
    make_ori_time_step("What year is it?", "ori_year",
                       lambda a: str(ctx["year"]) in a, ctx["year"]),
    make_ori_time_step("What season is it?", "ori_season",
                       lambda a: ctx["season"].lower() in a.lower(), ctx["season"]),
    step_ori_place(),
    step_fluency_animals,
    step_calculation,
    make_naming_step(1, "What do you call the object worn on the wrist that tells time?", "naming_watch", "watch"),
    make_naming_step(2, "What do you call the object used for writing?", "naming_pen", "pen"),
    make_naming_step(3, "What do you call the pet that barks and guards the house?", "naming_dog", "dog"),
    step_delayed_recall,
    step_results,
]

TOTAL_STEPS = len(STEPS)

if "step" not in st.session_state:
    st.session_state["step"] = 0

step_idx = st.session_state["step"]

st.markdown("""
    <style>
    div[data-testid="stAudioInput"] { transform: scale(1.3); transform-origin: left top; margin-bottom: 24px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**🧠 MindCheck**")
    st.caption("Cognitive + Speech Analysis")
    st.progress(step_idx / (TOTAL_STEPS - 1), text=f"Step {step_idx + 1} / {TOTAL_STEPS}")

st.progress(step_idx / (TOTAL_STEPS - 1), text=f"Step {step_idx + 1} / {TOTAL_STEPS}")

STEPS[step_idx]()

if step_idx < TOTAL_STEPS - 1:
    st.write("")
    col1, col2 = st.columns([1, 2])
    with col1:
        if step_idx > 0:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state["step"] -= 1
                st.rerun()
    with col2:
        if st.button("➡️ Next", use_container_width=True, type="primary"):
            st.session_state["step"] += 1
            st.rerun()
