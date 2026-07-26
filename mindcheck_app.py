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
# ─────────────────────────────────────────────
WORDS = ["face", "velvet", "church", "daisy", "red"]
SENTENCES = [
    "I only know that John is the one to help today",
    "The cat always hid under the couch when dogs were in the room",
    "The boy runs to the playground after school every day",
    "It rained so hard that the street in front of the house turned into a small river",
]
KEYWORDS = {
    "vehicle":     ["vehicle", "transport", "transportation", "wheels", "ride", "travel", "move"],
    "measurement": ["measure", "measurement", "tool", "instrument", "scale", "tell time", "numbers"],
    "fruit":       ["fruit", "eat", "sweet"],
    "furniture":   ["furniture", "wood", "sit", "house", "home"],
    "watch":       ["watch"],
    "pen":         ["pen", "pencil"],
    "dog":         ["dog"],
    "hammer":      ["hit", "nail", "pound", "strike", "hammer"],
    "scissors":    ["cut", "trim", "snip"],
    "proverb_hot": ["opportunity", "chance", "act quickly", "seize", "now"],
    "proverb_slow":["patience", "careful", "steady", "slow", "take your time"],
}

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

def pacing_factor(pause_ratio):
    return max(0.5, min(1.0, 1.0 - pause_ratio * 1.2))

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
            pause = analyze_pause_ratio(audio_bytes)
            st.session_state[f"{key}_pacing"] = pacing_factor(pause)
    text = st.session_state.get(f"{key}_text", "")
    if text:
        p = st.session_state.get(f"{key}_pacing", 1.0)
        st.caption(f"⏸️ Speech pacing: {p:.0%} — more pauses lower this answer's score.")
    return text

def get_pacing(key):
    return st.session_state.get(f"{key}_pacing", 1.0)

def show_answer(text):
    st.success(f"You said: **{text}**")

# ─────────────────────────────────────────────
# STEP FUNCTIONS
# ─────────────────────────────────────────────
def step_word_memory():
    st.subheader("📋 Word Memory")
    st.write("Memorize these words — you'll be asked again later:")
    cols = st.columns(len(WORDS))
    for i, w in enumerate(WORDS):
        cols[i].markdown(
            f"<div style='text-align:center;font-size:1.4rem;font-weight:700;"
            f"padding:1rem;background:#f0f4ff;border-radius:12px;'>{w}</div>",
            unsafe_allow_html=True
        )
    st.info("Read these carefully, then press **Next**.")

def step_immediate_recall():
    st.subheader("🔁 Immediate Recall")
    st.write("Say back all 5 words you just saw:")
    ans = voice_input("immediate_recall")
    if ans:
        show_answer(ans)
        found = [w for w in WORDS if w.lower() in ans.lower()]
        st.write(f"📊 {len(found)} / {len(WORDS)}")
        st.session_state["immediate_recall_count"] = len(found) * get_pacing("immediate_recall")

def step_forward_digits():
    st.subheader("🔢 Forward Digit Span")
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
        st.write("✅ Correct!" if ok else "❌ Not quite.")
        st.session_state["fwd_ok"] = ok

def step_backward_digits():
    st.subheader("🔢 Backward Digit Span")
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
        st.write("✅ Correct!" if ok else "❌ Not quite.")
        st.session_state["bwd_ok"] = ok

def make_sentence_step(i, sentence):
    def step():
        st.subheader(f"🗣️ Sentence Repetition ({i}/4)")
        st.write("Repeat this sentence exactly:")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;font-style:italic;'>\"{sentence}\"</div>",
            unsafe_allow_html=True
        )
        ans = voice_input(f"lang{i}")
        if ans:
            show_answer(ans)
            sc = similarity_score(ans, sentence) * get_pacing(f"lang{i}")
            st.progress(sc, text=f"Match: {sc:.0%}")
            st.session_state[f"lang{i}_score"] = sc
    return step

def make_abstraction_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🧩 Abstraction ({n}/4)")
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
            st.write("✅ Correct!" if ok else "❌ Not quite.")
            st.session_state[f"abs{n}_correct"] = ok
    return step

def make_ori_time_step(question, key, check_fn, truth):
    def step():
        st.subheader("🕐 Orientation to Time")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            ok = check_fn(ans)
            st.write("✅ Correct!" if ok else f"❌ It's actually: **{truth}**")
            st.session_state[f"{key}_ok"] = ok
    return step

def make_ori_place_step(question, key):
    def step():
        st.subheader("📍 Orientation to Place")
        st.caption("Your answer is recorded for a reviewer to verify.")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            st.session_state[f"{key}_answered"] = True
    return step

def step_fluency_animals():
    st.subheader("🦁 Verbal Fluency — Animals")
    st.write("⏱️ Name as many **animals** as you can in 1 minute:")
    ans = voice_input("fluency_animals")
    if ans:
        show_answer(ans)
        count = len(set(ans.lower().split()))
        st.write(f"📊 Approximate word count: **{count}**")
        st.session_state["fluency_animals_count"] = count

def step_fluency_fruits():
    st.subheader("🍎 Verbal Fluency — Fruits")
    st.write("⏱️ Name as many **fruits** as you can in 1 minute:")
    ans = voice_input("fluency_fruits")
    if ans:
        show_answer(ans)
        count = len(set(ans.lower().split()))
        st.write(f"📊 Approximate word count: **{count}**")
        st.session_state["fluency_fruits_count"] = count

def step_calculation():
    st.subheader("🧮 Calculation")
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
        st.write(f"📊 Correct: {correct} / {len(expected)}")
        st.session_state["calc_correct_count"] = correct

def make_naming_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🏷️ Naming ({n}/3)")
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
            st.write("✅ Correct!" if ok else "❌ Not quite.")
            st.session_state[f"{key}_ok"] = ok
    return step

def make_func_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🔧 Functional Description ({n}/2)")
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
            st.write("✅ Correct!" if ok else "❌ Not quite.")
            st.session_state[f"{key}_ok"] = ok
    return step

def make_proverb_step(n, question, key, kw_key):
    def step():
        st.subheader(f"📜 Proverb Interpretation ({n}/2)")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;font-style:italic;'>{question}</div>",
            unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            ok = ai_grade(question, ans)
            if ok is None:
                ok = contains_any(ans, KEYWORDS[kw_key])
            st.write("✅ Correct!" if ok else "⚠️ Try explaining again in your own words.")
            st.session_state[f"{key}_ok"] = ok
    return step

def step_delayed_recall():
    st.subheader("🧠 Delayed Recall")
    st.write("Say as many words as you can remember from the very beginning:")
    ans = voice_input("recall_widget")
    if ans:
        show_answer(ans)
        found = [w for w in WORDS if w.lower() in ans.lower()]
        pacing = get_pacing("recall_widget")
        pct = (len(found) / len(WORDS)) * pacing
        st.progress(pct, text=f"Accuracy: {pct:.0%}")
        st.write(f"📊 {len(found)} / {len(WORDS)} words")
        st.session_state["recall_score_count"] = len(found) * pacing

def step_results():
    st.subheader("📊 Results")
    score = 0.0
    details = []

    imm = st.session_state.get("immediate_recall_count", 0.0)
    score += imm
    details.append(f"🔁 Immediate Recall: {imm:.1f}/{len(WORDS)}")

    fwd_pt = get_pacing("fwd") if st.session_state.get("fwd_ok") else 0.0
    score += fwd_pt
    details.append(f"{'✅' if fwd_pt>0 else '❌'} Forward Digits: {fwd_pt:.2f}")

    bwd_pt = get_pacing("bwd") if st.session_state.get("bwd_ok") else 0.0
    score += bwd_pt
    details.append(f"{'✅' if bwd_pt>0 else '❌'} Backward Digits: {bwd_pt:.2f}")

    for i in range(1, 5):
        s = st.session_state.get(f"lang{i}_score", 0.0)
        if s >= 0.6:
            score += 1
            details.append(f"✅ Sentence {i}: {s:.0%}")
        else:
            details.append(f"❌ Sentence {i}: {s:.0%}")

    for i in range(1, 5):
        pt = get_pacing(f"abs{i}_widget") if st.session_state.get(f"abs{i}_correct") else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Abstraction {i}: {pt:.2f}")

    for key in ["ori_day_name","ori_date","ori_month","ori_year","ori_season"]:
        pt = get_pacing(key) if st.session_state.get(f"{key}_ok") else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Time ({key}): {pt:.2f}")

    for key in ["ori_country","ori_province","ori_place","ori_floor","ori_city"]:
        pt = get_pacing(key) if st.session_state.get(f"{key}_answered") else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Place ({key}): {pt:.2f}")

    for key in ["fluency_animals","fluency_fruits"]:
        count = st.session_state.get(f"{key}_count", 0)
        pt = get_pacing(key) if count >= 8 else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Fluency ({key}): {count} words")

    calc_raw = st.session_state.get("calc_correct_count", 0)
    calc_pt = calc_raw * get_pacing("calc_serial7")
    score += calc_pt
    details.append(f"🧮 Calculation: {calc_raw}/5 → {calc_pt:.2f}")

    for key in ["naming_watch","naming_pen","naming_dog"]:
        pt = get_pacing(key) if st.session_state.get(f"{key}_ok") else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Naming ({key}): {pt:.2f}")

    for key in ["func_hammer","func_scissors"]:
        pt = get_pacing(key) if st.session_state.get(f"{key}_ok") else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Function ({key}): {pt:.2f}")

    for key in ["proverb_water","proverb_slow"]:
        pt = get_pacing(key) if st.session_state.get(f"{key}_ok") else 0.0
        score += pt
        details.append(f"{'✅' if pt>0 else '❌'} Proverb ({key}): {pt:.2f}")

    found = st.session_state.get("recall_score_count", 0.0)
    score += found
    details.append(f"🧠 Delayed Recall: {found:.1f}/{len(WORDS)}")

    max_score = len(WORDS) + 2 + 4 + 4 + 5 + 5 + 2 + 5 + 3 + 2 + 2 + len(WORDS)

    st.markdown(f"""
        <div style='text-align:center;padding:2rem;background:#f0f4ff;
        border-radius:16px;margin:1rem 0;'>
            <div style='font-size:3rem;font-weight:700;'>{score:.1f} / {max_score}</div>
            <div style='font-size:1rem;color:#666;'>Total Score</div>
        </div>
    """, unsafe_allow_html=True)

    if score >= max_score * 0.7:
        st.success("🟢 Good performance")
    elif score >= max_score * 0.4:
        st.warning("🟡 Mild concern")
    else:
        st.error("🔴 Needs review")

    with st.expander("📋 Details"):
        for d in details:
            st.write(d)

    st.caption("⚠️ This is a prototype, not a medical diagnosis tool. Please consult a qualified professional.")

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
    step_immediate_recall,
    step_forward_digits,
    step_backward_digits,
    make_sentence_step(1, SENTENCES[0]),
    make_sentence_step(2, SENTENCES[1]),
    make_sentence_step(3, SENTENCES[2]),
    make_sentence_step(4, SENTENCES[3]),
    make_abstraction_step(1, "How are a Train and a Bicycle similar?", "abs1", "vehicle"),
    make_abstraction_step(2, "How are a Watch and a Ruler similar?", "abs2", "measurement"),
    make_abstraction_step(3, "How are an Orange and a Banana similar?", "abs3", "fruit"),
    make_abstraction_step(4, "How are a Table and a Chair similar?", "abs4", "furniture"),
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
    make_ori_place_step("What country are you in?", "ori_country"),
    make_ori_place_step("What state or province are you in?", "ori_province"),
    make_ori_place_step("Where are you right now? (e.g. home, hospital, clinic)", "ori_place"),
    make_ori_place_step("What floor of the building are you on?", "ori_floor"),
    make_ori_place_step("What city or town are you in?", "ori_city"),
    step_fluency_animals,
    step_fluency_fruits,
    step_calculation,
    make_naming_step(1, "What do you call the object worn on the wrist that tells time?", "naming_watch", "watch"),
    make_naming_step(2, "What do you call the object used for writing?", "naming_pen", "pen"),
    make_naming_step(3, "What do you call the pet that barks and guards the house?", "naming_dog", "dog"),
    make_func_step(1, "What is a hammer used for?", "func_hammer", "hammer"),
    make_func_step(2, "What are scissors used for?", "func_scissors", "scissors"),
    make_proverb_step(1, "What does \"strike while the iron is hot\" mean?", "proverb_water", "proverb_hot"),
    make_proverb_step(2, "What does \"slow and steady wins the race\" mean?", "proverb_slow", "proverb_slow"),
    step_delayed_recall,
    step_results,
]

TOTAL_STEPS = len(STEPS)

# ─────────────────────────────────────────────
# NAVIGATION STATE
# ─────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state["step"] = 0

step_idx = st.session_state["step"]

# ─────────────────────────────────────────────
# HEADER + PROGRESS BAR
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    div[data-testid="stAudioInput"] { transform: scale(1.3); transform-origin: left top; margin-bottom: 24px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**🧠 MindCheck**")
    st.progress(step_idx / (TOTAL_STEPS - 1), text=f"Step {step_idx + 1} / {TOTAL_STEPS}")

st.progress(step_idx / (TOTAL_STEPS - 1), text=f"Step {step_idx + 1} / {TOTAL_STEPS}")

# ─────────────────────────────────────────────
# RENDER CURRENT STEP
# ─────────────────────────────────────────────
STEPS[step_idx]()

# ─────────────────────────────────────────────
# NAVIGATION BUTTONS
# ─────────────────────────────────────────────
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
