# app.py
import os
import io
import csv
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Picking Helper", page_icon="📦", layout="wide")

# ---------- ENV / PATHS ----------
TODAY = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.getenv("PICKING_HELPER_LOG_DIR", "logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"picking-log-{TODAY}.csv")

# ---------- SESSION STATE ----------
ss = st.session_state
if "operator" not in ss:
    ss.operator = ""
if "current_location" not in ss:
    ss.current_location = ""
if "current_pallet" not in ss:
    ss.current_pallet = ""
if "starting_qty" not in ss:
    ss.starting_qty = None
if "picked_so_far" not in ss:
    ss.picked_so_far = {}  # pallet_id -> picked qty sum (int)
if "recent_scans" not in ss:
    ss.recent_scans = []
if "tag_bytes" not in ss:
    ss.tag_bytes = None

# ---------- HELPERS ----------
def is_location(code: str) -> bool:
    # Your locations are 8 digits (AAA BBB CC style like 11100101)
    # Adjust this rule if needed.
    c = code.strip()
    return c.isdigit() and len(c) == 8

def clean_scan(raw: str) -> str:
    return (raw or "").replace("\r", "").replace("\n", "").strip()

def append_log_row(row: dict):
    # Append to CSV on disk safely (header on first write)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def make_partial_tag_png(location: str, pallet: str, qty_remaining: int|None, operator: str) -> bytes:
    # Simple, bold tag you can print or show on screen
    W, H = 800, 500
    bg = (249, 249, 249)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    # Try system font fallback
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        font_body  = ImageFont.truetype("DejaVuSans.ttf", 34)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_body  = ImageFont.load_default()
        font_small = ImageFont.load_default()

    d.rectangle([0, 0, W, 90], fill=(6, 83, 164))
    d.text((24, 20), "PARTIAL PALLET TAG", font=font_title, fill="white")

    y = 120
    d.text((24, y),     f"Location:  {location or '—'}", font=font_body, fill=(30,30,30)); y += 60
    d.text((24, y),     f"Pallet ID: {pallet or '—'}",   font=font_body, fill=(30,30,30)); y += 60
    if qty_remaining is not None:
        d.text((24, y), f"Qty Remaining: {qty_remaining}", font=font_body, fill=(200, 33, 39))
        y += 60
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    d.text((24, y),     f"Generated: {ts}", font=font_small, fill=(80,80,80)); y += 36
    if operator:
        d.text((24, y), f"By: {operator}", font=font_small, fill=(80,80,80))

    # Footer stripe
    d.rectangle([0, H-18, W, H], fill=(6, 83, 164))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def upsert_picked(pallet_id: str, qty: int):
    current = ss.picked_so_far.get(pallet_id, 0)
    ss.picked_so_far[pallet_id] = current + qty

def get_remaining(starting: int|None, pallet_id: str) -> int|None:
    if starting is None:
        return None
    picked = ss.picked_so_far.get(pallet_id, 0)
    return max(starting - picked, 0)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ss.operator = st.text_input("Operator (optional)", value=ss.operator, placeholder="e.g., Carlos")
    st.write("**Log file:**", f"`{LOG_FILE}`")
    st.markdown("---")
    st.markdown("#### Start/Balance (optional)")
    ss.starting_qty = st.number_input(
        "Starting qty on current pallet",
        min_value=0, step=1, value=ss.starting_qty or 0,
        help="Set this if you want the app to compute remaining qty."
    )
    st.caption("Tip: Set this after you scan a pallet to track remaining.")

# ---------- HEADER ----------
st.title("📦 Picking Helper")
st.caption("Scan Location → Scan Pallet → Enter Qty Picked → Generate Partial Tag → CSV Log")

# ---------- OPTIONAL AUTO-FOCUS (works in most Android browsers) ----------
st.markdown("""
    <script>
    const focusScan = () => {
        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
        if (inputs && inputs.length) { inputs[0].focus(); inputs[0].select(); }
    };
    setTimeout(focusScan, 200);
    </script>
""", unsafe_allow_html=True)

# ---------- SCAN HANDLER ----------
def on_scan():
    raw = ss.scan or ""
    code = clean_scan(raw)
    if not code:
        return
    # Classify scan: if it's an 8-digit numeric => treat as Location, else Pallet
    if is_location(code):
        ss.current_location = code
        st.toast(f"Location set: {code}", icon="📍")
    else:
        ss.current_pallet = code
        st.toast(f"Pallet set: {code}", icon="🪵")
    # Keep history
    ss.recent_scans.insert(0, (datetime.now().strftime("%H:%M:%S"), code))
    ss.recent_scans = ss.recent_scans[:25]
    # Clear box for next scan
    ss.scan = ""

# ---------- UI: SCAN BOX ----------
st.subheader("Scan")
st.text_input("Scan here", key="scan", placeholder="Focus here and scan location or pallet…",
              label_visibility="collapsed", on_change=on_scan)

# ---------- CURRENT CONTEXT ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Location", ss.current_location or "—")
with c2:
    st.metric("Pallet ID", ss.current_pallet or "—")
with c3:
    rem = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet) if ss.current_pallet else None
    st.metric("Remaining (est.)", "—" if rem is None else rem)

with st.expander("Recent scans", expanded=True):
    if ss.recent_scans:
        for t, c in ss.recent_scans[:10]:
            st.write(f"🕒 {t} — **{c}**")
    else:
        st.info("No scans yet.")

st.markdown("---")

# ---------- PICK ENTRY ----------
st.subheader("Pick")
pick_qty = st.number_input("Qty picked", min_value=0, step=1, value=0, help="Enter the quantity you just picked.")

colA, colB, colC = st.columns([1,1,1])
with colA:
    do_log = st.button("➕ Log Pick", use_container_width=True)
with colB:
    do_tag = st.button("🏷️ Generate Partial Pallet Tag", use_container_width=True)
with colC:
    st.download_button(
        "⬇️ Download Today’s CSV Log",
        data=open(LOG_FILE, "rb").read() if os.path.exists(LOG_FILE) else b"",
        file_name=f"picking-log-{TODAY}.csv",
        mime="text/csv",
        disabled=not os.path.exists(LOG_FILE),
        use_container_width=True
    )

# ---------- ACTIONS ----------
if do_log:
    if not ss.current_location:
        st.error("Scan a **Location** first.")
    elif not ss.current_pallet:
        st.error("Scan a **Pallet ID** next.")
    elif pick_qty <= 0:
        st.error("Enter a **Qty picked** greater than zero.")
    else:
        upsert_picked(ss.current_pallet, int(pick_qty))
        remaining = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": ss.operator or "",
            "location": ss.current_location,
            "pallet_id": ss.current_pallet,
            "qty_picked": int(pick_qty),
            "starting_qty": (ss.starting_qty if ss.starting_qty != 0 else ""),
            "remaining_after": ("" if remaining is None else remaining)
        }
        append_log_row(row)
        st.success(f"Logged pick: Pallet {ss.current_pallet}  |  Qty {pick_qty}" + (f"  |  Remaining {remaining}" if remaining is not None else ""))
        # Clear qty only; keep context for next pick
        st.experimental_rerun()

if do_tag:
    if not ss.current_pallet:
        st.error("Scan a **Pallet ID** before generating a tag.")
    else:
        remaining = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet)
        ss.tag_bytes = make_partial_tag_png(
            location=ss.current_location,
            pallet=ss.current_pallet,
            qty_remaining=remaining,
            operator=ss.operator or ""
        )

# ---------- TAG PREVIEW / DOWNLOAD ----------
if ss.tag_bytes:
    st.subheader("Partial Pallet Tag")
    st.image(ss.tag_bytes, caption="Preview", use_column_width=True)
    st.download_button(
        "⬇️ Download Tag (PNG)",
        data=ss.tag_bytes,
        file_name=f"partial-tag-{ss.current_pallet or 'pallet'}.png",
        mime="image/png"
    )

# ---------- LOG TABLE PREVIEW ----------
st.markdown("---")
st.subheader("Today’s Log (preview)")
if os.path.exists(LOG_FILE):
    import pandas as pd
    df = pd.read_csv(LOG_FILE)
    st.dataframe(df.tail(50), use_container_width=True, height=300)
else:
    st.info("No log entries yet today.")