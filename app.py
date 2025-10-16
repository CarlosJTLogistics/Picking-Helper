# app.py
import os
import io
import re
import json
import csv
from typing import Optional, Dict, Tuple, List
from urllib.parse import urlparse, parse_qs, unquote_plus
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

# Always trim at first '8304'
START_MARKER = "8304"

# ---------- SESSION STATE ----------
ss = st.session_state
defaults = {
    "operator": "",
    "current_location": "",
    "current_pallet": "",
    "sku": "",
    "lot_number": "",
    "starting_qty": None,    # optional starting qty to compute remaining
    "picked_so_far": {},     # pallet_id -> picked qty sum (int)
    "recent_scans": [],
    "tag_bytes": None,
    "scan": "",
    "qty_staged": 0,
    "chosen_quick": None,
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# ---------- HELPERS ----------
def is_location(code: str) -> bool:
    c = (code or "").strip()
    return c.isdigit() and len(c) == 8

def clean_scan(raw: str) -> str:
    return (raw or "").replace("\r", "").replace("\n", "").strip()

def apply_start_marker(s: str, marker: str) -> Tuple[str, bool]:
    if not marker:
        return s, False
    idx = s.find(marker)
    if idx >= 0:
        return s[idx:], True
    return s, False

def append_log_row(row: dict):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        field_order = [
            "timestamp","operator","location","pallet_id","sku","lot_number",
            "qty_staged","qty_picked","starting_qty","remaining_after"
        ]
        for k in field_order:
            row.setdefault(k, "")
        writer = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in field_order})

def make_partial_tag_png(location: str, pallet: str, qty_remaining: Optional[int], operator: str) -> bytes:
    W, H = 800, 500
    bg = (249, 249, 249)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
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
    d.text((24, y), f"Location: {location or '—'}", font=font_body, fill=(30,30,30)); y += 60
    d.text((24, y), f"Pallet ID: {pallet or '—'}", font=font_body, fill=(30,30,30)); y += 60
    if qty_remaining is not None:
        d.text((24, y), f"Qty Remaining: {qty_remaining}", font=font_body, fill=(200, 33, 39)); y += 60
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    d.text((24, y), f"Generated: {ts}", font=font_small, fill=(80,80,80)); y += 36
    if operator:
        d.text((24, y), f"By: {operator}", font=font_small, fill=(80,80,80))
    d.rectangle([0, H-18, W, H], fill=(6, 83, 164))
    b = io.BytesIO(); img.save(b, format="PNG")
    return b.getvalue()

def upsert_picked(pallet_id: str, qty: int):
    current = ss.picked_so_far.get(pallet_id, 0)
    ss.picked_so_far[pallet_id] = current + qty

def get_remaining(starting: Optional[int], pallet_id: str) -> Optional[int]:
    if starting is None:
        return None
    picked = ss.picked_so_far.get(pallet_id, 0)
    return max(starting - picked, 0)

def normalize_lot(lot: Optional[str]) -> str:
    if not lot:
        return ""
    digits = re.sub(r"\D", "", lot)
    digits = digits.lstrip("0") or "0"
    return digits

# ---- Parsers ----
def try_parse_json(s: str) -> Optional[Dict[str,str]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def try_parse_query_or_kv(s: str) -> Optional[Dict[str,str]]:
    if "://" in s:
        parsed = urlparse(s)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        return {k.lower(): unquote_plus(v[-1]) if v else "" for k, v in qs.items()} or None
    sep_standardized = s.replace(";", "&").replace("|", "&")
    if "=" in sep_standardized:
        parts = [p for p in sep_standardized.split("&") if p]
        kv = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip().lower()] = unquote_plus(v.strip())
        return kv or None
    return None

def try_parse_gs1_paren(s: str) -> Optional[Dict[str,str]]:
    # Supports (21)=pallet, (10)=lot, (01)=gtin
    pairs = re.findall(r"\((\d{2,4})\)([^\(\)]+)", s)
    if not pairs:
        return None
    out: Dict[str,str] = {}
    for ai, val in pairs:
        val = val.strip()
        if ai == "21": out["pallet"] = val
        elif ai == "10": out["lot"] = val
        elif ai == "01": out["gtin"] = val
    return out or None

def normalize_keys(data: Dict[str,str]) -> Dict[str,str]:
    key_map = {
        "location": ["location","loc","bin","slot","staging","stg"],
        "pallet":   ["pallet","pallet_id","serial","id","license","lpn","sscc"],
        "sku":      ["sku","item","itemcode","product","part","material"],
        "lot":      ["lot","lot_number","batch","batchno"],
    }
    out: Dict[str,str] = {}
    lower_data = {k.lower(): str(v) for k, v in data.items()}
    for target, aliases in key_map.items():
        for a in aliases:
            if a in lower_data and lower_data[a]:
                out[target] = lower_data[a].strip()
                break
    if "lot" in out:
        out["lot"] = normalize_lot(out["lot"])
    return out

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ss["operator"] = st.text_input("Operator (optional)", value=ss.operator, placeholder="e.g., Carlos")
    st.write("**Log file:**", f"`{LOG_FILE}`")
    st.markdown("---")
    st.markdown("#### Start/Balance (optional)")
    ss["starting_qty"] = st.number_input(
        "Starting qty on current pallet",
        min_value=0, step=1, value=ss.starting_qty or 0,
        help="Set this if you want the app to compute remaining qty."
    )
    st.caption("Tip: Set this after you scan a pallet to track remaining.")

# ---------- HEADER ----------
st.title("📦 Picking Helper")
st.caption("Scan QR → trim at 8304 → auto-fill Pallet / Location / SKU / LOT → choose QTY staged → Log / Tag")

# ---------- Auto-focus ----------
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

    # 1) Lock trim at '8304'
    trimmed, was_trimmed = apply_start_marker(code, START_MARKER)
    # after marker (remove exactly the marker itself if at start)
    after = trimmed[len(START_MARKER):] if trimmed.startswith(START_MARKER) else trimmed

    # 2) Try rich parses on "after"
    parsed = (try_parse_json(after)
              or try_parse_query_or_kv(after)
              or try_parse_gs1_paren(after)
              or {})

    norm = normalize_keys(parsed) if parsed else {}

    # 3) Resolve fields, preferring parsed keys; fallback to treating "after" as bare Pallet ID
    pallet_val = norm.get("pallet")
    location = norm.get("location")
    sku       = norm.get("sku")
    lot       = norm.get("lot")

    # If no 'pallet' key parsed, assume the remainder is the pallet id
    if not pallet_val:
        pallet_val = after.strip()

    # Commit to session (Option 1 behavior)
    ss.current_pallet = pallet_val or ss.current_pallet
    if location: ss.current_location = location
    if sku is not None: ss.sku = sku or ""
    if lot is not None: ss.lot_number = lot or ""

    # Toast summary
    bits = []
    if was_trimmed: bits.append(f"Trimmed at '{START_MARKER}'")
    bits.append(f"Pallet {ss.current_pallet}")
    if ss.current_location: bits.append(f"Location {ss.current_location}")
    if ss.sku: bits.append(f"SKU {ss.sku}")
    if ss.lot_number: bits.append(f"LOT {ss.lot_number}")
    st.toast(" | ".join(bits), icon="✅")

    # History + clear
    ss.recent_scans.insert(0, (datetime.now().strftime("%H:%M:%S"), trimmed))
    ss.recent_scans = ss.recent_scans[:25]
    ss.scan = ""

# ---------- UI ----------
st.subheader("Scan")
st.text_input(
    "Scan here",
    key="scan",
    placeholder="Focus here and scan… (app trims at 8304 and parses keys)",
    label_visibility="collapsed",
    on_change=on_scan
)

# Current context
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Location", ss.current_location or "—")
with c2: st.metric("Pallet ID", ss.current_pallet or "—")
with c3: st.metric("SKU", ss.sku or "—")
with c4: st.metric("LOT Number", ss.lot_number or "—")
with c5:
    rem = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet) if ss.current_pallet else None
    st.metric("Remaining (est.)", "—" if rem is None else rem)

with st.expander("Recent scans", expanded=True):
    if ss.recent_scans:
        for t, c in ss.recent_scans[:10]:
            st.write(f"🕒 {t} — **{c}**")
    else:
        st.info("No scans yet.")

st.markdown("---")

# QTY staged
st.subheader("QTY staged")
colq1, colq2, colq3, colq4, colq5, colq6, colq7 = st.columns([1,1,1,1,1,1,2])
def _set_quick(v: int):
    st.session_state["qty_staged"] = v
    st.session_state["chosen_quick"] = v
with colq1: st.button("1", on_click=_set_quick, args=(1,))
with colq2: st.button("2", on_click=_set_quick, args=(2,))
with colq3: st.button("3", on_click=_set_quick, args=(3,))
with colq4: st.button("5", on_click=_set_quick, args=(5,))
with colq5: st.button("10", on_click=_set_quick, args=(10,))
with colq6: st.button("15", on_click=_set_quick, args=(15,))
with colq7:
    qty_staged = st.number_input("or type a value", min_value=0, step=1,
                                 value=st.session_state.get("qty_staged", 0),
                                 help="Select a chip or type a custom staged quantity.")
    st.session_state["qty_staged"] = qty_staged
st.caption("Only QTY staged is chosen by the user. Everything else auto-fills from the QR.")

# Pick section
st.subheader("Pick")
pick_qty = st.number_input("Qty picked", min_value=0, step=1, value=0, help="Optional running tracker.")
colA, colB, colC = st.columns([1,1,1])
with colA: do_log = st.button("➕ Log Entry", use_container_width=True)
with colB: do_tag = st.button("🏷️ Generate Partial Pallet Tag", use_container_width=True)
with colC:
    st.download_button(
        "⬇️ Download Today’s CSV Log",
        data=open(LOG_FILE, "rb").read() if os.path.exists(LOG_FILE) else b"",
        file_name=f"picking-log-{TODAY}.csv",
        mime="text/csv",
        disabled=not os.path.exists(LOG_FILE),
        use_container_width=True
    )

# Actions
if do_log:
    if not ss.current_location:
        st.error("Scan a **Location** first (or include it in the QR).")
    elif not ss.current_pallet:
        st.error("Scan a **Pallet ID** next.")
    elif st.session_state.get("qty_staged", 0) <= 0 and pick_qty <= 0:
        st.error("Choose a **QTY staged** or enter a **Qty picked** > 0.")
    else:
        if pick_qty > 0:
            upsert_picked(ss.current_pallet, int(pick_qty))
        remaining = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": ss.operator or "",
            "location": ss.current_location,
            "pallet_id": ss.current_pallet,
            "sku": ss.sku or "",
            "lot_number": ss.lot_number or "",
            "qty_staged": int(st.session_state.get("qty_staged", 0)) if st.session_state.get("qty_staged", 0) > 0 else "",
            "qty_picked": int(pick_qty) if pick_qty > 0 else "",
            "starting_qty": (ss.starting_qty if ss.starting_qty != 0 else ""),
            "remaining_after": ("" if remaining is None else remaining),
        }
        append_log_row(row)
        msg_bits = [f"Pallet {ss.current_pallet}"]
        if st.session_state.get("qty_staged", 0) > 0: msg_bits.append(f"Staged {st.session_state['qty_staged']}")
        if pick_qty > 0: msg_bits.append(f"Picked {pick_qty}")
        if remaining is not None: msg_bits.append(f"Remaining {remaining}")
        st.success(" | ".join(msg_bits))
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

# Tag preview
if ss.tag_bytes:
    st.subheader("Partial Pallet Tag")
    st.image(ss.tag_bytes, caption="Preview", use_column_width=True)
    st.download_button(
        "⬇️ Download Tag (PNG)",
        data=ss.tag_bytes,
        file_name=f"partial-tag-{ss.current_pallet or 'pallet'}.png",
        mime="image/png"
    )

# Log table preview
st.markdown("---")
st.subheader("Today’s Log (preview)")
if os.path.exists(LOG_FILE):
    import pandas as pd
    dfprev = pd.read_csv(LOG_FILE)
    st.dataframe(dfprev.tail(50), use_container_width=True, height=320)
else:
    st.info("No log entries yet today.")