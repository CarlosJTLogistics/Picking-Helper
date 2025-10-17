# app.py — Outbound Picking Helper (Batch Submit)
# v1.4.2 — QR compatibility hotfix: GS1 (single-paren) + GS1 FNC1 + label-style key:value; raw scan debugger
import os
import io
import re
import json
import csv
import sys
import time
import platform
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs, unquote_plus
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Picking Helper", page_icon="📦", layout="wide")

# ------------------ ENV / PATHS ------------------
TODAY = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = os.getenv("PICKING_HELPER_LOG_DIR", "logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"picking-log-{TODAY}.csv")

WEBHOOK_URL = os.getenv("PICKING_HELPER_WEBHOOK_URL", "").strip()

# Recipients for notifications (email/Teams handled in Flow)
NOTIFY_TO_ENV = os.getenv("PICKING_HELPER_NOTIFY_TO", "").strip()
DEFAULT_NOTIFY_TO = [
    "carlos.pacheco@jtlogistics.com",
    "Alex.Miller@jtlogistics.com",
    "Cody.Robles@JTlogistics.com",
]
def _parse_notify_to(s: str) -> List[str]:
    if not s:
        return DEFAULT_NOTIFY_TO
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]
NOTIFY_TO = _parse_notify_to(NOTIFY_TO_ENV)

KIOSK = os.getenv("PICKING_HELPER_KIOSK", "0").strip() == "1"
LOOKUP_FILE_ENV = os.getenv("PICKING_HELPER_LOOKUP_FILE", "").strip()

# ------------------ SESSION STATE ------------------
ss = st.session_state
defaults = {
    # identity / context
    "operator": "",
    "current_order": "",  # optional
    # live fields
    "current_location": "",  # source location
    "current_pallet": "",
    "sku": "",
    "lot_number": "",
    "staging_location_current": "",
    # scans/typing buffers
    "scan": "",  # pallet scan
    "source_scan": "",  # source location scan
    "typed_source": "",  # source location typed
    "typed_pallet_id": "",  # typed pallet
    "lot_scan": "",  # lot scan
    "typed_lot": "",  # typed lot
    "staging_scan": "",  # staging scan
    # qty
    "qty_staged": 0,  # numeric only (no presets)
    # misc
    "starting_qty": None,  # optional for Remaining calc
    "picked_so_far": {},  # pallet_id -> picked qty sum
    "recent_scans": [],
    "tag_bytes": None,
    # lookup
    "lookup_df": None,
    "lookup_cols": {"pallet": None, "sku": None, "lot": None, "location": None},
    # batch
    "batch_rows": [],
    # debug
    "last_raw_scan": "",
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v
# ------------------ HELPERS ------------------
def clean_scan(raw: str) -> str:
    # Keep control chars like \x1D for GS1 FNC1; only strip CR/LF and leading/trailing spaces.
    return (raw or "").replace("\r", "").replace("\n", "").strip()

def append_log_row(row: dict):
    file_exists = os.path.isfile(LOG_FILE)
    field_order = [
        "timestamp","operator","order_number","location","staging_location",
        "pallet_id","sku","lot_number","qty_staged","qty_picked",
        "starting_qty","remaining_after","batch_id","action"
    ]
    for k in field_order:
        row.setdefault(k, "")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in field_order})

def make_partial_tag_png(location: str, pallet: str, qty_remaining: Optional[int], operator: str) -> bytes:
    W, H = 800, 500
    img = Image.new("RGB", (W, H), (249, 249, 249))
    d = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        font_body = ImageFont.truetype("DejaVuSans.ttf", 34)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_small = ImageFont.load_default()
    d.rectangle([0, 0, W, 90], fill=(6, 83, 164))
    d.text((24, 20), "PARTIAL PALLET TAG", font=font_title, fill="white")
    y = 120
    d.text((24, y), f"Location: {location or '—'}", font=font_body, fill=(30,30,30)); y += 60
    d.text((24, y), f"Pallet ID: {pallet or '—'}", font=font_body, fill=(30,30,30)); y += 60
    if qty_remaining is not None:
        d.text((24, y), f"Qty Remaining: {qty_remaining}", font=font_body, fill=(200,33,39)); y += 60
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    d.text((24, y), f"Generated: {ts}", font=font_small, fill=(80,80,80)); y += 36
    if operator:
        d.text((24, y), f"By: {operator}", font=font_small, fill=(80,80,80))
    d.rectangle([0, H-18, W, H], fill=(6, 83, 164))
    b = io.BytesIO(); img.save(b, format="PNG")
    return b.getvalue()

def upsert_picked(pallet_id: str, qty: int):
    ss.picked_so_far[pallet_id] = ss.picked_so_far.get(pallet_id, 0) + qty

def get_remaining(starting: Optional[int], pallet_id: str) -> Optional[int]:
    if starting is None:
        return None
    return max(starting - ss.picked_so_far.get(pallet_id, 0), 0)

def normalize_lot(lot: Optional[str]) -> str:
    if not lot:
        return ""
    digits = re.sub(r"\D", "", lot)
    return digits.lstrip("0") or "0"

def _is_valid_source_location(loc: str) -> bool:
    return bool(loc) and loc.isdigit() and len(loc) == 8

# ------------------ LOOKUP SUPPORT ------------------
def _auto_guess_cols(columns: List[str]) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in columns}
    def pick(syns: List[str]) -> Optional[str]:
        for s in syns:
            if s in cols_lower:
                return cols_lower[s]
        for c in columns:
            cl = c.lower()
            if any(s in cl for s in syns):
                return c
        return None
    return {
        "pallet": pick(["pallet","pallet id","pallet_id","lpn","license","serial","sscc"]),
        "sku": pick(["sku","item","itemcode","product","part","material"]),
        "lot": pick(["lot","lot_number","batch","batchno"]),
        "location": pick(["location","loc","bin","slot","staging","stg"]),
    }

@st.cache_data(show_spinner=False)
def load_lookup(path: Optional[str], uploaded_bytes: Optional[bytes]):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(f"pandas not available: {e}")
    df = None
    try:
        if uploaded_bytes:
            try:
                try:
                    df = pd.read_excel(io.BytesIO(uploaded_bytes), engine="openpyxl")
                except Exception:
                    df = pd.read_csv(io.BytesIO(uploaded_bytes))
            except Exception as e:
                raise RuntimeError(f"Failed to read uploaded file: {e}")
        elif path and os.path.exists(path):
            try:
                p = path.lower()
                if p.endswith(".xls"):
                    df = pd.read_excel(path, engine="xlrd")
                elif p.endswith((".xlsx",".xlsm")):
                    df = pd.read_excel(path, engine="openpyxl")
                else:
                    df = pd.read_csv(path)
            except Exception as e:
                raise RuntimeError(f"Failed to read file at {path}: {e}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Lookup load error: {e}")
    if df is not None and len(df) > 0:
        df.columns = [str(c).strip() for c in df.columns]
        return df, _auto_guess_cols(list(df.columns))
    return None, None
def lookup_fields_by_pallet(df, colmap: Dict[str, Optional[str]], pallet_id: str) -> Dict[str,str]:
    out = {}
    if df is None or not pallet_id:
        return out
    pcol = colmap.get("pallet")
    if not pcol or pcol not in df.columns:
        return out
    needle = str(pallet_id).strip().upper()
    try:
        subset = df[df[pcol].astype(str).str.upper().str.strip() == needle]
        if not subset.empty:
            row = subset.iloc[0]
            if (c := colmap.get("sku")) and c in df.columns: out["sku"] = str(row[c]).strip()
            if (c := colmap.get("lot")) and c in df.columns: out["lot"] = normalize_lot(str(row[c]))
            if (c := colmap.get("location")) and c in df.columns: out["location"] = str(row[c]).strip()
    except Exception:
        pass
    return out
# ------------------ SCAN PARSERS (expanded) ------------------
def try_parse_json(s: str) -> Optional[Dict[str,str]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def try_parse_query_or_kv(s: str) -> Optional[Dict[str,str]]:
    # URL query or "k=v" separated by &, ;, commas, spaces, or newlines
    if "://" in s:
        parsed = urlparse(s)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        return {k.lower(): unquote_plus(v[-1]) if v else "" for k, v in qs.items()} or None
    s2 = s.replace(";", "&").replace("\n", "&").replace(",", "&").replace("  ", " ")
    parts = [p for p in re.split(r"[& ]", s2) if p]
    kv = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip().lower()] = unquote_plus(v.strip())
        elif ":" in p:
            k, v = p.split(":", 1)
            kv[k.strip().lower()] = v.strip()
    return kv or None

def try_parse_label_kv(s: str) -> Optional[Dict[str,str]]:
    """
    Parses human-readable labels like:
      Pallet ID: JTL00496
      LOT: 9062716
      Location: 11400804
    Lines may be separated by newline, ';', or ','.
    """
    lines = re.split(r"[\n;,]+", s)
    out = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower(); v = v.strip()
            if k and v:
                out[k] = v
    return out or None

def try_parse_gs1_paren(s: str) -> Optional[Dict[str,str]]:
    """
    Parses GS1 AI with *single* parentheses, e.g.:
      (21)SERIAL(10)LOT(01)GTIN
    """
    pairs = re.findall(r"\((\d{2,4})\)([^()]+)", s)
    if not pairs:
        return None
    out: Dict[str,str] = {}
    for ai, val in pairs:
        val = val.strip()
        if ai == "21": out["pallet"] = val  # Serial/LPN
        elif ai == "10": out["lot"] = val
        elif ai == "01": out["gtin"] = val
        elif ai == "00": out["pallet"] = val  # SSCC as pallet if present
    return out or None

def try_parse_gs1_fnc1(s: str) -> Optional[Dict[str,str]]:
    """
    Parses GS1 AI with FNC1 (ASCII 29) separators. Example raw:
      "010123456789012121JTL00496\x1D109062716"
    """
    GS = "\x1D"
    if GS not in s and not re.match(r"^\d{2,4}", s):
        return None

    # Known AIs
    AI_FIXED = {"00":18, "01":14, "414":13}  # SSCC, GTIN, GLN (if present)
    AI_VAR = {"10","21","240","241","420"}   # LOT, SERIAL/LPN, etc.

    i = 0
    n = len(s)
    out: Dict[str,str] = {}

    def read_ai(idx: int) -> Tuple[Optional[str], int]:
        # Try 4, 3, then 2-digit AI
        for L in (4,3,2):
            if idx+L <= n and s[idx:idx+L].isdigit():
                return s[idx:idx+L], idx+L
        return None, idx+1

    while i < n:
        if not s[i].isdigit():
            i += 1
            continue
        ai, j = read_ai(i)
        if not ai:
            i += 1
            continue
        if ai in AI_FIXED:
            L = AI_FIXED[ai]
            if j+L <= n:
                val = s[j:j+L]
                i = j+L
            else:
                val = s[j:]
                i = n
        elif ai in AI_VAR:
            # Read until GS or end
            k = s.find(GS, j)
            if k == -1:
                val = s[j:]
                i = n
            else:
                val = s[j:k]
                i = k+1
        else:
            # Unknown AI; skip to next separator or char
            next_gs = s.find(GS, j)
            if next_gs == -1:
                i = n
            else:
                i = next_gs+1
            continue

        val = val.strip()
        if ai == "21": out["pallet"] = val
        elif ai == "10": out["lot"] = val
        elif ai == "01": out["gtin"] = val
        elif ai == "00": out["pallet"] = val  # Use SSCC as pallet when 21 absent

    return out or None

def normalize_keys(data: Dict[str,str]) -> Dict[str,str]:
    key_map = {
        "location": ["location","loc","bin","slot","staging","stg"],
        "pallet": ["pallet","pallet_id","serial","id","license","lpn","sscc","pallet id"],
        "sku": ["sku","item","itemcode","product","part","material"],
        "lot": ["lot","lot_number","batch","batchno"],
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

def parse_any_scan(code: str) -> Dict[str,str]:
    """
    Try multiple strategies to extract pallet/location/sku/lot from a raw scan.
    Order: JSON -> query/kv -> GS1 (paren) -> GS1 (FNC1) -> label kv.
    """
    for fn in (try_parse_json, try_parse_query_or_kv, try_parse_gs1_paren, try_parse_gs1_fnc1, try_parse_label_kv):
        try:
            parsed = fn(code)
        except Exception:
            parsed = None
        if parsed:
            norm = normalize_keys(parsed)
            if norm:
                return norm
    return {}

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.info("BUILD: outbound-batch v1.4.2 (QR GS1/FNC1 hotfix)", icon="🧭")
    st.markdown("### ⚙️ Settings")
    ss["operator"] = st.text_input("Operator (optional)", value=ss.operator, placeholder="e.g., Carlos")
    ss["current_order"] = st.text_input("Order # (optional)", value=ss.current_order, placeholder="e.g., SO-12345")
    st.write("**Log file:**", f"`{LOG_FILE}`")
    st.caption(f"Recipients: {', '.join(NOTIFY_TO)}")
    if WEBHOOK_URL:
        st.caption("Submit will POST to configured Power Automate webhook.")
    else:
        st.warning("PICKING_HELPER_WEBHOOK_URL is not set. Submit will only write CSV locally.")

    st.markdown("---")
    st.markdown("#### Inventory Lookup (optional)")
    st.caption("Upload CSV/XLS/XLSX to auto-fill SKU / LOT / Location from Pallet ID.")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"], accept_multiple_files=False)
    up_bytes = uploaded.read() if uploaded is not None else None
    df, guessed = None, None
    lookup_error = None
    try:
        df, guessed = load_lookup(LOOKUP_FILE_ENV if not up_bytes else None, up_bytes)
        ss.lookup_df = df
    except Exception as e:
        lookup_error = str(e)
        ss.lookup_df = None
    if lookup_error:
        st.error(f"Lookup load error: {lookup_error}")
    if ss.lookup_df is not None:
        st.success(f"Loaded lookup with {len(ss.lookup_df):,} rows")
        cols = list(ss.lookup_df.columns)
        g = guessed or {"pallet": None, "sku": None, "lot": None, "location": None}
        c1, c2 = st.columns(2)
        with c1:
            ss.lookup_cols["pallet"] = st.selectbox("Pallet column", options=cols, index=cols.index(g["pallet"]) if g["pallet"] in cols else 0)
            ss.lookup_cols["sku"] = st.selectbox("SKU column", options=["(none)"] + cols, index=(cols.index(g["sku"]) + 1) if g["sku"] in cols else 0)
        with c2:
            ss.lookup_cols["lot"] = st.selectbox("LOT column", options=["(none)"] + cols, index=(cols.index(g["lot"]) + 1) if g["lot"] in cols else 0)
            ss.lookup_cols["location"] = st.selectbox("Location column", options=["(none)"] + cols, index=(cols.index(g["location"]) + 1) if g["location"] in cols else 0)
        for k in ["sku","lot","location"]:
            if ss.lookup_cols.get(k) == "(none)":
                ss.lookup_cols[k] = None

    st.markdown("---")
    st.markdown("#### Start/Balance (optional)")
    ss["starting_qty"] = st.number_input(
        "Starting qty on current pallet",
        min_value=0, step=1, value=ss.starting_qty or 0,
        help="Only needed if you want Remaining computed."
    )
    st.caption("Max staged per line remains 15 cases (validation).")
    if KIOSK:
        st.caption("KIOSK MODE enabled (menu/footer hidden).")

    with st.expander("Diagnostics", expanded=False):
        st.write({
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "streamlit": st.__version__,
            "LOOKUP_FILE_ENV": LOOKUP_FILE_ENV or "(empty)",
            "cwd": os.getcwd(),
            "files_in_cwd": sorted(os.listdir("."))[:50],
        })

# Optional kiosk CSS
if KIOSK:
    st.markdown("""
    <style>
    #MainMenu, footer {visibility:hidden;}
    </style>
    """, unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("📦 Picking Helper — Outbound (Batch)")
st.caption("Scan or type Pallet / Location / LOT → set QTY staged (max 15) → scan/type Staging → Add to batch → Review & Submit")

# Auto-focus the first text input (helps scan guns)
st.markdown("""
<script>
 const focusScan = () => {
   const inputs = window.parent.document.querySelectorAll('input[type="text"]');
   if (inputs && inputs.length) { inputs[0].focus(); inputs[0].select(); }
 };
 setTimeout(focusScan, 200);
</script>
""", unsafe_allow_html=True)

# ------------------ SCAN/TYPING HANDLERS ------------------
def on_pallet_scan():
    raw = ss.scan or ""
    code = clean_scan(raw)
    if not code:
        return
    ss.last_raw_scan = raw  # keep true raw (before clean) for debugging

    # Parse using multiple strategies
    norm = parse_any_scan(code)

    pallet_id = norm.get("pallet") or code
    ss.current_pallet = pallet_id

    loc = norm.get("location")
    sku = norm.get("sku")
    lot = norm.get("lot")

    # Lookup fallback
    if ss.lookup_df is not None and (not loc or not sku or not lot):
        lookup_map = {
            "pallet": ss.lookup_cols.get("pallet"),
            "sku": ss.lookup_cols.get("sku"),
            "lot": ss.lookup_cols.get("lot"),
            "location": ss.lookup_cols.get("location"),
        }
        looked = lookup_fields_by_pallet(ss.lookup_df, lookup_map, pallet_id)
        loc = loc or looked.get("location")
        sku = sku or looked.get("sku")
        lot = lot or looked.get("lot")

    if loc: ss.current_location = loc
    if sku is not None: ss.sku = sku or ""
    if lot is not None: ss.lot_number = lot or ""

    bits = [f"Pallet {ss.current_pallet}"]
    if ss.current_location: bits.append(f"Location {ss.current_location}")
    if ss.sku: bits.append(f"SKU {ss.sku}")
    if ss.lot_number: bits.append(f"LOT {ss.lot_number}")
    st.toast("\n".join(bits), icon="✅")

    ss.recent_scans.insert(0, (datetime.now().strftime("%H:%M:%S"), code))
    ss.recent_scans = ss.recent_scans[:25]
    ss.scan = ""

def on_source_scan():
    raw = ss.source_scan or ""
    code = clean_scan(raw)
    if not code:
        return
    ss.current_location = code
    st.toast(f"Source Location set: {code}", icon="📍")
    ss.source_scan = ""

def on_staging_scan():
    raw = ss.staging_scan or ""
    code = clean_scan(raw)
    if not code:
        return
    ss.staging_location_current = code
    st.toast(f"Staging Location set: {code}", icon="📍")
    ss.staging_scan = ""

def on_lot_scan():
    raw = ss.lot_scan or ""
    code = clean_scan(raw)
    if not code:
        return
    ss.lot_number = normalize_lot(code)
    st.toast(f"LOT set: {ss.lot_number}", icon="🧾")
    ss.lot_scan = ""

def set_typed_source():
    val = clean_scan(ss.typed_source)
    ss.current_location = val
    st.toast(f"Source Location set: {val}", icon="✍️")

def set_typed_pallet():
    val = clean_scan(ss.typed_pallet_id)
    if not val:
        return
    ss.current_pallet = val
    # try lookup for sku/lot/location
    if ss.lookup_df is not None:
        lookup_map = {
            "pallet": ss.lookup_cols.get("pallet"),
            "sku": ss.lookup_cols.get("sku"),
            "lot": ss.lookup_cols.get("lot"),
            "location": ss.lookup_cols.get("location"),
        }
        looked = lookup_fields_by_pallet(ss.lookup_df, lookup_map, val)
        if looked.get("location"): ss.current_location = looked["location"]
        if looked.get("sku") is not None: ss.sku = looked.get("sku") or ""
        if looked.get("lot") is not None: ss.lot_number = looked.get("lot") or ""
    st.toast(f"Pallet set: {val}", icon="📦")

def set_typed_lot():
    val = normalize_lot(ss.typed_lot)
    ss.lot_number = val
    st.toast(f"LOT set: {val}", icon="✍️")

# ------------------ MAIN UI ------------------
# A) Pallet — scan or type
st.subheader("Pallet ID")
cP1, cP2 = st.columns([2, 1])
with cP1:
    st.text_input("Scan pallet here", key="scan",
        placeholder="Scan pallet barcode (JSON/query/GS1/label parsed if present)",
        on_change=on_pallet_scan)
with cP2:
    st.text_input("…or type Pallet ID", key="typed_pallet_id", placeholder="e.g., JTL00496")
    st.button("Set Pallet", on_click=set_typed_pallet, use_container_width=True)

with st.expander("🔍 Raw Scan Debugger (last pallet scan)", expanded=False):
    if ss.last_raw_scan:
        raw = ss.last_raw_scan
        st.code(repr(raw), language="text")
        # Show code points (helpful to spot \x1D FNC1)
        hexes = " ".join(f"{ord(c):02X}" for c in raw)
        st.caption(f"Hex bytes: {hexes}")
        if "\x1D" in raw:
            st.success("Detected ASCII 29 (FNC1) in the raw scan — GS1 mode.", icon="✅")
        else:
            st.info("No FNC1 (ASCII 29) detected in raw scan.", icon="ℹ️")
    else:
        st.caption("No raw pallet scan captured yet.")

# B) Source Location — scan or type
st.subheader("Source Location")
cL1, cL2 = st.columns([2, 1])
with cL1:
    st.text_input("Scan source location here", key="source_scan",
        placeholder="Scan location barcode (e.g., 11400804)",
        on_change=on_source_scan)
with cL2:
    st.text_input("…or type location", key="typed_source", placeholder="e.g., 11400804")
    st.button("Set Location", on_click=set_typed_source, use_container_width=True)

# Inline warning icon if current_location isn't 8 digits numeric (visual cue only)
if ss.current_location:
    if not _is_valid_source_location(ss.current_location):
        st.warning("Expected 8 digits (e.g., 11400804). You can proceed, but please double‑check.", icon="⚠️")
    else:
        st.caption("✅ Source Location format looks good.")

# C) LOT — scan or type
st.subheader("LOT Number")
cLot1, cLot2 = st.columns([2, 1])
with cLot1:
    st.text_input("Scan LOT here", key="lot_scan",
        placeholder="Scan LOT barcode (digits preferred)",
        on_change=on_lot_scan)
with cLot2:
    st.text_input("…or type LOT", key="typed_lot", placeholder="e.g., 9062716")
    st.button("Set LOT", on_click=set_typed_lot, use_container_width=True)

# D) Staging Location — scan or type
st.subheader("Staging Location")
cS1, cS2 = st.columns([2, 1])
with cS1:
    st.text_input("Scan staging here", key="staging_scan",
        placeholder="Scan staging barcode (e.g., STAGE-01)",
        on_change=on_staging_scan)
with cS2:
    st.text_input("…or type staging", key="staging_location_current", placeholder="e.g., STAGE-01")

# Current context metrics
st.markdown("---")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("Order # (optional)", ss.current_order or "—")
with c2: st.metric("Source Location", ss.current_location or "—")
with c3: st.metric("Pallet ID", ss.current_pallet or "—")
with c4: st.metric("SKU", ss.sku or "—")
with c5: st.metric("LOT Number", ss.lot_number or "—")
with c6:
    rem = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet) if ss.current_pallet else None
    st.metric("Remaining (est.)", "—" if rem is None else rem)
with st.expander("Recent scans", expanded=False):
    if ss.recent_scans:
        for t, c in ss.recent_scans[:10]:
            st.write(f"🕒 {t} — **{c}**")
    else:
        st.info("No scans yet.")
st.markdown("---")

# QTY staged — numeric only (no presets)
st.subheader("QTY staged (1–15)")
qty_staged = st.number_input(
    "Enter QTY staged",
    min_value=0, max_value=15, step=1, value=ss.qty_staged or 0,
    help="Type the quantity staged for this line (max 15)."
)
ss.qty_staged = qty_staged

# Optional running tracker
st.subheader("Pick (optional tracker)")
pick_qty = st.number_input("Qty picked", min_value=0, step=1, value=0)
colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    add_to_batch = st.button("➕ Add to Batch", use_container_width=True)
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
with colD:
    clear_batch = st.button("🗑️ Clear Batch", use_container_width=True, disabled=not ss.batch_rows)

# ---- Add to Batch ----
if add_to_batch:
    if not ss.current_pallet:
        st.error("Set a **Pallet ID** (scan or type) before adding.")
    elif not ss.current_location:
        st.error("Set a **Source Location** (scan or type) before adding.")
    elif ss.qty_staged <= 0:
        st.error("Enter a **QTY staged** > 0.")
    else:
        if pick_qty > 0:
            upsert_picked(ss.current_pallet, int(pick_qty))
        line = {
            "order_number": ss.current_order or "",
            "source_location": ss.current_location,
            "staging_location": ss.staging_location_current or "",
            "pallet_id": ss.current_pallet,
            "sku": ss.sku or "",
            "lot_number": ss.lot_number or "",
            "qty_staged": int(ss.qty_staged),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        ss.batch_rows.append(line)
        st.success(
            f"Added: Pallet {line['pallet_id']} — QTY {line['qty_staged']}"
            + (f" → {line['staging_location']}" if line['staging_location'] else "")
        )
        # reset qty only
        ss.qty_staged = 0

# ---- Tag generation ----
if do_tag:
    if not ss.current_pallet:
        st.error("Set a **Pallet ID** before generating a tag.")
    else:
        remaining = get_remaining(ss.starting_qty if ss.starting_qty != 0 else None, ss.current_pallet)
        ss.tag_bytes = make_partial_tag_png(
            location=ss.current_location,
            pallet=ss.current_pallet,
            qty_remaining=remaining,
            operator=ss.operator or ""
        )
if ss.tag_bytes:
    st.subheader("Partial Pallet Tag")
    st.image(ss.tag_bytes, caption="Preview", use_column_width=True)
    st.download_button(
        "⬇️ Download Tag (PNG)",
        data=ss.tag_bytes,
        file_name=f"partial-tag-{ss.current_pallet or 'pallet'}.png",
        mime="image/png"
    )

# Clear batch
if clear_batch and ss.batch_rows:
    ss.batch_rows = []
    st.warning("Batch cleared.")

# ------------------ REVIEW & SUBMIT ------------------
st.markdown("---")
st.subheader("Review & Submit")
if not ss.batch_rows:
    st.info("No items in the batch yet. Add lines above.")
else:
    try:
        import pandas as pd
        df_batch = pd.DataFrame(ss.batch_rows)
    except Exception:
        df_batch = None
    if df_batch is not None:
        st.caption("Edit QTY or fields if needed (0-qty rows are dropped). Then **Apply Edits** → **Submit All**.")
        edited = st.data_editor(
            df_batch,
            use_container_width=True,
            hide_index=True,
            column_config={
                "order_number": "Order # (optional)",
                "source_location": "Source Location",
                "staging_location": "Staging Location",
                "pallet_id": "Pallet ID",
                "sku": "SKU",
                "lot_number": "LOT",
                "qty_staged": st.column_config.NumberColumn("QTY Staged", min_value=0, max_value=15),
                "timestamp": "Timestamp",
            },
            num_rows="fixed",
            key="batch_editor"
        )
        colR1, colR2, colR3 = st.columns([1,1,2])
        with colR1:
            apply_edits = st.button("💾 Apply Edits", use_container_width=True)
        with colR2:
            submit_all = st.button("✅ Submit All", use_container_width=True)
        with colR3:
            try:
                total_lines = len(edited)
                total_qty = int(edited["qty_staged"].sum())
            except Exception:
                total_lines = len(ss.batch_rows)
                total_qty = sum(int(r.get("qty_staged", 0)) for r in ss.batch_rows)
            st.metric("Batch Totals", f"{total_lines} lines / {total_qty} cases")

        if apply_edits:
            new_rows = []
            for _, r in edited.iterrows():
                qty = int(r.get("qty_staged", 0) or 0)
                if qty <= 0:
                    continue
                if qty > 15:
                    st.error(f"Line for pallet {r.get('pallet_id','?')}: qty {qty} > 15 (max). Using 15.")
                    qty = 15
                new_rows.append({
                    "order_number": str(r.get("order_number","")),
                    "source_location": str(r.get("source_location","")),
                    "staging_location": str(r.get("staging_location","")),
                    "pallet_id": str(r.get("pallet_id","")),
                    "sku": str(r.get("sku","")),
                    "lot_number": normalize_lot(str(r.get("lot_number",""))),
                    "qty_staged": qty,
                    "timestamp": str(r.get("timestamp","")),
                })
            ss.batch_rows = new_rows
            st.success("Edits applied.")

        if submit_all:
            if not ss.batch_rows:
                st.error("Batch is empty.")
            else:
                bad = []
                for r in ss.batch_rows:
                    if not r.get("pallet_id") or not r.get("source_location"):
                        bad.append(r)
                    q = int(r.get("qty_staged", 0))
                    if q <= 0 or q > 15:
                        bad.append(r)
                if bad:
                    st.error("Some lines are invalid (missing pallet/location or qty not in 1–15). Fix and try again.")
                else:
                    batch_id = f"BATCH-{datetime.now().isoformat(timespec='seconds')}-{(ss.operator or 'operator')}".replace(":", "")
                    submitted_at = datetime.now().isoformat(timespec="seconds")
                    totals_qty = sum(int(r["qty_staged"]) for r in ss.batch_rows)
                    payload = {
                        "batch_id": batch_id,
                        "submitted_at": submitted_at,
                        "operator": ss.operator or "",
                        "rows": ss.batch_rows,
                        "totals": {"lines": len(ss.batch_rows), "qty_staged_sum": totals_qty},
                        "notify_to": NOTIFY_TO,  # recipients for Flow
                    }
                    # Send to webhook (if configured)
                    sent_ok = False
                    send_error = None
                    if WEBHOOK_URL:
                        try:
                            import requests
                            r = requests.post(WEBHOOK_URL, json=payload, timeout=12)
                            r.raise_for_status()
                            sent_ok = True
                        except Exception as e:
                            send_error = str(e)
                    # Write CSV rows locally (audit)
                    # Also attempt to fill qty_picked & remaining_after if starting_qty present
                    for r in ss.batch_rows:
                        qty_picked = ""  # unknown per-line unless you use the optional tracker
                        remaining_after = ""
                        if ss.starting_qty and ss.current_pallet and r.get("pallet_id") == ss.current_pallet:
                            # best-effort: compute remaining using running tracker
                            current_picked = ss.picked_so_far.get(ss.current_pallet, 0)
                            remaining_after = max(ss.starting_qty - current_picked, 0)
                        append_log_row({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "operator": ss.operator or "",
                            "order_number": r.get("order_number",""),
                            "location": r.get("source_location",""),
                            "staging_location": r.get("staging_location",""),
                            "pallet_id": r.get("pallet_id",""),
                            "sku": r.get("sku",""),
                            "lot_number": r.get("lot_number",""),
                            "qty_staged": int(r.get("qty_staged", 0)) or "",
                            "qty_picked": qty_picked,
                            "starting_qty": (ss.starting_qty if ss.starting_qty != 0 else ""),
                            "remaining_after": remaining_after,
                            "batch_id": batch_id,
                            "action": "SUBMIT",
                        })
                    # Save JSON backup of payload
                    try:
                        with open(os.path.join(LOG_DIR, f"{batch_id}.json"), "w", encoding="utf-8") as jf:
                            json.dump(payload, jf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    if WEBHOOK_URL and not sent_ok:
                        st.error(f"Submit saved locally, but webhook failed: {send_error}")
                    else:
                        st.success("Submitted! Admin will receive the email/Teams notification.")
                    ss.batch_rows = []
                    st.toast(
                        f"Batch {batch_id} submitted — {payload['totals']['lines']} lines / {payload['totals']['qty_staged_sum']} cases.",
                        icon="📨"
                    )
                    time.sleep(0.5)
                    st.rerun()

# ------------------ Today’s Log Preview ------------------
st.markdown("---")
st.subheader("Today’s Log (preview)")
if os.path.exists(LOG_FILE):
    try:
        import pandas as pd
        dfprev = pd.read_csv(LOG_FILE)
        st.dataframe(dfprev.tail(50), use_container_width=True, height=320)
    except Exception:
        st.code(open(LOG_FILE, "r", encoding="utf-8", errors="ignore").read().splitlines()[-10:])
else:
    st.info("No log entries yet today.")