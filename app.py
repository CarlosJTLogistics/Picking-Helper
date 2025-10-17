# app.py — Outbound Picking Helper (Batch Submit)
# v1.8 — Persistent Lookup (mobile-safe) + Saved Column Mapping + Mobile UI tweaks
# v1.7.1 — TZ-aware + Remaining-cap enforcement + Teams webhook + preserves AIM/LOT/QTY defaults & auto-clear

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

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Picking Helper", page_icon="📦", layout="wide")

# --- Mobile-friendly CSS (larger touch targets + compact padding on phones)
st.markdown("""
<style>
/* Bigger buttons */
.stButton > button, .stDownloadButton > button {
    min-height: 48px;
    font-size: 1.05rem;
}
/* Inputs a bit larger for touch */
div[data-baseweb="input"] input, .stNumberInput input, .stTextInput input {
    min-height: 44px;
    font-size: 1.02rem;
}
/* Compact padding on narrow screens */
@media (max-width: 640px) {
  .block-container { padding-top: 0.6rem; padding-left: 0.5rem; padding-right: 0.5rem; }
  h1, h2, h3 { line-height: 1.2; }
}
</style>
""", unsafe_allow_html=True)

# -------------------- CONFIG (Secrets-first, env fallback) --------------------
def _get_cfg():
    sec = st.secrets.get("picking_helper", {}) if hasattr(st, "secrets") else {}
    def get(name, env, default=None):
        val = sec.get(name) if isinstance(sec, dict) else None
        if val in (None, "", []):
            val = os.getenv(env, default)
        return val
    cfg = {
        "webhook_url": get("webhook_url", "PICKING_HELPER_WEBHOOK_URL", ""),
        "teams_webhook_url": get("teams_webhook_url", "PICKING_HELPER_TEAMS_WEBHOOK_URL", ""),
        "notify_to": get("notify_to", "PICKING_HELPER_NOTIFY_TO", ""),
        "lookup_file": get("lookup_file", "PICKING_HELPER_LOOKUP_FILE", ""),
        "log_dir": get("log_dir", "PICKING_HELPER_LOG_DIR", "logs"),
        "kiosk": str(get("kiosk", "PICKING_HELPER_KIOSK", "0")).strip().lower() in ("1","true","yes"),
        "require_location": str(get("require_location", "PICKING_HELPER_REQUIRE_LOCATION", "1")).strip().lower() in ("1","true","yes"),
        "require_staging": str(get("require_staging", "PICKING_HELPER_REQUIRE_STAGING", "0")).strip().lower() in ("1","true","yes"),
        "scan_to_add": str(get("scan_to_add", "PICKING_HELPER_SCAN_TO_ADD", "1")).strip().lower() in ("1","true","yes"),
        "keep_staging_after_add": str(get("keep_staging_after_add", "PICKING_HELPER_KEEP_STAGING_AFTER_ADD", "1")).strip().lower() in ("1","true","yes"),
        # Explicit timezone (IANA). Default to America/Chicago for Central Time.
        "timezone": get("timezone", "PICKING_HELPER_TIMEZONE", "America/Chicago"),
    }
    nt = cfg["notify_to"]
    if isinstance(nt, list):
        notify = nt
    else:
        parts = re.split(r"[,;]", nt) if nt else []
        notify = [p.strip() for p in parts if p.strip()]
    if not notify:
        notify = [
            "carlos.pacheco@jtlogistics.com",
            "Alex.Miller@jtlogistics.com",
            "Cody.Robles@JTlogistics.com",
        ]
    cfg["notify_to"] = notify
    cfg["_source"] = "Secrets" if isinstance(sec, dict) and sec else "Env"
    return cfg
CFG = _get_cfg()

# -------------------- TIMEZONE HELPERS --------------------
def get_tz():
    tzname = (CFG.get("timezone") or "America/Chicago").strip()
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tzname)
    except Exception:
        return None
def now_local():
    tz = get_tz()
    return datetime.now(tz) if tz else datetime.now()
def ts12(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = now_local()
    else:
        try:
            tz = get_tz()
            if tz:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=tz)
                else:
                    dt = dt.astimezone(tz)
        except Exception:
            pass
    return dt.strftime("%Y-%m-%d %I:%M:%S %p")

# -------------------- ENV / PATHS --------------------
TODAY = now_local().strftime("%Y-%m-%d")  # local date for file naming
LOG_DIR = CFG["log_dir"] or "logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"picking-log-{TODAY}.csv")
WEBHOOK_URL = (CFG["webhook_url"] or "").strip()
TEAMS_WEBHOOK_URL = (CFG["teams_webhook_url"] or "").strip()
NOTIFY_TO: List[str] = CFG["notify_to"]
LOOKUP_FILE_ENV = (CFG["lookup_file"] or "").strip()
KIOSK = CFG["kiosk"]

# -- Persistent Lookup Paths
PERSIST_DIR = os.path.join(LOG_DIR, "lookup")
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
META_PATH = os.path.join(PERSIST_DIR, "lookup_meta.json")

def _read_lookup_meta() -> Dict:
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_lookup_meta(meta: Dict):
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _persist_uploaded_lookup(filename: str, raw_bytes: bytes) -> str:
    # Save to logs/lookup/lookup-latest.<ext>
    ext = ""
    if "." in filename:
        ext = "." + filename.split(".")[-1].lower().strip()
    if ext not in [".csv", ".xlsx", ".xls"]:
        # default to .csv if unknown
        ext = ".csv"
    save_path = os.path.join(PERSIST_DIR, f"lookup-latest{ext}")
    with open(save_path, "wb") as f:
        f.write(raw_bytes)
    meta = _read_lookup_meta()
    meta.update({
        "saved_path": save_path,
        "original_name": filename,
        "uploaded_at": ts12(),
        "size_bytes": len(raw_bytes),
    })
    _write_lookup_meta(meta)
    return save_path

def _get_persisted_lookup_path() -> Optional[str]:
    # Priority: Secrets/Env (LOOKUP_FILE_ENV) > meta.saved_path (if exists)
    if LOOKUP_FILE_ENV and os.path.exists(LOOKUP_FILE_ENV):
        return LOOKUP_FILE_ENV
    meta = _read_lookup_meta()
    p = meta.get("saved_path")
    return p if p and os.path.exists(p) else None

# -------------------- SESSION STATE --------------------
ss = st.session_state
defaults = {
    "operator": "",
    "current_location": "",
    "current_pallet": "",
    "sku": "",
    "lot_number": "",
    "staging_location_current": "",
    "scan": "",
    "staging_scan": "",
    "typed_staging": "",
    "qty_picked_str": "",
    "qty_staged": 0,
    "starting_qty": None,
    "picked_so_far": {},
    "recent_scans": [],
    "last_raw_scan": "",
    "focus_qty": False,
    "lookup_df": None,
    "lookup_cols": {"pallet": None, "sku": None, "lot": None, "location": None, "qty": None},
    "batch_rows": [],
    "undo_stack": [],
    "redo_stack": [],
    "start_qty_by_pallet": {},
    "pallet_qty": None,
    "clear_qty_next": False,
    "clear_top_next": False,
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# -------------------- HELPERS --------------------
def clean_scan(raw: str) -> str:
    return (raw or "").replace("\r", "").replace("\n", "").strip()
def strip_aim_prefix(s: str) -> str:
    m = re.match(r"^\][A-Za-z]\d", s)
    return s[m.end():] if m else s
def safe_int(x, default=0) -> int:
    try:
        if x is None: return default
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return int(x)
        s = str(x).strip()
        if s == "": return default
        s2 = re.sub(r"[^\d\-]", "", s)
        return int(s2) if s2 not in ("","-") else default
    except Exception:
        return default
def clamp_nonneg(n: Optional[int]) -> Optional[int]:
    if n is None: return None
    return max(int(n), 0)
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
def upsert_picked(pallet_id: str, qty: int):
    ss.picked_so_far[pallet_id] = ss.picked_so_far.get(pallet_id, 0) + max(safe_int(qty,0),0)
def get_start_qty(pallet_id: str) -> Optional[int]:
    if not pallet_id:
        return None
    if pallet_id in ss.start_qty_by_pallet:
        return safe_int(ss.start_qty_by_pallet[pallet_id], 0)
    if pallet_id == ss.current_pallet and ss.starting_qty not in (None, 0):
        return safe_int(ss.starting_qty, 0)
    return None
def get_remaining_for_pallet(pallet_id: str) -> Optional[int]:
    start = get_start_qty(pallet_id)
    if start is None:
        return None
    picked = safe_int(ss.picked_so_far.get(pallet_id, 0), 0)
    return clamp_nonneg(start - picked)
def normalize_lot(lot: Optional[str]) -> str:
    if not lot:
        return ""
    digits = re.sub(r"\D", "", str(lot))
    return digits.lstrip("0") or "0"

# -------------------- LOOKUP SUPPORT --------------------
def _auto_guess_cols(columns: List[str]) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in columns}
    def pick(syns: List[str]) -> Optional[str]:
        for s in syns:
            if s in cols_lower:
                return cols_lower[s]
        for c in columns:
            cl = str(c).lower()
            if any(s in cl for s in syns):
                return c
        return None
    return {
        "pallet": pick(["pallet","pallet id","pallet_id","lpn","license","serial","sscc"]),
        "sku": pick(["sku","item","itemcode","product","part","material"]),
        "lot": pick(["customerlotreference","customer lot reference","lot","lot_number","lot #","lot#","batch","batchno"]),
        "location": pick(["location","loc","bin","slot","binlocation","location code","staging","stg"]),
        "qty": pick(["qtyonhand","qty","quantity","cases","casecount","count","units","pallet_qty","onhand","on_hand"]),
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

def lookup_fields_by_pallet(df, colmap: Dict[str, Optional[str]], pallet_id: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
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
            if (c := colmap.get("sku")) and c in df.columns:
                out["sku"] = str(row[c]).strip()
            if (c := colmap.get("lot")) and c in df.columns:
                out["lot"] = normalize_lot(row[c])
            if (c := colmap.get("location")) and c in df.columns:
                out["location"] = str(row[c]).strip()
            if (c := colmap.get("qty")) and c in df.columns:
                out["pallet_qty"] = str(row[c]).strip()
    except Exception:
        pass
    return out

# -------------------- SCAN PARSERS --------------------
def try_parse_json(s: str) -> Optional[Dict[str,str]]:
    try: obj = json.loads(s); return obj if isinstance(obj, dict) else None
    except Exception: return None
def try_parse_query_or_kv(s: str) -> Optional[Dict[str,str]]:
    if "://" in s:
        parsed = urlparse(s)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        return {k.lower(): unquote_plus(v[-1]) if v else "" for k, v in qs.items()} or None
    s2 = s.replace(";", "&").replace("\n", "&").replace(",", "&").replace(" ", " ")
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
    lines = re.split(r"[\n;,]+", s)
    out = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower(); v = v.strip()
            if k and v: out[k] = v
    return out or None
def try_parse_label_compact(s: str) -> Optional[Dict[str,str]]:
    tokens = re.split(r"[,\\\s;]+", s.strip())
    out = {}
    i = 0
    while i < len(tokens) - 1:
        key = tokens[i].lower()
        val = tokens[i+1]
        if re.fullmatch(r"[a-zA-Z#]+", key):
            out[key] = val
            i += 2
        else:
            i += 1
    return out or None
def try_parse_gs1_paren(s: str) -> Optional[Dict[str,str]]:
    pairs = re.findall(r"\((\d{2,4})\)([^()]+)", s)
    if not pairs:
        return None
    out: Dict[str,str] = {}
    for ai, val in pairs:
        val = val.strip()
        if ai == "21": out["pallet"] = val
        elif ai == "10": out["lot"] = val
        elif ai == "01": out["gtin"] = val
        elif ai == "00": out["pallet"] = val
        elif ai in {"240","241"}: out["pallet"] = val
    return out or None
def try_parse_gs1_fnc1(s: str) -> Optional[Dict[str,str]]:
    GS = "\x1D"
    if GS not in s and not re.match(r"^\d{2,4}", s):
        return None
    AI_FIXED = {"00":18, "01":14}
    AI_VAR = {"10","21","240","241"}
    i = 0
    n = len(s)
    out: Dict[str,str] = {}
    def read_ai(idx: int) -> Tuple[Optional[str], int]:
        for L in (4,3,2):
            if idx+L <= n and s[idx:idx+L].isdigit():
                return s[idx:idx+L], idx+L
        return None, idx+1
    while i < n:
        if not s[i].isdigit():
            i += 1; continue
        ai, j = read_ai(i)
        if not ai:
            i += 1; continue
        if ai in AI_FIXED:
            L = AI_FIXED[ai]
            val = s[j:j+L] if j+L <= n else s[j:]; i = j+L if j+L <= n else n
        elif ai in AI_VAR:
            k = s.find(GS, j)
            if k == -1:
                val = s[j:]; i = n
            else:
                val = s[j:k]; i = k+1
        else:
            k = s.find(GS, j); i = (k+1) if k != -1 else n; continue
        val = val.strip()
        if ai in {"21","00","240","241"}:
            out["pallet"] = val
        elif ai == "10":
            out["lot"] = val
        elif ai == "01":
            out["gtin"] = val
    return out or None
def try_parse_gs1_numeric_naked(s: str) -> Optional[Dict[str,str]]:
    if not re.match(r"^\d{2,4}", s):
        return None
    out: Dict[str,str] = {}
    if s.startswith("00") and len(s) >= 20:
        out["pallet"] = s[2:20]
    elif s.startswith("01") and len(s) >= 16:
        out["gtin"] = s[2:16]
    elif s.startswith(("21","10")):
        val = s[2:]
        if s.startswith("21") and val: out["pallet"] = val
        elif s.startswith("10") and val: out["lot"] = val
    elif s.startswith(("240","241")):
        val = s[3:]
        if val: out["pallet"] = val
    return out or None
def normalize_keys(data: Dict[str,str]) -> Dict[str,str]:
    key_map = {
        "location": ["location","loc","bin","slot","staging","stg","binlocation","location code"],
        "pallet": ["pallet","pallet_id","pallet id","serial","id","license","lpn","sscc"],
        "sku": ["sku","item","itemcode","product","part","material"],
        "lot": ["customerlotreference","customer lot reference","lot","lot_number","lot #","lot#","batch","batchno"],
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
def parse_any_scan(raw_code: str) -> Dict[str,str]:
    code = strip_aim_prefix(raw_code)
    for fn in (
        try_parse_json,
        try_parse_query_or_kv,
        try_parse_gs1_paren,
        try_parse_gs1_fnc1,
        try_parse_gs1_numeric_naked,
        try_parse_label_kv,
        try_parse_label_compact,
    ):
        try:
            parsed = fn(code)
        except Exception:
            parsed = None
        if parsed:
            norm = normalize_keys(parsed)
            if norm:
                return norm
    return {}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.info("BUILD: outbound-batch v1.8 (Persistent Lookup • Saved Mapping • Mobile UI)", icon="🧭")
    st.caption(f"Config source: **{CFG['_source']}**")

    st.markdown("### ⚙️ Settings")
    ss["operator"] = st.text_input("Picker Name (required)", value=ss.operator, placeholder="e.g., Carlos")
    CFG["timezone"] = st.text_input(
        "Timezone (IANA)", value=CFG.get("timezone") or "America/Chicago",
        help="Examples: America/Chicago • America/Denver • UTC • America/New_York"
    )
    st.caption(f"Current app time: {ts12()} (TZ: {CFG['timezone'] or 'system'})")
    st.write("**Log file:**", f"`{LOG_FILE}`")
    st.caption(f"Recipients (email): {', '.join(NOTIFY_TO)}")
    if WEBHOOK_URL:
        st.caption("Power Automate webhook is configured (submissions POSTed there).")
    else:
        st.warning("Power Automate webhook URL is not set. Submissions will only write CSV locally.")
    if TEAMS_WEBHOOK_URL:
        st.caption("Teams Incoming Webhook configured — submissions will also post to that channel.")
    else:
        st.info("Optional: add `teams_webhook_url` (or env PICKING_HELPER_TEAMS_WEBHOOK_URL) to notify a Teams channel.")

    st.markdown("---")
    st.markdown("#### #### Inventory Lookup")
    st.caption("Upload your latest RAMP export (CSV/XLS/XLSX). Scans will auto-fill from this file + barcode contents.")

    # Figure out persisted path if any (unless explicit Secrets/Env path is set)
    persisted_path = _get_persisted_lookup_path()
    current_source_note = None
    if LOOKUP_FILE_ENV and os.path.exists(LOOKUP_FILE_ENV):
        current_source_note = f"Using Secrets/Env file: `{LOOKUP_FILE_ENV}`"
    elif persisted_path:
        current_source_note = f"Using saved lookup: `{persisted_path}`"
    else:
        current_source_note = "No saved lookup yet."

    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"], accept_multiple_files=False)

    cols_btn = st.columns([1,1,1])
    with cols_btn[0]:
        if st.button("↻ Clear lookup cache", use_container_width=True):
            try:
                load_lookup.clear()
            except Exception:
                pass
            st.toast("Lookup cache cleared — re-upload or rely on Secrets/saved file.", icon="🧹")
            st.rerun()
    with cols_btn[1]:
        # Remove persisted file + meta
        if st.button("🧹 Remove saved lookup file", use_container_width=True, disabled=not persisted_path):
            try:
                if persisted_path and os.path.exists(persisted_path):
                    os.remove(persisted_path)
            except Exception:
                pass
            try:
                if os.path.exists(META_PATH):
                    os.remove(META_PATH)
            except Exception:
                pass
            st.toast("Saved lookup file + metadata removed.", icon="🗑️")
            st.rerun()
    with cols_btn[2]:
        st.caption(current_source_note or "")

    # Read uploaded file (if any) and also persist it to disk for future sessions
    up_name = uploaded.name if uploaded is not None else None
    up_bytes = uploaded.read() if uploaded is not None else None

    # Decide which path to load from
    load_path = LOOKUP_FILE_ENV or (None if up_bytes else persisted_path)  # if uploading now, pass bytes

    df, guessed = None, None
    lookup_error = None
    try:
        df, guessed = load_lookup(load_path, up_bytes)
        ss.lookup_df = df
        # If this was a new upload and it parsed OK, persist it for next time
        if up_bytes and up_name and df is not None and len(df) > 0:
            saved_path = _persist_uploaded_lookup(up_name, up_bytes)
            st.toast(f"Saved lookup for future sessions: {saved_path}", icon="💾")
    except Exception as e:
        lookup_error = str(e)
        ss.lookup_df = None

    if lookup_error:
        st.error(f"Lookup load error: {lookup_error}")

    # Column mapping UI + defaults (from desired, guessed, and saved meta)
    saved_meta = _read_lookup_meta()
    saved_colmap = saved_meta.get("colmap") if isinstance(saved_meta.get("colmap"), dict) else {}

    if ss.lookup_df is not None:
        st.success(f"Loaded lookup with {len(ss.lookup_df):,} rows")
        cols = list(ss.lookup_df.columns)

        # desire known defaults first
        desired_defaults = {
            "pallet": "PalletID",
            "lot": "CustomerLotReference",
            "sku": "WarehouseSku",
            "location": "LocationName",
            "qty": None,
        }
        g = guessed or {"pallet": None, "sku": None, "lot": None, "location": None, "qty": None}

        # prefer saved mapping if columns exist
        for k, v in (saved_colmap or {}).items():
            if v and v in cols:
                g[k] = v

        # then well-known defaults if present
        for key, want in desired_defaults.items():
            if want and want in cols:
                g[key] = want
        qty_candidates = ["QTYOnHand", "QTY On Hand", "OnHandQty", "On Hand", "On_Hand"]
        for qn in qty_candidates:
            if g.get("qty"): break
            if qn in cols:
                g["qty"] = qn

        c1, c2 = st.columns(2)
        with c1:
            ss.lookup_cols["pallet"] = st.selectbox("Pallet column", options=cols, index=cols.index(g.get("pallet")) if g.get("pallet") in cols else 0)
            ss.lookup_cols["sku"] = st.selectbox("SKU column", options=["(none)"] + cols, index=(cols.index(g.get("sku")) + 1) if g.get("sku") in cols else 0)
        with c2:
            ss.lookup_cols["lot"] = st.selectbox("LOT column", options=["(none)"] + cols, index=(cols.index(g.get("lot")) + 1) if g.get("lot") in cols else 0)
            ss.lookup_cols["location"] = st.selectbox("Location column", options=["(none)"] + cols, index=(cols.index(g.get("location")) + 1) if g.get("location") in cols else 0)
            ss.lookup_cols["qty"] = st.selectbox("Pallet QTY column", options=["(none)"] + cols, index=(cols.index(g.get("qty")) + 1) if g.get("qty") in cols else 0)

        for k in ["sku","lot","location","qty"]:
            if ss.lookup_cols.get(k) == "(none)":
                ss.lookup_cols[k] = None

        # Save mapping button
        if st.button("💾 Save mapping as default", use_container_width=True):
            meta = _read_lookup_meta()
            meta["colmap"] = ss.lookup_cols
            _write_lookup_meta(meta)
            st.toast("Default mapping saved. It will auto-apply next time.", icon="✅")
    else:
        st.warning("No Inventory Lookup loaded — auto-fill is limited to what the barcode encodes (e.g., LPN via GS1 21/240, LOT via 10).", icon="ℹ️")

    st.markdown("---")
    st.markdown("#### #### Behavior")
    CFG["require_location"] = st.toggle("Require Location on Add", value=CFG["require_location"])
    CFG["require_staging"] = st.toggle("Require Staging on Add", value=CFG["require_staging"])
    CFG["scan_to_add"] = st.toggle("Scan-to-Add (after QTY, staging scan auto-adds)", value=CFG["scan_to_add"])
    CFG["keep_staging_after_add"] = st.toggle("Keep Staging Location after Add", value=CFG["keep_staging_after_add"])

    st.markdown("---")
    st.markdown("#### #### Start/Balance (optional)")
    ss["starting_qty"] = st.number_input(
        "Starting qty on current pallet (fallback if lookup lacks qty)",
        min_value=0, step=1, value=ss.starting_qty or 0,
        help="If your lookup provides pallet quantity, that value will override this for KPI & Remaining."
    )
    st.caption("Max picked per line remains 15 cases (validation).")
    if KIOSK:
        st.caption("KIOSK MODE enabled (menu/footer hidden).")

    with st.expander("Diagnostics", expanded=False):
        st.write({
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "streamlit": st.__version__,
            "LOOKUP_FILE (Secrets/Env)": LOOKUP_FILE_ENV or "(empty)",
            "persisted_lookup": _get_persisted_lookup_path() or "(none)",
            "cfg_source": CFG["_source"],
            "timezone": CFG.get("timezone"),
            "now_local": ts12(),
            "cwd": os.getcwd(),
            "files_in_cwd": sorted(os.listdir("."))[:50],
        })

if KIOSK:
    st.markdown("<style>#MainMenu, footer {visibility:hidden;}</style>", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("📦 Picking Helper — Outbound (Batch)")
st.caption("Scan Pallet → KPI shows Pallet QTY → QTY Picked (max 15, not above Remaining) → (optional) Staging → Add to Batch → Review & Submit")

def _focus_first_text():
    st.markdown("""
    <script>
    const f = () => {
      const inputs = window.parent.document.querySelectorAll('input[type="text"]');
      if (inputs && inputs.length) { inputs[0].focus(); inputs[0].select(); }
    };
    setTimeout(f, 200);
    </script>
    """, unsafe_allow_html=True)

def _focus_qty_input():
    st.markdown("""
    <script>
    const f = () => {
      const els = Array.from(window.parent.document.querySelectorAll('input[aria-label]'));
      const el = els.find(i => i.getAttribute('aria-label')?.toLowerCase()?.includes('enter qty picked'));
      if (el) {
        el.setAttribute('inputmode', 'numeric');  // helps mobile show numeric keyboard
        el.focus(); el.select();
      }
    };
    setTimeout(f, 150);
    </script>
    """, unsafe_allow_html=True)

# -------------------- SAFE PRE-CLEAR BEFORE UI --------------------
if ss.clear_top_next:
    ss.clear_top_next = False
    ss.current_pallet = ""
    ss.sku = ""
    ss.lot_number = ""
    ss.current_location = ""
    ss.pallet_qty = None
    ss.scan = ""  # pallet scan box
    if not CFG["keep_staging_after_add"]:
        ss.staging_location_current = ""
if ss.clear_qty_next:
    ss.clear_qty_next = False
    st.session_state["qty_picked_str"] = ""
_focus_first_text()

# -------------------- Pallet + Staging --------------------
def _apply_lookup_into_state(pallet_id: str):
    if ss.lookup_df is None:
        if ss.starting_qty and pallet_id:
            ss.pallet_qty = clamp_nonneg(safe_int(ss.starting_qty, 0))
            ss.start_qty_by_pallet[pallet_id] = ss.pallet_qty
        return
    lookup_map = {
        "pallet": ss.lookup_cols.get("pallet"),
        "sku": ss.lookup_cols.get("sku"),
        "lot": ss.lookup_cols.get("lot"),
        "location": ss.lookup_cols.get("location"),
        "qty": ss.lookup_cols.get("qty"),
    }
    looked = lookup_fields_by_pallet(ss.lookup_df, lookup_map, pallet_id)
    if looked.get("location"): ss.current_location = looked["location"]
    if looked.get("sku") is not None: ss.sku = looked.get("sku") or ""
    if looked.get("lot") is not None: ss.lot_number = looked.get("lot") or ""
    if looked.get("pallet_qty") is not None:
        qty_val = clamp_nonneg(safe_int(looked.get("pallet_qty"), 0))
        ss.pallet_qty = qty_val
        ss.start_qty_by_pallet[pallet_id] = qty_val
    elif ss.starting_qty and pallet_id:
        ss.pallet_qty = clamp_nonneg(safe_int(ss.starting_qty, 0))
        ss.start_qty_by_pallet[pallet_id] = ss.pallet_qty

def on_pallet_scan():
    raw = ss.scan or ""
    code = clean_scan(raw)
    if not code:
        return
    ss.last_raw_scan = raw
    norm = parse_any_scan(code)
    pallet_id = norm.get("pallet") or strip_aim_prefix(code) or code
    ss.current_pallet = pallet_id
    if norm.get("location"): ss.current_location = norm["location"]
    if norm.get("sku") is not None: ss.sku = norm.get("sku") or ""
    if norm.get("lot") is not None: ss.lot_number = norm.get("lot") or ""
    _apply_lookup_into_state(pallet_id)
    bits = [f"Pallet {ss.current_pallet}"]
    if ss.pallet_qty is not None: bits.append(f"QTY {ss.pallet_qty}")
    if ss.current_location: bits.append(f"Location {ss.current_location}")
    if ss.sku: bits.append(f"SKU {ss.sku}")
    if ss.lot_number: bits.append(f"LOT {ss.lot_number}")
    rem_now = get_remaining_for_pallet(ss.current_pallet)
    if rem_now is not None and rem_now <= 0:
        st.toast("Pallet fully picked — Remaining 0", icon="✅")
    else:
        st.toast("\n".join(bits), icon="✅")
    ss.recent_scans.insert(0, (now_local().strftime("%I:%M:%S %p"), strip_aim_prefix(code)))  # 12‑hour
    ss.recent_scans = ss.recent_scans[:25]
    ss.scan = ""
    ss.focus_qty = True

def on_staging_scan():
    raw = ss.staging_scan or ""
    code = clean_scan(raw)
    if not code:
        return
    ss.staging_location_current = code
    st.toast(f"Staging Location set: {code}", icon="📍")
    ss.staging_scan = ""
    if CFG["scan_to_add"] and ss.current_pallet and safe_int(ss.qty_staged,0) > 0:
        _add_current_line_to_batch()

def set_typed_staging():
    val = clean_scan(ss.typed_staging)
    ss.staging_location_current = val
    st.toast(f"Staging Location set: {val}", icon="✍️")

st.subheader("Pallet ID")
st.text_input("Scan pallet here", key="scan",
              placeholder="Scan pallet barcode (GS1/AIM/JSON/query/label parsed)",
              on_change=on_pallet_scan)
with st.expander("🔍 Raw Scan Debugger (last pallet scan)", expanded=False):
    if ss.last_raw_scan:
        raw = ss.last_raw_scan
        st.code(repr(raw), language="text")
        hexes = " ".join(f"{ord(c):02X}" for c in raw)
        st.caption(f"Hex bytes: {hexes}")
        if raw.startswith("]"):
            st.info(f"Detected AIM symbology prefix: {raw[:3]!r}", icon="🔖")
        if "\x1D" in raw:
            st.success("Detected ASCII 29 (FNC1) in the raw scan — GS1 mode.", icon="✅")
        else:
            st.info("No FNC1 (ASCII 29) detected in raw scan.", icon="ℹ️")
        st.caption(f"After stripping AIM: {strip_aim_prefix(clean_scan(raw))!r}")
    else:
        st.caption("No raw pallet scan captured yet.")

st.subheader("Staging Location")
cS1, cS2 = st.columns([2, 1])
with cS1:
    st.text_input("Scan staging here", key="staging_scan",
                  placeholder="Scan staging barcode (e.g., STAGE-01)",
                  on_change=on_staging_scan)
with cS2:
    st.text_input("…or type staging", key="typed_staging", placeholder="e.g., STAGE-01")
    st.button("Set Staging", on_click=set_typed_staging, use_container_width=True)

# -------------------- KPI / Metrics --------------------
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Picker Name", ss.operator or "—")
with c2: st.metric("Pallet ID", ss.current_pallet or "—")
with c3: st.metric("SKU", ss.sku or "—")
with c4: st.metric("LOT Number", ss.lot_number or "—")
with c5: st.metric("Source Location", ss.current_location or "—")
c6, c7 = st.columns(2)
with c6: st.metric("Pallet QTY", "—" if ss.pallet_qty is None else safe_int(ss.pallet_qty, 0))
with c7:
    rem = get_remaining_for_pallet(ss.current_pallet) if ss.current_pallet else None
    st.metric("Remaining", "—" if rem is None else rem)
with st.expander("Recent scans", expanded=False):
    if ss.recent_scans:
        for t, c in ss.recent_scans[:10]:
            st.write(f"🕒 {t} — **{c}**")
    else:
        st.info("No scans yet.")

st.markdown("---")

# -------------------- QTY Picked + Batch controls --------------------
st.subheader("QTY Picked (1–15)")
qty_str = st.text_input(
    "Enter QTY Picked",
    key="qty_picked_str",
    placeholder="e.g., 5",
    help="Type the quantity picked for this line (max 15 and not above Remaining)."
)
ss.qty_staged = safe_int(qty_str, 0)

# Compute add button disabled based on Remaining (if start qty is known)
_current_start = get_start_qty(ss.current_pallet)
_current_remaining = get_remaining_for_pallet(ss.current_pallet) if _current_start is not None else None
disable_add = (not bool(ss.operator)) or (
    _current_start is not None and _current_remaining is not None and _current_remaining <= 0
)
if ss.focus_qty:
    _focus_qty_input()
    ss.focus_qty = False

colA, colC, colD, colE = st.columns([1,1,1,1])
with colA:
    add_to_batch_click = st.button(
        "➕ Add to Batch",
        use_container_width=True,
        disabled=disable_add
    )
with colC:
    undo_last = st.button("↩ Undo Last Line", use_container_width=True, disabled=not ss.batch_rows and not ss.undo_stack)
with colD:
    st.download_button(
        "⬇️ Download Today’s CSV Log",
        data=open(LOG_FILE, "rb").read() if os.path.exists(LOG_FILE) else b"",
        file_name=f"picking-log-{TODAY}.csv",
        mime="text/csv",
        disabled=not os.path.exists(LOG_FILE),
        use_container_width=True
    )
with colE:
    clear_batch = st.button("🗑️ Clear Batch", use_container_width=True, disabled=not ss.batch_rows)

if disable_add and _current_start is not None and _current_remaining is not None and ss.current_pallet:
    st.info(f"Pallet **{ss.current_pallet}** is fully picked (Remaining 0).", icon="✅")

def _add_current_line_to_batch():
    if not ss.operator:
        st.error("Enter **Picker Name** before adding.")
        return False
    if not ss.current_pallet:
        st.error("Scan a **Pallet ID** before adding.")
        return False
    if CFG["require_location"] and not ss.current_location:
        st.error("No **Source Location** found for this pallet (lookup/scan).")
        return False
    if CFG["require_staging"] and not ss.staging_location_current:
        st.error("Scan/Type a **Staging Location** before adding.")
        return False
    q = safe_int(ss.qty_staged, 0)
    if q <= 0:
        st.error("Enter a **QTY Picked** > 0.")
        return False
    if q > 15:
        st.warning("QTY Picked capped at 15 cases per line.")
        q = 15
    start_qty = get_start_qty(ss.current_pallet)
    if start_qty is not None:
        remaining_before = get_remaining_for_pallet(ss.current_pallet) or 0
        if remaining_before <= 0:
            st.error(f"Pallet **{ss.current_pallet}** is fully picked. Remaining is 0 — cannot add more.")
            return False
        if q > remaining_before:
            st.warning(f"QTY {q} exceeds Remaining {remaining_before}. Using {remaining_before} instead.")
            q = remaining_before
    # Apply
    upsert_picked(ss.current_pallet, q)
    line = {
        "order_number": "",  # schema retained: left blank
        "source_location": ss.current_location or "",
        "staging_location": ss.staging_location_current or "",
        "pallet_id": ss.current_pallet,
        "sku": ss.sku or "",
        "lot_number": ss.lot_number or "",
        "qty_staged": int(q),
        "timestamp": ts12(),  # 12-hour local
    }
    ss.batch_rows.append(line)
    ss.undo_stack.append(("add", line))
    st.success(
        f"Added: Pallet {line['pallet_id']} — QTY {line['qty_staged']}"
        + (f" → {line['staging_location']}" if line['staging_location'] else "")
    )
    ss.clear_qty_next = True
    ss.clear_top_next = True
    ss.qty_staged = 0
    st.rerun()
    return True

if add_to_batch_click:
    _add_current_line_to_batch()
if clear_batch and ss.batch_rows:
    for r in ss.batch_rows:
        upsert_picked(r.get("pallet_id",""), -safe_int(r.get("qty_staged"),0))
    ss.undo_stack.append(("remove_all", ss.batch_rows.copy()))
    ss.batch_rows = []
    st.warning("Batch cleared.")
if undo_last:
    if ss.batch_rows:
        last = ss.batch_rows.pop()
        ss.undo_stack.append(("remove", last))
        upsert_picked(last.get("pallet_id",""), -safe_int(last.get("qty_staged"),0))
        st.warning(f"Removed last line for pallet {last.get('pallet_id','?')}. Click again to undo removal.", icon="↩")
    elif ss.undo_stack:
        act, row = ss.undo_stack.pop()
        if act == "remove":
            ss.batch_rows.append(row)
            upsert_picked(row.get("pallet_id",""), safe_int(row.get("qty_staged"),0))
            st.success(f"Restored line for pallet {row.get('pallet_id','?')}.")

# -------------------- REVIEW & SUBMIT --------------------
st.markdown("---")
st.subheader("Review & Submit")
if not ss.batch_rows:
    st.info("No items in the batch yet. Add lines above.")
else:
    try:
        import pandas as pd
        df = pd.DataFrame(ss.batch_rows)
    except Exception:
        df = None
    if df is not None and not df.empty:
        df["qty_staged"] = df["qty_staged"].apply(lambda x: max(safe_int(x,0),0))
        df["start_qty"] = df["pallet_id"].apply(lambda p: get_start_qty(p))
        df["cum_staged"] = df.groupby("pallet_id")["qty_staged"].cumsum()
        def rem_after(row):
            if row["start_qty"] is None:
                return None
            return max(safe_int(row["start_qty"],0) - safe_int(row["cum_staged"],0), 0)
        df["remaining_qty"] = df.apply(rem_after, axis=1)
        df["picker_name"] = ss.operator or ""
        view_cols = [
            "picker_name",
            "source_location",
            "staging_location",
            "pallet_id",
            "sku",
            "lot_number",
            "qty_staged",
            "remaining_qty",
            "timestamp",
        ]
        df_view = df[view_cols].copy()
        op_col1, op_col2 = st.columns([1,3])
        with op_col1:
            st.metric("Picker Name", ss.operator or "—")
        with op_col2:
            total_lines = len(df_view)
            total_qty = int(df["qty_staged"].sum())
            st.metric("Batch Totals", f"{total_lines} lines / {total_qty} cases")
        st.caption("Edit QTY or fields (0-qty rows are dropped). **Capped at Remaining** per pallet. Then **Apply Edits** → **Submit All**.")
        edited = st.data_editor(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "picker_name": st.column_config.TextColumn("Picker Name", disabled=True),
                "source_location": "Source Location",
                "staging_location": "Staging Location",
                "pallet_id": st.column_config.TextColumn("Pallet ID", disabled=True),
                "sku": st.column_config.TextColumn("SKU", disabled=True),
                "lot_number": st.column_config.TextColumn("LOT", disabled=True),
                "qty_staged": st.column_config.NumberColumn("QTY Picked", min_value=0, max_value=15),
                "remaining_qty": st.column_config.NumberColumn("Remaining QTY", disabled=True),
                "timestamp": st.column_config.TextColumn("Timestamp", disabled=True),
            },
            num_rows="fixed",
            key="batch_editor"
        )
        colR1, colR2 = st.columns([1,1])
        with colR1:
            apply_edits = st.button("💾 Apply Edits", use_container_width=True)
        with colR2:
            submit_all = st.button("✅ Submit All", use_container_width=True, disabled=(not bool(ss.operator)))

        if apply_edits:
            # Enforce per-pallet remaining caps while applying edits
            new_rows: List[Dict] = []
            warn_msgs: List[str] = []
            running_totals: Dict[str, int] = {}
            for _, r in edited.iterrows():
                pallet = str(r.get("pallet_id","")).strip()
                if not pallet:
                    continue
                qty = safe_int(r.get("qty_staged", 0), 0)
                if qty <= 0:
                    continue
                if qty > 15:
                    warn_msgs.append(f"Pallet {pallet}: qty {qty} > 15 — using 15.")
                    qty = 15
                start_q = get_start_qty(pallet)
                if start_q is not None:
                    used = running_totals.get(pallet, 0)
                    remaining_here = max(start_q - used, 0)
                    if remaining_here <= 0:
                        warn_msgs.append(f"Pallet {pallet}: fully picked — skipping this line.")
                        continue
                    if qty > remaining_here:
                        warn_msgs.append(f"Pallet {pallet}: qty {qty} exceeds Remaining {remaining_here} — using {remaining_here}.")
                        qty = remaining_here
                    running_totals[pallet] = used + qty
                # Build row
                new_rows.append({
                    "order_number": "",
                    "source_location": str(r.get("source_location","")),
                    "staging_location": str(r.get("staging_location","")),
                    "pallet_id": pallet,
                    "sku": str(r.get("sku","")),
                    "lot_number": normalize_lot(str(r.get("lot_number",""))),
                    "qty_staged": qty,
                    "timestamp": str(r.get("timestamp","")),
                })
            # Rebuild picked_so_far based on new rows
            ss.picked_so_far = {}
            for row in new_rows:
                upsert_picked(row.get("pallet_id",""), safe_int(row.get("qty_staged"),0))
            ss.batch_rows = new_rows
            if warn_msgs:
                st.warning("• " + "\n• ".join(warn_msgs))
            st.success("Edits applied (remaining caps enforced).")

        if submit_all:
            if not ss.operator:
                st.error("Picker Name is required.")
            elif not ss.batch_rows:
                st.error("Batch is empty.")
            else:
                # Final validation: do not allow any pallet to exceed its start qty
                overfull: List[str] = []
                totals: Dict[str, int] = {}
                for r in ss.batch_rows:
                    p = r.get("pallet_id","")
                    q = safe_int(r.get("qty_staged", 0), 0)
                    totals[p] = totals.get(p, 0) + q
                for p, s in totals.items():
                    start_q = get_start_qty(p)
                    if start_q is not None and s > start_q:
                        overfull.append(f"{p} (total {s} > start {start_q})")
                if overfull:
                    st.error("Cannot submit: these pallets would exceed their starting quantity:\n- " + "\n- ".join(overfull))
                else:
                    # Use local, tz-aware time for IDs and ISO
                    dt_now = now_local()
                    batch_id = f"BATCH-{dt_now.isoformat(timespec='seconds')}-{(ss.operator or 'operator')}".replace(":", "")
                    submitted_at_iso = dt_now.isoformat(timespec="seconds")
                    totals_qty = sum(safe_int(r["qty_staged"],0) for r in ss.batch_rows)
                    payload = {
                        "batch_id": batch_id,
                        "submitted_at": submitted_at_iso,
                        "submitted_at_local": ts12(),
                        "operator": ss.operator or "",
                        "rows": ss.batch_rows,
                        "totals": {"lines": len(ss.batch_rows), "qty_staged_sum": totals_qty},
                        "notify_to": NOTIFY_TO,
                    }
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
                    if TEAMS_WEBHOOK_URL:
                        try:
                            import requests
                            card = {
                                "text": f"**Picking Batch Submitted**\n"
                                        f"- Batch: `{batch_id}`\n"
                                        f"- Picker: **{payload['operator']}**\n"
                                        f"- Lines: **{payload['totals']['lines']}**\n"
                                        f"- Cases: **{payload['totals']['qty_staged_sum']}**\n"
                                        f"- Submitted: {payload['submitted_at_local']}"
                            }
                            requests.post(TEAMS_WEBHOOK_URL, json=card, timeout=10)
                        except Exception:
                            pass
                    # Write CSV log rows (12-hour local timestamps)
                    for r in ss.batch_rows:
                        pal = r.get("pallet_id","")
                        start_qty = get_start_qty(pal)
                        current_picked = safe_int(ss.picked_so_far.get(pal, 0), 0)
                        remaining_after = clamp_nonneg((start_qty - current_picked) if start_qty is not None else None)
                        append_log_row({
                            "timestamp": ts12(),
                            "operator": ss.operator or "",
                            "order_number": "",
                            "location": r.get("source_location",""),
                            "staging_location": r.get("staging_location",""),
                            "pallet_id": pal,
                            "sku": r.get("sku",""),
                            "lot_number": r.get("lot_number",""),
                            "qty_staged": safe_int(r.get("qty_staged", 0), 0) or "",
                            "qty_picked": safe_int(r.get("qty_staged", 0), 0) or "",
                            "starting_qty": (start_qty if start_qty is not None else ""),
                            "remaining_after": (remaining_after if remaining_after is not None else ""),
                            "batch_id": batch_id,
                            "action": "SUBMIT",
                        })
                    try:
                        with open(os.path.join(LOG_DIR, f"{batch_id}.json"), "w", encoding="utf-8") as jf:
                            json.dump(payload, jf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    if WEBHOOK_URL and not sent_ok:
                        st.error(f"Submit saved locally, but webhook failed: {send_error}")
                    else:
                        st.success("Submitted! Notifications sent and history captured locally.")
                    ss.batch_rows = []
                    st.toast(
                        f"Batch {batch_id} submitted — {payload['totals']['lines']} lines / {payload['totals']['qty_staged_sum']} cases.",
                        icon="📨"
                    )
                    time.sleep(0.5)
                    st.rerun()

# -------------------- Today’s Log Preview --------------------
st.markdown("---")
st.subheader("Today’s Log (preview)")
if os.path.exists(LOG_FILE):
    try:
        import pandas as pd
        dfprev = pd.read_csv(LOG_FILE)
        if "order_number" in dfprev.columns:
            dfprev = dfprev[[c for c in dfprev.columns if c != "order_number"]]
        st.dataframe(dfprev.tail(50), use_container_width=True, height=320)
    except Exception:
        st.code(open(LOG_FILE, "r", encoding="utf-8", errors="ignore").read().splitlines()[-10:])
else:
    st.info("No log entries yet today.")