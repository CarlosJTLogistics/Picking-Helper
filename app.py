# app.py — Outbound Picking Helper (Batch Submit)
# v1.6.5 — 12-hour timestamps; optional Teams Incoming Webhook notify; preserves AIM/LOT/QTY defaults & auto-clear

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
        "teams_webhook_url": get("teams_webhook_url", "PICKING_HELPER_TEAMS_WEBHOOK_URL", ""),  # NEW
        "notify_to": get("notify_to", "PICKING_HELPER_NOTIFY_TO", ""),
        "lookup_file": get("lookup_file", "PICKING_HELPER_LOOKUP_FILE", ""),
        "log_dir": get("log_dir", "PICKING_HELPER_LOG_DIR", "logs"),
        "kiosk": str(get("kiosk", "PICKING_HELPER_KIOSK", "0")).strip().lower() in ("1","true","yes"),
        "require_location": str(get("require_location", "PICKING_HELPER_REQUIRE_LOCATION", "1")).strip().lower() in ("1","true","yes"),
        "require_staging": str(get("require_staging", "PICKING_HELPER_REQUIRE_STAGING", "0")).strip().lower() in ("1","true","yes"),
        "scan_to_add": str(get("scan_to_add", "PICKING_HELPER_SCAN_TO_ADD", "1")).strip().lower() in ("1","true","yes"),
        "keep_staging_after_add": str(get("keep_staging_after_add", "PICKING_HELPER_KEEP_STAGING_AFTER_ADD", "1")).strip().lower() in ("1","true","yes"),
    }
    nt = cfg["notify_to"]
    if isinstance(nt, list):
        notify = nt
    else:
        parts = re.split(r"[;,]", nt) if nt else []
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

# -------------------- ENV / PATHS --------------------
TODAY = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = CFG["log_dir"] or "logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"picking-log-{TODAY}.csv")
WEBHOOK_URL = (CFG["webhook_url"] or "").strip()
TEAMS_WEBHOOK_URL = (CFG["teams_webhook_url"] or "").strip()
NOTIFY_TO: List[str] = CFG["notify_to"]
LOOKUP_FILE_ENV = (CFG["lookup_file"] or "").strip()
KIOSK = CFG["kiosk"]

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
    "qty_picked_str": "",          # text input so it starts blank
    "qty_staged": 0,               # internal numeric mirror
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
    "start_qty_by_pallet": {},     # pallet_id -> int
    "pallet_qty": None,            # KPI value for current pallet
    "clear_qty_next": False,       # safe clear for QTY input
    "clear_top_next": False,       # safe clear top pallet/fields before UI draw
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# -------------------- HELPERS --------------------
def clean_scan(raw: str) -> str:
    return (raw or "").replace("\r", "").replace("\n", "").strip()

def strip_aim_prefix(s: str) -> str:
    # Correctly remove patterns like "]C1", "]A0", etc.
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

# 12-hour timestamp helper
def ts12(dt: Optional[datetime] = None) -> str:
    """Return local 12-hour timestamp like '2025-10-17 09:14:05 AM'."""
    return (dt or datetime.now()).strftime("%Y-%m-%d %I:%M:%S %p")

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
        "pallet":   pick(["pallet","pallet id","pallet_id","lpn","license","serial","sscc"]),
        "sku":      pick(["sku","item","itemcode","product","part","material"]),
        # Default LOT -> CustomerLotReference (plus aliases)
        "lot":      pick(["customerlotreference","customer lot reference","lot","lot_number","lot #","lot#","batch","batchno"]),
        "location": pick(["location","loc","bin","slot","binlocation","location code","staging","stg"]),
        # Default QTY -> QTYOnHand (plus aliases)
        "qty":      pick(["qtyonhand","qty","quantity","cases","casecount","count","units","pallet_qty","onhand","on_hand"]),
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
            if k and v:
                out[k] = v
    return out or None

def try_parse_label_compact(s: str) -> Optional[Dict[str,str]]:
    tokens = re.split(r"[,\\s;]+", s.strip())
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
    pairs = re.findall(r"\((\d{2,4})\)([^\(\)]+)", s)
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
        "pallet":   ["pallet","pallet_id","pallet id","serial","id","license","lpn","sscc"],
        "sku":      ["sku","item","itemcode","product","part","material"],
        "lot":      ["customerlotreference","customer lot reference","lot","lot_number","lot #","lot#","batch","batchno"],
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
    st.info("BUILD: outbound-batch v1.6.5 (12‑hour timestamps • Teams webhook optional • auto-clear)", icon="🧭")
    st.caption(f"Config source: **{CFG['_source']}**")

    st.markdown("### ⚙️ Settings")
    ss["operator"] = st.text_input("Picker Name (required)", value=ss.operator, placeholder="e.g., Carlos")

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
    st.markdown("#### Inventory Lookup (manual)")
    st.caption("Upload your latest RAMP export (CSV/XLS/XLSX). Scans will auto-fill from this file + barcode contents.")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx","xls"], accept_multiple_files=False)

    cols_btn = st.columns([1,1])
    with cols_btn[0]:
        if st.button("↻ Clear lookup cache", use_container_width=True):
            try:
                load_lookup.clear()
            except Exception:
                pass
            st.toast("Lookup cache cleared — re-upload or rely on Secrets path.", icon="🧹")
            st.rerun()
    with cols_btn[1]:
        pass

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

        desired_defaults = {
            "pallet": "PalletID",
            "lot": "CustomerLotReference",
            "sku": "WarehouseSku",
            "location": "LocationName",
            "qty": None,  # handled below
        }
        g = guessed or {"pallet": None, "sku": None, "lot": None, "location": None, "qty": None}
        for key, want in desired_defaults.items():
            if want and want in cols:
                g[key] = want

        qty_candidates = ["QTYOnHand", "QTY On Hand", "OnHandQty", "On Hand", "On_Hand"]
        for qn in qty_candidates:
            if qn in cols:
                g["qty"] = qn
                break

        c1, c2 = st.columns(2)
        with c1:
            ss.lookup_cols["pallet"] = st.selectbox("Pallet column", options=cols, index=cols.index(g.get("pallet")) if g.get("pallet") in cols else 0)
            ss.lookup_cols["sku"]    = st.selectbox("SKU column", options=["(none)"] + cols, index=(cols.index(g.get("sku")) + 1) if g.get("sku") in cols else 0)
        with c2:
            ss.lookup_cols["lot"]      = st.selectbox("LOT column", options=["(none)"] + cols, index=(cols.index(g.get("lot")) + 1) if g.get("lot") in cols else 0)
            ss.lookup_cols["location"] = st.selectbox("Location column", options=["(none)"] + cols, index=(cols.index(g.get("location")) + 1) if g.get("location") in cols else 0)
            ss.lookup_cols["qty"]      = st.selectbox("Pallet QTY column", options=["(none)"] + cols, index=(cols.index(g.get("qty")) + 1) if g.get("qty") in cols else 0)
        for k in ["sku","lot","location","qty"]:
            if ss.lookup_cols.get(k) == "(none)":
                ss.lookup_cols[k] = None
    else:
        st.warning("No Inventory Lookup loaded — auto-fill is limited to what the barcode encodes (e.g., LPN via GS1 21/240, LOT via 10).", icon="ℹ️")

    st.markdown("---")
    st.markdown("#### Behavior")
    CFG["require_location"] = st.toggle("Require Location on Add", value=CFG["require_location"])
    CFG["require_staging"]  = st.toggle("Require Staging on Add",  value=CFG["require_staging"])
    CFG["scan_to_add"]      = st.toggle("Scan-to-Add (after QTY, staging scan auto-adds)", value=CFG["scan_to_add"])
    CFG["keep_staging_after_add"] = st.toggle("Keep Staging Location after Add", value=CFG["keep_staging_after_add"])
    st.markdown("---")
    st.markdown("#### Start/Balance (optional)")
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
            "LOOKUP_FILE": LOOKUP_FILE_ENV or "(empty)",
            "cfg_source": CFG["_source"],
            "cwd": os.getcwd(),
            "files_in_cwd": sorted(os.listdir("."))[:50],
        })

if KIOSK:
    st.markdown("<style>#MainMenu, footer {visibility:hidden;}</style>", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("📦 Picking Helper — Outbound (Batch)")
st.caption("Scan Pallet → KPI shows Pallet QTY → QTY Picked (max 15) → (optional) Staging → Add to Batch → Review & Submit")

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
      if (el) { el.focus(); el.select(); }
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
    st.toast("\n".join(bits), icon="✅")
    ss.recent_scans.insert(0, (datetime.now().strftime("%I:%M:%S %p"), strip_aim_prefix(code)))  # 12-hour
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
    help="Type the quantity picked for this line (max 15)."
)
ss.qty_staged = safe_int(qty_str, 0)

if ss.focus_qty:
    _focus_qty_input()
    ss.focus_qty = False

colA, colC, colD, colE = st.columns([1,1,1,1])
with colA:
    add_to_batch_click = st.button("➕ Add to Batch", use_container_width=True, disabled=(not bool(ss.operator)))
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
    upsert_picked(ss.current_pallet, q)
    line = {
        "order_number": "",  # schema retained: left blank
        "source_location": ss.current_location or "",
        "staging_location": ss.staging_location_current or "",
        "pallet_id": ss.current_pallet,
        "sku": ss.sku or "",
        "lot_number": ss.lot_number or "",
        "qty_staged": int(q),  # keep internal field name for compatibility
        "timestamp": ts12(),   # 12-hour
    }
    ss.batch_rows.append(line)
    ss.undo_stack.append(("add", line))
    st.success(
        f"Added: Pallet {line['pallet_id']} — QTY {line['qty_staged']}"
        + (f" → {line['staging_location']}" if line['staging_location'] else "")
    )
    # Request safe clears on the next run
    ss.clear_qty_next = True     # clears QTY Picked widget
    ss.clear_top_next = True     # clears top pallet fields (and staging if toggle is off)
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
            "timestamp",  # already 12-hour when created
        ]
        df_view = df[view_cols].copy()
        op_col1, op_col2 = st.columns([1,3])
        with op_col1:
            st.metric("Picker Name", ss.operator or "—")
        with op_col2:
            total_lines = len(df_view)
            total_qty = int(df["qty_staged"].sum())
            st.metric("Batch Totals", f"{total_lines} lines / {total_qty} cases")
        st.caption("Edit QTY or fields if needed (0-qty rows are dropped). Then **Apply Edits** → **Submit All**.")
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
            new_rows = []
            for _, r in edited.iterrows():
                qty = safe_int(r.get("qty_staged", 0), 0)
                if qty <= 0:
                    continue
                if qty > 15:
                    st.error(f"Line for pallet {r.get('pallet_id','?')}: qty {qty} > 15 (max). Using 15.")
                    qty = 15
                new_rows.append({
                    "order_number": "",  # keep blank
                    "source_location": str(r.get("source_location","")),
                    "staging_location": str(r.get("staging_location","")),
                    "pallet_id": str(r.get("pallet_id","")),
                    "sku": str(r.get("sku","")),
                    "lot_number": normalize_lot(str(r.get("lot_number",""))),
                    "qty_staged": qty,
                    "timestamp": str(r.get("timestamp","")),  # already 12-hour
                })
            ss.picked_so_far = {}
            for row in new_rows:
                upsert_picked(row.get("pallet_id",""), safe_int(row.get("qty_staged"),0))
            ss.batch_rows = new_rows
            st.success("Edits applied.")
        if submit_all:
            if not ss.operator:
                st.error("Picker Name is required.")
            elif not ss.batch_rows:
                st.error("Batch is empty.")
            else:
                bad = []
                for r in ss.batch_rows:
                    if (CFG["require_location"] and not r.get("source_location")) or not r.get("pallet_id"):
                        bad.append(r)
                    if CFG["require_staging"] and not r.get("staging_location"):
                        bad.append(r)
                    q = safe_int(r.get("qty_staged", 0), 0)
                    if q <= 0 or q > 15:
                        bad.append(r)
                if bad:
                    st.error("Some lines are invalid (missing required fields or qty not in 1–15). Fix and try again.")
                else:
                    batch_id = f"BATCH-{datetime.now().isoformat(timespec='seconds')}-{(ss.operator or 'operator')}".replace(":", "")
                    submitted_at_iso = datetime.now().isoformat(timespec="seconds")  # keep ISO for systems
                    totals_qty = sum(safe_int(r["qty_staged"],0) for r in ss.batch_rows)
                    payload = {
                        "batch_id": batch_id,
                        "submitted_at": submitted_at_iso,
                        "submitted_at_local": ts12(),  # helpful in Flow/cards
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
                    # Also post to Teams Incoming Webhook if configured (non-blocking)
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
                    # Write CSV log rows (12-hour timestamps)
                    for r in ss.batch_rows:
                        pal = r.get("pallet_id","")
                        start_qty = get_start_qty(pal)
                        current_picked = safe_int(ss.picked_so_far.get(pal, 0), 0)
                        remaining_after = clamp_nonneg((start_qty - current_picked) if start_qty is not None else None)
                        append_log_row({
                            "timestamp": ts12(),  # 12-hour
                            "operator": ss.operator or "",
                            "order_number": "",  # keep schema, blank value
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