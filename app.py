# app.py — Outbound Picking Helper (Batch Submit)
# v1.9.1 — Picker Name on main • Pallet scan gated by Picker • Sticky Action Bar • Color Status • Floating Summary • Global Batch Cap • Duplicate Merge
# Offline queue + retry • Voice feedback (toggle) • EN/ES i18n • Mobile UI
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

# --- Mobile-friendly CSS (placeholders – customize as needed)
st.markdown("""
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
        "timezone": get("timezone", "PICKING_HELPER_TIMEZONE", "America/Chicago"),
        # NEW: global batch cap + language + voice + offline
        "batch_max_qty": int(str(get("batch_max_qty", "PICKING_HELPER_BATCH_MAX_QTY", "200")).strip() or "200"),
        "language": (get("language", "PICKING_HELPER_LANG", "en") or "en").lower(),
        "voice_feedback": str(get("voice_feedback", "PICKING_HELPER_VOICE", "0")).strip().lower() in ("1","true","yes"),
        "retry_pending_on_start": str(get("retry_pending_on_start", "PICKING_HELPER_RETRY_PENDING", "1")).strip().lower() in ("1","true","yes"),
    }

    nt = cfg["notify_to"]
    if isinstance(nt, list):
        notify = nt
    else:
        import re as _re
        parts = _re.split(r"[ ,;]", nt) if nt else []
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

# -------------------- TIME / PATHS --------------------
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

TODAY = now_local().strftime("%Y-%m-%d")
LOG_DIR = CFG["log_dir"] or "logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"picking-log-{TODAY}.csv")

WEBHOOK_URL = (CFG["webhook_url"] or "").strip()
TEAMS_WEBHOOK_URL = (CFG["teams_webhook_url"] or "").strip()
NOTIFY_TO: List[str] = CFG["notify_to"]
LOOKUP_FILE_ENV = (CFG["lookup_file"] or "").strip()

# Persistent lookup paths (as in v1.8)
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
    ext = ""
    if "." in filename:
        ext = "." + filename.split(".")[-1].lower().strip()
    if ext not in [".csv", ".xlsx", ".xls"]:
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
    if LOOKUP_FILE_ENV and os.path.exists(LOOKUP_FILE_ENV):
        return LOOKUP_FILE_ENV
    meta = _read_lookup_meta()
    p = meta.get("saved_path")
    return p if p and os.path.exists(p) else None

# -------------------- OFFLINE QUEUE (webhook retry) --------------------
PENDING_DIR = os.path.join(LOG_DIR, "pending")
Path(PENDING_DIR).mkdir(parents=True, exist_ok=True)

def _queue_submission(payload: Dict, batch_id: str):
    try:
        with open(os.path.join(PENDING_DIR, f"{batch_id}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _pending_files() -> List[str]:
    try:
        return [p for p in os.listdir(PENDING_DIR) if p.endswith(".json")]
    except Exception:
        return []

def _retry_pending(webhook_url: str) -> Tuple[int,int]:
    """Returns (sent, failed)."""
    sent = failed = 0
    if not webhook_url:
        return (0, len(_pending_files()))
    try:
        import requests
    except Exception:
        return (0, len(_pending_files()))
    for fname in list(_pending_files()):
        fpath = os.path.join(PENDING_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            r = requests.post(webhook_url, json=payload, timeout=12)
            r.raise_for_status()
            os.remove(fpath)
            sent += 1
        except Exception:
            failed += 1
    return sent, failed

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
    # i18n
    "lang": CFG.get("language","en"),
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# -------------------- HELPERS --------------------
def clean_scan(raw: str) -> str:
    return (raw or "").replace("\r", "").replace("\n", "").strip()

def strip_aim_prefix(s: str) -> str:
    # AIM symbology prefixes look like ]Xn, e.g., ]C1, ]e0, ]d2
    m = re.match(r'^\][A-Za-z]\d', s)
    return s[m.end():] if m else s

def safe_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        s2 = re.sub(r"[^\d\-]", "", s)
        return int(s2) if s2 not in ("","-") else default
    except Exception:
        return default

def clamp_nonneg(n: Optional[int]) -> Optional[int]:
    if n is None:
        return None
    return max(int(n), 0)

def normalize_lot(lot: Optional[str]) -> str:
    if not lot:
        return ""
    digits = re.sub(r"\D", "", str(lot))
    return digits.lstrip("0") or "0"

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

# -------------------- LOOKUP (same as v1.8) --------------------
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

# -------------------- SCAN PARSERS (same as v1.8) --------------------
def try_parse_json(s: str) -> Optional[Dict[str,str]]:
    try:
        obj = json.loads(s); return obj if isinstance(obj, dict) else None
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
            k, v = p.split("=", 1); kv[k.strip().lower()] = unquote_plus(v.strip())
        elif ":" in p:
            k, v = p.split(":", 1); kv[k.strip().lower()] = v.strip()
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
        key = tokens[i].lower(); val = tokens[i+1]
        if re.fullmatch(r"[a-zA-Z#]+", key):
            out[key] = val; i += 2
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
        if ai == "21":
            out["pallet"] = val
        elif ai == "10":
            out["lot"] = val
        elif ai == "01":
            out["gtin"] = val
        elif ai == "00":
            out["pallet"] = val
        elif ai in {"240","241"}:
            out["pallet"] = val
    return out or None

def try_parse_gs1_fnc1(s: str) -> Optional[Dict[str,str]]:
    GS = "\x1D"
    if GS not in s and not re.match(r"^\d{2,4}", s):
        return None
    AI_FIXED = {"00":18, "01":14}
    AI_VAR = {"10","21","240","241"}
    i = 0; n = len(s); out: Dict[str,str] = {}
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
            L = AI_FIXED[ai]; val = s[j:j+L] if j+L <= n else s[j:]; i = j+L if j+L <= n else n
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
        if s.startswith("21") and val:
            out["pallet"] = val
        elif s.startswith("10") and val:
            out["lot"] = val
    elif s.startswith(("240","241")):
        val = s[3:]
        if val:
            out["pallet"] = val
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
                out[target] = lower_data[a].strip(); break
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

# -------------------- I18N --------------------
LANGS = {
    "en": {
        "build": "BUILD: outbound-batch v1.9.1 (Picker on main • UI • Guards • Offline • Voice • i18n)",
        "picker_required": "Picker Name (required)",
        "tz_label": "Timezone (IANA)",
        "time_now": "Current app time",
        "inv_lookup": "Inventory Lookup",
        "upload_label": "Upload CSV/XLSX",
        "clear_cache": "↻ Clear lookup cache",
        "remove_saved": "🧹 Remove saved lookup file",
        "saved_lookup": "Saved lookup file + metadata removed.",
        "saved_for_future": "Saved lookup for future sessions",
        "no_lookup": "No Inventory Lookup loaded — auto-fill limited to barcode contents.",
        "behavior": "Behavior",
        "start_balance": "Start/Balance (optional)",
        "start_qty_hint": "If your lookup provides pallet quantity, it overrides this.",
        "kiosk_on": "KIOSK MODE enabled (menu/footer hidden).",
        "pallet_id": "Pallet ID",
        "scan_pallet_here": "Scan pallet here",
        "raw_debug": "🔍 Raw Scan Debugger (last pallet scan)",
        "staging_loc": "Staging Location",
        "scan_staging_here": "Scan staging here",
        "type_staging": "…or type staging",
        "set_staging": "Set Staging",
        "picker_name": "Picker Name",
        "sku": "SKU",
        "lot_number": "LOT Number",
        "source_location": "Source Location",
        "pallet_qty": "Pallet QTY",
        "remaining": "Remaining",
        "recent_scans": "Recent scans",
        "qty_picked": "QTY Picked (1–15)",
        "enter_qty": "Enter QTY Picked",
        "add_batch": "➕ Add to Batch",
        "undo_last": "↩ Undo Last Line",
        "download_log": "⬇️ Download Today’s CSV Log",
        "clear_batch": "🗑️ Clear Batch",
        "review_submit": "Review & Submit",
        "apply_edits": "💾 Apply Edits",
        "submit_all": "✅ Submit All",
        "no_items": "No items in the batch yet. Add lines above.",
        "submit_ok": "Submitted! Notifications sent and history captured locally.",
        "pallet_full": "Pallet fully picked — Remaining 0",
        "added_line": "Added: Pallet {pallet} — QTY {qty}",
        "need_picker": "Enter **Picker Name** before adding.",
        "need_pallet": "Scan a **Pallet ID** before adding.",
        "need_location": "No **Source Location** for this pallet (lookup/scan).",
        "need_staging": "Scan/Type a **Staging Location** before adding.",
        "need_qty": "Enter a **QTY Picked** > 0.",
        "cap15": "QTY Picked capped at 15 cases per line.",
        "exceeds_remaining": "QTY {qty} exceeds Remaining {rem}. Using {rem} instead.",
        "pallet_fully_picked": "Pallet **{pallet}** is fully picked. Remaining is 0 — cannot add more.",
        "batch_totals": "Batch Totals",
        "status_ready": "Ready to Add",
        "status_warn": "Missing fields: {fields}",
        "status_err": "Validation error: {msg}",
        "total_lines_cases": "{lines} lines / {cases} cases",
        "global_cap_hit": "Adding {q} would exceed batch cap {cap}. Using {use} instead.",
        "duplicate_merged": "Duplicate pallet merged: {pallet} → new qty {qty}",
        "duplicate_warn": "Duplicate pallet detected in batch: {pallet}",
        "pending_q": "Pending submissions",
        "retry_now": "🔁 Retry pending now",
        "retry_result": "Retried pending: sent {sent}, failed {failed}.",
        "voice_toggle": "Voice feedback (beta)",
        "language": "Language",
        "english": "English",
        "spanish": "Español",
        "status_bar_ready": "✅ Ready",
        "status_bar_warn": "⚠️ Check",
        "status_bar_err": "⛔ Error",
    },
    "es": {
        "build": "BUILD: v1.9.1 (Nombre del Picker en principal • UI • Reglas • Offline • Voz • i18n)",
        "picker_required": "Nombre del Picker (requerido)",
        "tz_label": "Zona horaria (IANA)",
        "time_now": "Hora actual de la app",
        "inv_lookup": "Búsqueda de Inventario",
        "upload_label": "Subir CSV/XLSX",
        "clear_cache": "↻ Limpiar caché de búsqueda",
        "remove_saved": "🧹 Quitar archivo guardado",
        "saved_lookup": "Archivo de búsqueda y metadatos eliminados.",
        "saved_for_future": "Búsqueda guardada para futuras sesiones",
        "no_lookup": "Sin archivo de búsqueda — auto‑completar limitado al código de barras.",
        "behavior": "Comportamiento",
        "start_balance": "Inicio/Saldo (opcional)",
        "start_qty_hint": "Si el archivo tiene cantidad por tarima, eso tiene prioridad.",
        "kiosk_on": "MODO KIOSKO activado (menú/pie ocultos).",
        "pallet_id": "ID de Tarima",
        "scan_pallet_here": "Escanee la tarima aquí",
        "raw_debug": "🔍 Depurador de escaneo (última tarima)",
        "staging_loc": "Ubicación de Stage",
        "scan_staging_here": "Escanee el stage aquí",
        "type_staging": "…o escriba stage",
        "set_staging": "Fijar Stage",
        "picker_name": "Nombre del Picker",
        "sku": "SKU",
        "lot_number": "Número de Lote",
        "source_location": "Ubicación Origen",
        "pallet_qty": "QTY de Tarima",
        "remaining": "Restante",
        "recent_scans": "Escaneos recientes",
        "qty_picked": "QTY Picked (1–15)",
        "enter_qty": "Ingrese QTY Picked",
        "add_batch": "➕ Agregar al Lote",
        "undo_last": "↩ Deshacer última línea",
        "download_log": "⬇️ Descargar Log de Hoy",
        "clear_batch": "🗑️ Limpiar Lote",
        "review_submit": "Revisar y Enviar",
        "apply_edits": "💾 Aplicar Cambios",
        "submit_all": "✅ Enviar Todo",
        "no_items": "No hay líneas en el lote. Agregue arriba.",
        "submit_ok": "¡Enviado! Notificaciones y registro local guardado.",
        "pallet_full": "Tarima completa — Restante 0",
        "added_line": "Agregado: Tarima {pallet} — QTY {qty}",
        "need_picker": "Ingrese **Nombre del Picker** antes de agregar.",
        "need_pallet": "Escanee un **ID de Tarima** antes de agregar.",
        "need_location": "Sin **Ubicación Origen** para esta tarima (archivo/escaneo).",
        "need_staging": "Escanee/Escriba una **Ubicación de Stage** antes de agregar.",
        "need_qty": "Ingrese una **QTY Picked** > 0.",
        "cap15": "QTY Picked limitada a 15 por línea.",
        "exceeds_remaining": "QTY {qty} excede Restante {rem}. Se usa {rem}.",
        "pallet_fully_picked": "La tarima **{pallet}** está completa. Restante 0 — no se puede agregar.",
        "batch_totals": "Totales del Lote",
        "status_ready": "Listo para Agregar",
        "status_warn": "Campos faltantes: {fields}",
        "status_err": "Error de validación: {msg}",
        "total_lines_cases": "{lines} líneas / {cases} cajas",
        "global_cap_hit": "Agregar {q} excedería el límite {cap}. Se usa {use}.",
        "duplicate_merged": "Tarima duplicada combinada: {pallet} → nueva qty {qty}",
        "duplicate_warn": "Tarima duplicada detectada en el lote: {pallet}",
        "pending_q": "Pendientes por enviar",
        "retry_now": "🔁 Reintentar pendientes",
        "retry_result": "Reintentos: enviados {sent}, fallidos {failed}.",
        "voice_toggle": "Voz (beta)",
        "language": "Idioma",
        "english": "Inglés",
        "spanish": "Español",
        "status_bar_ready": "✅ Listo",
        "status_bar_warn": "⚠️ Revisar",
        "status_bar_err": "⛔ Error",
    }
}

def _t(key: str, **kwargs) -> str:
    lang = ss.get("lang","en")
    base = LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"].get(key, key))
    try:
        return base.format(**kwargs)
    except Exception:
        return base

def _voice(text: str):
    """Lightweight TTS via Web Speech API (if enabled)."""
    if not CFG.get("voice_feedback"):
        return
    # Escape simple quotes and newlines
    txt = (text or "").replace("\\", "\\\\").replace("`","").replace("\n", " ").replace("'", "\\'")
    st.markdown(f"""
<script>
try {{
  const u = new SpeechSynthesisUtterance('{txt}');
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}} catch(e) {{}}
</script>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.info(_t("build"), icon="🧭")
    st.caption(f"Config source: **{CFG['_source']}**")

    # Language selector + voice
    lang_col1, lang_col2 = st.columns([1,1])
    with lang_col1:
        chosen = st.selectbox(
            _t("language"),
            options=[("en", _t("english")), ("es", _t("spanish"))],
            format_func=lambda t: t[1],
            index=0 if ss.lang=="en" else 1
        )
        ss.lang = chosen[0]
    with lang_col2:
        CFG["voice_feedback"] = st.toggle(_t("voice_toggle"), value=CFG.get("voice_feedback", False))

    st.markdown("### ⚙️ Settings")
    # NOTE: Picker input REMOVED from sidebar; now on main panel per v1.9.1
    CFG["timezone"] = st.text_input(_t("tz_label"), value=CFG.get("timezone") or "America/Chicago")
    st.caption(f"{_t('time_now')}: {ts12()} (TZ: {CFG['timezone'] or 'system'})")
    st.write("**Log file:**", f"`{LOG_FILE}`")
    st.caption(f"Recipients (email): {', '.join(NOTIFY_TO)}")
    if WEBHOOK_URL:
        st.caption("Power Automate webhook is configured.")
    else:
        st.warning("Power Automate webhook URL is not set. Submissions write CSV locally only.", icon="ℹ️")

    # Pending queue + retry
    pend = len(_pending_files())
    if pend:
        st.warning(f"{_t('pending_q')}: **{pend}**", icon="📥")
        if st.button(_t("retry_now"), use_container_width=True, disabled=pend==0 or not WEBHOOK_URL):
            sent, failed = _retry_pending(WEBHOOK_URL)
            st.toast(_t("retry_result", sent=sent, failed=failed), icon="🔁")

# Auto retry once per session if enabled via Secrets/Env
if CFG.get("retry_pending_on_start") and WEBHOOK_URL and not ss.retried_on_start:
    sent, failed = _retry_pending(WEBHOOK_URL)
    ss.retried_on_start = True
    if sent or failed:
        st.toast(_t("retry_result", sent=sent, failed=failed), icon="🔁")

    st.markdown("---")
    st.markdown(f"#### #### {_t('inv_lookup')}")
    st.caption("Upload your latest RAMP export (CSV/XLS/XLSX).")
    persisted_path = _get_persisted_lookup_path()
    if LOOKUP_FILE_ENV and os.path.exists(LOOKUP_FILE_ENV):
        current_source_note = f"Using Secrets/Env file: `{LOOKUP_FILE_ENV}`"
    elif persisted_path:
        current_source_note = f"Using saved lookup: `{persisted_path}`"
    else:
        current_source_note = "No saved lookup yet."

    uploaded = st.file_uploader(_t("upload_label"), type=["csv","xlsx","xls"], accept_multiple_files=False)
    cols_btn = st.columns([1,1,1])
    with cols_btn[0]:
        if st.button(_t("clear_cache"), use_container_width=True):
            try:
                load_lookup.clear()
            except Exception:
                pass
            st.toast("Lookup cache cleared.", icon="🧹"); st.rerun()
    with cols_btn[1]:
        if st.button(_t("remove_saved"), use_container_width=True, disabled=not persisted_path):
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
            st.toast(_t("saved_lookup"), icon="🗑️"); st.rerun()
    with cols_btn[2]:
        st.caption(current_source_note or "", help="Secrets/Env overrides saved file")

    up_name = uploaded.name if uploaded is not None else None
    up_bytes = uploaded.read() if uploaded is not None else None
    load_path = LOOKUP_FILE_ENV or (None if up_bytes else persisted_path)
    df, guessed = None, None
    lookup_error = None
    try:
        df, guessed = load_lookup(load_path, up_bytes)
        ss.lookup_df = df
        if up_bytes and up_name and df is not None and len(df) > 0:
            saved_path = _persist_uploaded_lookup(up_name, up_bytes)
            st.toast(f"{_t('saved_for_future')}: {saved_path}", icon="💾")
    except Exception as e:
        lookup_error = str(e); ss.lookup_df = None

    if lookup_error:
        st.error(f"Lookup load error: {lookup_error}")

    saved_meta = _read_lookup_meta()
    saved_colmap = saved_meta.get("colmap") if isinstance(saved_meta.get("colmap"), dict) else {}

    if ss.lookup_df is not None:
        st.success(f"Loaded lookup with {len(ss.lookup_df):,} rows")
        cols = list(ss.lookup_df.columns)
        desired_defaults = {"pallet":"PalletID","lot":"CustomerLotReference","sku":"WarehouseSku","location":"LocationName","qty":None}
        g = guessed or {"pallet": None, "sku": None, "lot": None, "location": None, "qty": None}
        for k, v in (saved_colmap or {}).items():
            if v and v in cols:
                g[k] = v
        for key, want in desired_defaults.items():
            if want and want in cols:
                g[key] = want
        for qn in ["QTYOnHand","QTY On Hand","OnHandQty","On Hand","On_Hand"]:
            if not g.get("qty") and qn in cols:
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

        if st.button("💾 Save mapping as default", use_container_width=True):
            meta = _read_lookup_meta(); meta["colmap"] = ss.lookup_cols; _write_lookup_meta(meta)
            st.toast("Default mapping saved.", icon="✅")
    else:
        st.warning(_t("no_lookup"), icon="ℹ️")

    st.markdown("---")
    st.markdown(f"#### #### {_t('behavior')}")
    CFG["require_location"] = st.toggle("Require Location on Add", value=CFG["require_location"])
    CFG["require_staging"] = st.toggle("Require Staging on Add", value=CFG["require_staging"])
    CFG["scan_to_add"] = st.toggle("Scan-to-Add (after QTY, staging scan auto-adds)", value=CFG["scan_to_add"])
    CFG["keep_staging_after_add"] = st.toggle("Keep Staging Location after Add", value=CFG["keep_staging_after_add"])
    CFG["batch_max_qty"] = st.number_input("Global batch cap (cases)", min_value=1, max_value=5000, step=10, value=int(CFG.get("batch_max_qty",200)))

    st.markdown("---")
    st.markdown(f"#### #### {_t('start_balance')}")
    ss["starting_qty"] = st.number_input(
        "Starting qty on current pallet (fallback if lookup lacks qty)",
        min_value=0, step=1, value=ss.starting_qty or 0, help=_t("start_qty_hint")
    )
    st.caption("Max per line remains 15 cases.")

    if CFG["kiosk"]:
        st.caption(_t("kiosk_on"))

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
            "pending": _pending_files(),
        })

    if CFG["kiosk"]:
        st.markdown("", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.title("📦 Picking Helper — Outbound (Batch)")
st.caption("Scan Pallet → KPI shows Pallet QTY → QTY Picked (max 15, not above Remaining) → (optional) Staging → Add to Batch → Review & Submit")

def _focus_first_text():
    st.markdown("""
""", unsafe_allow_html=True)

def _focus_qty_input():
    st.markdown("""
""", unsafe_allow_html=True)

# -------------------- SAFE PRE-CLEAR BEFORE UI --------------------
if ss.clear_top_next:
    ss.clear_top_next = False
    ss.current_pallet = ""
    ss.sku = ""
    ss.lot_number = ""
    ss.current_location = ""
    ss.pallet_qty = None
    ss.scan = ""
    if not CFG["keep_staging_after_add"]:
        ss.staging_location_current = ""
if ss.clear_qty_next:
    ss.clear_qty_next = False
    st.session_state["qty_picked_str"] = ""
_focus_first_text()

# -------------------- Pallet + Staging Helpers --------------------
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
    if looked.get("location"):
        ss.current_location = looked["location"]
    if looked.get("sku") is not None:
        ss.sku = looked.get("sku") or ""
    if looked.get("lot") is not None:
        ss.lot_number = looked.get("lot") or ""
    if looked.get("pallet_qty") is not None:
        qty_val = clamp_nonneg(safe_int(looked.get("pallet_qty"), 0))
        ss.pallet_qty = qty_val
        ss.start_qty_by_pallet[pallet_id] = qty_val
    elif ss.starting_qty and pallet_id:
        ss.pallet_qty = clamp_nonneg(safe_int(ss.starting_qty, 0))
        ss.start_qty_by_pallet[pallet_id] = ss.pallet_qty

def on_pallet_scan():
    raw = ss.scan or ""; code = clean_scan(raw)
    if not code:
        return
    ss.last_raw_scan = raw
    norm = parse_any_scan(code)
    pallet_id = norm.get("pallet") or strip_aim_prefix(code) or code
    ss.current_pallet = pallet_id
    if norm.get("location"):
        ss.current_location = norm["location"]
    if norm.get("sku") is not None:
        ss.sku = norm.get("sku") or ""
    if norm.get("lot") is not None:
        ss.lot_number = norm.get("lot") or ""
    _apply_lookup_into_state(pallet_id)

    bits = [f"Pallet {ss.current_pallet}"]
    if ss.pallet_qty is not None:
        bits.append(f"QTY {ss.pallet_qty}")
    if ss.current_location:
        bits.append(f"Location {ss.current_location}")
    if ss.sku:
        bits.append(f"SKU {ss.sku}")
    if ss.lot_number:
        bits.append(f"LOT {ss.lot_number}")

    rem_now = get_remaining_for_pallet(ss.current_pallet)
    if rem_now is not None and rem_now <= 0:
        st.toast(_t("pallet_full"), icon="✅"); _voice(_t("pallet_full"))
    else:
        st.toast("\n".join(bits), icon="✅")

    ss.recent_scans.insert(0, (now_local().strftime("%I:%M:%S %p"), strip_aim_prefix(code)))
    ss.recent_scans = ss.recent_scans[:25]
    ss.scan = ""; ss.focus_qty = True

def on_staging_scan():
    raw = ss.staging_scan or ""; code = clean_scan(raw)
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

# -------------------- MAIN PANEL: Picker (FIRST), then Pallet Scan --------------------
st.subheader(_t("picker_required"))
st.text_input(
    _t("picker_required"),
    key="operator",
    value=ss.get("operator", ""),
    placeholder="e.g., Carlos",
    help="Required before scanning pallets. Used on notifications and CSV logs."
)
if not ss.get("operator"):
    st.warning("Enter the Picker Name above to begin.", icon="👆")

with st.expander(_t("raw_debug"), expanded=False):
    if ss.last_raw_scan:
        raw = ss.last_raw_scan
        st.code(repr(raw), language="text")
        hexes = " ".join(f"{ord(c):02X}" for c in raw)
        st.caption(f"Hex bytes: {hexes}")
        if raw.startswith("]"):
            st.info(f"Detected AIM prefix: {raw[:3]!r}", icon="🔖")
        if "\x1D" in raw:
            st.success("Detected ASCII 29 (FNC1) — GS1.", icon="✅")
        else:
            st.info("No FNC1 detected.", icon="ℹ️")
        st.caption(f"After stripping AIM: {strip_aim_prefix(clean_scan(raw))!r}")
    else:
        st.caption("No raw pallet scan captured yet.")

st.subheader(_t("staging_loc"))
cS1, cS2 = st.columns([2, 1])
with cS1:
    st.text_input(_t("scan_staging_here"), key="staging_scan", placeholder="e.g., STAGE-01", on_change=on_staging_scan)
with cS2:
    st.text_input(_t("type_staging"), key="typed_staging", placeholder="e.g., STAGE-01")
st.button(_t("set_staging"), on_click=set_typed_staging, use_container_width=True)

# -------------------- KPI / Metrics --------------------
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(_t("picker_name"), ss.operator or "—")
with c2:
    st.metric(_t("pallet_id"), ss.current_pallet or "—")
with c3:
    st.metric(_t("sku"), ss.sku or "—")
with c4:
    st.metric(_t("lot_number"), ss.lot_number or "—")
with c5:
    st.metric(_t("source_location"), ss.current_location or "—")

# --- Pallet Scan (added in v1.9.2) ------------------------------------------
st.subheader(_t("pallet_id"))
st.text_input(
    _t("scan_pallet_here"),
    key="scan",
    placeholder="Scan pallet barcode here",
    on_change=on_pallet_scan
)

c6, c7 = st.columns(2)
with c6:
    st.metric(_t("pallet_qty"), "—" if ss.pallet_qty is None else safe_int(ss.pallet_qty, 0))
with c7:
    rem = get_remaining_for_pallet(ss.current_pallet) if ss.current_pallet else None
    st.metric(_t("remaining"), "—" if rem is None else rem)

# -------------------- Status chip (color-coded readiness) --------------------
missing = []
if not ss.operator:
    missing.append(_t("picker_name"))
if not ss.current_pallet:
    missing.append(_t("pallet_id"))
if safe_int(ss.qty_picked_str,0) <= 0:
    missing.append(_t("enter_qty"))
if CFG["require_location"] and not ss.current_location:
    missing.append(_t("source_location"))
if CFG["require_staging"] and not ss.staging_location_current:
    missing.append(_t("staging_loc"))

if not missing:
    st.markdown(f"""
{_t('status_bar_ready')} — {_t('status_ready')}
""", unsafe_allow_html=True)
else:
    st.markdown(f"""
{_t('status_bar_warn')} — {_t('status_warn', fields=", ".join(missing))}
""", unsafe_allow_html=True)

# Floating batch summary (always visible)
total_lines = len(ss.batch_rows)
total_qty = sum(safe_int(r.get("qty_staged"),0) for r in ss.batch_rows)
st.markdown(f"""
**{_t('batch_totals')}** — {_t('total_lines_cases', lines=total_lines, cases=total_qty)}
Cap: {int(CFG.get('batch_max_qty',200))} cases
""", unsafe_allow_html=True)
st.markdown("---")

# -------------------- QTY Picked + Batch controls --------------------
st.subheader(_t("qty_picked"))
qty_str = st.text_input(
    _t("enter_qty"),
    key="qty_picked_str",
    placeholder="e.g., 5",
    help="Max 15; not above Remaining or batch cap."
)
ss.qty_staged = safe_int(qty_str, 0)

# Disable add if fully picked
_current_start = get_start_qty(ss.current_pallet)
_current_remaining = get_remaining_for_pallet(ss.current_pallet) if _current_start is not None else None
disable_add = (not bool(ss.operator)) or (
    _current_start is not None and _current_remaining is not None and _current_remaining <= 0
)

if ss.focus_qty:
    _focus_qty_input(); ss.focus_qty = False

# --- Add/Undo/Download/Clear row (top controls)
colA, colC, colD, colE = st.columns([1,1,1,1])
with colA:
    add_to_batch_click = st.button(_t("add_batch"), use_container_width=True, disabled=disable_add, key="add_top")
with colC:
    undo_last = st.button(_t("undo_last"), use_container_width=True, disabled=not ss.batch_rows and not ss.undo_stack, key="undo_top")
with colD:
    st.download_button(
        _t("download_log"),
        data=open(LOG_FILE, "rb").read() if os.path.exists(LOG_FILE) else b"",
        file_name=f"picking-log-{TODAY}.csv",
        mime="text/csv",
        disabled=not os.path.exists(LOG_FILE),
        use_container_width=True,
        key="dl_top"
    )
with colE:
    clear_batch = st.button(_t("clear_batch"), use_container_width=True, disabled=not ss.batch_rows, key="clear_top")

def _apply_global_cap_on_add(q_int: int) -> Tuple[int, Optional[str]]:
    cap = int(CFG.get("batch_max_qty",200))
    current = sum(safe_int(r.get("qty_staged"),0) for r in ss.batch_rows)
    if current + q_int <= cap:
        return q_int, None
    use = max(cap - current, 0)
    return use, _t("global_cap_hit", q=q_int, cap=cap, use=use)

def _merge_duplicate_if_any(line: Dict) -> bool:
    """Merge if same pallet_id AND same staging_location. Returns True if merged."""
    p = line.get("pallet_id","")
    stg = line.get("staging_location","")
    for existing in reversed(ss.batch_rows):
        if existing.get("pallet_id","") == p and existing.get("staging_location","") == stg:
            new_q = safe_int(existing.get("qty_staged"),0) + safe_int(line.get("qty_staged"),0)
            # keep per-line cap 15 as UI rule; if over 15, split across lines (simple: clip and leave remainder)
            if new_q <= 15:
                existing["qty_staged"] = new_q
                existing["timestamp"] = line.get("timestamp", existing.get("timestamp"))
                st.toast(_t("duplicate_merged", pallet=p, qty=new_q), icon="🧩"); _voice(_t("duplicate_merged", pallet=p, qty=new_q))
                return True
            else:
                remainder = new_q - 15
                existing["qty_staged"] = 15
                if remainder > 0:
                    ss.batch_rows.append({**line, "qty_staged": remainder})
                st.toast(_t("duplicate_warn", pallet=p), icon="⚠️")
                return True
    return False

def _add_current_line_to_batch():
    if not ss.operator:
        st.error(_t("need_picker")); _voice(_t("need_picker")); return False
    if not ss.current_pallet:
        st.error(_t("need_pallet")); _voice(_t("need_pallet")); return False
    if CFG["require_location"] and not ss.current_location:
        st.error(_t("need_location")); _voice(_t("need_location")); return False
    if CFG["require_staging"] and not ss.staging_location_current:
        st.error(_t("need_staging")); _voice(_t("need_staging")); return False

    q = safe_int(ss.qty_staged, 0)
    if q <= 0:
        st.error(_t("need_qty")); _voice(_t("need_qty")); return False
    if q > 15:
        st.warning(_t("cap15")); q = 15

    # remaining cap per pallet
    start_qty = get_start_qty(ss.current_pallet)
    if start_qty is not None:
        remaining_before = get_remaining_for_pallet(ss.current_pallet) or 0
        if remaining_before <= 0:
            st.error(_t("pallet_fully_picked", pallet=ss.current_pallet)); _voice(_t("pallet_fully_picked", pallet=ss.current_pallet)); return False
        if q > remaining_before:
            st.warning(_t("exceeds_remaining", qty=q, rem=remaining_before)); q = remaining_before

    # global batch cap
    q, cap_msg = _apply_global_cap_on_add(q)
    if cap_msg is not None:
        st.warning(cap_msg)
    if q <= 0:
        return False

    # Apply
    upsert_picked(ss.current_pallet, q)
    line = {
        "order_number": "",
        "source_location": ss.current_location or "",
        "staging_location": ss.staging_location_current or "",
        "pallet_id": ss.current_pallet,
        "sku": ss.sku or "",
        "lot_number": ss.lot_number or "",
        "qty_staged": int(q),
        "timestamp": ts12(),
    }
    # duplicate merge (same pallet + same staging)
    if not _merge_duplicate_if_any(line):
        ss.batch_rows.append(line)
        ss.undo_stack.append(("add", line))
        st.success(_t("added_line", pallet=line["pallet_id"], qty=line["qty_staged"])); _voice(_t("added_line", pallet=line["pallet_id"], qty=line["qty_staged"]))
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
        st.warning(f"Removed last line for pallet {last.get('pallet_id','?')}.", icon="↩")
    elif ss.undo_stack:
        act, row = ss.undo_stack.pop()
        if act == "remove":
            ss.batch_rows.append(row)
            upsert_picked(row.get("pallet_id",""), safe_int(row.get("qty_staged"),0))
            st.success(f"Restored line for pallet {row.get('pallet_id','?')}.")

# -------------------- REVIEW & SUBMIT --------------------
st.markdown("---")
st.subheader(_t("review_submit"))
if not ss.batch_rows:
    st.info(_t("no_items"))
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
            "picker_name","source_location","staging_location","pallet_id",
            "sku","lot_number","qty_staged","remaining_qty","timestamp",
        ]
        df_view = df[view_cols].copy()

        op_col1, op_col2 = st.columns([1,3])
        with op_col1:
            st.metric(_t("picker_name"), ss.operator or "—")
        with op_col2:
            total_lines = len(df_view)
            total_qty = int(df["qty_staged"].sum())
            st.metric(_t("batch_totals"), _t("total_lines_cases", lines=total_lines, cases=total_qty))

        st.caption("Edit QTY or fields (0-qty rows are dropped). **Capped at Remaining & batch cap**. Then **Apply Edits** → **Submit All**.")
        edited = st.data_editor(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "picker_name": st.column_config.TextColumn(_t("picker_name"), disabled=True),
                "source_location": _t("source_location"),
                "staging_location": _t("staging_loc"),
                "pallet_id": st.column_config.TextColumn(_t("pallet_id"), disabled=True),
                "sku": st.column_config.TextColumn(_t("sku"), disabled=True),
                "lot_number": st.column_config.TextColumn(_t("lot_number"), disabled=True),
                "qty_staged": st.column_config.NumberColumn(_t("enter_qty"), min_value=0, max_value=15),
                "remaining_qty": st.column_config.NumberColumn(_t("remaining"), disabled=True),
                "timestamp": st.column_config.TextColumn("Timestamp", disabled=True),
            },
            num_rows="fixed",
            key="batch_editor"
        )

        colR1, colR2 = st.columns([1,1])
        with colR1:
            apply_edits = st.button(_t("apply_edits"), use_container_width=True, key="apply_top")
        with colR2:
            submit_all = st.button(_t("submit_all"), use_container_width=True, disabled=(not bool(ss.operator)), key="submit_top")

        # Honor bottom-bar trigger too
        submit_all = submit_all or st.session_state.pop("__submit_request", False)

        if apply_edits:
            # Enforce per-pallet remaining caps + batch cap while applying edits
            new_rows: List[Dict] = []
            warn_msgs: List[str] = []
            running_totals: Dict[str, int] = {}
            batch_cap = int(CFG.get("batch_max_qty",200))
            current_sum = 0
            for _, r in edited.iterrows():
                pallet = str(r.get("pallet_id","")).strip()
                if not pallet:
                    continue
                qty = safe_int(r.get("qty_staged", 0), 0)
                if qty <= 0:
                    continue
                if qty > 15:
                    warn_msgs.append("qty > 15 — using 15."); qty = 15
                start_q = get_start_qty(pallet)
                if start_q is not None:
                    used = running_totals.get(pallet, 0)
                    remaining_here = max(start_q - used, 0)
                    if remaining_here <= 0:
                        warn_msgs.append(f"Pallet {pallet}: fully picked — skipping.")
                        continue
                    if qty > remaining_here:
                        warn_msgs.append(f"Pallet {pallet}: qty {qty} > Remaining {remaining_here} — using {remaining_here}.")
                        qty = remaining_here
                    running_totals[pallet] = used + qty

                # batch cap
                if current_sum + qty > batch_cap:
                    allowed = max(batch_cap - current_sum, 0)
                    if allowed <= 0:
                        warn_msgs.append(f"Batch cap {batch_cap} reached — skipping remaining rows.")
                        break
                    warn_msgs.append(f"Global cap: using {allowed} instead of {qty}.")
                    qty = allowed
                current_sum += qty

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

            ss.picked_so_far = {}
            for row in new_rows:
                upsert_picked(row.get("pallet_id",""), safe_int(row.get("qty_staged"),0))
            ss.batch_rows = new_rows
            if warn_msgs:
                st.warning("• " + "\n• ".join(warn_msgs))
            st.success("Edits applied (caps enforced).")

        if submit_all:
            if not ss.operator:
                st.error(_t("need_picker"))
            elif not ss.batch_rows:
                st.error(_t("no_items"))
            else:
                # Validate final against start qty per pallet (safety)
                overfull: List[str] = []
                totals: Dict[str, int] = {}
                for r in ss.batch_rows:
                    p = r.get("pallet_id",""); q = safe_int(r.get("qty_staged", 0), 0)
                    totals[p] = totals.get(p, 0) + q
                for p, s in totals.items():
                    start_q = get_start_qty(p)
                    if start_q is not None and s > start_q:
                        overfull.append(f"{p} (total {s} > start {start_q})")
                if overfull:
                    st.error("Cannot submit: these pallets would exceed starting quantity:\n- " + "\n- ".join(overfull))
                else:
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
                    sent_ok = False; send_error = None
                    if WEBHOOK_URL:
                        try:
                            import requests
                            r = requests.post(WEBHOOK_URL, json=payload, timeout=12)
                            r.raise_for_status()
                            sent_ok = True
                        except Exception as e:
                            send_error = str(e)
                            _queue_submission(payload, batch_id)  # offline queue
                    else:
                        # If no webhook, still store for history
                        _queue_submission(payload, batch_id)

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
                        st.error(f"Submit saved locally; webhook failed and was queued: {send_error}")
                        _voice("Submission queued for retry")
                    else:
                        st.success(_t("submit_ok")); _voice("Submitted")

                    ss.batch_rows = []
                    st.toast(
                        f"Batch {batch_id} — {payload['totals']['lines']} lines / {payload['totals']['qty_staged_sum']} cases.",
                        icon="📨"
                    )
                    time.sleep(0.5); st.rerun()

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
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()[-10:]
            st.code("\n".join(lines))
        except Exception:
            st.info("Unable to preview log file.")
else:
    st.info("No log entries yet today.")

# -------------------- Sticky Bottom Action Bar --------------------
st.markdown("""
""", unsafe_allow_html=True)
bar_c1, bar_c2, bar_c3, bar_c4 = st.columns(4)
with bar_c1:
    add_bottom = st.button(_t("add_batch"), key="add_bottom", use_container_width=True, disabled=disable_add)
with bar_c2:
    undo_bottom = st.button(_t("undo_last"), key="undo_bottom", use_container_width=True, disabled=not ss.batch_rows and not ss.undo_stack)
with bar_c3:
    submit_bottom = st.button(_t("submit_all"), key="submit_bottom", use_container_width=True, disabled=(not bool(ss.operator)))
with bar_c4:
    clear_bottom = st.button(_t("clear_batch"), key="clear_bottom", use_container_width=True, disabled=not ss.batch_rows)

# Wire bottom bar events to same handlers
if add_bottom:
    _add_current_line_to_batch()
if undo_bottom:
    if ss.batch_rows:
        last = ss.batch_rows.pop()
        ss.undo_stack.append(("remove", last))
        upsert_picked(last.get("pallet_id",""), -safe_int(last.get("qty_staged"),0))
        st.warning(f"Removed last line for pallet {last.get('pallet_id','?')}.", icon="↩")
    elif ss.undo_stack:
        act, row = ss.undo_stack.pop()
        if act == "remove":
            ss.batch_rows.append(row)
            upsert_picked(row.get("pallet_id",""), safe_int(row.get("qty_staged"),0))
            st.success(f"Restored line for pallet {row.get('pallet_id','?')}.")

if submit_bottom:
    st.session_state["__submit_request"] = True  # neutral flag, not a widget key
    st.rerun()

if clear_bottom:
    if ss.batch_rows:
        for r in ss.batch_rows:
            upsert_picked(r.get("pallet_id",""), -safe_int(r.get("qty_staged"),0))
        ss.undo_stack.append(("remove_all", ss.batch_rows.copy()))
        ss.batch_rows = []
        st.warning("Batch cleared.")