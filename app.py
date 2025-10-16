# Picking Helper – Streamlit prototype (with Scan Mode)
# Author: M365 Copilot for Carlos Pacheco
# Date: 2025-10-16
# Purpose: Bridge process to record partial picks and generate partial pallet tags without changing WMS (RAMP)

import os
import io
import re
import uuid
import json
from datetime import datetime
from urllib.parse import parse_qs

import pandas as pd
import streamlit as st

# Optional: PDF tag generation
try:
    import fitz  # pymupdf
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# Optional: QR generation (not required for parsing scans)
try:
    import qrcode
    from PIL import Image
    HAS_QR = True
except Exception:
    HAS_QR = False

APP_NAME = "Picking Helper"
FULL_PALLET_DEFAULT = 15  # default cases on a full pallet

# ----------------------------
# Utilities
# ----------------------------

def normalize_lot_number(lot_raw: str) -> str:
    """Normalize LOT Number to digits only and remove leading zeros."""
    if not lot_raw:
        return ""
    digits = re.sub(r"[^0-9]", "", str(lot_raw))
    digits = digits.lstrip("0")
    return digits if digits else ""

def get_default_log_dir() -> str:
    """
    Pick a writable log directory with robust fallbacks.

    Priority:
      1) Env var PICKING_HELPER_LOG_DIR (or BIN_HELPER_LOG_DIR for cross-compat)
      2) OneDrive - JT Logistics/picking-helper/logs under user home
      3) ./logs under current working directory
    """
    for env_key in ("PICKING_HELPER_LOG_DIR", "BIN_HELPER_LOG_DIR"):
        path = os.environ.get(env_key)
        if path:
            try:
                os.makedirs(path, exist_ok=True)
                testfile = os.path.join(path, ".write_test")
                with open(testfile, "w", encoding="utf-8") as f:
                    f.write("ok")
                os.remove(testfile)
                return path
            except Exception:
                pass
    primary = os.path.join(
        os.path.expanduser("~"),
        "OneDrive - JT Logistics",
        "picking-helper",
        "logs",
    )
    secondary = os.path.join(os.getcwd(), "logs")

    for path in (primary, secondary):
        try:
            os.makedirs(path, exist_ok=True)
            testfile = os.path.join(path, ".write_test")
            with open(testfile, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(testfile)
            return path
        except Exception:
            continue

    return os.getcwd()

def csv_paths(base_dir: str) -> dict:
    return {
        "picks": os.path.join(base_dir, "picks_log.csv"),
        "partials": os.path.join(base_dir, "partial_pallets.csv"),
        "config": os.path.join(base_dir, "config.json"),
    }

def init_storage(paths: dict):
    if not os.path.exists(paths["picks"]):
        pd.DataFrame(
            columns=[
                "timestamp", "picker", "order_id", "pallet_id", "sku",
                "lot_number", "location_from", "location_to", "cases_before",
                "qty_picked", "cases_remaining", "note"
            ]
        ).to_csv(paths["picks"], index=False)

    if not os.path.exists(paths["partials"]):
        pd.DataFrame(
            columns=[
                "tag_id", "created_at", "updated_at", "status", "pallet_id", "sku",
                "lot_number", "remaining_cases", "location_current", "order_id", "picker"
            ]
        ).to_csv(paths["partials"], index=False)

    if not os.path.exists(paths["config"]):
        cfg = {
            "full_pallet_cases": FULL_PALLET_DEFAULT,
            "theme": {"primary": "#1f77b4", "accent": "#d62728"},  # blue/red
        }
        with open(paths["config"], "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")

def safe_write_df(path: str, df: pd.DataFrame):
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def safe_append_csv(path: str, row: dict):
    df = safe_read_csv(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    safe_write_df(path, df)

def next_cases_before_for_pallet(partials_df: pd.DataFrame, pallet_id: str) -> int:
    """Prefill 'cases_before' using the last known remaining for this pallet."""
    if partials_df.empty or not pallet_id:
        return FULL_PALLET_DEFAULT
    active = partials_df[partials_df["pallet_id"].astype(str).str.upper() == str(pallet_id).upper()]
    if active.empty:
        return FULL_PALLET_DEFAULT
    active = active[~active["status"].isin(["Consumed"])]
    if active.empty:
        return FULL_PALLET_DEFAULT
    latest = active.sort_values("updated_at", ascending=False).iloc[0]
    try:
        val = int(latest.get("remaining_cases", FULL_PALLET_DEFAULT))
        return val if val > 0 else FULL_PALLET_DEFAULT
    except Exception:
        return FULL_PALLET_DEFAULT

# ----------------------------
# Scan payload parsing
# ----------------------------

KEY_ALIASES = {
    "PID": ["PID", "PALLET", "PAL", "PALLET_ID", "PALLETID"],
    "SKU": ["SKU", "ITEM", "PART"],
    "LOT": ["LOT", "LOT_NUMBER", "LOTNO", "LN"],
    "QTY": ["QTY", "QUANTITY", "Q"],
    "FROM": ["FROM", "LOC_FROM", "LOCATION_FROM", "BIN_FROM", "LOC", "BIN"],
    "TO": ["TO", "LOC_TO", "LOCATION_TO", "STAGE", "STAGING", "STAGING_LOC"],
    "ORDER": ["ORDER", "SO", "WAVE", "ORDER_ID", "WAVE_ID"],
    "PICKER": ["PICKER", "USER", "EMP", "PICKER_ID"],
    "TAG": ["TAG", "TAG_ID"],
}

def _key_to_canonical(k: str) -> str | None:
    k_up = (k or "").strip().upper()
    for canon, aliases in KEY_ALIASES.items():
        if k_up in aliases:
            return canon
    return None

def parse_scan_payload(payload: str) -> dict:
    """
    Accepts several formats and returns a dict with canonical keys:
      PID, SKU, LOT, QTY, FROM, TO, ORDER, PICKER, TAG
    Supported formats:
      - key:value or key=value separated by | , space
      - querystring: PID=...&QTY=...
      - JSON object: {"PID":"...","QTY":5}
      - Light GS1 hints: (not full GS1) try to detect QTY as (37)=n, LOT as (10)=..., PID as (21)=...
    """
    result = {}
    if not payload or not isinstance(payload, str):
        return result

    s = payload.strip()

    # Try JSON
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            for k, v in obj.items():
                canon = _key_to_canonical(k)
                if canon:
                    result[canon] = str(v).strip()
            # Normalize LOT
            if "LOT" in result:
                result["LOT"] = normalize_lot_number(result["LOT"])
            return result
        except Exception:
            pass

    # Try querystring (allows & or ;)
    if "=" in s and ("&" in s or ";" in s):
        try:
            qs = parse_qs(s.replace(";", "&"), keep_blank_values=True)
            for k, vlist in qs.items():
                canon = _key_to_canonical(k)
                if canon and vlist:
                    result[canon] = str(vlist[0]).strip()
            if "LOT" in result:
                result["LOT"] = normalize_lot_number(result["LOT"])
            return result
        except Exception:
            pass

    # Light GS1-style hints (VERY minimal, for convenience—not full spec)
    # Common AIs (not exhaustive): (10)=lot, (37)=qty, (21)=serial (we'll map to PID if present)
    # Accept formats like "(10)9062716 (37)5 (21)JTL00496"
    if "(" in s and ")" in s:
        try:
            pairs = re.findall(r"\((\d+)\)\s*([^\s]+)", s)
            for ai, val in pairs:
                if ai == "10":   # LOT
                    result["LOT"] = normalize_lot_number(val)
                elif ai == "37": # QTY
                    result["QTY"] = re.sub(r"[^0-9]", "", val)
                elif ai == "21": # Serial → use as PID fallback
                    result["PID"] = val
            if result:
                return result
        except Exception:
            pass

    # Generic delimited "key:value" or "key=value" split by | , or whitespace
    tokens = re.split(r"[|,\s]+", s)
    for tok in tokens:
        if not tok:
            continue
        if ":" in tok:
            k, v = tok.split(":", 1)
        elif "=" in tok:
            k, v = tok.split("=", 1)
        else:
            # Loose tokens: try to catch QTY like "QTY5" or "PIDJTL00496"
            m = re.match(r"([A-Za-z_]+)(.+)", tok)
            if m:
                k, v = m.group(1), m.group(2)
            else:
                continue
        canon = _key_to_canonical(k)
        if canon:
            result[canon] = v.strip()

    # Normalize known fields
    if "LOT" in result:
        result["LOT"] = normalize_lot_number(result["LOT"])
    if "QTY" in result:
        q = re.sub(r"[^0-9]", "", str(result["QTY"]))
        result["QTY"] = q

    return result

# ----------------------------
# Optional QR: encode tag dict to a compact payload (for labels)
# ----------------------------

def encode_qr_payload_from_tag(tag: dict) -> str:
    """
    Compact schema for our labels (what we parse in Scan Mode):
      PID:<pallet>|SKU:<sku>|LOT:<lot>|QTY:<remaining>|FROM:<loc>|ORDER:<order>|TAG:<tag_id>
    """
    parts = []
    if tag.get("pallet_id"): parts.append(f"PID:{tag['pallet_id']}")
    if tag.get("sku"): parts.append(f"SKU:{tag['sku']}")
    if tag.get("lot_number"): parts.append(f"LOT:{tag['lot_number']}")
    if tag.get("remaining_cases") is not None: parts.append(f"QTY:{tag['remaining_cases']}")
    if tag.get("location_current"): parts.append(f"FROM:{tag['location_current']}")
    if tag.get("order_id"): parts.append(f"ORDER:{tag['order_id']}")
    if tag.get("tag_id"): parts.append(f"TAG:{tag['tag_id']}")
    return "|".join(parts)

def make_qr_png_bytes(data: str) -> bytes:
    if not HAS_QR:
        return b""
    qr = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")  # PIL Image
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def insert_png_on_pdf(page, x, y, png_bytes: bytes, width: float = 120):
    """Insert a PNG image at (x,y) with width; height auto from aspect."""
    if not png_bytes:
        return
    # fitz can insert image from a stream
    rect = fitz.Rect(x, y, x + width, y + width)  # square QR area
    page.insert_image(rect, stream=png_bytes)

# ----------------------------
# PDF tag (with optional QR)
# ----------------------------

def make_tag_pdf_bytes(tag: dict) -> bytes:
    """Create a simple PDF tag (A5-ish) and return bytes. Requires pymupdf."""
    if not HAS_PDF:
        return b""
    doc = fitz.open()
    page = doc.new_page(width=595, height=420)  # landscape small sheet
    margin = 36
    primary = (31/255, 119/255, 180/255)  # blue
    accent = (214/255, 39/255, 40/255)    # red

    def draw_label(key, value, y, big=False):
        text = f"{key}: {value}"
        size = 18 if big else 14
        page.insert_text((margin, y), text, fontsize=size, color=(0, 0, 0))

    # Header
    page.draw_rect(fitz.Rect(0, 0, page.rect.width, 40), color=primary, fill=primary)
    page.insert_text((margin, 24), "PARTIAL PALLET TAG", fontsize=18, color=(1, 1, 1))

    y = 70
    draw_label("Tag ID", tag["tag_id"], y, big=True); y += 26
    draw_label("Pallet ID", tag["pallet_id"], y); y += 22
    draw_label("SKU", tag["sku"], y); y += 22
    draw_label("LOT", tag["lot_number"], y); y += 22

    # Big Remaining
    page.insert_text((margin, y+16), "REMAINING CASES", fontsize=12, color=accent)
    page.insert_text((margin+250, y), str(tag["remaining_cases"]), fontsize=72, color=(0,0,0))
    y += 90

    draw_label("Location", tag.get("location_current", ""), y); y += 22
    draw_label("Status", tag.get("status", "Staged"), y); y += 22
    draw_label("Order", tag.get("order_id", ""), y); y += 22
    draw_label("Picker", tag.get("picker", ""), y); y += 22
    draw_label("Generated", tag.get("created_at", datetime.utcnow().isoformat()), y)

    # Optional QR on the right
    try:
        payload = encode_qr_payload_from_tag(tag)
        if payload and HAS_QR:
            png_bytes = make_qr_png_bytes(payload)
            insert_png_on_pdf(page, x=420, y=70, png_bytes=png_bytes, width=140)
            page.insert_text((420, 220), "Scan to autofill", fontsize=10, color=(0,0,0))
    except Exception:
        pass

    bio = io.BytesIO()
    doc.save(bio)
    b = bio.getvalue()
    doc.close()
    return b

# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title=APP_NAME, page_icon="📦", layout="wide")

# Simple CSS styling
st.markdown(
    """
    <style>
    :root {
      --primary: #1f77b4;
      --accent: #d62728;
      --card: #f8fbff;
      --shadow: rgba(0,0,0,0.08);
    }
    .metric-card {
      background: var(--card);
      border: 1px solid #e6effa;
      border-radius: 12px; padding: 14px 16px; box-shadow: 0 6px 18px var(--shadow);
    }
    .muted {color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

LOG_DIR = get_default_log_dir()
PATHS = csv_paths(LOG_DIR)
init_storage(PATHS)

st.sidebar.success(f"Active log directory:\n{LOG_DIR}")

# Load data
picks_df = safe_read_csv(PATHS["picks"]) or pd.DataFrame()
partials_df = safe_read_csv(PATHS["partials"]) or pd.DataFrame()

st.title("📦 Picking Helper (Prototype)")

# KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"<div class='metric-card'><div class='muted'>Total Picks</div><h2>{len(picks_df)}</h2></div>",
        unsafe_allow_html=True,
    )
with col2:
    active_partials = 0 if partials_df.empty else int(
        (partials_df["status"].isin(["Staged", "Returned"])).sum()
    )
    st.markdown(
        f"<div class='metric-card'><div class='muted'>Active Partial Pallets</div><h2>{active_partials}</h2></div>",
        unsafe_allow_html=True,
    )
with col3:
    today = datetime.now().date().isoformat()
    picks_today = 0 if picks_df.empty else (picks_df["timestamp"].astype(str).str[:10] == today).sum()
    st.markdown(
        f"<div class='metric-card'><div class='muted'>Picks Today</div><h2>{picks_today}</h2></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='metric-card'><div class='muted'>Full Pallet Size</div><h2>{FULL_PALLET_DEFAULT}</h2></div>",
        unsafe_allow_html=True,
    )

# Tabs
pick_tab, partials_tab, export_tab, admin_tab = st.tabs(
    ["Pick", "Partial Pallets", "Reports / Export", "Admin"]
)

with pick_tab:
    st.subheader("Scan & Pick")
    st.caption("Use your scan gun like a keyboard. Enter the exact quantity you picked, or use Scan Mode to autofill.")

    # --- Scan Mode (QR/Barcode) ---
    with st.expander("🔍 Scan Mode (QR/Barcode) – paste or scan a single string to autofill", expanded=False):
        scan_payload = st.text_input(
            "Scan payload",
            key="scan_payload",
            placeholder="Example: PID:JTL00496|QTY:5|LOT:9062716|FROM:11400801|TO:STAGE-A|SKU:ABC123|ORDER:SO123|PICKER:CP",
            help="Scan a QR/barcode that encodes key:value pairs (|, space, or commas as separators). JSON and querystring also work."
        )
        parse_now = st.button("Parse Scan")
        if parse_now and scan_payload:
            parsed = parse_scan_payload(scan_payload)
            # Update session state to populate inputs below
            mapping = {
                "PID": "pallet_id",
                "FROM": "loc_from",
                "TO": "loc_to",
                "PICKER": "picker",
                "SKU": "sku",
                "LOT": "lot",
                "ORDER": "order",
            }
            for k_src, k_dest in mapping.items():
                if k_src in parsed:
                    st.session_state[k_dest] = parsed[k_src]
            if "QTY" in parsed:
                try:
                    st.session_state["qty_parsed"] = int(parsed["QTY"])
                except Exception:
                    st.session_state["qty_parsed"] = None
            st.info(f"Parsed fields: {', '.join(parsed.keys()) or 'none'}")

    # Pre-fill cases_before from last known remaining for this pallet
    pallet_input = st.text_input(
        "Pallet ID", key="pallet_id", help="Scan or type Pallet ID (alphanumeric preserved)"
    )
    default_cases_before = (
        next_cases_before_for_pallet(partials_df, pallet_input)
        if pallet_input else FULL_PALLET_DEFAULT
    )

    colA, colB, colC = st.columns(3)
    with colA:
        location_from = st.text_input("Location (From)", key="loc_from", help="Bin/location you are picking from")
    with colB:
        staging_loc = st.text_input("Staging Location (optional)", key="loc_to")
    with colC:
        picker = st.text_input("Picker Name/Initials", key="picker")

    col1, col2, col3 = st.columns(3)
    with col1:
        sku = st.text_input("SKU", key="sku")
    with col2:
        lot_raw = st.text_input("LOT Number", key="lot")
        lot_number = normalize_lot_number(lot_raw)
        if lot_raw and lot_number != lot_raw:
            st.caption(f"Normalized LOT: **{lot_number or '—'}** (digits only)")
    with col3:
        order_id = st.text_input("Order / Wave # (optional)", key="order")

    col4, col5, col6 = st.columns(3)
    with col4:
        cases_before = st.number_input(
            "Cases on Pallet (before pick)", min_value=1, max_value=9999, value=int(default_cases_before)
        )
    with col5:
        # If Scan Mode parsed qty, default to that for convenience
        default_qty = int(st.session_state.get("qty_parsed", 1)) if isinstance(st.session_state.get("qty_parsed"), int) else 1
        qty_picked = st.number_input("Quantity Picked (cases)", min_value=1, max_value=9999, value=default_qty)
    with col6:
        note = st.text_input("Note (optional)")

    # Validation
    errors = []
    if not pallet_input:
        errors.append("Pallet ID is required.")
    if not location_from:
        errors.append("Location (From) is required.")
    if qty_picked > cases_before:
        errors.append(f"You cannot pick {qty_picked} from {cases_before}.")

    for e in errors:
        st.warning(e)

    create_tag = st.checkbox("Generate Partial Pallet Tag if cases remain", value=True)

    if st.button("✅ Record Pick", disabled=bool(errors)):
        ts = datetime.now().isoformat(timespec="seconds")
        cases_remaining = max(int(cases_before) - int(qty_picked), 0)

        # Append to picks log
        safe_append_csv(
            PATHS["picks"],
            {
                "timestamp": ts,
                "picker": picker,
                "order_id": order_id,
                "pallet_id": pallet_input,
                "sku": sku,
                "lot_number": lot_number,
                "location_from": location_from,
                "location_to": staging_loc,
                "cases_before": int(cases_before),
                "qty_picked": int(qty_picked),
                "cases_remaining": int(cases_remaining),
                "note": note,
            },
        )

        new_tag_id = None
        tag_bytes = None
        tag_dict = None

        if create_tag and cases_remaining > 0:
            new_tag_id = str(uuid.uuid4())[:8].upper()
            tag_dict = {
                "tag_id": new_tag_id,
                "created_at": ts,
                "updated_at": ts,
                "status": "Staged" if staging_loc else "Returned",
                "pallet_id": pallet_input,
                "sku": sku,
                "lot_number": lot_number,
                "remaining_cases": int(cases_remaining),
                "location_current": staging_loc or location_from,
                "order_id": order_id,
                "picker": picker,
            }
            safe_append_csv(PATHS["partials"], tag_dict)
            if HAS_PDF:
                tag_bytes = make_tag_pdf_bytes(tag_dict)

        st.success(f"Pick recorded. Remaining on pallet: {cases_remaining} cases.")
        if create_tag and cases_remaining > 0:
            st.info(f"Partial Pallet Tag created: {new_tag_id}")
            if HAS_PDF and tag_bytes:
                st.download_button(
                    "⬇️ Download Tag PDF",
                    data=tag_bytes,
                    file_name=f"partial-tag-{new_tag_id}.pdf",
                    mime="application/pdf",
                )
            else:
                st.code(json.dumps(tag_dict, indent=2), language="json")
        # Clear parsed qty after submit
        st.session_state["qty_parsed"] = None
        st.rerun()

with partials_tab:
    st.subheader("Active & Historical Partial Pallets")

    if partials_df.empty:
        st.info("No partial pallets yet.")
    else:
        # Filters
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            f_sku = st.text_input("Filter SKU contains")
        with fcol2:
            f_lot = st.text_input("Filter LOT contains")
        with fcol3:
            f_pallet = st.text_input("Filter Pallet ID contains")
        with fcol4:
            f_status = st.multiselect(
                "Status",
                options=sorted(partials_df["status"].dropna().unique().tolist()),
                default=["Staged", "Returned"],
            )

        view = partials_df.copy()
        if f_sku:
            view = view[view["sku"].astype(str).str.contains(f_sku, case=False, na=False)]
        if f_lot:
            view = view[view["lot_number"].astype(str).str.contains(f_lot, case=False, na=False)]
        if f_pallet:
            view = view[view["pallet_id"].astype(str).str.contains(f_pallet, case=False, na=False)]
        if f_status:
            view = view[view["status"].isin(f_status)]

        st.dataframe(view.sort_values(["updated_at"], ascending=False), use_container_width=True, height=420)

        st.divider()
        st.markdown("**Update Selected Tag**")
        sel_tag = st.text_input("Enter Tag ID to update (e.g., 3F4A2B1C)")
        if sel_tag:
            match = partials_df[partials_df["tag_id"].astype(str).str.upper() == sel_tag.strip().upper()]
            if match.empty:
                st.warning("No tag found with that ID.")
            else:
                rec = match.iloc[0]
                ucol1, ucol2, ucol3 = st.columns(3)
                with ucol1:
                    new_status = st.selectbox(
                        "Status",
                        options=["Staged", "Returned", "Consumed"],
                        index=["Staged", "Returned", "Consumed"].index(rec["status"])
                        if rec["status"] in ["Staged", "Returned", "Consumed"]
                        else 0,
                    )
                with ucol2:
                    new_loc = st.text_input("Location", value=str(rec["location_current"]))
                with ucol3:
                    new_rem = st.number_input("Remaining Cases", min_value=0, max_value=9999, value=int(rec["remaining_cases"]))
                if st.button("💾 Save Update"):
                    df = partials_df.copy()
                    idx = df.index[df["tag_id"].astype(str).str.upper() == sel_tag.strip().upper()][0]
                    df.at[idx, "status"] = new_status
                    df.at[idx, "location_current"] = new_loc
                    df.at[idx, "remaining_cases"] = int(new_rem)
                    df.at[idx, "updated_at"] = datetime.now().isoformat(timespec="seconds")
                    safe_write_df(PATHS["partials"], df)
                    st.success("Tag updated.")
                    st.rerun()

with export_tab:
    st.subheader("Reports & Export")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Picks Log CSV**")
        if not picks_df.empty:
            data = picks_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download picks_log.csv", data=data, file_name="picks_log.csv", mime="text/csv")
        else:
            st.caption("No picks recorded yet.")
    with c2:
        st.markdown("**Partial Pallets CSV**")
        if not partials_df.empty:
            data = partials_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download partial_pallets.csv", data=data, file_name="partial_pallets.csv", mime="text/csv")
        else:
            st.caption("No partials yet.")

with admin_tab:
    st.subheader("Admin & Settings")
    st.write("You can safely change defaults and view the active storage location.")

    st.markdown(f"**Active Log Directory:** `{LOG_DIR}`")
    st.markdown("**Files**: `picks_log.csv`, `partial_pallets.csv`, `config.json`")

    # Load config
    try:
        with open(PATHS["config"], "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {"full_pallet_cases": FULL_PALLET_DEFAULT}

    new_full = st.number_input(
        "Default full pallet cases",
        min_value=1, max_value=1000,
        value=int(cfg.get("full_pallet_cases", FULL_PALLET_DEFAULT))
    )
    if st.button("Save Settings"):
        cfg["full_pallet_cases"] = int(new_full)
        with open(PATHS["config"], "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        st.success("Settings saved. Reload app to apply.")