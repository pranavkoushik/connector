# app.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO
from datetime import datetime
import os
import time
import openai
from math import ceil

st.set_page_config(page_title="Leads — Master Viewer", layout="wide")

# ---------------- CONFIG ----------------
SPREADSHEET_ID = "1dT52dtq62Z1XkinCRqgHaJA7lLmIxa3NeijDRc514do"  # change if different
MASTER_SHEET_NAME = "Master"  # name of the canonical sheet created by your Apps Script

# ---------------- AUTH / GSheet CLIENT ----------------
def get_gsheet_client():
    """
    Returns an authorized gspread client using Service Account.
    Supports both local (creds.json) and Streamlit Cloud (secrets) deployment.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    
    # Try Streamlit secrets first (for cloud deployment)
    try:
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
    except (KeyError, FileNotFoundError):
        # Fall back to local creds.json file
        credentials = Credentials.from_service_account_file("creds.json", scopes=scopes)
    
    client = gspread.authorize(credentials)
    return client

# ---------------- LOAD MASTER SHEET ----------------
@st.cache_data(ttl=60)
def load_master_df():
    """
    Loads the Master sheet into a pandas DataFrame using Service Account.
    Caches for 60 seconds by default — change ttl as needed.
    """
    client = get_gsheet_client()
    ss = client.open_by_key(SPREADSHEET_ID)
    
    try:
        sheet = ss.worksheet(MASTER_SHEET_NAME)
    except Exception as e:
        raise RuntimeError(f"Cannot open sheet '{MASTER_SHEET_NAME}'. Error: {e}")
    
    data = sheet.get_all_records(empty2zero=False)
    if not data:
        return pd.DataFrame()  # empty
    
    df = pd.DataFrame(data)

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Convert all columns to string to avoid Arrow serialization issues
    for col in df.columns:
        df[col] = df[col].astype(str)

    # Try to parse timestamps into datetimes
    for col_name in ["Created Time", "Fetch Timestamp", "created_time", "fetch_timestamp", "timestamp"]:
        if col_name in df.columns:
            try:
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
            except Exception:
                # leave as-is if parse fails
                pass

    # Ensure passes_condition is boolean
    if "passes_condition" in df.columns:
        df["passes_condition"] = df["passes_condition"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)
    else:
        # fallback heuristic: has email or phone
        df["passes_condition"] = df.get("Email", df.get("email", "")).notna()

    return df

# ---------------- UI ----------------
st.title("📥 Leads — Master Sheet Viewer")

# Top info row
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("Read-only view of the **Master** sheet. Use filters to narrow leads and download results as CSV.")
with col2:
    if st.button("Refresh data"):
        # clear cache and reload
        st.cache_data.clear()
        st.rerun()

# Load data
try:
    df = load_master_df()
except Exception as e:
    st.error(f"Failed to load Master sheet: {e}")
    st.stop()

if df.empty:
    st.warning("Master sheet is empty or not found. Make sure your Apps Script populated the Master sheet and that the service account has access.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Detect which columns exist
cols = df.columns.tolist()

# Form Name / Form ID filters
form_name_col = None
if "Form Name" in cols:
    form_name_col = "Form Name"
elif "FormName" in cols:
    form_name_col = "FormName"
elif "form_name" in cols:
    form_name_col = "form_name"

form_id_col = None
if "Form ID" in cols:
    form_id_col = "Form ID"
elif "FormId" in cols:
    form_id_col = "FormId"
elif "form_id" in cols:
    form_id_col = "form_id"

status_col = None
for candidate in ["Status", "status", "status "]:
    if candidate in cols:
        status_col = candidate
        break

# Form Name filter
if form_name_col:
    form_options = sorted(df[form_name_col].dropna().unique().tolist())
    selected_forms = st.sidebar.multiselect("Form Name", options=form_options, default=None)
else:
    selected_forms = None

# Status filter
if status_col:
    status_options = sorted(df[status_col].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect("Status", options=status_options, default=None)
else:
    selected_status = None

# passes_condition filter
if "passes_condition" in cols:
    pass_filter = st.sidebar.selectbox("Passes Condition", options=["All", "Passed", "Failed"], index=0)
else:
    pass_filter = "All"

# Date range filter — try Created Time or Fetch Timestamp
date_col = None
for c in ["Created Time", "created_time", "Fetch Timestamp", "Fetch Timestamp", "fetch_timestamp", "timestamp"]:
    if c in cols:
        date_col = c
        break

if date_col:
    min_date = pd.to_datetime(df[date_col], errors="coerce").min()
    max_date = pd.to_datetime(df[date_col], errors="coerce").max()
    if pd.isna(min_date):
        date_from = st.sidebar.date_input("From date")
        date_to = st.sidebar.date_input("To date")
    else:
        date_from = st.sidebar.date_input("From date", value=min_date.date())
        date_to = st.sidebar.date_input("To date", value=max_date.date())
else:
    date_from = None
    date_to = None

# Search box (name/email/phone)
search_text = st.sidebar.text_input("Search (name / email / phone)")

# Choose which columns to display
default_display_cols = ["Lead ID", "Created Time", "Form Name", "Email", "First Name", "Last Name", "Phone Number", "Status", "passes_condition"]
available_display_cols = [c for c in default_display_cols if c in cols] + [c for c in cols if c not in default_display_cols]
display_cols = st.sidebar.multiselect("Columns to display", options=available_display_cols, default=available_display_cols[:min(12, len(available_display_cols))])

# Apply filters
filtered = df.copy()

# Form name filter
if selected_forms and form_name_col:
    filtered = filtered[filtered[form_name_col].isin(selected_forms)]

# Status filter
if selected_status and status_col:
    filtered = filtered[filtered[status_col].isin(selected_status)]

# passes_condition
if "passes_condition" in cols and pass_filter != "All":
    if pass_filter == "Passed":
        filtered = filtered[filtered["passes_condition"] == True]
    else:
        filtered = filtered[filtered["passes_condition"] == False]

# date range
if date_col and date_from and date_to:
    try:
        start_dt = pd.to_datetime(date_from)
        # include whole day until end of date_to
        end_dt = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[pd.to_datetime(filtered[date_col], errors="coerce").between(start_dt, end_dt)]
    except Exception:
        pass

# search text
if search_text and search_text.strip() != "":
    s = search_text.strip().lower()
    mask = pd.Series(False, index=filtered.index)
    for c in ["First Name", "Last Name", "Email", "Phone Number", "Full Data JSON", "FirstName", "email", "phone"]:
        if c in filtered.columns:
            mask = mask | filtered[c].astype(str).str.lower().fillna("").str.contains(s)
    filtered = filtered[mask]

# Show counts
st.markdown(f"**Total leads:** {len(df):,} &nbsp;&nbsp;|&nbsp;&nbsp; **Filtered:** {len(filtered):,}")

# Display table
if display_cols:
    display_df = filtered[display_cols].reset_index(drop=True)
else:
    display_df = filtered.reset_index(drop=True)

st.dataframe(display_df, use_container_width=True, height=600)

# Download filtered CSV
def convert_df_to_csv_bytes(df_in):
    towrite = BytesIO()
    df_in.to_csv(towrite, index=False, encoding="utf-8")
    towrite.seek(0)
    return towrite.read()

csv_bytes = convert_df_to_csv_bytes(filtered)

st.download_button(
    label="📥 Download filtered CSV",
    data=csv_bytes,
    file_name=f"leads_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

# Optional: show JSON of first row when clicked (debug)
with st.expander("Preview first filtered row (raw JSON)"):
    if len(filtered) > 0:
        st.json(filtered.iloc[0].to_dict())
    else:
        st.write("No rows to preview.")

# --- START: AI Query / Sheet Control Panel ---

# load OpenAI key from secrets or env
OPENAI_API_KEY = None
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_BASE_URL = "https://llm-gateway.prod.joveo.com/"

try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
    OPENAI_MODEL = st.secrets["openai"].get("model", "gpt-4o-mini")
    OPENAI_BASE_URL = st.secrets["openai"].get("base_url", "https://llm-gateway.prod.joveo.com/")
except Exception:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-K3pdIROG5kzZ2-S_BtPXAg")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://llm-gateway.prod.joveo.com/")

if OPENAI_API_KEY is None:
    ai_enabled = False
else:
    openai.api_key = OPENAI_API_KEY
    openai.api_base = OPENAI_BASE_URL
    ai_enabled = True

st.markdown("---")
st.header("🤖 AI Query → Sheet Updater")

st.markdown(
    """
Use this panel to run an LLM prompt against each lead and write the result back into a new column in your **Master** sheet.

**Template placeholders:** use `{{First Name}}`, `{{Last Name}}`, `{{Email}}`, `{{Phone Number}}`, `{{Lead ID}}`, `{{Full Data JSON}}`, or any column name in double braces.
"""
)

with st.form("ai_query_form"):
    form_col1, form_col2 = st.columns([3,1])
    with form_col1:
        new_col_name = st.text_input("New column name to create (Sheet)", value="ai_result")
        prompt_template = st.text_area(
            "AI prompt (you can use placeholders like {{Email}} or {{Full Data JSON}})",
            value=(
                "You are a recruiter assistant. Based on the candidate data below, give a one-line "
                "recommendation (Qualified / Maybe / Not Qualified) and a 10-word reason.\n\n"
                "Candidate data:\n{{Full Data JSON}}\n\nOutput format: <Recommendation> — <short reason>"
            ),
            height=160,
        )
        apply_filtered_only = st.checkbox("Run only on currently filtered rows (recommended)", value=True)
        batch_size = st.number_input("Batch size (requests per batch)", min_value=1, max_value=200, value=20)
    with form_col2:
        st.write("AI status:")
        if not ai_enabled:
            st.error("No OpenAI API key configured. Set `st.secrets['openai']['api_key']` or OPENAI_API_KEY env.")
        else:
            st.success(f"Ready — model: {OPENAI_MODEL}")
        run_button = st.form_submit_button("Run AI and update sheet")

def substitute_placeholders(template: str, row: pd.Series):
    """
    Replace {{Field}} with the row value (stringified). Fall back to empty string.
    """
    out = template
    # include full data JSON
    try:
        full_json = row.to_dict()
    except Exception:
        full_json = {}
    out = out.replace("{{Full Data JSON}}", str(full_json))
    # replace each column placeholder
    for col in row.index:
        placeholder = "{{" + col + "}}"
        val = "" if pd.isna(row[col]) else str(row[col])
        out = out.replace(placeholder, val)
    # common aliases
    out = out.replace("{{First Name}}", str(row.get("First Name", "")))
    out = out.replace("{{Last Name}}", str(row.get("Last Name", "")))
    out = out.replace("{{Email}}", str(row.get("Email", "")))
    out = out.replace("{{Phone Number}}", str(row.get("Phone Number", "")))
    out = out.replace("{{Lead ID}}", str(row.get("Lead ID", "")))
    return out

def ensure_sheet_column(sheet, col_name):
    """
    Ensure the worksheet has the header col_name; if not, append column header.
    Returns the 1-based column index of the target column.
    """
    headers = sheet.row_values(1)
    if not headers:
        headers = []
    if col_name in headers:
        col_idx = headers.index(col_name) + 1
    else:
        # append header to first row
        headers.append(col_name)
        sheet.update("1:1", [headers])  # overwrite first row
        col_idx = len(headers)
    return col_idx

def call_llm_for_text(prompt_text, model, max_retries=3):
    """
    Simple wrapper for OpenAI ChatCompletion - adapt if you use another provider.
    """
    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content":prompt_text}],
                temperature=0.0,
                max_tokens=150
            )
            # extract text
            content = resp["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            # exponential backoff
            wait = 2 ** attempt
            time.sleep(wait)
            last_exc = e
    raise last_exc

def colnum_to_letter(n):
    """Convert column number to Excel-style letter (1=A, 2=B, etc.)"""
    string = ""
    while n > 0:
        n, remainder = divmod(n-1, 26)
        string = chr(65 + remainder) + string
    return string

if run_button and ai_enabled:
    # confirm
    if new_col_name.strip() == "":
        st.error("Please provide a column name.")
    else:
        # choose rows to process
        target_df = filtered if apply_filtered_only else df
        if target_df.empty:
            st.warning("No rows selected to process.")
        else:
            st.info(f"Processing {len(target_df)} rows — batching {batch_size} per request...")

            # get gsheet worksheet with write permissions
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            try:
                creds_dict = st.secrets["gcp_service_account"]
                credentials = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
            except (KeyError, FileNotFoundError):
                credentials = Credentials.from_service_account_file("creds.json", scopes=scopes)
            
            client = gspread.authorize(credentials)
            ss = client.open_by_key(SPREADSHEET_ID)
            try:
                ws = ss.worksheet(MASTER_SHEET_NAME)
            except Exception as e:
                st.error(f"Cannot open worksheet: {e}")
                st.stop()

            # ensure column exists and find index
            target_col_idx = ensure_sheet_column(ws, new_col_name)

            # prepare mapping row -> sheet row number
            sheet_headers = ws.row_values(1)
            lead_id_col_name = None
            lead_idx_in_sheet = None
            for candidate in ["Lead ID", "lead_id", "LeadID", "leadId"]:
                if candidate in sheet_headers:
                    lead_id_col_name = candidate
                    lead_idx_in_sheet = sheet_headers.index(candidate)  # 0-based
                    break

            # build list of (sheet_row_number, prompt_for_row)
            rows_to_update = []
            sheet_all_values = ws.get_all_values()
            # find mapping from lead_id to sheet row number (1-based)
            lead_to_rownum = {}
            if lead_id_col_name and lead_idx_in_sheet is not None:
                for r_idx, row_vals in enumerate(sheet_all_values, start=1):
                    if len(row_vals) > lead_idx_in_sheet:
                        lead_to_rownum[row_vals[lead_idx_in_sheet]] = r_idx

            for idx, r in target_df.reset_index(drop=True).iterrows():
                # decide sheet row number
                sheet_row_num = None
                if lead_id_col_name and str(r.get("Lead ID", "")) in lead_to_rownum:
                    sheet_row_num = lead_to_rownum[str(r.get("Lead ID", ""))]
                else:
                    # fallback: assume table order matches sheet and row 1 is headers
                    sheet_row_num = idx + 2

                prompt_text = substitute_placeholders(prompt_template, r)
                rows_to_update.append((sheet_row_num, prompt_text))

            # run in batches, call LLM for each row and write in batches back to sheet
            total = len(rows_to_update)
            progress = st.progress(0)
            results = []
            batch_updates = []
            for i, (sheet_row, ptext) in enumerate(rows_to_update, start=1):
                try:
                    ai_out = call_llm_for_text(ptext, OPENAI_MODEL)
                except Exception as e:
                    ai_out = f"ERROR: {str(e)}"
                results.append((sheet_row, ai_out))

                # create batch update entry for gspread (A1 notation)
                col_letter = colnum_to_letter(target_col_idx)
                cell_addr = f"{col_letter}{sheet_row}"
                batch_updates.append({'range': cell_addr, 'values': [[ai_out]]})

                # write in chunks to avoid huge single updates
                if len(batch_updates) >= batch_size or i == total:
                    try:
                        ws.batch_update(batch_updates)
                    except Exception as e:
                        st.warning(f"Partial write failed: {e}")
                        # try single cell writes as fallback
                        for bu in batch_updates:
                            rng = bu['range']
                            val = bu['values'][0][0]
                            try:
                                ws.update(rng, [[val]])
                            except Exception:
                                pass
                    batch_updates = []

                # polite pacing to avoid rate limits
                time.sleep(0.2)
                progress.progress(i/total)

            st.success(f"Completed {total} rows. Column '{new_col_name}' updated in sheet.")
            # clear cache and rerun so changes reflect
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()
# --- END: AI Query / Sheet Control Panel ---