# app.py
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO
from datetime import datetime
import os

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
