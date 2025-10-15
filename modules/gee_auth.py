import streamlit as st, tempfile, os, ee

def initialize_ee_from_secrets():
    if "ee" not in st.secrets or "service_account" not in st.secrets["ee"] or "private_key" not in st.secrets["ee"]:
        raise RuntimeError("Missing [ee] secrets.")
    sa_email = st.secrets["ee"]["service_account"]
    key_json = st.secrets["ee"]["private_key"]
    project = st.secrets["ee"].get("project")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(key_json)
        key_path = f.name
    try:
        creds = ee.ServiceAccountCredentials(sa_email, key_path)
        ee.Initialize(credentials=creds, project=project)
    finally:
        os.remove(key_path)

import ee
import streamlit as st

def initialize_ee():
    """Initialize Google Earth Engine with authentication."""
    try:
        ee.Initialize()
    except Exception as e:
        try:
            ee.Authenticate(auth_mode="notebook")
            ee.Initialize()
        except Exception as auth_error:
            st.error(f"Earth Engine authentication failed: {str(auth_error)}")
            raise