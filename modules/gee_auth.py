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