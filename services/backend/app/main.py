"""Deployment entrypoint for the FastAPI application.

This module ensures that the ASGI server (e.g. Uvicorn) can import
``services.backend.app.main:app`` without relying on Streamlit modules or
other optional dependencies.  It simply re-exports the FastAPI instance
from the root ``main.py`` module where all of the routes are defined.
"""

from fastapi import FastAPI

from main import app as ndvi_app

# Re-export the FastAPI application object expected by Uvicorn.
app: FastAPI = ndvi_app

__all__ = ["app"]
