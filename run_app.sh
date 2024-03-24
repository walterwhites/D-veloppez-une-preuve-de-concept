#!/bin/sh

# Run FastAPI application using uvicorn in the background
./custom_uvicorn.sh api.app.api_models:app &

# Wait for a moment to ensure the FastAPI application has started
sleep 5

# Run Streamlit application in the foreground
streamlit run api/app/app.py