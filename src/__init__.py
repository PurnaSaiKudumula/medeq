"""MedEQ source package.

Makes ``src`` importable as a proper package so both the FastAPI backend
(``uvicorn src.api:app``) and any internal module imports resolve cleanly.
"""