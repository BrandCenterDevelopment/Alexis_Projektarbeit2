import platform
import psutil
import cpuinfo
import pandas as pd
import streamlit as st
import numpy as np
import sklearn  # Import the sklearn module
import scipy  # Import the scipy module
import joblib  # Import the joblib module
import plotly

# System information
print(f"System: {platform.system()}")
print(f"Release: {platform.release()}")
print(f"Version: {platform.version()}")

# Processor information
print(f"Processor: {cpuinfo.get_cpu_info()['brand_raw']}")
print(f"Physical cores: {psutil.cpu_count(logical=False)}")
print(f"Total cores: {psutil.cpu_count(logical=True)}")

# Memory information
import plotly.graph_objects as go

svmem = psutil.virtual_memory()
print(f"Total memory: {svmem.total}")
print(f"Available memory: {svmem.available}")
print(f"Used memory: {svmem.used}")
print(f"Memory percent used: {svmem.percent}")

# Python and library versions
print(f"Python version: {platform.python_version()}")

print(f"Pandas version: {pd.__version__}")
print(f"Streamlit version: {st.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Scipy version: {scipy.__version__}")
print(f"Joblib version: {joblib.__version__}")
print(f"Plotly version: {plotly.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")