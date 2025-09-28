# Trauma Hemorrhage ML (Kawai et al.)

Reproduces the main analyses. Real patient data are **not** included.

## How to run
1. Install requirements: `pip install -r requirements.txt`
2. In Python:
```python
import pandas as pd
df_all = pd.read_csv("/path/to/private.csv")  # <- your private data
exec(open("main_analysis.py", encoding="utf-8").read())
