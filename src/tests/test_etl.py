import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.classes.ETL2 import ETL 
# Run this test with:
# python -m pytest src/tests/test_etl.py
def test_etl_merge():
    etl = ETL(data_dir="data", start_date="2010-01-01", end_date="2025-04-30")
    data = etl.run(stock_list=["BTC-USD"], indicators=["M2SL"], experiment_name="test")
    print(data.head())
    assert not data.empty
    assert "BTC-USD_Close" in data.columns
    assert "M2SL" in data.columns
    assert not data.index.duplicated().any()

