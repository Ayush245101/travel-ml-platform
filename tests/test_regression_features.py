import pandas as pd
from src.train_regression import build_pipeline

def test_pipeline_builds():
    cat_cols = ["from", "to", "flightType", "agency"]
    num_cols = ["time", "distance", "month", "dayofweek", "is_weekend"]
    pipe = build_pipeline(cat_cols, num_cols)
    assert pipe is not None

def test_pipeline_fit_predict():
    data = pd.DataFrame({
        "from": ["A","A","B"],
        "to": ["C","D","C"],
        "flightType": ["premium","economic","firstClass"],
        "time": [1.0, 1.5, 2.0],
        "distance": [100, 200, 300],
        "agency": ["X","Y","X"],
        "month": [1,2,3],
        "dayofweek": [0,3,6],
        "is_weekend": [0,0,1]
    })
    y = [100.0, 200.0, 300.0]
    pipe = build_pipeline(["from","to","flightType","agency"], ["time","distance","month","dayofweek","is_weekend"])
    pipe.fit(data, y)
    preds = pipe.predict(data)
    assert len(preds) == 3