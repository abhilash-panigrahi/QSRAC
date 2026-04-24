import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from crypto_provider import (
    generate_exchange_keypair, kem_encapsulate, 
    generate_signing_keypair, sign
)


def benchmark_inference(model, X):
    X_sample = X.iloc[:1000]
    latencies = []
    
    for i in range(len(X_sample)):
        single_row = X_sample.iloc[[i]]
        t0 = time.perf_counter()
        _ = model.predict_risk(single_row)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    res = {
        "Mean_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99)
    }
    pd.DataFrame([res]).to_csv("outputs/metrics/inference_latency.csv", index=False)