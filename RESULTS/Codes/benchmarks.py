import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from crypto_provider import (
    generate_exchange_keypair, kem_encapsulate, 
    generate_signing_keypair, sign, MockCryptoProvider
)

def benchmark_pqc_overhead():
    iterations = 50
    dummy_payload = b"login_response_payload_12345"
    
    _, client_pub = generate_exchange_keypair()
    server_sign_priv, _ = generate_signing_keypair()
    
    pqc_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _, _ = kem_encapsulate(client_pub)
        _ = sign(server_sign_priv, dummy_payload)
        pqc_times.append((time.perf_counter() - t0) * 1000)
        
    mock_provider = MockCryptoProvider()
    _, mock_client_pub = mock_provider.generate_exchange_keypair()
    mock_server_sign_priv, _ = mock_provider.generate_signing_keypair()
    
    mock_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _, _ = mock_provider.kem_encapsulate(mock_client_pub)
        _ = mock_provider.sign(mock_server_sign_priv, dummy_payload)
        mock_times.append((time.perf_counter() - t0) * 1000)
        
    print("\n[*] Cryptographic Handshake Benchmarks:")
    print(f"    MOCK Latency (ms): mean={np.mean(mock_times):.2f}, p95={np.percentile(mock_times, 95):.2f}, p99={np.percentile(mock_times, 99):.2f}")
    print(f"    PQC Latency  (ms): mean={np.mean(pqc_times):.2f}, p95={np.percentile(pqc_times, 95):.2f}, p99={np.percentile(pqc_times, 99):.2f}")

    plt.figure(figsize=(6, 5))
    means = [np.mean(mock_times), np.mean(pqc_times)]
    stds = [np.std(mock_times), np.std(pqc_times)]
    
    plt.bar(["MOCK Mode (SHA256+RNG)", "PQC Mode (Kyber512+Dilithium2)"], 
            means, yerr=stds, capsize=5, color=['#4daf4a', '#e41a1c'])
    plt.title("Login Handshake Latency: Real Mock vs PQC (Measured)")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig("outputs/plots/pqc_overhead_benchmark.png")
    plt.close()

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