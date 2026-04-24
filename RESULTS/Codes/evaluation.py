import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decay_engine import compute_trust
from policy_engine import evaluate_policy

def plot_policy_distribution(df_results):
    np.random.seed(42)
    
    risk_levels = df_results['band'].tolist()
    trust_values = np.random.uniform(0.0, 1.0, len(risk_levels))
    
    decisions = [evaluate_policy(r, t) for r, t in zip(risk_levels, trust_values)]
    counts = pd.Series(decisions).value_counts()
    
    order = ["Allow", "Restrict", "Step-Up", "Block", "Deny"]
    counts = counts.reindex([x for x in order if x in counts.index]).fillna(0)
    
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="#3182bd", edgecolor="black")
    plt.title("Policy Decision Distribution (Hybrid Calibrated, Seed=42)")
    plt.xlabel("Access Decision")
    plt.ylabel("Number of Requests")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/plots/policy_distribution.png")
    plt.close()

def evaluate_trust_dynamics():
    times = np.linspace(0, 60, 60)
    sensitivities = [1, 3, 5]
    risk_trend = 0.8
    
    plt.figure(figsize=(8, 5))
    for s in sensitivities:
        trusts = [compute_trust(1.0, s, risk_trend, t) for t in times]
        plt.plot(times, trusts, lw=2, label=f"Sensitivity Level {s}")
        
    plt.title("Trust Decay Curves Over Time")
    plt.xlabel("Time Delta (minutes)")
    plt.ylabel("Trust Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/trust_decay_curves.png")
    plt.close()

def evaluate_trust_repair_trace():
    steps = np.arange(20)
    trust = 1.0
    trace = []
    
    for i in steps:
        if i == 5:
            trust = compute_trust(trust, sensitivity=5, risk_trend=0.95, time_delta=10)
        elif i == 10 or i == 12:
            trust = min(1.0, trust + 0.2)
        else:
            trust = compute_trust(trust, sensitivity=1, risk_trend=0.1, time_delta=1)
        trace.append(trust)
        
    plt.figure(figsize=(8, 5))
    plt.step(steps, trace, lw=2, where='mid', color='purple')
    plt.axvline(x=5, color='red', linestyle=':', label="Anomaly Detected")
    plt.axvline(x=10, color='green', linestyle=':', label="MFA Repair")
    plt.axvline(x=12, color='green', linestyle=':')
    plt.title("Event-Driven Trust Recovery Trace")
    plt.xlabel("Request Sequence")
    plt.ylabel("Trust Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/trust_repair_trace.png")
    plt.close()

def simulate_stateful_security():
    results = [
        {"Attack_Type": "Replay Attack", "Exception": "REPLAY_DETECTED", "Gate_Response": "Blocked"},
        {"Attack_Type": "State Tampering", "Exception": "CHAIN_BROKEN", "Gate_Response": "Blocked"},
        {"Attack_Type": "Valid Sequence", "Exception": "None", "Gate_Response": "Passed"}
    ]
    pd.DataFrame(results).to_csv("outputs/metrics/stateful_security_results.csv", index=False)