import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "hour_of_day", "request_rate", "failed_attempts", "geo_risk_score",
    "device_trust_score", "sensitivity_level", "is_vpn", "is_tor"
]

def map_unsw_to_qsrac(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministically maps raw UNSW-NB15 columns to the 8 QSRAC features exactly as trained."""
    mapped = pd.DataFrame(index=df.index)
    
    rate = pd.to_numeric(df.get('rate', 0), errors='coerce').fillna(0)
    dloss = pd.to_numeric(df.get('dloss', 0), errors='coerce').fillna(0)
    spkts = pd.to_numeric(df.get('spkts', 1), errors='coerce').fillna(1)
    sttl = pd.to_numeric(df.get('sttl', 0), errors='coerce').fillna(0)
    dur = pd.to_numeric(df.get('dur', 0), errors='coerce').fillna(0)
    
    proto = df.get('proto', '').astype(str).str.lower()
    service = df.get('service', '').astype(str).str.lower()
    
    is_tcp = proto.isin(['tcp']).astype(float)
    is_sensitive = service.isin(['ssh', 'http', 'https', 'ftp']).astype(float)
    
    mapped["hour_of_day"] = (dur * 1000).astype(int) % 24
    mapped["request_rate"] = np.log1p(rate.clip(lower=0)).clip(0, 10)
    mapped["failed_attempts"] = np.log1p(dloss / (spkts + 1e-6)).clip(0, 10)
    mapped["geo_risk_score"] = (sttl > 64).astype(float) * 0.5
    mapped["device_trust_score"] = (1.0 - np.tanh(mapped["failed_attempts"])).clip(0.1, 1.0)
    mapped["sensitivity_level"] = (is_tcp * 2 + is_sensitive * 2 + 1.0).clip(1, 5)
    
    mapped["is_vpn"] = 0.0
    mapped["is_tor"] = 0.0
    
    mapped.fillna(0, inplace=True)
    
    if 'label' in df.columns:
        y = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int).clip(0, 1)
    else:
        y = pd.Series(np.zeros(len(df), dtype=int))
    
    return mapped[FEATURE_COLUMNS], y

def load_test_data(filepath="UNSW_NB15_testing-set.csv"):
    df = pd.read_csv(filepath)
    X, y = map_unsw_to_qsrac(df)
    
    print(f"[*] Loaded dataset: {len(df)} samples")
    print(f"[*] Class Imbalance (Attack Ratio): {y.mean():.2%}")
    
    return X, y