import pandas as pd
import numpy as np
from datetime import datetime, timedelta

###### Reproducibility ######
np.random.seed(42)

####### Core generation function #########

def generate_synthetic_data(start_date: datetime,
                             num_samples: int,
                             base_fault_prob: float = 0.15) -> pd.DataFrame:
    """
    Generate a synthetic three-phase transmission-line dataset.
    """

    # ── 1. Timestamps ────
    timestamps = [start_date + timedelta(minutes=15 * i)
                  for i in range(num_samples)]

    # ── 2. Nominal electrical values ──────
    V_ph  = 190.5   # kV  — phase-to-neutral RMS voltage (330 kV / √3)
    I_nom = 650.0   # A   — nominal load current per phase

    # ── 3. Independent per-phase noise (2 % std dev) ──────
    eps_v = np.random.normal(0.0, 0.02, (num_samples, 3))  # voltage noise
    eps_i = np.random.normal(0.0, 0.02, (num_samples, 3))  # current noise

    Va = V_ph  * (1.0 + eps_v[:, 0])
    Vb = V_ph  * (1.0 + eps_v[:, 1])
    Vc = V_ph  * (1.0 + eps_v[:, 2])

    Ia = I_nom * (1.0 + eps_i[:, 0])
    Ib = I_nom * (1.0 + eps_i[:, 1])
    Ic = I_nom * (1.0 + eps_i[:, 2])

    # ── 4. Environmental variables ────
    temperature = np.random.uniform(15.0, 45.0, num_samples)
    humidity    = np.random.uniform(30.0, 95.0, num_samples)
    wind_speed  = np.random.uniform( 0.0, 25.0, num_samples)

    # ── 5. Environmental fault-probability modifier ─────
    env_multiplier = np.where(
        (temperature > 35.0) | (humidity > 80.0),
        1.8, 1.0
    )

    # ── 6. Peak-load hour modifier ─────────
    hours = np.array([t.hour for t in timestamps])
    peak_mask       = ((hours >= 6) & (hours < 9)) | ((hours >= 18) & (hours < 21))
    peak_multiplier = np.where(peak_mask, 1.3, 1.0)

    # Combined effective fault probability per sample
    effective_fault_prob = np.clip(
        base_fault_prob * env_multiplier * peak_multiplier,
        0.0, 0.95
    )

    # ── 7. Fault-class proportions ────────
    fault_class_probs = [0.70, 0.20, 0.10]   # LG, LL, LLG
    fault_classes     = [2, 3, 4]

    # ── 8. Fault injection ──────────────
    fault_type = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        if np.random.rand() < effective_fault_prob[i]:
            fault_class = np.random.choice(fault_classes, p=fault_class_probs)
            fault_type[i] = fault_class

            if fault_class == 2:
                # ── LG Fault ─────────────────────────────────────────────────
                # One phase: voltage ↓ 40–80 %, current ↑ 300–700 %
                phase = np.random.randint(0, 3)
                v_factor = np.random.uniform(0.2, 0.6)
                i_factor = np.random.uniform(4.0, 8.0)
                if phase == 0:
                    Va[i] *= v_factor;  Ia[i] *= i_factor
                elif phase == 1:
                    Vb[i] *= v_factor;  Ib[i] *= i_factor
                else:
                    Vc[i] *= v_factor;  Ic[i] *= i_factor

            elif fault_class == 3:
                # ── LL Fault ──────────────────────────────────────────────────
                # Two phases: voltage ↓ 50–70 %, current ↑ 400–600 %
                phases = np.random.choice([0, 1, 2], 2, replace=False)
                for p in phases:
                    v_factor = np.random.uniform(0.3, 0.5)
                    i_factor = np.random.uniform(4.0, 6.0)
                    if p == 0:
                        Va[i] *= v_factor;  Ia[i] *= i_factor
                    elif p == 1:
                        Vb[i] *= v_factor;  Ib[i] *= i_factor
                    else:
                        Vc[i] *= v_factor;  Ic[i] *= i_factor

            elif fault_class == 4:
                # ── LLG Fault ─────────
                # Two phases: voltage ↓ 60–90 %, current ↑ 500–800 %
                phases = np.random.choice([0, 1, 2], 2, replace=False)
                for p in phases:
                    v_factor = np.random.uniform(0.1, 0.4)
                    i_factor = np.random.uniform(5.0, 8.0)
                    if p == 0:
                        Va[i] *= v_factor;  Ia[i] *= i_factor
                    elif p == 1:
                        Vb[i] *= v_factor;  Ib[i] *= i_factor
                    else:
                        Vc[i] *= v_factor;  Ic[i] *= i_factor

    # ── 9. Normal-class assignment ────────────────
    normal_mask = fault_type == 0
    n_normal    = normal_mask.sum()
    normal_labels = np.random.choice([0, 1, 5], size=n_normal, p=[0.15, 0.15, 0.70])
    fault_type[normal_mask] = normal_labels

    class5_mask = fault_type == 5
    n5          = class5_mask.sum()
    eps_v5 = np.random.normal(0.0, 0.003, (n5, 3))   # 0.3 % — very tight balance
    eps_i5 = np.random.normal(0.0, 0.003, (n5, 3))
    Va[class5_mask] = V_ph  * (1.0 + eps_v5[:, 0])
    Vb[class5_mask] = V_ph  * (1.0 + eps_v5[:, 1])
    Vc[class5_mask] = V_ph  * (1.0 + eps_v5[:, 2])
    Ia[class5_mask] = I_nom * (1.0 + eps_i5[:, 0])
    Ib[class5_mask] = I_nom * (1.0 + eps_i5[:, 1])
    Ic[class5_mask] = I_nom * (1.0 + eps_i5[:, 2])

    # ── 10. Sensor / EMI noise ─────────────
    sensor_std_v = np.random.uniform(0.03, 0.05) * V_ph
    sensor_std_i = np.random.uniform(0.03, 0.05) * I_nom

    non5_mask = ~class5_mask
    n_non5    = non5_mask.sum()

    Va[non5_mask] += np.random.normal(0.0, sensor_std_v, n_non5)
    Vb[non5_mask] += np.random.normal(0.0, sensor_std_v, n_non5)
    Vc[non5_mask] += np.random.normal(0.0, sensor_std_v, n_non5)
    Ia[non5_mask] += np.random.normal(0.0, sensor_std_i, n_non5)
    Ib[non5_mask] += np.random.normal(0.0, sensor_std_i, n_non5)
    Ic[non5_mask] += np.random.normal(0.0, sensor_std_i, n_non5)

    # ── 11. Assemble DataFrame ────
    df = pd.DataFrame({
        'Timestamp'  : timestamps,
        'Ia'         : Ia,
        'Ib'         : Ib,
        'Ic'         : Ic,
        'Va'         : Va,
        'Vb'         : Vb,
        'Vc'         : Vc,
        'Temperature': temperature,
        'Humidity'   : humidity,
        'Wind_Speed' : wind_speed,
        'Fault_Type' : fault_type
    })
    return df


# ───────────────────────
#  Sample-count helper
# ────────────────────────

def samples_in_period(start: datetime, end: datetime) -> int:
    """
    Return the exact number of 15-minute intervals between start and end
    (inclusive of start, exclusive of end — i.e. left-closed, right-open).
    """
    delta_minutes = int((end - start).total_seconds() / 60)
    return delta_minutes // 15


#  Dataset generation

# Training period: 01 Jan 2025 00:00 → 31 Mar 2025 23:45  (90 days)
TRAIN_START = datetime(2025, 1, 1, 0, 0, 0)
TRAIN_END   = datetime(2025, 4, 1, 0, 0, 0)
N_TRAIN     = samples_in_period(TRAIN_START, TRAIN_END)  # 8,640

# Prediction period: 01 Apr 2025 00:00 → 30 Jun 2025 23:45  (91 days)
TEST_START  = datetime(2025, 4, 1, 0, 0, 0)
TEST_END    = datetime(2025, 7, 1, 0, 0, 0)
N_TEST      = samples_in_period(TEST_START, TEST_END)    # 8,736

print("=" * 60)
print("Generating FIXED synthetic datasets …")
print(f"  Training samples  : {N_TRAIN:,}  ({TRAIN_START.date()} – "
      f"{(TRAIN_END - timedelta(minutes=15)).date()})")
print(f"  Prediction samples: {N_TEST:,}  ({TEST_START.date()} – "
      f"{(TEST_END  - timedelta(minutes=15)).date()})")
print("=" * 60)

# ── Generate & save training data ────────────────────────────────────────────
train_df = generate_synthetic_data(TRAIN_START, N_TRAIN)
train_df.to_csv('train_dataset.csv', index=False)
print("\n✅  train_dataset.csv saved successfully")

# ── Generate & save prediction data ──────────────────────────────────────────
test_df = generate_synthetic_data(TEST_START, N_TEST)
test_df.to_csv('test_dataset.csv', index=False)
print("✅  test_dataset.csv  saved successfully")

# ── Verification summaries ────────────────────────────────────────────────────
print("\n─── Fault-Type Distribution (Training) ───")
dist_train = train_df['Fault_Type'].value_counts().sort_index()
label_map = {0: 'Class 0 – Normal A',
             1: 'Class 1 – Normal B',
             2: 'Class 2 – LG fault',
             3: 'Class 3 – LL fault',
             4: 'Class 4 – LLG fault',
             5: 'Class 5 – Healthy'}
for cls, count in dist_train.items():
    pct = 100 * count / len(train_df)
    print(f"  {label_map.get(cls, cls):28s}  {count:5,}  ({pct:5.1f} %)")

print("\n─── Fault-Type Distribution (Prediction) ───")
dist_test = test_df['Fault_Type'].value_counts().sort_index()
for cls, count in dist_test.items():
    pct = 100 * count / len(test_df)
    print(f"  {label_map.get(cls, cls):28s}  {count:5,}  ({pct:5.1f} %)")

print("\n─── Electrical Signal Summary (Training, normal samples only) ───")
normal_train = train_df[train_df['Fault_Type'].isin([0, 1, 5])]
for col in ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']:
    s = normal_train[col]
    print(f"  {col}: mean={s.mean():8.3f}  std={s.std():7.3f}  "
          f"min={s.min():8.3f}  max={s.max():8.3f}")

print("\n─── Environmental Range Check (Training) ───")
for col in ['Temperature', 'Humidity', 'Wind_Speed']:
    s = train_df[col]
    print(f"  {col:12s}: min={s.min():6.1f}  max={s.max():6.1f}")

# Confirm LG dominance among fault-only samples
faults_only = train_df[train_df['Fault_Type'].isin([2, 3, 4])]
lg_pct = 100 * (faults_only['Fault_Type'] == 2).sum() / len(faults_only)
print(f"\n  LG fault share among all fault events: {lg_pct:.1f} %  "
      f"(target ≈ 70 %)")

print("\n✅  All datasets generated and verified.\n")