# Phoenix Rent Growth Forecast Update

**Forecast Date**: 2025-11-10
**Model**: Production-Validated Ensemble (LightGBM + SARIMA + Ridge)

---

## Executive Summary

### 2026-2028 Outlook
- **Average Annual Rent Growth**: **2.04%**
- **Range**: 1.90% to 2.58%

### Annual Breakdown

| Year | Average Rent Growth |
|------|---------------------|
| **2026** | **2.36%** |
| **2027** | **1.92%** |
| **2028** | **1.98%** |
| **2029** | **1.93%** |
| **2030** | **1.98%** |

### Quarterly Forecast Detail

| Quarter | Ensemble Prediction |
|---------|---------------------|
| 2026-Q%q | **2.57%** |
| 2026-Q%q | **2.58%** |
| 2026-Q%q | **2.37%** |
| 2026-Q%q | **1.93%** |
| 2027-Q%q | **1.90%** |
| 2027-Q%q | **1.90%** |
| 2027-Q%q | **1.90%** |
| 2027-Q%q | **1.98%** |
| 2028-Q%q | **1.98%** |
| 2028-Q%q | **1.98%** |
| 2028-Q%q | **1.98%** |
| 2028-Q%q | **1.98%** |
| 2029-Q%q | **1.92%** |
| 2029-Q%q | **1.92%** |
| 2029-Q%q | **1.92%** |
| 2029-Q%q | **1.98%** |
| 2030-Q%q | **1.98%** |
| 2030-Q%q | **1.98%** |
| 2030-Q%q | **1.98%** |
| 2030-Q%q | **1.98%** |

---

## Model Configuration

**Production-Validated Configuration** (from root cause analysis):
- **SARIMA**: Order (1,1,2), Seasonal (0,0,1,4)
- **LightGBM**: Early stopping 50 rounds, best iteration varies
- **Ridge**: Alpha range [0.1, 1.0, 10.0, 100.0, 1000.0]

---

**Auto-generated**: 2025-11-10 22:11:13
