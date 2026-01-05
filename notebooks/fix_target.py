# Quick fix script - run this cell first to fix target column detection

# Find the actual target column names
target_cols = [c for c in df_v2.columns if 'target' in c.lower() and 'direction' in c.lower()]
print(f"Available target columns: {target_cols}")

# Use GBPUSD_Target_Direction (the actual column name)
target_col = 'GBPUSD_Target_Direction'
if target_col not in df_v2.columns:
    # Fallback: find any GBPUSD target column
    gbp_targets = [c for c in df_v2.columns if 'gbpusd' in c.lower() and 'target' in c.lower()]
    if gbp_targets:
        target_col = gbp_targets[0]
        print(f"Using fallback target: {target_col}")
    else:
        raise ValueError(f"No GBPUSD target column found! Available: {[c for c in df_v2.columns if 'target' in c.lower()]}")

print(f"✅ Using target column: {target_col}")

# === COPY THIS FIXED CODE INTO CELL 8 AND 9 ===
