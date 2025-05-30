import pandas as pd
import numpy as np

# Load CSVs
df_ref = pd.read_csv('/media/rbccps/Kingston/synthetic_data_testing/ideal_robot_path_1697.csv')   # Accurate one
df_est = pd.read_csv('/media/rbccps/Kingston/synthetic_data_testing/diadem_rectangle3_1.csv')   # Estimate to evaluate

# Ensure they are aligned (same length, row-wise)
assert len(df_ref) == len(df_est), "CSV files are not aligned!"

# Compute position error (Euclidean)
pos_error = np.linalg.norm(df_ref[['X', 'Y']].values - df_est[['X', 'Y']].values, axis=1)

# Compute orientation error (absolute difference of RPY in degrees)
# If your angles are in radians, convert them to degrees or keep it consistent
rpy_ref = df_ref[['Roll', 'Pitch', 'Yaw']].values
rpy_est = df_est[['Roll', 'Pitch', 'Yaw']].values
rpy_error = np.abs(rpy_ref - rpy_est)

# Optional: Normalize angles between -180 to 180 if needed
rpy_error = (rpy_error + 180) % 360 - 180

# Aggregate errors
print("Position Error (Euclidean):")
print(f"Mean: {np.mean(pos_error):.4f}, Std: {np.std(pos_error):.4f}")

print("\nOrientation Error (RPY in degrees):")
print("Roll Mean Error:", np.mean(rpy_error[:, 0]))
print("Pitch Mean Error:", np.mean(rpy_error[:, 1]))
print("Yaw Mean Error:", np.mean(rpy_error[:, 2]))
