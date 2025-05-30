import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual file paths
file1 = '/media/rbccps/Kingston/synthetic_data_testing/ideal_robot_path_1697.csv'   # e.g., 'ideal_robot_path.csv'
file2 = '/media/rbccps/Kingston/synthetic_data_testing/diadem_rectangle3_1.csv'  # e.g., 'actual_robot_path.csv'

# Load the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Plot only the X and Y columns
plt.figure(figsize=(10, 6))
plt.plot(df1['X'], df1['Y'], color='blue', label='Expected path')
plt.plot(df2['X'], df2['Y'], color='red', label='Path as per Camera')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Trajectories')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Keeps aspect ratio equal for better shape comparison
plt.show()
