import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
file_path = "/media/rbccps/Kingston/synthetic_data_testing/diadem_rectangle3_1.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Plot X vs Y
plt.figure(figsize=(10,10))
plt.scatter(df["X"], df["Y"], label="Data points", color="blue", marker="o")  # Scatter plot
plt.plot(df["X"], df["Y"], linestyle="dashed", color="red", alpha=0.5)  # Connect points with a line

# Labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("X vs Y Plot")
plt.legend()
plt.grid(True)
 
# Show plot

plt.show()