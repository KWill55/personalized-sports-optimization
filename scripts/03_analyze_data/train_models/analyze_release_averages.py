

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================== 
# Parameters 
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / "data" / ATHLETE / SESSION
player_tracking_metrics_dir = session_dir / "extracted_metrics" / "player_tracking_metrics"

#inputs 
input_path = player_tracking_metrics_dir / "release" / "release_summary.csv"

output_csv_path = session_dir / "analysis" / "average_kinematics_by_outcome.csv"
output_plot_path = session_dir / "analysis" / "average_kinematics_plot.png"

# ======================================== 
# Load and Filter Data 
# ========================================

release_df = pd.read_csv(input_path)
filtered_df = release_df[release_df['outcome'].isin(['made', 'miss'])]

# ======================================== 
# Compute Averages 
# ========================================

numeric_cols = filtered_df.select_dtypes(include='number').columns.difference(['time'])
average_kinematics = filtered_df.groupby('outcome')[numeric_cols].mean()

# Save CSV
average_kinematics.to_csv(output_csv_path, index=True)
print(f"Average kinematics saved to {output_csv_path}")

# ======================================== 
# Plot Bar Graph 
# ========================================

# Transpose so each bar group is a joint angle
ax = average_kinematics.T.plot(
    kind='bar',
    figsize=(14, 6),
    title='Average Joint Angles at Release: Made vs Missed',
    ylabel='Angle (degrees)',
    xlabel='Joint',
    rot=45,
    colormap='Set2'
)

plt.tight_layout()
plt.savefig(output_plot_path)
plt.show()
print(f"Plot saved to {output_plot_path}")
