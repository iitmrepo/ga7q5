import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import io

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------
# Generate synthetic data
# -----------------------------
n_samples = 400
channels = ['Email', 'Phone', 'Chat', 'Social Media']

data = {
    'response_time': np.concatenate([
        np.random.normal(loc=20, scale=5, size=n_samples//4),  # Email: slower
        np.random.normal(loc=10, scale=3, size=n_samples//4),  # Phone: moderate
        np.random.normal(loc=5, scale=2, size=n_samples//4),   # Chat: fast
        np.random.normal(loc=15, scale=4, size=n_samples//4)   # Social Media: mid
    ]),
    'channel': np.repeat(channels, n_samples//4)
}

df = pd.DataFrame(data)

# Ensure no negative response times
df['response_time'] = df['response_time'].clip(lower=1)

# -----------------------------
# Seaborn styling
# -----------------------------
sns.set_style("whitegrid")
sns.set_context("talk")

# -----------------------------
# Create violinplot
# -----------------------------
plt.figure(figsize=(8, 8))  # ensures 512x512 with dpi=64

sns.violinplot(
    data=df,
    x="channel",
    y="response_time",
    palette="Set2",
    inner="quartile"
)

# -----------------------------
# Titles and labels
# -----------------------------
plt.title("Customer Support Response Time Distribution\nby Channel",
          fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Support Channel", fontsize=14, fontweight="semibold")
plt.ylabel("Response Time (minutes)", fontsize=14, fontweight="semibold")

sns.despine()
plt.tight_layout()

# -----------------------------
# Save as 512x512 PNG
# -----------------------------
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=64, bbox_inches='tight')
buf.seek(0)

# Resize to exactly 512x512 pixels
img = Image.open(buf)
img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
img_resized.save("chart.png", "PNG", optimize=True)
buf.close()

print("Chart generated successfully with Seaborn violinplot!")

plt.show()
