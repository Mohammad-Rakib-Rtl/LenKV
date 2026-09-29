import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your new Qasper data
df = pd.read_csv('results_sweep/qasper_Qwen2.5-1.5B-Instruct.csv')

# 2. Extract the Full Memory scores and the 512-budget scores
full_kv = df[df['budget'] == 0].set_index('idx')['score']
compressed = df[df['budget'] == 512].set_index('idx')['score']

# 3. Calculate how much accuracy was lost for each specific request
loss = (full_kv - compressed).dropna()

# 4. Create a clean, professional histogram
plt.figure(figsize=(7, 4))
loss.hist(bins=15, color='#e67e22', edgecolor='black', grid=False)
plt.title("Distribution of Per-Request Score Loss (Qasper @ 512 tokens)")
plt.xlabel("Score Loss (Full KV Score minus Compressed Score)")
plt.ylabel("Number of Requests")

# Add a line to show where "Zero Loss" is
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('results_sweep/fig_loss_distribution.png', dpi=200)
print("Success! Saved fig_loss_distribution.png to your results_sweep folder.")