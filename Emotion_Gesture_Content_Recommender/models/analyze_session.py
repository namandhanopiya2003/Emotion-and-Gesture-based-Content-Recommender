import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data/session_logs.csv', header=None, names=[
    'timestamp', 'emotion', 'emotion_confidence', 'gesture',
    'gesture_score', 'recommended_content', 'vibe_score'
])

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

plt.figure(figsize=(12, 4))
sns.scatterplot(data=df, x='timestamp', y='emotion', hue='emotion', palette='tab10', s=100)
plt.title('User Emotion Over Time')
plt.xlabel('Time')
plt.ylabel('Emotion')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
sns.lineplot(data=df, x='timestamp', y='vibe_score', color='purple', marker='o')
plt.title('Vibe Score Over Time')
plt.xlabel('Time')
plt.ylabel('Vibe Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='recommended_content', hue='emotion', palette='Set2')
plt.title('Recommended Content by Emotion')
plt.xlabel('Count')
plt.ylabel('Content')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='gesture_score', y='vibe_score', hue='emotion', palette='coolwarm', s=100)
plt.title('Gesture Score vs Vibe Score (colored by Emotion)')
plt.xlabel('Gesture Score')
plt.ylabel('Vibe Score')
plt.tight_layout()
plt.show()

print("\nAnalyzing Recommendation Effectiveness...\n")

content_stats = df.groupby('recommended_content')['vibe_score'].agg(['mean', 'std', 'count']).reset_index()
content_stats.columns = ['recommended_content', 'avg_vibe_score', 'score_stddev', 'count']

content_stats['confidence_score'] = content_stats['score_stddev'].apply(lambda x: round(1 / (x + 1e-5), 2))

top_content = content_stats.sort_values(by='avg_vibe_score', ascending=False).head(5)

print("Top 5 Most Effective Recommendations (by Vibe Score):\n")
print(top_content[['recommended_content', 'avg_vibe_score', 'confidence_score', 'count']].to_string(index=False))

df.head(0).to_csv('data/session_logs.csv', index=False, header=False)
