positive_freqs = np.mean(traingenerator.labels, axis = 0)
negative_freqs = 1 - positive_freqs
data = {
    'Class': labels,
    'Positive': positive_freqs,
    'Negative':negative_freqs
}

X_axis = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(X_axis-0.2, data['Positive'], width=0.4, color='b', label = "Positive")
ax.bar(X_axis+0.2, data['Negative'], width=0.4, color='r', label = 'Negative')
plt.xticks(X_axis, labels, rotation = 90)
plt.legend()
plt.figure(figsize=(20,15))
{"mode":"full","isActive":false}

data = {
    'Class': labels,
    'Positive': positive_freqs*negative_freqs,
    'Negative':negative_freqs*positive_freqs
}

X_axis = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(X_axis-0.2, data['Positive'], width=0.4, color='b', label = "Positive")
ax.bar(X_axis+0.2, data['Negative'], width=0.4, color='r', label = 'Negative')
plt.xticks(X_axis, labels, rotation = 90)
plt.legend()
plt.figure(figsize=(20,15))
{"mode":"full","isActive":false}