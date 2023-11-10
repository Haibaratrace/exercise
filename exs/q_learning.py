import numpy as np

r = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]])

q = np.zeros((6, 6))
alpha, gamma = 1, 0.8

for epoch in range(20):
    for s in range(6):
        for a in range(6):
            if r[s][a] >= 0:
                q[s][a] = q[s][a] + alpha * (r[s][a] + gamma * max(q[a]) - q[s][a])
            else:
                continue
print(q / np.max(q) * 100)
