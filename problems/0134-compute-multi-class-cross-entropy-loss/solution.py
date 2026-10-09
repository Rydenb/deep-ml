import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:

    p = np.clip(predicted_probs, epsilon, 1-epsilon)
    y = np.clip(true_labels, epsilon, 1-epsilon)
    n = len(predicted_probs)
    c = len(predicted_probs[0])
    output = 0
    for i in range(n):
        for j in range(c):
            output+= y[i][j] * np.log(p[i][j])
    return (output/n)*-1