from sklearn.datasets import make_circles
import pandas as pd
import argparse

n_samples = 1000

X, y = make_circles(
    n_samples,
    noise=0.03,
    random_state=42
)

print(len(X), len(y))
print(X[:5], y[:5])

circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})

print(circles.head(5))

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", action="store_true", help="Show scatter plot")
args = parser.parse_args()

if args.visualize:
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
    plt.show()

print(type(X))