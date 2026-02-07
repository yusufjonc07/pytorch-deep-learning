import matplotlib.pyplot as plt
from dataset import X_train, y_train, X_test, y_test

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Ensure predictions are on CPU and converted to numpy for matplotlib
    if hasattr(predictions, 'detach'):
        predictions = predictions.detach().cpu().numpy()
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
  plt.show()


def plot_loss_curve(epoch_count, train_loss_values, test_loss_values):
    # Debug prints
    print(f"Epoch count: {epoch_count}")
    print(f"Train losses: {train_loss_values}")
    print(f"Test losses: {test_loss_values}")
    print(f"Lengths - epochs: {len(epoch_count)}, train: {len(train_loss_values)}, test: {len(test_loss_values)}")
    
    # Check for issues
    print(f"\nTrain loss - Min: {min(train_loss_values)}, Max: {max(train_loss_values)}")
    print(f"Test loss - Min: {min(test_loss_values)}, Max: {max(test_loss_values)}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_count, train_loss_values, label="Train loss", marker='o')
    plt.plot(epoch_count, test_loss_values, label="Test loss", marker='o')
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("train_test_loss.jpg", dpi=150)
    plt.show()