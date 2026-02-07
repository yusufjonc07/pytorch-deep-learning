import matplotlib.pyplot as plt
from dataset import X_train, y_train, X_test, y_test

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  Uses the first feature (e.g., area) for x-axis.
  """
  plt.figure(figsize=(10, 7))

  # Move all tensors to CPU and numpy for plotting
  train_data_np = train_data.detach().cpu().numpy() if hasattr(train_data, 'detach') else train_data
  train_labels_np = train_labels.detach().cpu().numpy() if hasattr(train_labels, 'detach') else train_labels
  test_data_np = test_data.detach().cpu().numpy() if hasattr(test_data, 'detach') else test_data
  test_labels_np = test_labels.detach().cpu().numpy() if hasattr(test_labels, 'detach') else test_labels

  # Use the first feature (e.g., area) for x-axis
  train_x = train_data_np[:, 0]
  test_x = test_data_np[:, 0]

  plt.scatter(train_x, train_labels_np, c="b", s=4, label="Training data")
  plt.scatter(test_x, test_labels_np, c="g", s=4, label="Testing data")

  if predictions is not None:
      if hasattr(predictions, 'detach'):
          predictions = predictions.detach().cpu().numpy()
      plt.scatter(test_x, predictions, c="r", s=4, label="Predictions")

  plt.legend(prop={"size": 14})
  plt.xlabel("Area (first feature)")
  plt.ylabel("Price")
  plt.title("House Price Prediction (Area vs Price)")
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