def print_accuracy(m: RunMetrics):
    plt.figure(figsize=(7,4))
    plt.plot(m.acc, label="Train Accuracy")
    plt.plot(m.val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy (Train vs Val)")
    plt.legend()
    plt.show()

print_accuracy(m)
print("Finished")