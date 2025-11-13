def print_loss(m: RunMetrics):
    plt.figure(figsize=(7,4))
    plt.plot(m.loss, label="Train Loss")
    plt.plot(m.val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss (Train vs Val)")
    plt.legend()
    plt.show()

print_loss(m)
print("Finished")
