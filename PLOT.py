import matplotlib.pyplot as plt

models = ['LSTM', 'CNN+LSTM', 'CNN', 'GRU', 'MoveNet', 'Swin', 'VIT']
val_acc = [0.76, 0.79, 0.74, 0.70, 0.67, 0.55, .60]

plt.bar(models, val_acc, color=['red','orange','green','purple','blue','gray'])
plt.title('Model Validation Accuracy Comparison')
plt.ylabel('Validation Accuracy')
plt.ylim(0,5)
plt.show()
