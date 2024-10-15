import pickle
import matplotlib.pyplot as plt
from itertools import cycle

files = ['model_result-resnet50-self.pickle', 'model_result-resnet50x2-self.pickle', 'model_result-resnet200x2-self.pickle']

models = ['Resnet50', 'Resnet50x2', 'Resnet200x2']

n_classes = 3

fig, axs = plt.subplots(1, n_classes, figsize=(15, 5))

for i in range(n_classes):
    for file, model in zip(files, models):
        with open(file, 'rb') as handle:
            loaded_data = pickle.load(handle)

        fpr = loaded_data['fpr']
        tpr = loaded_data['tpr']
        roc_auc = loaded_data['roc_auc']
    
        axs[i].plot(fpr[i], tpr[i], 
                     label = '{1} (area = {2:0.3f})'
                     ''.format(i, model, roc_auc[i]))
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])
        axs[i].set_xlabel('False Positive Rate', fontsize = 18)
        axs[i].set_ylabel('True Positive Rate', fontsize = 18)

        axs[0].set_title('(a) ROC for NEG', fontsize = 18)
        axs[1].set_title('(b) ROC for DEL', fontsize = 18)
        axs[2].set_title('(c) ROC for INS', fontsize = 18)

        axs[i].legend(loc="lower right")
    axs[i].plot([0, 1], [0, 1], 'k--')
plt.tight_layout()

plt.show()
