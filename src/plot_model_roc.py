import pickle
import matplotlib.pyplot as plt
from itertools import cycle

files = ['model_result-resnet50-self.pickle', 'model_result-resnet50x2-self.pickle', 'model_result-resnet200x2-self.pickle']
# files = ['model_result-mobilenet_v2.pickle', 'model_result-resnet34.pickle', 'model_result-resnet50.pickle']

models = ['Resnet50', 'Resnet50x2', 'Resnet200x2']

n_classes = 3

fig, axs = plt.subplots(1, n_classes, figsize=(15, 5))

# 对于每一个类别
for i in range(n_classes):
    # 对于每一个模型
    for file, model in zip(files, models):
        # 使用pickle从文件加载数据
        with open(file, 'rb') as handle:
            loaded_data = pickle.load(handle)

        # 现在loaded_data就是你之前保存的数据
        fpr = loaded_data['fpr']
        tpr = loaded_data['tpr']
        roc_auc = loaded_data['roc_auc']
    
        # 在子图上绘制ROC曲线
        axs[i].plot(fpr[i], tpr[i], 
                     label = '{1} (area = {2:0.3f})'
                     ''.format(i, model, roc_auc[i]))
        # 设置x，y轴的范围
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.05])

        # 设置x，y轴的标签
        axs[i].set_xlabel('False Positive Rate', fontsize = 14)
        axs[i].set_ylabel('True Positive Rate', fontsize = 14)

        # 设置子图的标题
        # axs[i].set_title('Receiver Operating Characteristic for class {}'.format(i))
        axs[0].set_title('(a) ROC for NEG', fontsize = 14)
        axs[1].set_title('(b) ROC for DEL', fontsize = 14)
        axs[2].set_title('(c) ROC for INS', fontsize = 14)

        # 设置子图的图例
        axs[i].legend(loc="lower right")

    # 绘制对角线
    axs[i].plot([0, 1], [0, 1], 'k--')

# 调整子图之间的空间
plt.tight_layout()

# 显示图像
plt.show()

# for file, model in zip(files, models):
#     with open(file, 'rb') as handle:
#         loaded_data = pickle.load(handle)

#     fpr = loaded_data['fpr']
#     tpr = loaded_data['tpr']
#     roc_auc = loaded_data['roc_auc']

#     lw = 2

#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

#     for color in colors:
#             plt.plot(fpr[0], tpr[0], color=color, lw=lw,
#                      label='ROC curve of class {0} form {1} (area = {1:0.3f})'
#                      ''.format(0, model, roc_auc[0]))


# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# # plt.savefig("resnet50.pdf", dpi=1000, bbox_inches='tight')
# plt.show()

