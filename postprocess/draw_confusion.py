import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
f,ax=plt.subplots()


a = [[ 748,    9,   61,    0,    5,    5,  113,    0,   37,    2],
        [   0, 1082,    5,    0,   11,    2,   24,    0,   11,    0],
        [   1,   18,  950,    2,   20,    0,   26,    2,   13,    0],
        [  17,    0,  168,  574,   51,   50,   10,    2,   73,   65],
        [   0,    8,   10,    0,  884,    0,   54,    0,   15,   11],
        [  20,   25,   13,  129,   24,  442,   81,    1,   85,   72],
        [   6,    6,    3,    0,    2,    7,  920,    0,   14,    0],
        [   9,   22,  176,   29,   38,    1,    5,  623,   29,   96],
        [  10,    8,   31,    4,   26,   10,   31,    0,  848,    6],
        [   3,   17,   46,   13,  128,    1,    4,    7,   89,  701]]

sns.heatmap(a,annot=True,ax=ax)
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴

plt.savefig("confusion_matrix.png")