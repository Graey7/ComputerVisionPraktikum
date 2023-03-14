import matplotlib.pyplot as plt
import numpy as np

def make_plots(path, title, train_data, valid_data):
    fig, ax = plt.subplots()
    ax.set_title(title)

    for i in range(len(train_data)):
        ax.plot(train_data[i], color='lightsteelblue', linewidth=0.5)
   
    for i in range(len(valid_data)):
        ax.plot(valid_data[i], color='moccasin', linewidth=0.5)
    
    ax.plot(np.mean(train_data, axis=0), label='train mean', color='#1f77b4', linewidth=1)
    ax.plot(np.mean(valid_data, axis=0), label='valid mean', color='#ff7f0e', linewidth=1)

    ax.set_xlabel('epoch')
    ax.legend()

    plt.savefig(f'{path}{title}.png') #, bbox_inches='tight')
    plt.close()

def make_bars(path, title, data):
    fig, ax = plt.subplots()
    ax.set_title(title)

    for i, d in enumerate(data):
        p = ax.bar(i, round(d, 2), width=0.5, color='#1f77b4')
        ax.bar_label(p, label_type='center', color='white')
    ax.axhline(np.mean(data), color='red', label='mean')
    ax.text(1.02, np.mean(data), str(round(np.mean(data),2)), va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5), transform=ax.get_yaxis_transform(), color='red')
    ax.set_xlabel('Fold')
    ax.legend()
    plt.savefig(f'{path}{title}.png')
    plt.close()