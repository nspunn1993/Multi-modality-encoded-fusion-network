import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from train import config

PATH_PRED_DIR = "/home/sonali/BDALAB/.sonali/modelmaster/prediction/"
LOG_FILE = config["training_log"]
SCORE_FILE = "brats_scores.csv"
DICEBOX_IMG = "validation_scores_boxplot.png"
LOSS_IMG = "loss_graph.png"

def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def save_animation(pred_image, gt_image, folder_dir):
    fig = plt.figure()
    ims = []
    for i in range(pred_image.shape[-1]):
        plt.subplot(121)
        plt.title('GT')
        plt.axis('off')
        im1 = plt.imshow(gt_image[:,:,i],cmap='gray')
        plt.subplot(122)
        plt.title('P')
        plt.axis('off')
        im2 = plt.imshow(pred_image[:,:,i],cmap='gray')
        ims.append([im1,im2])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=500)
    writer = PillowWriter(fps=20)
    ani.save(folder_dir+"/demo3.gif", writer=writer)
    plt.close()

header = ("WholeTumor", "TumorCore", "EnhancingTumor")
masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
rows = list()
subject_ids = list()

for case_folder in glob.glob(PATH_PRED_DIR+"/*"):
    if not os.path.isdir(case_folder):
        continue
    subject_ids.append(os.path.basename(case_folder))
    truth_file = os.path.join(case_folder, "truth.nii.gz")
    truth_image = nib.load(truth_file)
    truth = truth_image.get_data()
    prediction_file = os.path.join(case_folder, "prediction.nii.gz")
    prediction_image = nib.load(prediction_file)
    prediction = prediction_image.get_data()
    rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
    save_animation(prediction,truth,case_folder)

df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
df.to_csv(PATH_PRED_DIR+SCORE_FILE)

scores = dict()
for index, score in enumerate(df.columns):
    values = df.values.T[index]
    scores[score] = values[np.isnan(values) == False]

plt.boxplot(list(scores.values()), labels=list(scores.keys()))
plt.ylabel("Dice Coefficient")
plt.savefig(PATH_PRED_DIR+DICEBOX_IMG)
plt.close()

if os.path.exists(LOG_FILE):
    training_df = pd.read_csv(LOG_FILE).set_index('epoch')
    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig(PATH_PRED_DIR+LOSS_IMG)