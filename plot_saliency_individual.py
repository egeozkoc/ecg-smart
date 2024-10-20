import torch
import numpy as np
from captum.attr import Saliency
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
import pandas as pd
from matplotlib.cm import ScalarMappable

def get_density_plot(y_pred, y):
    # plot a bar graph of the predictions for the two classes

    y_pred_0 = y_pred[y == 0]
    y_pred_1 = y_pred[y == 1]

    plt.hist([y_pred_0, y_pred_1], bins=50, label=['No OMI', 'OMI'], density=True)
    plt.legend()
    plt.show()

def get_prediction_probs(model, test_ids, test_omi):
    probs = []
    for id in test_ids:
        ecg = np.load(id, allow_pickle=True).item()['ecg']
        ecg = ecg[:, 150:-50] / 1000

        max_val = np.max(np.abs(ecg), axis=-1)
        ecg = ecg.T
        ecg /= max_val
        ecg = ecg.T

        ecg[3, :] *= -1

        if num_leads == 8:
            ecg = ecg[[0, 1, 6, 7, 8, 9, 10, 11], :]

        ecg = torch.Tensor(ecg)
        ecg = ecg.unsqueeze(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ecg = ecg.to(device)
        # ecg.requires_grad_()

        predicted = model(ecg)
        predicted = torch.softmax(predicted, dim=-1).squeeze().cpu().detach().numpy()
        probs.append(predicted[1])
    probs = np.array(probs)

    get_density_plot(probs, test_omi)

    idx = np.argsort(probs)
    probs = probs[idx]
    test_ids = test_ids[idx]
    test_omi = test_omi[idx]

    # get top 3 and bottom 3 predictions
    # top_idx = np.where(test_omi == 1)[0][0:20]
    bottom_idx = np.where(test_omi == 0)[0][0:5]

    top_idx = np.where((test_omi == 1) & (probs > 0.38))[0]



    top_ids = test_ids[top_idx]
    bottom_ids = test_ids[bottom_idx]

    return top_ids, bottom_ids

resolution = 1

def on_resize(event):
    fig = event.canvas.figure
    fig.set_size_inches(9* resolution, 8 * resolution)  # Set the desired fixed size

num_leads = 12
target_class = 1

if num_leads == 12:
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
elif num_leads == 8:
    leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

model = torch.load('results/model_12lead_2d.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
saliency = Saliency(model)

fold = np.load('omi_folds.npy', allow_pickle=True).item()

test_ids = fold['test_ids']
test_omi = fold['test_omi']

# test_ids = test_ids[test_omi == target_class]
# test_omi = test_omi[test_omi == target_class]

fs = 500

top_ids, bottom_ids = get_prediction_probs(model, test_ids, test_omi)

if target_class == 1:
    chosen_ids = top_ids
else:
    chosen_ids = bottom_ids

for id in chosen_ids:
    print(id)
    ecg = np.load(id, allow_pickle=True).item()['ecg']
    ecg = ecg[:, 150:-50] / 1000

    ecg1 = ecg.copy()

    max_val = np.max(np.abs(ecg), axis=-1)
    ecg = ecg.T
    ecg /= max_val
    ecg = ecg.T

    ecg[3,:] *= -1

    if num_leads == 8:
        ecg = ecg[[0,1,6,7,8,9,10,11], :]
        ecg1 = ecg1[[0,1,6,7,8,9,10,11], :]

    fiducial_id = id.replace('median_beats', 'fiducials')
    fiducials = np.load(fiducial_id)
    fiducials -= 150
    fiducials[fiducials < 0] = 0
    ecg_rms = np.sqrt(np.mean(ecg**2, axis=0))
    t_peak = np.argmax(ecg_rms[fiducials[4]:fiducials[5]]) + fiducials[4]

    ecg = torch.Tensor(ecg)
    ecg = ecg.unsqueeze(0)
    ecg = ecg.to(device)
    ecg.requires_grad_()

    predicted = model(ecg)
    predicted = torch.softmax(predicted, dim=-1).squeeze().cpu().detach().numpy()
    predicted = predicted[1]
    print(predicted)

    saliency_map = saliency.attribute(ecg, target=target_class)
    saliency_map = saliency_map.squeeze().cpu().detach().numpy()

    contributions = np.sum(saliency_map, axis=1) / np.sum(saliency_map) * 100
    print(contributions)
    
    for i in range(0,400,20):
        for j in range(num_leads):
            saliency_map[j, i:i+20] = np.mean(saliency_map[j, i:i+20], axis=-1)

    saliency_min = np.min(saliency_map)
    saliency_max = np.max(saliency_map)
    saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)

    # saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    # cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])

    color_points = [
        (0.0, (1,0.9,0)),
        (0.275, (1,.45,0)),
        (0.55, (1,0,0)),
        (1.0, (0.25,0,0))
    ]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_points)
    color0 = (1,0.9,0)

    start = fiducials[0]
    end = fiducials[5]
    
    saliency_colors = cmap(saliency_map)

    ecg = ecg.squeeze().cpu().detach().numpy()


    fig, ax = plt.subplots(figsize=(9*resolution, 8*resolution))
    fig.canvas.mpl_connect('resize_event', on_resize)
    offset = 0
    x = np.arange(0,400,1)
    for i in range(0,12,3):
        if i < num_leads:
            colors_lead = saliency_colors[i, :, :3]  # Extract the RGB colors for the lead
            for j in range(400 - 1):
                if j <= start or j >= end:
                    ax.plot(x[j:j+2] + offset, ecg1[i, j:j+2], color=color0, linewidth=2*resolution)
                else:
                    ax.plot(x[j:j+2] + offset, ecg1[i, j:j+2], color=colors_lead[j:j+1], linewidth=2*resolution)
            ax.text(x=10 + offset, y=1, s=leads[i] + '({0:.2f}%)'.format(contributions[i]), fontsize=20*resolution, weight='bold')

        if i+1 < num_leads:
            colors_lead = saliency_colors[i+1, :, :3]
            for j in range(400 - 1):
                if j <= start or j >= end:
                    ax.plot(x[j:j+2] + offset, ecg1[i+1, j:j+2] - 3, color=color0, linewidth=2*resolution)
                else:
                    ax.plot(x[j:j+2] + offset, ecg1[i+1, j:j+2] - 3, color=colors_lead[j:j+1], linewidth=2*resolution)
            ax.text(x=10 + offset, y=1-3, s=leads[i+1] + '({0:.2f}%)'.format(contributions[i+1]), fontsize=20*resolution, weight='bold')

        if i+2 < num_leads:
            colors_lead = saliency_colors[i+2, :, :3]
            for j in range(400 - 1):
                if j <= start or j >= end:
                    ax.plot(x[j:j+2] + offset, ecg1[i+2, j:j+2] - 6, color=color0, linewidth=2*resolution)
                else: 
                    ax.plot(x[j:j+2] + offset, ecg1[i+2, j:j+2] - 6, color=colors_lead[j:j+1], linewidth=2*resolution)
            ax.text(x=10 + offset, y=1-6, s=leads[i+2] + '({0:.2f}%)'.format(contributions[i+2]), fontsize=20*resolution, weight='bold')

        offset += 400

    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5*resolution, color=(0.1,0.1,0.1), alpha=1)
    ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5*resolution, color=(0.1,0.1,0.1), alpha=0.2)

    minor_locator = MultipleLocator(0.04*fs)
    major_locator = MultipleLocator(0.2*fs)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_locator(major_locator)

    minor_locator = MultipleLocator(0.1)
    major_locator = MultipleLocator(0.5)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_major_locator(major_locator)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_facecolor((0.85,0.85,0.85))

    plt.axvline(x=0.8*fs, color='k', linestyle='-', linewidth=2*resolution)
    plt.axvline(x=0.8*fs*2, color='k', linestyle='-', linewidth=2*resolution)
    plt.axvline(x=0.8*fs*3, color='k', linestyle='-', linewidth=2*resolution)
    plt.xlim((0, 0.8*fs*4))
    plt.ylim((-7.5, 1.5))
    plt.title(id.split('\\')[-1][:-4], fontsize=20)
    fig.tight_layout()
    # add colorbar on x axis
    # cmappable = ScalarMappable(norm=Normalize(0,1), cmap = cmap)
    # cbar = plt.colorbar(cmappable, ax=ax, orientation='horizontal')
    plt.show()
    # plt.savefig('C:/Users/nater/OneDrive/Desktop/true positives/{}.png'.format(id.split('\\')[-1][:-4]), dpi=300)
    # plt.close()



print('done')