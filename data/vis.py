import numpy as np
import matplotlib.pyplot as plt

data = np.load('pts_in_hull.npy')
probs = np.load('prior_probs.npy')
# plt.figure(figsize=(15, 15))
# gs = gridspec.GridSpec(1, 1)
# ax = plt.subplot(gs[0])
# for i in range(q_ab.shape[0]):
#     ax.scatter(q_ab[:, 0], q_ab[:, 1])
#     ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
#     ax.set_xlim([-110,110])
#     ax.set_ylim([-110,110])
