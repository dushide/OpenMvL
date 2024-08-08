

import numpy as np


def calc_bins(preds,labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def one_hot(labels, num_classes=None):
    """
    将标签转换为独热编码
    :param labels: 标签，可以是list、tuple、ndarray等
    :param num_classes: 标签总数，如果不指定则根据labels中的值自动确定
    :return: 独热编码矩阵
    """
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]

def get_metrics(preds,labels):
  ECE = 0
  MCE = 0

  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds,labels)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE