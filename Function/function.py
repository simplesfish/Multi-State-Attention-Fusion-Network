import random
import numpy as np
import torch
from tqdm import tqdm


# --------------------------------------SEED--------------------------------------#
def seed_func():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------Calculate accuracy--------------------------------------#


def cal_accuracy_fewshot(loader, net):
    total_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for support_images, support_labels, query_images, query_labels in tqdm(
        loader, desc="Evaluating"
    ):
        support_images = support_images.squeeze(0).to(device)
        support_labels = support_labels.squeeze(0).to(device)
        query_images = query_images.squeeze(0).to(device)
        query_labels = query_labels.squeeze(0).to(device)

        logits = net(support_images, support_labels, query_images)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == query_labels).float().mean().item()
        total_acc += acc
    return total_acc / len(loader)
