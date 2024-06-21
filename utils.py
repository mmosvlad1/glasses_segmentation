import os
import torch
import torchvision
from torch.utils.data import DataLoader

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

class EyeglassDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        parts = img_name.split('_')
        mask_name = '_'.join(parts[:2]) + '_mask.png'

        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image


def get_test_loader(test_dir, transform):
    test_ds = TestDataset(test_dir, transform)
    return DataLoader(test_ds)


def get_loaders(
    test_dir,
    test_mask_dir,
    val_transform,
    train_transform=None,
    train_dir=None,
    train_mask_dir=None,
    val_dir=None,
    val_mask_dir=None,
    batch_size=None,
    pin_memory=True,
    testing = False
):
    
    test_ds = EyeglassDataset(
    image_dir=test_dir,
    mask_dir=test_mask_dir,
    transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        pin_memory=pin_memory,
        shuffle=True,
    )

    if testing:
        return test_loader

    train_ds = EyeglassDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = EyeglassDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader

def check_metrics(loader, model, loss_fn, device="cuda"):
    num_correct = 0
    num_pixels = 0
    total_loss = 0.0
    all_precisions = []
    all_recalls = []
    all_dices = []
    all_ious = []

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(dim=1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            total_loss += loss_fn(preds, y)

            precision, recall, dice, iou = calculate_metrics(y, preds)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_dices.append(dice)
            all_ious.append(iou)

    accuracy = (num_correct / num_pixels) * 100
    avg_loss = total_loss / len(loader)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_dice = np.mean(all_dices)
    avg_iou = np.mean(all_ious)

    print(f"Acc: {accuracy:.2f}%")
    model.train()
    return accuracy, avg_loss.item(), avg_precision, avg_recall, avg_dice, avg_iou


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy().astype(int)
    y_pred = y_pred.cpu().detach().numpy().astype(int)

    precision = precision_score(y_true.flatten(), y_pred.flatten())
    recall = recall_score(y_true.flatten(), y_pred.flatten())
    dice = f1_score(y_true.flatten(), y_pred.flatten())
    iou = jaccard_score(y_true.flatten(), y_pred.flatten())

    return precision, recall, dice, iou


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print("model saved successfully")


def load_checkpoint(filepath, model, device, optimizer=None):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])


def save_predictions_as_images(
        loader, model, folder="saved_images/", device="cuda", orig_img=False
):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Directory '{folder}' created successfully")

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.inference_mode():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        if orig_img:
            x_cpu = x.squeeze(dim=0).cpu()
            y_cpu = y.repeat(3, 1, 1).cpu()
            preds_cpu = preds.squeeze(dim=0).repeat(3, 1, 1).cpu()
            torchvision.utils.save_image([x_cpu, y_cpu, preds_cpu], f"{folder + str(idx) + '.png'}")
        else:
            preds_cpu = preds.cpu()
            torchvision.utils.save_image(preds_cpu, f"{folder + str(idx) + '.png'}")

    model.train()


def plot_random_samples(loader, model, device, num_samples=6):
    model.eval()
    random_indices = random.sample(range(len(loader.dataset)), num_samples)

    rows, cols = int(num_samples/3+0.7), 3

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    axs = axs.flatten()

    for i, idx in enumerate(random_indices):
        if len(loader.dataset[idx]) == 2:
            x, _ = loader.dataset[idx]
        else:
            x = loader.dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.inference_mode():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float().squeeze().cpu().numpy()

        x = np.einsum('ijkm->ikmj', x).squeeze()
        axs[i].imshow(x)
        axs[i].imshow(preds, cmap="gray", alpha=0.5)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
    model.train()