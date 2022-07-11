import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from PIL import Image
import torchvision
import torchvision.transforms as T

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 12   # 设置字体的大小为10

def draw_chart(model_name, show_test_loss=False, save=False, path='./result_fig/'):
    train_loss_l = np.load('./logs/'+model_name+'_train_loss_l.npy', allow_pickle=True).flatten()
    test_loss_l = np.load('./logs/'+model_name+'_test_loss_l.npy', allow_pickle=True)
    test_acc_l = np.load('./logs/'+model_name+'_test_acc_l.npy', allow_pickle=True)
    if show_test_loss:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
        axes[0].plot(range(len(train_loss_l)), train_loss_l)
        axes[0].set_title('训练损失曲线')
        axes[0].set_xlabel('训练步数')   
        axes[0].set_ylabel('训练损失') 
        axes[1].plot(range(len(test_acc_l)), test_acc_l)
        axes[1].set_title('测试准确率曲线')
        axes[1].set_xlabel('测试次数')   
        axes[1].set_ylabel('训准确率')  
        axes[1].set_ylim(0,1.1)
        axes[2].plot(range(len(test_loss_l)), test_loss_l)
        axes[2].set_title('测试损失曲线')
        axes[2].set_xlabel('测试次数')   
        axes[2].set_ylabel('测试损失')  
        if save:
            figure.savefig(path+model_name+'.jpg')
    else:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        axes[0].plot(range(len(train_loss_l)), train_loss_l)
        axes[0].set_title('训练损失曲线')
        axes[0].set_xlabel('训练步数')   
        axes[0].set_ylabel('训练损失')   
        axes[1].plot(range(len(test_acc_l)), test_acc_l)
        axes[1].set_title('测试准确率曲线')
        axes[1].set_xlabel('测试次数')   
        axes[1].set_ylabel('训准确率')
        axes[1].set_ylim(0,1)
        if save:
            figure.savefig(path+model_name+'.jpg')

def normalize(mask, vmin=None, vmax=None, percentile=99):
    if vmax is None:
        vmax = np.percentile(mask, percentile)
    if vmin is None:
        vmin = np.min(mask)
    return (mask - vmin) / (vmax - vmin + 1e-10)


def make_grayscale(mask):
    return np.sum(mask, axis=2)


def make_black_white(mask):
    return make_grayscale(np.abs(mask))

def cut_image_with_mask(image_path, mask, title='', percentile=70, axis=None):
    image = np.moveaxis(load_image(image_path, size=mask.shape[0], preprocess=False).numpy().squeeze(), 0, -1)
    mask = mask > np.percentile(mask, percentile)
    image[~mask] = 0

    if axis is None:
        plt.imshow(image, interpolation='lanczos')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        axis.imshow(image, interpolation='lanczos')
        if title:
            axis.set_title(title)
        axis.axis('off')


def show_mask_on_image(image_path, mask, title='', cmap=cc.cm.bmy, alpha=0.7, axis=None):
    image = load_image(image_path, size=mask.shape[0], color_mode='L', preprocess=False).numpy().squeeze()
    if axis is None:
        plt.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    else:
        axis.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    show_mask(mask, title, cmap, alpha, norm=False, axis=axis)


def pil_loader(path, color_mode='RGB'):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(color_mode)


def load_imagenet_img(path, color_mode='RGB'):
    pil_image = pil_loader(path, color_mode)
    shape = np.array(pil_image).shape
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225])
    ])
    return transform(pil_image).unsqueeze(0)

def show_img(img, transpose=False, to_cpu=False, inv=False, axis=None, title='',vmin=0, vmax=1, std=[1/0.229,1/0.224,1/0.225],mean=[-0.485,-0.456,-0.406]):
    if inv:
        invTrans = T.Compose([T.Normalize(mean = [ 0., 0., 0. ],
                               std = std),
                       T.Normalize(mean = mean,
                               std = [ 1., 1., 1. ]),])
        img = invTrans(img)
    if to_cpu:
        img = img.cpu().numpy()
    if img.ndim==4:
        img = img[0]
    if transpose:
        img = np.transpose(img, (1,2,0))
    if axis:
        axis.imshow(img, vmin=vmin, vmax=vmax)
        if title:
            axis.set_title(title)
        axis.axis('off')
    else:
        plt.imshow(img, vmin=vmin, vmax=vmax)
        if title:
            plt.set_title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()