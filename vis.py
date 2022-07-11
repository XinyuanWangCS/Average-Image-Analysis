import torch
import numpy as np
import matplotlib.pyplot as plt 
from utils import show_img

def min_max_normalize(img):
    for i in range(img.shape[0]):
        img[i] -= img[i].min()
        img[i] /= img[i].max()
    return img
        
def show_mask(mask, title='', transpose=True, norm=False, clip=False, axis=None):
    if mask.ndim==4:
        mask = mask[0]
    if norm:
        mask = min_max_normalize(mask)
    if clip:
        mask = np.clip(mask, 0, 1)
    if transpose:
        mask = np.transpose(mask, (1,2,0))
    (vmin, vmax) = (0, 1)
    if axis is None:
        plt.imshow(mask, interpolation='lanczos')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        axis.imshow(mask, interpolation='lanczos')
        if title:
            axis.set_title(title)
        axis.axis('off')
        
def visualize_imagenet(model, img, title='', norm=False, clip=False, save=False, transpose=True, inv=False, to_cpu=True):
    visualize = VanillaGradient(model)
    vg_mask = visualize.get_mask(image_tensor=img)
    torch.cuda.empty_cache()
    
    visualize = Deconvolution(model)
    deconv_mask = visualize.get_mask(image_tensor=img)
    
    torch.cuda.empty_cache()
    visualize = GuidedBackprop(model)
    gb_mask = visualize.get_mask(image_tensor=img)
    
    torch.cuda.empty_cache()
    visualize = IntegratedGradients(model)
    ig_mask = visualize.get_ig_mask(image_tensor=img)
    torch.cuda.empty_cache()
    
    figure, axes = plt.subplots(1, 5, figsize=(16, 5), tight_layout=True)
    show_img(img, inv=inv, title=title, axis=axes[0], transpose=transpose, to_cpu=to_cpu)
    show_mask(vg_mask, title='Vanilla Gradient', axis=axes[1], norm=norm, clip=clip, transpose=transpose)
    show_mask(deconv_mask, title='Deconvolution', axis=axes[2], norm=norm, clip=clip, transpose=transpose)
    show_mask(gb_mask, title='Guided Backpropgation', axis=axes[3], norm=norm, clip=clip, transpose=transpose)
    show_mask(ig_mask, title='Integrated Gradients', axis=axes[4], norm=norm, clip=clip, transpose=transpose)
    figure.show()
    if save:
        figure.savefig('./images/vis_results/'+title+'.jpg')
        

class SaliencyMask(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradient = None
        self.hooks = list()

    def get_mask(self, image_tensor, target_class=None):
        raise NotImplementedError('A derived class should implemented this method')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

class VanillaGradient(SaliencyMask):
    def __init__(self, model):
        super(VanillaGradient, self).__init__(model)

    def get_mask(self, image_tensor, target_class=None):
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        logits = self.model(image_tensor)
        target = torch.zeros_like(logits)
        target[0][target_class if target_class else logits.topk(1, dim=1)[1]] = 1
        self.model.zero_grad()
        logits.backward(target)
        return image_tensor.grad.detach().cpu().numpy()[0]
    
    def get_smoothed_mask(self, image_tensor, target_class=None, samples=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((width, height, channels))
        for sample in range(samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples

    @staticmethod
    def apply_region(mask, region):
        return mask * region[..., np.newaxis]

class Deconvolution(VanillaGradient):
    def __init__(self, model):
        super(Deconvolution, self).__init__(model)
        self.update_relus()

    def update_relus(self):
        def clip_gradient(module, grad_input, grad_output):
            return (grad_output[0] * (grad_output[0] > 0.).float(),)

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_backward_hook(clip_gradient))
                
class GuidedBackprop(VanillaGradient):
    def __init__(self, model):
        super(GuidedBackprop, self).__init__(model)
        self.relu_inputs = list()
        self.update_relus()

    def update_relus(self):
        def clip_gradient(module, grad_input, grad_output):
            relu_input = self.relu_inputs.pop()
            return (grad_output[0] * (grad_output[0] > 0.).float() * (relu_input > 0.).float(),)

        def save_input(module, input, output):
            self.relu_inputs.append(input[0])

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_forward_hook(save_input))
                self.hooks.append(module.register_backward_hook(clip_gradient))
                
class IntegratedGradients(VanillaGradient):
    def get_ig_mask(self, image_tensor, target_class=None, baseline='black', steps=25, process=lambda x: x):
        if baseline is 'black':
            baseline = torch.ones_like(image_tensor) * torch.min(image_tensor).detach().cpu()
        elif baseline is 'white':
            baseline = torch.ones_like(image_tensor) * torch.max(image_tensor).detach().cpu()
        else:
            baseline = torch.zeros_like(image_tensor)

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((channels, width, height))
        
        image_diff = image_tensor - baseline
        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            image_step = baseline + alpha * image_diff
            grad_sum += self.get_mask(image_step, target_class)
        return grad_sum * image_diff.detach().cpu().numpy()[0] / steps

'''class IntegratedGradients(VanillaGradient):
    def get_mask(self, image_tensor, target_class=None, baseline='black', steps=25, process=lambda x: x):
        if baseline is 'black':
            baseline = torch.ones_like(image_tensor) * torch.min(image_tensor).detach().cpu()
        elif baseline is 'white':
            baseline = torch.ones_like(image_tensor) * torch.max(image_tensor).detach().cpu()
        else:
            baseline = torch.zeros_like(image_tensor)

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((channels, width, height))
        image_diff = image_tensor - baseline

        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            image_step = baseline + alpha * image_diff
            grad_sum += process(super(IntegratedGradients, self).get_mask(image_step, target_class))
        return grad_sum * image_diff.detach().cpu().numpy()[0] / steps'''
