import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from utils import show_img
from sklearn.metrics.pairwise import cosine_similarity


def load_ave_data(model_name, ave_type, proportion=False, path='./results'):
    if proportion:
        ave_data_path = os.path.join(path, ave_type,'ave_data', model_name+'_ave_data_proportion.npy')
    else:
        ave_data_path = os.path.join(path, ave_type,'ave_data', model_name+'_ave_data.npy')
    ave_data = np.load(ave_data_path, allow_pickle=True)
    return torch.from_numpy(ave_data)

class SaliencyMaskKernel(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradient = None
        self.hooks = list()
        self.count = 0
        self.conv_count = []
        self.model_structure = []
        self.conv_locations = []
        self.conv_kernel_nums = []
        
        self.get_conv_locations()
        
    def get_kernel_mask(self, image_tensor, layer_num, kernel_num):
        raise NotImplementedError('A derived class should implemented this method')
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def _DFS(self, module):
        if len(list(module.children())) == 0:
            if isinstance(module, nn.Conv2d):
                self.conv_count.append(self.count)
                self.conv_kernel_nums.append(module.out_channels)
                self.count+=1
                return self.count-1
            else:
                self.count+=1
                return self.count-1
        else:
            subset = []
            for child in module.children():
                subset+=[self._DFS(child)]
            return subset
        
    def _include_conv(self, structure_list, conv_num):
        for i in structure_list:
            if isinstance(i, int):
                if conv_num==i:
                    return True
            else:
                if self._include_conv(i, conv_num):
                    return True
        return False


    def _locate_conv(self, structure_list, conv_num, location = []):
        count = 0
        for i in structure_list:
            if isinstance(i, int):
                if conv_num==i:
                    location.append(count)
                    return
                else:
                    count+=1
            else:
                if not self._include_conv(i, conv_num):
                    count+=1
                else:
                    location.append(count)
                    loc = self._locate_conv(i, conv_num, location)

    def get_conv_locations(self):
        self.model_structure = self._DFS(self.model)
        locations = []
        for i in self.conv_count:
            location = []
            self._locate_conv(self.model_structure, i, location)
            locations.append(location)
        self.conv_locations = locations
                       
class GuidedBackpropKernel(SaliencyMaskKernel):
    def __init__(self, model):
        super(GuidedBackpropKernel, self).__init__(model)
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
    
    def get_cls_mask(self, image_tensor, target_cls=None):
        image_tensor = image_tensor.clone().unsqueeze(0)
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        logits = self.model(image_tensor)
        target = torch.zeros_like(logits)
        target[0][target_cls if target_cls else logits.topk(1, dim=1)[1]] = 1
        self.model.zero_grad()
        logits.backward(target)
        return image_tensor.grad.detach()[0]
    
    def get_kernel_mask(self, image_tensor, layer_num, kernel_num, debug=False):
        loc = self.conv_locations[layer_num]
        
        image_tensor = image_tensor.clone().unsqueeze(0)
        image_tensor.requires_grad = True
        image_tensor.retain_grad()
        
        children = self.model
        x = image_tensor
        
        for i in range(len(loc)):
            children = list(children.children())
            for j in range(loc[i]+1):
                if j!=loc[i]:
                    x = children[j](x)
            children = children[j]
        x = children(x)
        
        self.model.zero_grad()
        conv_output = torch.sum(torch.abs(x[0, kernel_num]))
        if debug:
            print(conv_output)
        conv_output.backward()

        return image_tensor.grad.detach().cpu().numpy()[0]
    
    def get_sim_value(self, image_tensor, vis_tensor):
        return cosine_similarity(image_tensor.reshape(1, -1), vis_tensor.reshape(1, -1))
    
    def make_layer_dirs(self, path='./results', model_name='',layer=''):
        path = os.path.join(path, model_name)
        sim_path = os.path.join(path, 'sims')
        vis_path = os.path.join(path, 'vis_results')

        if not os.path.exists(path):
            os.mkdir(path)
            os.mkdir(sim_path)
            os.mkdir(vis_path)

        sim_path = os.path.join(path, 'sims', 'layer_'+str(layer))
        vis_path = os.path.join(path, 'vis_results', 'layer_'+str(layer))

        if not os.path.exists(sim_path):
            os.mkdir(sim_path)
            os.mkdir(vis_path)
            for i in range(10):
                os.mkdir(os.path.join(vis_path, str(i)))
        return sim_path, vis_path
    
    def normalize(self, img_tensor):
        for i in range(img_tensor.shape[0]):
            img_tensor[i] -= img_tensor[i].min()
            img_tensor[i] /= img_tensor[i].max()
        return img_tensor
        
    def get_layer_mask_and_sim_value(self, image_tensor, target_cls, layer_num, path='./results', model_name='',norm=True, save=True, debug=False):
        kernel_num = self.conv_kernel_nums[layer_num]
        sim_path, vis_path = self.make_layer_dirs(path=path, model_name=model_name, layer=str(layer_num))
        
        sim_list = []
        error_count = 0
        image_numpy = image_tensor.cpu().numpy()
        for i in range(kernel_num):
            vis_numpy = self.get_kernel_mask(image_tensor, layer_num, i)
            try:
                sim = self.get_sim_value(image_numpy, vis_numpy)#vis结果不做任何处理（normalize）
            except BaseException:
                sim_list.append(0)
                error_count += 1
            else:
                sim_list.append(sim[0])
            if debug:
                print(vis_numpy)
                print(sim)
            if norm:
                vis_numpy = self.normalize(vis_numpy)
            if save:
                if vis_numpy.shape[0]==3:
                    vis_numpy = np.transpose(vis_numpy, (1, 2, 0))
                else:
                    vis_numpy = vis_numpy[0]
                matplotlib.image.imsave(os.path.join(vis_path,str(target_cls),str(i).zfill(3)+'.jpg'),vis_numpy)
        sim_list = np.array(sim_list)
        if save:
            np.save(os.path.join(sim_path, str(target_cls)), sim_list)
        print('Model:{0}  Class:{1}  Layer:{2}  Filter_num:{3}  Mean Sim:{4:.3f}  Errors:{5}'.format(model_name, target_cls, layer_num, kernel_num, sim_list.mean(), error_count))
        
    def get_layer_mask_and_two_vis_sim_value(self, image_tensor, target_cls, layer_num, path='./results', model_name='',norm=True, save=True, debug=False):
        cls_mask = self.get_cls_mask(image_tensor=image_tensor, target_cls=target_cls)
        self.get_layer_mask_and_sim_value(image_tensor=cls_mask, 
                                   target_cls=target_cls, 
                                   layer_num=layer_num, 
                                   path=path, 
                                   model_name=model_name,
                                   norm=norm, 
                                   save=save,
                                   debug=debug
                                   )
            
    def get_all_layer_mask_and_sim_value(self, image_tensor, target_cls, path='./results', model_name='',norm=True, save=True):
        layer_num = len(self.conv_kernel_nums)
        for i in range(layer_num):
            self.get_layer_mask_and_sim_value(image_tensor=image_tensor, 
                                   target_cls=target_cls, 
                                   layer_num=i, 
                                   path=path, 
                                   model_name=model_name,
                                   norm=norm, 
                                   save=save
                                   )
    
    