import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go


def load_sims(model_name='',path='./results', show=True):
    sim_path = os.path.join(path, model_name, 'sims')
    layer_names = os.listdir(sim_path)
    all_layer_sims = []
    for layer_name in layer_names:
        layer_sims = []
        layer_path = os.path.join(sim_path, layer_name)
        for sim_name in os.listdir(layer_path):
            layer_sims.append(np.load(os.path.join(layer_path, sim_name), allow_pickle=True).squeeze())
        all_layer_sims.append(np.array(layer_sims))
    if show:
        print('Load all layer similarities(Layer-Class-Kernel): ', model_name, len(all_layer_sims))
    return all_layer_sims
    
def load_vis_results(model_name='',path='./results', show=True):
    vis_path = os.path.join(path, model_name, 'vis_results')
    layer_names = os.listdir(vis_path)
    all_layer_imgs = []
    for layer_name in layer_names:
        layer_vis = []
        layer_path = os.path.join(vis_path, layer_name)
        for cls_name in os.listdir(layer_path):
            cls_path = os.path.join(layer_path, cls_name)
            cls_vis = []
            for img_name in os.listdir(cls_path):
                cls_vis.append(plt.imread(os.path.join(cls_path, img_name)))
            layer_vis.append(np.array(cls_vis))
        all_layer_imgs.append(np.array(layer_vis))
    if show:
        print('Load all layer vis results(Layer-Class-Kernel): ', model_name, len(all_layer_imgs))
    return all_layer_imgs

def load_sims_and_vis_results(model_name='',path='./results'):
    all_layer_sims = load_sims(model_name=model_name, path=path)
    all_layer_vis_results = load_vis_results(model_name=model_name, path=path)
    return all_layer_sims, all_layer_vis_results

def draw_sim_one_layer(sim_data_one_layer, model_name=''):
    data = []
    for i in range(sim_data_one_layer.shape[0]):
        data.append(go.Scatter(x=np.arange(sim_data_one_layer[i].shape[0]), y=sim_data_one_layer[i], name='cls_'+str(i),mode='markers'))
    data.append(go.Scatter(x=np.arange(sim_data_one_layer[i].shape[0]), y=sim_data_one_layer.mean(axis=0), name='均值',mode='lines+markers'))
    fig = go.Figure(data)
    fig.update_layout(
        title=model_name+'卷积核相似值',
        xaxis_title='卷积核编号',
        yaxis_title='相似值'
    )
    fig.show()

def draw_sim_all_layers(sim_data, model_name=''):
    for i in range(len(sim_data)):
        draw_sim_one_layer(sim_data_one_layer=sim_data[i], model_name=model_name+'层'+str(i))

def draw_sim_mean_all_class(sim_data, model_name=''):
    layer_num = len(sim_data)
    class_num = sim_data[0].shape[0]
    cls_sim_data = []
    for layer in range(layer_num):
        cls_sim_data.append(sim_data[layer].mean(axis=1))
    cls_sim_data = np.array(cls_sim_data).T
    
    data = []
    for i in range(class_num):
        data.append(go.Scatter(x=np.arange(layer_num), y=cls_sim_data[i], name='cls_'+str(i),mode='lines+markers'))
    data.append(go.Scatter(x=np.arange(layer_num), y=cls_sim_data.mean(axis=0), name='均值',mode='lines+markers'))
    fig = go.Figure(data)
    fig.update_layout(
        title=model_name+'层相似值均值',
        xaxis_title='卷积层编号',
        yaxis_title='相似值'
    )
    fig.show()

def draw_sim_std_all_class(sim_data, model_name=''):
    layer_num = len(sim_data)
    class_num = sim_data[0].shape[0]
    cls_sim_data = []
    for layer in range(layer_num):
        cls_sim_data.append(sim_data[layer].std(axis=1))
    cls_sim_data = np.array(cls_sim_data).T
    
    data = []
    for i in range(class_num):
        data.append(go.Scatter(x=np.arange(layer_num), y=cls_sim_data[i], name='cls_'+str(i),mode='lines+markers'))
    data.append(go.Scatter(x=np.arange(layer_num), y=cls_sim_data.mean(axis=0), name='均值',mode='lines+markers'))
    fig = go.Figure(data)
    fig.update_layout(
        title=model_name+'层相似值标准差',
        xaxis_title='卷积层编号',
        yaxis_title='标准差'
    )
    fig.show()

def draw_vis_results_one_layer(sim_one_layer, layer_path, layer_num, model_name='', index=None, pic_one_row=6, head_and_tail=True):
    kernel_num = sim_one_layer.shape[1]
    
    if index is None:
        index = []
        for i in range(sim_one_layer.shape[0]):
            index.append(list(range(kernel_num))) 
            
    for cls, cls_name in enumerate(os.listdir(layer_path)):
        cls_path = os.path.join(layer_path, cls_name)
        if not head_and_tail:
            lines = kernel_num//pic_one_row
            if kernel_num  % pic_one_row != 0:
                lines+=1

            figure, axes = plt.subplots(lines, pic_one_row, figsize=(16, 3*lines), tight_layout=True, clear=True)
            for i in range(lines):    
                for j in range(pic_one_row):
                    if i*pic_one_row+j==kernel_num:
                        break
                    axe = axes[i, j]
                    pic_path = os.path.join(cls_path, str(index[cls][i*pic_one_row+j]).zfill(3)+'.jpg')
                    img = plt.imread(pic_path)
                    axe.imshow(img)
                    round_sim = np.round(sim_one_layer[cls][index[cls][i*pic_one_row+j]],3)
                    axe.set_title('Kernel: '+str(index[cls][i*pic_one_row+j])+'\nSim: '+str(round_sim))   
            figure.suptitle('Model:'+model_name+' Layer:'+str(layer_num)+' Class:'+cls_name,fontsize=10)
            figure.tight_layout()
        else:
            lines=pic_one_row//8
            figure, axes = plt.subplots(lines, 8, figsize=(16, 3*lines), tight_layout=True, clear=True)
            for i in range(lines): 
                
                for j in range(8):
                    axe = axes[i,j]
                    pic_path = os.path.join(cls_path, str(index[cls][i*pic_one_row+j]).zfill(3)+'.jpg')
                    img = plt.imread(pic_path)
                    axe.imshow(img)
                    round_sim = np.round(sim_one_layer[cls][index[cls][i*pic_one_row+j]],3)
                    axe.set_title('Kernel: '+str(index[cls][i*pic_one_row+j])+'\nSim: '+str(round_sim))   
                figure.suptitle('Model:'+model_name+' Layer:'+str(layer_num)+' Class:'+cls_name,fontsize=15)
                
            figure, axes = plt.subplots(lines, 8, figsize=(16, 3*lines), tight_layout=True, clear=True)   
            for i in range(lines): 
                
                for j in range(8):
                    axe = axes[i,j]
                    pic_path = os.path.join(cls_path, str(index[cls][-(i*pic_one_row+j+1)]).zfill(3)+'.jpg')
                    img = plt.imread(pic_path)
                    axe.imshow(img)
                    round_sim = np.round(sim_one_layer[cls][index[cls][-(i*pic_one_row+j+1)]],3)
                    axe.set_title('Kernel: '+str(index[cls][-(i*pic_one_row+j+1)])+'\nSim: '+str(round_sim))   
            
                    
def draw_sorted_vis_results_one_layer(sim_one_layer, layer_path, layer_num, model_name='', pic_one_row=6, head_and_tail=True):
    argsort_sims = sim_one_layer.argsort(axis=1)
    draw_vis_results_one_layer(sim_one_layer=sim_one_layer, 
                               layer_path=layer_path, 
                               layer_num=layer_num,
                               model_name=model_name, 
                               index=argsort_sims, 
                               pic_one_row=pic_one_row,
                               head_and_tail=head_and_tail)
    
def draw_vis_results_one_cls(sim_one_layer, layer_path, layer_num, cls, model_name='', index=None, pic_one_row=6, head_and_tail=True):
    kernel_num = sim_one_layer.shape[1]
    
    if index is None:
        index = []
        for i in range(sim_one_layer.shape[0]):
            index.append(list(range(kernel_num))) 
    cls_name = os.listdir(layer_path)[cls]    
    cls_path = os.path.join(layer_path, cls_name)

    lines=pic_one_row//8
    figure, axes = plt.subplots(lines, 8, figsize=(16, 3*lines), tight_layout=True, clear=True)
    for i in range(lines): 
        for j in range(8):
            axe = axes[i,j]
            pic_path = os.path.join(cls_path, str(index[cls][i*pic_one_row+j]).zfill(3)+'.jpg')
            img = plt.imread(pic_path)
            axe.imshow(img)
            round_sim = np.round(sim_one_layer[cls][index[cls][i*pic_one_row+j]],3)
            axe.set_title('Kernel: '+str(index[cls][i*pic_one_row+j])+'\nSim: '+str(round_sim))   
    figure.suptitle('Model:'+model_name+' Layer:'+str(layer_num)+' Class:'+cls_name,fontsize=15)
                
    figure, axes = plt.subplots(lines, 8, figsize=(16, 3*lines), tight_layout=True, clear=True)   
    for i in range(lines):       
        for j in range(8):
            axe = axes[i,j]
            pic_path = os.path.join(cls_path, str(index[cls][-(i*pic_one_row+j+1)]).zfill(3)+'.jpg')
            img = plt.imread(pic_path)
            axe.imshow(img)
            round_sim = np.round(sim_one_layer[cls][index[cls][-(i*pic_one_row+j+1)]],3)
            axe.set_title('Kernel: '+str(index[cls][-(i*pic_one_row+j+1)])+'\nSim: '+str(round_sim))

def draw_sorted_vis_results_one_cls(sim_one_layer, layer_path, layer_num, cls, model_name='', pic_one_row=6, head_and_tail=True):
    argsort_sims = sim_one_layer.argsort(axis=1)
    draw_vis_results_one_cls(sim_one_layer=sim_one_layer, 
                               layer_path=layer_path, 
                               layer_num=layer_num,
                               cls=cls,
                               model_name=model_name, 
                               index=argsort_sims, 
                               pic_one_row=pic_one_row,
                               head_and_tail=head_and_tail)
            
def draw_vis_results_one_cls_all_layers(vis_results, sims, cls, path='./results', model_name='', pic_one_row=6, sort=True, head_and_tail=True):
    path = os.path.join(path, model_name, 'vis_results')
    for i, layer_name in enumerate(os.listdir(path)):
        layer_path = os.path.join(path, layer_name)
        draw_sorted_vis_results_one_cls(sim_one_layer=sims[i], 
                                              layer_path=layer_path, 
                                              layer_num=i,
                                        cls=cls,
                                              model_name=model_name, 
                                              pic_one_row=pic_one_row,
                                             head_and_tail=head_and_tail)

            
def draw_vis_results_all_layers(vis_results, sims, path='./results', model_name='', pic_one_row=6, sort=True, head_and_tail=True):
    path = os.path.join(path, model_name, 'vis_results')
    for i, layer_name in enumerate(os.listdir(path)):
        layer_path = os.path.join(path, layer_name)
        if sort:
            draw_sorted_vis_results_one_layer(sim_one_layer=sims[i], 
                                              layer_path=layer_path, 
                                              layer_num=i,
                                              model_name=model_name, 
                                              pic_one_row=pic_one_row,
                                             head_and_tail=head_and_tail)
        else:
            draw_vis_results_one_layer(flat_sim_layer=sims[i], 
                                       layer_path=layer_path, 
                                       layer_num=i,
                                       model_name=model_name, 
                                       index=None, 
                                       pic_one_row=pic_one_row,
                                      head_and_tail=head_and_tail)