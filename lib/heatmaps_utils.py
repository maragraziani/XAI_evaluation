import cv2
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# pip install lime==0.1.1.37 (latest version available working with python2.7)
from lime import lime_image

from keras import backend as K 
from keras.layers import Conv2D
from keras.models import Model

from skimage.measure import compare_ssim as ssim
from skimage.segmentation import mark_boundaries, felzenszwalb, quickshift, slic, watershed

from sklearn.metrics import average_precision_score, precision_recall_curve, auc

### CAM ###
def get_cam_model(model, last_conv_layer, pred_layer):
    n_classes = model.output_shape[-1]
    final_params = model.get_layer(pred_layer).get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, n_classes), final_params[1])

    last_conv_output = model.get_layer(last_conv_layer).output
    # upgrade keras to 2.2.3 in order to use UpSampling2D with bilinear interpolation
    # x = UpSampling2D(size=(32, 32))(last_conv_output)
    x = Conv2D(filters=n_classes, kernel_size=(1, 1), name='predictions_2')(last_conv_output)

    cam_model = Model(inputs=model.input, outputs=[model.output, x])
    cam_model.get_layer('predictions_2').set_weights(final_params)
    return cam_model

def postprocess(preds, cams, top_k=1):
    idxes = np.argsort(preds[0])[-top_k:]
    class_activation_map = np.zeros_like(cams[0, :, :, 0])
    for i in idxes:
        class_activation_map += cams[0, :, :, i]
    return class_activation_map

def cam(model, img, last_conv, original_size, pred_layer=None):
    if pred_layer is None:
        pred_layer = model.layers[-1].name
    cam_model = get_cam_model(model, last_conv_layer=last_conv, pred_layer=pred_layer)
    preds, cams = cam_model.predict(img)
    class_activation_map = postprocess(preds, cams)
    return cv2.resize(class_activation_map, original_size)

### gradCAM ###
def gradcam(model, img, layer_name, original_size):
    """ Grad-CAM function """
    
    cls = np.argmax(model.predict(img))
    
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    # Get outputs and grads
    gradient_function = K.function([model.input], [conv_output, grads])
    output, grads_val = gradient_function([img])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    
    weights = np.mean(grads_val, axis=(0, 1)) # Passing through GlobalAveragePooling

    cam = np.dot(output, weights) # multiply
    cam = np.maximum(cam, 0)      # Passing through ReLU
    cam /= np.max(cam)            # scale 0 to 1.0

    cam = cv2.resize(cam, original_size)
    return cam

### gradCAM++ ###
def gradcam_plus_plus(model, img, layer_name, original_size):
    """ Grad-CAM++ function """
    
    cls = np.argmax(model.predict(img))
    y_c = model.output[0, cls]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    first = K.exp(y_c) * grads
    second = K.exp(y_c) * grads * grads
    third = K.exp(y_c) * grads * grads * grads

    gradient_function = K.function([model.input], [y_c, first, second, third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom # 0


    weights = np.maximum(conv_first_grad[0], 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0) # 0
    alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2])) # NAN
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)

    cam = np.sum(deep_linearization_weights * conv_output[0], axis=2)
    cam = np.maximum(cam, 0) # Passing through ReLU
    cam /= np.max(cam)       # scale 0 to 1.0  

    cam = cv2.resize(cam, original_size)
    return cam

# Superpixels segementations algorithms (used for LIME)
def segments_box(image, n, input_size):
    size = (input_size[0]/n, input_size[1]/n)
    _segs = []
    for k in range(n):
        _segs.append(k*np.ones(size, dtype=np.int32))
    segs_row = np.concatenate(tuple(_segs), axis=1)
    _segs = []
    for k in range(n):
        _segs.append(segs_row+n*k)
    return np.concatenate(tuple(_segs), axis=0)

def segments_fz(image):
    return felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

def segments_slic(image):
    return slic(image, n_segments=250, compactness=10, sigma=1)

def segments_quick(image):
    return quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)


def compute_cams(heatmaps):
    _tmp = heatmaps['CAM'] + heatmaps['gradCAM'] + heatmaps['gradCAM++']
    _tmp /= 3
    heatmaps['CAMs'] = _tmp
    return heatmaps

def compute_avg(heatmaps):
    _tmp = heatmaps['Quickshift'] + heatmaps['SLIC'] + heatmaps['Felzenszwalb']
    _tmp /= 3
    heatmaps['AVG'] = _tmp
    return heatmaps

def compute_squaregrid(heatmaps):
    _tmp = 0
#     _n = 0
    for m, h in heatmaps.items():
        if m.startswith('boxes'):
            _tmp += h
#             _n +=1
#     _tmp /= _n
    heatmaps['Squaregrid'] = _tmp
    return heatmaps

    
def get_method_num(method):
    if method == 'original':
        return 0
    #elif method == 'mask':
    #    return 1
    elif method == 'CAM':
        return 1
    #elif method == 'CAM norm':
    #    return 2.5
    elif method == 'gradCAM':
        return 3
    elif method == 'gradCAM++':
        return 4
    elif method == 'CAMs':
        return 5
    elif method == 'Quickshift':
        return 6
    elif method == 'SLIC':
        return 7
    elif method == 'Felzenszwalb':
        return 8
    elif method == 'Watershed':
        return 9
    elif method == 'AVG':
        return 10
    elif method == 'Squaregrid':
        return 11
    elif method.startswith('boxes'):
        return 3*int(method.split('_')[-1])
    else:
        raise ValueError('Unknown method {}'.format(method))

def find_extreme_values(res):
    _dict = copy.deepcopy(res)
    if 'original' in res:
        del _dict['original']
    if 'mask' in res:
        del _dict['mask']
    min_value = np.min(_dict.values())
    max_value = np.max(_dict.values())
    return min_value, max_value
    
def plot_heatmaps(heatmaps, global_colorbar=True, symmetrical_colorbar=False, cmap='RdBu', save=None):
    nrows, ncols = len(heatmaps), len(heatmaps[0])
    figsize = (6.4*ncols, 4.8*nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)
    methods = heatmaps[0].keys()
    methods = sorted(methods, key=lambda method: get_method_num(method))
    for i in range(nrows):
        for j in range(ncols):
            vmin, vmax = find_extreme_values(heatmaps[i])
#             print vmin, vmax
            if symmetrical_colorbar:
                if -vmin > vmax:
                    vmax = -vmin
                vmin = -vmax
            method = methods[j]
            axes[0, j].set_title(method, size='xx-large')
            axes[i, j].axis('off')
            if method == 'original':
                axes[i, j].imshow(heatmaps[i]['original'])
            elif method == 'mask':
                if not heatmaps[i]['mask'] is None:
                    axes[i, j].imshow(heatmaps[i]['mask'])
            else:
                heatmap = heatmaps[i][method]
                if not global_colorbar:
                    vmin, vmax  = heatmap.min(), heatmap.max()
#                     print i, method, vmin, vmax
                    if symmetrical_colorbar:
                        if -vmin > vmax:
                            vmax = -vmin    
                        vmin = -vmax
                    im = axes[i, j].imshow(heatmap, cmap = cmap, vmin  = vmin, vmax = vmax)
                    cax, kw = mpl.colorbar.make_axes([axes[i, j ]])
                    plt.colorbar(im, cax=cax, **kw)
                else:
                    im = axes[i, j].imshow(heatmap, cmap = cmap, vmin  = vmin, vmax = vmax)
    if global_colorbar:
        cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
    if not save is None:
        plt.savefig(new_folder+'/'+save+'.png')
    plt.show()


def normalize_heatmap(heatmap, vmin=0, vmax=1):
    if heatmap is None:
        return None
    ht = heatmap
    #h = copy.deepcopy(heatmap)
    #return (vmax-vmin)*(ht-np.min(ht))/(np.max(ht)-np.min(ht))+vmin
    return (ht-vmin)/(vmax-vmin)

def compute_ssim(heatmaps, method1, method2):
    h1 = normalize_heatmap(np.asarray(heatmaps[method1], dtype=np.float64))
    h2 = normalize_heatmap(np.asarray(heatmaps[method2], dtype=np.float64))
    return ssim(h1, h2)

def auprc(mask, heatmap, plot=False):
    mask = normalize_heatmap(mask)
    heatmap = normalize_heatmap(heatmap)
    if plot:
        figsize = (12.8, 4.8)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        axes[0].imshow(mask)
        axes[1].imshow(heatmap, cmap='jet')
        plt.show() 
    precision, recall, _ = precision_recall_curve(mask.flatten(), heatmap.flatten())
    return auc(recall, precision), precision, recall

def auprc_heatmaps(heatmaps, method1, method2, nb_step=11, plot_each=False, plot_final=True):
    thresholds = [threshold for threshold in np.linspace(0, 1, nb_step)]
    aucs = []
    for threshold in thresholds[1:-1]:
        threshold = round(threshold, 3)
        
        mask = normalize_heatmap(heatmaps[method1])
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        
        heatmap = heatmaps[method2]
        _auc, precision, recall = auprc(mask, heatmap, plot=plot_each)
        plt.plot(recall, precision, label=threshold)
        aucs.append(_auc)
    if not plot_each and plot_final:
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()
    return aucs, thresholds