import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import copy


def make_gif(filename):

    gt_dirname= 'segmentation_gt'
    seg_map_3_dirname = 'frankfurt_0_segmentation'
    rgb_map_dirname = 'frankfurt_0_rgb'

    gt_filename= os.path.join('segmentation_gt',filename+'_maskrcnntargetsemanticsegmentation.png')
    seg_map_3_filename= os.path.join('segmentation_gt',filename+'_maskrcnntargetsemanticsegmentation.png')


    gt = cv.imread(gt_filename)
    seg_map_3 = cv.imread(filename+"_prediction_3.png")
    rgb_map = cv.imread("frankfurt_0_rgb/frankfurt_000000_000576_leftImg8bit.png")
    pastel_cmap=plt.get_cmap('Pastel1')
    red_mask =np.zeros(seg_map_3.shape)
    red_mask[:,:,2] = 255

    # numcolors=len(pastel_cmap.colors)
    # seg_map_color=np.zeros((seg_map.shape[0],seg_map.shape[1],3))
    # for category,color in enumerate(pastel_cmap.colors):
    #     keep_mask = seg_map == category
    #     if category==0:
    #         keep_mask_0 = np.asarray([keep_mask,keep_mask,keep_mask])
    #         seg_map_color[keep_mask, :] = (0.0,0.0,0.0)
    #         continue
    #     seg_map_color[keep_mask,:] = color

    diff_mask = seg_map_3==gt

    seg_map_mixed = 0.5*red_mask*(~diff_mask) + rgb_map/255.0
    #seg_map_mixed = 100*(seg_map-gt)*(seg_map-gt) + 0.5*rgb_map/255.0
    #seg_map_mixed = 0.7*seg_map_color + 0.3*rgb_map/255.0


    set_map=plt.imshow(seg_map_mixed)
    plt.show()
    cv.imwrite('frankfurt_000000_000577_gt_discrepancy.png',seg_map_mixed*255)
