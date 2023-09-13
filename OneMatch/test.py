import torch
import cv2
from model.OneMatch import OneMatch
import matplotlib.pyplot as plt
import numpy as np
from einops.einops import rearrange
from tool import RGB2YCrCb, make_matching_figure

if __name__ == '__main__':
    # config
    img0_pth = './data/vi/1.jpg'
    img1_pth = './data/ir/1.jpg'
    model_name = 'OneMatch-G'
    input_size = (480,360)

    if model_name == 'OneMatch-G':
        matcher = OneMatch(dim=256, input_size=input_size)
        matcher.load_state_dict(torch.load('./weights/OneMatch-G.ckpt'))

    elif model_name == 'OneMatch-C':
        matcher = OneMatch(dim=64, input_size=input_size)
        matcher.load_state_dict(torch.load('./weights/OneMatch-C.ckpt'))


    matcher = matcher.eval().cuda()

    img0_raw = cv2.imread(img0_pth)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, input_size)
    img1_raw = cv2.resize(img1_raw, input_size)

    img0 = torch.from_numpy(img0_raw)[None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    img0_raw = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
    img0 = rearrange(img0, 'n h w c ->  n c h w')
    vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

    mkpts0, mkpts1 = matcher(vi_Y, img1)
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()

    h, prediction =cv2.findHomography(mkpts1, mkpts0, cv2.USAC_MAGSAC, 5, confidence=0.99999, maxIters=100000)
    prediction = np.array(prediction, dtype=bool).reshape([-1])
    mkpts0_t = mkpts0[prediction]
    mkpts1_t = mkpts1[prediction]

    out_img = cv2.warpPerspective(img1_raw, h, (img1_raw.shape[1], img1_raw.shape[0]))


    fig = make_matching_figure(out_img, img0_raw, img1_raw, mkpts0_t, mkpts1_t)
    plt.show()


