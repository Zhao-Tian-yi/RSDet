import os
import pdb
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot, inference_detector_kaist, write_result_txt_img


def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')

    parser.add_argument('--img_path', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--img_res', help='Image file')
    parser.add_argument('--checkpoint_start', type=int, default=3)
    parser.add_argument('--checkpoint_end', type=int, default=12)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    args = parser.parse_args()

    img_path = args.img_path
    imgList = os.listdir(img_path)
    imgList = sorted(imgList)
    import pdb
    pdb.set_trace()
    for index in range(args.checkpoint_start, args.checkpoint_end+1):
        # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint+'epoch_{}.pth'.format(index), device=args.device)
        print(args.checkpoint+'{}.pth'.format(index))
        for i in range(0, len(imgList), 2):
            lwir_img_path = os.path.join(img_path, imgList[i])
            vis_img_path = os.path.join(img_path, imgList[i+1])
            # test a single image
            result = inference_detector_kaist(model, vis_img_path, lwir_img_path)
            name_slit = vis_img_path.split('.')[0].split('/')[-1].split('_')
            imgname = name_slit[0]+'_' + name_slit[1] + '_' + name_slit[2]
            # show the results
            write_result_txt_img(model, lwir_img_path, imgname, args.img_res+'{}'.format(index), result, score_thr=args.score_thr)
        print('write in {} done!'.format(args.img_res+'{}'.format(index)))

if __name__ == '__main__':
    main()
