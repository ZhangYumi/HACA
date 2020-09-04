from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot,show_result
import cv2

def main():
    parser = ArgumentParser()
    #parser.add_argument('img_dir',help='Image file Dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()

    import os
    img_dir = r"/home/amax/anaconda3/envs/mmdetection/mmdetection/demo/demo_image"
    out_dir = "/home/amax/anaconda3/envs/mmdetection/mmdetection/demo/"+args.config[8:-3]
    if(os.path.exists(out_dir) is None):
        os.makedirs(out_dir)
    for img_name in os.listdir(img_dir):
        img_path = img_dir + "/" +img_name
        out_file = out_dir + "/" + img_name
        img = cv2.imread(img_path)
    # build the model from a config file and a checkpoint file
        model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
        result = inference_detector(model, img)
    # show the results
    #show_result_pyplot(img, result, model.CLASSES,score_thr=args.score_thr)

        show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1,out_file=out_file)


if __name__ == '__main__':
    main()