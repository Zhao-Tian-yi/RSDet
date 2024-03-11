# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
# from mmcv.parallel import collate, scatter
from mmengine.runner import load_checkpoint

from mmdet.registry import DATASETS
from mmdet.utils import ConfigType
from ..evaluation import get_classes
from ..registry import MODELS
from ..structures import DetDataSample, SampleList
from ..utils import get_test_pipeline_cfg


def init_detector(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    palette: str = 'none',
    device: str = 'cuda:0',
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        palette (str): Color palette used for visualization. If palette
            is stored in checkpoint, use checkpoint's palette first, otherwise
            use externally passed palette. Currently, supports 'coco', 'voc',
            'citys' and 'random'. Defaults to none.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    scope = config.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(config.get('default_scope', 'mmdet'))

    model = MODELS.build(config.model)
    model = revert_sync_batchnorm(model)
    if checkpoint is None:
        warnings.simplefilter('once')
        warnings.warn('checkpoint is None, use COCO classes by default.')
        model.dataset_meta = {'classes': get_classes('coco')}
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})

        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

    # Priority:  args.palette -> config -> checkpoint
    if palette != 'none':
        model.dataset_meta['palette'] = palette
    else:
        test_dataset_cfg = copy.deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

class LoadImage(object):
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(
    model: nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_['text'] = text_prompt
            data_['custom_entities'] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list
class LoadImageKaist(object):
    """A simple pipeline to load image"""

    def __call__(self, results):
        """Call function to load images into results

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        import pdb
        pdb.set_trace()
        if isinstance(results['img'], str) and isinstance(results['img_lwir'], str):
            results['img_path'] = results['img']
            results['img_lwir_path'] = results['img_lwir']
            # name_slit = results['img'].split('.')[0].split('/')[-1].split('_')
            # results['ori_filename'] = 'set'+name_slit[0]+'_V'+name_slit[1].zfill(3)+'_'+name_slit[2]
            results['ori_filename'] = results['img'].split('.')[0].split('/')[-1][:-2]
            results['ori_filename_vis']=results['img'].split('.')[0].split('/')[-1]
            results['ori_filename_lwir'] = results['img_lwir'].split('.')[0].split('/')[-1]
        else:

            results['img_path'] =None
            results['img_lwir_path'] =None

        img = mmcv.imread(results['img'])
        img_lwir = mmcv.imread(results['img_lwir'])
        results['img'] = img
        results['img_lwir'] = img_lwir
        results['img_fields'] = ['img', 'img_lwir']
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_id'] = 0
        import pdb
        pdb.set_trace()
        return results

def inference_detector_kaist(model, imgs, imgs_lwir):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """


    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    # test_pipeline = cfg.test_pipeline
    test_pipeline = [LoadImageKaist()] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    import pdb
    pdb.set_trace()
    # prepare data
    data = dict(img=imgs, img_lwir=imgs_lwir)
    data = test_pipeline(data)
    data = default_collate([data])

    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    # Use torchvision ops for CPU mode instead
    for m in model.modules():
        if isinstance(m, (RoIPool, RoIAlign)):
            if not m.aligned:
                # aligned=False is not implemented on CPU
                # set use_torchvision on-the-fly
                m.use_torchvision = True
    warnings.warn('We set use_torchvision=True in CPU mode.')
    # just get the actual data from DataContainer
    data['img_metas'] = data['img_metas'][0].data

        # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

# TODO: Awaiting refactoring
async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNDArray'

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    for m in model.modules():
        assert not isinstance(
            m,
            RoIPool), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(data, rescale=True)
    return results

def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

def write_result_txt_img(model, lwir_img_path, img, img_path, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    txt_path = os.path.join(img_path, 'det')
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    result_path = os.path.join(img_path, img+'.png')
    if hasattr(model, 'module'):
        model = model.module
    # lwir_img = model.show_result(lwir_img_path, result, score_thr=score_thr, show=False)
    result_shape = result[0][0][0].shape

    with open(os.path.join(txt_path, img+'.txt'), 'w') as txt:

        for i in range(0, result_shape[0]):

            bboxes, scores = result[0][0][0][i, :-1], result[0][0][0][i, -1]
            sx,sy,ex,ey = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
            if scores < score_thr:
                continue
            line= 'person ' + str(sx) +' '+str(sy)+' '+str(ex)+' '+str(ey)+' '+str(scores) + '\n'
            txt.write(line)
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
    # mmcv.imwrite( mmcv.bgr2rgb(lwir_img),result_path)
def build_test_pipeline(cfg: ConfigType) -> ConfigType:
    """Build test_pipeline for mot/vis demo. In mot/vis infer, original
    test_pipeline should remove the "LoadImageFromFile" and
    "LoadTrackAnnotations".

    Args:
         cfg (ConfigDict): The loaded config.
    Returns:
         ConfigType: new test_pipeline
    """
    # remove the "LoadImageFromFile" and "LoadTrackAnnotations" in pipeline
    transform_broadcaster = cfg.test_dataloader.dataset.pipeline[0].copy()
    for transform in transform_broadcaster['transforms']:
        if transform['type'] == 'Resize':
            transform_broadcaster['transforms'] = transform
    pack_track_inputs = cfg.test_dataloader.dataset.pipeline[-1].copy()
    test_pipeline = Compose([transform_broadcaster, pack_track_inputs])

    return test_pipeline


def inference_mot(model: nn.Module, img: np.ndarray, frame_id: int,
                  video_len: int) -> SampleList:
    """Inference image(s) with the mot model.

    Args:
        model (nn.Module): The loaded mot model.
        img (np.ndarray): Loaded image.
        frame_id (int): frame id.
        video_len (int): demo video length
    Returns:
        SampleList: The tracking data samples.
    """
    cfg = model.cfg
    data = dict(
        img=[img.astype(np.float32)],
        frame_id=[frame_id],
        ori_shape=[img.shape[:2]],
        img_id=[frame_id + 1],
        ori_video_length=[video_len])

    test_pipeline = build_test_pipeline(cfg)
    data = test_pipeline(data)

    if not next(model.parameters()).is_cuda:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        data = default_collate([data])
        result = model.test_step(data)[0]
    return result


def init_track_model(config: Union[str, Config],
                     checkpoint: Optional[str] = None,
                     detector: Optional[str] = None,
                     reid: Optional[str] = None,
                     device: str = 'cuda:0',
                     cfg_options: Optional[dict] = None) -> nn.Module:
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (Optional[str], optional): Checkpoint path. Defaults to
            None.
        detector (Optional[str], optional): Detector Checkpoint path, use in
            some tracking algorithms like sort.  Defaults to None.
        reid (Optional[str], optional): Reid checkpoint path. use in
            some tracking algorithms like sort. Defaults to None.
        device (str, optional): The device that the model inferences on.
            Defaults to `cuda:0`.
        cfg_options (Optional[dict], optional): Options to override some
            settings in the used config. Defaults to None.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # Weights converted from elsewhere may not have meta fields.
        checkpoint_meta = checkpoint.get('meta', {})
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            if 'CLASSES' in checkpoint_meta['dataset_meta']:
                value = checkpoint_meta['dataset_meta'].pop('CLASSES')
                checkpoint_meta['dataset_meta']['classes'] = value
            model.dataset_meta = checkpoint_meta['dataset_meta']

    if detector is not None:
        assert not (checkpoint and detector), \
            'Error: checkpoint and detector checkpoint cannot both exist'
        load_checkpoint(model.detector, detector, map_location='cpu')

    if reid is not None:
        assert not (checkpoint and reid), \
            'Error: checkpoint and reid checkpoint cannot both exist'
        load_checkpoint(model.reid, reid, map_location='cpu')

    # Some methods don't load checkpoints or checkpoints don't contain
    # 'dataset_meta'
    # VIS need dataset_meta, MOT don't need dataset_meta
    if not hasattr(model, 'dataset_meta'):
        warnings.warn('dataset_meta or class names are missed, '
                      'use None by default.')
        model.dataset_meta = {'classes': None}

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model
