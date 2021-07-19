import numpy as np

from .plotter import Plotter

from cosypose.datasets.wrappers.augmentation_wrapper import AugmentationWrapper
from cosypose.datasets.augmentations import CropResizeToAspectAugmentation


def filter_predictions(preds, scene_id, view_id=None, th=None):
    mask = preds.infos['scene_id'] == scene_id
    if view_id is not None:
        mask = np.logical_and(mask, preds.infos['view_id'] == view_id)
    if th is not None:
        mask = np.logical_and(mask, preds.infos['score'] >= th)
    keep_ids = np.where(mask)[0]
    preds = preds[keep_ids]
    return preds


def render_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480)):
    pred = pred.cpu()
    camera.update(TWC=np.eye(4))

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
        )
        list_objects.append(obj)
    rgb_rendered = renderer.render_scene(list_objects, [camera])[0]['rgb']
    return rgb_rendered

def make_my_plots(renderer, predictions, rgb_input, detections=None):
    
    plotter = Plotter()

    figures = dict()

    figures['input_im'] = plotter.plot_image(rgb_input)

    camera_state = dict()
    camera_state['T0C']=np.array([[ 0.398836  ,  0.267093  , -0.877263  ,  0.76227163],
       [ 0.916524  , -0.0845743 ,  0.390936  , -0.40550313],
       [ 0.0302225 , -0.959952  , -0.278529  ,  0.35137333],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    camera_state['K']=np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
       [0.000000e+00, 1.067487e+03, 2.413109e+02],
       [0.000000e+00, 0.000000e+00, 1.000000e+00]])
       
    camera_state['TWC']=np.array([[ 0.398836  ,  0.267093  , -0.877263  ,  0.76227163],
       [ 0.916524  , -0.0845743 ,  0.390936  , -0.40550313],
       [ 0.0302225 , -0.959952  , -0.278529  ,  0.35137333],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    camera_state['resolution']=rgb_input.size()[0:2]


    pred_rendered = render_prediction_wrt_camera(renderer, predictions, camera=camera_state)
    figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    figures['pred_overlay'] = plotter.plot_overlay(rgb_input, pred_rendered)

    if detections is not None:
        fig_dets = plotter.plot_image(rgb_input)
        fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections, text_auto=False)
        figures['detections'] = fig_dets
    return figures



def make_singleview_prediction_plots(scene_ds, renderer, predictions, detections=None, resolution=(640, 480), scene_id=None, view_id=None):
    plotter = Plotter()

    if scene_id is None or view_id is None:
        scene_id, view_id = np.unique(predictions.infos['scene_id']).item(), np.unique(predictions.infos['view_id']).item()

    scene_ds_index = scene_ds.frame_index
    scene_ds_index['ds_idx'] = np.arange(len(scene_ds_index))
    scene_ds_index = scene_ds_index.set_index(['scene_id', 'view_id'])
    idx = scene_ds_index.loc[(scene_id, view_id), 'ds_idx']

    augmentation = CropResizeToAspectAugmentation(resize=resolution)
    scene_ds = AugmentationWrapper(scene_ds, augmentation)
    rgb_input, mask, state = scene_ds[idx]
    figures = dict()

    figures['input_im'] = plotter.plot_image(rgb_input)

    if detections is not None:
        fig_dets = plotter.plot_image(rgb_input)
        fig_dets = plotter.plot_maskrcnn_bboxes(fig_dets, detections, text_auto=False)
        figures['detections'] = fig_dets

    pred_rendered = render_prediction_wrt_camera(renderer, predictions, camera=state['camera'])
    figures['pred_rendered'] = plotter.plot_image(pred_rendered)
    figures['pred_overlay'] = plotter.plot_overlay(rgb_input, pred_rendered)
    return figures
