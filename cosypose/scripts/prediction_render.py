#notebook cosypose/notebooks/visualize_singleview_predictions.ipynb as py-file
import os
import sys

sys.path.append('/home/bv-user/cosypose-ordner/cosypose')

site_pkgs='/vol/tiago/bildverarbeitung-cosypose/lib/python3.6/site-packages'
sys.path.append(site_pkgs)

import torch
from cosypose.config import LOCAL_DATA_DIR
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import make_singleview_prediction_plots, filter_predictions
from cosypose.visualization.singleview import filter_predictions
from bokeh.plotting import gridplot
from bokeh.io import show, export_png, output_notebook; output_notebook()
from pathlib import Path
 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# result_id = 'tless-siso-n_views=1--684390594'
# ds_name, urdf_ds_name = 'tless.primesense.test', 'tless.cad'
# pred_key = 'pix2pose_detections/refiner/iteration=4'

result_id = 'ycbv-n_views=1--5154971130'
ds_name, urdf_ds_name = 'ycbv.test.keyframes', 'ycbv'
pred_key = 'posecnn_init/refiner/iteration=2'


results = LOCAL_DATA_DIR / 'results' / result_id / 'results.pth.tar'
scene_ds = make_scene_dataset(ds_name, keyframes_path=Path('/home/bv-user/cosypose-ordner/keyframe.txt'))
results = torch.load(results)['predictions']
results[pred_key].infos.loc[:, ['scene_id', 'view_id']].groupby('scene_id').first()
# scene_id, view_id = 12, 103

# Replace this here, you can use the dataframe above to get examples of scene/view ids.
scene_id, view_id = 48, 733

this_preds = filter_predictions(results[pred_key], scene_id, view_id)
renderer = BulletSceneRenderer(urdf_ds_name)
figures = make_singleview_prediction_plots(scene_ds, renderer, this_preds)
renderer.disconnect()
# print(this_preds)
print(type(figures['pred_overlay']))

#not enough imports yet
#export_png(figures['pred_overlay'], 'pred1.png')

#only for jupyter notebooks
# show(figures['input_im'])
# show(figures['pred_overlay'])