import paths
import functions
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('DL'))

# functions.save_roi_images(paths.raw_annotations,
#                           paths.original_images,
#                           paths.roi_images)
#
# functions.save_gt_spike_annotations(paths.raw_annotations,
#                                     paths.spikes_annotation,
#                                     paths.point_scale)


ga_out = functions.ga()
np.save(os.path.join(paths.results_path, 'ga_out_position.npy'), ga_out.bestsol.position, allow_pickle=True)
np.save(os.path.join(paths.results_path, 'ga_out_bestcost.npy'), ga_out.bestcost, allow_pickle=True)

print('end')