import json
import numpy as np
import pandas as pd
import cv2
import os
import shutil
import albumentations as A
from ypstruct import structure
import paths
import matplotlib.pyplot as plt
import shutil

#! This is 
#? This is 
#// This is 
#todo This is 
#* This is 
# print(json.dumps(data, indent = 4, sort_keys=True))

#######################################################################################################################

def crop(image):
    """
    Crop the image to its non-zero amounts
    """
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

#######################################################################################################################

def roi_creator(data_json, region, data_image, name_image):
    """
    Create roi image with transparent background from the annotations
    """
    x = np.array([data_json[name_image]['regions'][region]['shape_attributes']['all_points_x']]).T
    y = np.array([data_json[name_image]['regions'][region]['shape_attributes']['all_points_y']]).T
    data_points = np.hstack((x, y)).astype(np.int32)
    data_mask = np.zeros((data_image.shape[0], data_image.shape[1]))
    cv2.fillConvexPoly(data_mask, data_points, 1)
    data_mask = data_mask.astype(bool)
    data_roi = np.zeros_like(data_image)
    data_roi[data_mask] = data_image[data_mask]
    data_roi = crop(data_roi)
    _,alpha = cv2.threshold(cv2.cvtColor(data_roi, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(data_roi) 
    data_roi = cv2.merge([b,g,r, alpha],4)
    return data_roi

#######################################################################################################################

def name_roi_creator(label, counter, name): 
    """
    Create the roi name according to its label, image name, and counter
    """
    name_roi = label + '_' + name + '_' + str(counter) + '.png'
    return name_roi

#######################################################################################################################

def save_roi_images(raw_annotations_path, original_images_path, roi_images_path):
    """
    Save the created roi images
    """
    #! Get the list of annotation names
    names = [fn[0:-5] for fn in os.listdir(raw_annotations_path) if any(fn.endswith(ext) for ext in ['json'])]

    #! Create the roi_images folder and remove that if exists
    if os.path.exists(roi_images_path):
        shutil.rmtree(roi_images_path)
    os.mkdir(roi_images_path)

    #! Loop for extracting informaiton from each image
    for name in names:
        #! Preparing the names
        name_json = name + '.json'
        name_image = name + '.png'

        #! Loading the image and annotations
        data_image = cv2.imread(os.path.join(original_images_path, name_image))
        data_json = json.load(open(os.path.join(raw_annotations_path, name_json)))

        #! Loop over all of the objects
        spike_count, ground_count, stem_count, leaf_count = 0, 0, 0, 0
        regions = list(data_json[name_image]['regions'].keys())
        for region in regions:
            label = data_json[name_image]['regions'][region]['region_attributes']['label']
            data_roi = roi_creator(data_json, region, data_image, name_image)

            #! Correct the saving names
            if label == 'spike':
                name_roi = name_roi_creator(label, spike_count, name)
                spike_count += 1
            elif label == 'ground':
                name_roi = name_roi_creator(label, ground_count, name)
                ground_count += 1
            elif label == 'leaf':
                name_roi = name_roi_creator(label, leaf_count, name)
                leaf_count += 1
            elif label == 'stem':
                name_roi = name_roi_creator(label, stem_count, name)
                stem_count += 1

            #! Save the roi images
            cv2.imwrite(os.path.join(roi_images_path, name_roi), data_roi, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

#######################################################################################################################

def save_gt_spike_annotations(raw_annotations_path, spikes_annotation_path, point_scale):
    # ! Get the list of annotation names
    names = [fn[0:-4] for fn in os.listdir(raw_annotations_path) if any(fn.endswith(ext) for ext in ['csv'])]

    # ! Create the spikes_annotation folder and remove that if exists
    if os.path.exists(spikes_annotation_path):
        shutil.rmtree(spikes_annotation_path)
    os.mkdir(spikes_annotation_path)

    # ! Loop over the csv files
    for name in names:

        # ! Preaparing the names
        name_csv = name + '.csv'
        data_csv = pd.read_csv(os.path.join(raw_annotations_path, name_csv), header=None)

        # ! Loop over the spike images
        for i in range(len(data_csv)):
            # ! Generate the ground truth image
            spike_sample = data_csv.loc[i, :]
            image_width = spike_sample.loc[4]
            image_height = spike_sample.loc[5]
            point_x = spike_sample.loc[1]
            point_y = spike_sample.loc[2]
            data_spike_gt = np.zeros((image_height, image_width), dtype=np.uint8)
            data_spike_gt[point_y, point_x] = point_scale
            data_spike_gt[point_y - 1, point_x] = point_scale
            data_spike_gt[point_y + 1, point_x] = point_scale
            data_spike_gt[point_y, point_x - 1] = point_scale
            data_spike_gt[point_y, point_x + 1] = point_scale
            data_spike_gt[point_y - 1, point_x - 1] = point_scale
            data_spike_gt[point_y + 1, point_x + 1] = point_scale
            data_spike_gt[point_y - 1, point_x + 1] = point_scale
            data_spike_gt[point_y + 1, point_x - 1] = point_scale
            data_spike_gt[point_y - 2, point_x] = point_scale
            data_spike_gt[point_y + 2, point_x] = point_scale
            data_spike_gt[point_y, point_x - 2] = point_scale
            data_spike_gt[point_y, point_x + 2] = point_scale
            data_spike_gt[point_y - 2, point_x - 2] = point_scale
            data_spike_gt[point_y + 2, point_x + 2] = point_scale
            data_spike_gt[point_y - 2, point_x + 2] = point_scale
            data_spike_gt[point_y + 2, point_x - 2] = point_scale
            data_spike_gt[point_y + 2, point_x - 1] = point_scale
            data_spike_gt[point_y + 2, point_x + 1] = point_scale
            data_spike_gt[point_y - 2, point_x - 1] = point_scale
            data_spike_gt[point_y - 2, point_x + 1] = point_scale
            data_spike_gt[point_y - 1, point_x + 2] = point_scale
            data_spike_gt[point_y + 1, point_x + 2] = point_scale
            data_spike_gt[point_y - 1, point_x - 2] = point_scale
            data_spike_gt[point_y + 1, point_x - 2] = point_scale

            # ! Generate the name for ground truth image
            name_spike = spike_sample.loc[3]
            name_spike_gt = name_spike[:-4] + "_" + "gt" + ".png"

            # ! Save the ground truth images
            cv2.imwrite(os.path.join(spikes_annotation_path, name_spike_gt), data_spike_gt,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

            """
            #! In order to read that without change
            read_data_spike_gt = cv2.imread(os.path.join('annotations_spikes', name_spike_gt), 
                                            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            """

#######################################################################################################################

def denormalize(min_val, max_val, magnitude):
    magnitude = magnitude * (max_val - min_val) + min_val
    return magnitude

#######################################################################################################################

def determine_elements_no(min_val, max_val, magnitude):
    magnitude = int(round(denormalize(min_val, max_val, magnitude)))
    return magnitude

#######################################################################################################################

def ga_cost_func(x):

    generate_augmented_images(x, 5, paths.roi_images, False, 100)

    curdir = os.getcwd()
    os.chdir ('DL')
    import train
    cost = train.train()
    os.chdir(curdir)
    return cost

#######################################################################################################################

def crossover(p1, p2, gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma, 1 + gamma, *c1.position.shape)
    c1.position = alpha * p1.position + (1 - alpha) * p2.position
    c2.position = alpha * p2.position + (1 - alpha) * p1.position
    return c1, c2

#######################################################################################################################

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma * np.random.randn(*ind.shape)
    return y

#######################################################################################################################

def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

#######################################################################################################################

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

#######################################################################################################################

def run(problem, params):
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc * npop / 2) * 2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iterations
    bestcost = np.empty(maxit)

    # Main Loop
    for it in range(maxit):
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(-beta * costs)

        popc = []
        for _ in range(nc // 2):

            # Select Parents
            # q = np.random.permutation(npop)
            # p1 = pop[q[0]]
            # p2 = pop[q[1]]

            # Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bounds
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)

            # Evaluate First Offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

#######################################################################################################################

def ga(nvar=24, varmin=0, varmax=1, maxit=50, npop=4, beta=1, pc=1, gamma=0.1, mu=0.01, sigma=0.1):
    # Problem Definition
    problem = structure()
    problem.costfunc = ga_cost_func
    problem.nvar = nvar
    problem.varmin = varmin
    problem.varmax = varmax

    # GA Parameters
    params = structure()
    params.maxit = maxit
    params.npop = npop
    params.beta = beta
    params.pc = pc
    params.gamma = gamma
    params.mu = mu
    params.sigma = sigma

    # Run GA
    out = run(problem, params)

    return out

# # Results
# # plt.plot(out.bestcost)
# plt.semilogy(out.bestcost)
# plt.xlim(0, params.maxit)
# plt.xlabel('Iterations')
# plt.ylabel('Best Cost')
# plt.title('Genetic Algorithm (GA)')
# plt.grid(True)
# plt.show()

#######################################################################################################################

#! Add padding
def add_padding(rgb, trans, gt, spike):
    width_old = trans.shape[1]
    height_old = trans.shape[0]
    #! Create new images of desired size for padding
    width_new = width_old * 2
    height_new = height_old * 2
    rgb_out = np.full((height_new, width_new, 3), (0, 0, 0), dtype=np.uint8)
    trans_out = np.zeros((height_new, width_new), dtype=np.uint8)
    #! Compute the center offset
    x_center = (width_new - width_old) // 2
    y_center = (height_new - height_old) // 2
    #! Copy the images into the center of result image
    rgb_out[y_center:y_center+height_old, x_center:x_center+width_old] = rgb
    trans_out[y_center:y_center+height_old, x_center:x_center+width_old] = trans
    if spike:
        gt_out = np.zeros((height_new, width_new), dtype=np.uint8)
        gt_out[y_center:y_center+height_old, x_center:x_center+width_old] = gt

    #! Return
    if spike:
        return rgb_out, trans_out, gt_out
    else:
        return rgb_out, trans_out

#######################################################################################################################

#! Get a 4d image and return a 3d RGB and 1d Transparent images
def generate_image_mask(image):
    imageBgr = image[:, :, 0:3]
    imageRgb = cv2.cvtColor(imageBgr, cv2.COLOR_BGR2RGB)
    imageTrans = image[:, :, 3]
    return imageRgb, imageTrans

#######################################################################################################################

def augmentation_transformer_2aug(name_stem_roi,
                             image_trans,
                             aug_pipeline,
                             spike=True,
                             visualization=True):

    rgb, trans = generate_image_mask(image_trans)

    if spike:
        gt_name = name_stem_roi[:-4] + '_gt.png'
        gt = cv2.imread(os.path.join(paths.spikes_annotation, gt_name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        rgb, trans, gt = add_padding(rgb, trans, gt, spike)
        masks = [trans, gt]
    else:
        gt = None
        rgb, trans = add_padding(rgb, trans, gt, spike)
        masks = [trans]

    if visualization: visualize_augmented('BEFORE_AUG_', 200, rgb, trans, gt, spike)

    if aug_pipeline.do_aug:
        augmented_rgb_trans = aug_pipeline.transform_1(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        masks = [trans]
        if spike:
            gt = augmented_rgb_trans['masks'][1]
            masks = [trans, gt]

        if visualization: visualize_augmented('AFTER_1st_AUG_', 200, rgb, trans, gt, spike)

        augmented_rgb_trans = aug_pipeline.transform_2(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        if spike:
            gt = augmented_rgb_trans['masks'][1]

        if visualization: visualize_augmented('AFTER_2nd_AUG_', 200, rgb, trans, gt, spike)

        if visualization:
            plt.pause(3)
            plt.close("all")

    image = cv2.merge([rgb, trans])

    if spike:
        return image, gt
    if not spike:
        return image

#######################################################################################################################

def augmentation_transformer_5aug(name_stem_roi,
                             image_trans,
                             aug_pipeline,
                             spike=True,
                             visualization=True):

    rgb, trans = generate_image_mask(image_trans)

    if spike:
        gt_name = name_stem_roi[:-4] + '_gt.png'
        gt = cv2.imread(os.path.join(paths.spikes_annotation, gt_name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        rgb, trans, gt = add_padding(rgb, trans, gt, spike)
        masks = [trans, gt]
    else:
        gt = None
        rgb, trans = add_padding(rgb, trans, gt, spike)
        masks = [trans]

    if visualization: visualize_augmented('BEFORE_AUG_', 200, rgb, trans, gt, spike)

    if aug_pipeline.do_aug:
        augmented_rgb_trans = aug_pipeline.transform_1(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        masks = [trans]
        if spike:
            gt = augmented_rgb_trans['masks'][1]
            masks = [trans, gt]

        augmented_rgb_trans = aug_pipeline.transform_2(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        masks = [trans]
        if spike:
            gt = augmented_rgb_trans['masks'][1]
            masks = [trans, gt]

        augmented_rgb_trans = aug_pipeline.transform_3(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        masks = [trans]
        if spike:
            gt = augmented_rgb_trans['masks'][1]
            masks = [trans, gt]

        augmented_rgb_trans = aug_pipeline.transform_4(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        masks = [trans]
        if spike:
            gt = augmented_rgb_trans['masks'][1]
            masks = [trans, gt]

        augmented_rgb_trans = aug_pipeline.transform_5(image=rgb, masks=masks)
        rgb = augmented_rgb_trans['image']
        trans = augmented_rgb_trans['masks'][0]
        if spike:
            gt = augmented_rgb_trans['masks'][1]

        if visualization:
            plt.pause(3)
            plt.close("all")

    image = cv2.merge([rgb, trans])

    if spike:
        return image, gt
    if not spike:
        return image

#######################################################################################################################

def visualize_augmented(name, location, rgb, trans, gt=None, spike=True):

    if spike:
        fig, axs = plt.subplots(3, sharex=True, sharey=True)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(location,100,640, 545)
        # fig.set_size_inches(7, 12)
        name = name + 'SPIKE'
        fig.suptitle(name)
        axs[0].imshow(rgb)
        axs[1].imshow(trans)
        axs[2].imshow(gt)

    elif not spike:
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(location,100,640, 545)
        # fig.set_size_inches(7, 12)
        name = name + 'OTHERS'
        fig.suptitle(name)
        axs[0].imshow(rgb)
        axs[1].imshow(trans)

#######################################################################################################################

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    h = overlay.shape[0]
    w = overlay.shape[1]

    flag_x = True
    flag_y = True

    if x + int(w / 2) > background_width:
        temp = w - ((x + int(w / 2)) - background_width)
        overlay = overlay[:, :temp]
        w = overlay.shape[1]
        x = background_width - int(w / 2)

    if x - int(w / 2) < 0:
        flag_x = False
        temp = w - (0 - (x - int(w / 2)))
        overlay = overlay[:, temp:]
        w = overlay.shape[1]
        x = 0 + int(w / 2)

    if y + int(h / 2) > background_height:
        temp = h - ((y + int(h / 2)) - background_height)
        overlay = overlay[:temp]
        h = overlay.shape[0]
        y = background_width - int(h / 2)

    if y - int(h / 2) < 0:
        flag_y = False
        temp = h - (0 - (y - int(h / 2)))
        overlay = overlay[temp:]
        h = overlay.shape[0]
        y = 0 + int(h / 2)

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    x_min = x - int(w / 2)
    x_max = x + int(w / 2)
    y_min = y - int(h / 2)
    y_max = y + int(h / 2)

    if w % 2 == 1 and flag_x == True:
        x_min = x - int(w / 2) - 1
    if h % 2 == 1 and flag_y == True:
        y_min = y - int(h / 2) - 1
    if w % 2 == 1 and flag_x == False:
        x_max = x + int(w / 2) + 1
    if h % 2 == 1 and flag_y == False:
        y_max = y + int(h / 2) + 1

    if x_min < 0:
        x_min = 0
        x_max = x_max + 1
    if y_min < 0:
        y_min = 0
        y_max = y_max + 1

    background[y_min:y_max, x_min:x_max] = (1.0 - mask) * background[y_min:y_max, x_min:x_max] + mask * overlay_image

    return background

#######################################################################################################################

def spike_overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

#######################################################################################################################

#! Find the coordinates of spikes gt on the image
def find_gt_coordinates(spike_gt):
    points = np.argwhere(spike_gt == 100)
    y = int(points[:, 0].mean())
    x = int(points[:, 1].mean())
    spike_gt = np.zeros(spike_gt.shape).astype(np.uint8)
    spike_gt[y, x] = 100
    return x, y

#######################################################################################################################

#! Visualize the transformed image, mask, and gt
# def visualize(transformed_image, transformed_masks):
#     fig, axs = plt.subplots(3, sharex=True, sharey=True)
#     fig.set_size_inches(10, 18)
#     axs[0].imshow(transformed_image)
#     axs[1].imshow(transformed_masks[0])
#     axs[2].imshow(transformed_masks[1])
#     plt.show(block=False)
#     plt.pause(1)
#     plt.close('all')

#######################################################################################################################

def parse_opt_out_2aug(opt_out):
    opt_out_parsed = structure()

    opt_out_parsed.spikes_no = determine_elements_no(30, 60, opt_out[0])
    opt_out_parsed.leaves_no = determine_elements_no(30, 60, opt_out[1])
    opt_out_parsed.stems_no = determine_elements_no(30, 60, opt_out[2])
    opt_out_parsed.grounds_no = determine_elements_no(150, 250, opt_out[3])

    opt_out_parsed.c1t1t = determine_aug_type(opt_out[4])
    opt_out_parsed.c1t2t = determine_aug_type(opt_out[5])
    opt_out_parsed.c2t1t = determine_aug_type(opt_out[6])
    opt_out_parsed.c2t2t = determine_aug_type(opt_out[7])
    opt_out_parsed.c3t1t = determine_aug_type(opt_out[8])
    opt_out_parsed.c3t2t = determine_aug_type(opt_out[9])
    opt_out_parsed.c4t1t = determine_aug_type(opt_out[10])
    opt_out_parsed.c4t2t = determine_aug_type(opt_out[11])
    opt_out_parsed.c5t1t = determine_aug_type(opt_out[12])
    opt_out_parsed.c5t2t = determine_aug_type(opt_out[13])

    opt_out_parsed.c1t1m = opt_out[14]
    opt_out_parsed.c1t2m = opt_out[15]
    opt_out_parsed.c2t1m = opt_out[16]
    opt_out_parsed.c2t2m = opt_out[17]
    opt_out_parsed.c3t1m = opt_out[18]
    opt_out_parsed.c3t2m = opt_out[19]
    opt_out_parsed.c4t1m = opt_out[20]
    opt_out_parsed.c4t2m = opt_out[21]
    opt_out_parsed.c5t1m = opt_out[22]
    opt_out_parsed.c5t2m = opt_out[23]

    return opt_out_parsed

#######################################################################################################################

def parse_opt_out_5aug(opt_out):
    opt_out_parsed = structure()

    opt_out_parsed.spikes_no = determine_elements_no(30, 60, opt_out[0])
    opt_out_parsed.leaves_no = determine_elements_no(30, 60, opt_out[1])
    opt_out_parsed.stems_no = determine_elements_no(30, 60, opt_out[2])
    opt_out_parsed.grounds_no = determine_elements_no(150, 250, opt_out[3])

    opt_out_parsed.c1t1t = determine_aug_type(opt_out[4])
    opt_out_parsed.c1t2t = determine_aug_type(opt_out[5])
    opt_out_parsed.c1t3t = determine_aug_type(opt_out[6])
    opt_out_parsed.c1t4t = determine_aug_type(opt_out[7])
    opt_out_parsed.c1t5t = determine_aug_type(opt_out[8])
    opt_out_parsed.c2t1t = determine_aug_type(opt_out[9])
    opt_out_parsed.c2t2t = determine_aug_type(opt_out[10])
    opt_out_parsed.c2t3t = determine_aug_type(opt_out[11])
    opt_out_parsed.c2t4t = determine_aug_type(opt_out[12])
    opt_out_parsed.c2t5t = determine_aug_type(opt_out[13])

    opt_out_parsed.c1t1m = opt_out[14]
    opt_out_parsed.c1t2m = opt_out[15]
    opt_out_parsed.c1t3m = opt_out[16]
    opt_out_parsed.c1t4m = opt_out[17]
    opt_out_parsed.c1t5m = opt_out[18]
    opt_out_parsed.c2t1m = opt_out[19]
    opt_out_parsed.c2t2m = opt_out[20]
    opt_out_parsed.c2t3m = opt_out[21]
    opt_out_parsed.c2t4m = opt_out[22]
    opt_out_parsed.c2t5m = opt_out[23]

    return opt_out_parsed

#######################################################################################################################

def determine_aug_pipeline_2aug(opt_out_parsed):
    probs = np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.2])
    aug_pipeline_no = roulette_wheel_selection(probs) + 1

    aug_pipeline = structure()

    if aug_pipeline_no == 6:
        aug_pipeline.do_aug = False

    else:
        aug_pipeline.do_aug = True
        exec('aug_pipeline.aug_type_1 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't1t')
        exec('aug_pipeline.aug_type_2 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't2t')
        exec('aug_pipeline.aug_magnitude_1 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't1m')
        exec('aug_pipeline.aug_magnitude_2 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't2m')
        aug_pipeline.transform_1 = determine_transformation(aug_pipeline.aug_type_1, aug_pipeline.aug_magnitude_1)
        aug_pipeline.transform_2 = determine_transformation(aug_pipeline.aug_type_2, aug_pipeline.aug_magnitude_2)

    return aug_pipeline

#######################################################################################################################

def determine_aug_pipeline_5aug(opt_out_parsed):
    probs = np.array([0.3, 0.3, 0.4])
    aug_pipeline_no = roulette_wheel_selection(probs) + 1

    aug_pipeline = structure()

    if aug_pipeline_no == 3:
        aug_pipeline.do_aug = False

    else:
        aug_pipeline.do_aug = True
        exec('aug_pipeline.aug_type_1 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't1t')
        exec('aug_pipeline.aug_type_2 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't2t')
        exec('aug_pipeline.aug_type_3 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't3t')
        exec('aug_pipeline.aug_type_4 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't4t')
        exec('aug_pipeline.aug_type_5 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't5t')

        exec('aug_pipeline.aug_magnitude_1 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't1m')
        exec('aug_pipeline.aug_magnitude_2 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't2m')
        exec('aug_pipeline.aug_magnitude_3 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't3m')
        exec('aug_pipeline.aug_magnitude_4 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't4m')
        exec('aug_pipeline.aug_magnitude_5 =' + 'opt_out_parsed.c' + str(aug_pipeline_no) + 't5m')

        aug_pipeline.transform_1 = determine_transformation (aug_pipeline.aug_type_1, aug_pipeline.aug_magnitude_1)
        aug_pipeline.transform_2 = determine_transformation (aug_pipeline.aug_type_2, aug_pipeline.aug_magnitude_2)
        aug_pipeline.transform_3 = determine_transformation (aug_pipeline.aug_type_3, aug_pipeline.aug_magnitude_3)
        aug_pipeline.transform_4 = determine_transformation (aug_pipeline.aug_type_4, aug_pipeline.aug_magnitude_4)
        aug_pipeline.transform_5 = determine_transformation (aug_pipeline.aug_type_5, aug_pipeline.aug_magnitude_5)

    return aug_pipeline

#######################################################################################################################

def determine_aug_type(magnitude):
    min_val, max_val = 0, 20
    magnitude = denormalize(min_val, max_val, magnitude)

    if False:
        pass
    elif 0 <= magnitude < 1:
        aug_type = "Blur"
    elif 1 <= magnitude < 2:
        aug_type = "MedianBlur"
    elif 2 <= magnitude < 3:
        aug_type = "ISONoise"
    elif 3 <= magnitude < 4:
        aug_type = "CLAHE"
    elif 4 <= magnitude < 5:
        aug_type = "Downscale"
    elif 5 <= magnitude < 6:
        aug_type = "Emboss"
    elif 6 <= magnitude < 7:
        aug_type = "RGBShift"
    elif 7 <= magnitude < 8:
        aug_type = "HueSaturationValue"
    elif 8 <= magnitude < 9:
        aug_type = "MultiplicativeNoise"
    elif 9 <= magnitude < 10:
        aug_type = "RandomBrightnessContrast"
    elif 10 <= magnitude < 11:
        aug_type = "RandomToneCurve"
    elif 11 <= magnitude < 12:
        aug_type = "CoarseDropout"
    elif 12 <= magnitude < 13:
        aug_type = "GridDistortion"
    elif 13 <= magnitude < 14:
        aug_type = "ElasticTransform"
    elif 14 <= magnitude < 15:
        aug_type = "Perspective"
    elif 15 <= magnitude < 16:
        aug_type = "PiecewiseAffine"
    elif 16 <= magnitude < 17:
        aug_type = "Flip"
    elif 17 <= magnitude < 18:
        aug_type = "AffineRotate"
    elif 18 <= magnitude < 19:
        aug_type = "AffineScale"
    elif 19 <= magnitude <= 20:
        aug_type = "AffineShear"

    return aug_type

#######################################################################################################################

def determine_transformation(aug_type, magnitude):
    if False:
        pass

    elif aug_type == "Blur":
        min_val, max_val = 3, 7
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.Blur(blur_limit=[magnitude, max_val],
                           p=1)

    elif aug_type == "MedianBlur":
        min_val, max_val = 3, 7
        magnitude = denormalize(min_val, max_val, magnitude)
        magnitude = round(magnitude)
        if magnitude % 2 == 0: magnitude += 1
        transform = A.MedianBlur(blur_limit=[magnitude, max_val],
                                 p=1)

    elif aug_type == "ISONoise":
        min_val, max_val = 0.01, 0.2
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.ISONoise(color_shift=(0.01, 1),
                               intensity=(magnitude, max_val),
                               p=1)

    elif aug_type == "CLAHE":
        min_val, max_val = 0, 100
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.CLAHE(clip_limit=magnitude,
                            tile_grid_size=(8, 8),
                            p=1)

    elif aug_type == "Downscale":
        min_val, max_val = 0.7, 0.99
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.Downscale(scale_min=magnitude,
                                scale_max=max_val,
                                p=1)

    elif aug_type == "Emboss":
        max_val = 1
        transform = A.Emboss(alpha=(magnitude, max_val),
                             strength=(0.2, 0.7),
                             p=1)

    elif aug_type == "RGBShift":
        min_val, max_val = 10, 40
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.RGBShift(r_shift_limit=magnitude,
                               g_shift_limit=magnitude,
                               b_shift_limit=magnitude,
                               p=1)

    elif aug_type == "HueSaturationValue":
        min_val, max_val = 10, 40
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.HueSaturationValue(hue_shift_limit=magnitude,
                                         sat_shift_limit=magnitude,
                                         val_shift_limit=magnitude,
                                         p=1)

    elif aug_type == "MultiplicativeNoise":
        min_val, half_range, average = 0, 0.4, 1
        magnitude = denormalize(min_val, half_range, magnitude)
        transform = A.MultiplicativeNoise(multiplier=(average-magnitude, average+magnitude),
                                          per_channel=True,
                                          elementwise=False,
                                          p=1)

    elif aug_type == "RandomBrightnessContrast":
        min_val, max_val = 0, 0.4
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.RandomBrightnessContrast(brightness_limit=magnitude,
                                               contrast_limit=magnitude,
                                               brightness_by_max=True,
                                               p=1)

    elif aug_type == "RandomToneCurve":
        min_val, max_val = 0, 0.5
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.RandomToneCurve(scale=magnitude,
                                      p=1)

    elif aug_type == "CoarseDropout":
        min_val, max_val = 4, 30
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.CoarseDropout(max_holes=int(round(magnitude)),
                                    max_height=int(round(magnitude/4)),
                                    max_width=int(round(magnitude/4)),
                                    fill_value=0,
                                    p=1)

    elif aug_type == "GridDistortion":
        min_val, half_range, average = 0, 0.4, 0
        magnitude = denormalize(min_val, half_range, magnitude)
        transform = A.GridDistortion(num_steps=5,
                                     distort_limit=(average-magnitude, average+magnitude),
                                     interpolation=1,
                                     border_mode=1,
                                     p=1)

    elif aug_type == "ElasticTransform":
        min_val, max_val = 0, 10
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.ElasticTransform(alpha=magnitude,
                                       sigma=1,
                                       alpha_affine=1,
                                       interpolation=1,
                                       border_mode=4,
                                       value=None,
                                       mask_value=None,
                                       approximate=False,
                                       same_dxdy=False,
                                       p=1)

    elif aug_type == "Perspective":
        min_val, max_val = 0, 0.2
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.Perspective(scale=(min_val, magnitude),
                                  keep_size=True,
                                  pad_mode=0,
                                  pad_val=0,
                                  mask_pad_val=0,
                                  fit_output=True,
                                  interpolation=1,
                                  p=1)

    elif aug_type == "PiecewiseAffine":
        min_val, max_val = 0.01, 0.05
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.PiecewiseAffine(scale=(magnitude, max_val),
                                      nb_rows=4,
                                      nb_cols=4,
                                      interpolation=1,
                                      mask_interpolation=0,
                                      cval=0,
                                      cval_mask=0,
                                      mode='constant',
                                      absolute_scale=False,
                                      keypoints_threshold=0.01,
                                      p=1)

    elif aug_type == "Flip":
        # magnitude is not used for this transformation
        transform = A.Flip(p=1)

    elif aug_type == "AffineRotate":
        min_val, half_range, average = 0, 90, 0
        magnitude = denormalize(min_val, half_range, magnitude)
        transform = A.Affine(scale=(1, 1),
                             translate_percent=None,
                             translate_px=None,
                             rotate=(average-magnitude, average+magnitude),
                             shear=(0, 0),
                             interpolation=1,
                             mask_interpolation=0,
                             cval=0,
                             cval_mask=0,
                             mode=0,
                             fit_output=False,
                             p=1)

    elif aug_type == "AffineScale":
        min_val, max_val = 0.5, 1.2
        magnitude = denormalize(min_val, max_val, magnitude)
        transform = A.Affine(scale=(magnitude, max_val),
                             translate_percent=None,
                             translate_px=None,
                             rotate=(0, 0),
                             shear=(0, 0),
                             interpolation=1,
                             mask_interpolation=0,
                             cval=0,
                             cval_mask=0,
                             mode=0,
                             fit_output=False,
                             p=1)

    elif aug_type == "AffineShear":
        min_val, half_range, average = 0, 30, 0
        magnitude = denormalize(min_val, half_range, magnitude)
        transform = A.Affine(scale=(1, 1),
                             translate_percent=None,
                             translate_px=None,
                             rotate=(0, 0),
                             shear=(average-magnitude, average+magnitude),
                             interpolation=1,
                             mask_interpolation=0,
                             cval=0,
                             cval_mask=0,
                             mode=0,
                             fit_output=False,
                             p=1)

    # image = transform(image=image)['image']
    return transform

#######################################################################################################################

def generate_augmented_images(opt_output, sample_number, roi_images_path, visualize_sample, point_scale):
    opt_out_parsed = parse_opt_out_2aug(opt_output)

    name_total_roi = os.listdir(paths.roi_images)
    name_spike_rois = [string for string in name_total_roi if 'spike' in string]
    name_ground_rois = [string for string in name_total_roi if 'ground' in string]
    name_stem_rois = [string for string in name_total_roi if 'stem' in string]
    name_leaf_rois = [string for string in name_total_roi if 'leaf' in string]

    # temoprary path for the generated images
    augmented_images = paths.augmented_images_path
    if os.path.exists(augmented_images):
        shutil.rmtree(augmented_images)
    os.mkdir(augmented_images)
    os.mkdir(os.path.join(augmented_images, 'images'))
    os.mkdir(os.path.join(augmented_images, 'actual_labels'))

    for sample in range(sample_number):
        sample_image = np.zeros((1024, 1024, 3)).astype(np.uint8)
        sample_gt = np.zeros((1024, 1024)).astype(np.uint8)

        # ! Create the ground in the samples
        for i in range(opt_out_parsed.grounds_no):
            name_roi = name_ground_rois[np.random.randint(0, len(name_ground_rois))]
            image_trans = cv2.imread(os.path.join(roi_images_path, name_roi),
                                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

            aug_pipeline = determine_aug_pipeline_2aug(opt_out_parsed)
            roi = augmentation_transformer_2aug(name_roi,
                                                image_trans,
                                                aug_pipeline,
                                                spike=False,
                                                visualization=False)
            x, y = np.random.randint(0, 1024), np.random.randint(0, 1024)
            sample_image = overlay_transparent(sample_image, roi, x, y)

        for i in range(sample_image.shape[2]):
            temp = sample_image[:, :, i]
            temp_mean = int(temp[temp != 0].mean())
            temp[temp == 0] = temp_mean
            sample_image[:, :, i] = temp

        # ! Create the stem in the samples
        for i in range(opt_out_parsed.stems_no):
            name_roi = name_stem_rois[np.random.randint(0, len(name_stem_rois))]
            image_trans = cv2.imread(os.path.join(roi_images_path, name_roi),
                                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

            aug_pipeline = determine_aug_pipeline_2aug(opt_out_parsed)
            roi = augmentation_transformer_2aug(name_roi,
                                                image_trans,
                                                aug_pipeline,
                                                spike=False,
                                                visualization=False)
            x, y = np.random.randint(0, 1024), np.random.randint(0, 1024)
            sample_image = overlay_transparent(sample_image, roi, x, y)

        # ! Create the leaf in the samples
        for i in range(opt_out_parsed.leaves_no):
            name_roi = name_leaf_rois[np.random.randint(0, len(name_leaf_rois))]
            image_trans = cv2.imread(os.path.join(roi_images_path, name_roi),
                                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

            aug_pipeline = determine_aug_pipeline_2aug(opt_out_parsed)
            roi = augmentation_transformer_2aug(name_roi,
                                           image_trans,
                                           aug_pipeline,
                                           spike=False,
                                           visualization=False)
            x, y = np.random.randint(0, 1024), np.random.randint(0, 1024)
            sample_image = overlay_transparent(sample_image, roi, x, y)

        # ! Create the spike in the samples
        for i in range(opt_out_parsed.spikes_no):
            gt_correct = False
            while gt_correct == False:
                name_roi = name_spike_rois[np.random.randint(0, len(name_spike_rois))]
                image_trans = cv2.imread(os.path.join(roi_images_path, name_roi),
                                         cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

                aug_pipeline = determine_aug_pipeline_2aug(opt_out_parsed)
                roi, gt = augmentation_transformer_2aug(name_roi,
                                                   image_trans,
                                                   aug_pipeline,
                                                   spike=True,
                                                   visualization=False)
                if gt.max() != 0:
                    gt_correct = True

            points = False
            while points == False:
                x, y = np.random.randint(100, 1024 - 100), np.random.randint(100, 1024 - 100)
                point_x, point_y = find_gt_coordinates(gt)
                point_x = x + point_x
                point_y = y + point_y
                if point_x <= 1023 and point_y <= 1023:
                    points = True

            sample_image = spike_overlay_transparent(sample_image, roi, x, y)
            sample_gt[point_y, point_x] = point_scale

        # visualize the generaged image
        if visualize_sample:
            plt.imshow(sample_image)
            plt.pause(2)
            plt.close('all')

        # save the images in the temporary path
        image_name = 'image_' + str(sample) + '.png'
        gt_name = 'gt_' + str(sample) + '.png'
        cv2.imwrite(os.path.join(augmented_images, 'images', image_name), sample_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(os.path.join(augmented_images, 'actual_labels', gt_name), sample_gt, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# opt_output = np.random.uniform(0, 1, 24)
# sample_number = 100
# roi_images_path = paths.roi_images
# generate_augmented_images(opt_output, sample_number, roi_images_path, True)

#######################################################################################################################























