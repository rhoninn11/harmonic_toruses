import time
import numpy as np
import pye57

import generators.torus_generator as tg


def generate_poitions(point_count, grid_size):
    # linspace 0 to z
    x_num = grid_size[0]
    y_num = grid_size[1]
    z_num = grid_size[2]

    x = np.linspace(1, x_num, x_num, dtype=np.dtype('float64'))
    x = np.tile(x, y_num*z_num)
    x = x - 0.5*float(x_num)

    y = np.linspace(1, y_num, y_num, dtype=np.dtype('float64'))
    y = np.repeat(y, x_num)
    y = np.tile(y, z_num)
    y = y - 0.5*float(y_num)

    z = np.linspace(1, z_num, z_num, dtype=np.dtype('float64'))
    z = np.repeat(z, x_num*y_num)
    z = z - 0.5*float(z_num)

    return x, y, z

def calc_min_max(xyz_data):
    min_v = [min(axis_data) for axis_data in xyz_data]
    max_v = [max(axis_data) for axis_data in xyz_data]

    return min_v, max_v

def rescale_for_grid(xyz_data, grid_size):
    min_v, max_v = calc_min_max(xyz_data)

    scale = [max_v[i] - min_v[i] for i in range(len(xyz_data))]
    scale_up = [grid_size[i]/scale[i] for i in range(len(xyz_data))]

    min_idx = np.argmin(scale_up)
    chosen_scale = scale_up[min_idx]

    print(f"min_idx: {scale}, chosen_scale: {scale_up}")
    print(f"min_idx: {min_idx}, max_v: {chosen_scale}")
    # push valuse to positive side
    for i in range(len(xyz_data)):
        xyz_data[i] = xyz_data[i] - min_v[i]
        xyz_data[i] = xyz_data[i]*chosen_scale*0.75 + scale[i]*chosen_scale*0.125

    return xyz_data

def calc_volumetric_shape_idx_arr():
    width = 5
    offsets = np.linspace(-width, width, width*2+1, dtype=np.dtype('int64'))
    z_off = np.repeat(offsets, len(offsets)*len(offsets))
    x_off = np.tile(offsets, len(offsets)*len(offsets))
    y_off = np.repeat(offsets, len(offsets))
    y_off = np.tile(y_off, len(offsets))

    mask = np.array([x_off, y_off, z_off])
    # volumetric cube mask to volumetric sphere mask
    mask = mask[:, np.sum(mask**2, axis=0) <= (width+0.5)**2]

    return mask

def dense_grid_point_mod_idx(xyz_idx, grid_size, volumertic_shape_idx_arr):
    x_scl, y_scl, z_scl = grid_size
    xyz_off = volumertic_shape_idx_arr
    xyz_off = xyz_off.transpose() + xyz_idx

    # copilot optimised:D ~10k elements from 0.0083s to 0.0001s
    x_idx, y_idx, z_idx = xyz_off.T
    linear_idx = x_idx + x_scl*y_idx + x_scl*y_scl*z_idx
    linear_idx = linear_idx.astype(np.dtype('uint64'))
    idx_list = linear_idx.tolist()
    return linear_idx


def generate_color(point_count, grid_size):

    rgb_data = np.ones((point_count,3), dtype=np.dtype('uint8'))*250
    print(f"rgb_data: {rgb_data.shape}")

    torus_data = tg.SingleTorus.single_torus()
    
    xyz_data = [torus_data["cartesianX"], torus_data["cartesianY"], torus_data["cartesianZ"]]
    color_data = [torus_data["colorRed"], torus_data["colorGreen"], torus_data["colorBlue"]]
    
    xyz_data = rescale_for_grid(xyz_data, grid_size)
    np_xyz_data = np.rint(np.array(xyz_data))
    # convert to uint64 type
    np_xyz_data = np_xyz_data.astype(np.dtype('uint64'))

    np_point_data = np.concatenate((np_xyz_data, np.array(color_data)), axis=0)
    np_point_data = np_point_data.transpose()

    idxarr_precalc = calc_volumetric_shape_idx_arr()

    # baseline 32.5s -> 1.8s
    tic = time.perf_counter()
    for point in np_point_data:
        xyz_data = point[:3]
        mod_idx = dense_grid_point_mod_idx(xyz_data, grid_size, idxarr_precalc)
        rgb_data[mod_idx] = point[3:]
    toc = time.perf_counter()
    print(f"=== dense_grid_point_mod_idx: {toc - tic:0.4f} seconds")
    # shape of rgb_data
    print(f"rgb_data shape: {rgb_data.shape}")
    print(f"rgb_data dtype: {rgb_data.dtype}")

    # get r g b as separate np arrays 
    r = rgb_data[:,0]
    g = rgb_data[:,1]
    b = rgb_data[:,2]
    return r, g, b

def generate_data(point_count, grid_size):
    x,y,z = generate_poitions(point_count, grid_size)
    r,g,b = generate_color(point_count, grid_size)

    # its wierd but it seem unreal use red as blue and blue as read
    tmp_r = r
    r = b
    b = tmp_r

    intensity = np.ones((point_count), dtype=np.dtype('float32'))
    invalidState = np.zeros((point_count), dtype=np.dtype('uint8')) #other then 0 raise errro form library
    data = {
        "cartesianX": x,
        "cartesianY": y,
        "cartesianZ": z,
        "colorRed": r,
        "colorGreen": g,
        "colorBlue": b,
        "intensity": intensity,
        "cartesianInvalidState": invalidState
    }
    # print(data)
    return data

def script():
    grid_size = (500, 500, 200)
    point_count = grid_size[0]*grid_size[1]*grid_size[2]

    print(f"=== grid_size: {grid_size}")
    print(f"=== generate data")
    data_raw = generate_data(point_count, grid_size)
    print(f"=== data generated")
    e57_out = pye57.E57(f"fs/out/grid_sample.e57", mode='w')
    print(f"=== saving data")
    e57_out.write_scan_raw(data_raw)
    print(f"=== data saved")