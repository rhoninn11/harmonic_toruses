from utils.spectra import plus_minus_one2spectrum
import numpy as np
import pye57


V_ZERO = np.array([0,0,0])
V_X_BASE = np.array([1,0,0])
V_Y_BASE = np.array([0,1,0])
V_Z_BASE = np.array([0,0,1])


def print_header(e57_cloud, index=0):
    header = e57_cloud.get_header(index)
    print(header.point_count)

    # all the header information can be printed using:
    for line in header.pretty_print():
        print(line)

class spatial_osc():
    def __init__(self, point_count, freq, amp=(0,0)):
        self.freq = freq
    
        self.amp_x = amp[0]
        self.amp_y = amp[1]
    
        self.phase = 0
        self.delta = (2*np.pi*self.freq)/(point_count-1)

    def _calc(self, x_base, y_base, phase):
        x_p = self.amp_x*np.cos(phase) * x_base
        y_p = self.amp_y*np.sin(phase) * y_base
        return x_p + y_p

    def calc(self, x_base, y_base):
        return self._calc(x_base, y_base, self.phase)

    def calc_next(self, x_base, y_base):
        next_phase = self.phase + self.delta
        return self._calc(x_base, y_base, next_phase)

    def inc(self):
        self.phase += self.delta

class spatial_ref():
    def __init__(self, p=V_ZERO, base=V_Z_BASE):
        self.base = base
        self.p = p

class spatial_memory():
    def __init__(self, point_count):
        self.memory = [spatial_ref()] * point_count
        self.count = point_count
        self.index = 0

        self.osc = None

    def bind_osc(self, osc):
        self.osc = osc
        print(f"osc binded {self.osc}")
    
    def _append(self, p, base):
        s_ref = spatial_ref(p, base)
        self.memory[self.index] = s_ref
        self.index = (self.index + 1)%self.count


    def next_point(self, x_base=V_X_BASE, y_base=V_Y_BASE, z_base=V_Z_BASE, p_ref=V_ZERO):
        local_space_p = self.osc.calc(x_base, y_base)
        self.osc.inc()
        
        global_space_p = p_ref + local_space_p
        self._append(global_space_p, z_base)

        result = [global_space_p]
        return result

    
class sub_space_memory(spatial_memory):
    def __init__(self, point_count):
        spatial_memory.__init__(self, point_count)

        self.parent_space = None

    def safe_ref_sample(self, index):
        index_minus_one = index - 1
        index_plus_one = index + 1

        if index_minus_one < 0:
            index_minus_one = self.count + index_minus_one
        
        if index_plus_one >= self.count:
            index_plus_one = index_plus_one - self.count

        idx_list = [index_minus_one, index, index_plus_one]
        samples = [self.memory[idx] for idx in idx_list ]
        return samples


    def reconstruct(self, index):
        ref_list = self.safe_ref_sample(index)
        p_list = [s_ref.p for s_ref in ref_list]

        new_z_base = ((p_list[1] - p_list[0]) + (p_list[2] - p_list[1]))/2
        new_z_base = new_z_base / np.linalg.norm(new_z_base)
        
        main_ref = ref_list[1]
        new_x_base = main_ref.base

        new_y_base = np.cross(new_z_base, new_x_base)
        new_y_base = new_y_base / np.linalg.norm(new_y_base)

        new_x_base = np.cross(new_y_base, new_z_base)
        new_x_base = new_x_base / np.linalg.norm(new_x_base)

        return new_x_base, new_y_base, new_z_base, main_ref.p

    def bind_parent_space(self, parent_space):
        self.parent_space = parent_space

    def next_point(self, x_base=V_X_BASE, y_base=V_Y_BASE, z_base=V_Z_BASE, p_ref=V_ZERO):
        if self.parent_space:
            x,y,z,p = self.parent_space.reconstruct(self.index)
            return spatial_memory.next_point(self, x, y, z, p)
        
        return spatial_memory.next_point(self, x_base, y_base, z_base, p_ref)


def spatial_osc_init(point_count, recipe_list):

    memory_list = []
    for recipe in recipe_list:
        frequency = recipe[0]
        size = recipe[1]

        osc = spatial_osc(point_count, frequency, (size,size))
        spc_mem = sub_space_memory(point_count)
        spc_mem.bind_osc(osc)
        memory_list.append(spc_mem)

    for i in range(len(memory_list) - 1):
        memory_list[i + 1].bind_parent_space(memory_list[i])

    return memory_list

    

def generate_poitions(point_count, recipe_list):

    x = np.zeros((point_count), dtype=np.dtype('float64'))
    y = np.zeros((point_count), dtype=np.dtype('float64'))
    z = np.zeros((point_count), dtype=np.dtype('float64'))
    
    spaces2calculate = spatial_osc_init(point_count, recipe_list)

    for space in spaces2calculate:
        for i in range(point_count):
            result = space.next_point()
            x[i], y[i], z[i] = result[-1]

    return x, y, z

def generate_color(point_count):

    r = intensity = np.zeros((point_count), dtype=np.dtype('uint8'))
    g = intensity = np.zeros((point_count), dtype=np.dtype('uint8'))
    b = intensity = np.zeros((point_count), dtype=np.dtype('uint8'))

    # form -1 to 1 seep
    pos = -1.0
    delta = 2.0/(point_count-1)
    for i in range(point_count):
        rgb_tupla = plus_minus_one2spectrum(pos)
        r[i], g[i], b[i] = rgb_tupla
        pos += delta
    print(f"r: {r.dtype}, g: {g.dtype}, b: {b.dtype}")
    return r, g, b


def generate_data(point_count, recipe_list):
    x,y,z = generate_poitions(point_count, recipe_list)
    r,g,b = generate_color(point_count)
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



def permut(list_2_permut):
    permut_result = []
    for i in range(len(list_2_permut)):
        lcp = list_2_permut.copy()
        pre_merge = [lcp[i]]
        del lcp[i]
        
        if len(lcp):
            sub_permut_list = permut(lcp)
            for sub_permut in sub_permut_list:
                single_result = pre_merge.copy()
                single_result.extend(sub_permut)
                permut_result.append(single_result)
        else:
            permut_result.append(pre_merge)
    
    return permut_result


def prepare_recipe(size=1):
    stages = [0,1,2,3]
    freqs = [1, 9, 81, 729]
    ring_size = [12, 4.68, 1.83, 0.71]
    # scale
    for i in range(len(ring_size)):
        ring_size[i] = ring_size[i] * size

    permutations = []
    for i in range(len(stages)):
        permutations.extend(permut(stages[0:i+1]))
        
    recipe_data_list = []
    for permutation in permutations:
        single_recipe = []
        for i in range(len(permutation)):
            size_index = i
            freq_index = permutation[i]
            single_recipe.append((freqs[freq_index], ring_size[size_index]))
        recipe_data = {"recipe": single_recipe, "permut": permutation}
        recipe_data_list.append(recipe_data)

    return recipe_data_list

def add_permut_info2name(base_name, permutation):
    str_tmp = [base_name]
    for p in permutation:
        str_tmp.append(f"_{p}")

    return "".join(str_tmp)

def script():
    point_num = 10000
    recipe_data_list = prepare_recipe()
    for recipe_data in recipe_data_list:
        base_name = "synth"
        base_name = add_permut_info2name(base_name, recipe_data["permut"])
        recipe = recipe_data["recipe"]
        data_raw = generate_data(point_num, recipe)
        e57_out = pye57.E57(f"fs/out/{base_name}.e57", mode='w')
        e57_out.write_scan_raw(data_raw)

def filter_recipe_list(recipe_data_list, golden_permut):
    golder_recipe_data_list = []
    for recipe_data in recipe_data_list:
        a = recipe_data["permut"]
        b = golden_permut
        if len(a) == len(b):
            c = 0
            for i in range(len(a)):
                if a[i] == b[i]:
                    c += 1
            if c == len(a):
                golder_recipe_data_list.append(recipe_data)
    return golder_recipe_data_list

# static method
class SingleTorus():
    def script():
        point_num = 10000
        recipe_data_list = prepare_recipe()
        filtered_data_list = filter_recipe_list(recipe_data_list, [1,2,0,3])
        base_name = "single_torus"
        for recipe_data in filtered_data_list:
            recipe = recipe_data["recipe"]
            data_raw = generate_data(point_num, recipe)
            e57_out = pye57.E57(f"fs/out/{base_name}.e57", mode='w')
            e57_out.write_scan_raw(data_raw)

    def single_torus():
        point_num = 10000
        recipe_data_list = prepare_recipe()
        filtered_data_list = filter_recipe_list(recipe_data_list, [1,2,0,3])
        base_name = "single_torus"
        for recipe_data in filtered_data_list:
            recipe = recipe_data["recipe"]
            data_raw = generate_data(point_num, recipe)
            return data_raw

