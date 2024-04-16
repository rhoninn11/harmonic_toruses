import numpy as np
import pye57


base_name = "test"
# e57 = pye57.E57(f"fs/out/{base_name}.e57")
e57 = pye57.E57(f"fs/in/{base_name}.e57")
# read scan at index 0
header = e57.get_header(0)
print(header.point_count)

# all the header information can be printed using:
for line in header.pretty_print():
    print(line)

data = e57.read_scan(0)
print(data)
# 'data' is a dictionary with the point types as keys
assert isinstance(data["cartesianX"], np.ndarray)
assert isinstance(data["cartesianY"], np.ndarray)
assert isinstance(data["cartesianZ"], np.ndarray)

# other attributes can be read using:
data = e57.read_scan(0, intensity=True, colors=True, row_column=False)
assert isinstance(data["cartesianX"], np.ndarray)
assert isinstance(data["cartesianY"], np.ndarray)
assert isinstance(data["cartesianZ"], np.ndarray)
assert isinstance(data["intensity"], np.ndarray)
assert isinstance(data["colorRed"], np.ndarray)
assert isinstance(data["colorGreen"], np.ndarray)
assert isinstance(data["colorBlue"], np.ndarray)

data_raw = e57.read_scan_raw(0)
print(data_raw)
# the 'read_scan' method filters points using the 'cartesianInvalidState' field
# if you want to get everything as raw, untransformed data, use:

# the ScanHeader object wraps most of the scan information:
