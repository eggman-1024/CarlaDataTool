import numpy as np
from vedo import Points,Plotter,show
a=np.load("raw_data/2208261034_96fov1/vehicle.tesla.model3.master/velodyne_semantic/0000000075.npy",allow_pickle=True)
format_data = []
for l in a:
    b=list(l)
    format_data.append(b)

data=np.array(format_data)
data=data[:,0:3]
RGBA = np.array([[0,0,0,255]]*data.shape[0])
pts = Points(data, r=2, c=RGBA)
plt = Plotter(interactive=False, offscreen=True)

#plt.lookAt()
# 导出图片

cam1 = dict(
    pos=(-18.43110, -455.6388, 208.1018),
    focalPoint=(-8.490190, 10.34274, 11.72185),
    viewup=(-0.07181808, 0.3886525, 0.9185812),
    distance=505.7694,
    clippingRange=(345.8786, 708.2590),
)
cam2 = dict(
    pos=(-18.43110, -455.6388, 208.1018),
    focalPoint=(-8.490190, 10.34274, 11.72185),
    viewup=(-0.07181808, 0.3886525, 0.9185812),
    distance=505.7694,
    clippingRange=(345.8786, 708.2590),
)
plt.moveCamera(camstart=cam1,camstop=cam2,fraction=0)
plt.show(pts)
plt.screenshot("out1.png", scale=2).close()
cam3 = dict(
    pos=(-18.43110, -455.6388, 208.1018),
    focalPoint=(-8.490190, 10.34274, 11.72185),
    viewup=(-0.07181808, 0.3886525, 0.9185812),
    distance=505.7694,
    clippingRange=(345.8786, 708.2590),
)
show(pts, __doc__, axes=True,camera=cam3).close()