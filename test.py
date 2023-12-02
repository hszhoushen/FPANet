import scipy.io as sio


color_mat = sio.loadmat('color150.mat')
print('color_mat:', color_mat['__header__'])
print('color_mat:', color_mat['__version__'])
print('color_mat:', color_mat['__globals__'])

print('color_mat:', color_mat['colors'], len(color_mat['colors']))

