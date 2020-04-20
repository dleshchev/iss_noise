import numpy as np
import matplotlib.pyplot as plt
from denoising import Denoising


d = np.genfromtxt(r'C:\work\Experiments\2020\Mark_Cobalt_03\CoK_IE200_reduction(1) 0035.raw')
di = np.genfromtxt(r'C:\work\Experiments\2020\Mark_Cobalt_03\eli.xas')

e_bins = di[:, 0]
e_edges = np.hstack((e_bins[0] - (e_bins[1] - e_bins[0])/2,
                     e_bins[:-1] + np.diff(e_bins),
                     e_bins[-1] + (e_bins[-1] - e_bins[-2])/2))

t = d[:, 0]
t -= t.min()
e = d[:, 1]
i0 = d[:, 2]
iff = d[:, 5]


t_sort = np.argsort(t)

t_uni = np.linspace(t.min(), t.max(), t.size)
e_uni = np.interp(t_uni, t[t_sort], e[t_sort])
iff_uni = np.interp(t_uni, t[t_sort], iff[t_sort])
i0_uni = np.interp(t_uni, t[t_sort], i0[t_sort])


# dns = Denoising(t_uni, iff_uni)
dns = Denoising(t_uni, i0_uni)
dns.denoise()
