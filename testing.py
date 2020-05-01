import numpy as np
import matplotlib.pyplot as plt
from denoising import SpectrumDenoising
from scipy.signal import medfilt, savgol_filter

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
#dns = SpectrumDenoising(t_uni, i0_uni)
dns = SpectrumDenoising(t_uni, iff_uni)
dns.denoise()

#%%

x = np.array([1,4,6])
y = np.array([1,2,-1])

from denoising import find_curvature

find_curvature(x,y)



#%%

kf = KFold(n_splits=5, shuffle=True)

bla = t_uni[:100]
bla_y = i0[:100]
bla_idx = [i for i in kf.split(bla)][0]


plt.figure(100)
plt.clf()

plt.plot(bla[bla_idx[0]], bla_y[bla_idx[0]], 'k.')
plt.plot(bla[bla_idx[1]], bla_y[bla_idx[1]], 'ro')




#%%


e_smooth = savgol_filter(e_uni, 1001, 3)

#print(t_traj[-1], t_uni[-1])


N = 2000
w = 2
#t_uni_sel = (t_uni>19.5) & (t_uni<19.55)
t_uni_sel = (t_uni>(t_uni[N]-w/2)) & (t_uni<(t_uni[N]+w/2))

def fft_plot(t, x, color='k'):
    f = np.fft.fftfreq(t.size, d=t[1]-t[0])
    x_fft = np.fft.fft(x)
#    plt.plot(f[f>0], np.angle(x_fft)[f>0], 'k.-')
    plt.plot(f[f>0], np.abs(x_fft)[f>0], '.-', color=color)
    plt.grid()
    plt.xlim(1, 250)


plt.figure(1)
plt.clf()
#
plt.subplot(221)
#plt.plot(e_bins, t_per_bin)
#plt.plot(e_bins, t_per_bin_upd)

plt.plot(t_uni[t_uni_sel], i0_uni[t_uni_sel])

plt.subplot(222)
fft_plot(t_uni[t_uni_sel], i0_uni[t_uni_sel])

plt.subplot(223)
#plt.plot(t_traj - t_traj[-1], e_bins)
#plt.plot(t_traj_upd-t_traj_upd[-1], e_bins)
plt.plot(t_uni[t_uni_sel], (e_uni - e_smooth)[t_uni_sel] )
#plt.plot(t_uni[t_uni_sel], (e_smooth[t_uni_sel] - np.mean(e_uni[t_uni_sel])))

#plt.plot(t_uni, e_uni - e_smooth)
#plt.plot(t_uni, e_smooth)

plt.subplot(224)
fft_plot(t_uni[t_uni_sel], (e_uni - e_smooth)[t_uni_sel])

plt.figure(99)
plt.subplot(212)
fft_plot(t_uni[t_uni_sel], (e_uni - e_smooth)[t_uni_sel], color='r')




#%%


n_bins, _ = np.histogram(e, e_edges)

t_edges = np.interp(e_edges, e_uni[np.argsort(e_uni)], t_uni[np.argsort(e_uni)])

energy_bin = e_edges[:-3] + np.diff(e_edges[:-2])/2


time_per_bin = -np.diff(t_edges[:-2])
time_per_bin_upd = time_per_bin.copy()
time_per_bin_upd[time_per_bin_upd<0.05] = 0.05


e_traj = energy_bin
t_traj = np.cumsum(-time_per_bin)

t_traj -= t_traj.min()

#dt = t_uni[-1]/t_uni.size
#
#t_per_bin = n_bins*dt
#t_per_bin_upd = t_per_bin.copy()
#t_per_bin_upd[t_per_bin_upd<0.05] = 0.05
##t_per_bin_upd[t_per_bin_upd>0.1] = 0.15
#
#t_traj = np.cumsum(-t_per_bin[:-1])
#t_traj = np.hstack((0, t_traj))
#t_traj_upd = np.cumsum(-t_per_bin_upd)


plt.figure(2)
plt.clf()


plt.subplot(211)
plt.plot(energy_bin, time_per_bin*2)
plt.plot(energy_bin, time_per_bin_upd)

plt.subplot(212)
plt.plot(t_traj, e_traj, 'k.-')
#plt.plot(t_uni[], e_uni)
#plt.plot(energy_bin, time_per_bin)
#plt.hlines(e_edges, 0, 30)


#plt.plot(e_uni)

