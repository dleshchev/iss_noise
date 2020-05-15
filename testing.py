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
dns = SpectrumDenoising(t_uni, i0_uni)
#dns = SpectrumDenoising(t_uni, iff_uni) 
dns.denoise()
#%%


plt.figure(1)
plt.clf()

plt.subplot(321)
plt.plot(t_uni, i0_uni)

plt.xlabel('time, s')
plt.ylabel('i0')


plt.subplot(322)
freq = np.fft.fftfreq(np.sum([t_uni<1]), d=t_uni[1]-t_uni[0])
i0_fft = np.fft.fft(i0_uni[t_uni<1])

plt.plot(freq[np.argsort(freq)], np.abs(i0_fft[np.argsort(freq)]))

plt.xlabel('frquency, Hz')
plt.ylim([0, 1e-4])

plt.subplot(323)
plt.plot(t_uni, iff_uni)

plt.xlabel('time, s')
plt.ylabel('iff')

plt.subplot(325)
plt.plot(t_uni, iff_uni/i0_uni)
plt.xlabel('time, s')
plt.ylabel('mu')


plt.figure(2)
plt.clf()

plt.subplot(311)
plt.plot(t_uni, e_uni)



mask = (t_uni>19) & (t_uni<20)
plt.subplot(312)
plt.plot(t_uni[mask], (e_uni - savgol_filter(e_uni, 201, 3))[mask])


plt.subplot(312)
plt.plot(t_uni[mask], (e_uni - savgol_filter(e_uni, 201, 3))[mask])


freq_e = np.fft.fftfreq(t_uni[mask].size, t_uni[1]-t_uni[0])
e_fft = np.fft.fft((e_uni - savgol_filter(e_uni, 201, 3))[mask])

plt.subplot(313)
plt.plot(freq_e, np.abs(e_fft))



#%%

from cvxopt import matrix, normal, spdiag, misc, lapack
from ubsdp import ubsdp

m, n = 50, 50
A = normal(m**2, n)

# Z0 random positive definite with maximum e.v. less than 1.0.
Z0 = normal(m,m)
Z0 = Z0 * Z0.T
w = matrix(0.0, (m,1))
a = +Z0
lapack.syev(a, w, jobz = 'V')
wmax = max(w)
if wmax > 0.9:  w = (0.9/wmax) * w
Z0 = a * spdiag(w) * a.T

# c = -A'(Z0)
c = matrix(0.0, (n,1))
misc.sgemv(A, Z0, c, dims = {'l': 0, 'q': [], 's': [m]}, trans = 'T', alpha = -1.0)

# Z1 = I - Z0
Z1 = -Z0
Z1[::m+1] += 1.0

x0 = normal(n,1)
X0 = normal(m,m)
X0 = X0*X0.T
S0 = normal(m,m)
S0 = S0*S0.T
# B = A(x0) - X0 + S0
B = matrix(A*x0 - X0[:] + S0[:], (m,m))

X = ubsdp(c, A, B)


#%%

# Import packages.
import numpy as np
import cvxpy as cp

# Generate a random SDP.
n = 3
p = 3
np.random.seed(1)
C = np.random.randn(n, n)
A = []
b = []
for i in range(p):
    A.append(np.random.randn(n, n))
    b.append(np.random.randn())

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
X = cp.Variable((n,n), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0]
constraints += [
    cp.trace(A[i] @ X) == b[i] for i in range(p)
]
prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                  constraints)
prob.solve(verbose=True)

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)
#%%

def bla():
    t = np.linspace(0, 1, 101)
    f = np.array([1, 2, 3, 4, 5, 6])
    a = np.array([0.0, 1.0, 0.0,  0.0, -1.0, 0.0,
                -0.1, 0.0, 0.5, -0.2, 0.5, 0.1])
    ksi = 2*np.pi*f[None,:]*t[:, None]
    X = np.hstack((np.cos(ksi), np.sin(ksi)))
    
    y = (X @ a)**2
    XTX_inv = np.linalg.pinv(X.T @ X)
    y_red = X.T @ np.diag(y) @ X - 0.5 * 0 * np.eye(a.size) 
    A = XTX_inv @ y_red @ XTX_inv
#    print(A.shape, np.linalg.matrix_rank(A))
#    print(np.linalg.matrix_rank(X.T @ X)
    hhh = a[:, None] @ a[None, :]
#    print(hhh.shape, np.linalg.matrix_rank(hhh))
    
    Y = y[:, None] * X @ X.T
    u, s, v = np.linalg.svd(Y)
    
    plt.figure(2)
    plt.clf()
#    plt.plot(t, np.diag(g))
    plt.subplot(211)
    plt.plot(t, y)
    plt.plot(t, np.diag(X @ A @ X.T)*15-1)
    
    plt.subplot(212)
    plt.imshow(X @ A @ X.T)


bla()

#x = np.array([1,-1,3])
#
##A = x[:, None] @ x[None, :]
#A = x[:, None] @ x[None, :] + np.random.randn(x.size, x.size)*1e-4
#
#
#u,s,v = np.linalg.svd(A)
#print(u @ np.sqrt(np.diag(s)))

#%%


#from patsy import dmatrix

vvv = np.linspace(-10, 10, 101)


#bla = dmatrix("bs(x, df=10, degree=3, include_intercept=True) - 1", {"x": vvv})

plt.figure(1)
plt.clf()

plt.plot(vvv, 2*(np.sqrt(vvv**2+1)-1))

#function
#B = bspline(x,xl, xr, ndx,bdeg)
#dx = (xr - xl) / ndx;
#t = xl + dx * [-bdeg:ndx-1];
#T = (0 * x + 1) * t;
#X = x * (0 * t + 1);
#P = (X - T) / dx;
#B = (T <= X) & (X < (T + dx));
#r = [2:length(t) 1];
#for k = 1:bdeg
#B = (P .* B + (k + 1 - P) .* B(:,r)) / k;
#end;
#end;


#%%


from denoising import interp_spline
#t_sel = (t_uni>14.5) & (t_uni<15.5)
t_sel = (t_uni>15.23) & (t_uni<15.24)

def interpolate(t, x, f=10):
    n = t.size
    t_pad = np.linspace(t.min(), t.max(), n*f+1)
    x_pad = interp_spline(t_pad, t, x)
    return t_pad, x_pad
#    plt.figure(52)
#    plt.clf()
##    plt.plot(np.abs(x_fft))
##    plt.plot(np.abs(x_fft_pad))
#    
#    plt.plot(t, x, 'k.')
#    plt.plot(t_pad, x_pad, 'r-')
#
#interpolate(t_uni[t_sel], iff_uni[t_sel])

def diff_interp(x, n, f=10):
    t = np.arange(x.size)
    t_pad, x_pad = interpolate(t, x, f=10)
    dx_pad = np.diff(x_pad, n)*f**n
    t_pad_diff = t_pad[n:] - (t_pad[1] - t_pad[0])/2*n
    
    return np.interp(t, t_pad_diff, dx_pad)
#    return dx_pad


plt.figure(23)
plt.clf()

plt.subplot(221)
plt.plot(t_uni, iff_uni, 'k-')
plt.plot(t_uni[t_sel], iff_uni[t_sel], 'r-')

plt.subplot(422)
#plt.plot(t_uni[:-1] + np.diff(t_uni)/2, np.diff(iff_uni), 'k-')
#plt.plot(t_uni[t_sel][:-1] + np.diff(t_uni[t_sel])/2, np.diff(iff_uni[t_sel]), 'r-')
plt.plot(np.diff(iff_uni[t_sel],0), 'r-')
plt.plot(diff_interp(iff_uni[t_sel], 0), 'g--')



plt.subplot(424)
plt.plot(np.diff(iff_uni[t_sel], 1), 'r-')
plt.plot(diff_interp(iff_uni[t_sel], 1), 'g--')


plt.subplot(426)
plt.plot(np.diff(iff_uni[t_sel], 2), 'r-')
plt.plot(diff_interp(iff_uni[t_sel], 2), 'g--')

plt.subplot(428)
plt.plot(np.diff(iff_uni[t_sel], 6), 'r-')
plt.plot(diff_interp(iff_uni[t_sel], 6), 'g--')

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

