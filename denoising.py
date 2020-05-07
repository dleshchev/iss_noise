import numpy as np
import matplotlib.pyplot as plt

import time
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from sklearn.model_selection import KFold
from patsy import dmatrix
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class NonlinearSemiparametricSolver:
    def __init__(self, t, y, _F, _gradF, theta0, dof=4, deg=3, ndiff=2):
        assert t.size == y.size, 't and y mist have the same size'

        self.t = t
        # self.y = y
        self.y = (y - y.min())/(y.max() - y.min())
        self.m = t.size

        self._F = _F
        self._gradF = _gradF

        self.theta0 = theta0

        str_input = f"bs(y, df={dof}, degree={deg}, include_intercept=True) - 1"
        self.A = dmatrix(str_input, {"y": t})
        self.D = L(dof, ndiff)
        # self.D = L(self.m, ndiff) @ self.A

        # self.A = np.eye(self.m)
        # self.D = L(self.m, ndiff) @ self.A
        self.B = self.D.T @ self.D

    def F(self, theta):
        return self._F(theta, self.t)

    def gradF(self, theta):
        return self._gradF(theta, self.t)

    def y_theta(self, theta):
        return self.y - self.F(theta)

    def _update_B_kappa(self, kappa):
        self._z,_,_,_ = np.linalg.lstsq(self.A.T @ self.A + kappa * self.m * self.B, self.A.T, rcond=-1)
        _I = np.eye(self.m)
        self.B_kappa = np.vstack(((_I - self.A @ self._z),
                                 (np.sqrt(self.m * kappa) * self.D @ self._z)))

    def resid_theta(self, theta):
        return self.B_kappa @ self.y_theta(theta)

    def jac_resid_theta(self, theta):
        return -self.B_kappa @ self.gradF(theta)

    def solve_kappa(self, kappa):
        self._update_B_kappa(kappa)
        result = least_squares(self.resid_theta, self.theta0,
                               method='lm', jac=self.jac_resid_theta)
        theta_kappa = result['x']
        a_kappa = self._z @ self.y_theta(theta_kappa)
        F_kappa = self.F(theta_kappa)
        g_kappa = self.A @ a_kappa
        y_kappa = F_kappa + g_kappa

        # plt.figure(10)
        # plt.clf()
        #
        # plt.plot(self.t, self.y, 'k.-')
        # plt.plot(self.t, y_kappa, 'r-')
        # plt.plot(self.t, g_kappa, 'b--')


        gradF_kappa = self.gradF(theta_kappa)
        T_kappa = np.hstack((gradF_kappa, self.A))


        m11 = gradF_kappa.T @ gradF_kappa
        m12 = gradF_kappa.T @ self.A
        m22 = self.A.T @ self.A + self.m * kappa * self.B
        # print(m11.shape, m12.shape)
        # print(m12.T.shape, m22.shape)
        # self.theta0 = theta_kappa
        H_kappa = np.block([[m11,   m12],
                            [m12.T, m22]])
        S_kappa = T_kappa @ np.linalg.pinv(H_kappa) @ T_kappa.T
        GCV_kappa = self.m * np.linalg.norm(y_kappa - self.y)**2 / np.trace(np.eye(self.m) - S_kappa)

        return theta_kappa, a_kappa, y_kappa, F_kappa, g_kappa, GCV_kappa


    def compute_GCV_grid(self, kappas):
        thetas = np.zeros((self.theta0.size, kappas.size))
        aas = np.zeros((self.A.shape[1], kappas.size))
        GCV = np.zeros(kappas.shape)
        ys = np.zeros((self.t.size, kappas.size))
        Fs = np.zeros((self.t.size, kappas.size))
        gs = np.zeros((self.t.size, kappas.size))
        for i, kappa in enumerate(kappas):
            thetas[:, i], aas[:, i], ys[:, i], Fs[:, i], gs[:, i], GCV[i] = self.solve_kappa(kappa)

        # idx = np.argmin(GCV)
        idx = np.argmin(np.sum(Fs**2, axis=0))
        plt.figure(10)
        plt.clf()

        plt.plot(self.t, self.y, 'k.-')
        plt.plot(self.t, ys[:, idx], 'r-')
        plt.plot(self.t, gs[:, idx], 'b--')

        plt.figure(11)
        plt.clf()
        plt.loglog(kappas, GCV, 'k.-')
        plt.vlines(kappas[idx], GCV.min(), GCV.max(), colors='r')

        plt.figure(12)
        plt.clf()
        plt.subplot(321)
        plt.plot(self.t, Fs, alpha=0.5)
        plt.plot(self.t, Fs[:, idx], 'r')

        plt.subplot(322)
        plt.imshow(Fs)


        jj = int(self.theta0.size/2)

        bbb = np.sqrt(thetas[jj:, :]**2 + thetas[:jj, :]**2)
        plt.subplot(323)
        plt.plot(thetas[:jj, :], alpha=0.5)
        plt.plot(thetas[:jj, idx], 'r-')
        # plt.plot(thetas[jj:, :], 'b.-')

        # plt.plot(bbb, alpha=0.5)
        # plt.plot(bbb[:, idx], 'r-')

        plt.subplot(324)
        # plt.imshow(np.sqrt(thetas[jj:, :]**2 + thetas[:jj, :]**2))
        plt.plot(np.sum(Fs**2, axis=0), 'm.-')

        plt.subplot(325)
        plt.plot(aas, alpha=0.5)
        plt.plot(aas[:, idx], 'r')


        plt.subplot(326)
        plt.semilogy(GCV, 'k.-')
        # plt.semilogy(bbb[3, :], 'r.-')
        # plt.semilogy(aas[0, :], 'm.-')

        # plt.semilogy(np.sum((self.D @ aas)**2, axis=0), 'g.-')
        # plt.semilogy(np.sum((ys - self.y[:, None]) ** 2, axis=0), 'b.-')
        plt.vlines(idx, GCV.min(), GCV.max(), colors='b')




class DenoisePiece:

    def __init__(self, t, y, theta0, w_base=0.005, fmin=0, fmax=200):
        self.t = t
        self.y = y
        self.m = t.size


        self.get_fk(fmin, fmax)

        # self.get_conditioner()
        # self.y = self.condition_piece(self.y)

        # self.kappa = 1
        self.theta0 = theta0

    def _F(self, theta, t):
        return osc_sum_square(theta, self.fk, t)

    def _gradF(self, theta, t):
        return grad_osc_sum_square(theta, self.fk, t)


    def _y_resid(self, theta, y):
        return y - self._F(theta, self.t)

    def _grady_resid(self, theta):
        return -self._gradF(theta, self.t)

    # def cross_validate(self, kappas=None, n_splits=2):
    #     self.process()
    #     if not kappas:
    #         kappas = 10**np.linspace(-1, 1, 11)
    #     kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    #     gen_error = np.zeros((kappas.size, n_splits))
    #     # for train_indey, test_indey in kf.split(self.t):
    #     print([i for i in kf.split(self.t)][0])
    #     for train_indey, test_indey in [[i for i in kf.split(self.t)][0]]:
    #         t_train, t_test = self.t[train_indey], self.t[test_indey]
    #         y_train, y_test = self.y[train_indey], self.y[test_indey]
    #         # for kappa in kappas:
    #         for kappa in [5e0]:
    #             print('bla')
    #             dp_cv = DenoisePiece(t_train, y_train, self.theta0, kappa, fk=self.fk)
    #             dp_cv.process(plotting=True)
    #
    #             plt.figure(3)
    #             plt.subplot(211)
    #             plt.plot(self.t, self.y, '.-', color=[0.7, 0.7, 0.7])


    # def optimize(self, kappas=None):
    #     # self.process()
    #     if not kappas:
    #         kappas = np.hstack((10**-3, 10**np.linspace(-2, 0, 55), 10**1))
    #
    #     rho = np.zeros(kappas.shape)
    #     ksi = np.zeros(kappas.shape)
    #     for k, kappa in enumerate(kappas):
    #         self.kappa = kappa
    #         self.process()
    #         # rho[k] = np.log10(np.sum((self.y_fit - self.y)**2)) # SE
    #         # ksi[k] = np.log10(np.sum((self.ridge.L @ self.y_smooth)**2)) # reg norm
    #         rho[k] = (np.sum((self.y_fit - self.y) ** 2))  # SE
    #         # ksi[k] = (np.sum((self.ridge.L @ self.y_smooth) ** 2))  # reg norm
    #         ksi[k] = self.ridge.penalty
    #
    #
    #     # drho, dksi = np.diff(np.log(rho)), np.diff(np.log(ksi))
    #     # drho, dksi = np.diff(rho), np.diff(ksi)
    #     # ddrho, ddksi = np.diff(drho), np.diff(dksi)
    #
    #     # curvature = 2 * (drho[:-1]*ddksi - ddrho*dksi[:-1]) / (drho[:-1]**2 + dksi[:-1]**2)**1.5
    #     # curvature = find_curvature(rho, ksi)
    #     # idy = np.argmay(curvature) + 1
    #
    #     fom = np.sqrt(((rho-rho.min())/(rho.max()-rho.min()))**2 +
    #                   ((ksi-ksi.min())/(ksi.max()-ksi.min()))**2)
    #     dists = np.diff(np.log(rho))**2 + np.diff(np.log(ksi))**2
    #     fom2 = np.sqrt(dists[1:] + dists[:-1])
    #
    #     idy = np.argmin(fom[1:-1])+1
    #     idy2 = np.argmin(fom2)+1
    #
    #     plt.figure(4)
    #     plt.clf()
    #     plt.subplot(211)
    #     plt.loglog(rho, ksi, 'k.-')
    #     plt.loglog(rho[idy], ksi[idy], 'ro')
    #     plt.loglog(rho[idy2], ksi[idy2], 'bo')
    #     # plt.plot(rho, ksi, 'k.-')
    #     # plt.plot(rho[idy], ksi[idy], 'ro')
    #     # plt.plot(rho[idy2], ksi[idy2], 'bo')
    #     plt.ylabel('MSE norm')
    #     plt.ylabel('penalty norm')
    #
    #     plt.subplot(223)
    #     # plt.semilogy(kappas[1:-1], curvature, 'k.-')
    #     plt.semilogy(kappas, fom, 'k.-')
    #     # plt.vlines(kappas[idy], curvature.min(), curvature.may(), colors='r')
    #     plt.vlines(kappas[idy], fom.min(), fom.may(), colors='r')
    #     plt.vlines(kappas[idy2], fom.min(), fom.may(), colors='b')
    #     # plt.loglog(SE, PN, 'k.-')
    #
    #     plt.subplot(224)
    #     # plt.semilogy(kappas[1:-1], curvature, 'k.-')
    #     plt.semilogy(kappas[2:-2], fom2[1:-1], 'k.-')
    #     # plt.vlines(kappas[idy], curvature.min(), curvature.may(), colors='r')
    #     plt.vlines(kappas[idy], fom2[1:-1].min(), fom2[1:-1].may(), colors='r')
    #     plt.vlines(kappas[idy2], fom2[1:-1].min(), fom2[1:-1].may(), colors='b')
    #
    #     self.kappa = kappas[idy2]
    #     self.process(plotting=True)







    def process(self, w_base=0.005, plotting=False):

        if self.theta0 is None:
            y_base = self.predef_baseline(w_base)
            self.theta0 = self.generate_starting_guess(self.y - y_base)

        # plt.figure(1)
        # plt.clf()
        #
        # plt.plot(self.t, self.y)
        # # plt.plot(self.t, y_base)
        # plt.plot(self.t, self._F(self.theta0, self.t)+y_base)


        nss = NonlinearSemiparametricSolver(self.t, self.y, self._F, self._gradF, self.theta0)

        # kappa = 0.01
        # nss.solve_kappa(kappa)

        kappas = 10**np.linspace(-15, 3, 51)
        nss.compute_GCV_grid(kappas)

        # self.ridge = LRidge(self.t, self.kappa)
        # self.ridge = LRidge_spline(self.t, 10, 3, self.kappa)
        # self.theta = least_squares(self.fit_resid_nonparametric, self.theta0,
        #                         method='lm', jac=self.fit_resid_nonparametric_jac)['x']

    #     osci = self.F(self.theta)
    #     bottom = self.get_y_bkg(osci)
    #     self.y_fit = osci + bottom
    #     self.y_smooth = bottom
    #
    #     # self.y = self.uncondition_piece(self.y)
    #     # self.y_fit = self.uncondition_piece(self.y_fit)
    #     # self.y_smooth = self.uncondition_piece(self.y_smooth)
    #
    #     if plotting:
    #         plt.figure(3)
    #         plt.clf()
    #
    #         plt.subplot(221)
    #         plt.plot(self.t, self.y, 'k.-')
    #         plt.plot(self.t, self.y_fit, 'r-')
    #         # plt.plot(self.t, osci, 'g--')
    #         # plt.plot(self.t, self.y - osci, 'b--')
    #         plt.plot(self.t, self.y_smooth, 'b--')
    #         #
    #         plt.subplot(223)
    #         plt.plot(self.fk, np.sqrt(self.theta[:self.fk_size]**2 + self.theta[self.fk_size:]**2), 'k.-')
    #         plt.plot(self.fk, self.theta[:self.fk_size], 'r.-')
    #         plt.plot(self.fk, self.theta[self.fk_size:], 'b.-')
    #         # plt.plot(self.fk, np.angle(1j*self.theta0[:self.fk_size] + self.theta0[self.fk_size:]), 'k.-')
    #         # plt.plot(self.t, self.y - osci, 'k.-')
    #
    #         plt.subplot(422)
    #         plt.plot(np.diff(self.y), 'k.-')
    #         plt.plot(np.diff(self.y_fit), 'r-')
    #         plt.plot(np.diff(osci), 'g-')
    #
    #         plt.subplot(424)
    #         plt.plot(np.diff(self.y, 2), 'k.-')
    #         plt.plot(np.diff(self.y_fit, 2), 'r-')
    #         plt.plot(np.diff(osci, 2), 'g-')
    #
    #         plt.subplot(426)
    #         plt.plot(np.diff(self.y, 3), 'k.-')
    #         plt.plot(np.diff(self.y_fit, 3), 'r-')
    #         plt.plot(np.diff(osci, 3), 'g-')
    #
    #
    #
    def predef_baseline(self, w_base):
        n_int = int(np.ceil((self.t.max() - self.t.min()) / w_base))
        t_edges = np.linspace(self.t.min(), self.t.max(), n_int + 1)
        t_bins = t_edges[:-1] + np.diff(t_edges) / 2
        y_mins = np.zeros(t_bins.shape)
        for i in range(n_int):
            t_sel = (self.t >= t_edges[i]) & (self.t <= t_edges[i + 1])
            y_mins[i] = self.y[t_sel].min()

        y_base = interp_spline(self.t, t_bins, y_mins)
        y_base += np.percentile(self.y - y_base, 5)
        return y_base



    def get_conditioner(self):
        offset = self.y.min()
        scale = self.y.may() - self.y.min()
        self.flattener = Flattener(scale, offset)



    def condition_piece(self, y):
        return (y - self.flattener.offset) / self.flattener.scale



    def uncondition_piece(self, y):
        return y * self.flattener.scale + self.flattener.offset


    def get_fk(self, fmin, fmax):
        freq = np.fft.fftfreq(self.m, d=self.t[1] - self.t[0])
        dfreq = freq[1] - freq[0]
        fmin_actual = freq[np.argmin(np.abs(freq - fmin))]
        if np.isclose(fmin_actual, 0): fmin_actual = dfreq
        fmax_actual = freq[np.argmin(np.abs(freq - fmax))]

        self.fk = np.arange(fmin_actual, fmax_actual + dfreq, dfreq )
        self.fk_size = self.fk.size


    def generate_starting_guess(self, y):
        freq = np.fft.fftfreq(self.m, d=self.t[1] - self.t[0])
        y_fft = np.fft.fft(y) / y.size / self.fk.size
        amp_k = np.sqrt(np.interp(2 * self.fk, freq, np.abs(y_fft)) * 4)
        ph_k = np.interp(2 * self.fk, freq, np.angle(y_fft)) / 2

        ak_0 = amp_k * np.cos(ph_k)
        bk_0 = -amp_k * np.sin(ph_k)

        def resid_loc(theta):
            return self._y_resid(theta, y)



            # return np.hstack((r, self.mu * p))
            # return np.hstack((r, self.mu * (self.Afft @ self.oscPart_fiy_freq(p))))

        theta0 = np.hstack((ak_0, bk_0))
        theta0 = least_squares(resid_loc, theta0, method='lm', jac=self._grady_resid)['x']

        # ak_0, bk_0 = theta0[:self.fk_size], theta0[self.fk_size:]
        return theta0

    #
    # def sineSum(self, pars):
    #     amps = pars[:self.fk_size]
    #     bmps = pars[self.fk_size:]
    #
    #     y = np.zeros(self.t.shape)
    #     for amp, bmp, f in zip(amps, bmps, self.fk):
    #         y += (amp * np.cos(2 * np.pi * f * self.t) +
    #               bmp * np.sin(2 * np.pi * f * self.t))
    #     return y
    #
    # def F(self, pars):
    #     return self.sineSum(pars) ** 2
    #
    #
    #
    # def fit_resid_nonparametric(self, p):
    #     y_osc = self.F(p)
    #     y_bkg = self.get_y_bkg(y_osc)
    #     r = y_osc + y_bkg - self.y
    #     return r
    #     # return np.hstack((r, self.mu*p))
    #     # return np.hstack((r, self.mu*(self.Afft @ y_osc)))
    #
    # def fit_resid_nonparametric_jac(self, p):
    #     cos = np.cos(2 * np.pi * self.fk[None, :] * self.t[:, None])
    #     sin = np.sin(2 * np.pi * self.fk[None, :] * self.t[:, None])
    #     j = np.hstack((cos, sin))
    #     j *= 2 * self.sineSum(p)[:, None]
    #     return j
    #     # return np.vstack((j, self.mu*np.eye(p.size)))
    #     # return np.vstack((j, self.mu*(self.Afft @ j)))
    #
    #
    #
    #
    # def get_y_bkg(self, y_osc):
    #     y_bkg = self.ridge.solve(self.y - y_osc)
    #     return y_bkg




def osc_sum(pars, fs, t):
    n = int(pars.size/2)
    amps = pars[:n]
    bmps = pars[n:]

    y = np.zeros(t.shape)
    for amp, bmp, f in zip(amps, bmps, fs):
        y += (amp * np.cos(2 * np.pi * f * t) +
              bmp * np.sin(2 * np.pi * f * t))
    return y


def osc_sum_square(pars, fs, t):
    return osc_sum(pars, fs, t) ** 2


def grad_osc_sum_square(p, fs, t):
    cos = np.cos(2 * np.pi * fs[None, :] * t[:, None])
    sin = np.sin(2 * np.pi * fs[None, :] * t[:, None])
    j = np.hstack((cos, sin))
    j *= 2 * osc_sum(p, fs, t)[:, None]
    return j


class Flattener:
    def __init__(self, scale=None, offset=None):
        self.scale = scale
        self.offset = offset




def interp_spline(y, yp, fp):
    cs = CubicSpline(yp, fp)
    return cs(y)


def L2(N):
    L = np.zeros((N - 2, N))
    for i in range(N - 2):
        L[i, i] = 1
        L[i, i + 1] = -2
        L[i, i + 2] = 1
    return L


def L1(N):
    L = np.zeros((N - 1, N))
    for i in range(N - 1):
        L[i, i] = 1
        L[i, i + 1] = -1
    return L



def L(N, n):
    L = np.eye(N)
    for i in range(0, n):
        L = L1(N-i) @ L
    return L



def L1_t(t):
    N = t.size
    t_diff = np.diff(t)
    t_diff /= np.min(t_diff)
    L = np.zeros((N - 1, N))
    for i in range(N - 1):
        L[i, i] = 1
        L[i, i + 1] = -1
    return L / t_diff[:, None]



class LRidge:

    def __init__(self, t, kappa):
        A = np.eye(t.size)
        #            L = L2(y_osc.size)
        self.L = L1_t(t)
        self.At = np.vstack((A, kappa * self.L))
        self.bt = np.zeros(t.size * 2 - 1)


    def solve(self, b):
        self.bt[:b.size] = b
        y, _, _, _ = np.linalg.lstsq(self.At, self.bt, rcond=-1)
        return y



class LRidge_spline:

    def __init__(self, t, dof, deg, kappa):
        str_input = f"bs(y, df={dof}, degree={deg}, include_intercept=True) - 1"
        self.A = dmatrix(str_input, {"y": t})
        self.L = L2(dof)
        self.At = np.vstack((self.A, kappa * self.L))
        self.bt = np.zeros(t.size + dof - 2)

    def solve(self, b):
        self.bt[:b.size] = b
        y, _, _, _ = np.linalg.lstsq(self.At, self.bt, rcond=-1)
        self.penalty = np.linalg.norm(self.L @ y)**2
        self.resid = np.linalg.norm(self.A @ y - b)**2
        return self.A @ y


# def find_curvature(y,y):
#
#     phi = np.linspace(0, 2*np.pi, 361)
#
#     plt.figure(25)
#     plt.clf()
#
#     c = np.zeros(y.size-2)
#     for i in range(1,y.size-1):
#         y1, y2, y3 = y[i-1], y[i], y[i+1]
#         y1, y2, y3 = y[i-1], y[i], y[i+1]
#         r, y0, y0 =  find_circle(y1, y1, y2, y2, y3, y3)
#         c[i - 1] = 1/r
#         plt.plot([y[i-1], y[i], y[i+1]], [y[i-1], y[i], y[i+1]], 'k.')
#         plt.plot(y0 + np.abs(r) * np.cos(phi), y0 + np.abs(r) * np.sin(phi), 'r-')
#
#     return c
#
#
# def find_circle(y1, y1, y2, y2, y3, y3):
#
#     A = np.array([[y2-y1, y2-y1],
#                   [y3-y1, y3-y1]])
#     b = 0.5*np.array([y2**2 - y1**2 + y2**2 - y1**2,
#                       y3**2 - y1**2 + y3**2 - y1**2])
#     center,_,_,_ = np.linalg.lstsq(A, b, rcond=-1)
#     y0, y0 = center
#     r = np.sqrt((y0-y1)**2 +(y0-y1)**2)
#
#     if (y0<y2) & (y0<y2):
#         sgn = -1
#     else:
#         sgn = 1
#     return sgn*r, y0, y0
#
#     # print("Centre = (", h, ", ", k, ")");
#     # print("Radius = ", r);



class SpectrumDenoising:

    def __init__(self, t, y, w=0.07):
        self.t = t - t.min()
        self.y = y

        self.T = t.max()-t.min()
        self.nw = np.floor(self.T / w)

        self.t_edges = np.linspace(0, self.T, self.nw+1)
        self.theta = None



    def denoise(self):

        # for idy_cen in [16750]:  # mono glitch
        # for idy_cen in [21730]:  # mono glitch
            #        for idy_cen in range(16750-10, 16750+10): # mono glitch
        self.y_smooth = np.zeros(self.y.size)
        # for i in range(self.nw):
        for i in [76]:
            cur_time = time.time_ns()
            idxs = self.choose_piece(i)
            # piece = DenoisePiece(self.t[idys], self.y[idys], self.p, self.kappa)
            plt.figure(99)
            plt.clf()

            plt.plot(self.t, self.y, 'k-')
            plt.plot(self.t[idxs], self.y[idxs], 'r-')

            piece = DenoisePiece(self.t[idxs], self.y[idxs], self.theta)
            piece.process()
            # piece.optimize()
            # self.y_smooth[idys] = piece.y_smooth
            print(i, 'took %.1f' % ((time.time_ns() - cur_time) * 1e-6), 'ms')


    def choose_piece(self, i):
        return (self.t>=self.t_edges[i]) & (self.t<=self.t_edges[i+1])







