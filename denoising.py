import numpy as np
import matplotlib.pyplot as plt

import time
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from sklearn.model_selection import KFold


class DenoisePiece:

    def __init__(self, t, x, p0, w_base=0.005, F_low=40, F_high=150, fk_0=None):
        self.t = t
        self.x = x
        self.dt = t[1] - t[0]

        self.npt = t.size
        self.F = np.fft.fftfreq(self.npt, d=self.dt)
        self.dF = self.F[1] - self.F[0]
        self.F_low, self.F_high = F_low, F_high
        if fk_0 is None: self.get_fk_0()
        else: self.fk_0, self.fk_0_size = fk_0, fk_0.size



        self.get_conditioner()
        self.x = self.condition_piece(self.x)


        self.p0 = p0
        # if kappa:
        #     self.kappa = kappa
        # else:
        #     self.kappa = 5e0
        #
        # self.ridge = LRidge(self.t, self.kappa)



    # def cross_validate(self, kappas=None, n_splits=2):
    #     self.process()
    #     if not kappas:
    #         kappas = 10**np.linspace(-1, 1, 11)
    #     kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    #     gen_error = np.zeros((kappas.size, n_splits))
    #     # for train_index, test_index in kf.split(self.t):
    #     print([i for i in kf.split(self.t)][0])
    #     for train_index, test_index in [[i for i in kf.split(self.t)][0]]:
    #         t_train, t_test = self.t[train_index], self.t[test_index]
    #         x_train, x_test = self.x[train_index], self.x[test_index]
    #         # for kappa in kappas:
    #         for kappa in [5e0]:
    #             print('bla')
    #             dp_cv = DenoisePiece(t_train, x_train, self.p0, kappa, fk_0=self.fk_0)
    #             dp_cv.process(plotting=True)
    #
    #             plt.figure(3)
    #             plt.subplot(211)
    #             plt.plot(self.t, self.x, '.-', color=[0.7, 0.7, 0.7])


    def optimize(self, kappas=None):
        # self.process()
        if not kappas:
            kappas = np.hstack((10**-3, 10**np.linspace(-1, 1, 105), 10**3))

        rho = np.zeros(kappas.shape)
        ksi = np.zeros(kappas.shape)
        for k, kappa in enumerate(kappas):
            self.kappa = kappa
            self.process()
            # rho[k] = np.log10(np.sum((self.x_fit - self.x)**2)) # SE
            # ksi[k] = np.log10(np.sum((self.ridge.L @ self.x_smooth)**2)) # reg norm
            rho[k] = (np.sum((self.x_fit - self.x) ** 2))  # SE
            ksi[k] = (np.sum((self.ridge.L @ self.x_smooth) ** 2))  # reg norm


        # drho, dksi = np.diff(np.log(rho)), np.diff(np.log(ksi))
        # drho, dksi = np.diff(rho), np.diff(ksi)
        # ddrho, ddksi = np.diff(drho), np.diff(dksi)

        # curvature = 2 * (drho[:-1]*ddksi - ddrho*dksi[:-1]) / (drho[:-1]**2 + dksi[:-1]**2)**1.5
        # curvature = find_curvature(rho, ksi)
        # idx = np.argmax(curvature) + 1

        fom = np.sqrt(((rho-rho.min())/(rho.max()-rho.min()))**2 +
                      ((ksi-ksi.min())/(ksi.max()-ksi.min()))**2)
        idx = np.argmin(fom)

        plt.figure(4)
        plt.clf()
        plt.subplot(211)
        plt.loglog(rho, ksi, 'k.-')
        plt.loglog(rho[idx], ksi[idx], 'ro')
        # plt.plot(rho, ksi, 'k.-')
        # plt.plot(rho[idx], ksi[idx], 'ro')

        plt.subplot(212)
        # plt.semilogx(kappas[1:-1], curvature, 'k.-')
        plt.semilogx(kappas, fom, 'k.-')
        # plt.vlines(kappas[idx], curvature.min(), curvature.max(), colors='r')
        plt.vlines(kappas[idx], fom.min(), fom.max(), colors='r')
        # plt.loglog(SE, PN, 'k.-')

        self.kappa = kappas[idx]
        self.process(plotting=True)







    def process(self, w_base=0.005, plotting=False):


        if self.p0 is None:
            x_base = self.predef_baseline(w_base)
            self.p0 = self.generate_starting_guess(self.x - x_base)

        self.ridge = LRidge(self.t, self.kappa)
        self.p = least_squares(self.fit_resid_nonparametric, self.p0,
                                method='lm', jac=self.fit_resid_nonparametric_jac)['x']

        osci = self.oscPart_fix_freq(self.p)
        bottom = self.get_x_bkg(osci)
        self.x_fit = osci + bottom
        self.x_smooth = bottom

        # self.x = self.uncondition_piece(self.x)
        # self.x_fit = self.uncondition_piece(self.x_fit)
        # self.x_smooth = self.uncondition_piece(self.x_smooth)

        if plotting:
            plt.figure(3)
            plt.clf()

            plt.subplot(211)
            plt.plot(self.t, self.x, 'k.-')
            plt.plot(self.t, self.x_fit, 'r-')
            plt.plot(self.t, self.x_smooth, 'b--')
            #
            plt.subplot(212)
            # plt.plot(self.fk_0, np.sqrt(self.p0[:self.fk_0_size]**2 + self.p0[self.fk_0_size:]**2), 'k.-')
            plt.plot(self.fk_0, self.p0[:self.fk_0_size], 'r.-')
            plt.plot(self.fk_0, self.p0[self.fk_0_size:], 'b.-')
            # plt.plot(self.fk_0, np.angle(1j*self.p0[:self.fk_0_size] + self.p0[self.fk_0_size:]), 'k.-')


    def predef_baseline(self, w_base):
        n_int = int(np.ceil((self.t.max() - self.t.min()) / w_base))
        t_edges = np.linspace(self.t.min(), self.t.max(), n_int + 1)
        t_bins = t_edges[:-1] + np.diff(t_edges) / 2
        x_mins = np.zeros(t_bins.shape)
        for i in range(n_int):
            t_sel = (self.t >= t_edges[i]) & (self.t <= t_edges[i + 1])
            x_mins[i] = self.x[t_sel].min()

        x_base = interp_spline(self.t, t_bins, x_mins)
        x_base += np.percentile(self.x - x_base, 5)
        return x_base



    def get_conditioner(self):
        offset = self.x.min()
        scale = self.x.max() - self.x.min()
        self.flattener = Flattener(scale, offset)



    def condition_piece(self, x):
        return (x - self.flattener.offset) / self.flattener.scale



    def uncondition_piece(self, x):
        return x * self.flattener.scale + self.flattener.offset


    def get_fk_0(self):
        F_low_actual = self.F[np.argmin(np.abs(self.F - self.F_low))]
        if np.isclose(F_low_actual, 0): F_low_actual = self.dF
        F_high_actual = self.F[np.argmin(np.abs(self.F - self.F_high))]

        self.fk_0 = np.arange(F_low_actual, F_high_actual, self.dF / 2)
        self.fk_0_size = self.fk_0.size



    def generate_starting_guess(self, x):
        x_fft = np.fft.fft(x) / x.size / self.fk_0.size
        amp_k = np.sqrt(np.interp(2 * self.fk_0, self.F, np.abs(x_fft)) * 4)
        ph_k = np.interp(2 * self.fk_0, self.F, np.angle(x_fft)) / 2

        ak_0 = amp_k * np.cos(ph_k)
        bk_0 = -amp_k * np.sin(ph_k)

        def resid_loc(p):
            r = x - self.oscPart_fix_freq(p)
            return r

        p0 = np.hstack((ak_0, bk_0))
        p0 = least_squares(resid_loc, p0, method='lm', jac=self.fit_resid_nonparametric_jac)['x']

        # ak_0, bk_0 = p0[:self.fk_0_size], p0[self.fk_0_size:]
        return p0


    def sineSum(self, pars):
        amps = pars[:self.fk_0_size]
        bmps = pars[self.fk_0_size:]

        x = np.zeros(self.t.shape)
        for amp, bmp, f in zip(amps, bmps, self.fk_0):
            x += (amp * np.cos(2 * np.pi * f * self.t) +
                  bmp * np.sin(2 * np.pi * f * self.t))
        return x

    def oscPart_fix_freq(self, pars):
        return self.sineSum(pars) ** 2

    def polyPart(self, pars):
        x = np.zeros(self.t.shape)
        for i, p in enumerate(pars):
            x += p * self.t ** i
        return x

    def sineSquare_fix_freq(self, pars, full_output=False):
        pars_osc = pars[:self.fk_0_size * 2]
        pars_bkg = pars[self.fk_0_size * 2:]
        s_osc = self.oscPart_fix_freq(pars_osc)
        s_poly = self.polyPart(pars_bkg)

        if full_output:
            return s_poly + s_osc, s_poly, s_osc
        else:
            return s_poly + s_osc

    def fit_resid(self, p):
        r = (self.x - self.sineSquare_fix_freq(p))
        #        return np.hstack((r, np.sqrt(np.abs(1e-12*p[self.fk_0_size*2:]))))
        #        return np.hstack((r, 1e-4*p[self.fk_0_size*2:]))
        return r

    def fit_resid_nonparametric(self, p):
        x_osc = self.oscPart_fix_freq(p)
        x_bkg = self.get_x_bkg(x_osc)
        r = x_osc + x_bkg - self.x
        return r

    def fit_resid_nonparametric_jac(self, p):
        cos = np.cos(2 * np.pi * self.fk_0[None, :] * self.t[:, None])
        sin = np.sin(2 * np.pi * self.fk_0[None, :] * self.t[:, None])
        j = np.hstack((cos, sin))
        j *= 2 * self.sineSum(p)[:, None]
        return j

    def get_x_bkg(self, x_osc):
        x_bkg = self.ridge.solve(self.x - x_osc)
        return x_bkg





class Flattener:
    def __init__(self, scale=None, offset=None):
        self.scale = scale
        self.offset = offset




def interp_spline(x, xp, fp):
    cs = CubicSpline(xp, fp)
    return cs(x)


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
        #            L = L2(x_osc.size)
        self.L = L1_t(t)
        self.At = np.vstack((A, kappa * self.L))
        self.bt = np.zeros(t.size * 2 - 1)


    def solve(self, b):
        self.bt[:b.size] = b
        x, _, _, _ = np.linalg.lstsq(self.At, self.bt, rcond=-1)
        return x



def find_curvature(x,y):

    phi = np.linspace(0, 2*np.pi, 361)

    plt.figure(25)
    plt.clf()

    c = np.zeros(y.size-2)
    for i in range(1,y.size-1):
        x1, x2, x3 = x[i-1], x[i], x[i+1]
        y1, y2, y3 = y[i-1], y[i], y[i+1]
        r, x0, y0 =  find_circle(x1, y1, x2, y2, x3, y3)
        c[i - 1] = 1/r
        plt.plot([x[i-1], x[i], x[i+1]], [y[i-1], y[i], y[i+1]], 'k.')
        plt.plot(x0 + np.abs(r) * np.cos(phi), y0 + np.abs(r) * np.sin(phi), 'r-')

    return c


def find_circle(x1, y1, x2, y2, x3, y3):

    A = np.array([[x2-x1, y2-y1],
                  [x3-x1, y3-y1]])
    b = 0.5*np.array([x2**2 - x1**2 + y2**2 - y1**2,
                      x3**2 - x1**2 + y3**2 - y1**2])
    center,_,_,_ = np.linalg.lstsq(A, b, rcond=-1)
    x0, y0 = center
    r = np.sqrt((x0-x1)**2 +(y0-y1)**2)

    if (x0<x2) & (y0<y2):
        sgn = -1
    else:
        sgn = 1
    return sgn*r, x0, y0

    # print("Centre = (", h, ", ", k, ")");
    # print("Radius = ", r);



class SpectrumDenoising:

    def __init__(self, t, x, w=0.05):
        self.t = t - t.min()
        self.x = x

        self.T = t.max()-t.min()
        self.nw = np.floor(self.T / w)

        self.t_edges = np.linspace(0, self.T, self.nw+1)
        self.p, self.kappa = None, None



    def denoise(self):

        # for idx_cen in [16750]:  # mono glitch
        # for idx_cen in [21730]:  # mono glitch
            #        for idx_cen in range(16750-10, 16750+10): # mono glitch
        self.x_smooth = np.zeros(self.x.size)
        # for i in range(self.nw):
        for i in [100]:
            cur_time = time.time_ns()
            idxs = self.choose_piece(i)
            # piece = DenoisePiece(self.t[idxs], self.x[idxs], self.p, self.kappa)
            piece = DenoisePiece(self.t[idxs], self.x[idxs], self.p)
            # piece.process()
            piece.optimize()
            # self.x_smooth[idxs] = piece.x_smooth
            print(i, 'took %.1f' % ((time.time_ns() - cur_time) * 1e-6), 'ms')


    def choose_piece(self, i):
        return (self.t>=self.t_edges[i]) & (self.t<=self.t_edges[i+1])







