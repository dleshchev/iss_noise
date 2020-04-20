import numpy as np
import matplotlib.pyplot as plt

import time
from scipy.optimize import least_squares


class Denoising:

    def __init__(self, t, x, w=0.05):
        self.t = t
        self.x = x

        self.w = w

        self.dt = t[1] - t[0]
        self.n_t = t.size

        self.nptw = np.sum(t < w)  # define number of descrete points in the window
        if self.nptw % 2 == 0:  # make it odd
            self.nptw += 1

        self.F = np.fft.fftfreq(self.nptw, d=self.dt)
        self.dF = self.F[1] - self.F[0]

        self.p0 = None
        self.At = None
        self.kappa = 1e0

    def denoise(self):

        for idx_cen in [16750]:  # mono glitch
            #        for idx_cen in range(16750-10, 16750+10): # mono glitch

            #        for idx_cen in [18750]: # xanes
            #        for idx_cen in [2000]: # flat
            cur_time = time.time_ns()
            self.choose_piece(idx_cen)
            self.denoise_piece()
            print(idx_cen, 'took %.1f' % ((time.time_ns() - cur_time) * 1e-6), 'ms')

    def choose_piece(self, idx_cen):
        idx1 = np.max((idx_cen - int((self.nptw - 1) / 2), 0))
        idx2 = np.min((idx_cen + int((self.nptw - 1) / 2 + 1), self.n_t))
        self.tw_cur = self.t[idx1:idx2]
        self.xw_cur = self.x[idx1:idx2]

    #        assert self.xw_cur.size == self.tw_cur.size == self.nptw, 'size of selected signal is not sufficient'

    def denoise_piece(self, F_low=20, F_high=200, w_base=0.005):
        self.get_conditioner()

        plt.figure(99)
        plt.clf()
        #        plt.plot(self.t, self.x)
        #        plt.plot(self.tw_cur, self.xw_cur)

        self.xw_cur = self.condition_piece(self.xw_cur)
        x_base = self.predef_baseline(w_base)

        #        plt.plot(self.tw_cur, self.xw_cur, 'k.-')
        #        plt.plot(self.tw_cur, x_base)
        #        plt.plot(self.tw_cur, self.xw_cur - x_base)

        if self.p0 is None:
            self.p0 = self.generate_starting_guess(self.xw_cur - x_base, F_low, F_high)

        self.p0 = least_squares(self.fit_resid_nonparametric, self.p0,
                                method='lm', jac=self.fit_resid_nonparametric_jac)['x']

        #        osci = self.oscPart_fix_freq(self.p0)
        #        bottom = self.get_x_bkg(osci)
        #        plt.plot(self.tw_cur, bottom + osci, 'r-')
        #        plt.plot(self.tw_cur, bottom, 'b--')

        osci = self.oscPart_fix_freq(self.p0)
        bottom = self.get_x_bkg(osci)
        self.xw_cur_est = osci + bottom
        self.xw_cur_optimum = bottom
        ###
        ###
        ###
        #        result = least_squares(self.fit_resid, self.p0)
        #        self.p0 = result['x']
        #        self.xw_cur_est, self.xw_cur_optimum, _ = self.sineSquare_fix_freq(self.p0,
        #                                                                          full_output=True)
        self.xw_cur = self.uncondition_piece(self.xw_cur)
        self.xw_cur_est = self.uncondition_piece(self.xw_cur_est)
        self.xw_cur_optimum = self.uncondition_piece(self.xw_cur_optimum)

        #
        #        plt.figure(99)
        #        plt.clf()
        #
        plt.plot(self.tw_cur, self.xw_cur, 'k.-')
        plt.plot(self.tw_cur, self.xw_cur_est, 'r-')
        plt.plot(self.tw_cur, self.xw_cur_optimum, 'b--')

    #
    #        plt.xlim(self.tw_cur.min() - self.w/2, self.tw_cur.max() + self.w/2)
    #        plt.ylim(self.xw_cur.min() - self.flattener.scale*0.5, self.xw_cur.max() + self.flattener.scale*0.5)

    def predef_baseline(self, w):
        n_int = int(np.ceil(self.w / w))
        t_edges = np.linspace(self.tw_cur.min(), self.tw_cur.max(), n_int + 1)
        t_bins = t_edges[:-1] + np.diff(t_edges) / 2
        x_mins = np.zeros(t_bins.shape)
        for i in range(n_int):
            t_sel = (self.tw_cur >= t_edges[i]) & (self.tw_cur <= t_edges[i + 1])
            x_mins[i] = self.xw_cur[t_sel].min()

        #        p = np.polyfit(t_bins, x_base, n_poly)
        #        p[-1] += -np.mean(x_base) + np.mean(self.xw_cur)

        x_base = interp_spline(self.tw_cur, t_bins, x_mins)
        x_base += np.percentile(self.xw_cur - x_base, 5)

        #        plt.plot(t_bins, x_mins, 'b.')
        #        plt.plot(self.tw_cur, x_base, 'r-')
        return x_base

    def get_conditioner(self):
        offset = self.xw_cur.min()
        scale = self.xw_cur.max() - self.xw_cur.min()
        self.flattener = Flattener(scale, offset)

    def condition_piece(self, x):
        return (x - self.flattener.offset) / self.flattener.scale

    def uncondition_piece(self, x):
        return x * self.flattener.scale + self.flattener.offset

    def generate_starting_guess(self, xw, F_low, F_high):

        F_low_actual = self.F[np.argmin(np.abs(self.F - F_low))]
        if np.isclose(F_low_actual, 0): F_low_actual = self.dF
        F_high_actual = self.F[np.argmin(np.abs(self.F - F_high))]

        self.fk_0 = np.arange(F_low_actual, F_high_actual, self.dF)
        self.fk_0_size = self.fk_0.size

        xw_fft = np.fft.fft(xw) / xw.size / self.fk_0.size
        amp_k = np.sqrt(np.interp(2 * self.fk_0, self.F, np.abs(xw_fft)) * 4)
        ph_k = np.interp(2 * self.fk_0, self.F, np.angle(xw_fft)) / 2

        ak_0 = amp_k * np.cos(ph_k)
        bk_0 = -amp_k * np.sin(ph_k)

        def resid_loc(p):
            r = xw - self.oscPart_fix_freq(p)
            return r

        p0 = np.hstack((ak_0, bk_0))
        p0 = least_squares(resid_loc, p0, method='lm', jac=self.fit_resid_nonparametric_jac)['x']

        #        plt.plot(self.tw_cur, self.oscPart_fix_freq(p0))

        ak_0, bk_0 = p0[:self.fk_0_size], p0[self.fk_0_size:]

        #        plt.figure(100)
        #        plt.clf()
        #        plt.subplot(211)
        #        plt.plot(self.fk_0, np.sqrt(ak_0**2 + bk_0**2), 'k.-')
        #
        #        plt.subplot(212)
        #        plt.plot(self.fk_0, ak_0, 'b.-')
        #        plt.plot(self.fk_0, bk_0, 'r.-')

        #        p0_poly = np.zeros(n_poly+1)
        #        p0_poly[0] = self.xw_cur.min()

        #        return np.hstack((ak_0, bk_0, p0_poly))
        return p0

    def sineSum(self, pars):
        amps = pars[:self.fk_0_size]
        bmps = pars[self.fk_0_size:]

        x = np.zeros(self.tw_cur.shape)
        for amp, bmp, f in zip(amps, bmps, self.fk_0):
            x += (amp * np.cos(2 * np.pi * f * self.tw_cur) +
                  bmp * np.sin(2 * np.pi * f * self.tw_cur))
        return x

    def oscPart_fix_freq(self, pars):
        return self.sineSum(pars) ** 2

    def polyPart(self, pars):
        x = np.zeros(self.tw_cur.shape)
        for i, p in enumerate(pars):
            x += p * self.tw_cur ** i
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
        r = (self.xw_cur - self.sineSquare_fix_freq(p))
        #        return np.hstack((r, np.sqrt(np.abs(1e-12*p[self.fk_0_size*2:]))))
        #        return np.hstack((r, 1e-4*p[self.fk_0_size*2:]))
        return r

    def fit_resid_nonparametric(self, p):
        x_osc = self.oscPart_fix_freq(p)
        x_bkg = self.get_x_bkg(x_osc)
        r = x_osc + x_bkg - self.xw_cur
        return r

    def fit_resid_nonparametric_jac(self, p):
        cos = np.cos(2 * np.pi * self.fk_0[None, :] * self.tw_cur[:, None])
        sin = np.sin(2 * np.pi * self.fk_0[None, :] * self.tw_cur[:, None])
        j = np.hstack((cos, sin))
        j *= 2 * self.sineSum(p)[:, None]
        return j

    def get_x_bkg(self, x_osc):
        if self.At is None:
            A = np.eye(x_osc.size)
            #            L = L2(x_osc.size)
            L = L1(x_osc.size)
            self.At = np.vstack((A, self.kappa * L))
            #            self.bt = np.zeros(x_osc.size*2-2)
            self.bt = np.zeros(x_osc.size * 2 - 1)

        self.bt[:x_osc.size] = self.xw_cur - x_osc
        x_bkg, _, _, _ = np.linalg.lstsq(self.At, self.bt, rcond=-1)
        return x_bkg


class Flattener:
    def __init__(self, scale=None, offset=None, x_flattener=None):
        self.scale = scale
        self.offset = offset
        self.x_flattener = x_flattener


from scipy.interpolate import CubicSpline


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