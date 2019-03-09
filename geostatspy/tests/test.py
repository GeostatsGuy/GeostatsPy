import numpy as np
import pandas as pd
import unittest

from geostatspy import geostats


class UnitTest(unittest.TestCase):

    def setUp(self):
        # Any inputs go here
        self.sample_data = pd.read_csv('sample_data.csv', sep=',')

        return

    def test_locate(self):
        # Inputs
        # xx =
        # iis =
        # iie =
        # x =

        # Test answers
        # j_test =

        # Call function
        # j_output = geostats.locate(xx, iis, iie, x)

        # Testing
        # np.testing.assert_allclose(j_output, j_test, atol=0.01)

        return

    def test_dlocate(self):
        # Inputs
        # xx =
        # iis =
        # iie =
        # x =

        # Test answers
        # j_test =

        # Call function
        # j_output = geostats.locate(xx, iis, iie, x)

        # Testing
        # np.testing.assert_allclose(j_output, j_test, atol=0.01)

        return

    def test_powint(self):
        # Inputs
        # xlow =
        # xhigh =
        # ylow =
        # yhigh =
        # xval =
        # power =

        # Test answers
        # powint_test =

        # Call function
        # powint_output = geostats.powint(self, xlow, xhigh, ylow, yhigh, xval, power)

        # Testing
        # np.testing.assert_allclose(powint_output, powint_test, atol=0.01)

        return

    def test_dsortem(self):
        # Inputs
        # ib =
        # ie =
        # a =
        # iperm =

        # Test answers
        # a_test =
        # b_test =
        # c_test =
        # d_test =
        # e_test =
        # f_test =
        # g_test =
        # h_test =
        # dsortem_test = [a_test, b_test, c_test, d_test, e_test, f_test, g_test, h_test]

        # Call function
        # dsortem_output = geostats.dsortem(ib, ie, a, iperm, b=0, c=0, d=0, e=0, f=0, g=0, h=0)

        # Testing
        # np.testing.assert_allclose(dsortem_output, dsortem_test, atol=0.01)

        return

    def test_gauinv(self):
        # Inputs
        # p =

        # Test answers
        # xp_test =

        # Call function
        # xp_output = geostats.gauinv(p)

        # Testing
        # np.testing.assert_allclose(xp_output, xp_test, atol=0.01)

        return

    def test_gcum(self):
        # Inputs
        # x =

        # Test answers
        # gcum_test =

        # Call function
        # gcum_output = geostats.gcum(x)

        # Testing
        # np.testing.assert_allclose(gcum_output, gcum_test, atol=0.01)

        return

    def test_dpowint(self):
        # Inputs
        # xlow =
        # xhigh =
        # ylow =
        # yhigh =
        # xval =
        # pwr =

        # Test answers
        # dpowint_test =

        # Call function
        # dpowint_output = geostats.dpowint(xlow, xhigh, ylow, yhigh, xval, pwr)

        # Testing
        # np.testing.assert_allclose(lag_output, lag_test, atol=0.01)

        return

    def test_setup_rotmat(self):
        # Inputs
        # c0 =
        # nst =
        # it =
        # cc =
        # ang =
        # pmx =

        # Test answers
        # rotmat_test =
        # maxcov_test =

        # Call function
        # rotmat_output, maxcov_output = geostats.setup_rotmat(c0, nst, it, cc, ang, pmx)

        # Testing
        # np.testing.assert_allclose(rotmat_output, rotmat_test, atol=0.01)
        # np.testing.assert_allclose(maxcov_output, maxcov_test, atol=0.01)

        return

    def test_cova2(self):
        # Inputs
        # x1 =
        # y1 =
        # x2 =
        # y2 =
        # nst =
        # c0 =
        # pmx =
        # cc =
        # aa =
        # it =
        # ang =
        # anis =
        # rotmat =
        # maxcov =

        # Test answers
        # cova2_test =

        # Call function
        # cova2_output = geostats.cova2(x1, y1, x2, y2, nst, c0, pmx, cc, aa, it, ang, anis, rotmat, maxcov)

        # Testing
        # np.testing.assert_allclose(cova2_output, cova2_test, atol=0.01)

        return

    def test_ksol_numpy(self):
        # Inputs
        # neq =
        # a =
        # r =

        # Test answers
        # s_test =

        # Call function
        # s_output = geostats.ksol_numpy(neq, a, r)

        # Testing
        # np.testing.assert_allclose(s_output, s_test, atol=0.01)

        return

    def test_correct_trend(self):
        # Inputs
        # trend =

        # Test answers
        # trend_test =

        # Call function
        # trend_output = geostats.correct_trend(trend)

        # Testing
        # np.testing.assert_allclose(trend_output, trend_test, atol=0.01)

        return

    def test_ordrel(self):
        # Inputs
        # ivtype =

        # Test answers
        # ccdfo_test =

        # Call function
        # ccdfo_output = geostats.ordrel(ivtype)

        # Testing
        # np.testing.assert_allclose(ccdfo_output, ccdfo_test, atol=0.01)

        return

    def test_declus(self):
        # Inputs
        # df =
        # xcol =
        # ycol =
        # vcol =
        # iminmax =
        # noff =
        # ncell =
        # cmin =
        # cmax =

        # Test answers
        # wtopt_test =
        # xcs_mat_test =
        # vrcr_mat_test =

        # Call function
        # wtopt_output, xcs_mat_output, vrcr_mat_output = geostats.declus(df, xcol, ycol, vcol, iminmax, noff, ncell, cmin, cmax)

        # Testing
        # np.testing.assert_allclose(wtopt_output, wtopt_test, atol=0.01)
        # np.testing.assert_allclose(xcs_mat_output, xcs_mat_test, atol=0.01)
        # np.testing.assert_allclose(vrcr_mat_output, vrcr_mat_test, atol=0.01)

        return

    def test_gam(self):
        # Inputs
        # array =
        # tmin =
        # tmax =
        # xsiz =
        # ysiz =
        # ixd =
        # iyd =
        # nlag =
        # isill =

        # Test answers
        # lag_test =
        # vario_test =
        # npp_test =

        # Call function
        # lag_output, vario_output, npp_output = geostats.gam(array, tmin, tmax, xsiz, ysiz, ixd, iyd, isill)

        # Testing
        # np.testing.assert_allclose(lag_output, lag_test, atol=0.01)
        # np.testing.assert_allclose(vario_output, vario_test, atol=0.01)
        # np.testing.assert_allclose(npp_output, npp_test, atol=0.01)

        return

    def test_gamv(self):
        # Inputs
        df = self.sample_data
        xcol = "X"
        ycol = "Y"
        vcol = "Porosity"
        tmin = 0
        tmax = 1500
        xlag = 100
        xltol = 25
        nlag = 7
        azm = 0
        atol = 90
        bandwh = 25
        isill = 1

        # Test answers
        dis_test = [0.0, 17.31, 93.002, 199.258, 300.349, 400.174, 501.166, 600.773, 0.0]
        vario_test = [0.0, 0.056, 0.139, 0.201, 0.381, 0.357, 1.058, 0.172, 0.0]
        npp_test = [522.0, 5032.0, 2168.0, 504.0, 372.0, 376.0, 312.0, 116.0, 0.0]

        # Call function
        dis_output, vario_output, npp_output = geostats.gamv(df, xcol, ycol, vcol, tmin, tmax, xlag, xltol, nlag, azm,
                                                             atol, bandwh, isill)

        # Testing
        np.testing.assert_allclose(dis_output, dis_test, atol=0.01)
        np.testing.assert_allclose(vario_output, vario_test, atol=0.01)
        np.testing.assert_allclose(npp_output, npp_test, atol=0.01)

        return

    def test_variogram_loop(self):
        # Inputs
        # x =
        # y =
        # vr =
        # xlag =
        # xltol =
        # nlag =
        # azm =
        # atol =
        # bandwh =

        # Test answers
        # dis_test =
        # vario_test =
        # npp_test =

        # Call function
        # dis_output, vario_output, npp_output = geostats.variogram_loop(x, y, vr, xlag, xltol, nlag, azm, atol, bandwh)

        # Testing
        # np.testing.assert_allclose(dis_output, dis_test, atol=0.01)
        # np.testing.assert_allclose(vario_output, vario_test, atol=0.01)
        # np.testing.assert_allclose(npp_output, npp_test, atol=0.01)

        return

    def test_varmapv(self):
        # Inputs
        # df =
        # xcol =
        # ycol =
        # tmin =
        # tmax =
        # nxlag =
        # nylag =
        # dxlag =
        # dylag =
        # minnp =
        # isill =

        # Test answers
        # gamf_test =
        # nppf_test =

        # Call function
        # gamf_output, nppf_output = geostats.varmapv(df, xcol, ycol, vcol, tmin, tmax, nxlag, nylag, dxlag, dylag, minnp, isill)

        # Testing
        # np.testing.assert_allclose(gamf_output, gamf_test, atol=0.01)
        # np.testing.assert_allclose(nppf_output, nppf_test, atol=0.01)

        return

    def test_vmodel(self):
        # Inputs
        # nlag =
        # xlag =
        # azm =
        # vario =

        # Test answers
        # index_test =
        # h_test =
        # gam_test =
        # cov_test =
        # ro_test =

        # Call function
        # index_output, h_output, gam_output, cov_output, ro_output = geostats.vmodel(nlag, xlag, azm, vario)

        # Testing
        # np.testing.assert_allclose(index_output, index_test, atol=0.01)
        # np.testing.assert_allclose(h_output, h_test, atol=0.01)
        # np.testing.assert_allclose(gam_output, gam_test, atol=0.01)
        # np.testing.assert_allclose(cov_output, cov_test, atol=0.01)
        # np.testing.assert_allclose(ro_output, ro_test, atol=0.01)

        return

    def test_nscore(self):
        # Inputs
        # df =
        # vcol =
        # wcol = None
        # ismooth = False
        # dfsmooth = None
        # smcol = 0
        # smwcol = 0

        # Test answers
        # ns_test =
        # vr_test =
        # wt_ns_test =

        # Call function
        # ns_output, vr_output, wt_ns_output = geostats.nscore(df, vcol, wcol=None, ismooth=False, dfsmooth=None, smcol=0, smwcol=0)

        # Testing
        # np.testing.assert_allclose(ns_output, ns_test, atol=0.01)
        # np.testing.assert_allclose(vr_output, vr_test, atol=0.01)
        # np.testing.assert_allclose(wt_ns_output, wt_ns_test, atol=0.01)

        return

    def test_kb2d(self):
        # Inputs
        # df =
        # xcol =
        # ycol =
        # vcol =
        # tmin =
        # tmax =
        # nx =
        # xmn =
        # xsiz =
        # ny =
        # ymn =
        # ysiz =
        # nxdis =
        # nydis =
        # ndmin =
        # ndmax =
        # radius =
        # ktype =
        # skmean =
        # vario =

        # Test answers
        # kmap_test =
        # vmap_test =

        # Call function
        # kmap_output, vmap_output = geostats.kb2d(df, xcol, ycol, vcol, tmin, tmax, nx, xmn, xsiz, ny, ymn, ysiz, nxdis, nydis, ndmin, ndmax, radius, ktype, skmean, vario)

        # Testing
        # np.testing.assert_allclose(kmap_output, kmap_test, atol=0.01)
        # np.testing.assert_allclose(vmap_output, vmap_test, atol=0.01)

        return

    def test_ik2d(self):
        # Inputs
        # df =
        # xcol =
        # ycol =
        # vcol =
        # ivtype =
        # tmax =
        # koption =
        # ncut =
        # thresh =
        # gcdf =
        # trend =
        # tmin =
        # tmax =
        # nx =
        # xmn =
        # xsiz =
        # ny =
        # ymn =
        # ysiz =
        # ndmin =
        # ndmax =
        # radius =
        # ktype =
        # vario =

        # Test answers
        # ikout_test =

        # Call function
        # ikout_output = geostats.ik2d(df, xcol, ycol, vcol, ivtype, koption, ncut, thresh, gcdf, trend, tmin, tmax, nx, xmn, xsiz, ny, ymn, ysiz, ndmin, ndmax, radius, ktype, vario)

        # Testing
        # np.testing.assert_allclose(ikout_output, ikout_test, atol=0.01)

        return

    if __name__ == '__main__':
        unittest.main()
