"""
Unit tests for the get_pc_type routing function.

These tests verify the dataset-to-power-correction-type mapping without
requiring the NNPDF infrastructure.
"""

import pytest

from validphys.theorycovariance.higher_twist_functions import (
    get_pc_type,
    F2P_exps,
    F2D_exps,
    NC_SIGMARED_P_EM,
    NC_SIGMARED_P_EP,
    NC_SIGMARED_P_EAVG,
)


class TestDISRouting:
    """Tests for DIS dataset -> PC type mapping."""

    @pytest.mark.parametrize("exp_name", F2P_exps)
    def test_f2_proton_experiments(self, exp_name):
        assert get_pc_type(exp_name, "DIS_NCE") == "f2p"

    @pytest.mark.parametrize("exp_name", F2D_exps)
    def test_f2_deuteron_experiments(self, exp_name):
        assert get_pc_type(exp_name, "DIS_NCE") == "f2d"

    def test_nmc_ratio_returns_tuple(self):
        result = get_pc_type("NMC_NC_NOTFIXED_EM-F2", "DIS_NCE")
        assert result == ("f2p", "f2d")

    @pytest.mark.parametrize("exp_name", NC_SIGMARED_P_EM)
    def test_nc_sigmared_em(self, exp_name):
        assert get_pc_type(exp_name, "DIS_NCE") == "f2p"

    @pytest.mark.parametrize("exp_name", NC_SIGMARED_P_EP)
    def test_nc_sigmared_ep(self, exp_name):
        assert get_pc_type(exp_name, "DIS_NCE") == "f2p"

    @pytest.mark.parametrize("exp_name", NC_SIGMARED_P_EAVG)
    def test_nc_sigmared_eavg(self, exp_name):
        assert get_pc_type(exp_name, "DIS_NCE") == "f2p"

    @pytest.mark.parametrize(
        "exp_name",
        [
            "CHORUS_CC_NOTFIXED_FE_NB-SIGMARED",
            "CHORUS_CC_NOTFIXED_FE_NU-SIGMARED",
        ],
    )
    def test_chorus_cc(self, exp_name):
        assert get_pc_type(exp_name, "DIS_CC") == "dis_cc"

    @pytest.mark.parametrize(
        "exp_name",
        [
            "NUTEV_CC_NOTFIXED_FE_NB-SIGMARED",
            "NUTEV_CC_NOTFIXED_FE_NU-SIGMARED",
        ],
    )
    def test_nutev_cc(self, exp_name):
        assert get_pc_type(exp_name, "DIS_CC") == "dis_cc"

    @pytest.mark.parametrize(
        "exp_name",
        [
            "HERA_CC_318GEV_EM-SIGMARED",
            "HERA_CC_318GEV_EP-SIGMARED",
        ],
    )
    def test_hera_cc(self, exp_name):
        assert get_pc_type(exp_name, "DIS_CC") == "dis_cc"

    def test_emc_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            get_pc_type("EMC_NC_250GEV_FOO", "DIS_NCE")

    def test_unknown_dis_raises_value_error(self):
        with pytest.raises(ValueError, match="has not been implemented"):
            get_pc_type("UNKNOWN_DATASET", "DIS_NCE")

    def test_dis_process_type_prefix_matching(self):
        """Various DIS process type strings should all route correctly."""
        for ptype in ["DIS_NCE", "DIS_NCP", "DIS_CC", "DIS_NC"]:
            result = get_pc_type("BCDMS_NC_NOTFIXED_P_EM-F2", ptype)
            assert result == "f2p"


class TestJETRouting:
    """Tests for JET dataset -> PC type mapping."""

    def test_jet_returns_hj(self):
        assert get_pc_type("ATLAS_1JET_8TEV_R06", "JET") == "Hj"

    def test_any_jet_dataset_returns_hj(self):
        assert get_pc_type("CMS_1JET_7TEV", "JET") == "Hj"


class TestDIJETRouting:
    """Tests for DIJET dataset -> PC type mapping."""

    def test_atlas_dijet_specific_key(self):
        pc_dict = {"H2j_star": {"nodes": [0, 1], "yshift": [0.1, 0.2]}}
        result = get_pc_type("ATLAS_2JET_7TEV", "DIJET", experiment="ATLAS", pc_dict=pc_dict)
        assert result == "H2j_ystar"

    def test_cms_dijet_specific_key(self):
        pc_dict = {"H2j_ymax": {"nodes": [0, 1], "yshift": [0.1, 0.2]}}
        result = get_pc_type("CMS_2JET_7TEV", "DIJET", experiment="CMS", pc_dict=pc_dict)
        assert result == "H2j_ymax"

    def test_atlas_dijet_without_pc_dict_returns_specific(self):
        """Without pc_dict, should return experiment-specific key."""
        result = get_pc_type("ATLAS_2JET_7TEV", "DIJET", experiment="ATLAS")
        assert result == "H2j_ystar"

    def test_cms_dijet_without_pc_dict_returns_specific(self):
        result = get_pc_type("CMS_2JET_7TEV", "DIJET", experiment="CMS")
        assert result == "H2j_ymax"

    def test_unknown_dijet_experiment_raises(self):
        with pytest.raises(ValueError, match="is not implemented for DIJET"):
            get_pc_type("FOO_2JET_7TEV", "DIJET", experiment="FOO")

    def test_dijet_without_experiment_raises(self):
        with pytest.raises(ValueError, match="experiment"):
            get_pc_type("ATLAS_2JET_7TEV", "DIJET")
    
    def test_dijet_3d(self):
        result = get_pc_type("CMS_2JET_13TEV_M12-YSTAR-YB-R08", "DIJET_3D", experiment="CMS")
        assert result == "H2j_ystar"


class TestUnknownProcessType:
    """Tests for unsupported process types."""

    def test_unknown_process_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="has not been implemented"):
            get_pc_type("SOME_DATASET", "UNKNOWN_PROCESS")
