#ifndef GPUFIT_MATERIAL_BASIS_CUH_INCLUDED
#define GPUFIT_MATERIAL_BASIS_CUH_INCLUDED

/* Description of the calculate_material_basis function
* =====================================================
*
* This function calculates the values of a material basis decomposition
* model and its partial derivatives with respect to the model parameters.
*
* Unlike the COMPTON_PE model which uses analytic Klein-Nishina and
* photoelectric basis functions, MATERIAL_BASIS uses two tabulated
* material LAC curves (m1_cp, m2_cp) passed through user_info.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: Material 1 line integral coefficient
*             p[1]: Material 2 line integral coefficient
*
* n_fits: The number of fits (sinogram pixels).
*
* n_points: The number of data points per fit (2: high and low energy).
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index (0 = high energy, 1 = low energy).
*
* fit_index: The fit index.
*
* chunk_index: The chunk index.
*
* user_info: An input vector containing user information.
*            Layout:
*            [scc, scp, pch, pcl, n_kev_h, n_kev_l,
*             m1_cp[n_kev_h], m2_cp[n_kev_l],
*             kev_h[n_kev_h], kev_l[n_kev_l],
*             spctrm_h[n_kev_h], spctrm_l[n_kev_l],
*             spctrm_h_ph[n_kev_h], spctrm_l_ph[n_kev_l],
*             spctrm_h_kn[n_kev_h], spctrm_l_kn[n_kev_l]]
*
* user_info_size: The size of user_info in bytes.
*
*/

__device__ void calculate_material_basis(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // parameters
    REAL const * param = parameters;

    // user info
    // layout is:
    // scc, scp, pch, pcl, n_kev_h, n_kev_l,
    // m1_cp, m2_cp, kev_h, kev_l,
    // spctrm_h, spctrm_l, spctrm_h_ph, spctrm_l_ph, spctrm_h_kn, spctrm_l_kn

    REAL * uif = (REAL *) user_info;
    REAL const scc = uif[0];
    REAL const scp = uif[1];
    REAL const pch = uif[2];
    REAL const pcl = uif[3];
    int const n_kev_h = (int) uif[4];
    int const n_kev_l = (int) uif[5];
    int ci = 6;

    // tabulated material basis function values
    REAL * m1_cp = uif + ci; ci += n_kev_h;
    REAL * m2_cp = uif + ci; ci += n_kev_l;

    // energy levels
    REAL * kev_h = uif + ci; ci += n_kev_h;
    REAL * kev_l = uif + ci; ci += n_kev_l;

    // normalized spectra
    REAL * spctrm_h = uif + ci; ci += n_kev_h;
    REAL * spctrm_l = uif + ci; ci += n_kev_l;

    // pre-multiplied spectra for derivatives
    REAL * spctrm_h_ph = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_ph = uif + ci; ci += n_kev_l;
    REAL * spctrm_h_kn = uif + ci; ci += n_kev_h;
    REAL * spctrm_l_kn = uif + ci; ci += n_kev_l;

    // value
    REAL a1 = param[0], a2 = param[1];
    REAL h = 0, l = 0;
    REAL dh_da1 = 0, dh_da2 = 0;
    REAL dl_da1 = 0, dl_da2 = 0;
    REAL tmp = 0;

    a1 /= scc; a2 /= scp;

    REAL * current_derivative = derivative + point_index;

    if (point_index == 0)
    {
        // for high energy projection
        for (int i = 0; i < n_kev_h; ++i)
        {
            tmp = exp(-(a1 * m1_cp[i] + a2 * m2_cp[i]));
            h += tmp * spctrm_h[i];
            dh_da1 += tmp * spctrm_h_kn[i];
            dh_da2 += tmp * spctrm_h_ph[i];
        }
        value[0] = -log(h * pch / pch);
        current_derivative[0 * n_points] = dh_da1 / h / scc;
        current_derivative[1 * n_points] = dh_da2 / h / scp;
    }
    else
    {
        // for low energy projection
        for (int i = 0; i < n_kev_l; ++i)
        {
            tmp = exp(-(a1 * m1_cp[i] + a2 * m2_cp[i]));
            l += tmp * spctrm_l[i];
            dl_da1 += tmp * spctrm_l_kn[i];
            dl_da2 += tmp * spctrm_l_ph[i];
        }
        value[1] = -log(l * pcl / pcl);
        current_derivative[0 * n_points] = dl_da1 / l / scc;
        current_derivative[1 * n_points] = dl_da2 / l / scp;
    }
}

#endif
