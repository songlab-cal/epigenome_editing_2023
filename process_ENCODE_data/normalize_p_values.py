import numpy as np
import sys
import scipy.optimize as optimize
from random import sample
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

Original_Data_Path = "../Data/Original_25bp_Data/"
Transformed_Data_Path = "../Data/Transformed_25bp_Data/"

Avocado_list = ["T07_A03", "T11_A04", "T12_A07", "T08_A01", "T09_A01",
                "T10_A01", "T11_A01", "T12_A01", "T13_A01", "T13_A03"]

linear_scaling_value_reference = 1.0
linear_scaling_value_target = 1.0


# define a function to transform -log10(p-values)
def transform(x, alpha, beta, multiplicative_scale, linear_scaling_value):
    y = np.maximum((linear_scaling_value + x) / multiplicative_scale, 0.0)
    return alpha * np.power(y, beta)


if __name__ == '__main__':

    cell_type = sys.argv[1]
    assay_type = sys.argv[2]
    cell_types = ["T01", cell_type]

    f_S3norm_parameters = open(Transformed_Data_Path + cell_type + assay_type +
                               ".S3norm.txt", 'w')

    common_peak_indices = {}
    common_peak_values = {}
    common_peak_values[1] = []
    common_peak_values[2] = []
    common_background_indices = {}
    common_background_values = {}
    common_background_values[1] = []
    common_background_values[2] = []
    remaining_values = {}
    remaining_values[1] = []
    remaining_values[2] = []

    for chrom in list(range(1, 23)) + ["X"]:

        x1 = np.load(Original_Data_Path + cell_types[0] + assay_type +
                     ".chr" + str(chrom) + ".npy")
        x2 = np.load(Original_Data_Path + cell_types[1] + assay_type +
                     ".chr" + str(chrom) + ".npy")

        x1_pvalues = np.power(10, -1.0 * x1)
        x2_pvalues = np.power(10, -1.0 * x2)

        chromosome_length = x1.shape[0]
        # print("Number of 25bp bins in chr"+str(chrom)+" = "
        #       + str(chromosome_length))

        x1_FDR = sm.stats.fdrcorrection(x1_pvalues, alpha=0.05, method='indep')
        x1_peaks = [idx for idx, v in enumerate(x1_FDR[0])
                    if bool(v) is True]
        x1_background = [idx for idx, v in enumerate(x1_FDR[0])
                         if bool(v) is False]

        if(cell_types[1]+"_"+assay_type in Avocado_list):
            # Find the smallest 1000 p-values and assign them as peaks
            # (as per Jacob's suggestion on October 11 2021)
            x2_peaks = np.argpartition(x2_pvalues, 10000)[:10000]

            x2_background = list(set(list(range(chromosome_length))) -
                                 set(list(x2_peaks)))
        else:
            x2_FDR = sm.stats.fdrcorrection(x2_pvalues, alpha=0.05,
                                            method='indep')
            x2_peaks = [idx for idx, v in enumerate(x2_FDR[0])
                        if bool(v) is True]
            x2_background = [idx for idx, v in enumerate(x2_FDR[0])
                             if bool(v) is False]

        common_peaks_chrom = list(set(x1_peaks) & set(x2_peaks))
        # print("Number of common peaks for chr" + str(chrom) + " = "
        #       + str(len(common_peaks_chrom)))
        common_peak_indices[chrom] = common_peaks_chrom
        common_peak_values[1].extend(x1[common_peaks_chrom])
        common_peak_values[2].extend(x2[common_peaks_chrom])

        common_background_chrom = list(set(x1_background) & set(x2_background))
        # print("Number of common background for chr" + str(chrom) + " = "
        #       + str(len(common_background_chrom)))
        common_background_indices[chrom] = common_background_chrom
        common_background_values[1].extend(x1[common_background_chrom])
        common_background_values[2].extend(x2[common_background_chrom])

        remaining_indices = list(set(list(set(range(chromosome_length)) -
                                          set(common_peaks_chrom))) -
                                 set(common_background_chrom))
        remaining_values[1].extend(x1[remaining_indices])
        remaining_values[2].extend(x2[remaining_indices])

    # Perform linear and multiplicative scaling of -log10(p-values)
    scale_reference = min(common_peak_values[1])
    common_peaks_mean_1 = np.mean([(linear_scaling_value_reference + x) /
                                   scale_reference for x in
                                   common_peak_values[1]])
    common_background_mean_1 = [np.maximum((linear_scaling_value_reference + x)
                                           / scale_reference, 0.0) for x in
                                common_background_values[1]]
    common_background_mean_1 = np.mean(common_background_mean_1)

    scale_target = min(common_peak_values[2])
    common_peaks_2 = [(linear_scaling_value_target + x) / scale_target for x
                      in common_peak_values[2]]

    # sub-sample from background of target cell type
    common_background_2 = sample(common_background_values[2], 100000)

    common_background_2 = [np.maximum((linear_scaling_value_target + x) /
                                      scale_target, 0.0) for x
                           in common_background_2]

    # Now we solve the S3norm optimization problem to match
    # the peak and background means of the second cell type to the first
    def log_linear(parameters):
        a, b = parameters

        LHS_1 = common_peaks_mean_1
        RHS_1 = np.mean(a * np.power(common_peaks_2, b))

        LHS_2 = common_background_mean_1
        RHS_2 = np.mean(a * np.power(common_background_2, b))

        output = (np.power(LHS_1 - RHS_1, 2) + np.power(LHS_2 - RHS_2, 2))
        return output

    initial_guess = [1, 1]
    fitted_parameters = optimize.minimize(log_linear, initial_guess,
                                          method='Powell',
                                          options={"maxiter": 100000})
    alpha, beta = fitted_parameters.x
    print(cell_types[1], assay_type, alpha, beta, log_linear([alpha, beta]),
          file=f_S3norm_parameters)
    f_S3norm_parameters.close()

    # Now write the transformed npy arrays
    for chrom in list(range(1, 23)) + ["X"]:
        if(cell_types[1] == "T02"):
            # Only write out the transformed reference cell type once
            x = np.load(Original_Data_Path + cell_types[0] + assay_type +
                        ".chr" + str(chrom) + ".npy")
            np.save(Transformed_Data_Path + cell_types[0] + assay_type +
                    ".chr" + str(chrom) + ".npy",
                    transform(x,
                              1.0,
                              1.0,
                              scale_reference,
                              linear_scaling_value_reference))

        # Always write the target cell type
        assert(cell_types[0] == "T01")
        assert(cell_types[1] != "T01")
        x = np.load(Original_Data_Path + cell_types[1] + assay_type +
                    ".chr" + str(chrom) + ".npy")
        np.save(Transformed_Data_Path + cell_types[1] + assay_type +
                ".chr" + str(chrom) + ".npy",
                transform(x,
                          alpha,
                          beta,
                          scale_target,
                          linear_scaling_value_target))

