import uproot as u
import numpy as np
import ROOT as R

import matplotlib.pyplot as plt
# import boost_histogram as bh
import mplhep as hep
# from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend that does not show plots on the screen
plt.style.use(hep.style.ROOT)

# from scipy.optimize import curve_fit
from multiprocessing import Pool
from tqdm import tqdm

import os

from configs.energy_plot_config import path, reco_files, channel_map_ref, channel_map_test
from utils.utils import save_plot_batch, selfit_landau, test_module_energy_processor, plot_energy_sum_steps, plot_energy_spectrums

def plot_processor(runset):
    run_idx = runset[0]
    reco_file = runset[1]
    
    tin = u.open(reco_file, num_workers=4)['data']
    analyzer = tin.arrays(["energy", "channelIdx"], library="np")
    
    # # create a list of key-value pairs (FOR REFERENCE MODULE)
    # key_vals_list = [(key, vals) for key, vals in channel_map_ref.items()]
    # # create a Pool with 16 worker processes
    # with Pool(16) as p:
    #     # call process_ref for each key-value pair in key_vals_list
    #     sum_energy_list_ref = p.map(energy_summer_ref, key_vals_list)
    # sum_energy_arr_ref = np.array(sum_energy_list_ref)
    # succ_plot_energy_sum_ref = plot_energy_sum_steps(run_idx, sum_energy_arr_ref)
    
    # create a list of key-value pairs (FOR TEST MODULE)
    key_vals_list = [(analyzer, key, vals) for key, vals in channel_map_test.items()]
    energy_spectrum_list_test_results = []
    
    print('Processing Reco Data...')
    with tqdm(total=len(key_vals_list)) as pbar:
        for i, results in enumerate(Pool(16).imap_unordered(test_module_energy_processor, key_vals_list)):
            energy_spectrum_list_test_results.append(results)
            pbar.update()
        
    # sort the results
    energy_spectrum_list_test_results = sorted(energy_spectrum_list_test_results, key=lambda x: int(x[0]))
    sum_energy_list_test = []
    energy_spectrum_list = []
    for result in energy_spectrum_list_test_results:
        sum_energy_list_test.append(result[2])
        energy_spectrum_list.append((run_idx,result[0],result[1],'test')) # run index; ibar; energy lists; module types;
        
    sum_energy_arr_test = np.array(sum_energy_list_test)
    succ_plot_energy_sum_test = plot_energy_sum_steps(run_idx, sum_energy_arr_test, 'test')
    
    print('Sum of energy plot, Done.')
    
    results = []
    print('Plotting energy spectrum...')
    with tqdm(total=len(energy_spectrum_list)) as pbar:
        for i, result in enumerate(Pool(16).imap_unordered(plot_energy_spectrums, energy_spectrum_list)):
            results.append(result)
            pbar.update()
    if results == [0]*16: succ_plot_energy_spectrum_test = 0
    else: succ_plot_energy_spectrum_test = 1
    
    return (succ_plot_energy_sum_test, succ_plot_energy_spectrum_test)

if __name__ == "__main__":
    
    succ_results = []
    for runset in reco_files:
        print(f'Processing run {runset[0]}...')
        succ_results.append([runset[0], plot_processor(runset)])
    
    for succ_info in succ_results:
        print(f'Run {succ_info[0]}:\n')
        print(f'energy sum plots for test module... {"succeeded" if succ_info[1][0]==0 else "failed"}.')
        print(f'energy spectrum plots for test module... {"succeeded" if succ_info[1][1]==0 else "failed"}.\n')
        
    exit(0)
