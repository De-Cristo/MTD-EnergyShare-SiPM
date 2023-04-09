import numpy as np
import ROOT as R
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend that does not show plots on the screen
plt.style.use(hep.style.ROOT)
from multiprocessing import Pool
from tqdm import tqdm
import os


def save_plot_batch( plot_str ):
    plt.savefig(plot_str)
    # plt.show()
    plt.close()
    
def selfit_landau(th, low_edge, high_edge, maximum):
    landau = R.TF1("fit","landau",low_edge,high_edge)
    landau.SetParLimits(1,maximum-30,maximum+30)
    th.Fit("fit","Q0")
    par = landau.GetParameters()
    par_err = landau.GetParErrors()
    return par[0], par[1], par[2], par_err[0], par_err[1], par_err[2]

def landau_eval(x, amp, mpv, sigma):
    landau = R.TF1("fit",f"{str(amp)}*TMath::Landau(x,{str(mpv)},{str(sigma)})",-200,2200)
    return landau.Eval(x)
    
# define a function to be called by each process
def energy_summer_ref(key_vals):
    key, vals = key_vals
    energy_list = []
    for _evt in range(len(analyzer['channelIdx'])):
        if (analyzer['channelIdx'][_evt][vals[0]]>=0):
            if key == "9":
                energy_list.append(0.)
            else:
                energy_list.append(analyzer['energy'][_evt][analyzer['channelIdx'][_evt][vals[0]]])
        if (analyzer['channelIdx'][_evt][vals[1]]>=0):
            energy_list.append(analyzer['energy'][_evt][analyzer['channelIdx'][_evt][vals[1]]])
    return np.sum(np.array(energy_list))/len(energy_list)

# define a function to be called by each process
def test_module_energy_processor(arg_list):
    analyzer, key, vals = arg_list
    energy_list = [[]]*4
    energy_list[0] = [] # on the 'left'
    energy_list[1] = [] # on the 'right'
    energy_list[2] = [] # on the 'combined'
    energy_list[3] = [] # on the 'sumation'
    
    for _evt in range(len(analyzer['channelIdx'])):
        if (analyzer['channelIdx'][_evt][vals[0]]>=0):
            energy_list[0].append(analyzer['energy'][_evt][analyzer['channelIdx'][_evt][vals[0]]])
            if (analyzer['channelIdx'][_evt][vals[1]]>=0):
                energy_list[2].append(analyzer['energy'][_evt][analyzer['channelIdx'][_evt][vals[0]]]\
                                      +analyzer['energy'][_evt][analyzer['channelIdx'][_evt][vals[1]]])
        if (analyzer['channelIdx'][_evt][vals[1]]>=0):
            energy_list[1].append(analyzer['energy'][_evt][analyzer['channelIdx'][_evt][vals[1]]])
    energy_list[3] = energy_list[0] + energy_list[1]
    sumation = np.sum(np.array(energy_list[3]))
    # key = bar index; energy_list contains four lists of the energy of the triggerred events; sumation is a float that equals all the energy recorded on certain bar
    return (key, energy_list, sumation)
    
def plot_energy_sum_steps(run_idx, bar_array, module_type):
    if len(bar_array) == 16:
        num_bins = 16
        bin_edges = np.arange(num_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        fig, ax = plt.subplots()
        hist, _, _ = ax.hist(np.arange(num_bins), 
                             bins=bin_edges, 
                             weights=bar_array, 
                             alpha=0.6, 
                             edgecolor='black', 
                             color = '#958CDD',
                             log=False
                            )
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(np.arange(num_bins))
        ax.set_xlabel(f"Bar Number [{module_type} mudule] run {run_idx}")
        ax.set_ylabel("Sum of energy [L+R]")
        if not os.path.isdir('SumOfEnergy'):
            os.makedirs('SumOfEnergy')
        save_plot_batch(f'SumOfEnergy/Sum_energy_{module_type}_{run_idx}.png')
        return 0
    else:
        return 1
        
def plot_energy_spectrums(arg_list):
    run_idx, ibar, bar_list, module_type = arg_list
    energy_spectrum_left = bar_list[0]
    energy_spectrum_right = bar_list[1]
    energy_spectrum_com = bar_list[2]
    energy_spectrum_sum = bar_list[3]
    
    energy_spectrum_left = np.array(energy_spectrum_left)
    energy_spectrum_right = np.array(energy_spectrum_right)
    energy_spectrum_com = np.array(energy_spectrum_com)
    energy_spectrum_sum = np.array(energy_spectrum_sum)
    
    plot_dict = {}
    plot_dict['left'] = np.array(energy_spectrum_left)
    plot_dict['right'] = np.array(energy_spectrum_right)
    plot_dict['combination'] = np.array(energy_spectrum_com)
    plot_dict['summation'] = np.array(energy_spectrum_sum)
    
    for key, val in plot_dict.items():
        upper_limit = 1000
        lower_edge = 60
        upper_edge = 800
        if key == 'combination':
            upper_limit = 2000
            lower_edge = 100
            upper_edge = 1600
        fig, ax = plt.subplots()
        counts, bins, _ = ax.hist(val,
                                  bins = np.linspace(0, upper_limit, 100), 
                                  edgecolor='#958CDD', 
                                  alpha=1.0, 
                                  histtype='step',
                                  log = True,
                                  label = 'Energy Spectrum'
                                 )
        # create a ROOT TH1 histogram
        root_hist = R.TH1F("hist", "hist", 100, 0, upper_limit)
        # fill the ROOT histogram with the numpy histogram data
        for i in range(len(counts)):
            root_hist.SetBinContent(i+1, counts[i])
        constant, mpv, sigma, constant_err, mpv_err, sigma_err = selfit_landau(root_hist, lower_edge, upper_edge, \
                      (np.argmax(counts[round(lower_edge*100./upper_limit):round(upper_edge*100./upper_limit)])+round(lower_edge*100./upper_limit))*upper_limit/100.)
        fit_x = np.linspace(lower_edge,upper_edge,5000)
        fit_y = []
        for x in fit_x:
            fit_y.append(landau_eval(x, constant, mpv, sigma))
        fit_y = np.array(fit_y)
        ax.plot(fit_x, fit_y, 'r-', label=f'{constant:.2e}*Landau({mpv:.2e},{sigma:.2e})')
        ax.axvline(x=lower_edge, linestyle='--', color='gray')
        ax.axvline(x=upper_edge, linestyle='--', color='gray')
        
        # calculate bin widths
        bin_widths = bins[1:] - bins[:-1]
        # calculate bin errors (assuming Poisson statistics)
        scale = len(val) / sum(counts)
        err   = np.sqrt(counts * scale) / scale
        # plot error bars as filled rectangles
        label = 'Stat. Uncertainty'
        for i in range(len(bins)-1):
            x = bins[i] + bin_widths[i]/2
            y = counts[i]
            error = err[i]
            ax.add_patch(plt.Rectangle((x-bin_widths[i]/2, y-error), 
                                       bin_widths[i], 
                                       2*error,
                                       facecolor='#FFDDAB',
                                       alpha=1.0, 
                                       edgecolor='none'
                                      )
                        )
        ax.legend([label])
        ax.set_xlabel(f'Bar{ibar} run{run_idx} {key} Energy (ADC)')
        ax.set_ylabel(f'Ref Module Events({len(val)})')
        ax.legend(fontsize=18)
        hep.cms.label("Preliminary", loc=0, ax=ax, data=True, rlabel="Test Beam")
        if not os.path.isdir(f'{module_type}MuduleEnergySprectrumRun{run_idx}'):
            os.makedirs(f'{module_type}MuduleEnergySprectrumRun{run_idx}')
            
        save_plot_batch(f'{module_type}MuduleEnergySprectrumRun{run_idx}/{module_type}_module_Bar{ibar}_{key}_energy_spectrum_run{run_idx}.png')
            
    return 0
