    
# path = "/gwdata/users/lzhang/public/cmstestbeam/2023_03_cmstiming_BTL/TOFHIR/RecoData/"
path = "/gwdata/users/lzhang/public/cmstestbeam/2023_03_cmstiming_BTL/TOFHIR2/RecoData_v2/"

reco_files = []
root_suffix = '.root'
for run in range(67962,67967):
    reco_file_name = f'run{str(run)}_e'+root_suffix
    reco_files.append([run, path + reco_file_name])
    
channel_map_ref = {}
for ibar in range(0,16):
    if ibar <= 7: icL = 14 - 2 * ibar
    else: icL = 1 + 2 * (ibar - 8)
    icR = 31 - icL
    channel_map_ref[str(ibar)] = [icL, icR]
    
channel_map_test = {}
for ibar in range(0,16):
    channel_map_test[str(ibar)] = [channel_map_ref[str(ibar)][0]+64, channel_map_ref[str(ibar)][1]+64]
