from helios.analysis import Analysis

save_dir = './output/'+str('test_11-04-2023_12-13')

analysis = Analysis(window_size=0.1)
analysis.convergence_analysis(results_table_dir= save_dir+'/Standard_Experiment/Qlearntab_Language_1', 
                                save_dir=save_dir,show_figures="Y")