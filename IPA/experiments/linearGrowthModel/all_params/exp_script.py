from IPA.model import model as m
from IPA.informationProfiler import informationProfiler as ip

# This script is to have a first glance at the Information Profile Algorithm
# Test Case I: Linear Growth Model
# instantiate model
file_linear_model = 'IPA/modelRepository/linear_growth_model.mmt'
linear_model = m.SingleOutputModel(file_linear_model)
linear_model_true_params = [1, 2]  # [init drug, lambda]
parameter_boundaries = [[0, 100], [0, 100]]

# instantiate IP
profiler = ip.informationProfiler(model=linear_model,
                                  parameters=linear_model_true_params,
                                  boundaries=parameter_boundaries
                                  )

# generate data (without noise)
print('')
print('generate data:')
start = 0.0
end = 100.0
steps = 100
profiler.generate_data(start=start,
                       end=end,
                       steps=steps
                       )

# # find sample size
# print('')
# print('test sample size:')
# profiler.test_subset_size(subset_size=10)

# # find information profile
# print('')
# print('find information profile')
# profiler.find_IP(subset_size=10,
#                  iterations=200,
#                  opt_per_iter=1,
#                  no_successful=1
#                  )

# # save information profile
# profiler.save_profile(path='IPA/experiments/linearGrowthModel',
#                       parameter_names=['init value', 'growth_factor']
#                       )

# generate plot of model
profiler.plot_data()

# # generate plots of information profile
# profiler.plot_information_profile()

# # plot from files
# profiler.plot_from_files(path='IPA/experiments/linearGrowthModel/all_params',
#                          parameter_names=['init value', 'growth_factor']
#                          )
