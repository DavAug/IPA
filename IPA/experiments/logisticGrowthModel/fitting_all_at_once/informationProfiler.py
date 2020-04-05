from IPA.model import model as m
from IPA.informationProfiler import informationProfiler as ip

# This script is to have a first glance at the Information Profile Algorithm
# Test Case I: Logistic Growth Model
# instantiate model
file_logistic_model = 'IPA/modelRepository/logistic_growth_model.mmt'
logistic_model = m.SingleOutputModel(file_logistic_model)
logistic_model_true_params = [0.001, 25]  # [init drug, lambda]
parameter_boundaries = [[0, 1], [0, 50]]

# instantiate IP
profiler = ip.informationProfiler(model=logistic_model,
                                  parameters=logistic_model_true_params,
                                  boundaries=parameter_boundaries
                                  )

# generate data (without noise)
print('')
print('generate data:')
start = 0.0
end = 1.0
steps = 100
profiler.generate_data(start=start,
                       end=end,
                       steps=steps
                       )

# # find sample size
# print('')
# print('test sample size:')
# profiler.test_subset_size(subset_size=4)

# # find information profile
# print('')
# print('find information profile')
# profiler.find_IP(subset_size=4,
#                  iterations=200,
#                  opt_per_iter=1,
#                  no_successful=1
#                  )

# # save information profile
# profiler.save_profile(path='IPA/experiments/logisticGrowthModel',
#                       parameter_names=['init value', 'growth_factor']
#                       )

# # generate plot of model
# profiler.plot_data()

# # generate plots of information profile
# profiler.plot_information_profile()

# plot model and IP from files
profiler.plot_from_files(path='IPA/experiments/logisticGrowthModel',
                         parameter_names=['init value', 'growth_factor']
                         )
