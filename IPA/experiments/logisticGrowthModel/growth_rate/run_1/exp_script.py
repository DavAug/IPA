"""This script is to evaulate the IPA of a logistic growth model with respect
to its growth rate. Capacity and inital value are fixed at 1 and 0,
respectively. We explore different growth rates and see how the IPA changes.
"""
from IPA.model import model as m
from IPA.informationProfiler import informationProfiler as ip

# This script is to have a first glance at the Information Profile Algorithm
# Test Case I: Logistic Growth Model
# instantiate model
file_logistic_model = 'IPA/modelRepository/logistic_growth_model.mmt'
logistic_model = m.SingleOutputModel(file_logistic_model)

# fix the initial condition and the capacity
names = ['central_compartment.drug', 'central_compartment.capacity']
values = [0.0001, 1.0]
logistic_model.fix_model_dof(names=names, values=values)

# set true growth rate
true_growth_rate = [25]
parameter_boundaries = [[0, 50]]

# instantiate IP
profiler = ip.informationProfiler(model=logistic_model,
                                  parameters=true_growth_rate,
                                  boundaries=parameter_boundaries
                                  )

# generate data (without noise)
print('')
print('generate data:')
start = 0.0
end = 5.0
steps = 100
profiler.generate_data(start=start,
                       end=end,
                       steps=steps
                       )

# # find sample size
# print('')
# print('test sample size:')
# profiler.test_subset_size(subset_size=2)

# # find information profile
# print('')
# print('find information profile')
# profiler.find_IP(subset_size=2,
#                  iterations=500,
#                  opt_per_iter=1,
#                  no_successful=1
#                  )

# # save information profile
# profiler.save_profile(path='IPA/experiments/logisticGrowthModel/growth_rate',
#                       parameter_names=['growth_factor']
#                       )

# # generate plot of model
# profiler.plot_data()

# # generate plots of information profile
# profiler.plot_information_profile()

# plot model and IP from files
profiler.plot_from_files(path='IPA/experiments/logisticGrowthModel/growth_rate',
                         parameter_names=['growth_factor']
                         )
