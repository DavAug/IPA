import matplotlib.pyplot as plt
import myokit
import numpy as np
import pints
import pytest
import seaborn as sns

from IPA.inference import inference as inf


class informationProfiler(object):

    def __init__(self, model, parameters, boundaries):
        self._model = model
        self.parameters = parameters
        self.boundaries = boundaries
        self.data = None
        self.times = None

    def generate_data(self,
                      start=0.0,
                      end=100.0,
                      steps=100,
                      optimiser=pints.PSO
                      ):
        # generate data (without noise)
        self.times = np.linspace(start, end, steps)
        self.data = np.array(self._model.simulate(self.parameters,
                                                  self.times
                                                  )
                             )
        print(self.times)
        print(self.data)
        print(self._model.mdof_values)
        print(self._model.output_name)

        # instantiate inverse problem
        problem = inf.SingleOutputInverseProblem(
            models=[self._model],
            times=[self.times],
            values=[self.data]
            )

        # set optimiser
        problem.set_optimiser(optimiser=optimiser)

        # solve inverse problem
        problem.optimise(initial_parameter=self.parameters,
                         number_of_iterations=1)
        estimated_parameters = problem.estimated_parameters

        # check whether inferred parameters are close to true parameters
        for param_id, true_value in enumerate(self.parameters):
            estimated_value = estimated_parameters[param_id]
            if estimated_value == pytest.approx(true_value, rel=0.05):
                print(
                    'Parameter %d has been recovered. True: %.2f, Est: %.2f.' %
                    (param_id, true_value, estimated_value)
                )
            else:
                print(
                    'ERROR!: Parameter %d has not been recovered. True: %.2f'
                    % (param_id, true_value) + ', Est: %.2f.' %
                    (estimated_value)
                )

    def test_subset_size(self, subset_size=10, maximal_optimisations=10):
        for opt in range(maximal_optimisations):
            # subssample mask
            number_data_points = len(self.times)
            mask = np.zeros(number_data_points, dtype=bool)  # init False array
            mask[:subset_size] = np.ones(subset_size, dtype=bool)  # ones
            np.random.shuffle(mask)

            # subsample data
            time_subset = self.times[mask]
            data_subset = self.data[mask]

            # instantiate inverse problem
            problem = inf.SingleOutputInverseProblem(
                models=[self._model],
                times=[time_subset],
                values=[data_subset]
                )

            # set parameter boundaries
            boundaries_array = np.array(self.boundaries)
            problem.set_parameter_boundaries(
                boundaries=[boundaries_array[:, 0],
                            boundaries_array[:, 1]
                            ]
                )

            # draw initial parameters for optimisation
            initial_parameters = []
            for parameter_boundaries in self.boundaries:
                minimum, maximum = parameter_boundaries
                initial_value = np.random.uniform(low=float(minimum),
                                                  high=float(maximum)
                                                  )
                initial_parameters.append(initial_value)

            try:
                # solve inverse problem
                problem.optimise(initial_parameter=initial_parameters,
                                 number_of_iterations=1
                                 )
                estimated_parameters = problem.estimated_parameters
            except myokit.SimulationError:
                # set flag
                flag = False

                # print message to user
                print('The inverse problem could not be solved in run %d'
                      % opt
                      )
            else:
                # set flag
                flag = True

                # check whether inferred parameters are close to true
                # parameters
                for param_id, true_value in enumerate(self.parameters):
                    estimated_value = estimated_parameters[param_id]
                    if estimated_value == pytest.approx(true_value, rel=0.05):
                        print('Parameter %d has been recovered.' % param_id)
                        print('True: %.5f, Est: %.5f.' % (true_value,
                                                          estimated_value
                                                          )
                              )
                    else:
                        print('ERROR: Parameter %d has not been recovered.'
                              % param_id
                              )
                        print('True: %.5f, Est: %.5f.' % (true_value,
                                                          estimated_value
                                                          )
                              )

            # break the for loop, if one identification was successful
            if flag:
                break

    def find_IP(self,
                subset_size=10,
                iterations=100,
                opt_per_iter=10,
                no_successful=7
                ):
        # get number of parameters
        number_of_parameters = len(self.parameters)

        # create data container for each parameter
        self.data_container = np.empty(shape=(number_of_parameters,
                                              iterations,
                                              subset_size
                                              )
                                       )

        # create container for parameter values within an iteration
        parameter_container = np.empty(shape=(number_of_parameters,
                                              opt_per_iter
                                              )
                                       )

        # create nan array
        nan_array = np.array([np.nan] * subset_size)

        # initialise sampling mask
        number_data_points = len(self.times)
        mask = np.zeros(number_data_points, dtype=bool)  # init False array
        mask[:subset_size] = np.ones(subset_size, dtype=bool)  # ones

        # run subsample and fitting routine
        for iteration_id in range(iterations):
            # shuffle mask
            np.random.shuffle(mask)

            # subsample data
            time_subset = self.times[mask]
            data_subset = self.data[mask]

            # instantiate inverse problem
            problem = inf.SingleOutputInverseProblem(
                models=[self._model],
                times=[time_subset],
                values=[data_subset]
                )

            # set parameter boundaries
            boundaries_array = np.array(self.boundaries)
            problem.set_parameter_boundaries(
                boundaries=[boundaries_array[:, 0],
                            boundaries_array[:, 1]
                            ]
                )

            try:
                # run optimisation opt_per_iter times
                for opt_id in range(opt_per_iter):
                    # draw initial parameters for optimisation
                    initial_parameters = []
                    for parameter_boundaries in self.boundaries:
                        minimum, maximum = parameter_boundaries
                        initial_value = np.random.uniform(
                            low=float(minimum),
                            high=float(maximum)
                            )
                        initial_parameters.append(initial_value)

                    # solve inverse problem
                    problem.optimise(initial_parameter=initial_parameters,
                                     number_of_iterations=1
                                     )
                    est = problem.estimated_parameters
                    parameter_container[:, opt_id] = est

            except myokit.SimulationError:
                # if optimisatoin breaks, add nan entry to container
                self.data_container[:,
                                    iteration_id,
                                    :
                                    ] = nan_array

            else:
                # iterate through parameters and check whether 0.7 if the
                # optimisations recovered the true parameter
                for param_id, true_parameter in enumerate(self.parameters):
                    optimised_parameters = parameter_container[param_id, :]
                    successful_mask = np.isclose(optimised_parameters,
                                                 true_parameter,
                                                 rtol=0.05
                                                 )
                    correct_optimisations = np.sum(successful_mask)

                    # add subset to container if >=0.7 opt were successful
                    if correct_optimisations >= no_successful:
                        self.data_container[param_id,
                                            iteration_id,
                                            :
                                            ] = time_subset

                    # if less are correct, fill with nans
                    else:
                        self.data_container[param_id,
                                            iteration_id,
                                            :
                                            ] = nan_array

    def save_profile(self, path, parameter_names):
        # write header
        header = 'This file stores the subsets of the data leading to recovery'

        # iterate through parameters and save container to .txt
        for param_id, param in enumerate(parameter_names):
            filename = path + '/' + param + '.csv'
            np.savetxt(fname=filename,
                       X=self.data_container[param_id, ...],
                       delimiter=',',
                       header=header
                       )

    def plot_data(self):
        # plot data
        plt.plot(self.times, self.data, label='data')

        # add x anc y label
        plt.xlabel('time [arbitrary units]')
        plt.ylabel('state [arbitrary units]')

        # generate legend
        plt.legend()

        # show plot
        plt.show()

    def plot_information_profile(self):
        # iterate through parameter containers
        for param_id, container in enumerate(self.data_container):
            sampled_time_points = container.flatten()
            sns.distplot(sampled_time_points, hist=False, kde=True,
                         label='%d' % param_id
                         )
            # plt.legend()
            plt.show()

    def plot_from_files(self, path, parameter_names):
        # get number of parameters
        number_parameters = len(parameter_names)

        # instantiate figure
        _, axes = plt.subplots(nrows=1+number_parameters,
                               sharex=True
                               )

        # plot the model to ax 0
        axes[0].plot(self.times, self.data, label='data')

        # set y label
        axes[0].set_ylabel('state')

        # create legend
        axes[0].legend()

        # iterate through parameters and add their profile to plot
        for param_id, param in enumerate(parameter_names):
            # reconstruct data file
            filename = path + '/' + param + '.csv'

            # load data
            data = np.loadtxt(fname=filename,
                              delimiter=','
                              )

            # flatten data container
            data = data.flatten()

            # plot IP
            sns.distplot(data,
                         hist=False,
                         kde=True,
                         label='Parameter %s' % param,
                         ax=axes[param_id+1]
                         )

        # add x label
        axes[-1].set_xlabel('time')

        # show figure
        plt.show()
