fgrape (feedback GRAPE)
========================

.. automodule:: feedback_grape.fgrape
    :members:
    :show-inheritance:


    .. note::

        The below code is the default predefined RNN used in feedback GRAPE. 
        For another example of how to use a custom RNN, see: \
        :doc:`tutorials/feedbackGRAPE-tutorials/example_E_rnn`
    .. code-block:: python

        class RNN(nn.Module):
            hidden_size: int  # number of features in the hidden state
            output_size: int  # number of features in the output (inferred from the number of parameters) just provide those attributes to the class

            @nn.compact
            def __call__(self, measurement, hidden_state):

                gru_cell = nn.GRUCell(features=self.hidden_size)

                if measurement.ndim == 1:
                    measurement = measurement.reshape(1, -1)

                ###############
                ### Free to change whatever you want below as long as hidden layers have size self.hidden_size
                ### and output layer has size self.output_size
                ###############

                new_hidden_state, _ = gru_cell(hidden_state, measurement)
                # this returns the povm_params after linear regression through the hidden state which contains
                # the information of the previous time steps and this is optimized to output best povm_params
                # new_hidden_state = nn.Dense(features=self.hidden_size)(new_hidden_state)
                output = nn.Dense(
                    features=self.output_size,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.constant(jnp.pi),
                )(new_hidden_state)
                output = nn.relu(output)
                
                ###############
                ### Do not change the return statement
                ###############

                return output[0], new_hidden_state


