# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from neon import NervanaObject
from neon.optimizers.optimizer import GradientDescentMomentum, get_param_list
from neon.util.persist import load_class
import numpy as np


class GradientDescentMomentumNesterov(GradientDescentMomentum):
    """
    Nesterov Stochastic gradient descent with momentum
    """

    def optimize(self, layer_list, epoch):
        """
        Apply the learning rule to all the layers and update the states.
        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """
        lrate = self.schedule.get_learning_rate(self.learning_rate, epoch)
        param_list = get_param_list(layer_list)

        scale_factor = self.clip_gradient_norm(param_list, self.gradient_clip_norm)

        for (param, grad), states in param_list:
            param.rounding = self.stochastic_round
            if len(states) == 0:
                states.append(self.be.zeros_like(grad))
                states.append(self.be.zeros_like(grad))
            grad = grad / self.be.bsz
            grad = self.clip_gradient_value(grad, self.gradient_clip_value)

            velocity = states[0]
            velocity_backup = states[-1]

            velocity_backup[:] = velocity
            velocity[:] = (self.momentum_coef * velocity -
                           lrate * (scale_factor * grad + self.wdecay * param))
            param[:] = (param + velocity * (1 + self.momentum_coef) -
                        self.momentum_coef * velocity_backup)

