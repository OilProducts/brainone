from flax import linen as nn

from layers import STDPLinear

class SimpleSpikeNetwork(nn.Module):
    batch_size: int = 128
    a_pos: float = 0.005
    a_neg: float = 0.005
    plasticity_reward: float = 1
    plasticity_punish: float = 1

    def setup(self):
        self.layer_1 = STDPLinear(
            in_features=784,
            out_features=500,
            a_pos=self.a_pos,
            a_neg=self.a_neg,
            plasticity_reward=self.plasticity_reward,
            plasticity_punish=self.plasticity_punish,
            batch_size=self.batch_size
        )

        self.layer_2 = STDPLinear(
            in_features=500,
            out_features=200,
            a_pos=self.a_pos,
            a_neg=self.a_neg,
            plasticity_reward=self.plasticity_reward,
            plasticity_punish=self.plasticity_punish,
            batch_size=self.batch_size
        )

        self.layer_3 = STDPLinear(
            in_features=200,
            out_features=10,
            a_pos=self.a_pos,
            a_neg=self.a_neg,
            plasticity_reward=self.plasticity_reward,
            plasticity_punish=self.plasticity_punish,
            batch_size=self.batch_size
        )

    def __call__(self, x, labels, train=True):
        layer_1_out = self.layer_1(x)
        layer_2_out = self.layer_2(layer_1_out)
        layer_3_out = self.layer_3(layer_2_out)

        return layer_3_out



    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)

    def reset_hidden_state(self):
        self.layer_1.reset_hidden_state()
        self.layer_2.reset_hidden_state()
        self.layer_3.reset_hidden_state()
