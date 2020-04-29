"""
Classes to apply compression to the output of the cochleagram function.
Include modifications of the gradients to improve stability for back
propagation.
"""

import torch as ch

class CompressionFunction(ch.nn.Module):
    """
    Base class for all compression functions. Forward pass must take in
    a ch tensor and apply the compression to each element. 
    """
    def __init__(self, scale=1., offset=0., fake_relu=True):
        """
        Compression functions have a scale and offset, which will be 
        optionally applied before the compression.
        """
        super(CompressionFunction, self).__init__()
        self.scale = scale
        self.offset = offset
        self.fake_relu = fake_relu
        self.fake_relu_func = FakeReLU.apply

    def _apply_scale_and_offset(self, x):
        # TODO: we only want to have positive values passed into the 
        # network, because sometimes a sound results in negative values
        # due to downsampling. We want to eliminate these, however
        # we should maintain gradients for the zero values (so that we 
        # can optimize them efficiently)
        if self.fake_relu:
            x = self.fake_relu_func(x)
        else:
            x = ch.nn.functional.relu(x)
        x = x + self.offset
        return (self.scale * x)

    def forward(self, x):
        """
        Each compression function should implement their own forward
        pass.

        Args:
            x (Tensor): The tensor to be compressed elementwise
        Returns: 
            Tensor: the compressed signal, same shape as x

        """
        raise NotImplementedError('Forward Pass is not implemented')
 

class LinearCompression(CompressionFunction):
    """
    Performs "Linear" compression, resulting in no compression in the 
    typical sense, but applying the specified scale and offset.
    """

    def forward(self, x):
       return self._apply_scale_and_offset(x)


class PowerCompression(CompressionFunction):
    """
    Performs power compression, raising each element to the specified 
    power. Note: "compression" in the typical sense will only happen 
    if the power is < 1. Human compression of the cochleagram is estimated
    at 0.3. After scale the input is clipped at 0 to avoid NaNs. 
    """
    def __init__(self, scale=1., offset=0., power=0.3):
       super().__init__(scale, offset)
       self.power = power

    def forward(self, x):
       x = self._apply_scale_and_offset(x)
       x = ch.pow(x, self.power)
       return x


class ClippedGradPowerCompression(PowerCompression):
    """
    Performs power compression, raising each element to the specified
    power in the forward pass, but clipping the gradients during the
    backwards pass for stability (forward pass is same as
    PowerCompression)
    """
    def __init__(self, scale=1., offset=0.,
                 power=0.3, clip_value=1.):
        super().__init__(scale, offset, power)
        self.clip_value = clip_value
        self.clipped_power = ClippedPower.apply

    def forward(self, x):
        x = self._apply_scale_and_offset(x)
        x = self.clipped_power(x, self.clip_value, self.power)
        return x


class ClippedPower(ch.autograd.Function):
    """
    Takes the power of a signal and clips its gradients to the 
    provided values in the backwards pass
    """
    @staticmethod
    def forward(ctx, x, clip_value, power):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        ctx.power = power
        return ch.pow(x, power)
 
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = ctx.power * ch.pow(x, ctx.power-1)
        return grad_output * ch.clamp(g, -ctx.clip_value, ctx.clip_value), None, None


class FakeReLU(ch.autograd.Function):
    """
    Applies a ReLU in the forward pass, but all gradients=1 in the backwards
    pass. 
    """
    @staticmethod
    def forward(ctx, x):
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

