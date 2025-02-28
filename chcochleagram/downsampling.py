import numpy as np
import torch as ch
import scipy.signal as signal
try:
    from scipy.signal import kaiser
except ImportError:
    from scipy.signal.windows import kaiser

class DownsampleEnvelopes(ch.nn.Module):
    """
    Base class for downsampling operations for the cochlear envelopes.
    """
    def __init__(self, sr, env_sr):
        """
        Base class for downsampling cochlear envelopes.
      
        Args:
            sr (int): the sampling rate for the input signal.
            env_sr (int): the desired sampling rate for the cochlear
                envelopes. 
        """
        super(DownsampleEnvelopes, self).__init__()
        self.sr = sr
        self.env_sr = env_sr
        self.downsample_factor = self.sr/self.env_sr

    def forward(self, x):
        """
        Define the forward pass within each downsample envelope class. 

        Args:
            x (Tensor): the tensor to perform downsampling. Downsampling is 
                performed on the last dimension of this tensor. 

        Returns: 
            (Tensor): the downsampled envelope of the signal. 
        """
        raise NotImplementedError('Forward Pass is not implemented')


class SincWithKaiserWindow(DownsampleEnvelopes):
    """
    Makes a downsampling window that is a sinc function windowed with a kaiser window.
    """
    def __init__(self, sr, env_sr, window_size=1001, padding=None):
        super(SincWithKaiserWindow, self).__init__(sr, env_sr)
        self.window_size = window_size
        self.padding = padding
        if self.sr % self.env_sr !=0:
            raise ValueError('SincWithKaiserWindow downsampling is only '
                             'supported with integer downsampling factors')
        self.downsample_factor = int(self.downsample_factor)
        downsample_filter_np = self._make_downsample_filter_response()
        self.register_buffer('downsample_filter', ch.from_numpy(downsample_filter_np).float())
        
    def forward(self, x): # TODO: implement different padding
        # TODO: is this the fastest way to apply the weighted average?
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        x_shape = x.shape
        if len(x_shape)>2:
            x = x.view(x_shape[0]*x_shape[-2], 1, -1)
        else: # Handle the case where there is no batch dimension
            x = x.view(x_shape[0], 1, -1)

        if (type(self.padding) is not int) and (self.padding is not None):
            # If we have different padding on both sides, need to
            # have a custom padding before the conv
            if len(self.padding) == 2:
                x = ch.nn.functional.pad(x, self.padding)
            # TODO: Does padding get applied appropriately if it is an int? It doesn't seem tobe applied in the conv1d line below. 
        x = ch.nn.functional.conv1d(x, self.downsample_filter, 
                                    stride=self.downsample_factor)
        return x.view(x_shape[0:-1] + (-1,))

    def _make_downsample_filter_response(self):
        downsample_filter_times = np.arange(-self.window_size/2,
                                            int(self.window_size/2))
        downsample_filter_response_orig = (np.sinc(downsample_filter_times / 
                                                  self.downsample_factor) /
                                              self.downsample_factor)
        downsample_filter_window = kaiser(self.window_size, 5)
        downsample_filter_response = (downsample_filter_window * 
                                         downsample_filter_response_orig)
        downsample_filter_response = np.expand_dims(np.expand_dims(
                                         downsample_filter_response, 0), 0)

        return downsample_filter_response

class HannPooling1d(DownsampleEnvelopes):
    """
    1D Weighted average pooling with a Hann window.

    Inputs
    ------
    padding (string): how much padding for the convolution
    normalize (bool): if true, divide the filter by the sum of its values, so that the
        smoothed signal is the same amplitude as the original.
    """
    def __init__(self, sr, env_sr, window_size=9, padding=0, normalize=True):
        super(HannPooling1d, self).__init__(sr, env_sr)
        self.padding = padding
        self.normalize = normalize
        self.window_size = window_size
        if self.sr % self.env_sr !=0:
            raise ValueError('HannPooling1d downsampling is only '
                             'supported with integer downsampling factors')
        self.downsample_factor = int(self.downsample_factor)

        downsample_filter_np = self._make_hann_window()
        self.register_buffer('downsample_filter', ch.from_numpy(downsample_filter_np).float())

    def forward(self, x): # TODO: implement different padding
        # TODO: is this the fastest way to apply the weighted average?
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        x_shape = x.shape
        if len(x_shape)>2:
            x = x.view(x_shape[0]*x_shape[-2], 1, -1)
        else: # Handle the case where there is no batch dimension
            x = x.view(x_shape[0], 1, -1)

        if (type(self.padding) is not int):
            # If we have different padding on both sides, need to
            # have a custom padding before the conv
            if len(self.padding) == 2:
                x = ch.nn.functional.pad(x, self.padding)
            # TODO: Should this overwrite the self.padding specification? It seems like it should? 

        x = ch.nn.functional.conv1d(x, self.downsample_filter,
                                    stride=self.downsample_factor,
                                    padding=self.padding)
        return x.view(x_shape[0:-1] + (-1,))

    def _make_hann_window(self):
        hann_window_w = np.hanning(self.window_size)

        # Add a channel dimensiom to the filter
        hann_window1d = np.expand_dims(np.expand_dims(hann_window_w,0),0)

        if self.normalize:
            hann_window1d = hann_window1d/(sum(hann_window1d.ravel()))
        return hann_window1d


def calculate_same_padding(input_size,
                           kernel_size,
                           stride=1,
                           dilation=1):
    """
    Calculates padding to match "same" padding in tensorflow
    Useful for calculating the integer padding used withe downsampling operations
    """
    pad = ((((input_size + stride - 1) // stride - 1) *
              stride + kernel_size - input_size) * dilation)
    return pad // 2, pad - pad // 2
