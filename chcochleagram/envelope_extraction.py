import numpy as np
import torch as ch
import sys

class EnvelopeExtraction(ch.nn.Module):
    """
    Base class for envelope extraction functions. 
    Extracts the envelope of a time-signal.
    """
    def __init__(self, signal_size, sr, 
                 use_rfft, pad_factor):
        """
        Base class for each of the envelope extraction functions.

        Args:
            signal_size (int) : the length of the input signal. Must be fixed
                at initialization if the filters are applied in the fourier
                domain. Otherwise, argument is ignored (can be set to None)
            sr (int) : the sampling rate for the input signal.
            use_rfft (Boolean) : If true, use rfft/ifft operations if in the 
                fourier domain.
            pad_factor (int) : If None (default), the signal will not be
                padded. Otherwise, assume the waveform signal will be padded
                to length pad_factor*signal_length.
        """
        super(EnvelopeExtraction, self).__init__()
        self.signal_size = signal_size
        self.sr = sr
        self.use_rfft = use_rfft
        self.pad_factor = pad_factor

    def forward(self, x):
        """
        Each envelope extraction function should implement their own 
        forward pass. 

        Args:
            x (Tensor): the tensor to perform envelope extraction
        Returns:
            (Tensor): the envelope extracted signal, in the time domain. 
        """
        raise NotImplementedError('Forward Pass is not implemented')


class HilbertEnvelopeExtraction(EnvelopeExtraction):
    def __init__(self, signal_size, sr,
                         use_rfft, pad_factor):
        super(HilbertEnvelopeExtraction, self).__init__(signal_size, sr, use_rfft, pad_factor)
        self.apply_in_fourier = True
        if (self.signal_size is None):
            raise ValueError('Must specify signal_size (currently None) if '
                             'performing envelope extraction via Hilbert')

        if not self.use_rfft:
            if self.pad_factor is not None: 
                freq_signal = np.fft.fftfreq(self.signal_size*self.pad_factor, 
                                             1./self.sr)
            else:
                freq_signal = np.fft.fftfreq(self.signal_size,
                                             1./self.sr)
            self.step_tensor = self._make_step_tensor(freq_signal)
        else: 
            if self.pad_factor is not None:
                self.hilbert_pad = int((self.signal_size * self.pad_factor) / 2) - 1
            else:
                self.hilbert_pad = int(self.signal_size / 2) - 1
     
    def forward(self, x):
        if not self.use_rfft:
            x = ch.mul(x, self.step_tensor)
        else:
            # Last dim is the real/imaginary channel, so do not pad it. 
            x = ch.nn.functional.pad(x, (0, 0, 0, self.hilbert_pad),
                                     mode='constant', value=0)
        if "torch.fft" not in sys.modules:
            x = ch.ifft(x, 1)
        else:
            x = ch.view_as_real(ch.fft.ifft(ch.view_as_complex(x), dim=-1)).contiguous()
        # Remove the padding that was applied -- only return signal size 
        if self.pad_factor is not None:
            x, _ = ch.split(x,
                            [self.signal_size, 
                                 int(self.pad_factor*self.signal_size)-self.signal_size],
                            dim=-2)
        x = complex_abs(x)  
        return x 
        
        
    def _make_step_tensor(self, freq_signal):
        """
        Make step tensor for calculating the anlyatic envelopes.
      
        Args:
            freq_signal (array): numpy array containing the frequencies of the audio signal
     
        Returns: 
            step_tensor (tensor): tensor with dimensions [len(freq_signal)] as a step function
               where frequencies > 0 are 1 and frequencies < 0 are 0. 
    
        """
        step_func = (freq_signal>=0).astype(np.int)*2 # wikipedia says that this should be 2x the original.
        step_func[freq_signal==0] = 0 # https://en.wikipedia.org/wiki/Analytic_signal| 
        # Must have a real and imaginary component
        step_tensor = ch.Tensor(np.stack([step_func, step_func],-1))
        return step_tensor

class RectifySubbands(EnvelopeExtraction):
    """
    Combining rectified subbands with a lowpass filter results in an envelope
    extraction. Rather than performing the lowpass filter here, it should be 
    applied as the antialiasing filter step of the downsampling operation. 
    """
    def __init__(self, signal_size, sr,
                 use_rfft, pad_factor):
        super(RectifySubbands, self).__init__(signal_size, sr,
                                              use_rfft, pad_factor)
        self.apply_in_fourier = True
        if pad_factor is not None:
            self.signal_size_with_padding = int(self.pad_factor*self.signal_size )
        else:
            self.signal_size_with_padding = self.signal_size
        
    def forward(self, x):
        if "torch.fft" not in sys.modules:
            if self.use_rfft:
                x = ch.irfft(x, 1, signal_sizes=[self.signal_size_with_padding])
            else:
                x = ch.ifft(x, 1)
                # Get only the real part, because these are waveforms
                # TODO(jfeather): Use complex numbers
                x, _ = ch.split(x, 1, dim=-1)
        else:
            if self.use_rfft:
                x = ch.fft.irfft(ch.view_as_complex(x), dim=-1, 
                                 n=self.signal_size_with_padding)
            else:
                x = ch.fft.ifft(ch.view_as_complex(x), dim=-1)
                x, _ = ch.split(ch.view_as_real(x), 1, dim=-1)
                x = x.contiguous()

        # Remove the padding that was applied -- only return signal size
        if self.pad_factor is not None:
            x, _ = ch.split(x,
                            [self.signal_size,
                                 int(self.pad_factor*self.signal_size)-self.signal_size],
                            dim=-1)

        # Rectified subbands -- combined with downsampling they will be envelopes
        x = ch.nn.functional.relu(x)
        return x 

class AbsSubbands(EnvelopeExtraction):
    """
    Combining the magnitude of subbands with a lowpass filter results in an envelope
    extraction. Rather than performing the lowpass filter here, it should be
    applied as the antialiasing filter step of the downsampling operation.
    """
    def __init__(self, signal_size, sr,
                 use_rfft, pad_factor):
        super(AbsSubbands, self).__init__(signal_size, sr,
                                              use_rfft, pad_factor)
        self.apply_in_fourier = True
        if pad_factor is not None:
            self.signal_size_with_padding = int(self.pad_factor*self.signal_size )
        else:
            self.signal_size_with_padding = self.signal_size

    def forward(self, x):
        if "torch.fft" not in sys.modules:
            if self.use_rfft:
                x = ch.irfft(x, 1, signal_sizes=[self.signal_size_with_padding])
            else:
                x = ch.ifft(x, 1)
                # Get only the real part, because these are waveforms
                x, _ = ch.split(x, 1, dim=-1)
        else:
            if self.use_rfft:
                x = ch.fft.irfft(ch.view_as_complex(x), dim=-1, 
                                 n=self.signal_size_with_padding)
            else:
                x = ch.fft.ifft(ch.view_as_complex(x), dim=-1)
                x, _ = ch.split(ch.view_as_real(x), 1, dim=-1)
                x = x.contiguous()

        # Remove the padding that was applied -- only return signal size
        if self.pad_factor is not None:
            x, _ = ch.split(x,
                            [self.signal_size,
                                 int(self.pad_factor*self.signal_size)-self.signal_size],
                            dim=-1)
        # Abs subbands -- combined with downsampling they will be envelopes
        x = ch.abs(x)

        return x
        
# for Nx2 tensors
def complex_abs(t1):
  real1, imag1 = [a.squeeze(-1) for a in ch.split(t1, 1, dim=-1)]
  return ch.sqrt(ch.clamp((real1**2 + imag1**2), min=1e-16))

