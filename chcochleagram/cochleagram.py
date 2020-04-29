from . import compression
from . import cochlear_filters
from . import envelope_extraction
import torch as ch
import numpy as np
import sys

class Cochleagram(ch.nn.Module):
    """
    Generates a cochleagram, passing an audio signal through a set of bandpass
    filters designed to mimic an ear, followed by envelope extraction, downsampling,
    and a compressive nonlinearity. 
    """
    def __init__(self, filter_object, envelope_extraction, downsampling, 
                 compression=None, downsample_then_compress=True):
        """
        Makes the torch components used for the cochleagram generation. Construct 
        the filter_object, envelope_extraction, downsampling, and (optional) 
        compression modules before using this to piece them together. 
        
        Args: 
            filter_object (cochlear_filters.CochlearFilter) : a set of torch
                filters that will be used for the cochleagram generation. Many 
                parameters for cochleagram generation are inherited from the 
                filters that are constructed. 
            envelope_extraction (envelope_extraction.EnvelopeExtraction) : the 
                torch module that will perform envelope extraction using the 
                filter_object
            downsampling (downsampling.DownsampleEnvelopes) : the torch module 
                to perform downsampling on the cochlear envelopes. 
            compression (compression.CompressionFunction) : the torch module 
                that will perform compression. 
            downsample_then_compress (bool) : If True (default) the envelopes 
                are downsampled before they are compressed. If False, reverses the 
                order and applies the compression operation before the 
                downsampling operation.          
        """
        super(Cochleagram, self).__init__()

        self.numpy_coch_filters = filter_object.coch_filters
        self.numpy_coch_filter_extras = filter_object.filter_extras
        self.sr = filter_object.sr
        self.signal_size = filter_object.signal_size
        self.apply_in_fourier = filter_object.apply_in_fourier
        self.use_rfft = filter_object.use_rfft
        self.downsample_then_compress = downsample_then_compress

        if not self.use_rfft:
            raise NotImplementedError('not using rfft for speed is not tested')
        if envelope_extraction is not None:
            if filter_object.apply_in_fourier != envelope_extraction.apply_in_fourier:
                raise ValueError('Filter objects and envelope extraction operations '
                                 'are not compatible. Check apply_in_fourier.')

        self.pad_factor = filter_object.pad_factor

        self.compute_subbands = ComputeSubbands(self.numpy_coch_filters, 
                                                self.apply_in_fourier,
                                                self.use_rfft, 
                                                self.pad_factor,
                                                self.signal_size)

        self.envelope_extraction = envelope_extraction
        self.downsampling = downsampling
        self.compression = compression

    def forward(self, x, return_latent=False):
        subbands = self.compute_subbands(x)
        envelopes = self.envelope_extraction(subbands)
        
        if self.downsample_then_compress:
            downsampled = self.downsampling(envelopes)
            if self.compression is not None:
                x = self.compression(downsampled)
            else:
                x = downsampled
        elif not self.downsample_then_compress: 
            if self.compression is not None:
                compress = self.compression(envelopes)
            else:
                compress = envelopes
            downsampled = self.downsampling(compress)
            x = downsampled

        if not return_latent:
            return x 
        else:
            return x, {'subbands':subbands,
                       'envelopes':envelopes,
                       'downsampled':downsampled, 
                       'cochleagram':x}
        

class ComputeSubbands(ch.nn.Module):
    """
    Takes the FFT or RFFT of the input waveform, in order to apply cochlear 
    filtering. 
    """
    def __init__(self, coch_filters, apply_in_fourier, use_rfft, 
                 pad_factor, signal_size):
        super(ComputeSubbands, self).__init__()
        self.apply_in_fourier = apply_in_fourier
        self.use_rfft = use_rfft
        self.pad_factor = pad_factor
        self.signal_size = signal_size
    
        self.register_buffer("coch_filters", ch.from_numpy(coch_filters).float())

    def forward(self, x): 
        if self.pad_factor is not None:
            # TODO: Make padding more flexible
            x = ch.nn.functional.pad(x, 
                                     (0,int(self.signal_size*(self.pad_factor-1))), 
                                     mode="constant", value=0) 

        if self.apply_in_fourier:
            x = self._apply_filt_in_fourier(x)
        else: 
            x = self._apply_filt_in_time(x)

        return x

    def _apply_filt_in_fourier(self, x): 
        if "torch.fft" not in sys.modules:
            if self.use_rfft:
                x_fft = ch.rfft(x, 1).unsqueeze_(-3) # Add channel dim
            else:
                x = ch.stack([x, ch.zeros(x.shape)], dim=-1)
                x_fft = ch.fft(x, 1).unsqueeze_(-3) # Add channel dim
        else: # transition to using complex ops in pytorch
            if self.use_rfft:
               x_fft = ch.fft.rfft(x, dim=-1).unsqueeze_(-2) # Add channel dim
            else:
               x_fft = ch.fft(x, dim=-1).unsqueeze_(-2) # Add channel dim
            x_fft = ch.view_as_real(x_fft)
        filtered_signal = self._complex_multiplication(x_fft, self.coch_filters)

        return filtered_signal

    def _apply_filt_in_time(self, x):
        raise NotImplementedError('Applying cochlear filters in the time '
                                      'domain is not yet implemented')

    # for Nx2 tensors
    def _complex_multiplication(self, t1, t2):
      real1, imag1 = [a.squeeze(-1) for a in ch.split(t1, 1, dim=-1)]
      real2, imag2 = [a.squeeze(-1) for a in ch.split(t2, 1, dim=-1)]
      return ch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)
