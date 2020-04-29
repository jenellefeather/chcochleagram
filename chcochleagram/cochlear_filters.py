import numpy as np
import torch as ch
import sys
from .helpers import erb_filters as erb


class CochlearFilter:
    """
    Filterbanks designed to model the processing in the human cochlea. 
    """
    def __init__(self, signal_size, sr, apply_in_fourier=True, 
                 use_rfft=True, pad_factor=None, filter_kwargs={}):
        """
        Base class for each of the cochlear filters. 
       
        Args: 
            signal_size (int) : the length of the input signal. Must be fixed
                at initialization if the filters are applied in the fourier
                domain. Otherwise, argument is ignored (can be set to None)
            sr (int) : the sampling rate for the input signal.
            apply_in_fourier (Boolean) : If true, filters are designed in 
                the fourier domain, and we should multiply the fft of the
                signal with the appropriate filters
            use_rfft (Boolean) : If true, make filters that will be applied
                to real valued signals only. 
            pad_factor (int) : If None (default), the signal will not be 
                padded before filtering. Otherwise, the filters will be 
                created assuming the waveform signal will be padded to length 
                pad_factor*signal_length by adding zeros to the end of the 
                signal. 
        """
        super(CochlearFilter, self).__init__()
           
        self.signal_size = signal_size
        self.sr = sr
        self.apply_in_fourier = apply_in_fourier
        self.use_rfft = use_rfft
        self.pad_factor = pad_factor
        self.filter_kwargs = filter_kwargs
        if (self.signal_size is None) and self.apply_in_fourier:
            raise ValueError('Must specify signal_size (currently None) if ' 
                             'apply_in_fourier is True')
        self.coch_filters, self.filter_extras  = self._chfilters(
                                                     self.filter_kwargs)

    def _npfilters(self, **kwargs):
        """ 
        Contains cochlear filters in a numpy array.
        This function should be defined in the specific class type. 
        
        Args:
            **kwargs : kwargs that will be used by the filter constructor
        Returns: 
            filters (numpy array) : filters that were constructed
            filter_extras (dict) : extra returns from the filter constructor, 
                for instance the center frequencies of the filters or the 
                frequency mapping.
        """
        raise NotImplementedError("Numpy filters are not implemented")

    def _chfilters(self, filter_kwargs):
        """
        Contains versions of the filters that have operations applied to 
        work with torch (ie adding extra dim for complex numbers)
        """
        np_filters, filter_extras = self._npfilters(**filter_kwargs)
        if np_filters.dtype=='float64':
            np_filters = np_filters.astype(np.float32)

        if self.apply_in_fourier:
            # Currently complex valued numbers are handled by stacking the
            # real and complex components.
            # TODO: update to use complex operations for ch>1.8
            if len(np_filters.shape) == 2:
                np_filters = np.stack([np_filters,
                                       np.zeros(np_filters.shape)],
                                       axis=-1)

        chfilters = np_filters
        return chfilters, filter_extras


class ERBCosFilters(CochlearFilter):
    """
    Makes Half cosine filters on an ERB scale. 
    """
    def _npfilters(self, n=40, low_lim=20, high_lim=10000, sample_factor=4, 
                  **kwargs):
        """
        Generates the numpy filters, with arguments used by make_cos_filters_nx
       
        Args:
            n (int) : number of filters to uniquely span the frequency space
            low_lim (int) : Lower frequency limits for the filters.
            high_lim (int) : Higher frequency limits for the filters.
            sample_factor (int) : number of times to overcomplete the filters.
            kwargs (dict): additional kwargs to erb.make_erb_cos_filters

        Returns:
            npfilters (array): numpy array containing the erb filters
            filter_extras (dict): dictionary containing the center frequencies
                of the filters and the frequency mapping.
        """
        if not self.apply_in_fourier:
            raise ValueError('CosFilters are designed to be applied in the '
                             'fourier domain, but apply_in_fourier is False')
        if self.pad_factor is not None:
            padding_size = int((self.pad_factor-1)*self.signal_size)
        else:
            padding_size = None
        npfilters, cf, freqs = erb.make_erb_cos_filters_nx(self.signal_size, self.sr,
                                                       n, low_lim, high_lim, 
                                                       sample_factor,
                                                       padding_size=padding_size,
                                                       **kwargs)
    
        # Use NCHW convention, so channels should be in the first axis. 
        if self.use_rfft: # TODO: pycochleagram is not consistent with dimension
            npfilters = npfilters.T

        filter_extras = {'cf': cf, 'freqs': freqs}
        return npfilters, filter_extras
        
