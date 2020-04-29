import torch as ch
import chcochleagram

class AudioToCochleagram(ch.nn.Module):
    """
    Converts audio to cochleagram
     
    Wrapper that takes in cochleagram kwargs as a dictionary. 
    """
    def __init__(self, cgram_kwargs):
        super(AudioToCochleagram, self).__init__()
        self.cgram_kwargs = cgram_kwargs

        # Args used for multiple of the cochleagram operations
        self.signal_size = self.cgram_kwargs['signal_size']
        self.sr = self.cgram_kwargs['sr']
        self.pad_factor = self.cgram_kwargs['pad_factor']
        self.use_rfft = self.cgram_kwargs['use_rfft']

        # Define cochlear filters
        self.coch_filter_kwargs = self.cgram_kwargs['coch_filter_kwargs']

        self.make_coch_filters = self.cgram_kwargs['coch_filter_type']
        self.filters = self.make_coch_filters(self.signal_size,
                                              self.sr, 
                                              use_rfft=self.use_rfft,
                                              pad_factor=self.pad_factor,
                                              filter_kwargs=self.coch_filter_kwargs)

        # Define an envelope extraction operation
        self.env_extraction = self.cgram_kwargs['env_extraction_type']
        self.envelope_extraction = self.env_extraction(self.signal_size, 
                                                       self.sr, 
                                                       self.use_rfft, 
                                                       self.pad_factor)

        # Define a downsampling operation
        self.downsampling = self.cgram_kwargs['downsampling_type']
        self.env_sr = self.cgram_kwargs['env_sr']
        self.downsampling_kwargs = self.cgram_kwargs['downsampling_kwargs']
        self.downsampling_op = self.downsampling(self.sr, self.env_sr, 
                                                 **self.downsampling_kwargs)

        # Define a compression operation
        self.compression = self.cgram_kwargs['compression_type']
        self.compression_kwargs = self.cgram_kwargs['compression_kwargs']
        self.compression_op = self.compression(**self.compression_kwargs)
        
        # Make the full cochleagram 
        cochleagram = chcochleagram.cochleagram.Cochleagram(
                                           self.filters, 
                                           self.envelope_extraction,
                                           self.downsampling_op,
                                           compression=self.compression_op)

        self.Cochleagram = cochleagram

    def forward(self, input_audio):
        """
        Args:
            input_audio (torch.Tensor): the waveform that will be used as
                the audio sample
        """
        return self.Cochleagram(input_audio)
