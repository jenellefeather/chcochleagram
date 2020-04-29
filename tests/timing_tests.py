import sys
sys.path.append('../')
import chcochleagram
import torch as ch
import numpy as np
import matplotlib.pylab as plt


signal_sizes = [30000, 2**15, 40000, 50000, 2**16]
env_extractions = [chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
                   chcochleagram.envelope_extraction.AbsSubbands,
                   chcochleagram.envelope_extraction.RectifySubbands]

save_results = {}
for signal_size in signal_sizes:
    save_results[signal_size] = {}
    for env_extraction in env_extractions:
        print('%s: Signal Size %d'%(env_extraction, signal_size))

        # Args used for multiple of the cochleagram operations
        sr = 16000
        pad_factor = 1.25
        use_rfft = True

        # Define cochlear filters
        half_cos_filter_kwargs = {
            'n':40,
            'low_lim':50,
            'high_lim':8000,
            'sample_factor':4,
            'include_highpass':False,
            'include_lowpass':False,
            'full_filter':False, 
        }
        coch_filter_kwargs = {'use_rfft':use_rfft,
                              'pad_factor':pad_factor, # Circular convolution when applying the filters
                              'filter_kwargs':half_cos_filter_kwargs}

        filters = chcochleagram.cochlear_filters.ERBCosFilters(signal_size,
                                                               sr, 
                                                               **coch_filter_kwargs)

        # Define an envelope extraction operation
        envelope_extraction = env_extraction(signal_size, sr, use_rfft, pad_factor)


        # Define a downsampling operation
        env_sr = 200
        downsampling_kwargs = {'window_size':1001}
        downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(sr, env_sr, **downsampling_kwargs)


        # Define a compression operation
        compression_kwargs = {'power':0.3,
                              'offset':1e-8,
                              'scale':1,
                              'clip_value':100}

        compression = chcochleagram.compression.ClippedGradPowerCompression(**compression_kwargs)



        cochleagram = chcochleagram.cochleagram.Cochleagram(filters, 
                                                            envelope_extraction,
                                                            downsampling_op,
                                                            compression=compression)
        
        # Build x and generate a cochleagram
        freq = 100
        time_x = np.arange(0, signal_size)/20000;
        # Amplitude of the sine wave is sine of a variable like time=
        amplitude = np.sin(2 * np.pi * freq * time_x)

        freq2 = 120
        amplitude += np.sin(2 * np.pi * freq2 * time_x)

        freq2 = 110
        amplitude += np.sin(2 * np.pi * freq2 * time_x)

        amplitude = np.expand_dims(amplitude, 0)

        x = ch.autograd.Variable(ch.Tensor(amplitude), requires_grad=True)
        y = cochleagram(x)

        import time

        time_start = time.time()

        for i in range(100): 
            y = cochleagram(x)

        print('Total time: %f | Output size %s'%(time.time()-time_start,y.shape))

        save_results[signal_size][env_extraction] = time.time()-time_start

import pickle
pickle.dump(save_results,open( "timing_tests.pckl", "wb" ))
