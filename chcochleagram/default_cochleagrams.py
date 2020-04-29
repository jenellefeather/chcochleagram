import chcochleagram

cochleagram_1 = {'rep_type': 'cochleagram',
                 'signal_size':40000,
                'sr':20000,
                'env_sr': 200,
                'pad_factor':None,
                'use_rfft':True,
                'coch_filter_type': chcochleagram.cochlear_filters.ERBCosFilters,
                'coch_filter_kwargs': {
                    'n':50,
                    'low_lim':50,
                    'high_lim':10000,
                    'sample_factor':4,
                    'full_filter':False,
                    },
                'env_extraction_type': chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
                'downsampling_type': chcochleagram.downsampling.SincWithKaiserWindow,
                'downsampling_kwargs': {
                    'window_size':1001},
                 'compression_type': chcochleagram.compression.ClippedGradPowerCompression,
                 'compression_kwargs': {'scale': 1,
                                        'offset':1e-8,
                                        'clip_value': 5, # This wil clip cochleagram values < ~0.04
                                        'power': 0.3},
                }

