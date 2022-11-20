# chcochleagram

PyTorch modules for cochleagram generation, allowing for gradient computations on the cochleagram generation graph. Cochleagrams are a variation on spectrograms  with filter shapes and widths motivated by human perception. Default arguments use half cosine filters at erb spacing. Custom filters can alternatively be provided. After initial (bandpass) filtering, the signals are envelope extracted, compressed, and downsampled to construct the cochleagram representation. 

## Installation
The easiest way to install chcochleagram is by pip installing directly from git: 

`pip install git+git://github.com/jfeather/chcochleagram`

Alternatively, you can clone the respository and run `pip install .` or other installation methods using the setup.py file. 

## Getting Started
A demonstration of cochleagram generation is provided in [notebooks/ExampleCochleagramGeneration.ipynb](notebooks/ExampleCochleagramGeneration.ipynb)

### Prerequisites
```
pytorch
numpy
scipy
```

## Related Repositories
* tfcochleagram (tensorflow cochleagram generation, used in [2,3]): https://github.com/jenellefeather/tfcochleagram
* pycochleagram (python cochleagram generation): https://github.com/mcdermottLab/pycochleagram

## Authors
* **Jenelle Feather** (https://github.com/jfeather)

## Contributors
* Ray Gonzalez (pycochleagram filters)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* McDermott Lab (https://github.com/mcdermottLab)

## Citation
This repository was released with the following publication. If you use this repository in your research, please cite as: 

[Feather, J., Leclerc, G., Mądry, A., & McDermott, J. H. (2022). Model metamers illuminate divergences between biological and artificial neural networks. bioRxiv.](https://www.biorxiv.org/content/10.1101/2022.05.19.492678v1)

```
@article{feather2022model,
  title={Model metamers illuminate divergences between biological and artificial neural networks},
  author={Feather, Jenelle and Leclerc, Guillaume and M{\k{a}}dry, Aleksander and McDermott, Josh H},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

## References
[1] McDermott J. and Simoncelli E. Sound Texture Perception via Statistics of the Auditory Periphery: Evidence from Sound Synthesis. Neuron (2011). 

[2] Feather J. and McDermott J. Auditory texture synthesis from task-optimized convolutional neural networks. Conference on Cognitive Computational Neuroscience (2018). 

[3] Feather J., Durango A., Gonzalez R., and McDermott J. Metamers of neural networks reveal divergence from human perceptual systems. Advances in Neural Information Processing Systems (2019). 
