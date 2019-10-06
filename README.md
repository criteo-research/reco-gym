# RecoGym

Python code for the RecSys 2018 REVEAL workshop paper entitled 'RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in Online Advertising'. A pre-print version of the paper can be found here - https://arxiv.org/abs/1808.00720

_RecoGym_ is a Open-AI gym RL environment for recommendation, which is defined by a model of user traffic patterns on e-commerce and the users response to recommendations on the publisher websites. We hope that _RecoGym_ will be an important step forward for the field of recommendation systems research, that could open up an avenue of collaboration between the recommender systems and reinforcement learning communities and lead to better alignment between offline and online performance metrics.

For getting starting with _RecoGym_ please view the 'Getting Started' Jupyter Notebook which will explain the functionality of the environment and detail the creation of a simple agent. The 'Compare Agent' notebook compares the recommendation performance of a selection of our included agents. The agent we include with _RecoGym_ can be found in the agents directory of this repository. 

## Dependencies and Requirements
The code has been designed to support python 3.6+ only. The project has the following dependencies
 and version requirements:

- MarkupSafe==1.1.1
- Send2Trash==1.5.0
- appnope==0.1.0
- attrs==19.2.0
- backcall==0.1.0
- bleach==3.1.0
- cloudpickle==1.2.2
- cycler==0.10.0
- datetime==4.3
- decorator==4.4.0
- defusedxml==0.6.0
- entrypoints==0.3
- future==0.17.1
- gym==0.14.0
- icc==rt-2019.0
- intel==numpy-1.15.1
- intel==openmp-2019.0
- intel==scipy-1.1.0
- ipykernel==5.1.2
- ipython==7.8.0
- ipython==genutils-0.2.0
- ipywidgets==7.5.1
- jedi==0.15.1
- jinja2==2.10.3
- joblib==0.14.0
- jsonschema==3.0.2
- jupyter==1.0.0
- jupyter==client-5.3.3
- jupyter==console-6.0.0
- jupyter==core-4.5.0
- kiwisolver==1.1.0
- llvmlite==0.29.0
- matplotlib==3.1.1
- mistune==0.8.4
- mkl==2019.0
- mkl-fft==1.0.6
- mkl-random==1.0.1.1
- nbconvert==5.6.0
- nbformat==4.4.0
- notebook==6.0.1
- numba==0.45.1
- numpy==1.17.2
- pandas==0.25.1
- pandocfilters==1.4.2
- parso==0.5.1
- pexpect==4.7.0
- pickleshare==0.7.5
- prometheus==client-0.7.1
- prompt==toolkit-2.0.10
- ptyprocess==0.6.0
- pyglet==1.3.2
- pygments==2.4.2
- pyparsing==2.4.2
- pyrsistent==0.15.4
- python==dateutil-2.8.0
- pytz==2019.2
- pyzmq==18.1.0
- qtconsole==4.5.5
- recogym==0.1.2.3
- scikit==learn-0.21.3
- scipy==1.3.1
- simplegeneric==0.8.1
- six==1.12.0
- tbb==2019.0
- tbb4py==2019.0
- terminado==0.8.2
- testpath==0.4.2
- torch==1.2.0
- tornado==6.0.3
- tqdm==4.36.1
- traitlets==4.3.3
- wcwidth==0.1.7
- webencodings==0.5.1
- widgetsnbextension==3.5.1
- zope.interface==4.6.0

In this repository we provide a Anaconda environment setup file with all the required python packages and versions all ready configured. You can install it as follows:

```bash
# install conda env
conda create -n reco-gym python=3.6
conda activate reco-gym

pip install recogym==0.1.2.3
```

For MacOS users, you shall also install _`libomp`_:
```bash
brew install libomp
```

## Cite

Please cite the associated paper for this work if you use this code:

```
@article{rohde2018recogym,
  title={RecoGym: A Reinforcement Learning Environment for the problem of Product Recommendation in Online Advertising},
  author={Rohde, David and Bonner, Stephen and Dunlop, Travis and Vasile, Flavian and Karatzoglou, Alexandros},
  journal={arXiv preprint arXiv:1808.00720},
  year={2018}
}
```

## License

Copyright CRITEO

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
