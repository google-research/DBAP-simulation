DISCLAIMER: Not an official Google product.

These are environments for the DBAP paper submitted to CoRL 2021 in simulation

## Installation

```bash
git clone "sso://adept/adept_envs"
cd adept_envs
pip install -e .
```

## Example Usage

```python
import adept_envs
import gym

env = gym.make('franka_knob_cabinet_slider_resetfree_newcode-v1')
```


