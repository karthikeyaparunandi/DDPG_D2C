### Introduction

---

<span style="float:right;">[[source]](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py#L89)</span>
### DQNAgent

```python
rl.agents.dqn.DQNAgent(model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg')
```


__Arguments __

model__: A Keras model. 
policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py). 
test_policy__: A Keras-rl policy. 
enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting. 
enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al. 
dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581). 
	`avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta))) 
	`max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta))) 
	`naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta) 
 


---

### References
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), Mnih et al., 2013
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html), Mnih et al., 2015
- [Deep Reinforcement Learning with Double Q-learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/doubledqn.pdf), van Hasselt et al., 2015
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), Wang et al., 2016
