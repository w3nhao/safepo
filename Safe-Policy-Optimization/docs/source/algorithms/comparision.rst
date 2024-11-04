Trustworthy Implementation
==========================

To ensure that SafePO's implementation is trustworthy, we have compared 
our algorithms' performance with open source implementations of the same algorithms.
As some of the algorithms can not be found in open source, we selected
``PPO-Lag``, ``TRPO-Lag``, ``CPO`` and ``FOCOPS`` for comparison. 

We have compared the following algorithms:

- ``PPO-Lag``: `OpenAI Baselines: Safety Starter Agents <https://github.com/openai/safety-starter-agents>`_
- ``TRPO-Lag``: `OpenAI Baselines: Safety Starter Agents <https://github.com/openai/safety-starter-agents>`_, `RL Safety Algorithms <https://github.com/SvenGronauer/RL-Safety-Algorithms>`_
- ``CPO``: `OpenAI Baselines: Safety Starter Agents <https://github.com/openai/safety-starter-agents>`_, `RL Safety Algorithms <https://github.com/SvenGronauer/RL-Safety-Algorithms>`_
- ``FOCOPS``: `Original Implementation <https://github.com/ymzhang01/focops>`_

We compared those algorithms in tasks from `Safety-Gymnasium <https://github.com/PKU-Alignment/safety-gymnasium>`_,

.. warning::

     It may takes some time to load the results.
     If you can not see the results, please directly visit `wandb.ai <https://wandb.ai/pku_rl/SafePO/reports?view=table>`_.

The results are shown as follows.

.. tab-set::

    .. tab-item:: PPO-Lag

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Comparison-of-PPO-Lag-s-Implementation--Vmlldzo1MTgxOTkx" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>

    .. tab-item:: TRPO-Lag

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Comparison-of-TRPO-Lag-s-Implementation--Vmlldzo1MTgyMDAz" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>

    .. tab-item:: CPO

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Comparison-of-CPO-s-Implementation--Vmlldzo1MTgyMDA2" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>

    .. tab-item:: FOCOPS

      .. raw:: html

         <iframe src="https://wandb.ai/pku_rl/SafePO/reports/Comparison-of-FOCOPS-s-Implementation--Vmlldzo1MTgxOTg3" style="border:none;width:90%; height:1000px" >

      .. raw:: html

         </iframe>