#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The financial decisions that consumers need to make in their present lifetime, become increasingly more complex. A good example of this phenomenon is the shift from defined benefits to defined contributions in which consumers take on greater individual responsibility and risks. The evolution in the abstruseness of financial products has become challenging for consumers who possess low financial knowledge and limiting numeracy skills {cite}`bi2017financial`. Combined with uncertainty about the future, the consumer is necessitated to be more aware of his financial well-being than ever before. Looking back into the past, Porteba et al, {cite}`poterba2011were` conducted an examination of preparedness in retirement for Children of Depression, War Baby, and the Early Baby Boomer in the Health and Retirement Study and Asset and Health Dynamics Among the Oldest Old cohorts. They found that 46.1 percent die with less than 10 000 dollars. With this amount of assets, they would not have the capacity to pay for unexpected events and one might wonder if it is adequate asset levels for retirement. Furthermore, saving behavior has not kept pace with increasing life expectation and the expected prolonged lifespan of the coming generations are unprecedented {cite}`hershfield2011future`. All these elements give a painstakingly clear picture that having a vital understanding of one's financial situation has become one of the greatest challenges in life.
# 
# To combat these difficulties, consumers require additional undertakings in planning for their future prosperity. One of the approaches to tackle this issue, is by using financial planning tools. These tools give the consumer the capability to estimate complex intertemporal calculations {cite}`bi2020limitations`. They also enhance financial behavior, increase household wealth accumulation and they are a complement to other planning aid like a financial advisor {cite}`bi2017financial`. Although financial planning tools can greatly benefit consumers, it can also be a double-edged sword. More specifically, when consumers are misinformed about the capabilities of the tool, or when the design of the tool is inadequate, the consumer can be given sub-optimal advice or even misleading advice {cite}`dorman2018efficacy`. Insufficiencies in design can arise when not all essential input variables are included, not all risks are considered, and when accuracy is sacrificed for the ease of use {cite}`bi2020limitations`. On top of that, there are wide variations in results because of the various methodology and assumptions used in the models {cite}`dorman2018efficacy`. For example, assumptions based on inflation and the use of different financial products have a large impact on the results. On the side of the consumer, the possibility of misunderstanding the implications of the results due to a lack of financial knowledge, is a matter of great concern in the eyes of financial educators {cite}`bi2020limitations`. Clarifying the results is therefore an essential part of making models operational. To improve upon these deficiencies, Dorman et al., {cite}`dorman2018efficacy` found that when the models handle additional theoretical variables, the accuracy will improve. Besides, they found that the consumer requires unique solutions that better capture their financial situation. Meaning planning tools need to be more flexible. They should be able to operate in different financial settings and have the ability to look at the impact of changes in input variables. To address the variability in results and the adaptability of models to different settings, this paper will look at reinforcement learning techniques in an intertemporal setting. Reinforcement Learning enables an increase in the flexibility of the model while keeping fundamental theoritical aspects like Optimal Control Theory at its core. 
# 
# For the remainder of the paper, the general theory of Reinforcement Learning (RL) will first be introduced. Then, some challenges will be discussed together with Deep Reinforcement Learning. Next, The possible implications of RL for financial planning are considered.  Next, a deep Backward Stochastic Differential Equation method is discussed which will solve the Terminal Partial Differential Equation of the dynammic programming system in higher dimensions. Lastly, an example which will implement this method is presented. 
