## base包：

1. controller：控制所有网络，执行嵌入操作。拥有所有节点和链路的属性与资源信息的读写，能够决定节点映射和链路映射的方式。大部分操作都有safely和unsafely，前者保证操作满足某些约束。一些动作支持undo

   ​	a. place, 将一个虚拟节点安置到一个物理节点上

   ​	b. route，将虚拟网络中的一个虚拟链路映射到物理网络中的一个path中

   ​	c. place_and_route, 执行place，成功place则对虚拟机点的所有已安置的邻节点进行route	

2. counter：计算过程中的各类数据，如收益、开销等。

3. environment：基础环境，可以执行step()，返回(next_state, reward, done, info)。以及状态转换。或者说是运行一次项目的环境，它包含controller、counter、recorder，虚拟网络嵌入请求模拟器，物理网络

4. recorder：负责记录任务中的各个event的执行信息，以及环境当前的状态的信息

5. scenario：一次业务场景。即在env环境下，使用solver进行了切片嵌入训练。相关设置由config指定。

6. solution：对某一个切片嵌入请求的解决方案，可能是一个成功的方案，也可能是一个失败的反馈。相当于强化学习中agent的动作（solver接受instance，分析后给出solution）。

整个过程就是在一个指定的env下，我们跑n次scenario，每次都是用某个solver来解决一套数据集。

## data包：

​	关于底层物理网络和虚拟网络的定义

## solver包：

​	我们仅关注pg_cnn。PgCnnSolver同时继承了InstanceAgent, PGSolver。PGSolver又继承自RLSolver，RLSolver继承自Solver，其中只包含了一些基础信息

## pg_cnn的环境逐层解剖：

1. `class SubEnv(PlaceStepSubRLEnv):`负责计算奖励、计算下一个观测状态
1. `class PlaceStepSubRLEnv(SubRLEnv):`负责执行step()。先后执行节点映射和链路映射。返回四元组
1. `class SubRLEnv(VNERLEnv):`拥有物理虚拟网络、counter、recorder、controller。并能够执行reset()、reject()、revoke()。同时有一个calcuate_graph_metrics()方法，用于计算图的属性
1. class VNERLEnv(gym.Env):主要是关于能否reject和revoke的参数设置，有许多获取节点部署信息的方法

我们执行使用pg_cnn时，主程序会在base.environment中执行，但是controller会将当前状态s交给solver，让它在自身的subEnv中解决



GNN的提出论文：F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini,“The graph neural network model,” IEEE Transactions on Neural Networks, vol. 20, no. 1, pp. 61–80, 2008



我们更改了train_epochs（1000->100),v_net_nu ms(1000->500)



p_net：PhysicalNetwork with 100 nodes and 482 edges

## PG_CNN的执行流程（从sceario.run()开始）：

​	进入到scenario，执行scenario.ready():

​		如果solver有load_model()方法且pretrained_model_path有意义，则加载预训练的模型

​		如果solver需要训练，且训练轮数＞0，则执行solver.learn(), 进入到solver：

​			for 每一轮训练：

​				重置环境，得到初始状态instance。其中包含目前需要映射的虚拟网络v_net，和物理网络p_net

​					for 每一个待隐射的虚拟网络：

​						执行RLSolver的learn_with_instacne(instance):

​							初始化缓冲区sub_buffer，初始化sovler的子环境sub_env，执行子环境的get_observation得到sub_obs。包含节点属性信息、边属性信息、平均距离、节点度信息）

​							while 目前虚拟网络嵌入操作未完成：

​								预处理sub_obs，让它适配solver的select_action()

​								执行solver.select_action():

​									执行solver.policy.act(obs)得到动作概率。

​									对动作概率执行mask操作

​									采样得到具体动作，即对于目前的虚拟网络节点，应该映射到哪个物理网络节点

​									返回执行动作和动作的对数概率

​								使用solver.estimate_obs(obs)评估得到value，即调用policy的critic()

​								sub_env接收动作，虚拟网络的一个节点映射完毕，进入下一个状态，返回四元组next_sub_obs, sub_reward, sub_done, sub_info

​								将本次操作的sub_obs, action, sub_reward, sub_done, action_logprob, value录入缓冲区

​								如果虚拟网络嵌入完成，结束，否则sub_obs = next_sub_obs

​							对于本次instance，我们的agent给出了一个solution，它或许可行。

​							计算关于最后一次sub_obs的评估值last_value

​							返回soluton，sub_buffer，last_value。

​						执行solver.self.merge_instance_experience(instance, solution, sub_buffer, last_value):

​							如果方案可行，计算sub_buffer的总奖励，并将其归并到buffer

​						如果方案可行，则成功次数+1，记录本次solution的r2c

​						如果当前缓冲区积累的节点映射经验超过256次，则更新agent的网络参数sovler.update():

​							获得buffer中的observations，actions，returns，masks

​							执行self.evaluate_actions(observations, actions, masks=masks, return_others=True)得到当前policy对所有actions的对数概率。

​							loss = - (action_logprobs * returns).mean()

​							loss.backward()

​							清空buffer，返回loss





​						

## a3c_gcn_seq2seq的执行流程（从sceario.run()开始）：

​	进入到scenario，执行scenario.ready():

​		如果solver有load_model()方法且pretrained_model_path有意义，则加载预训练的模型

​		如果solver需要训练，且训练轮数＞0，则执行solver.learn(), 进入到solver：

​			for 每一轮训练：

​				重置环境，得到初始状态instance。其中包含目前需要映射的虚拟网络v_net，和物理网络p_net

​					for 每一个待隐射的虚拟网络：

​						执行A3CGcnSeq2SeqSolver的learn_with_instacne(instance):

​							初始化缓冲区sub_buffer，初始化sovler的子环境sub_env，执行子环境的get_observation得到encoder_obs，包含图神经网络需要的输入p_net_x，p_net_edge_index）

​							对encoder_obs进行处理，让它被policy.encode()接收，得到encoder_outputs。同时记录虚拟网络的初始隐藏信息hidden_state。

​							改变encoder_outputs的形状，初始化sub_obs，包含**上一次选取的物理节点id、hidden_state，图神经网络的输入，encoder_outputs**

​							while 目前虚拟网络嵌入操作未完成：

​								更新hidden_state为最新的隐藏信息

​								计算掩码mask

​								预处理sub_obs，让它适配solver的select_action()

​								执行solver.select_action():

​									执行solver.policy.act(obs)得到动作概率。

​									对动作概率执行mask操作

​									采样得到具体动作，即对于目前的虚拟网络节点，应该映射到哪个物理网络节点

​									返回执行动作和动作的对数概率

​								使用solver.estimate_obs(obs)评估得到value，即调用policy的critic()。得到value

​								sub_env接收动作，虚拟网络的一个节点映射完毕，进入下一个状态，返回四元组next_sub_obs, sub_reward, sub_done, sub_info。当前的next_sub_obs其实是下一状态的enconder_obs

​								将本次操作的sub_obs, action, sub_reward, sub_done, action_logprob, value录入缓冲区

​								如果虚拟网络嵌入完成，结束，否则sub_obs = next_sub_obs

​							对于本次instance，我们的agent给出了一个solution，它或许可行。

​							计算关于最后一次sub_obs的评估值last_value

​							返回soluton，sub_buffer，last_value。

​						执行solver.self.merge_instance_experience(instance, solution, sub_buffer, last_value):

​							如果方案可行，计算sub_buffer的总奖励，并将其归并到buffer

​						如果方案可行，则成功次数+1，记录本次solution的r2c

​						如果当前缓冲区积累的节点映射经验超过256次，则更新agent的网络参数sovler.update():

​							获得buffer中的observations，actions，returns，masks

​							执行self.evaluate_actions(observations, actions, masks=masks, return_others=True)得到当前policy对所有actions的对数概率。

​							loss = - (action_logprobs * returns).mean()

​							loss.backward()

​							清空buffer，返回loss

100node，498link的pnet1

![image-20230515145453630](C:\Users\81970\AppData\Roaming\Typora\typora-user-images\image-20230515145453630.png)



初始化scenario：初始化counter、controller：分析节点属性和链路属性，指定链路寻找使用bfs。然后初始化record，主要是定义保存路径。读取物理网络的拓扑结构，如果没有则随即生成。接着初始化环境和solver。初始化RLsolver时可以指定是否使用一个基准解法，默认可以是grc，以及超参数。GNN的神经网络入度为节点数（100），物理网络节点特征维度为5，虚拟网络节点特征维度为2，嵌入维度为128。演员和评论员的学习率均为0.001





初始环境：物理网络信息、下一个带创建的切片信息

状态：当前物理网络信息，下一个待映射节点

动作空间：一个[0,p_net_num)的整数，指定当前节点映射到哪个物理节点

奖励：成功映射则该动作到 1/v_net_nums 奖励，否则 -1/v_net_nums 奖励



对于一个方案的奖励，成功则奖励为1+r2c

大环境检验方案的可行性，给出下一个初始环境





agent对状态的观测：

​	p_net_x:(p_node_nums,5)

​	p_net_edge_index(2,p_link_nums*2)

​	v_net_x:(p_node_nums,5,2)

执行前agent会将观测状态转为图神经网络的输入pyg.BatchData形式。输出动作，及对数概率



环境接收动作。首先会判断该动作是否有效，另外如果这个切片创建与环境交互了太多次，则自动放弃它，如果选择了重复点（已经给切片中其他节点用的物理节点）则直接失败。如果动作可行，则放置节点，并使用bfs进行路由。如果成功放置节点并且成功路由，则返回下一观测状态。否则直接失败。

收益：v_net_node_revenue + v_net_link_revenue / v_net_node_revenue + v_net_link_cost



更新：当收录超过256次经验（solution）时进行更新。returns是历史收益（得分）。 演员接收历史数据bs*p_num，5。重新学习后输出bs，p_num再动作对数概率action_logprobs，它是对于之前的执行的老动作在目前actor心中的对数概率，我们要学习的就是以目前的眼光来看，曾经选择这个老动作是不是不错。评论员也接受历史数据bs*p_num，5。重新评判后输出bs，1再评分values。

advantage就是returns-再评分values，并且会归一化。之后演员和评论员的损失如下

```
actor_loss = - (action_logprobs * advantages).mean() 均值
critic_loss = F.mse_loss(returns, values) #均方误差的平均值  Σ(x-y)^2
```

总损失则是al+cl+el



输入的node_data：100*5：特征：（剩余CPU、CPU、邻边带宽总和、邻边最大带宽总和，是否已选）

