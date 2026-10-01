import os
import tqdm
from config import get_config
from base import BasicScenario
from solver import REGISTRY

class ProgressBarScenario(BasicScenario):
    """
    自定义场景类，在运行过程中显示进度条
    """
    def run(self):
        self.ready()
        for epoch_id in range(self.config.start_epoch, self.config.start_epoch + self.config.num_epochs):
            instance = self.env.reset()
            # 每个 VNR 有进入和离开两个事件，总事件数为 2 * num_v_nets
            total_steps = self.env.num_v_nets * 2
            pbar = tqdm.tqdm(total=total_steps, desc=f"模拟中 ({self.config.solver_name} | {self.config.p_net_setting['num_nodes']} 节点 | {self.env.num_v_nets} 请求)")
            
            while True:
                solution = self.solver.solve(instance)
                next_instance, _, done, info = self.env.step(solution)
                
                pbar.update(1)
                
                if done:
                    break
                instance = next_instance
            pbar.close()
            # 最终总结数据
            self.env.summary_records()

def run_experiment(num_nodes, solver_name, num_v_nets=None, wm_alpha=0.4, wm_beta=0.2):
    """
    运行单个实验配置
    """
    # 如果没有指定请求数，默认比例为 1:10
    if num_v_nets is None:
        num_v_nets = num_nodes * 10
    
    config = get_config()
    
    # 算法与模型设置
    config.solver_name = solver_name
    if solver_name == 'pg_cnn':
        config.pretrained_model_path = os.path.join('pretrained', 'pg_cnn', 'model-99.pkl')
    elif solver_name == 'pg_gnn':
        config.pretrained_model_path = os.path.join('pretrained', 'pg_gnn', 'model-89.pkl')
    else:
        config.pretrained_model_path = None
        
    config.num_train_epochs = 0
    config.num_epochs = 1
    config.verbose = 1 
    config.renew_v_net_simulator = True  # 强制重新生成虚拟网络，确保 num_v_nets 生效
    
    # 物理网络设置
    config.p_net_setting['num_nodes'] = num_nodes
    config.p_net_setting['topology']['type'] = 'waxman'
    config.p_net_setting['topology']['wm_alpha'] = wm_alpha
    config.p_net_setting['topology']['wm_beta'] = wm_beta
    # 移除可能存在的路径配置，确保从 setting 生成而非加载旧数据
    if 'path' in config.p_net_setting: config.p_net_setting.pop('path')
    if 'file_path' in config.p_net_setting['topology']: config.p_net_setting['topology'].pop('file_path')
        
    # 虚拟网络设置
    config.v_sim_setting['num_v_nets'] = num_v_nets
    if 'path' in config.v_sim_setting: config.v_sim_setting.pop('path')

    # 获取环境和求解器类
    solver_info = REGISTRY.get(config.solver_name)
    Env, Solver = solver_info['env'], solver_info['solver']
    
    # 使用带进度条的场景类
    scenario = ProgressBarScenario.from_config(Env, Solver, config)
    scenario.run()

if __name__ == '__main__':
    # 根据 数据收集.md 的要求执行实验
    
    # 1. 启发式算法 (GRC) 测试
    print("\n>>> 开始 GRC 实验 (50/150 节点)...")
    run_experiment(50, 'grc_rank', 500)
    run_experiment(150, 'grc_rank', 1500)
    
    # 2. 预训练 CNN 测试
    print("\n>>> 开始 CNN 实验 (150 节点)...")
    run_experiment(150, 'pg_cnn', 1500)
    
    # 3. 预训练 GNN 测试 (不同负载规模)
    print("\n>>> 开始 GNN 实验 (150 节点, 不同请求数)...")
    for nv in [2000, 2500, 3000]:
        run_experiment(150, 'pg_gnn', nv)
    
    print("\n[全部完成] 所有实验数据已保存至 save/ 目录。")
