import csv

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from config import get_config, show_config, save_config, load_config
from data import Generator
from base import BasicScenario
from solver import REGISTRY
from fastapi.responses import JSONResponse
app = FastAPI()


# 定义一个返回字典实例的函数
def create_dict_from_file_content(content: str):
    # 假设文件内容为JSON字符串，解析为字典
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}


# JSON请求体的Pydantic模型
class Item(BaseModel):
    data: dict


p_net_path = './dataset/p_net/p_net-cpu_[50-100]-max_cpu_None-bw_[50-100]-max_bw_None-ltc_[0.0-1.0]'
after_p_net_path = './dataset/p_net/aftersolution'
v_net_path = './dataset/v_nets/1-[2-5]-random-1000-0.04-cpu_[0-50]-bw_[0-50]'
n_net_path = 'save/pg_gnn/gnn_cpu_bw_default/model/model-89.pkl'


@app.post("/drl/")
async def process_gml_data(gml_data: dict):
    gml_files = gml_data.get("gml_files", [])
    saved_files = []
    for gml_file in gml_files:
        filename = gml_file.get("filename")
        content = gml_file.get("content")
        if filename == "p_net.gml":
            file_location = os.path.join(p_net_path, filename)
        else:
            file_location = os.path.join(v_net_path, 'v_nets', filename)
        # 将 GML 数据写入 .gml 文件
        with open(file_location, "w") as file:
            file.write(content)

    config = get_config(args=[])
    config.solver_name = 'pg_gnn'  # modify the algorithm of the solver
    config.pretrained_model_path = n_net_path
    config.num_train_epochs = 0
    config.p_net_setting['path'] = p_net_path
    config.v_sim_setting['path'] = v_net_path
    summary_info, record_path, summary_path = run(config)
    #print(record_path)
    with open(record_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        solution = next(reader)

    with open(os.path.join(after_p_net_path, 'p_net.gml'), 'r') as gml_file:
        p_net_content = gml_file.read()
    #print(loaded_data)
    #print(type(loaded_data))
    return JSONResponse(content={
        "solution": solution,   # 返回字典内容
        "p_net": p_net_content  # 将GML内容作为字符串包含在JSON响应中
    })


def run(config):
    solver_info = REGISTRY.get(config.solver_name)
    Env, Solver = solver_info['env'], solver_info['solver']
    scenario = BasicScenario.from_config(Env, Solver, config)
    return scenario.run()
