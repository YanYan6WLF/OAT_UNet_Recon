# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       visualizer.py
   Project Name:    project_name
   Author :         Yan
   Date:            2025/10/25
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/10/25:
-------------------------------------------------
"""

import visdom # # 
import numpy as np


class Visualizer(object):
    def __init__(self, env='main', port=31430, **kwargs):
        self.viz = visdom.Visdom(env=env, port=port, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}

    
    # plot line
    def plot_multi_win(self, d, loop_flag=None): 
        '''
        一次plot多个或者一个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        long_update = True 
        if loop_flag == 0:
            long_update = False
        for k, v in d.items():
            self.plot(k, v, long_update)

    def plot(self, name, y, long_update, **kwargs):# **kwargs 额外选项（如 nrow 每行显示几张图）
        '''
        self.plot('loss', 1.00)
        One mame, one win: only one lie in a win.
        '''
        x = self.index.get(
            name, 0)  # dict.get(key, default=None). 返回指定键的值，如果值不在字典中返回default值;
            # 第一次调用 → 取默认 0 下一次 → 取上次记录的值（例如 3）
            # 这样就能实现“自动递增横坐标”。
        self.viz.line(Y=np.array([y]), X=np.array([x]),
                    win=name,
                    opts=dict(title=name),
                    update='append' if (x > 0 and long_update) else None, # 可以通过控制long_update操控update，人为控制加点还是重绘？
                    **kwargs)
        self.index[name] = x + 1    # Maintain the X

    def img(self, img_, name, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.viz.images(#  → 调用 visdom 的图片显示函数。  可以一次性显示多张图（例如 [100, 3, 64, 64]）。
                        img_.cpu().numpy(),#  → 把 PyTorch Tensor 转换成 numpy 数组（Visdom 只能处理 numpy）。
                        win=name,
                        opts=dict(title=name), #  → 设置图片窗口的标题（显示在网页上）。
                        **kwargs
                        )
