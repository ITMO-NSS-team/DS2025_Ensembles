#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install kaleido')


# In[1]:


import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch
import numpy as np
import pandas as pd

import dill as pickle

import seaborn as sns
import random
import math


# Burgers equation
# 
# 
# $$
# \\ 256\times256, x \in [0; 1], t \in [0; 1].
# $$

# In[2]:


grid_res = 255
title = 'burgers'
#df = pd.read_csv('C:\\Users\\YOUR_PATH\\burgers_sln_256.csv', header=None)
initial_data = torch.from_numpy(df.to_numpy()).reshape(-1)

x_grid = np.linspace(0, 1, grid_res +1)
t_grid = np.linspace(0, 1, grid_res +1)

params = [x_grid, t_grid]

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()


# In[3]:


#PATH_TO SIMPLE ENSEMBLE DATA
#u_sde = torch.load('C:\\Users\\YOUR_PATH\\pt\\sols_stacked_1_solutions_shape_torch.Size([1, 256, 256]).pt', weights_only=True)


# In[4]:


u_sde = u_sde.squeeze().T.detach().cpu().numpy()


# In[5]:


u_sde_mean_tens = u_sde.reshape(*[len(i) for i in params])#.detach().cpu().numpy()


# In[6]:


#PATH_TO BN ENSEMBLE DATA
#u_main = torch.load('C:\\Users\\YOUR_PATH\\file_u_main_[32, 256, 256]_mat_[0].pt', weights_only=False)


# In[7]:


u_main.shape


# In[8]:


def calculate_statistics(u_main):

    mean_arr = np.zeros((u_main.shape[1], u_main.shape[2]))
    var_arr = np.zeros((u_main.shape[1], u_main.shape[2]))
    s_g_arr = np.zeros((u_main.shape[1], u_main.shape[2])) # population standard deviation of data.
    s_arr = np.zeros((u_main.shape[1], u_main.shape[2])) # sample standard deviation of data
    
    for i in range(u_main.shape[1]):
        for j in range(u_main.shape[2]):
            mean_arr[i, j] = np.mean(u_main[:, i, j])
            var_arr[i, j] = np.var(u_main[:, i, j])
            m = np.mean(u_main[:, i, j])
            s_arr[i, j] = math.sqrt(np.sum(list(map(lambda x: (x - m)**2, u_main[:, i, j])))/(len(u_main[:, i, j]) - 1))
    
    mean_tens = torch.from_numpy(mean_arr)
    var_tens = torch.from_numpy(var_arr)
    s_g_arr = torch.from_numpy(var_arr) ** (1/2)
    s_arr = torch.from_numpy(s_arr)
    
    # Confidence region for the mean
    upper_bound = mean_tens + 1.96 * s_arr / math.sqrt(len(u_main))
    lower_bound = mean_tens - 1.96 * s_arr / math.sqrt(len(u_main))
    
    mean_tens = mean_tens.reshape(-1)
    upper_bound = upper_bound.reshape(-1)
    lower_bound = lower_bound.reshape(-1)
    
    return mean_tens, upper_bound, lower_bound


# In[9]:


u_bs_mean_tens, u_bs_upper_bound, u_bs_lower_bound = calculate_statistics(u_main)


# In[10]:


# building 3-dimensional graph
fig = go.Figure(data=[
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=initial_data, name='Initial field',
              legendgroup='i', showlegend=True, color='rgb(139,224,164)',
              opacity=0.5),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=u_bs_mean_tens, name='Solution field - M[BN]',
              legendgroup='s', showlegend=True, color='lightpink',
              opacity=1),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=u_bs_upper_bound, name='Confidence region',
              legendgroup='c', showlegend=True, color='blue',
              opacity=0.20),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=u_bs_lower_bound, name='Confidence region',
              legendgroup='c', color='blue', opacity=0.20),
    
   go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=u_sde_mean_tens.reshape(-1), name='Solution field - M[SDE]',
              legendgroup='sde', showlegend=True, color='rgb(139,200,164)',),

    

])

fig.update_layout(scene_aspectmode='auto')
fig.update_layout(showlegend=True,
                  scene=dict(
                      xaxis_title='x1',
                      yaxis_title='x2',
                      zaxis_title='u',
                      zaxis=dict(nticks=10, dtick=100),
                      aspectratio={"x": 1, "y": 1, "z": 1}
                  ),
                  height=800, width=800
                  )



# Sobol's indices method is based on the expansion of variance $f(\mathbf{x})$ on the contributions of various input parameters. The formula looks like this:
# 
# $$
# f(\mathbf{x}) = f_0 + \sum_{i=1}^n f_i(x_i) + \sum_{1 \leq i < j \leq n} f_{ij}(x_i, x_j) + \cdots + f_{1,2,\ldots,n}(x_1, x_2, \ldots, x_n),
# $$
# where:
# 
# SDE: Represents components associated with first-order variance, i.e. with the influence of individual stochastic parameters.
# 
# BS: Mainly corresponds to second-order variances, reflecting conditional probabilities and dependencies between parameters.
# 
# $$
# M[BN−M[SDE]]
# $$
# BN — result of applying ensemble with Bayesian presence.
# 
# SDE — result of applying a simple ensemble in the form of a stochastic differential equation.

# In[11]:


u_temp = u_main - u_sde_mean_tens

print(u_temp.shape)

result_1 = np.mean(u_temp , axis=0)

print(result_1)

result_1.shape


# In[12]:


# building 3-dimensional graph
fig = go.Figure(data=[
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=initial_data, name='Initial field',
              legendgroup='i', showlegend=True, color='rgb(139,224,164)',
              opacity=0.5),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=result_1.reshape(-1), name='M[BN−M[SDE]]',
              legendgroup='s', showlegend=True, color='lightpink',
              opacity=1),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=u_sde_mean_tens.reshape(-1), name='Solution field - M[SDE]',
              legendgroup='sde', showlegend=True, color='rgb(139,200,164)',),
])

fig.update_layout(scene_aspectmode='auto')
fig.update_layout(showlegend=True,
                  scene=dict(
                      xaxis_title='x1',
                      yaxis_title='x2',
                      zaxis_title='u',
                      zaxis=dict(nticks=10, dtick=100),
                      aspectratio={"x": 1, "y": 1, "z": 1}
                  ),
                  height=800, width=800
                  )



# $$
# M[BS]−M[SDE
# $$

# In[13]:


u_main_bs = np.mean(u_main, axis=0)
result_2 = u_main_bs - u_sde_mean_tens
print(result_2)
result_2.shape


# In[14]:


result_2 = u_main_bs - u_sde_mean_tens
print(result_2)


# In[15]:


rmse = np.sqrt(np.mean((result_1 - result_2) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[16]:


# building 3-dimensional graph
fig = go.Figure(data=[
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=initial_data, name='Initial field',
              legendgroup='i', showlegend=True, color='rgb(139,224,164)',
              opacity=0.5),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=result_1.reshape(-1), name='M[BN−M[SDE]]',
              legendgroup='s', showlegend=True, color='lightpink',
              opacity=1),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=result_2.reshape(-1), name='M[BN]−M[SDE]',
              legendgroup='c', showlegend=True, color='blue',
              opacity=1),
    go.Mesh3d(x=grid[:, 0], y=grid[:, 1], z=u_sde_mean_tens.reshape(-1), name='Solution field - M[SDE]',
              legendgroup='sde', showlegend=True, color='rgb(139,200,164)',),
])


fig.update_layout(scene_aspectmode='auto')
fig.update_layout(showlegend=True,
                  scene=dict(
                      xaxis_title='x1',
                      yaxis_title='x2',
                      zaxis_title='u',
                      zaxis=dict(nticks=10, dtick=100),
                      aspectratio={"x": 1, "y": 1, "z": 1}
                  ),
                  height=800, width=800
                  )

fig.show()


# In[17]:


u_sde_mean_tens


# In[18]:


max_value = np.max(result_1)
variance = np.mean(u_sde)

print(f"max M[BN - M[SDE]]: {max_value}")
print(f"D SDE: {variance}")


# In[24]:


diff = u_main_mean - u_sde_mean_tens
sq_diff = diff ** 2
mean_sq_diff = np.mean(sq_diff, axis=(0)) 

print('RMSE:', np.min(mean_sq_diff))


# In[19]:


u_sde.shape, u_main.shape


# In[21]:


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)


# In[33]:


fig = make_subplots(
    rows=1, 
    cols=2, 
    subplot_titles=('Solution', 'M[SDE]'), 
    horizontal_spacing=0.1 
)

fig.add_trace(
    go.Heatmap(
        z=u_sde_mean_tens,
        x=x,
        y=y,
        colorscale='magma',
        opacity=1,
        showscale=True,
        colorbar=dict(
            x=0.46,       
            len=0.8,       
            thickness=20   
        )
    ),
    row=1, 
    col=1
)

fig.update_layout(
    height=500,
    width=700,  
    showlegend=False,
    margin=dict(
        l=40,   
        r=60,   
        t=60,   
        b=40    
    )
)

fig.show()


# In[34]:


x = np.linspace(-5, 5, 100)  
y = np.linspace(-5, 5, 100)  

fig = make_subplots(
    rows=1, cols=1,
    subplot_titles=('<b>M[BN]−M[SDE]</b>'))
fig = go.Figure(
    data=go.Heatmap(
        z=result_2,
        x=x,
        y=y,
        colorscale='magma',
        opacity=1,
        showscale=True,
        colorbar=dict(x=1.02)  
    )
)
fig.update_layout(
    title={
        'text': "<b>M[BN]−M[SDE]</b>",  
        'y': 0.9,                      
        'x': 0.5,                    
        'xanchor': 'center',           
        'yanchor': 'top',              
    },
    height=600,  
    width=600,  
    showlegend=False,
    xaxis=dict(scaleanchor="y"),  
    yaxis=dict(scaleanchor="x"),  
    
    xaxis2=dict(scaleanchor="y2"),  
    yaxis2=dict(scaleanchor="x2")
)
font=dict(
        family="ATimes New Roman, Times, serif",  
        size=16,  
        color="black"  
    ),
title=dict(
        font=dict(
            family="Times New Roman, Times, serif",
            size=20,
            color="black"
         )
)

fig.show()

