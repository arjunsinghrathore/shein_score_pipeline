from brainscore.tolerance import get_benchmark
import numpy as np
import pandas as pd
import pickle
import argparse

import json

import plotly.graph_objects as go 
import plotly.express as px 

from brainscore import score_model
from model_tools.brain_transformation import LayerMappedModel, TemporalIgnore
from candidate_models.model_commitments import brain_translated_pool, base_model_pool

import chart_studio
import chart_studio.plotly as py

chart_studio.tools.set_credentials_file(username='aarjun1', api_key='DRKG55Wr2s8OQOukeuUY')

#
# identifier = 'alexnet'
# model = brain_translated_pool[identifier]
# score = score_model(model_identifier=identifier, model=model, benchmark_identifier='dicarlo.MajajHong2015.IT-pls')
# print(score)


parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=float, default=0.5)
parser.add_argument('--test_size', type=float, default=0.5)
parser.add_argument('--assembly_name', type=str, default="dicarlo.MajajHong2015.private")

args = parser.parse_args()


result_fname = './results/ceiling_{}_{}.pkl'
benchmark = get_benchmark('tol_ceiling', assembly_name=args.assembly_name, baseline=True, train_size=args.train_size, test_size=args.test_size, baseline_splits=1)
result = benchmark()

print('result result result : ',result)
# print('result center center : ',result.sel(aggregation='center').values)
# print('result center center : ',result.sel(aggregation='center').values.shape)
print('result result result : ',result.attrs['raw'].data)
print('result result result : ',result.attrs['raw'].data.shape)
print('result result result : ',np.median(np.mean(result.attrs['raw'].data, axis = 0)))

int_cons_data = np.array(np.mean(result.attrs['raw'].data, axis = 0)).reshape(-1)

df = pd.DataFrame({'Internal_Consistency':int_cons_data, 'Neuroids':range(199)})

# simple bar graph
fig = px.histogram(           # replace line with bar in line chart code 
    df, 
    # x = 'Neuroids',
    # y = 'Internal_Consistency',
    x = 'Internal_Consistency',
    text_auto=True,
    nbins = 7,
    # color = 'Neuroids' # changes the color as per the changes in the column 
)

# line = go.Scatter(y= [198, 198],
#                   x= [0, 1],
#                   mode= 'lines',
#                   showlegend= False,
#                   hoverinfo='none')

fig.update_layout(
    width=2500,
    height=1000,
    title = 'Internal Consistency over Individual Neuroids(Sheinberg IT Data)',
    # xaxis_title = 'Neuroid No.',
    xaxis_title = 'Number of Neuroids(Count)',
    yaxis_title = 'Internal Consistency',
    xaxis = dict(           # attribures for x axis 
        showline = True,
        showgrid = True,
        linecolor = 'black',
        tickfont = dict(
            family = 'Calibri'
        )
    ),
    yaxis = dict(           # attribures for y axis 
        showline = True,
        showgrid = True,
        linecolor = 'black',
        tickfont = dict(
            family = 'Times New Roman'
        )
    ),
    plot_bgcolor = 'white' 
)
# fig.update_traces(width=10)

fig.show()

fig.write_image("./results/ceiling_fig_shein_histo_prop.png")
# fig.write_image("./results/ceiling_fig_shein.webp")
# fig.write_html("./results/ceiling_fig_majaj.html")

chart_studio = py.plot(fig, filename = 'Sheinberg_histo_prop', auto_open=True)

print('chart_studio', chart_studio)

with open(result_fname.format(args.train_size, args.test_size), 'wb') as f:
    pickle.dump(result, f)
