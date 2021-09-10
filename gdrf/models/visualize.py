import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image

from typing import Union

def png_bytes_to_numpy(png):
    """Convert png bytes to numpy array

    Source: https://gist.github.com/eric-czech/fea266e546efac0e704d99837a52b35f
    Credit: Eric Czech

    """
    return np.array(Image.open(BytesIO(png)))

def categorical_stackplot(data: pd.DataFrame, title: str = None, label: str = 'topic'):
    fig = go.Figure()
    for i, column in enumerate(data.columns):
        color = px.colors.qualitative.Light24[i % 24]
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            name=column,
            mode='lines',
            line={'width':0.5, 'color': color},
            stackgroup='one',
            groupnorm='fraction'
        ))
    fig.update_layout(title=title)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Probability', range=(0, 1))
    return png_bytes_to_numpy(fig.to_image(format="png"))

def matrix_plot(data: np.ndarray, title: str=None, xlabels: list[str]=None, ylabels: list[str]=None, value: str=None):
    fig = px.imshow(
        img=data,
        range_color=[-5, 0],
        aspect='auto',
        x=xlabels,
        y=ylabels,
        title=title)
    fig.update_layout(coloraxis_colorbar={
        'title': value,
        'tickvals': [-5, -4, -3, -2, -1, 0],
        'ticktext': ['1e-5', '1e-4', '1e-3', '0.01', '0.1', '1']
    })
    return png_bytes_to_numpy(fig.to_image(format='png'))

