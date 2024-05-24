from django.shortcuts import render

from django.db import reset_queries
from django.db import connection
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.decorators import login_required, permission_required

from well_mapping.models import Experiment
import os, sys, json, glob, gc
import time

import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import io
import urllib, base64

import math

from typing import Any

from pathlib import Path
from io import BytesIO
import base64
from PIL import Image

import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts


#___________________________________________________________________________________________
def vast_handler(doc: bokeh.document.Document) -> None:
    print('****************************  vast_handler ****************************')
    #TO BE CHANGED WITH ASYNC?????
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    im_size = 2048
    x_range = bokeh.models.Range1d(start=0, end=im_size)
    y_range = bokeh.models.Range1d(start=0, end=im_size)

    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    y = ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']


    x_labels = []
    y_labels = []
    for xi in x:
        for yi in y:
            x_labels.append(xi)
            y_labels.append(yi)

    source_labels = bokeh.models.ColumnDataSource(data=dict(x=x_labels, y=y_labels))

    plot_96_wellplate   = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x), y_range=bokeh.models.FactorRange(*y), width=750, height=500, title="96 well plate")
    plot_96_wellplate.xaxis.major_label_text_font_size = "15pt"  # Increase x-axis tick label size
    plot_96_wellplate.yaxis.major_label_text_font_size = "15pt"  # Increase y-axis tick label size


    #___________________________________________________________________________________________
    def select_tap_callback():
        return """
        const indices = cb_data.source.selected.indices;

        if (indices.length > 0) {
            const index = indices[0];
            other_source.data = {'index': [index]};
            other_source.change.emit();  
        }
        """

    index_source = bokeh.models.ColumnDataSource(data=dict(index=[]))  # Data source for the image
    tap_tool = bokeh.models.TapTool(callback=bokeh.models.CustomJS(args=dict(other_source=index_source),code=select_tap_callback()))


    #___________________________________________________________________________________________
    def select_image_96_well_plate(attr, old, new):
        current_path="/Users/helsens/Software/github/EPFL-TOP/VAST-DS/inputData"

        print('attr=',attr,'  old=',old,'  new=',new)
        print('labels ',x_labels[new['index'][0]], y_labels[new['index'][0]])

        xlab=x_labels[new['index'][0]]
        if len(xlab)==1:
            xlab='0'+xlab
        bf_image  = glob.glob(os.path.join(current_path, 'Leica_Well_{}{}*BF*.tiff'.format(y_labels[new['index'][0]], xlab)))
        yfp_image = glob.glob(os.path.join(current_path, 'Leica_Well_{}{}*YFP*.tiff'.format(y_labels[new['index'][0]], xlab)))
        
        print('bf_image = ',bf_image)
        print('yfp_image= ',yfp_image)

        if len(bf_image)==1:  
            image_bf  = imread(bf_image[0])
            max_value = np.max(image_bf)
            min_value = np.min(image_bf)
            intensity_normalized_bf = (image_bf - min_value)/(max_value-min_value)*255
            intensity_normalized_bf = intensity_normalized_bf.astype(np.uint8)
            source_img_bf.data  = {'img':[np.flip(intensity_normalized_bf,0)]}
        else:
            source_img_bf.data  = {'img':[]}

        if len(yfp_image)==1: 
            image_yfp = imread(yfp_image[0])
            max_value = np.max(image_yfp)
            min_value = np.min(image_yfp)
            intensity_normalized_yfp = (image_yfp - min_value)/(max_value-min_value)*255
            intensity_normalized_yfp = intensity_normalized_yfp.astype(np.uint8)
            source_img_yfp.data = {'img':[np.flip(intensity_normalized_yfp,0)]}
        else:
            source_img_yfp.data  = {'img':[]}
    index_source.on_change('data', select_image_96_well_plate)
    #___________________________________________________________________________________________


    plot_96_wellplate.add_tools(tap_tool)

    color_mapper = bokeh.models.LinearColorMapper(palette="Greys256", low=0, high=255)

    data_img_bf   = {'img':[]}
    source_img_bf = bokeh.models.ColumnDataSource(data=data_img_bf)
    plot_img_bf   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    plot_img_bf.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_bf, color_mapper=color_mapper)

    data_img_yfp   = {'img':[]}
    source_img_yfp = bokeh.models.ColumnDataSource(data=data_img_yfp)
    plot_img_yfp   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    plot_img_yfp.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_yfp)


    plot_96_wellplate.circle('x', 'y', source=source_labels, size=40, line_color='blue')

    norm_layout = bokeh.layouts.column(bokeh.layouts.row(plot_96_wellplate),
                                       bokeh.layouts.row(plot_img_bf, plot_img_yfp))

    doc.add_root(norm_layout)


#___________________________________________________________________________________________
#@login_required
def index(request: HttpRequest) -> HttpResponse:
    context={}

    return render(request, 'well_mapping/index.html', context=context)


#___________________________________________________________________________________________
#@login_required
def bokeh_dashboard(request: HttpRequest) -> HttpResponse:

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())
    context = {'script': script}

    return render(request, 'well_mapping/bokeh_dashboard.html', context=context)
