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
from PIL import Image


import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts

from well_mapping.models import Experiment, SourceWellPlate, DestWellPlate, SourceWellPosition, DestWellPosition, Drug, DestWellProperties

import vast_leica_mapping as vlm

LOCALPATH_CH = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
LOCALPATH_HIVE= r'Y:\raw_data\microscopy\vast'
LOCALPATH_RAID5 =r'D:\vast'

LOCALPATH = LOCALPATH_HIVE
if os.path.exists(LOCALPATH_CH):
    LOCALPATH = LOCALPATH_CH

#___________________________________________________________________________________________
def vast_handler(doc: bokeh.document.Document) -> None:
    print('****************************  vast_handler ****************************')
    #TO BE CHANGED WITH ASYNC?????
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"


    experiments = ['Select experiment']
    for exp in Experiment.objects.all():
        experiments.append(exp.name)

    experiments=sorted(experiments)
    dropdown_exp  = bokeh.models.Select(value='Select experiment', title='Experiment', options=experiments)

    x_96 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    y_96 = ['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
    x_labels_96 = []
    y_labels_96 = []
    for xi in x_96:
        for yi in y_96:
            x_labels_96.append(xi)
            y_labels_96.append(yi)
    source_labels_96 = bokeh.models.ColumnDataSource(data=dict(x=x_labels_96, y=y_labels_96))

    x_48 = ['1', '2', '3', '4', '5', '6', '7', '8']
    y_48 = ['F', 'E', 'D', 'C', 'B', 'A']
    x_labels_48 = []
    y_labels_48 = []
    for xi in x_48:
        for yi in y_48:
            x_labels_48.append(xi)
            y_labels_48.append(yi)
    source_labels_48 = bokeh.models.ColumnDataSource(data=dict(x=x_labels_48, y=y_labels_48))

    x_24 = ['1', '2', '3', '4', '5', '6']
    y_24 = ['D', 'C', 'B', 'A']
    x_labels_24 = []
    y_labels_24 = []
    for xi in x_24:
        for yi in y_24:
            x_labels_24.append(xi)
            y_labels_24.append(yi)
    source_labels_24 = bokeh.models.ColumnDataSource(data=dict(x=x_labels_24, y=y_labels_24))

    cds_labels_dest   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2 = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_dest_present   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_present = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    drug_message    = bokeh.models.Div(visible=False)
    image_message    = bokeh.models.Div(visible=False)

    plot_wellplate_dest   = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x_96), y_range=bokeh.models.FactorRange(*y_96), title='',width=900, height=600, tools="box_select,box_zoom,reset,undo")
    plot_wellplate_dest.xaxis.major_label_text_font_size = "15pt"
    plot_wellplate_dest.yaxis.major_label_text_font_size = "15pt"
    plot_wellplate_dest.grid.visible = False
    plot_wellplate_dest.axis.visible = False

    plot_wellplate_dest_2   = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x_96), y_range=bokeh.models.FactorRange(*y_96), title='',width=900, height=600, tools="box_select,box_zoom,reset,undo")
    plot_wellplate_dest_2.xaxis.major_label_text_font_size = "15pt"
    plot_wellplate_dest_2.yaxis.major_label_text_font_size = "15pt"
    plot_wellplate_dest_2.grid.visible = False
    plot_wellplate_dest_2.axis.visible = False

    plot_wellplate_dest.circle('x', 'y', 
                               size='size', 
                               source=cds_labels_dest, 
                               line_color='blue', fill_color="white",
                               selection_fill_color="orange",    # when selected
                               selection_line_color="firebrick",
                               selection_fill_alpha=0.9,
                               nonselection_fill_alpha=0.0,      # style for non-selected
                               nonselection_fill_color="white",
                               nonselection_line_color="blue",)


    plot_wellplate_dest_2.circle('x', 'y', 
                               size='size', 
                               source=cds_labels_dest_2, 
                               line_color='blue', fill_color="white",
                               selection_fill_color="orange",    # when selected
                               selection_line_color="firebrick",
                               selection_fill_alpha=0.9,
                               nonselection_fill_alpha=0.0,      # style for non-selected
                               nonselection_fill_color="white",
                               nonselection_line_color="blue",)

    plot_wellplate_dest.circle('x', 'y', 
                               size='size', 
                               source=cds_labels_dest_present, 
                               line_color='blue', fill_color="black",
                               fill_alpha=0.5,
                                selection_fill_color="orange",    # when selected
                               selection_line_color="firebrick",
                               selection_fill_alpha=0.9,
                               selection_line_width=2,
                               nonselection_fill_alpha=0.2,      # style for non-selected
                               nonselection_fill_color="black",
                               nonselection_line_color="blue",)


    plot_wellplate_dest_2.circle('x', 'y', 
                               size='size', 
                               source=cds_labels_dest_2_present, 
                               line_color='blue', fill_color="black",
                               fill_alpha=0.5,
                                selection_fill_color="orange",    # when selected
                               selection_line_color="firebrick",
                               selection_fill_alpha=0.9,
                               selection_line_width=2,
                               nonselection_fill_alpha=0.2,      # style for non-selected
                               nonselection_fill_color="black",
                               nonselection_line_color="blue",)



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



    #___________________________________________________________________________________________
    def get_well_mapping(indices):
        print('------------------->>>>>>>>> get_well_mapping')

        n_well = len(cds_labels_dest.data['x'])

        positions = []
        print('get_well_mapping indices=',indices)
        print('n_well=',n_well)
        if n_well == 96:
            i=0
            for xi in x_96:
                for yi in y_96:
                    if i in indices:
                        positions.append(('{}'.format(xi),'{}'.format(yi)))
                    i+=1
        elif n_well == 48:
            i=0
            for xi in x_48:
                for yi in y_48:
                    if i in indices:
                        positions.append(('{}'.format(xi),'{}'.format(yi)))
                    i+=1
        elif n_well == 24:
            i=0
            for xi in x_24:
                for yi in y_24:
                    if i in indices:
                        positions.append(('{}'.format(xi),'{}'.format(yi)))
                    i+=1
       
        print('positions=', positions)
        return positions
    

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
    def dest_plate_visu(attr, old, new):
        if len(cds_labels_dest.selected.indices) == 0: 
            return
        cds_labels_dest_2_present.selected.indices = []
        cds_labels_dest_2.selected.indices = []
        position = get_well_mapping(cds_labels_dest.selected.indices)



        path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 1', 'Well_{}{}'.format(position[0][1], position[0][0]))
        if int(position[0][0]) < 10:
            path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 1', 'Well_{}0{}'.format(position[0][1], position[0][0]))  
        files = glob.glob(os.path.join(path_leica, '*_norm.tiff'))

        for f in files:
            if 'BF' in f:
                file_BF = f
            else:
                file_YFP = f

        if len(files) == 0:
            print('No files found in path:', path_leica)
            source_img_bf.data  = {'img':[]}
            source_img_yfp.data = {'img':[]}
            source_img_vast.data = {'img':[]}
            drug_message.text = ""
            drug_message.visible = False
            image_message.text = "<b style='color:red; font-size:18px;'>No images found for selected well {}</b>".format(position[0][1] + position[0][0])
            image_message.visible = True
            return

        image_message.text = ""
        image_message.visible = False

        image_bf  = imread(file_BF)
        source_img_bf.data  = {'img':[np.flip(image_bf,0)]}

        image_yfp = imread(file_YFP)
        source_img_yfp.data = {'img':[np.flip(image_yfp,0)]}

        path_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 1', 'Well_{}{}'.format(position[0][1], position[0][0]))
        if int(position[0][0]) < 10:
            path_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 1', 'Well_{}0{}'.format(position[0][1], position[0][0]))  
        files = glob.glob(os.path.join(path_vast, '*.tiff'))

        img_list= []
        for f in files: 
            image = Image.open(f).convert('RGBA')
            img_list.append(image)

        merged_array = np.concatenate(img_list, axis=0)  # horizontal
        height, width, channels = merged_array.shape
        rgba_image = np.empty((height, width), dtype=np.uint32)
        view = rgba_image.view(dtype=np.uint8).reshape((height, width, 4))
        view[:, :, :] = merged_array
        source_img_vast.data = {'img': [rgba_image]}


        well_plate = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=1).first()
        dest = DestWellPosition.objects.filter(well_plate=well_plate, position_col=position[0][0], position_row=position[0][1])
        if dest[0].source_well is None:
            print('No source well found for dest position:', position[0])
            drug_message.text = "<b style='color:red; font-size:18px;'>No source well found for selected well {}</b>".format(position[0][1] + position[0][0])
            drug_message.visible = True
            return
        drugs = dest[0].source_well.drugs.all()
        items_html = "".join(
            f"<li style='color:navy; font-size:14px; "
            f"margin-bottom:4px;'>{drug}</li>"
            for drug in drugs)

        drug_message.text = f"""
        <b style='color:green; font-size:18px;'>
            Drug(s) in selected well {position[0][1]}{position[0][0]}:
        </b>
        <ul style='margin-top:0;'>
            {items_html} <br> <b style='color:black; font-size:14px;'> comments={dest[0].source_well.comments}, valid well={dest[0].source_well.valid}</b>
        </ul>
        """
        drug_message.visible = True


        # Set the dropdowns if properties exist
        try:
            dest_well_properties = DestWellProperties.objects.get(dest_well=dest[0])
            print('Found properties for dest well:', dest, ' properties:', dest_well_properties)
            if dest_well_properties.n_good_somites is not None:
                dropdown_good_somites.value = str(dest_well_properties.n_good_somites)
            else:
                dropdown_good_somites.value = 'Select a value'
            if dest_well_properties.n_bad_somites is not None:
                dropdown_bad_somites.value  = str(dest_well_properties.n_bad_somites)
            else:
                dropdown_bad_somites.value = 'Select a value'
            dropdown_good_somites_err.value = str(dest_well_properties.n_good_somites_err)
            dropdown_bad_somites_err.value  = str(dest_well_properties.n_bad_somites_err)
            if dest_well_properties.valid:
                dropdown_good_image.value = 'Yes'
            else:
                dropdown_good_image.value = 'No'
            if dest_well_properties.comments is not None:
                images_comments.value = dest_well_properties.comments
            else:
                images_comments.value = ''
        except DestWellProperties.DoesNotExist:
            print('No properties found for dest well:', dest)
            dropdown_good_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_good_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            images_comments.value = ''



    cds_labels_dest.selected.on_change('indices', lambda attr, old, new: dest_plate_visu(attr, old, new))

    #___________________________________________________________________________________________
    def dest_plate_2_visu(attr, old, new):
        if len(cds_labels_dest_2.selected.indices) == 0:
            return
        cds_labels_dest_present.selected.indices = []
        cds_labels_dest.selected.indices = []

        position = get_well_mapping(cds_labels_dest_2.selected.indices) 

        path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 2', 'Well_{}{}'.format(position[0][1], position[0][0]))
        if int(position[0][0]) < 10:
            path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 2', 'Well_{}0{}'.format(position[0][1], position[0][0]))  
        files = glob.glob(os.path.join(path_leica, '*_norm.tiff'))

        for f in files:
            if 'BF' in f:
                file_BF = f
            else:
                file_YFP = f

        if len(files) == 0:
            print('No files found in path:', path_leica)
            source_img_bf.data  = {'img':[]}
            source_img_yfp.data = {'img':[]}
            source_img_vast.data = {'img':[]}
            drug_message.text = ""
            drug_message.visible = False
            image_message.text = "<b style='color:red; font-size:18px;'>No images found for selected well {}</b>".format(position[0][1] + position[0][0])
            image_message.visible = True
            return


        image_bf  = imread(file_BF)
        source_img_bf.data  = {'img':[np.flip(image_bf,0)]}

        image_yfp = imread(file_YFP)
        source_img_yfp.data = {'img':[np.flip(image_yfp,0)]}

        path_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 2', 'Well_{}{}'.format(position[0][1], position[0][0]))
        if int(position[0][0]) < 10:
            path_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 2', 'Well_{}0{}'.format(position[0][1], position[0][0]))  
        files = glob.glob(os.path.join(path_vast, '*.tiff'))

        img_list= []
        for f in files: 
            image = Image.open(f).convert('RGBA')
            img_list.append(image)

        merged_array = np.concatenate(img_list, axis=0)  # horizontal
        height, width, channels = merged_array.shape
        rgba_image = np.empty((height, width), dtype=np.uint32)
        view = rgba_image.view(dtype=np.uint8).reshape((height, width, 4))
        view[:, :, :] = merged_array
        source_img_vast.data = {'img': [rgba_image]}

        well_plate = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=2).first()
        dest = DestWellPosition.objects.filter(well_plate=well_plate, position_col=position[0][0], position_row=position[0][1])
        if dest[0].source_well is None:
            print('No source well found for dest position:', position[0])
            drug_message.text = "<b style='color:red; font-size:18px;'>No source well found for selected well {}</b>".format(position[0][1] + position[0][0])
            drug_message.visible = True
            return
        drugs = dest[0].source_well.drugs.all()
        items_html = "".join(
            f"<li style='color:navy; font-size:14px; "
            f"margin-bottom:4px;'>{drug}</li>"
            for drug in drugs)

        drug_message.text = f"""
        <b style='color:green; font-size:18px;'>
            Drug(s) in selected well {position[0][1]}{position[0][0]}:
        </b>
        <ul style='margin-top:0;'>
            {items_html} <br> <b style='color:black; font-size:14px;'> comments={dest[0].source_well.comments}, valid well={dest[0].source_well.valid}</b>
        </ul>
        """
        drug_message.visible = True

        # Set the dropdowns if properties exist
        try:
            dest_well_properties = DestWellProperties.objects.get(dest_well=dest[0])
            print('Found properties for dest well:', dest, ' properties:', dest_well_properties)
            if dest_well_properties.n_good_somites is not None:
                dropdown_good_somites.value = str(dest_well_properties.n_good_somites)
            else:
                dropdown_good_somites.value = 'Select a value'
            if dest_well_properties.n_bad_somites is not None:
                dropdown_bad_somites.value  = str(dest_well_properties.n_bad_somites)
            else:
                dropdown_bad_somites.value = 'Select a value'
            dropdown_good_somites_err.value = str(dest_well_properties.n_good_somites_err)
            dropdown_bad_somites_err.value  = str(dest_well_properties.n_bad_somites_err)
            if dest_well_properties.valid:
                dropdown_good_image.value = 'Yes'
            else:
                dropdown_good_image.value = 'No'
            if dest_well_properties.comments is not None:
                images_comments.value = dest_well_properties.comments
            else:
                images_comments.value = ''
        except DestWellProperties.DoesNotExist:
            print('No properties found for dest well:', dest)
            dropdown_good_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_good_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            images_comments.value = ''

    cds_labels_dest_2.selected.on_change('indices', lambda attr, old, new: dest_plate_2_visu(attr, old, new))



    #___________________________________________________________________________________________
    def load_experiment(attr, old, new):

        experiment  = Experiment.objects.get(name=new)
        dest_well_plates   = DestWellPlate.objects.filter(experiment=experiment)
        print('dest_well_plates=', dest_well_plates)

        if len(dest_well_plates)==0:
            print('No destination well plates found for experiment:', new)
            return

        n_plates = len(dest_well_plates)
        print('n_plates=', n_plates)

        if n_plates==1 or n_plates==2:
            dest_well_plate = dest_well_plates[0]
            if dest_well_plate.plate_type == '96':
                plot_wellplate_dest.x_range.factors = x_96
                plot_wellplate_dest.y_range.factors = y_96
                plot_wellplate_dest.title.text = "96 well plate"
                cds_labels_dest.data = dict(source_labels_96.data, size=[50]*len(source_labels_96.data['x']))
                plot_wellplate_dest.axis.visible = True

            elif dest_well_plate.plate_type == '48':
                plot_wellplate_dest.x_range.factors = x_48
                plot_wellplate_dest.y_range.factors = y_48
                plot_wellplate_dest.title.text = "48 well plate"
                cds_labels_dest.data = dict(source_labels_48.data, size=[65]*len(source_labels_48.data['x']))
                plot_wellplate_dest.axis.visible = True

            elif dest_well_plate.plate_type == '24':
                plot_wellplate_dest.x_range.factors = x_24
                plot_wellplate_dest.y_range.factors = y_24
                plot_wellplate_dest.title.text = "24 well plate"
                cds_labels_dest.data = dict(source_labels_24.data, size=[80]*len(source_labels_24.data['x']))
                plot_wellplate_dest.axis.visible = True

        if n_plates==2:
            dest_well_plate_2 = dest_well_plates[1]
            if dest_well_plate_2.plate_type == '96':
                plot_wellplate_dest_2.x_range.factors = x_96
                plot_wellplate_dest_2.y_range.factors = y_96
                plot_wellplate_dest_2.title.text = "96 well plate"
                cds_labels_dest_2.data = dict(source_labels_96.data, size=[50]*len(source_labels_96.data['x']))
                plot_wellplate_dest_2.axis.visible = True

            elif dest_well_plate_2.plate_type == '48':
                plot_wellplate_dest_2.x_range.factors = x_48
                plot_wellplate_dest_2.y_range.factors = y_48
                plot_wellplate_dest_2.title.text = "48 well plate"
                cds_labels_dest_2.data = dict(source_labels_48.data, size=[65]*len(source_labels_48.data['x']))
                plot_wellplate_dest_2.axis.visible = True

            elif dest_well_plate_2.plate_type == '24':
                plot_wellplate_dest_2.x_range.factors = x_24
                plot_wellplate_dest_2.y_range.factors = y_24
                plot_wellplate_dest_2.title.text = "24 well plate"
                cds_labels_dest_2.data = dict(source_labels_24.data, size=[80]*len(source_labels_24.data['x']))
                plot_wellplate_dest_2.axis.visible = True


        path_plate_1_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 1', 'Well_*')
        path_plate_2_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 2', 'Well_*')
        path_plate_1_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 1', 'Well_*')
        path_plate_2_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 2', 'Well_*')        

        wells_plate_1_leica = [os.path.split(f)[-1] for f in glob.glob(path_plate_1_leica)]
        wells_plate_2_leica = [os.path.split(f)[-1] for f in glob.glob(path_plate_2_leica)]

        wells_plate_1_vast = [os.path.split(f)[-1] for f in glob.glob(path_plate_1_vast)]
        wells_plate_2_vast = [os.path.split(f)[-1] for f in glob.glob(path_plate_2_vast)]

        if len(wells_plate_1_leica)==0 and len(wells_plate_2_leica)==0:
            print('No wells found for experiment:', new)
            cds_labels_dest.data = dict(x=[], y=[], size=[])
            cds_labels_dest_2.data = dict(x=[], y=[], size=[])
            cds_labels_dest_present.data = dict(x=[], y=[], size=[])
            cds_labels_dest_2_present.data = dict(x=[], y=[], size=[])
            source_img_bf.data  = {'img':[]}
            source_img_yfp.data = {'img':[]}
            source_img_vast.data = {'img':[]}
            drug_message.text = ""
            drug_message.visible = False
            image_message.text = "<b style='color:red; font-size:18px;'>No Leica images found for experiment {} need to run mapping</b>".format(new)
            image_message.visible = True
            return

        x_dest_1=[]
        y_dest_1=[]
        size_dest_1=[]
        for w in wells_plate_1_leica:
            if w not in wells_plate_1_vast:
                print('well not in both leica and vast...')
            row=w.split("_")[-1][0:1]
            col=w.split("_")[-1][1:3]
            x_dest_1.append(str(int(col)))
            y_dest_1.append(row)
            size_dest_1.append(cds_labels_dest.data['size'][0])
            cds_labels_dest_present.data = {'x':x_dest_1, 'y':y_dest_1, 'size':size_dest_1}


        x_dest_2=[]
        y_dest_2=[]
        size_dest_2=[]
        for w in wells_plate_2_leica:
            if w not in wells_plate_2_vast:
                print('well not in both leica and vast...')
            row=w.split("_")[-1][0:1]
            col=w.split("_")[-1][1:3]
            x_dest_2.append(str(int(col)))
            y_dest_2.append(row)
            size_dest_2.append(cds_labels_dest_2.data['size'][0])
            cds_labels_dest_2_present.data = {'x':x_dest_2, 'y':y_dest_2, 'size':size_dest_2}

    dropdown_exp.on_change("value", load_experiment)


    #___________________________________________________________________________________________
    def mapping_callback():
        print('------------------->>>>>>>>> mapping_callback')
        if dropdown_exp.value == 'Select experiment':
            print('Please select an experiment first')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select an experiment first</b>"
            image_message.visible = True
            return
        print('Mapping for experiment:', dropdown_exp.value, ' in path:', LOCALPATH)
        vlm.map_well_to_vast(LOCALPATH, dropdown_exp.value)
        load_experiment(None, None, dropdown_exp.value)
        well_mapping_button.label = "Well mapping"
        well_mapping_button.button_type = "success"
    well_mapping_button = bokeh.models.Button(label="Well mapping", button_type="success", width=150)


    #___________________________________________________________________________________________
    def mapping_callback_short():
        well_mapping_button.label = "Processing"
        well_mapping_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(mapping_callback)
    well_mapping_button.on_click(mapping_callback_short)

    #___________________________________________________________________________________________

    color_low=0
    color_high=65535

   #___________________________________________________________________________________________
    def update_contrast(attr, old, new):
        low, high = new 
        color_mapper.low = int(low*655.35)
        color_mapper.high = int(high*655.35)

    contrast_slider = bokeh.models.RangeSlider(start=0, end=100, value=(0, 100), step=1, title="Contrast", width=200)
    contrast_slider.on_change('value', update_contrast)


    dropdown_good_somites      = bokeh.models.Select(value='Select a value', title='# good somites', options=['Select a value', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'])
    dropdown_bad_somites       = bokeh.models.Select(value='Select a value', title='# bad somites',  options=['Select a value','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'])
    dropdown_good_somites_err  = bokeh.models.Select(value='0', title='# good somites error', options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    dropdown_bad_somites_err   = bokeh.models.Select(value='0', title='# bad somites error',  options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    dropdown_good_image        = bokeh.models.Select(value='Yes', title='Good image', options=['Yes', 'No'])
    images_comments            = bokeh.models.widgets.TextAreaInput(title="Comments if any:", value='', rows=7, width=200, css_classes=["font-size:18px"])

    #___________________________________________________________________________________________
    def saveimages_callback():
        print('------------------->>>>>>>>> saveimages_callback')
        if dropdown_exp.value == 'Select experiment':
            print('Please select an experiment first')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select an experiment first</b>"
            image_message.visible = True
            saveimages_button.label = "Save"
            saveimages_button.button_type = "success"
            return
        print('Saving properties for experiment:', dropdown_exp.value)
        if len(cds_labels_dest.selected.indices) == 0 and len(cds_labels_dest_2.selected.indices) == 0:
            print('Please select a well first')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select a well first</b>"
            image_message.visible = True
            saveimages_button.label = "Save"
            saveimages_button.button_type = "success"
            return
        
        if len(cds_labels_dest.selected.indices) > 0 and len(cds_labels_dest_2.selected.indices) > 0:
            print('Please select a well in only one plate')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select a well in only one plate</b>"
            image_message.visible = True
            saveimages_button.label = "Save"
            saveimages_button.button_type = "success"
            return


        if dropdown_good_somites.value == 'Select a value' and dropdown_bad_somites.value == 'Select a value':
            print('Please select at least one of # good somites or # bad somites')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select at least one of # good somites or # bad somites</b>"
            image_message.visible = True
            saveimages_button.label = "Save"
            saveimages_button.button_type = "success"
            return

        dest=None
        if len(cds_labels_dest.selected.indices) > 0:
            position = get_well_mapping(cds_labels_dest.selected.indices)
            well_plate = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=1).first()

        elif len(cds_labels_dest_2.selected.indices) > 0:
            position = get_well_mapping(cds_labels_dest_2.selected.indices)
            well_plate = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=2).first()
            dest = DestWellPosition.objects.filter(well_plate=well_plate, position_col=position[0][0], position_row=position[0][1]).first()

        dest = DestWellPosition.objects.filter(well_plate=well_plate, position_col=position[0][0], position_row=position[0][1]).first()
        #dest_well_properties = DestWellProperties(dest_well=dest)
        dest_well_properties, created = DestWellProperties.objects.get_or_create(dest_well=dest)
        dest_well_properties.num_good_somites = int(dropdown_good_somites.value) if dropdown_good_somites.value != 'Select a value' else None
        dest_well_properties.num_bad_somites  = int(dropdown_bad_somites.value)  if dropdown_bad_somites.value != 'Select a value' else None
        dest_well_properties.num_good_somites_err = int(dropdown_good_somites_err.value)
        dest_well_properties.num_bad_somites_err  = int(dropdown_bad_somites_err.value)
        dest_well_properties.good_image = True if dropdown_good_image.value == 'Yes' else False
        dest_well_properties.comments = images_comments.value
        dest_well_properties.save()
        print('Saved properties for dest well:', dest, ' properties:', dest_well_properties)

        saveimages_button.label = "Save"
        saveimages_button.button_type = "success"
    saveimages_button = bokeh.models.Button(label="Save", button_type="success", width=150)


    #___________________________________________________________________________________________
    def saveimages_callback_short():
        saveimages_button.label = "Processing"
        saveimages_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(saveimages_callback)
    saveimages_button.on_click(saveimages_callback_short)







    plot_wellplate_dest.add_tools(tap_tool)
    plot_wellplate_dest_2.add_tools(tap_tool)

    color_mapper = bokeh.models.LinearColorMapper(palette="Greys256", low=color_low, high=color_high)

    data_img_bf   = {'img':[]}
    source_img_bf = bokeh.models.ColumnDataSource(data=data_img_bf)
    plot_img_bf   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    plot_img_bf.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_bf, color_mapper=color_mapper)

    data_img_yfp   = {'img':[]}
    source_img_yfp = bokeh.models.ColumnDataSource(data=data_img_yfp)
    plot_img_yfp   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=550, height=550)
    plot_img_yfp.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_yfp, color_mapper=color_mapper)

    data_img_vast   = {'img':[]}
    source_img_vast = bokeh.models.ColumnDataSource(data=data_img_vast)
    x_range_2 = bokeh.models.Range1d(start=0, end=1024)
    y_range_2 = bokeh.models.Range1d(start=0, end=200*4)
    plot_img_vast   = bokeh.plotting.figure(x_range=x_range_2, y_range=y_range_2, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=1110, height=217*4)
    #plot_img_vast   = bokeh.plotting.figure(tools="box_select,wheel_zoom,box_zoom,reset,undo",width=1024, height=200)
    plot_img_vast.image_rgba(image='img', x=0, y=0, dw=1024, dh=200*4, source=source_img_vast)




    indent = bokeh.models.Spacer(width=30)

    norm_layout = bokeh.layouts.column(bokeh.layouts.row(indent,bokeh.layouts.column(dropdown_exp, well_mapping_button), bokeh.models.Spacer(width=20),    bokeh.layouts.column(image_message,drug_message)),
                                       bokeh.layouts.Spacer(width=50),
                                       bokeh.layouts.row(indent,  bokeh.layouts.column(plot_wellplate_dest, plot_wellplate_dest_2),
                                                         bokeh.layouts.column(bokeh.layouts.row(bokeh.layouts.Spacer(width=10), contrast_slider, dropdown_good_somites, dropdown_good_somites_err, dropdown_bad_somites, dropdown_bad_somites_err, dropdown_good_image, saveimages_button,images_comments),
                                                                              bokeh.layouts.row(plot_img_bf, bokeh.layouts.Spacer(width=10),plot_img_yfp),
                                                                              bokeh.layouts.row(plot_img_vast))))

    plot_img_bf.axis.visible   = False
    plot_img_bf.grid.visible   = False
    plot_img_yfp.axis.visible  = False
    plot_img_yfp.grid.visible  = False
    plot_img_vast.axis.visible = False
    plot_img_vast.grid.visible = False


    doc.add_root(norm_layout)




#___________________________________________________________________________________________
#@login_required
def index(request: HttpRequest) -> HttpResponse:
    context={}
    return render(request, 'well_explorer/index.html', context=context)


#___________________________________________________________________________________________
#@login_required
def bokeh_dashboard(request: HttpRequest) -> HttpResponse:

    script = bokeh.embed.server_document(request.build_absolute_uri())
    print("request.build_absolute_uri() ",request.build_absolute_uri())
    context = {'script': script}

    return render(request, 'well_explorer/bokeh_dashboard.html', context=context)
