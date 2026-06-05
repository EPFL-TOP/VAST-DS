from django.shortcuts import render

from django.db import reset_queries
from django.db import connection
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.decorators import login_required, permission_required

from well_mapping.models import Experiment
import os, sys, json, glob, gc
import time
import shutil
import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from PIL import Image
import random

import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts


from well_mapping.models import (
    Experiment, SourceWellPlate, DestWellPlate, SourceWellPosition,
    DestWellPosition, Drug, DestWellProperties, DestWellPropertiesPredicted,
    latest_prediction,
)

# Default model_name used when reading/writing predictions from the original
# ResNet18 stack. Will be joined by 'sam_v1', 'multitask_v1', etc., as new
# models land — see DestWellPropertiesPredicted in well_mapping/models.py.
RESNET_MODEL_NAME = 'resnet_v1'

from somiteCounting.training import SomiteCounter_freeze, FishQualityClassifier
from somiteCounting.training_orientation import OrientationClassifier, preprocess_image

import somiteCounting.orientfish as of

def load_and_prepare_image(img_path, resize=(224,224)):
    img_raw = np.array(Image.open(img_path)).astype(np.float32)
    img_raw /= img_raw.max()  # scale to 0-1

    img_pil = Image.fromarray((img_raw*65535).astype(np.uint16))
    img_pil = img_pil.resize(resize, resample=Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32)/65535.0).unsqueeze(0).unsqueeze(0)
    return img_raw, img_tensor

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = SomiteCounter().to(device)
model = SomiteCounter_freeze().to(device)
checkpoint_path=r"C:\Users\helsens\software\VAST-DS\somiteCounting\checkpoints\somite_counting_best.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
except Exception as e:
    print('exception laoding model ',e)



model_fish = FishQualityClassifier().to(device)
checkpoint_path_fish=r"C:\Users\helsens\software\VAST-DS\somiteCounting\checkpoints\fish_quality_best.pth"
try:
    checkpoint_fish = torch.load(checkpoint_path_fish, map_location=device)
    model_fish.load_state_dict(checkpoint_fish["model_state_dict"])
    model_fish.eval()
except Exception as e:
    print('exception laoding model ',e)

model_orientation = OrientationClassifier().to(device)
checkpoint_path_ori=r"C:\Users\helsens\software\VAST-DS\somiteCounting\checkpoints\orientation_best.pth"
try:
    checkpoint_orientation = torch.load(checkpoint_path_ori, map_location=device)
    model_orientation.load_state_dict(checkpoint_orientation["model_state_dict"])
    model_orientation.eval()
except Exception as e: 
    print('exception laoding model ',e)



import vast_leica_mapping as vlm

LOCALPATH_CH = "/Users/helsens/Software/github/EPFL-TOP/VAST-DS/data"
LOCALPATH_HIVE= r'Y:\raw_data\microscopy\vast\VAST-DS'
LOCALPATH_RAID5 =r'D:\vast\VAST-DS'
LOCALPATH_TRAINING=r'D:\vast\VAST-DS\training_data'

LOCALPATH = LOCALPATH_HIVE
if os.path.exists(LOCALPATH_CH):
    LOCALPATH = LOCALPATH_CH


from pathlib import Path

def to_media_url(path):
    p = Path(path)
    # remove D:\vast
    rel = p.relative_to(r"D:\vast")
    return f"/media/{rel.as_posix()}"


def clamp_somite_count(x):
    """Round a raw float somite prediction to a non-negative integer.

    Used at inference time so what we store in DestWellPropertiesPredicted
    matches the discrete, non-negative nature of somite counts (the network
    outputs unbounded floats). Returns 0 on any error so saving never fails."""
    try:
        return max(0, int(round(float(x))))
    except (TypeError, ValueError):
        return 0

#___________________________________________________________________________________________
def vast_handler(doc: bokeh.document.Document) -> None:
    print('****************************  vast_handler ****************************')
    #TO BE CHANGED WITH ASYNC?????
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    nzoom_wells = 1
    ncrop = 0

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

    cds_labels_dest_filled    = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_filled   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_dest_filled_bad    = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_filled_bad   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    drug_message    = bokeh.models.Div(visible=False)
    image_message    = bokeh.models.Div(visible=False)
    prediction_message    = bokeh.models.Div(visible=False)

    plot_wellplate_dest   = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x_96), y_range=bokeh.models.FactorRange(*y_96), title='',
                                                  width=560, height=380, tools="box_select,box_zoom,reset,undo")
    plot_wellplate_dest.xaxis.major_label_text_font_size = "15pt"
    plot_wellplate_dest.yaxis.major_label_text_font_size = "15pt"
    plot_wellplate_dest.grid.visible = False
    plot_wellplate_dest.axis.visible = False

    plot_wellplate_dest_2   = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x_96), y_range=bokeh.models.FactorRange(*y_96), title='',
                                                    width=560, height=380, tools="box_select,box_zoom,reset,undo")
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
                               fill_alpha=0.3,
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
                               fill_alpha=0.3,
                                selection_fill_color="orange",    # when selected
                               selection_line_color="firebrick",
                               selection_fill_alpha=0.9,
                               selection_line_width=2,
                               nonselection_fill_alpha=0.2,      # style for non-selected
                               nonselection_fill_color="black",
                               nonselection_line_color="blue",)

    plot_wellplate_dest.circle('x', 'y', 
                               size='size', 
                               source=cds_labels_dest_filled, 
                               line_color='green', fill_color="white",
                               fill_alpha=0.0,
                               line_width=4,
                               nonselection_line_width=4,
                               selection_line_width=4)

    plot_wellplate_dest_2.circle('x', 'y', 
                                 size='size', 
                                 source=cds_labels_dest_2_filled, 
                                 line_color='green', fill_color="white",
                                 fill_alpha=0.0,
                                 line_width=4,
                                 nonselection_line_width=4,
                                 selection_line_width=4)
    

    plot_wellplate_dest.circle('x', 'y', 
                               size='size', 
                               source=cds_labels_dest_filled_bad, 
                               line_color='black', fill_color="white",
                               fill_alpha=0.0,
                               line_width=4,
                               nonselection_line_width=4,
                               selection_line_width=4)

    plot_wellplate_dest_2.circle('x', 'y', 
                                 size='size', 
                                 source=cds_labels_dest_2_filled_bad, 
                                 line_color='black', fill_color="white",
                                 fill_alpha=0.0,
                                 line_width=4,
                                 nonselection_line_width=4,
                                 selection_line_width=4)

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



    zoom_in_wells = bokeh.models.Button(label="Zoom in wells")
    zoom_out_wells = bokeh.models.Button(label="Zoom out wells")

    zoom_in_fish = bokeh.models.Button(label="Zoom in fish")
    zoom_out_fish = bokeh.models.Button(label="Zoom out fish")

    #___________________________________________________________________________________________
    def zoom_size(factor, cds):

        if len(cds.data['size'])>0:
            new_size = int(cds.data['size'][0] * factor)
            data = dict(cds.data)
            data["size"] = [new_size] * len(data["x"])
            cds.data = data


    #___________________________________________________________________________________________
    def make_zoom_cb_wells(factor):
        def zoom_cb():
            nonlocal nzoom_wells
            nzoom_wells *= factor
            plot_wellplate_dest.width  = int(plot_wellplate_dest.width * factor)
            plot_wellplate_dest.height = int(plot_wellplate_dest.height * factor)
            plot_wellplate_dest_2.width  = int(plot_wellplate_dest_2.width * factor)
            plot_wellplate_dest_2.height = int(plot_wellplate_dest_2.height * factor)
            zoom_size(factor, cds_labels_dest)
            zoom_size(factor, cds_labels_dest_2)
            zoom_size(factor, cds_labels_dest_present)
            zoom_size(factor, cds_labels_dest_2_present)
            zoom_size(factor, cds_labels_dest_filled)
            zoom_size(factor, cds_labels_dest_2_filled)
            zoom_size(factor, cds_labels_dest_filled_bad)
            zoom_size(factor, cds_labels_dest_2_filled_bad)
        return zoom_cb


    #___________________________________________________________________________________________
    def make_zoom_cb_fish(factor):
        def zoom_cb():
            plot_img_bf.width  = int(plot_img_bf.width * factor)
            plot_img_bf.height = int(plot_img_bf.height * factor)
            plot_img_yfp.width  = int(plot_img_yfp.width * factor)
            plot_img_yfp.height = int(plot_img_yfp.height * factor)
            plot_img_yfp_cropped.width  = int(plot_img_yfp_cropped.width * factor)
            plot_img_yfp_cropped.height = int(plot_img_yfp_cropped.height * factor)
            plot_img_vast.width  = int(plot_img_vast.width * factor)
            plot_img_vast.height = int(plot_img_vast.height * factor)

        return zoom_cb

    zoom_in_wells.on_click(make_zoom_cb_wells(1.2))
    zoom_out_wells.on_click(make_zoom_cb_wells(1./1.2))

    zoom_in_fish.on_click(make_zoom_cb_fish(1.2))
    zoom_out_fish.on_click(make_zoom_cb_fish(1./1.2))


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
    def update_filled_wells():
        print('------------------->>>>>>>>> update_filled_wells')
        well_plate_1 = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=1).first()
        dest_1 = DestWellPosition.objects.filter(well_plate=well_plate_1)

        x_dest_1_filled = []
        y_dest_1_filled = []
        size_dest_1_filled = []

        x_dest_1_filled_bad = []
        y_dest_1_filled_bad = []
        size_dest_1_filled_bad = []
        for dest in dest_1:
            try:
                props = dest.dest_well_properties  # reverse OneToOne accessor
                if props.valid:
                    x_dest_1_filled.append(dest.position_col)
                    y_dest_1_filled.append(dest.position_row)
                    size_dest_1_filled.append(cds_labels_dest.data['size'][0])
                else:
                    x_dest_1_filled_bad.append(dest.position_col)
                    y_dest_1_filled_bad.append(dest.position_row)
                    size_dest_1_filled_bad.append(cds_labels_dest.data['size'][0])
            except DestWellProperties.DoesNotExist:
                pass

        cds_labels_dest_filled.data = {'x':x_dest_1_filled, 'y':y_dest_1_filled, 'size':size_dest_1_filled}
        cds_labels_dest_filled_bad.data = {'x':x_dest_1_filled_bad, 'y':y_dest_1_filled_bad, 'size':size_dest_1_filled_bad}

        well_plate_2 = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=2).first()
        dest_2 = DestWellPosition.objects.filter(well_plate=well_plate_2)
        x_dest_2_filled = []
        y_dest_2_filled = []
        size_dest_2_filled = []

        x_dest_2_filled_bad = []
        y_dest_2_filled_bad = []
        size_dest_2_filled_bad = []
        for dest in dest_2:
            try:
                props = dest.dest_well_properties  # reverse OneToOne accessor
                if props.valid:
                    x_dest_2_filled.append(dest.position_col)
                    y_dest_2_filled.append(dest.position_row)
                    size_dest_2_filled.append(cds_labels_dest_2.data['size'][0])
                else:
                    x_dest_2_filled_bad.append(dest.position_col)
                    y_dest_2_filled_bad.append(dest.position_row)
                    size_dest_2_filled_bad.append(cds_labels_dest_2.data['size'][0])
            except DestWellProperties.DoesNotExist:
                pass
        cds_labels_dest_2_filled.data = {'x':x_dest_2_filled, 'y':y_dest_2_filled, 'size':size_dest_2_filled}
        cds_labels_dest_2_filled_bad.data = {'x':x_dest_2_filled_bad, 'y':y_dest_2_filled_bad, 'size':size_dest_2_filled_bad}


    use_corrected_checkbox = bokeh.models.Checkbox(label="Use corrected", active=True)

    #___________________________________________________________________________________________
    def dest_plate_visu(attr, old, new):
        if len(cds_labels_dest.selected.indices) == 0:
            if len(cds_labels_dest_2.selected.indices) == 0:
                source_img_bf.data  = {'img':[]}
                source_img_yfp.data = {'img':[]}
                source_img_yfp_cropped.data = {'img':[]}
                source_img_vast.data = {'img':[]}
            return
        cds_labels_dest_2_present.selected.indices = []
        cds_labels_dest_2.selected.indices = []
        cds_labels_dest_2_filled.selected.indices = []
        cds_labels_dest_2_filled_bad.selected.indices = []
        position = get_well_mapping(cds_labels_dest.selected.indices)

        prediction_message.visible = False

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, dropdown_exp.value)):
            LOCALPATH = LOCALPATH_RAID5

        print('=======================LOCALPATH=', LOCALPATH)

        path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 1', 'Well_{}{}'.format(position[0][1], position[0][0]), 'corrected_orientation' if use_corrected_checkbox.active else '')
        if int(position[0][0]) < 10:
            path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 1', 'Well_{}0{}'.format(position[0][1], position[0][0]), 'corrected_orientation' if use_corrected_checkbox.active else '')  
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
            source_img_yfp_cropped.data = {'img':[]}
            source_img_vast.data = {'img':[]}
            drug_message.text = ""
            drug_message.visible = False
            image_message.text = "<b style='color:red; font-size:18px;'>No images found for selected well {}</b>".format(position[0][1] + position[0][0])
            image_message.visible = True
            prediction_message.visible = False


            dropdown_total_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_total_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            dropdown_good_orientation.value = 'Not set'
            images_comments.value = ''

            return

        image_message.text = ""
        image_message.visible = False

        image_bf  = imread(file_BF)
        source_img_bf.data  = {'img':[np.flip(image_bf,0)]}

        image_yfp = imread(file_YFP)
        source_img_yfp.data = {'img':[np.flip(image_yfp,0)]}
        # Crop the YFP image around the fish

        nonlocal ncrop
        cropped_yfp = image_yfp[768+ncrop:1280+ncrop, :]
        source_img_yfp_cropped.data = {'img':[np.flip(cropped_yfp,0)]}
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

        #predict_callback()

        well_plate = DestWellPlate.objects.filter(experiment__name=dropdown_exp.value, plate_number=1).first()
        dest = DestWellPosition.objects.filter(well_plate=well_plate, position_col=position[0][0], position_row=position[0][1])
        if dest[0].source_well is None:
            print('No source well found for dest position:', position[0])
            drug_message.text = "<b style='color:red; font-size:18px;'>No source well found for selected well {}</b>".format(position[0][1] + position[0][0])
            drug_message.visible = True
            dropdown_total_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_total_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            dropdown_good_orientation.value = 'Not set'
            images_comments.value = ''
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
            if dest_well_properties.n_total_somites is not None:
                dropdown_total_somites.value = str(dest_well_properties.n_total_somites)
            else:
                dropdown_total_somites.value = 'Select a value'
            if dest_well_properties.n_bad_somites is not None:
                dropdown_bad_somites.value  = str(dest_well_properties.n_bad_somites)
            else:
                dropdown_bad_somites.value = 'Select a value'
            dropdown_total_somites_err.value = str(dest_well_properties.n_total_somites_err)
            dropdown_bad_somites_err.value  = str(dest_well_properties.n_bad_somites_err)
            if dest_well_properties.valid:
                dropdown_good_image.value = 'Yes'
            else:
                dropdown_good_image.value = 'No'
            if dest_well_properties.correct_orientation == True:
                dropdown_good_orientation.value = 'Yes'
            elif dest_well_properties.correct_orientation == False:
                dropdown_good_orientation.value = 'No'
            else:
                dropdown_good_orientation.value = 'Not set'

            if dest_well_properties.comments is not None:
                images_comments.value = dest_well_properties.comments
            else:
                images_comments.value = ''
        except DestWellProperties.DoesNotExist:
            print('No properties found for dest well:', dest)
            dropdown_total_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_total_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            dropdown_good_orientation.value = 'Not set'
            images_comments.value = ''


    cds_labels_dest.selected.on_change('indices', lambda attr, old, new: dest_plate_visu(attr, old, new))
    use_corrected_checkbox.on_change("active", lambda attr, old, new: dest_plate_visu(attr, old, new))

    #___________________________________________________________________________________________
    def dest_plate_2_visu(attr, old, new):
        if len(cds_labels_dest_2.selected.indices) == 0:
            if len(cds_labels_dest.selected.indices) == 0:
                source_img_bf.data  = {'img':[]}
                source_img_yfp.data = {'img':[]}
                source_img_yfp_cropped.data = {'img':[]}
                source_img_vast.data = {'img':[]}
            return
        cds_labels_dest_present.selected.indices = []
        cds_labels_dest.selected.indices = []
        cds_labels_dest_filled.selected.indices = []
        cds_labels_dest_filled_bad.selected.indices = []

        prediction_message.visible = False

        position = get_well_mapping(cds_labels_dest_2.selected.indices) 

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, dropdown_exp.value)):
            LOCALPATH = LOCALPATH_RAID5

        print('=======================LOCALPATH=', LOCALPATH)

        path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 2', 'Well_{}{}'.format(position[0][1], position[0][0]), 'corrected_orientation' if use_corrected_checkbox.active else '')
        if int(position[0][0]) < 10:
            path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 2', 'Well_{}0{}'.format(position[0][1], position[0][0]), 'corrected_orientation' if use_corrected_checkbox.active else '')  
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
            source_img_yfp_cropped.data = {'img':[]}
            source_img_vast.data = {'img':[]}
            drug_message.text = ""
            drug_message.visible = False
            image_message.text = "<b style='color:red; font-size:18px;'>No images found for selected well {}</b>".format(position[0][1] + position[0][0])
            image_message.visible = True
            prediction_message.visible = False
            dropdown_total_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_total_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            dropdown_good_orientation.value = 'Not set'
            images_comments.value = ''
            return


        image_bf  = imread(file_BF)
        source_img_bf.data  = {'img':[np.flip(image_bf,0)]}

        image_yfp = imread(file_YFP)
        source_img_yfp.data = {'img':[np.flip(image_yfp,0)]}

        nonlocal ncrop
        cropped_yfp = image_yfp[768+ncrop:1280+ncrop, :]
        source_img_yfp_cropped.data = {'img':[np.flip(cropped_yfp,0)]}

        path_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 2', 'Well_{}{}'.format(position[0][1], position[0][0]))
        if int(position[0][0]) < 10:
            path_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 2', 'Well_{}0{}'.format(position[0][1], position[0][0]))  
        files = glob.glob(os.path.join(path_vast, '*.tiff'))

        #predict_callback()


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
            dropdown_total_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_total_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            dropdown_good_orientation.value = 'Not set'
            images_comments.value = ''
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
            if dest_well_properties.n_total_somites is not None:
                dropdown_total_somites.value = str(dest_well_properties.n_total_somites)
            else:
                dropdown_total_somites.value = 'Select a value'
            if dest_well_properties.n_bad_somites is not None:
                dropdown_bad_somites.value  = str(dest_well_properties.n_bad_somites)
            else:
                dropdown_bad_somites.value = 'Select a value'
            dropdown_total_somites_err.value = str(dest_well_properties.n_total_somites_err)
            dropdown_bad_somites_err.value  = str(dest_well_properties.n_bad_somites_err)
            if dest_well_properties.valid:
                dropdown_good_image.value = 'Yes'
            else:
                dropdown_good_image.value = 'No'

            if dest_well_properties.correct_orientation == True:
                dropdown_good_orientation.value = 'Yes'
            elif dest_well_properties.correct_orientation == False:
                dropdown_good_orientation.value = 'No'
            else:
                dropdown_good_orientation.value = 'Not set'
            if dest_well_properties.comments is not None:
                images_comments.value = dest_well_properties.comments
            else:
                images_comments.value = ''
        except DestWellProperties.DoesNotExist:
            print('No properties found for dest well:', dest)
            dropdown_total_somites.value = 'Select a value'
            dropdown_bad_somites.value  = 'Select a value'
            dropdown_total_somites_err.value = '0'
            dropdown_bad_somites_err.value  = '0'
            dropdown_good_image.value = 'Yes'
            dropdown_good_orientation.value = 'Not set'
            images_comments.value = ''


    cds_labels_dest_2.selected.on_change('indices', lambda attr, old, new: dest_plate_2_visu(attr, old, new))
    use_corrected_checkbox.on_change("active", lambda attr, old, new: dest_plate_2_visu(attr, old, new))


    #___________________________________________________________________________________________
    def move_crop_up():
        nonlocal ncrop
        data = source_img_yfp.data['img']
        if len(data) == 0:
            return
        data = data[0]

        ncrop += 10
        y1=int(768-ncrop)
        y2=int(1280-ncrop)
        # Crop the YFP image around the fish
        cropped_yfp = data[y1:y2, :]
        source_img_yfp_cropped.data = {'img':[cropped_yfp]}

    move_crop_up_button = bokeh.models.Button(label="move crop up")
    move_crop_up_button.on_click(move_crop_up)

    #___________________________________________________________________________________________
    def move_crop_down():
        nonlocal ncrop
        data = source_img_yfp.data['img']
        if len(data) == 0:
            return
        data = data[0]
        ncrop -= 10
        y1=int(768-ncrop)
        y2=int(1280-ncrop)
        # Crop the YFP image around the fish
        cropped_yfp = data[y1:y2, :]
        source_img_yfp_cropped.data = {'img':[cropped_yfp]}

    move_crop_down_button = bokeh.models.Button(label="move crop down")
    move_crop_down_button.on_click(move_crop_down)


    #___________________________________________________________________________________________
    def load_experiment(attr, old, new):
        nonlocal nzoom_wells
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
                cds_labels_dest.data = dict(source_labels_96.data, size=[30*nzoom_wells]*len(source_labels_96.data['x']) if cds_labels_dest.data['size']==[] else cds_labels_dest.data['size'])
                plot_wellplate_dest.axis.visible = True

            elif dest_well_plate.plate_type == '48':
                plot_wellplate_dest.x_range.factors = x_48
                plot_wellplate_dest.y_range.factors = y_48
                plot_wellplate_dest.title.text = "48 well plate"
                cds_labels_dest.data = dict(source_labels_48.data, size=[42*nzoom_wells]*len(source_labels_48.data['x']) if cds_labels_dest.data['size']==[] else cds_labels_dest.data['size'])
                plot_wellplate_dest.axis.visible = True

            elif dest_well_plate.plate_type == '24':
                plot_wellplate_dest.x_range.factors = x_24
                plot_wellplate_dest.y_range.factors = y_24
                plot_wellplate_dest.title.text = "24 well plate"
                cds_labels_dest.data = dict(source_labels_24.data, size=[55*nzoom_wells]*len(source_labels_24.data['x']) if cds_labels_dest.data['size']==[] else cds_labels_dest.data['size'])
                plot_wellplate_dest.axis.visible = True

        if n_plates==2:
            dest_well_plate_2 = dest_well_plates[1]
            if dest_well_plate_2.plate_type == '96':
                plot_wellplate_dest_2.x_range.factors = x_96
                plot_wellplate_dest_2.y_range.factors = y_96
                plot_wellplate_dest_2.title.text = "96 well plate"
                cds_labels_dest_2.data = dict(source_labels_96.data, size=[30*nzoom_wells]*len(source_labels_96.data['x']) if cds_labels_dest_2.data['size']==[] else cds_labels_dest_2.data['size'])
                plot_wellplate_dest_2.axis.visible = True

            elif dest_well_plate_2.plate_type == '48':
                plot_wellplate_dest_2.x_range.factors = x_48
                plot_wellplate_dest_2.y_range.factors = y_48
                plot_wellplate_dest_2.title.text = "48 well plate"
                cds_labels_dest_2.data = dict(source_labels_48.data, size=[42*nzoom_wells]*len(source_labels_48.data['x']) if cds_labels_dest_2.data['size']==[] else cds_labels_dest_2.data['size'])
                plot_wellplate_dest_2.axis.visible = True

            elif dest_well_plate_2.plate_type == '24':
                plot_wellplate_dest_2.x_range.factors = x_24
                plot_wellplate_dest_2.y_range.factors = y_24
                plot_wellplate_dest_2.title.text = "24 well plate"
                cds_labels_dest_2.data = dict(source_labels_24.data, size=[55*nzoom_wells]*len(source_labels_24.data['x']) if cds_labels_dest_2.data['size']==[] else cds_labels_dest_2.data['size'])
                plot_wellplate_dest_2.axis.visible = True

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, dropdown_exp.value)):
            LOCALPATH = LOCALPATH_RAID5

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
            cds_labels_dest_filled.data = dict(x=[], y=[], size=[])
            cds_labels_dest_filled_bad.data = dict(x=[], y=[], size=[])
            cds_labels_dest_2_filled.data = dict(x=[], y=[], size=[])
            cds_labels_dest_2_filled_bad.data = dict(x=[], y=[], size=[])
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

        update_filled_wells()

        cds_labels_dest_present.selected.indices = []
        cds_labels_dest_2_present.selected.indices = []
        cds_labels_dest.selected.indices = []
        cds_labels_dest_2.selected.indices = []

        source_img_bf.data  = {'img':[]}
        source_img_yfp.data = {'img':[]}
        source_img_yfp_cropped.data = {'img':[]}
        source_img_vast.data = {'img':[]}

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
        of.orient_fish(LOCALPATH, dropdown_exp.value)
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


    dropdown_total_somites     = bokeh.models.Select(value='Select a value', title='# total somites', options=['Select a value', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'])
    dropdown_bad_somites       = bokeh.models.Select(value='Select a value', title='# bad somites',  options=['Select a value','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'])
    dropdown_total_somites_err = bokeh.models.Select(value='0', title='# total somites error', options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    dropdown_bad_somites_err   = bokeh.models.Select(value='0', title='# bad somites error',  options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    dropdown_good_image        = bokeh.models.Select(value='Yes', title='Good image', options=['Yes', 'No'])
    dropdown_good_orientation  = bokeh.models.Select(value='Not set', title='Good orientation', options=['Not set', 'Yes', 'No'])
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


        if dropdown_total_somites.value == 'Select a value' or dropdown_bad_somites.value == 'Select a value':
            print('Please select at least one of # total somites or # bad somites')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select at least one of # total somites or # bad somites</b>"
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
        dest_well_properties.n_total_somites = int(dropdown_total_somites.value) if dropdown_total_somites.value != 'Select a value' else None
        dest_well_properties.n_bad_somites  = int(dropdown_bad_somites.value)  if dropdown_bad_somites.value != 'Select a value' else None
        dest_well_properties.n_total_somites_err = int(dropdown_total_somites_err.value)
        dest_well_properties.n_bad_somites_err  = int(dropdown_bad_somites_err.value)
        dest_well_properties.valid = True if dropdown_good_image.value == 'Yes' else False
        if dropdown_good_orientation.value == 'Not set':
            pass
        else:
            dest_well_properties.correct_orientation = True if dropdown_good_orientation.value == 'Yes' else False
        dest_well_properties.comments = images_comments.value
        dest_well_properties.save()
        print('Saved properties for dest well:', dest, ' properties:', dest_well_properties)

        saveimages_button.label = "Save"
        saveimages_button.button_type = "success"
        update_filled_wells()

    saveimages_button = bokeh.models.Button(label="Save", button_type="success", width=150)


    #___________________________________________________________________________________________
    def saveimages_callback_short():
        saveimages_button.label = "Processing"
        saveimages_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(saveimages_callback)
    saveimages_button.on_click(saveimages_callback_short)

    predict_button = bokeh.models.Button(label="Predict", button_type="success", width=150)
    predict_button_fullwell = bokeh.models.Button(label="Predict Full Plate", button_type="success", width=150)

#___________________________________________________________________________________________
    def predict_callback():
        print('------------------->>>>>>>>> predict_callback')
        if dropdown_exp.value == 'Select experiment':
            print('Please select an experiment first')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select an experiment first</b>"
            image_message.visible = True
            predict_button.label = "Predict"
            predict_button.button_type = "success"
            return
        print('Predicting properties for experiment:', dropdown_exp.value)

        if len(cds_labels_dest.selected.indices) == 0 and len(cds_labels_dest_2.selected.indices) == 0:
            print('Please select a well first')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select a well first</b>"
            image_message.visible = True
            predict_button.label = "Predict"
            predict_button.button_type = "success"
            return
        
        if len(cds_labels_dest.selected.indices) > 0 and len(cds_labels_dest_2.selected.indices) > 0:
            print('Please select a well in only one plate')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select a well in only one plate</b>"
            image_message.visible = True
            predict_button.label = "Predict"
            predict_button.button_type = "success"
            return
        
        position=None
        plate="1"
        if len(cds_labels_dest.selected.indices) > 0:
            position = get_well_mapping(cds_labels_dest.selected.indices)
        elif len(cds_labels_dest_2.selected.indices) > 0:
            position = get_well_mapping(cds_labels_dest_2.selected.indices) 
            plate="2"

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, dropdown_exp.value)):
            LOCALPATH = LOCALPATH_RAID5



        path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate {}'.format(plate), 'Well_{}{}'.format(position[0][1], position[0][0]), 'corrected_orientation' if use_corrected_checkbox.active else '')
        if int(position[0][0]) < 10:
            path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate {}'.format(plate), 'Well_{}0{}'.format(position[0][1], position[0][0]), 'corrected_orientation' if use_corrected_checkbox.active else '')  
        files = glob.glob(os.path.join(path_leica, '*.tiff'))


        for f in files:
            if 'YFP' in f and 'norm' not in f:
                file_YFP = f
                img_raw, img_tensor = load_and_prepare_image(file_YFP)
                img_tensor = img_tensor.to(device)
                # Prediction
                with torch.no_grad():
                    pred = model(img_tensor).cpu().numpy().flatten()
                    logit = model_fish(img_tensor.to(device))    # shape [1,1]
                    prob = torch.sigmoid(logit)[0,0].item()
            if 'BF' in f and 'norm' not in f:
                file_BF = f
                img_np = np.array(Image.open(file_BF)).astype(np.float32)
                img = preprocess_image(img_np)
                img = img.unsqueeze(0).cuda()   # 1,1,H,W
                #img_raw, img_tensor = load_and_prepare_image(file_BF)
                #img_tensor = img_tensor.to(device)
                # Prediction
                with torch.no_grad():
                    #logit_ori = model_orientation(img_tensor.to(device))
                    #prob_ori = torch.sigmoid(logit_ori).item()  # scalar

                    logit_ori = model_orientation(img.to(device))
                    prob_ori = torch.sigmoid(logit_ori)


        pred_total_raw, pred_def_raw = pred
        pred_total = clamp_somite_count(pred_total_raw)
        pred_def   = clamp_somite_count(pred_def_raw)
        prediction_message.text = (
            "<b style='color:blue; font-size:18px;'>Predicting Total {} (raw {:.2f})"
            "  --  defective {} (raw {:.2f})"
            "  --  Valid Fish {}"
            "  --  Prob orientation {:.2} +/- {:.2} </b>"
        ).format(pred_total, pred_total_raw, pred_def, pred_def_raw,
                 'Yes' if prob > 0.5 else 'No', prob_ori.mean(), prob_ori.std())
        prediction_message.visible = True

        predict_button.label = "Predict"
        predict_button.button_type = "success"

#___________________________________________________________________________________________
    def predict_callback_short():
        predict_button.label = "Processing"
        predict_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(predict_callback)
    predict_button.on_click(predict_callback_short)

#___________________________________________________________________________________________
    def predict_callback_fullwell():
        print('------------------->>>>>>>>> predict_callback')
        if dropdown_exp.value == 'Select experiment':
            print('Please select an experiment first')
            image_message.text = "<b style='color:red; font-size:18px;'>Please select an experiment first</b>"
            image_message.visible = True
            predict_button_fullwell.label = "Predict Full Plate"
            predict_button_fullwell.button_type = "success"
            return

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, dropdown_exp.value)):
            LOCALPATH = LOCALPATH_RAID5

        experiment  = Experiment.objects.get(name=dropdown_exp.value)
        dest_well_plates   = DestWellPlate.objects.filter(experiment=experiment)
        print('------dest_well_plates=', dest_well_plates)
        for dest_well_plate in dest_well_plates:
            print('     ---------dest well plate ',dest_well_plate)
            dest_well_positions = DestWellPosition.objects.filter(well_plate=dest_well_plate)
            for dest in dest_well_positions:
                print('       =============dest ',dest)
                path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate {}'.format(dest_well_plate.plate_number), 'Well_{}{}'.format(dest.position_row, dest.position_col), 'corrected_orientation' if use_corrected_checkbox.active else '')
                if int(dest.position_col) < 10:
                    path_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate {}'.format(dest_well_plate.plate_number), 'Well_{}0{}'.format(dest.position_row, dest.position_col), 'corrected_orientation' if use_corrected_checkbox.active else '')  
                files = glob.glob(os.path.join(path_leica, '*.tiff'))

                print('path_leica ',path_leica)
                pred_somite=None
                prob_valid=None
                prob_ori=None
                for f in files:
                    if 'YFP' in f and 'norm' not in f:
                        file_YFP = f
                        img_raw, img_tensor = load_and_prepare_image(file_YFP)
                        img_tensor = img_tensor.to(device)
                        # Prediction
                        with torch.no_grad():
                            pred_somite = model(img_tensor).cpu().numpy().flatten()
                            logit = model_fish(img_tensor.to(device))    # shape [1,1]
                            prob_valid = torch.sigmoid(logit)[0,0].item()
                    if 'BF' in f and 'norm' not in f:
                        file_BF = f
                        img_np = np.array(Image.open(file_BF)).astype(np.float32)
                        img = preprocess_image(img_np)
                        img = img.unsqueeze(0).cuda()   # 1,1,H,W
                        with torch.no_grad():
                            logit_ori = model_orientation(img.to(device))
                            prob_ori = torch.sigmoid(logit_ori)

                if len(files)>0:
                    pred_total_raw, pred_def_raw = pred_somite
                    DestWellPropertiesPredicted.objects.update_or_create(
                        dest_well=dest,
                        model_name=RESNET_MODEL_NAME,
                        model_version='',
                        defaults={
                            'n_total_somites':     clamp_somite_count(pred_total_raw),
                            'n_bad_somites':       clamp_somite_count(pred_def_raw),
                            'valid':               bool(prob_valid > 0.5),
                            'correct_orientation': bool(prob_ori.mean() > 0.5),
                        },
                    )

        predict_button_fullwell.label = "Predict Full Plate"
        predict_button_fullwell.button_type = "success"

#___________________________________________________________________________________________
    def predict_callback_fullwell_short():
        predict_button_fullwell.label = "Processing"
        predict_button_fullwell.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(predict_callback_fullwell)
    predict_button_fullwell.on_click(predict_callback_fullwell_short)

    #___________________________________________________________________________________________
    def create_training_callback():
        print('------------------->>>>>>>>> create_training_callback')
        experiments = Experiment.objects.all()

        if os.path.exists(LOCALPATH_TRAINING):
            shutil.rmtree(LOCALPATH_TRAINING)
        if os.path.exists(LOCALPATH_TRAINING) is False:
            os.mkdir(LOCALPATH_TRAINING)
        for sub in ('train', 'valid', 'test'):
            sub_path = os.path.join(LOCALPATH_TRAINING, sub)
            if os.path.exists(sub_path) is False:
                os.mkdir(sub_path)
        

        for experiment in experiments:
            print('Creating training set for experiment:', experiment.name)


            LOCALPATH = LOCALPATH_HIVE
            if os.path.exists(os.path.join(LOCALPATH_RAID5, experiment.name)):
                LOCALPATH = LOCALPATH_RAID5

            dest_well_plates   = DestWellPlate.objects.filter(experiment=experiment)
            print('dest_well_plates=', dest_well_plates)
            for dest_well_plate in dest_well_plates:
                dest_well_positions = DestWellPosition.objects.filter(well_plate=dest_well_plate)
                for dest in dest_well_positions:
                    try:
                        props = dest.dest_well_properties  # reverse OneToOne accessor
                        # Project rule: never include valid=False annotations
                        # in any training data. (Same rule applied by the
                        # resplit_training_data management command.)
                        if not props.valid:
                            continue
                        if props.n_total_somites>=0 and props.n_bad_somites >=0:
                            # Pick the bucket. If the annotation already has a
                            # flag set (e.g. assigned by a previous run or by
                            # the resplit_training_data command), respect it.
                            # Otherwise do a fresh 70/15/15 split.
                            if (not props.use_for_training and
                                not props.use_for_validation and
                                not props.use_for_test):
                                rand = random.uniform(0, 1)
                                if rand < 0.70:
                                    bucket = 'train'
                                    props.use_for_training = True
                                elif rand < 0.85:
                                    bucket = 'valid'
                                    props.use_for_validation = True
                                else:
                                    bucket = 'test'
                                    props.use_for_test = True
                                props.save()
                            elif props.use_for_training:
                                bucket = 'train'
                            elif props.use_for_validation:
                                bucket = 'valid'
                            elif props.use_for_test:
                                bucket = 'test'
                            else:
                                continue
                            outdir = os.path.join(LOCALPATH_TRAINING, bucket)

                            position_col = dest.position_col
                            position_row = dest.position_row
                            path_leica = os.path.join(LOCALPATH, experiment.name,'Leica images', 'Plate {}'.format(dest_well_plate.plate_number), 'Well_{}{}'.format(position_row, position_col), 'corrected_orientation' if use_corrected_checkbox.active else '')
                            if int(position_col) < 10:
                                path_leica = os.path.join(LOCALPATH, experiment.name,'Leica images', 'Plate {}'.format(dest_well_plate.plate_number), 'Well_{}0{}'.format(position_row, position_col), 'corrected_orientation' if use_corrected_checkbox.active else '')  
                            files_YFP = glob.glob(os.path.join(path_leica, '*YFP*.tiff'))
                            files_BF  = glob.glob(os.path.join(path_leica, '*BF*.tiff'))
                            for f in files_YFP:
                                if 'norm' in f:
                                    continue
                                file_YFP = f

                            for f in files_BF:
                                if 'norm' in f:
                                    continue
                                file_BF = f
                            if len(files_YFP)==0 or len(files_BF)== 0:
                                print('No files found in path:', path_leica)
                                continue

                            # Copy the files to the training set folder with a new name
                            new_name_yfp = experiment.name + '_Plate' + str(dest_well_plate.plate_number) + '_' + position_row + position_col + '_YFP.tiff'
                            new_name_bf  = experiment.name + '_Plate' + str(dest_well_plate.plate_number) + '_' + position_row + position_col + '_BF.tiff'
                            shutil.copy(file_YFP, os.path.join(outdir, new_name_yfp))
                            shutil.copy(file_BF, os.path.join(outdir, new_name_bf))
                            out_json_yfp = new_name_yfp.replace('.tiff', '.json')
                            out_json_bf = new_name_bf.replace('.tiff', '.json')

                            data = {
                                'n_total_somites': props.n_total_somites,
                                'n_bad_somites': props.n_bad_somites,
                                'n_total_somites_err': props.n_total_somites_err,
                                'n_bad_somites_err': props.n_bad_somites_err,
                                'valid': props.valid,
                                'correct_orientation': props.correct_orientation,
                                'comments': props.comments,
                            }
                            with open(os.path.join(outdir, out_json_yfp), 'w') as f:
                                json.dump(data, f, indent=4)
                            print('Copied files to training set:', new_name_yfp)
                            with open(os.path.join(outdir, out_json_bf), 'w') as f:
                                json.dump(data, f, indent=4)
                            print('Copied files to training set:', new_name_bf)

                    except DestWellProperties.DoesNotExist:
                        pass

        create_training_button.label = "Create Training Set"
        create_training_button.button_type = "success"

    create_training_button = bokeh.models.Button(label="Create Training Set", button_type="success", width=150)

    #___________________________________________________________________________________________
    def create_training_callback_short():
        create_training_button.label = "Processing"
        create_training_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(create_training_callback)
    create_training_button.on_click(create_training_callback_short)




    plot_wellplate_dest.add_tools(tap_tool)
    plot_wellplate_dest_2.add_tools(tap_tool)

    color_mapper = bokeh.models.LinearColorMapper(palette="Greys256", low=color_low, high=color_high)

    data_img_bf   = {'img':[]}
    source_img_bf = bokeh.models.ColumnDataSource(data=data_img_bf)
    # All four image plots share roughly the same target width inside the
    # tabs container so the strip is visually consistent. The cropped YFP
    # keeps its wide aspect (it's a strip across the fish); VAST is naturally
    # tall (multi-frame stack) but is now constrained so the dashboard fits.
    IMG_TAB_W = 600
    plot_img_bf   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="pan,box_select,wheel_zoom,box_zoom,reset,undo",width=IMG_TAB_W, height=IMG_TAB_W)
    plot_img_bf.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_bf, color_mapper=color_mapper)

    data_img_yfp   = {'img':[]}
    source_img_yfp = bokeh.models.ColumnDataSource(data=data_img_yfp)
    plot_img_yfp   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="pan,box_select,wheel_zoom,box_zoom,reset,undo",width=IMG_TAB_W, height=IMG_TAB_W)
    plot_img_yfp.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_yfp, color_mapper=color_mapper)

    data_img_yfp_cropped   = {'img':[]}
    source_img_yfp_cropped = bokeh.models.ColumnDataSource(data=data_img_yfp_cropped)
    plot_img_yfp_cropped   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="pan,box_select,wheel_zoom,box_zoom,reset,undo",width=IMG_TAB_W, height=200)
    plot_img_yfp_cropped.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_yfp_cropped, color_mapper=color_mapper)

    data_img_vast   = {'img':[]}
    source_img_vast = bokeh.models.ColumnDataSource(data=data_img_vast)
    x_range_2 = bokeh.models.Range1d(start=0, end=1024)
    y_range_2 = bokeh.models.Range1d(start=0, end=200*4)
    plot_img_vast   = bokeh.plotting.figure(x_range=x_range_2, y_range=y_range_2, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=IMG_TAB_W, height=520)
    plot_img_vast.image_rgba(image='img', x=0, y=0, dw=1024, dh=200*4, source=source_img_vast)




    indent = bokeh.models.Spacer(width=30)

    plot_img_bf.axis.visible   = False
    plot_img_bf.grid.visible   = False
    plot_img_yfp.axis.visible  = False
    plot_img_yfp.grid.visible  = False
    plot_img_yfp_cropped.axis.visible = False
    plot_img_yfp_cropped.grid.visible = False
    plot_img_vast.axis.visible = False
    plot_img_vast.grid.visible = False

    # ---- Layout assembly (Tier-2 reorg) -----------------------------------
    # All widgets and callbacks above are unchanged. We only group the
    # existing widgets into clearly-labelled regions:
    #   top bar (experiment + actions + status messages)
    #   left  panel: destination plates + plate-zoom controls
    #   center panel: image tabs (BF / YFP / YFP cropped / VAST) + image controls
    #   right panel: annotation form + prediction controls
    def _section_header(text, w=None):
        d = bokeh.models.Div(
            text=(f'<div style="font-size:13px; font-weight:700; color:#1a2340;'
                  f' border-bottom:2px solid #5b8dee;'
                  f' padding:6px 4px; margin:4px 0 8px;">{text}</div>'),
        )
        if w is not None:
            d.width = w
        return d

    top_bar = bokeh.layouts.row(
        indent,
        bokeh.layouts.column(
            _section_header("Experiment"),
            dropdown_exp,
        ),
        bokeh.models.Spacer(width=20),
        bokeh.layouts.column(
            _section_header("Actions"),
            bokeh.layouts.row(well_mapping_button, create_training_button),
        ),
        bokeh.models.Spacer(width=40),
        bokeh.layouts.column(
            _section_header("Status"),
            image_message,
            drug_message,
        ),
    )

    plates_panel = bokeh.layouts.column(
        _section_header("Destination plates"),
        plot_wellplate_dest,
        plot_wellplate_dest_2,
        _section_header("Plate zoom"),
        bokeh.layouts.row(zoom_in_wells, zoom_out_wells),
    )

    image_tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.TabPanel(child=plot_img_yfp_cropped, title="YFP cropped"),
        bokeh.models.TabPanel(child=plot_img_yfp,         title="YFP"),
        bokeh.models.TabPanel(child=plot_img_bf,          title="BF"),
        bokeh.models.TabPanel(child=plot_img_vast,        title="VAST"),
    ])

    image_panel = bokeh.layouts.column(
        _section_header("Image view"),
        image_tabs,
        contrast_slider,
        _section_header("Fish zoom &amp; crop"),
        bokeh.layouts.row(zoom_in_fish, zoom_out_fish),
        bokeh.layouts.row(move_crop_up_button, move_crop_down_button),
    )

    annotation_panel = bokeh.layouts.column(
        _section_header("Annotation"),
        bokeh.layouts.row(dropdown_total_somites, dropdown_total_somites_err),
        bokeh.layouts.row(dropdown_bad_somites,   dropdown_bad_somites_err),
        bokeh.layouts.row(dropdown_good_image,    dropdown_good_orientation),
        images_comments,
        saveimages_button,
    )

    prediction_panel = bokeh.layouts.column(
        _section_header("Prediction"),
        bokeh.layouts.row(predict_button, predict_button_fullwell),
        use_corrected_checkbox,
        prediction_message,
    )

    right_col = bokeh.layouts.column(
        annotation_panel,
        bokeh.layouts.Spacer(height=20),
        prediction_panel,
    )

    main_row = bokeh.layouts.row(
        indent,
        plates_panel,
        bokeh.layouts.Spacer(width=20),
        image_panel,
        bokeh.layouts.Spacer(width=20),
        right_col,
    )

    norm_layout = bokeh.layouts.column(
        top_bar,
        bokeh.layouts.Spacer(height=15),
        main_row,
    )

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



# views.py
from django.shortcuts import render

def drug_list(request):

    drugs_data = []
    source_wells = SourceWellPosition.objects.all()

    for sw in source_wells:
        #print('Source well:', sw, ' has drugs:', sw.drugs.all())
        dest_wells = DestWellPosition.objects.filter(source_well=sw)
        n_dest_wells = dest_wells.count()
        n_fish_valid = 0
        n_fish_notvalid = 0
        n_total_somites = []
        n_bad_somites = []
        dest_wp_1 = []
        dest_wp_2 = []
        image_list_valid = []
        image_list_not_valid = []

        well_name_valid =[]
        well_name_invalid =[]

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, sw.well_plate.experiment.name)):
            LOCALPATH = LOCALPATH_RAID5

        print('=======================LOCALPATH drug list page =', LOCALPATH)


        for dest in dest_wells:
            try:
                props = dest.dest_well_properties  # reverse OneToOne accessor
                path_leica = os.path.join(LOCALPATH, sw.well_plate.experiment.name,'Leica images', 'Plate {}'.format(dest.well_plate.plate_number), 'Well_{}{}'.format(dest.position_row, dest.position_col), 'corrected_orientation')
                if int(dest.position_col) < 10:
                    path_leica = os.path.join(LOCALPATH, sw.well_plate.experiment.name,'Leica images', 'Plate {}'.format(dest.well_plate.plate_number), 'Well_{}0{}'.format(dest.position_row, dest.position_col), 'corrected_orientation')  
                print('path_leica=', path_leica)
                files = glob.glob(os.path.join(path_leica, '*YFP*_norm8.png'))
                if len(files) == 0:
                    tiff_path = os.path.join(path_leica, '*YFP*_norm8.tiff')
                    tiff_files = glob.glob(tiff_path)
                    for tiff_file in tiff_files:
                        png_path = tiff_file.replace('.tiff', '.png')
                        img = Image.open(tiff_file)
                        #imgd = ImageDraw.Draw(img)
                        #myFont = ImageFont.truetype('FreeMono.ttf', 40)
                        #mf = ImageFont.truetype('font.ttf', 25)
                        #myfont = ImageFont.truetype("sans-serif.ttf", 40)
                        #imgd.text((10,10), "Plate {} Well {}{} ".format(dest.well_plate.plate_number, dest.position_row, dest.position_col), font=myfont, fill=(255,255,255))

                        img.save(png_path)
                    files = glob.glob(os.path.join(path_leica, '*YFP*_norm8.png'))
                if props.valid:
                    n_fish_valid +=1
                    if props.n_total_somites is not None:
                        n_total_somites.append(props.n_total_somites)
                    if props.n_bad_somites is not None:
                        n_bad_somites.append(props.n_bad_somites)
                    for f in files:
                        image_list_valid.append(f)
                    well_name_valid.append('Plate {} Well {}{}'.format(dest.well_plate.plate_number, dest.position_row, dest.position_col))
                else:
                    n_fish_notvalid +=1
                    for f in files:
                        image_list_not_valid.append(f)
                    well_name_invalid.append('Plate {} Well {}{}'.format(dest.well_plate.plate_number, dest.position_row, dest.position_col))
                if dest.well_plate.plate_number == 1:
                    dest_wp_1.append('{}{}'.format(dest.position_row, dest.position_col))
                if dest.well_plate.plate_number == 2:
                    dest_wp_2.append('{}{}'.format(dest.position_row, dest.position_col))
            except DestWellProperties.DoesNotExist:
                pass


        image_list_valid = [
            to_media_url(p) for p in image_list_valid
        ]

        image_list_not_valid = [
            to_media_url(p) for p in image_list_not_valid
        ]

        print("image_list_valid=", image_list_valid)
        print("image_list_not_valid=", image_list_not_valid)

        dest_plate_info = '' 
        if len(dest_wp_1) > 0:
            dest_plate_info += 'Plate1: ' + ', '.join(dest_wp_1)
        if len(dest_wp_2) > 0:
            if len(dest_plate_info) > 0:
                dest_plate_info += ' | '
            dest_plate_info += 'Plate2: ' + ', '.join(dest_wp_2)



        image_list_valid_pairs = []
        for img, name in zip(image_list_valid, well_name_valid):
            image_list_valid_pairs.append((img, name))

        image_list_invalid_pairs = []
        for img, name in zip(image_list_not_valid, well_name_invalid):
            image_list_invalid_pairs.append((img, name))    

       

        well_data = {
            "exp": sw.well_plate.experiment.name,
            "well": f"{sw.position_row}{sw.position_col}",
            "valid": sw.valid,
            "dest_wells": dest_plate_info,
            "drugs": [{"name": drug.derivation_name, "conc": f"{drug.concentration} µM"} for drug in sw.drugs.all()],
            "number_of_drugs": sw.drugs.count(),
            "number_of_dest_wells": n_dest_wells,
            "number_of_fish": n_fish_notvalid+n_fish_valid,
            "number_of_fish_valid": n_fish_valid,
            "number_of_fish_notvalid": n_fish_notvalid,
            "avg_total_somites": np.mean(n_total_somites) if len(n_total_somites) > 0 else None,
            "avg_bad_somites": np.mean(n_bad_somites) if len(n_bad_somites) > 0 else None,
            "total_somites_err": np.std(n_total_somites) if len(n_total_somites) > 0 else None,
            "bad_somites_err": np.std(n_bad_somites) if len(n_bad_somites) > 0 else None,
            "fraction_bad_somites": (np.mean(n_bad_somites) / np.mean(n_total_somites)) if len(n_total_somites) > 0 else None,
            "images_valid": image_list_valid_pairs,
            "images_invalid": image_list_invalid_pairs,

        }
        if len(well_data["drugs"])>0:  # Only add wells that have drugs
            drugs_data.append(well_data)

    return render(request, "well_explorer/drugs_listing.html", {"rows": drugs_data})

#___________________________________________________________________________________________
def experiment_list(request: HttpRequest) -> HttpResponse:
    data=[]
    experiments = Experiment.objects.all()

    n_fish_valid_total = 0
    n_fish_notvalid_total = 0
    n_imaged_total = 0
    n_fish_total = 0
    n_dest_wells_total = 0
    n_mapped_wells_total = 0
    for exp in experiments:

        data.append({'name': exp.name, 'date_created': exp.date, 'description': exp.description})
        n_fish_valid = 0
        n_fish_notvalid = 0
        n_fish = 0
        n_imaged = 0
        n_dest_wells = 0
        n_mapped_wells = 0

        dest_well_plates = DestWellPlate.objects.filter(experiment=exp)
        for plate in dest_well_plates:
            dest_well_positions = DestWellPosition.objects.filter(well_plate=plate)
            n_dest_wells += dest_well_positions.count()
            n_mapped_wells += dest_well_positions.filter(source_well__isnull=False).count()

            for dest in dest_well_positions:
                has_ann = False
                has_pred = False
                try:
                    props = dest.dest_well_properties
                    has_ann = True
                    if props.valid:
                        n_fish_valid += 1
                    else:
                        n_fish_notvalid += 1
                except DestWellProperties.DoesNotExist:
                    pass
                if latest_prediction(dest, model_name=RESNET_MODEL_NAME) is not None:
                    has_pred = True
                if has_ann or has_pred:
                    n_imaged += 1

        data[-1]['n_dest_wells'] = n_dest_wells
        data[-1]['n_mapped_wells'] = n_mapped_wells
        data[-1]['n_imaged'] = n_imaged
        data[-1]['n_fish_valid'] = n_fish_valid
        data[-1]['n_fish_notvalid'] = n_fish_notvalid
        data[-1]['fraction_valid'] = (n_fish_valid / (n_fish_valid + n_fish_notvalid)) * 100 if (n_fish_valid + n_fish_notvalid) > 0 else 0
        data[-1]['vast_efficiency'] = (n_imaged / n_mapped_wells) * 100 if n_mapped_wells > 0 else 0
        n_fish = n_fish_valid + n_fish_notvalid
        data[-1]['n_fish'] = n_fish
        n_fish_valid_total += n_fish_valid
        n_fish_notvalid_total += n_fish_notvalid
        n_fish_total += n_fish
        n_imaged_total += n_imaged
        n_dest_wells_total += n_dest_wells
        n_mapped_wells_total += n_mapped_wells
    data_total = {
        'name': 'TOTAL', 'date_created': '', 'description': '',
        'n_dest_wells': n_dest_wells_total,
        'n_mapped_wells': n_mapped_wells_total,
        'n_imaged': n_imaged_total,
        'n_fish_valid': n_fish_valid_total,
        'n_fish_notvalid': n_fish_notvalid_total,
        'fraction_valid': (n_fish_valid_total / (n_fish_valid_total + n_fish_notvalid_total)) * 100 if (n_fish_valid_total + n_fish_notvalid_total) > 0 else 0,
        'vast_efficiency': (n_imaged_total / n_mapped_wells_total) * 100 if n_mapped_wells_total > 0 else 0,
        'n_fish': n_fish_total,
    }
    return render(request, 'well_explorer/experiment_listing.html', {'rows': data, 'data_total': data_total})

#___________________________________________________________________________________________
def stats_list(request: HttpRequest) -> HttpResponse:
    experiments = Experiment.objects.all().order_by('name')
    rows = []

    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    def _std(lst):
        if not lst or len(lst) < 2:
            return None
        m = sum(lst) / len(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

    def _subset_metrics(b):
        n = b['n']
        return {
            'n': n,
            'agree_valid_pct': (b['agree_valid'] / n * 100) if n > 0 else None,
            'agree_orient_pct': (b['agree_orient'] / n * 100) if n > 0 else None,
            'mae_total': _mean(b['somite_diffs']),
            'mae_bad': _mean(b['bad_diffs']),
        }

    def _aggregate_buckets(buckets):
        agg = {
            'n': sum(b['n'] for b in buckets),
            'agree_valid': sum(b['agree_valid'] for b in buckets),
            'agree_orient': sum(b['agree_orient'] for b in buckets),
            'somite_diffs': [v for b in buckets for v in b['somite_diffs']],
            'bad_diffs': [v for b in buckets for v in b['bad_diffs']],
        }
        return _subset_metrics(agg)

    tk = {
        'n_dest_wells': 0, 'n_mapped_wells': 0, 'n_imaged': 0,
        'n_valid': 0, 'n_invalid': 0,
        'n_annotated': 0, 'n_predicted': 0, 'n_both': 0,
        'n_training': 0, 'n_validation': 0, 'n_test': 0,
    }
    t_ann_total, t_ann_bad = [], []
    t_ann_total_tr, t_ann_bad_tr = [], []
    t_ann_total_val, t_ann_bad_val = [], []
    t_ann_total_te, t_ann_bad_te = [], []
    # per-subset comparison accumulators (training / validation / held-out test)
    SUBSETS = ('train', 'val', 'test')
    t_subset = {s: {'n': 0, 'agree_valid': 0, 'agree_orient': 0,
                    'somite_diffs': [], 'bad_diffs': []} for s in SUBSETS}

    for exp in experiments:
        n_dest_wells = 0
        n_mapped_wells = 0
        n_imaged = 0
        n_valid = 0
        n_invalid = 0
        n_annotated = 0
        n_predicted = 0
        n_both = 0
        n_training = 0
        n_validation = 0
        n_test = 0
        ann_total, ann_bad = [], []
        ann_total_tr, ann_bad_tr = [], []
        ann_total_val, ann_bad_val = [], []
        ann_total_te, ann_bad_te = [], []
        # per-subset comparison accumulators for this experiment
        sub = {s: {'n': 0, 'agree_valid': 0, 'agree_orient': 0,
                   'somite_diffs': [], 'bad_diffs': []} for s in SUBSETS}

        dest_well_plates = DestWellPlate.objects.filter(experiment=exp)
        for plate in dest_well_plates:
            dest_well_positions = DestWellPosition.objects.filter(well_plate=plate)
            n_dest_wells += dest_well_positions.count()
            n_mapped_wells += dest_well_positions.filter(source_well__isnull=False).count()

            for dest in dest_well_positions:
                props = None
                pred = None
                has_ann = False
                has_pred = False

                try:
                    props = dest.dest_well_properties
                    has_ann = True
                    n_annotated += 1
                    if props.valid:
                        n_valid += 1
                    else:
                        n_invalid += 1
                    # Somite counts are only meaningful on valid fish — invalid
                    # ones aren't counted properly and aren't used for analysis,
                    # so we exclude them from the somite-count statistics.
                    if props.valid:
                        if props.n_total_somites != -9999:
                            ann_total.append(props.n_total_somites)
                        if props.n_bad_somites != -9999:
                            ann_bad.append(props.n_bad_somites)
                    if props.use_for_training:
                        n_training += 1
                        if props.valid:
                            if props.n_total_somites != -9999:
                                ann_total_tr.append(props.n_total_somites)
                            if props.n_bad_somites != -9999:
                                ann_bad_tr.append(props.n_bad_somites)
                    if props.use_for_validation:
                        n_validation += 1
                        if props.valid:
                            if props.n_total_somites != -9999:
                                ann_total_val.append(props.n_total_somites)
                            if props.n_bad_somites != -9999:
                                ann_bad_val.append(props.n_bad_somites)
                    if props.use_for_test:
                        n_test += 1
                        if props.valid:
                            if props.n_total_somites != -9999:
                                ann_total_te.append(props.n_total_somites)
                            if props.n_bad_somites != -9999:
                                ann_bad_te.append(props.n_bad_somites)
                except DestWellProperties.DoesNotExist:
                    pass

                pred = latest_prediction(dest, model_name=RESNET_MODEL_NAME)
                if pred is not None:
                    has_pred = True
                    n_predicted += 1

                if has_ann or has_pred:
                    n_imaged += 1

                # Prediction-vs-annotation comparison is restricted to wells
                # whose manual annotation is `valid=True`: invalid fish have
                # unreliable somite counts and are excluded from analysis, so
                # comparing predictions against them would be noise.
                if has_ann and has_pred and props.valid:
                    # Bucket by the explicit subset flag. Annotations with
                    # NO flag set (use_for_training / _validation / _test all
                    # false) are skipped — we don't know which subset they
                    # belong to, and the "honest" test column would be
                    # misleading if we lumped them in.
                    if props.use_for_training:
                        s_key = 'train'
                    elif props.use_for_validation:
                        s_key = 'val'
                    elif props.use_for_test:
                        s_key = 'test'
                    else:
                        continue
                    n_both += 1
                    bucket = sub[s_key]
                    bucket['n'] += 1
                    if props.valid == pred.valid:
                        bucket['agree_valid'] += 1
                    if props.correct_orientation == pred.correct_orientation:
                        bucket['agree_orient'] += 1
                    if props.n_total_somites != -9999 and pred.n_total_somites != -9999:
                        bucket['somite_diffs'].append(abs(props.n_total_somites - pred.n_total_somites))
                    if props.n_bad_somites != -9999 and pred.n_bad_somites != -9999:
                        bucket['bad_diffs'].append(abs(props.n_bad_somites - pred.n_bad_somites))

        rows.append({
            'name': exp.name, 'date': exp.date,
            'n_dest_wells': n_dest_wells,
            'n_mapped_wells': n_mapped_wells,
            'plate_fill_eff': (n_mapped_wells / n_dest_wells * 100) if n_dest_wells > 0 else 0,
            'n_imaged': n_imaged,
            'vast_efficiency': (n_imaged / n_mapped_wells * 100) if n_mapped_wells > 0 else 0,
            'n_valid': n_valid,
            'n_invalid': n_invalid,
            'frac_valid': (n_valid / (n_valid + n_invalid) * 100) if (n_valid + n_invalid) > 0 else 0,
            'n_annotated': n_annotated,
            'mean_total_somites': _mean(ann_total),
            'std_total_somites': _std(ann_total),
            'mean_bad_somites': _mean(ann_bad),
            'std_bad_somites': _std(ann_bad),
            'n_training': n_training,
            'mean_total_tr': _mean(ann_total_tr),
            'mean_bad_tr': _mean(ann_bad_tr),
            'n_validation': n_validation,
            'mean_total_val': _mean(ann_total_val),
            'mean_bad_val': _mean(ann_bad_val),
            'n_test': n_test,
            'mean_total_te': _mean(ann_total_te),
            'mean_bad_te': _mean(ann_bad_te),
            'n_predicted': n_predicted,
            'n_both': n_both,
            'cmp_all':   _aggregate_buckets([sub[s] for s in SUBSETS]),
            'cmp_train': _subset_metrics(sub['train']),
            'cmp_val':   _subset_metrics(sub['val']),
            'cmp_test':  _subset_metrics(sub['test']),
        })

        for k, v in {'n_dest_wells': n_dest_wells, 'n_mapped_wells': n_mapped_wells,
                     'n_imaged': n_imaged, 'n_valid': n_valid, 'n_invalid': n_invalid,
                     'n_annotated': n_annotated, 'n_predicted': n_predicted,
                     'n_both': n_both, 'n_training': n_training,
                     'n_validation': n_validation, 'n_test': n_test}.items():
            tk[k] += v
        t_ann_total += ann_total
        t_ann_bad += ann_bad
        t_ann_total_tr += ann_total_tr
        t_ann_bad_tr += ann_bad_tr
        t_ann_total_val += ann_total_val
        t_ann_bad_val += ann_bad_val
        t_ann_total_te += ann_total_te
        t_ann_bad_te += ann_bad_te
        for s in SUBSETS:
            t_subset[s]['n'] += sub[s]['n']
            t_subset[s]['agree_valid'] += sub[s]['agree_valid']
            t_subset[s]['agree_orient'] += sub[s]['agree_orient']
            t_subset[s]['somite_diffs'] += sub[s]['somite_diffs']
            t_subset[s]['bad_diffs'] += sub[s]['bad_diffs']

    nd, nm, nb = tk['n_dest_wells'], tk['n_mapped_wells'], tk['n_both']
    nv = tk['n_valid'] + tk['n_invalid']
    total_row = {
        'name': 'TOTAL',
        'n_dest_wells': nd,
        'n_mapped_wells': nm,
        'plate_fill_eff': (nm / nd * 100) if nd > 0 else 0,
        'n_imaged': tk['n_imaged'],
        'vast_efficiency': (tk['n_imaged'] / nm * 100) if nm > 0 else 0,
        'n_valid': tk['n_valid'],
        'n_invalid': tk['n_invalid'],
        'frac_valid': (tk['n_valid'] / nv * 100) if nv > 0 else 0,
        'n_annotated': tk['n_annotated'],
        'mean_total_somites': _mean(t_ann_total),
        'std_total_somites': _std(t_ann_total),
        'mean_bad_somites': _mean(t_ann_bad),
        'std_bad_somites': _std(t_ann_bad),
        'n_training': tk['n_training'],
        'mean_total_tr': _mean(t_ann_total_tr),
        'mean_bad_tr': _mean(t_ann_bad_tr),
        'n_validation': tk['n_validation'],
        'mean_total_val': _mean(t_ann_total_val),
        'mean_bad_val': _mean(t_ann_bad_val),
        'n_test': tk['n_test'],
        'mean_total_te': _mean(t_ann_total_te),
        'mean_bad_te': _mean(t_ann_bad_te),
        'n_predicted': tk['n_predicted'],
        'n_both': nb,
        'cmp_all':   _aggregate_buckets([t_subset[s] for s in SUBSETS]),
        'cmp_train': _subset_metrics(t_subset['train']),
        'cmp_val':   _subset_metrics(t_subset['val']),
        'cmp_test':  _subset_metrics(t_subset['test']),
    }
    return render(request, 'well_explorer/stats_listing.html', {'rows': rows, 'data_total': total_row})


#___________________________________________________________________________________________
def drug_plot_handler(doc: bokeh.document.Document) -> None:
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    all_drug_names = sorted(list(Drug.objects.values_list('derivation_name', flat=True).distinct()))
    all_exp_names  = sorted([e.name for e in Experiment.objects.all()])

    if not all_drug_names:
        doc.add_root(bokeh.models.Div(text='<b>No drugs found in the database.</b>'))
        return

    # --- Widgets ---
    drug_select = bokeh.models.Select(
        title='Drug derivation', options=all_drug_names,
        value=all_drug_names[0], width=350,
    )
    exp_multi = bokeh.models.MultiSelect(
        title='Experiments (Ctrl+click for multi-select)',
        options=all_exp_names, value=all_exp_names,
        height=180, width=350,
    )
    # Concentration filter — narrows histograms + scatter to wells dosed at
    # one specific concentration. Options auto-refresh when drug/experiment
    # changes. Does NOT affect the dose-response plot (that always shows all).
    ALL_CONC_LABEL = 'All concentrations'
    concentration_select = bokeh.models.Select(
        title='Concentration filter',
        value=ALL_CONC_LABEL,
        options=[ALL_CONC_LABEL],
        width=350,
    )
    source_radio = bokeh.models.RadioGroup(
        labels=['Predicted only', 'Annotated only', 'Both (overlay)'],
        active=0,
    )
    valid_radio = bokeh.models.RadioGroup(
        labels=['Valid fish only',
                'Invalid fish only',
                'All fish',
                'Valid in BOTH ann & pred (intersection)'],
        active=0,
    )
    ann_subset_radio = bokeh.models.RadioGroup(
        labels=['All annotations', 'Training set', 'Validation set'],
        active=0,
    )
    update_button = bokeh.models.Button(label='Update plot', button_type='primary', width=350)
    status_div = bokeh.models.Div(text='', width=350, styles={'color': '#666', 'font-style': 'italic'})

    # --- Figures ---
    p_total = bokeh.plotting.figure(
        title='Total somites', width=580, height=360,
        x_axis_label='Number of somites', y_axis_label='Density',
        tools='pan,wheel_zoom,box_zoom,reset,save',
    )
    p_bad = bokeh.plotting.figure(
        title='Defective somites', width=580, height=360,
        x_axis_label='Number of somites', y_axis_label='Density',
        tools='pan,wheel_zoom,box_zoom,reset,save',
    )

    # --- Data sources ---
    src_total_pred = bokeh.models.ColumnDataSource(data=dict(top=[], left=[], right=[]))
    src_total_ann  = bokeh.models.ColumnDataSource(data=dict(top=[], left=[], right=[]))
    src_bad_pred   = bokeh.models.ColumnDataSource(data=dict(top=[], left=[], right=[]))
    src_bad_ann    = bokeh.models.ColumnDataSource(data=dict(top=[], left=[], right=[]))

    p_total.quad(top='top', bottom=0, left='left', right='right',
                 source=src_total_pred, fill_color='#2196F3', fill_alpha=0.65,
                 line_color='white', legend_label='Predicted')
    p_total.quad(top='top', bottom=0, left='left', right='right',
                 source=src_total_ann, fill_color='#FF9800', fill_alpha=0.65,
                 line_color='white', legend_label='Annotated')
    p_total.legend.click_policy = 'hide'
    p_total.legend.location = 'top_right'

    p_bad.quad(top='top', bottom=0, left='left', right='right',
               source=src_bad_pred, fill_color='#2196F3', fill_alpha=0.65,
               line_color='white', legend_label='Predicted')
    p_bad.quad(top='top', bottom=0, left='left', right='right',
               source=src_bad_ann, fill_color='#FF9800', fill_alpha=0.65,
               line_color='white', legend_label='Annotated')
    p_bad.legend.click_policy = 'hide'
    p_bad.legend.location = 'top_right'

    # --- Scatter plots: predicted vs annotated, paired per dest well ---
    SUBSET_COLORS = {'train': '#FF9800', 'val': '#4CAF50', 'test': '#2196F3'}
    SUBSET_LABELS = {'train': 'Train', 'val': 'Val', 'test': 'Test'}

    p_scatter_total = bokeh.plotting.figure(
        title='Total somites — predicted vs annotated', width=580, height=400,
        x_axis_label='Annotated', y_axis_label='Predicted',
        tools='pan,wheel_zoom,box_zoom,reset,save,hover',
        match_aspect=True,
    )
    p_scatter_bad = bokeh.plotting.figure(
        title='Defective somites — predicted vs annotated', width=580, height=400,
        x_axis_label='Annotated', y_axis_label='Predicted',
        tools='pan,wheel_zoom,box_zoom,reset,save,hover',
        match_aspect=True,
    )

    # y=x reference line
    for p in (p_scatter_total, p_scatter_bad):
        p.add_layout(bokeh.models.Slope(gradient=1, y_intercept=0,
                                        line_color='#888', line_dash='dashed', line_width=1))

    src_scatter_total = {s: bokeh.models.ColumnDataSource(data=dict(x=[], y=[])) for s in SUBSET_COLORS}
    src_scatter_bad   = {s: bokeh.models.ColumnDataSource(data=dict(x=[], y=[])) for s in SUBSET_COLORS}

    for s, color in SUBSET_COLORS.items():
        p_scatter_total.scatter('x', 'y', source=src_scatter_total[s], size=8,
                                fill_color=color, line_color='white', fill_alpha=0.7,
                                legend_label=SUBSET_LABELS[s])
        p_scatter_bad.scatter('x', 'y', source=src_scatter_bad[s], size=8,
                              fill_color=color, line_color='white', fill_alpha=0.7,
                              legend_label=SUBSET_LABELS[s])

    for p in (p_scatter_total, p_scatter_bad):
        p.legend.click_policy = 'hide'
        p.legend.location = 'top_left'

    # --- Dose-response plots: somite count vs drug concentration ---
    # Same drug derivation can have several `Drug` rows with different
    # `concentration` values (one row per concentration); we aggregate
    # somite measurements per concentration here. Log x-axis by default —
    # drug screens typically span several orders of magnitude.
    p_dose_total = bokeh.plotting.figure(
        title='Total somites vs concentration',
        width=580, height=380,
        x_axis_label='Concentration', y_axis_label='Total somites (mean ± SEM)',
        x_axis_type='log',
        tools='pan,wheel_zoom,box_zoom,reset,save,hover',
    )
    p_dose_bad = bokeh.plotting.figure(
        title='Defective somites vs concentration',
        width=580, height=380,
        x_axis_label='Concentration', y_axis_label='Defective somites (mean ± SEM)',
        x_axis_type='log',
        tools='pan,wheel_zoom,box_zoom,reset,save,hover',
    )

    # CDS columns: x (concentration), y (mean), upper / lower (y ± SEM),
    # n (sample count per concentration — used in hover tooltip).
    src_dose_total_pred = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], upper=[], lower=[], n=[]))
    src_dose_total_ann  = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], upper=[], lower=[], n=[]))
    src_dose_bad_pred   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], upper=[], lower=[], n=[]))
    src_dose_bad_ann    = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], upper=[], lower=[], n=[]))

    def _add_dose_series(fig, src, color, legend):
        # connecting line (helps see dose-response trend across concentrations)
        fig.line('x', 'y', source=src, color=color, line_width=2, alpha=0.6,
                 legend_label=legend)
        # markers
        fig.scatter('x', 'y', source=src, size=10, fill_color=color,
                    line_color='white', legend_label=legend)
        # error bars
        fig.add_layout(bokeh.models.Whisker(
            base='x', upper='upper', lower='lower', source=src,
            line_color=color, line_width=1.5,
            upper_head=bokeh.models.TeeHead(size=10, line_color=color),
            lower_head=bokeh.models.TeeHead(size=10, line_color=color),
        ))

    _add_dose_series(p_dose_total, src_dose_total_pred, '#2196F3', 'Predicted')
    _add_dose_series(p_dose_total, src_dose_total_ann,  '#FF9800', 'Annotated')
    _add_dose_series(p_dose_bad,   src_dose_bad_pred,   '#2196F3', 'Predicted')
    _add_dose_series(p_dose_bad,   src_dose_bad_ann,    '#FF9800', 'Annotated')

    for p in (p_dose_total, p_dose_bad):
        p.legend.click_policy = 'hide'
        p.legend.location = 'top_right'
        # Hover shows concentration / mean / sample count
        hover = p.select(dict(type=bokeh.models.HoverTool))
        if hover:
            hover[0].tooltips = [
                ('Concentration', '@x'),
                ('Mean',          '@y{0.00}'),
                ('N',             '@n'),
            ]

    dose_note = bokeh.models.Div(
        text=('<i style="color:#666; font-size:12px">'
              'Dose-response: somite counts (valid fish only) grouped by '
              '<code>Drug.concentration</code> for the selected derivation. '
              'Log x-axis; concentrations &le; 0 are dropped. Each point is '
              'the mean across all wells at that concentration; error bars are '
              'standard error of the mean.</i>'),
        width=1180)

    stats_div = bokeh.models.Div(text='', width=600, styles={'font-size': '13px', 'line-height': '1.6'})

    # --- Helpers ---
    def _shared_hist(vals_a, vals_b, n_bins=25):
        combined = vals_a + vals_b
        if not combined:
            empty = dict(top=[], left=[], right=[])
            return empty, empty
        arr = np.array(combined, dtype=float)
        bins = np.linspace(arr.min(), arr.max() + 1e-9, n_bins + 1)

        def _to_dict(vals):
            if not vals:
                return dict(top=[0.0] * n_bins,
                            left=bins[:-1].tolist(), right=bins[1:].tolist())
            h, _ = np.histogram(np.array(vals, dtype=float), bins=bins, density=True)
            return dict(top=h.tolist(), left=bins[:-1].tolist(), right=bins[1:].tolist())

        return _to_dict(vals_a), _to_dict(vals_b)

    def _fmt(label, vals):
        if not vals:
            return f'<b>{label}</b>: no data<br>'
        a = np.array(vals, dtype=float)
        return (f'<b>{label}</b> &mdash; N={len(a)}, '
                f'mean={a.mean():.2f}, median={float(np.median(a)):.2f}, '
                f'std={a.std():.2f}<br>')

    def _refresh_concentrations():
        """Repopulate `concentration_select` whenever the drug or selected
        experiments change. Preserves the current selection if it still exists
        in the new option list; falls back to 'All concentrations' otherwise."""
        drug = drug_select.value
        selected = exp_multi.value
        if not drug or not selected:
            concentration_select.options = [ALL_CONC_LABEL]
            concentration_select.value = ALL_CONC_LABEL
            return
        raw = (Drug.objects
               .filter(derivation_name=drug,
                       position__well_plate__experiment__name__in=selected)
               .values_list('concentration', flat=True)
               .distinct())
        concs = sorted({float(c) for c in raw
                        if c is not None and c > 0 and c != -9999})
        # Pretty-print: trim trailing zeros, keep at least one decimal place.
        def _fmt_conc(c):
            s = f'{c:.6g}'
            return s
        opts = [ALL_CONC_LABEL] + [_fmt_conc(c) for c in concs]
        prev = concentration_select.value
        concentration_select.options = opts
        concentration_select.value = prev if prev in opts else ALL_CONC_LABEL

    drug_select.on_change('value',
                           lambda attr, old, new: _refresh_concentrations())
    exp_multi.on_change('value',
                         lambda attr, old, new: _refresh_concentrations())
    _refresh_concentrations()   # populate once on page load

    # --- Update callback ---
    def do_update():
        drug         = drug_select.value
        selected     = exp_multi.value
        show_pred    = source_radio.active in (0, 2)
        show_ann     = source_radio.active in (1, 2)
        # Validity filter modes:
        #   0: Valid fish only        — `valid=True`  on whichever side we're querying
        #   1: Invalid fish only      — `valid=False`
        #   2: All fish               — no filter on validity
        #   3: Valid in BOTH (∩)      — same as mode 0 on the primary side, AND we also
        #                               require the OTHER side's `valid=True` via a join.
        #                               This is the "compare like-for-like" mode where
        #                               both annotation and prediction agree the fish
        #                               is valid before we score the somite counts.
        valid_map      = {0: True, 1: False, 2: None, 3: True}
        valid_filter   = valid_map[valid_radio.active]
        intersect_valid = (valid_radio.active == 3)
        ann_filter   = {0: None, 1: 'training', 2: 'validation'}[ann_subset_radio.active]

        if not selected:
            for s in (src_total_pred, src_total_ann, src_bad_pred, src_bad_ann):
                s.data = dict(top=[], left=[], right=[])
            for s in SUBSET_COLORS:
                src_scatter_total[s].data = dict(x=[], y=[])
                src_scatter_bad[s].data = dict(x=[], y=[])
            for s in (src_dose_total_pred, src_dose_total_ann,
                      src_dose_bad_pred, src_dose_bad_ann):
                s.data = dict(x=[], y=[], upper=[], lower=[], n=[])
            status_div.text = 'No experiments selected.'
            stats_div.text = ''
            return

        status_div.text = 'Fetching data…'

        # Resolve concentration filter (None = all concentrations)
        conc_filter = None
        if concentration_select.value != ALL_CONC_LABEL:
            try:
                conc_filter = float(concentration_select.value)
            except ValueError:
                conc_filter = None

        # Build the drug-related filters as a single kwargs dict so all M2M
        # conditions land in the same Django join — that way derivation_name
        # AND concentration apply to the SAME Drug row, not two different
        # drugs at the source well.
        drug_filters = {
            'dest_well__source_well__isnull': False,
            'dest_well__source_well__drugs__derivation_name': drug,
            'dest_well__well_plate__experiment__name__in': selected,
        }
        if conc_filter is not None:
            drug_filters['dest_well__source_well__drugs__concentration'] = conc_filter

        pred_total, pred_bad = [], []
        if show_pred:
            # NOTE: when SAM lands and produces 'sam_v1' rows, this query
            # will need a 'which model?' dropdown (Phase 3c). For now we
            # always read from the original ResNet predictions.
            qs = DestWellPropertiesPredicted.objects.filter(
                model_name=RESNET_MODEL_NAME,
                **drug_filters,
            ).distinct()
            if valid_filter is not None:
                qs = qs.filter(valid=valid_filter)
            if intersect_valid:
                # Intersection: also require the annotation on the same
                # dest_well to be valid=True. Walks the reverse OneToOne
                # from DestWellPosition to DestWellProperties.
                qs = qs.filter(dest_well__dest_well_properties__valid=True)
            for d in qs:
                if d.n_total_somites != -9999:
                    pred_total.append(d.n_total_somites)
                if d.n_bad_somites != -9999:
                    pred_bad.append(d.n_bad_somites)

        ann_total, ann_bad = [], []
        if show_ann:
            qs = DestWellProperties.objects.filter(**drug_filters).distinct()
            if valid_filter is not None:
                qs = qs.filter(valid=valid_filter)
            if ann_filter == 'training':
                qs = qs.filter(use_for_training=True)
            elif ann_filter == 'validation':
                qs = qs.filter(use_for_validation=True)
            if intersect_valid:
                # Intersection: also require a prediction on the same
                # dest_well with valid=True. Both filter conditions live
                # in the same .filter() call so they apply to the SAME
                # prediction row (multi-row FK reverse manager).
                qs = qs.filter(
                    dest_well__predictions__valid=True,
                    dest_well__predictions__model_name=RESNET_MODEL_NAME,
                )
            for d in qs:
                if d.n_total_somites != -9999:
                    ann_total.append(d.n_total_somites)
                if d.n_bad_somites != -9999:
                    ann_bad.append(d.n_bad_somites)

        hist_total_pred, hist_total_ann = _shared_hist(pred_total, ann_total)
        hist_bad_pred,   hist_bad_ann   = _shared_hist(pred_bad,   ann_bad)

        src_total_pred.data = hist_total_pred
        src_total_ann.data  = hist_total_ann
        src_bad_pred.data   = hist_bad_pred
        src_bad_ann.data    = hist_bad_ann

        # --- Paired (annotated, predicted) data for scatter plots, split by subset ---
        pair_total = {s: {'ann': [], 'pred': []} for s in SUBSET_COLORS}
        pair_bad   = {s: {'ann': [], 'pred': []} for s in SUBSET_COLORS}

        qs_pair = DestWellProperties.objects.filter(
            **drug_filters
        ).select_related('dest_well').distinct()
        if valid_filter is not None:
            qs_pair = qs_pair.filter(valid=valid_filter)
        # In intersection mode the pred-side validity is enforced when we
        # look up the paired prediction below.

        for ann in qs_pair:
            pred_obj = latest_prediction(ann.dest_well, model_name=RESNET_MODEL_NAME)
            if pred_obj is None:
                continue
            if intersect_valid and not pred_obj.valid:
                continue
            # Bucket by explicit subset flag; skip annotations with no flag
            # set so the "Test" colour stays an honest generalisation set.
            if ann.use_for_training:
                s_key = 'train'
            elif ann.use_for_validation:
                s_key = 'val'
            elif ann.use_for_test:
                s_key = 'test'
            else:
                continue
            if ann.n_total_somites != -9999 and pred_obj.n_total_somites != -9999:
                pair_total[s_key]['ann'].append(ann.n_total_somites)
                pair_total[s_key]['pred'].append(pred_obj.n_total_somites)
            if ann.n_bad_somites != -9999 and pred_obj.n_bad_somites != -9999:
                pair_bad[s_key]['ann'].append(ann.n_bad_somites)
                pair_bad[s_key]['pred'].append(pred_obj.n_bad_somites)

        for s in SUBSET_COLORS:
            src_scatter_total[s].data = dict(x=pair_total[s]['ann'], y=pair_total[s]['pred'])
            src_scatter_bad[s].data   = dict(x=pair_bad[s]['ann'],   y=pair_bad[s]['pred'])

        def _pair_mae(d):
            if not d['ann']:
                return None
            diffs = [abs(a - p) for a, p in zip(d['ann'], d['pred'])]
            return sum(diffs) / len(diffs)

        # --- Dose-response: aggregate per concentration ---
        # We pull every Drug row with this derivation_name, walk through its
        # source wells → dest wells → measurements, and build {concentration ->
        # {total: [...], bad: [...]}} buckets. Two passes: one for predicted
        # (resnet_v1), one for annotated. Both reuse the existing valid/subset
        # filters that drive the histograms above.
        #
        # Important: the same dest well can be linked from a Drug row only via
        # SourceWellPosition; we go through Drug.position to be explicit.
        def _by_concentration(use_predictions: bool):
            """Return {conc -> {'total': [...], 'bad': [...]}} for valid wells
            of the selected drug in the selected experiments."""
            buckets: dict = {}
            drug_qs = Drug.objects.filter(
                derivation_name=drug,
                position__well_plate__experiment__name__in=selected,
            ).distinct().prefetch_related('position')
            for d in drug_qs:
                conc = d.concentration
                if conc is None or conc <= 0 or conc == -9999:
                    continue
                source_wells = list(d.position.all())
                if not source_wells:
                    continue
                # Wells dosed at this concentration in the selected experiments
                dest_qs = DestWellPosition.objects.filter(
                    source_well__in=source_wells,
                    well_plate__experiment__name__in=selected,
                )
                if use_predictions:
                    pred_qs = DestWellPropertiesPredicted.objects.filter(
                        dest_well__in=dest_qs,
                        model_name=RESNET_MODEL_NAME,
                    )
                    if valid_filter is not None:
                        pred_qs = pred_qs.filter(valid=valid_filter)
                    if intersect_valid:
                        pred_qs = pred_qs.filter(
                            dest_well__dest_well_properties__valid=True)
                    rows = pred_qs
                else:
                    ann_qs = DestWellProperties.objects.filter(dest_well__in=dest_qs)
                    if valid_filter is not None:
                        ann_qs = ann_qs.filter(valid=valid_filter)
                    if intersect_valid:
                        ann_qs = ann_qs.filter(
                            dest_well__predictions__valid=True,
                            dest_well__predictions__model_name=RESNET_MODEL_NAME,
                        )
                    if ann_filter == 'training':
                        ann_qs = ann_qs.filter(use_for_training=True)
                    elif ann_filter == 'validation':
                        ann_qs = ann_qs.filter(use_for_validation=True)
                    rows = ann_qs
                b = buckets.setdefault(conc, {'total': [], 'bad': []})
                for r in rows:
                    if r.n_total_somites != -9999:
                        b['total'].append(r.n_total_somites)
                    if r.n_bad_somites != -9999:
                        b['bad'].append(r.n_bad_somites)
            return buckets

        def _to_dose_cds(buckets: dict, field: str):
            """{conc -> {'total':..., 'bad':...}} → CDS dict for plotting."""
            xs, ys, ups, lows, ns = [], [], [], [], []
            for conc in sorted(buckets.keys()):
                vals = buckets[conc][field]
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                mean = float(arr.mean())
                sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                xs.append(float(conc)); ys.append(mean)
                ups.append(mean + sem); lows.append(mean - sem)
                ns.append(len(arr))
            return dict(x=xs, y=ys, upper=ups, lower=lows, n=ns)

        if show_pred:
            buckets_pred = _by_concentration(use_predictions=True)
            src_dose_total_pred.data = _to_dose_cds(buckets_pred, 'total')
            src_dose_bad_pred.data   = _to_dose_cds(buckets_pred, 'bad')
        else:
            src_dose_total_pred.data = dict(x=[], y=[], upper=[], lower=[], n=[])
            src_dose_bad_pred.data   = dict(x=[], y=[], upper=[], lower=[], n=[])

        if show_ann:
            buckets_ann = _by_concentration(use_predictions=False)
            src_dose_total_ann.data = _to_dose_cds(buckets_ann, 'total')
            src_dose_bad_ann.data   = _to_dose_cds(buckets_ann, 'bad')
        else:
            src_dose_total_ann.data = dict(x=[], y=[], upper=[], lower=[], n=[])
            src_dose_bad_ann.data   = dict(x=[], y=[], upper=[], lower=[], n=[])

        # Count distinct concentrations for the title / banner
        all_xs = set(src_dose_total_pred.data['x']) | set(src_dose_total_ann.data['x'])
        n_conc = len(all_xs)

        valid_label = {0: 'valid', 1: 'invalid', 2: 'all',
                        3: 'valid in both ann & pred'}[valid_radio.active]
        p_total.title.text = f'Total somites — {drug} ({valid_label} fish)'
        p_bad.title.text   = f'Defective somites — {drug} ({valid_label} fish)'
        p_scatter_total.title.text = f'Predicted vs annotated — total somites — {drug}'
        p_scatter_bad.title.text   = f'Predicted vs annotated — defective somites — {drug}'
        p_dose_total.title.text = (
            f'Total somites vs concentration — {drug} '
            f'({n_conc} concentration{"s" if n_conc != 1 else ""})')
        p_dose_bad.title.text = (
            f'Defective somites vs concentration — {drug} '
            f'({n_conc} concentration{"s" if n_conc != 1 else ""})')

        html  = f'<h3>{drug}</h3>'
        html += '<b>Total somites — distribution</b><br>'
        html += _fmt('Predicted', pred_total)
        html += _fmt('Annotated', ann_total)
        html += '<br><b>Defective somites — distribution</b><br>'
        html += _fmt('Predicted', pred_bad)
        html += _fmt('Annotated', ann_bad)
        html += '<br><b>Pred vs Ann — MAE per subset</b><br>'
        for s in SUBSET_COLORS:
            mt = _pair_mae(pair_total[s])
            mb = _pair_mae(pair_bad[s])
            n_t = len(pair_total[s]['ann'])
            n_b = len(pair_bad[s]['ann'])
            mt_str = f'{mt:.2f}' if mt is not None else '—'
            mb_str = f'{mb:.2f}' if mb is not None else '—'
            html += (f'<span style="color:{SUBSET_COLORS[s]}">&#9632;</span> '
                     f'<b>{SUBSET_LABELS[s]}</b>: '
                     f'total N={n_t}, MAE={mt_str} &nbsp;·&nbsp; '
                     f'def. N={n_b}, MAE={mb_str}<br>')
        stats_div.text = html
        status_div.text = ''

    update_button.on_click(do_update)

    # --- Layout ---
    controls = bokeh.layouts.column(
        bokeh.models.Div(text='<h3 style="margin:0 0 8px">Controls</h3>'),
        drug_select,
        exp_multi,
        concentration_select,
        bokeh.models.Div(text='<b>Data source:</b>'),
        source_radio,
        bokeh.models.Div(text='<b>Fish validity:</b>'),
        valid_radio,
        bokeh.models.Div(text='<b>Annotation subset:</b>'),
        ann_subset_radio,
        update_button,
        status_div,
        width=370,
    )
    plots = bokeh.layouts.column(
        bokeh.models.Div(text='<h4 style="margin:0">Dose-response (somite count vs concentration)</h4>'),
        dose_note,
        bokeh.layouts.row(p_dose_total, p_dose_bad),
        bokeh.models.Div(text='<h4 style="margin:18px 0 0">Distributions (histograms)</h4>'),
        bokeh.layouts.row(p_total, p_bad),
        bokeh.models.Div(text='<h4 style="margin:18px 0 0">Predicted vs annotated (paired wells)</h4>'),
        bokeh.layouts.row(p_scatter_total, p_scatter_bad),
        stats_div,
    )
    doc.add_root(bokeh.layouts.row(controls, plots))


#___________________________________________________________________________________________
def drug_plot_page(request: HttpRequest) -> HttpResponse:
    script = bokeh.embed.server_document(request.build_absolute_uri())
    return render(request, 'well_explorer/drug_plot.html', {'script': script})


#___________________________________________________________________________________________
def docs_page(request: HttpRequest) -> HttpResponse:
    """In-app documentation: project overview, page reference, retraining
    instructions. Mirrors the most actionable parts of the README."""
    return render(request, 'well_explorer/docs.html', {})


#___________________________________________________________________________________________
# SAM segmentation dashboard
#
# Workflow:
#   1. user picks experiment → plate → well
#   2. dashboard loads the canonicalised YFP image
#   3. user clicks somites in the image → each click adds a positive
#      point prompt to a CDS
#   4. "Segment" button runs SAM with each point prompt → one mask per click
#   5. masks are turned into per-somite records (centroid / area /
#      AP position / severity=0 default) and shown as markers + table
#   6. "Save" writes a DestWellPropertiesPredicted row with model_name='sam_v1'
#      and per_somite_data populated
#
# Per-somite severity editing and grid-prompt auto-segment land in a follow-up.
SAM_MODEL_NAME = 'sam_v1'


def sam_handler(doc: bokeh.document.Document) -> None:
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    # Lazy-import the SAM wrapper. Don't load weights yet — wait until the
    # user clicks "Segment", since loading is slow and not every visitor
    # to the page wants to run inference.
    from somiteCounting.sam_segmenter import (
        DEFAULT_SAM_CHECKPOINT, DEFAULT_SAM_MODEL_TYPE,
        extract_per_somite_data, get_sam_instance,
    )

    # ----- experiment / plate / well selectors -----
    exp_options = ['Select experiment'] + sorted([e.name for e in Experiment.objects.all()])
    dropdown_exp   = bokeh.models.Select(title='Experiment', value=exp_options[0],
                                         options=exp_options, width=320)
    dropdown_plate = bokeh.models.Select(title='Plate', value='1',
                                         options=['1', '2'], width=120)
    dropdown_well  = bokeh.models.Select(title='Well', value='Select well',
                                         options=['Select well'], width=160)

    # ----- main image figure -----
    img_size = 2048
    img_fig = bokeh.plotting.figure(
        title='YFP image (click somites to add prompts)',
        width=720, height=720,
        x_range=(0, img_size), y_range=(0, img_size),
        tools='pan,wheel_zoom,box_zoom,reset,save,tap',
        match_aspect=True,
    )
    img_fig.axis.visible = False
    img_fig.grid.visible = False

    img_source = bokeh.models.ColumnDataSource(data=dict(image=[], dw=[], dh=[]))
    img_fig.image(image='image', x=0, y=0, dw='dw', dh='dh', source=img_source,
                  palette='Greys256')

    # Mask overlay (a single RGBA image stacking all per-somite masks).
    mask_source = bokeh.models.ColumnDataSource(data=dict(image=[], dw=[], dh=[]))
    img_fig.image_rgba(image='image', x=0, y=0, dw='dw', dh='dh',
                       source=mask_source, alpha=0.45)

    # Click prompts (the user's positive seeds).
    prompts_source = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))
    img_fig.scatter('x', 'y', source=prompts_source, size=14,
                    fill_color='#ff3030', line_color='white', line_width=2,
                    marker='circle')

    # Per-somite centroids (after segmentation).
    centroids_source = bokeh.models.ColumnDataSource(
        data=dict(x=[], y=[], idx=[]))
    img_fig.scatter('x', 'y', source=centroids_source, size=12,
                    fill_color='#2196F3', line_color='white', line_width=1,
                    marker='diamond')
    img_fig.text(x='x', y='y', text='idx', source=centroids_source,
                 text_color='white', text_font_size='10pt',
                 x_offset=-3, y_offset=4)

    # ----- right-column widgets -----
    btn_segment   = bokeh.models.Button(label='Segment', button_type='primary',  width=160)
    btn_clear     = bokeh.models.Button(label='Clear prompts',  button_type='warning', width=160)
    btn_save      = bokeh.models.Button(label='Save', button_type='success',     width=160)
    status_div    = bokeh.models.Div(width=400, height=80, styles={'font-size': '13px'})
    table_source  = bokeh.models.ColumnDataSource(data=dict(
        index=[], centroid_x=[], centroid_y=[], area=[], ap_position=[], severity=[]))
    table = bokeh.models.DataTable(
        source=table_source,
        columns=[
            bokeh.models.TableColumn(field='index',       title='#',     width=40),
            bokeh.models.TableColumn(field='ap_position', title='AP',
                formatter=bokeh.models.NumberFormatter(format='0.00'), width=60),
            bokeh.models.TableColumn(field='area',        title='Area',  width=70),
            bokeh.models.TableColumn(field='centroid_x',  title='cx',    width=60,
                formatter=bokeh.models.NumberFormatter(format='0')),
            bokeh.models.TableColumn(field='centroid_y',  title='cy',    width=60,
                formatter=bokeh.models.NumberFormatter(format='0')),
            bokeh.models.TableColumn(field='severity',    title='Sev.',  width=50),
        ],
        width=400, height=380, index_position=None,
    )

    # Mutable container for the loaded numpy image, so callbacks can read it.
    state = {
        'img_np':     None,   # 2-D float (H, W)
        'masks':      [],     # list of 2-D bool masks
        'somites':    [],     # list of dict (per-somite metadata)
        'dest':       None,   # DestWellPosition currently displayed
    }

    def _set_status(html: str):
        status_div.text = html

    # ----- helpers -----
    def _load_well_image():
        """Look up the canonical YFP image for the current selection and load
        it into `state['img_np']` + the figure's ColumnDataSource."""
        exp_name = dropdown_exp.value
        plate_n  = dropdown_plate.value
        well_str = dropdown_well.value
        if exp_name == 'Select experiment' or well_str == 'Select well':
            return

        # well_str is like 'A03' — split into row letter + col number
        position_row = well_str[0]
        position_col = well_str[1:].lstrip('0') or '0'

        try:
            plate = DestWellPlate.objects.get(experiment__name=exp_name,
                                              plate_number=int(plate_n))
            dest = DestWellPosition.objects.get(well_plate=plate,
                                                position_row=position_row,
                                                position_col=position_col)
        except (DestWellPlate.DoesNotExist, DestWellPosition.DoesNotExist):
            _set_status(f'<i>Well {well_str} not in database for {exp_name} P{plate_n}.</i>')
            return

        state['dest'] = dest
        state['masks'] = []
        state['somites'] = []
        prompts_source.data = dict(x=[], y=[])
        centroids_source.data = dict(x=[], y=[], idx=[])
        mask_source.data = dict(image=[], dw=[], dh=[])
        table_source.data = dict(
            index=[], centroid_x=[], centroid_y=[], area=[], ap_position=[], severity=[])

        # Find the canonicalised YFP image on disk.
        localpath = None
        for cand in (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH):
            if os.path.isdir(os.path.join(cand, exp_name)):
                localpath = cand
                break
        if localpath is None:
            _set_status(f'<i>No local path found for experiment {exp_name}.</i>')
            return

        pad = position_col if int(position_col) >= 10 else f"0{position_col}"
        well_dir = os.path.join(localpath, exp_name, 'Leica images',
                                f'Plate {plate_n}', f'Well_{position_row}{pad}',
                                'corrected_orientation')
        files = glob.glob(os.path.join(well_dir, '*YFP*.tiff'))
        files = [f for f in files if 'norm' not in f.lower()]
        if not files:
            _set_status(f'<i>No canonicalised YFP image found in {well_dir}. '
                        f'Did you run <code>refresh_orientation</code>?</i>')
            return
        try:
            img_np = imread(files[0]).astype(np.float32)
        except Exception as e:
            _set_status(f'<i>Could not read {files[0]}: {e}</i>')
            return

        h, w = img_np.shape[:2]
        # Bokeh's image glyph wants the image flipped vertically so y increases upward
        img_for_bokeh = np.flipud(img_np / max(img_np.max(), 1e-9))
        img_source.data = dict(image=[img_for_bokeh], dw=[w], dh=[h])
        img_fig.x_range.start = 0
        img_fig.x_range.end   = w
        img_fig.y_range.start = 0
        img_fig.y_range.end   = h
        state['img_np'] = img_np
        _set_status(f'<b>Loaded</b> {os.path.basename(files[0])} ({w}×{h}). '
                    f'Click somites to add prompts, then press <b>Segment</b>.')

    def _populate_well_dropdown():
        """When experiment/plate changes, populate `dropdown_well` with the
        wells that exist in the database for that plate."""
        exp_name = dropdown_exp.value
        plate_n  = dropdown_plate.value
        if exp_name == 'Select experiment':
            dropdown_well.options = ['Select well']
            dropdown_well.value = 'Select well'
            return
        try:
            plate = DestWellPlate.objects.get(
                experiment__name=exp_name, plate_number=int(plate_n))
        except DestWellPlate.DoesNotExist:
            dropdown_well.options = ['Select well']
            dropdown_well.value = 'Select well'
            return
        wells = []
        for d in DestWellPosition.objects.filter(well_plate=plate):
            try:
                col = int(d.position_col)
            except (TypeError, ValueError):
                continue
            wells.append(f'{d.position_row}{col:02d}')
        wells = sorted(set(wells))
        dropdown_well.options = ['Select well'] + wells
        dropdown_well.value = 'Select well'

    # ----- click → add prompt -----
    def _on_image_tap(event):
        if state['img_np'] is None:
            _set_status('<i>Select a well first.</i>')
            return
        h = state['img_np'].shape[0]
        x = event.x
        y_disp = event.y
        # Bokeh y is flipped vs numpy (we flipud-ed the image), reverse.
        y = h - y_disp
        d = dict(prompts_source.data)
        d['x'] = list(d['x']) + [x]
        d['y'] = list(d['y']) + [y_disp]   # store display-space for the marker
        prompts_source.data = d

    img_fig.on_event(bokeh.events.Tap, _on_image_tap)

    # ----- buttons -----
    def _on_clear():
        prompts_source.data = dict(x=[], y=[])
        centroids_source.data = dict(x=[], y=[], idx=[])
        mask_source.data = dict(image=[], dw=[], dh=[])
        table_source.data = dict(
            index=[], centroid_x=[], centroid_y=[], area=[], ap_position=[], severity=[])
        state['masks'] = []
        state['somites'] = []
        _set_status('Prompts cleared.')

    btn_clear.on_click(_on_clear)

    def _on_segment():
        if state['img_np'] is None:
            _set_status('<i>Select a well first.</i>'); return
        prompts = prompts_source.data
        if not prompts['x']:
            _set_status('<i>Add at least one point prompt by clicking on a somite.</i>'); return
        sam, err = get_sam_instance()
        if sam is None:
            _set_status(
                '<b style="color:#c00;">SAM not available.</b><br>'
                f'<small>{err or "Unknown error"}</small>')
            return
        _set_status('Running SAM…')

        # Convert display-space prompts (y was flipped for the image glyph)
        # back into image-space.
        h = state['img_np'].shape[0]
        points_xy = list(zip(list(prompts['x']),
                             [h - y for y in list(prompts['y'])]))

        try:
            sam.set_image(state['img_np'])
            masks = sam.segment_at_points(points_xy)
        except Exception as e:
            _set_status(f'<b style="color:#c00;">SAM error:</b> {e}'); return

        state['masks'] = masks
        somites = extract_per_somite_data(masks, state['img_np'].shape[:2])
        state['somites'] = somites

        # Update centroids overlay (display-space y)
        cx = [s['centroid_x'] for s in somites]
        cy = [h - s['centroid_y'] for s in somites]
        idx_strs = [str(s['index']) for s in somites]
        centroids_source.data = dict(x=cx, y=cy, idx=idx_strs)

        # Mask overlay: composite the masks into an RGBA image with one
        # colour per somite (cycle through a small palette).
        if masks:
            H, W = state['img_np'].shape[:2]
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            palette = [
                (255, 87, 51), (255, 195, 0), (76, 175, 80),
                (33, 150, 243), (156, 39, 176), (244, 67, 54),
                (0, 188, 212), (139, 195, 74),
            ]
            for i, m in enumerate(masks):
                r, g, b = palette[i % len(palette)]
                rgba[m, 0] = r
                rgba[m, 1] = g
                rgba[m, 2] = b
                rgba[m, 3] = 180
            mask_source.data = dict(
                image=[np.flipud(rgba.view(np.uint32).reshape(H, W))],
                dw=[W], dh=[H])

        # Update table
        table_source.data = dict(
            index=[s['index'] for s in somites],
            centroid_x=[round(s['centroid_x'], 1) for s in somites],
            centroid_y=[round(s['centroid_y'], 1) for s in somites],
            area=[s['area'] for s in somites],
            ap_position=[s['ap_position'] for s in somites],
            severity=[s['severity'] for s in somites],
        )
        _set_status(f'Segmented <b>{len(somites)}</b> somite(s). '
                    f'Press <b>Save</b> to record.')

    btn_segment.on_click(_on_segment)

    def _on_save():
        dest = state['dest']
        somites = state['somites']
        if dest is None or not somites:
            _set_status('<i>Nothing to save — segment a well first.</i>'); return
        n_total = len(somites)
        n_bad = sum(1 for s in somites if int(s.get('severity', 0)) > 0)
        DestWellPropertiesPredicted.objects.update_or_create(
            dest_well=dest,
            model_name=SAM_MODEL_NAME,
            model_version='',
            defaults={
                'n_total_somites': n_total,
                'n_bad_somites':   n_bad,
                'per_somite_data': somites,
            },
        )
        _set_status(f'<b>Saved.</b> {n_total} somite(s) recorded under '
                    f'<code>model_name="{SAM_MODEL_NAME}"</code>.')

    btn_save.on_click(_on_save)

    # ----- callback wiring on selectors -----
    dropdown_exp.on_change('value',
                            lambda attr, old, new: _populate_well_dropdown())
    dropdown_plate.on_change('value',
                              lambda attr, old, new: _populate_well_dropdown())
    dropdown_well.on_change('value',
                             lambda attr, old, new: _load_well_image())

    # ----- info banner about SAM availability -----
    sam_now, sam_err = get_sam_instance()
    if sam_now is None:
        _set_status('<b style="color:#c00;">SAM not loaded.</b><br>'
                    f'<small>{sam_err}</small><br>'
                    '<small>Default expected checkpoint: '
                    f'<code>{DEFAULT_SAM_CHECKPOINT}</code> '
                    f'(model type <code>{DEFAULT_SAM_MODEL_TYPE}</code>). '
                    'You can still browse images, but Segment will not run.</small>')
    else:
        _set_status('<i>Pick an experiment, plate, and well to begin. '
                    'Click somites to add prompts, then press <b>Segment</b>.</i>')

    # ----- layout -----
    def _section(text):
        return bokeh.models.Div(
            text=(f'<div style="font-size:13px; font-weight:700; color:#1a2340;'
                  f' border-bottom:2px solid #5b8dee;'
                  f' padding:6px 4px; margin:4px 0 8px;">{text}</div>'))

    selectors = bokeh.layouts.row(dropdown_exp, dropdown_plate, dropdown_well)
    actions = bokeh.layouts.row(btn_segment, btn_clear, btn_save)
    right_col = bokeh.layouts.column(
        _section('Selection'), selectors,
        _section('Actions'),   actions,
        _section('Status'),    status_div,
        _section('Per-somite data'), table,
        width=420,
    )
    layout = bokeh.layouts.row(
        bokeh.layouts.column(_section('Image'), img_fig),
        bokeh.layouts.Spacer(width=20),
        right_col,
    )
    doc.add_root(layout)


#___________________________________________________________________________________________
def sam_dashboard(request: HttpRequest) -> HttpResponse:
    """Serve the SAM segmentation dashboard (Bokeh embed)."""
    script = bokeh.embed.server_document(request.build_absolute_uri())
    return render(request, 'well_explorer/sam_dashboard.html', {'script': script})


#___________________________________________________________________________________________
# Model evaluation dashboard
#
# Designed for the "is the model agreeing with the biologist?" question, with
# explicit support for spotting microscope-driven per-well issues.
#
# Workflow:
#   1. Pick an experiment (single, by design — disagreements are usually
#      experiment-specific so we don't mix them up).
#   2. Pick which prediction model_name to score (defaults to resnet_v1).
#   3. Pick the subset of annotations to score against. Default is
#      "Test + Unflagged" — these are wells the model never saw during
#      training (either explicitly flagged use_for_test=True or never
#      assigned to any of the three flags, which covers annotations
#      added AFTER the model was trained).
#   4. The page shows:
#        - aggregate metrics (MAE/RMSE/R² for somites, confusion matrix for
#          valid flag, agreement % for orientation)
#        - a per-plate heatmap of |Δ total somites| so spatial patterns
#          (microscope quadrant issues) jump out
#        - a sortable per-well disagreement table
#        - an image preview pane (YFP/BF tabs) that updates when you click
#          a heatmap cell OR a table row
def model_eval_handler(doc: bokeh.document.Document) -> None:
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

    # ----- Experiment + model choices -----
    all_exp = sorted([e.name for e in Experiment.objects.all()])
    if not all_exp:
        doc.add_root(bokeh.models.Div(text='<b>No experiments in the database.</b>'))
        return

    model_names = sorted(
        set(DestWellPropertiesPredicted.objects
            .values_list('model_name', flat=True).distinct())) or [RESNET_MODEL_NAME]

    exp_select   = bokeh.models.Select(title='Experiment',
                                        value=all_exp[0], options=all_exp, width=320)
    model_select = bokeh.models.Select(title='Model', value=model_names[0],
                                        options=model_names, width=180)
    plate_select = bokeh.models.Select(title='Plate', value='Both',
                                        options=['Both', '1', '2'], width=110)
    subset_radio = bokeh.models.RadioGroup(
        labels=['Test + Unflagged (honest)',
                'Test only (use_for_test=True)',
                'Unflagged only',
                'All annotated (incl. training/validation — biased!)'],
        active=0,
    )
    valid_radio = bokeh.models.RadioGroup(
        labels=['All fish', 'Valid only', 'Invalid only'],
        active=0,
    )
    # Heatmap colouring metric — count comparisons are only meaningful when
    # the annotator marked the fish valid (otherwise the somite counts
    # are leftover defaults), so this dropdown drives just the heatmap
    # colour, not the underlying filter. Cells where ann.valid=False are
    # always greyed out regardless of which metric is chosen.
    COLOR_BY_OPTIONS = {
        'Max(|Δ total|, |Δ def|)': 'max',
        '|Δ total somites|':        'total',
        '|Δ defective somites|':    'def',
    }
    color_by_select = bokeh.models.Select(
        title='Heatmap colour metric',
        value='Max(|Δ total|, |Δ def|)',
        options=list(COLOR_BY_OPTIONS.keys()),
        width=320,
    )
    update_button = bokeh.models.Button(label='Update', button_type='primary', width=320)
    metrics_div = bokeh.models.Div(
        text='<i>Click Update to compute metrics.</i>',
        width=900, styles={'font-size': '13px', 'line-height': '1.6'})
    status_div = bokeh.models.Div(text='', width=900,
                                   styles={'color': '#666', 'font-style': 'italic'})

    # ----- Plate heatmaps (96-well; visualise |Δ total| per well) -----
    X_LABELS = [str(i) for i in range(1, 13)]
    Y_LABELS = list('HGFEDCBA')   # top row = H so axis reads A at the bottom

    def _make_plate_fig(title):
        f = bokeh.plotting.figure(
            x_range=bokeh.models.FactorRange(*X_LABELS),
            y_range=bokeh.models.FactorRange(*Y_LABELS),
            title=title, width=440, height=320,
            tools='tap,reset,save,hover', toolbar_location='above',
        )
        f.xaxis.major_label_text_font_size = '10pt'
        f.yaxis.major_label_text_font_size = '10pt'
        f.grid.visible = False
        return f

    plate_p1_fig = _make_plate_fig('Plate 1 — |Δ total somites|')
    plate_p2_fig = _make_plate_fig('Plate 2 — |Δ total somites|')

    # CDS columns include the full annotation + prediction so the hover
    # tooltip can show everything at a glance, plus two display fields
    # that the heatmap-colouring logic writes to:
    #   color_value  — what the fill colour reads (NaN where ann.valid is
    #                  False, so the cell renders in `nan_color`)
    #   line_color   — per-cell border colour; red when the valid flag
    #                  disagrees between annotation and prediction
    def _empty_plate_cds():
        return dict(
            x=[], y=[], abs_delta=[], idx=[], label=[],
            drug=[], concentration=[],
            ann_total=[], pred_total=[], dt=[],
            ann_bad=[], pred_bad=[], dd=[],
            ann_valid=[], pred_valid=[],
            ann_orient=[], pred_orient=[],
            color_value=[], line_color=[],
        )
    cds_plate_p1 = bokeh.models.ColumnDataSource(data=_empty_plate_cds())
    cds_plate_p2 = bokeh.models.ColumnDataSource(data=_empty_plate_cds())

    # One shared colour mapper for both plate heatmaps + the colorbar.
    # GREEN = good (Δ=0), RED = bad. NaN → light gray (count comparison
    # not meaningful, typically because ann.valid=False).
    #
    # Palette is hand-coded green→red so there's zero ambiguity about
    # which end is which — `reversed(bokeh.palettes.RdYlGn11)` worked but
    # was easy to second-guess.
    GREEN_TO_RED = [
        '#1a9850',  # dark green     (Δ ≈ 0,   perfect agreement)
        '#66bd63',
        '#a6d96a',
        '#d9ef8b',
        '#fee08b',
        '#fdae61',
        '#f46d43',
        '#d73027',
        '#a50026',  # dark red       (Δ ≈ high, worst disagreement)
    ]
    shared_cmap = bokeh.models.LinearColorMapper(
        palette=GREEN_TO_RED,
        low=0, high=5,
        high_color='#67000d',   # off-scale (>= high) → even darker red
        nan_color='#eee',
    )

    for fig, src in ((plate_p1_fig, cds_plate_p1), (plate_p2_fig, cds_plate_p2)):
        fig.rect(x='x', y='y', source=src, width=0.92, height=0.92,
                 fill_color={'field': 'color_value', 'transform': shared_cmap},
                 line_color='line_color', line_width=2)
        # Hover tooltip — show full annotation + prediction so the user can
        # see WHY a cell is red without leaving the heatmap.
        hover = fig.select(dict(type=bokeh.models.HoverTool))
        if hover:
            hover[0].tooltips = [
                ('Well',          '@label'),
                ('Drug',          '@drug'),
                ('Concentration', '@concentration'),
                ('Δ total',       '@dt{+0}'),
                ('Δ defective',   '@dd{+0}'),
                ('Total somites', 'ann @ann_total · pred @pred_total'),
                ('Defective',     'ann @ann_bad · pred @pred_bad'),
                ('Valid',         'ann @ann_valid · pred @pred_valid'),
                ('Orientation',   'ann @ann_orient · pred @pred_orient'),
            ]

    # Colorbar uses the same mapper so domain changes auto-propagate.
    color_bar = bokeh.models.ColorBar(color_mapper=shared_cmap, width=12,
                                       location=(0, 0), title='|Δ|')
    plate_p1_fig.add_layout(color_bar, 'right')

    # ----- Per-well disagreement table -----
    table_source = bokeh.models.ColumnDataSource(data=dict(
        idx=[], plate=[], well=[],
        ann_total=[], pred_total=[], dt=[],
        ann_bad=[], pred_bad=[], dd=[],
        ann_valid=[], pred_valid=[],
        ann_orient=[], pred_orient=[],
        abs_dt=[],
    ))
    table = bokeh.models.DataTable(
        source=table_source,
        columns=[
            bokeh.models.TableColumn(field='plate',       title='P',     width=30),
            bokeh.models.TableColumn(field='well',        title='Well',  width=60),
            bokeh.models.TableColumn(field='ann_total',   title='Ann T', width=60),
            bokeh.models.TableColumn(field='pred_total',  title='Pred T', width=60),
            bokeh.models.TableColumn(field='dt',          title='Δ T',   width=50),
            bokeh.models.TableColumn(field='ann_bad',     title='Ann D', width=60),
            bokeh.models.TableColumn(field='pred_bad',    title='Pred D', width=60),
            bokeh.models.TableColumn(field='dd',          title='Δ D',   width=50),
            bokeh.models.TableColumn(field='ann_valid',   title='Ann V', width=55),
            bokeh.models.TableColumn(field='pred_valid',  title='Pred V', width=55),
            bokeh.models.TableColumn(field='ann_orient',  title='Ann O', width=55),
            bokeh.models.TableColumn(field='pred_orient', title='Pred O', width=55),
        ],
        width=900, height=320, index_position=None, sortable=True,
        selectable=True,
    )

    # ----- Image preview pane -----
    IMG_W = 520
    img_size = 2048
    img_yfp_fig = bokeh.plotting.figure(
        width=IMG_W, height=IMG_W, x_range=(0, img_size), y_range=(0, img_size),
        tools='pan,wheel_zoom,box_zoom,reset,save', toolbar_location='above')
    img_bf_fig = bokeh.plotting.figure(
        width=IMG_W, height=IMG_W, x_range=(0, img_size), y_range=(0, img_size),
        tools='pan,wheel_zoom,box_zoom,reset,save', toolbar_location='above')
    for f in (img_yfp_fig, img_bf_fig):
        f.axis.visible = False
        f.grid.visible = False

    src_img_yfp = bokeh.models.ColumnDataSource(data=dict(image=[], dw=[], dh=[]))
    src_img_bf  = bokeh.models.ColumnDataSource(data=dict(image=[], dw=[], dh=[]))
    img_yfp_fig.image(image='image', x=0, y=0, dw='dw', dh='dh', source=src_img_yfp,
                       palette='Greys256')
    img_bf_fig.image(image='image', x=0, y=0, dw='dw', dh='dh', source=src_img_bf,
                      palette='Greys256')

    img_tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.TabPanel(child=img_yfp_fig, title='YFP'),
        bokeh.models.TabPanel(child=img_bf_fig,  title='BF'),
    ])
    img_caption = bokeh.models.Div(
        text='<i style="color:#666">Click a heatmap cell or a table row '
             'to preview the well image.</i>',
        width=IMG_W)

    # ----- Shared state -----
    state = {'records': []}   # list of dicts, one per (well with both ann + pred)

    # ----- Helpers -----
    def _subset_filter(qs, mode):
        """Apply the subset radio's choice to an annotation queryset."""
        if mode == 0:   # test + unflagged
            return qs.filter(
                Q(use_for_test=True) |
                (Q(use_for_training=False) &
                 Q(use_for_validation=False) &
                 Q(use_for_test=False)))
        if mode == 1:   # test only
            return qs.filter(use_for_test=True)
        if mode == 2:   # unflagged only
            return qs.filter(use_for_training=False,
                             use_for_validation=False,
                             use_for_test=False)
        return qs       # mode 3: all

    def _valid_filter(qs, mode):
        if mode == 1:
            return qs.filter(valid=True)
        if mode == 2:
            return qs.filter(valid=False)
        return qs

    def _confmat(records):
        """Confusion matrix for the valid flag (annotation = ground truth)."""
        tp = tn = fp = fn = 0
        for r in records:
            a, p = bool(r['ann_valid']), bool(r['pred_valid'])
            if a and p: tp += 1
            elif (not a) and (not p): tn += 1
            elif (not a) and p: fp += 1
            else: fn += 1
        return tp, tn, fp, fn

    def _color_for_record(r):
        """Compute (color_value, line_color) for one record based on the
        currently selected `color_by_select` metric.

        - color_value is NaN when ann.valid=False — those cells render in
          the mapper's `nan_color` (gray) because comparing somite counts
          on an invalid fish to the annotator's leftover defaults is
          meaningless.
        - line_color is red when the valid flag disagrees between
          annotation and prediction, white otherwise. This way a gray
          cell with a red border immediately reads as "annotator said
          invalid, but model predicts valid" (or vice versa) without
          conflating it with the count disagreement signal.
        """
        mode = COLOR_BY_OPTIONS[color_by_select.value]
        if not r['ann_valid']:
            cv = float('nan')
        elif mode == 'total':
            cv = float(abs(r['dt'])) if r['dt'] is not None else float('nan')
        elif mode == 'def':
            cv = float(abs(r['dd'])) if r['dd'] is not None else float('nan')
        else:   # 'max'
            parts = []
            if r['dt'] is not None: parts.append(abs(r['dt']))
            if r['dd'] is not None: parts.append(abs(r['dd']))
            cv = float(max(parts)) if parts else float('nan')
        lc = '#c00' if (bool(r['ann_valid']) != bool(r['pred_valid'])) else 'white'
        return cv, lc

    def _apply_cmap_for_mode():
        """Tighten the colour-mapper domain based on the selected metric."""
        mode = COLOR_BY_OPTIONS[color_by_select.value]
        # Defective counts are typically 0–3; total can be 0–5+. Tighter
        # `high` saturation is more informative for defective alone.
        if mode == 'def':
            shared_cmap.high = 3
            color_bar.title = '|Δ def|'
        elif mode == 'total':
            shared_cmap.high = 5
            color_bar.title = '|Δ total|'
        else:
            shared_cmap.high = 5
            color_bar.title = 'max |Δ|'

    # ----- Main update -----
    def do_update():
        status_div.text = 'Fetching…'
        exp = exp_select.value
        model_name = model_select.value
        plate_choice = plate_select.value

        # Annotations in the chosen experiment + subset + validity. Also
        # prefetch the source well and its drugs so we can show what was
        # in each well on the heatmap hover and the image caption without
        # an N+1 query.
        ann_qs = DestWellProperties.objects.filter(
            dest_well__well_plate__experiment__name=exp,
        ).select_related(
            'dest_well__well_plate',
            'dest_well__source_well',
        ).prefetch_related('dest_well__source_well__drugs')
        if plate_choice in ('1', '2'):
            ann_qs = ann_qs.filter(dest_well__well_plate__plate_number=int(plate_choice))
        ann_qs = _subset_filter(ann_qs, subset_radio.active)
        ann_qs = _valid_filter(ann_qs, valid_radio.active)

        # Build records: only wells where a prediction with the chosen
        # model_name also exists. We use latest_prediction() so the schema
        # change is opaque here.
        # Note: `.iterator()` requires an explicit `chunk_size` when
        # `prefetch_related()` is in play (Django ≥4), otherwise it errors
        # to prevent prefetch caches blowing memory on big tables. 2000 is
        # comfortable for our scale (single-experiment, <1000 wells).
        from well_mapping.models import latest_prediction as _latest
        records = []
        for ann in ann_qs.iterator(chunk_size=2000):
            pred = _latest(ann.dest_well, model_name=model_name)
            if pred is None:
                continue
            dest = ann.dest_well
            plate = dest.well_plate.plate_number
            col_str = f'{int(dest.position_col):02d}'
            well = f'{dest.position_row}{col_str}'

            # Drug(s) dosed in this well: read through source_well → drugs.
            # Cocktails / multi-drug wells get a comma-separated label.
            drug_names, drug_concs = [], []
            src = dest.source_well
            if src is not None:
                for drg in src.drugs.all():
                    drug_names.append(drg.derivation_name or '—')
                    drug_concs.append('—' if drg.concentration in (None, -9999)
                                       else f'{drg.concentration:g}')
            drug_label = ', '.join(drug_names) if drug_names else '—'
            conc_label = ', '.join(drug_concs) if drug_concs else '—'

            ann_t = ann.n_total_somites if ann.n_total_somites != -9999 else None
            pred_t = pred.n_total_somites if pred.n_total_somites != -9999 else None
            ann_d = ann.n_bad_somites if ann.n_bad_somites != -9999 else None
            pred_d = pred.n_bad_somites if pred.n_bad_somites != -9999 else None
            dt = (pred_t - ann_t) if (ann_t is not None and pred_t is not None) else None
            dd = (pred_d - ann_d) if (ann_d is not None and pred_d is not None) else None
            records.append({
                'plate': plate,
                'pos_row': dest.position_row,
                'pos_col': int(dest.position_col),
                'well': well,
                'drug': drug_label,
                'concentration': conc_label,
                'ann_total': ann_t, 'pred_total': pred_t, 'dt': dt,
                'ann_bad':   ann_d, 'pred_bad':   pred_d, 'dd': dd,
                'ann_valid':  bool(ann.valid),
                'pred_valid': bool(pred.valid),
                'ann_orient':  bool(ann.correct_orientation),
                'pred_orient': bool(pred.correct_orientation),
                'abs_dt': abs(dt) if dt is not None else float('nan'),
            })

        # Sort by |Δ total| descending so worst cases are on top
        records.sort(key=lambda r: (-(r['abs_dt']
                                       if not np.isnan(r['abs_dt']) else -1)))
        state['records'] = records

        # ---- Aggregate metrics ----
        if records:
            t_arr  = np.array([(r['ann_total'], r['pred_total']) for r in records
                               if r['dt'] is not None], dtype=float)
            d_arr  = np.array([(r['ann_bad'],   r['pred_bad'])   for r in records
                               if r['dd'] is not None], dtype=float)

            def _scoring_block(arr, label):
                if arr.size == 0:
                    return f'<b>{label}</b>: no data<br>'
                a, p = arr[:, 0], arr[:, 1]
                diff = p - a
                mae = float(np.abs(diff).mean())
                rmse = float(np.sqrt((diff ** 2).mean()))
                bias = float(diff.mean())
                if a.var() > 0:
                    r2 = 1.0 - float(((a - p) ** 2).sum()) / float(((a - a.mean()) ** 2).sum())
                else:
                    r2 = float('nan')
                return (f'<b>{label}</b> &mdash; N={len(arr)}, '
                        f'MAE={mae:.2f}, RMSE={rmse:.2f}, '
                        f'bias={bias:+.2f}, R²={r2:.2f}<br>')

            tp, tn, fp, fn = _confmat(records)
            n_v = tp + tn + fp + fn
            acc = (tp + tn) / max(n_v, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)

            n_orient = sum(1 for r in records)
            agree_orient = sum(1 for r in records
                                if r['ann_orient'] == r['pred_orient'])
            orient_pct = (agree_orient / n_orient * 100) if n_orient else 0

            html = f'<b>Experiment:</b> {exp} · model = <code>{model_name}</code> · N = {len(records)}<br>'
            html += _scoring_block(t_arr, 'Total somites')
            html += _scoring_block(d_arr, 'Defective somites')
            html += (f'<b>Valid flag</b> &mdash; '
                     f'TP={tp} TN={tn} FP={fp} FN={fn} · '
                     f'Acc={acc*100:.1f}% Prec={prec*100:.1f}% '
                     f'Rec={rec*100:.1f}% F1={f1*100:.1f}%<br>')
            html += (f'<b>Orientation</b> &mdash; agreement '
                     f'{orient_pct:.1f}% ({agree_orient}/{n_orient})')
            metrics_div.text = html
        else:
            metrics_div.text = (f'<b>Experiment:</b> {exp} · '
                                f'<i>No wells matched.</i> Try widening the subset '
                                f'or validity filter.')

        # ---- Plate heatmaps ----
        _apply_cmap_for_mode()
        for plate_num, src in ((1, cds_plate_p1), (2, cds_plate_p2)):
            buckets = _empty_plate_cds()
            for i, r in enumerate(records):
                if r['plate'] != plate_num:
                    continue
                if str(r['pos_col']) not in X_LABELS:
                    continue
                if r['pos_row'] not in Y_LABELS:
                    continue
                buckets['x'].append(str(r['pos_col']))
                buckets['y'].append(r['pos_row'])
                buckets['abs_delta'].append(r['abs_dt'])
                buckets['idx'].append(i)
                buckets['label'].append(f"P{plate_num} {r['well']}")
                buckets['drug'].append(r['drug'])
                buckets['concentration'].append(r['concentration'])
                # Use blank string instead of None so hover renders cleanly.
                buckets['ann_total'].append(
                    r['ann_total'] if r['ann_total'] is not None else '—')
                buckets['pred_total'].append(
                    r['pred_total'] if r['pred_total'] is not None else '—')
                buckets['dt'].append(r['dt'] if r['dt'] is not None else 0)
                buckets['ann_bad'].append(
                    r['ann_bad'] if r['ann_bad'] is not None else '—')
                buckets['pred_bad'].append(
                    r['pred_bad'] if r['pred_bad'] is not None else '—')
                buckets['dd'].append(r['dd'] if r['dd'] is not None else 0)
                buckets['ann_valid'].append('Y' if r['ann_valid'] else 'N')
                buckets['pred_valid'].append('Y' if r['pred_valid'] else 'N')
                buckets['ann_orient'].append('Y' if r['ann_orient'] else 'N')
                buckets['pred_orient'].append('Y' if r['pred_orient'] else 'N')
                cv, lc = _color_for_record(r)
                buckets['color_value'].append(cv)
                buckets['line_color'].append(lc)
            src.data = buckets
            (plate_p1_fig if plate_num == 1 else plate_p2_fig).visible = (
                plate_choice in ('Both', str(plate_num)))

        # ---- Table ----
        table_source.data = dict(
            idx=list(range(len(records))),
            plate=[r['plate'] for r in records],
            well=[r['well'] for r in records],
            ann_total=[r['ann_total'] for r in records],
            pred_total=[r['pred_total'] for r in records],
            dt=[r['dt'] for r in records],
            ann_bad=[r['ann_bad'] for r in records],
            pred_bad=[r['pred_bad'] for r in records],
            dd=[r['dd'] for r in records],
            ann_valid=[r['ann_valid'] for r in records],
            pred_valid=[r['pred_valid'] for r in records],
            ann_orient=[r['ann_orient'] for r in records],
            pred_orient=[r['pred_orient'] for r in records],
            abs_dt=[r['abs_dt'] for r in records],
        )

        status_div.text = f'Loaded {len(records)} well(s) with both annotation and prediction.'

    def _recolor_heatmaps():
        """Light-weight refresh — recompute only the colour fields of the
        existing plate CDS, without re-running database queries. Triggered
        when the user changes the 'Heatmap colour metric' dropdown."""
        _apply_cmap_for_mode()
        for plate_num, src in ((1, cds_plate_p1), (2, cds_plate_p2)):
            d = dict(src.data)
            if not d.get('idx'):
                continue
            new_cv, new_lc = [], []
            for k in d['idx']:
                k = int(k)
                if 0 <= k < len(state['records']):
                    cv, lc = _color_for_record(state['records'][k])
                else:
                    cv, lc = float('nan'), 'white'
                new_cv.append(cv); new_lc.append(lc)
            d['color_value'] = new_cv
            d['line_color']  = new_lc
            src.data = d

    update_button.on_click(do_update)
    color_by_select.on_change('value',
                                lambda attr, old, new: _recolor_heatmaps())

    # ----- Image loading on selection -----
    def _load_image_for_record(record):
        """Read the canonicalised YFP + BF images for `record['well']`.

        On failure (no LOCALPATH, no well folder, no canonicalised images)
        the caption explicitly shows what was tried so the user can
        diagnose. Common causes: experiment folder lives somewhere we don't
        probe, or `manage.py refresh_orientation` hasn't been run yet so
        there's no `corrected_orientation/` subfolder."""
        exp = exp_select.value
        plate_n = record['plate']
        row = record['pos_row']
        col = record['pos_col']
        pad = f'{col:02d}'

        # ----- 1. Locate the experiment root -----
        tried_roots = []
        localpath = None
        for cand in (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH):
            full = os.path.join(cand, exp)
            tried_roots.append(full)
            if os.path.isdir(full):
                localpath = cand
                break
        if localpath is None:
            src_img_yfp.data = dict(image=[], dw=[], dh=[])
            src_img_bf.data  = dict(image=[], dw=[], dh=[])
            img_caption.text = (
                f'<b style="color:#c00">Experiment folder not found.</b><br>'
                f'<small>Tried these paths:<br><code>'
                + '<br>'.join(tried_roots)
                + '</code></small>'
            )
            return

        # ----- 2. Locate the well's canonicalised image folder -----
        well_dir = os.path.join(localpath, exp, 'Leica images',
                                f'Plate {plate_n}', f'Well_{row}{pad}',
                                'corrected_orientation')

        if not os.path.isdir(well_dir):
            src_img_yfp.data = dict(image=[], dw=[], dh=[])
            src_img_bf.data  = dict(image=[], dw=[], dh=[])
            img_caption.text = (
                f'<b style="color:#c00">No canonicalised image folder for '
                f'P{plate_n} · {record["well"]}.</b><br>'
                f'<small>Expected: <code>{well_dir}</code><br>'
                f'Run <code>python manage.py refresh_orientation</code> '
                f'to generate it.</small>'
            )
            return

        # ----- 3. Load each channel; report which ones came back empty -----
        def _load(channel):
            # Accept .tif / .tiff and either case
            files = (glob.glob(os.path.join(well_dir, f'*{channel}*.tiff'))
                     + glob.glob(os.path.join(well_dir, f'*{channel}*.tif')))
            files = [f for f in files if 'norm' not in os.path.basename(f).lower()]
            if not files:
                return None, f'no *{channel}*.tiff in {well_dir}'
            try:
                arr = imread(files[0]).astype(np.float32)
            except Exception as e:
                return None, f'imread failed for {files[0]}: {e}'
            arr = arr / max(arr.max(), 1e-9)
            return arr, files[0]

        yfp, yfp_info = _load('YFP')
        bf,  bf_info  = _load('BF')

        if yfp is not None:
            h, w = yfp.shape[:2]
            src_img_yfp.data = dict(image=[np.flipud(yfp)], dw=[w], dh=[h])
        else:
            src_img_yfp.data = dict(image=[], dw=[], dh=[])
        if bf is not None:
            h, w = bf.shape[:2]
            src_img_bf.data = dict(image=[np.flipud(bf)], dw=[w], dh=[h])
        else:
            src_img_bf.data = dict(image=[], dw=[], dh=[])

        # ----- 4. Caption -----
        dt = record['dt']
        dd = record['dd']
        dt_str = f'{dt:+d}' if dt is not None else '—'
        dd_str = f'{dd:+d}' if dd is not None else '—'

        info_parts = []
        if yfp is None:
            info_parts.append(
                f'<span style="color:#c00">YFP not loaded ({yfp_info})</span>')
        else:
            info_parts.append(
                f'<small style="color:#888">YFP: {os.path.basename(yfp_info)}</small>')
        if bf is None:
            info_parts.append(
                f'<span style="color:#c00">BF not loaded ({bf_info})</span>')
        else:
            info_parts.append(
                f'<small style="color:#888">BF: {os.path.basename(bf_info)}</small>')

        img_caption.text = (
            f'<b>Plate {plate_n} · {record["well"]}</b> '
            f'&nbsp;·&nbsp; <span style="color:#2e4070">'
            f'{record.get("drug", "—")} @ {record.get("concentration", "—")}'
            f'</span><br>'
            f'<small>Annotated: total={record["ann_total"]} '
            f'def={record["ann_bad"]} valid={record["ann_valid"]} '
            f'orient={record["ann_orient"]}</small><br>'
            f'<small>Predicted: total={record["pred_total"]} '
            f'def={record["pred_bad"]} valid={record["pred_valid"]} '
            f'orient={record["pred_orient"]}</small><br>'
            f'<small>Δ total = <b>{dt_str}</b>, Δ defective = <b>{dd_str}</b></small>'
            f'<br>{"<br>".join(info_parts)}'
        )

    def _on_table_select(attr, old, new):
        if not new:
            return
        try:
            i = int(new[0])
        except (TypeError, ValueError, IndexError):
            return
        if 0 <= i < len(state['records']):
            _load_image_for_record(state['records'][i])

    table_source.selected.on_change('indices', _on_table_select)

    def _on_plate_select(attr, old, new, src):
        # The TapTool already updates src.selected.indices when a rect is
        # tapped — much more robust than parsing event.x/event.y because
        # Bokeh maps FactorRange factors to numeric positions internally.
        if not new:
            return
        try:
            k = int(new[0])
        except (TypeError, ValueError, IndexError):
            return
        idx_list = src.data.get('idx', [])
        if k >= len(idx_list):
            return
        rec_idx = int(idx_list[k])
        if 0 <= rec_idx < len(state['records']):
            # Mirror the selection in the disagreement table for visual
            # consistency, then load the image.
            table_source.selected.indices = [rec_idx]
            _load_image_for_record(state['records'][rec_idx])

    cds_plate_p1.selected.on_change(
        'indices',
        lambda attr, old, new: _on_plate_select(attr, old, new, cds_plate_p1))
    cds_plate_p2.selected.on_change(
        'indices',
        lambda attr, old, new: _on_plate_select(attr, old, new, cds_plate_p2))

    # ----- Layout -----
    def _section(text):
        return bokeh.models.Div(
            text=(f'<div style="font-size:13px; font-weight:700; color:#1a2340;'
                  f' border-bottom:2px solid #5b8dee;'
                  f' padding:6px 4px; margin:4px 0 8px;">{text}</div>'))

    filter_col = bokeh.layouts.column(
        _section('Filters'),
        exp_select, model_select, plate_select,
        bokeh.models.Div(text='<b>Subset:</b>'),
        subset_radio,
        bokeh.models.Div(text='<b>Validity:</b>'),
        valid_radio,
        update_button, status_div,
        _section('Heatmap'),
        color_by_select,
        bokeh.models.Div(text=(
            '<small style="color:#666">Gray cells = ann.valid=False '
            '(count comparison not meaningful).<br>'
            'Red border = valid-flag disagreement between annotation and '
            'prediction.</small>'), width=320),
        width=340,
    )

    plate_col = bokeh.layouts.column(
        _section('Spatial disagreement'),
        plate_p1_fig, plate_p2_fig,
    )
    image_col = bokeh.layouts.column(
        _section('Selected well'),
        img_caption, img_tabs,
    )

    main_col = bokeh.layouts.column(
        _section('Aggregate metrics'),
        metrics_div,
        bokeh.layouts.row(plate_col, bokeh.layouts.Spacer(width=20), image_col),
        _section('Per-well disagreement (sorted by |Δ total|)'),
        table,
    )

    doc.add_root(bokeh.layouts.row(filter_col, main_col))

    # Auto-run once with default selection.
    do_update()


#___________________________________________________________________________________________
def model_eval_page(request: HttpRequest) -> HttpResponse:
    """Serve the model-evaluation dashboard (Bokeh embed)."""
    script = bokeh.embed.server_document(request.build_absolute_uri())
    return render(request, 'well_explorer/model_eval.html', {'script': script})


#___________________________________________________________________________________________
# Profile-based somite dashboard
#
# Alternative to SAM: scan the canonicalised YFP image in y, build per-strip
# intensity profiles, find peaks, cluster them across strips. Gives:
#   - per-somite x position
#   - per-somite asymmetry (upper-half confidence vs lower-half confidence)
#   - body length
# Per-somite bounding boxes are also stored so future work can train a
# dedicated defect-severity classifier on the cropped tiles.
PROFILE_MODEL_NAME = 'profile_v1'


def profile_handler(doc: bokeh.document.Document) -> None:
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    from somiteCounting._common import preprocess_image
    from somiteCounting.profile_analysis import (
        DEFAULTS as PA_DEFAULTS, analyze_image,
    )

    all_exp = sorted([e.name for e in Experiment.objects.all()])
    if not all_exp:
        doc.add_root(bokeh.models.Div(text='<b>No experiments in the database.</b>'))
        return

    # ----- Selectors + algorithm knobs -----
    exp_select   = bokeh.models.Select(title='Experiment', value=all_exp[0],
                                        options=all_exp, width=320)
    plate_select = bokeh.models.Select(title='Plate', value='1',
                                        options=['1', '2'], width=120)
    well_select  = bokeh.models.Select(title='Well', value='Select well',
                                        options=['Select well'], width=160)

    prominence_slider = bokeh.models.Slider(
        title='Peak prominence', value=PA_DEFAULTS['peak_prominence'],
        start=0.01, end=0.30, step=0.01, width=320)
    distance_slider = bokeh.models.Slider(
        title='Min peak distance (px)', value=PA_DEFAULTS['peak_distance'],
        start=5, end=80, step=1, width=320)
    n_strips_slider = bokeh.models.Slider(
        title='Number of y-strips', value=PA_DEFAULTS['n_strips'],
        start=4, end=60, step=2, width=320)
    sigma_slider = bokeh.models.Slider(
        title='Smoothing σ (px)', value=PA_DEFAULTS['smoothing_sigma'],
        start=0.0, end=15.0, step=0.5, width=320)

    analyze_button = bokeh.models.Button(label='Analyse',
                                          button_type='primary', width=200)
    save_button   = bokeh.models.Button(label='Save', button_type='success',
                                         width=120)
    status_div    = bokeh.models.Div(text='', width=520,
                                      styles={'color': '#666',
                                              'font-style': 'italic'})
    summary_div   = bokeh.models.Div(text='', width=820,
                                      styles={'font-size': '13px',
                                              'line-height': '1.6'})

    # ----- Image figure -----
    img_size = 2048
    img_fig = bokeh.plotting.figure(
        title='YFP image', width=720, height=480,
        x_range=(0, img_size), y_range=(0, img_size),
        tools='pan,wheel_zoom,box_zoom,reset,save',
        match_aspect=True)
    img_fig.axis.visible = False
    img_fig.grid.visible = False

    img_src = bokeh.models.ColumnDataSource(data=dict(image=[], dw=[], dh=[]))
    img_fig.image(image='image', x=0, y=0, dw='dw', dh='dh', source=img_src,
                   palette='Greys256')

    # Spine ROI horizontal band shown via two horizontal lines
    roi_top    = bokeh.models.Span(location=0, dimension='width',
                                    line_color='#5b8dee', line_dash='dashed',
                                    line_width=1)
    roi_bottom = bokeh.models.Span(location=0, dimension='width',
                                    line_color='#5b8dee', line_dash='dashed',
                                    line_width=1)
    img_fig.add_layout(roi_top)
    img_fig.add_layout(roi_bottom)

    # Vertical line at every detected somite; colour by severity. Drawn via
    # a `segment` glyph so we can reflect severity per-cell.
    SEV_COLORS = ['#1a9850', '#fdae61', '#f46d43', '#a50026']
    SEV_LABELS = ['healthy', 'dim', 'asymmetric', 'weak detection']
    src_lines = bokeh.models.ColumnDataSource(
        data=dict(x0=[], y0=[], x1=[], y1=[], color=[], idx=[]))
    img_fig.segment(x0='x0', y0='y0', x1='x1', y1='y1',
                     source=src_lines, color='color', line_width=3)

    # Per-somite ROI bounding boxes (thin overlay)
    src_bboxes = bokeh.models.ColumnDataSource(
        data=dict(left=[], right=[], top=[], bottom=[]))
    img_fig.quad(left='left', right='right', top='top', bottom='bottom',
                  source=src_bboxes, fill_alpha=0, line_color='#888',
                  line_width=1, line_dash='dotted')

    # Spine centerline overlay (the polynomial fit) — shows the user the
    # curve we're following before straightening.
    src_centerline = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))
    img_fig.line('x', 'y', source=src_centerline,
                  color='#5b8dee', line_width=1.5, line_dash='dashed')

    # ----- Straightened-image figure (second view) -----
    img_straight_fig = bokeh.plotting.figure(
        title='Straightened image (spine forced horizontal)',
        width=720, height=260,
        x_range=img_fig.x_range,            # ← share x-axis with the original
        y_range=(0, img_size),
        tools='pan,wheel_zoom,box_zoom,reset,save')
    img_straight_fig.axis.visible = False
    img_straight_fig.grid.visible = False
    src_img_straight = bokeh.models.ColumnDataSource(
        data=dict(image=[], dw=[], dh=[]))
    img_straight_fig.image(image='image', x=0, y=0, dw='dw', dh='dh',
                            source=src_img_straight, palette='Greys256')

    # ----- Profile (1-D) figure, x-axis LINKED to image -----
    profile_fig = bokeh.plotting.figure(
        title='Mean intensity profile (smoothed)', width=720, height=240,
        x_range=img_fig.x_range,     # ← shared with image
        tools='pan,wheel_zoom,box_zoom,reset,save')
    profile_fig.xaxis.axis_label = 'x (AP axis)'
    profile_fig.yaxis.axis_label = 'mean intensity'
    src_profile = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))
    profile_fig.line('x', 'y', source=src_profile,
                      color='#2196F3', line_width=2)
    src_peaks = bokeh.models.ColumnDataSource(
        data=dict(x=[], y=[], color=[], conf=[], idx=[]))
    profile_fig.scatter('x', 'y', source=src_peaks, size=10,
                         fill_color='color', line_color='white')

    # ----- Kymograph (per-strip profiles as a 2-D heatmap) -----
    # Each row is one y-strip's smoothed intensity profile. Chevron-shaped
    # somites show as V-shaped bright streaks here — the kymograph is the
    # ground-truth view of the data the algorithm is actually working on.
    kymo_fig = bokeh.plotting.figure(
        title='Per-strip profile kymograph (y-strip vs x)',
        width=720, height=200,
        x_range=img_fig.x_range,             # share x with image+profile
        y_range=(0, 1),                       # placeholder; reset on analyse
        tools='pan,wheel_zoom,box_zoom,reset,save,hover')
    kymo_fig.xaxis.axis_label = 'x (AP axis)'
    kymo_fig.yaxis.axis_label = 'strip #  (top↑)'
    src_kymo = bokeh.models.ColumnDataSource(
        data=dict(image=[], dw=[], dh=[]))
    kymo_color_mapper = bokeh.models.LinearColorMapper(
        palette='Viridis256', low=0, high=1)
    kymo_fig.image(image='image', x=0, y=0, dw='dw', dh='dh',
                    source=src_kymo, color_mapper=kymo_color_mapper)
    # Detected peaks overlaid as scatter — column index = x, row index = strip
    src_kymo_peaks = bokeh.models.ColumnDataSource(
        data=dict(x=[], y=[]))
    kymo_fig.scatter('x', 'y', source=src_kymo_peaks, size=4,
                      fill_color='#ff3030', line_color='white', line_width=0.5)
    hover = kymo_fig.select(dict(type=bokeh.models.HoverTool))
    if hover:
        hover[0].tooltips = [('x', '$x{0}'), ('strip', '$y{0}'),
                              ('intensity', '@image{0.000}')]

    # ----- Per-somite table -----
    table_src = bokeh.models.ColumnDataSource(data=dict(
        idx=[], centroid_x=[], confidence=[],
        upper_confidence=[], lower_confidence=[],
        intensity=[], severity=[], severity_reason=[],
        ap_position=[],
    ))
    table = bokeh.models.DataTable(
        source=table_src,
        columns=[
            bokeh.models.TableColumn(field='idx',              title='#',         width=35),
            bokeh.models.TableColumn(field='centroid_x',       title='x',         width=60,
                formatter=bokeh.models.NumberFormatter(format='0')),
            bokeh.models.TableColumn(field='ap_position',      title='AP',        width=55,
                formatter=bokeh.models.NumberFormatter(format='0.00')),
            bokeh.models.TableColumn(field='confidence',       title='Conf',      width=60,
                formatter=bokeh.models.NumberFormatter(format='0.00')),
            bokeh.models.TableColumn(field='upper_confidence', title='Conf ↑',    width=60,
                formatter=bokeh.models.NumberFormatter(format='0.00')),
            bokeh.models.TableColumn(field='lower_confidence', title='Conf ↓',    width=60,
                formatter=bokeh.models.NumberFormatter(format='0.00')),
            bokeh.models.TableColumn(field='intensity',        title='I',         width=55,
                formatter=bokeh.models.NumberFormatter(format='0.00')),
            bokeh.models.TableColumn(field='severity',         title='Sev',       width=40),
            bokeh.models.TableColumn(field='severity_reason',  title='Notes',     width=260),
        ],
        width=820, height=320, index_position=None, selectable=True,
    )

    # ----- Shared state -----
    state = {
        'image_np':  None,         # 2-D float [0,1], canonicalised YFP
        'dest':      None,
        'result':    None,
    }

    def _set_status(msg, color='#666'):
        status_div.text = (f'<span style="color:{color}">{msg}</span>'
                            if color != '#666' else msg)

    # ----- Well dropdown auto-populate -----
    def _refresh_wells():
        exp_name = exp_select.value
        plate_n  = plate_select.value
        try:
            plate = DestWellPlate.objects.get(experiment__name=exp_name,
                                              plate_number=int(plate_n))
        except DestWellPlate.DoesNotExist:
            well_select.options = ['Select well']
            well_select.value = 'Select well'
            return
        wells = []
        for d in DestWellPosition.objects.filter(well_plate=plate):
            try:
                col = int(d.position_col)
            except (TypeError, ValueError):
                continue
            wells.append(f'{d.position_row}{col:02d}')
        wells = sorted(set(wells))
        well_select.options = ['Select well'] + wells
        well_select.value = 'Select well'

    exp_select.on_change('value',   lambda a, o, n: _refresh_wells())
    plate_select.on_change('value', lambda a, o, n: _refresh_wells())
    _refresh_wells()

    # ----- Image loading -----
    def _load_well_image():
        exp = exp_select.value
        plate_n = plate_select.value
        well_str = well_select.value
        if well_str == 'Select well':
            return
        row = well_str[0]
        try:
            col = int(well_str[1:])
        except ValueError:
            return
        pad = f'{col:02d}'
        try:
            plate = DestWellPlate.objects.get(experiment__name=exp,
                                               plate_number=int(plate_n))
            dest = DestWellPosition.objects.get(well_plate=plate,
                                                 position_row=row,
                                                 position_col=col)
        except (DestWellPlate.DoesNotExist, DestWellPosition.DoesNotExist):
            _set_status('Well not in DB.', '#c00')
            return
        state['dest'] = dest

        localpath = None
        for cand in (LOCALPATH_RAID5, LOCALPATH_HIVE, LOCALPATH_CH):
            if os.path.isdir(os.path.join(cand, exp)):
                localpath = cand
                break
        if localpath is None:
            _set_status(f'No LOCALPATH found for {exp}.', '#c00')
            return
        well_dir = os.path.join(localpath, exp, 'Leica images',
                                f'Plate {plate_n}', f'Well_{row}{pad}',
                                'corrected_orientation')
        files = glob.glob(os.path.join(well_dir, '*YFP*.tiff'))
        files = [f for f in files if 'norm' not in os.path.basename(f).lower()]
        if not files:
            _set_status(f'No YFP image in {well_dir}.', '#c00')
            return
        try:
            raw = imread(files[0]).astype(np.float32)
        except Exception as e:
            _set_status(f'imread failed: {e}', '#c00')
            return

        # Normalise via the same percentile-clip used everywhere else.
        # preprocess_image returns a tensor (1,H,W); take .numpy()[0].
        normed = preprocess_image(raw, resize=raw.shape[:2]).numpy()[0]

        h, w = normed.shape
        img_src.data = dict(image=[np.flipud(normed)], dw=[w], dh=[h])
        img_fig.x_range.start = 0; img_fig.x_range.end = w
        img_fig.y_range.start = 0; img_fig.y_range.end = h
        state['image_np'] = normed
        # Clear previous analysis overlays
        src_lines.data   = dict(x0=[], y0=[], x1=[], y1=[], color=[], idx=[])
        src_bboxes.data  = dict(left=[], right=[], top=[], bottom=[])
        src_centerline.data = dict(x=[], y=[])
        src_img_straight.data = dict(image=[], dw=[], dh=[])
        src_profile.data = dict(x=[], y=[])
        src_peaks.data   = dict(x=[], y=[], color=[], conf=[], idx=[])
        src_kymo.data    = dict(image=[], dw=[], dh=[])
        src_kymo_peaks.data = dict(x=[], y=[])
        table_src.data   = dict(idx=[], centroid_x=[], confidence=[],
                                upper_confidence=[], lower_confidence=[],
                                intensity=[], severity=[],
                                severity_reason=[], ap_position=[])
        summary_div.text = ''
        _set_status(
            f'Loaded {os.path.basename(files[0])} ({w}×{h}). '
            f'Press <b>Analyse</b>.')

    well_select.on_change('value', lambda a, o, n: _load_well_image())

    # ----- Analyse -----
    def _do_analyze():
        if state['image_np'] is None:
            _set_status('Load a well image first.', '#c00')
            return
        img = state['image_np']
        h, w = img.shape
        _set_status('Analysing…')
        result = analyze_image(
            img,
            n_strips=int(n_strips_slider.value),
            smoothing_sigma=float(sigma_slider.value),
            peak_prominence=float(prominence_slider.value),
            peak_distance=int(distance_slider.value),
        )
        state['result'] = result

        # Spine ROI band markers (image is flipped vertically for Bokeh,
        # so we draw at h - y_center ± dy)
        y_lo_disp = h - (result['spine_y_center'] + result['spine_dy'])
        y_hi_disp = h - (result['spine_y_center'] - result['spine_dy'])
        roi_top.location    = y_hi_disp
        roi_bottom.location = y_lo_disp

        # Spine centerline (the polynomial fit) shown on the ORIGINAL image.
        # Flip y for display.
        y_spine_orig = result.get('y_spine_original')
        if y_spine_orig is not None:
            src_centerline.data = dict(
                x=list(range(len(y_spine_orig))),
                y=(h - y_spine_orig).tolist(),
            )

        # Straightened image — second figure underneath the original
        straight = result.get('straightened_image')
        if straight is not None:
            sh, sw = straight.shape
            src_img_straight.data = dict(image=[np.flipud(straight)],
                                          dw=[sw], dh=[sh])
            img_straight_fig.y_range.start = 0
            img_straight_fig.y_range.end = sh

        # Kymograph — per-strip profiles stacked as a 2-D heatmap
        kym = result.get('kymograph')
        if kym is not None and kym.size > 0:
            ks, kw = kym.shape
            # Auto-scale colour mapper to this image's intensity range
            kymo_color_mapper.low  = float(kym.min())
            kymo_color_mapper.high = float(kym.max())
            src_kymo.data = dict(image=[kym], dw=[kw], dh=[ks])
            kymo_fig.y_range.start = 0
            kymo_fig.y_range.end   = ks

        # Profile
        prof = result['mean_profile']
        src_profile.data = dict(x=list(range(len(prof))), y=prof.tolist())

        somites = result['somites']

        # Peak markers on the profile (one circle per somite, coloured by severity)
        src_peaks.data = dict(
            x=[s['centroid_x'] for s in somites],
            y=[float(prof[int(s['centroid_x'])]) if 0 <= s['centroid_x'] < len(prof) else 0
               for s in somites],
            color=[SEV_COLORS[s['severity']] for s in somites],
            conf=[s['confidence'] for s in somites],
            idx=[s['index'] for s in somites],
        )

        # Vertical lines on the image, one per somite, severity-coloured
        src_lines.data = dict(
            x0=[s['centroid_x'] for s in somites],
            x1=[s['centroid_x'] for s in somites],
            y0=[max(0, h - (result['spine_y_center'] + result['spine_dy']) - 8)
                for _ in somites],
            y1=[min(h, h - (result['spine_y_center'] - result['spine_dy']) + 8)
                for _ in somites],
            color=[SEV_COLORS[s['severity']] for s in somites],
            idx=[s['index'] for s in somites],
        )

        # Per-somite bounding boxes (dotted gray; only on demand later if it
        # clutters the view, the user can switch off the glyph)
        src_bboxes.data = dict(
            left  =[s['bbox'][0] for s in somites],
            right =[s['bbox'][2] for s in somites],
            top   =[h - s['bbox'][1] for s in somites],     # flip for display
            bottom=[h - s['bbox'][3] for s in somites],
        )

        # Table
        table_src.data = dict(
            idx=             [s['index']             for s in somites],
            centroid_x=      [s['centroid_x']        for s in somites],
            confidence=      [s['confidence']        for s in somites],
            upper_confidence=[s['upper_confidence']  for s in somites],
            lower_confidence=[s['lower_confidence']  for s in somites],
            intensity=       [s['intensity']         for s in somites],
            severity=        [s['severity']          for s in somites],
            severity_reason= [s['severity_reason']   for s in somites],
            ap_position=     [s['ap_position']       for s in somites],
        )

        # Summary
        n = len(somites)
        n_def = sum(1 for s in somites if s['severity'] > 0)
        sev_counts = [sum(1 for s in somites if s['severity'] == k) for k in range(4)]
        body_len_px = result['body_length']
        sev_html = ' · '.join(
            f'<span style="color:{SEV_COLORS[k]}">■</span> {SEV_LABELS[k]}: {sev_counts[k]}'
            for k in range(4))
        summary_div.text = (
            f'<b>{n}</b> somite(s) detected · body length ≈ <b>{body_len_px:.0f} px</b> '
            f'· <b>{n_def}</b> non-healthy<br>'
            f'<small>{sev_html}</small>'
        )
        _set_status('Done. Press <b>Save</b> to write to '
                    f'<code>DestWellPropertiesPredicted</code> '
                    f'(model_name=<code>{PROFILE_MODEL_NAME}</code>).')

    analyze_button.on_click(_do_analyze)

    # ----- Save -----
    def _do_save():
        dest = state['dest']
        result = state['result']
        if dest is None or result is None:
            _set_status('Nothing to save — analyse first.', '#c00')
            return
        somites = result['somites']
        n_total = len(somites)
        n_bad   = sum(1 for s in somites if s['severity'] > 0)

        DestWellPropertiesPredicted.objects.update_or_create(
            dest_well=dest,
            model_name=PROFILE_MODEL_NAME,
            model_version='',
            defaults={
                'n_total_somites': n_total,
                'n_bad_somites':   n_bad,
                'per_somite_data': {
                    'somites':     somites,
                    'body_length': result['body_length'],
                    'algorithm_params': dict(
                        n_strips=int(n_strips_slider.value),
                        peak_prominence=float(prominence_slider.value),
                        peak_distance=int(distance_slider.value),
                        smoothing_sigma=float(sigma_slider.value),
                    ),
                },
            },
        )
        _set_status(
            f'Saved {n_total} somite(s) under '
            f'<code>model_name="{PROFILE_MODEL_NAME}"</code>.', '#1a9850')

    save_button.on_click(_do_save)

    # ----- Layout -----
    def _section(text):
        return bokeh.models.Div(
            text=(f'<div style="font-size:13px; font-weight:700; color:#1a2340;'
                  f' border-bottom:2px solid #5b8dee;'
                  f' padding:6px 4px; margin:4px 0 8px;">{text}</div>'))

    legend_div = bokeh.models.Div(
        text=('<small>Severity legend: '
              + ' &nbsp;·&nbsp; '.join(
                  f'<span style="color:{SEV_COLORS[k]}">■</span> {SEV_LABELS[k]}'
                  for k in range(4))
              + '</small>'),
        width=820)

    left = bokeh.layouts.column(
        _section('Selection'),
        exp_select, plate_select, well_select,
        _section('Algorithm knobs'),
        prominence_slider, distance_slider, n_strips_slider, sigma_slider,
        bokeh.layouts.row(analyze_button, save_button),
        status_div,
        width=360,
    )
    right = bokeh.layouts.column(
        _section('Image (original) + spine centerline'),
        img_fig,
        _section('Image (straightened)'),
        img_straight_fig,
        _section('Mean profile + detected peaks'),
        profile_fig,
        _section('Kymograph (per-strip profile · chevron pattern)'),
        kymo_fig,
        _section('Per-somite detections'),
        summary_div,
        legend_div,
        table,
    )
    doc.add_root(bokeh.layouts.row(left, right))


#___________________________________________________________________________________________
def profile_dashboard(request: HttpRequest) -> HttpResponse:
    """Serve the profile-based somite dashboard (Bokeh embed)."""
    script = bokeh.embed.server_document(request.build_absolute_uri())
    return render(request, 'well_explorer/profile_dashboard.html', {'script': script})