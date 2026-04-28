from django.shortcuts import render

from django.db import reset_queries
from django.db import connection
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
                cds_labels_dest.data = dict(source_labels_96.data, size=[50*nzoom_wells]*len(source_labels_96.data['x']) if cds_labels_dest.data['size']==[] else cds_labels_dest.data['size'])
                plot_wellplate_dest.axis.visible = True

            elif dest_well_plate.plate_type == '48':
                plot_wellplate_dest.x_range.factors = x_48
                plot_wellplate_dest.y_range.factors = y_48
                plot_wellplate_dest.title.text = "48 well plate"
                cds_labels_dest.data = dict(source_labels_48.data, size=[65*nzoom_wells]*len(source_labels_48.data['x']) if cds_labels_dest.data['size']==[] else cds_labels_dest.data['size'])
                plot_wellplate_dest.axis.visible = True

            elif dest_well_plate.plate_type == '24':
                plot_wellplate_dest.x_range.factors = x_24
                plot_wellplate_dest.y_range.factors = y_24
                plot_wellplate_dest.title.text = "24 well plate"
                cds_labels_dest.data = dict(source_labels_24.data, size=[80*nzoom_wells]*len(source_labels_24.data['x']) if cds_labels_dest.data['size']==[] else cds_labels_dest.data['size'])
                plot_wellplate_dest.axis.visible = True

        if n_plates==2:
            dest_well_plate_2 = dest_well_plates[1]
            if dest_well_plate_2.plate_type == '96':
                plot_wellplate_dest_2.x_range.factors = x_96
                plot_wellplate_dest_2.y_range.factors = y_96
                plot_wellplate_dest_2.title.text = "96 well plate"
                cds_labels_dest_2.data = dict(source_labels_96.data, size=[50*nzoom_wells]*len(source_labels_96.data['x']) if cds_labels_dest_2.data['size']==[] else cds_labels_dest_2.data['size'])
                plot_wellplate_dest_2.axis.visible = True

            elif dest_well_plate_2.plate_type == '48':
                plot_wellplate_dest_2.x_range.factors = x_48
                plot_wellplate_dest_2.y_range.factors = y_48
                plot_wellplate_dest_2.title.text = "48 well plate"
                cds_labels_dest_2.data = dict(source_labels_48.data, size=[65*nzoom_wells]*len(source_labels_48.data['x']) if cds_labels_dest_2.data['size']==[] else cds_labels_dest_2.data['size'])
                plot_wellplate_dest_2.axis.visible = True

            elif dest_well_plate_2.plate_type == '24':
                plot_wellplate_dest_2.x_range.factors = x_24
                plot_wellplate_dest_2.y_range.factors = y_24
                plot_wellplate_dest_2.title.text = "24 well plate"
                cds_labels_dest_2.data = dict(source_labels_24.data, size=[80*nzoom_wells]*len(source_labels_24.data['x']) if cds_labels_dest_2.data['size']==[] else cds_labels_dest_2.data['size'])
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
    source_radio = bokeh.models.RadioGroup(
        labels=['Predicted only', 'Annotated only', 'Both (overlay)'],
        active=0,
    )
    valid_radio = bokeh.models.RadioGroup(
        labels=['Valid fish only', 'Invalid fish only', 'All fish'],
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

    # --- Update callback ---
    def do_update():
        drug         = drug_select.value
        selected     = exp_multi.value
        show_pred    = source_radio.active in (0, 2)
        show_ann     = source_radio.active in (1, 2)
        valid_map    = {0: True, 1: False, 2: None}
        valid_filter = valid_map[valid_radio.active]
        ann_filter   = {0: None, 1: 'training', 2: 'validation'}[ann_subset_radio.active]

        if not selected:
            for s in (src_total_pred, src_total_ann, src_bad_pred, src_bad_ann):
                s.data = dict(top=[], left=[], right=[])
            for s in SUBSET_COLORS:
                src_scatter_total[s].data = dict(x=[], y=[])
                src_scatter_bad[s].data = dict(x=[], y=[])
            status_div.text = 'No experiments selected.'
            stats_div.text = ''
            return

        status_div.text = 'Fetching data…'

        pred_total, pred_bad = [], []
        if show_pred:
            # NOTE: when SAM lands and produces 'sam_v1' rows, this query
            # will need a 'which model?' dropdown (Phase 3c). For now we
            # always read from the original ResNet predictions.
            qs = DestWellPropertiesPredicted.objects.filter(
                model_name=RESNET_MODEL_NAME,
                dest_well__source_well__isnull=False,
                dest_well__source_well__drugs__derivation_name=drug,
                dest_well__well_plate__experiment__name__in=selected,
            ).distinct()
            if valid_filter is not None:
                qs = qs.filter(valid=valid_filter)
            for d in qs:
                if d.n_total_somites != -9999:
                    pred_total.append(d.n_total_somites)
                if d.n_bad_somites != -9999:
                    pred_bad.append(d.n_bad_somites)

        ann_total, ann_bad = [], []
        if show_ann:
            qs = DestWellProperties.objects.filter(
                dest_well__source_well__isnull=False,
                dest_well__source_well__drugs__derivation_name=drug,
                dest_well__well_plate__experiment__name__in=selected,
            ).distinct()
            if valid_filter is not None:
                qs = qs.filter(valid=valid_filter)
            if ann_filter == 'training':
                qs = qs.filter(use_for_training=True)
            elif ann_filter == 'validation':
                qs = qs.filter(use_for_validation=True)
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
            dest_well__source_well__isnull=False,
            dest_well__source_well__drugs__derivation_name=drug,
            dest_well__well_plate__experiment__name__in=selected,
        ).select_related('dest_well').distinct()
        if valid_filter is not None:
            qs_pair = qs_pair.filter(valid=valid_filter)

        for ann in qs_pair:
            pred_obj = latest_prediction(ann.dest_well, model_name=RESNET_MODEL_NAME)
            if pred_obj is None:
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

        valid_label = {0: 'valid', 1: 'invalid', 2: 'all'}[valid_radio.active]
        p_total.title.text = f'Total somites — {drug} ({valid_label} fish)'
        p_bad.title.text   = f'Defective somites — {drug} ({valid_label} fish)'
        p_scatter_total.title.text = f'Predicted vs annotated — total somites — {drug}'
        p_scatter_bad.title.text   = f'Predicted vs annotated — defective somites — {drug}'

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
        bokeh.models.Div(text='<h4 style="margin:0">Distributions (histograms)</h4>'),
        bokeh.layouts.row(p_total, p_bad),
        bokeh.models.Div(text='<h4 style="margin:14px 0 0">Predicted vs annotated (paired wells)</h4>'),
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