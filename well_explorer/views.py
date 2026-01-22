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
from PIL import ImageFont
from PIL import ImageDraw
import random

import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts

from well_mapping.models import Experiment, SourceWellPlate, DestWellPlate, SourceWellPosition, DestWellPosition, Drug, DestWellProperties

from somiteCounting.training import SomiteCounter_freeze, FishQualityClassifier
from somiteCounting.training_orientation import OrientationClassifier
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
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


model_fish = FishQualityClassifier().to(device)
checkpoint_path_fish=r"C:\Users\helsens\software\VAST-DS\somiteCounting\checkpoints\fish_quality_best.pth"
checkpoint_fish = torch.load(checkpoint_path_fish, map_location=device)
model_fish.load_state_dict(checkpoint_fish["model_state_dict"])
model_fish.eval()

model_orientation = OrientationClassifier().to(device)
checkpoint_path_ori=r"C:\Users\helsens\software\VAST-DS\somiteCounting\checkpoints\orientation_best.pth"
checkpoint_orientation = torch.load(checkpoint_path_ori, map_location=device)
model_orientation.load_state_dict(checkpoint_orientation["model_state_dict"])
model_orientation.eval()




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

    cds_labels_dest_filled    = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_filled   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_dest_filled_bad    = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_filled_bad   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    drug_message    = bokeh.models.Div(visible=False)
    image_message    = bokeh.models.Div(visible=False)
    prediction_message    = bokeh.models.Div(visible=False)

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
            source_img_vast.data = {'img':[]}
            drug_message.text = ""
            drug_message.visible = False
            image_message.text = "<b style='color:red; font-size:18px;'>No images found for selected well {}</b>".format(position[0][1] + position[0][0])
            image_message.visible = True
            prediction_message.visible = False

            return


        image_bf  = imread(file_BF)
        source_img_bf.data  = {'img':[np.flip(image_bf,0)]}

        image_yfp = imread(file_YFP)
        source_img_yfp.data = {'img':[np.flip(image_yfp,0)]}

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

        LOCALPATH = LOCALPATH_HIVE
        if os.path.exists(os.path.join(LOCALPATH_RAID5, dropdown_exp.value)):
            LOCALPATH = LOCALPATH_RAID5

        path_plate_1_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 1', 'Well_*')
        path_plate_2_leica = os.path.join(LOCALPATH, dropdown_exp.value,'Leica images', 'Plate 2', 'Well_*')
        path_plate_1_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 1', 'Well_*')
        path_plate_2_vast = os.path.join(LOCALPATH, dropdown_exp.value,'VAST images', 'Plate 2', 'Well_*')        

        print('path_plate_1_leica=', path_plate_1_leica)
        print('path_plate_2_leica=', path_plate_2_leica)
        print('path_plate_1_vast=', path_plate_1_vast)
        print('path_plate_2_vast=', path_plate_2_vast)

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
                img_raw, img_tensor = load_and_prepare_image(file_BF)
                img_tensor = img_tensor.to(device)
                # Prediction
                with torch.no_grad():
                    logit_ori = model_orientation(img_tensor.to(device))
                    prob_ori = torch.sigmoid(logit_ori).item()  # scalar
        pred_total, pred_def = pred
        prediction_message.text = "<b style='color:blue; font-size:18px;'>Predicting Total {:.1f}  --  defective {:.1f}  --  Valid Fish {}  --  Prob orientation {:.2} </b>".format(pred_total,pred_def, 'Yes' if prob>0.5 else 'No', prob_ori)
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
    def create_training_callback():
        print('------------------->>>>>>>>> create_training_callback')
        experiments = Experiment.objects.all()

        if os.path.exists(LOCALPATH_TRAINING):
            shutil.rmtree(LOCALPATH_TRAINING)
        if os.path.exists(LOCALPATH_TRAINING) is False:
            os.mkdir(LOCALPATH_TRAINING)
        if os.path.exists(os.path.join(LOCALPATH_TRAINING,'train')) is False:
            os.mkdir(os.path.join(LOCALPATH_TRAINING,'train'))
        if os.path.exists(os.path.join(LOCALPATH_TRAINING,'valid')) is False:
            os.mkdir(os.path.join(LOCALPATH_TRAINING,'valid'))
        

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
                        #if props.valid and props.n_total_somites>=0 and props.n_bad_somites >=0:
                        #Add false to train other model
                        if props.n_total_somites>=0 and props.n_bad_somites >=0:
                            rand=random.uniform(0,1)
                            if rand>0.2: outdir=os.path.join(LOCALPATH_TRAINING,'train')
                            else: outdir=os.path.join(LOCALPATH_TRAINING,'valid')

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
    plot_img_bf   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,pan,reset,undo",width=550, height=550)
    plot_img_bf.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_bf, color_mapper=color_mapper)

    data_img_yfp   = {'img':[]}
    source_img_yfp = bokeh.models.ColumnDataSource(data=data_img_yfp)
    plot_img_yfp   = bokeh.plotting.figure(x_range=x_range, y_range=y_range, tools="box_select,wheel_zoom,box_zoom,pan,reset,undo",width=550, height=550)
    plot_img_yfp.image(image='img', x=0, y=0, dw=im_size, dh=im_size, source=source_img_yfp, color_mapper=color_mapper)

    data_img_vast   = {'img':[]}
    source_img_vast = bokeh.models.ColumnDataSource(data=data_img_vast)
    x_range_2 = bokeh.models.Range1d(start=0, end=1024)
    y_range_2 = bokeh.models.Range1d(start=0, end=200*4)
    plot_img_vast   = bokeh.plotting.figure(x_range=x_range_2, y_range=y_range_2, tools="box_select,wheel_zoom,box_zoom,reset,undo",width=1110, height=217*4)
    #plot_img_vast   = bokeh.plotting.figure(tools="box_select,wheel_zoom,box_zoom,reset,undo",width=1024, height=200)
    plot_img_vast.image_rgba(image='img', x=0, y=0, dw=1024, dh=200*4, source=source_img_vast)




    indent = bokeh.models.Spacer(width=30)

    norm_layout = bokeh.layouts.column(bokeh.layouts.row(indent,bokeh.layouts.column(dropdown_exp, well_mapping_button, create_training_button), bokeh.models.Spacer(width=20),    bokeh.layouts.column(image_message,drug_message)),
                                       bokeh.layouts.Spacer(width=50),
                                       bokeh.layouts.row(indent,  bokeh.layouts.column(plot_wellplate_dest, plot_wellplate_dest_2),
                                                         bokeh.layouts.column(bokeh.layouts.row(bokeh.layouts.Spacer(width=10), bokeh.layouts.column(contrast_slider,predict_button, use_corrected_checkbox), dropdown_total_somites, dropdown_total_somites_err, dropdown_bad_somites, dropdown_bad_somites_err, dropdown_good_image, dropdown_good_orientation, saveimages_button,images_comments),
                                                                              bokeh.layouts.row(prediction_message),
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
    n_wells_total  = 0
    n_fish_total = 0
    for exp in experiments:

        data.append({'name': exp.name, 'date_created': exp.date, 'description': exp.description})
        n_fish_valid = 0
        n_fish_notvalid = 0
        n_fish = 0
        n_wells = 0


        dest_well_plates = DestWellPlate.objects.filter(experiment=exp)
        for plate in dest_well_plates:
            dest_well_positions = DestWellPosition.objects.filter(well_plate=plate)

            for dest in dest_well_positions:
                try:
                    props = dest.dest_well_properties  # reverse OneToOne accessor
                    n_wells += 1
                    if props.valid:
                        n_fish_valid +=1
                    else:
                        n_fish_notvalid +=1
                except DestWellProperties.DoesNotExist:
                    pass

        data[-1]['n_wells'] = n_wells
        data[-1]['n_fish_valid'] = n_fish_valid
        data[-1]['n_fish_notvalid'] = n_fish_notvalid
        data[-1]['fraction_valid'] = (n_fish_valid / (n_fish_valid + n_fish_notvalid)) * 100 if (n_fish_valid + n_fish_notvalid) > 0 else 0
        n_fish = n_fish_valid + n_fish_notvalid
        data[-1]['n_fish'] = n_fish
        n_fish_valid_total += n_fish_valid
        n_fish_notvalid_total += n_fish_notvalid
        n_fish_total += n_fish
        n_wells_total += n_wells
    data_total = {'name': 'TOTAL', 'date_created': '', 'description': '', 'n_wells': n_wells_total, 'n_fish_valid': n_fish_valid_total, 'n_fish_notvalid': n_fish_notvalid_total, 'fraction_valid': (n_fish_valid_total / (n_fish_valid_total + n_fish_notvalid_total)) * 100 if (n_fish_valid_total + n_fish_notvalid_total) > 0 else 0,  'n_fish': n_fish_total}
    return render(request, 'well_explorer/experiment_listing.html', {'rows': data, 'data_total': data_total})