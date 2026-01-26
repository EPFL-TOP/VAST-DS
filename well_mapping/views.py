from django.shortcuts import render
from django.db import reset_queries
from django.db import connection
from django.http import HttpRequest, HttpResponse
from django.contrib.auth.decorators import login_required, permission_required
from django.db.models import Q

import bokeh.models
import bokeh.palettes
import bokeh.plotting
import bokeh.embed
import bokeh.layouts
import bokeh.io

from slims.slims import Slims
from slims.util import display_results
import slims.criteria as slims_cr

import os, sys, json, glob
from datetime import date

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from well_mapping.models import Experiment, SourceWellPlate, DestWellPlate, SourceWellPosition, DestWellPosition, Drug
import accesskeys as accessk

from requests.auth import HTTPBasicAuth
import requests
username_pyrat = accessk.PYRAT_username
password_pyrat = accessk.PYRAT_password
auth = HTTPBasicAuth(username_pyrat, password_pyrat)
print(auth)

base_url = 'https://sv-pyrat-aquatic.epfl.ch/pyrat-aquatic/api/v3/'
api_url_tanks     = base_url +'tanks'
api_url_crossings = base_url + 'tanks/crossings'
api_url_strains   = base_url + 'strains'

_programmatic_change = False
global NZOOM_WELLS_SOURCE
global NZOOM_WELLS_DEST

NZOOM_WELLS_SOURCE = 1
NZOOM_WELLS_DEST   = 1
#___________________________________________________________________________________________
def vast_handler(doc: bokeh.document.Document) -> None:
    print('****************************  vast_handler ****************************')
    #TO BE CHANGED WITH ASYNC?????
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

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

    source_filled_well = bokeh.models.ColumnDataSource(data={'x':[], 'y':[]})

    plot_wellplate_source = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x_96), y_range=bokeh.models.FactorRange(*y_96), 
                                                  title='',width=900, height=600, tools="box_select,box_zoom,reset,undo")
    plot_wellplate_source.xaxis.major_label_text_font_size = "15pt"
    plot_wellplate_source.yaxis.major_label_text_font_size = "15pt"
    plot_wellplate_source.grid.visible = False
    plot_wellplate_source.axis.visible = False

    plot_wellplate_source_supp = bokeh.plotting.figure(x_range=bokeh.models.FactorRange(*x_96), y_range=bokeh.models.FactorRange(*y_96), 
                                                       title='',width=900, height=200, tools="box_select,box_zoom,reset,undo")
    plot_wellplate_source_supp.xaxis.major_label_text_font_size = "15pt"
    plot_wellplate_source_supp.yaxis.major_label_text_font_size = "15pt"
    plot_wellplate_source_supp.grid.visible = False
    plot_wellplate_source_supp.axis.visible = False



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

    cds_labels_source = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_source_supp = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_source_drug = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[], drug=[]))
    cds_labels_source_supp_drug = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[], drug=[]))


    cds_labels_dest_1_drug = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_drug = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_dest   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2 = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_dest_mapping   = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_mapping = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))

    cds_labels_dest_1_drug_control = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))
    cds_labels_dest_2_drug_control = bokeh.models.ColumnDataSource(data=dict(x=[], y=[], size=[]))


    well_plates = ['Select well plate', '96-wells', '48-wells', '24-wells']
    dropdown_well_plate_source  = bokeh.models.Select(value='Select well plate', title='Source Well Plate', options=well_plates)
    dropdown_well_plate_dest    = bokeh.models.Select(value='Select well plate', title='Destination Well Plate', options=well_plates)

    delete_experiment_button = bokeh.models.Button(label="Delete experiment", button_type="danger")

    n_dest_wellplates = [1, 2]
    dropdown_n_dest_wellplates = bokeh.models.Select(value='1', title='N dest WP', options=[str(i) for i in n_dest_wellplates])

    n_supp_sourcewells = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    dropdown_n_supp_sourcewell = bokeh.models.Select(value='0', title='N supp wells', options=[str(i) for i in n_supp_sourcewells])


    experiment_name    = bokeh.models.TextInput(title="Experiment name:", value='')#does not work, css_classes=["custom-input"])
    experiment_message = bokeh.models.Div(visible=False)
    
    pyrat_id  = bokeh.models.TextInput(title="Pyrat Xid (use -9999 if no Xid):", value='', width=150)
    pyrat_message = bokeh.models.Div(visible=False)

    experiment_date        = bokeh.models.widgets.DatePicker(title="Select a date:", value=date.today(), min_date=date(2000, 1, 1), max_date=date.today())
    experiment_description = bokeh.models.widgets.TextAreaInput(title="Enter a description:", value='', rows=7, width=550, css_classes=["font-size:18px"])

    experiments = ['Select experiment']
    for exp in Experiment.objects.all():
        experiments.append(exp.name)

    experiments=sorted(experiments)
    dropdown_exp  = bokeh.models.Select(value='Select experiment', title='Experiment', options=experiments)

    add_drug_button = bokeh.models.Button(label="Add drug",  button_type="success")
    remove_drug_button = bokeh.models.Button(label="Remove drug",  button_type="danger")
    map_drug_button = bokeh.models.Button(label="Map drug",  button_type="success")
    unmap_drug_button = bokeh.models.Button(label="Unmap drug",  button_type="danger")
    add_drug_other_wells_button = bokeh.models.Button(label="Add drug to other wells",  button_type="success")
    force_add_drug_button = bokeh.models.Button(label="Force add drug",  button_type="success")
    valid_wellcluster_button = bokeh.models.Button(label="Enter comment/Valid",  button_type="success")
    drug_message    = bokeh.models.Div(visible=False)
    mapping_message = bokeh.models.Div(visible=False)   
    wellvalid_message = bokeh.models.Div(visible=False)   

    slimsid_name        = bokeh.models.TextInput(title="Slims ID: (eg: OA_DS_00024)", value='' )
    drug_concentration  = bokeh.models.TextInput(title="Concentration (uM) or Percentage (%)", value='', width=200)
    valid_wellcluster   = bokeh.models.Select(value='True', title='Valid well cluster', options=['True','False'])
    wellcluster_comment = bokeh.models.widgets.TextAreaInput(title="Comment:", value='', rows=7, width=300, css_classes=["font-size:18px"])

    lines_source = bokeh.models.ColumnDataSource(data=dict(x_start=[], y_start=[], x_end=[], y_end=[]))
    p_lines = bokeh.plotting.figure(width=1500, height=500, match_aspect=True, tools="", toolbar_location=None)
    p_lines.segment(x0='x_start', y0='y_start', x1='x_end', y1='y_end', source=lines_source, line_color="red", line_width=1)
    p_lines.grid.visible = False
    p_lines.axis.visible = False


    slims = Slims(name="slims", url=accessk.end_point, username=accessk.user_name, password=accessk.password)

    plot_wellplate_source.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_source, 
                                 line_color='blue', fill_color="white",
                                 selection_fill_color="orange",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.9,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="white",
                                 nonselection_line_color="blue",)
    
    plot_wellplate_source.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_source_drug, 
                                 fill_alpha=0.5,line_width=3,
                                 line_color='black', fill_color="black",
                                 selection_fill_color="red",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.7,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="black",
                                 nonselection_line_color="black",)

    labels = bokeh.models.LabelSet(x = 'x',y = 'y', text = 'drug',
                                   source = cds_labels_source_drug,
                                    x_offset = 0, y_offset = 0,
                                    text_align = 'center',
                                    text_baseline = 'middle',
                                    text_font_size = '12px',
                                    text_color = 'navy'
    )

    labels_supp = bokeh.models.LabelSet(x = 'x',y = 'y', text = 'drug',
                                   source = cds_labels_source_supp_drug,
                                    x_offset = 0, y_offset = 0,
                                    text_align = 'center',
                                    text_baseline = 'middle',
                                    text_font_size = '12px',
                                    text_color = 'navy'
    )

    plot_wellplate_source.add_layout(labels)

    hover_grid = bokeh.modelsHoverTool(tooltips=[("ID", "@id"), ("(X,Y)", "(@x{0.1f}, @y{0.1f})")])
    hover_grid.mode = 'mouse' 
    hover_grid.point_policy = 'snap_to_data'
    plot_wellplate_source.add_tools(hover_grid)


    plot_wellplate_source_supp.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_source_supp, 
                                 line_color='blue', fill_color="white",
                                 selection_fill_color="orange",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.9,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="white",
                                 nonselection_line_color="blue",)
    
    plot_wellplate_source_supp.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_source_supp_drug, 
                                 fill_alpha=0.5,line_width=3,
                                 line_color='black', fill_color="black",
                                 selection_fill_color="red",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.7,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="black",
                                 nonselection_line_color="black",)

    plot_wellplate_source_supp.add_layout(labels_supp)

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
                                 source=cds_labels_dest_1_drug, 
                                 fill_alpha=0.5,line_width=3,
                                 line_color='black', fill_color="black",
                                 selection_fill_color="red",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.7,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="black",
                                 nonselection_line_color="black",)

    plot_wellplate_dest.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_dest_1_drug_control, 
                                 fill_alpha=0.2,line_width=3,
                                 line_color='black', fill_color="black",
                                 selection_fill_color="red",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.5,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="black",
                                 nonselection_line_color="black",)


    plot_wellplate_dest_2.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_dest_2_drug, 
                                 fill_alpha=0.5,line_width=3,
                                 line_color='black', fill_color="black",
                                 selection_fill_color="red",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.9,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="black",
                                 nonselection_line_color="black",)
    
    plot_wellplate_dest_2.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_dest_2_drug_control, 
                                 fill_alpha=0.2,line_width=3,
                                 line_color='black', fill_color="black",
                                 selection_fill_color="red",    # when selected
                                 selection_line_color="firebrick",
                                 selection_fill_alpha=0.9,
                                 nonselection_fill_alpha=0.0,      # style for non-selected
                                 nonselection_fill_color="black",
                                 nonselection_line_color="black",)
    
    plot_wellplate_dest.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_dest_mapping, 
                                 fill_alpha=0.5,line_width=3,
                                 line_color='black', fill_color="yellow",
                                 selection_fill_alpha=0.7,
                                 nonselection_fill_alpha=0.0)

    plot_wellplate_dest_2.circle('x', 'y', 
                                 size='size',
                                 source=cds_labels_dest_2_mapping, 
                                 fill_alpha=0.5,line_width=3,
                                 line_color='black', fill_color="yellow",
                                 selection_fill_alpha=0.7,
                                 nonselection_fill_alpha=0.0)

    #___________________________________________________________________________________________
    def get_well_mapping(indices, issupp=False, issource=True):
        print('------------------->>>>>>>>> get_well_mapping')
        if issource:
            n_well = len(cds_labels_source.data['x'])
        else:
            n_well = len(cds_labels_dest.data['x'])

        positions = []
        print('get_well_mapping indices=',indices)
        print('n_well=',n_well)
        if n_well == 96 and not issupp:
            i=0
            for xi in x_96:
                for yi in y_96:
                    if i in indices:
                        positions.append(('{}'.format(xi),'{}'.format(yi)))
                    i+=1
        elif n_well == 48 and not issupp:
            i=0
            for xi in x_48:
                for yi in y_48:
                    if i in indices:
                        positions.append(('{}'.format(xi),'{}'.format(yi)))
                    i+=1
        elif n_well == 24 and not issupp:
            i=0
            for xi in x_24:
                for yi in y_24:
                    if i in indices:
                        positions.append(('{}'.format(xi),'{}'.format(yi)))
                    i+=1
        elif issupp:
            for idx in indices:
                positions.append((cds_labels_source_supp.data['x'][idx], cds_labels_source_supp.data['y'][idx]))
        print('positions=', positions)
        return positions
    
    #___________________________________________________________________________________________
    def add_source_well(attr, old, new):
        print('------------------->>>>>>>>> add_source_well')
        n_well_supp = int(new)

        x_supp=[str(i+1) for i in range(n_well_supp)]
        y_supp=['Z' for i in range(n_well_supp)]

        plot_wellplate_source_supp.axis.visible = True
        plot_wellplate_source_supp.title.text = "Supplementary plate"
        cds_labels_source_supp.data = dict(x=x_supp, y=y_supp, size=[50]*len(x_supp))
        plot_wellplate_source_supp.x_range.factors = x_supp
        plot_wellplate_source_supp.y_range.factors = ['Z']

    dropdown_n_supp_sourcewell.on_change("value", add_source_well)


    #___________________________________________________________________________________________
    def modify_experiment():
        print('------------------->>>>>>>>> modify_experiment')
        exp_name = experiment_name.value
        experiment = Experiment.objects.filter(name=exp_name).first()

        print('experiment=',experiment)
        print('experiment.dest_plate.count() ',experiment.dest_plate.count())
        print('dropdown_n_dest_wellplates.value=',dropdown_n_dest_wellplates.value)

        if dropdown_n_dest_wellplates.value == '2' and experiment.dest_plate.count() == 1:
            print('create second dest well plate')
            dest_well_plate_2 = DestWellPlate(plate_type=dropdown_well_plate_dest.value.replace('-wells', ''), experiment=experiment, plate_number=2)
            dest_well_plate_2.save()

            #create dest well positions
            if '96' in dropdown_well_plate_dest.value:
                for xi in x_96:
                    for yi in y_96:
                        pos = DestWellPosition(well_plate=dest_well_plate_2, position_col=xi, position_row=yi)
                        pos.save()
            elif '48' in dropdown_well_plate_dest.value:
                for xi in x_48:
                    for yi in y_48:
                        pos = DestWellPosition(well_plate=dest_well_plate_2, position_col=xi, position_row=yi)
                        pos.save()
            elif '24' in dropdown_well_plate_dest.value:
                for xi in x_24:
                    for yi in y_24:
                        pos = DestWellPosition(well_plate=dest_well_plate_2, position_col=xi, position_row=yi)
                        pos.save()


    modify_experiment_button = bokeh.models.Button(label="Modify experiment", button_type="success")
    modify_experiment_button.on_click(modify_experiment)

    #___________________________________________________________________________________________
    def create_experiment():
        print('------------------->>>>>>>>> create_experiment')
        global _programmatic_change
        exp_name = experiment_name.value
        experiment = Experiment.objects.filter(name=exp_name)
        pyrat_status=add_pyrat_id()
        if experiment.exists():
            experiment_message.text    = f"<b style='color:red; ; font-size:18px;'> Error: The experiment '{exp_name}' already exists.</b>"
            experiment_message.visible = True
            return

        elif exp_name == '':
            experiment_message.text    = f"<b style='color:red; ; font-size:18px;'> Error: Can not define empty experiment name.</b>"
            experiment_message.visible = True
            return

        elif dropdown_well_plate_source.value == 'Select well plate' or dropdown_well_plate_dest.value == 'Select well plate':
            experiment_message.text    = f"<b style='color:red; ; font-size:18px;'> Error: Can not define an experiment without selecting a source and a destination well plate.</b>"
            experiment_message.visible = True            
            return

        elif not pyrat_status:
            return

        else:
            experiments.append(exp_name)
            experiment = Experiment(name=exp_name,
                                    date=experiment_date.value,
                                    description=experiment_description.value.strip(),
                                    pyrat_id=pyrat_id.value.strip())
            experiment.save()

            source_well_plate = SourceWellPlate(plate_type=dropdown_well_plate_source.value.replace('-wells', ''), 
                                                experiment=experiment, 
                                                n_well_supp=len(cds_labels_source_supp.data['x']))
            source_well_plate.save()


            for well in range(len(cds_labels_source_supp.data['x'])):
                pos = SourceWellPosition(well_plate=source_well_plate, 
                                             position_col=cds_labels_source_supp.data['x'][well], 
                                             position_row=cds_labels_source_supp.data['y'][well],
                                             is_supp=True, 
                                             )
                pos.save()

            dest_well_plate = DestWellPlate(plate_type=dropdown_well_plate_dest.value.replace('-wells', ''), experiment=experiment, plate_number=1)
            dest_well_plate.save()

            if dropdown_n_dest_wellplates.value == '2':
                dest_well_plate_2 = DestWellPlate(plate_type=dropdown_well_plate_dest.value.replace('-wells', ''), experiment=experiment, plate_number=2)
                dest_well_plate_2.save()

            #create source well positions 
            if '96' in dropdown_well_plate_source.value:
                for xi in x_96:
                    for yi in y_96:
                        pos = SourceWellPosition(well_plate=source_well_plate, position_col=xi, position_row=yi, is_supp=False)
                        pos.save()
            elif '48' in dropdown_well_plate_source.value:
                for xi in x_48:
                    for yi in y_48:
                        pos = SourceWellPosition(well_plate=source_well_plate, position_col=xi, position_row=yi, is_supp=False)
                        pos.save()
            elif '24' in dropdown_well_plate_source.value:
                for xi in x_24:
                    for yi in y_24:
                        pos = SourceWellPosition(well_plate=source_well_plate, position_col=xi, position_row=yi, is_supp=False)
                        pos.save()
            #create dest well positions
            if '96' in dropdown_well_plate_dest.value:
                for xi in x_96:
                    for yi in y_96:
                        pos = DestWellPosition(well_plate=dest_well_plate, position_col=xi, position_row=yi)
                        pos.save()
                        if dropdown_n_dest_wellplates.value == '2':
                            pos = DestWellPosition(well_plate=dest_well_plate_2, position_col=xi, position_row=yi)
                            pos.save()
            elif '48' in dropdown_well_plate_dest.value:
                for xi in x_48:
                    for yi in y_48:
                        pos = DestWellPosition(well_plate=dest_well_plate, position_col=xi, position_row=yi)
                        pos.save()
                        if dropdown_n_dest_wellplates.value == '2':
                            pos = DestWellPosition(well_plate=dest_well_plate_2, position_col=xi, position_row=yi)
                            pos.save()
            elif '24' in dropdown_well_plate_dest.value:
                for xi in x_24:
                    for yi in y_24:
                        pos = DestWellPosition(well_plate=dest_well_plate, position_col=xi, position_row=yi)
                        pos.save()
                        if dropdown_n_dest_wellplates.value == '2':
                            pos = DestWellPosition(well_plate=dest_well_plate_2, position_col=xi, position_row=yi)
                            pos.save()

            experiments_sorted = sorted(experiments)
            dropdown_exp.options = ["Select experiment"] + experiments_sorted
            dropdown_exp.value = "Select experiment"  # reset selection to first item
            experiment_message.text    = f"<b style='color:green; ; font-size:18px;'> The Experiment '{exp_name}' has been created.</b>"
            experiment_message.visible = True

            _programmatic_change = True
            dropdown_exp.value = exp_name
            _programmatic_change = False

    create_experiment_button = bokeh.models.Button(label="Create experiment", button_type="success")
    create_experiment_button.on_click(create_experiment)

    #___________________________________________________________________________________________
    def add_pyrat_id():
        print('------------------->>>>>>>>> add_pyrat_id')

        if pyrat_id.value == '':
            pyrat_message.text    = f"<b style='color:red; ; font-size:18px;'> ERROR: Please provide a pyrat Xid</b>"
            pyrat_message.visible = True
            return False
        
        if int(pyrat_id.value) == -9999:
            pyrat_message.text    = f"<b style='color:orange; ; font-size:18px;'> WARNING: Using a dummy pyrat Xid</b>"
            pyrat_message.visible = True
            return True
        
        query = {'tk':['strain_id', 'strain_name', 'strain_name_id','strain_name_with_id','mutations','date_of_birth'],'crossing_id':'{}'.format(str(pyrat_id.value))}
        response = requests.get(api_url_crossings, auth=auth, params=query )
        response_json = response.json()
        print('response_json=',response_json)

        if type(response_json)==dict:
            if response_json['detail'] == 'Invalid credentials (not found).':
                pyrat_message.text    = f"<b style='color:red; ; font-size:18px;'> ERROR: Invalid pyrat credentials (not found).</b>"
                pyrat_message.visible = True
                return False
            else:
                pyrat_message.text    = f"<b style='color:red; ; font-size:18px;'> ERROR: Please provide a pyrat Xid</b>"
                pyrat_message.visible = True
                return False
        if len(response_json) == 0:
            pyrat_message.text    = f"<b style='color:red; ; font-size:18px;'> ERROR: The pyrat Xid '{pyrat_id.value}' does not exist.</b>"
            pyrat_message.visible = True
            return False
        
        elif len(response_json) == 1:
            print('response_json[0]=',response_json[0])
            pyrat_message.text  =  f"<b style='color:green; ; font-size:18px;'> PYRAT informations for Xid '{pyrat_id.value}' are</b><br>"

            pyrat_message.text  += "<ul style='margin-top:0;'>"

            pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> date of reccord: {response_json[0]['date_of_record'].split('T')[0]}, {response_json[0]['date_of_record'].split('T')[-1]}</li>"
            pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> strain name with id: {response_json[0]['strain_name_with_id']}</li>"
            for idx, p in enumerate(response_json[0]['tanks']['parents']):
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> parent {idx}</li>"
                pyrat_message.text  += "<ul style='margin-top:0;'>"

                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> tank id {p['tank_id']}</li>"
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> strain name with id {p['strain_name_with_id']}</li>"
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> mutations {p['mutations']}</li>"
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> number of male {p['number_of_male']}</li>"
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> number of female {p['number_of_female']}</li>"
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> number of unknown {p['number_of_unknown']}</li>"
                pyrat_message.text  += f"<li style='color:navy; font-size:14px; margin-bottom:4px;'> date of birth {p['date_of_birth'].split('T')[0]}, {p['date_of_birth'].split('T')[-1]}</li>"
                pyrat_message.text  +="</ul>"
                
            pyrat_message.text  +="</ul>"
            pyrat_message.visible = False
        return True
    



    #___________________________________________________________________________________________
    def check_pyrat():
        add_pyrat_id()
        pyrat_message.visible = True
    check_pyrat_id_button = bokeh.models.Button(label="Check pyrat", button_type="success")
    check_pyrat_id_button.on_click(check_pyrat)

  
    #___________________________________________________________________________________________
    def load_experiment(attr, old, new):
        if _programmatic_change:
            return
        print('------------------->>>>>>>>> load_experiment')

        try:
            experiment  = Experiment.objects.get(name=new)
            experiment_date.value        = experiment.date
            experiment_name.value        = new
            experiment_description.value = experiment.description
            experiment_message.text      = f"<b style='color:green; ; font-size:18px;'> The experiment '{experiment_name.value}' has been loaded.</b>"
            experiment_message.visible   = True
            print('experiment=',experiment)
            dropdown_well_plate_source.value    = experiment.source_plate.plate_type + '-wells'
            print('experiment.dest_plate.all()=',experiment.dest_plate.all())
            print('exp ',exp)
            for idx, plate in enumerate(experiment.dest_plate.all()):
                print(plate, idx)
                if idx==0:dropdown_well_plate_dest.value      = plate.plate_type + '-wells'
                if idx==1:dropdown_n_dest_wellplates.value='2'

            dropdown_n_supp_sourcewell.value = str(experiment.source_plate.n_well_supp)

            #load drugs
            display_drugs_source_wellplate()
            display_drugs_dest_wellplate()

            pyrat_id.value = experiment.pyrat_id

        except Experiment.DoesNotExist:
            experiment_message.text    = f"<b style='color:red; ; font-size:18px;'> ERROR: The experiment '{new}' does not exist.</b>"
            experiment_message.visible = True
            return

        drug_message.visible = False
    dropdown_exp.on_change("value", load_experiment)

    #___________________________________________________________________________________________
    def delete_experiment():
        print('------------------->>>>>>>>> delete_experiment')
        global _programmatic_change
        if dropdown_exp.value == 'Select experiment': 
            experiment_message.text    = f"<b style='color:red; ; font-size:18px;'> Error: No experiment selected.</b>"
            experiment_message.visible = True
            return
        try:
            experiment = Experiment.objects.get(name=dropdown_exp.value)
            experiment.delete()

            experiment_message.text    = f"<b style='color:green; ; font-size:18px;'> The experiment '{dropdown_exp.value}' has been deleted.</b>"
            experiment_message.visible = True
            _programmatic_change = True
            dropdown_exp.options.remove(dropdown_exp.value)
            dropdown_exp.value = 'Select experiment'
            _programmatic_change = False
            experiment_name.value = ''
            cds_labels_dest.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_2.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_source.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_source_supp.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_source_drug.data = {'x':[], 'y':[], 'size':[], 'drug':[]}
            cds_labels_source_supp_drug.data = {'x':[], 'y':[], 'size':[], 'drug':[]}
            cds_labels_dest_1_drug.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_2_drug.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_1_drug_control.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_2_drug_control.data = {'x':[], 'y':[], 'size':[]}
            source_filled_well.data = {'x':[], 'y':[], 'size':[]}
            dropdown_well_plate_source.value = 'Select well plate'
            dropdown_well_plate_dest.value = 'Select well plate'
            dropdown_n_supp_sourcewell.value = '0'  
            dropdown_n_dest_wellplates.value = '1'
            experiment_date.value = date.today()
            experiment_description.value = ''
            pyrat_id.value = ''
            pyrat_message.visible = False
            drug_message.visible = False
            mapping_message.visible = False
            experiment_name.value = ''
        except Experiment.DoesNotExist:
            return
    delete_experiment_button = bokeh.models.Button(label="Delete experiment", button_type="danger")
    delete_experiment_button.on_click(delete_experiment)


    #___________________________________________________________________________________________
    def load_well_plate_source(attr, old, new):
        print('------------------->>>>>>>>> load_well_plate_source')
        if '96' in new:
            plot_wellplate_source.x_range.factors = x_96
            plot_wellplate_source.y_range.factors = y_96
            plot_wellplate_source.title.text = "96 well plate"
            cds_labels_source.data = dict(source_labels_96.data, size=[50*NZOOM_WELLS_SOURCE]*len(source_labels_96.data['x']))
            plot_wellplate_source.axis.visible = True

        elif '48' in new:
            plot_wellplate_source.x_range.factors = x_48
            plot_wellplate_source.y_range.factors = y_48
            plot_wellplate_source.title.text = "48 well plate"
            cds_labels_source.data = dict(source_labels_48.data, size=[65*NZOOM_WELLS_SOURCE]*len(source_labels_48.data['x']))
            plot_wellplate_source.axis.visible = True

        elif '24' in new:
            print('24 well plate')
            plot_wellplate_source.x_range.factors = x_24
            plot_wellplate_source.y_range.factors = y_24
            plot_wellplate_source.title.text = "24 well plate"
            cds_labels_source.data = dict(source_labels_24.data, size=[80*NZOOM_WELLS_SOURCE]*len(source_labels_24.data['x']))
            plot_wellplate_source.axis.visible = True

        else:
            cds_labels_source.data = {'x':[], 'y':[]}
            plot_wellplate_source.title.text = ""
            plot_wellplate_source.axis.visible = False
    dropdown_well_plate_source.on_change("value", load_well_plate_source)


    #___________________________________________________________________________________________
    def load_well_plate_dest(attr, old, new):
        print('------------------->>>>>>>>> load_well_plate_dest')
        if '96' in new:
            plot_wellplate_dest.x_range.factors = x_96
            plot_wellplate_dest.y_range.factors = y_96
            plot_wellplate_dest.title.text = "96 well plate"
            cds_labels_dest.data = dict(source_labels_96.data, size=[50*NZOOM_WELLS_DEST]*len(source_labels_96.data['x']))
            plot_wellplate_dest.axis.visible = True

        elif '48' in new:
            plot_wellplate_dest.x_range.factors = x_48
            plot_wellplate_dest.y_range.factors = y_48
            plot_wellplate_dest.title.text = "48 well plate"
            cds_labels_dest.data = dict(source_labels_48.data, size=[65*NZOOM_WELLS_DEST]*len(source_labels_48.data['x']))
            plot_wellplate_dest.axis.visible = True

        elif '24' in new:
            plot_wellplate_dest.x_range.factors = x_24
            plot_wellplate_dest.y_range.factors = y_24
            plot_wellplate_dest.title.text = "24 well plate"
            cds_labels_dest.data = dict(source_labels_24.data, size=[80*NZOOM_WELLS_DEST]*len(source_labels_24.data['x']))
            plot_wellplate_dest.axis.visible = True

        else:
            cds_labels_dest.data = {'x':[], 'y':[]}
            plot_wellplate_dest.title.text = ""
            plot_wellplate_dest.axis.visible = False

        load_well_plate_dest_2(None, None, None)  # Update second destination plate if needed
    dropdown_well_plate_dest.on_change("value", load_well_plate_dest)

    #___________________________________________________________________________________________
    def load_well_plate_dest_2(attr, old, new):
        print('------------------->>>>>>>>> load_well_plate_dest_2')
        if dropdown_n_dest_wellplates.value == '2':
            if '96' in dropdown_well_plate_dest.value:
                plot_wellplate_dest_2.x_range.factors = x_96
                plot_wellplate_dest_2.y_range.factors = y_96
                plot_wellplate_dest_2.title.text = "96 well plate"
                cds_labels_dest_2.data = dict(source_labels_96.data, size=[50*NZOOM_WELLS_DEST]*len(source_labels_96.data['x']))
                plot_wellplate_dest_2.axis.visible = True

            elif '48' in dropdown_well_plate_dest.value:
                plot_wellplate_dest_2.x_range.factors = x_48
                plot_wellplate_dest_2.y_range.factors = y_48
                plot_wellplate_dest_2.title.text = "48 well plate"
                cds_labels_dest_2.data = dict(source_labels_48.data, size=[65*NZOOM_WELLS_DEST]*len(source_labels_48.data['x']))
                plot_wellplate_dest_2.axis.visible = True

            elif '24' in dropdown_well_plate_dest.value:
                plot_wellplate_dest_2.x_range.factors = x_24
                plot_wellplate_dest_2.y_range.factors = y_24
                plot_wellplate_dest_2.title.text = "24 well plate"
                cds_labels_dest_2.data = dict(source_labels_24.data, size=[80*NZOOM_WELLS_DEST]*len(source_labels_24.data['x']))
                plot_wellplate_dest_2.axis.visible = True

            else:
                cds_labels_dest_2.data = {'x':[], 'y':[]}
                plot_wellplate_dest_2.title.text = ""
                plot_wellplate_dest_2.axis.visible = False
        else:
            cds_labels_dest_2.data = {'x':[], 'y':[]}
            plot_wellplate_dest_2.title.text = ""
            plot_wellplate_dest_2.axis.visible = False
    dropdown_n_dest_wellplates.on_change("value", load_well_plate_dest_2)



    #___________________________________________________________________________________________
    def add_drug_to_well(drug):
        print('------------------->>>>>>>>> add_drug_to_well')

        experiement   = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiement:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Select a valid experiment.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return

        plate = experiement.source_plate

        positions = get_well_mapping(cds_labels_source.selected.indices)
        positions_supp = get_well_mapping(cds_labels_source_supp.selected.indices, issupp=True)
        print('add drug to well positions=', positions)
        print('add drug to well positions supp=', positions_supp)   

        wells=[]
        for pos in positions:
            print('pos=', pos)
            try:
                source_well_pos = SourceWellPosition.objects.get(well_plate=plate, position_col=pos[0], position_row=pos[1], is_supp=False)
                print('source_well_pos=', source_well_pos)
                drug.position.add(source_well_pos)
                wells.append(f"{pos[1]}{pos[0]}")
            except SourceWellPosition.DoesNotExist:
                print(f"Source well position {pos} does not exist in the source well plate.")

        for pos in positions_supp:
            print('pos supp=', pos)
            try:
                source_well_pos_supp = SourceWellPosition.objects.get(well_plate=plate, position_col=pos[0], position_row=pos[1], is_supp=True)
                print('source_well_pos_supp=', source_well_pos_supp)
                drug.position.add(source_well_pos_supp) 
                wells.append(f"{pos[1]}{pos[0]}")

            except SourceWellPosition.DoesNotExist:
                print(f"Source Supp well position {pos} does not exist in the source well plate.")
        return wells
    
    #___________________________________________________________________________________________
    def display_drugs_source_wellplate():
        print('------------------->>>>>>>>> display_drugs_source_wellplate')
        cds_labels_source_drug.data = {'x':[], 'y':[], 'size':[], 'drug':[]}
        cds_labels_source_supp_drug.data = {'x':[], 'y':[], 'size':[], 'drug':[]}

        experiment = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiment:
            return

        source_well_plate = SourceWellPlate.objects.filter(experiment=experiment).first()
        if not source_well_plate:
            return

        source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=False)
        source_well_positions_supp = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=True)

        x_filled = []
        y_filled = []
        size_filled = []
        drug_filled = []
        for well_pos in source_well_positions:
            drug = Drug.objects.filter(position=well_pos)
            if len(drug) != 0:
                x_filled.append(well_pos.position_col)
                y_filled.append(well_pos.position_row)
                size_filled.append(cds_labels_source.data['size'][cds_labels_source.data['x'].index(well_pos.position_col)])
                drug_filled.append('\n '.join([f'{str(d.derivation_name)}\n{d.concentration}muMol' for d in drug]))
        cds_labels_source_drug.data={'x':x_filled, 'y':y_filled, 'size':size_filled, 'drug':drug_filled}

        x_supp = []
        y_supp = []
        size_supp = []
        drug_supp = []
        for well_pos in source_well_positions_supp:
            drug = Drug.objects.filter(position=well_pos)
            if len(drug) != 0:
                x_supp.append(well_pos.position_col)
                y_supp.append(well_pos.position_row)
                size_supp.append(cds_labels_source_supp.data['size'][cds_labels_source_supp.data['x'].index(well_pos.position_col)])
                drug_supp.append('\n '.join([f'{str(d.derivation_name)}\n{d.concentration}muMol' for d in drug]))

        cds_labels_source_supp_drug.data={'x':x_supp, 'y':y_supp, 'size':size_supp,'drug':drug_supp}


    #___________________________________________________________________________________________
    def display_drugs_dest_wellplate():
        print('------------------->>>>>>>>> display_drugs_dest_wellplate')
        cds_labels_dest_1_drug.data = {'x':[], 'y':[], 'size':[]}
        cds_labels_dest_2_drug.data = {'x':[], 'y':[], 'size':[]}
        cds_labels_dest_1_drug_control.data = {'x':[], 'y':[], 'size':[]}
        cds_labels_dest_2_drug_control.data = {'x':[], 'y':[], 'size':[]}
        experiment = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiment:
            return
        
        source_well_plate = SourceWellPlate.objects.filter(experiment=experiment).first()
        if not source_well_plate:
            return
        
        dest_well_plate = DestWellPlate.objects.filter(experiment=experiment)

        if len(dest_well_plate)==0:
            return
        
        dest_well_plate_1 = dest_well_plate.filter(plate_number=1).first()
        dest_well_plate_2 = dest_well_plate.filter(plate_number=2).first()

        print('dest_well_plate_1=', dest_well_plate_1)
        print('dest_well_plate_2=', dest_well_plate_2)
        

        dest_well_positions_1 = DestWellPosition.objects.filter(well_plate=dest_well_plate_1)
        x_dest_1 = []
        y_dest_1 = []
        size_dest_1 = []
        x_dest_1_control = []
        y_dest_1_control = []
        size_dest_1_control = []
        for well_pos in dest_well_positions_1:
            if well_pos.source_well!= None:
                drug = well_pos.source_well.drugs.all()
                print('well_pos=', well_pos, ' -- ', len(drug))
                if len(drug) == 1:
                    x_dest_1_control.append(well_pos.position_col)
                    y_dest_1_control.append(well_pos.position_row)
                    size_dest_1_control.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
                elif len(drug) > 1:
                    x_dest_1.append(well_pos.position_col)
                    y_dest_1.append(well_pos.position_row)
                    size_dest_1.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
        cds_labels_dest_1_drug.data = {'x':x_dest_1, 'y':y_dest_1, 'size':size_dest_1}
        cds_labels_dest_1_drug_control.data = {'x':x_dest_1_control, 'y':y_dest_1_control, 'size':size_dest_1_control}
        dest_well_positions_2 = DestWellPosition.objects.filter(well_plate=dest_well_plate_2)
        x_dest_2 = []
        y_dest_2 = []
        size_dest_2 = []
        x_dest_2_control = []
        y_dest_2_control = []
        size_dest_2_control = []
        for well_pos in dest_well_positions_2:
            if well_pos.source_well!= None:
                drug = well_pos.source_well.drugs.all()
                print('well_pos=', well_pos, ' -- ', len(drug))
                if len(drug) == 1:
                    x_dest_2_control.append(well_pos.position_col)
                    y_dest_2_control.append(well_pos.position_row)
                    size_dest_2_control.append(cds_labels_dest_2.data['size'][cds_labels_dest_2.data['x'].index(well_pos.position_col)])
                elif len(drug) > 1:
                    x_dest_2.append(well_pos.position_col)
                    y_dest_2.append(well_pos.position_row)
                    size_dest_2.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
        cds_labels_dest_2_drug.data = {'x':x_dest_2, 'y':y_dest_2, 'size':size_dest_2}
        cds_labels_dest_2_drug_control.data = {'x':x_dest_2_control, 'y':y_dest_2_control, 'size':size_dest_2_control}

    #___________________________________________________________________________________________
    def display_drug_name(attr, old, new):
        print('------------------->>>>>>>>> display_drug_name')
        cds_labels_source_supp_drug.selected.indices = []

        global _programmatic_change
        if _programmatic_change: 
            return

        _programmatic_change = True
        cds_labels_source_supp.selected.indices = []
        _programmatic_change = False

        experiment = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiment:
            return
        source_well_plate = SourceWellPlate.objects.filter(experiment=experiment).first()
        if not source_well_plate:
            return 
        if len(new) == 0:
            print('No drug selected')
            drug_message.text = ''
            drug_message.visible = False
            mapping_message.text = ''
            mapping_message.visible = False
            cds_labels_dest_2_mapping.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_mapping.data = {'x':[], 'y':[], 'size':[]}
            return
        if len(new) > 1:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Can not display more than 1 drug info.</b>"
            drug_message.visible = True
            return
        well_position = get_well_mapping(new)
        source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=False, position_col=well_position[0][0], position_row=well_position[0][1])
        drugs = Drug.objects.filter(position__in=source_well_positions)
        print("--------source_well_positions ",source_well_positions)

        items_html = "".join(
            f"<li style='color:navy; font-size:14px; "
            f"margin-bottom:4px;'>{drug}</li>"
            for drug in drugs)

        drug_message.text = f"""
        <b style='color:green; font-size:18px;'>
            Drug(s) in selected well {well_position[0][1]}{well_position[0][0]}:
        </b>
        <ul style='margin-top:0;'>
            {items_html} <br> <b style='color:black; font-size:14px;'> comments={source_well_positions[0].comments}, valid well={source_well_positions[0].valid}</b>
        </ul>
        """

        drug_message.visible = True
        add_drug_button.label = "Add drug"
        add_drug_button.button_type = "success"

        dests = source_well_positions.first().destwellposition_set.all()
        print('dests=', dests)


        if len(dests) == 0:
            mapping_message.text = f"<b style='color:red; font-size:18px;'> No destination wells mapped for well {well_position[0][1]}{well_position[0][0]}.</b>"
            mapping_message.visible = True
            cds_labels_dest_2_mapping.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_mapping.data = {'x':[], 'y':[], 'size':[]}
            return
        
        str_1 = ''
        str_2 = ''
        for well_pos in dests:
            if well_pos.source_well!= None:
                if well_pos.well_plate.plate_number == 1:
                    if str_1 == "":
                        str_1 = f'{str(well_pos.position_row)}{well_pos.position_col}'
                    else:
                        str_1+=f', {str(well_pos.position_row)}{well_pos.position_col}'
                if well_pos.well_plate.plate_number == 2:
                    if str_2 == "":
                        str_2 = f'{str(well_pos.position_row)}{well_pos.position_col}'
                    else:
                        str_2+=f', {str(well_pos.position_row)}{well_pos.position_col}'
        if str_1 != '':
            mapping_message.text = f"<b style='color:green; font-size:18px;'> Mapped well {well_position[0][1]}{well_position[0][0]} to {str_1} for plate number 1.</b><br>"
        if str_2 != '' and str_1 != '':
            mapping_message.text += f"<b style='color:green; font-size:18px;'> Mapped well {well_position[0][1]}{well_position[0][0]} to {str_2} for plate number 2.</b>"
        if str_2 != '' and str_1 == '':
            mapping_message.text = f"<b style='color:green; font-size:18px;'> Mapped well {well_position[0][1]}{well_position[0][0]} to {str_2} for plate number 2.</b>"

        mapping_message.visible = True

        x_dest_1 = []
        y_dest_1 = []
        size_dest_1 = []
        x_dest_2 = []
        y_dest_2 = []
        size_dest_2 = []
        for well_pos in dests:
            if well_pos.source_well!= None:
                if well_pos.well_plate.plate_number == 1:
                    x_dest_1.append(well_pos.position_col)
                    y_dest_1.append(well_pos.position_row)
                    size_dest_1.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
                if well_pos.well_plate.plate_number == 2:
                    x_dest_2.append(well_pos.position_col)
                    y_dest_2.append(well_pos.position_row)
                    size_dest_2.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
        cds_labels_dest_mapping.data = {'x':x_dest_1, 'y':y_dest_1, 'size':size_dest_1}
        cds_labels_dest_2_mapping.data = {'x':x_dest_2, 'y':y_dest_2, 'size':size_dest_2}
        
    cds_labels_source.selected.on_change('indices',display_drug_name)



    #___________________________________________________________________________________________
    def display_drug_supp_name(attr, old, new):
        print('------------------->>>>>>>>> display_drug_supp_name')
        cds_labels_source_drug.selected.indices = []
        global _programmatic_change
        if _programmatic_change: 
            return

        _programmatic_change = True
        cds_labels_source.selected.indices = []
        _programmatic_change = False

        experiment = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiment:
            return
        source_well_plate = SourceWellPlate.objects.filter(experiment=experiment).first()
        if not source_well_plate:
            return 
        if len(new) == 0:
            drug_message.text = ''
            drug_message.visible = False
            mapping_message.text = ''
            mapping_message.visible = False
            cds_labels_dest_2_mapping.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_mapping.data = {'x':[], 'y':[], 'size':[]}
            return
        well_position = get_well_mapping(new, issupp=True)
        source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=True, position_col=well_position[0][0], position_row=well_position[0][1])
        drugs = Drug.objects.filter(position__in=source_well_positions)

        items_html = "".join(
            f"<li style='color:navy; font-size:14px; "
            f"margin-bottom:4px;'>{drug}</li>"
            for drug in drugs
        )

        drug_message.text = f"""
        <b style='color:green; font-size:18px;'>
            Drug(s) in selected well {well_position[0][1]}{well_position[0][0]}:
        </b>
        <ul style='margin-top:0;'>
            {items_html}
        </ul>
        """

        #drug_message.text = f"<b style='color:green; ; font-size:18px;'> Drug(s) in selected well {well_position[1]}{well_position[0]}.</b>"
        drug_message.visible = True
        add_drug_button.label = "Add drug"
        add_drug_button.button_type = "success"


        dests = source_well_positions.first().destwellposition_set.all()
        print('dests=', dests)
        if len(dests) == 0:
            mapping_message.text = f"<b style='color:red; font-size:18px;'> No destination wells mapped for well {well_position[0][1]}{well_position[0][0]}.</b>"
            mapping_message.visible = True
            cds_labels_dest_2_mapping.data = {'x':[], 'y':[], 'size':[]}
            cds_labels_dest_mapping.data = {'x':[], 'y':[], 'size':[]}
            return
        
        str_1 = ''
        str_2 = ''
        for well_pos in dests:
            if well_pos.source_well!= None:
                if well_pos.well_plate.plate_number == 1:
                    if str_1 == "":
                        str_1 = f'{str(well_pos.position_row)}{well_pos.position_col}'
                    else:
                        str_1+=f', {str(well_pos.position_row)}{well_pos.position_col}'
                if well_pos.well_plate.plate_number == 2:
                    if str_2 == "":
                        str_2 = f'{str(well_pos.position_row)}{well_pos.position_col}'
                    else:
                        str_2+=f', {str(well_pos.position_row)}{well_pos.position_col}'
        if str_1 != '':
            mapping_message.text = f"<b style='color:green; font-size:18px;'> Mapped well {well_position[0][1]}{well_position[0][0]} to {str_1} for plate number 1.</b><br>"
        if str_2 != '' and str_1 != '':
            mapping_message.text += f"<b style='color:green; font-size:18px;'> Mapped well {well_position[0][1]}{well_position[0][0]} to {str_2} for plate number 2.</b>"
        if str_2 != '' and str_1 == '':
            mapping_message.text = f"<b style='color:green; font-size:18px;'> Mapped well {well_position[0][1]}{well_position[0][0]} to {str_2} for plate number 2.</b>"

        mapping_message.visible = True

        x_dest_1 = []
        y_dest_1 = []
        size_dest_1 = []
        x_dest_2 = []
        y_dest_2 = []
        size_dest_2 = []
        for well_pos in dests:
            if well_pos.source_well!= None:
                if well_pos.well_plate.plate_number == 1:
                    x_dest_1.append(well_pos.position_col)
                    y_dest_1.append(well_pos.position_row)
                    size_dest_1.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
                if well_pos.well_plate.plate_number == 2:
                    x_dest_2.append(well_pos.position_col)
                    y_dest_2.append(well_pos.position_row)
                    size_dest_2.append(cds_labels_dest.data['size'][cds_labels_dest.data['x'].index(well_pos.position_col)])
        cds_labels_dest_mapping.data = {'x':x_dest_1, 'y':y_dest_1, 'size':size_dest_1}
        cds_labels_dest_2_mapping.data = {'x':x_dest_2, 'y':y_dest_2, 'size':size_dest_2}
        
    cds_labels_source_supp.selected.on_change('indices',display_drug_supp_name)


    #___________________________________________________________________________________________
    def add_wellcluster_comment_valid():        
        if cds_labels_source.selected.indices == [] and cds_labels_source_supp.selected.indices == []:
            wellvalid_message.text = f"<b style='color:red; ; font-size:18px;'> Error: No source well selected.</b>"
            wellvalid_message.visible = True
            return

        if cds_labels_source.selected.indices != [] and cds_labels_source_supp.selected.indices != []:
            wellvalid_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Can not select a source well from main plate and supp plate.</b>"
            wellvalid_message.visible = True
            return
        
        if len(cds_labels_source.selected.indices) + len(cds_labels_source_supp.selected.indices) >1:
            wellvalid_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Can not add to multiple source wells.</b>"
            wellvalid_message.visible = True
            return
        
        positions = get_well_mapping(cds_labels_source.selected.indices)
        positions_supp = get_well_mapping(cds_labels_source_supp.selected.indices, issupp=True)
        print('add drug to well positions=', positions)
        print('add drug to well positions supp=', positions_supp)   

        experiement   = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiement:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Select a valid experiment.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return

        plate = experiement.source_plate
        is_supp = False
        if len(positions_supp)==1: 
            is_supp = True
            positions=positions_supp
        
        try:
            source_well_pos = SourceWellPosition.objects.get(well_plate=plate, position_col=positions[0][0], position_row=positions[0][1], is_supp=is_supp)
            print('source_well_pos=', source_well_pos)
            source_well_pos.comments = wellcluster_comment.value
            source_well_pos.valid = valid_wellcluster.value
            source_well_pos.save()
        except SourceWellPosition.DoesNotExist:
            print(f"Source well position {positions[0]} does not exist in the source well plate.")

        wellvalid_message.text = f"<b style='color:green; ; font-size:18px;'> Added comment to well {positions[0][1]}{positions[0][0]}.</b>"
        wellvalid_message.visible = True

        wellcluster_comment.value = ''
        valid_wellcluster.value = 'True'

    valid_wellcluster_button.on_click(add_wellcluster_comment_valid)

    #___________________________________________________________________________________________
    #this function adds a drug to the source well plate and to the database
    def add_drug():
        print('------------------->>>>>>>>> add_drug')
        experiement   = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiement:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Select a valid experiment.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return

        slims_deriv = slims.fetch("Content", slims_cr.equals("cntn_id", slimsid_name.value))
        if len(slims_deriv)==0:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid slims ID.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return
        drug_deriv = slims_deriv[0].json_entity['columns']

        #print('drug_deriv')
        #for d in drug_deriv:
        #    print('   ----   ',d)

        stock_id = ''
        deriv_name = ''

        for element in drug_deriv:
            if element["name"]=="cntn_cf_name": deriv_name= element["value"]
            if element["name"]=="cntn_fk_originalContent": stock_id= element["displayValue"]

        slims_stock = slims.fetch("Content", slims_cr.equals("cntn_id", stock_id))
        if len(slims_stock)==0 and 'OA_CH' not in slimsid_name.value:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Could not find powder id {stock_id} for the derivation {slimsid_name.value} in slims</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return
       
        if drug_concentration.value == '':
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid concentration or percentage.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return

        try:
            x = float(drug_concentration.value)
        except ValueError:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid numerical concentration.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return
        
        if cds_labels_source.selected.indices == [] and cds_labels_source_supp.selected.indices == []:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Need to select at least one source well.</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            return

        drug_exists = Drug.objects.filter(slims_id=slimsid_name.value, concentration=str(drug_concentration.value)).exists()

        if drug_exists:

            drug_exp = Drug.objects.filter(slims_id=slimsid_name.value, 
                                       concentration=str(drug_concentration.value),
                                       position__well_plate__experiment__name=experiment_name.value).distinct()
            
            if len(drug_exp) == 0:
                exp_qs = Experiment.objects.filter(source_plate__sourcewellposition__drugs__slims_id=slimsid_name.value,
                                                            source_plate__sourcewellposition__drugs__concentration=drug_concentration.value).distinct()
                print('exp_qs=', exp_qs)
                name_list = [exp.name for exp in exp_qs]
                drug_message.text = (f"<b style='color:orange; ; font-size:18px;'> Warning: Drug with SLIMS ID {slimsid_name.value} and concentration {drug_concentration.value}</b><br>"
                                     f"<b style='color:orange; ; font-size:18px;'> already exists in the database for other experiment(s) {name_list} .</b><br>"
                                     f"<b style='color:orange; ; font-size:18px;'> Use the 'Force add drug' button to add it again to this experiment.</b>")
                drug_message.visible = True
                add_drug_button.label = "Add drug"
                add_drug_button.button_type = "success"
                return
            else:
                drug_message.text = (f"<b style='color:orange; ; font-size:18px;'> Warning: Drug with SLIMS ID {slimsid_name.value} and concentration {drug_concentration.value}</b><br>"
                                     f"<b style='color:orange; ; font-size:18px;'> already exists in the database for experiment {experiment_name.value}</b><br>"
                                     f"<b style='color:orange; ; font-size:18px;'> Use the 'Add drug to other wells' button to add it to other wells for this experiment.</b>")
                drug_message.visible = True
                add_drug_button.label = "Add drug"
                add_drug_button.button_type = "success"
                return
                
        else:

            drug = Drug(slims_id=slimsid_name.value, 
                        concentration=drug_concentration.value, 
                        valid=valid_wellcluster.value,
                        derivation_name=deriv_name)
            drug.save()

            wells=', '.join(add_drug_to_well(drug))

            drug_message.text = f"<b style='color:green; ; font-size:18px;'>Added drug {deriv_name} with concentration/percentage {drug_concentration.value} in well(s) {wells}</b>"
            drug_message.visible = True
            add_drug_button.label = "Add drug"
            add_drug_button.button_type = "success"
            
            display_drugs_source_wellplate()
            display_drugs_dest_wellplate()

            print('about ot call display_drugs_source_wellplate')
            global _programmatic_change
            _programmatic_change = True
            cds_labels_source.selected.indices = []
            cds_labels_source_supp.selected.indices = []
            _programmatic_change = False
        print('cds_labels_source.data     ',cds_labels_source.data)
        print('cds_labels_source.indices  ',cds_labels_source.selected.indices)
        print('cds_labels_source_supp.data     ',cds_labels_source_supp.data)
        print('cds_labels_source_supp.indices  ',cds_labels_source_supp.selected.indices)
        print('cds_labels_source_drug.data     ',cds_labels_source_drug.data)
        print('cds_labels_source_drug.indices  ',cds_labels_source_drug.selected.indices)
        print('cds_labels_source_supp_drug.data     ',cds_labels_source_supp_drug.data)
        print('cds_labels_source_supp_drug.indices  ',cds_labels_source_supp_drug.selected.indices)



    #___________________________________________________________________________________________
    def force_add_drug():
        print('------------------->>>>>>>>> force_add_drug')

        pos=get_well_mapping(cds_labels_source.selected.indices)
        pos_supp=get_well_mapping(cds_labels_source_supp.selected.indices, issupp=True)

        print('positions source ', pos)
        print('positions source supp ', pos_supp)
        print('slims id: ',slimsid_name.value)

        slims_deriv = slims.fetch("Content", slims_cr.equals("cntn_id", slimsid_name.value))
        if len(slims_deriv)==0:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid slims ID.</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"
            return
        drug_deriv = slims_deriv[0].json_entity['columns']

        deriv_name = ''
        stock_id = ''

        for element in drug_deriv:
            if element["name"]=="cntn_cf_name":
                print('derivation name=',element["value"])
                deriv_name = element["value"]
            if element["name"]=="cntn_fk_originalContent":
                print('stock ID=',element["displayValue"])
                stock_id = element["displayValue"]

        slims_stock = slims.fetch("Content", slims_cr.equals("cntn_id", stock_id))
        if len(slims_stock)==0 and 'OA_CH' not in slimsid_name.value:

            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid slims ID.</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"
            return

        if drug_concentration.value == '' :
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid concentration.</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"
            return

        try:
            x = float(drug_concentration.value)
        except ValueError:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid numerical concentration.</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"
            return
        
        if cds_labels_source.selected.indices == [] and cds_labels_source_supp.selected.indices == []:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Need to select a source well.</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"
            return

        drug = Drug.objects.filter(slims_id=slimsid_name.value, 
                                   concentration=str(drug_concentration.value),
                                   position__well_plate__experiment__name=experiment_name.value).distinct()
        print('drug=', drug)

        if len(drug)==0:

            drug = Drug(slims_id=slimsid_name.value,
                        concentration=drug_concentration.value, 
                        valid=valid_wellcluster.value,
                        derivation_name=deriv_name)
            drug.save()

            wells=add_drug_to_well(drug)

            drug_message.text = f"<b style='color:green; ; font-size:18px;'>Added drug {deriv_name} with concentration/percentage {drug_concentration.value} in well(s) {wells}</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"
            display_drugs_source_wellplate()
            display_drugs_dest_wellplate()

        else:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Drug with SLIMS ID {slimsid_name.value} and concentration {drug_concentration.value} </b><br>"
            drug_message.text += f"<b style='color:red; ; font-size:18px;'> already exists in the database for the experiment {experiment_name.value}.</b><br>"
            drug_message.text += f"<b style='color:red; ; font-size:18px;'> Please use 'Add drug to other wells' button.</b>"
            drug_message.visible = True
            force_add_drug_button.label = "Force add drug"
            force_add_drug_button.button_type = "success"

    #___________________________________________________________________________________________
    def add_drug_other_wells(): 
        print('------------------->>>>>>>>> add_drug_other_wells')
        experiement   = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiement:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Select a valid experiment.</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return

        slims_deriv = slims.fetch("Content", slims_cr.equals("cntn_id", slimsid_name.value))
        if len(slims_deriv)==0:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid slims ID.</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return
        drug_deriv = slims_deriv[0].json_entity['columns']

     
        stock_id = ''
        deriv_name = ''

        for element in drug_deriv:
            if element["name"]=="cntn_cf_name": deriv_name= element["value"]
            if element["name"]=="cntn_fk_originalContent": stock_id= element["displayValue"]

        slims_stock = slims.fetch("Content", slims_cr.equals("cntn_id", stock_id))
        if len(slims_stock)==0 and 'OA_CH' not in slimsid_name.value:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Could not find powder id {stock_id} for the derivation {slimsid_name.value} in slims</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return
       
        if drug_concentration.value == '' :
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid concentration.</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"

        try:
            x = float(drug_concentration.value)
        except ValueError:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Enter a valid numerical concentration.</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return
        
        if cds_labels_source.selected.indices == [] and cds_labels_source_supp.selected.indices == []:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Need to select at least one source well.</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return

        drug_exists_overall = Drug.objects.filter(slims_id=slimsid_name.value, concentration=str(drug_concentration.value)).exists()

        drug_exists = Drug.objects.filter(slims_id=slimsid_name.value, 
                                          concentration=str(drug_concentration.value), 
                                          position__well_plate__experiment__name=experiment_name.value).distinct()

        print('drug_exists=', drug_exists)
        print('drug_exists_overall=', drug_exists_overall)
        if not drug_exists_overall:
            drug_message.text = (f"<b style='color:orange; ; font-size:18px;'> Warning: Drug with SLIMS ID {slimsid_name.value} and concentration {drug_concentration.value}</b><br>"
                                 f"<b style='color:orange; ; font-size:18px;'> does not already exists in the database. Use the 'Add drug' button to add the drug to the database.</b>")
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return

        if len(drug_exists) == 0:
            exp_qs = Experiment.objects.filter(source_plate__sourcewellposition__drugs__slims_id=slimsid_name.value,
                                               source_plate__sourcewellposition__drugs__concentration=drug_concentration.value).distinct()
            print('exp_qs=', exp_qs)
            name_list = [exp.name for exp in exp_qs]
            drug_message.text = (f"<b style='color:orange; ; font-size:18px;'> Warning: Drug with SLIMS ID {slimsid_name.value} and concentration {drug_concentration.value}</b><br>"
                                 f"<b style='color:orange; ; font-size:18px;'> already exists in the database for other experiment(s) {name_list}.</b><br>"
                                 f"<b style='color:orange; ; font-size:18px;'> Use the 'Force add drug' button to add it again to this experiment.</b>")
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            return
        
        else:

            wells=add_drug_to_well(drug_exists[0])

            #drug_message.text = f"<b style='color:green; ; font-size:18px;'>Added drug {deriv_name} in well(s) {wells}</b>"
            drug_message.text = f"<b style='color:green; ; font-size:18px;'>Added drug {deriv_name} with concentration/percentage {drug_concentration.value} in well(s) {wells}</b>"
            drug_message.visible = True
            add_drug_other_wells_button.label = "Add drug to other wells"
            add_drug_other_wells_button.button_type = "success"
            display_drugs_source_wellplate()
            display_drugs_dest_wellplate()



    #___________________________________________________________________________________________
    def map_drugs_to_wellplate():
        print('------------------->>>>>>>>> map_drugs_to_wellplate')
        if dropdown_well_plate_dest.value == 'Select a well plate':
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Select a destination well plate.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"
            return
        
        if cds_labels_source_supp.selected.indices == [] and cds_labels_source.selected.indices == []:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Need to select a source well for the mapping.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"
            return

        if len(cds_labels_source_supp.selected.indices)+len(cds_labels_source.selected.indices)>1:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Can not associate from 2 different source wells for the mapping.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"
            return

        if cds_labels_dest.selected.indices == [] and cds_labels_dest_2.selected.indices == []:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Need to select a destination well for the mapping.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"
            return

        if cds_labels_dest.selected.indices != [] and cds_labels_dest_2.selected.indices != []:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: can not associate to two different destination well-plates for the mapping.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"
            return

        experiment = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiment:
            return
        source_well_plate = SourceWellPlate.objects.filter(experiment=experiment).first()
        if not source_well_plate:
            return 

        dest_well_plate = DestWellPlate.objects.filter(experiment=experiment,plate_number=1).first()
        if not dest_well_plate:
            return 
        print('===============dest_well_plate=', dest_well_plate)  
        dest_well_plate_supp = DestWellPlate.objects.filter(experiment=experiment)
        print('===============dest_well_plate_supp=', dest_well_plate_supp)
        print('===============len(dest_well_plate_supp)=', len(dest_well_plate_supp))
        if len(dest_well_plate_supp) == 2:
            dest_well_plate_2 = dest_well_plate_supp.filter(plate_number=2).first()

        source_well_pos = cds_labels_source.selected.indices
        source_well_pos_supp = cds_labels_source_supp.selected.indices
        print('source_well_pos=', source_well_pos)
        if cds_labels_source.selected.indices != []:
            well_position = get_well_mapping(source_well_pos)
            print('well_position in if=', well_position)
            source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=False, position_col=well_position[0][0], position_row=well_position[0][1])

        else:
            well_position = get_well_mapping(source_well_pos_supp, issupp=True)
            print('well_position in else=', well_position)
            source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=True, position_col=well_position[0][0], position_row=well_position[0][1])

        drugs = Drug.objects.filter(position__in=source_well_positions)

        if len(drugs) == 0:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: No drug in the selected source well.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"
            return

        dest_well_pos = cds_labels_dest.selected.indices
        dest_well_pos_2 = cds_labels_dest_2.selected.indices

        dest_well_position = get_well_mapping(dest_well_pos, issource=False)
        dest_well_position_2 = get_well_mapping(dest_well_pos_2, issource=False)

        print('----dest_well_pos=', dest_well_pos)
        print('----dest_well_pos_2=', dest_well_pos_2)
        print('----dest_well_position=', dest_well_position)
        print('----dest_well_position_2=', dest_well_position_2)


        if cds_labels_dest.selected.indices != []:
            dest_well_string = ''
            for i in range(len(dest_well_position)):
                if i==0:
                    dest_well_string = '{}{}'.format(dest_well_position[i][1], dest_well_position[i][0])
                else:
                    dest_well_string += ', {}{}'.format(dest_well_position[i][1], dest_well_position[i][0])
            source_well_string = ''
            for i in range(len(well_position)):
                if i==0:
                    source_well_string = '{}{}'.format(well_position[i][1], well_position[i][0])
                else:
                    source_well_string += ', {}{}'.format(well_position[i][1], well_position[i][0])


            q = Q()
            pairs = [(x[0], x[1]) for x in dest_well_position]
            for col, row in pairs:
                q |= Q(position_col=col, position_row=row)

            dest_well_positions = DestWellPosition.objects.filter(well_plate=dest_well_plate).filter(q)

            print('>>>>dest_well_positions NEW =', dest_well_positions)
            for dest_well_p in dest_well_positions:
                dest_well_p.source_well = source_well_positions[0]
                dest_well_p.save()
            
            mapping_message.text = f"<b style='color:green; ; font-size:18px;'> Mapped source well {source_well_string} to destination wells {dest_well_string} in plate {dest_well_plate.plate_number}.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"

        if cds_labels_dest_2.selected.indices != []:
            dest_well_string_2 = ''
            for i in range(len(dest_well_position_2)):
                if i==0:
                    dest_well_string_2 = '{}{}'.format(dest_well_position_2[i][1], dest_well_position_2[i][0])
                else:
                    dest_well_string_2 += ', {}{}'.format(dest_well_position_2[i][1], dest_well_position_2[i][0])
            source_well_string_2 = ''
            for i in range(len(well_position)):
                if i==0:
                    source_well_string_2 = '{}{}'.format(well_position[i][1], well_position[i][0])
                else:
                    source_well_string_2 += ', {}{}'.format(well_position[i][1], well_position[i][0])

            q = Q()
            pairs = [(x[0], x[1]) for x in dest_well_position_2]
            for col, row in pairs:
                q |= Q(position_col=col, position_row=row)

            dest_well_positions_2 = DestWellPosition.objects.filter(well_plate=dest_well_plate_2).filter(q)

            print('>>>>dest_well_positions_2=', dest_well_positions_2)
            for dest_well_p in dest_well_positions_2:
                dest_well_p.source_well = source_well_positions[0]
                dest_well_p.save()
        
            if mapping_message.text != '':
                mapping_message.text += '<br>'
                mapping_message.text += f"<b style='color:green; ; font-size:18px;'> Mapped source well {source_well_string_2} to destination wells {dest_well_string_2} in plate {dest_well_plate_2.plate_number}.</b>"
            else:
                mapping_message.text = f"<b style='color:green; ; font-size:18px;'> Mapped source well {source_well_string_2} to destination wells {dest_well_string_2} in plate {dest_well_plate_2.plate_number}.</b>"
            mapping_message.visible = True
            map_drug_button.label = "Map drug"
            map_drug_button.button_type = "success"

        cds_labels_dest.selected.indices = []
        cds_labels_dest_2.selected.indices =[]
        display_drugs_dest_wellplate()


    #___________________________________________________________________________________________
    def map_drugs_to_wellplate_short():
        map_drug_button.label = "Processing"
        map_drug_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(map_drugs_to_wellplate)
    map_drug_button.on_click(map_drugs_to_wellplate_short)

    #___________________________________________________________________________________________
    def unmap_drugs_to_wellplate():
        
        if dropdown_well_plate_dest.value == 'Select a well plate':
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Select a destination well plate.</b>"
            mapping_message.visible = True
            unmap_drug_button.label = "Unmap drug"
            return
        
        if cds_labels_source_supp.selected.indices == [] and cds_labels_source.selected.indices == []:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Need to select a source well for the unmapping.</b>"
            mapping_message.visible = True
            unmap_drug_button.label = "Unmap drug"
            return

        if len(cds_labels_source_supp.selected.indices)+len(cds_labels_source.selected.indices)>1:
            mapping_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Can not unmap several source wells at the same time.</b>"
            mapping_message.visible = True
            unmap_drug_button.label = "Unmap drug"
            return

        experiment = Experiment.objects.filter(name=dropdown_exp.value).first()
        if not experiment:
            return
        source_well_plate = SourceWellPlate.objects.filter(experiment=experiment).first()
        if not source_well_plate:
            return 
        
        source_well_pos = cds_labels_source.selected.indices
        source_well_pos_supp = cds_labels_source_supp.selected.indices
        print('source_well_pos=', source_well_pos)
        print('source_well_pos_supp=', source_well_pos_supp)
        if cds_labels_source.selected.indices != []:
            well_position = get_well_mapping(source_well_pos)
            print('well_position in if=', well_position)
            source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=False, position_col=well_position[0][0], position_row=well_position[0][1]).first()

        else:
            well_position = get_well_mapping(source_well_pos_supp, issupp=True)
            print('well_position in else=', well_position)
            source_well_positions = SourceWellPosition.objects.filter(well_plate=source_well_plate, is_supp=True, position_col=well_position[0][0], position_row=well_position[0][1]).first()

        print('source_well_positions=', source_well_positions)

        #swp = SourceWellPosition.objects.get(pk=source_well_id)
        count = source_well_positions.unmap_dest_wells()

        mapping_message.text = f"<b style='color:gree; ; font-size:18px;'> Succesfully unmapped {count} wells.</b>"
        mapping_message.visible = True
        unmap_drug_button.label = "Unmap drug"
        display_drugs_dest_wellplate()
        cds_labels_dest.selected.indices = []
        cds_labels_dest_2.selected.indices =[]
        cds_labels_dest_2_mapping.data = {'x':[], 'y':[], 'size':[]}
        cds_labels_dest_mapping.data = {'x':[], 'y':[], 'size':[]}

    #___________________________________________________________________________________________
    def unmap_drugs_to_wellplate_short():
        unmap_drug_button.label = "Processing"
        unmap_drug_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(unmap_drugs_to_wellplate)
    unmap_drug_button.on_click(unmap_drugs_to_wellplate_short)


    zoom_in_wells_source = bokeh.models.Button(label="Zoom in wells source")
    zoom_out_wells_source = bokeh.models.Button(label="Zoom out wells source")

    zoom_in_wells_dest = bokeh.models.Button(label="Zoom in wells dest")
    zoom_out_wells_dest = bokeh.models.Button(label="Zoom out wells dest")

    #___________________________________________________________________________________________
    def zoom_size(factor, cds):

        if len(cds.data['size'])>0:
            new_size = int(cds.data['size'][0] * factor)
            data = dict(cds.data)
            data["size"] = [new_size] * len(data["x"])
            cds.data = data


    #___________________________________________________________________________________________
    def make_zoom_cb_wells_source(factor):
        def zoom_cb():
            global NZOOM_WELLS_SOURCE
            NZOOM_WELLS_SOURCE *= factor

            plot_wellplate_source.width  = int(plot_wellplate_source.width * factor)
            plot_wellplate_source.height = int(plot_wellplate_source.height * factor)
            plot_wellplate_source_supp.width  = int(plot_wellplate_source_supp.width * factor)
            plot_wellplate_source_supp.height = int(plot_wellplate_source_supp.height * factor)

            zoom_size(factor, cds_labels_source)
            zoom_size(factor, cds_labels_source_supp)
            zoom_size(factor, cds_labels_source_drug)
            zoom_size(factor, cds_labels_source_supp_drug)


        return zoom_cb

    #___________________________________________________________________________________________
    def make_zoom_cb_wells_dest(factor):
        def zoom_cb():
            global NZOOM_WELLS_DEST
            NZOOM_WELLS_DEST *= factor
            plot_wellplate_dest.width  = int(plot_wellplate_dest.width * factor)
            plot_wellplate_dest.height = int(plot_wellplate_dest.height * factor)
            plot_wellplate_dest_2.width  = int(plot_wellplate_dest_2.width * factor)
            plot_wellplate_dest_2.height = int(plot_wellplate_dest_2.height * factor)
         
            zoom_size(factor, cds_labels_dest)
            zoom_size(factor, cds_labels_dest_2)
            zoom_size(factor, cds_labels_dest_1_drug)
            zoom_size(factor, cds_labels_dest_2_drug)
            zoom_size(factor,cds_labels_dest_mapping)
            zoom_size(factor,cds_labels_dest_2_mapping)
            zoom_size(factor,cds_labels_dest_1_drug_control)
            zoom_size(factor,cds_labels_dest_2_drug_control)

        return zoom_cb

    zoom_in_wells_source.on_click(make_zoom_cb_wells_source(1.2))
    zoom_out_wells_source.on_click(make_zoom_cb_wells_source(1./1.2))

    zoom_in_wells_dest.on_click(make_zoom_cb_wells_dest(1.2))
    zoom_out_wells_dest.on_click(make_zoom_cb_wells_dest(1./1.2))
 


    #_______________________________________________________
    def add_drug_short():
        add_drug_button.label = "Processing"
        add_drug_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(add_drug)
    add_drug_button.on_click(add_drug_short)

    #_______________________________________________________
    def force_add_drug_short():
        force_add_drug_button.label = "Processing"
        force_add_drug_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(force_add_drug)
    force_add_drug_button.on_click(force_add_drug_short)

    #_______________________________________________________
    def add_drug_other_wells_short():
        add_drug_other_wells_button.label = "Processing"
        add_drug_other_wells_button.button_type = "danger"
        bokeh.io.curdoc().add_next_tick_callback(add_drug_other_wells)
    add_drug_other_wells_button.on_click(add_drug_other_wells_short)

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
    def remove_drug():

        if cds_labels_source.selected.indices == [] and cds_labels_source_supp.selected.indices == []:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Must select one drug to be removed</b>"
            drug_message.visible = True
            return
        
        if cds_labels_source.selected.indices != [] and cds_labels_source_supp.selected.indices != []:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Must select only from source plate or supp plate not both</b>"
            drug_message.visible = True
            return

        if len(cds_labels_source.selected.indices) + len(cds_labels_source_supp.selected.indices) >1:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: Must select only one well to remove drugs</b>"
            drug_message.visible = True
            return

        positions = get_well_mapping(cds_labels_source.selected.indices)
        positions_supp = get_well_mapping(cds_labels_source_supp.selected.indices, issupp=True)
        is_supp = False
        if len(positions_supp)==1: 
            positions=positions_supp
            is_supp = True

        experiement   = Experiment.objects.filter(name=dropdown_exp.value).first()
        plate = experiement.source_plate


        source_well_pos = SourceWellPosition.objects.get(well_plate=plate, position_col=positions[0][0], position_row=positions[0][1], is_supp=is_supp)
        print('source_well_pos=', source_well_pos)
        drugs = source_well_pos.drugs.all()
        drugs_name = ', '.join([f'{drug.derivation_name} {drug.slims_id}' for drug in drugs])

        if len(drugs) == 0:
            drug_message.text = f"<b style='color:red; ; font-size:18px;'> Error: No drug in the selected source well.</b>"
            drug_message.visible = True
            return


        for drug in drugs:

            source_well_pos.remove_drug(drug)

        display_drugs_source_wellplate()
        display_drugs_dest_wellplate()
        cds_labels_source.selected.indices = []
        cds_labels_source_supp.selected.indices = []
        cds_labels_source_drug.selected.indices = []
        cds_labels_source_supp_drug.selected.indices = []

        drug_message.text = f"<b style='color:green; ; font-size:18px;'> Removed drug(s) {drugs_name}</b>"
        drug_message.visible = True

    remove_drug_button.on_click(remove_drug)




    plot_wellplate_source.add_tools(tap_tool)
    plot_wellplate_dest.add_tools(tap_tool)
    plot_wellplate_dest_2.add_tools(tap_tool)
    plot_wellplate_source_supp.add_tools(tap_tool)


    indent = bokeh.models.Spacer(width=30)

    text_layout = bokeh.layouts.column(bokeh.layouts.row(experiment_message),
                                       bokeh.layouts.row(drug_message),
                                       bokeh.layouts.row(mapping_message),
                                       bokeh.layouts.row(pyrat_message),
                                       bokeh.layouts.row(wellvalid_message))

    well_layout = bokeh.layouts.row(indent, bokeh.layouts.column(plot_wellplate_source,plot_wellplate_source_supp), bokeh.layouts.column(plot_wellplate_dest, plot_wellplate_dest_2))
    
    exp_layout = bokeh.layouts.column(bokeh.layouts.row(indent,zoom_in_wells_source, zoom_out_wells_source),
                                      bokeh.layouts.row(dropdown_exp, dropdown_well_plate_source,dropdown_n_supp_sourcewell, dropdown_well_plate_dest, dropdown_n_dest_wellplates),
                                      bokeh.layouts.row(experiment_name, experiment_date, pyrat_id),
                                      bokeh.layouts.row(experiment_description), 
                                      bokeh.layouts.row(create_experiment_button, delete_experiment_button, check_pyrat_id_button, modify_experiment_button))
    

    drug_layout = bokeh.layouts.column(bokeh.layouts.row(zoom_in_wells_dest, zoom_out_wells_dest),
                                       bokeh.layouts.row(slimsid_name, drug_concentration),
                                       bokeh.layouts.row(add_drug_button, add_drug_other_wells_button, force_add_drug_button, remove_drug_button),
                                       bokeh.layouts.row(wellcluster_comment,bokeh.layouts.column(valid_wellcluster,valid_wellcluster_button)),
                                       bokeh.layouts.row(map_drug_button, unmap_drug_button),)

   

    norm_layout = bokeh.layouts.column(bokeh.layouts.row(indent,exp_layout, bokeh.layouts.Spacer(width=50), drug_layout, text_layout),
                                       bokeh.layouts.row(bokeh.layouts.Spacer(height=50)),
                                       well_layout,
                                       bokeh.layouts.row(bokeh.layouts.Spacer(height=50)))

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



#KEEP THIS FUNCTION, IT IS USED TO NAVIGATE SLIMS DRUGS
#___________________________________________________________________________________________
def navigate_slims(drug_deriv):

    deriv_name = ''
    stock_id = ''
    stock_name = ''
    batch_number = ''
    powder_diluent = ''
    stock_concentration = ''
    ultrasonic = False
    storage = ''
    for i in drug_deriv:
        print(i)
    for element in drug_deriv:
        if element["name"]=="cntn_cf_name":
            print('derivation name=',element["value"])
            deriv_name = element["value"]
        if element["name"]=="cntn_fk_originalContent":
            print('stock ID=',element["displayValue"])
            stock_id = element["displayValue"]
        if element["name"]=="cntn_cf_batchNumber":
            print('batch number=',element["value"])
            batch_number = element["value"]
        if element["name"]=="cntn_cf_fk_powderDiluent":
            print('powder diluent=',element["displayValue"])
            powder_diluent = element["displayValue"]
        if element["name"]=="cntn_cf_molarConcentration":
            print('stock_concentration=',element["value"])
            stock_concentration = element["value"]
        if element["name"]=="cntn_cf_ultrasonic":
            print('ultrasonic=',element["value"])
            ultrasonic = element["value"]
        if element["name"]=="locationPath":
            print('storage=',element["value"])
            storage = element["value"]

    slim_drug = SlimsDrug.objects.filter(slims_id=stock_id, name=stock_name)
    if len(slim_drug) == 0:
        cas_number = ''
        reference = ''
        storage = ''
        quantity=''
        hazard=''
        provider=''
        for stock in drug_dstock:
            print('   ---- ------ --------- -------  ',stock)
            if stock["name"]=="cntn_cf_casNumberLinkToWebsiteCatalog":
                print('cas_number=',stock["value"])
                cas_number = stock["value"] 
            if stock["name"]=="cntn_cf_reference":
                print('reference=',stock["value"])
                reference = stock["value"] 
            if stock["name"]=="locationPath":
                print('reference=',stock["value"])
                storage = stock["value"]                     
            if stock["name"]=="cntn_cf_drugPowderWeight":
                print('reference=',stock["value"])
                quantity = stock["value"]                     
            if stock["name"]=="cntn_cf_fk_hazards":
                print('hazard=',stock["value"])
                hazard = stock["value"] 
            if stock["name"]=="cntn_cf_provider":
                print('provider=',stock["value"])
                provider = stock["value"]                     
