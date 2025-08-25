from django.contrib import admin
from django import forms

# Register your models here.
from .models import Experiment, SourceWellPlate, DestWellPlate, SourceWellPosition, DestWellPosition, Drug

class ExperimentAdmin(admin.ModelAdmin):
    search_fields = ["name"]
    ordering = ["name"]

class SourceWellPlateAdmin(admin.ModelAdmin):
    search_fields = ["experiment__name"]
    ordering = ["experiment__name"]

class DestWellPlateAdmin(admin.ModelAdmin):
    search_fields = ["experiment__name"]
    ordering = ["experiment__name"]

class SourceWellPositionAdmin(admin.ModelAdmin):
    search_fields = ["well_plate__experiment__name"]
    ordering = ["well_plate__experiment__name", "position_row", "position_col"]

class DestWellPositionAdmin(admin.ModelAdmin):
    search_fields = ["well_plate__experiment__name"]
    ordering = ["well_plate__experiment__name", "well_plate__plate_number", "position_row"]


admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(SourceWellPlate, SourceWellPlateAdmin)
admin.site.register(DestWellPlate, DestWellPlateAdmin)
admin.site.register(SourceWellPosition, SourceWellPositionAdmin)
admin.site.register(DestWellPosition, DestWellPositionAdmin)

admin.site.register(Drug)

