from django.urls import path
from . import views
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView


urlpatterns = [
    path(r"", views.index, name="index"),
    path(r"", views.sortable_table, name="drug_listing"),
    path(r"bokeh_dashboard", views.bokeh_dashboard, name="bokeh_dashboard"),
]
