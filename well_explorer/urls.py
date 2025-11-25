from django.urls import path
from . import views
from django.contrib.staticfiles.storage import staticfiles_storage
from django.views.generic.base import RedirectView


urlpatterns = [
    path(r"", views.index, name="index"),
    path(r"drugs_listing", views.sortable_table, name="drugs_listing"),
    path(r"bokeh_dashboard", views.bokeh_dashboard, name="bokeh_dashboard"),
]
