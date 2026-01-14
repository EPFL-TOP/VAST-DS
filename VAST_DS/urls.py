"""
URL configuration for VAST_DS project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from django.urls import include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

from django.contrib.staticfiles.storage import staticfiles_storage

from well_mapping import views as views_wm
from well_explorer import views as views_we
from bokeh_django import autoload, directory, document, static_extensions



urlpatterns = [

    path("", TemplateView.as_view(template_name="index.html"), name="index"),

    path(r"well_mapping/", views_wm.index, name="index"),
    path(r"well_mapping/bokeh_dashboard", views_wm.bokeh_dashboard, name="bokeh_dashboard_well_mapping"),
    
    path(r"well_explorer/", views_we.index, name="index"),
    path(r"well_explorer/drugs_listing", views_we.drug_list, name="drugs_listing"),
    path(r"well_explorer/experiment_listing", views_we.experiment_list, name="experiment_listing"),
    path(r"well_explorer/bokeh_dashboard", views_we.bokeh_dashboard, name="bokeh_dashboard_well_explorer"),
    path('admin/', admin.site.urls),
]


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


bokeh_apps = [
    autoload("well_mapping/bokeh_dashboard", views_wm.vast_handler),
    autoload("well_explorer/bokeh_dashboard", views_we.vast_handler),
]