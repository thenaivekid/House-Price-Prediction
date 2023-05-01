from django.contrib import admin
from django.urls import path
from prediction.views import (
    prediction,
    predict,
    )

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', prediction),
    path('predict', predict, name='predict'),
]
