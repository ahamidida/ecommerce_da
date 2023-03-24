from django.db import models
from django.utils.translation import gettext_lazy as _

class Order(models.Model):
    order_id = models.IntegerField(_('order_id'),primary_key=True)
    date = models.DateField(_('date'))
    user_id = models.IntegerField(_('user_id'))
    total_purchase = models.FloatField(_('total_purchase'))
    latitude = models.FloatField(_('latitude'))
    longitude = models.FloatField(_('longitude'))

    def __str__(self):
        return f"Order {self.order_id}"
